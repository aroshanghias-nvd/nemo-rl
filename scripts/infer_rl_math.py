"""

srun -p interactive -A llmservice_fm_vision -N 1 --pty \
    --container-image /lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/vllm-3225729-cuda-12.8.1.sqsh \
    --container-mounts "/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/jseppanen/dev:/code" \
    --gpus 8 \
    --exclusive \
    --job-name "nemo-rl-dev:interactive" \
    -t 04:00:00 \
    bash -l

vllm serve nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8 \
    --trust-remote-code \
    --quantization modelopt \
    --mamba_ssm_cache_dtype float32 \
    --video-pruning-rate 0 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --api-server-count=4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    >vllm.log 2>&1 &
"""

import json
import base64
import logging
import os
import queue
import socket
import subprocess
import threading
import time
from pathlib import Path

import click
import openai

# CONFIG
MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
MAX_TOKENS = 16384
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 0.95
NUM_TILES = 12
PORTS = list(range(8000, 8008))
GPU_IDS = list(range(8))

# nano-v2 won't really follow formats that deviate from the SFT data
# PROMPT = "Answer the question and output ONLY the final answer followed by a newline."
# "Answer the question after looking at the image. You should output only a single uppercase character (A, B, C, D, ...)."
# "Reason and answer the question. Give your final answer between the <answer>...</answer> tags."
# "Solve the following question step-by-step. Output ONLY the FINAL ANSWER in this format:\n\n\\boxed{your_final_answer_here}"
PROMPT = "Think step-by-step and write the final answer in this format:\n\nThe answer is \\(...\\)."


def detect_mime_type(path):
    """Return a MIME type for an image path."""
    ext = Path(path).suffix.lower()
    if ext in [".png"]:
        return "image/png"
    if ext in [".webp"]:
        return "image/webp"
    if ext in [".gif"]:
        return "image/gif"
    return "image/jpeg"


def image_to_data_url(path):
    """Return a base64 data URL for an image file path."""
    mime = detect_mime_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def read_samples(jsonl_path):
    """Yield (image, question, answer) from a JSONL file."""
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            image = row.get("image")
            conv = row["conversations"]
            question = conv[0]["value"]
            answer = conv[1]["value"]
            dataset = row.get("dataset")
            source_path = row.get("source_path")
            source_index = row.get("source_index")
            metadata = {
                "dataset": dataset,
                "source_path": source_path,
                "source_index": source_index,
            }
            yield image, question, answer, metadata


def build_messages(question, image_data_url=None, reasoning=False):
    """Build OpenAI chat messages with optional image."""
    messages = [{"role": "system", "content": "/think" if reasoning else "/no_think"}]
    if image_data_url:
        content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": question})
    return messages


def launch_vllm_server(port, gpu_id):
    """Start one vLLM server on a port pinned to one GPU."""
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--trust-remote-code",
        "--api-server-count",
        "2",
        "--quantization",
        "modelopt",
        "--mamba_ssm_cache_dtype",
        "float32",
        "--video-pruning-rate",
        "0",
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(0.9),
        "--max-model-len",
        str(MAX_TOKENS),
        "--port",
        str(port),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log = open(f"vllm_{port}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def wait_for_port(host, port, timeout):
    """Block until TCP port opens or timeout."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready")


def vllm_worker(port, request_q, response_q):
    """Consume items, call chat completions on one vLLM server, produce rows."""
    client = openai.OpenAI(api_key="dummy", base_url=f"http://localhost:{port}/v1")
    while True:
        item = request_q.get()
        if item is None:
            return
        image, question, answer, metadata = item
        try:
            image_data_url = image_to_data_url(image) if image else None
            question = question + "\n" + PROMPT
            messages = build_messages(question, image_data_url=image_data_url, reasoning=True)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                stream=False,
                extra_body={
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    "mm_processor_kwargs": {"max_num_tiles": NUM_TILES},
                },
            )
            pred = resp.choices[0].message.content.strip() if resp and resp.choices else ""
            if "</think>" in pred and "<think>" not in pred:
                pred = "<think>\n" + pred
        except Exception:
            logging.exception("Error in vLLM worker")
            pred = None
        response_q.put({
            "image": image,
            "question": question,
            "answer": answer,
            "prediction": pred,
            **metadata,
        })


def sample_reader(input_path, request_q, n_workers, counters, done_event):
    """Read samples and enqueue requests, then send sentinels."""
    for item in read_samples(input_path):
        request_q.put(item)
        counters["enqueued"] += 1
    for _ in range(n_workers):
        request_q.put(None)
    done_event.set()


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path, output_path):
    procs = []
    try:
        for port, gpu in zip(PORTS, GPU_IDS):
            proc, log = launch_vllm_server(port, gpu)
            procs.append((proc, log))
        for port in PORTS:
            wait_for_port("localhost", port, timeout=600)

        request_q = queue.Queue(maxsize=100)
        response_q = queue.Queue(maxsize=100)

        threads = []
        for port in PORTS:
            t = threading.Thread(target=vllm_worker, args=(port, request_q, response_q), daemon=True)
            t.start()
            threads.append(t)

        counters = {"enqueued": 0}
        done_event = threading.Event()
        reader = threading.Thread(
            target=sample_reader,
            args=(input_path, request_q, len(PORTS), counters, done_event),
            daemon=True,
        )
        reader.start()

        received = 0
        with open(output_path, "w", buffering=1, encoding="utf-8") as f:
            while True:
                row = response_q.get()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                received += 1
                if done_event.is_set() and received >= counters["enqueued"]:
                    break

        for t in threads:
            t.join(timeout=1)
        reader.join(timeout=1)
    finally:
        for proc, log in procs:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                log.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
