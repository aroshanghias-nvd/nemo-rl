"""
srun -p interactive -A llmservice_fm_vision -N 1 --pty \
    --container-image /lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/vllm-3225729-cuda-12.8.1.sqsh \
    --container-mounts "/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/jseppanen/dev:/code" \
    --gpus 1 \
    --job-name "nemo-rl-dev:interactive" \
    -t 04:00:00 \
    bash -l
"""

import json
import base64
import logging
import os
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import openai
from tqdm import tqdm

# CONFIG
MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
GENERATIONS_PER_PROMPT = 4
MAX_TOKENS = 16384
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 0.95
NUM_TILES = 12
BATCH_SIZE = 64
CONCURRENCY = 2 * BATCH_SIZE

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
    assert isinstance(path, str), f"path must be a string, got {type(path)}"
    mime = detect_mime_type(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def read_samples(jsonl_path, shard_id=0, num_shards=1):
    """Yield (image, question, answer) from a JSONL file."""
    with open(jsonl_path, "r") as f:
        for idx, line in enumerate(f):
            if (idx % num_shards) != shard_id:
                continue
            row = json.loads(line)
            image = row.get("image")
            conv = row["conversations"]
            question = conv[0]["value"]
            answer = conv[1]["value"]
            dataset = row.get("dataset")
            source_path = row.get("source_path")
            source_index = row.get("source_index")
            sample_id = row.get("id")
            metadata = {
                "dataset": dataset,
                "source_path": source_path,
                "source_index": source_index,
                "id": sample_id,
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


def launch_vllm_server(port):
    """Start one vLLM server on a port."""
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
        "--max-num-seqs",
        str(BATCH_SIZE),
        "--max-num-batched-tokens",
        str(MAX_TOKENS),
        # "--mm-processor-kwargs",
        # '{"use_fast": true}',
        "--port",
        str(port),
    ]
    env = os.environ.copy()
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
    log = open(f"vllm_{task_id}.log", "w")
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


def _infer_one(args):
    """Execute one completion and return (sample, total_tokens)."""
    client, messages, sample = args
    try:
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
        if resp and resp.choices:
            pred = resp.choices[0].message.content.strip()
            if "</think>" in pred and "<think>" not in pred:
                pred = "<think>\n" + pred
            result = dict(
                sample,
                prediction=pred,
                finish_reason=resp.choices[0].finish_reason,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
            return result, resp.usage.completion_tokens
    except Exception:
        logging.exception("Error in inference task")
        result = dict(sample, finish_reason="error")
        return result, 0


def run_inference_over_shard(client, input_path, output_path, shard_id, num_shards):
    """Run inference concurrently over one shard and write JSONL outputs."""

    def job_iter():
        """Yield (messages, sample) for each generation task."""
        for image, question, answer, metadata in read_samples(
            input_path, shard_id, num_shards
        ):
            image_data_url = image_to_data_url(image) if image else None
            q = question + "\n" + PROMPT
            messages = build_messages(q, image_data_url=image_data_url, reasoning=True)
            sample = {
                "image": image,
                "question": q,
                "answer": answer,
                **metadata,
            }
            for _ in range(GENERATIONS_PER_PROMPT):
                yield client, messages, sample

    total_samples = count_samples(input_path, shard_id, num_shards)
    progress = tqdm(total=GENERATIONS_PER_PROMPT * total_samples)
    start_times = []
    token_counts = []
    with open(output_path, "w", buffering=1, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            start_times.append(time.perf_counter())
            for row, output_tokens in executor.map(_infer_one, job_iter()):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                token_counts.append(output_tokens)
                start_times = start_times[-int(1.1 * CONCURRENCY) :]
                token_counts = token_counts[-int(1.1 * CONCURRENCY) :]
                now = time.perf_counter()
                tps = sum(token_counts) / (now - start_times[0])
                progress.set_description(f"{tps:.1f} TPS")
                progress.update()
                start_times.append(now)
    progress.close()


def count_samples(jsonl_path, shard_id=0, num_shards=1):
    """Return number of non-empty lines in a JSONL file."""
    with open(jsonl_path, "r") as f:
        return sum(1 for idx, line in enumerate(f) if (idx % num_shards) == shard_id)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--shard-id", type=int, default=0)
@click.option("--num-shards", type=int, default=1)
def main(input_path, output_path, shard_id, num_shards):
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
    port = 18000 + task_id

    proc = None
    log = None
    try:
        proc, log = launch_vllm_server(port)
        wait_for_port("localhost", port, timeout=600)
        client = openai.OpenAI(api_key="dummy", base_url=f"http://localhost:{port}/v1")
        run_inference_over_shard(client, input_path, output_path, shard_id, num_shards)
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if log is not None:
            try:
                log.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
