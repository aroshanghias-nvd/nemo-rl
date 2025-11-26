"""
srun -p interactive -A llmservice_fm_vision -N 1 --pty \
    --container-image /lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/vllm-c799126-cuda-12.8.1.sqsh \
    --container-mounts "/lustre:/lustre,/lustre/fsw/portfolios/llmservice/users/jseppanen/dev:/code" \
    --gpus 4 \
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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from contextlib import contextmanager

import click
import openai
from tqdm import tqdm

INPUT_PATH = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr_1_2_verifiable_1126.jsonl"

# batch 1 (failed with timeouts)
# HARD_SAMPLE_IDS = [json.loads(l) for l in open("/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/mmpr_nanov2_hard_sample_ids_v1.jsonl")]
# HARD_SAMPLE_IDS = set(s["id"] for s in HARD_SAMPLE_IDS)

# batch 2 (remaining samples from batch 1)
HARD_SAMPLE_IDS = set(int(l.replace("[", "").replace("]", "")) for l in open("/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/hard_ids2"))

MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
GENERATIONS_PER_PROMPT = 1
MAX_TOKENS = 16384
# https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#thinking-models
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.95

# reduce batch size to prevent timeouts for long/slow answers
BATCH_SIZE = 16
CONCURRENCY = BATCH_SIZE
TIMEOUT = 600


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


def build_messages(question, images=None, reasoning=False):
    """Build OpenAI chat messages with optional images."""
    system_prompt = "/think" if reasoning else "/no_think"
    content = [{"type": "text", "text": question}]
    if images:
        assert isinstance(images, list), f"images must be a list, got {type(images)}"
        for image in images:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}
            )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    return messages


def launch_vllm_server(port):
    """Start one vLLM server on a port."""
    # https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html#running-qwen3-vl
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--chat-template-content-format", "openai",
        "--tensor-parallel-size", "4",
        "--limit-mm-per-prompt.video", "0",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len",
        str(2 * MAX_TOKENS),
        "--max-num-seqs",
        str(BATCH_SIZE),
        "--max-num-batched-tokens",
        str(MAX_TOKENS),
        "--port",
        str(port),
    ]
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["OMP_NUM_THREADS"] = "1"
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
    log = open(f"vllm_{task_id}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def wait_for_port(host, port, timeout, proc=None):
    """Block until TCP port opens, process exits, or timeout."""
    end = time.time() + timeout
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited with code {proc.returncode} before port {port} became ready"
            )
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Port {port} not ready")


def _infer_one(args):
    """Execute one completion and return (sample, total_tokens)."""
    client, messages, sample = args
    retries = 5
    for retry in range(retries):
        try:
            start_time = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False,
                extra_body={
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                },
            )
            if not (resp and resp.choices):
                raise ValueError(f"No response: {resp}")
            latency = time.perf_counter() - start_time
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
            return result, resp.usage.completion_tokens, latency
        except openai.BadRequestError as e:
            logging.error(f"Bad request, skipping sample {sample['id']}: {e}")
            return None, 0, 0
        except Exception as e:
            if retry < retries - 1:
                logging.warning(
                    f"Retry {retry + 1}/{retries}: Inference failed for sample {sample['id']}: {e}"
                )
                time.sleep(2**retry)
                continue
            logging.exception(f"Error in inference task for sample {sample['id']}")
            raise


def run_inference_over_shard(client, input_path, output_path, shard_id, num_shards):
    """Run inference concurrently over one shard and write JSONL outputs."""

    def job_iter():
        """Yield (messages, sample) for each generation task."""
        for _, line in read_lines(input_path, shard_id, num_shards):
            sample = json.loads(line)
            if sample["id"] not in HARD_SAMPLE_IDS:
                continue
            messages = build_messages(
                sample["question"], images=sample["images"], reasoning=True
            )
            for _ in range(GENERATIONS_PER_PROMPT):
                yield client, messages, sample

    total_samples = len(HARD_SAMPLE_IDS) // num_shards  # mmpr-1.2 hard samples
    with open(output_path, "w", buffering=1, encoding="utf-8") as f:
        with show_progress(GENERATIONS_PER_PROMPT * total_samples) as progress:
            for row, output_tokens, latency in concurrent_map(_infer_one, job_iter()):
                if row is not None:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.update(output_tokens, latency)


def concurrent_map(fn, jobs):
    """Compute unordered map(fn, jobs) concurrently in a thread pool."""
    jobs_iter = iter(jobs)
    tasks = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        for _ in range(CONCURRENCY):
            try:
                tasks.append(executor.submit(fn, next(jobs_iter)))
            except StopIteration:
                break

        while tasks:
            done, not_done = wait(tasks, return_when=FIRST_COMPLETED)
            tasks = list(not_done)
            for fut in done:
                yield fut.result()
                try:
                    tasks.append(executor.submit(fn, next(jobs_iter)))
                except StopIteration:
                    pass


@contextmanager
def show_progress(total):
    """Yield a tracker that updates tqdm progress and QPH/TPS."""
    progress = tqdm(total=total)
    start_times = []
    token_counts = []
    latencies = []
    start_times.append(time.perf_counter())

    class _Tracker:
        def update(self, token_count: int, latency: float):
            now = time.perf_counter()
            token_counts.append(token_count)
            latencies.append(latency)
            while len(start_times) >= 2 and (now - start_times[0]) > TIMEOUT:
                start_times.pop(0)
                token_counts.pop(0)
                latencies.pop(0)
            qph = len(start_times) / (now - start_times[0]) * 3600
            tps = sum(token_counts) / (now - start_times[0])
            lat_mean = sum(latencies) / len(latencies)
            lat_max = max(latencies)
            progress.set_description(
                f"{qph:.0f} QPH, {tps:.0f} TPS, {lat_mean:.0f}/{lat_max:.0f} s"
            )
            progress.update()
            start_times.append(now)

    try:
        yield _Tracker()
    finally:
        progress.close()


def read_lines(path, shard_id=0, num_shards=1):
    """Yield lines from a files based on shard ID."""
    with open(path, "r") as f:
        for sample_idx, line in enumerate(f):
            if (sample_idx % num_shards) != shard_id:
                continue
            yield sample_idx, line


@click.command()
@click.argument("input_path", type=click.Path(exists=True), default=INPUT_PATH)
@click.argument("output_path", type=click.Path())
@click.option("--shard-id", type=int, default=0)
@click.option("--num-shards", type=int, default=1)
def main(input_path, output_path, shard_id, num_shards):
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
    port = 18765 + task_id

    logging.basicConfig(level=logging.WARNING)
    proc = None
    log = None
    try:
        proc, log = launch_vllm_server(port)
        # Qwen3-VL-235B-A22B-Thinking-FP8 takes ~10 mins to start
        wait_for_port("localhost", port, timeout=1200, proc=proc)
        client = openai.OpenAI(
            api_key="dummy", base_url=f"http://localhost:{port}/v1", timeout=TIMEOUT
        )
        run_inference_over_shard(client, input_path, output_path, shard_id, num_shards)
    except Exception as e:
        logging.exception(f"Error in main: {e}")
        raise
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
