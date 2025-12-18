"""
srun -p interactive -A llmservice_fm_vision -N 1 --pty \
    --container-image /lustre/fsw/portfolios/llmservice/users/jseppanen/sqsh/vllm-c799126-cuda-12.8.1.sqsh \
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
import re
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from contextlib import contextmanager

import click
import openai
from tqdm import tqdm

INPUT_PATH = "/lustre/fs1/portfolios/llmservice/users/jseppanen/dev/nemo-rl-n5p5-mmpr-filtered/raven_output.jsonl"
INPUT_SIZE = 42000

MODEL = "openai/gpt-oss-120b"
GENERATIONS_PER_PROMPT = 1
MAX_TOKENS = 32768

REASONING_EFFORT = "medium"
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 0.95

BATCH_SIZE = 16
CONCURRENCY = BATCH_SIZE
TIMEOUT = 600

REPHRASE_PROMPT = """
Here are two reasoning traces for the same visual puzzle:

<reference>
{reference}
</reference>

<prediction>
{prediction}
</prediction>

Your task: Edit the PREDICTION to fix any errors, using the REFERENCE as ground truth.

**What to fix:**
- Factual errors about the image (wrong colors, shapes, positions, counts)
- Logical reasoning mistakes
- Wrong final answer (must match the reference)

**What to preserve:**
- The prediction's literal text as much as possible
- Sentence structure and writing style
- Level of detail (don't add or remove reasoning steps)

**Rules:**
- Don't summarize steps or patterns, but instead write out all details in full
- Don't ever mention the reference solution; the prediction must be self-contained

Write your corrected version within <prediction>...</prediction> tags.
""".strip()

FINAL_FORMATTING_PROMPT = (
    "Write the final answer in this format:\n\nAnswer: \\boxed{...}."
)


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


def build_messages(prompt, *, images=None, system_prompt=None):
    """Build OpenAI chat messages with optional images."""
    content = [{"type": "text", "text": prompt}]
    if images:
        assert isinstance(images, list), f"images must be a list, got {type(images)}"
        for image in images:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}
            )
    messages = []
    if system_prompt:
        messages.append(
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        )
    messages.append({"role": "user", "content": content})
    return messages


def launch_vllm_server(port):
    """Start one vLLM server on a port."""
    # https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#recipe-for-nvidia-blackwell-hopper-hardware
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--async-scheduling",
        "--no-enable-prefix-caching",
        "--max-cudagraph-capture-size", "2048",
        "--max-num-batched-tokens", "8192",
        "--chat-template-content-format",
        "openai",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        str(2 * MAX_TOKENS),
        "--max-num-seqs",
        str(BATCH_SIZE),
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


def _process_sample(args):
    client, sample = args

    # relaxed parsing of natural thinking trace (Qwen can miss the thinking trace)
    if "<think>" not in sample["prediction"]:
        natural_think = sample["prediction"].replace("</think>", "").strip()
    elif "</think>" in sample["prediction"]:
        natural_think = re.search(r"<think>(.*)</think>", sample["prediction"], re.DOTALL).group(1).strip()
    else:
        # truncated
        natural_think = re.search(r"<think>(.*)", sample["prediction"], re.DOTALL).group(1).strip()
    synthetic_think = sample["gt_think"].replace("<think>", "").replace("</think>", "").strip()
    resp, tokens, latency = _llm_call(
        client,
        prompt=REPHRASE_PROMPT.format(prediction=natural_think, reference=synthetic_think),
        sample_id=sample["id"],
    )
    if resp is None:
        return None, 0, 0
    rephrase_prediction = resp["prediction"]
    rephrased_think = re.sub(
        r"<think>(.*?)</think>", "", rephrase_prediction, flags=re.DOTALL
    ).strip()
    # relaxed parsing of rephrased thinking trace (gpt-oss-120b can miss the output formatting)
    if "<prediction>" in rephrased_think:
        rephrased_think = rephrased_think.split("<prediction>")[-1].strip()
    if "</prediction>" in rephrased_think:
        rephrased_think = rephrased_think.split("</prediction>")[0].strip()
    prediction = (
        f"<think>\n{rephrased_think}\n</think>\n\nAnswer: \\boxed{{{sample['answer']}}}"
    )
    resp = dict(
        sample,
        prediction=prediction,
        qwen_prediction=sample["prediction"],
        rephrase_prediction=rephrase_prediction,
        finish_reason=resp["finish_reason"],
        prompt_tokens=sample["prompt_tokens"] + resp["prompt_tokens"],
        completion_tokens=sample["completion_tokens"] + resp["completion_tokens"],
        total_tokens=sample["total_tokens"] + resp["total_tokens"],
    )
    return resp, tokens, latency


def _llm_call(client, *, prompt=None, images=None, system_prompt=None, sample_id=None):
    """Execute one completion and return (sample, total_tokens)."""
    messages = build_messages(prompt, images=images, system_prompt=system_prompt)
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
                    "reasoning_effort": REASONING_EFFORT,
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
            # gpt-oss
            if resp.choices[0].message.reasoning_content:
                pred = f"<think>\n{resp.choices[0].message.reasoning_content}\n</think>\n\n{pred}"
            result = dict(
                prediction=pred,
                finish_reason=resp.choices[0].finish_reason,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
            )
            return result, resp.usage.completion_tokens, latency
        except openai.BadRequestError as e:
            logging.error(f"Bad request, skipping sample {sample_id}: {e}")
            return None, 0, 0
        except Exception as e:
            if retry < retries - 1:
                logging.warning(
                    f"Retry {retry + 1}/{retries}: Inference failed for sample {sample_id}: {e}"
                )
                time.sleep(2**retry)
                continue
            logging.exception(f"Error in inference task for sample {sample_id}")
            raise


def run_inference_over_shard(client, input_path, output_path, shard_id, num_shards):
    """Run inference concurrently over one shard and write JSONL outputs."""

    def jobs():
        """Yield samples for each generation task."""
        for _, line in read_lines(input_path, shard_id, num_shards):
            sample = json.loads(line)
            for _ in range(GENERATIONS_PER_PROMPT):
                yield client, sample

    total_samples = INPUT_SIZE // num_shards
    with open(output_path, "w", buffering=1, encoding="utf-8") as f:
        with show_progress(GENERATIONS_PER_PROMPT * total_samples) as progress:
            for row, output_tokens, latency in concurrent_map(_process_sample, jobs()):
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


@contextmanager
def vllm_server(port, *, shutdown=True):
    try:
        wait_for_port("localhost", port, timeout=1)
        print(f"vLLM server ready on port {port}", file=sys.stderr)
        yield
        return
    except TimeoutError:
        pass

    proc = log = None
    try:
        print(f"Launching vLLM server on port {port}", file=sys.stderr)
        proc, log = launch_vllm_server(port)
        # Qwen3-VL-235B-A22B-Thinking-FP8 takes ~10 mins to start
        wait_for_port("localhost", port, timeout=1200, proc=proc)
        print(f"vLLM server ready on port {port}", file=sys.stderr)
        yield
    finally:
        if shutdown:
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


# Run inference and start vllm server if not already running:
#
#     $ infer_raven_paraphrase.py input.jsonl output_0.jsonl --shard-id=0 --num-shards=10
#
# Start vllm server only:
#
#     $ infer_raven_paraphrase.py
#
@click.command()
@click.argument("input_path", type=click.Path(exists=True), default=INPUT_PATH)
@click.argument("output_path", type=click.Path(), required=False)
@click.option("--shard-id", type=int, default=0)
@click.option("--num-shards", type=int, default=1)
def main(input_path, output_path, shard_id, num_shards):
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
    port = 18765 + task_id

    if output_path is None:
        with vllm_server(port, shutdown=False):
            return

    logging.basicConfig(level=logging.WARNING)
    try:
        with vllm_server(port):
            client = openai.OpenAI(
                api_key="dummy", base_url=f"http://localhost:{port}/v1", timeout=TIMEOUT
            )
            run_inference_over_shard(
                client, input_path, output_path, shard_id, num_shards
            )
    except Exception as e:
        logging.exception(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
