

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
from typing import Any

import click
import openai
from tqdm import tqdm
import json
import re
import random
import sys
from collections import defaultdict
from glob import glob
import requests
from mathruler.grader import grade_answer

MODEL = "/lustre/fsw/portfolios/llmservice/users/jseppanen/checkpoints/mmpr_mpo_sft_n5p5_12b_300k_13p52_cot_ruler_only_from_iter_2400_1011_step_425"
PRECISION = "bf16"
GENERATIONS_PER_PROMPT = 5
MAX_TOKENS = 16384
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 0.95
NUM_TILES = 12
BATCH_SIZE = 64
CONCURRENCY = 2 * BATCH_SIZE
TIMEOUT = 600
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
    if path.startswith(("http://", "https://")):
        mime = detect_mime_type(path)  # guess from extension
        resp = requests.get(path, timeout=30)
        resp.raise_for_status()
        b64 = base64.b64encode(resp.content).decode("utf-8")
        return f"data:{mime};base64,{b64}"
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
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--trust-remote-code",
        "--chat-template-content-format", "openai",
        "--mamba_ssm_cache_dtype",
        "float32",
        "--video-pruning-rate",
        "0",
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
    if PRECISION == "fp8":
        cmd.extend(["--quantization", "modelopt"])
    elif PRECISION == "bf16":
        cmd.extend(["--dtype", "bfloat16"])
    else:
        raise ValueError(f"Unsupported precision: {PRECISION}")
    env = os.environ.copy()
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
            if not (resp and resp.choices):
                raise ValueError(f"No response: {resp}")
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
        except openai.BadRequestError as e:
            logging.error(f"Bad request, skipping sample {sample['id']}: {e}")
            return None, 0
        except Exception as e:
            if retry < retries - 1:
                logging.warning(
                    f"Retry {retry + 1}/{retries}: Inference failed for sample {sample['id']}: {e}"
                )
                time.sleep(2**retry)
                continue
            logging.exception(f"Error in inference task for sample {sample['id']}")
            return None, 0


def run_inference_over_shard(client, input_path, output_path, image_root, shard_id, num_shards):
    """Run inference concurrently over one shard and write JSONL outputs."""
    # Determine chunk boundaries and total lines for progress bar
    total_lines = count_samples(input_path)
    chunk_size = total_lines // num_shards
    chunk_start = shard_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, total_lines)
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk start: {chunk_start}")
    print(f"Chunk end: {chunk_end}")
    def job_iter():
        """Yield (messages, sample) for each generation task."""
        for _, line in read_lines(input_path, chunk_start, chunk_end):
            if input_path.endswith(".jsonl"):
                sample = json.loads(line)
            else:
                sample = line
            if not sample['image'].startswith(("http://", "https://")):
                sample["images"] = [os.path.join(image_root, sample["image"])]
            else:
                sample["images"] = [sample["image"]]
            
            question = sample["question"] + "\n" + PROMPT
            messages = build_messages(
                question, images=sample["images"], reasoning=True
            )
            for _ in range(GENERATIONS_PER_PROMPT):
                yield client, messages, sample
    total_samples = count_samples(input_path)//num_shards
    print(f"Total samples: {total_samples} in shard {shard_id} with num_shards {num_shards}")
    with open(output_path, "w", buffering=1, encoding="utf-8") as f:
        with show_progress(GENERATIONS_PER_PROMPT * total_samples) as progress:
            for row, output_tokens in concurrent_map(_infer_one, job_iter()):
                if row is not None:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.update(output_tokens)

def count_samples(input_path):
    """Count number of samples in a shard."""
    with open(input_path, "r") as f:
        if input_path.endswith(".jsonl"):
            return sum(1 for _ in f)
        elif input_path.endswith(".json"):
            return len(json.load(f))

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
    """Yield a tracker that updates tqdm progress and TPS."""
    progress = tqdm(total=total)
    start_times = []
    token_counts = []
    start_times.append(time.perf_counter())

    class _Tracker:
        def update(self, token_count: int):
            now = time.perf_counter()
            token_counts.append(token_count)
            if len(start_times) > CONCURRENCY:
                # Calculate tokens/sec throughput but avoid inflated values due to concurrent & batched inference
                tps = min(
                    sum(token_counts[-i:]) / (now - start_times[-i])
                    for i in range(CONCURRENCY, min(len(start_times), 2 * CONCURRENCY))
                )
                progress.set_description(f"{tps:.1f} TPS")
            progress.update()
            start_times.append(now)

    try:
        yield _Tracker()
    finally:
        progress.close()


def read_lines(path, chunk_start, chunk_end):
    """Yield lines from a files based on shard ID."""
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            for sample_idx, line in enumerate(f):
                if sample_idx < chunk_start:
                    continue
                if sample_idx >= chunk_end:
                    break
                yield sample_idx, line
        elif path.endswith(".json"):
            data = json.load(f)
            for sample_idx, line in enumerate(data):
                if sample_idx < chunk_start:
                    continue
                if sample_idx >= chunk_end:
                    break
                yield sample_idx, line


def evaluate_filter(input_paths, output_path):
    correctness = defaultdict(int)
    inner_pattern = r"([^\n]+?)"
    result_pattern = rf"(?:\\\({inner_pattern}\\\)|\${inner_pattern}\$|{inner_pattern})"
    answer_patterns = [
        rf"[Tt]he answer is {result_pattern}\.?$",
        # rf"[Tt]he final answer is {result_pattern}\.?$",
        # rf"[Tt]he result is {result_pattern}\.?$",
    ]
    for path, idx, sample in read_jsonls(input_paths):
        correct_answer = str(sample["answer"])
        output_text = sample["prediction"]
        output_text = re.sub(r"<think>.*</think>", "", output_text, flags=re.DOTALL)
        response = None
        for answer_pattern in answer_patterns:
            response = re.search(answer_pattern, output_text, flags=re.DOTALL)
            if not response:
                continue
            response = next(g for g in response.groups() if g is not None)
            if not response:
                continue
            response = response.strip()
            break
        if not response:
            continue
        try:
            is_correct = grade_answer(response, correct_answer)
        except Exception as e:
            logging.exception(f"Error in grade_answer: {e}")
            is_correct = False
        image = sample["image"]
        if is_correct:
            correctness[str({"question":sample["question"],
            "images": tuple(image),
            "answer": sample["answer"]})] += 1/GENERATIONS_PER_PROMPT
        else:
            correctness[str({"question":sample["question"],
            "images": tuple(image),
            "answer": sample["answer"]})] += 0.0
    samples_distribution = [0 for _ in range(GENERATIONS_PER_PROMPT+1)]
    with open(output_path, "w") as f:
        for sample, pass_rate in correctness.items():
            sample = eval(sample)
            sample['pass_rate'] = pass_rate
            samples_distribution[int(pass_rate * GENERATIONS_PER_PROMPT)] += 1
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Total samples : {len(correctness)}")
    print(f"distribution of samples: {samples_distribution}")

# def evaluate_filter_llm(client, input_paths, output_path):
#     PROMPT = """
    
    
    
#     """
#     correctness = defaultdict(int)
#     inner_pattern = r"([^\n]+?)"
#     result_pattern = rf"(?:\\\({inner_pattern}\\\)|\${inner_pattern}\$|{inner_pattern})"
#     answer_patterns = [
#         rf"[Tt]he answer is {result_pattern}\.?$",
#         # rf"[Tt]he final answer is {result_pattern}\.?$",
#         # rf"[Tt]he result is {result_pattern}\.?$",
#     ]
#     for path, idx, sample in read_jsonls(input_paths):
#         correct_answer = str(sample["answer"])
#         output_text = sample["prediction"]
#         output_text = re.sub(r"<think>.*</think>", "", output_text, flags=re.DOTALL)
#         response = None
#         for answer_pattern in answer_patterns:
#             response = re.search(answer_pattern, output_text, flags=re.DOTALL)
#             if not response:
#                 continue
#             response = next(g for g in response.groups() if g is not None)
#             if not response:
#                 continue
#             response = response.strip()
#             break
#         if not response:
#             continue
#         try:
#             is_correct = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": PROMPT + output_text}],
#                 temperature=TEMPERATURE,
#                 stream=False,
#             )
#             if not (resp and resp.choices):
#                 raise ValueError(f"No response: {resp}")
#             pred = resp.choices[0].message.content.strip()



def read_jsonls(pattern):
    paths = glob(pattern)
    for path in sorted(paths):
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                yield path, idx, sample

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.argument("image_root", type=click.Path(exists=True))
@click.option("--shard-id", type=int, default=0)
@click.option("--num-shards", type=int, default=1)
@click.option("--mode", type=str, default="infer")
def main(input_path, output_dir, image_root, shard_id, num_shards, mode):
    output_path = output_dir + f"_shard_{shard_id}.jsonl"
    if mode == "infer":
        task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
        port = 18765 + task_id
        logging.basicConfig(level=logging.WARNING)
        proc = None
        log = None
        try:
            proc, log = launch_vllm_server(port)
            wait_for_port("localhost", port, timeout=1200, proc=proc)
            client = openai.OpenAI(
                api_key="dummy", base_url=f"http://localhost:{port}/v1", timeout=TIMEOUT
            )
            run_inference_over_shard(client, input_path, output_path, image_root, shard_id, num_shards)
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
    evaluate_filter(output_path, output_dir+f"_shard{shard_id}_filtered.jsonl")
    #evaluate_filter_llm(client, output_path, output_dir+f"_shard{shard_id}_filtered_llm.jsonl")

if __name__ == "__main__":
    main()