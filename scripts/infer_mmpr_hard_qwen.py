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

HARD_SAMPLE_IDS = [json.loads(l) for l in open("/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/mmpr_nanov2_hard_sample_ids_v1.jsonl")]
HARD_SAMPLE_IDS = set(s["id"] for s in HARD_SAMPLE_IDS)

# throughput about 5000 samples per hour per 4 gpus

MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"
GENERATIONS_PER_PROMPT = 1
MAX_TOKENS = 16384
# https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#thinking-models
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.95
BATCH_SIZE = 64
CONCURRENCY = 2 * BATCH_SIZE

# nano-v2 won't really follow formats that deviate from the SFT data
# FORMATTING_PROMPT = "Answer the question and output ONLY the final answer followed by a newline."
# "Answer the question after looking at the image. You should output only a single uppercase character (A, B, C, D, ...)."
# "Reason and answer the question. Give your final answer between the <answer>...</answer> tags."
# "Solve the following question step-by-step. Output ONLY the FINAL ANSWER in this format:\n\n\\boxed{your_final_answer_here}"
# "Please answer the question and put the final answer within \\boxed{...}."
# FORMATTING_PROMPT = "Think step-by-step and write the final answer in this format:\n\nThe answer is \\(...\\)."
# FORMATTING_PROMPT = "Think step-by-step and write the final answer in this format:\n\nFinal answer: ..."
# FORMATTING_PROMPT = "Answer the preceding question. The last line of your response should follow this format:\n\nAnswer: \\boxed{$FINAL_ANSWER}."
FORMATTING_PROMPT = "Please answer the question and put the final answer in this format:\n\nAnswer: \\boxed{...}."

# verifiable but missing formatting instructions
NEED_FORMATTING_PROMPT = """
mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_wo_image
""".strip().split()

# non-verifiable preference data
SKIP = """
mmpr-1.2-inat_train2018_merge_gpt4o_en_20240819_sr0.50_wo_image
mmpr-1.2-spot_the_diff_en_20240910_sr0.50_wo_image
mmpr-1.2-ai2d_cap_gpt4o_en_20240410
mmpr-1.2-sam_cap_review_negative_en_20240918
mmpr-1.2-llavar_inhouse_sft_longcap_en_20240521
mmpr-1.2-gaokao_chemistry_ocr_zh_20240623_sr0.50_wo_image
mmpr-1.2-gaokao_chemistry_zh_20240623_sr0.50_wo_image
mmpr-1.2-gaokao_math_jieti_zh_20240805_sr0.50_wo_image
mmpr-1.2-gaokao_math_ocr_zh_20240623_sr0.50_wo_image
mmpr-1.2-gaokao_physics_ocr_zh_20240623_sr0.50_wo_image
mmpr-1.2-gaokao_physics_zh_20240623_sr0.50_wo_image
mmpr-1.2-gaokao_politics_zh_20240623_sr0.50_wo_image
mmpr-1.2-openbmb_RLAIF-V-Dataset
mmpr-1.2-wildvision_gpt4o_en_20240903.jsonl_extracted_sr0.0_with_image
mmpr-1.2-wildvision_gpt4v_to_gpt4o_en_20240903.jsonl_extracted_sr0.0_with_image
mmpr-1.2-wildvision_gpt4o_en_20240903.jsonl_extracted_sr0.5_with_image
mmpr-1.2-wildvision_gpt4v_to_gpt4o_en_20240903.jsonl_extracted_sr0.5_with_image
mmpr-1.2-RLAIF-V-Dataset_sr0.5_wo_image
mmpr-1.2-wildvision_gpt4o_en_20240903.jsonl_extracted_sr0.5_wo_image
mmpr-1.2-wildvision_gpt4v_to_gpt4o_en_20240903.jsonl_extracted_sr0.5_wo_image
""".strip().split()


def unify_answer_format(dataset: str, question: str) -> str:
    # mmpr already has output formatting instructions in some questions, but not all
    if dataset in NEED_FORMATTING_PROMPT:
        question = question + "\n" + FORMATTING_PROMPT
    # unify format to be \boxed{...}
    if "\"Final answer: ..\"" in question:
        question = question.replace("\"Final answer: ..\"", "\"\\boxed{...}\"")
    assert (
        "\\boxed{" in question or
        dataset in [
            "mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image",  # python list
            "mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules",
            "mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_direct_rules",
            "mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules",
            "mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_direct_rules",
        ]
    ), f"question missing formatting: {question} ({dataset})"
    return question


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


def read_mmpr_samples(dataset_path, shard_id=0, num_shards=1):
    """Yield (image, question, answer) from MMPR-1.2 dataset directory."""
    dataset_path = Path(dataset_path)
    meta = json.loads((dataset_path / "meta.json").read_text(encoding="utf-8"))
    root = dataset_path.parent
    sample_idx = 0
    for subset_idx, (subset_name, subset) in enumerate(meta.items()):
        if subset_name == "dpo_hallucination":
            continue
        skip = 0
        total = 0
        subset_path = root / subset["annotation"]
        for file_idx, line in read_lines(subset_path):
            total += 1
            row = json.loads(line)
            images = row.get("image")
            if not images:
                images = []
            elif not isinstance(images, list):
                images = [images]
            images = [root / subset["root"] / i for i in images]
            for img in images:
                if not img.exists():
                    print(f"image not found: {img}")
                    skip += 1
                    continue
            images = [str(img) for img in images]
            question = row["question"]
            if "answer" in row:
                answer = row["answer"]
            elif "answer_gt" in row:
                answer = row["answer_gt"]
            elif "chosen" in row:
                # some preference data subsets are verifiable
                if subset_name in [
                    "inat_train2018_merge_en_20240811_sr0.50_wo_image",
                    "mavis_function_abs_pairs_vqa_direct_rules",
                    "geometry3k_en_20240402_extracted_pairs_vqa_direct_rules",
                    "m3cot_train_extracted_pairs_vqa_direct_rules",
                    "scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_direct_rules",
                ]:
                    answer = row["chosen"]
                else:
                    skip += 1
                    continue
            else:
                raise ValueError(f"Unknown answer type: {subset_path}:{row}")
            # shard only after filtering for verifiable samples because some subsets get skipped as a whole
            sample_idx += 1
            if (sample_idx % num_shards) != shard_id:
                continue
            sample_id = 100_000_000 * (subset_idx + 1) + sample_idx
            if sample_id not in HARD_SAMPLE_IDS:
                continue
            metadata = {
                "dataset": f"mmpr-1.2-{subset_name}",
                "source_path": str(subset_path),
                "source_index": file_idx,
                "id": 100_000_000 * (subset_idx + 1) + sample_idx,
            }
            yield images, question, answer, metadata
        if skip:
            print(f"skipped {skip}/{total} samples in {subset['annotation']}")


def build_messages(question, images=None, reasoning=False):
    """Build OpenAI chat messages with optional images."""
    messages = [{"role": "system", "content": "/think" if reasoning else "/no_think"}]
    if images:
        assert isinstance(images, list), f"images must be a list, got {type(images)}"
        content = [{"type": "text", "text": question}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": question})
    return messages


def launch_vllm_server(port):
    """Start one vLLM server on a port."""
    # https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html#running-qwen3-vl
    cmd = [
        "vllm",
        "serve",
        MODEL,
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
                logging.warning(f"Retry {retry + 1}/{retries}: Inference failed for sample {sample['id']}: {e}")
                time.sleep(2 ** retry)
                continue
            logging.exception("Error in inference task")
            raise


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


def run_inference_over_shard(client, input_path, output_path, shard_id, num_shards):
    """Run inference concurrently over one shard and write JSONL outputs."""

    def job_iter():
        """Yield (messages, sample) for each generation task."""
        for images, question, answer, metadata in read_mmpr_samples(input_path, shard_id, num_shards):
            question = unify_answer_format(metadata["dataset"], question)
            messages = build_messages(question, images=images, reasoning=True)
            sample = {
                "images": images,
                "question": question,
                "answer": answer,
                **metadata,
            }
            for _ in range(GENERATIONS_PER_PROMPT):
                yield client, messages, sample

    total_samples = 98575 // num_shards  # mmpr-1.2 hard samples
    with open(output_path, "w", buffering=1, encoding="utf-8") as f:
        with show_progress(GENERATIONS_PER_PROMPT * total_samples) as progress:
            for row, output_tokens in concurrent_map(_infer_one, job_iter()):
                if row is not None:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.update(output_tokens)


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


def read_lines(path, shard_id=0, num_shards=1):
    """Yield lines from a files based on shard ID."""
    with open(path, "r") as f:
        for sample_idx, line in enumerate(f):
            if (sample_idx % num_shards) != shard_id:
                continue
            yield sample_idx, line


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--shard-id", type=int, default=0)
@click.option("--num-shards", type=int, default=1)
def main(input_path, output_path, shard_id, num_shards):
    task_id = int(os.getenv("SLURM_ARRAY_TASK_ID") or "0")
    port = 18765 + task_id

    proc = None
    log = None
    try:
        proc, log = launch_vllm_server(port)
        # Qwen3-VL-235B-A22B-Thinking-FP8 takes ~10 mins to start
        wait_for_port("localhost", port, timeout=1200, proc=proc)
        client = openai.OpenAI(api_key="dummy", base_url=f"http://localhost:{port}/v1", timeout=300)
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
