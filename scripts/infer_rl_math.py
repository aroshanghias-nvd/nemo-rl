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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import openai

# CONFIG
BASE_URL = "http://localhost:8000/v1"
MODEL = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
MAX_TOKENS = 16384
TEMPERATURE = 0.6
TOP_K = 50
TOP_P = 0.95
NUM_TILES = 12

# nano-v2 won't really follow formats that deviate from the SFT data
# PROMPT = "Answer the question and output ONLY the final answer followed by a newline."
# "Answer the question after looking at the image. You should output only a single uppercase character (A, B, C, D, ...)."
# "Reason and answer the question. Give your final answer between the <answer>...</answer> tags."
# "Solve the following question step-by-step. Output ONLY the FINAL ANSWER in this format:\n\n\\boxed{your_final_answer_here}"
PROMPT = "Solve the following question step-by-step. Write the final answer in this format:\n\nThe answer is \\(...\\)."

CLIENT = openai.OpenAI(api_key="dummy", base_url=BASE_URL.rstrip("/"))


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
            conv = row.get("conversations", [])
            question = conv[0]["value"]
            answer = conv[1]["value"]
            yield image, question, answer


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


def process(item):
    image, question, answer = item
    image_data_url = image_to_data_url(image) if image else None
    question2 = question + "\n" + PROMPT
    messages = build_messages(question2, image_data_url=image_data_url, reasoning=True)
    create_params = {
        "model": MODEL,
        "messages": messages,
        "temperature": TEMPERATURE,
        "stream": False,
    }
    extra_body = {
        "top_k": TOP_K,
        "top_p": TOP_P,
        "mm_processor_kwargs": {"max_num_tiles": NUM_TILES},
    }
    resp = CLIENT.chat.completions.create(**create_params, extra_body=extra_body)
    pred = resp.choices[0].message.content.strip() if resp and resp.choices else ""
    if "</think>" in pred and "<think>" not in pred:
        pred = "<think>\n" + pred
    out_row = {
        "image": image,
        "question": question2,
        "answer": answer,
        "prediction": pred,
    }
    return out_row


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def main(input_path, output_path):
    samples = read_samples(input_path)
    with open(output_path, "w", buffering=1, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=20) as ex:
            future_to_idx = {ex.submit(process, item): item[0] for item in samples}
            for fut in as_completed(future_to_idx):
                row = fut.result()
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
