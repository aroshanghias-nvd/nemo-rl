
import json
import glob
import random
import re
from itertools import groupby


INPUT_PATHS = sorted(
    glob.glob("/lustre/fs1/portfolios/llmservice/users/jseppanen/dev/nemo-rl-n5p5-mmpr-filtered/geometry3k_qwen_output_*.jsonl") +
    glob.glob("/lustre/fs1/portfolios/llmservice/users/jseppanen/dev/nemo-rl-n5p5-mmpr-filtered/geometry3k_qwen_output2_*.jsonl")
)

OUTPUT_PATH = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_jsonl/geometry3k_qwen_cot.jsonl.sorted"


def read_jsonls(paths):
    for path in sorted(paths):
        with open(path, "r") as f:
            for line in f:
                yield json.loads(line)


def extract_boxed_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    candidates = extract_all_boxed(text)
    if candidates:
        # take last boxed answer if many
        return candidates[-1]
    return ""


def extract_all_boxed(text: str) -> list[str]:
    if "\\boxed{" not in text:
        return []
    results = []
    # require that boxed can't be nested
    parts = text.split("\\boxed{")[1:]
    for part in parts:
        depth = 1
        for i, char in enumerate(part):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == 0:
                results.append(part[:i])
                break
        # if parens are not balanced, the answer is ignored
    return results


def verify_exact_string_match(gt_answer: str, pred_answer: str) -> float:
    return pred_answer.lower() == gt_answer.lower()

data = list(read_jsonls(INPUT_PATHS))
data.sort(key=lambda x: x["id"])

# NB. shuffle output file
with open(OUTPUT_PATH, "w") as fout:
    for sample_id, group in groupby(data, key=lambda x: x["id"]):
        group_correct = []
        for sample in group:
            pred_answer = extract_boxed_answer(sample["prediction"])
            correct = verify_exact_string_match(sample["answer"], pred_answer)
            if correct:
                group_correct.append(sample)
        if not group_correct:
            continue
        sample = random.choice(group_correct)
        converted = {
            "image": sample["images"][0].replace("/lustre/fs1/portfolios/llmservice/users/jseppanen/data/geometry3k/", ""),
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n" + sample["question"],
                },
                {
                    "from": "gpt",
                    "value": sample["prediction"],
                },
            ],
            "id": sample["id"],
        }
        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
