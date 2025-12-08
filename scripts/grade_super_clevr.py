# srun -A llmservice_fm_vision -p cpu_interactive -t 4:00:00 --cpus-per-task=96 --mem=165G --exclusive --pty bash -l
# uvx python scripts/grade_super_clevr.py >super_clevr_grading.jsonl

import json
import re
from concurrent.futures import ProcessPoolExecutor
from glob import glob


path_pattern = "/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/jseppanen/dev/nemo-rl-n5p5-mmpr-filtered/super_clevr_output_*.jsonl"


def read_jsonls(pattern):
    paths = glob(pattern)
    for path in sorted(paths):
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                yield path, idx, sample


def extract_boxed_answer(text: str) -> str:
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


def verify_fuzzy_string_match(pred_answer: str, gt_answer: str) -> float:
    pred_answer = pred_answer.lower()
    gt_answer = gt_answer.lower()
    if _normalize_punctuation(pred_answer) == _normalize_punctuation(gt_answer):
        return 1.0
    elif _normalize_numbers(pred_answer) == _normalize_numbers(gt_answer):
        return 1.0
    elif _normalize_synonyms(pred_answer) == _normalize_synonyms(gt_answer):
        return 1.0
    else:
        return 0.0


def _normalize_punctuation(text: str) -> str:
    norm_text = text.rstrip(".!?")
    return norm_text or text


def _normalize_numbers(text: str) -> str:
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    text = text.replace("\\%", "%")
    return text


def _normalize_synonyms(text: str) -> str:
    if text == "metallic":
        return "metal"
    elif text == "wooden":
        return "wood"
    elif text == "fighter jet":
        return "fighter"
    elif text == "double-decker bus":
        return "double bus"
    elif text == "regular bus":
        return "bus"
    elif text == "minivan":
        return "van"
    elif text == "airliner":
        return "airplane"
    elif text == "tandem bike":
        return "tandem bicycle"
    elif text == "large":
        return "big"
    elif text == "pickup truck":
        return "truck"
    elif text == "dirtbike":
        return "dirt bike"
    else:
        return text


def process(path):
    rows = []
    for _, _, sample in read_jsonls(path):
        gt_answer = sample["answer"]
        prediction = re.sub(r"<think>.*?</think>", "", sample["prediction"], flags=re.DOTALL).strip()
        pred_answer = extract_boxed_answer(prediction)
        sample["pred_answer"] = pred_answer
        sample["score"] = verify_fuzzy_string_match(pred_answer, gt_answer)
        rows.append(sample)
    return rows


def main():
    with ProcessPoolExecutor() as executor:
        for rows in executor.map(process, sorted(glob(path_pattern))):
            for row in rows:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
