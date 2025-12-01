# uvx --with mathruler --with sympy --with pylatexenc --with tqdm python scripts/eval_rl_math.py >incorrect.jsonl

import json
import re
import random
import sys
from collections import defaultdict
from glob import glob

from mathruler.grader import grade_answer


path_pattern = "/lustre/fs1/portfolios/llmservice/users/jseppanen/dev/nemo-rl-n5p5-mmpr-tiny/rl_math_output_*.jsonl"

def read_jsonls(pattern):
    paths = glob(pattern)
    for path in sorted(paths):
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                yield path, idx, sample


def main():
    total = defaultdict(int)
    correct = defaultdict(int)
    malformed = defaultdict(int)
    inner_pattern = r"([^\n]+?)"
    result_pattern = rf"(?:\\\({inner_pattern}\\\)|\${inner_pattern}\$|{inner_pattern})"
    answer_patterns = [
        rf"[Tt]he answer is {result_pattern}\.?$",
        # rf"[Tt]he final answer is {result_pattern}\.?$",
        # rf"[Tt]he result is {result_pattern}\.?$",
    ]
    for path, idx, sample in read_jsonls(path_pattern):
        correct_answer = sample["answer"]
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
        total[sample["dataset"]] += 1
        if not response:
            malformed[sample["dataset"]] += 1
            # print("malformed:", output_text)
            row = {
                "source_path": sample["source_path"],
                "source_index": sample["source_index"],
            }
            print(json.dumps(row, ensure_ascii=False))
            continue
        is_correct = grade_answer(response, correct_answer)
        if not is_correct:
            # print("incorrect:", response, "!=", correct_answer)
            row = {
                "source_path": sample["source_path"],
                "source_index": sample["source_index"],
            }
            print(json.dumps(row, ensure_ascii=False))
        correct[sample["dataset"]] += is_correct
        # if is_correct and random.random() < 0.2:
        #     row = {
        #         "source_path": sample["source_path"],
        #         "source_index": sample["source_index"],
        #     }
        #     print(json.dumps(row, ensure_ascii=False))
    total["overall"] = sum(total.values())
    correct["overall"] = sum(correct.values())
    malformed["overall"] = sum(malformed.values())
    for dataset in total:
        print(
            f"{dataset}: total: {total[dataset]}, correct: {correct[dataset]}, "
            f"incorrect: {total[dataset] - correct[dataset] - malformed[dataset]}, "
            f"malformed: {malformed[dataset]}, "
            f"accuracy: {correct[dataset] / total[dataset]:.2f}",
            file=sys.stderr
        )


if __name__ == "__main__":
    main()
