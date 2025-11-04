# uvx --with mathruler --with sympy --with pylatexenc --with tqdm python scripts/eval_rl_math.py >incorrect.jsonl

import json
import re
import sys
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
    total = correct = malformed = 0
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
        total += 1
        if not response:
            malformed += 1
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
        correct += is_correct
    print(f"total: {total}, correct: {correct}, incorrect: {total - correct}, malformed: {malformed}", file=sys.stderr)
    print(f"accuracy: {correct / total}", file=sys.stderr)


if __name__ == "__main__":
    main()
