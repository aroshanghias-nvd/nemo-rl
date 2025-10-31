import json
import sys
import re
import random

root = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data"


prompt = "Answer the question and output ONLY the final answer followed by a newline."
# "Answer the question after looking at the image. You should output only a single uppercase character (A, B, C, D, ...)."
# "Reason and answer the question. Give your final answer between the <answer> and </answer> tags."
# "Solve the following question step-by-step. Output ONLY the FINAL ANSWER in this format:\n\n\\boxed{your_final_answer_here}"


def read_jsonl(path):
    with open(path, "r") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            if len(sample["conversations"]) > 2:
                print(f"[{idx}] malformed sample: {json.dumps(sample, indent=2)}", file=sys.stderr)
                continue
            assert sample["conversations"][0]["from"] == "human"
            assert sample["conversations"][1]["from"] == "gpt"
            image = sample.get("image")
            input_text = sample["conversations"][0]["value"]
            output_text = sample["conversations"][1]["value"]
            input_text = input_text.replace("<image>", "").strip()
            yield idx, sample, image, input_text, output_text


def format_mulberry():
    path = f"{root}/sft_jsonl/mulberry_sft/vision_r1_mulberry_sft_full_nmh5r_legal_cleanedup_v2.jsonl"
    subsets = [
        "CLEVR-Math",
        "geo3k",
        "geoqa_plus",
        "GEOS",
        "UniGeo",
        "mathvision",
    ]
    old_prompt = 'Return the final answer as "Final Answer:option".'
    answer_patterns = [
        r"Final Answer:(-?\d+(?:\.\d+)?)\.?$",
        r"Final Answer:([-\d\w\\:,. ]+?)\.?$",
        r"Final Answer:([\d\w]+?)\. [^\n]*$",
        r"Final Answer:([\d\w ]+?) \([^\n]*\)$",
        r"Final Answer:[^=]+ = (\d+)\.?$",
        r"Final Answer:(\w+?)\.\nAnswer:\1$",
        r"Final Answer:option (\w+?)\.?$",
        r"Final Answer:option (\w+?)\.\nAnswer:\1$",
        r"Final Answer:(No|Yes), [^\n]*\.?$",
    ]
    for idx, sample, image, input_text, output_text in read_jsonl(path):
        subset = re.search(r"mulberry_images/([^/]+)", image)
        if not subset:
            print(f"[{idx}] malformed image path: {image}", file=sys.stderr)
            continue
        subset = subset.group(1)
        if subset not in subsets:
            continue
        assert input_text.endswith(old_prompt), f"[{idx}] malformed question: {sample}"
        question = input_text.replace(old_prompt, prompt)
        answer = None
        for answer_pattern in answer_patterns:
            answer = re.search(answer_pattern, output_text, flags=re.DOTALL)
            if answer:
                answer = answer.group(1).strip()
                break
        assert answer, f"[{idx}] malformed answer: {sample}"
        row = {
            "image": image,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
        }
        row["source_path"] = path
        row["source_index"] = idx
        yield row


def format_geomverse():
    # path = f"{root}/sft_jsonl/internvl_cot/geomverse_en_aug_nmh5r.jsonl"  # geomverse_cot
    path = f"{root}/sft_jsonl/format4/cauldron_geomverse_base.jsonl"
    answer_patterns = [
        r"The answer is ([^\n]+)\.$",
        r"Therefore the final answer is ([^\n]+)\.$",
    ]
    for idx, sample, image, input_text, output_text in read_jsonl(path):
        question = input_text + "\n" + prompt
        answer = None
        for answer_pattern in answer_patterns:
            answer = re.search(answer_pattern, output_text, flags=re.DOTALL)
            if answer:
                answer = answer.group(1).strip()
                break
        assert answer, f"[{idx}] malformed answer: {sample}"
        row = {
            "image": image,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
        }
        row["source_path"] = path
        row["source_index"] = idx
        yield row


def format_metamathqa():
    # path = f"{root}/sft_jsonl/internvl_cot/metamathqa_en_nmh5r_clean.jsonl"
    path = f"{root}/sft_jsonl/internvl_cot/metamathqa_en.jsonl"
    inner_pattern = r"([^\n]+?)"
    result_pattern = rf"{inner_pattern}|\\\({inner_pattern}\\\)|\${inner_pattern}\$"
    answer_patterns = [
        rf"he answer is:? {result_pattern}\.?$",
        rf"he final answer is:? {result_pattern}\.?$",
        rf"he result is:? {result_pattern}\.?$",
        rf" earned {result_pattern} dollars\.$",
        rf" has {result_pattern} eggs\.$",
        rf"Thus, the [^.\n]* is {result_pattern}\.$",
        rf"So, the [^.\n]* is {result_pattern}\.$",
        rf"Thus, [^.\n]* received {result_pattern} dandelion puffs\.?$",
        rf"Thus, [^.\n]* owns {result_pattern} hoodies\.$",
        rf"Thus, [^.\n]* are {result_pattern}\.$",
        rf"Thus, [^.\n]* contribute {result_pattern}\.$",
        rf"Thus, [^.\n]* sell {result_pattern} candy bars\.$",
        rf"So, [^.\n]* sell {result_pattern} candy bars\.$",
        rf"So, [^.\n]* receive {result_pattern} balloons\.$",
        rf"So, [^.\n]* earn {result_pattern} in a year\.$",
        rf"Thus, the number that does not round to 65\.14 is [Oo]ption {result_pattern}\.$",
        rf"\nAnswer: {result_pattern}\.?$",
        rf"\n\*\*Answer:\*\* {result_pattern}\.?$",
        r"\n(\d+(?:\.\d+)?)$",
    ]
    for idx, sample, image, input_text, output_text in read_jsonl(path):
        assert input_text.endswith("Solve the math problem in the image.")
        question = "Solve the math problem in the image.\n" + prompt
        answer = None
        for answer_pattern in answer_patterns:
            answer = re.search(answer_pattern, output_text, flags=re.DOTALL)
            if not answer:
                continue
            answer = next(g for g in answer.groups() if g is not None)
            if not answer:
                continue
            answer = answer.strip()
            break
        if not answer:
            print(f"[{idx}] skipping malformed answer: {output_text[-70:].strip()}", file=sys.stderr)
            continue
        row = {
            "image": image,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ],
        }
        row["source_path"] = path
        row["source_index"] = idx
        yield row


def format_educhat_math():
    # path = f"{root}/internvl_data/image_data/educhat_math/cmm_math_cot_zh_nmh5r.jsonl"
    path = f"{root}/internvl_data/image_data/educhat_math/cmm_math_cot_zh.jsonl"
    old_prompt = "当你准备好给出答案时，请使用以下格式：\"答案: ...\""
    inner_pattern = r"([^\n]+?)"
    result_pattern = rf"{inner_pattern}|\\\({inner_pattern}\\\)|\${inner_pattern}\$"
    answer_patterns = [
        rf"答案[:：]?\s*{result_pattern}[.。]?$",
        rf"答案为[:：]?\s*{result_pattern}[.。]?$",
        rf"答案是[:：]?\s*{result_pattern}[.。]?$",
        rf"故答案为[:：]?\s*{result_pattern}[.。]?$",
        rf"\n\*\*答案[:：]?\s*{result_pattern}\*\*[.。]?$",
        rf"\n\*\*答案\*\*[:：]?\s*{result_pattern}[.。]?$",
        rf"\n\*\*答案:\*\*\s*{result_pattern}[.。]?$",
        rf"\n##* 答案[:：]?\s*{result_pattern}[.。]?$",
    ]
    for idx, sample, image, input_text, output_text in read_jsonl(path):
        assert old_prompt in input_text
        question = input_text.replace(old_prompt, "") + "\n" + prompt
        answer = None
        for answer_pattern in answer_patterns:
            answer = re.search(answer_pattern, output_text, flags=re.DOTALL)
            if not answer:
                continue
            answer = next(g for g in answer.groups() if g is not None)
            if not answer:
                continue
            answer = answer.strip()
            break
        if not answer:
            print(f"[{idx}] skipping malformed answer: {output_text[-70:].strip()}", file=sys.stderr)
            continue
        row = {"image": image} if image else {}
        row["conversations"] = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
        row["source_path"] = path
        row["source_index"] = idx
        yield row

def main():
    rows = []
    rows.extend(format_mulberry())
    rows.extend(format_geomverse())
    rows.extend(format_metamathqa())
    rows.extend(format_educhat_math())
    random.seed(0)
    random.shuffle(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))

if __name__ == "__main__":
    main()
