# srun -A llmservice_fm_vision -p cpu_interactive -t 4:00:00 --cpus-per-task=96 --mem=165G --exclusive --pty bash -l
# uvx --with mathruler --with sympy --with pylatexenc --with tqdm python scripts/postprocess_hard_qwen.py >generations_hard_qwen_postproc.jsonl

import ast
import json
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from glob import glob

from mathruler.grader import extract_boxed_content, grade_answer
from tqdm import tqdm


path_pattern = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/generations_hard_qwen/mmpr_*.jsonl"


PYTHON_LIST_VERIFIER = """
mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image
""".strip().split()

MATH_VERIFIER = """
mmpr-1.2-CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules
mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_format_rules
mmpr-1.2-geo170k_extracted_full_pairs_vqa_correctness_rules
mmpr-1.2-geo170k_extracted_full_pairs_vqa_format_rules
mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geo170k_extracted_pairs_vqa_format_rules
mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-geomverse_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geomverse_extracted_pairs_vqa_format_rules
mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_correctness_rules
mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_format_rules
mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules
mmpr-1.2-MathV360K_prompts_pairs_vqa_correctness_rules
mmpr-1.2-MathV360K_prompts_pairs_vqa_format_rules
mmpr-1.2-mavis_function_abs_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_abs_pairs_vqa_format_rules
mmpr-1.2-mavis_function_cos_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_cos_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_cos_pairs_vqa_format_rules
mmpr-1.2-mavis_function_log_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_log_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_log_pairs_vqa_format_rules
mmpr-1.2-mavis_function_poly_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_poly_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_poly_pairs_vqa_format_rules
mmpr-1.2-mavis_function_sin_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_sin_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_sin_pairs_vqa_format_rules
mmpr-1.2-mavis_function_tan_pairs_vqa_correctness_rules
mmpr-1.2-mavis_function_tan_pairs_vqa_direct_rules
mmpr-1.2-mavis_function_tan_pairs_vqa_format_rules
mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_correctness_rules
mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_format_rules
mmpr-1.2-tallyqa_vg_en_20240816_cot_pairs_vqa_correctness_rules
mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_correctness_rules
mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_format_rules
""".strip().split()

MULTIPLE_CHOICE_VERIFIER = """
mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules
mmpr-1.2-koniq10k_en_20240403_pairs_vqa_correctness_rules
mmpr-1.2-koniq10k_en_20240403_pairs_vqa_format_rules
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules
mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_correctness_rules
mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_format_rules
mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules
""".strip().split()


def read_jsonls(pattern):
    paths = glob(pattern)
    for path in sorted(paths):
        with open(path, "r") as f:
            for idx, line in enumerate(f):
                sample = json.loads(line)
                yield path, idx, sample


def split_response(text: str) -> tuple[str, str, str]:
    if "</think>" in text:
        thinking, text = text.rsplit("</think>", 1)
        thinking = thinking + "</think>\n\n"
        text = text.strip()
    else:
        thinking = ""
    if "\\boxed{" not in text:
        match = re.search(r"\\boxed.?(\[[^]]*?\])", text)
        if match:
            text = text[:match.start()] + "\\boxed{" + match.group(1) + "}"
        else:
            raise ValueError(f"no boxed in {text}")
    explanation, answer, suffix = extract_last_boxed(text)
    explanation = remove_all_boxed(explanation).strip()
    return thinking, explanation, answer


def extract_last_boxed(text: str) -> tuple[str, str, str]:
    if "\\boxed{" not in text:
        return text, "", ""
    prefix, suffix = text.rsplit("\\boxed{", 1)
    depth = 1
    for i, char in enumerate(suffix):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            return prefix, suffix[:i], suffix[i+1:]
    # unbalanced braces
    return prefix, suffix, ""


def remove_all_boxed(text: str) -> str:
    while "\\boxed{" in text:
        prefix, boxed, suffix = extract_last_boxed(text)
        text = prefix + boxed + suffix
    return text


def verify_math(pred_answer: str, gt_answer: str) -> float:
    # normalize unicode to latex for mathruler
    gt_answer = (
        gt_answer.replace("°", "^\\circ")
        .replace("²", "^2")
        .replace("³", "^3")
        .replace("⁴", "^4")
        .replace("⁵", "^5")
        .replace("⁶", "^6")
        .replace("⁷", "^7")
        .replace("⁸", "^8")
        .replace("⁹", "^9")
        .replace("√", "\\sqrt")
        .replace("﹣", "-")
        .replace("﹢", "+")
        .replace("﹦", "=")
        .replace("﹤", "<")
        .replace("﹥", ">")
        .replace("：", ":")
        .replace("π", "\\pi")
    )
    try:
        float(pred_answer)
    except ValueError:
        pass
    else:
        gt_answer = (
            gt_answer.replace("cm2", "")
            .replace("cm", "")
            .replace("m2", "")
            .replace("m", "")
            .replace("kg", "")
            .replace("克", "")
        )
    return float(grade_answer(pred_answer, gt_answer))


def process_multiple_choice(question: str, gt_answer: str, pred_answer: str) -> str:
    choices = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pred_answer_uc = pred_answer.upper()
    gt_answer = "".join(ch for ch in gt_answer.upper() if ch in choices)
    assert len(gt_answer) == 1, f"gt_answer: {gt_answer}"
    if not pred_answer:
        return ""
    elif len(pred_answer_uc) == 1 and pred_answer_uc[0] in choices:
        return pred_answer_uc
    elif len(pred_answer_uc) > 1 and pred_answer_uc[0] in choices and pred_answer_uc[1] in (".", ",", ":", ";"):
        return pred_answer_uc[:1]
    else:
        values = re.findall(r"^([A-Z])[.:] (.*)$", question, flags=re.MULTILINE)
        for choice, value in values:
            if verify_math(pred_answer, value):
                return choice
    return pred_answer


def process(path):
    samples = []
    for _, _, sample in read_jsonls(path):
        if sample["finish_reason"] == "length":
            # truncated
            samples.append(sample)
            continue
        question = sample["question"]
        if "prediction" not in sample:
            print("missing prediction", sample["source_path"], sample["source_index"], file=sys.stderr)
            continue
        prediction = sample["prediction"]
        thinking, explanation, pred_answer = split_response(prediction)
        if sample["dataset"] == "mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image":
            # prediction = process_python_list(prediction)
            pass
        elif sample["dataset"] in MULTIPLE_CHOICE_VERIFIER:
            pred_answer = process_multiple_choice(question, sample["answer"], pred_answer)
        sample["prediction"] = thinking + explanation + "\n\\boxed{" + pred_answer + "}"
        sample["raw_prediction"] = prediction
        samples.append(sample)
    return samples


def main():
    with ProcessPoolExecutor() as executor:
        for rows in executor.map(process, sorted(glob(path_pattern))):
            for row in rows:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
