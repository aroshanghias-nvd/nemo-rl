# srun -A llmservice_fm_vision -p cpu_interactive -t 4:00:00 --cpus-per-task=96 --mem=165G --exclusive --pty bash -l
# uvx --with mathruler --with sympy --with pylatexenc --with tqdm python scripts/grade_mmpr.py >mmpr_grading.jsonl

import ast
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob

from mathruler.grader import extract_boxed_content, grade_answer
from tqdm import tqdm


path_pattern = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/generations/mmpr_output_*.jsonl"

# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_pairs_vqa_format_rules
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-cocorem_exist_yorn_en_20241016_pairs_vqa_correctness_rules
# mmpr-1.2-cocorem_exist_yorn_en_20241016_pairs_vqa_format_rules
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_correctness_rules
# mmpr-1.2-dvqa_en_20240402_extracted_int_only_pairs_vqa_format_rules
# mmpr-1.2-figureqa_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-figureqa_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-geo170k_extracted_full_pairs_vqa_correctness_rules
# mmpr-1.2-geo170k_extracted_full_pairs_vqa_format_rules
# mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geo170k_extracted_pairs_vqa_format_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_direct_rules
# mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-geomverse_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geomverse_extracted_pairs_vqa_format_rules
# mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-geoqa+_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_correctness_rules
# mmpr-1.2-geoqa+_extracted_en_version_pairs_vqa_format_rules
# mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-geos_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-geos_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-iconqa_train_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-iconqa_train_extracted_pairs_vqa_format_rules
# mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_pairs_vqa_format_rules
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-koniq10k_en_20240403_pairs_vqa_correctness_rules
# mmpr-1.2-koniq10k_en_20240403_pairs_vqa_format_rules
# mmpr-1.2-m3cot_train_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules
# mmpr-1.2-m3cot_train_extracted_pairs_vqa_format_rules
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-mapqa_suv_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-MathV360K_prompts_pairs_vqa_correctness_rules
# mmpr-1.2-MathV360K_prompts_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_abs_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules
# mmpr-1.2-mavis_function_abs_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_cos_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_cos_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_log_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_log_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_poly_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_poly_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_sin_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_sin_pairs_vqa_format_rules
# mmpr-1.2-mavis_function_tan_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_function_tan_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth0_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth0_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth1_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth1_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth2_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth2_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-mavis_geo_depth3_text_dominant_vision_dominant_en_pairs_vqa_correctness_rules
# mmpr-1.2-mavis_geo_depth3_text_dominant_vision_dominant_en_pairs_vqa_format_rules
# mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_correctness_rules
# mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_format_rules
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_direct_rules
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_format_rules
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_pairs_vqa_correctness_rules
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_pairs_vqa_format_rules
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_correctness_rules
# mmpr-1.2-super_clevr_en_20240402_int_pairs_vqa_format_rules
# mmpr-1.2-super_clevr_en_20240402_yorn_pairs_vqa_correctness_rules
# mmpr-1.2-super_clevr_en_20240402_yorn_pairs_vqa_format_rules
# mmpr-1.2-tabmwp_en_20240402_cot_pairs_vqa_correctness_rules
# mmpr-1.2-tallyqa_vg_en_20240816_cot_pairs_vqa_correctness_rules
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_correctness_rules
# mmpr-1.2-unigeo_calc_en_20240402_extracted_open_ended_only_pairs_vqa_format_rules
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_correctness_rules
# mmpr-1.2-vqav2_en_20240402_int_pairs_vqa_format_rules
# mmpr-1.2-vsr_en_20240402_cot_ques_pairs_vqa_correctness_rules

# EXACT_VERIFIER = """
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-ai2d_train_12k_en_20240410_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-chartqa_trainval_30k_w_csv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-CLEVR_math_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-docvqa_train_56k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-figureqa_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-SROIE_information_extraction_multi_turn_20240620_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-textvqa_train_21k_wo_ocr_en_20240611_extracted_prefix_pair_sr0.5_wo_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.0_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_with_image
# mmpr-1.2-vqav2_en_20240402_extracted_prefix_pair_sr0.5_wo_image
# """.strip().split()

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


def extract_boxed_answer(text: str) -> str:
    if text.count("\\boxed{") != 1:
        return ""
    return extract_boxed_content(text)


def extract_final_answer(text: str) -> str:
    matches = list(re.finditer(r"^Final answer: *(.*)\.?$", text, flags=re.MULTILINE))
    if matches:
        answers = [match.group(1).strip() for match in matches]
        if all(ans == answers[0] for ans in answers):
            return answers[0]
    return ""


def extract_python_list(text: str) -> str:
    try:
        text = text.replace("```python\n", "").replace("```", "")
        return repr(ast.literal_eval(text))
    except Exception:
        return ""


def verify_math(pred_answer: str, gt_answer: str) -> float:
    # heuristics
    gt_answer = gt_answer.replace("°", "^\\circ")
    return float(grade_answer(pred_answer, gt_answer))


def verify_multiple_choice(pred_answer: str, gt_answer: str) -> float:
    pred_answer = pred_answer.upper()
    gt_answer = "".join(ch for ch in gt_answer.upper() if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert len(gt_answer) == 1, f"gt_answer: {gt_answer}"
    if len(pred_answer) > 1:
        if pred_answer[1] in (".", ",", ":", ";"):
            pred_answer = pred_answer[:1]
        else:
            return 0
    return int(pred_answer[0] == gt_answer[0])


def verify_python_list(pred_answer: str, gt_answer: str) -> float:
    try:
        pred_list = ast.literal_eval(pred_answer)
        gt_list = ast.literal_eval(gt_answer)
        correct = sum(pred == gt for pred, gt in zip(pred_list, gt_list))
        return correct / max(len(pred_list), len(gt_list))
    except Exception:
        return 0.0


def verify_string_match(pred_answer: str, gt_answer: str) -> float:
    pred_answer = pred_answer.lower()
    gt_answer = gt_answer.lower()
    if _normalize_numbers(pred_answer) == _normalize_numbers(gt_answer):
        return 1.0
    elif _normalize_lists(pred_answer) == _normalize_lists(gt_answer):
        return 1.0
    elif _normalize_states(pred_answer) == _normalize_states(gt_answer):
        return 1.0
    else:
        return 0.0


def _normalize_numbers(text: str) -> str:
    text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
    text = text.replace("\\%", "%")
    return text


def _normalize_lists(text: str) -> str:
    text = text.replace(",", " ").replace(";", " ")
    text = text.replace("and", " ").replace("or", " ")
    text = " ".join(text.split())
    return text


def _normalize_states(text: str) -> str:
    text = (
        text
        .replace("alabama", "AL")
        .replace("alaska", "AK")
        .replace("arizona", "AZ")
        .replace("arkansas", "AR")
        .replace("california", "CA")
        .replace("colorado", "CO")
        .replace("connecticut", "CT")
        .replace("delaware", "DE")
        .replace("district of columbia", "DC")
        .replace("florida", "FL")
        .replace("georgia", "GA")
        .replace("hawaii", "HI")
        .replace("idaho", "ID")
        .replace("illinois", "IL")
        .replace("indiana", "IN")
        .replace("iowa", "IA")
        .replace("kansas", "KS")
        .replace("kentucky", "KY")
        .replace("louisiana", "LA")
        .replace("maine", "ME")
        .replace("maryland", "MD")
        .replace("massachusetts", "MA")
        .replace("michigan", "MI")
        .replace("minnesota", "MN")
        .replace("mississippi", "MS")
        .replace("missouri", "MO")
        .replace("montana", "MT")
        .replace("nebraska", "NE")
        .replace("nevada", "NV")
        .replace("new hampshire", "NH")
        .replace("new jersey", "NJ")
        .replace("new mexico", "NM")
        .replace("new york", "NY")
        .replace("north carolina", "NC")
        .replace("north dakota", "ND")
        .replace("ohio", "OH")
        .replace("oklahoma", "OK")
        .replace("oregon", "OR")
        .replace("pennsylvania", "PA")
        .replace("rhode island", "RI")
        .replace("south carolina", "SC")
        .replace("south dakota", "SD")
        .replace("tennessee", "TN")
        .replace("texas", "TX")
        .replace("utah", "UT")
        .replace("vermont", "VT")
        .replace("virginia", "VA")
        .replace("washington", "WA")
        .replace("west virginia", "WV")
        .replace("wisconsin", "WI")
        .replace("wyoming", "WY")
    )
    return _normalize_lists(text)


def process(path):
    rows = []
    for _, _, sample in read_jsonls(path):
        question = sample["question"]
        if "prediction" not in sample:
            print("missing prediction", sample["source_path"], sample["source_index"], file=sys.stderr)
            continue
        prediction = re.sub(r"<think>.*</think>", "", sample["prediction"], flags=re.DOTALL).strip()
        if "\\boxed{" in question:
            pred_answer = extract_boxed_answer(prediction)
        elif "\"Final answer: ..\"" in question:
            pred_answer = extract_final_answer(prediction)
        elif sample["dataset"] == "mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image":
            pred_answer = extract_python_list(prediction)
        elif sample["dataset"] in [
            "mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules",
            "mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_direct_rules",
            "mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules",
            "mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_direct_rules",
        ]:
            pred_answer = prediction  # string match
        else:
            raise ValueError(f"unknown answer format: {sample['dataset']}: {question}")

        gt_answer = sample["answer"]
        row = {
            "id": sample["id"],
            "dataset": sample["dataset"],
            "truncated": sample["finish_reason"] == "length",
            "answered": bool(pred_answer),
        }
        try:
            if not pred_answer:
                row["score"] = 0
                row["verifier"] = "unanswered"
            elif sample["dataset"] in MATH_VERIFIER:
                row["score"] = verify_math(pred_answer, gt_answer)
                row["verifier"] = "mathruler"
            elif sample["dataset"] in MULTIPLE_CHOICE_VERIFIER:
                row["score"] = verify_multiple_choice(pred_answer, gt_answer)
                row["verifier"] = "multiple-choice"
            elif sample["dataset"] in PYTHON_LIST_VERIFIER:
                row["score"] = verify_python_list(pred_answer, gt_answer)
                row["verifier"] = "python-list"
            else:
                # case insensitive string match
                row["score"] = verify_string_match(pred_answer, gt_answer)
                row["verifier"] = "string-match"
        except Exception as e:
            raise ValueError(f"verification failed for {sample['dataset']}: {question} -> {pred_answer} -> {gt_answer}") from e
        rows.append(row)
    return rows


def main():
    with ProcessPoolExecutor() as executor:
        for rows in executor.map(process, sorted(glob(path_pattern))):
            for row in rows:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
