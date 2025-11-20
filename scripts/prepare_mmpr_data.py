"""Preprocess MMPR-1.2 dataset into unified format for RL training with verifiable rewards."""

"""
srun -p cpu_interactive -A llmservice_fm_vision -N 1 \
    --job-name "nemo-rl-dev:interactive" \
    -t 04:00:00 \
    --pty \
    bash -l

uvx --with tqdm python scripts/prepare_mmpr_data.py
"""

import json
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

ROOT = Path("/lustre/fsw/portfolios/llmservice/users/smohsenitahe/forked_rl/nano_v2/RL/MMPR-v1.2")

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
APPEND_FORMATTING_PROMPT = """
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
mmpr-1.2-geometry3k_en_20240402_extracted_pairs_vqa_direct_rules
mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-gqa_train_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-inat_train2018_merge_en_20240811_sr0.50_wo_image
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-infographics_20240403_qa_20240407_v2_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-m3cot_train_extracted_pairs_vqa_direct_rules
mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-m3cot_train_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-mapqa_suv_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-mavis_function_abs_pairs_vqa_direct_rules
mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.0_with_image
mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_with_image
mmpr-1.2-okvqa_train_9k_en_20240402_extracted_prefix_pair_sr0.5_wo_image
mmpr-1.2-scienceqa_multi_choice_en_20240402_extracted_pairs_vqa_direct_rules
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
    if dataset in APPEND_FORMATTING_PROMPT:
        question = question + "\n" + FORMATTING_PROMPT
    # unify format to be \boxed{...}
    if "\"Final answer: ..\"" in question:
        question = question.replace("\"Final answer: ..\"", "\"\\boxed{...}\"")
    assert "\\boxed{" in question, f"question missing formatting: {question} ({dataset})"
    return question


def prepare_mmpr_samples(args):
    subset_idx, (name, subset) = args
    if name == "dpo_hallucination":
        return []
    path = ROOT.parent / subset["annotation"]
    skip = 0
    total = 0
    samples = []
    with open(path) as f:
        for line_num, line in enumerate(f):
            total += 1
            row = json.loads(line)
            images = row.get("image")
            if not images:
                images = []
            elif not isinstance(images, list):
                images = [images]
            images = [ROOT.parent / subset["root"] / i for i in images]
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
                if name in [
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
                raise ValueError(f"Unknown answer type: {path}:{row}")
            question = unify_answer_format(f"mmpr-1.2-{name}", question)
            sample = {
                "dataset": f"mmpr-1.2-{name}",
                "images": images,
                "question": question,
                "answer": answer,
                "source_path": str(path),
                "source_index": line_num,
                "subset_idx": subset_idx,
            }
            samples.append(sample)
    if skip:
        print(f"skipped {skip}/{total} samples in {subset['annotation']}")
    return samples


meta = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))

with ProcessPoolExecutor() as executor:
    results = executor.map(prepare_mmpr_samples, enumerate(meta.items()))
    samples = [
        sample
        for shard_samples in tqdm(results, total=len(meta))
        for sample in shard_samples
    ]

samples = [
    dict(
        sample,
        id=100_000_000 * (sample.pop("subset_idx") + 1) + (idx + 1),
    )
    for idx, sample in enumerate(samples)
]

random.seed(0)
random.shuffle(samples)

with open("mmpr_1_2_verifiable_1120.jsonl", "w", buffering=1, encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
