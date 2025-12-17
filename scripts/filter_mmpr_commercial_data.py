import json

ROOT = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr_1.2_commercial/MMPR-v1.2"


# https://jirasw.nvidia.com/browse/DGPTT-4039
COMMERCIAL_APPROVED = """
dpo_hallucination
mmpr_nanov2_qwen_paired_commercial
ai2d_cap_gpt4o_en
ai2d_train_12k_en
chartqa_trainval
CLEVR_math_en
cocorem_exist_yorn_en
docvqa_train_56k_en
figureqa_en
gaokao
geo170k
geometry3k_en
geomverse
geoqa+
geos_en
gqa_train_en
infographics
koniq10k_en
mapqa_suv_en
MathV360K_no_super_clevr
mavis_function
mavis_geo
nlvr2_en
okvqa_train_9k_en
spot_the_diff_en
SROIE_information_extraction
tabmwp_en
tallyqa_vg_en
textvqa_train_21k_wo_ocr_en
unigeo_calc_en
vqav2_en
vsr_en
""".strip().split()

NONCOMMERCIAL = """
dvqa_en
iconqa
inat_train2018
llavar
m3cot_train
openbmb/RLAIF-V-Dataset
RLAIF-V-Dataset
sam_cap_review_negative_en
scienceqa_multi_choice_en
super_clevr_en
wildvision
"""


# remove image folders:
# rm -rf images/dvqa images/iconqa images/inat2018 images/LLaVAR images/M3CoT images/RLAIF-V images/SA-1B images/ScienceQA images/Super-CLEVR images/wildvision

# remove annotations:
# rm annotations/dvqa_en*.jsonl annotations/iconqa*.jsonl annotations/inat_train2018*.jsonl annotations/llavar*.jsonl annotations/m3cot_train*.jsonl annotations/*RLAIF-V-Dataset*.jsonl annotations/sam_cap_review_negative_en*.jsonl annotations/scienceqa_multi_choice_en*.jsonl annotations/super_clevr_en*.jsonl annotations/wildvision*.jsonl


# process meta.json:

def check_key(key):
    if any(key.startswith(prefix) for prefix in COMMERCIAL_APPROVED):
        return True
    if any(key.startswith(prefix) for prefix in NONCOMMERCIAL):
        return False
    raise ValueError(f"Key is not known to be either commercial or non-commercial: {key}")


meta_nc = json.load(open(f"{ROOT}/meta_noncommercial.json"))
meta = {k: meta_nc[k] for k in meta_nc.keys() if check_key(k)}

with open(f"{ROOT}/meta.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=4)


# process mmpr_nanov2_qwen_paired:

BANNED_FOLDERS = "dvqa iconqa inat2018 LLaVAR M3CoT RLAIF-V SA-1B ScienceQA Super-CLEVR wildvision".split()

with open(f"{ROOT}/mmpr_nanov2_qwen_paired_noncommercial.jsonl", "r") as fin:
    with open(f"{ROOT}/mmpr_nanov2_qwen_paired_commercial.jsonl", "w") as fout:
        for line in fin:
            sample = json.loads(line)
            images = [
                path.replace("/lustre/fsw/portfolios/llmservice/users/smohsenitahe/forked_rl/nano_v2/RL/", "")
                for path in sample["image"]
            ]
            assert all(p.startswith("MMPR-v1.2/images/") for p in images), "logic error"
            if any(p.startswith(f"MMPR-v1.2/images/{folder}/") for p in images for folder in BANNED_FOLDERS):
                continue
            sample["image"] = [
                f"{ROOT.replace('/MMPR-v1.2', '')}/{p}" for p in images
            ]
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
