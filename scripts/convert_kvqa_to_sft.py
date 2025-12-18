import glob
import os
import json
import shutil
from tqdm import tqdm

INPUT_DIR = "/lustre/fs1/portfolios/llmservice/users/jseppanen/data/kvqa/extracted"
OUTPUT_PATH = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_jsonl/kvqa.jsonl"
IMAGE_OUTPUT_DIR = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_data/kvqa"

os.makedirs(IMAGE_OUTPUT_DIR + "KVQAimgs", exist_ok=True)

# please shuffle the data afterwards
with open(OUTPUT_PATH + ".tmp", "w") as fout:
    for path in tqdm(glob.glob(os.path.join(INPUT_DIR, "*.json"))):
        with open(path, "r") as fin:
            data = json.load(fin)
        image_src_path = path.replace(".json", ".img")
        image_dst_path = os.path.join(IMAGE_OUTPUT_DIR + data["image"])
        shutil.move(image_src_path, image_dst_path)
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")
