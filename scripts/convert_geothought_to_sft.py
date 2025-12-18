# srun -p cpu_interactive -A llmservice_fm_vision -N 1 --job-name "nemo-rl-dev:interactive" -t 04:00:00 --pty bash -l
# uvx --with pandas --with pyarrow --with tqdm python scripts/convert_geothought_to_sft.py

import json
import os
import re
import pandas as pd
from tqdm import tqdm

INPUT_PATH = "/lustre/fs1/portfolios/llmservice/users/jseppanen/data/Geo-Thought/Geo-Thought-Augmented-10K.parquet"
OUTPUT_JSON_PATH = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_jsonl/geo_thought_augmented_10k.jsonl"
OUTPUT_IMAGE_DIR = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_data/geo_thought"

os.makedirs(OUTPUT_IMAGE_DIR + "/images", exist_ok=True)
df = pd.read_parquet(INPUT_PATH)
df["index"] = range(len(df))
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
with open(OUTPUT_JSON_PATH, "w") as fout:
    for _, sample in tqdm(df.iterrows(), total=len(df)):
        image_relpath = f"images/{sample['index']:05d}.png"
        image_path = os.path.join(OUTPUT_IMAGE_DIR, image_relpath)
        with open(image_path, "wb") as img_out:
            img_out.write(sample["images"]["bytes"])
        response = (
            sample["solution"].replace("<answer>", "").replace("</answer>", "")
            .replace("<think>", "<think>\n").replace("</think>", "\n</think>")
            .strip()
        )
        response = re.sub(r" +\n", "\n", response)
        converted = {
            "index": sample["index"],
            "image": image_relpath,
            "conversations": [
                {
                    "from": "human",
                    "value": sample["problem"],
                },
                {
                    "from": "gpt",
                    "value": response,
                },
            ],
        }
        fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
