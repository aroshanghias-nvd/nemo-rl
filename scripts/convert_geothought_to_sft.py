# srun -p cpu_interactive -A llmservice_fm_vision -N 1 --job-name "nemo-rl-dev:interactive" -t 04:00:00 --pty bash -l
# uvx --with pandas --with pyarrow --with tqdm python scripts/convert_geothought_to_sft.py

import json
import os
import re
import pandas as pd
from tqdm import tqdm

INPUT_PATH = "/lustre/fs1/portfolios/llmservice/users/jseppanen/data/Geo-Thought/Geo-Thought-Augmented-10K.parquet"
OUTPUT_DIR = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/geo_thought"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "geo_thought_augmented_10k.jsonl")
OUTPUT_IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
df = pd.read_parquet(INPUT_PATH)
df["index"] = range(len(df))
df = df.sample(frac=1, random_state=0).reset_index(drop=True)
with open(OUTPUT_JSON_PATH, "w") as fout:
    for _, sample in tqdm(df.iterrows(), total=len(df)):
        image_filename = f"{sample['index']:05d}.png"
        image_path = os.path.join(OUTPUT_IMAGE_DIR, image_filename)
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
            "image": os.path.relpath(image_path, OUTPUT_DIR),
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
