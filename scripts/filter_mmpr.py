# uvx --with numpy --with pandas --with tqdm python scripts/filter_mmpr.py

import numpy as np
import pandas as pd
import json
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

path_pattern = "/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/generations/mmpr_output_*.jsonl"

# run grade_mmpr.py first
with open("/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/mmpr_grading.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# save dataset of samples with 20–80% correct
# 1. sort each dataset by increasing difficulty
# 2. break ties randomly
# 3. merge datasets randomly

np.random.seed(0)

# filter samples by accuracy
df["verifier"] = df["verifier"].apply(lambda x: "" if x == "unanswered" else x)
train_ids = df.groupby(["id"]).agg({"score": "mean", "dataset": "first", "verifier": "max"}).reset_index()
train_ids = train_ids[(train_ids["score"] > 0) & (train_ids["score"] < 1)]

# sort by difficulty (with random tie-breaker)
train_ids["random"] = np.random.rand(len(train_ids))
train_ids.sort_values(by=["score", "random"], ascending=False, inplace=True)

# shuffle samples
train_ids2 = train_ids.copy()
train_ids2["random"] = np.random.rand(len(train_ids2))
train_ids2.sort_values(by="random", inplace=True)

# merge samples by difficulty level within dataset and random order across datasets
for dataset in train_ids["dataset"].unique():
    train_ids2.loc[train_ids2["dataset"] == dataset] = train_ids[train_ids["dataset"] == dataset]

train_ids2[["id"]].to_csv("mmpr_train_ids.csv", index=False, header=False)
ids = set(train_ids2["id"].tolist())

def filter_data(path):
    filtered_data = {}
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            sample_id = sample["id"]
            if sample_id in ids and sample_id not in filtered_data:
                if "prediction" in sample:
                    del sample["prediction"]
                sample["grade"] = train_ids2[train_ids2["id"] == sample_id]["score"].values[0]
                sample["verifier"] = train_ids2[train_ids2["id"] == sample_id]["verifier"].values[0]
                filtered_data[sample_id] = sample
    return filtered_data

filtered_data = {}
with ProcessPoolExecutor() as executor:
    for shard_data in executor.map(filter_data, sorted(glob(path_pattern))):
        filtered_data.update(shard_data)

with open("mmpr_nanov2_filtered_v1.jsonl", "w") as f:
    for id in tqdm(train_ids2["id"].tolist(), desc="writing"):
        f.write(json.dumps(filtered_data[id], ensure_ascii=False) + "\n")
