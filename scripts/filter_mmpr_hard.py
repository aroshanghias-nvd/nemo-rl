# filter hard samples for labeling with Qwen3-VL
# uvx --with numpy --with pandas --with tqdm python scripts/filter_mmpr_hard.py

import numpy as np
import pandas as pd
import json
import math

# run infer_mmpr.py and grade_mmpr.py first
with open("/lustre/fs1/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/rl_data/mmpr1.2_nanov2_filtered/mmpr_grading.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

np.random.seed(0)

# filter samples by accuracy
df.loc[df["verifier"] == "unanswered", "score"] = -1
ids = df.groupby(["id"]).agg({"score": "mean", "dataset": "first"}).reset_index()
ids = ids[ids["score"] < 1]
ids["score"] = ids["score"].apply(lambda x: max(-1, math.floor(5 * x)))

# sample max. 1k per dataset and difficulty (total ~100k)
def sample(limit, dataset, score):
    subset = ids[(ids["dataset"] == dataset) & (ids["score"] == score)]
    return subset.sample(n=min(limit, len(subset)))

ids = pd.concat([
    sample(1000, dataset, score)
    for dataset in ids["dataset"].unique()
    for score in ids["score"].unique()
])

# shuffle samples
ids = ids.sample(frac=1).reset_index(drop=True)
ids.to_json("mmpr_nanov2_hard_sample_ids_v1.jsonl", orient="records", lines=True)
