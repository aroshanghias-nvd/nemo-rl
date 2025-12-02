# uvx --with numpy --with pandas --with tqdm python scripts/join_mmpr_nanov2_qwen.py


import pandas as pd
import json


# run grade_mmpr.py first (both nanov2 and generations_hard_qwen_postproc.jsonl)
with open("mmpr_nanov2_grading.jsonl", "r") as f:
    nanov2_df = pd.DataFrame([json.loads(line) for line in f])

with open("mmpr_hard_qwen_grading.jsonl", "r") as f:
    qwen_df = pd.DataFrame([json.loads(line) for line in f])


df = pd.merge(
    qwen_df,
    nanov2_df[[
        "id", "score", "prediction", "pred_answer", "verifier",
        "finish_reason", "prompt_tokens", "completion_tokens", "total_tokens"
    ]],
    on="id", how="inner", suffixes=("_qwen", "_nanov2")
)

# sample one example per question where Qwen is correct but Nano-v2 is wrong
paired = (
    df[(df["score_qwen"] == 1) & (df["score_nanov2"] == 0)]
    .groupby("id", as_index=False)
    .sample(n=1, random_state=0)
)

print(paired.groupby("dataset").size().reset_index(name="count").sort_values("count", ascending=False))

# row = paired[(paired["dataset"] == "mmpr-1.2-geo170k_extracted_pairs_vqa_correctness_rules") & (paired["verifier_nanov2"] == "unanswered") & (paired["finish_reason_nanov2"] != "length")].sample(1).iloc[0]; print(row["prediction_nanov2"])
# row = paired[(paired["dataset"] == "mmpr-1.2-nlvr2_en_20240910_ov_pairs_vqa_correctness_rules") & (paired["verifier_nanov2"] == "unanswered") & (paired["finish_reason_nanov2"] != "length")].sample(1).iloc[0]; print(row["prediction_nanov2"])
# row = paired[(paired["dataset"] == "mmpr-1.2-vsr_en_20240402_cot_ques_pairs_vqa_correctness_rules") & (paired["finish_reason_nanov2"] != "length")].sample(1).iloc[0]; print(row["prediction_nanov2"])

out = pd.DataFrame({
    "image": paired["images"],
    "question": paired["question"],
    "answer": paired["answer"],
    "chosen": paired["prediction_qwen"],
    "rejected": paired["prediction_nanov2"],
    "dataset": paired["dataset"],
    "id": paired["id"],
})
out = out.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle
out.to_json("mmpr_nanov2_qwen_paired.jsonl", orient="records", lines=True)
