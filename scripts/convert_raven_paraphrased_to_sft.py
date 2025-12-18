
import json
import os
import glob


INPUT_PATHS = sorted(
    glob.glob("/lustre/fs1/portfolios/llmservice/users/jseppanen/dev/nemo-rl-n5p5-mmpr-filtered/raven_paraphrase_output_*.jsonl") +
    glob.glob("/lustre/fs1/portfolios/llmservice/users/jseppanen/dev/nemo-rl-n5p5-mmpr-filtered/raven_paraphrase_output2__*.jsonl")
)

OUTPUT_PATH = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_jsonl/raven_paraphrased_train.jsonl"

with open(OUTPUT_PATH, "w") as fout:
    for path in INPUT_PATHS:
        with open(path, "r") as fin:
            for line in fin:
                sample = json.loads(line)
                converted = {
                    "image": os.path.basename(sample["images"][0]),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\n" + sample["question"],
                        },
                        {
                            "from": "gpt",
                            "value": sample["prediction"],
                        },
                    ],
                    "id": sample["id"],
                }
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
