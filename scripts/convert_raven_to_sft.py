
import json
import os
    
INPUT_PATH = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_data/RAVEN/prepared/raven_train.jsonl"
OUTPUT_PATH = "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/image_data/commercial_sft_jsonl/raven_train.jsonl"
FORMATTING_PROMPT = "Please answer the question and put the final answer in this format:\n\nAnswer: \\boxed{...}."

with open(INPUT_PATH, "r") as fin:
    with open(OUTPUT_PATH, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            prompt = "<image>\n" + sample["question"] + "\n" + FORMATTING_PROMPT
            response = sample["gt_think"] + "\n\nAnswer: \\boxed{" + sample["answer"] + "}"
            converted = {
                "image": os.path.basename(sample["images"][0]),
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt,
                    },
                    {
                        "from": "gpt",
                        "value": response,
                    },
                ],
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
