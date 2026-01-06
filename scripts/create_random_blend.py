import numpy as np
import random
import argparse
import json
from conf.dataset_path import Dataset_Path
from conf.config import *

# python create_progressive_blend.py \
#     --output_path /lustre/fsw/portfolios/llmservice/users/smohsenitahe/data/progressive_blends/humanlabel_ocr/humanlabel_ocr_progressive_blend.json \
#     --batch_size 256 \
#     --sigma 0.5 \
#     --start_pass_rate 0.8 \
#     --end_pass_rate 0


PYTHON_LIST_VERIFIER = """
""".strip().split()

MATH_VERIFIER = """
gameqa_140k
OCR_human_label
""".strip().split()

MULTIPLE_CHOICE_VERIFIER = """
""".strip().split()

epsilon = 0.001


def filter_dataset(data, upper_bound_pass_rate, lower_bound_pass_rate):
    if data['pass_rate'] > upper_bound_pass_rate:
        return False
    if data['pass_rate'] < lower_bound_pass_rate:
        return False
    return True

def write_jsonl(data, output_path):
    with open(output_path, "w") as f:
        for data in data:
            f.write(json.dumps(data) + "\n")

def get_verifier(dataset_name):
    if dataset_name in PYTHON_LIST_VERIFIER:
        return "python-list"
    if dataset_name in MATH_VERIFIER:
        return "mathruler"
    if dataset_name in MULTIPLE_CHOICE_VERIFIER:
        return "multiple-choice"
    return "string-match"

from collections import defaultdict

def build_random_samples(dataset, sample_ratio=0.8, num_buckets=5):
    buckets = defaultdict(list)

    pass_rates = np.array([sample["pass_rate"] for sample in dataset])
    min_pass_rate = pass_rates.min() - epsilon
    max_pass_rate = pass_rates.max() + epsilon

    bin_edges = np.linspace(min_pass_rate, max_pass_rate, num_buckets + 1)
    for idx in range(len(bin_edges)-1):
        for element in dataset:
            if element["pass_rate"] >= bin_edges[idx] and element["pass_rate"] < bin_edges[idx+1]:
                buckets[idx].append(element)

    sample_size = int(len(dataset) * sample_ratio)
    for idx in range(num_buckets):
        # no oversampling
        buckets[idx] = random.sample(buckets[idx], min(sample_size//num_buckets, len(buckets[idx])))

    samples = [sample for idx in range(num_buckets) for sample in buckets[idx]]
    remaining = sample_size - len(samples)

    if remaining > 0:
        #samples from the whole dataset.
        #Oversampling is allowed
        samples.extend(random.sample(dataset, remaining))

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create progressive blend of dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--upper_bound_pass_rate", default=0.8, type=float, help="Upper bound pass rate")
    parser.add_argument("--lower_bound_pass_rate", default=0.1, type=float, help="Lower bound pass rate")
    parser.add_argument("--data_blend", type=str, help="data blend from the config file")
    args = parser.parse_args()
    BLEND = eval(args.data_blend)
    all_dataset = []
    for dataset_name, dataset_path in Dataset_Path.items():
        dataset = []
        with open(dataset_path["train"], "r") as f:
            for line in f:
                data = json.loads(line)
                if "dataset" not in data.keys():
                    data["dataset"] = dataset_name
                    data["verifier"] = get_verifier(dataset_name)
                else:
                    data["verifier"] = "mathruler"
                if "pass_rate" not in data.keys():
                    if "grade" in data:
                        data["pass_rate"] = data["grade"]
                    else:
                        raise ValueError(f"grade not in data: {data}")
                if filter_dataset(data, args.upper_bound_pass_rate + epsilon, args.lower_bound_pass_rate - epsilon):
                    dataset.append(data)

        random.shuffle(dataset)
        dataset = build_random_samples(dataset, sample_ratio=BLEND[dataset_name])
        print(f"length of dataset {dataset_name}: {len(dataset)}")
        all_dataset.extend(dataset)
    
    random.shuffle(all_dataset)

    write_jsonl(all_dataset, args.output_path)
