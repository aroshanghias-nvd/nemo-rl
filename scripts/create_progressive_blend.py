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
chart_ocr
""".strip().split()

MULTIPLE_CHOICE_VERIFIER = """
dec2
nov4
nov23
""".strip().split()

epsilon = 0.001


def merge_empty_bins(bin_edges, bins):
    """
    Merge any zero-length bins into an adjacent bin until none remain.
    bin_edges: np.array of length len(bins)+1
    bins: list of lists of indices
    Returns updated (bin_edges, bins)
    """
    while True:
        empty_idx = next((i for i, b in enumerate(bins) if len(b) == 0), None)
        if empty_idx is None:
            break  # no empty bins

        i = empty_idx
        # choose neighbor: prefer previous, otherwise next
        if i > 0:
            target = i - 1
            edge_to_drop = i          # drop the shared edge between target and i
        else:
            target = i + 1
            edge_to_drop = i + 1      # drop the upper edge of bin 0

        bins[target].extend(bins[i])  # merges (empty) into neighbor
        bins.pop(i)
        bin_edges = np.delete(bin_edges, edge_to_drop)

    return bin_edges, bins

def create_progressive_blend(pass_rates, start_pass_rate, end_pass_rate, batch_size, num_steps, sigma=0.15):
    """
    Create a progressive blend of a dataset based on pass rates.
    Args:
        pass_rates: list of pass rates
        start_pass_rate: starting pass rate for easy samples (higher pass rate)
        end_pass_rate: end pass rate for hard samples (lower pass rate)
        batch_size: batch size
        num_steps: number of steps
        sigma: sigma for the Gaussian distribution
    Returns:
        index_order: list of indices in the order of the progressive blend
    """
    N = len(pass_rates)
    seed = 42
    random.seed(seed)
    # Discretize pass rates into K bins
    K = 10

    bin_edges = np.linspace(end_pass_rate, start_pass_rate, K + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pre-group indices by bin
    bins = [[] for _ in range(K)]
    for idx, p in enumerate(pass_rates):
        k = np.searchsorted(bin_edges, p, side="right") - 1
        k = np.clip(k, 0, K - 1)
        bins[k].append(idx)

    #Update bin edges in case of zero bins
    print(f"distribution of bins before merging: {[len(bin) for bin in bins ]}")
    bin_edges, bins = merge_empty_bins(bin_edges, bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    K = len(bins)
    print(f"distribution of bins after merging: {[len(bin) for bin in bins ]}")
    # import pdb; pdb.set_trace()

    # Shuffle within each bin for randomness
    for b in bins:
        random.shuffle(b)

    # Track how many left per bin
    bin_ptr = [0] * K  # pointer into each bin list

    all_indices = []
    print(bin_centers)
    for t in range(num_steps):
        # Schedule mean difficulty mu_t: start easy, end hard (~0.0)
        progress = t / max(num_steps - 1, 1)
        mu_t = start_pass_rate - progress * (start_pass_rate - end_pass_rate)  # Start at start_pass_rate (easy), end at 0.0 (hard)
        #mu_t = end_pass_rate + progress * (start_pass_rate - end_pass_rate)  # Start hard, end easy

        # Gaussian weights over bins
        weights = np.exp(-0.5 * ((bin_centers - mu_t) / sigma) ** 2)
        if weights.sum() == 0:
            weights[:] = 1.0
        weights /= weights.sum()

        # Target counts per bin for this batch
        target_counts = np.floor(weights * batch_size).astype(int)
        # Fix rounding so total == batch_size
        diff = batch_size - target_counts.sum()
        # Add or remove 1 from bins with largest weights
        if diff != 0:
            order = np.argsort(-weights)
            for k in order:
                if diff == 0:
                    break
                target_counts[k] += np.sign(diff)
                diff -= np.sign(diff)

        batch_idxs = []
        # First pass: take up to target_counts[k] from each bin
        for k in range(K):
            need = target_counts[k]
            have = len(bins[k]) - bin_ptr[k]
            take = min(need, have)
            if take > 0:
                batch_idxs.extend(bins[k][bin_ptr[k]:bin_ptr[k]+take])
                bin_ptr[k] += take

        # If batch not full, fill from bins with most remaining
        while len(batch_idxs) < batch_size:
            remaining = [len(bins[k]) - bin_ptr[k] for k in range(K)]
            # indices of bins that still have samples
            candidate_bins = [k for k, r in enumerate(remaining) if r > 0]
            if not candidate_bins:
                break  # exhausted all samples
            # pick a random bin among those with remaining
            k = random.choice(candidate_bins)
            batch_idxs.append(bins[k][bin_ptr[k]])
            bin_ptr[k] += 1
        random.shuffle(batch_idxs)
        all_indices.extend(batch_idxs)
    return np.array(all_indices)

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
        # samples from the whole dataset.
        #Oversampling is allowed
        samples.extend(random.sample(dataset, remaining))

    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create progressive blend of dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma")
    parser.add_argument("--start_pass_rate", type=float, required=True, help="Start pass rate")
    parser.add_argument("--end_pass_rate", type=float, required=True, help="End pass rate")
    parser.add_argument("--upper_bound_pass_rate", default=0.8, type=float, help="Upper bound pass rate")
    parser.add_argument("--lower_bound_pass_rate", default=0.0, type=float, help="Lower bound pass rate")
    parser.add_argument("--data_blend", type=str, help="data blend from the config file")
    args = parser.parse_args()

    all_dataset = []
    for dataset_name, dataset_path in Dataset_Path.items():
        dataset = []
        with open(dataset_path["train"], "r") as f:
            for line in f:
                data = json.loads(line)
                if "grade" in data:
                    data["pass_rate"] = data["grade"]
                if "dataset" not in data.keys() and "batch" in data.keys():
                    data["dataset"] = "ocr_" + data["batch"]
                    data["verifier"] = get_verifier(data["batch"])
                else:
                    data["verifier"] = "mathruler"
                if filter_dataset(data, args.upper_bound_pass_rate + epsilon, args.lower_bound_pass_rate - epsilon):
                    dataset.append(data)

        random.shuffle(dataset)
        dataset = build_random_samples(dataset, sample_ratio=eval(args.data_blend)[dataset_name])
        print(f"length of dataset {dataset_name}: {len(dataset)}")
        all_dataset.extend(dataset)
    
    random.shuffle(all_dataset)
    num_steps = len(all_dataset) // args.batch_size
    pass_rates = np.array([sample["pass_rate"] for sample in all_dataset])
    print(f"length of all_dataset: {len(all_dataset)}")
    index_order = create_progressive_blend(pass_rates, args.start_pass_rate, args.end_pass_rate, args.batch_size, num_steps, args.sigma)
    output_data = []
    for idx in index_order:
        output_data.append(all_dataset[idx])

    write_jsonl(output_data, args.output_path)
