import numpy as np
import random
import argparse
import json


# python create_progressive_blend.py \
#     --dataset_path /lustre/fsw/portfolios/llmservice/users/smohsenitahe/data/progressive_blends/humanlabel_ocr/humanlabel_ocr_nov4_23_dec2_plus_chart_diff_ratio.json \
#     --output_path /lustre/fsw/portfolios/llmservice/users/smohsenitahe/data/progressive_blends/humanlabel_ocr/humanlabel_ocr_progressive_blend.json \
#     --batch_size 256 \
#     --sigma 0.5 \
#     --start_pass_rate 0.8 \
#     --end_pass_rate 0

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
    K = num_steps

    bin_edges = np.linspace(end_pass_rate, start_pass_rate, K + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Pre-group indices by bin
    bins = [[] for _ in range(K)]
    for idx, p in enumerate(pass_rates):
        k = np.searchsorted(bin_edges, p, side="right") - 1
        k = np.clip(k, 0, K - 1)
        bins[k].append(idx)

    # Shuffle within each bin for randomness
    for b in bins:
        random.shuffle(b)

    # Track how many left per bin
    bin_ptr = [0] * K  # pointer into each bin list

    all_indices = []

    for t in range(num_steps):
        # Schedule mean difficulty mu_t: start easy, end hard (~0.0)
        progress = t / max(num_steps - 1, 1)
        mu_t = start_pass_rate - progress * (start_pass_rate)  # Start at start_pass_rate (easy), end at 0.0 (hard)

        print(f"Progress: {progress}, mu_t: {mu_t}")
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
        print(f"Target counts: {target_counts}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create progressive blend of dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--sigma", type=float, required=True, help="Sigma")
    parser.add_argument("--start_pass_rate", type=float, required=True, help="Start pass rate")
    parser.add_argument("--end_pass_rate", type=float, required=True, help="End pass rate")
    args = parser.parse_args()

    dataset = json.load(open(args.dataset_path))
    num_steps = len(dataset) // args.batch_size
    pass_rates = np.array([sample["pass_rate"] for sample in dataset])
    index_order = create_progressive_blend(pass_rates, args.start_pass_rate, args.end_pass_rate, args.batch_size, num_steps, args.sigma)
    output_data = []
    for idx in index_order:
        output_data.append(dataset[idx])

    json.dump(output_data, open(args.output_path, "w"))
