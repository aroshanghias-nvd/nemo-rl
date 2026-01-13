#!/usr/bin/env python3
"""Truncate Nemotron Nano v3 VL checkpoint for interactive development."""

import json
import re
import shutil
from pathlib import Path

import click
from safetensors import safe_open
from safetensors.torch import save_file


def should_keep_weight(name: str, num_layers: int, num_experts: int) -> bool:
    """Return True if weight should be kept in truncated model."""
    layer_match = re.search(r"layers\.(\d+)\.", name)
    if layer_match:
        layer_idx = int(layer_match.group(1))
        if layer_idx >= num_layers:
            return False

    expert_match = re.search(r"\.experts\.(\d+)\.", name)
    if expert_match:
        expert_idx = int(expert_match.group(1))
        if expert_idx >= num_experts:
            return False

    return True


def truncate_tensor(name: str, tensor, num_experts: int):
    """Truncate tensors that have per-expert dimensions (e.g., router weights, biases)."""
    if "gate.weight" in name and tensor.shape[0] > num_experts:
        return tensor[:num_experts, :]
    if "e_score_correction_bias" in name and tensor.shape[0] > num_experts:
        return tensor[:num_experts]
    return tensor


def truncate_config(config: dict, num_layers: int, num_experts: int) -> dict:
    """Modify config for truncated model."""
    config = json.loads(json.dumps(config))

    llm = config.get("llm_config", config)
    original_pattern = llm.get("hybrid_override_pattern", "")

    llm["num_hidden_layers"] = num_layers
    llm["n_routed_experts"] = num_experts
    if original_pattern:
        llm["hybrid_override_pattern"] = original_pattern[:num_layers]

    return config


def collect_safetensor_files(input_dir: Path) -> list[Path]:
    """Find all safetensor files in input directory."""
    index_file = input_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        return [input_dir / f for f in files]

    single_file = input_dir / "model.safetensors"
    if single_file.exists():
        return [single_file]

    return sorted(input_dir.glob("*.safetensors"))


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--num-layers", default=8, help="Number of LLM layers to keep")
@click.option("--num-experts", default=8, help="Number of experts per MoE layer to keep")
@click.option("-f", "--force", is_flag=True, help="Overwrite output directory if it exists")
def main(input_dir: Path, output_dir: Path, num_layers: int, num_experts: int, force: bool):
    if num_experts < 6:
        raise click.UsageError("Number of experts must be at least 6")

    if output_dir.exists():
        if not force:
            raise click.UsageError(f"Output directory {output_dir} exists. Use -f/--force to replace.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for src in input_dir.iterdir():
        if src.suffix == ".safetensors" or src.name == "model.safetensors.index.json":
            continue
        dst = output_dir / src.name
        if src.is_file():
            if src.name == "config.json":
                with open(src) as f:
                    config = json.load(f)
                config = truncate_config(config, num_layers, num_experts)
                with open(dst, "w") as f:
                    json.dump(config, f, indent=2)
                click.echo(f"Wrote truncated config to {dst}")
            else:
                shutil.copy2(src, dst)
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    safetensor_files = collect_safetensor_files(input_dir)
    click.echo(f"Processing {len(safetensor_files)} safetensor files...")

    kept_tensors = {}
    total_original = 0
    total_kept = 0

    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                total_original += 1
                if should_keep_weight(name, num_layers, num_experts):
                    tensor = f.get_tensor(name)
                    kept_tensors[name] = truncate_tensor(name, tensor, num_experts)
                    total_kept += 1

    click.echo(f"Kept {total_kept}/{total_original} tensors")

    output_file = output_dir / "model.safetensors"
    save_file(kept_tensors, output_file)
    click.echo(f"Saved truncated model to {output_file}")

    total_params = sum(t.numel() for t in kept_tensors.values())
    total_bytes = sum(t.numel() * t.element_size() for t in kept_tensors.values())
    click.echo(f"Total params: {total_params:,} ({total_bytes / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
