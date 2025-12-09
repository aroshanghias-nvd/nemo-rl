"""Convert RAVEN dataset to unified JSONL format with PNG images.

srun -p cpu_interactive -A llmservice_fm_vision -N 1 \
    --job-name "nemo-rl-dev:interactive" \
    --exclusive \
    -t 04:00:00 \
    --pty \
    bash -l

uvx --with tqdm --with pillow --with numpy python scripts/prepare_raven_data.py
"""

import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

RAVEN_ROOT = Path("/lustre/fs1/portfolios/llmservice/users/jseppanen/data/RAVEN/RAVEN-10000")
OUTPUT_DIR = Path("/lustre/fsw/portfolios/llmservice/users/jseppanen/data/RAVEN/prepared")
IMAGE_SIZE = 160
BASE_PADDING = 10
BASE_DIVIDER_HEIGHT = 20
BASE_FONT_SIZE = 24

QUESTION = """Look at the 3x3 puzzle on top and the 8 answer options on bottom. Which answer option (A–H) best completes the pattern?

Please answer the question and put the final answer in this format:

Answer: \\boxed{...}."""

CONFIGS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
]


def parse_rules_from_xml(xml_path: Path) -> dict:
    """Parse the XML file and return a nested dictionary of rules."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    first_panel = root.find("Panels/Panel")
    if first_panel is None:
        return {}

    components = first_panel.findall("Struct/Component")
    component_names = {}
    for comp in components:
        comp_id = comp.get("id")
        comp_name = comp.get("name", "Unknown")
        component_names[comp_id] = comp_name

    rules_elem = root.find("Rules")
    if rules_elem is None:
        return {}

    rules_dict = {}
    for rule_group in rules_elem.findall("Rule_Group"):
        group_id = rule_group.get("id", "0")
        comp_name = component_names.get(group_id, f"Component_{group_id}")

        component_rules = {}
        for rule in rule_group.findall("Rule"):
            attr = rule.get("attr", "unknown")
            name = rule.get("name", "unknown")
            if attr == "Number/Position":
                component_rules["Number"] = name
                component_rules["Position"] = name
            else:
                component_rules[attr] = name

        if component_rules:
            rules_dict[comp_name] = component_rules

    return rules_dict


def create_composite_image(images: np.ndarray, *, seed: int = None) -> Image.Image:
    """Create a single image showing the 3x3 problem grid and 8 answer options."""
    rng = random.Random(seed) if seed is not None else random
    padding = round(2 ** rng.uniform(-1, 1) * BASE_PADDING)
    divider_height = round(2 ** rng.uniform(-1, 1) * BASE_DIVIDER_HEIGHT)
    font_size = round(2 ** rng.uniform(-1, 1) * BASE_FONT_SIZE)
    label_height = round(font_size * 30 / 24)

    grid_width = 3 * IMAGE_SIZE + 4 * padding
    grid_height = 3 * IMAGE_SIZE + 4 * padding
    answer_cols = rng.randint(2, 5)
    answer_rows = math.ceil(8 / answer_cols)
    answers_width = answer_cols * IMAGE_SIZE + (answer_cols + 1) * padding
    answers_row_height = IMAGE_SIZE + padding + label_height
    answers_height = answer_rows * answers_row_height + (answer_rows + 1) * padding

    total_width = max(grid_width, answers_width)
    total_height = grid_height + divider_height + answers_height

    composite = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    outline_tone = rng.randint(0, 255)
    outline_color = (outline_tone, outline_tone, outline_tone)
    grid_x_offset = (total_width - grid_width) // 2
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            x = grid_x_offset + padding + col * (IMAGE_SIZE + padding)
            y = padding + row * (IMAGE_SIZE + padding)

            if idx < 8:
                img = Image.fromarray(images[idx]).convert("RGB")
                composite.paste(img, (x, y))
                draw.rectangle([x - 1, y - 1, x + IMAGE_SIZE, y + IMAGE_SIZE], outline=outline_color, width=2)
            else:
                draw.rectangle([x, y, x + IMAGE_SIZE - 1, y + IMAGE_SIZE - 1], outline=outline_color, width=2)
                bbox = draw.textbbox((0, 0), "?", font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x + (IMAGE_SIZE - text_width) // 2
                text_y = y + (IMAGE_SIZE - text_height) // 2
                draw.text((text_x, text_y), "?", fill=(0, 0, 0), font=font)

    divider_y = grid_height + divider_height // 2
    divider_width = round(rng.uniform(0.0, 1.0) * (total_width - 2 * padding))
    divider_x = (total_width - divider_width) // 2
    draw.line([(divider_x, divider_y), (divider_x + divider_width, divider_y)], fill=(200, 200, 200), width=2)

    answers_x_offset = (total_width - answers_width) // 2
    answers_y_start = grid_height + divider_height
    labels = "ABCDEFGH"

    for i in range(8):
        row = i // answer_cols
        col = i % answer_cols
        x = answers_x_offset + padding + col * (IMAGE_SIZE + padding)
        y = answers_y_start + padding + row * answers_row_height

        img = Image.fromarray(images[8 + i]).convert("RGB")
        composite.paste(img, (x, y))
        draw.rectangle([x - 1, y - 1, x + IMAGE_SIZE, y + IMAGE_SIZE], outline=outline_color, width=2)

        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        label_x = x + (IMAGE_SIZE - text_width) // 2
        label_y = y + IMAGE_SIZE + 5
        draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

    scale = 2 ** rng.uniform(-1, 0)
    new_size = (round(total_width * scale), round(total_height * scale))
    resample = rng.choice([Image.Resampling.NEAREST, Image.Resampling.BOX, Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
    composite = composite.resize(new_size, resample=resample)
    return composite


def process_npz_file(args):
    """Process a single npz file and return sample dict."""
    npz_path, output_image_dir, sample_id, split = args

    data = np.load(npz_path)
    images = data["image"]
    target = int(data["target"])

    composite = create_composite_image(images, seed=sample_id)

    rel_path = npz_path.relative_to(RAVEN_ROOT)
    image_name = rel_path.with_suffix(".png").as_posix().replace("/", "_")
    image_path = output_image_dir / image_name
    composite.save(image_path)

    xml_path = npz_path.with_suffix(".xml")
    gt_rules = parse_rules_from_xml(xml_path) if xml_path.exists() else {}

    answer_letter = chr(ord("A") + target)
    config = npz_path.parent.name

    sample = {
        "dataset": f"raven-{config}",
        "images": [str(image_path)],
        "question": QUESTION,
        "answer": answer_letter,
        "gt_rules": gt_rules,
        "source_path": str(npz_path),
        "id": sample_id,
    }
    return sample, split


def collect_npz_files(split: str):
    """Find all npz files for a given split in the RAVEN dataset."""
    npz_files = []
    for config in CONFIGS:
        config_dir = RAVEN_ROOT / config
        if not config_dir.exists():
            print(f"Warning: config directory not found: {config_dir}")
            continue
        for npz_file in sorted(config_dir.glob(f"*_{split}.npz")):
            npz_files.append(npz_file)
    return npz_files


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(exist_ok=True)

    args_list = []
    current_id = 1
    for split in ["train", "val", "test"]:
        npz_files = sorted(collect_npz_files(split))
        print(f"Found {len(npz_files)} {split} npz files")
        for npz_path in npz_files:
            args_list.append((npz_path, image_dir, current_id, split))
            current_id += 1

    with ProcessPoolExecutor() as executor:
        processed = list(tqdm(
            executor.map(process_npz_file, args_list),
            total=len(args_list),
            desc="Processing RAVEN files"
        ))

    samples_by_split = {"train": [], "val": [], "test": []}
    for sample, split in processed:
        samples_by_split[split].append(sample)

    random.seed(0)
    random.shuffle(samples_by_split["train"])

    for split in ["train", "val", "test"]:
        output_path = OUTPUT_DIR / f"raven_{split}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples_by_split[split]:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"Wrote {len(samples_by_split[split])} samples to {output_path}")


if __name__ == "__main__":
    main()
