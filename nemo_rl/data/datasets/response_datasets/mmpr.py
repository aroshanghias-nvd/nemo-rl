# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import io
from typing import Any, Optional

from datasets import load_dataset
from PIL import Image

from nemo_rl.data.interfaces import TaskDataSpec


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Converts a PIL Image object to a base64 encoded string.

    Args:
        image: The PIL Image object to convert.
        format: The image format (e.g., "PNG", "JPEG"). Defaults to "PNG".

    Returns:
        A base64 encoded string representation of the image.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def format_mmpr_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the MMPR dataset into an OpenAI-API-like message log for DPO training.

    Expected MMPR format:
    {
        "image": PIL.Image or path,
        "question": str,
        "chosen_response": str,  # Preferred response
        "rejected_response": str,  # Non-preferred response
        ...
    }
    """
    if isinstance(example["image"], str):
        pil_img = Image.open(example["image"]).convert("RGB")
    elif hasattr(example["image"], "convert"):  # already PIL
        pil_img = example["image"]
    else:
        pil_img = Image.fromarray(example["image"]).convert("RGB")

    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(pil_img),
        },
        {
            "type": "text",
            "text": str(example["question"]),
        },
    ]

    # For DPO, we need both chosen and rejected responses
    # MMPR typically provides preference pairs
    chosen_content = str(example.get("chosen_response", example.get("chosen", "")))
    rejected_content = str(example.get("rejected_response", example.get("rejected", "")))

    return {
        "context": [{"role": "user", "content": user_content}],
        "completions": [
            {
                "rank": 0,
                "completion": [
                    {"role": "assistant", "content": chosen_content}
                ],
            },
            {
                "rank": 1,
                "completion": [
                    {"role": "assistant", "content": rejected_content}
                ],
            },
        ],
    }
    # ret = {
    #     "messages": [
    #         {"role": "user", "content": user_content},
    #         {
    #             "role": "assistant",
    #             "content": chosen_content,
    #             "rank": 0,  # Chosen response
    #         },
    #         {
    #             "role": "assistant",
    #             "content": rejected_content,
    #             "rank": 1,  # Rejected response
    #         },
    #     ],
    #     "task_name": "mmpr",
    # }
    # return ret


def prepare_mmpr_dataset(
    split: str = "train", task_name: Optional[str] = None
):
    """Prepare the MMPR dataset for training."""
    if task_name is None:
        task_name = "mmpr"

    try:
        # Load the MMPR dataset from HuggingFace
        print("Loading MMPR dataset from HuggingFace...")

        # Try multiple loading strategies due to dataset structure complexity
        full_dataset = None
        import json
        import os
        from datasets import Dataset
        with open(f"./MMPR-v1.1/meta.json", "r") as f:
            meta_data = json.load(f)
        
        for dataset_name, dataset_info in meta_data.items():
            if dataset_name == "ai2d_cap_gpt4o_en_20240410":
                image_root = dataset_info["root"]
                annotation_file = dataset_info["annotation"]
                dataset = []
                with open(annotation_file, "r") as f:
                    for line in f:
                        rec = json.loads(line)
                        rec["image"] = os.path.join(image_root, rec["image"])
                        dataset.append(rec)
                        
                    full_dataset = Dataset.from_list(dataset)

        train_dataset = full_dataset
        # Create a small validation set from train data
        train_size = len(train_dataset)
        val_size = min(500, train_size // 10)
        val_dataset = train_dataset.select(range(val_size))
        train_dataset = train_dataset.select(range(val_size, train_size))

        print(f"Successfully loaded MMPR dataset with {len(train_dataset)} training samples")

    except Exception as e:
        print(f"Error loading MMPR dataset: {e}")

    # Add task_name column
    train_dataset = train_dataset.add_column("task_name", [task_name] * len(train_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": train_dataset,
        "validation": val_dataset,
    }


class MMPRDataset:
    """Dataset class for MMPR (Multimodal Preference Ranking) dataset.

    This dataset contains multimodal preference pairs for DPO training.
    Each example includes an image, a question, and preferred/rejected response pairs.

    Args:
        split: The split of the dataset to use ('train', 'validation', 'test')
        prompt_file: Optional file containing custom prompts for the dataset
    """

    def __init__(
        self,
        split: str = "train",
        prompt_file: Optional[str] = None,
    ):
        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train', 'validation', or 'test'."
            )

        self.task_name = "mmpr"
        self.formatted_ds = prepare_mmpr_dataset(
            split=split, task_name=self.task_name
        )

        self.task_spec = TaskDataSpec(
            task_name="MMPR",
            prompt_file=prompt_file,
        )