# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import torch
from einops import rearrange
from megatron.core.packed_seq_params import PackedSeqParams


def adjust_image_tokens(
    input_ids: torch.Tensor,
    num_tiles: int | list[int],
    img_start_token_id: int,
    img_end_token_id: int,
) -> torch.Tensor:
    """Ensures the input_ids tensor contains the correct number of <image> tokens as specified by num_tiles.

    This adjustment is necessary to bridge the gap between from HF processor to Megatron LLaVAModel.

    Example:
        input_ids decoded may look like this
        System: ...
        User:...
        Image 1: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[0]
        Image 2: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[1]
        ...
        etc
    Args:
        input_ids: The input_ids tensor (output of HF processor) with shape [batch, seq]
        num_tiles: The number of <image> tokens to ensure, either a single int or a list of ints
        img_start_token_id: The token id of <img>
        img_end_token_id: The token id of </img>
    Returns:
        The input_ids tensor with the correct number of <image> tokens
    """
    if isinstance(num_tiles, int):
        num_tiles = [num_tiles]

    for i, num_tile in enumerate(num_tiles):
        image_start_pos = (
            (input_ids[0] == img_start_token_id).nonzero(as_tuple=True)[0][i].item()
        )
        image_end_pos = (
            (input_ids[0] == img_end_token_id).nonzero(as_tuple=True)[0][i].item()
        )
        media_token_id = input_ids[0, image_start_pos + 1]
        existing = image_end_pos - image_start_pos + 1

        if num_tile > existing:
            repeat = num_tile + 2 - existing
            repeat_tokens = torch.full(
                (1, repeat), media_token_id, dtype=input_ids.dtype, device=input_ids.device
            )
            input_ids = torch.cat(
                [input_ids[:, :image_start_pos + 1], repeat_tokens, input_ids[:, image_start_pos + 1:]],
                dim=1,
            )
        elif num_tile < existing:
            keep_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)
            positions = (
                input_ids[0][image_start_pos:image_end_pos + 1] == media_token_id
            ).nonzero(as_tuple=True)[0] + image_start_pos
            drop_positions = positions[num_tile:].tolist()
            keep_tokens_mask[0, drop_positions] = False
            input_ids = input_ids[keep_tokens_mask].unsqueeze(0)

    return input_ids


def collapse_image_tokens_for_megatron(
    input_ids: torch.Tensor,
    img_start_token_id: int,
    img_end_token_id: int,
    pad_token_id: int = 0,
    input_lengths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Collapse expanded image tokens to single tokens per image for Megatron LLaVA.

    HF processors expand <image> to multiple tokens (one per tile), but Megatron
    expects exactly ONE <image> token per image, using num_image_tiles for tile counts.
    """
    batch_size = input_ids.shape[0]
    collapsed_list = []
    new_lengths = []

    for b in range(batch_size):
        sample_ids = input_ids[b]
        if input_lengths is not None:
            valid_len = input_lengths[b].item()
            sample_ids = sample_ids[:valid_len]

        num_images = (sample_ids == img_start_token_id).sum().item()
        num_tiles = [1] * num_images
        collapsed = adjust_image_tokens(
            sample_ids.unsqueeze(0), num_tiles, img_start_token_id, img_end_token_id
        )[0]
        collapsed_list.append(collapsed)
        new_lengths.append(collapsed.shape[0])

    max_len = max(c.shape[0] for c in collapsed_list)
    padded = torch.full(
        (batch_size, max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device
    )
    for b, collapsed in enumerate(collapsed_list):
        padded[b, :collapsed.shape[0]] = collapsed

    updated_lengths = None
    if input_lengths is not None:
        updated_lengths = torch.tensor(new_lengths, dtype=input_lengths.dtype, device=input_lengths.device)

    return padded, updated_lengths


def get_image_token_ids_from_model(model) -> Optional[tuple[int, int]]:
    """Extract <img> and </img> token IDs from Megatron model."""
    inner_model = model
    while hasattr(inner_model, 'module'):
        inner_model = inner_model.module

    has_llava = hasattr(inner_model, 'llava_model')
    if has_llava:
        inner_model = inner_model.llava_model

    img_start = getattr(inner_model, 'img_start_token_id', None)
    img_end = getattr(inner_model, 'img_end_token_id', None)
    if img_start is not None and img_end is not None:
        return img_start, img_end

    if hasattr(inner_model, 'config'):
        config = inner_model.config
        img_start = getattr(config, 'img_start_token_id', None)
        img_end = getattr(config, 'img_end_token_id', None)
        if img_start is not None and img_end is not None:
            return img_start, img_end

    raise ValueError("No image token IDs found in model")


def prepare_multimodal_tokens(data_dict: dict, model) -> None:
    """Collapse expanded image tokens in data_dict if model is a VLM. Modifies data_dict in place."""
    image_token_ids = get_image_token_ids_from_model(model)
    if image_token_ids is None or "pixel_values" not in data_dict:
        return

    img_start_id, img_end_id = image_token_ids
    collapsed_ids, collapsed_lengths = collapse_image_tokens_for_megatron(
        data_dict["input_ids"], img_start_id, img_end_id,
        input_lengths=data_dict.get("input_lengths"),
    )
    data_dict["input_ids"] = collapsed_ids
    if collapsed_lengths is not None:
        data_dict["input_lengths"] = collapsed_lengths


def process_images_for_dynamic_resolution(
    images: torch.Tensor,
    imgs_sizes: torch.Tensor,
    patch_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PackedSeqParams]:
    """Patchify images for dynamic resolution VLMs.

    For dynamic resolution, RADIO vision model expects pre-patchified images.
    This converts from [N, C, H, W] to [1, total_patches, C*patch_dim*patch_dim].

    Returns:
        patchified_images: [1, total_patches, patch_features] packed patches
        imgs_sizes: [N, 2] image sizes (H, W) in pixels (unchanged)
        num_image_tiles: [N] tile count per image (1 for pure dynamic res)
        vision_packed_seq_params: PackedSeqParams for packed attention
    """

    def rearrange_img(x):
        py = x.shape[-2] // patch_dim
        px = x.shape[-1] // patch_dim
        return rearrange(
            x, 'c (py yy) (px xx) -> (py px) (c yy xx)',
            py=py, yy=patch_dim, px=px, xx=patch_dim,
        )

    patches_list = [rearrange_img(img) for img in images]

    current_length = 0
    max_length = 0
    vision_cu_lengths = [0]
    for patch in patches_list:
        seq_len = patch.shape[0]
        if max_length < seq_len:
            max_length = seq_len
        current_length += seq_len
        vision_cu_lengths.append(current_length)

    vision_cu_lengths = torch.tensor(vision_cu_lengths, dtype=torch.int32, device=images.device)
    vision_max_lengths = torch.tensor(max_length, dtype=torch.int32, device=images.device)

    # num_image_tiles = 1 per image for pure dynamic resolution (no tiling).
    # This matches SFT semantics where num_image_tiles counts tiles, not patches.
    # The model uses imgs_sizes to derive patch grid dimensions internally.
    num_image_tiles = torch.ones(len(images), dtype=torch.int, device=images.device)

    patchified_images = torch.cat(patches_list, dim=0).unsqueeze(0)

    vision_packed_seq_params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=vision_cu_lengths,
        cu_seqlens_kv=vision_cu_lengths,
        max_seqlen_q=vision_max_lengths,
        max_seqlen_kv=vision_max_lengths,
    )

    return patchified_images, imgs_sizes, num_image_tiles, vision_packed_seq_params


def get_model_dynamic_resolution_config(model):
    """Extract dynamic resolution config from a Megatron model."""
    inner_model = model
    while hasattr(inner_model, 'module'):
        inner_model = inner_model.module

    if hasattr(inner_model, 'llava_model'):
        inner_model = inner_model.llava_model

    if hasattr(inner_model, '_dynamic_resolution') and hasattr(inner_model, 'vision_model'):
        patch_dim = getattr(inner_model.vision_model, 'patch_dim', 16)
        return inner_model._dynamic_resolution, patch_dim

    return False, None


def prepare_multimodal_data(multimodal_data: dict, model) -> None:
    """Prepare multimodal data for Megatron model forward pass.

    Handles pixel_values -> images conversion and patchification for dynamic resolution.
    Modifies multimodal_data in place.
    """
    if "pixel_values" not in multimodal_data:
        return

    images = multimodal_data.pop("pixel_values").to(torch.bfloat16)
    dynamic_resolution, patch_dim = get_model_dynamic_resolution_config(model)
    if dynamic_resolution and "imgs_sizes" in multimodal_data:
        images, _, num_image_tiles, vision_packed_seq_params = process_images_for_dynamic_resolution(
            images, multimodal_data["imgs_sizes"], patch_dim
        )
        multimodal_data["num_image_tiles"] = num_image_tiles
        multimodal_data["vision_packed_seq_params"] = vision_packed_seq_params
    multimodal_data["images"] = images
