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


def collapse_multimodal_tokens(data_dict: dict, model) -> tuple[dict, dict]:
    """Collapse N image tokens to 1 token per image for Megatron LLaVA forward pass.

    vLLM uses N tokens per image (1:1 token-to-embedding), while Megatron uses 1 token
    per image (1:N via imgs_sizes). This collapses <img><image>×N</img> to <img><image></img>.

    Returns (data_dict, metadata) where metadata contains info needed to expand back.
    """
    image_token_ids = _get_image_token_ids(model)
    if image_token_ids is None or "pixel_values" not in data_dict:
        return data_dict, {}

    input_ids = data_dict["input_ids"]
    input_lengths = data_dict.get("input_lengths")
    img_start_id, img_end_id = image_token_ids
    batch_size = input_ids.shape[0]
    original_seq_len = input_ids.shape[1]

    keep_masks = []
    collapsed_list = []
    new_lengths = []

    for b in range(batch_size):
        valid_len = input_lengths[b].item() if input_lengths is not None else input_ids.shape[1]
        sample = input_ids[b, :valid_len]

        keep_mask = torch.ones(valid_len, dtype=torch.bool, device=input_ids.device)
        for start_pos in (sample == img_start_id).nonzero(as_tuple=True)[0]:
            end_pos = (sample[start_pos:] == img_end_id).nonzero(as_tuple=True)[0][0] + start_pos
            keep_mask[start_pos + 2 : end_pos] = False

        keep_masks.append(keep_mask)
        collapsed_list.append(sample[keep_mask])
        new_lengths.append(keep_mask.sum().item())

    max_collapsed_len = max(new_lengths)
    collapsed_ids = torch.zeros(
        batch_size, max_collapsed_len, dtype=input_ids.dtype, device=input_ids.device
    )
    for b, collapsed in enumerate(collapsed_list):
        collapsed_ids[b, : len(collapsed)] = collapsed

    new_data_dict = data_dict.copy()
    new_data_dict["input_ids"] = collapsed_ids
    if input_lengths is not None:
        new_data_dict["input_lengths"] = torch.tensor(
            new_lengths, dtype=input_lengths.dtype, device=input_lengths.device
        )

    mm_metadata = {
        "original_seq_len": original_seq_len,
        "batch_size": batch_size,
        "keep_masks": keep_masks,
    }
    return new_data_dict, mm_metadata


def expand_multimodal_tokens(tensor: torch.Tensor, mm_metadata: dict) -> torch.Tensor:
    """Expand collapsed tensor back to original sequence length (inverse of collapse)."""
    if not mm_metadata:
        return tensor
    original_seq_len = mm_metadata["original_seq_len"]
    batch_size = mm_metadata["batch_size"]
    keep_masks = mm_metadata["keep_masks"]
    if tensor.shape[1] == original_seq_len:
        return tensor
    extra_dims = tensor.shape[2:] if tensor.dim() > 2 else ()
    result = torch.zeros(
        batch_size, original_seq_len, *extra_dims, dtype=tensor.dtype, device=tensor.device
    )
    for b, mask in enumerate(keep_masks):
        result[b, : len(mask)][mask] = tensor[b, : mask.sum()]
    return result


def _get_image_token_ids(model) -> Optional[tuple[int, int]]:
    """Extract <img> and </img> token IDs from Megatron model."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    for obj in [inner, getattr(inner, "config", None)]:
        if obj is None:
            continue
        start = getattr(obj, "img_start_token_id", None)
        end = getattr(obj, "img_end_token_id", None)
        if start is not None and end is not None:
            return start, end
    return None


def prepare_multimodal_data(multimodal_data: dict, model) -> None:
    """Prepare pixel_values for Megatron forward (patchification for dynamic resolution)."""
    if "pixel_values" not in multimodal_data:
        return

    images = multimodal_data.pop("pixel_values").to(torch.bfloat16)

    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    if hasattr(inner, "llava_model"):
        inner = inner.llava_model

    dynamic_res = getattr(inner, "_dynamic_resolution", False)
    if dynamic_res and "imgs_sizes" in multimodal_data:
        patch_dim = getattr(inner.vision_model, "patch_dim", 16)
        images, num_tiles, vision_params = _patchify_for_dynamic_resolution(
            images, multimodal_data["imgs_sizes"], patch_dim
        )
        multimodal_data["num_image_tiles"] = num_tiles
        multimodal_data["vision_packed_seq_params"] = vision_params

    multimodal_data["images"] = images


def _patchify_for_dynamic_resolution(
    images: torch.Tensor,
    imgs_sizes: torch.Tensor,
    patch_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, PackedSeqParams]:
    """Convert images to packed patches for dynamic resolution RADIO vision encoder."""

    def to_patches(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        img = img[:, :h, :w]
        py, px = h // patch_dim, w // patch_dim
        return rearrange(
            img, "c (py yy) (px xx) -> (py px) (c yy xx)", py=py, yy=patch_dim, px=px, xx=patch_dim
        )

    patches_list = [to_patches(img, *imgs_sizes[i].tolist()) for i, img in enumerate(images)]

    cu_seqlens = [0]
    for p in patches_list:
        cu_seqlens.append(cu_seqlens[-1] + p.shape[0])

    max_seqlen = max(p.shape[0] for p in patches_list)
    return (
        torch.cat(patches_list, dim=0).unsqueeze(0),
        torch.ones(len(images), dtype=torch.int, device=images.device),
        PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=images.device),
            cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32, device=images.device),
            max_seqlen_q=torch.tensor(max_seqlen, dtype=torch.int32, device=images.device),
            max_seqlen_kv=torch.tensor(max_seqlen, dtype=torch.int32, device=images.device),
        ),
    )
