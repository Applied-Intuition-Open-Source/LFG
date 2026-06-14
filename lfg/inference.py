# Copyright 2026 Applied Intuition, Inc.
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

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .io import Frame, preprocess_frames
from .visualization import save_visualizations


DEFAULT_OUTPUT_KEYS = (
    "points",
    "local_points",
    "conf",
    "camera_poses",
    "segmentation",
    "motion",
    "flow",
)


def _finite_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def predict_window(
    model: torch.nn.Module,
    frames: list[Frame],
    config: ModelConfig,
    *,
    device: str | torch.device,
    target_size: int,
    resize_mode: str,
    keep_ratio: bool,
) -> dict[str, Any]:
    images = preprocess_frames(
        frames,
        target_size=target_size,
        mode=resize_mode,
        keep_ratio=keep_ratio,
        patch_size=14,
    )
    batch = images.unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(batch)
    outputs["n_current_frames"] = config.m
    outputs["n_future_frames"] = config.n
    return outputs


def predictions_to_numpy(
    predictions: dict[str, Any],
    *,
    save_features: bool = False,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    allowed = set(DEFAULT_OUTPUT_KEYS)
    for key, value in predictions.items():
        if not save_features and key not in allowed:
            continue
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.shape[:1] == (1,):
                tensor = tensor.squeeze(0)
            arrays[key] = tensor.numpy()
        elif isinstance(value, (int, float, np.number)):
            arrays[key] = np.asarray(value)
    return arrays


def window_metadata(
    *,
    window_index: int,
    start_index: int,
    frames: list[Frame],
    padded: list[bool],
    config: ModelConfig,
) -> dict[str, Any]:
    input_frames = []
    for slot, (frame, was_padded) in enumerate(zip(frames, padded)):
        input_frames.append(
            {
                "slot": slot,
                "source": frame.source,
                "frame_index": frame.frame_index,
                "timestamp_sec": _finite_float_or_none(frame.timestamp_sec),
                "padded": was_padded,
            }
        )

    predicted_slots = []
    for slot in range(config.m + config.n):
        predicted_slots.append(
            {
                "slot": slot,
                "kind": "current" if slot < config.m else "future",
                "nominal_source_index": start_index + slot,
            }
        )

    return {
        "window_index": window_index,
        "start_index": start_index,
        "input_frames": input_frames,
        "predicted_slots": predicted_slots,
    }


def save_window_result(
    output_dir: Path,
    *,
    window_index: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_npz: bool = True,
    save_viz: bool = True,
) -> Path:
    window_dir = output_dir / f"window_{window_index:06d}"
    window_dir.mkdir(parents=True, exist_ok=True)

    if save_npz:
        np.savez_compressed(window_dir / "predictions.npz", **arrays)
    if save_viz:
        save_visualizations(window_dir, arrays)

    with (window_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    return window_dir
