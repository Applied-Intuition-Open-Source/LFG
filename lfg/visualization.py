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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SEGMENTATION_PALETTE = np.array(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
    ],
    dtype=np.uint8,
)


def _finite_percentile_range(values: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def colorize_scalar(values: np.ndarray, *, cmap_name: str = "turbo") -> np.ndarray:
    vmin, vmax = _finite_percentile_range(values)
    normalized = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    colormap = plt.get_cmap(cmap_name)
    return (colormap(normalized)[..., :3] * 255).astype(np.uint8)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_visualizations(window_dir: Path, predictions: dict[str, np.ndarray]) -> None:
    if "local_points" in predictions:
        depth = predictions["local_points"][..., 2]
        for frame_idx in range(depth.shape[0]):
            save_png(window_dir / "depth" / f"{frame_idx:03d}.png", colorize_scalar(depth[frame_idx]))

    if "conf" in predictions:
        conf = predictions["conf"][..., 0]
        conf = _sigmoid(conf)
        for frame_idx in range(conf.shape[0]):
            save_png(window_dir / "confidence" / f"{frame_idx:03d}.png", colorize_scalar(conf[frame_idx], cmap_name="magma"))

    if "segmentation" in predictions:
        labels = predictions["segmentation"].argmax(axis=-1)
        for frame_idx in range(labels.shape[0]):
            colors = SEGMENTATION_PALETTE[labels[frame_idx] % len(SEGMENTATION_PALETTE)]
            save_png(window_dir / "segmentation" / f"{frame_idx:03d}.png", colors)

    if "motion" in predictions:
        motion = predictions["motion"][..., 0]
        motion = _sigmoid(motion)
        for frame_idx in range(motion.shape[0]):
            save_png(window_dir / "motion" / f"{frame_idx:03d}.png", colorize_scalar(motion[frame_idx], cmap_name="inferno"))

    if "flow" in predictions:
        flow = predictions["flow"]
        magnitude = np.linalg.norm(flow, axis=-1)
        for frame_idx in range(magnitude.shape[0]):
            save_png(window_dir / "flow" / f"{frame_idx:03d}.png", colorize_scalar(magnitude[frame_idx], cmap_name="viridis"))
