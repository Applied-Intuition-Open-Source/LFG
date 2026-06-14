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

import importlib.util
import sys
from pathlib import Path

from .config import ModelConfig


SUPPORTED_ENCODERS = {"dinov2"}
SUPPORTED_DECODER_SIZES = {"small", "base", "large"}
SUPPORTED_POINT_HEADS = {"linear", "refined", "conv", "simple_conv"}
MAX_TOTAL_FRAMES = 15


def _ensure_local_pi3_on_path() -> None:
    try:
        if importlib.util.find_spec("pi3.models.pi3") is not None:
            return
    except ModuleNotFoundError:
        pass

    root = Path(__file__).resolve().parents[1]
    pi3_root = root / "Pi3"
    if not pi3_root.exists():
        raise ImportError("Could not find bundled Pi3 model package.")
    if str(pi3_root) not in sys.path:
        sys.path.insert(0, str(pi3_root))


def model_config_error(config: ModelConfig) -> str | None:
    if config.m <= 0:
        return "m must be positive"
    if config.n < 0:
        return "n must be non-negative"
    if config.m + config.n > MAX_TOTAL_FRAMES:
        return f"m + n must be <= {MAX_TOTAL_FRAMES} for this LFG checkpoint"
    if config.encoder_name not in SUPPORTED_ENCODERS:
        return (
            f"Unsupported encoder_name '{config.encoder_name}'. "
            "This inference package currently supports dinov2 LFG checkpoints."
        )
    if config.decoder_size not in SUPPORTED_DECODER_SIZES:
        return "decoder_size must be one of: small, base, large"
    if config.ar_n_heads <= 0:
        return "ar_n_heads must be positive"
    if config.ar_n_layers <= 0:
        return "ar_n_layers must be positive"
    if not 0.0 <= config.ar_dropout < 1.0:
        return "ar_dropout must be in [0, 1)"
    if config.segmentation_num_classes <= 0:
        return "segmentation_num_classes must be positive"
    if config.point_head_type not in SUPPORTED_POINT_HEADS:
        return "point_head_type must be one of: linear, refined, conv, simple_conv"
    return None


def validate_model_config(config: ModelConfig) -> None:
    error = model_config_error(config)
    if error is not None:
        raise ValueError(error)


def build_model(config: ModelConfig):
    """Instantiate the LFG model."""

    validate_model_config(config)
    _ensure_local_pi3_on_path()
    from pi3.models.pi3 import LFG

    return LFG(
        decoder_size=config.decoder_size,
        encoder_name=config.encoder_name,
        n_future_frames=config.n,
        ar_n_heads=config.ar_n_heads,
        ar_n_layers=config.ar_n_layers,
        ar_dropout=config.ar_dropout,
        use_segmentation_head=config.use_segmentation_head,
        segmentation_num_classes=config.segmentation_num_classes,
        use_motion_head=config.use_motion_head,
        use_flow_head=config.use_flow_head,
        point_head_type=config.point_head_type,
        pretrained_encoder=False,
    )
