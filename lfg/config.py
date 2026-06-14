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

import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings needed to build the LFG model."""

    m: int = 3
    n: int = 3
    encoder_name: str = "dinov2"
    decoder_size: str = "large"
    ar_n_heads: int = 16
    ar_n_layers: int = 8
    ar_dropout: float = 0.1
    use_segmentation_head: bool = False
    segmentation_num_classes: int = 7
    use_motion_head: bool = False
    use_flow_head: bool = False
    point_head_type: str = "linear"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_updates(self, **updates: Any) -> "ModelConfig":
        clean = {key: value for key, value in updates.items() if value is not None}
        return replace(self, **clean)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_path(obj: Any, path: tuple[str, ...], default: Any = None) -> Any:
    current = obj
    for key in path:
        current = _get(current, key, None)
        if current is None:
            return default
    return current


def _get_any(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = _get(obj, key, None)
        if value is not None:
            return value
    return default


def _first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _infer_optional_heads(state_dict: Mapping[str, Any] | None) -> dict[str, Any]:
    if not state_dict:
        return {}

    inferred: dict[str, Any] = {}
    if any(key.startswith("segmentation_head.") for key in state_dict):
        inferred["use_segmentation_head"] = True
        weight = state_dict.get("segmentation_head.proj.weight")
        if weight is not None and hasattr(weight, "shape"):
            # LinearPts3d expands each class into patch_size**2 channels.
            patch_area = 14 * 14
            if weight.shape[0] % patch_area == 0:
                inferred["segmentation_num_classes"] = int(weight.shape[0] // patch_area)
    if any(key.startswith("motion_head.") for key in state_dict):
        inferred["use_motion_head"] = True
    if any(key.startswith("flow_head.") for key in state_dict):
        inferred["use_flow_head"] = True
    return inferred


def _infer_architecture_from_state(state_dict: Mapping[str, Any] | None) -> dict[str, Any]:
    if not state_dict:
        return {}

    inferred: dict[str, Any] = {}

    decoder_width = state_dict.get("decoder.0.norm1.weight")
    if decoder_width is not None and hasattr(decoder_width, "shape") and decoder_width.shape:
        inferred["decoder_size"] = {
            384: "small",
            768: "base",
            1024: "large",
        }.get(int(decoder_width.shape[0]), "large")

    block_indices = []
    for key in state_dict:
        match = re.match(r"autoregressive_transformer\.blocks\.(\d+)\.", key)
        if match:
            block_indices.append(int(match.group(1)))
    if block_indices:
        inferred["ar_n_layers"] = max(block_indices) + 1

    if any(key.startswith("point_head.initial_proj.") for key in state_dict):
        inferred["point_head_type"] = "conv"
    elif any(key.startswith("point_head.input_proj.") for key in state_dict):
        inferred["point_head_type"] = "refined"
    elif any(key.startswith("point_head.conv_net.") for key in state_dict):
        inferred["point_head_type"] = "simple_conv"
    elif any(key.startswith("point_head.proj.") for key in state_dict):
        inferred["point_head_type"] = "linear"

    return inferred


def model_config_from_checkpoint(
    checkpoint_config: Any = None,
    state_dict: Mapping[str, Any] | None = None,
) -> ModelConfig:
    """Build a ModelConfig from common LFG checkpoint payload formats."""

    model = _get(checkpoint_config, "MODEL", None)

    config = ModelConfig(
        m=int(_first(_get_any(model, "M", "m"), _get_any(checkpoint_config, "M", "m"), default=3)),
        n=int(_first(_get_any(model, "N", "n"), _get_any(checkpoint_config, "N", "n"), default=3)),
        encoder_name=str(
            _first(
                _get_any(model, "ENCODER_NAME", "encoder_name"),
                _get(checkpoint_config, "encoder_name"),
                default="dinov2",
            )
        ).lower(),
        decoder_size=str(
            _first(
                _get_path(checkpoint_config, ("MULTIVIEW", "DECODER_SIZE")),
                _get_any(model, "DECODER_SIZE", "decoder_size"),
                _get(checkpoint_config, "decoder_size"),
                default="large",
            )
        ).lower(),
        ar_n_heads=int(
            _first(_get_any(model, "AR_N_HEADS", "ar_n_heads"), _get(checkpoint_config, "ar_n_heads"), default=16)
        ),
        ar_n_layers=int(
            _first(_get_any(model, "AR_N_LAYERS", "ar_n_layers"), _get(checkpoint_config, "ar_n_layers"), default=8)
        ),
        ar_dropout=float(
            _first(_get_any(model, "AR_DROPOUT", "ar_dropout"), _get(checkpoint_config, "ar_dropout"), default=0.1)
        ),
        use_segmentation_head=_to_bool(
            _first(
                _get_any(model, "USE_SEGMENTATION_HEAD", "use_segmentation_head"),
                _get(checkpoint_config, "use_segmentation"),
                _get(checkpoint_config, "use_segmentation_head"),
            ),
            default=False,
        ),
        segmentation_num_classes=int(
            _first(_get_any(model, "SEGMENTATION_NUM_CLASSES", "segmentation_num_classes"), default=7)
        ),
        use_motion_head=_to_bool(
            _first(
                _get_any(model, "USE_MOTION_HEAD", "use_motion_head"),
                _get(checkpoint_config, "use_motion"),
                _get(checkpoint_config, "use_motion_head"),
            ),
            default=False,
        ),
        use_flow_head=_to_bool(
            _first(_get_any(model, "USE_FLOW_HEAD", "use_flow_head"), _get(checkpoint_config, "use_flow_head")),
            default=False,
        ),
        point_head_type=str(_first(_get_any(model, "POINT_HEAD_TYPE", "point_head_type"), default="linear")).lower(),
    )

    inferred = {}
    inferred.update(_infer_architecture_from_state(state_dict))
    inferred.update(_infer_optional_heads(state_dict))
    return config.with_updates(**inferred)


def checkpoint_looks_multiview(
    checkpoint_config: Any = None,
    state_dict: Mapping[str, Any] | None = None,
) -> bool:
    model = _get(checkpoint_config, "MODEL", None)
    architecture = str(
        _first(
            _get_any(model, "ARCHITECTURE", "architecture"),
            _get_any(checkpoint_config, "ARCHITECTURE", "architecture"),
            default="",
        )
    ).lower()
    if "multiview" in architecture:
        return True

    multiview_enabled = _to_bool(_get_path(checkpoint_config, ("MULTIVIEW", "ENABLE")), default=False)
    num_cameras = _first(
        _get_path(checkpoint_config, ("MULTIVIEW", "NUM_CAMERAS")),
        _get(checkpoint_config, "num_cameras"),
    )
    try:
        if num_cameras is not None and int(num_cameras) > 1 and (multiview_enabled or not architecture):
            return True
    except (TypeError, ValueError):
        pass

    if str(_get(checkpoint_config, "model_version", "")).lower() in {"v2", "v3", "multiview"}:
        return True

    if not state_dict:
        return False

    multiview_prefixes = (
        "camera_embedding.",
        "egomotion_head.",
        "cross_view",
        "cross_camera",
    )
    if "scale_token" in state_dict:
        return True
    return any(key.startswith(multiview_prefixes) for key in state_dict)


def checkpoint_has_autoregressive_state(state_dict: Mapping[str, Any]) -> bool:
    return any(key.startswith("autoregressive_transformer.") for key in state_dict)
