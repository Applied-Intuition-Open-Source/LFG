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

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from .config import (
    ModelConfig,
    checkpoint_has_autoregressive_state,
    checkpoint_looks_multiview,
    model_config_from_checkpoint,
)
from .model import build_model
from .paths import ensure_local_path


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded_keys: int
    model_key_count: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    skipped_shape_mismatches: list[str]
    ignored_checkpoint_keys: list[str]

    @property
    def loaded_fraction(self) -> float:
        if self.model_key_count == 0:
            return 0.0
        return self.loaded_keys / self.model_key_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded_keys": self.loaded_keys,
            "model_key_count": self.model_key_count,
            "loaded_fraction": self.loaded_fraction,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "skipped_shape_mismatches": self.skipped_shape_mismatches,
            "ignored_checkpoint_keys": self.ignored_checkpoint_keys,
        }


def _add_torch_safe_globals() -> None:
    safe_globals: list[type[Any]] = [argparse.Namespace, SimpleNamespace]
    try:
        import yacs.config

        safe_globals.append(yacs.config.CfgNode)
    except Exception:
        pass
    try:
        import numpy as np
        from numpy.core.multiarray import scalar as numpy_scalar

        safe_globals.extend([numpy_scalar, np.dtype])
        if hasattr(np, "dtypes"):
            safe_globals.extend(
                dtype_class
                for name in (
                    "BoolDType",
                    "Float16DType",
                    "Float32DType",
                    "Float64DType",
                    "Int16DType",
                    "Int32DType",
                    "Int64DType",
                    "UInt8DType",
                )
                if (dtype_class := getattr(np.dtypes, name, None)) is not None
            )
    except Exception:
        pass
    torch.serialization.add_safe_globals(safe_globals)


def _strip_known_prefixes(key: str) -> str:
    prefixes = ("module.", "_orig_mod.", "model.", "student.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {_strip_known_prefixes(key): value for key, value in state_dict.items()}


def load_checkpoint_file(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    ensure_local_path(str(path), kind="Checkpoint")
    _add_torch_safe_globals()
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Checkpoint could not be loaded with PyTorch's safe weights-only loader. "
            "Use a checkpoint saved as a tensor/primitive dictionary with optional argparse, "
            "SimpleNamespace, or YACS config metadata."
        ) from exc


def extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _normalize_state_dict(value)
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return _normalize_state_dict(checkpoint)
    raise ValueError("Checkpoint does not contain a model state dict.")


def _json_safe_metadata_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return _json_safe_metadata_value(value.item())
        return _json_safe_metadata_value(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe_metadata_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata_value(item) for item in value]
    return str(value)


def checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key in ("epoch", "global_step", "best_loss", "val_loss", "best_val_loss"):
        if key in checkpoint:
            metadata[key] = _json_safe_metadata_value(checkpoint[key])
    return metadata


def checkpoint_config(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, Mapping):
        return None
    return checkpoint.get("config") or checkpoint.get("args")


def _filter_state_dict_for_model(
    state_dict: Mapping[str, Any],
    model_state: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    filtered: dict[str, Any] = {}
    shape_mismatches: list[str] = []
    ignored: list[str] = []

    for key, value in state_dict.items():
        if key not in model_state:
            ignored.append(key)
            continue
        if hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint {tuple(value.shape)} != model {tuple(model_state[key].shape)}"
            )
            continue
        filtered[key] = value
    return filtered, shape_mismatches, ignored


def _preview_keys(keys: list[str], *, limit: int = 8) -> str:
    preview = ", ".join(keys[:limit])
    if len(keys) > limit:
        preview = f"{preview}, ... (+{len(keys) - limit} more)"
    return preview


def _validate_full_model_load(report: CheckpointLoadReport) -> None:
    if not report.missing_keys and not report.skipped_shape_mismatches:
        return

    details = [
        f"loaded {report.loaded_keys}/{report.model_key_count} model tensors",
    ]
    if report.missing_keys:
        details.append(f"missing {len(report.missing_keys)} keys: {_preview_keys(report.missing_keys)}")
    if report.skipped_shape_mismatches:
        details.append(
            f"shape mismatches {len(report.skipped_shape_mismatches)}: "
            f"{_preview_keys(report.skipped_shape_mismatches, limit=4)}"
        )
    raise ValueError(
        "Checkpoint did not fully load into the model; refusing to run inference "
        f"with randomly initialized weights ({'; '.join(details)})."
    )


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> tuple[torch.nn.Module, ModelConfig, CheckpointLoadReport, dict[str, Any]]:
    """Load an LFG model from a local checkpoint file."""

    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    config_payload = checkpoint_config(checkpoint)

    if checkpoint_looks_multiview(config_payload, state_dict):
        raise ValueError(
            "This checkpoint is not compatible with this inference repo."
        )
    if not checkpoint_has_autoregressive_state(state_dict):
        raise ValueError("This checkpoint does not contain LFG weights.")

    model_config = model_config_from_checkpoint(config_payload, state_dict)
    model = build_model(model_config)
    model_state = model.state_dict()
    filtered_state, shape_mismatches, ignored_keys = _filter_state_dict_for_model(state_dict, model_state)
    if not filtered_state:
        raise ValueError("No checkpoint tensors matched the model architecture.")

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    report = CheckpointLoadReport(
        loaded_keys=len(filtered_state),
        model_key_count=len(model_state),
        missing_keys=list(missing),
        unexpected_keys=list(unexpected),
        skipped_shape_mismatches=shape_mismatches,
        ignored_checkpoint_keys=ignored_keys,
    )
    _validate_full_model_load(report)

    model.to(device)
    model.eval()
    return model, model_config, report, checkpoint_metadata(checkpoint)
