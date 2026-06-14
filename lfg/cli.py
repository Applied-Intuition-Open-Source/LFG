#!/usr/bin/env python3
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
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from lfg.checkpoint import load_model_from_checkpoint
from lfg.inference import (
    predictions_to_numpy,
    predict_window,
    save_window_result,
    window_metadata,
)
from lfg.io import iter_frame_windows, iter_input_frames
from lfg.paths import ensure_local_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LFG inference on a video or image sequence.")
    parser.add_argument("input", help="Video file, image file, image directory, or quoted image glob.")
    parser.add_argument("--checkpoint", required=True, help="Path to an LFG checkpoint.")
    parser.add_argument("--output-dir", default="outputs/lfg_inference", help="Directory for predictions and visualizations.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")

    parser.add_argument(
        "--target-size",
        type=int,
        default=518,
        help="Resize width/maximum side before inference. Must be divisible by 14.",
    )
    parser.add_argument("--resize-mode", choices=["crop", "pad"], default="crop", help="Resize policy.")
    parser.add_argument("--keep-ratio", action="store_true", help="Do not center-crop tall resized images.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Read every Nth input frame.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index to read from video/image sequence.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum sampled frames to read.")
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Advance N sampled frames between model windows. Defaults to the checkpoint history length.",
    )

    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw predictions as compressed NPZ files.",
    )
    parser.add_argument(
        "--save-visualizations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save depth/confidence/segmentation/motion/flow PNG visualizations.",
    )
    parser.add_argument("--save-features", action="store_true", help="Also save large internal feature tensors.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(device_arg)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device '{device_arg}'. Use auto, cuda, cuda:0, or cpu.") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device_arg}'. Use auto, cuda, cuda:0, or cpu.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Device '{device_arg}' was requested, but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Device '{device_arg}' was requested, but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
    return device_arg


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive when set.")
    if args.window_stride is not None and args.window_stride <= 0:
        raise ValueError("--window-stride must be positive when set.")
    if args.target_size <= 0:
        raise ValueError("--target-size must be positive.")
    if args.target_size % 14 != 0:
        raise ValueError("--target-size must be divisible by 14.")


def validate_output_path(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {output_dir}")


def prepare_output_dir(output_dir: Path) -> None:
    validate_output_path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Could not create output directory {output_dir}: {exc}") from exc


def run() -> None:
    args = parse_args()
    validate_args(args)
    ensure_local_path(args.input, kind="Input")
    ensure_local_path(str(args.checkpoint), kind="Checkpoint")
    ensure_local_path(args.output_dir, kind="Output directory")
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    validate_output_path(output_dir)

    model, config, load_report, checkpoint_info = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
    )
    window_stride = args.window_stride if args.window_stride is not None else config.m

    run_metadata: dict[str, Any] = {
        "input": args.input,
        "checkpoint": str(args.checkpoint),
        "checkpoint_info": checkpoint_info,
        "device": device,
        "model_config": config.to_dict(),
        "load_report": {
            "loaded_keys": load_report.loaded_keys,
            "model_key_count": load_report.model_key_count,
            "loaded_fraction": load_report.loaded_fraction,
            "missing_key_count": len(load_report.missing_keys),
            "unexpected_key_count": len(load_report.unexpected_keys),
            "skipped_shape_mismatch_count": len(load_report.skipped_shape_mismatches),
            "ignored_checkpoint_key_count": len(load_report.ignored_checkpoint_keys),
        },
        "frame_stride": max(1, args.frame_stride),
        "window_stride": window_stride,
        "target_size": args.target_size,
        "resize_mode": args.resize_mode,
        "windows": [],
    }

    frame_stream = iter_input_frames(
        args.input,
        frame_stride=max(1, args.frame_stride),
        max_frames=args.max_frames,
        start_frame=max(0, args.start_frame),
    )

    prepare_output_dir(output_dir)
    windows = iter_frame_windows(frame_stream, history=config.m, window_stride=window_stride)
    for window_index, (start_index, window_frames, padded) in enumerate(
        tqdm(windows, desc="LFG inference")
    ):
        predictions = predict_window(
            model,
            window_frames,
            config,
            device=device,
            target_size=args.target_size,
            resize_mode=args.resize_mode,
            keep_ratio=args.keep_ratio,
        )
        arrays = predictions_to_numpy(predictions, save_features=args.save_features)
        metadata = window_metadata(
            window_index=window_index,
            start_index=start_index,
            frames=window_frames,
            padded=padded,
            config=config,
        )
        window_dir = save_window_result(
            output_dir,
            window_index=window_index,
            arrays=arrays,
            metadata=metadata,
            save_npz=args.save_npz,
            save_viz=args.save_visualizations,
        )
        run_metadata["windows"].append(
            {
                "window_index": window_index,
                "path": str(window_dir),
                "start_index": start_index,
                "input_frame_indices": [frame.frame_index for frame in window_frames],
                "padded": padded,
            }
        )

    run_metadata["written_windows"] = len(run_metadata["windows"])
    if run_metadata["written_windows"] == 0:
        raise ValueError("No input frames were found.")
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, allow_nan=False)
    print(f"Wrote {run_metadata['written_windows']} window(s) to {output_dir}")


def main() -> None:
    try:
        run()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
