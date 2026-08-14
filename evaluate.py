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

"""Evaluate depth, semantics and trajectory on clips of consecutive driving frames.

Each clip is six consecutive frames. LFG is given the first three and predicts all six;
baselines that are not future-predicting (pi3, vggt, segformer) are given all six, so for
them the "predicted" split simply names the same frames LFG had to extrapolate. Metrics
are reported over all frames ("overall"), the frames the model observed ("observed") and
the frames it had to predict ("predicted").

  depth       AbsRel, RMSE (m) and delta<1.25 against LiDAR, after a least-squares scale
              and shift alignment fitted once per clip.
  semantics   pixel accuracy and mIoU over seven classes, averaged per frame over the
              classes present in that frame.
  trajectory  ATE (m) after a similarity alignment, plus rotation (deg) and translation
              (% of distance travelled) error relative to the first frame.

Clips come from a list file (see eval/clips/), one `<sequence>:<first frame>` per line.

Examples:
  python evaluate.py --checkpoint checkpoints/lfg_seg_motion_m3n3.pt \
      --dataset kitti360 --data-root /data/KITTI-360

  python evaluate.py --model pi3 \
      --dataset waymo --data-root /data/waymo_v2/validation \
      --clip-list eval/clips/waymo_200.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lfg.checkpoint import load_model_from_checkpoint
from lfg.inference import predict_window
from lfg.io import Frame, preprocess_frames

CLIP_LENGTH = 6
IGNORE_LABEL = 255
TARGET_WIDTH = 518          # model input width; frames keep their aspect and are centre-cropped
MAX_DEPTH = 80.0            # ground truth beyond this range is ignored
MIN_VALID_POINTS = 100      # frames with fewer ground-truth pixels are not scored
MIN_TRAVEL = 1.0            # clips where the vehicle barely moved carry no trajectory signal
WAYMO_FRONT_CAMERA = 1
# OpenCV camera axes (x-right, y-down, z-forward) expressed in Waymo's (x-fwd, y-left, z-up).
OPENCV_TO_WAYMO_CAMERA = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)

# The segmentation head's 7 classes, in index order.
CLASS_NAMES = ["road", "vehicle", "person", "traffic light", "traffic sign", "sky",
               "building/grass/background"]

# Cityscapes labelId -> class index. Cityscapes "void" ids carry no supervision and are
# excluded from the metrics rather than folded into the background class.
CITYSCAPES_LABEL_TO_CLASS = {
    7: 0,                                                    # road
    26: 1, 27: 1, 28: 1, 31: 1, 32: 1, 33: 1,                # car/truck/bus/train/motorcycle/bicycle
    24: 2, 25: 2,                                            # person, rider
    19: 3,                                                   # traffic light
    20: 4,                                                   # traffic sign
    23: 5,                                                   # sky
    8: 6, 11: 6, 12: 6, 13: 6, 17: 6, 21: 6, 22: 6,          # sidewalk/building/wall/fence/pole/vegetation/terrain
}


# SegFormer is trained on Cityscapes train ids; this is the same grouping the LFG
# segmentation head was distilled with.
CITYSCAPES_TRAIN_ID_TO_CLASS = {
    0: 0,                                              # road
    6: 3, 7: 4, 10: 5,                                 # traffic light, traffic sign, sky
    11: 2, 12: 2,                                      # person, rider
    13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1,          # car/truck/bus/train/motorcycle/bicycle
    1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 8: 6, 9: 6,          # sidewalk/building/wall/fence/pole/vegetation/terrain
}


@dataclass
class Clip:
    """One evaluation clip: frame images plus a lazy loader for ground-truth depth."""

    name: str
    load_images: Callable[[], list[np.ndarray]]
    """load_images() -> CLIP_LENGTH RGB frames as uint8 arrays."""
    load_depth: Callable[[int, int, int], np.ndarray]
    """load_depth(frame_slot, out_height, out_width) -> sparse metric depth, NaN where absent."""
    load_poses: Callable[[], np.ndarray] | None = None
    """load_poses() -> (CLIP_LENGTH, 4, 4) ground-truth camera-to-world poses, if available."""
    load_labels: Callable[[int, int, int], np.ndarray] | None = None
    """load_labels(frame_slot, out_height, out_width) -> class indices, IGNORE_LABEL for void."""


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def affine_align(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares scale and shift taking `pred` onto `target` over `mask`."""
    source = pred[mask]
    design = np.stack([source, np.ones_like(source)], axis=1)
    scale, shift = np.linalg.lstsq(design, target[mask], rcond=None)[0]
    return scale * pred + shift


def depth_errors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """AbsRel, RMSE (metres) and delta<1.25 over an already-aligned prediction."""
    estimate = np.clip(pred[mask], 1e-3, None)
    truth = target[mask]
    abs_rel = float(np.mean(np.abs(estimate - truth) / truth))
    rmse = float(np.sqrt(np.mean((estimate - truth) ** 2)))
    ratio = np.maximum(estimate / truth, truth / estimate)
    return abs_rel, rmse, float(np.mean(ratio < 1.25))


def frame_segmentation_metrics(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Per-frame pixel accuracy and mIoU.

    Class averages cover the classes present in the prediction or the ground truth of that
    frame, so a class absent from a frame neither helps nor hurts.
    """
    classes = len(CLASS_NAMES)
    valid = (truth != IGNORE_LABEL) & (predicted < classes)
    predicted, truth = predicted[valid], truth[valid]
    if predicted.size == 0:
        return np.nan, np.nan

    accuracy = float((predicted == truth).mean())
    ious = []
    for index in range(classes):
        prediction_mask, truth_mask = predicted == index, truth == index
        union = int((prediction_mask | truth_mask).sum())
        if union == 0:      # a class in neither the prediction nor the truth is not scored
            continue
        ious.append(int((prediction_mask & truth_mask).sum()) / union)
    return (accuracy, float(np.mean(ious))) if ious else (accuracy, np.nan)


def load_depth_cached(clip: "Clip", slot: int, height: int, width: int,
                      cache_dir: Path | None, max_depth: float, stride: int) -> np.ndarray:
    """Ground-truth depth for one frame, reusing a cached copy when one is available.

    Decoding ground truth is the slowest part of a run and every model repeats it over the
    same clips. Anything that changes the depth map is part of the key.
    """
    if cache_dir is None:
        return clip.load_depth(slot, height, width)

    key = f"{clip.name.replace('/', '_')}_s{stride}_{slot}_{height}x{width}_{max_depth:g}.npz"
    path = cache_dir / key
    if path.exists():
        with np.load(path) as stored:
            return stored["depth"]

    depth = clip.load_depth(slot, height, width)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, depth=depth)
    return depth


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, rotation, translation) taking `source` points onto `target`."""
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centred, target_centred = source - source_mean, target - target_mean
    u, singular_values, vt = np.linalg.svd(target_centred.T @ source_centred / len(source))
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = (source_centred ** 2).sum() / len(source)
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance) if variance > 0 else 1.0
    return scale, rotation, target_mean - scale * rotation @ source_mean


def trajectory_errors(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """ATE (m), rotation error (deg) and translation error as a share of distance travelled.

    Both trajectories are expressed relative to their first frame. Predicted poses are only
    defined up to a similarity, so a scale is fitted against the ground-truth positions; ATE
    additionally applies the full similarity before measuring position error, while the
    relative-pose errors compare each frame's pose against frame 0.

    Translation error is a share of the distance covered, which stays comparable across
    frame rates and datasets. Callers should skip stationary clips, where it is undefined.
    """
    relative = lambda poses: np.linalg.inv(poses[0]) @ poses  # noqa: E731
    predicted, truth = relative(predicted), relative(truth)
    scale, rotation, translation = umeyama(predicted[:, :3, 3], truth[:, :3, 3])

    aligned = (scale * (rotation @ predicted[:, :3, 3].T).T) + translation
    ate = float(np.sqrt((np.linalg.norm(aligned - truth[:, :3, 3], axis=1) ** 2).mean()))

    residual = np.einsum("nij,nkj->nik", predicted[1:, :3, :3], truth[1:, :3, :3])
    cosines = (np.trace(residual, axis1=1, axis2=2) - 1.0) / 2.0
    rotation_error = float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).mean())

    # Pooled rather than a mean of per-frame ratios: early frames cover little ground, so
    # dividing frame by frame lets the first step dominate.
    offsets = np.linalg.norm(scale * predicted[1:, :3, 3] - truth[1:, :3, 3], axis=1)
    travelled = np.linalg.norm(truth[1:, :3, 3], axis=1)
    return ate, rotation_error, float(100.0 * offsets.sum() / travelled.sum())


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "count": 0}
    return {"mean": float(array.mean()), "std": float(array.std()), "count": int(array.size)}


# --------------------------------------------------------------------------------------
# Dataset adapters
# --------------------------------------------------------------------------------------


def _sparse_depth_map(
    rows: np.ndarray,
    cols: np.ndarray,
    depth: np.ndarray,
    source_hw: tuple[int, int],
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Rasterise scattered depth samples onto the model's output grid, nearest surface wins."""
    source_h, source_w = source_hw
    out_h, out_w = out_hw
    y = np.clip((((rows + 0.5) / source_h) * out_h).astype(int), 0, out_h - 1)
    x = np.clip((((cols + 0.5) / source_w) * out_w).astype(int), 0, out_w - 1)
    canvas = np.full((out_h, out_w), np.nan, dtype=np.float32)
    order = np.argsort(-depth)
    canvas[y[order], x[order]] = depth[order]
    return canvas


@lru_cache(maxsize=6)
def _waymo_arrow(root: Path, component: str, segment: str, columns: tuple[str, ...]):
    """One component of one Waymo segment, left in Arrow.

    Cached and never converted wholesale: the LiDAR columns hold millions of values, so rows
    are selected first and only those are turned into numpy.
    """
    import pyarrow.parquet as pq

    return pq.read_table(root / component / f"{segment}.parquet", columns=list(columns))


def _waymo_at(table, stamp: int):
    """The rows of a component belonging to one frame."""
    import pyarrow.compute as pc

    return table.filter(pc.equal(table.column("key.frame_timestamp_micros"), stamp))


def _waymo_front_calibration(root: Path, segment: str):
    """Intrinsics, image size and camera-to-vehicle pose (OpenCV axes) for the front camera."""
    prefix = "[CameraCalibrationComponent]"
    table = _waymo_arrow(root, "camera_calibration", segment, (
        "key.camera_name", f"{prefix}.intrinsic.f_u", f"{prefix}.intrinsic.f_v",
        f"{prefix}.intrinsic.c_u", f"{prefix}.intrinsic.c_v",
        f"{prefix}.width", f"{prefix}.height", f"{prefix}.extrinsic.transform",
    )).to_pydict()
    row = table["key.camera_name"].index(WAYMO_FRONT_CAMERA)
    intrinsics = tuple(float(table[f"{prefix}.intrinsic.{k}"][row]) for k in ("f_u", "f_v", "c_u", "c_v"))
    size = (int(table[f"{prefix}.width"][row]), int(table[f"{prefix}.height"][row]))
    vehicle_from_camera = np.asarray(table[f"{prefix}.extrinsic.transform"][row], float).reshape(4, 4)
    # Waymo camera axes are x-forward/y-left/z-up; rotate so the pose uses OpenCV axes.
    vehicle_from_camera[:3, :3] = vehicle_from_camera[:3, :3] @ OPENCV_TO_WAYMO_CAMERA
    return intrinsics, size, vehicle_from_camera


def _waymo_depth(root, segment, stamp, camera_size, intrinsics, out_hw, max_depth) -> np.ndarray:
    """Camera-frame depth for one frame, from the LiDAR range images and their projections."""
    lidar = _waymo_at(_waymo_arrow(root, "lidar", segment, (
        "key.frame_timestamp_micros",
        "[LiDARComponent].range_image_return1.values",
        "[LiDARComponent].range_image_return1.shape")), stamp)
    projection = _waymo_at(_waymo_arrow(root, "lidar_camera_projection", segment, (
        "key.frame_timestamp_micros",
        "[LiDARCameraProjectionComponent].range_image_return1.values",
        "[LiDARCameraProjectionComponent].range_image_return1.shape")), stamp)

    focal_u, focal_v, centre_u, centre_v = intrinsics
    width, height = camera_size
    rows, cols, depths = [], [], []
    for laser in range(min(lidar.num_rows, projection.num_rows)):
        shape = np.asarray(lidar.column(2)[laser].values)
        ranges = np.asarray(lidar.column(1)[laser].values).reshape(tuple(shape))[:, :, 0]
        proj = np.asarray(projection.column(1)[laser].values).reshape(
            tuple(np.asarray(projection.column(2)[laser].values)))
        for offset in (0, 3):     # a return may project into two cameras
            hit = (proj[:, :, offset] == WAYMO_FRONT_CAMERA) & (ranges > 0)
            if not hit.any():
                continue
            u = proj[:, :, offset + 1][hit].astype(float)
            v = proj[:, :, offset + 2][hit].astype(float)
            # range is measured along the ray, so convert it to a camera-frame z
            x, y = (u - centre_u) / focal_u, (v - centre_v) / focal_v
            depths.append(ranges[hit] / np.sqrt(x * x + y * y + 1.0))
            rows.append(v)
            cols.append(u)

    if not depths:
        return np.full(out_hw, np.nan, np.float32)
    depth = np.concatenate(depths)
    keep = depth <= max_depth
    return _sparse_depth_map(np.concatenate(rows)[keep], np.concatenate(cols)[keep],
                             depth[keep], (height, width), out_hw)


def waymo_clips(root: Path, max_depth: float, stride: int = 1) -> Iterator[Clip]:
    """Waymo Open Dataset v2, read from the released parquet components.

    Layout: <root>/{camera_image,lidar,lidar_camera_projection,camera_calibration,
    vehicle_pose}/<segment>.parquet -- one split directory exactly as distributed.
    """
    import io as _io

    components = ("lidar", "lidar_camera_projection", "camera_calibration", "vehicle_pose")
    for path in sorted((root / "camera_image").glob("*.parquet")):
        segment = path.stem
        if not all((root / c / f"{segment}.parquet").exists() for c in components):
            continue
        intrinsics, camera_size, vehicle_from_camera = _waymo_front_calibration(root, segment)
        images = _waymo_arrow(root, "camera_image", segment,
                              ("key.frame_timestamp_micros", "key.camera_name",
                               "[CameraImageComponent].image"))
        import pyarrow.compute as pc
        front = images.filter(pc.equal(images.column("key.camera_name"), WAYMO_FRONT_CAMERA))
        stamps = sorted(front.column("key.frame_timestamp_micros").to_pylist())

        for start in range(len(stamps) - stride * (CLIP_LENGTH - 1)):
            window = tuple(stamps[start + stride * step] for step in range(CLIP_LENGTH))

            def load_images(window=window, segment=segment) -> list[np.ndarray]:
                table = _waymo_arrow(root, "camera_image", segment,
                                     ("key.frame_timestamp_micros", "key.camera_name",
                                      "[CameraImageComponent].image"))
                table = table.filter(pc.equal(table.column("key.camera_name"), WAYMO_FRONT_CAMERA))
                return [np.asarray(Image.open(_io.BytesIO(
                    _waymo_at(table, stamp).column("[CameraImageComponent].image")[0].as_py()
                )).convert("RGB")) for stamp in window]

            def load_depth(slot, out_h, out_w, window=window, segment=segment,
                           camera_size=camera_size, intrinsics=intrinsics) -> np.ndarray:
                return _waymo_depth(root, segment, window[slot], camera_size, intrinsics,
                                    (out_h, out_w), max_depth)

            def load_poses(window=window, segment=segment,
                           vehicle_from_camera=vehicle_from_camera) -> np.ndarray:
                table = _waymo_arrow(root, "vehicle_pose", segment,
                                     ("key.frame_timestamp_micros",
                                      "[VehiclePoseComponent].world_from_vehicle.transform")).to_pydict()
                by_time = dict(zip(table["key.frame_timestamp_micros"],
                                   table["[VehiclePoseComponent].world_from_vehicle.transform"]))
                return np.stack([np.asarray(by_time[s], float).reshape(4, 4) @ vehicle_from_camera
                                 for s in window])

            yield Clip(name=f"{segment}:{window[0]}", load_images=load_images,
                       load_depth=load_depth, load_poses=load_poses)


def _kitti360_calibration(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_rect_00, T_cam0_from_velo) for the left perspective camera."""
    perspective = {}
    with (root / "calibration" / "perspective.txt").open() as handle:
        for line in handle:
            key, _, values = line.partition(":")
            try:
                perspective[key.strip()] = np.array(values.split(), dtype=float)
            except ValueError:
                continue        # non-numeric entries, such as the calibration date
    projection = perspective["P_rect_00"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = perspective["R_rect_00"].reshape(3, 3)

    cam_to_velo = np.loadtxt(root / "calibration" / "calib_cam_to_velo.txt").reshape(3, 4)
    velo_from_cam = np.eye(4)
    velo_from_cam[:3, :4] = cam_to_velo
    return projection, rect @ np.linalg.inv(velo_from_cam)


def kitti360_clips(root: Path, max_depth: float, stride: int = 1) -> Iterator[Clip]:
    """KITTI-360 in its official layout, LiDAR projected into the left perspective camera.

    Layout: data_2d_raw/<drive>/image_00/data_rect/<idx>.png
            data_3d_raw/<drive>/velodyne_points/data/<idx>.bin
            calibration/{perspective.txt,calib_cam_to_velo.txt}
    """
    projection, cam_from_velo = _kitti360_calibration(root)
    for drive in sorted((root / "data_2d_raw").iterdir()):
        image_dir = drive / "image_00" / "data_rect"
        velodyne_dir = root / "data_3d_raw" / drive.name / "velodyne_points" / "data"
        if not image_dir.is_dir():
            continue
        has_lidar = velodyne_dir.is_dir()
        indices = sorted(int(p.stem) for p in image_dir.glob("*.png"))
        available = set(indices)
        for first in indices:
            window = [first + stride * step for step in range(CLIP_LENGTH)]
            if not available.issuperset(window):
                continue
            if has_lidar and not all((velodyne_dir / f"{i:010d}.bin").exists() for i in window):
                continue

            def load_depth(
                slot: int,
                out_h: int,
                out_w: int,
                window=window,
                image_dir=image_dir,
                velodyne_dir=velodyne_dir,
                has_lidar=has_lidar,
            ) -> np.ndarray:
                if not has_lidar:  # semantics-only run
                    return np.full((out_h, out_w), np.nan, dtype=np.float32)

                scan = np.fromfile(velodyne_dir / f"{window[slot]:010d}.bin", dtype=np.float32)
                points = scan.reshape(-1, 4)[:, :3]
                homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
                in_camera = homogeneous @ cam_from_velo.T
                forward = in_camera[:, 2]
                in_camera = in_camera[forward > 0]
                pixels = in_camera @ projection.T
                depth = pixels[:, 2]
                cols, rows = pixels[:, 0] / depth, pixels[:, 1] / depth
                width, height = Image.open(image_dir / f"{window[slot]:010d}.png").size
                keep = (
                    (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height) & (depth <= max_depth)
                )
                return _sparse_depth_map(
                    rows[keep], cols[keep], depth[keep], (height, width), (out_h, out_w)
                )

            semantic_dir = root / "data_2d_semantics" / "train" / drive.name / "image_00" / "semantic"

            def load_labels(
                slot: int, out_h: int, out_w: int, window=window, semantic_dir=semantic_dir
            ) -> np.ndarray:
                label_path = semantic_dir / f"{window[slot]:010d}.png"
                label_ids = np.asarray(
                    Image.open(label_path).resize((out_w, out_h), Image.NEAREST)
                )
                classes = np.full(label_ids.shape, IGNORE_LABEL, dtype=np.uint8)
                for label_id, class_index in CITYSCAPES_LABEL_TO_CLASS.items():
                    classes[label_ids == label_id] = class_index
                return classes

            # data_poses.zip extracts either under data_poses/ or straight to the root
            pose_file = next(
                (candidate for candidate in (
                    root / "data_poses" / drive.name / "cam0_to_world.txt",
                    root / drive.name / "cam0_to_world.txt",
                ) if candidate.exists()),
                root / "data_poses" / drive.name / "cam0_to_world.txt",
            )

            def load_poses(window=window, pose_file=pose_file) -> np.ndarray:
                table = np.loadtxt(pose_file)
                by_frame = {int(row[0]): row[1:].reshape(4, 4) for row in table}
                if not all(i in by_frame for i in window):
                    raise KeyError("clip has frames without a pose")
                return np.stack([by_frame[i] for i in window])

            if semantic_dir.is_dir() and not all(
                (semantic_dir / f"{i:010d}.png").exists() for i in window
            ):
                continue

            yield Clip(
                name=f"{drive.name}:{window[0]:010d}",
                load_images=(lambda window=window, image_dir=image_dir: [
                    np.asarray(Image.open(image_dir / f"{i:010d}.png").convert("RGB")) for i in window
                ]),
                load_depth=load_depth,
                load_poses=load_poses if pose_file.exists() else None,
                load_labels=load_labels if semantic_dir.is_dir() else None,
            )


ADAPTERS = {"kitti360": kitti360_clips, "waymo": waymo_clips}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def load_pi3_baseline(weights: str, device: str):
    """The pi3 teacher, which is given every frame of the clip rather than only the history.

    Needs the upstream pi3 package (https://github.com/yyfz/Pi3); it is not vendored here.
    Weights are downloaded from the Hub unless a local file is given.
    """
    try:
        from pi3.models.pi3 import Pi3
    except ImportError as exc:  # pragma: no cover - depends on an optional package
        raise SystemExit(
            "The pi3 baseline needs the upstream pi3 package on PYTHONPATH "
            "(https://github.com/yyfz/Pi3)."
        ) from exc

    model = Pi3()
    if not weights:
        from huggingface_hub import snapshot_download

        weights = str(Path(snapshot_download("yyfz233/Pi3")) / "model.safetensors")
    if weights.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(weights)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_segformer_baseline(device: str):
    """The SegFormer teacher, given every RGB frame of the clip."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    return (SegformerImageProcessor.from_pretrained(name),
            SegformerForSemanticSegmentation.from_pretrained(name).eval().to(device))


def load_maskformer_baseline(device: str):
    """MaskFormer trained on Cityscapes, given every RGB frame of the clip."""
    from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor

    name = "facebook/maskformer-resnet101-cityscapes"
    return (MaskFormerImageProcessor.from_pretrained(name),
            MaskFormerForInstanceSegmentation.from_pretrained(name).eval().to(device))


def load_da3_baseline(weights: str, device: str):
    """Depth Anything 3, a depth-only baseline given every frame of the clip.

    Built from the released config rather than the package's high-level API, which pulls in
    video-export dependencies this script does not need.
    """
    try:
        from depth_anything_3.cfg import create_object
    except ImportError as exc:  # pragma: no cover - depends on an optional package
        raise SystemExit(
            "The da3 baseline needs the depth-anything-3 package "
            "(pip install --no-deps git+https://github.com/ByteDance-Seed/Depth-Anything-3.git)."
        ) from exc
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = Path(weights) if weights else Path(snapshot_download("depth-anything/DA3METRIC-LARGE"))
    model = create_object(json.loads((path / "config.json").read_text())["config"])
    state = load_file(path / "model.safetensors")
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    return model.eval().to(device)


def load_vggt_baseline(device: str):
    """VGGT, a depth/pose baseline that -- like pi3 -- is given every frame of the clip."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:  # pragma: no cover - depends on an optional package
        raise SystemExit(
            "The vggt baseline needs the vggt package "
            "(pip install git+https://github.com/facebookresearch/vggt.git)."
        ) from exc
    return VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)


def as_frames(images: list[np.ndarray]) -> list[Frame]:
    """Wrap decoded RGB arrays in the Frame records the preprocessing expects."""
    return [Frame(rgb=image, source=f"frame_{index}", frame_index=index)
            for index, image in enumerate(images)]


NEEDS_CHECKPOINT = {"lfg"}
SEGMENTATION_ONLY = {"segformer", "maskformer", "static"}


def build_predictor(args: argparse.Namespace):
    """Return (predict, future_start, description).

    `predict` maps a clip's image paths to model outputs. `future_start` is the first slot
    the model had to predict rather than observe: LFG only sees its history, whereas the
    pi3 baseline is given every frame, so for pi3 the split marks the same frames the paper
    reports as "predicted" even though pi3 observed them.
    """
    if args.model in NEEDS_CHECKPOINT and not args.checkpoint:
        raise SystemExit(f"--checkpoint is required for --model {args.model}.")

    if args.model == "pi3":
        model = load_pi3_baseline(args.checkpoint, args.device)

        def predict(images: list[np.ndarray]) -> dict:
            batch = preprocess_frames(
                as_frames(images), target_size=TARGET_WIDTH, mode="crop",
                keep_ratio=False, patch_size=14,
            )
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return model(batch.unsqueeze(0).to(args.device))

        return predict, CLIP_LENGTH // 2, {"model": "pi3", "frames_seen": CLIP_LENGTH}

    if args.model == "segformer":
        processor, model = load_segformer_baseline(args.device)

        mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

        def predict(images: list[np.ndarray]) -> dict:
            # Fed at the same input resolution as LFG so the comparison is like for like,
            # rather than at the processor's much larger default.
            batch = preprocess_frames(
                as_frames(images), target_size=TARGET_WIDTH, mode="crop",
                keep_ratio=False, patch_size=14,
            )
            pixels = ((batch - mean) / std).to(args.device)
            with torch.inference_mode():
                logits = model(pixel_values=pixels).logits      # (S, 19, h, w)
            classes = torch.full(
                (logits.shape[0], len(CLASS_NAMES), *logits.shape[-2:]), -1e4, device=logits.device
            )
            for train_id, index in CITYSCAPES_TRAIN_ID_TO_CLASS.items():
                classes[:, index] = torch.maximum(classes[:, index], logits[:, train_id])
            return {"segmentation": classes.permute(0, 2, 3, 1).unsqueeze(0)}

        return predict, CLIP_LENGTH // 2, {"model": "segformer", "frames_seen": CLIP_LENGTH}

    if args.model == "da3":
        model = load_da3_baseline(args.checkpoint, args.device)

        def predict(images: list[np.ndarray]) -> dict:
            batch = preprocess_frames(
                as_frames(images), target_size=TARGET_WIDTH, mode="crop",
                keep_ratio=False, patch_size=14,
            ).unsqueeze(0).to(args.device)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                depth = model(batch)["depth"]                   # (1, S, H, W)
            points = torch.zeros(*depth.shape, 3, device=depth.device)
            points[..., 2] = depth
            return {"local_points": points}                     # depth only: no pose head

        return predict, CLIP_LENGTH // 2, {"model": "da3", "frames_seen": CLIP_LENGTH}

    if args.model == "maskformer":
        processor, model = load_maskformer_baseline(args.device)
        mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

        def predict(images: list[np.ndarray]) -> dict:
            batch = preprocess_frames(
                as_frames(images), target_size=TARGET_WIDTH, mode="crop",
                keep_ratio=False, patch_size=14,
            )
            height, width = batch.shape[-2:]
            with torch.inference_mode():
                outputs = model(pixel_values=((batch - mean) / std).to(args.device))
            maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(height, width)] * len(images))
            # post-processing returns label maps, so express them as logits for the scorer
            logits = torch.zeros(len(images), height, width, len(CLASS_NAMES))
            for frame, labels in enumerate(maps):
                labels = labels.cpu()
                for train_id, index in CITYSCAPES_TRAIN_ID_TO_CLASS.items():
                    logits[frame, :, :, index][labels == train_id] = 1.0
            return {"segmentation": logits.unsqueeze(0)}

        return predict, CLIP_LENGTH // 2, {"model": "maskformer", "frames_seen": CLIP_LENGTH}

    if args.model == "static":
        # No prediction of its own: SegFormer segments the last observed frame and that map
        # stands in for every future frame, measuring how much of the future is the present.
        processor, model = load_segformer_baseline(args.device)

        mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std).view(1, 3, 1, 1)

        def predict(images: list[np.ndarray]) -> dict:
            observed = CLIP_LENGTH // 2
            batch = preprocess_frames(
                as_frames(images[:observed]), target_size=TARGET_WIDTH, mode="crop",
                keep_ratio=False, patch_size=14,
            )
            pixels = ((batch - mean) / std).to(args.device)
            with torch.inference_mode():
                logits = model(pixel_values=pixels).logits
            classes = torch.full(
                (logits.shape[0], len(CLASS_NAMES), *logits.shape[-2:]), -1e4, device=logits.device
            )
            for train_id, index in CITYSCAPES_TRAIN_ID_TO_CLASS.items():
                classes[:, index] = torch.maximum(classes[:, index], logits[:, train_id])
            frozen = classes[observed - 1].expand(CLIP_LENGTH, -1, -1, -1)
            return {"segmentation": frozen.permute(0, 2, 3, 1).unsqueeze(0), "static": True}

        return predict, CLIP_LENGTH // 2, {"model": "static", "frames_seen": CLIP_LENGTH // 2}

    if args.model == "vggt":
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model = load_vggt_baseline(args.device)

        def predict(images: list[np.ndarray]) -> dict:
            batch = preprocess_frames(
                as_frames(images), target_size=TARGET_WIDTH, mode="crop",
                keep_ratio=False, patch_size=14,
            ).unsqueeze(0).to(args.device)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
            depth = outputs["depth"][..., 0]                        # (1, S, H, W)
            points = torch.zeros(*depth.shape, 3, device=depth.device)
            points[..., 2] = depth
            extrinsic, _ = pose_encoding_to_extri_intri(outputs["pose_enc"], batch.shape[-2:])
            square = torch.eye(4, device=depth.device).repeat(1, extrinsic.shape[1], 1, 1)
            square[:, :, :3, :4] = extrinsic                        # world-to-camera
            return {"local_points": points, "camera_poses": torch.linalg.inv(square)}

        return predict, CLIP_LENGTH // 2, {"model": "vggt", "frames_seen": CLIP_LENGTH}

    model, config, _, _ = load_model_from_checkpoint(args.checkpoint, device=args.device)
    if config.m + config.n < CLIP_LENGTH:
        print(
            f"Warning: checkpoint predicts {config.m}+{config.n} frames; "
            f"only those are scored against the {CLIP_LENGTH}-frame clips."
        )

    def predict(images: list[np.ndarray]) -> dict:
        return predict_window(
            model, as_frames(images[: config.m]), config, device=args.device,
            target_size=TARGET_WIDTH, resize_mode="crop", keep_ratio=False,
        )

    return predict, config.m, {"model": "lfg", **config.to_dict()}


def evaluate(args: argparse.Namespace) -> dict:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    root = Path(args.data_root)
    expected = {"kitti360": ("data_2d_raw", "calibration"),
                "waymo": ("camera_image", "lidar", "camera_calibration")}[args.dataset]
    missing = [name for name in expected if not (root / name).is_dir()]
    if missing:
        raise SystemExit(
            f"{root} does not look like a {args.dataset} root: missing {', '.join(missing)}. "
            "See the Evaluation section of the README for the expected layout."
        )
    clips = list(ADAPTERS[args.dataset](root, MAX_DEPTH, args.frame_stride))
    if not clips:
        raise SystemExit(f"No usable {CLIP_LENGTH}-frame clips found under {args.data_root}")
    wanted = [line.strip() for line in Path(args.clip_list).read_text().splitlines() if line.strip()]
    by_name = {clip.name: clip for clip in clips}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        sequences = sorted({name.split(":")[0] for name in missing})
        raise SystemExit(
            f"{len(missing)} of {len(wanted)} clips in {args.clip_list} are not under "
            f"{root}. Sequences needed: {', '.join(sequences)}"
        )
    if args.model in SEGMENTATION_ONLY and not any(clip.load_labels for clip in clips):
        raise SystemExit(
            f"{args.dataset} has no semantic labels, so the {args.model} baseline has nothing "
            "to score. Segmentation is evaluated on KITTI-360."
        )

    # Score sequence by sequence: the metrics are order-independent, but reading a clip is far
    # cheaper when the sequence it belongs to is already the one held in memory.
    clips = sorted((by_name[name] for name in wanted), key=lambda clip: clip.name)
    predict, future_start, model_info = build_predictor(args)
    print(f"{len(clips)} clips from {args.clip_list}")

    scored: dict[str, list[float]] = {k: [] for k in ("overall_absrel", "overall_rmse",
                                                      "predicted_absrel", "predicted_rmse",
                                                      "observed_absrel", "observed_rmse",
                                                      "overall_delta1", "predicted_delta1",
                                                      "observed_delta1",
                                                      "trajectory_ate", "trajectory_rot",
                                                      "trajectory_trans_pct")
                                     + tuple(f"{split}_{metric}"
                                             for split in ("overall", "predicted")
                                             for metric in ("pa", "miou"))}
    for clip in tqdm(clips, desc="evaluating"):
        images = clip.load_images()
        outputs = predict(images)
        if "local_points" in outputs:
            predicted = outputs["local_points"][0].float().cpu().numpy()[..., 2]
            slots = min(CLIP_LENGTH, predicted.shape[0])
            height, width = predicted.shape[1], predicted.shape[2]
        else:  # segmentation-only baselines: score on the same grid the models use
            predicted = None
            slots = CLIP_LENGTH
            source_h, source_w = images[0].shape[:2]
            width = TARGET_WIDTH
            height = max(14, round(source_h * (width / source_w) / 14) * 14)

        if clip.load_poses is not None and "camera_poses" in outputs:
            try:
                gt_poses = clip.load_poses()[:slots]
                relative = np.linalg.inv(gt_poses[0]) @ gt_poses
                if np.linalg.norm(relative[-1, :3, 3]) < MIN_TRAVEL:
                    raise KeyError("stationary clip")      # no trajectory to score
                pred_poses = outputs["camera_poses"][0].float().cpu().numpy()[:slots]
                ate, rot, trans_pct = trajectory_errors(pred_poses, gt_poses)
                scored["trajectory_ate"].append(ate)
                scored["trajectory_rot"].append(rot)
                scored["trajectory_trans_pct"].append(trans_pct)
            except (KeyError, OSError):     # this clip has no ground-truth poses
                pass

        if clip.load_labels is not None and "segmentation" in outputs:
            raw = outputs["segmentation"][0].permute(0, 3, 1, 2).float()       # (S, C, h, w)
            if raw.shape[-2:] != (height, width):
                raw = torch.nn.functional.interpolate(
                    raw, size=(height, width), mode="bilinear", align_corners=False
                )
            logits = raw.permute(0, 2, 3, 1).cpu().numpy()[:slots]
            per_frame_seg = []
            for slot in range(slots):
                try:
                    labels = clip.load_labels(slot, height, width)
                except (OSError, ValueError):
                    continue
                prediction = logits[slot].argmax(-1)
                per_frame_seg.append(frame_segmentation_metrics(prediction, labels))

            seg = np.array(per_frame_seg, dtype=float) if per_frame_seg else np.zeros((0, 4))
            # The static baseline repeats its own frame-3 output, so scoring the observed
            # frames would just restate the SegFormer row.
            splits = (("predicted", seg[future_start:]),) if outputs.get("static") else \
                     (("overall", seg), ("predicted", seg[future_start:]))
            for split, rows in splits:
                if len(rows) and np.isfinite(rows[:, 0]).any():
                    for column, metric in enumerate(("pa", "miou")):
                        scored[f"{split}_{metric}"].append(np.nanmean(rows[:, column]))

        if predicted is None:
            continue
        truth = [load_depth_cached(clip, slot, height, width, cache_dir, MAX_DEPTH,
                                   args.frame_stride)
                 for slot in range(slots)]
        masks = [np.isfinite(gt) & (np.nan_to_num(gt) > 0) for gt in truth]

        # Point maps are defined up to one scale and shift per clip, so the alignment is
        # fitted once over the whole clip rather than per frame.
        stacked_mask = np.stack(masks)
        if stacked_mask.sum() < MIN_VALID_POINTS:
            continue
        aligned = list(affine_align(predicted[:slots], np.nan_to_num(np.stack(truth)), stacked_mask))

        per_frame = []
        for slot in range(slots):
            if masks[slot].sum() < MIN_VALID_POINTS or not np.isfinite(aligned[slot]).any():
                per_frame.append((np.nan, np.nan, np.nan))
                continue
            per_frame.append(depth_errors(aligned[slot], truth[slot], masks[slot]))

        errors = np.array(per_frame, dtype=float)
        if not np.isfinite(errors[:, 0]).any():
            continue
        for split, rows in (("overall", errors), ("observed", errors[:future_start]),
                            ("predicted", errors[future_start:])):
            if len(rows) and np.isfinite(rows[:, 0]).any():
                for column, metric in enumerate(("absrel", "rmse", "delta1")):
                    scored[f"{split}_{metric}"].append(np.nanmean(rows[:, column]))

    results = {name: summarize(values) for name, values in scored.items()}
    return {
        "checkpoint": Path(args.checkpoint).name,
        "dataset": args.dataset,
        "clips_scored": max(entry["count"] for entry in results.values()),
        "clips_evaluated": len(clips),
        "clip_list": args.clip_list,
        "frame_stride": args.frame_stride,
        "max_depth": MAX_DEPTH,
        "model": model_info,
        "metrics": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Path to the model weights. Required for lfg; baselines fetch their own.",
    )
    parser.add_argument(
        "--model",
        choices=["lfg", "pi3", "vggt", "da3", "segformer", "maskformer", "static"],
        default="lfg",
        help="lfg (default) sees only the history; the pi3 baseline is given every frame.",
    )
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS), help="Dataset adapter.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument(
        "--clip-list",
        default="eval/clips/kitti360_200.txt",
        help="File of clip names, one per line, formatted <sequence>:<first frame>.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Spacing between the six frames of a clip, in source frames. Both datasets "
             "record at 10 Hz, so 5 gives a 2 Hz clip.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Reuse decoded ground-truth depth across runs. Recommended for Waymo.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Write results JSON here.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args)
    metrics = results["metrics"]
    print(f"\n{results['clips_scored']} clips scored ({results['dataset']})")
    if metrics["overall_absrel"]["count"]:
        print(f"{'split':<12}{'AbsRel':>18}{'RMSE (m)':>18}{'delta<1.25':>14}")
        for split in ("overall", "observed", "predicted"):
            absrel, rmse = metrics[f"{split}_absrel"], metrics[f"{split}_rmse"]
            delta = metrics[f"{split}_delta1"]
            print(f"{split:<12}{absrel['mean']:>10.3f} ±{absrel['std']:<6.3f}"
                  f"{rmse['mean']:>10.2f} ±{rmse['std']:<6.2f}{delta['mean']:>13.3f}")

    scored_splits = [s for s in ("overall", "predicted") if metrics[f"{s}_pa"]["count"]]
    if scored_splits:
        print(f"\n{'segmentation':<12}{'PA':>10}{'mIoU':>10}")
        for split in scored_splits:
            row = "".join(f"{metrics[f'{split}_{m}']['mean']:>10.3f}"
                          for m in ("pa", "miou"))
            print(f"{split:<12}{row}")

    if metrics["trajectory_ate"]["count"]:
        ate, rot, pct = (metrics[f"trajectory_{k}"] for k in ("ate", "rot", "trans_pct"))
        print(f"\ntrajectory ({ate['count']} clips){'ATE (m)':>13}{'Rot (deg)':>18}{'Trans (%)':>18}")
        print(f"{'':<24}{ate['mean']:>7.2f} ±{ate['std']:<5.2f}{rot['mean']:>10.2f} ±{rot['std']:<6.2f}"
              f"{pct['mean']:>10.1f} ±{pct['std']:<6.1f}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
