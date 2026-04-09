#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sim_bridge.factory import create_bridge
from coord_transform_utils import airsim_ned_to_enu_points, local_points_to_world_enu
from progress_utils import ProgressBar, StageLogger
from pipeline_common import (
    append_unified_scene_log,
    build_unified_stage_event,
    build_unified_bridge_config,
    compute_airsim_lidar_range_profile,
    format_unified_startup_ports_message,
    prepare_airsim_runtime_unified,
    resolve_output_dir_name,
    resolve_scene_root,
)


@dataclass
class WorkerBinding:
    worker_id: int
    vehicle: str


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("config yaml root must be a mapping")
    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_bindings(config: dict[str, Any], worker_count: int) -> list[WorkerBinding]:
    parallel = config.get("parallel", {}) or {}
    raw_bindings = parallel.get("bindings", []) or []

    bindings: list[WorkerBinding] = []
    if raw_bindings:
        for item in raw_bindings:
            worker_id = int(item.get("worker_id"))
            vehicle = str(item.get("vehicle"))
            bindings.append(WorkerBinding(worker_id=worker_id, vehicle=vehicle))
    else:
        for worker_id in range(worker_count):
            bindings.append(WorkerBinding(worker_id=worker_id, vehicle=f"drone_{worker_id + 1}"))

    if len(bindings) < worker_count:
        next_id = len(bindings)
        for worker_id in range(next_id, worker_count):
            bindings.append(WorkerBinding(worker_id=worker_id, vehicle=f"drone_{worker_id + 1}"))

    return sorted(bindings, key=lambda item: item.worker_id)


def normalize_airsim_vehicle_name(name: str, fallback_index: int) -> str:
    candidate = str(name).strip()
    if not candidate:
        candidate = f"drone_{fallback_index + 1}"

    m = re.match(r"^drone[_-]?(\d+)$", candidate, flags=re.IGNORECASE)
    if m:
        return f"drone_{int(m.group(1))}"

    m = re.match(r"^Drone[_-]?(\d+)$", candidate)
    if m:
        return f"drone_{int(m.group(1))}"

    return candidate


def make_pose_list(config: dict[str, Any]) -> tuple[list[tuple[float, float, float, float, float, float]], dict[str, Any]]:
    collect_cfg = config.get("collect", {}) or {}
    traj_map = config.get("traj_map", {}) or {}

    map_bound = traj_map.get("MapBound", None)
    lidar_delta = traj_map.get("LidarDelta", None)
    if not (isinstance(map_bound, (list, tuple)) and len(map_bound) >= 6):
        raise ValueError("traj_map.MapBound must be a list of 6 numbers: [x_min, x_max, y_min, y_max, z_min, z_max]")
    if not (isinstance(lidar_delta, (list, tuple)) and len(lidar_delta) >= 3):
        raise ValueError("traj_map.LidarDelta must be a list of 3 numbers: [dx, dy, dz]")

    x_min = float(map_bound[0])
    x_max = float(map_bound[1])
    y_min = float(map_bound[2])
    y_max = float(map_bound[3])
    z_min = float(map_bound[4])
    z_max = float(map_bound[5])
    dx = float(lidar_delta[0])
    dy = float(lidar_delta[1])
    dz = float(lidar_delta[2])

    if not (x_max > x_min and y_max > y_min and z_max > z_min):
        raise ValueError("Invalid traj_map.MapBound: max must be greater than min for x/y/z")
    if not (dx > 0 and dy > 0 and dz > 0):
        raise ValueError("Invalid traj_map.LidarDelta: dx/dy/dz must be > 0")

    x_coords = np.arange(x_min, x_max, dx, dtype=np.float32)
    y_coords = np.arange(y_min, y_max, dy, dtype=np.float32)
    z_coords = np.arange(z_min, z_max, dz, dtype=np.float32)
    if len(x_coords) == 0 or len(y_coords) == 0 or len(z_coords) == 0:
        raise ValueError("Empty sampling grid: check MapBound and LidarDelta")

    yaws_raw = collect_cfg.get("yaw_list_deg", [0.0])
    yaws = [float(value) for value in yaws_raw] if isinstance(yaws_raw, (list, tuple)) and len(yaws_raw) > 0 else [0.0]

    poses: list[tuple[float, float, float, float, float, float]] = []
    for z in z_coords:
        for x in x_coords:
            for y in y_coords:
                for yaw in yaws:
                    poses.append((float(x), float(y), float(z), float(yaw), 0.0, 0.0))

    meta = {
        "sampling_mode": "traj_map_3d_grid",
        "bounds_source": "traj_map.MapBound",
        "sampling_bounds_xyz": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
        },
        "lidar_delta_xyz": {"dx": dx, "dy": dy, "dz": dz},
        "x_count": int(len(x_coords)),
        "y_count": int(len(y_coords)),
        "z_count": int(len(z_coords)),
        "yaw_values_deg": [float(value) for value in yaws],
    }
    return poses, meta


def resolve_max_frames_per_worker(
    config: dict[str, Any],
    args: argparse.Namespace,
    total_pose_count: int,
    worker_count: int,
) -> tuple[int, bool]:
    if args.max_frames is not None:
        value = max(1, int(args.max_frames))
        return value, False

    collect_cfg = config.get("collect", {}) or {}
    from_cfg = collect_cfg.get("max_frames_per_worker", None)
    if from_cfg is not None:
        value = max(1, int(from_cfg))
        return value, False

    auto_value = max(1, int(math.ceil(float(total_pose_count) / float(worker_count))))
    return auto_value, True


def split_tasks_round_robin(
    poses: list[tuple[float, float, float, float, float, float]],
    worker_count: int,
) -> list[list[tuple[float, float, float, float, float, float]]]:
    shards: list[list[tuple[float, float, float, float, float, float]]] = [list() for _ in range(worker_count)]
    for index, pose in enumerate(poses):
        shards[index % worker_count].append(pose)
    return shards


def sample_pose_shard_uniform(
    shard: list[tuple[float, float, float, float, float, float]],
    max_frames: int,
) -> list[tuple[float, float, float, float, float, float]]:
    if max_frames <= 0 or len(shard) <= max_frames:
        return shard

    indices = np.linspace(0, len(shard) - 1, num=max_frames, dtype=np.int64)
    unique_indices: list[int] = []
    seen: set[int] = set()
    for idx in indices.tolist():
        i = int(idx)
        if i not in seen:
            seen.add(i)
            unique_indices.append(i)

    return [shard[i] for i in unique_indices]


def airsim_ned_to_enu(points_xyz: np.ndarray) -> np.ndarray:
    return airsim_ned_to_enu_points(points_xyz)


def normalize_lidar(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)

    payload = points
    if isinstance(points, dict):
        payload = points.get("points", points.get("point_cloud", None))
        if payload is None:
            return np.empty((0, 3), dtype=np.float32)

    arr = np.asarray(payload, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] % 4 == 0:
            arr = arr.reshape(-1, 4)
        elif arr.shape[0] % 3 == 0:
            arr = arr.reshape(-1, 3)
        else:
            return np.empty((0, 3), dtype=np.float32)

    if arr.ndim < 2 or arr.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    return arr[:, :3]


def normalize_lidar_segmentation(points: Any, expected_count: int) -> np.ndarray:
    n = max(0, int(expected_count))
    if n == 0:
        return np.empty((0,), dtype=np.uint32)

    if not isinstance(points, dict):
        return np.zeros((n,), dtype=np.uint32)

    payload = None
    for key in ("segmentation", "semantic_raw", "segmentation_ids", "class_ids"):
        if key in points:
            payload = points.get(key)
            break

    if payload is None:
        return np.zeros((n,), dtype=np.uint32)

    try:
        arr = np.asarray(payload).reshape(-1).astype(np.uint32, copy=False)
    except Exception:
        return np.zeros((n,), dtype=np.uint32)

    if arr.size == n:
        return arr
    if arr.size > n:
        return arr[:n]

    out = np.zeros((n,), dtype=np.uint32)
    if arr.size > 0:
        out[: arr.size] = arr
    return out


def normalize_depth(depth: Any, depth_scale: float = 1.0) -> np.ndarray:
    if depth is None:
        return np.empty((0, 0), dtype=np.float32)

    if isinstance(depth, bytes):
        try:
            arr = np.load(io.BytesIO(depth))
            arr = np.asarray(arr, dtype=np.float32)
            return arr * float(depth_scale)
        except Exception:
            decoded = cv2.imdecode(np.frombuffer(depth, np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                return np.empty((0, 0), dtype=np.float32)
            return np.asarray(decoded, dtype=np.float32) * float(depth_scale)

    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    return arr * float(depth_scale)


def normalize_segmentation_raw(seg: Any) -> np.ndarray:
    if seg is None:
        return np.empty((0, 0), dtype=np.uint32)
    arr = np.asarray(seg)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.uint32)
    if arr.ndim == 2:
        return arr.astype(np.uint32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        b = arr[:, :, 0].astype(np.uint32)
        g = arr[:, :, 1].astype(np.uint32)
        r = arr[:, :, 2].astype(np.uint32)
        return b + (g << 8) + (r << 16)
    return np.empty((0, 0), dtype=np.uint32)


def depth_to_points_local_with_pixels(
    depth: np.ndarray,
    fov_deg: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    height, width = depth.shape
    if height <= 1 or width <= 1:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    step = max(1, int(sample_step))
    yy = np.arange(0, height, step, dtype=np.int32)
    xx = np.arange(0, width, step, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    z = depth[grid_y, grid_x]

    valid = np.isfinite(z) & (z >= float(min_depth)) & (z <= float(max_depth))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    x_valid = grid_x[valid].astype(np.float32)
    y_valid = grid_y[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)

    fov_rad = math.radians(float(fov_deg))
    fx = (width * 0.5) / math.tan(max(1e-6, fov_rad * 0.5))
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    x3 = (x_valid - cx) / fx * z_valid
    y3 = (y_valid - cy) / fy * z_valid
    pts = np.stack([x3, y3, z_valid], axis=1).astype(np.float32)
    return pts, grid_x[valid].astype(np.int32), grid_y[valid].astype(np.int32)


def _nearest_attribute_transfer(
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_colors: np.ndarray,
    source_labels: np.ndarray,
    max_dist_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(target_points.shape[0])
    colors = np.zeros((n, 3), dtype=np.uint8)
    labels = np.zeros((n,), dtype=np.uint32)
    conf = np.zeros((n,), dtype=np.float32)
    if n == 0 or source_points.shape[0] == 0:
        return colors, labels, conf

    max_dist = float(max(1e-3, max_dist_m))
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(source_points)
        dist, idx = tree.query(target_points, k=1)
        valid = np.isfinite(dist) & (dist <= max_dist)
        if np.any(valid):
            ii = idx[valid].astype(np.int64)
            colors[valid] = source_colors[ii]
            labels[valid] = source_labels[ii]
            conf[valid] = np.clip(1.0 - (dist[valid] / max_dist), 0.0, 1.0).astype(np.float32)
        return colors, labels, conf
    except Exception:
        chunk = 2048
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            q = target_points[start:end]
            d2 = np.sum((q[:, None, :] - source_points[None, :, :]) ** 2, axis=2)
            nn = np.argmin(d2, axis=1)
            dd = np.sqrt(np.take_along_axis(d2, nn[:, None], axis=1)[:, 0])
            valid = dd <= max_dist
            if np.any(valid):
                rows = np.where(valid)[0]
                ii = nn[rows]
                colors[start:end][rows] = source_colors[ii]
                labels[start:end][rows] = source_labels[ii]
                conf[start:end][rows] = np.clip(1.0 - (dd[rows] / max_dist), 0.0, 1.0).astype(np.float32)
        return colors, labels, conf


def fuse_lidar_with_multimodal_frame(
    lidar_points_enu: np.ndarray,
    frame_rgb: Any,
    frame_depth: Any,
    frame_seg: Any,
    pose_enu: tuple[float, float, float, float, float, float],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lidar_points_enu.size == 0:
        return (
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.float32),
        )

    mm_cfg = ((config.get("collect", {}) or {}).get("multimodal_fusion", {}) or {})
    depth_cfg = config.get("depth_backproject", {}) or {}

    rgb = normalize_rgb(frame_rgb)
    depth = normalize_depth(frame_depth, depth_scale=float(depth_cfg.get("depth_scale", 1.0)))
    seg_raw = normalize_segmentation_raw(frame_seg)
    if rgb.size == 0 or depth.size == 0 or seg_raw.size == 0:
        n = lidar_points_enu.shape[0]
        return np.zeros((n, 3), dtype=np.uint8), np.zeros((n,), dtype=np.uint32), np.zeros((n,), dtype=np.float32)

    if rgb.shape[:2] != depth.shape[:2]:
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
    if seg_raw.shape[:2] != depth.shape[:2]:
        seg_raw = cv2.resize(seg_raw.astype(np.uint32), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    pts_local, px, py = depth_to_points_local_with_pixels(
        depth=depth,
        fov_deg=float(depth_cfg.get("fov_deg", config.get("camera", {}).get("fov", 90.0))),
        sample_step=int(depth_cfg.get("sample_step", 4)),
        min_depth=float(depth_cfg.get("min_depth", 0.3)),
        max_depth=float(depth_cfg.get("max_depth", 120.0)),
    )
    if pts_local.size == 0:
        n = lidar_points_enu.shape[0]
        return np.zeros((n, 3), dtype=np.uint8), np.zeros((n,), dtype=np.uint32), np.zeros((n,), dtype=np.float32)

    depth_world = local_to_world_enu(pts_local, pose_enu=pose_enu)
    src_colors = rgb[py, px][:, :3].astype(np.uint8)
    src_labels = seg_raw[py, px].astype(np.uint32)

    max_assign = float(mm_cfg.get("max_assign_dist_m", 1.2))
    return _nearest_attribute_transfer(
        target_points=lidar_points_enu,
        source_points=depth_world,
        source_colors=src_colors,
        source_labels=src_labels,
        max_dist_m=max_assign,
    )


def depth_to_points_local(
    depth: np.ndarray,
    fov_deg: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    height, width = depth.shape
    if height <= 1 or width <= 1:
        return np.empty((0, 3), dtype=np.float32)

    step = max(1, int(sample_step))
    yy = np.arange(0, height, step, dtype=np.int32)
    xx = np.arange(0, width, step, dtype=np.int32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    z = depth[grid_y, grid_x]

    valid = np.isfinite(z) & (z >= float(min_depth)) & (z <= float(max_depth))
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    x_valid = grid_x[valid].astype(np.float32)
    y_valid = grid_y[valid].astype(np.float32)
    z_valid = z[valid].astype(np.float32)

    fov_rad = math.radians(float(fov_deg))
    fx = (width * 0.5) / math.tan(max(1e-6, fov_rad * 0.5))
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    x3 = (x_valid - cx) / fx * z_valid
    y3 = (y_valid - cy) / fy * z_valid
    pts = np.stack([x3, y3, z_valid], axis=1).astype(np.float32)
    return pts


def local_to_world_enu(points_local: np.ndarray, pose_enu: tuple[float, float, float, float, float, float]) -> np.ndarray:
    return local_points_to_world_enu(points_local, pose_enu)

def normalize_rgb(rgb: Any) -> np.ndarray:
    if rgb is None:
        return np.empty((0, 0, 3), dtype=np.uint8)

    arr = np.asarray(rgb)
    if arr.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    if arr.ndim == 2:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGRA2BGR)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return arr.astype(np.uint8)


def render_lidar_bev(
    points: np.ndarray,
    width: int,
    height: int,
    range_xy_m: float,
    max_points: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)

    for grid_step in (0.25, 0.5, 0.75):
        x = int(width * grid_step)
        y = int(height * grid_step)
        cv2.line(canvas, (x, 0), (x, height - 1), (34, 34, 34), 1)
        cv2.line(canvas, (0, y), (width - 1, y), (34, 34, 34), 1)

    cv2.line(canvas, (width // 2, 0), (width // 2, height - 1), (80, 80, 80), 1)
    cv2.line(canvas, (0, height // 2), (width - 1, height // 2), (80, 80, 80), 1)

    if points.size == 0:
        cv2.putText(canvas, 'No LiDAR points', (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2, cv2.LINE_AA)
        return canvas

    pts = points
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    radial = np.sqrt(x * x + y * y)
    if radial.size > 0:
        auto_half = float(np.percentile(radial, 92.0))
    else:
        auto_half = 0.0
    half = max(20.0, min(max(25.0, auto_half * 1.25), float(range_xy_m)))

    u = ((x / half) * 0.5 + 0.5) * float(width - 1)
    v = (0.5 - (y / half) * 0.5) * float(height - 1)

    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        cv2.putText(canvas, 'LiDAR out of BEV range', (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        return canvas

    u_valid = u[valid].astype(np.int32)
    v_valid = v[valid].astype(np.int32)
    z_valid = z[valid]

    z_min = float(np.min(z_valid))
    z_max = float(np.max(z_valid))
    denom = max(1e-6, z_max - z_min)
    z_norm = (z_valid - z_min) / denom

    layers = np.zeros((height, width), dtype=np.uint8)
    intensity = (64.0 + 191.0 * z_norm).astype(np.uint8)
    layers[v_valid, u_valid] = np.maximum(layers[v_valid, u_valid], intensity)
    layers = cv2.dilate(layers, np.ones((3, 3), np.uint8), iterations=1)
    colorized = cv2.applyColorMap(layers, cv2.COLORMAP_TURBO)

    mask = layers > 0
    canvas[mask] = colorized[mask]

    cv2.putText(canvas, f'points={int(points.shape[0])}', (24, height - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'bev_half={half:.1f}m', (24, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return canvas


def compose_sampling_frame(
    rgb: Any,
    lidar_enu: np.ndarray,
    output_width: int,
    output_height: int,
    lidar_bev_range_m: float,
    lidar_max_points: int,
    frame_idx: int | None = None,
    total_frames: int | None = None,
    worker_id: int | None = None,
    vehicle_or_actor: str | None = None,
    pose_enu: tuple[float, float, float, float, float, float] | None = None,
    point_source: str | None = None,
) -> np.ndarray:
    panel_width = max(320, int(output_width // 2))
    panel_height = max(240, int(output_height))

    rgb_img = normalize_rgb(rgb)
    if rgb_img.size == 0:
        rgb_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        cv2.putText(rgb_panel, 'RGB unavailable', (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 180, 255), 2, cv2.LINE_AA)
    else:
        rgb_panel = cv2.resize(rgb_img, (panel_width, panel_height), interpolation=cv2.INTER_LINEAR)

    lidar_panel = render_lidar_bev(
        points=lidar_enu,
        width=panel_width,
        height=panel_height,
        range_xy_m=lidar_bev_range_m,
        max_points=lidar_max_points,
    )

    frame = np.concatenate([rgb_panel, lidar_panel], axis=1)
    cv2.putText(frame, 'RGB', (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'LiDAR BEV', (panel_width + 24, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    overlay_lines: list[str] = []
    if frame_idx is not None and total_frames is not None and total_frames > 0:
        progress = (float(frame_idx + 1) / float(total_frames)) * 100.0
        overlay_lines.append(f'progress={frame_idx + 1}/{total_frames} ({progress:.1f}%)')
    if worker_id is not None:
        overlay_lines.append(f'worker={worker_id}')
    if vehicle_or_actor:
        overlay_lines.append(f'vehicle={vehicle_or_actor}')
    if pose_enu is not None:
        x, y, z, yaw, pitch, roll = pose_enu
        overlay_lines.append(f'pose_enu x={x:.1f} y={y:.1f} z={z:.1f}')
        overlay_lines.append(f'attitude yaw={yaw:.1f} pitch={pitch:.1f} roll={roll:.1f}')
    if point_source:
        overlay_lines.append(f'source={point_source}')

    if overlay_lines:
        line_height = 26
        box_top = max(54, panel_height - (line_height * len(overlay_lines) + 24))
        box_left = 20
        box_right = min(frame.shape[1] - 20, int(frame.shape[1] * 0.62))
        box_bottom = box_top + line_height * len(overlay_lines) + 12

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_left, box_top), (box_right, box_bottom), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

        for idx, line in enumerate(overlay_lines):
            y_text = box_top + 26 + idx * line_height
            cv2.putText(frame, line, (box_left + 12, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

    return frame


def build_bridge_config(
    config: dict[str, Any],
    engine: str,
    sim_port: int,
    worker_binding: WorkerBinding,
    vehicle_names: list[str] | None = None,
) -> dict[str, Any]:
    bridge_cfg = build_unified_bridge_config(
        config=config,
        engine=str(engine),
        vehicle_name=str(worker_binding.vehicle),
        sim_port=int(sim_port),
        vehicle_names=[str(item) for item in vehicle_names] if vehicle_names else None,
        default_width=3840,
        default_height=2160,
        default_fov=90.0,
    )
    if str(engine).strip().lower() == "airsim":
        lidar_profile = compute_airsim_lidar_range_profile(config)
        bridge_cfg["lidar_range"] = float(lidar_profile["range_m"])
    return bridge_cfg


def build_depth_points(
    bridge: Any,
    pose: tuple[float, float, float, float, float, float],
    config: dict[str, Any],
) -> np.ndarray:
    depth_cfg = config.get("depth_backproject", {}) or {}
    depth = normalize_depth(bridge.capture_depth(), depth_scale=float(depth_cfg.get("depth_scale", 1.0)))
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pts_local = depth_to_points_local(
        depth=depth,
        fov_deg=float(depth_cfg.get("fov_deg", 90.0)),
        sample_step=int(depth_cfg.get("sample_step", 4)),
        min_depth=float(depth_cfg.get("min_depth", 0.3)),
        max_depth=float(depth_cfg.get("max_depth", 120.0)),
    )
    return local_to_world_enu(pts_local, pose_enu=pose)


def collect_worker(
    config: dict[str, Any],
    engine: str,
    scene_id: str,
    sim_port: int,
    worker_binding: WorkerBinding,
    worker_poses: list[tuple[float, float, float, float, float, float]],
    root_dir: Path,
    shared_frames: list[dict[str, Any]],
    shared_lock: threading.Lock,
    stop_on_error: bool,
) -> dict[str, Any]:
    worker_dir = root_dir / "raw" / engine / f"worker_{worker_binding.worker_id}"
    ensure_dir(worker_dir)

    collect_cfg = config.get("collect", {}) or {}
    pose_settle_sec = max(0.0, float(collect_cfg.get("pose_settle_sec", 0.05)))
    multimodal_cfg = collect_cfg.get("multimodal_fusion", {}) or {}
    multimodal_enabled = bool(multimodal_cfg.get("enabled", False))
    lidar_seg_cfg = collect_cfg.get("lidar_segmentation", {}) or {}
    lidar_seg_enabled = bool(lidar_seg_cfg.get("enabled", engine == "airsim")) and engine == "airsim"

    bridge_cfg = build_bridge_config(config=config, engine=engine, sim_port=sim_port, worker_binding=worker_binding)
    if engine == "airsim":
        bridge_cfg["sim_port"] = int(sim_port)
        bridge_cfg["launch_sim"] = False
        bridge_cfg["connect_on_init"] = True
        bridge_cfg["auto_select_port_on_conflict"] = False
        if lidar_seg_enabled:
            bridge_cfg["lidar_segmentation_enabled"] = True
    bridge = create_bridge(engine=engine, scene_id=scene_id, config=bridge_cfg)

    rows: list[dict[str, Any]] = []
    point_count_total = 0

    engine_cfg = (config.get("engine_params", {}) or {}).get(engine, {}) or {}
    default_depth_fallback = False if engine == "airsim" else True
    depth_fallback_enabled = bool(engine_cfg.get("depth_fallback_when_lidar_empty", default_depth_fallback))
    lidar_retry_count = max(1, int(engine_cfg.get("lidar_retry_count", 5 if engine == "airsim" else 1)))
    lidar_retry_interval_sec = max(0.0, float(engine_cfg.get("lidar_retry_interval_sec", 0.03 if engine == "airsim" else 0.0)))

    video_cfg = collect_cfg.get("sampling_video", {}) or {}
    sampling_video_enabled = bool(video_cfg.get("enabled", True))
    sampling_video_fps = max(1.0, float(video_cfg.get("fps", 10.0)))
    sampling_video_width = max(640, int(video_cfg.get("width", 1920)))
    sampling_video_height = max(360, int(video_cfg.get("height", 1080)))
    lidar_bev_range_m = max(20.0, float(video_cfg.get("lidar_bev_range_m", 120.0)))
    lidar_max_points = max(1000, int(video_cfg.get("lidar_max_points_per_frame", 20000)))

    video_dir = root_dir / "vis"
    sampling_video_path = video_dir / f"sampling_worker_{worker_binding.worker_id}.mp4"
    video_writer: cv2.VideoWriter | None = None

    try:
        for frame_idx, pose in enumerate(worker_poses):
            x, y, z, yaw, pitch, roll = pose
            try:
                bridge.set_uav_pose(
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    vehicle_or_actor=worker_binding.vehicle,
                )
                if pose_settle_sec > 0.0:
                    time.sleep(pose_settle_sec)

                lidar = np.empty((0, 3), dtype=np.float32)
                lidar_seg = np.empty((0,), dtype=np.uint32)
                source = "lidar"
                for retry_idx in range(lidar_retry_count):
                    lidar_packet = bridge.get_lidar()
                    lidar = normalize_lidar(lidar_packet)
                    if lidar_seg_enabled:
                        lidar_seg = normalize_lidar_segmentation(lidar_packet, expected_count=int(lidar.shape[0]))
                    if lidar.shape[0] > 0:
                        break
                    if retry_idx + 1 < lidar_retry_count and lidar_retry_interval_sec > 0.0:
                        time.sleep(lidar_retry_interval_sec)

                if lidar.shape[0] == 0:
                    if depth_fallback_enabled:
                        lidar = build_depth_points(bridge=bridge, pose=pose, config=config)
                        source = "depth_backproject"
                    else:
                        source = "lidar_empty"

                if engine == "airsim" and source == "lidar":
                    lidar = airsim_ned_to_enu(lidar)

                if source != "lidar" or not lidar_seg_enabled:
                    lidar_seg = np.zeros((lidar.shape[0],), dtype=np.uint32)
                elif lidar_seg.shape[0] != lidar.shape[0]:
                    aligned_seg = np.zeros((lidar.shape[0],), dtype=np.uint32)
                    copy_count = min(aligned_seg.shape[0], lidar_seg.shape[0])
                    if copy_count > 0:
                        aligned_seg[:copy_count] = lidar_seg[:copy_count]
                    lidar_seg = aligned_seg

                chunk_name = f"chunk_{frame_idx:06d}.npy"
                chunk_path = worker_dir / chunk_name
                np.save(chunk_path, lidar)

                if lidar_seg_enabled:
                    lidar_seg_chunk_path = worker_dir / f"chunk_{frame_idx:06d}.lidarseg.npz"
                    np.savez_compressed(
                        lidar_seg_chunk_path,
                        points=lidar.astype(np.float32),
                        semantic_raw=lidar_seg.astype(np.uint32),
                    )
                else:
                    lidar_seg_chunk_path = None

                frame_data = None
                if sampling_video_enabled or multimodal_enabled:
                    try:
                        frame_data = bridge.capture_frame(
                            include_rgb=True,
                            include_depth=multimodal_enabled,
                            include_seg=multimodal_enabled,
                            include_lidar=False,
                        )
                    except Exception:
                        frame_data = None

                if multimodal_enabled:
                    rgb_m = frame_data.rgb if frame_data is not None else None
                    depth_m = frame_data.depth if frame_data is not None else None
                    seg_m = frame_data.seg if frame_data is not None else None
                    colors, semantic_raw, semantic_conf = fuse_lidar_with_multimodal_frame(
                        lidar_points_enu=lidar,
                        frame_rgb=rgb_m,
                        frame_depth=depth_m,
                        frame_seg=seg_m,
                        pose_enu=pose,
                        config=config,
                    )
                    mm_chunk_path = worker_dir / f"chunk_{frame_idx:06d}.mm.npz"
                    np.savez_compressed(
                        mm_chunk_path,
                        points=lidar.astype(np.float32),
                        colors=colors.astype(np.uint8),
                        semantic_raw=semantic_raw.astype(np.uint32),
                        semantic_conf=semantic_conf.astype(np.float32),
                    )
                else:
                    mm_chunk_path = None
                    colors = np.zeros((lidar.shape[0], 3), dtype=np.uint8)
                    semantic_raw = np.zeros((lidar.shape[0],), dtype=np.uint32)
                    semantic_conf = np.zeros((lidar.shape[0],), dtype=np.float32)

                if sampling_video_enabled:
                    rgb_frame = frame_data.rgb if frame_data is not None else None
                    frame_img = compose_sampling_frame(
                        rgb=rgb_frame,
                        lidar_enu=lidar,
                        output_width=sampling_video_width,
                        output_height=sampling_video_height,
                        lidar_bev_range_m=lidar_bev_range_m,
                        lidar_max_points=lidar_max_points,
                        frame_idx=frame_idx,
                        total_frames=len(worker_poses),
                        worker_id=worker_binding.worker_id,
                        vehicle_or_actor=worker_binding.vehicle,
                        pose_enu=pose,
                        point_source=source,
                    )

                    if video_writer is None:
                        ensure_dir(video_dir)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(sampling_video_path),
                            fourcc,
                            sampling_video_fps,
                            (frame_img.shape[1], frame_img.shape[0]),
                        )
                    if video_writer is not None and video_writer.isOpened():
                        video_writer.write(frame_img)

                ts = time.time()
                row = {
                    "scene_id": scene_id,
                    "engine": engine,
                    "worker_id": worker_binding.worker_id,
                    "vehicle_or_actor": worker_binding.vehicle,
                    "frame_idx": frame_idx,
                    "timestamp": ts,
                    "pose_enu": {
                        "x": x,
                        "y": y,
                        "z": z,
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                    },
                    "lidar_path": str(chunk_path.as_posix()),
                    "point_count": int(lidar.shape[0]),
                    "point_source": source,
                    "lidar_segmentation_path": str(lidar_seg_chunk_path.as_posix()) if lidar_seg_chunk_path is not None else None,
                    "lidar_segmentation_count": int(np.count_nonzero(lidar_seg > 0)),
                    "multimodal_path": str(mm_chunk_path.as_posix()) if mm_chunk_path is not None else None,
                    "multimodal_colored_count": int(np.count_nonzero(np.any(colors > 0, axis=1))),
                    "multimodal_semantic_count": int(np.count_nonzero(semantic_raw > 0)),
                }
                rows.append(row)
                point_count_total += int(lidar.shape[0])

                with shared_lock:
                    shared_frames.append(row)
            except Exception as exc:
                error_row = {
                    "scene_id": scene_id,
                    "engine": engine,
                    "worker_id": worker_binding.worker_id,
                    "vehicle_or_actor": worker_binding.vehicle,
                    "frame_idx": frame_idx,
                    "error": str(exc),
                }
                rows.append(error_row)
                if stop_on_error:
                    raise
    finally:
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
        try:
            bridge.shutdown()
        except Exception:
            pass

    worker_manifest = worker_dir / f"manifest_worker_{worker_binding.worker_id}.jsonl"
    write_jsonl(worker_manifest, rows)

    return {
        "worker_id": worker_binding.worker_id,
        "vehicle": worker_binding.vehicle,
        "frames": len(worker_poses),
        "points": point_count_total,
        "manifest_path": str(worker_manifest.as_posix()),
        "sampling_video_path": str(sampling_video_path.as_posix()) if sampling_video_enabled and sampling_video_path.exists() else None,
    }


def gather_multimodal_chunks(root_dir: Path, engine: str) -> dict[str, np.ndarray]:
    chunks = sorted((root_dir / "raw" / engine).glob("worker_*/chunk_*.mm.npz"))
    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []

    for path in chunks:
        try:
            data = np.load(path)
            points = normalize_lidar(data.get("points"))
            colors = np.asarray(data.get("colors"), dtype=np.uint8)
            labels = np.asarray(data.get("semantic_raw"), dtype=np.uint32).reshape(-1)
            conf = np.asarray(data.get("semantic_conf"), dtype=np.float32).reshape(-1)
            n = points.shape[0]
            if n == 0:
                continue
            if colors.shape[0] != n:
                colors = np.zeros((n, 3), dtype=np.uint8)
            if labels.shape[0] != n:
                labels = np.zeros((n,), dtype=np.uint32)
            if conf.shape[0] != n:
                conf = np.zeros((n,), dtype=np.float32)
            all_points.append(points)
            all_colors.append(colors)
            all_labels.append(labels)
            all_conf.append(conf)
        except Exception:
            continue

    if not all_points:
        return {
            "points": np.empty((0, 3), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.uint8),
            "semantic_raw": np.empty((0,), dtype=np.uint32),
            "semantic_conf": np.empty((0,), dtype=np.float32),
        }

    return {
        "points": np.concatenate(all_points, axis=0),
        "colors": np.concatenate(all_colors, axis=0),
        "semantic_raw": np.concatenate(all_labels, axis=0),
        "semantic_conf": np.concatenate(all_conf, axis=0),
    }


def gather_lidar_segmentation_chunks(root_dir: Path, engine: str) -> dict[str, np.ndarray]:
    chunks = sorted((root_dir / "raw" / engine).glob("worker_*/chunk_*.lidarseg.npz"))
    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for chunk in chunks:
        try:
            data = np.load(chunk)
            points = normalize_lidar(data.get("points"))
            if points.shape[0] == 0:
                continue
            labels = np.asarray(data.get("semantic_raw"), dtype=np.uint32).reshape(-1)
            if labels.shape[0] != points.shape[0]:
                aligned = np.zeros((points.shape[0],), dtype=np.uint32)
                copy_count = min(aligned.shape[0], labels.shape[0])
                if copy_count > 0:
                    aligned[:copy_count] = labels[:copy_count]
                labels = aligned
            all_points.append(points)
            all_labels.append(labels)
        except Exception:
            continue

    if not all_points:
        return {
            "points": np.empty((0, 3), dtype=np.float32),
            "semantic_raw": np.empty((0,), dtype=np.uint32),
        }

    return {
        "points": np.concatenate(all_points, axis=0),
        "semantic_raw": np.concatenate(all_labels, axis=0),
    }


def voxel_dedup_with_indices(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0 or voxel_size <= 0:
        idx = np.arange(points.shape[0], dtype=np.int64)
        return points, idx
    vox = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(vox, axis=0, return_index=True)
    keep = np.sort(unique_indices)
    return points[keep], keep.astype(np.int64)


def write_ascii_pcd_xyzrgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    ensure_dir(path.parent)
    if points.shape[0] != colors.shape[0]:
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    with path.open("w", encoding="utf-8") as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z r g b\n")
        file.write("SIZE 4 4 4 1 1 1\n")
        file.write("TYPE F F F U U U\n")
        file.write("COUNT 1 1 1 1 1 1\n")
        file.write(f"WIDTH {points.shape[0]}\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {points.shape[0]}\n")
        file.write("DATA ascii\n")
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0])
            file.write(f"{x} {y} {z} {r} {g} {b}\n")


def write_ascii_pcd_instance(
    path: Path,
    points: np.ndarray,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    semantic_raw: np.ndarray,
    semantic_conf: np.ndarray,
    observed: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    n = int(points.shape[0])
    if class_ids.shape[0] != n:
        class_ids = np.zeros((n,), dtype=np.uint32)
    if instance_ids.shape[0] != n:
        instance_ids = np.zeros((n,), dtype=np.uint32)
    if semantic_raw.shape[0] != n:
        semantic_raw = np.zeros((n,), dtype=np.uint32)
    if semantic_conf.shape[0] != n:
        semantic_conf = np.zeros((n,), dtype=np.float32)
    if observed.shape[0] != n:
        observed = np.zeros((n,), dtype=np.uint8)

    with path.open("w", encoding="utf-8") as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z class_id instance_id semantic_raw semantic_conf observed\n")
        file.write("SIZE 4 4 4 4 4 4 4 1\n")
        file.write("TYPE F F F U U U F U\n")
        file.write("COUNT 1 1 1 1 1 1 1 1\n")
        file.write(f"WIDTH {n}\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {n}\n")
        file.write("DATA ascii\n")
        for i in range(n):
            x, y, z = points[i]
            file.write(
                f"{x} {y} {z} {int(class_ids[i])} {int(instance_ids[i])} "
                f"{int(semantic_raw[i])} {float(semantic_conf[i]):.6f} {int(observed[i])}\n"
            )


def write_ascii_ply_xyzrgb(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    ensure_dir(path.parent)
    if points.shape[0] != colors.shape[0]:
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0])
            file.write(f"{x} {y} {z} {r} {g} {b}\n")


def write_ascii_ply_semantic_instance(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    semantic_raw: np.ndarray,
    semantic_conf: np.ndarray,
    observed: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    n = int(points.shape[0])
    if colors.shape[0] != n:
        colors = np.zeros((n, 3), dtype=np.uint8)
    if class_ids.shape[0] != n:
        class_ids = np.zeros((n,), dtype=np.uint32)
    if instance_ids.shape[0] != n:
        instance_ids = np.zeros((n,), dtype=np.uint32)
    if semantic_raw.shape[0] != n:
        semantic_raw = np.zeros((n,), dtype=np.uint32)
    if semantic_conf.shape[0] != n:
        semantic_conf = np.zeros((n,), dtype=np.float32)
    if observed.shape[0] != n:
        observed = np.zeros((n,), dtype=np.uint8)

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {n}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("property uint class_id\n")
        file.write("property uint instance_id\n")
        file.write("property uint semantic_raw\n")
        file.write("property float semantic_conf\n")
        file.write("property uchar observed\n")
        file.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            r, g, b = int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0])
            file.write(
                f"{x} {y} {z} {r} {g} {b} {int(class_ids[i])} {int(instance_ids[i])} "
                f"{int(semantic_raw[i])} {float(semantic_conf[i]):.6f} {int(observed[i])}\n"
            )


def _neighbors26_key(key: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = key
    out: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                out.append((x + dx, y + dy, z + dz))
    return out


def extract_instances_from_semantic_points(
    points: np.ndarray,
    class_ids: np.ndarray,
    min_points: int,
    voxel_size: float,
) -> np.ndarray:
    out = np.zeros((points.shape[0],), dtype=np.uint32)
    if points.size == 0:
        return out

    next_id = 1
    classes = [int(c) for c in np.unique(class_ids) if int(c) > 0]
    for cid in classes:
        idxs = np.where(class_ids == cid)[0]
        if idxs.size == 0:
            continue
        vox_map: dict[tuple[int, int, int], list[int]] = {}
        for idx in idxs.tolist():
            p = points[idx]
            key = (
                int(math.floor(float(p[0]) / voxel_size)),
                int(math.floor(float(p[1]) / voxel_size)),
                int(math.floor(float(p[2]) / voxel_size)),
            )
            vox_map.setdefault(key, []).append(idx)

        visited: set[tuple[int, int, int]] = set()
        for key in list(vox_map.keys()):
            if key in visited:
                continue
            q = [key]
            visited.add(key)
            comp_vox: list[tuple[int, int, int]] = []
            head = 0
            while head < len(q):
                cur = q[head]
                head += 1
                comp_vox.append(cur)
                for nb in _neighbors26_key(cur):
                    if nb in visited or nb not in vox_map:
                        continue
                    visited.add(nb)
                    q.append(nb)
            pids: list[int] = []
            for v in comp_vox:
                pids.extend(vox_map[v])
            if len(pids) < int(min_points):
                continue
            arr = np.asarray(pids, dtype=np.int64)
            out[arr] = np.uint32(next_id)
            next_id += 1

    return out


def gather_raw_points(root_dir: Path, engine: str, show_progress: bool = False) -> np.ndarray:
    chunks = sorted((root_dir / "raw" / engine).glob("worker_*/chunk_*.npy"))
    all_points: list[np.ndarray] = []

    progress: ProgressBar | None = None
    if show_progress and len(chunks) > 0:
        progress = ProgressBar(total=len(chunks), label="step2.fuse")
        progress.update(0, detail="loading raw chunks")

    try:
        for idx, chunk in enumerate(chunks):
            points = np.load(chunk)
            points = normalize_lidar(points)
            if points.size > 0:
                all_points.append(points)
            if progress is not None:
                progress.update(idx + 1, detail=f"chunk={idx + 1}/{len(chunks)}")
    finally:
        if progress is not None:
            progress.finish(detail="raw chunk loading complete")

    if not all_points:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(all_points, axis=0)


def voxel_dedup(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.size == 0 or voxel_size <= 0:
        return points
    vox = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(vox, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def sor_filter(points: np.ndarray, k: int = 8, z_thresh: float = 2.0) -> np.ndarray:
    if points.shape[0] <= max(16, k + 1):
        return points

    k_eff = max(2, int(k) + 1)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(points)
        try:
            dists, _ = tree.query(points, k=k_eff, workers=-1)
        except TypeError:
            dists, _ = tree.query(points, k=k_eff)

        if dists.ndim == 1:
            return points

        mean_knn = dists[:, 1:].mean(axis=1)
    except Exception:
        sample_count = min(points.shape[0], 4000)
        indices = np.random.choice(points.shape[0], size=sample_count, replace=False)
        sample = points[indices]

        dists = np.sqrt(np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2))
        dists.sort(axis=1)
        mean_knn_sample = dists[:, 1:k_eff].mean(axis=1)

        mu_s = float(mean_knn_sample.mean())
        sigma_s = float(mean_knn_sample.std() + 1e-6)
        thresh_s = float(mu_s + float(z_thresh) * sigma_s)

        approx = np.sqrt(np.sum((points - points.mean(axis=0, keepdims=True)) ** 2, axis=1))
        mu_a = float(approx.mean())
        sigma_a = float(approx.std() + 1e-6)
        if sigma_a <= 1e-9:
            return points

        normalized = (approx - mu_a) / sigma_a
        mean_knn = normalized
        thresh_s = float(normalized.mean() + float(z_thresh) * (normalized.std() + 1e-6))
        keep = mean_knn <= thresh_s
        kept = points[keep]
        if kept.shape[0] == 0:
            return points
        return kept

    mu = float(mean_knn.mean())
    sigma = float(mean_knn.std() + 1e-6)
    keep = mean_knn <= (mu + float(z_thresh) * sigma)
    kept = points[keep]
    if kept.shape[0] == 0:
        return points
    return kept


def compute_coverage_metrics(points: np.ndarray, grid_res_m: float = 1.0) -> dict[str, float]:
    if points.shape[0] == 0:
        return {"coverage_ratio": 0.0, "hole_ratio": 1.0}

    x_min, y_min = float(points[:, 0].min()), float(points[:, 1].min())
    x_max, y_max = float(points[:, 0].max()), float(points[:, 1].max())

    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)

    width = int(np.ceil(x_span / grid_res_m)) + 1
    height = int(np.ceil(y_span / grid_res_m)) + 1

    xi = np.floor((points[:, 0] - x_min) / grid_res_m).astype(np.int32)
    yi = np.floor((points[:, 1] - y_min) / grid_res_m).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    mask[yi[valid], xi[valid]] = 1

    occupied = int(mask.sum())
    total = int(mask.size)
    coverage = float(occupied / max(1, total))
    hole = float(1.0 - coverage)
    return {"coverage_ratio": coverage, "hole_ratio": hole}


def write_ascii_pcd(path: Path, points: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z\n")
        file.write("SIZE 4 4 4\n")
        file.write("TYPE F F F\n")
        file.write("COUNT 1 1 1\n")
        file.write(f"WIDTH {points.shape[0]}\n")
        file.write("HEIGHT 1\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write(f"POINTS {points.shape[0]}\n")
        file.write("DATA ascii\n")
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")


def prepare_airsim_runtime(
    config: dict[str, Any],
    scene_id: str,
    sim_port: int,
    worker_binding: WorkerBinding,
    vehicle_names: list[str],
) -> tuple[int, Any | None, bool]:
    bootstrap_cfg = build_bridge_config(
        config=config,
        engine="airsim",
        sim_port=sim_port,
        worker_binding=worker_binding,
        vehicle_names=vehicle_names,
    )
    resolved_port, bootstrap_bridge, launched_by_bridge, _ = prepare_airsim_runtime_unified(
        config=config,
        scene_id=scene_id,
        base_bridge_cfg=bootstrap_cfg,
        vehicle_name=str(worker_binding.vehicle),
        vehicle_names=[str(v) for v in vehicle_names],
    )
    return resolved_port, bootstrap_bridge, launched_by_bridge


def run_collect(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    logger = StageLogger("step2.collect")
    if bool(getattr(args, "multimodal_fusion", False)):
        collect_cfg = config.setdefault("collect", {})
        mm_cfg = collect_cfg.setdefault("multimodal_fusion", {})
        mm_cfg["enabled"] = True
    if bool(getattr(args, "lidar_segmentation", False)):
        collect_cfg = config.setdefault("collect", {})
        ls_cfg = collect_cfg.setdefault("lidar_segmentation", {})
        ls_cfg["enabled"] = True
        engine_params = config.setdefault("engine_params", {})
        airsim_cfg = engine_params.setdefault("airsim", {})
        airsim_cfg["lidar_segmentation_enabled"] = True
    task_cfg = config.get("task", {}) or {}
    engine = str(args.engine or task_cfg.get("engine", "airsim")).lower()
    scene_id = str(args.scene_id or task_cfg.get("scene_id", "env_airsim_16"))

    supported_engines = {"airsim", "carla", "unrealcv", "sibr", "ue", "gs", "3dgs", "ue5"}
    if engine not in supported_engines:
        raise NotImplementedError(f"engine not supported in M2: {engine}")

    engine_alias = {
        "ue": "unrealcv",
        "ue5": "unrealcv",
        "gs": "sibr",
        "3dgs": "sibr",
    }
    engine = engine_alias.get(engine, engine)

    collect_cfg = config.get("collect", {}) or {}
    parallel_cfg = config.get("parallel", {}) or {}
    engine_cfg = (config.get("engine_params", {}) or {}).get(engine, {}) or {}

    worker_count = int(args.workers or parallel_cfg.get("workers", 2))
    if worker_count <= 0:
        raise ValueError("workers must be >= 1")

    sim_port = int(args.control_port or engine_cfg.get("sim_port", collect_cfg.get("sim_port", 41451)))
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    stage1_dir_name = resolve_output_dir_name(config, key="stage1_dir", default="pcd_map")
    root_dir = Path(args.output_root) if args.output_root else (scene_root / stage1_dir_name)
    ensure_dir(root_dir)
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage1",
        step="collect_raw",
        message="collect_raw_started",
        payload=build_unified_stage_event(
            stage="stage1",
            step="collect_raw",
            scene_id=scene_id,
            engine=engine,
            status="started",
            extra={"workers": int(worker_count), "output_dir": str(root_dir.as_posix())},
        ),
    )

    raw_engine_dir = root_dir / "raw" / engine
    vis_dir = root_dir / "vis"
    shutil.rmtree(raw_engine_dir, ignore_errors=True)
    shutil.rmtree(vis_dir, ignore_errors=True)
    for stale in [
        root_dir / "frames_manifest.jsonl",
        root_dir / f"{scene_id}.frames_manifest.jsonl",
        root_dir / "sampling_preview.mp4",
    ]:
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    poses, pose_meta = make_pose_list(config)
    airsim_lidar_range_profile = compute_airsim_lidar_range_profile(config) if engine == "airsim" else None
    if airsim_lidar_range_profile is not None:
        pose_meta["airsim_lidar_range"] = airsim_lidar_range_profile
        logger.info(
            "airsim lidar range profile "
            f"range_m={float(airsim_lidar_range_profile['range_m']):.1f} "
            f"mode={airsim_lidar_range_profile['mode']} "
            f"source={airsim_lidar_range_profile['source']}"
        )
    max_frames_per_worker, max_frames_auto = resolve_max_frames_per_worker(
        config=config,
        args=args,
        total_pose_count=len(poses),
        worker_count=worker_count,
    )

    bindings = parse_bindings(config, worker_count)
    if engine == "airsim":
        for idx, binding in enumerate(bindings):
            binding.vehicle = normalize_airsim_vehicle_name(binding.vehicle, idx)
    vehicle_names = [item.vehicle for item in bindings]
    shards = split_tasks_round_robin(poses, worker_count)
    shards = [sample_pose_shard_uniform(worker_shard, max_frames_per_worker) for worker_shard in shards]
    requested_frame_count = int(sum(len(worker_shard) for worker_shard in shards))
    logger.info(f"scene={scene_id} engine={engine} workers={worker_count} requested_frames={requested_frame_count}")

    shared_frames: list[dict[str, Any]] = []
    shared_lock = threading.Lock()
    results: list[dict[str, Any]] = []
    errors: list[str] = []

    bootstrap_bridge: Any | None = None
    configured_port = int(sim_port)
    runtime_port = int(configured_port)
    launched_by_bridge = False

    try:
        if engine == "airsim":
            logger.info(f"prepare airsim runtime (preferred_port={sim_port})")
            runtime_port, bootstrap_bridge, launched_by_bridge = prepare_airsim_runtime(
                config=config,
                scene_id=scene_id,
                sim_port=sim_port,
                worker_binding=bindings[0],
                vehicle_names=vehicle_names,
            )
        startup_port_msg = format_unified_startup_ports_message(
            stage="stage1",
            engine=engine,
            configured_sim_port=int(configured_port),
            runtime_sim_port=int(runtime_port),
            launched_by_bridge=bool(launched_by_bridge),
        )
        logger.info(startup_port_msg)
        append_unified_scene_log(
            config=config,
            scene_root=scene_root,
            stage="stage1",
            step="startup",
            message=startup_port_msg,
            payload=build_unified_stage_event(
                stage="stage1",
                step="startup",
                scene_id=scene_id,
                engine=engine,
                status="ready",
                extra={
                    "configured_sim_port": int(configured_port),
                    "runtime_sim_port": int(runtime_port),
                    "launched_by_bridge": bool(launched_by_bridge),
                },
            ),
        )

        def _worker_main(slot_index: int) -> None:
            binding = bindings[slot_index]
            try:
                result = collect_worker(
                    config=config,
                    engine=engine,
                    scene_id=scene_id,
                    sim_port=runtime_port,
                    worker_binding=binding,
                    worker_poses=shards[slot_index],
                    root_dir=root_dir,
                    shared_frames=shared_frames,
                    shared_lock=shared_lock,
                    stop_on_error=bool(args.stop_on_error),
                )
                with shared_lock:
                    results.append(result)
            except Exception as exc:
                with shared_lock:
                    errors.append(f"worker_{binding.worker_id}: {exc}")

        threads: list[threading.Thread] = []
        for slot_index in range(worker_count):
            thread = threading.Thread(target=_worker_main, args=(slot_index,), daemon=False)
            thread.start()
            threads.append(thread)

        progress = ProgressBar(total=requested_frame_count, label="step2.collect")
        progress.update(0, detail="workers started")

        while True:
            alive = any(thread.is_alive() for thread in threads)
            with shared_lock:
                current_rows = len(shared_frames)
                current_errors = len(errors)
            progress.update(current_rows, detail=f"errors={current_errors}")
            if not alive:
                break
            time.sleep(0.5)

        for thread in threads:
            thread.join()

        shared_frames_sorted = sorted(
            shared_frames,
            key=lambda item: (item.get("worker_id", -1), item.get("frame_idx", -1)),
        )

        frames_manifest = root_dir / "frames_manifest.jsonl"
        write_jsonl(frames_manifest, shared_frames_sorted)

        frames_manifest_alias = root_dir / f"{scene_id}.frames_manifest.jsonl"
        write_jsonl(frames_manifest_alias, shared_frames_sorted)

        progress.finish(detail="collection complete")

        worker_results_sorted = sorted(results, key=lambda item: item["worker_id"])
        sampling_videos = [
            str(item["sampling_video_path"])
            for item in worker_results_sorted
            if item.get("sampling_video_path")
        ]
        sampling_preview = None
        if sampling_videos:
            sampling_preview = root_dir / "sampling_preview.mp4"
            try:
                shutil.copy2(sampling_videos[0], sampling_preview)
            except Exception:
                sampling_preview = Path(sampling_videos[0])

        summary = {
            "mode": "collect_raw",
            "scene_id": scene_id,
            "engine": engine,
            "parallel_mode": "single_instance_multi_thread",
            "workers": worker_count,
            "vehicle_names": vehicle_names,
            "grid_pose_count": len(poses),
            "pose_sampling": pose_meta,
            "airsim_lidar_range": airsim_lidar_range_profile,
            "max_frames_per_worker": max_frames_per_worker,
            "max_frames_auto_from_yaml": bool(max_frames_auto),
            "runtime_sim_port": runtime_port,
            "scene_launched_by_bridge": bool(launched_by_bridge),
            "total_requested_frames": requested_frame_count,
            "collected_rows": len(shared_frames_sorted),
            "worker_results": worker_results_sorted,
            "worker_errors": errors,
            "frames_manifest": str(frames_manifest.as_posix()),
            "frames_manifest_alias": str(frames_manifest_alias.as_posix()),
            "sampling_videos": sampling_videos,
            "sampling_preview_video": str(sampling_preview.as_posix()) if sampling_preview else None,
            "root_dir": str(root_dir.as_posix()),
        }

        logger.info(f"collected_rows={len(shared_frames_sorted)} worker_errors={len(errors)}")
        if len(shared_frames_sorted) == 0 and len(errors) > 0:
            raise RuntimeError(
                f"collect_raw failed: no frames collected on port={runtime_port}; errors={errors}"
            )

        append_unified_scene_log(
            config=config,
            scene_root=scene_root,
            stage="stage1",
            step="collect_raw",
            message="collect_raw_finished",
            payload=build_unified_stage_event(
                stage="stage1",
                step="collect_raw",
                scene_id=scene_id,
                engine=engine,
                status="finished",
                extra={
                    "requested_frames": int(requested_frame_count),
                    "collected_rows": int(len(shared_frames_sorted)),
                    "worker_errors": int(len(errors)),
                },
            ),
        )

        return summary
    finally:
        if bootstrap_bridge is not None:
            try:
                bootstrap_bridge.shutdown()
            except Exception:
                pass


def run_fuse(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    logger = StageLogger("step2.fuse")
    task_cfg = config.get("task", {}) or {}
    scene_id = str(args.scene_id or task_cfg.get("scene_id", "env_airsim_16"))
    engine = str(args.engine or task_cfg.get("engine", "airsim")).lower()

    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    stage1_dir_name = resolve_output_dir_name(config, key="stage1_dir", default="pcd_map")
    root_dir = Path(args.output_root) if args.output_root else (scene_root / stage1_dir_name)
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage1",
        step="fuse",
        message="fuse_started",
        payload=build_unified_stage_event(
            stage="stage1",
            step="fuse",
            scene_id=scene_id,
            engine=engine,
            status="started",
            extra={"output_dir": str(root_dir.as_posix())},
        ),
    )
    raw_points = gather_raw_points(root_dir=root_dir, engine=engine, show_progress=True)
    logger.info(f"gather_raw_points={int(raw_points.shape[0])} scene={scene_id} engine={engine}")

    merge_cfg = config.get("merge", {}) or {}
    voxel_size = float(args.voxel_size or merge_cfg.get("voxel_size", 0.15))
    use_sor = bool(args.use_sor or merge_cfg.get("use_sor", False))
    dedup_enabled_cfg = bool(merge_cfg.get("dedup", True))
    dedup_enabled = bool(dedup_enabled_cfg and not args.no_dedup)
    sor_k = int(merge_cfg.get("sor_k", 8))
    sor_z = float(merge_cfg.get("sor_z", 2.0))
    coverage_grid_res_m = float(merge_cfg.get("coverage_grid_res_m", 1.0))

    raw_count = int(raw_points.shape[0])

    stage_progress = ProgressBar(total=5, label="step2.fuse")
    stage_progress.update(0, detail="post-fuse stages")

    if dedup_enabled:
        dedup_points = voxel_dedup(raw_points, voxel_size=voxel_size)
        stage_progress.update(1, detail="voxel dedup complete")
    else:
        dedup_points = raw_points
        stage_progress.update(1, detail="no-dedup mode")

    use_sor_effective = bool(use_sor and dedup_enabled)
    if use_sor_effective:
        dedup_points = sor_filter(dedup_points, k=sor_k, z_thresh=sor_z)
        stage_progress.update(2, detail="sor filter complete")
    else:
        if use_sor and not dedup_enabled:
            logger.info("fuse no-dedup mode enabled; skip SOR to preserve direct accumulation")
        stage_progress.update(2, detail="sor skipped")

    metrics = compute_coverage_metrics(dedup_points, grid_res_m=coverage_grid_res_m)
    stage_progress.update(3, detail="coverage metrics complete")

    primary_pcd_path = root_dir / f"{scene_id}.pcd"
    meta_path = root_dir / f"{scene_id}.meta.json"
    raw_pcd_path = root_dir / f"{scene_id}.raw.pcd"
    raw_pcd_path_nested = root_dir / "merged" / f"{scene_id}.raw.pcd"

    output_cfg = config.get("stage1_output", {}) or {}
    output_formats = output_cfg.get("formats", {}) or {}
    output_pcd_enabled = bool(output_formats.get("pcd", True))
    output_ply_enabled = bool(output_formats.get("ply", True))
    primary_cloud_type = str(output_cfg.get("primary_cloud_type", "semantic_lidar")).lower().strip()
    if primary_cloud_type not in {"semantic_lidar", "raw", "instance_semantic", "colorized"}:
        primary_cloud_type = "semantic_lidar"

    if output_pcd_enabled:
        write_ascii_pcd(raw_pcd_path, dedup_points)
        write_ascii_pcd(raw_pcd_path_nested, dedup_points)

    multimodal_cfg = (config.get("collect", {}) or {}).get("multimodal_fusion", {}) or {}
    multimodal_enabled = bool(multimodal_cfg.get("enabled", False))
    lidar_seg_cfg = (config.get("collect", {}) or {}).get("lidar_segmentation", {}) or {}
    lidar_seg_enabled = bool(lidar_seg_cfg.get("enabled", engine == "airsim")) and engine == "airsim"
    color_pcd_path = root_dir / f"{scene_id}.colorized.pcd"
    inst_pcd_path = root_dir / f"{scene_id}.instance_semantic.pcd"
    color_ply_path = root_dir / f"{scene_id}.colorized.ply"
    inst_ply_path = root_dir / f"{scene_id}.instance_semantic.ply"
    combo_ply_path = root_dir / f"{scene_id}.colorized_semantic_instance.ply"
    multimodal_meta: dict[str, Any] = {"enabled": multimodal_enabled, "available": False}
    lidar_sem_pcd_path = root_dir / f"{scene_id}.semantic_lidar.pcd"
    lidar_sem_ply_path = root_dir / f"{scene_id}.semantic_lidar.ply"
    lidar_seg_meta: dict[str, Any] = {"enabled": lidar_seg_enabled, "available": False}

    if multimodal_enabled:
        mm = gather_multimodal_chunks(root_dir=root_dir, engine=engine)
        mm_points = mm["points"]
        if mm_points.shape[0] > 0:
            mm_colors = mm["colors"]
            mm_sem_raw = mm["semantic_raw"]
            mm_sem_conf = mm["semantic_conf"]

            mm_dedup_points, keep_idx = voxel_dedup_with_indices(mm_points, voxel_size=voxel_size)
            mm_colors = mm_colors[keep_idx]
            mm_sem_raw = mm_sem_raw[keep_idx]
            mm_sem_conf = mm_sem_conf[keep_idx]

            raw_to_compact: dict[int, int] = {}
            class_compact = np.zeros((mm_sem_raw.shape[0],), dtype=np.uint16)
            for rv in np.unique(mm_sem_raw):
                rvi = int(rv)
                if rvi <= 0:
                    continue
                if rvi not in raw_to_compact:
                    raw_to_compact[rvi] = len(raw_to_compact) + 1
                class_compact[mm_sem_raw == rv] = np.uint16(raw_to_compact[rvi])

            instance_voxel = float(multimodal_cfg.get("instance_voxel_size", 1.0))
            instance_min_points = int(multimodal_cfg.get("instance_min_points", 80))
            instance_ids = extract_instances_from_semantic_points(
                points=mm_dedup_points,
                class_ids=class_compact.astype(np.uint32),
                min_points=instance_min_points,
                voxel_size=instance_voxel,
            )

            observed_mask = ((mm_sem_conf > 0.0) & (mm_sem_raw > 0)).astype(np.uint8)

            if output_pcd_enabled:
                write_ascii_pcd_xyzrgb(color_pcd_path, mm_dedup_points, mm_colors)
                write_ascii_pcd_instance(
                    inst_pcd_path,
                    mm_dedup_points,
                    class_compact.astype(np.uint32),
                    instance_ids,
                    mm_sem_raw.astype(np.uint32),
                    mm_sem_conf.astype(np.float32),
                    observed_mask,
                )
            if output_ply_enabled:
                write_ascii_ply_xyzrgb(color_ply_path, mm_dedup_points, mm_colors)
                write_ascii_ply_semantic_instance(
                    inst_ply_path,
                    mm_dedup_points,
                    mm_colors,
                    class_compact.astype(np.uint32),
                    instance_ids,
                    mm_sem_raw.astype(np.uint32),
                    mm_sem_conf.astype(np.float32),
                    observed_mask,
                )
                write_ascii_ply_semantic_instance(
                    combo_ply_path,
                    mm_dedup_points,
                    mm_colors,
                    class_compact.astype(np.uint32),
                    instance_ids,
                    mm_sem_raw.astype(np.uint32),
                    mm_sem_conf.astype(np.float32),
                    observed_mask,
                )
            np.save(root_dir / f"{scene_id}.semantic_raw.npy", mm_sem_raw.astype(np.uint32))
            np.save(root_dir / f"{scene_id}.semantic_compact.npy", class_compact.astype(np.uint16))
            np.save(root_dir / f"{scene_id}.semantic_conf.npy", mm_sem_conf.astype(np.float32))
            np.save(root_dir / f"{scene_id}.instance_ids.npy", instance_ids.astype(np.uint32))

            multimodal_meta = {
                "enabled": True,
                "available": True,
                "points_raw": int(mm_points.shape[0]),
                "points_dedup": int(mm_dedup_points.shape[0]),
                "colored_points": int(np.count_nonzero(np.any(mm_colors > 0, axis=1))),
                "semantic_points": int(np.count_nonzero(class_compact > 0)),
                "observed_points": int(np.count_nonzero(observed_mask > 0)),
                "instance_count": int(np.max(instance_ids)) if instance_ids.size > 0 else 0,
                "colorized_pcd": str(color_pcd_path.as_posix()) if output_pcd_enabled else "",
                "instance_pcd": str(inst_pcd_path.as_posix()) if output_pcd_enabled else "",
                "colorized_ply": str(color_ply_path.as_posix()) if output_ply_enabled else "",
                "instance_ply": str(inst_ply_path.as_posix()) if output_ply_enabled else "",
                "colorized_semantic_instance_ply": str(combo_ply_path.as_posix()) if output_ply_enabled else "",
            }

    if lidar_seg_enabled:
        ls = gather_lidar_segmentation_chunks(root_dir=root_dir, engine=engine)
        ls_points = ls["points"]
        if ls_points.shape[0] > 0:
            ls_sem_raw = ls["semantic_raw"]
            ls_dedup_points, ls_keep_idx = voxel_dedup_with_indices(ls_points, voxel_size=voxel_size)
            ls_sem_raw = ls_sem_raw[ls_keep_idx]

            raw_to_compact: dict[int, int] = {}
            ls_class_compact = np.zeros((ls_sem_raw.shape[0],), dtype=np.uint32)
            for rv in np.unique(ls_sem_raw):
                rvi = int(rv)
                if rvi <= 0:
                    continue
                if rvi not in raw_to_compact:
                    raw_to_compact[rvi] = len(raw_to_compact) + 1
                ls_class_compact[ls_sem_raw == rv] = np.uint32(raw_to_compact[rvi])

            instance_voxel = float(lidar_seg_cfg.get("instance_voxel_size", 1.0))
            instance_min_points = int(lidar_seg_cfg.get("instance_min_points", 80))
            ls_instance_ids = extract_instances_from_semantic_points(
                points=ls_dedup_points,
                class_ids=ls_class_compact.astype(np.uint32),
                min_points=instance_min_points,
                voxel_size=instance_voxel,
            )
            ls_observed = (ls_sem_raw > 0).astype(np.uint8)
            ls_conf = ls_observed.astype(np.float32)
            ls_colors = np.zeros((ls_dedup_points.shape[0], 3), dtype=np.uint8)

            if output_pcd_enabled:
                write_ascii_pcd_instance(
                    lidar_sem_pcd_path,
                    ls_dedup_points,
                    ls_class_compact.astype(np.uint32),
                    ls_instance_ids,
                    ls_sem_raw.astype(np.uint32),
                    ls_conf,
                    ls_observed,
                )
            if output_ply_enabled:
                write_ascii_ply_semantic_instance(
                    lidar_sem_ply_path,
                    ls_dedup_points,
                    ls_colors,
                    ls_class_compact.astype(np.uint32),
                    ls_instance_ids,
                    ls_sem_raw.astype(np.uint32),
                    ls_conf,
                    ls_observed,
                )
            np.save(root_dir / f"{scene_id}.semantic_lidar_raw.npy", ls_sem_raw.astype(np.uint32))
            np.save(root_dir / f"{scene_id}.semantic_lidar_compact.npy", ls_class_compact.astype(np.uint32))
            np.save(root_dir / f"{scene_id}.semantic_lidar_instance.npy", ls_instance_ids.astype(np.uint32))

            lidar_seg_meta = {
                "enabled": True,
                "available": True,
                "points_raw": int(ls_points.shape[0]),
                "points_dedup": int(ls_dedup_points.shape[0]),
                "semantic_points": int(np.count_nonzero(ls_class_compact > 0)),
                "instance_count": int(np.max(ls_instance_ids)) if ls_instance_ids.size > 0 else 0,
                "semantic_lidar_pcd": str(lidar_sem_pcd_path.as_posix()) if output_pcd_enabled else "",
                "semantic_lidar_ply": str(lidar_sem_ply_path.as_posix()) if output_ply_enabled else "",
            }

    primary_source_map = {
        "semantic_lidar": lidar_sem_pcd_path,
        "raw": raw_pcd_path,
        "instance_semantic": inst_pcd_path,
        "colorized": color_pcd_path,
    }
    selected_primary_source = primary_source_map.get(primary_cloud_type, lidar_sem_pcd_path)
    if output_pcd_enabled:
        fallback_candidates = [selected_primary_source, lidar_sem_pcd_path, raw_pcd_path, inst_pcd_path, color_pcd_path]
        final_primary_source = None
        for candidate in fallback_candidates:
            if candidate is not None and candidate.exists():
                final_primary_source = candidate
                break
        if final_primary_source is not None and final_primary_source != primary_pcd_path:
            ensure_dir(primary_pcd_path.parent)
            shutil.copy2(final_primary_source, primary_pcd_path)
    else:
        final_primary_source = None

    stage_progress.update(4, detail="pcd write complete")

    dedup_count = int(dedup_points.shape[0])
    dedup_ratio = float(dedup_count / max(1, raw_count))

    meta = {
        "scene_id": scene_id,
        "engine": engine,
        "raw_points": raw_count,
        "dedup_points": dedup_count,
        "dedup_ratio": dedup_ratio,
        "voxel_size": voxel_size,
        "dedup_enabled": dedup_enabled,
        "fuse_strategy": "direct_accumulate" if not dedup_enabled else "voxel_dedup",
        "use_sor": use_sor_effective,
        "coverage_ratio": metrics["coverage_ratio"],
        "hole_ratio": metrics["hole_ratio"],
        "timestamp": time.time(),
        "source_root": str(root_dir.as_posix()),
        "merged_nested": str(raw_pcd_path_nested.as_posix()) if output_pcd_enabled else "",
        "multimodal": multimodal_meta,
        "lidar_segmentation": lidar_seg_meta,
        "stage1_output": {
            "formats": {"pcd": output_pcd_enabled, "ply": output_ply_enabled},
            "primary_cloud_type": primary_cloud_type,
            "primary_pcd_path": str(primary_pcd_path.as_posix()) if output_pcd_enabled else "",
            "raw_pcd_path": str(raw_pcd_path.as_posix()) if output_pcd_enabled else "",
        },
        "stage1_output": {
            "formats": {"pcd": output_pcd_enabled, "ply": output_ply_enabled},
            "primary_cloud_type": primary_cloud_type,
            "primary_pcd_path": str(primary_pcd_path.as_posix()) if output_pcd_enabled else "",
            "raw_pcd_path": str(raw_pcd_path.as_posix()) if output_pcd_enabled else "",
        },
    }
    ensure_dir(meta_path.parent)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    stage_progress.finish(detail="meta write complete")

    logger.info(f"fuse done raw={raw_count} dedup={dedup_count} coverage={metrics['coverage_ratio']:.4f}")
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage1",
        step="fuse",
        message="fuse_finished",
        payload=build_unified_stage_event(
            stage="stage1",
            step="fuse",
            scene_id=scene_id,
            engine=engine,
            status="finished",
            extra={
                "raw_points": int(raw_count),
                "dedup_points": int(dedup_count),
                "coverage_ratio": float(metrics["coverage_ratio"]),
            },
        ),
    )
    return {
        "mode": "fuse",
        "scene_id": scene_id,
        "engine": engine,
        "raw_points": raw_count,
        "dedup_points": dedup_count,
        "dedup_ratio": dedup_ratio,
        "coverage_ratio": metrics["coverage_ratio"],
        "hole_ratio": metrics["hole_ratio"],
        "pcd_path": str(primary_pcd_path.as_posix()) if output_pcd_enabled else "",
        "pcd_path_nested": str(raw_pcd_path_nested.as_posix()) if output_pcd_enabled else "",
        "meta_path": str(meta_path.as_posix()),
        "multimodal": multimodal_meta,
        "lidar_segmentation": lidar_seg_meta,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAV-DualCog Stage 1: collect and fuse point clouds")
    parser.add_argument("--config", type=str, default="configs/uav_dualcog/task_airsim_env_7.yaml")
    parser.add_argument("--mode", type=str, default="all", choices=["collect_raw", "fuse", "all"])
    parser.add_argument("--engine", type=str, default="airsim")
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--use-sor", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--multimodal-fusion", action="store_true")
    parser.add_argument("--lidar-segmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = StageLogger("step2")
    config = load_yaml(Path(args.config))

    summaries: list[dict[str, Any]] = []
    logger.info(f"mode={args.mode} config={args.config}")
    if args.mode in {"collect_raw", "all"}:
        summaries.append(run_collect(config=config, args=args))
    if args.mode in {"fuse", "all"}:
        summaries.append(run_fuse(config=config, args=args))

    print(json.dumps({"ok": True, "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
