#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sim_bridge.factory import create_bridge


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_pipeline_common = _load_module_from_path("pipeline_common_dynamic", SCRIPT_DIR / "pipeline_common.py")
_stage1_common = _load_module_from_path("stage1_collect_pcd_dynamic", SCRIPT_DIR / "stage1_collect_pcd.py")

build_unified_bridge_config = _pipeline_common.build_unified_bridge_config
prepare_airsim_runtime_unified = _pipeline_common.prepare_airsim_runtime_unified
parse_bindings = _stage1_common.parse_bindings
normalize_airsim_vehicle_name = _stage1_common.normalize_airsim_vehicle_name


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be mapping")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _log_progress(message: str) -> None:
    print(f"[probe][{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _boundary_probe_source(probe_source: str) -> str:
    source = str(probe_source or "").strip().lower()
    return "lidar" if source == "hybrid" else source


def depth_to_cloud_count(depth: Any, sample_step: int, min_depth: float, max_depth: float) -> int:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.size == 0:
        return 0
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        return 0
    sampled = arr[:: max(1, int(sample_step)), :: max(1, int(sample_step))]
    finite = np.isfinite(sampled)
    if not np.any(finite):
        return 0
    valid = finite & (sampled >= float(min_depth)) & (sampled <= float(max_depth))
    return int(np.count_nonzero(valid))


def lidar_to_cloud_count(
    lidar: Any,
    min_range: float,
    max_range: float,
    origin_xyz: tuple[float, float, float] | None = None,
) -> int:
    if lidar is None:
        return 0
    points_raw = lidar.get("points", None) if isinstance(lidar, dict) else lidar
    if points_raw is None:
        return 0
    arr = np.asarray(points_raw, dtype=np.float32)
    if arr.size == 0:
        return 0
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return 0
        arr = arr.reshape(-1, 3)
    elif arr.ndim == 2 and arr.shape[1] >= 3:
        arr = arr[:, :3]
    else:
        return 0
    finite = np.isfinite(arr).all(axis=1)
    if not np.any(finite):
        return 0
    pts = arr[finite]
    if origin_xyz is None:
        ranges = np.linalg.norm(pts, axis=1)
    else:
        origin = np.asarray(origin_xyz, dtype=np.float32).reshape(1, 3)
        ranges = np.linalg.norm(pts - origin, axis=1)
    valid = (ranges >= float(min_range)) & (ranges <= float(max_range))
    return int(np.count_nonzero(valid))


def lidar_surface_z_from_pose(
    x: float,
    y: float,
    z: float,
    yaws: list[float],
    settle_sec: float,
    local_xy_radius: float,
    percentile: float,
    lidar_min_range: float,
    lidar_max_range: float,
) -> float | None:
    bridge = _get_thread_bridge()
    z_samples: list[float] = []
    radius = float(max(1.0, local_xy_radius))
    q = float(np.clip(percentile, 0.1, 50.0))
    for yaw in yaws:
        bridge.set_uav_pose(float(x), float(y), float(z), float(yaw), 0.0, 0.0)
        if settle_sec > 0:
            time.sleep(float(settle_sec))
        lidar = bridge.get_lidar()
        points_raw = lidar.get("points", None) if isinstance(lidar, dict) else lidar
        if points_raw is None:
            continue
        pts = np.asarray(points_raw, dtype=np.float32)
        if pts.size == 0:
            continue
        if pts.ndim == 1:
            if pts.size % 3 != 0:
                continue
            pts = pts.reshape(-1, 3)
        elif pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        else:
            continue
        finite = np.isfinite(pts).all(axis=1)
        if not np.any(finite):
            continue
        pts = pts[finite]
        ranges = np.linalg.norm(pts, axis=1)
        in_range = (ranges >= float(lidar_min_range)) & (ranges <= float(lidar_max_range))
        if not np.any(in_range):
            continue
        pts = pts[in_range]
        dxy = np.linalg.norm(pts[:, :2] - np.array([float(x), float(y)], dtype=np.float32)[None, :], axis=1)
        near = dxy <= radius
        if not np.any(near):
            continue
        pts = pts[near]
        z_samples.extend([float(v) for v in pts[:, 2].tolist()])
    if not z_samples:
        return None
    return float(np.percentile(np.asarray(z_samples, dtype=np.float32), q))


_THREAD_LOCAL = threading.local()


def _set_thread_bridge(bridge: Any) -> None:
    _THREAD_LOCAL.bridge = bridge


def _get_thread_bridge() -> Any:
    bridge = getattr(_THREAD_LOCAL, "bridge", None)
    if bridge is None:
        raise RuntimeError("thread bridge is not initialized")
    return bridge


def _has_cloud_at_pose(
    x: float,
    y: float,
    z: float,
    yaws: list[float],
    settle_sec: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
    min_cloud_points: int,
    probe_source: str,
    lidar_min_range: float,
    lidar_max_range: float,
) -> tuple[bool, int]:
    bridge = _get_thread_bridge()
    best = 0
    source = str(probe_source).strip().lower()
    for yaw in yaws:
        bridge.set_uav_pose(float(x), float(y), float(z), float(yaw), 0.0, 0.0)
        if settle_sec > 0:
            time.sleep(float(settle_sec))
        depth_count = 0
        lidar_count = 0
        if source in {"depth", "hybrid"}:
            depth = bridge.capture_depth()
            depth_count = depth_to_cloud_count(depth, sample_step, min_depth, max_depth)
            best = max(best, depth_count)
            if source == "depth" and depth_count >= int(min_cloud_points):
                return True, best
        if source in {"lidar", "hybrid"}:
            lidar = bridge.get_lidar()
            lidar_count = lidar_to_cloud_count(
                lidar,
                min_range=lidar_min_range,
                max_range=lidar_max_range,
                origin_xyz=(float(x), float(y), float(z)),
            )
            best = max(best, lidar_count)
            if source == "lidar" and lidar_count >= int(min_cloud_points):
                return True, best
        if source == "hybrid" and max(depth_count, lidar_count) >= int(min_cloud_points):
            return True, best
    return False, best


def has_cloud(
    x: float,
    y: float,
    z: float,
    yaws: list[float],
    settle_sec: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
    min_cloud_points: int,
    probe_source: str,
    lidar_min_range: float,
    lidar_max_range: float,
) -> tuple[bool, int]:
    return _has_cloud_at_pose(
        x=x,
        y=y,
        z=z,
        yaws=yaws,
        settle_sec=settle_sec,
        sample_step=sample_step,
        min_depth=min_depth,
        max_depth=max_depth,
        min_cloud_points=min_cloud_points,
        probe_source=probe_source,
        lidar_min_range=lidar_min_range,
        lidar_max_range=lidar_max_range,
    )


def _extract_lidar_near_stats(
    lidar: Any,
    lidar_min_range: float,
    lidar_max_range: float,
    origin_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    if lidar is None:
        return {"count": 0, "seg_counter": Counter()}
    points_raw = lidar.get("points", None) if isinstance(lidar, dict) else lidar
    if points_raw is None:
        return {"count": 0, "seg_counter": Counter()}
    pts = np.asarray(points_raw, dtype=np.float32)
    if pts.size == 0:
        return {"count": 0, "seg_counter": Counter()}
    if pts.ndim == 1:
        if pts.size % 3 != 0:
            return {"count": 0, "seg_counter": Counter()}
        pts = pts.reshape(-1, 3)
    elif pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]
    else:
        return {"count": 0, "seg_counter": Counter()}
    finite_mask = np.isfinite(pts).all(axis=1)
    if not np.any(finite_mask):
        return {"count": 0, "seg_counter": Counter()}
    pts = pts[finite_mask]
    seg = None
    if isinstance(lidar, dict):
        seg_raw = lidar.get("segmentation", None)
        if seg_raw is not None:
            try:
                seg = np.asarray(seg_raw, dtype=np.uint32).reshape(-1)
            except Exception:
                seg = None
    if seg is not None and seg.shape[0] == finite_mask.shape[0]:
        seg = seg[finite_mask]
    ranges = np.linalg.norm(pts, axis=1)
    near_mask = (ranges >= float(lidar_min_range)) & (ranges <= float(lidar_max_range))
    if not np.any(near_mask):
        return {"count": 0, "seg_counter": Counter()}
    near_count = int(np.count_nonzero(near_mask))
    seg_counter: Counter[int] = Counter()
    if seg is not None and seg.shape[0] == near_mask.shape[0]:
        seg_near = seg[near_mask]
        seg_counter.update(int(v) for v in seg_near.tolist())
    return {"count": near_count, "seg_counter": seg_counter}


def sample_lidar_summary_at_pose(
    x: float,
    y: float,
    z: float,
    yaws: list[float],
    settle_sec: float,
    lidar_min_range: float,
    lidar_max_range: float,
) -> dict[str, Any]:
    bridge = _get_thread_bridge()
    counts: list[int] = []
    agg_counter: Counter[int] = Counter()
    for yaw in yaws:
        bridge.set_uav_pose(float(x), float(y), float(z), float(yaw), 0.0, 0.0)
        if settle_sec > 0:
            time.sleep(float(settle_sec))
        lidar = bridge.get_lidar()
        stats = _extract_lidar_near_stats(
            lidar=lidar,
            lidar_min_range=lidar_min_range,
            lidar_max_range=lidar_max_range,
            origin_xyz=(float(x), float(y), float(z)),
        )
        counts.append(int(stats["count"]))
        agg_counter.update(stats["seg_counter"])
    return {
        "counts": counts,
        "seg_counter": agg_counter,
        "near_total": int(sum(counts)),
    }


def boundary_lidar_support(
    x: float,
    y: float,
    z: float,
    yaws: list[float],
    settle_sec: float,
    lidar_min_range: float,
    lidar_max_range: float,
    min_points_per_yaw: int,
    min_valid_yaws: int,
    seg_min_points_per_id: int,
    seg_stop_max_distinct_ids: int,
    seg_stop_max_total_points: int,
) -> dict[str, Any]:
    summary = sample_lidar_summary_at_pose(
        x=x,
        y=y,
        z=z,
        yaws=yaws,
        settle_sec=settle_sec,
        lidar_min_range=lidar_min_range,
        lidar_max_range=lidar_max_range,
    )
    counts = list(summary.get("counts", []))
    seg_counter = summary.get("seg_counter", Counter())
    valid_yaws = sum(1 for value in counts if int(value) >= int(min_points_per_yaw))
    max_count = max(counts) if counts else 0
    mean_count = float(sum(counts) / len(counts)) if counts else 0.0
    near_total = int(summary.get("near_total", 0))
    distinct_seg_ids = sum(1 for value in seg_counter.values() if int(value) >= int(seg_min_points_per_id))
    top_seg_ids = [
        [int(seg_id), int(seg_count)]
        for seg_id, seg_count in seg_counter.most_common(6)
    ]
    density_ok = bool(valid_yaws >= int(min_valid_yaws))
    seg_stop = bool(
        near_total <= int(seg_stop_max_total_points)
        and distinct_seg_ids <= int(seg_stop_max_distinct_ids)
    )
    return {
        "ok": bool(density_ok and not seg_stop),
        "counts": [int(v) for v in counts],
        "valid_yaws": int(valid_yaws),
        "max_count": int(max_count),
        "mean_count": float(mean_count),
        "near_total": int(near_total),
        "distinct_seg_ids": int(distinct_seg_ids),
        "top_seg_ids": top_seg_ids,
        "seg_stop": bool(seg_stop),
    }


def scan_line_boundary(
    axis: str,
    sign: int,
    center: dict[str, float],
    fixed_other_axis_value: float,
    yaws: list[float],
    coarse_step: float,
    refine_tol: float,
    max_steps: int,
    settle_sec: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
    min_cloud_points: int,
    probe_source: str,
    lidar_min_range: float,
    lidar_max_range: float,
    boundary_lidar_max_range: float,
    boundary_min_points_per_yaw: int,
    boundary_min_valid_yaws: int,
    boundary_seg_min_points_per_id: int,
    boundary_seg_stop_max_distinct_ids: int,
    boundary_seg_stop_max_total_points: int,
    hard_limit_value: float | None = None,
) -> dict[str, Any]:
    del sample_step, min_depth, max_depth, min_cloud_points, probe_source, lidar_max_range
    direction = 1.0 if sign >= 0 else -1.0
    lo = 0.0
    hi: float | None = None
    probe = float(max(0.1, coarse_step))
    last_support = {
        "counts": [],
        "valid_yaws": 0,
        "max_count": 0,
        "mean_count": 0.0,
    }
    coarse_passes = 0
    hit_hard_limit = False

    for _ in range(max_steps):
        if hard_limit_value is not None:
            remaining = (float(hard_limit_value) - float(center[axis])) * direction
            if remaining <= 0.0:
                lo = 0.0
                hit_hard_limit = True
                break
            if probe >= remaining:
                lo = float(remaining)
                hit_hard_limit = True
                break
        x = float(center["x"])
        y = float(center["y"])
        z = float(center["z"])
        if axis == "x":
            x = float(center["x"] + direction * probe)
            y = float(fixed_other_axis_value)
        else:
            y = float(center["y"] + direction * probe)
            x = float(fixed_other_axis_value)
        support = boundary_lidar_support(
            x=x,
            y=y,
            z=z,
            yaws=yaws,
            settle_sec=settle_sec,
            lidar_min_range=lidar_min_range,
            lidar_max_range=boundary_lidar_max_range,
            min_points_per_yaw=boundary_min_points_per_yaw,
            min_valid_yaws=boundary_min_valid_yaws,
            seg_min_points_per_id=boundary_seg_min_points_per_id,
            seg_stop_max_distinct_ids=boundary_seg_stop_max_distinct_ids,
            seg_stop_max_total_points=boundary_seg_stop_max_total_points,
        )
        if support["ok"]:
            lo = probe
            coarse_passes += 1
            last_support = support
            probe += coarse_step
        else:
            hi = probe
            break

    hit_limit = hi is None and not hit_hard_limit
    refine_passes = 0
    if not hit_limit and not hit_hard_limit:
        for _ in range(max_steps * 2):
            if hi - lo <= refine_tol:
                break
            mid = 0.5 * (lo + hi)
            x = float(center["x"])
            y = float(center["y"])
            z = float(center["z"])
            if axis == "x":
                x = float(center["x"] + direction * mid)
                y = float(fixed_other_axis_value)
            else:
                y = float(center["y"] + direction * mid)
                x = float(fixed_other_axis_value)
            support = boundary_lidar_support(
                x=x,
                y=y,
                z=z,
                yaws=yaws,
                settle_sec=settle_sec,
                lidar_min_range=lidar_min_range,
                lidar_max_range=boundary_lidar_max_range,
                min_points_per_yaw=boundary_min_points_per_yaw,
                min_valid_yaws=boundary_min_valid_yaws,
                seg_min_points_per_id=boundary_seg_min_points_per_id,
                seg_stop_max_distinct_ids=boundary_seg_stop_max_distinct_ids,
                seg_stop_max_total_points=boundary_seg_stop_max_total_points,
            )
            refine_passes += 1
            if support["ok"]:
                lo = mid
                last_support = support
            else:
                hi = mid

    return {
        "value": float(center[axis] + direction * lo),
        "hit_limit": bool(hit_limit),
        "coarse_passes": int(coarse_passes),
        "refine_passes": int(refine_passes),
        "last_ok_probe": float(lo),
        "last_ok_count": int(last_support.get("max_count", 0)),
        "last_ok_counts": [int(v) for v in last_support.get("counts", [])],
        "last_ok_valid_yaws": int(last_support.get("valid_yaws", 0)),
        "last_ok_mean_count": float(last_support.get("mean_count", 0.0)),
        "last_ok_near_total": int(last_support.get("near_total", 0)),
        "last_ok_distinct_seg_ids": int(last_support.get("distinct_seg_ids", 0)),
        "last_ok_top_seg_ids": list(last_support.get("top_seg_ids", [])),
        "seg_stop": bool(last_support.get("seg_stop", False)),
        "first_fail_probe": None if hi is None else float(hi),
        "hit_hard_limit": bool(hit_hard_limit),
        "hard_limit_value": None if hard_limit_value is None else float(hard_limit_value),
        "effective_probe_source": "lidar_nearby",
    }


def scan_up_boundary(
    x: float,
    y: float,
    z0: float,
    yaws: list[float],
    coarse_step: float,
    refine_tol: float,
    max_steps: int,
    settle_sec: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
    min_cloud_points: int,
    probe_source: str,
    lidar_min_range: float,
    lidar_max_range: float,
    boundary_lidar_max_range: float,
    boundary_min_points_per_yaw: int,
    z_max_limit: float | None = None,
) -> dict[str, Any]:
    del sample_step, min_depth, max_depth, min_cloud_points, probe_source, lidar_max_range
    lo = 0.0
    hi: float | None = None
    probe = float(max(0.1, coarse_step))
    last_ok_count = 0
    coarse_passes = 0
    hit_hard_limit = False

    for _ in range(max_steps):
        if z_max_limit is not None:
            remaining = float(z_max_limit) - float(z0)
            if remaining <= 0.0:
                lo = 0.0
                hit_hard_limit = True
                break
            if probe >= remaining:
                lo = float(remaining)
                hit_hard_limit = True
                break
        ok, count = has_cloud(
            x=float(x),
            y=float(y),
            z=float(z0 + probe),
            yaws=yaws,
            settle_sec=settle_sec,
            sample_step=1,
            min_depth=0.0,
            max_depth=0.0,
            min_cloud_points=int(boundary_min_points_per_yaw),
            probe_source="lidar",
            lidar_min_range=lidar_min_range,
            lidar_max_range=boundary_lidar_max_range,
        )
        if ok:
            lo = probe
            coarse_passes += 1
            last_ok_count = max(last_ok_count, int(count))
            probe += coarse_step
        else:
            hi = probe
            break

    hit_limit = hi is None and not hit_hard_limit
    refine_passes = 0
    if not hit_limit and not hit_hard_limit:
        for _ in range(max_steps * 2):
            if hi - lo <= refine_tol:
                break
            mid = 0.5 * (lo + hi)
            ok, _ = has_cloud(
                x=float(x),
                y=float(y),
                z=float(z0 + mid),
                yaws=yaws,
                settle_sec=settle_sec,
                sample_step=1,
                min_depth=0.0,
                max_depth=0.0,
                min_cloud_points=int(boundary_min_points_per_yaw),
                probe_source="lidar",
                lidar_min_range=lidar_min_range,
                lidar_max_range=boundary_lidar_max_range,
            )
            refine_passes += 1
            if ok:
                lo = mid
            else:
                hi = mid

    return {
        "value": float(z0 + lo),
        "hit_limit": bool(hit_limit),
        "coarse_passes": int(coarse_passes),
        "refine_passes": int(refine_passes),
        "last_ok_probe": float(lo),
        "last_ok_count": int(last_ok_count),
        "hit_hard_limit": bool(hit_hard_limit),
        "hard_limit_value": None if z_max_limit is None else float(z_max_limit),
        "effective_probe_source": "lidar_nearby",
    }


def scan_down_boundary(
    x: float,
    y: float,
    z0: float,
    yaws: list[float],
    coarse_step: float,
    refine_tol: float,
    max_steps: int,
    settle_sec: float,
    sample_step: int,
    min_depth: float,
    max_depth: float,
    min_cloud_points: int,
    probe_source: str,
    lidar_min_range: float,
    lidar_max_range: float,
    boundary_lidar_max_range: float,
    boundary_min_points_per_yaw: int,
    z_min_limit: float | None = None,
) -> dict[str, Any]:
    del sample_step, min_depth, max_depth, min_cloud_points, probe_source, lidar_max_range
    lo = 0.0
    hi: float | None = None
    probe = float(max(0.1, coarse_step))
    last_ok_count = 0
    coarse_passes = 0
    hit_hard_limit = False

    for _ in range(max_steps):
        if z_min_limit is not None:
            remaining = float(z0) - float(z_min_limit)
            if remaining <= 0.0:
                lo = 0.0
                hit_hard_limit = True
                break
            if probe >= remaining:
                lo = float(remaining)
                hit_hard_limit = True
                break
        ok, count = has_cloud(
            x=float(x),
            y=float(y),
            z=float(z0 - probe),
            yaws=yaws,
            settle_sec=settle_sec,
            sample_step=1,
            min_depth=0.0,
            max_depth=0.0,
            min_cloud_points=int(boundary_min_points_per_yaw),
            probe_source="lidar",
            lidar_min_range=lidar_min_range,
            lidar_max_range=boundary_lidar_max_range,
        )
        if ok:
            lo = probe
            coarse_passes += 1
            last_ok_count = max(last_ok_count, int(count))
            probe += coarse_step
        else:
            hi = probe
            break

    hit_limit = hi is None and not hit_hard_limit
    refine_passes = 0
    if not hit_limit and not hit_hard_limit:
        for _ in range(max_steps * 2):
            if hi - lo <= refine_tol:
                break
            mid = 0.5 * (lo + hi)
            ok, _ = has_cloud(
                x=float(x),
                y=float(y),
                z=float(z0 - mid),
                yaws=yaws,
                settle_sec=settle_sec,
                sample_step=1,
                min_depth=0.0,
                max_depth=0.0,
                min_cloud_points=int(boundary_min_points_per_yaw),
                probe_source="lidar",
                lidar_min_range=lidar_min_range,
                lidar_max_range=boundary_lidar_max_range,
            )
            refine_passes += 1
            if ok:
                lo = mid
            else:
                hi = mid

    return {
        "value": float(z0 - lo),
        "hit_limit": bool(hit_limit),
        "coarse_passes": int(coarse_passes),
        "refine_passes": int(refine_passes),
        "last_ok_probe": float(lo),
        "last_ok_count": int(last_ok_count),
        "hit_hard_limit": bool(hit_hard_limit),
        "hard_limit_value": None if z_min_limit is None else float(z_min_limit),
        "effective_probe_source": "lidar_nearby",
    }


def _build_worker_bridges(
    cfg: dict[str, Any],
    scene_id: str,
    sim_port: int,
    headless: bool,
    workers: int,
) -> tuple[list[Any], Any, int, int]:
    worker_count = max(1, int(workers))
    bindings = parse_bindings(cfg, worker_count=worker_count)
    for idx, binding in enumerate(bindings):
        binding.vehicle = normalize_airsim_vehicle_name(binding.vehicle, idx)

    first_vehicle = str(bindings[0].vehicle)
    base_cfg = build_unified_bridge_config(
        cfg,
        engine="airsim",
        vehicle_name=first_vehicle,
        sim_port=int(sim_port),
    )
    base_cfg["headless"] = bool(headless)
    base_cfg["launch_sim"] = bool(((cfg.get("engine_params", {}) or {}).get("airsim", {}) or {}).get("launch_sim", True))
    base_cfg["connect_on_init"] = True
    base_cfg["auto_select_port_on_conflict"] = True

    runtime_port, bootstrap_bridge, _launched_by_bridge, configured_port = prepare_airsim_runtime_unified(
        config=cfg,
        scene_id=scene_id,
        base_bridge_cfg=base_cfg,
        vehicle_name=first_vehicle,
        vehicle_names=[str(b.vehicle) for b in bindings],
    )

    bridges: list[Any] = []
    for binding in bindings:
        bridge_cfg = build_unified_bridge_config(
            cfg,
            engine="airsim",
            vehicle_name=str(binding.vehicle),
            sim_port=int(runtime_port),
        )
        bridge_cfg["headless"] = bool(headless)
        bridge_cfg["launch_sim"] = False
        bridge_cfg["connect_on_init"] = True
        bridge_cfg["auto_select_port_on_conflict"] = False
        bridge_cfg["vehicle_names"] = [str(b.vehicle) for b in bindings]
        bridge_obj = create_bridge(engine="airsim", scene_id=scene_id, config=bridge_cfg)
        bridges.append(bridge_obj)
    return bridges, bootstrap_bridge, int(runtime_port), int(configured_port)


def _safe_shutdown_bridge(bridge_obj: Any, timeout_sec: float = 2.0) -> bool:
    if bridge_obj is None:
        return True
    done = {"ok": False}

    def _runner() -> None:
        try:
            bridge_obj.shutdown()
            done["ok"] = True
        except Exception:
            done["ok"] = True

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(max(0.1, float(timeout_sec)))
    return bool(done.get("ok", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe AirSim scene bounds by four-edge sweep + upward scan.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--scene-id", default=None, type=str)
    parser.add_argument("--sim-port", default=None, type=int)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--center-x", default=0.0, type=float)
    parser.add_argument("--center-y", default=0.0, type=float)
    parser.add_argument("--center-z", default=None, type=float)
    parser.add_argument("--sweep-half-span", default=120.0, type=float)
    parser.add_argument("--sweep-lateral-step", default=30.0, type=float)
    parser.add_argument("--coarse-step", default=15.0, type=float)
    parser.add_argument("--refine-tol", default=1.0, type=float)
    parser.add_argument("--max-steps", default=120, type=int)
    parser.add_argument("--settle-sec", default=0.03, type=float)
    parser.add_argument("--workers", default=6, type=int)
    parser.add_argument("--probe-source", choices=["depth", "lidar", "hybrid"], default="hybrid")
    parser.add_argument("--yaws", nargs="*", type=float, default=[0.0, 90.0, 180.0, 270.0])
    parser.add_argument("--sample-step", default=4, type=int)
    parser.add_argument("--min-depth", default=0.3, type=float)
    parser.add_argument("--max-depth", default=120.0, type=float)
    parser.add_argument("--min-lidar-range", default=None, type=float)
    parser.add_argument("--max-lidar-range", default=None, type=float)
    parser.add_argument("--boundary-lidar-max-range", default=50.0, type=float)
    parser.add_argument("--boundary-min-points-per-yaw", default=250, type=int)
    parser.add_argument("--boundary-min-valid-yaws", default=2, type=int)
    parser.add_argument("--boundary-seg-min-points-per-id", default=50, type=int)
    parser.add_argument("--boundary-seg-stop-max-distinct-ids", default=4, type=int)
    parser.add_argument("--boundary-seg-stop-max-total-points", default=800, type=int)
    parser.add_argument("--xy-boundary-quantile", default=0.2, type=float, help="robust XY boundary quantile; 0 uses raw min/max, 0.2 ignores sparse outer outliers")
    parser.add_argument("--min-cloud-points", default=20, type=int)
    parser.add_argument("--surface-percentile", default=5.0, type=float)
    parser.add_argument("--surface-local-xy-radius", default=120.0, type=float)
    parser.add_argument("--surface-z-margin", default=8.0, type=float)
    parser.add_argument(
        "--surface-robust-low-percentile",
        default=35.0,
        type=float,
        help="robust low-percentile anchor for multi-point surface aggregation",
    )
    parser.add_argument(
        "--surface-center-bias-max-above-low",
        default=3.0,
        type=float,
        help="max allowed center estimate above low anchor to suppress high outliers",
    )
    parser.add_argument("--down-probe", action="store_true", default=True)
    parser.add_argument("--write-back", dest="write_back", action="store_true")
    parser.add_argument("--no-write-back", dest="write_back", action="store_false")
    parser.set_defaults(write_back=True)
    parser.add_argument(
        "--min-map-bound",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="minimum allowed MapBound envelope; lower bounds will not exceed upward past XMIN/YMIN/ZMIN, and upper bounds will not shrink below XMAX/YMAX/ZMAX",
    )
    parser.add_argument(
        "--hard-map-bound",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="hard probe bounds applied during actual x/y/z scans; an axis is enabled only when MIN < MAX",
    )
    parser.add_argument("--output", default=".runtime/probe_mapbound_result.json", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    task_cfg = cfg.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim") or "airsim").strip().lower()
    scene_id = str(args.scene_id or task_cfg.get("scene_id", "env_7"))
    scene_name = scene_id if scene_id.startswith(f"{engine_name}_") else f"{engine_name}_{scene_id}"
    traj_map_cfg = cfg.get("traj_map", {}) or {}
    airsim_cfg = ((cfg.get("engine_params", {}) or {}).get("airsim", {}) or {})
    sim_port = int(args.sim_port if args.sim_port is not None else airsim_cfg.get("sim_port", 41007))
    probe_source = str(args.probe_source).strip().lower()
    workers = max(1, int(args.workers))
    lidar_min_range = float(args.min_lidar_range if args.min_lidar_range is not None else airsim_cfg.get("lidar_min_range_m", 2.0))
    lidar_max_range = float(args.max_lidar_range if args.max_lidar_range is not None else airsim_cfg.get("lidar_range", 500.0))
    boundary_lidar_max_range = float(max(5.0, args.boundary_lidar_max_range))
    boundary_min_points_per_yaw = max(1, int(args.boundary_min_points_per_yaw))
    boundary_min_valid_yaws = max(1, int(args.boundary_min_valid_yaws))
    boundary_seg_min_points_per_id = max(1, int(args.boundary_seg_min_points_per_id))
    boundary_seg_stop_max_distinct_ids = max(1, int(args.boundary_seg_stop_max_distinct_ids))
    boundary_seg_stop_max_total_points = max(1, int(args.boundary_seg_stop_max_total_points))
    xy_boundary_quantile = float(np.clip(float(args.xy_boundary_quantile), 0.0, 0.49))
    hard_map_bound: dict[str, float] | None = None
    if isinstance(args.hard_map_bound, (list, tuple)) and len(args.hard_map_bound) == 6:
        x_min_h, x_max_h, y_min_h, y_max_h, z_min_h, z_max_h = [float(v) for v in args.hard_map_bound]
        if x_min_h < x_max_h or y_min_h < y_max_h or z_min_h < z_max_h:
            hard_map_bound = {
                "x_min": float(x_min_h),
                "x_max": float(x_max_h),
                "y_min": float(y_min_h),
                "y_max": float(y_max_h),
                "z_min": float(z_min_h),
                "z_max": float(z_max_h),
            }

    z_default = float((cfg.get("collect", {}) or {}).get("altitude_m", 20.0))
    center = {
        "x": float(args.center_x),
        "y": float(args.center_y),
        "z": float(args.center_z if args.center_z is not None else z_default),
    }

    yaws = [float(v) for v in (args.yaws or [0.0, 90.0, 180.0, 270.0])]
    sweep_half_span = float(max(0.0, args.sweep_half_span))
    sweep_lateral_step = float(max(1.0, args.sweep_lateral_step))
    lateral_offsets = np.arange(-sweep_half_span, sweep_half_span + 1e-6, sweep_lateral_step, dtype=np.float32).tolist()
    if 0.0 not in lateral_offsets:
        lateral_offsets.append(0.0)
        lateral_offsets = sorted(set(float(v) for v in lateral_offsets))

    _log_progress(
        "start "
        f"scene_id={scene_id} sim_port={sim_port} workers={workers} "
        f"probe_source={probe_source} boundary_source=lidar_nearby "
        f"boundary_lidar_max_range={boundary_lidar_max_range:.1f} "
        f"boundary_min_points_per_yaw={boundary_min_points_per_yaw} "
        f"boundary_min_valid_yaws={boundary_min_valid_yaws} "
        f"boundary_seg_min_points_per_id={boundary_seg_min_points_per_id} "
        f"boundary_seg_stop_max_distinct_ids={boundary_seg_stop_max_distinct_ids} "
        f"boundary_seg_stop_max_total_points={boundary_seg_stop_max_total_points} "
        f"xy_boundary_quantile={xy_boundary_quantile:.2f} "
        f"hard_map_bound={hard_map_bound}"
    )

    bridges, bootstrap_bridge, runtime_port, configured_port = _build_worker_bridges(
        cfg=cfg,
        scene_id=scene_id,
        sim_port=sim_port,
        headless=bool(args.headless),
        workers=workers,
    )
    _log_progress(
        f"bridges_ready runtime_port={runtime_port} configured_port={configured_port} "
        f"lateral_offsets={len(lateral_offsets)}"
    )
    t0 = time.time()

    try:
        line_tasks: list[tuple[str, float, int, float]] = []
        for dy in lateral_offsets:
            y_fixed = float(center["y"] + float(dy))
            line_tasks.append(("x", float(dy), -1, y_fixed))
            line_tasks.append(("x", float(dy), 1, y_fixed))
        for dx in lateral_offsets:
            x_fixed = float(center["x"] + float(dx))
            line_tasks.append(("y", float(dx), -1, x_fixed))
            line_tasks.append(("y", float(dx), 1, x_fixed))

        init_lock = threading.Lock()
        init_idx = {"value": 0}

        def _executor_initializer() -> None:
            with init_lock:
                idx = int(init_idx["value"] % len(bridges))
                init_idx["value"] += 1
            _set_thread_bridge(bridges[idx])

        def _run_line_task(task: tuple[str, float, int, float]) -> dict[str, Any]:
            axis, offset, sign, fixed = task
            hard_limit_value = None
            if hard_map_bound is not None:
                if axis == "x" and sign < 0 and float(hard_map_bound["x_min"]) < float(hard_map_bound["x_max"]):
                    hard_limit_value = float(hard_map_bound["x_min"])
                elif axis == "x" and sign > 0 and float(hard_map_bound["x_min"]) < float(hard_map_bound["x_max"]):
                    hard_limit_value = float(hard_map_bound["x_max"])
                elif axis == "y" and sign < 0 and float(hard_map_bound["y_min"]) < float(hard_map_bound["y_max"]):
                    hard_limit_value = float(hard_map_bound["y_min"])
                elif axis == "y" and sign > 0 and float(hard_map_bound["y_min"]) < float(hard_map_bound["y_max"]):
                    hard_limit_value = float(hard_map_bound["y_max"])
            item = scan_line_boundary(
                axis=axis,
                sign=sign,
                center=center,
                fixed_other_axis_value=float(fixed),
                yaws=yaws,
                coarse_step=float(args.coarse_step),
                refine_tol=float(args.refine_tol),
                max_steps=int(args.max_steps),
                settle_sec=float(args.settle_sec),
                sample_step=int(args.sample_step),
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                min_cloud_points=int(args.min_cloud_points),
                probe_source=probe_source,
                lidar_min_range=lidar_min_range,
                lidar_max_range=lidar_max_range,
                boundary_lidar_max_range=boundary_lidar_max_range,
                boundary_min_points_per_yaw=boundary_min_points_per_yaw,
                boundary_min_valid_yaws=boundary_min_valid_yaws,
                boundary_seg_min_points_per_id=boundary_seg_min_points_per_id,
                boundary_seg_stop_max_distinct_ids=boundary_seg_stop_max_distinct_ids,
                boundary_seg_stop_max_total_points=boundary_seg_stop_max_total_points,
                hard_limit_value=hard_limit_value,
            )
            item.update(
                {
                    "axis": axis,
                    "offset": float(offset),
                    "sign": int(sign),
                    "fixed_other_axis_value": float(fixed),
                }
            )
            return item

        _log_progress(f"line_scan_start total_tasks={len(line_tasks)}")
        x_neg_candidates: list[float] = []
        x_pos_candidates: list[float] = []
        y_neg_candidates: list[float] = []
        y_pos_candidates: list[float] = []
        line_details: list[dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            initializer=_executor_initializer,
        ) as executor:
            line_futs = [executor.submit(_run_line_task, task) for task in line_tasks]
            total_line_tasks = len(line_futs)
            for done_index, fut in enumerate(concurrent.futures.as_completed(line_futs), start=1):
                item = fut.result()
                line_details.append(item)
                axis = str(item["axis"])
                sign = int(item["sign"])
                val = float(item["value"])
                if axis == "x" and sign < 0:
                    x_neg_candidates.append(val)
                elif axis == "x" and sign > 0:
                    x_pos_candidates.append(val)
                elif axis == "y" and sign < 0:
                    y_neg_candidates.append(val)
                elif axis == "y" and sign > 0:
                    y_pos_candidates.append(val)
                _log_progress(
                    f"line_scan_progress {done_index}/{total_line_tasks} "
                    f"axis={axis} sign={sign:+d} offset={float(item['offset']):.1f} "
                    f"value={val:.2f} hit_limit={bool(item['hit_limit'])} "
                    f"hit_hard_limit={bool(item.get('hit_hard_limit', False))} "
                    f"valid_yaws={int(item.get('last_ok_valid_yaws', 0))} "
                    f"max_count={int(item.get('last_ok_count', 0))} "
                    f"near_total={int(item.get('last_ok_near_total', 0))} "
                    f"distinct_seg_ids={int(item.get('last_ok_distinct_seg_ids', 0))} "
                    f"seg_stop={bool(item.get('seg_stop', False))}"
                )

        x_neg_arr = np.asarray(x_neg_candidates, dtype=np.float32)
        x_pos_arr = np.asarray(x_pos_candidates, dtype=np.float32)
        y_neg_arr = np.asarray(y_neg_candidates, dtype=np.float32)
        y_pos_arr = np.asarray(y_pos_candidates, dtype=np.float32)
        raw_min_x = float(np.min(x_neg_arr))
        raw_max_x = float(np.max(x_pos_arr))
        raw_min_y = float(np.min(y_neg_arr))
        raw_max_y = float(np.max(y_pos_arr))
        if xy_boundary_quantile > 0.0:
            min_x = float(np.quantile(x_neg_arr, xy_boundary_quantile))
            max_x = float(np.quantile(x_pos_arr, 1.0 - xy_boundary_quantile))
            min_y = float(np.quantile(y_neg_arr, xy_boundary_quantile))
            max_y = float(np.quantile(y_pos_arr, 1.0 - xy_boundary_quantile))
        else:
            min_x = raw_min_x
            max_x = raw_max_x
            min_y = raw_min_y
            max_y = raw_max_y
        _log_progress(
            f"xy_bounds raw=({raw_min_x:.2f},{raw_max_x:.2f},{raw_min_y:.2f},{raw_max_y:.2f}) "
            f"robust=({min_x:.2f},{max_x:.2f},{min_y:.2f},{max_y:.2f}) q={xy_boundary_quantile:.2f}"
        )

        z_probe_points = [
            (center["x"], center["y"]),
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
        ]

        def _run_surface_task(task: tuple[int, tuple[float, float]]) -> dict[str, Any]:
            point_index, pt = task
            px, py = pt
            surface_value = lidar_surface_z_from_pose(
                x=float(px),
                y=float(py),
                z=float(center["z"]),
                yaws=yaws,
                settle_sec=float(args.settle_sec),
                local_xy_radius=float(args.surface_local_xy_radius),
                percentile=float(args.surface_percentile),
                lidar_min_range=lidar_min_range,
                lidar_max_range=lidar_max_range,
            )
            return {
                "point_index": int(point_index),
                "x": float(px),
                "y": float(py),
                "surface_value": None if surface_value is None else float(surface_value),
            }

        def _run_z_task(pt: tuple[float, float]) -> dict[str, Any]:
            px, py = pt
            z_max_limit = None
            if hard_map_bound is not None and float(hard_map_bound["z_min"]) < float(hard_map_bound["z_max"]):
                z_max_limit = float(hard_map_bound["z_max"])
            item = scan_up_boundary(
                x=float(px),
                y=float(py),
                z0=float(center["z"]),
                yaws=yaws,
                coarse_step=float(args.coarse_step),
                refine_tol=float(args.refine_tol),
                max_steps=int(args.max_steps),
                settle_sec=float(args.settle_sec),
                sample_step=int(args.sample_step),
                min_depth=float(args.min_depth),
                max_depth=float(args.max_depth),
                min_cloud_points=int(args.min_cloud_points),
                probe_source=probe_source,
                lidar_min_range=lidar_min_range,
                lidar_max_range=lidar_max_range,
                boundary_lidar_max_range=boundary_lidar_max_range,
                boundary_min_points_per_yaw=boundary_min_points_per_yaw,
                z_max_limit=z_max_limit,
            )
            item.update({"x": float(px), "y": float(py)})
            return item

        _log_progress(f"z_up_scan_start points={len(z_probe_points)}")
        z_candidates: list[float] = []
        z_details: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, max(1, len(z_probe_points))),
            initializer=_executor_initializer,
        ) as z_executor:
            z_futs = [z_executor.submit(_run_z_task, pt) for pt in z_probe_points]
            total_z_tasks = len(z_futs)
            for done_index, fut in enumerate(concurrent.futures.as_completed(z_futs), start=1):
                item = fut.result()
                z_details.append(item)
                z_candidates.append(float(item["value"]))
                _log_progress(
                    f"z_up_scan_progress {done_index}/{total_z_tasks} x={float(item['x']):.1f} y={float(item['y']):.1f} "
                    f"value={float(item['value']):.2f} hit_limit={bool(item['hit_limit'])} "
                    f"hit_hard_limit={bool(item.get('hit_hard_limit', False))} max_count={int(item['last_ok_count'])}"
                )

        _log_progress(f"surface_scan_start points={len(z_probe_points)}")
        surface_candidates: list[float] = []
        surface_details: list[dict[str, Any]] = []
        center_surface_z: float | None = None
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, max(1, len(z_probe_points))),
            initializer=_executor_initializer,
        ) as s_executor:
            s_futs = [s_executor.submit(_run_surface_task, (idx, pt)) for idx, pt in enumerate(z_probe_points)]
            total_surface_tasks = len(s_futs)
            for done_index, fut in enumerate(concurrent.futures.as_completed(s_futs), start=1):
                item = fut.result()
                surface_details.append(item)
                surface_z = item["surface_value"]
                if surface_z is not None and np.isfinite(surface_z):
                    surface_candidates.append(float(surface_z))
                    if int(item["point_index"]) == 0:
                        center_surface_z = float(surface_z)
                _log_progress(
                    f"surface_scan_progress {done_index}/{total_surface_tasks} x={float(item['x']):.1f} y={float(item['y']):.1f} "
                    f"surface_z={item['surface_value']}"
                )

        old_map_bound = traj_map_cfg.get("MapBound", [-400, 400, -400, 400, 0, 200])
        old_min_z = float(old_map_bound[4]) if isinstance(old_map_bound, list) and len(old_map_bound) == 6 else 0.0
        max_z = float(np.max(np.asarray(z_candidates, dtype=np.float32)))
        surface_z_est = None
        if surface_candidates:
            surface_arr = np.asarray(surface_candidates, dtype=np.float32)
            low_q = float(np.clip(float(args.surface_robust_low_percentile), 5.0, 50.0))
            low_anchor = float(np.percentile(surface_arr, low_q))
            median_anchor = float(np.median(surface_arr))
            surface_z_est = float(0.5 * (low_anchor + median_anchor))
            if center_surface_z is not None and np.isfinite(center_surface_z):
                center_v = float(center_surface_z)
                window = float(max(1.0, float(args.surface_center_bias_max_above_low) * 2.0))
                if abs(center_v - surface_z_est) <= window:
                    surface_z_est = float(0.8 * surface_z_est + 0.2 * center_v)

        down_candidates: list[float] = []
        down_details: list[dict[str, Any]] = []
        if bool(args.down_probe):
            z0_down = float(center["z"])
            if surface_z_est is not None:
                z0_down = max(z0_down, float(surface_z_est + max(2.0, float(args.surface_z_margin))))

            def _run_down_task(pt: tuple[float, float]) -> dict[str, Any]:
                px, py = pt
                z_min_limit = None
                if hard_map_bound is not None and float(hard_map_bound["z_min"]) < float(hard_map_bound["z_max"]):
                    z_min_limit = float(hard_map_bound["z_min"])
                item = scan_down_boundary(
                    x=float(px),
                    y=float(py),
                    z0=float(z0_down),
                    yaws=yaws,
                    coarse_step=float(args.coarse_step),
                    refine_tol=float(args.refine_tol),
                    max_steps=int(args.max_steps),
                    settle_sec=float(args.settle_sec),
                    sample_step=int(args.sample_step),
                    min_depth=float(args.min_depth),
                    max_depth=float(args.max_depth),
                    min_cloud_points=int(args.min_cloud_points),
                    probe_source=probe_source,
                    lidar_min_range=lidar_min_range,
                    lidar_max_range=lidar_max_range,
                    boundary_lidar_max_range=boundary_lidar_max_range,
                    boundary_min_points_per_yaw=boundary_min_points_per_yaw,
                    z_min_limit=z_min_limit,
                )
                item.update({"x": float(px), "y": float(py)})
                return item

            _log_progress(f"z_down_scan_start points={len(z_probe_points)} z0_down={z0_down:.2f}")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, max(1, len(z_probe_points))),
                initializer=_executor_initializer,
            ) as d_executor:
                d_futs = [d_executor.submit(_run_down_task, pt) for pt in z_probe_points]
                total_down_tasks = len(d_futs)
                for done_index, fut in enumerate(concurrent.futures.as_completed(d_futs), start=1):
                    item = fut.result()
                    down_details.append(item)
                    down_candidates.append(float(item["value"]))
                    _log_progress(
                        f"z_down_scan_progress {done_index}/{total_down_tasks} x={float(item['x']):.1f} y={float(item['y']):.1f} "
                        f"value={float(item['value']):.2f} hit_limit={bool(item['hit_limit'])} "
                        f"hit_hard_limit={bool(item.get('hit_hard_limit', False))} max_count={int(item['last_ok_count'])}"
                    )

        min_z_pool: list[float] = []
        if down_candidates:
            min_z_pool.append(float(np.min(np.asarray(down_candidates, dtype=np.float32))))
        if surface_z_est is not None:
            min_z_pool.append(float(surface_z_est - float(args.surface_z_margin)))
        detected_min_z_raw = float(np.min(np.asarray(min_z_pool, dtype=np.float32))) if min_z_pool else float(old_min_z)
        z_min_lower_limit = -50.0
        detected_min_z = float(max(detected_min_z_raw, z_min_lower_limit))

        new_map_bound = [
            int(np.floor(min_x)),
            int(np.ceil(max_x)),
            int(np.floor(min_y)),
            int(np.ceil(max_y)),
            int(np.floor(detected_min_z)),
            int(np.ceil(max_z)),
        ]

        min_map_bound = None
        if isinstance(args.min_map_bound, (list, tuple)) and len(args.min_map_bound) == 6:
            min_map_bound = [int(v) for v in args.min_map_bound]
            min_map_bound[4] = max(int(min_map_bound[4]), int(z_min_lower_limit))
            new_map_bound = [
                min(int(new_map_bound[0]), int(min_map_bound[0])),
                max(int(new_map_bound[1]), int(min_map_bound[1])),
                min(int(new_map_bound[2]), int(min_map_bound[2])),
                max(int(new_map_bound[3]), int(min_map_bound[3])),
                max(int(new_map_bound[4]), int(min_map_bound[4])),
                max(int(new_map_bound[5]), int(min_map_bound[5])),
            ]
        if hard_map_bound is not None:
            if float(hard_map_bound["x_min"]) < float(hard_map_bound["x_max"]):
                new_map_bound[0] = max(int(new_map_bound[0]), int(np.floor(float(hard_map_bound["x_min"]))))
                new_map_bound[1] = min(int(new_map_bound[1]), int(np.ceil(float(hard_map_bound["x_max"]))))
            if float(hard_map_bound["y_min"]) < float(hard_map_bound["y_max"]):
                new_map_bound[2] = max(int(new_map_bound[2]), int(np.floor(float(hard_map_bound["y_min"]))))
                new_map_bound[3] = min(int(new_map_bound[3]), int(np.ceil(float(hard_map_bound["y_max"]))))
            if float(hard_map_bound["z_min"]) < float(hard_map_bound["z_max"]):
                new_map_bound[4] = max(int(new_map_bound[4]), int(np.floor(float(hard_map_bound["z_min"]))))
                new_map_bound[5] = min(int(new_map_bound[5]), int(np.ceil(float(hard_map_bound["z_max"]))))
        new_map_bound[4] = max(int(new_map_bound[4]), int(z_min_lower_limit))

        hit_limit_summary = {
            "line": bool(any(bool(item.get("hit_limit")) for item in line_details)),
            "z_up": bool(any(bool(item.get("hit_limit")) for item in z_details)),
            "z_down": bool(any(bool(item.get("hit_limit")) for item in down_details)),
        }
        if any(hit_limit_summary.values()):
            _log_progress(f"warning hit_limit_summary={json.dumps(hit_limit_summary, ensure_ascii=False)}")

        result = {
            "ok": True,
            "scene_name": scene_name,
            "scene_id": scene_id,
            "engine": engine_name,
            "config": str(config_path),
            "elapsed_sec": float(time.time() - t0),
            "sim_port": int(runtime_port),
            "configured_sim_port": int(configured_port),
            "center": center,
            "probe_source": probe_source,
            "boundary_probe_source": "lidar_nearby",
            "hard_map_bound": hard_map_bound,
            "boundary_rules": {
                "lidar_max_range": float(boundary_lidar_max_range),
                "min_points_per_yaw": int(boundary_min_points_per_yaw),
                "min_valid_yaws": int(boundary_min_valid_yaws),
                "seg_min_points_per_id": int(boundary_seg_min_points_per_id),
                "seg_stop_max_distinct_ids": int(boundary_seg_stop_max_distinct_ids),
                "seg_stop_max_total_points": int(boundary_seg_stop_max_total_points),
                "xy_boundary_quantile": float(xy_boundary_quantile),
            },
            "workers": int(workers),
            "probe_ranges": {
                "depth": {"min_depth": float(args.min_depth), "max_depth": float(args.max_depth)},
                "lidar": {"min_range": float(lidar_min_range), "max_range": float(lidar_max_range)},
                "boundary_lidar": {"min_range": float(lidar_min_range), "max_range": float(boundary_lidar_max_range)},
            },
            "lateral_offsets": [float(v) for v in lateral_offsets],
            "xy_bounds_raw": {"min_x": raw_min_x, "max_x": raw_max_x, "min_y": raw_min_y, "max_y": raw_max_y},
            "xy_bounds_robust": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
            "line_task_details": line_details,
            "z_up_details": z_details,
            "surface_details": surface_details,
            "z_down_details": down_details,
            "candidates": {
                "x_neg": x_neg_candidates,
                "x_pos": x_pos_candidates,
                "y_neg": y_neg_candidates,
                "y_pos": y_pos_candidates,
                "z_up": z_candidates,
                "surface_z": surface_candidates,
                "z_down": down_candidates,
            },
            "surface_center_z": center_surface_z,
            "surface_robust_low_percentile": float(args.surface_robust_low_percentile),
            "surface_center_bias_max_above_low": float(args.surface_center_bias_max_above_low),
            "surface_estimate_z": surface_z_est,
            "detected_min_z_raw": detected_min_z_raw,
            "z_min_lower_limit": z_min_lower_limit,
            "detected_min_z": detected_min_z,
            "map_bound_old": old_map_bound,
            "min_map_bound": min_map_bound,
            "hit_limit_summary": hit_limit_summary,
            "map_bound_new": new_map_bound,
            "written": False,
        }

        if bool(args.write_back):
            cfg.setdefault("traj_map", {})
            cfg["traj_map"]["MapBound"] = [int(v) for v in new_map_bound]
            if surface_z_est is not None:
                cfg["traj_map"]["EstimatedSurfaceZ"] = float(surface_z_est)
            cfg["traj_map"]["DetectedMinZ"] = float(detected_min_z)
            stage2_cfg = cfg.setdefault("stage2", {})
            if surface_z_est is not None:
                stage2_cfg["collect_surface_exclude_below_z"] = float(surface_z_est)
            save_yaml(config_path, cfg)
            result["written"] = True

        output_arg = str(args.output or "").strip()
        if output_arg == ".runtime/probe_mapbound_result.json":
            output_arg = f".runtime/probe_{scene_name}_mapbound_result.json"
        out_path = Path(output_arg).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _log_progress(
            f"finished elapsed_sec={result['elapsed_sec']:.2f} map_bound_new={result['map_bound_new']} written={result['written']}"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        for bridge_obj in bridges:
            _safe_shutdown_bridge(bridge_obj, timeout_sec=2.0)
        _safe_shutdown_bridge(bootstrap_bridge, timeout_sec=2.0)


if __name__ == "__main__":
    main()
