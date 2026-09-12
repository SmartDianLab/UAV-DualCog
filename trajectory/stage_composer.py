from __future__ import annotations

import math
from typing import Any

import numpy as np

from .behaviors import sample_waypoints
from .minsnap_connector import _keepout_min_distance


def _bbox_diag(landmark: dict[str, Any]) -> float:
    raw_bbox = landmark.get("bbox_3d", []) or [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0]
    if isinstance(raw_bbox, dict):
        size = raw_bbox.get("size", [3.0, 3.0, 3.0])
        yaw_deg = float(raw_bbox.get("yaw_deg", 0.0) or 0.0)
        if isinstance(size, list) and len(size) >= 3:
            bbox = [0.0, 0.0, 0.0, float(size[0]), float(size[1]), float(size[2]), yaw_deg]
        else:
            bbox = [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, yaw_deg]
    else:
        bbox = list(raw_bbox)
    sx = float(bbox[3]) if len(bbox) > 3 else 3.0
    sy = float(bbox[4]) if len(bbox) > 4 else 3.0
    sz = float(bbox[5]) if len(bbox) > 5 else 3.0
    return float(max(1.0, math.sqrt(sx * sx + sy * sy + sz * sz)))


def _bbox_xy_half_diag(landmark: dict[str, Any]) -> float:
    raw_bbox = landmark.get("bbox_3d", []) or [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0]
    if isinstance(raw_bbox, dict):
        size = raw_bbox.get("size", [3.0, 3.0, 3.0])
        if isinstance(size, list) and len(size) >= 2:
            sx, sy = float(size[0]), float(size[1])
        else:
            sx, sy = 3.0, 3.0
    else:
        bbox = list(raw_bbox)
        sx = float(bbox[3]) if len(bbox) > 3 else 3.0
        sy = float(bbox[4]) if len(bbox) > 4 else 3.0
    return 0.5 * float(math.sqrt(max(1e-6, sx * sx + sy * sy)))


def _legal_radius_range(landmark: dict[str, Any], *, safety_distance: float) -> tuple[float, float]:
    base = _bbox_xy_half_diag(landmark)
    r_min = max(0.5, base + max(0.2, float(safety_distance)))
    r_max = max(r_min + 8.0, 8.0 * max(1.0, base))
    return float(r_min), float(r_max)


def _preferred_start_radius(element_instance: dict[str, Any]) -> float | None:
    params = dict(element_instance.get("params", {}) or {})
    element_key = str(element_instance.get("element_class", "") or "")
    if element_key in {"circular_orbit", "square_orbit", "triangular_orbit", "figure8_orbit"}:
        radius = params.get("effective_radius_m", None)
    elif element_key == "spiral_orbit":
        radius = params.get("effective_start_radius_m", None)
    else:
        radius = None
    try:
        value = float(radius)
    except Exception:
        return None
    return value if value > 1e-3 else None


def _requires_legal_start_projection(element_instance: dict[str, Any]) -> bool:
    element_key = str(element_instance.get("element_class", "") or "")
    return element_key in {
        "circular_orbit",
        "spiral_orbit",
        "square_orbit",
        "triangular_orbit",
        "figure8_orbit",
        "comet",
        "surface_mapping",
    }


def _project_to_legal_start(landmark: dict[str, Any], start_pos: np.ndarray, *, preferred_radius: float | None = None, safety_distance: float = 2.0) -> tuple[np.ndarray, bool, float]:
    center = np.asarray(landmark.get("center_3d", [0.0, 0.0, 0.0]), dtype=np.float32)
    start = start_pos.astype(np.float32)
    vec_xy = start[:2] - center[:2]
    distance = float(np.linalg.norm(vec_xy))
    r_min, r_max = _legal_radius_range(landmark, safety_distance=float(safety_distance))

    if distance < 1e-6:
        projected = start.copy()
        projected[0] = center[0] + float(r_min)
        projected[1] = center[1]
        return projected, True, float(np.linalg.norm(projected - start_pos))

    preferred = None
    if preferred_radius is not None:
        preferred = min(max(float(preferred_radius), r_min), r_max)
    if preferred is not None and abs(distance - preferred) > max(0.75, 0.04 * max(1.0, preferred)):
        projected = start.copy()
        projected[:2] = center[:2] + vec_xy / distance * preferred
        return projected.astype(np.float32), True, float(np.linalg.norm(projected - start_pos))

    if r_min <= distance <= r_max:
        return start.astype(np.float32), False, 0.0

    target_r = min(max(distance, r_min), r_max)
    projected = start.copy()
    projected[:2] = center[:2] + vec_xy / distance * target_r
    return projected.astype(np.float32), True, float(np.linalg.norm(projected - start_pos))


def _bridge_points(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    if float(np.linalg.norm(end - start)) < 1e-3:
        return start.reshape(1, 3).astype(np.float32)
    count = max(6, int(n))
    t = np.linspace(0.0, 1.0, num=count, dtype=np.float32).reshape(-1, 1)
    pts = (start.reshape(1, 3) * (1.0 - t) + end.reshape(1, 3) * t).astype(np.float32)
    return pts


def _repair_points_by_obstacles(
    points_xyz: np.ndarray,
    *,
    obstacles_xyz: np.ndarray | None,
    keepout_boxes: list[dict[str, Any]] | None,
    safety_distance: float,
    max_lift_step_m: float,
    max_total_lift_m: float,
    keepout_margin_xy: float = 0.0,
    keepout_margin_z: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if (obstacles_xyz is None or obstacles_xyz.size == 0) and (not keepout_boxes) or points_xyz.size == 0:
        return points_xyz.astype(np.float32), {"repair_applied": False, "lifted_points": 0, "max_lift_m": 0.0}
    pts = points_xyz.astype(np.float32).copy()
    obs = obstacles_xyz.astype(np.float32) if isinstance(obstacles_xyz, np.ndarray) else np.zeros((0, 3), dtype=np.float32)
    lifted = 0
    max_lift = 0.0
    safe = max(0.2, float(safety_distance))
    step = max(0.2, float(max_lift_step_m))
    max_total = max(step, float(max_total_lift_m))
    for idx in range(pts.shape[0]):
        p = pts[idx]
        lift = 0.0
        while lift <= max_total:
            dist_min = float("inf")
            if obs.size > 0:
                dist_min = float(np.min(np.linalg.norm(obs - p.reshape(1, 3), axis=1)))
            box_dist = _keepout_min_distance(
                p.astype(np.float32),
                list(keepout_boxes or []),
                margin_xy=float(keepout_margin_xy),
                margin_z=float(keepout_margin_z),
            )
            if dist_min >= safe and box_dist >= 0.0:
                break
            p[2] += step
            lift += step
        pts[idx] = p
        if lift > 0.0:
            lifted += 1
            max_lift = max(max_lift, lift)
    return pts, {
        "repair_applied": bool(lifted > 0),
        "lifted_points": int(lifted),
        "max_lift_m": float(max_lift),
    }


def _target_landmark_for_instance(
    *,
    primary_landmark: dict[str, Any],
    landmark_lookup: dict[str, dict[str, Any]],
    element_instance: dict[str, Any],
) -> dict[str, Any]:
    target_id = str(element_instance.get("target_instance_id", "") or "")
    if target_id and target_id in landmark_lookup:
        return landmark_lookup[target_id]
    return primary_landmark


def compose_trajectory(
    *,
    primary_landmark: dict[str, Any],
    element_instances: list[dict[str, Any]],
    landmark_lookup: dict[str, dict[str, Any]],
    start_pos: np.ndarray,
    points_per_element: int = 60,
    pose_fps: float = 10.0,
    obstacles_xyz: np.ndarray | None = None,
    keepout_boxes: list[dict[str, Any]] | None = None,
    safety_distance: float = 2.0,
    repair_max_lift_step_m: float = 2.0,
    repair_max_total_lift_m: float = 24.0,
    keepout_margin_xy: float = 0.0,
    keepout_margin_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not element_instances:
        return start_pos.reshape(1, 3).astype(np.float32), np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), []

    current = start_pos.astype(np.float32)
    all_points: list[np.ndarray] = []
    all_forwards: list[np.ndarray] = []
    segments: list[dict[str, Any]] = []

    for index, element_instance in enumerate(element_instances):
        had_previous_points = bool(all_points)
        target_landmark = _target_landmark_for_instance(
            primary_landmark=primary_landmark,
            landmark_lookup=landmark_lookup,
            element_instance=element_instance,
        )
        if _requires_legal_start_projection(element_instance):
            projected_start, projected, project_distance = _project_to_legal_start(
                landmark=target_landmark,
                start_pos=current,
                preferred_radius=_preferred_start_radius(element_instance),
                safety_distance=float(safety_distance),
            )
        else:
            projected_start = current.astype(np.float32).copy()
            projected = False
            project_distance = 0.0

        if float(np.linalg.norm(projected_start - current)) > max(1.0, 0.12 * _bbox_diag(target_landmark)):
            bridge = _bridge_points(current, projected_start, max(8, points_per_element // 4))
            if all_points:
                bridge = bridge[1:]
            if bridge.size > 0:
                bridge_fwd = np.repeat((projected_start - current).reshape(1, 3), repeats=bridge.shape[0], axis=0).astype(np.float32)
                norms = np.linalg.norm(bridge_fwd, axis=1, keepdims=True)
                bridge_fwd = bridge_fwd / np.where(norms < 1e-6, 1.0, norms)
                all_points.append(bridge.astype(np.float32))
                all_forwards.append(bridge_fwd.astype(np.float32))

        previous_terminal_point = all_points[-1][-1].astype(np.float32).copy() if all_points else None

        preview_points = max(24, int(points_per_element))
        seg_points_preview, _ = sample_waypoints(
            element_instance=element_instance,
            landmark=target_landmark,
            start_pos=projected_start,
            num_points=preview_points,
        )
        preview_lengths = np.linalg.norm(np.diff(seg_points_preview, axis=0), axis=1) if seg_points_preview.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
        path_len_m = float(np.sum(preview_lengths)) if preview_lengths.size > 0 else 0.0
        speed_mps = max(1.0, float((element_instance.get("params", {}) or {}).get("speed_mps", 8.0) or 8.0))
        target_duration_sec = max(0.8, path_len_m / speed_mps) if path_len_m > 1e-3 else 0.8
        target_points = max(24, int(round(target_duration_sec * max(1.0, float(pose_fps)))) + 1)
        seg_points_full, seg_forwards_full = sample_waypoints(
            element_instance=element_instance,
            landmark=target_landmark,
            start_pos=projected_start,
            num_points=target_points,
        )
        seg_points_full, repair_meta = _repair_points_by_obstacles(
            seg_points_full,
            obstacles_xyz=obstacles_xyz,
            keepout_boxes=keepout_boxes,
            safety_distance=float(safety_distance),
            max_lift_step_m=float(repair_max_lift_step_m),
            max_total_lift_m=float(repair_max_total_lift_m),
            keepout_margin_xy=float(keepout_margin_xy),
            keepout_margin_z=float(keepout_margin_z),
        )

        seg_points = seg_points_full
        seg_forwards = seg_forwards_full

        if had_previous_points:
            seg_points = seg_points[1:]
            seg_forwards = seg_forwards[1:]

        if seg_points.shape[0] == 0:
            continue

        all_points.append(seg_points.astype(np.float32))
        all_forwards.append(seg_forwards.astype(np.float32))
        start_idx = int(sum(part.shape[0] for part in all_points[:-1]))
        end_idx = int(start_idx + seg_points.shape[0] - 1)
        shared_start = bool(
            had_previous_points
            and previous_terminal_point is not None
            and seg_points_full.shape[0] > 0
            and float(np.linalg.norm(seg_points_full[0] - previous_terminal_point)) <= 1e-3
        )
        logical_start_idx = int(max(0, start_idx - 1)) if shared_start else int(start_idx)
        logical_num_points = int(seg_points.shape[0] + 1) if shared_start else int(seg_points.shape[0])
        logical_start_point = (
            previous_terminal_point.astype(float).tolist()
            if shared_start and previous_terminal_point is not None
            else (seg_points_full[0].astype(float).tolist() if seg_points_full.shape[0] > 0 else seg_points[0].astype(float).tolist())
        )
        logical_end_point = seg_points_full[-1].astype(float).tolist() if seg_points_full.shape[0] > 0 else seg_points[-1].astype(float).tolist()
        segments.append(
            {
                "segment_id": f"seg_{index:02d}",
                "element_instance_id": str(element_instance.get("element_instance_id", "") or f"elem_{index:02d}"),
                "element_class": str(element_instance.get("element_class", "") or ""),
                "element_display_name": str(element_instance.get("element_display_name", "") or ""),
                "target_instance_id": str(target_landmark.get("instance_id", "") or ""),
                "target_binding": str(element_instance.get("target_binding", "primary") or "primary"),
                "params": dict(element_instance.get("params", {}) or {}),
                "start": logical_start_point,
                "end": logical_end_point,
                "num_points": int(logical_num_points),
                "num_points_unique": int(seg_points.shape[0]),
                "shared_start_with_previous": bool(shared_start),
                "waypoint_start_idx": int(logical_start_idx),
                "waypoint_end_idx": int(end_idx),
                "projected_start": bool(projected),
                "project_distance_m": float(project_distance),
                "repair": repair_meta,
            }
        )
        current = seg_points[-1]

    if not all_points:
        raw = start_pos.reshape(1, 3).astype(np.float32)
        forwards = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    else:
        raw = np.concatenate(all_points, axis=0).astype(np.float32)
        forwards = np.concatenate(all_forwards, axis=0).astype(np.float32)

    return raw, forwards, segments
