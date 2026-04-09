from __future__ import annotations

import math
from typing import Any

import numpy as np


def _bbox_axes_from_list(bbox_list: list[float]) -> tuple[float, float, float]:
    sx = float(bbox_list[3]) if len(bbox_list) > 3 else 3.0
    sy = float(bbox_list[4]) if len(bbox_list) > 4 else 3.0
    sz = float(bbox_list[5]) if len(bbox_list) > 5 else 3.0
    return max(1e-3, sx), max(1e-3, sy), max(1e-3, sz)


def _yaw_rotation_inv(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.asarray([[c, s], [-s, c]], dtype=np.float32)


def _obb_signed_distance(point_xyz: np.ndarray, keepout_box: dict[str, Any], *, margin_xy: float, margin_z: float) -> float:
    center = np.asarray(list(keepout_box.get("center_3d", [0.0, 0.0, 0.0])[:3]), dtype=np.float32)
    bbox = list(keepout_box.get("bbox_list", []) or [])
    sx, sy, sz = _bbox_axes_from_list(bbox)
    yaw_deg = float(bbox[6]) if len(bbox) > 6 else 0.0
    local_xy = (point_xyz[:2].astype(np.float32) - center[:2]) @ _yaw_rotation_inv(yaw_deg).T
    local = np.asarray(
        [
            float(local_xy[0]),
            float(local_xy[1]),
            float(point_xyz[2] - center[2]),
        ],
        dtype=np.float32,
    )
    half = np.asarray(
        [
            0.5 * float(sx) + float(margin_xy),
            0.5 * float(sy) + float(margin_xy),
            0.5 * float(sz) + float(margin_z),
        ],
        dtype=np.float32,
    )
    q = np.abs(local) - half
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(float(q[0]), max(float(q[1]), float(q[2]))), 0.0)
    return float(outside + inside)


def _keepout_min_distance(point_xyz: np.ndarray, keepout_boxes: list[dict[str, Any]], *, margin_xy: float, margin_z: float) -> float:
    if not keepout_boxes:
        return float("inf")
    return min(_obb_signed_distance(point_xyz, box, margin_xy=margin_xy, margin_z=margin_z) for box in keepout_boxes)


def _resample_polyline(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    if points.shape[0] <= 1:
        return points.astype(np.float32)

    out = [points[0]]
    k = max(1, int(samples_per_segment))
    for i in range(points.shape[0] - 1):
        p0 = points[i]
        p1 = points[i + 1]
        for j in range(1, k + 1):
            t = j / k
            out.append((1.0 - t) * p0 + t * p1)
    return np.asarray(out, dtype=np.float32)


def _moving_average(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or points.shape[0] < 3:
        return points
    w = max(1, int(window))
    pad = w // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(points)
    for i in range(points.shape[0]):
        out[i] = padded[i : i + w].mean(axis=0)
    return out.astype(np.float32)


def smooth_trajectory(points: np.ndarray, samples_per_segment: int = 4, smooth_window: int = 5) -> np.ndarray:
    dense = _resample_polyline(points, samples_per_segment=samples_per_segment)
    smooth = _moving_average(dense, window=smooth_window)
    if smooth.shape[0] >= 2:
        smooth[0] = dense[0]
        smooth[-1] = dense[-1]
    return smooth


def build_poses(points: np.ndarray, fps: float) -> list[dict[str, float]]:
    poses: list[dict[str, float]] = []
    if points.shape[0] == 0:
        return poses

    dt = 1.0 / max(1e-6, float(fps))
    for i in range(points.shape[0]):
        p = points[i]
        if i < points.shape[0] - 1:
            q = points[i + 1]
        else:
            q = points[i]
        dx = float(q[0] - p[0])
        dy = float(q[1] - p[1])
        yaw = math.degrees(math.atan2(dy, dx)) if abs(dx) + abs(dy) > 1e-9 else (poses[-1]["yaw"] if poses else 0.0)
        poses.append(
            {
                "frame": float(i),
                "t": float(i * dt),
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "yaw": float(yaw),
                "pitch": 0.0,
                "roll": 0.0,
            }
        )
    return poses


def check_constraints(
    poses: list[dict[str, float]],
    limits: dict[str, float],
) -> dict[str, Any]:
    v_max = float(limits.get("v_max", 8.0))
    a_max = float(limits.get("a_max", 10.0))
    yaw_rate_max = float(limits.get("yaw_rate_max", 120.0))

    speed_peak = 0.0
    accel_peak = 0.0
    yaw_rate_peak = 0.0
    violations: list[dict[str, Any]] = []

    if len(poses) < 2:
        return {
            "feasible": True,
            "speed_peak": speed_peak,
            "accel_peak": accel_peak,
            "yaw_rate_peak": yaw_rate_peak,
            "violations": violations,
        }

    dt = max(1e-6, poses[1]["t"] - poses[0]["t"])
    speeds: list[float] = []
    for i in range(1, len(poses)):
        dx = poses[i]["x"] - poses[i - 1]["x"]
        dy = poses[i]["y"] - poses[i - 1]["y"]
        dz = poses[i]["z"] - poses[i - 1]["z"]
        speed = float(math.sqrt(dx * dx + dy * dy + dz * dz) / dt)
        speed_peak = max(speed_peak, speed)
        speeds.append(speed)
        if speed > v_max:
            violations.append({"frame": i, "type": "speed", "value": speed, "limit": v_max})

        dyaw = abs(poses[i]["yaw"] - poses[i - 1]["yaw"])
        if dyaw > 180.0:
            dyaw = 360.0 - dyaw
        yaw_rate = float(dyaw / dt)
        yaw_rate_peak = max(yaw_rate_peak, yaw_rate)
        if yaw_rate > yaw_rate_max:
            violations.append({"frame": i, "type": "yaw_rate", "value": yaw_rate, "limit": yaw_rate_max})

    for i in range(1, len(speeds)):
        accel = abs(speeds[i] - speeds[i - 1]) / dt
        accel_peak = max(accel_peak, accel)
        if accel > a_max:
            violations.append({"frame": i + 1, "type": "accel", "value": accel, "limit": a_max})

    return {
        "feasible": len(violations) == 0,
        "speed_peak": speed_peak,
        "accel_peak": accel_peak,
        "yaw_rate_peak": yaw_rate_peak,
        "violations": violations,
    }


def check_collision(
    points: np.ndarray,
    obstacles_xyz: np.ndarray,
    safety_distance: float = 2.0,
    keepout_boxes: list[dict[str, Any]] | None = None,
    keepout_margin_xy: float = 0.0,
    keepout_margin_z: float = 0.0,
) -> dict[str, Any]:
    if points.shape[0] == 0 and (not keepout_boxes):
        return {"collision_free": True, "min_distance": float("inf"), "violations": []}

    min_d = float("inf")
    violations: list[dict[str, Any]] = []
    sample_points = points[:: max(1, points.shape[0] // 600)] if points.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)

    for idx, point in enumerate(sample_points):
        d_min = float("inf")
        if obstacles_xyz.shape[0] > 0:
            delta = obstacles_xyz - point.reshape(1, 3)
            d = np.sqrt(np.sum(delta * delta, axis=1))
            d_min = float(d.min())
        box_d_min = _keepout_min_distance(
            point.astype(np.float32),
            list(keepout_boxes or []),
            margin_xy=float(keepout_margin_xy),
            margin_z=float(keepout_margin_z),
        )
        min_d = min(min_d, d_min, box_d_min)
        collided = (d_min < float(safety_distance)) or (box_d_min < 0.0)
        if collided:
            violations.append(
                {
                    "sample_index": idx,
                    "distance": float(min(d_min, box_d_min)),
                    "pointcloud_threshold": float(safety_distance),
                    "inside_keepout": bool(box_d_min < 0.0),
                }
            )

    return {"collision_free": len(violations) == 0, "min_distance": min_d, "violations": violations}
