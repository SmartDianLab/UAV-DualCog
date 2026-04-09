from __future__ import annotations

import math
from typing import Any

import numpy as np


def enu_to_airsim_ned_position(x: float, y: float, z: float) -> tuple[float, float, float]:
    return float(x), float(-y), float(-z)


def airsim_ned_to_enu_points(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.size == 0:
        return points_xyz
    out = np.asarray(points_xyz, dtype=np.float32).copy()
    out[:, 1] = -out[:, 1]
    out[:, 2] = -out[:, 2]
    return out


def yaw_rotation_matrix_enu(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    return np.array(
        [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def local_points_to_world_enu(
    points_local: np.ndarray,
    pose_enu: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    if points_local.size == 0:
        return points_local
    x, y, z, yaw, _pitch, _roll = pose_enu
    rot = yaw_rotation_matrix_enu(yaw)
    pts_rot = np.asarray(points_local, dtype=np.float32) @ rot.T
    pts_rot[:, 0] += float(x)
    pts_rot[:, 1] += float(y)
    pts_rot[:, 2] += float(z)
    return pts_rot


def build_bev_info_from_points(points: np.ndarray, resolution_m: float) -> dict[str, float]:
    if points.shape[0] == 0:
        return {
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "resolution": float(resolution_m),
            "width": 1,
            "height": 1,
        }

    x_min, y_min = float(points[:, 0].min()), float(points[:, 1].min())
    x_max, y_max = float(points[:, 0].max()), float(points[:, 1].max())
    return build_bev_info_from_bounds(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, resolution_m=resolution_m)


def build_bev_info_from_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution_m: float,
) -> dict[str, float]:
    width = int(np.ceil((float(x_max) - float(x_min)) / float(resolution_m))) + 1
    height = int(np.ceil((float(y_max) - float(y_min)) / float(resolution_m))) + 1
    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "resolution": float(resolution_m),
        "width": int(max(1, width)),
        "height": int(max(1, height)),
    }


def xy_to_bev_indices(
    x: np.ndarray,
    y: np.ndarray,
    bev_info: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    res = float(bev_info["resolution"])
    x_min = float(bev_info["x_min"])
    y_min = float(bev_info["y_min"])
    xi = np.floor((np.asarray(x, dtype=np.float32) - x_min) / res).astype(np.int32)
    yi = np.floor((np.asarray(y, dtype=np.float32) - y_min) / res).astype(np.int32)
    return xi, yi


def points_to_bev_indices(points: np.ndarray, bev_info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return xy_to_bev_indices(points[:, 0], points[:, 1], bev_info)


def bev_index_valid_mask(x_idx: np.ndarray, y_idx: np.ndarray, bev_info: dict[str, Any]) -> np.ndarray:
    width = int(bev_info.get("width", 0))
    height = int(bev_info.get("height", 0))
    return (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)
