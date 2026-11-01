from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

from .base import BaseSimBridge


class GaussianSplatBridge(BaseSimBridge):
    """HTTP bridge for headless Gaussian-splat rendering and static geometry."""

    def __init__(self, scene_id: str, config: Optional[dict] = None) -> None:
        super().__init__(scene_id, config)
        self.render_url = str(self.config.get("render_url", "http://127.0.0.1:18080/render"))
        self.service_url = self.render_url.rsplit("/render", 1)[0]
        self.request_timeout_sec = float(self.config.get("request_timeout_sec", 180.0))
        self.image_width = int(self.config.get("image_width", 1920))
        self.image_height = int(self.config.get("image_height", 1080))
        self.fov = float(self.config.get("fov", 72.0))
        self.scene_manifest = str(self.config.get("scene_manifest", "") or "")
        self.geometry_path = str(self.config.get("geometry_path", "") or "")
        self.raycast_clearance_m = float(self.config.get("raycast_clearance_m", 0.75))
        self._geometry_world: Optional[np.ndarray] = None

        if bool(self.config.get("connect_on_init", True)):
            self.healthcheck()
            if self.scene_manifest:
                self.reset_scene(scene_id)

    def healthcheck(self) -> dict[str, Any]:
        response = requests.get(f"{self.service_url}/health", timeout=self.request_timeout_sec)
        response.raise_for_status()
        payload = response.json()
        if not bool(payload.get("ok", False)):
            raise RuntimeError(f"3DGS render service is not ready: {payload}")
        return payload

    def reset_scene(self, scene_id: str) -> None:
        payload: dict[str, Any] = {"scene_id": str(scene_id)}
        if self.scene_manifest:
            payload["manifest_path"] = self.scene_manifest
        response = requests.post(
            f"{self.service_url}/scene/load",
            json=payload,
            timeout=max(self.request_timeout_sec, 120.0),
        )
        response.raise_for_status()
        result = response.json()
        if not bool(result.get("ok", False)):
            raise RuntimeError(f"failed to load 3DGS scene: {result}")
        self.scene_id = str(scene_id)
        self._geometry_world = None

    def set_uav_pose(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        pitch: float,
        roll: float,
        vehicle_or_actor: Optional[str] = None,
    ) -> None:
        del vehicle_or_actor
        self._update_last_pose_enu(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

    def _intrinsics(self) -> dict[str, float]:
        focal = 0.5 * self.image_width / math.tan(math.radians(self.fov) * 0.5)
        return {
            "width": self.image_width,
            "height": self.image_height,
            "fx": focal,
            "fy": focal,
            "cx": self.image_width * 0.5,
            "cy": self.image_height * 0.5,
        }

    def _render_raw(self, mode: str) -> requests.Response:
        pose = self._last_pose_enu
        response = requests.post(
            self.render_url,
            json={
                "scene_id": self.scene_id,
                "mode": mode,
                "pose_enu": {
                    "x": pose.x,
                    "y": pose.y,
                    "z": pose.z,
                    "yaw": pose.yaw,
                    "pitch": pose.pitch,
                    "roll": pose.roll,
                },
                "camera": self._intrinsics(),
            },
            timeout=self.request_timeout_sec,
        )
        response.raise_for_status()
        return response

    def capture_rgb(self) -> Any:
        response = self._render_raw("rgb")
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("3DGS service returned an invalid RGB image")
        return image

    def capture_depth(self) -> Any:
        response = self._render_raw("depth")
        try:
            with np.load(io.BytesIO(response.content)) as payload:
                depth = payload["depth"]
        except Exception:
            depth = np.load(io.BytesIO(response.content))
        return np.asarray(depth, dtype=np.float32)

    def capture_seg(self) -> Any:
        response = self._render_raw("seg")
        image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError("3DGS service returned an invalid alpha mask")
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _load_local_geometry(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            points = np.load(path)
        elif path.suffix.lower() == ".npz":
            with np.load(path) as payload:
                key = "points" if "points" in payload else payload.files[0]
                points = payload[key]
        else:
            raise ValueError("geometry_path must be .npy or .npz; convert PLY once during scene setup")
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"invalid geometry array shape: {points.shape}")
        return points[:, :3]

    def get_lidar(self) -> Any:
        if self._geometry_world is None:
            if self.geometry_path:
                self._geometry_world = self._load_local_geometry(Path(self.geometry_path))
            else:
                response = requests.get(f"{self.service_url}/geometry", timeout=max(self.request_timeout_sec, 120.0))
                response.raise_for_status()
                with np.load(io.BytesIO(response.content)) as payload:
                    self._geometry_world = np.asarray(payload["points"], dtype=np.float32)
        pose = self._last_pose_enu
        translated = self._geometry_world - np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        yaw = math.radians(float(pose.yaw))
        world_to_local = np.array(
            [[math.cos(yaw), math.sin(yaw), 0.0], [-math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return translated @ world_to_local.T

    def raycast(self, origin: tuple[float, float, float], target: tuple[float, float, float]) -> Optional[bool]:
        if self._geometry_world is None:
            self.get_lidar()
        points = np.asarray(self._geometry_world, dtype=np.float32)
        if points.size == 0:
            return None
        start = np.asarray(origin, dtype=np.float32)
        end = np.asarray(target, dtype=np.float32)
        direction = end - start
        length_sq = float(np.dot(direction, direction))
        if length_sq <= 1e-8:
            return False
        t = np.clip(((points - start) @ direction) / length_sq, 0.0, 1.0)
        closest = start + t[:, None] * direction
        distance_sq = np.sum((points - closest) ** 2, axis=1)
        interior = (t > 0.01) & (t < 0.99)
        return bool(np.any(interior & (distance_sq <= self.raycast_clearance_m**2)))

    def shutdown(self) -> None:
        return None
