from __future__ import annotations

import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests

from .base import BaseSimBridge


class SIBRBridge(BaseSimBridge):
    def __init__(self, scene_id: str, config: Optional[dict] = None) -> None:
        super().__init__(scene_id, config)
        self._sim_process: Optional[subprocess.Popen] = None
        self.render_url = self.config.get("render_url", "http://localhost:18080/render")
        self._last_pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        if self.config.get("launch_sim", False):
            self._launch_simulator()
            time.sleep(float(self.config.get("launch_wait_sec", 10)))

    def _launch_simulator(self) -> None:
        viewer = Path("envs/gs/SIBR_viewers/install/bin/SIBR_gaussianHierarchyViewer_app")
        dataset_dir = Path("envs/gs") / self.scene_id
        if not viewer.exists():
            raise FileNotFoundError(f"SIBR viewer not found: {viewer}")
        if not dataset_dir.exists():
            raise FileNotFoundError(f"SIBR scene dir not found: {dataset_dir}")

        command = [
            str(viewer),
            "--path",
            str(dataset_dir / "camera_calibration/aligned"),
            "--scaffold",
            str(dataset_dir / "output/scaffold/point_cloud/iteration_30000"),
            "--model-path",
            str(dataset_dir / "output/merged.hier"),
            "--images-path",
            str(dataset_dir / "camera_calibration/rectified/images"),
        ]
        self._sim_process = subprocess.Popen(command)

    def reset_scene(self, scene_id: str) -> None:
        self.scene_id = scene_id

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
        self._last_pose = (x, y, z, yaw, pitch, roll)
        self._update_last_pose_enu(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

    def _render_raw(self, mode: str = "rgb") -> bytes:
        payload = {
            "mode": mode,
            "pose": {
                "x": self._last_pose[0],
                "y": self._last_pose[1],
                "z": self._last_pose[2],
                "yaw": self._last_pose[3],
                "pitch": self._last_pose[4],
                "roll": self._last_pose[5],
            },
        }
        response = requests.post(self.render_url, data={"payload": json.dumps(payload)}, timeout=5)
        response.raise_for_status()
        return bytes(response.content)

    def capture_rgb(self) -> Any:
        raw = self._render_raw(mode="rgb")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return img

    def capture_depth(self) -> Any:
        raw = self._render_raw(mode="depth")
        try:
            arr = np.load(io.BytesIO(raw))
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return np.empty((0, 0), dtype=np.float32)
            return np.asarray(img, dtype=np.float32)

    def capture_seg(self) -> Any:
        raw = self._render_raw(mode="seg")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return img

    def get_lidar(self) -> Any:
        return np.empty((0, 3), dtype=np.float32)

    def raycast(self, origin: tuple[float, float, float], target: tuple[float, float, float]) -> Optional[bool]:
        return None

    def shutdown(self) -> None:
        if self._sim_process is not None and self._sim_process.poll() is None:
            self._sim_process.terminate()
