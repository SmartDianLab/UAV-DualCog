from __future__ import annotations

import io
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseSimBridge


class UnrealCVBridge(BaseSimBridge):
    def __init__(self, scene_id: str, config: Optional[dict] = None) -> None:
        super().__init__(scene_id, config)
        self._client = None
        self._sim_process: Optional[subprocess.Popen] = None
        self.camera_id = int(self.config.get("camera_id", 1))
        self.sim_ip = self.config.get("sim_ip", "127.0.0.1")
        self.sim_port = int(self.config.get("sim_port", 9000))

        if self.config.get("launch_sim", False):
            self._launch_simulator()
            time.sleep(float(self.config.get("launch_wait_sec", 15)))

        if self.config.get("connect_on_init", True):
            self._connect()

    def _launch_simulator(self) -> None:
        start_script = Path("envs/ue") / self.scene_id / "CitySample.sh"
        if not start_script.exists():
            raise FileNotFoundError(f"UE start script not found: {start_script}")
        self._sim_process = subprocess.Popen(["bash", str(start_script)])

    def _connect(self) -> None:
        try:
            from unrealcv import Client
        except ImportError as exc:
            raise ImportError("`unrealcv` package is required for UnrealCVBridge") from exc

        client = Client((self.sim_ip, self.sim_port))
        if not client.connect():
            raise RuntimeError(f"Failed to connect UnrealCV at {self.sim_ip}:{self.sim_port}")
        self._client = client

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("UnrealCV client is not connected. Set `connect_on_init: true`.")
        return self._client

    def _to_bytes(self, data: Any) -> bytes:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(data, str):
            return data.encode("latin1", errors="ignore")
        return bytes(data)

    def _decode_png(self, data: Any, flags: int) -> np.ndarray:
        raw = self._to_bytes(data)
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
        if arr is None:
            return np.empty((0, 0), dtype=np.uint8)
        return arr

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
        client = self._require_client()
        x_cm, y_cm, z_cm = x * 100.0, -y * 100.0, z * 100.0
        yaw_ue = -yaw
        client.request(f"vset /camera/{self.camera_id}/location {x_cm} {y_cm} {z_cm}")
        client.request(f"vset /camera/{self.camera_id}/rotation {pitch} {yaw_ue} {roll}")
        self._update_last_pose_enu(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

    def capture_rgb(self) -> Any:
        client = self._require_client()
        data = client.request(f"vget /camera/{self.camera_id}/lit png")
        return self._decode_png(data, cv2.IMREAD_COLOR)

    def capture_depth(self) -> Any:
        client = self._require_client()
        data = client.request(f"vget /camera/{self.camera_id}/depth npy")
        raw = self._to_bytes(data)
        try:
            return np.asarray(np.load(io.BytesIO(raw)), dtype=np.float32)
        except Exception:
            img = self._decode_png(raw, cv2.IMREAD_UNCHANGED)
            return np.asarray(img, dtype=np.float32)

    def capture_seg(self) -> Any:
        client = self._require_client()
        data = client.request(f"vget /camera/{self.camera_id}/object_mask png")
        return self._decode_png(data, cv2.IMREAD_COLOR)

    def get_lidar(self) -> Any:
        return np.empty((0, 3), dtype=np.float32)

    def raycast(self, origin: tuple[float, float, float], target: tuple[float, float, float]) -> Optional[bool]:
        return None

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.disconnect()
        if self._sim_process is not None and self._sim_process.poll() is None:
            self._sim_process.terminate()
