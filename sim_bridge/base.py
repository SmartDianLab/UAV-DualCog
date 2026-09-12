from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Optional


@dataclass
class Pose6D:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float


@dataclass
class BridgeFrame:
    timestamp: float
    pose_enu: Pose6D
    rgb: Optional[Any] = None
    depth: Optional[Any] = None
    seg: Optional[Any] = None
    lidar: Optional[Any] = None


class BaseSimBridge(ABC):
    def __init__(self, scene_id: str, config: Optional[dict] = None) -> None:
        self.scene_id = scene_id
        self.config = config or {}
        self._last_pose_enu = Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def _update_last_pose_enu(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float,
        pitch: float,
        roll: float,
    ) -> None:
        self._last_pose_enu = Pose6D(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

    def _retry_call(self, fn: Any, default: Any = None) -> Any:
        retries = int(self.config.get("capture_retries", 1))
        backoff_sec = float(self.config.get("capture_retry_backoff_sec", 0.02))
        last_error: Optional[Exception] = None

        for _ in range(max(1, retries)):
            try:
                return fn()
            except Exception as exc:
                last_error = exc
                time.sleep(backoff_sec)

        if bool(self.config.get("raise_on_capture_error", False)) and last_error is not None:
            raise last_error
        return default

    @abstractmethod
    def reset_scene(self, scene_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def capture_rgb(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def capture_depth(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def capture_seg(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_lidar(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def raycast(self, origin: tuple[float, float, float], target: tuple[float, float, float]) -> Optional[bool]:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError

    def capture_frame(
        self,
        include_rgb: bool = True,
        include_depth: bool = True,
        include_seg: bool = True,
        include_lidar: bool = False,
    ) -> BridgeFrame:
        timestamp = time.time()
        rgb = self._retry_call(self.capture_rgb, default=None) if include_rgb else None
        depth = self._retry_call(self.capture_depth, default=None) if include_depth else None
        seg = self._retry_call(self.capture_seg, default=None) if include_seg else None
        lidar = self._retry_call(self.get_lidar, default=None) if include_lidar else None

        return BridgeFrame(
            timestamp=timestamp,
            pose_enu=self._last_pose_enu,
            rgb=rgb,
            depth=depth,
            seg=seg,
            lidar=lidar,
        )
