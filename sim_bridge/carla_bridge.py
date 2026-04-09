from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import BaseSimBridge


class CarlaBridge(BaseSimBridge):
    def __init__(self, scene_id: str, config: Optional[dict] = None) -> None:
        super().__init__(scene_id, config)
        self._client = None
        self._world = None
        self._ego_actor = None
        self._rgb_sensor = None
        self._depth_sensor = None
        self._seg_sensor = None
        self._lidar_sensor = None

        self.sim_ip = self.config.get("sim_ip", "127.0.0.1")
        self.sim_port = int(self.config.get("sim_port", 2000))
        self.timeout_sec = float(self.config.get("timeout_sec", 5.0))
        self.town = self.config.get("town")
        self.actor_role_name = self.config.get("actor_role_name", "ego")

        self._latest_rgb = None
        self._latest_depth = None
        self._latest_seg = None
        self._latest_lidar = None

        if self.config.get("connect_on_init", True):
            self._connect()
            if self.town:
                self._load_world(self.town)

    def _connect(self) -> None:
        try:
            import carla  # type: ignore
        except ImportError as exc:
            raise ImportError("`carla` package is required for CarlaBridge") from exc

        client = carla.Client(self.sim_ip, self.sim_port)
        client.set_timeout(self.timeout_sec)
        world = client.get_world()

        self._client = client
        self._world = world
        self._ego_actor = self._resolve_actor_by_role_name(self.actor_role_name)

    def _require_world(self) -> Any:
        if self._world is None:
            raise RuntimeError("CARLA world is not connected. Set `connect_on_init: true`.")
        return self._world

    def _require_actor(self) -> Any:
        if self._ego_actor is None:
            raise RuntimeError(
                "CARLA ego actor not found. Configure `actor_role_name` or set `vehicle_or_actor` in set_uav_pose."
            )
        return self._ego_actor

    def _load_world(self, town: str) -> None:
        if self._client is None:
            raise RuntimeError("CARLA client is not connected.")
        self._world = self._client.load_world(town)
        self._ego_actor = self._resolve_actor_by_role_name(self.actor_role_name)

    def _resolve_actor_by_role_name(self, role_name: str) -> Any:
        world = self._require_world()
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == role_name:
                return actor
        return None

    def _resolve_actor(self, vehicle_or_actor: Optional[str]) -> Any:
        world = self._require_world()

        if vehicle_or_actor is None:
            return self._require_actor()

        if vehicle_or_actor.isdigit():
            actor = world.get_actor(int(vehicle_or_actor))
            if actor is None:
                raise RuntimeError(f"CARLA actor id not found: {vehicle_or_actor}")
            return actor

        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == vehicle_or_actor:
                return actor
        raise RuntimeError(f"CARLA actor role_name not found: {vehicle_or_actor}")

    def reset_scene(self, scene_id: str) -> None:
        self.scene_id = scene_id
        if self._client is not None:
            self._load_world(scene_id)

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
        actor = self._resolve_actor(vehicle_or_actor)
        world = self._require_world()

        import carla  # type: ignore

        transform = carla.Transform(
            location=carla.Location(x=float(x), y=float(y), z=float(z)),
            rotation=carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=float(roll)),
        )
        actor.set_transform(transform)
        self._update_last_pose_enu(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

        if self.config.get("tick_after_set_pose", True):
            world.tick()

    def _convert_carla_image_to_bgr(self, image: Any) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        return arr[:, :, :3][:, :, ::-1].copy()

    def _convert_carla_depth_to_meters(self, image: Any) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4)).astype(np.float32)
        normalized = (arr[:, :, 2] * 65536.0 + arr[:, :, 1] * 256.0 + arr[:, :, 0]) / 16777215.0
        return (1000.0 * normalized).astype(np.float32)

    def _convert_carla_seg_to_label(self, image: Any) -> np.ndarray:
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))
        return arr[:, :, 2].copy()

    def _capture_from_sensor(self, sensor: Any, timeout_sec: float = 2.0) -> Any:
        if sensor is None:
            return None
        queue = self._get_sensor_queue(sensor)
        if queue is None:
            return None
        try:
            return queue.get(timeout=timeout_sec)
        except Exception:
            return None

    def _get_sensor_queue(self, sensor: Any) -> Any:
        return getattr(sensor, "_frame_queue", None)

    def _ensure_sensor_queue(self, sensor: Any) -> None:
        from queue import Queue

        if getattr(sensor, "_frame_queue", None) is not None:
            return
        sensor._frame_queue = Queue(maxsize=4)

        def _callback(data: Any) -> None:
            q = sensor._frame_queue
            if q.full():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put(data)

        sensor.listen(_callback)

    def _attach_sensors_if_needed(self) -> None:
        if not self.config.get("attach_sensors", False):
            return
        if self._rgb_sensor is not None:
            return

        world = self._require_world()
        actor = self._require_actor()
        blueprint_library = world.get_blueprint_library()

        image_width = str(int(self.config.get("image_width", 1920)))
        image_height = str(int(self.config.get("image_height", 1080)))
        fov = str(float(self.config.get("fov", 90.0)))

        import carla  # type: ignore

        sensor_transform = carla.Transform(carla.Location(x=1.5, z=2.0))

        def _spawn_camera(bp_name: str) -> Any:
            bp = blueprint_library.find(bp_name)
            bp.set_attribute("image_size_x", image_width)
            bp.set_attribute("image_size_y", image_height)
            bp.set_attribute("fov", fov)
            return world.spawn_actor(bp, sensor_transform, attach_to=actor)

        self._rgb_sensor = _spawn_camera("sensor.camera.rgb")
        self._depth_sensor = _spawn_camera("sensor.camera.depth")
        self._seg_sensor = _spawn_camera("sensor.camera.semantic_segmentation")

        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", str(float(self.config.get("lidar_range", 120.0))))
        lidar_bp.set_attribute("rotation_frequency", str(float(self.config.get("lidar_rotation_frequency", 10.0))))
        lidar_bp.set_attribute("channels", str(int(self.config.get("lidar_channels", 32))))
        lidar_bp.set_attribute("points_per_second", str(int(self.config.get("lidar_points_per_second", 56000))))
        self._lidar_sensor = world.spawn_actor(lidar_bp, sensor_transform, attach_to=actor)

        for sensor in [self._rgb_sensor, self._depth_sensor, self._seg_sensor, self._lidar_sensor]:
            self._ensure_sensor_queue(sensor)

    def capture_rgb(self) -> Any:
        self._attach_sensors_if_needed()
        if self._rgb_sensor is None:
            return self._latest_rgb
        image = self._capture_from_sensor(self._rgb_sensor)
        if image is None:
            return self._latest_rgb
        self._latest_rgb = self._convert_carla_image_to_bgr(image)
        return self._latest_rgb

    def capture_depth(self) -> Any:
        self._attach_sensors_if_needed()
        if self._depth_sensor is None:
            return self._latest_depth
        image = self._capture_from_sensor(self._depth_sensor)
        if image is None:
            return self._latest_depth
        self._latest_depth = self._convert_carla_depth_to_meters(image)
        return self._latest_depth

    def capture_seg(self) -> Any:
        self._attach_sensors_if_needed()
        if self._seg_sensor is None:
            return self._latest_seg
        image = self._capture_from_sensor(self._seg_sensor)
        if image is None:
            return self._latest_seg
        self._latest_seg = self._convert_carla_seg_to_label(image)
        return self._latest_seg

    def get_lidar(self) -> Any:
        self._attach_sensors_if_needed()
        if self._lidar_sensor is None:
            return np.empty((0, 4), dtype=np.float32)

        data = self._capture_from_sensor(self._lidar_sensor)
        if data is None:
            if self._latest_lidar is None:
                return np.empty((0, 4), dtype=np.float32)
            return self._latest_lidar

        arr = np.frombuffer(data.raw_data, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        self._latest_lidar = arr.reshape(-1, 4)
        return self._latest_lidar

    def raycast(self, origin: tuple[float, float, float], target: tuple[float, float, float]) -> Optional[bool]:
        world = self._require_world()
        try:
            import carla  # type: ignore
        except ImportError:
            return None

        origin_loc = carla.Location(x=float(origin[0]), y=float(origin[1]), z=float(origin[2]))
        target_loc = carla.Location(x=float(target[0]), y=float(target[1]), z=float(target[2]))
        hit_result = world.cast_ray(origin_loc, target_loc)
        return len(hit_result) == 0

    def shutdown(self) -> None:
        for sensor in [self._rgb_sensor, self._depth_sensor, self._seg_sensor, self._lidar_sensor]:
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
                try:
                    sensor.destroy()
                except Exception:
                    pass

        self._rgb_sensor = None
        self._depth_sensor = None
        self._seg_sensor = None
        self._lidar_sensor = None
