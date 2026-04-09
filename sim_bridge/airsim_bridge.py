from __future__ import annotations

import copy
import glob
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseSimBridge
from coord_transform_utils import enu_to_airsim_ned_position


AIRSIM_SETTINGS_TEMPLATE = {
    "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
    "SettingsVersion": 1.2,
    "SimMode": "ComputerVision",
    "ViewMode": "NoDisplay",
    "ClockSpeed": 1,
    "CameraDefaults": {
        "CaptureSettings": [
            {
                "ImageType": 0,
                "Width": 3840,
                "Height": 2160,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03,
            },
            {
                "ImageType": 2,
                "Width": 3840,
                "Height": 2160,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03,
            },
            {
                "ImageType": 3,
                "Width": 3840,
                "Height": 2160,
                "FOV_Degrees": 90,
                "AutoExposureMaxBrightness": 1,
                "AutoExposureMinBrightness": 0.03,
            },
        ],
        "X": 0,
        "Y": 0,
        "Z": 0,
        "Pitch": 0,
        "Roll": 0,
        "Yaw": 0,
    },
    "Recording": {"RecordInterval": 0.001, "Enabled": False, "Cameras": []},
    "SubWindows": [],
    "Vehicles": {},
}

AIRSIM_CAMERA_CAPTURE_SETTING_TEMPLATES = {
    int(item["ImageType"]): copy.deepcopy(item)
    for item in AIRSIM_SETTINGS_TEMPLATE["CameraDefaults"]["CaptureSettings"]
    if isinstance(item, dict) and "ImageType" in item
}


class AirSimBridge(BaseSimBridge):
    def __init__(self, scene_id: str, config: Optional[dict] = None) -> None:
        super().__init__(scene_id, config)
        self._client = None
        self._sim_process: Optional[subprocess.Popen] = None
        self._settings_path: Optional[Path] = None
        self.sim_ip = self.config.get("sim_ip", "127.0.0.1")
        self.sim_port = int(self.config.get("sim_port", 41451))
        self.camera_name = self.config.get("camera_name", "front_custom")
        self.vehicle_name = self.config.get("vehicle_name", "Drone_1")
        self._active_camera_name: Optional[str] = None

        if self.config.get("auto_select_port_on_conflict", True):
            self.sim_port = self._resolve_available_port(self.sim_port)
            self.config["sim_port"] = self.sim_port

        if self.config.get("launch_sim", False):
            self._launch_simulator()
            self._wait_for_launch_ready()

        if self.config.get("connect_on_init", True):
            self._connect()
            self._apply_segmentation_ids_if_needed()

    def _apply_segmentation_ids_if_needed(self) -> None:
        if not bool(self.config.get("setup_segmentation_ids", False)):
            return
        client = self._require_client()
        mappings = self.config.get("segmentation_object_ids", [])
        if not isinstance(mappings, (list, tuple)):
            return
        for item in mappings:
            if not isinstance(item, dict):
                continue
            pattern = str(item.get("pattern", "")).strip()
            if not pattern:
                continue
            object_id = int(item.get("id", 0))
            is_regex = bool(item.get("regex", True))
            try:
                client.simSetSegmentationObjectID(pattern, object_id, is_regex)
            except Exception:
                continue

    def _is_port_open(self, host: str, port: int, timeout_sec: float = 0.4) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_sec)
            return sock.connect_ex((host, port)) == 0

    def _resolve_available_port(self, preferred_port: int) -> int:
        if not self._is_port_open(self.sim_ip, preferred_port):
            return preferred_port

        if not self.config.get("auto_select_port_on_conflict", True):
            raise RuntimeError(f"AirSim port occupied: {preferred_port}")

        scan_max = int(self.config.get("port_scan_max_tries", 200))
        for offset in range(1, max(2, scan_max + 1)):
            candidate = preferred_port + offset
            if not self._is_port_open(self.sim_ip, candidate):
                return candidate

        raise RuntimeError(f"No free AirSim port found near {preferred_port}")

    def _resolve_scene_script(self) -> Path:
        scene_id = str(self.scene_id)
        env_name = scene_id if scene_id.lower().startswith("env_") else f"env_{scene_id}"

        patterns = [
            Path("envs") / "airsim" / env_name / env_name / "LinuxNoEditor" / "start.sh",
            Path("envs") / "airsim" / env_name / env_name / "LinuxNoEditor" / "AirVLN.sh",
            Path("envs") / "airsim" / env_name / "LinuxNoEditor" / "start.sh",
            Path("envs") / "airsim" / env_name / "LinuxNoEditor" / "AirVLN.sh",
            Path("envs") / "**" / env_name / "LinuxNoEditor" / "start.sh",
            Path("envs") / "**" / env_name / "LinuxNoEditor" / "AirVLN.sh",
        ]

        candidates: list[Path] = []
        for pattern in patterns:
            for item in glob.glob(str(pattern), recursive=True):
                path_obj = Path(item).resolve()
                if path_obj not in candidates:
                    candidates.append(path_obj)

        if candidates:
            candidates.sort(key=lambda x: (0 if "/envs/airsim/" in x.as_posix() else 1, len(x.as_posix()), x.as_posix()))
            return candidates[0]

        raise FileNotFoundError(f"AirSim scene script not found for {scene_id}")

    def _launch_simulator(self) -> None:
        start_script = self._resolve_scene_script()

        binary_config = start_script.parent / "AirVLN" / "Config" / "BinaryConfig.ini"
        binary_config.parent.mkdir(parents=True, exist_ok=True)

        reset_binary_config = bool(self.config.get("reset_binary_config", True))
        if reset_binary_config and binary_config.exists():
            try:
                binary_config.unlink()
            except Exception:
                pass

        write_binary_config = bool(self.config.get("write_binary_config", False))
        if write_binary_config:
            binary_config_content = str(
                self.config.get(
                    "binary_config_content",
                    "[/Script/EngineSettings.GeneralProjectSettings]\nProjectName=AirVLN\n",
                )
            )
            binary_config.write_text(binary_config_content, encoding="utf-8")

        script_dir = start_script.parent
        headless = bool(self.config.get("headless", False))
        extra_args = list(self.config.get("launch_extra_args", []))

        self._settings_path = self._write_runtime_settings(headless=headless)

        if not headless and not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "Headed mode requires GUI DISPLAY. Please run in desktop session or set `headless: true`."
            )

        launch_args = []
        if headless:
            launch_args.extend(["-RenderOffscreen", "-NoSound", "-NoSplash", "-NoVSync"])
        else:
            launch_args.extend(["-windowed", "-NoSound", "-NoVSync"])
        graphics_adapter = self.config.get("graphics_adapter")
        if graphics_adapter is not None:
            launch_args.append(f"-GraphicsAdapter={int(graphics_adapter)}")
        launch_args.append(f"-settings={self._settings_path}")
        launch_args.extend(extra_args)

        self._sim_process = subprocess.Popen(
            ["bash", str(start_script), *launch_args],
            cwd=str(script_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def _wait_for_launch_ready(self) -> None:
        if self._sim_process is None:
            return

        timeout_sec = float(self.config.get("launch_ready_timeout_sec", 45.0))
        check_interval_sec = float(self.config.get("launch_ready_check_interval_sec", 0.5))
        end_time = time.time() + timeout_sec

        while time.time() < end_time:
            if self._sim_process.poll() is not None:
                raise RuntimeError("AirSim process exited before ready")

            if self._is_port_open(self.sim_ip, self.sim_port):
                return

            time.sleep(check_interval_sec)

        raise RuntimeError(f"AirSim launch timeout; port not ready: {self.sim_ip}:{self.sim_port}")

    def _build_lidar_sensor_settings(self) -> dict[str, dict[str, Any]]:
        custom = self.config.get("lidar_sensor_configs", None)
        if isinstance(custom, dict) and len(custom) > 0:
            out: dict[str, dict[str, Any]] = {}
            for sensor_name, sensor_cfg in custom.items():
                if isinstance(sensor_cfg, dict):
                    out[str(sensor_name)] = copy.deepcopy(sensor_cfg)
            if out:
                return out

        lidar_range = float(self.config.get("lidar_range", 500.0))
        rotations_per_second = float(self.config.get("lidar_rotation_frequency", 10.0))
        points_per_second = int(self.config.get("lidar_points_per_second", 50000))
        channels_sensor1 = int(self.config.get("lidar_channels_sensor1", self.config.get("lidar_channels_horizontal", 128)))
        channels_sensor2 = int(self.config.get("lidar_channels_sensor2", self.config.get("lidar_channels_vertical", 128)))
        lidar_seg_enabled = bool(
            self.config.get(
                "lidar_segmentation_enabled",
                self.config.get("lidar_enable_segmentation", False),
            )
        )

        sensor1 = {
            "SensorType": 6,
            "Enabled": True,
            "NumberOfChannels": channels_sensor1,
            "Range": lidar_range,
            "PointsPerSecond": points_per_second,
            "RotationsPerSecond": rotations_per_second,
            "VerticalFOVUpper": float(self.config.get("lidar_vertical_fov_upper_sensor1", self.config.get("lidar_vertical_fov_upper_horizontal", 45.0))),
            "VerticalFOVLower": float(self.config.get("lidar_vertical_fov_lower_sensor1", self.config.get("lidar_vertical_fov_lower_horizontal", -45.0))),
            "HorizontalFOVStart": float(self.config.get("lidar_horizontal_fov_start", -180.0)),
            "HorizontalFOVEnd": float(self.config.get("lidar_horizontal_fov_end", 180.0)),
            "X": float(self.config.get("lidar_sensor1_x", 0.0)),
            "Y": float(self.config.get("lidar_sensor1_y", 0.0)),
            "Z": float(self.config.get("lidar_sensor1_z", -1.0)),
            "Roll": float(self.config.get("lidar_sensor1_roll", 0.0)),
            "Pitch": float(self.config.get("lidar_sensor1_pitch", 90.0)),
            "Yaw": float(self.config.get("lidar_sensor1_yaw", 0.0)),
            "DrawDebugPoints": bool(self.config.get("lidar_draw_debug_points", True)),
            "DataFrame": str(self.config.get("lidar_data_frame", "SensorLocalFrame")),
            "SegmentationEnabled": lidar_seg_enabled,
        }

        sensor2 = {
            "SensorType": 6,
            "Enabled": True,
            "NumberOfChannels": channels_sensor2,
            "Range": lidar_range,
            "PointsPerSecond": int(self.config.get("lidar_points_per_second_sensor2", points_per_second)),
            "RotationsPerSecond": rotations_per_second,
            "VerticalFOVUpper": float(self.config.get("lidar_vertical_fov_upper_sensor2", self.config.get("lidar_vertical_fov_upper_vertical", 45.0))),
            "VerticalFOVLower": float(self.config.get("lidar_vertical_fov_lower_sensor2", self.config.get("lidar_vertical_fov_lower_vertical", -45.0))),
            "HorizontalFOVStart": float(self.config.get("lidar_horizontal_fov_start", -180.0)),
            "HorizontalFOVEnd": float(self.config.get("lidar_horizontal_fov_end", 180.0)),
            "X": float(self.config.get("lidar_sensor2_x", 0.0)),
            "Y": float(self.config.get("lidar_sensor2_y", 0.0)),
            "Z": float(self.config.get("lidar_sensor2_z", -1.0)),
            "Roll": float(self.config.get("lidar_sensor2_roll", 0.0)),
            "Pitch": float(self.config.get("lidar_sensor2_pitch", 0.0)),
            "Yaw": float(self.config.get("lidar_sensor2_yaw", 0.0)),
            "DrawDebugPoints": bool(self.config.get("lidar_draw_debug_points", True)),
            "DataFrame": str(self.config.get("lidar_data_frame", "SensorLocalFrame")),
            "SegmentationEnabled": lidar_seg_enabled,
        }

        return {"LidarSensor1": sensor1, "LidarSensor2": sensor2}

    def _build_runtime_settings(self, headless: bool) -> dict:
        settings = copy.deepcopy(AIRSIM_SETTINGS_TEMPLATE)

        image_width = int(self.config.get("image_width", 3840))
        image_height = int(self.config.get("image_height", 2160))
        fov = float(self.config.get("fov", 90.0))
        uav_mode = bool(self.config.get("uav_mode", True))
        capture_image_types = list(self.config.get("camera_capture_image_types", [0, 2, 3]) or [0, 2, 3])
        capture_settings: list[dict[str, Any]] = []
        for image_type_raw in capture_image_types:
            image_type = int(image_type_raw)
            setting = copy.deepcopy(
                AIRSIM_CAMERA_CAPTURE_SETTING_TEMPLATES.get(
                    image_type,
                    AIRSIM_CAMERA_CAPTURE_SETTING_TEMPLATES[0],
                )
            )
            setting["ImageType"] = image_type
            setting["Width"] = image_width
            setting["Height"] = image_height
            setting["FOV_Degrees"] = fov
            capture_settings.append(setting)
        if not capture_settings:
            capture_settings = [copy.deepcopy(AIRSIM_CAMERA_CAPTURE_SETTING_TEMPLATES[0])]
            capture_settings[0]["Width"] = image_width
            capture_settings[0]["Height"] = image_height
            capture_settings[0]["FOV_Degrees"] = fov

        settings["ApiServerPort"] = int(self.sim_port)
        settings["LocalHostIp"] = str(self.sim_ip)
        settings["ViewMode"] = "NoDisplay" if headless else "Fpv"
        settings["SimMode"] = "Multirotor" if uav_mode else "ComputerVision"
        if uav_mode:
            settings["PhysicsEngineName"] = "ExternalPhysicsEngine"

        settings["CameraDefaults"]["CaptureSettings"] = copy.deepcopy(capture_settings)

        vehicle_type = "SimpleFlight" if uav_mode else "ComputerVision"
        lidar_sensors = self._build_lidar_sensor_settings() if uav_mode else {}
        configured_names = self.config.get("vehicle_names", None)
        if isinstance(configured_names, (list, tuple)) and len(configured_names) > 0:
            vehicle_names = [str(name) for name in configured_names if str(name).strip()]
        else:
            vehicle_names = [str(self.vehicle_name or "Drone_1")]

        for drone_name in vehicle_names:
            settings["Vehicles"][drone_name] = {
                "VehicleType": vehicle_type,
                "DefaultVehicleState": "Armed",
                "EnableCollisionPassthrogh": False,
                "EnableCollisions": True,
                "AllowAPIAlways": True,
                "RC": {
                    "RemoteControlID": 0,
                    "AllowAPIWhenDisconnected": True,
                },
                "Cameras": {
                    self.camera_name: {
                        "CaptureSettings": copy.deepcopy(capture_settings),
                        "X": 0.5,
                        "Y": 0.0,
                        "Z": 0.0,
                        "Pitch": 0.0,
                        "Roll": 0.0,
                        "Yaw": 0.0,
                    }
                },
                "Sensors": copy.deepcopy(lidar_sensors),
                "X": 0,
                "Y": 0,
                "Z": 0,
                "Pitch": 0,
                "Roll": 0,
                "Yaw": 0,
            }

        return settings

    def _write_runtime_settings(self, headless: bool) -> Path:
        settings = self._build_runtime_settings(headless=headless)
        settings_text = json.dumps(settings, indent=2)

        settings_dir = Path(".airsim_runtime") / "settings"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_path = settings_dir / f"{self.scene_id}_{self.sim_port}.json"
        settings_path.write_text(settings_text, encoding="utf-8")

        default_paths = [
            Path.home() / "Documents" / "AirSim" / "settings.json",
            Path.home() / "AirSim" / "settings.json",
            Path.home() / ".config" / "AirSim" / "settings.json",
        ]
        for default_path in default_paths:
            try:
                default_path.parent.mkdir(parents=True, exist_ok=True)
                default_path.write_text(settings_text, encoding="utf-8")
            except Exception:
                pass

        return settings_path

    def _connect(self) -> None:
        try:
            import airsim
        except ImportError as exc:
            raise ImportError("`airsim` package is required for AirSimBridge") from exc

        timeout_sec = float(self.config.get("connect_timeout_sec", 30.0))
        retry_interval_sec = float(self.config.get("connect_retry_interval_sec", 1.0))
        end_time = time.time() + timeout_sec
        last_error: Optional[Exception] = None

        while time.time() < end_time:
            try:
                client = airsim.MultirotorClient(ip=self.sim_ip, port=self.sim_port)
                client.confirmConnection()

                strict_vehicle_name = bool(self.config.get("strict_vehicle_name", True))
                if self.vehicle_name:
                    try:
                        client.enableApiControl(True, self.vehicle_name)
                        client.armDisarm(True, self.vehicle_name)
                    except Exception as exc:
                        if strict_vehicle_name:
                            raise RuntimeError(
                                f"AirSim vehicle not available: {self.vehicle_name}"
                            ) from exc
                        client.enableApiControl(True)
                        client.armDisarm(True)
                        self.vehicle_name = ""
                else:
                    client.enableApiControl(True)
                    client.armDisarm(True)

                self._client = client
                return
            except Exception as exc:
                last_error = exc
                time.sleep(retry_interval_sec)

        raise RuntimeError(f"Failed to connect to AirSim within {timeout_sec}s") from last_error

    def _client_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        client = self._require_client()
        method = getattr(client, method_name)

        if self.vehicle_name:
            kwargs.setdefault("vehicle_name", self.vehicle_name)

        return method(*args, **kwargs)

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("AirSim client is not connected. Set `connect_on_init: true`.")
        return self._client

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
        import airsim

        vehicle_name = str(vehicle_or_actor or self.vehicle_name or "").strip()
        strict_vehicle_name = bool(self.config.get("strict_vehicle_name", True))
        verify_pose_after_set = bool(self.config.get("verify_pose_after_set", True))
        verify_tol_m = max(1e-3, float(self.config.get("verify_pose_tolerance_m", 0.20)))
        verify_retries = max(1, int(self.config.get("verify_pose_retries", 3)))
        verify_wait_sec = max(0.0, float(self.config.get("verify_pose_wait_sec", 0.01)))

        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        x_ned, y_ned, z_ned = enu_to_airsim_ned_position(x=x, y=y, z=z)
        pose = airsim.Pose(
            airsim.Vector3r(x_ned, y_ned, z_ned),
            airsim.to_quaternion(pitch_rad, roll_rad, yaw_rad),
        )

        client = self._require_client()

        def _apply_pose(target_vehicle: str) -> None:
            if target_vehicle:
                client.simSetVehiclePose(pose, True, vehicle_name=target_vehicle)
            else:
                client.simSetVehiclePose(pose, True)

        def _fetch_pose(target_vehicle: str):
            if target_vehicle:
                return client.simGetVehiclePose(vehicle_name=target_vehicle)
            return client.simGetVehiclePose()

        active_vehicle = vehicle_name
        applied = False
        last_error: Optional[Exception] = None

        for _ in range(verify_retries):
            try:
                _apply_pose(active_vehicle)
                applied = True
            except Exception as exc:
                last_error = exc
                if active_vehicle and strict_vehicle_name:
                    raise RuntimeError(f"simSetVehiclePose failed for vehicle={active_vehicle}") from exc
                if active_vehicle and not strict_vehicle_name:
                    active_vehicle = ""
                    try:
                        _apply_pose(active_vehicle)
                        applied = True
                    except Exception as exc2:
                        last_error = exc2
                        continue
                else:
                    continue

            if not verify_pose_after_set:
                break
            try:
                got = _fetch_pose(active_vehicle)
                dx = float(got.position.x_val) - float(x_ned)
                dy = float(got.position.y_val) - float(y_ned)
                dz = float(got.position.z_val) - float(z_ned)
                err = math.sqrt(dx * dx + dy * dy + dz * dz)
                if err <= verify_tol_m:
                    break
            except Exception:
                pass

            if verify_wait_sec > 0.0:
                time.sleep(verify_wait_sec)

        if not applied:
            if last_error is not None:
                raise RuntimeError("simSetVehiclePose failed") from last_error
            raise RuntimeError("simSetVehiclePose failed: unknown error")

        if active_vehicle:
            self.vehicle_name = active_vehicle

        self._update_last_pose_enu(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)

    def set_camera_fov(
        self,
        fov_deg: float,
        camera_name: Optional[str] = None,
        vehicle_or_actor: Optional[str] = None,
    ) -> None:
        client = self._require_client()
        camera = str(camera_name or self._active_camera_name or self.camera_name)
        vehicle_name = vehicle_or_actor or self.vehicle_name
        try:
            if vehicle_name:
                client.simSetCameraFov(camera, float(fov_deg), vehicle_name=vehicle_name)
            else:
                client.simSetCameraFov(camera, float(fov_deg))
        except TypeError:
            client.simSetCameraFov(camera, float(fov_deg))

    def _capture_image(self, image_type: int, as_float: bool = False) -> Any:
        import airsim

        retries = int(self.config.get("capture_retries", 2))
        fallback_camera_names = self.config.get(
            "fallback_camera_names",
            ["front_center", "front_custom", "front_0", "0", "1"],
        )

        camera_candidates: list[str] = []
        if self._active_camera_name:
            camera_candidates.append(str(self._active_camera_name))
        camera_candidates.append(str(self.camera_name))
        for candidate in fallback_camera_names:
            name = str(candidate)
            if name not in camera_candidates:
                camera_candidates.append(name)

        strict_vehicle_name = bool(self.config.get("strict_vehicle_name", True))
        vehicle_candidates: list[Optional[str]] = []
        if self.vehicle_name:
            vehicle_candidates.append(str(self.vehicle_name))
            if not strict_vehicle_name:
                vehicle_candidates.append(None)
        else:
            vehicle_candidates.append(None)

        request_modes = [(as_float, False)]

        last_error: Optional[Exception] = None
        for _ in range(max(1, retries)):
            for camera_name in camera_candidates:
                for vehicle_name in vehicle_candidates:
                    for request_as_float, request_compress in request_modes:
                        try:
                            request = airsim.ImageRequest(camera_name, image_type, request_as_float, request_compress)
                            responses = self._client_call(
                                "simGetImages",
                                [request],
                                **({"vehicle_name": vehicle_name} if vehicle_name else {}),
                            )
                            response = responses[0]

                            if response.height == 0 or response.width == 0:
                                continue

                            self._active_camera_name = camera_name

                            if response.pixels_as_float:
                                return np.array(response.image_data_float, dtype=np.float32).reshape(
                                    response.height,
                                    response.width,
                                )

                            image_bytes = bytes(response.image_data_uint8)
                            arr = np.frombuffer(image_bytes, dtype=np.uint8)
                            expected = response.height * response.width * 3
                            if arr.size < expected:
                                continue
                            return arr[:expected].reshape(response.height, response.width, 3)
                        except Exception as exc:
                            last_error = exc

        if last_error is not None:
            raise RuntimeError(
                f"Failed to capture image type={image_type} after trying camera/vehicle fallbacks"
            ) from last_error
        raise RuntimeError(f"Failed to capture image type={image_type}: empty response")

    def capture_rgb(self) -> Any:
        import airsim

        return self._capture_image(airsim.ImageType.Scene, as_float=False)

    def capture_depth(self) -> Any:
        import airsim

        return self._capture_image(airsim.ImageType.DepthPlanar, as_float=True)

    def capture_seg(self) -> Any:
        import airsim

        return self._capture_image(airsim.ImageType.Segmentation, as_float=False)

    def get_lidar(self) -> Any:
        lidar_names_cfg = self.config.get("lidar_names", None)
        if isinstance(lidar_names_cfg, (list, tuple)) and len(lidar_names_cfg) > 0:
            lidar_names = [str(name) for name in lidar_names_cfg if str(name).strip()]
        else:
            lidar_names = [str(self.config.get("lidar_name", "LidarSensor1"))]

        update_frequency_hz = max(1.0, float(self.config.get("lidar_update_frequency_hz", 400.0)))
        capture_window_sec = max(0.0, float(self.config.get("lidar_capture_window_sec", 0.005)))
        configured_samples = self.config.get("lidar_samples_per_pose", None)
        if configured_samples is None:
            sample_count = max(1, int(round(capture_window_sec * update_frequency_hz)))
        else:
            sample_count = max(1, int(configured_samples))

        sample_interval_sec = self.config.get("lidar_sample_interval_sec", None)
        if sample_interval_sec is None:
            sample_interval_sec = 1.0 / update_frequency_hz
        sample_interval_sec = max(0.0, float(sample_interval_sec))
        min_points = max(1, int(self.config.get("lidar_min_points_per_pose", 1)))
        segmentation_enabled = bool(
            self.config.get(
                "lidar_segmentation_enabled",
                self.config.get("lidar_enable_segmentation", False),
            )
        )

        vehicle_candidates: list[str] = []
        seen_vehicle_names: set[str] = set()

        def _add_vehicle_candidate(name: Any) -> None:
            if name is None:
                return
            candidate = str(name).strip()
            if not candidate:
                return
            if candidate in seen_vehicle_names:
                return
            seen_vehicle_names.add(candidate)
            vehicle_candidates.append(candidate)

        _add_vehicle_candidate(self.vehicle_name)
        _add_vehicle_candidate(self.config.get("vehicle_name", None))
        cfg_vehicle_names = self.config.get("vehicle_names", None)
        if isinstance(cfg_vehicle_names, (list, tuple)):
            for item in cfg_vehicle_names:
                _add_vehicle_candidate(item)

        accumulated: list[np.ndarray] = []
        accumulated_seg: list[np.ndarray] = []
        client = self._require_client()

        def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
            ww, xx, yy, zz = w * w, x * x, y * y, z * z
            wx, wy, wz = w * x, w * y, w * z
            xy, xz, yz = x * y, x * z, y * z
            return np.array(
                [
                    [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
                    [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
                    [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
                ],
                dtype=np.float32,
            )

        def _extract_segmentation_ids(lidar_data_obj: Any, expected_count: int) -> Optional[np.ndarray]:
            for attr_name in ("segmentation", "segmentation_data", "segmentation_ids", "segmentation_id"):
                if not hasattr(lidar_data_obj, attr_name):
                    continue
                raw_seg = getattr(lidar_data_obj, attr_name, None)
                if raw_seg is None:
                    continue
                try:
                    seg = np.asarray(raw_seg).reshape(-1)
                except Exception:
                    continue
                if seg.size == 0:
                    continue
                seg = seg.astype(np.uint32, copy=False)
                if seg.size < expected_count:
                    pad = np.zeros((expected_count,), dtype=np.uint32)
                    pad[: seg.size] = seg
                    seg = pad
                elif seg.size > expected_count:
                    seg = seg[:expected_count]
                return seg
            return None

        for sample_idx in range(sample_count):
            for lidar_name in lidar_names:
                lidar_data = None
                used_vehicle_name = ""

                for vehicle_name in vehicle_candidates:
                    try:
                        lidar_data = client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)
                    except Exception:
                        continue
                    if len(lidar_data.point_cloud) >= 3:
                        used_vehicle_name = vehicle_name
                        break

                if (lidar_data is None or len(lidar_data.point_cloud) < 3):
                    try:
                        lidar_data = client.getLidarData(lidar_name=lidar_name)
                        used_vehicle_name = ""
                    except Exception:
                        lidar_data = None

                if lidar_data is None or len(lidar_data.point_cloud) < 3:
                    continue

                if used_vehicle_name and self.vehicle_name != used_vehicle_name:
                    self.vehicle_name = used_vehicle_name

                points_local = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
                if points_local.size == 0:
                    continue

                seg_ids = None
                if segmentation_enabled:
                    seg_ids = _extract_segmentation_ids(lidar_data, expected_count=int(points_local.shape[0]))

                finite_mask = np.isfinite(points_local).all(axis=1)
                if not np.any(finite_mask):
                    continue
                points_local = points_local[finite_mask]
                if seg_ids is not None and seg_ids.shape[0] == finite_mask.shape[0]:
                    seg_ids = seg_ids[finite_mask]

                min_range_m = float(self.config.get("lidar_min_range_m", 1.0))
                if min_range_m > 0.0:
                    ranges = np.linalg.norm(points_local, axis=1)
                    range_mask = ranges >= min_range_m
                    points_local = points_local[range_mask]
                    if seg_ids is not None and seg_ids.shape[0] == range_mask.shape[0]:
                        seg_ids = seg_ids[range_mask]

                if points_local.size == 0:
                    continue

                try:
                    position = lidar_data.pose.position
                    orientation = lidar_data.pose.orientation
                    rot = _quat_to_rot(
                        float(orientation.w_val),
                        float(orientation.x_val),
                        float(orientation.y_val),
                        float(orientation.z_val),
                    )
                    trans = np.array(
                        [float(position.x_val), float(position.y_val), float(position.z_val)],
                        dtype=np.float32,
                    )
                    points_world = points_local @ rot.T + trans
                except Exception:
                    points_world = points_local

                accumulated.append(points_world.astype(np.float32))
                if segmentation_enabled:
                    if seg_ids is None or seg_ids.shape[0] != points_world.shape[0]:
                        seg_ids = np.zeros((points_world.shape[0],), dtype=np.uint32)
                    accumulated_seg.append(seg_ids.astype(np.uint32, copy=False))

            if sample_idx + 1 < sample_count and sample_interval_sec > 0.0:
                time.sleep(sample_interval_sec)

        if not accumulated:
            if segmentation_enabled:
                return {
                    "points": np.empty((0, 3), dtype=np.float32),
                    "segmentation": np.empty((0,), dtype=np.uint32),
                }
            return np.empty((0, 3), dtype=np.float32)

        merged = np.concatenate(accumulated, axis=0)
        if merged.shape[0] < min_points:
            if segmentation_enabled:
                return {
                    "points": np.empty((0, 3), dtype=np.float32),
                    "segmentation": np.empty((0,), dtype=np.uint32),
                }
            return np.empty((0, 3), dtype=np.float32)

        if segmentation_enabled:
            merged_seg = np.concatenate(accumulated_seg, axis=0) if accumulated_seg else np.zeros((merged.shape[0],), dtype=np.uint32)
            if merged_seg.shape[0] != merged.shape[0]:
                aligned = np.zeros((merged.shape[0],), dtype=np.uint32)
                copy_count = min(aligned.shape[0], merged_seg.shape[0])
                if copy_count > 0:
                    aligned[:copy_count] = merged_seg[:copy_count]
                merged_seg = aligned
            return {"points": merged, "segmentation": merged_seg}

        return merged

    def raycast(self, origin: tuple[float, float, float], target: tuple[float, float, float]) -> Optional[bool]:
        return None

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                if self.vehicle_name:
                    self._client.armDisarm(False, self.vehicle_name)
                    self._client.enableApiControl(False, self.vehicle_name)
                else:
                    self._client.armDisarm(False)
                    self._client.enableApiControl(False)
            except Exception:
                pass
            finally:
                self._client = None

        if self._sim_process is not None:
            if self._sim_process.poll() is None:
                term_timeout = float(self.config.get("shutdown_term_timeout_sec", 8.0))
                kill_timeout = float(self.config.get("shutdown_kill_timeout_sec", 2.0))
                try:
                    os.killpg(os.getpgid(self._sim_process.pid), signal.SIGTERM)
                except Exception:
                    try:
                        self._sim_process.terminate()
                    except Exception:
                        pass

                try:
                    self._sim_process.wait(timeout=term_timeout)
                except Exception:
                    try:
                        os.killpg(os.getpgid(self._sim_process.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            self._sim_process.kill()
                        except Exception:
                            pass
                    try:
                        self._sim_process.wait(timeout=kill_timeout)
                    except Exception:
                        pass

            self._sim_process = None
