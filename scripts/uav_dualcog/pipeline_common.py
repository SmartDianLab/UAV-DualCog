from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sim_bridge.factory import create_bridge


BRIDGE_OPTIONAL_KEYS: tuple[str, ...] = (
    "camera_id",
    "render_url",
    "attach_sensors",
    "actor_role_name",
    "town",
    "lidar_name",
    "lidar_names",
    "lidar_range",
    "lidar_min_range_m",
    "lidar_samples_per_pose",
    "lidar_sample_interval_sec",
    "lidar_min_points_per_pose",
    "lidar_segmentation_enabled",
    "lidar_enable_segmentation",
    "headless",
    "auto_select_port_on_conflict",
    "launch_ready_timeout_sec",
    "launch_ready_check_interval_sec",
    "connect_retry_interval_sec",
    "launch_extra_args",
    "vehicle_names",
)


def list_running_airsim_processes() -> list[str]:
    patterns = [
        "AirVLN-Linux-Shipping",
        "LinuxNoEditor/AirVLN.sh",
    ]
    try:
        proc = subprocess.run(["ps", "-ef"], capture_output=True, text=True, check=False)
    except Exception:
        return []
    rows = []
    for line in (proc.stdout or "").splitlines():
        text = str(line or "")
        if any(pattern in text for pattern in patterns):
            rows.append(text.strip())
    return rows


def _list_running_airsim_process_infos() -> list[dict[str, Any]]:
    patterns = [
        "AirVLN-Linux-Shipping",
        "LinuxNoEditor/AirVLN.sh",
    ]
    try:
        proc = subprocess.run(["ps", "-ef"], capture_output=True, text=True, check=False)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in (proc.stdout or "").splitlines():
        text = str(line or "").strip()
        if not text or not any(pattern in text for pattern in patterns):
            continue
        parts = text.split(None, 7)
        if len(parts) < 8:
            continue
        try:
            pid = int(parts[1])
        except Exception:
            continue
        rows.append({"pid": int(pid), "cmd": text})
    return rows


def ensure_single_airsim_process(stage: str) -> None:
    rows = list_running_airsim_processes()
    if len(rows) <= 0:
        return
    raise RuntimeError(
        f"{stage}_airsim_process_conflict_detected: count={len(rows)} | processes={rows[:6]}"
    )


def cleanup_airsim_processes(stage: str, *, timeout_sec: float = 8.0) -> dict[str, Any]:
    infos = _list_running_airsim_process_infos()
    if not infos:
        return {
            "stage": str(stage),
            "found_count": 0,
            "terminated_count": 0,
            "killed_count": 0,
            "remaining_count": 0,
            "processes": [],
        }
    terminated = 0
    killed = 0
    pids = [int(row["pid"]) for row in infos]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated += 1
        except ProcessLookupError:
            pass
        except Exception:
            pass
    deadline = time.time() + max(0.5, float(timeout_sec))
    while time.time() < deadline:
        remaining_infos = _list_running_airsim_process_infos()
        remaining_pids = {int(row["pid"]) for row in remaining_infos}
        if not any(pid in remaining_pids for pid in pids):
            break
        time.sleep(0.2)
    remaining_infos = _list_running_airsim_process_infos()
    remaining_pids = {int(row["pid"]) for row in remaining_infos}
    for pid in pids:
        if pid not in remaining_pids:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except ProcessLookupError:
            pass
        except Exception:
            pass
    time.sleep(0.2)
    final_infos = _list_running_airsim_process_infos()
    return {
        "stage": str(stage),
        "found_count": len(infos),
        "terminated_count": int(terminated),
        "killed_count": int(killed),
        "remaining_count": len(final_infos),
        "processes": [str(row.get("cmd", "") or "") for row in infos[:6]],
    }


def compute_airsim_lidar_range_profile(
    config: dict[str, Any],
    *,
    default_range: float = 500.0,
) -> dict[str, Any]:
    engine_cfg = (config.get("engine_params", {}) or {}).get("airsim", {}) or {}
    traj_map = config.get("traj_map", {}) or {}

    mode = str(engine_cfg.get("lidar_range_mode", "auto") or "auto").strip().lower()
    if mode not in {"auto", "fixed"}:
        mode = "auto"

    configured_range_raw = engine_cfg.get("lidar_range", None)
    configured_range = float(configured_range_raw) if configured_range_raw is not None else None
    auto_scale = max(0.1, float(engine_cfg.get("lidar_range_auto_scale", 1.5)))
    auto_min = float(engine_cfg.get("lidar_range_auto_min", 60.0))
    auto_max = float(engine_cfg.get("lidar_range_auto_max", 140.0))
    if auto_min > auto_max:
        auto_min, auto_max = auto_max, auto_min

    dx = None
    dy = None
    lidar_delta = traj_map.get("LidarDelta", None)
    if isinstance(lidar_delta, (list, tuple)) and len(lidar_delta) >= 2:
        try:
            dx = abs(float(lidar_delta[0]))
            dy = abs(float(lidar_delta[1]))
        except (TypeError, ValueError):
            dx = None
            dy = None

    diag = None
    if dx is not None and dy is not None and dx > 0.0 and dy > 0.0:
        diag = math.hypot(dx, dy)

    effective_mode = mode
    source = "engine_params.airsim.lidar_range"
    if mode == "fixed":
        range_m = configured_range if configured_range is not None else float(default_range)
        if configured_range is None:
            source = "default"
            effective_mode = "fallback_default"
    elif diag is not None and diag > 0.0:
        range_m = max(auto_min, min(auto_max, diag * auto_scale))
        source = "traj_map.LidarDelta"
        effective_mode = "auto"
    elif configured_range is not None:
        range_m = configured_range
        effective_mode = "fallback_config"
    else:
        range_m = float(default_range)
        source = "default"
        effective_mode = "fallback_default"

    return {
        "range_m": float(range_m),
        "mode": str(effective_mode),
        "source": str(source),
        "configured_range_m": float(configured_range) if configured_range is not None else None,
        "grid_step_xy": {
            "dx": float(dx) if dx is not None else None,
            "dy": float(dy) if dy is not None else None,
            "diag": float(diag) if diag is not None else None,
        },
        "auto_rule": {
            "formula": "clip(hypot(dx,dy) * scale, auto_min, auto_max)",
            "scale": float(auto_scale),
            "min_m": float(auto_min),
            "max_m": float(auto_max),
        },
    }


def build_unified_bridge_config(
    config: dict[str, Any],
    *,
    engine: str,
    vehicle_name: str,
    sim_port: int | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    fov: float | None = None,
    vehicle_names: list[str] | None = None,
    default_width: int = 3840,
    default_height: int = 2160,
    default_fov: float = 90.0,
) -> dict[str, Any]:
    engine_norm = str(engine or "airsim").lower()
    engine_cfg = (config.get("engine_params", {}) or {}).get(engine_norm, {}) or {}
    camera_cfg = config.get("camera", {}) or {}

    w = int(image_width) if image_width is not None else int(camera_cfg.get("width", default_width))
    h = int(image_height) if image_height is not None else int(camera_cfg.get("height", default_height))
    fov_val = float(fov) if fov is not None else float(camera_cfg.get("fov", default_fov))
    port_val = int(sim_port) if sim_port is not None else int(engine_cfg.get("sim_port", 41471))

    bridge_cfg: dict[str, Any] = {
        "sim_ip": str(engine_cfg.get("sim_ip", "127.0.0.1")),
        "sim_port": int(engine_cfg.get("sim_port", port_val)),
        "connect_on_init": bool(engine_cfg.get("connect_on_init", True)),
        "launch_sim": bool(engine_cfg.get("launch_sim", False)),
        "vehicle_name": str(engine_cfg.get("vehicle_name", vehicle_name)),
        "camera_name": str(engine_cfg.get("camera_name", "front_0")),
        "connect_timeout_sec": float(engine_cfg.get("connect_timeout_sec", 30.0)),
        "capture_retries": int(engine_cfg.get("capture_retries", 2)),
        "tick_after_set_pose": bool(engine_cfg.get("tick_after_set_pose", True)),
        "strict_vehicle_name": bool(engine_cfg.get("strict_vehicle_name", False)),
        "image_width": int(w),
        "image_height": int(h),
        "fov": float(fov_val),
    }

    for key in BRIDGE_OPTIONAL_KEYS:
        if key in engine_cfg:
            bridge_cfg[key] = engine_cfg[key]

    if vehicle_names:
        bridge_cfg["vehicle_names"] = [str(v) for v in vehicle_names]

    return bridge_cfg


def prepare_airsim_runtime_unified(
    config: dict[str, Any],
    *,
    scene_id: str,
    base_bridge_cfg: dict[str, Any],
    vehicle_name: str,
    vehicle_names: list[str] | None = None,
) -> tuple[int, Any, bool, int]:
    airsim_cfg = (config.get("engine_params", {}) or {}).get("airsim", {}) or {}

    bootstrap_cfg = dict(base_bridge_cfg)
    bootstrap_cfg["vehicle_name"] = str(vehicle_name)
    if vehicle_names:
        bootstrap_cfg["vehicle_names"] = [str(v) for v in vehicle_names]

    configured_port = int(bootstrap_cfg.get("sim_port", 41471))
    bootstrap_cfg["launch_sim"] = bool(airsim_cfg.get("launch_sim", bootstrap_cfg.get("launch_sim", False)))
    bootstrap_cfg["connect_on_init"] = True
    bootstrap_cfg["auto_select_port_on_conflict"] = bool(bootstrap_cfg.get("launch_sim", False))

    bootstrap_bridge = create_bridge(engine="airsim", scene_id=scene_id, config=bootstrap_cfg)
    runtime_port = int(getattr(bootstrap_bridge, "sim_port", configured_port))
    launched_by_bridge = bool(bootstrap_cfg.get("launch_sim", False))
    return runtime_port, bootstrap_bridge, launched_by_bridge, configured_port


def validate_complete_indices(indices: list[int], total_count: int, *, name: str) -> None:
    total = int(total_count)
    if total <= 0:
        return
    seen: set[int] = set()
    for idx in indices:
        v = int(idx)
        if v < 0 or v >= total:
            raise RuntimeError(f"{name}_out_of_range: idx={v}, total={total}")
        if v in seen:
            raise RuntimeError(f"{name}_duplicate: idx={v}")
        seen.add(v)
    if len(seen) != total:
        missing = [i for i in range(total) if i not in seen]
        raise RuntimeError(f"{name}_missing: expected={total}, got={len(seen)}, missing={missing[:12]}")


def format_unified_startup_ports_message(
    *,
    stage: str,
    engine: str,
    configured_sim_port: int,
    runtime_sim_port: int,
    launched_by_bridge: bool,
) -> str:
    stage_name = str(stage).strip().lower()
    engine_name = str(engine).strip().lower()
    return (
        f"[{stage_name}][{engine_name}] startup ports: "
        f"configured_sim_port={int(configured_sim_port)}, "
        f"runtime_sim_port={int(runtime_sim_port)}, "
        f"launched_by_bridge={bool(launched_by_bridge)}"
    )


def write_unified_startup_log(output_dir: Path, message: str, *, filename: str = "startup_ports.log") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / filename
    ts = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"{ts} {str(message).strip()}\n")
    return log_path


def resolve_base_dir(config: dict[str, Any], *, workspace_root: Path | None = None) -> Path:
    task_cfg = config.get("task", {}) or {}
    base_dir_cfg = Path(str(task_cfg.get("base_dir", "scene_data")))
    if base_dir_cfg.is_absolute():
        return base_dir_cfg
    cwd_base = (Path.cwd() / base_dir_cfg).resolve()
    if cwd_base.exists():
        return cwd_base
    if workspace_root is not None:
        return (Path(workspace_root) / base_dir_cfg).resolve()
    return cwd_base


def resolve_scene_root(
    config: dict[str, Any],
    *,
    scene_id: str,
    engine: str | None = None,
    workspace_root: Path | None = None,
) -> Path:
    task_cfg = config.get("task", {}) or {}
    output_layout = config.get("output_layout", {}) or {}
    include_engine = bool(output_layout.get("scene_dir_include_engine", True))
    engine_name = str(engine or task_cfg.get("engine", "airsim")).lower().strip()
    explicit_scene_dir = str(task_cfg.get("scene_dir_name", "")).strip() or str(output_layout.get("scene_dir_name", "")).strip()
    base_dir = resolve_base_dir(config, workspace_root=workspace_root)
    if explicit_scene_dir:
        scene_dir_name = explicit_scene_dir
    elif include_engine:
        # New default layout: <engine>_<scene_id>.
        # Keep backward compatibility with historical <scene_id>_<engine> directories.
        preferred = f"{engine_name}_{str(scene_id)}"
        legacy = f"{str(scene_id)}_{engine_name}"
        preferred_root = base_dir / preferred
        legacy_root = base_dir / legacy
        if preferred_root.exists():
            return preferred_root
        if legacy_root.exists():
            return legacy_root
        scene_dir_name = preferred
    else:
        scene_dir_name = str(scene_id)
    return base_dir / scene_dir_name


def resolve_task_pipeline_base_dir(
    config: dict[str, Any] | None = None,
    *,
    workspace_root: Path | None = None,
    default: str = "task_pipeline_data",
) -> Path:
    cfg = dict(config or {})
    task_pipeline_cfg = cfg.get("task_pipeline", {}) or {}
    base_dir_cfg = Path(str(task_pipeline_cfg.get("root_dir", default) or default))
    if base_dir_cfg.is_absolute():
        return base_dir_cfg
    root = Path(workspace_root) if workspace_root is not None else Path.cwd()
    return (root / base_dir_cfg).resolve()


def resolve_task_pipeline_task_root(
    config: dict[str, Any] | None = None,
    *,
    task_name: str | None = None,
    workspace_root: Path | None = None,
) -> Path | None:
    cfg = dict(config or {})
    task_pipeline_cfg = cfg.get("task_pipeline", {}) or {}
    task_name_value = str(task_name or task_pipeline_cfg.get("task_name", "") or "").strip()
    if not task_name_value:
        return None
    return resolve_task_pipeline_base_dir(cfg, workspace_root=workspace_root) / task_name_value


def resolve_task_pipeline_scene_root(
    config: dict[str, Any] | None,
    *,
    scene_id: str,
    engine: str,
    task_name: str | None = None,
    workspace_root: Path | None = None,
) -> Path | None:
    task_root = resolve_task_pipeline_task_root(config, task_name=task_name, workspace_root=workspace_root)
    if task_root is None:
        return None
    scene_dir_name = f"{str(engine).strip().lower()}_{str(scene_id).strip()}"
    return task_root / scene_dir_name


def list_task_pipeline_tasks(
    config: dict[str, Any] | None = None,
    *,
    workspace_root: Path | None = None,
) -> list[str]:
    base_dir = resolve_task_pipeline_base_dir(config, workspace_root=workspace_root)
    if not base_dir.exists():
        return []
    out: list[str] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        if (
            (path / "task_pipeline").exists()
            or (path / "video_tasks").exists()
            or (path / "image_tasks").exists()
            or (path / "stage3_tasks").exists()
            or (path / "qa").exists()
        ):
            out.append(path.name)
    return out


def resolve_output_dir_name(config: dict[str, Any], *, key: str, default: str) -> str:
    output_layout = config.get("output_layout", {}) or {}
    if key in output_layout:
        raw = str(output_layout.get(key, default)).strip()
        return raw or default

    # Backward-compatible fallbacks used by older stage scripts/configs.
    output_dirs = config.get("output_dirs", {}) or {}
    if key in output_dirs:
        raw = str(output_dirs.get(key, default)).strip()
        return raw or default

    task_cfg = config.get("task", {}) or {}
    if key in task_cfg:
        raw = str(task_cfg.get(key, default)).strip()
        return raw or default

    raw = str(default).strip()
    return raw or default


def scene_id_variants(scene_id: str | None) -> list[str]:
    raw = str(scene_id or "").strip()
    if not raw:
        return []
    out: list[str] = []

    def _append(value: str | None) -> None:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)

    _append(raw)
    if raw.startswith("env_"):
        suffix = raw[4:].strip()
        _append(suffix)
        if suffix.isdigit():
            _append(f"env_{int(suffix)}")
    elif raw.isdigit():
        _append(str(int(raw)))
        _append(f"env_{int(raw)}")
    else:
        match = re.fullmatch(r"(?:.*\b)?env_(\d+)", raw)
        if match:
            numeric = match.group(1)
            _append(str(int(numeric)))
            _append(f"env_{int(numeric)}")
    return out


def resolve_scene_artifact_path(directory: Path, scene_id: str | None, artifact_suffix: str) -> Path:
    base_dir = Path(directory)
    variants = scene_id_variants(scene_id)
    if not variants:
        return base_dir / artifact_suffix.lstrip(".")
    for variant in variants:
        candidate = base_dir / f"{variant}{artifact_suffix}"
        if candidate.exists():
            return candidate
    return base_dir / f"{variants[0]}{artifact_suffix}"


def append_unified_scene_log(
    *,
    config: dict[str, Any],
    scene_root: Path,
    stage: str,
    step: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> Path:
    logs_dir_name = resolve_output_dir_name(config, key="logs_dir", default="logs")
    logs_dir = scene_root / logs_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": str(stage),
        "step": str(step),
        "message": str(message),
        "payload": payload or {},
    }
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log_path


def build_unified_stage_event(
    *,
    stage: str,
    step: str,
    scene_id: str,
    engine: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "stage": str(stage),
        "step": str(step),
        "scene_id": str(scene_id),
        "engine": str(engine),
        "status": str(status),
    }
    if isinstance(extra, dict) and extra:
        out.update(extra)
    return out
