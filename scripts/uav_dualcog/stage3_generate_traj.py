#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
try:
    from flask import Flask, jsonify, redirect, render_template_string, request, send_file
except Exception:
    Flask = None
    redirect = None
    jsonify = None
    render_template_string = None
    request = None
    send_file = None

try:
    import cv2
except Exception:
    cv2 = None

from image_compression_utils import compression_cfg as build_image_compression_cfg
from image_compression_utils import preferred_output_path, save_bgr_image
from progress_utils import StageLogger

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
COMMON_STAGE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "uav_dualcog" / "common_stage_configs.yaml"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from trajectory.minsnap_connector import build_poses, check_collision, check_constraints, smooth_trajectory
from trajectory.stage_composer import compose_trajectory
from trajectory.behaviors import (
    BEHAVIOR_SET,
    ELEMENT_LIBRARY,
    SET_LIBRARY,
    build_element_instance,
    sample_waypoints,
)
from sim_bridge.factory import create_bridge
from pipeline_common import (
    append_unified_scene_log,
    build_unified_stage_event,
    build_unified_bridge_config,
    ensure_single_airsim_process,
    format_unified_startup_ports_message,
    prepare_airsim_runtime_unified,
    resolve_base_dir,
    resolve_output_dir_name,
    resolve_scene_root,
    resolve_task_pipeline_scene_root,
    validate_complete_indices,
)

from stage4_qa_generate_and_eval import (
    _discover_scene_catalog,
    _draw_reference_bbox,
    _extract_response_text,
    _iso_now,
    _load_scene_config_from_catalog,
    _load_valid_instances,
    _now_ts,
    _path_for_json,
    _prepare_upload_image,
    _resolve_api_settings,
    _safe_name,
    ApiRateLimiter,
    CancelledExperimentError,
    ExperimentJobManager,
)
from stage3_task_suite import generate_manifest as generate_stage3_manifest
from stage3_task_suite import register_stage3_task_routes
from stage3_task_suite import run_experiment_once as run_stage3_experiment_once

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid yaml root type: {path}")
    return data


def _load_stage3_behavior_defaults() -> dict[str, Any]:
    def _normalize_set_entry(set_key: str, entry: dict[str, Any]) -> dict[str, Any]:
        out = dict(entry or {})
        step_rows = [dict(item) for item in list(out.get("element_steps", []) or []) if isinstance(item, dict)]
        if step_rows:
            out["behavior_sequence"] = [
                str(item.get("element_class", "") or "")
                for item in step_rows
                if str(item.get("element_class", "") or "").strip()
            ]
            param_overrides: dict[str, Any] = {}
            auto_rules: dict[str, Any] = {}
            for idx, step in enumerate(step_rows):
                params = dict(step.get("params", {}) or {})
                rules = dict(step.get("auto_rules", {}) or {})
                if params:
                    param_overrides[str(idx)] = params
                if rules:
                    auto_rules[str(idx)] = rules
            if param_overrides:
                out["element_param_overrides"] = param_overrides
            if auto_rules:
                out["element_auto_rules"] = auto_rules
        return out

    path = COMMON_STAGE_CONFIG_PATH
    if not path.exists():
        return {}
    payload = load_yaml(path)
    block = payload.get("stage3_behavior_library", {}) or {}
    raw_sets = dict(block.get("sets", {}) or {}) if isinstance(block, dict) else {}
    return {str(k): _normalize_set_entry(str(k), dict(v)) for k, v in raw_sets.items() if isinstance(v, dict)}


def _load_common_stage_cfg() -> dict[str, Any]:
    path = COMMON_STAGE_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        payload = load_yaml(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(dict(out.get(key, {}) or {}), value)
        else:
            out[key] = value
    return out


def _load_stage3_behavior_shared() -> dict[str, Any]:
    path = COMMON_STAGE_CONFIG_PATH
    if not path.exists():
        return {}
    payload = load_yaml(path)
    block = payload.get("stage3_behavior_library", {}) or {}
    return dict(block.get("shared", {}) or {}) if isinstance(block, dict) else {}


def _load_stage3_behavior_defaults_raw() -> dict[str, Any]:
    path = COMMON_STAGE_CONFIG_PATH
    if not path.exists():
        return {}
    payload = load_yaml(path)
    block = payload.get("stage3_behavior_library", {}) or {}
    return dict(block.get("sets", {}) or {}) if isinstance(block, dict) else {}


def _build_default_set_step_rows(set_key: str, profile: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = [str(x).strip() for x in list(profile.get("behavior_sequence", []) or []) if str(x).strip()]
    if not sequence:
        spec = dict(SET_LIBRARY.get(set_key, {}) or {})
        if isinstance(spec.get("element_steps", None), list) and list(spec.get("element_steps", []) or []):
            return [dict(item) for item in list(spec.get("element_steps", []) or []) if isinstance(item, dict)]
        sequence = [str(x).strip() for x in list(spec.get("element_template", []) or []) if str(x).strip()]
    param_overrides = dict(profile.get("element_param_overrides", {}) or {})
    auto_rules = dict(profile.get("element_auto_rules", {}) or {})
    rows: list[dict[str, Any]] = []
    for idx, element_key in enumerate(sequence):
        row: dict[str, Any] = {"element_class": str(element_key)}
        params = param_overrides.get(str(idx), {})
        rules = auto_rules.get(str(idx), {})
        if isinstance(params, dict) and params:
            row["params"] = dict(params)
        if isinstance(rules, dict) and rules:
            row["auto_rules"] = dict(rules)
        rows.append(row)
    return rows


def _write_stage3_behavior_defaults(sets_payload: dict[str, Any]) -> Path:
    path = COMMON_STAGE_CONFIG_PATH
    payload = load_yaml(path) if path.exists() else {}
    if not isinstance(payload, dict):
        payload = {}
    existing_block = dict(payload.get("stage3_behavior_library", {}) or {})
    elements_block = dict(existing_block.get("elements", {}) or {})
    shared_block = dict(existing_block.get("shared", {}) or {})
    normalized_sets: dict[str, Any] = {}
    for set_key, profile in dict(sets_payload or {}).items():
        if not isinstance(profile, dict):
            continue
        normalized_sets[str(set_key)] = {
            "generation_kind": str(profile.get("generation_kind", "auto") or "auto"),
            "allow_interleave_repeat": bool(profile.get("allow_interleave_repeat", False)),
            "max_total_elements": int(profile.get("max_total_elements", 0) or 0),
            "element_steps": _build_default_set_step_rows(str(set_key), dict(profile)),
        }
    payload["stage3_behavior_library"] = {
        "updated_at": _utc_now_iso() if "_utc_now_iso" in globals() else datetime.now(timezone.utc).isoformat(),
        "shared": shared_block,
        "elements": elements_block,
        "sets": normalized_sets,
    }
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


STAGE3_DEFAULT_TASKS = [
    "self_instance_recognition_joint",
    "self_set_instance_recognition",
    "self_element_instance_recognition",
    "self_element_instance_localization",
    "env_visible_count",
    "env_visible_intervals",
]

DIFFICULTY_BANDS: list[tuple[str, tuple[int, int | None]]] = [
    ("1", (1, 1)),
    ("2-3", (2, 3)),
    ("4-5", (4, 5)),
    ("6+", (6, None)),
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def _make_mp4_web_compatible(path: Path, bitrate: str | None = None) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None or not path.exists() or path.suffix.lower() != ".mp4":
        return False
    tmp_path = path.with_name(f"{path.stem}.webtmp.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    br = str(bitrate or "").strip()
    if br:
        cmd.extend(["-b:v", br, "-maxrate", br])
    cmd.extend(["-an", str(tmp_path)])
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=False)
        if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(path)
            return _is_mp4_web_compatible(path)
        elif tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return False


def _export_web_mp4_variant(src_path: Path, dst_path: Path, bitrate: str = "2M") -> Path | None:
    if not src_path.exists() or src_path.suffix.lower() != ".mp4":
        return None
    try:
        ensure_dir(dst_path.parent)
        shutil.copy2(src_path, dst_path)
        ok = _make_mp4_web_compatible(dst_path, bitrate=str(bitrate or "2M"))
        if not ok and dst_path.exists():
            dst_path.unlink(missing_ok=True)
        if ok and dst_path.exists() and dst_path.stat().st_size > 0 and _is_mp4_web_compatible(dst_path):
            return dst_path
    except Exception:
        pass
    return None


def _write_frame_manifest(
    manifest_path: Path,
    frames: list[str],
    fps: float,
    *,
    width: int | None = None,
    height: int | None = None,
) -> None:
    base = manifest_path.parent.resolve()
    normalized: list[str] = []
    for item in frames:
        p = Path(str(item))
        try:
            rel = p.resolve().relative_to(base)
            normalized.append(str(rel.as_posix()))
        except Exception:
            normalized.append(str(p.as_posix()))
    write_json(
        manifest_path,
        {
            "fps": float(max(1.0, fps)),
            "frame_count": int(len(frames)),
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
            "frames": normalized,
        },
    )


def _resolve_path_near(base_dir: Path, raw_path: Any) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    cand = Path(text)
    candidates = [cand]
    if not cand.is_absolute():
        candidates.append((base_dir / cand).resolve())
        candidates.append((WORKSPACE_ROOT / cand).resolve())
    for item in candidates:
        try:
            resolved = item.resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None


def _load_existing_final_task_plan(
    *,
    final_meta_path: Path,
    final_index_map_path: Path,
    waypoints_xyz: np.ndarray,
    default_source_pose_fps: float,
    default_speedup: float,
) -> dict[str, Any] | None:
    existing_meta = read_json_if_exists(final_meta_path, default={})
    if not isinstance(existing_meta, dict) or not existing_meta:
        return None
    video = dict(existing_meta.get("video", {}) or {})
    index_map_path = _resolve_path_near(final_meta_path.parent, video.get("frame_index_map", "")) or final_index_map_path
    index_map = read_json_if_exists(index_map_path, default={}) if index_map_path.exists() else {}
    if not isinstance(index_map, dict):
        return None
    rows = [dict(item) for item in list(index_map.get("frames", []) or []) if isinstance(item, dict)]
    if not rows:
        return None
    sampled_idx: list[int] = []
    last = -1
    total_points = int(waypoints_xyz.shape[0])
    for row in rows:
        try:
            src_idx = int(row.get("source_idx", -1) or -1)
        except Exception:
            return None
        if src_idx < 0 or src_idx >= total_points or src_idx <= last:
            return None
        sampled_idx.append(src_idx)
        last = src_idx
    if not sampled_idx:
        return None
    fps = float(video.get("fps", index_map.get("fps", 0.0)) or 0.0)
    if fps <= 0.0:
        return None
    source_pose_fps = float(video.get("source_pose_fps", default_source_pose_fps) or default_source_pose_fps)
    speedup = float(video.get("speedup", default_speedup) or default_speedup)
    playback_duration_sec = float(video.get("duration_sec", float(max(0, len(sampled_idx) - 1)) / float(max(1.0, fps))) or float(max(0, len(sampled_idx) - 1)) / float(max(1.0, fps)))
    source_duration_sec = float(video.get("source_duration_sec", float(max(0, sampled_idx[-1] - sampled_idx[0])) / float(max(1e-6, source_pose_fps))) or float(max(0, sampled_idx[-1] - sampled_idx[0])) / float(max(1e-6, source_pose_fps)))
    return {
        "meta": existing_meta,
        "video": video,
        "sampled_idx": sampled_idx,
        "fps": float(fps),
        "source_pose_fps": float(source_pose_fps),
        "speedup": float(speedup),
        "playback_duration_sec": float(playback_duration_sec),
        "source_duration_sec": float(source_duration_sec),
    }


def _is_mp4_web_compatible(path: Path) -> bool:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None or (not path.exists()) or path.suffix.lower() != ".mp4":
        return False
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_tag_string,pix_fmt",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
        if proc.returncode != 0:
            return False
        out = [line.strip().lower() for line in (proc.stdout or "").splitlines() if line.strip()]
        if len(out) < 3:
            return False
        codec_name, codec_tag, pix_fmt = out[0], out[1], out[2]
        return codec_name == "h264" and codec_tag in {"avc1", "h264"} and pix_fmt == "yuv420p"
    except Exception:
        return False


def _ensure_mp4_web_playable(path: Path) -> None:
    if path.suffix.lower() != ".mp4" or not path.exists():
        return
    if _is_mp4_web_compatible(path):
        return
    _make_mp4_web_compatible(path)


def read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_final_task_inputs_from_mission_dir(mission_dir: Path) -> dict[str, Any]:
    mission_dir = Path(mission_dir).resolve()
    constraint_path = mission_dir / "constraint_report.json"
    segments_path = mission_dir / "composed_segments.json"
    waypoints_path = mission_dir / "waypoints.npy"
    forwards_path = mission_dir / "forwards.npy"
    raw_waypoints_path = mission_dir / "composed_path_raw.npy"
    if not constraint_path.exists():
        raise FileNotFoundError(f"constraint_report_missing: {constraint_path}")
    if not segments_path.exists():
        raise FileNotFoundError(f"composed_segments_missing: {segments_path}")
    if not waypoints_path.exists():
        raise FileNotFoundError(f"waypoints_missing: {waypoints_path}")
    if not forwards_path.exists():
        raise FileNotFoundError(f"forwards_missing: {forwards_path}")

    constraint = read_json_if_exists(constraint_path, default={})
    segments_payload = read_json_if_exists(segments_path, default={})
    if not isinstance(constraint, dict):
        raise RuntimeError(f"invalid_constraint_report: {constraint_path}")
    if not isinstance(segments_payload, dict):
        raise RuntimeError(f"invalid_composed_segments: {segments_path}")

    center = list(constraint.get("landmark_center_3d", []) or [])
    bbox_obj = dict(constraint.get("landmark_bbox_3d", {}) or {})
    size = list(bbox_obj.get("size", []) or [])
    if len(center) < 3 or len(size) < 3:
        raise RuntimeError(f"missing_target_geometry: {constraint_path}")

    mission_meta = {
        "task_type": str(constraint.get("task_type", constraint.get("task_family", "")) or ""),
        "task_subtype": str(constraint.get("task_subtype", constraint.get("task_subtype", "")) or ""),
        "task_difficulty": str(constraint.get("task_difficulty", "") or ""),
        "task_difficulty_score": float(constraint.get("task_difficulty_score", 0.0) or 0.0),
        "set_instance": dict(constraint.get("set_instance", {}) or {}),
        "element_instances": list(constraint.get("element_instances", []) or []),
        "self_state": {
            "landmark_order": list((constraint.get("set_instance", {}) or {}).get("landmark_order", []) or []),
        },
        "mode_sequence": list(constraint.get("mode_sequence", []) or []),
        "event_sequence": list(constraint.get("event_sequence", []) or []),
        "secondary_instance_ids": list(constraint.get("secondary_instance_ids", []) or []),
        "landmark_set_map": dict((constraint.get("summary", {}) or {}).get("landmark_set_map", {}) or {}) if isinstance(constraint.get("summary", {}), dict) else {},
    }
    target_bbox_list = [
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(size[0]),
        float(size[1]),
        float(size[2]),
        float(bbox_obj.get("yaw_deg", 0.0) or 0.0),
    ]
    return {
        "mission_dir": mission_dir,
        "constraint": constraint,
        "segments": list(segments_payload.get("segments", []) or []),
        "waypoints": np.load(waypoints_path).astype(np.float32),
        "forwards": np.load(forwards_path).astype(np.float32),
        "raw_waypoints": np.load(raw_waypoints_path).astype(np.float32) if raw_waypoints_path.exists() else None,
        "source_pose_fps": float(((constraint.get("summary", {}) or {}).get("source_pose_fps", (constraint.get("summary", {}) or {}).get("fps", 10.0)) or 10.0)),
        "target_center_3d": [float(center[0]), float(center[1]), float(center[2])],
        "target_bbox_list": target_bbox_list,
        "mission_meta": mission_meta,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sample_preview_points(points: np.ndarray, max_points: int = 240) -> dict[str, list[list[float]]]:
    if points.size == 0:
        return {"xy": [], "xz": []}
    count = int(points.shape[0])
    if count <= max_points:
        sampled = points
    else:
        idx = np.linspace(0, count - 1, num=max_points, endpoint=True).astype(np.int64)
        sampled = points[idx]
    xy = [[float(p[0]), float(p[1])] for p in sampled]
    xz = [[float(p[0]), float(p[2])] for p in sampled]
    return {"xy": xy, "xz": xz}


def _build_run_args(args: argparse.Namespace, overrides: dict[str, Any]) -> argparse.Namespace:
    payload = vars(copy.deepcopy(args))
    payload.update(overrides)
    return argparse.Namespace(**payload)


def _resolve_base_dir(config: dict[str, Any]) -> Path:
    return resolve_base_dir(config, workspace_root=WORKSPACE_ROOT)


def _stage3_cfg(config: dict[str, Any]) -> dict[str, Any]:
    common_cfg = dict((_load_common_stage_cfg().get("stage3_runtime_defaults", {}) or {}))
    if not common_cfg:
        raise RuntimeError("missing_stage3_runtime_defaults_in_common_stage_configs")
    scene_cfg = dict(config.get("stage3", {}) or config.get("trajectory", {}) or {})
    return _deep_merge_dict(common_cfg, scene_cfg)


def _stage3_temporal_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("stage3_temporal", {}) or {})


def _resolve_stage3_layout(config: dict[str, Any], scene_id: str) -> dict[str, Path]:
    task_cfg = config.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim")).lower().strip()
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine_name, workspace_root=WORKSPACE_ROOT)
    artifact_scene_root = resolve_task_pipeline_scene_root(
        config,
        scene_id=scene_id,
        engine=engine_name,
        workspace_root=WORKSPACE_ROOT,
    ) or scene_root
    task_root_name = resolve_output_dir_name(config, key="stage3_task_root_dir", default="stage3_tasks")
    stage3_root = artifact_scene_root / task_root_name
    missions_root = stage3_root / resolve_output_dir_name(config, key="stage3_mission_dir", default="missions")
    review_root = stage3_root / resolve_output_dir_name(config, key="stage3_review_dir", default="review")
    datasets_root = stage3_root / resolve_output_dir_name(config, key="stage3_dataset_dir", default="datasets")
    experiments_root = stage3_root / resolve_output_dir_name(config, key="stage3_experiment_dir", default="experiments")
    reports_root = stage3_root / resolve_output_dir_name(config, key="stage3_report_dir", default="reports")
    cache_root = stage3_root / resolve_output_dir_name(config, key="stage3_cache_dir", default="cache")
    return {
        "scene_root": scene_root,
        "artifacts_scene_root": artifact_scene_root,
        "stage3_root": stage3_root,
        "missions_root": missions_root,
        "review_root": review_root,
        "datasets_root": datasets_root,
        "experiments_root": experiments_root,
        "reports_root": reports_root,
        "cache_root": cache_root,
    }


def _resolve_stage3_dirs(config: dict[str, Any], scene_id: str) -> tuple[Path, Path, Path]:
    layout = _resolve_stage3_layout(config=config, scene_id=scene_id)
    return layout["scene_root"], layout["missions_root"], layout["review_root"]


def _resolve_scene_id(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.scene_id:
        return str(args.scene_id)
    task_cfg = config.get("task", {}) or {}
    scene_id = str(task_cfg.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError("scene_id is required (cli --scene-id or task.scene_id in config)")
    return scene_id


def _to_bbox_list(bbox_3d: Any) -> list[float]:
    if isinstance(bbox_3d, list):
        vals = [float(v) for v in bbox_3d]
        if len(vals) >= 6:
            return vals
    if isinstance(bbox_3d, dict):
        size = bbox_3d.get("size", [3.0, 3.0, 3.0])
        yaw_deg = float(bbox_3d.get("yaw_deg", 0.0) or 0.0)
        if isinstance(size, list) and len(size) >= 3:
            sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        else:
            pmin = bbox_3d.get("min", [0.0, 0.0, 0.0])
            pmax = bbox_3d.get("max", [3.0, 3.0, 3.0])
            if isinstance(pmin, list) and isinstance(pmax, list) and len(pmin) >= 3 and len(pmax) >= 3:
                sx, sy, sz = float(pmax[0] - pmin[0]), float(pmax[1] - pmin[1]), float(pmax[2] - pmin[2])
            else:
                sx, sy, sz = 3.0, 3.0, 3.0
        return [0.0, 0.0, 0.0, max(1e-3, sx), max(1e-3, sy), max(1e-3, sz), yaw_deg]
    return [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0]


def _normalize_landmark_item(item: dict[str, Any]) -> dict[str, Any]:
    center = item.get("center_3d", [0.0, 0.0, 0.0])
    if not isinstance(center, list) or len(center) < 3:
        center = [0.0, 0.0, 0.0]
    center3 = [float(center[0]), float(center[1]), float(center[2])]
    bbox_list = _to_bbox_list(item.get("bbox_3d", None))

    out = dict(item)
    out["center_3d"] = center3
    out["bbox_3d"] = bbox_list
    out["landmark_category"] = str(
        out.get("landmark_category")
        or out.get("auto_label_category")
        or out.get("class_name")
        or ""
    ).strip()
    out["landmark_subcategory"] = str(
        out.get("landmark_subcategory")
        or out.get("auto_label_subcategory")
        or ""
    ).strip()
    out["landmark_description"] = str(
        out.get("landmark_description")
        or out.get("auto_label_description")
        or ""
    ).strip()
    return out


def _build_keepout_boxes(instances: list[dict[str, Any]], *, margin_xy: float, margin_z: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in list(instances or []):
        if not isinstance(raw, dict):
            continue
        item = _normalize_landmark_item(raw)
        center = list(item.get("center_3d", []) or [])
        bbox = list(item.get("bbox_3d", []) or [])
        if len(center) < 3 or len(bbox) < 7:
            continue
        rows.append(
            {
                "instance_id": str(item.get("instance_id", "") or ""),
                "center_3d": [float(center[0]), float(center[1]), float(center[2])],
                "bbox_list": [float(v) for v in bbox[:7]],
                "margin_xy": float(margin_xy),
                "margin_z": float(margin_z),
            }
        )
    return rows


def _repair_summary_from_segments(segments: list[dict[str, Any]], *, waypoint_count: int) -> dict[str, Any]:
    max_lift = 0.0
    lifted_points = 0
    repaired_segments = 0
    for seg in list(segments or []):
        repair = dict((seg or {}).get("repair", {}) or {})
        lift = float(repair.get("max_lift_m", 0.0) or 0.0)
        pts = int(repair.get("lifted_points", 0) or 0)
        if lift > 0.0 or pts > 0:
            repaired_segments += 1
        max_lift = max(max_lift, lift)
        lifted_points += pts
    lifted_fraction = float(lifted_points) / float(max(1, int(waypoint_count))) if waypoint_count > 0 else 0.0
    return {
        "repaired_segments": int(repaired_segments),
        "lifted_points": int(lifted_points),
        "lifted_fraction": float(lifted_fraction),
        "max_lift_m": float(max_lift),
    }


def _landmark_category(item: dict[str, Any]) -> str:
    return str(item.get("landmark_category", "") or item.get("class_name", "") or "landmark").strip()


def _landmark_subcategory(item: dict[str, Any]) -> str:
    return str(item.get("landmark_subcategory", "") or "").strip()


def _landmark_description(item: dict[str, Any]) -> str:
    desc = str(item.get("landmark_description", "") or "").strip()
    if desc:
        return desc
    desc = str(item.get("description", "") or "").strip()
    if desc:
        return desc
    return str(item.get("instance_id", "") or "").strip() or "landmark"


def _normalize_generation_kind(raw: Any) -> str:
    text = str(raw or "auto").strip().lower()
    if text == "atomic-only":
        return "atomic-only"
    if text == "composite-driven":
        return "composite-driven"
    return "auto"


def _difficulty_band_from_score(score: float) -> str:
    if score < 0.40:
        return "easy"
    if score < 0.70:
        return "medium"
    return "hard"


def _resolve_source_instances(
    scene_root: Path,
    scene_id: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[str, Path, list[dict[str, Any]]]:
    if args.instances_json:
        source_path = Path(str(args.instances_json))
        payload = read_json_if_exists(source_path, default={})
        if not isinstance(payload, dict):
            raise ValueError(f"invalid json payload: {source_path}")
        if "valid_instances" in payload:
            instances = list(payload.get("valid_instances", []) or [])
            source_name = "valid_instances"
        else:
            instances = list(payload.get("instances", []) or [])
            source_name = "instances"
    else:
        review_dir_name = resolve_output_dir_name(config, key="stage2_review_dir", default="landmarks_review")
        source_path = scene_root / review_dir_name / f"{scene_id}.valid_instances.json"
        payload = read_json_if_exists(source_path, default={})
        if not isinstance(payload, dict):
            raise ValueError(f"invalid json payload: {source_path}")
        instances = list(payload.get("valid_instances", []) or [])
        source_name = "valid_instances"

    instances = [it for it in instances if isinstance(it, dict)]
    if source_name == "valid_instances":
        instances = [it for it in instances if str(it.get("annotation_status", "") or "").strip().lower() == "labeled"]
        instances = [it for it in instances if str(it.get("review_action", "") or "").strip().lower() in {"keep", ""}]
        instances = [it for it in instances if str(it.get("landmark_description", "") or it.get("description", "") or "").strip()]
    if not instances:
        if source_name == "valid_instances":
            raise FileNotFoundError(f"no usable valid_instances in source: {source_path}")
        raise FileNotFoundError(f"no usable instances in source: {source_path}")
    return source_name, source_path, instances


def _pick_landmark(instances: list[dict[str, Any]], landmark_id: str | None) -> dict[str, Any]:
    if landmark_id:
        target = str(landmark_id)
        for item in instances:
            if str(item.get("instance_id", "")) == target:
                return item
        raise ValueError(f"landmark_id not found: {landmark_id}")

    ranked = sorted(instances, key=lambda it: int(it.get("point_count", 0) or 0), reverse=True)
    return ranked[0]


def _select_secondary_landmarks(
    *,
    instances: list[dict[str, Any]],
    primary: dict[str, Any],
    radius_m: float,
    max_secondary: int,
) -> list[dict[str, Any]]:
    primary_id = str(primary.get("instance_id", "") or "")
    primary_center = list(primary.get("center_3d", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
    out: list[tuple[float, dict[str, Any]]] = []
    for item in instances:
        if str(item.get("instance_id", "") or "") == primary_id:
            continue
        center = list(item.get("center_3d", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])
        if len(center) < 3:
            continue
        dist = float(math.sqrt(sum((float(center[i]) - float(primary_center[i])) ** 2 for i in range(3))))
        if dist > float(radius_m):
            continue
        score = dist - 0.01 * float(int(item.get("point_count", 0) or 0))
        out.append((score, item))
    out.sort(key=lambda pair: pair[0])
    return [dict(pair[1]) for pair in out[: max(0, int(max_secondary))]]


def _difficulty_band_from_visible_count(visible_count: int) -> str:
    count = max(0, int(visible_count))
    for label, (lo, hi) in DIFFICULTY_BANDS:
        if count >= lo and (hi is None or count <= hi):
            return label
    return "0"


def _compute_self_state_difficulty(
    *,
    duration_sec: float,
    landmark_count: int,
    element_count: int,
    revisit_count: int,
) -> tuple[float, str]:
    d_time = min(1.0, max(0.0, float(duration_sec) / 80.0))
    d_landmark = min(1.0, max(0.0, float(landmark_count) / 5.0))
    d_element = min(1.0, max(0.0, float(element_count) / 8.0))
    d_revisit = min(1.0, max(0.0, float(revisit_count) / 3.0))
    d_transition = min(1.0, max(0.0, float(max(0, element_count - 1)) / 7.0))
    score = 0.30 * d_time + 0.25 * d_landmark + 0.30 * d_element + 0.10 * d_transition + 0.05 * d_revisit
    return float(score), _difficulty_band_from_score(float(score))


def _compute_env_difficulty(
    *,
    duration_sec: float,
    visible_count: int,
    visible_duration_ratio: float,
    mean_visible_bbox_area_ratio: float,
) -> tuple[float, str]:
    d_time = min(1.0, max(0.0, float(duration_sec) / 80.0))
    d_count = min(1.0, max(0.0, float(visible_count) / 8.0))
    r_spatiotemporal = 0.5 * float(mean_visible_bbox_area_ratio) + 0.5 * float(visible_duration_ratio)
    d_spatiotemporal = 1.0 - min(1.0, max(0.0, r_spatiotemporal))
    score = 0.30 * d_time + 0.35 * d_count + 0.35 * d_spatiotemporal
    if score < 0.35:
        band = "easy"
    elif score < 0.68:
        band = "medium"
    else:
        band = "hard"
    return float(score), str(band)


def _auto_pick_landmark(instances: list[dict[str, Any]]) -> dict[str, Any]:
    def _score(item: dict[str, Any]) -> tuple[float, int]:
        bbox = _to_bbox_list(item.get("bbox_3d", None))
        diag = _estimate_scale_from_bbox(bbox)
        points = int(item.get("point_count", 0) or 0)
        bird = 1 if any(bool(v.get("is_query_view", False)) for v in list(item.get("rgb_views", []) or [])) else 0
        return (diag * 4.0 + math.log1p(max(0, points)) + bird * 2.0, points)

    ranked = sorted(instances, key=_score, reverse=True)
    return ranked[0]


def _matching_set_keys_for_mode(mode: str) -> list[str]:
    mode_norm = str(mode or "single-landmark").strip().lower()
    keys: list[str] = []
    for key, spec in SET_LIBRARY.items():
        scope = str(spec.get("scope", "single-landmark") or "single-landmark").strip().lower()
        if mode_norm == "multi-landmark":
            if "multi" in scope:
                keys.append(str(key))
        else:
            if "multi" not in scope:
                keys.append(str(key))
    return keys


def _single_landmark_component_set_keys() -> list[str]:
    keys: list[str] = []
    for key, spec in SET_LIBRARY.items():
        if str(spec.get("scope", "single-landmark") or "single-landmark") != "single-landmark":
            continue
        if bool(spec.get("multi_landmark_component", False)):
            keys.append(str(key))
    return keys


def _heuristic_set_key_for_landmark(*, landmark: dict[str, Any], mode: str = "single-landmark") -> str:
    category = _landmark_category(landmark).lower()
    diag = _estimate_scale_from_bbox(_to_bbox_list(landmark.get("bbox_3d", None)))
    if diag <= 8.0:
        return "atomic_gradual_approach"
    if "building" in category or "tower" in category or diag >= 22.0:
        return "spiral_inspection"
    if "square" in category:
        return "square_inspection"
    if "triangle" in category:
        return "triangular_inspection"
    return "circular_inspection"


def _expand_set_step_defs(set_spec: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(set_spec.get("element_steps", None), list) and list(set_spec.get("element_steps", []) or []):
        return [dict(item) for item in list(set_spec.get("element_steps", []) or []) if isinstance(item, dict)]
    return [{"element_class": str(x).strip(), "params": {}, "target_binding": "primary"} for x in list(set_spec.get("element_template", []) or []) if str(x).strip()]


def _build_multi_landmark_composite_set(
    *,
    selected_landmarks: list[dict[str, Any]],
    landmark_set_map: dict[str, str] | None,
    allowed_set_types: list[str] | None,
    auto_rule: str,
    seed: int,
    explicit_multi_set_key: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    explicit_multi = str(explicit_multi_set_key or "").strip()
    if explicit_multi and explicit_multi in SET_LIBRARY and str((SET_LIBRARY.get(explicit_multi, {}) or {}).get("scope", "")) == "single-landmark":
        chosen_map = {str(item.get("instance_id", "") or ""): explicit_multi for item in selected_landmarks}
        composite_steps: list[dict[str, Any]] = []
        set_spec = dict(SET_LIBRARY.get(explicit_multi, {}) or {})
        for idx, landmark in enumerate(selected_landmarks):
            landmark_id = str(landmark.get("instance_id", "") or "")
            for step in _expand_set_step_defs(set_spec):
                step_copy = dict(step)
                step_copy["target_instance_id"] = landmark_id
                step_copy["target_binding"] = "primary" if idx == 0 else "secondary"
                composite_steps.append(step_copy)
        return {
            "set_key": "multi_landmark_composite_inspection",
            "display_name": f"多地标巡检组合-{explicit_multi}",
            "scope": "multi-landmark",
            "description": f"对多个地标统一执行 {explicit_multi} 复合巡检，并在地标之间做快速直飞衔接。",
            "element_steps": composite_steps,
            "landmark_count_default": len(selected_landmarks),
            "allow_revisit": False,
            "selected_component_set_map": chosen_map,
            "component_sequence": [explicit_multi for _ in selected_landmarks],
        }, chosen_map
    component_choices = [str(x).strip() for x in list(allowed_set_types or []) if str(x).strip()]
    if not component_choices:
        component_choices = _single_landmark_component_set_keys()
    chosen_map: dict[str, str] = {}
    composite_steps: list[dict[str, Any]] = []
    set_name_parts: list[str] = []
    for idx, landmark in enumerate(selected_landmarks):
        landmark_id = str(landmark.get("instance_id", "") or "")
        explicit_set = str((landmark_set_map or {}).get(landmark_id, "") or "").strip() or None
        set_spec = _select_set_template(
            landmark=landmark,
            set_type=explicit_set,
            mode="single-landmark",
            allowed_set_types=component_choices,
            auto_rule=auto_rule,
            seed=int(seed) + idx * 131,
        )
        set_key = str(set_spec.get("set_key", "") or "")
        chosen_map[landmark_id] = set_key
        set_name_parts.append(set_key)
        for step in _expand_set_step_defs(set_spec):
            step_copy = dict(step)
            step_copy["target_instance_id"] = landmark_id
            step_copy["target_binding"] = "primary" if idx == 0 else "secondary"
            composite_steps.append(step_copy)
    composite_set = {
        "set_key": "multi_landmark_composite_inspection",
        "display_name": "多地标巡检组合",
        "scope": "multi-landmark",
        "description": "按所选地标逐个拼接单地标巡检 set。",
        "element_steps": composite_steps,
        "landmark_count_default": len(selected_landmarks),
        "allow_revisit": False,
        "selected_component_set_map": chosen_map,
        "component_sequence": list(set_name_parts),
    }
    return composite_set, chosen_map


def _select_set_template(
    *,
    landmark: dict[str, Any],
    set_type: str | None,
    mode: str = "single-landmark",
    allowed_set_types: list[str] | None = None,
    auto_rule: str = "heuristic",
    seed: int | None = None,
) -> dict[str, Any]:
    if set_type:
        if set_type not in SET_LIBRARY:
            raise ValueError(f"unsupported set_type: {set_type}")
        out = dict(SET_LIBRARY[set_type])
        out["set_key"] = str(set_type)
        return out

    candidate_keys = _matching_set_keys_for_mode(mode)
    allowed = [str(x).strip() for x in list(allowed_set_types or []) if str(x).strip()]
    if allowed:
        filtered = [key for key in candidate_keys if key in allowed]
        if filtered:
            candidate_keys = filtered
    if not candidate_keys:
        candidate_keys = _matching_set_keys_for_mode(mode)

    preferred_key = _heuristic_set_key_for_landmark(landmark=landmark, mode=mode)
    rule = str(auto_rule or "heuristic").strip().lower()
    if rule not in {"heuristic", "random", "round_robin"}:
        rule = "heuristic"
    if rule == "random":
        rng = random.Random(int(seed or 42))
        chosen_key = rng.choice(candidate_keys)
    elif rule == "round_robin":
        pivot = int(seed or 0)
        chosen_key = candidate_keys[pivot % max(1, len(candidate_keys))]
    else:
        chosen_key = preferred_key if preferred_key in candidate_keys else candidate_keys[0]

    out = dict(SET_LIBRARY[chosen_key])
    out["set_key"] = chosen_key
    out["auto_set_rule"] = rule
    out["allowed_set_types"] = list(candidate_keys)
    return out


def _resolve_generation_kind(args: argparse.Namespace, set_spec: dict[str, Any]) -> str:
    kind = _normalize_generation_kind(getattr(args, "generation_kind", "auto"))
    if kind != "auto":
        return kind
    if args.behavior_sequence:
        return "atomic-only"
    if str(set_spec.get("scope", "single-landmark")) == "multi-landmark":
        return "composite-driven"
    if str(set_spec.get("set_key", "") or "").startswith("atomic_"):
        return "atomic-only"
    return "composite-driven"


def _normalize_angle_deg(value: float) -> float:
    out = float(value)
    while out > 180.0:
        out -= 360.0
    while out < -180.0:
        out += 360.0
    return float(out)


def _snap_numeric_with_spec(value: float, spec: dict[str, Any]) -> float:
    out = float(value)
    min_v = spec.get("min")
    max_v = spec.get("max")
    step = spec.get("step")
    if min_v is not None:
        out = max(float(min_v), out)
    if max_v is not None:
        out = min(float(max_v), out)
    if step is not None:
        step_f = float(step)
        if step_f > 0.0:
            base = float(min_v) if min_v is not None else 0.0
            out = base + round((out - base) / step_f) * step_f
            if min_v is not None:
                out = max(float(min_v), out)
            if max_v is not None:
                out = min(float(max_v), out)
    default = spec.get("default")
    if isinstance(default, int) and float(out).is_integer():
        return float(int(round(out)))
    return float(out)


def _extract_step_param_map(raw: Any, *, step_index: int, element_key: str) -> dict[str, Any]:
    if isinstance(raw, list):
        if 0 <= int(step_index) < len(raw) and isinstance(raw[int(step_index)], dict):
            return dict(raw[int(step_index)])
        return {}
    if not isinstance(raw, dict):
        return {}
    for key in [str(step_index), element_key, f"{element_key}#{step_index}"]:
        value = raw.get(key, None)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _sample_param_from_auto_rule(
    *,
    rng: np.random.Generator,
    spec: dict[str, Any],
    rule: dict[str, Any],
) -> float | None:
    if not bool(rule.get("enabled", False)):
        return None
    if spec.get("choices"):
        choices = [choice for choice in list(spec.get("choices", []) or []) if str(choice).strip()]
        if not choices:
            return None
        picked = str(rule.get("choice", "") or "").strip()
        if picked and picked in choices:
            return picked
        return str(choices[int(rng.integers(0, len(choices)))])
    min_v = float(rule.get("min", spec.get("min", 0.0)) if rule.get("min", None) is not None else spec.get("min", 0.0) or 0.0)
    max_v = float(rule.get("max", spec.get("max", min_v)) if rule.get("max", None) is not None else spec.get("max", min_v) or min_v)
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    method = str(rule.get("method", "random") or "random").strip().lower()
    if method not in {"random", "normal"}:
        method = "random"
    if method == "normal":
        mean = float(rule.get("mean", (min_v + max_v) * 0.5) or (min_v + max_v) * 0.5)
        std = float(rule.get("std", max((max_v - min_v) / 6.0, float(spec.get("step", 1.0) or 1.0))) or max((max_v - min_v) / 6.0, float(spec.get("step", 1.0) or 1.0)))
        sampled = float(rng.normal(loc=mean, scale=max(1e-6, std)))
    else:
        sampled = float(rng.uniform(min_v, max_v))
    merged_spec = dict(spec)
    if rule.get("min", None) is not None:
        merged_spec["min"] = min_v
    if rule.get("max", None) is not None:
        merged_spec["max"] = max_v
    if rule.get("step", None) is not None:
        merged_spec["step"] = float(rule.get("step"))
    return _snap_numeric_with_spec(sampled, merged_spec)


def _sample_param_from_spec_default(
    *,
    rng: np.random.Generator,
    spec: dict[str, Any],
) -> Any:
    choices = [choice for choice in list(spec.get("choices", []) or []) if str(choice).strip()]
    if choices:
        method = str(spec.get("auto_method_default", "fixed") or "fixed").strip().lower()
        if method == "random":
            return str(choices[int(rng.integers(0, len(choices)))])
        return spec.get("default", choices[0])
    if spec.get("min", None) is None or spec.get("max", None) is None:
        return spec.get("default")
    min_v = float(spec.get("min", 0.0) or 0.0)
    max_v = float(spec.get("max", min_v) or min_v)
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    method = str(spec.get("auto_method_default", "fixed") or "fixed").strip().lower()
    if method == "normal":
        mean = float(spec.get("auto_center", spec.get("default", (min_v + max_v) * 0.5)) or (min_v + max_v) * 0.5)
        step = float(spec.get("step", 1.0) or 1.0)
        std = float(spec.get("auto_std", max((max_v - min_v) / 6.0, step)) or max((max_v - min_v) / 6.0, step))
        sampled = float(rng.normal(loc=mean, scale=max(1e-6, std)))
    elif method == "random":
        sampled = float(rng.uniform(min_v, max_v))
    else:
        return spec.get("default")
    return _snap_numeric_with_spec(sampled, dict(spec))


def _is_orbit_like_element(element_key: str) -> bool:
    return str(element_key or "") in {"circular_orbit", "spiral_orbit", "square_orbit", "triangular_orbit", "figure8_orbit"}


def _orbit_anchor_radius_m(landmark: dict[str, Any], *, safety_distance_m: float = 2.0) -> float:
    bbox = _to_bbox_list(landmark.get("bbox_3d", None))
    sx = float(bbox[3]) if len(bbox) > 3 else 3.0
    sy = float(bbox[4]) if len(bbox) > 4 else 3.0
    half_xy_diag = 0.5 * float(math.sqrt(max(1e-6, sx * sx + sy * sy)))
    return max(0.5, half_xy_diag + max(0.2, float(safety_distance_m)))


def _orbit_join_radius_from_step(
    *,
    step: dict[str, Any] | None,
    landmark: dict[str, Any],
    safety_distance_m: float,
) -> float | None:
    if not isinstance(step, dict):
        return None
    element_key = str(step.get("element_key", "") or "")
    if not _is_orbit_like_element(element_key):
        return None
    params = dict(step.get("base_params", {}) or {})
    params_spec = dict((ELEMENT_LIBRARY.get(element_key, {}) or {}).get("params", {}) or {})
    anchor = _orbit_anchor_radius_m(landmark, safety_distance_m=safety_distance_m)

    def _value(name: str, fallback: float) -> float:
        if name in params and params.get(name, None) is not None:
            try:
                return float(params.get(name))
            except Exception:
                pass
        try:
            return float(dict(params_spec.get(name, {}) or {}).get("default", fallback) or fallback)
        except Exception:
            return float(fallback)

    if element_key == "spiral_orbit":
        return max(anchor, anchor + _value("start_extension_m", 0.0))
    return max(anchor, anchor + _value("extension_m", 0.0))


def _build_base_explicit_params(
    *,
    element_key: str,
    step_def: dict[str, Any],
    param_overrides: Any,
    auto_param_rules: Any,
    step_index: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], set[str]]:
    params_spec = dict((ELEMENT_LIBRARY.get(element_key, {}) or {}).get("params", {}) or {})
    manual_map = _extract_step_param_map(param_overrides, step_index=step_index, element_key=element_key)
    auto_map = _extract_step_param_map(auto_param_rules, step_index=step_index, element_key=element_key)
    explicit_params: dict[str, Any] = dict(step_def.get("params", {}) or {})
    locked_keys: set[str] = set(str(k) for k in explicit_params.keys())
    for key, raw in manual_map.items():
        if key not in params_spec:
            continue
        if raw in {"", None}:
            continue
        rule = dict(params_spec.get(key, {}) or {})
        if rule.get("choices"):
            explicit_params[key] = str(raw)
        else:
            try:
                explicit_params[key] = _snap_numeric_with_spec(float(raw), rule)
            except Exception:
                continue
        locked_keys.add(str(key))
    for key, spec in params_spec.items():
        if key in explicit_params:
            continue
        rule = auto_map.get(key, None)
        if isinstance(rule, dict):
            sampled = _sample_param_from_auto_rule(rng=rng, spec=dict(spec or {}), rule=rule)
            if sampled is not None:
                explicit_params[key] = sampled
            continue
        sampled = _sample_param_from_spec_default(rng=rng, spec=dict(spec or {}))
        if sampled is not None:
            explicit_params[key] = sampled
    return explicit_params, locked_keys


def _apply_adaptive_param_hints(
    *,
    element_key: str,
    landmark: dict[str, Any],
    current_pos: np.ndarray,
    explicit_params: dict[str, Any],
    locked_param_keys: set[str] | None = None,
    previous_step: dict[str, Any] | None = None,
    next_step: dict[str, Any] | None = None,
    safety_distance_m: float = 2.0,
) -> dict[str, Any]:
    out = dict(explicit_params)
    locked = set(str(x) for x in list(locked_param_keys or set()))
    bbox = _to_bbox_list(landmark.get("bbox_3d", None))
    center = np.asarray(list(landmark.get("center_3d", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])[:3], dtype=np.float32)
    vec_xy = current_pos[:2] - center[:2]
    curr_radius = float(np.linalg.norm(vec_xy))
    yaw_deg = float(bbox[6]) if len(bbox) > 6 else 0.0
    heading_out = math.degrees(math.atan2(float(vec_xy[1]), float(vec_xy[0]))) if curr_radius > 1e-3 else yaw_deg
    z_base_no_offset = float(center[2] + 0.8 * (float(bbox[5]) if len(bbox) > 5 else 3.0))
    prev_same_landmark = (
        isinstance(previous_step, dict)
        and str(((previous_step.get("landmark", {}) or {}).get("instance_id", "") or "")) == str(landmark.get("instance_id", "") or "")
    )
    next_same_landmark = (
        isinstance(next_step, dict)
        and str(((next_step.get("landmark", {}) or {}).get("instance_id", "") or "")) == str(landmark.get("instance_id", "") or "")
    )
    if prev_same_landmark:
        out.setdefault("adaptive_start_from_current", True)
        out.setdefault("adaptive_altitude_m", float(current_pos[2]))
    if element_key == "gradual_approach":
        approach_spec = dict((ELEMENT_LIBRARY.get("gradual_approach", {}) or {}).get("params", {}) or {})
        next_join_radius = _orbit_join_radius_from_step(
            step=next_step,
            landmark=landmark,
            safety_distance_m=float(safety_distance_m),
        ) if next_same_landmark else None
        should_align_to_current_ray = prev_same_landmark or (next_join_radius is not None)
        if should_align_to_current_ray and curr_radius > 1e-3 and "yaw_offset_deg" not in locked:
            desired_offset = _normalize_angle_deg(float(heading_out - yaw_deg))
            out["yaw_offset_deg"] = _snap_numeric_with_spec(
                desired_offset,
                dict(approach_spec.get("yaw_offset_deg", {}) or {}),
            )
            out["adaptive_heading_deg"] = float(heading_out)
        if next_join_radius is not None:
            out["adaptive_end_radius_m"] = float(next_join_radius)
            if "travel_distance_m" not in locked and curr_radius > 1e-3:
                dist_spec = dict(approach_spec.get("travel_distance_m", {}) or {})
                desired_dist = max(curr_radius, next_join_radius + 8.0)
                out["travel_distance_m"] = _snap_numeric_with_spec(desired_dist, dist_spec)
        next_altitude = _step_target_altitude_m(step=next_step, landmark=landmark) if next_same_landmark else None
        if next_altitude is not None:
            out["adaptive_end_altitude_m"] = float(next_altitude)
    if element_key in {"circular_orbit", "square_orbit", "triangular_orbit", "figure8_orbit"}:
        spec = dict((ELEMENT_LIBRARY.get(element_key, {}) or {}).get("params", {}).get("extension_m", {}) or {})
        anchor = _orbit_anchor_radius_m(landmark, safety_distance_m=safety_distance_m)
        desired_ext = max(0.0, curr_radius - anchor)
        if "extension_m" not in locked:
            out["extension_m"] = _snap_numeric_with_spec(desired_ext if desired_ext > 0.0 else float(spec.get("default", 12.0) or 12.0), spec)
        if "altitude_offset_m" not in locked:
            out["altitude_offset_m"] = float(current_pos[2] - z_base_no_offset)
        out["adaptive_altitude_m"] = float(current_pos[2])
    if element_key == "spiral_orbit":
        anchor = _orbit_anchor_radius_m(landmark, safety_distance_m=safety_distance_m)
        start_spec = dict((ELEMENT_LIBRARY.get("spiral_orbit", {}) or {}).get("params", {}).get("start_extension_m", {}) or {})
        end_spec = dict((ELEMENT_LIBRARY.get("spiral_orbit", {}) or {}).get("params", {}).get("end_extension_m", {}) or {})
        desired_start_ext = max(0.0, curr_radius - anchor)
        if "start_extension_m" not in locked:
            out["start_extension_m"] = _snap_numeric_with_spec(
                desired_start_ext if desired_start_ext > 0.0 else float(start_spec.get("default", 10.0) or 10.0),
                start_spec,
            )
        if "end_extension_m" not in locked:
            min_end = float(out.get("start_extension_m", desired_start_ext if desired_start_ext > 0.0 else float(start_spec.get("default", 10.0) or 10.0))) + 4.0
            out["end_extension_m"] = _snap_numeric_with_spec(
                max(min_end, float(end_spec.get("default", 24.0) or 24.0)),
                end_spec,
            )
        if "altitude_offset_m" not in locked:
            out["altitude_offset_m"] = float(current_pos[2] - z_base_no_offset)
        out["adaptive_altitude_m"] = float(current_pos[2])
    if element_key == "sky_rise":
        if "top_extension_m" not in locked and prev_same_landmark:
            top_spec = dict((ELEMENT_LIBRARY.get("sky_rise", {}) or {}).get("params", {}).get("top_extension_m", {}) or {})
            desired_top = max(0.0, curr_radius)
            out["top_extension_m"] = _snap_numeric_with_spec(
                desired_top if desired_top > 0.0 else float(top_spec.get("default", 4.0) or 4.0),
                top_spec,
            )
        out["adaptive_altitude_m"] = float(current_pos[2])
    if element_key == "comet":
        comet_spec = dict((ELEMENT_LIBRARY.get("comet", {}) or {}).get("params", {}).get("extension_m", {}) or {})
        anchor = _orbit_anchor_radius_m(landmark, safety_distance_m=safety_distance_m)
        desired_ext = max(0.0, curr_radius - anchor)
        if "extension_m" not in locked and prev_same_landmark:
            out["extension_m"] = _snap_numeric_with_spec(
                desired_ext if desired_ext > 0.0 else float(comet_spec.get("default", 18.0) or 18.0),
                comet_spec,
            )
        out["adaptive_altitude_m"] = float(current_pos[2])
    if element_key == "gradual_depart":
        depart_spec = dict((ELEMENT_LIBRARY.get("gradual_depart", {}) or {}).get("params", {}) or {})
        if "yaw_offset_deg" not in locked and curr_radius > 1e-3:
            desired_offset = _normalize_angle_deg(float(heading_out - yaw_deg - 180.0))
            out["yaw_offset_deg"] = _snap_numeric_with_spec(
                desired_offset,
                dict(depart_spec.get("yaw_offset_deg", {}) or {}),
            )
            out["adaptive_heading_deg"] = float(heading_out)
        if "travel_distance_m" not in locked:
            dist_spec = dict(depart_spec.get("travel_distance_m", {}) or {})
            delta = max(12.0, min(40.0, curr_radius * 0.45))
            desired_dist = curr_radius + delta
            out["travel_distance_m"] = _snap_numeric_with_spec(
                desired_dist if desired_dist > 1e-3 else float(dist_spec.get("default", 30.0) or 30.0),
                dist_spec,
            )
        out.setdefault("adaptive_start_from_current", True)
        out.setdefault("start_radius_m", max(6.0, curr_radius))
    return out


def _is_obstacle_sensitive_element(element_key: str) -> bool:
    return str(element_key or "") in {
        "circular_orbit",
        "spiral_orbit",
        "square_orbit",
        "triangular_orbit",
        "figure8_orbit",
        "comet",
        "surface_mapping",
    }


def _step_target_altitude_m_from_params(
    *,
    element_key: str,
    landmark: dict[str, Any],
    params: dict[str, Any],
) -> float | None:
    bbox = _to_bbox_list(landmark.get("bbox_3d", None))
    center = np.asarray(list(landmark.get("center_3d", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])[:3], dtype=np.float32)
    sz = float(bbox[5]) if len(bbox) > 5 else 3.0
    adaptive_altitude = params.get("adaptive_altitude_m", None)
    if adaptive_altitude is None:
        base_alt = float(center[2] + 0.8 * sz + float(params.get("altitude_offset_m", 0.0) or 0.0))
    else:
        base_alt = float(adaptive_altitude)
    if str(element_key or "") == "surface_mapping":
        return float(base_alt + float(params.get("altitude_offset_m", 0.0) or 0.0))
    return float(base_alt)


def _step_target_altitude_m(
    *,
    step: dict[str, Any] | None,
    landmark: dict[str, Any],
) -> float | None:
    if not isinstance(step, dict):
        return None
    element_key = str(step.get("element_key", step.get("element_class", "")) or "")
    params = dict(step.get("base_params", step.get("params", {})) or {})
    if not element_key:
        return None
    return _step_target_altitude_m_from_params(
        element_key=element_key,
        landmark=landmark,
        params=params,
    )


def _crop_obstacles_for_param_search(
    *,
    obstacles_xyz: np.ndarray,
    landmark: dict[str, Any],
    current_pos: np.ndarray,
) -> np.ndarray:
    if not isinstance(obstacles_xyz, np.ndarray) or obstacles_xyz.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    center = np.asarray(list(landmark.get("center_3d", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])[:3], dtype=np.float32)
    p0 = current_pos.astype(np.float32)
    pts = np.vstack([center.reshape(1, 3), p0.reshape(1, 3)]).astype(np.float32)
    min_x = float(np.min(pts[:, 0])) - 70.0
    max_x = float(np.max(pts[:, 0])) + 70.0
    min_y = float(np.min(pts[:, 1])) - 70.0
    max_y = float(np.max(pts[:, 1])) + 70.0
    min_z = float(np.min(pts[:, 2])) - 35.0
    max_z = float(np.max(pts[:, 2])) + 55.0
    mask = (
        (obstacles_xyz[:, 0] >= min_x)
        & (obstacles_xyz[:, 0] <= max_x)
        & (obstacles_xyz[:, 1] >= min_y)
        & (obstacles_xyz[:, 1] <= max_y)
        & (obstacles_xyz[:, 2] >= min_z)
        & (obstacles_xyz[:, 2] <= max_z)
    )
    cropped = obstacles_xyz[mask]
    if cropped.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if cropped.shape[0] > 25000:
        cropped, _ = _sample_points(cropped.astype(np.float32), max_points=25000)
    return cropped.astype(np.float32)


def _estimate_element_terminal_pos(
    *,
    element_instance: dict[str, Any],
    landmark: dict[str, Any],
    start_pos: np.ndarray,
    preview_points: int,
) -> np.ndarray | None:
    try:
        pts, _ = sample_waypoints(
            element_instance=element_instance,
            landmark=landmark,
            start_pos=start_pos,
            num_points=max(24, int(preview_points)),
        )
    except Exception:
        return None
    if not isinstance(pts, np.ndarray) or pts.size == 0:
        return None
    return pts[-1].astype(np.float32)


def _attach_element_target_metadata(
    *,
    element_instance: dict[str, Any],
    landmark: dict[str, Any],
) -> None:
    element_instance["target_center_3d"] = list(landmark.get("center_3d", []) or [])
    element_instance["target_bbox_3d"] = list(landmark.get("bbox_3d", []) or [])
    element_instance["target_category"] = _landmark_category(landmark)
    element_instance["target_description"] = _landmark_description(landmark)


def _apply_obstacle_aware_param_hints(
    *,
    element_key: str,
    landmark: dict[str, Any],
    current_pos: np.ndarray,
    explicit_params: dict[str, Any],
    locked_param_keys: set[str] | None,
    element_instance_id: str,
    seed: int,
    target_binding: str,
    class_instance_index: int,
    obstacles_xyz: np.ndarray | None,
    keepout_boxes: list[dict[str, Any]] | None,
    safety_distance_m: float,
    preview_points: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = dict(explicit_params or {})
    locked = set(str(x) for x in list(locked_param_keys or set()))
    if not _is_obstacle_sensitive_element(element_key):
        return out, {"checked": False, "reason": "element_not_supported"}
    if not isinstance(obstacles_xyz, np.ndarray) or obstacles_xyz.size == 0:
        return out, {"checked": False, "reason": "no_obstacles"}

    local_obs = _crop_obstacles_for_param_search(
        obstacles_xyz=obstacles_xyz,
        landmark=landmark,
        current_pos=current_pos,
    )
    if local_obs.size == 0:
        return out, {"checked": False, "reason": "no_local_obstacles"}

    params_spec = dict((ELEMENT_LIBRARY.get(element_key, {}) or {}).get("params", {}) or {})
    radius_deltas = [0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0]
    altitude_deltas = [0.0, 2.0, -2.0, 4.0, -4.0, 6.0, -6.0, 8.0, -8.0]
    scan_deltas = [0.0, -2.0, 2.0, -4.0, 4.0]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _register_candidate(candidate: dict[str, Any], *, cost: float) -> None:
        normalized = dict(candidate)
        for key, spec in params_spec.items():
            if key not in normalized:
                continue
            if spec.get("choices"):
                continue
            try:
                normalized[key] = _snap_numeric_with_spec(float(normalized[key]), dict(spec or {}))
            except Exception:
                pass
        if element_key == "spiral_orbit":
            start_ext = float(normalized.get("start_extension_m", 10.0) or 10.0)
            end_ext = float(normalized.get("end_extension_m", 24.0) or 24.0)
            normalized["end_extension_m"] = max(end_ext, start_ext + 2.0)
        key = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        candidates.append({"params": normalized, "cost": float(cost)})

    if element_key in {"circular_orbit", "square_orbit", "triangular_orbit", "figure8_orbit", "comet"}:
        for radius_delta in radius_deltas:
            for altitude_delta in altitude_deltas:
                candidate = dict(out)
                if "extension_m" in params_spec and "extension_m" not in locked:
                    candidate["extension_m"] = float(candidate.get("extension_m", params_spec["extension_m"].get("default", 12.0) or 12.0)) + float(radius_delta)
                if "altitude_offset_m" in params_spec and "altitude_offset_m" not in locked:
                    candidate["altitude_offset_m"] = float(candidate.get("altitude_offset_m", params_spec["altitude_offset_m"].get("default", 8.0) or 8.0)) + float(altitude_delta)
                _register_candidate(candidate, cost=abs(radius_delta) + abs(altitude_delta))
    elif element_key == "spiral_orbit":
        for radius_delta in radius_deltas:
            for altitude_delta in altitude_deltas:
                candidate = dict(out)
                if "start_extension_m" in params_spec and "start_extension_m" not in locked:
                    candidate["start_extension_m"] = float(candidate.get("start_extension_m", params_spec["start_extension_m"].get("default", 10.0) or 10.0)) + float(radius_delta)
                if "end_extension_m" in params_spec and "end_extension_m" not in locked:
                    candidate["end_extension_m"] = float(candidate.get("end_extension_m", params_spec["end_extension_m"].get("default", 24.0) or 24.0)) + float(radius_delta)
                if "altitude_offset_m" in params_spec and "altitude_offset_m" not in locked:
                    candidate["altitude_offset_m"] = float(candidate.get("altitude_offset_m", params_spec["altitude_offset_m"].get("default", 8.0) or 8.0)) + float(altitude_delta)
                _register_candidate(candidate, cost=abs(radius_delta) + abs(altitude_delta))
    elif element_key == "surface_mapping":
        for scan_delta in scan_deltas:
            for altitude_delta in altitude_deltas:
                candidate = dict(out)
                if "extension_x_m" in params_spec and "extension_x_m" not in locked:
                    candidate["extension_x_m"] = float(candidate.get("extension_x_m", params_spec["extension_x_m"].get("default", 18.0) or 18.0)) + float(scan_delta)
                if "extension_y_m" in params_spec and "extension_y_m" not in locked:
                    candidate["extension_y_m"] = float(candidate.get("extension_y_m", params_spec["extension_y_m"].get("default", 18.0) or 18.0)) + float(scan_delta)
                if "altitude_offset_m" in params_spec and "altitude_offset_m" not in locked:
                    candidate["altitude_offset_m"] = float(candidate.get("altitude_offset_m", params_spec["altitude_offset_m"].get("default", 18.0) or 18.0)) + float(altitude_delta)
                _register_candidate(candidate, cost=abs(scan_delta) + abs(altitude_delta))
    else:
        _register_candidate(dict(out), cost=0.0)

    if not candidates:
        return out, {"checked": False, "reason": "no_candidate_variants"}

    evaluated: list[dict[str, Any]] = []
    for row in candidates:
        candidate_params = dict(row.get("params", {}) or {})
        try:
            candidate_instance = build_element_instance(
                element_key,
                landmark=landmark,
                element_instance_id=str(element_instance_id),
                seed=int(seed),
                explicit_params=candidate_params,
                target_binding=str(target_binding or "primary"),
                class_instance_index=int(class_instance_index),
            )
            pts, _ = sample_waypoints(
                element_instance=candidate_instance,
                landmark=landmark,
                start_pos=current_pos,
                num_points=max(24, int(preview_points)),
            )
            collision = check_collision(
                points=pts,
                obstacles_xyz=local_obs,
                safety_distance=float(safety_distance_m),
                keepout_boxes=list(keepout_boxes or []),
                keepout_margin_xy=max(0.1, float(safety_distance_m) * 0.25),
                keepout_margin_z=max(0.1, float(safety_distance_m) * 0.15),
            )
            evaluated.append(
                {
                    "params": candidate_params,
                    "cost": float(row.get("cost", 0.0) or 0.0),
                    "collision_free": bool(collision.get("collision_free", False)),
                    "min_distance": float(collision.get("min_distance", 0.0) or 0.0),
                    "violation_count": int(len(list(collision.get("violations", []) or []))),
                }
            )
        except Exception:
            continue

    if not evaluated:
        return out, {"checked": False, "reason": "candidate_evaluation_failed"}

    collision_free = [row for row in evaluated if bool(row.get("collision_free", False))]
    if collision_free:
        chosen = min(collision_free, key=lambda row: (float(row["cost"]), -float(row["min_distance"])))
    else:
        chosen = max(evaluated, key=lambda row: (float(row["min_distance"]), -float(row["cost"])))

    chosen_params = dict(chosen.get("params", {}) or {})
    changed = json.dumps(chosen_params, ensure_ascii=False, sort_keys=True) != json.dumps(out, ensure_ascii=False, sort_keys=True)
    return chosen_params, {
        "checked": True,
        "applied": bool(changed),
        "collision_free": bool(chosen.get("collision_free", False)),
        "min_distance": float(chosen.get("min_distance", 0.0) or 0.0),
        "violation_count": int(chosen.get("violation_count", 0) or 0),
        "candidate_count": int(len(evaluated)),
        "cost": float(chosen.get("cost", 0.0) or 0.0),
    }


def _extract_set_profile(raw_profiles: Any, set_key: str) -> dict[str, Any]:
    if not isinstance(raw_profiles, dict):
        return {}
    value = raw_profiles.get(str(set_key), None)
    return dict(value) if isinstance(value, dict) else {}


def _build_element_instances(
    *,
    set_spec: dict[str, Any],
    primary: dict[str, Any],
    secondary: list[dict[str, Any]],
    start_pos: np.ndarray,
    seed: int,
    generation_kind: str,
    explicit_sequence: list[str] | None = None,
    param_overrides: Any = None,
    auto_param_rules: Any = None,
    adaptive_sequential_params: bool = True,
    allow_interleave_repeat: bool = False,
    max_total_elements: int = 0,
    safety_distance_m: float = 2.0,
    obstacles_xyz: np.ndarray | None = None,
    keepout_boxes: list[dict[str, Any]] | None = None,
    preview_points_per_element: int = 40,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    targets = [primary] + list(secondary or [])
    raw_steps = []
    if explicit_sequence:
        raw_steps = [{"element_class": str(x).strip(), "params": {}, "target_binding": "primary"} for x in list(explicit_sequence or []) if str(x).strip()]
    elif isinstance(set_spec.get("element_steps", None), list) and list(set_spec.get("element_steps", []) or []):
        raw_steps = [dict(item) for item in list(set_spec.get("element_steps", []) or []) if isinstance(item, dict)]
    else:
        raw_steps = [{"element_class": str(x).strip(), "params": {}, "target_binding": "primary"} for x in list(set_spec.get("element_template", []) or []) if str(x).strip()]
    if generation_kind == "atomic-only" and not raw_steps:
        raw_steps = [{"element_class": "gradual_approach", "params": {}, "target_binding": "primary"}]
    if generation_kind == "composite-driven" and not raw_steps:
        raw_steps = [
            {"element_class": "gradual_approach", "params": {}, "target_binding": "primary"},
            {"element_class": "circular_orbit", "params": {}, "target_binding": "primary"},
            {"element_class": "gradual_depart", "params": {}, "target_binding": "primary"},
        ]
    element_classes = [str(item.get("element_class", "") or "").strip() for item in raw_steps if str(item.get("element_class", "") or "").strip()]

    landmarks_for_plan: list[dict[str, Any]]
    if generation_kind == "atomic-only":
        landmarks_for_plan = [primary]
    else:
        default_count = int(set_spec.get("landmark_count_default", len(targets) or 1) or 1)
        landmarks_for_plan = list(targets[: max(1, default_count)])
        if not landmarks_for_plan:
            landmarks_for_plan = [primary]
        if bool(allow_interleave_repeat) and len(landmarks_for_plan) > 1 and int(max_total_elements or 0) > 0 and not isinstance(set_spec.get("element_steps", None), list):
            per_landmark_cost = max(1, len(element_classes))
            max_landmark_slots = max(1, int(math.ceil(float(max_total_elements) / float(per_landmark_cost))))
            base_seq = list(landmarks_for_plan)
            append_idx = 0
            while len(landmarks_for_plan) < max_landmark_slots:
                landmarks_for_plan.append(dict(base_seq[append_idx % len(base_seq)]))
                append_idx += 1
        elif bool(set_spec.get("allow_revisit", False)) and len(landmarks_for_plan) > 1 and not isinstance(set_spec.get("element_steps", None), list):
            landmarks_for_plan.append(dict(landmarks_for_plan[0]))

    set_instance: dict[str, Any] | None = None
    if generation_kind == "composite-driven":
        set_instance = {
            "set_id": str(set_spec.get("set_key", _safe_name(str(set_spec.get("display_name", "set")) or "set"))),
            "set_name": str(set_spec.get("display_name", "") or "复合任务"),
            "set_scope": str(set_spec.get("scope", "single-landmark") or "single-landmark"),
            "landmark_order": [str(item.get("instance_id", "") or "") for item in landmarks_for_plan],
            "allow_revisit": bool(set_spec.get("allow_revisit", False)),
            "allow_interleave_repeat": bool(allow_interleave_repeat),
            "max_total_elements": int(max_total_elements or 0),
            "element_template": list(element_classes),
        }

    element_instances: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    step_start_positions: list[np.ndarray] = []
    rng = np.random.default_rng(int(seed) * 9973 + 17)
    current_pos = np.asarray(start_pos, dtype=np.float32).copy()
    counter = 0
    step_plan: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    if isinstance(set_spec.get("element_steps", None), list) and list(set_spec.get("element_steps", []) or []):
        step_defs = [dict(item) for item in list(set_spec.get("element_steps", []) or []) if isinstance(item, dict)]
        target_lookup = {str(item.get("instance_id", "") or ""): item for item in [primary, *list(secondary or [])]}
        for local_idx, step_def in enumerate(step_defs):
            target_id = str(step_def.get("target_instance_id", "") or "").strip()
            target_landmark = target_lookup.get(target_id, primary)
            step_plan.append((target_landmark, step_def, local_idx))
    else:
        for landmark_idx, landmark in enumerate(landmarks_for_plan):
            for local_idx, step_def in enumerate(raw_steps):
                step_plan.append((landmark, dict(step_def), local_idx))
                if generation_kind == "atomic-only":
                    break
            if generation_kind == "atomic-only":
                break

    prepared_steps: list[dict[str, Any]] = []
    for plan_idx, (landmark, step_def, local_idx) in enumerate(step_plan):
        element_key = str(step_def.get("element_class", "") or "").strip()
        if not element_key:
            continue
        prepared_steps.append(
            {
                "plan_index": int(plan_idx),
                "landmark": landmark,
                "step_def": dict(step_def),
                "local_idx": int(local_idx),
                "element_key": element_key,
                **(lambda built: {"base_params": built[0], "locked_param_keys": built[1]})(
                    _build_base_explicit_params(
                        element_key=element_key,
                        step_def=dict(step_def),
                        param_overrides=param_overrides,
                        auto_param_rules=auto_param_rules,
                        step_index=len(prepared_steps),
                        rng=rng,
                    )
                ),
            }
        )

    if bool(adaptive_sequential_params) and len(prepared_steps) >= 2:
        first_step = prepared_steps[0]
        second_step = prepared_steps[1]
        first_landmark = first_step.get("landmark", {}) or {}
        second_landmark = second_step.get("landmark", {}) or {}
        if (
            str(first_step.get("element_key", "") or "") == "gradual_approach"
            and _is_orbit_like_element(str(second_step.get("element_key", "") or ""))
            and str(first_landmark.get("instance_id", "") or "") == str(second_landmark.get("instance_id", "") or "")
        ):
            join_radius = _orbit_join_radius_from_step(
                step=second_step,
                landmark=dict(first_landmark),
                safety_distance_m=float(safety_distance_m),
            )
            if join_radius is not None:
                center = np.asarray(list((first_landmark or {}).get("center_3d", [0.0, 0.0, 0.0])[:3]), dtype=np.float32)
                vec_xy = current_pos[:2] - center[:2]
                curr_radius = float(np.linalg.norm(vec_xy))
                if curr_radius > 1e-3 and curr_radius < float(join_radius) + 8.0:
                    desired_radius = float(join_radius) + 8.0
                    direction_xy = (vec_xy / curr_radius).astype(np.float32)
                    current_pos[:2] = center[:2] + direction_xy * desired_radius

    for step_idx, prepared in enumerate(prepared_steps):
        landmark = prepared["landmark"]
        step_def = dict(prepared["step_def"])
        local_idx = int(prepared["local_idx"])
        element_key = str(prepared["element_key"])
        element_key = str(step_def.get("element_class", "") or "").strip()
        if not element_key:
            continue
        if int(max_total_elements or 0) > 0 and len(element_instances) >= int(max_total_elements):
            break
        counter += 1
        element_instance_id = f"ei_{counter:02d}_{_safe_name(element_key)}"
        step_start_pos = current_pos.astype(np.float32).copy()
        explicit_params: dict[str, Any] = dict(prepared.get("base_params", {}) or {})
        if bool(adaptive_sequential_params):
            explicit_params = _apply_adaptive_param_hints(
                element_key=element_key,
                landmark=landmark,
                current_pos=step_start_pos,
                explicit_params=explicit_params,
                locked_param_keys=set(prepared.get("locked_param_keys", set()) or set()),
                previous_step=prepared_steps[step_idx - 1] if step_idx > 0 else None,
                next_step=prepared_steps[step_idx + 1] if step_idx + 1 < len(prepared_steps) else None,
                safety_distance_m=float(safety_distance_m),
            )
        obstacle_adjustment: dict[str, Any] = {"checked": False, "reason": "not_needed"}
        if _is_obstacle_sensitive_element(element_key):
            explicit_params, obstacle_adjustment = _apply_obstacle_aware_param_hints(
                element_key=element_key,
                landmark=landmark,
                current_pos=step_start_pos,
                explicit_params=explicit_params,
                locked_param_keys=set(prepared.get("locked_param_keys", set()) or set()),
                element_instance_id=element_instance_id,
                seed=int(seed) + counter * 37,
                target_binding=str(step_def.get("target_binding", "primary") or "primary"),
                class_instance_index=local_idx,
                obstacles_xyz=obstacles_xyz,
                keepout_boxes=keepout_boxes,
                safety_distance_m=float(safety_distance_m),
                preview_points=int(preview_points_per_element),
            )
            if step_idx > 0:
                prev_prepared = prepared_steps[step_idx - 1]
                prev_landmark = prev_prepared.get("landmark", {}) or {}
                prev_element = element_instances[-1] if element_instances else None
                prev_key = str(prev_prepared.get("element_key", "") or "")
                same_landmark = str(prev_landmark.get("instance_id", "") or "") == str(landmark.get("instance_id", "") or "")
                if isinstance(prev_element, dict) and prev_key == "gradual_approach" and same_landmark and step_start_positions:
                    prev_start_pos = step_start_positions[-1].astype(np.float32).copy()
                    next_step_override = dict(prepared)
                    next_step_override["base_params"] = dict(explicit_params)
                    revised_prev_params = dict(prev_prepared.get("base_params", {}) or {})
                    revised_prev_params = _apply_adaptive_param_hints(
                        element_key=prev_key,
                        landmark=prev_landmark,
                        current_pos=prev_start_pos,
                        explicit_params=revised_prev_params,
                        locked_param_keys=set(prev_prepared.get("locked_param_keys", set()) or set()),
                        previous_step=prepared_steps[step_idx - 2] if step_idx > 1 else None,
                        next_step=next_step_override,
                        safety_distance_m=float(safety_distance_m),
                    )
                    revised_prev = build_element_instance(
                        prev_key,
                        landmark=prev_landmark,
                        element_instance_id=str(prev_element.get("element_instance_id", "") or f"ei_{counter-1:02d}_{_safe_name(prev_key)}"),
                        seed=int(prev_element.get("seed", int(seed) + max(1, counter - 1) * 37) or int(seed) + max(1, counter - 1) * 37),
                        explicit_params=revised_prev_params if revised_prev_params else None,
                        target_binding=str((prev_prepared.get("step_def", {}) or {}).get("target_binding", prev_element.get("target_binding", "primary")) or prev_element.get("target_binding", "primary") or "primary"),
                        class_instance_index=int(prev_prepared.get("local_idx", max(0, local_idx - 1)) or max(0, local_idx - 1)),
                    )
                    _attach_element_target_metadata(element_instance=revised_prev, landmark=prev_landmark)
                    if "obstacle_adaptation" in prev_element:
                        revised_prev["obstacle_adaptation"] = dict(prev_element.get("obstacle_adaptation", {}) or {})
                    element_instances[-1] = revised_prev
                    plan_rows[-1]["element_display_name"] = str(revised_prev.get("element_display_name", "") or plan_rows[-1].get("element_display_name", ""))
                    revised_prev_end = _estimate_element_terminal_pos(
                        element_instance=revised_prev,
                        landmark=prev_landmark,
                        start_pos=prev_start_pos,
                        preview_points=int(preview_points_per_element),
                    )
                    if isinstance(revised_prev_end, np.ndarray) and revised_prev_end.size >= 3:
                        step_start_pos = revised_prev_end.astype(np.float32)
                        current_pos = step_start_pos.astype(np.float32)
                        explicit_params, obstacle_adjustment = _apply_obstacle_aware_param_hints(
                            element_key=element_key,
                            landmark=landmark,
                            current_pos=step_start_pos,
                            explicit_params=explicit_params,
                            locked_param_keys=set(prepared.get("locked_param_keys", set()) or set()),
                            element_instance_id=element_instance_id,
                            seed=int(seed) + counter * 37,
                            target_binding=str(step_def.get("target_binding", "primary") or "primary"),
                            class_instance_index=local_idx,
                            obstacles_xyz=obstacles_xyz,
                            keepout_boxes=keepout_boxes,
                            safety_distance_m=float(safety_distance_m),
                            preview_points=int(preview_points_per_element),
                        )
        prepared["base_params"] = dict(explicit_params)
        element_instance = build_element_instance(
            element_key,
            landmark=landmark,
            element_instance_id=element_instance_id,
            seed=int(seed) + counter * 37,
            explicit_params=explicit_params if explicit_params else None,
            target_binding=str(step_def.get("target_binding", "primary") or "primary"),
            class_instance_index=local_idx,
        )
        if bool(obstacle_adjustment.get("checked", False)):
            element_instance["obstacle_adaptation"] = dict(obstacle_adjustment)
        try:
            est_end = _estimate_element_terminal_pos(
                element_instance=element_instance,
                landmark=landmark,
                start_pos=step_start_pos,
                preview_points=int(preview_points_per_element),
            )
            if isinstance(est_end, np.ndarray) and est_end.size >= 3:
                current_pos = est_end.astype(np.float32)
        except Exception:
            pass
        _attach_element_target_metadata(element_instance=element_instance, landmark=landmark)
        element_instances.append(element_instance)
        step_start_positions.append(step_start_pos.astype(np.float32))
        plan_rows.append(
            {
                "element_instance_id": element_instance_id,
                "element_class": element_key,
                "element_display_name": element_instance.get("element_display_name", ""),
                "target_instance_id": str(landmark.get("instance_id", "") or ""),
                "target_description": _landmark_description(landmark),
            }
        )
    if set_instance is not None:
        set_instance["landmark_order"] = [str(row.get("target_instance_id", "") or "") for row in plan_rows]
    return set_instance, element_instances, plan_rows


def _build_flight_description(
    *,
    set_spec: dict[str, Any],
    set_instance: dict[str, Any] | None,
    element_instances: list[dict[str, Any]],
    primary: dict[str, Any],
    secondary: list[dict[str, Any]],
) -> str:
    primary_desc = _landmark_description(primary)
    primary_cat = _landmark_category(primary)
    element_names = " -> ".join(str(item.get("element_display_name", item.get("element_class", ""))) for item in element_instances)
    if set_instance is None:
        return (
            f"本次 flight mission 以 {primary_cat} 为目标，目标描述为“{primary_desc}”。"
            f"任务仅包含一个 atomic instance：{element_names}。"
        )
    secondary_text = ""
    if secondary:
        secondary_text = "；次目标包括 " + "，".join(_landmark_description(item) for item in secondary[:4])
    return (
        f"本次 flight mission 使用 composite instance “{set_instance.get('set_name', '')}”，"
        f"主目标为“{primary_desc}”（{primary_cat}），atomic instances 序列为 {element_names}{secondary_text}。"
    )


def _estimate_scale_from_bbox(bbox_list: list[float]) -> float:
    sx = float(bbox_list[3]) if len(bbox_list) > 3 else 3.0
    sy = float(bbox_list[4]) if len(bbox_list) > 4 else 3.0
    sz = float(bbox_list[5]) if len(bbox_list) > 5 else 3.0
    d_box = max(1.0, float(math.sqrt(sx * sx + sy * sy + sz * sz)))
    return d_box


def _build_start_pos(center_3d: list[float], bbox_list: list[float]) -> np.ndarray:
    d_box = _estimate_scale_from_bbox(bbox_list)
    start = np.asarray([
        center_3d[0] + 2.8 * d_box,
        center_3d[1],
        center_3d[2] + 0.8 * d_box,
    ], dtype=np.float32)
    return start


def _parse_ascii_pcd_header(path: Path) -> tuple[list[str], int, int]:
    fields: list[str] = []
    data_line = -1
    points_total = 0
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for idx, line in enumerate(file):
            s = line.strip()
            if not s:
                continue
            u = s.upper()
            if u.startswith("FIELDS "):
                fields = s.split()[1:]
            elif u.startswith("POINTS "):
                try:
                    points_total = int(float(s.split()[1]))
                except Exception:
                    points_total = 0
            elif u.startswith("DATA "):
                if "ASCII" not in u:
                    raise ValueError(f"Only ASCII PCD supported: {path}")
                data_line = idx + 1
                break
    if data_line < 0 or not fields:
        raise ValueError(f"Invalid PCD header: {path}")
    return fields, data_line, max(0, points_total)


def _class_color_bgr(class_id: int) -> tuple[int, int, int]:
    if cv2 is None:
        base = int(abs(int(class_id)))
        return int((base * 53) % 255), int((base * 97) % 255), int((base * 193) % 255)
    hue = int((int(class_id) * 37) % 180)
    hsv = np.uint8([[[hue, 190, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _try_load_obstacles_semantic(
    scene_root: Path,
    scene_id: str,
    max_points: int = 220000,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config if isinstance(config, dict) else {}
    pcd_dir = scene_root / resolve_output_dir_name(cfg, key="stage1_dir", default="pcd_map")
    semantic_path = pcd_dir / f"{scene_id}.semantic_lidar.pcd"
    plain_path = pcd_dir / f"{scene_id}.pcd"
    source = semantic_path if semantic_path.exists() else plain_path
    if not source.exists():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    fields, data_line, points_total = _parse_ascii_pcd_header(source)
    field_idx = {name: i for i, name in enumerate(fields)}
    has_class = "class_id" in field_idx
    x_idx = field_idx.get("x", 0)
    y_idx = field_idx.get("y", 1)
    z_idx = field_idx.get("z", 2)
    class_idx = field_idx.get("class_id", -1)

    step = 1
    if points_total > max_points and max_points > 0:
        step = max(1, int(math.ceil(float(points_total) / float(max_points))))

    xyz_list: list[list[float]] = []
    color_list: list[list[int]] = []
    with source.open("r", encoding="utf-8", errors="ignore") as file:
        for i, line in enumerate(file):
            if i < data_line:
                continue
            row_i = i - data_line
            if step > 1 and (row_i % step) != 0:
                continue
            parts = line.strip().split()
            if len(parts) < len(fields):
                continue
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
            except Exception:
                continue
            xyz_list.append([x, y, z])

            if has_class and class_idx >= 0:
                try:
                    cid = int(float(parts[class_idx]))
                except Exception:
                    cid = 0
                color_list.append(list(_class_color_bgr(cid)))
            else:
                color_list.append([185, 185, 185])

    if not xyz_list:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(xyz_list, dtype=np.float32), np.asarray(color_list, dtype=np.uint8)


def _write_poses_csv(path: Path, poses: list[dict[str, float]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "t", "x", "y", "z", "yaw", "pitch", "roll"])
        for pose in poses:
            writer.writerow([
                int(pose.get("frame", 0.0)),
                f"{float(pose.get('t', 0.0)):.6f}",
                f"{float(pose.get('x', 0.0)):.6f}",
                f"{float(pose.get('y', 0.0)):.6f}",
                f"{float(pose.get('z', 0.0)):.6f}",
                f"{float(pose.get('yaw', 0.0)):.6f}",
                f"{float(pose.get('pitch', 0.0)):.6f}",
                f"{float(pose.get('roll', 0.0)):.6f}",
            ])


def _build_bridge_config_for_stage3(
    config: dict[str, Any],
    image_width_override: int | None = None,
    image_height_override: int | None = None,
) -> tuple[str, str, dict[str, Any]]:
    task_cfg = config.get("task", {}) or {}
    engine = str(task_cfg.get("engine", "airsim")).lower()
    engine_cfg = (config.get("engine_params", {}) or {}).get(engine, {}) or {}
    vehicle = str(engine_cfg.get("vehicle_name", "drone_1") or "drone_1")
    bridge_cfg = build_unified_bridge_config(
        config=config,
        engine=engine,
        vehicle_name=vehicle,
        image_width=int(image_width_override) if image_width_override is not None else None,
        image_height=int(image_height_override) if image_height_override is not None else None,
        default_width=4096,
        default_height=3072,
        default_fov=72.0,
    )
    bridge_cfg["camera_capture_image_types"] = [0]
    return engine, vehicle, bridge_cfg


def _parse_stage3_capture_vehicles(config: dict[str, Any], stage3_cfg: dict[str, Any], worker_count: int) -> list[str]:
    if worker_count <= 0:
        return []
    raw = stage3_cfg.get("final_capture_bindings", [])
    if not isinstance(raw, list) or len(raw) == 0:
        raw = ((config.get("parallel", {}) or {}).get("bindings", []) or [])

    vehicles: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            vehicle = str(item.get("vehicle", "") or "").strip()
            if vehicle:
                vehicles.append(vehicle)

    if not vehicles:
        vehicles = [f"drone_{idx + 1}" for idx in range(worker_count)]

    if len(vehicles) < worker_count:
        for idx in range(len(vehicles), worker_count):
            vehicles.append(f"drone_{idx + 1}")
    return vehicles[:worker_count]


def _split_contiguous_segments(total_count: int, worker_count: int) -> list[tuple[int, int]]:
    if total_count <= 0 or worker_count <= 0:
        return []
    worker_count = min(int(worker_count), int(total_count))
    base = total_count // worker_count
    rem = total_count % worker_count
    segments: list[tuple[int, int]] = []
    start = 0
    for wid in range(worker_count):
        size = base + (1 if wid < rem else 0)
        end = start + size
        if end > start:
            segments.append((start, end))
        start = end
    return segments


def _forward_to_yaw_pitch_deg(forward_vec: np.ndarray) -> tuple[float, float]:
    f = forward_vec.astype(np.float32)
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        return 0.0, 0.0
    f = f / n
    yaw = math.degrees(math.atan2(-float(f[1]), float(f[0])))
    pitch = math.degrees(math.atan2(float(f[2]), max(1e-6, float(np.linalg.norm(f[:2])))))
    return float(yaw), float(pitch)


def _center_crop_to_aspect(image_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        return np.zeros((max(1, int(target_h)), max(1, int(target_w)), 3), dtype=np.uint8)
    h, w = image_bgr.shape[:2]
    target_aspect = float(target_w) / float(target_h)
    src_aspect = float(w) / float(max(1, h))
    if src_aspect > target_aspect:
        crop_h = h
        crop_w = int(round(float(h) * target_aspect))
    else:
        crop_w = w
        crop_h = int(round(float(w) / target_aspect))
    crop_w = max(1, min(w, crop_w))
    crop_h = max(1, min(h, crop_h))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    cropped = image_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
        cropped = cv2.resize(cropped, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    return cropped


def _camera_axes_from_forward(forward_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = forward_vec.astype(np.float32)
    fn = float(np.linalg.norm(f))
    if fn < 1e-6:
        f = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        f = f / fn
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(f, world_up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / rn
    cam_up = np.cross(right, f)
    un = float(np.linalg.norm(cam_up))
    if un < 1e-6:
        cam_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        cam_up = cam_up / un
    return f, right, cam_up


def _project_target_bbox_2d(
    target_center_3d: list[float],
    target_bbox_list: list[float],
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> list[int] | None:
    corners = _target_bbox_corners_world(target_center_3d, target_bbox_list)
    rel = corners - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1e-3
    if not np.any(valid):
        return None
    x_cam = x_cam[valid]
    y_cam = y_cam[valid]
    z_cam = z_cam[valid]
    f = 0.5 * float(width) / max(1e-6, math.tan(math.radians(float(fov_deg)) * 0.5))
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    px = (x_cam / z_cam) * f + cx
    py = cy - (y_cam / z_cam) * f
    in_frame = (px >= 0.0) & (px < float(width)) & (py >= 0.0) & (py < float(height))
    if not np.any(in_frame):
        return None
    px = px[in_frame]
    py = py[in_frame]
    x0, y0 = int(np.floor(float(np.min(px)))), int(np.floor(float(np.min(py))))
    x1, y1 = int(np.ceil(float(np.max(px)))), int(np.ceil(float(np.max(py))))
    x0 = max(0, min(width - 1, x0))
    y0 = max(0, min(height - 1, y0))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    w = max(0, x1 - x0 + 1)
    h = max(0, y1 - y0 + 1)
    if w < 4 or h < 4:
        return None
    return [x0, y0, w, h]


def _sample_target_bbox_surface_points(center_3d: list[float], bbox_list: list[float], grid_n: int = 8) -> np.ndarray:
    sx = float(bbox_list[3]) if len(bbox_list) > 3 else 3.0
    sy = float(bbox_list[4]) if len(bbox_list) > 4 else 3.0
    sz = float(bbox_list[5]) if len(bbox_list) > 5 else 3.0
    yaw_deg = float(bbox_list[6]) if len(bbox_list) > 6 else 0.0
    hx, hy, hz = max(0.2, sx * 0.5), max(0.2, sy * 0.5), max(0.2, sz * 0.5)
    n = max(4, int(grid_n))
    ux = np.linspace(-hx, hx, num=n, endpoint=True, dtype=np.float32)
    uy = np.linspace(-hy, hy, num=n, endpoint=True, dtype=np.float32)
    uz = np.linspace(-hz, hz, num=n, endpoint=True, dtype=np.float32)
    points_local: list[list[float]] = []
    for x in ux:
        for y in uy:
            points_local.append([float(x), float(y), -hz])
            points_local.append([float(x), float(y), hz])
    for x in ux:
        for z in uz:
            points_local.append([float(x), -hy, float(z)])
            points_local.append([float(x), hy, float(z)])
    for y in uy:
        for z in uz:
            points_local.append([-hx, float(y), float(z)])
            points_local.append([hx, float(y), float(z)])

    arr = np.asarray(points_local, dtype=np.float32)
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    world = arr @ rot.T
    world[:, 0] += float(center_3d[0])
    world[:, 1] += float(center_3d[1])
    world[:, 2] += float(center_3d[2])
    return world.astype(np.float32)


def _project_points_for_visibility(
    points_xyz: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_xyz.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    rel = points_xyz - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid_z = z_cam > 1e-3
    if not np.any(valid_z):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    x_cam = x_cam[valid_z]
    y_cam = y_cam[valid_z]
    z_cam = z_cam[valid_z]
    f = 0.5 * float(width) / max(1e-6, math.tan(math.radians(float(fov_deg)) * 0.5))
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    px = np.rint((x_cam / z_cam) * f + cx).astype(np.int32)
    py = np.rint(cy - (y_cam / z_cam) * f).astype(np.int32)
    in_frame = (px >= 0) & (px < int(width)) & (py >= 0) & (py < int(height))
    if not np.any(in_frame):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    return px[in_frame], py[in_frame], z_cam[in_frame].astype(np.float32)


def _compute_target_visibility_and_mask(
    target_points_world: np.ndarray,
    obstacles_xyz: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    depth_margin_m: float,
    min_visible_points: int,
    min_visible_ratio: float,
) -> tuple[np.ndarray, float, int, int, list[int] | None]:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    tx, ty, tz = _project_points_for_visibility(
        points_xyz=target_points_world,
        eye=eye,
        right=right,
        cam_up=cam_up,
        forward=forward,
        width=width,
        height=height,
        fov_deg=fov_deg,
    )
    target_inframe = int(tx.shape[0])
    if target_inframe <= 0:
        return mask, 0.0, 0, 0, None

    ox, oy, oz = _project_points_for_visibility(
        points_xyz=obstacles_xyz,
        eye=eye,
        right=right,
        cam_up=cam_up,
        forward=forward,
        width=width,
        height=height,
        fov_deg=fov_deg,
    )
    depth_map = np.full((int(height), int(width)), np.float32(np.inf), dtype=np.float32)
    if ox.shape[0] > 0:
        np.minimum.at(depth_map, (oy, ox), oz)

    depth_samples = depth_map[ty, tx]
    visible_bool = np.isinf(depth_samples) | (tz <= (depth_samples + float(depth_margin_m)))
    vis_count = int(np.count_nonzero(visible_bool))
    vis_ratio = float(vis_count) / float(max(1, target_inframe))

    if vis_count <= 0:
        return mask, vis_ratio, vis_count, target_inframe, None

    vx = tx[visible_bool]
    vy = ty[visible_bool]
    pts = np.stack([vx, vy], axis=1).astype(np.int32)
    if pts.shape[0] >= 3:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        for p in pts:
            cv2.circle(mask, (int(p[0]), int(p[1])), 2, 255, -1, lineType=cv2.LINE_AA)

    if (vis_count < int(min_visible_points)) or (vis_ratio < float(min_visible_ratio)):
        return mask, vis_ratio, vis_count, target_inframe, None

    x0, y0 = int(np.min(vx)), int(np.min(vy))
    x1, y1 = int(np.max(vx)), int(np.max(vy))
    x0 = max(0, min(int(width) - 1, x0))
    y0 = max(0, min(int(height) - 1, y0))
    x1 = max(0, min(int(width) - 1, x1))
    y1 = max(0, min(int(height) - 1, y1))
    bw = max(0, x1 - x0 + 1)
    bh = max(0, y1 - y0 + 1)
    if bw < 4 or bh < 4:
        return mask, vis_ratio, vis_count, target_inframe, None
    return mask, vis_ratio, vis_count, target_inframe, [x0, y0, bw, bh]


def _frames_to_intervals(frame_ids: list[int], fps: float) -> tuple[list[dict[str, int]], list[dict[str, float]]]:
    if not frame_ids:
        return [], []
    sorted_ids = sorted(set(int(v) for v in frame_ids))
    ranges: list[tuple[int, int]] = []
    st = sorted_ids[0]
    prev = sorted_ids[0]
    for v in sorted_ids[1:]:
        if v == prev + 1:
            prev = v
            continue
        ranges.append((st, prev))
        st = v
        prev = v
    ranges.append((st, prev))

    by_frame = [{"start_frame": int(a), "end_frame": int(b)} for a, b in ranges]
    by_time = [
        {
            "start_sec": float(a) / float(max(1e-6, fps)),
            "end_sec": float(b) / float(max(1e-6, fps)),
        }
        for a, b in ranges
    ]
    return by_frame, by_time


def _build_target_specs_from_mission_meta(
    mission_meta: dict[str, Any] | None,
    *,
    fallback_target_center_3d: list[float],
    fallback_target_bbox_list: list[float],
    mask_grid_n: int,
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for row in list((mission_meta or {}).get("element_instances", []) or []):
        if not isinstance(row, dict):
            continue
        target_id = str(row.get("target_instance_id", "") or "").strip()
        center = list(row.get("target_center_3d", []) or [])
        bbox = list(row.get("target_bbox_3d", []) or [])
        if not target_id or len(center) < 3 or len(bbox) < 7:
            continue
        if target_id in specs:
            continue
        specs[target_id] = {
            "target_instance_id": target_id,
            "target_center_3d": [float(center[0]), float(center[1]), float(center[2])],
            "target_bbox_list": [float(v) for v in bbox[:7]],
            "target_category": str(row.get("target_category", "") or ""),
            "target_description": str(row.get("target_description", "") or ""),
        }
    if not specs:
        specs["primary"] = {
            "target_instance_id": "primary",
            "target_center_3d": [float(fallback_target_center_3d[0]), float(fallback_target_center_3d[1]), float(fallback_target_center_3d[2])],
            "target_bbox_list": [float(v) for v in list(fallback_target_bbox_list)[:7]],
            "target_category": "",
            "target_description": "",
        }
    for row in specs.values():
        row["surface_points_world"] = _sample_target_bbox_surface_points(
            center_3d=row["target_center_3d"],
            bbox_list=row["target_bbox_list"],
            grid_n=int(mask_grid_n),
        )
    return specs


def _build_raw_to_dense_index_map(raw_xyz: np.ndarray, dense_xyz: np.ndarray) -> list[int]:
    raw = np.asarray(raw_xyz, dtype=np.float32)
    dense = np.asarray(dense_xyz, dtype=np.float32)
    if raw.ndim != 2 or dense.ndim != 2 or raw.shape[0] <= 0 or dense.shape[0] <= 0:
        return []
    mapping: list[int] = []
    start = 0
    n_dense = int(dense.shape[0])
    for i in range(int(raw.shape[0])):
        if start >= n_dense:
            mapping.append(n_dense - 1)
            continue
        slice_xyz = dense[start:, :3]
        d2 = np.sum((slice_xyz - raw[i, :3]) ** 2, axis=1)
        rel = int(np.argmin(d2))
        idx = start + rel
        mapping.append(idx)
        start = idx
    return mapping


def _segment_waypoint_ranges(
    segments: list[dict[str, Any]],
    total_points: int,
    *,
    raw_waypoints_xyz: np.ndarray | None = None,
    dense_waypoints_xyz: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if total_points <= 0 or not segments:
        return []
    normalized_segments: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        if isinstance(seg, dict):
            normalized_segments.append(dict(seg))
        elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
            normalized_segments.append(
                {
                    "segment_id": f"seg_{idx:02d}",
                    "behavior_id": str(seg[0]),
                    "event_label": str(seg[0]),
                    "num_points": int(seg[1]) if str(seg[1]).strip() else 1,
                }
            )
    if not normalized_segments:
        return []
    has_raw_ranges = all(
        isinstance(seg.get("waypoint_start_idx", None), (int, float))
        and isinstance(seg.get("waypoint_end_idx", None), (int, float))
        for seg in normalized_segments
    )
    if has_raw_ranges:
        raw_to_dense: list[int] = []
        raw_arr = np.asarray(raw_waypoints_xyz, dtype=np.float32) if isinstance(raw_waypoints_xyz, np.ndarray) else None
        dense_arr = np.asarray(dense_waypoints_xyz, dtype=np.float32) if isinstance(dense_waypoints_xyz, np.ndarray) else None
        if isinstance(raw_arr, np.ndarray) and isinstance(dense_arr, np.ndarray) and raw_arr.ndim == 2 and dense_arr.ndim == 2 and raw_arr.shape[0] > 0 and dense_arr.shape[0] > 0:
            raw_to_dense = _build_raw_to_dense_index_map(raw_arr, dense_arr)
        use_raw_dense_map = bool(raw_to_dense)
        raw_last = max(int(seg.get("waypoint_end_idx", 0) or 0) for seg in normalized_segments)
        raw_last = max(1, raw_last)
        out: list[dict[str, Any]] = []
        prev_end = -1
        for idx, seg in enumerate(normalized_segments):
            raw_st = max(0, int(seg.get("waypoint_start_idx", 0) or 0))
            raw_ed = max(raw_st, int(seg.get("waypoint_end_idx", raw_st) or raw_st))
            if use_raw_dense_map:
                raw_st = min(raw_st, len(raw_to_dense) - 1)
                raw_ed = min(raw_ed, len(raw_to_dense) - 1)
                start_idx = int(raw_to_dense[raw_st])
                end_idx = int(raw_to_dense[raw_ed])
            else:
                start_idx = int(round(float(raw_st) / float(raw_last) * float(max(0, total_points - 1))))
                end_idx = int(round(float(raw_ed) / float(raw_last) * float(max(0, total_points - 1))))
            if idx == len(normalized_segments) - 1:
                end_idx = int(total_points - 1)
            if start_idx < 0:
                start_idx = 0
            if end_idx < start_idx:
                end_idx = start_idx
            if prev_end >= 0 and start_idx > prev_end + 1:
                start_idx = prev_end + 1
            row = dict(seg)
            row["segment_index"] = int(idx)
            row["behavior_id"] = str(seg.get("behavior_id", seg.get("behavior", "")) or "")
            row["waypoint_start_idx"] = int(start_idx)
            row["waypoint_end_idx"] = int(end_idx)
            out.append(row)
            prev_end = int(end_idx)
        if out:
            out[-1]["waypoint_end_idx"] = int(total_points - 1)
        return out
    raw_counts = [max(1, int(seg.get("num_points", 1) or 1)) for seg in normalized_segments]
    total_raw = max(1, sum(raw_counts))
    out: list[dict[str, Any]] = []
    cursor = 0
    for idx, (seg, count) in enumerate(zip(normalized_segments, raw_counts)):
        start_idx = cursor
        if idx == len(raw_counts) - 1:
            end_idx = total_points - 1
        else:
            frac = float(count) / float(total_raw)
            span = max(1, int(round(frac * float(total_points))))
            end_idx = min(total_points - 1, start_idx + span - 1)
        if end_idx < start_idx:
            end_idx = start_idx
        row = dict(seg)
        row["segment_index"] = int(idx)
        row["behavior_id"] = str(seg.get("behavior_id", seg.get("behavior", "")) or "")
        row["waypoint_start_idx"] = int(start_idx)
        row["waypoint_end_idx"] = int(end_idx)
        out.append(row)
        cursor = end_idx + 1
    if out:
        out[-1]["waypoint_end_idx"] = int(total_points - 1)
    return out


def _event_visibility_rows(
    *,
    segment_ranges: list[dict[str, Any]],
    sampled_idx: list[int],
    fps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not segment_ranges:
        return [], []
    event_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for seg in segment_ranges:
        st = int(seg.get("waypoint_start_idx", 0) or 0)
        ed = int(seg.get("waypoint_end_idx", st) or st)
        matched_frames = [out_fid for out_fid, src_idx in enumerate(sampled_idx) if st <= int(src_idx) <= ed]
        intervals_frame, intervals_sec = _frames_to_intervals(matched_frames, fps=float(fps))
        event_rows.append(
            {
                "event_id": str(seg.get("event_id", seg.get("segment_id", "")) or ""),
                "event_label": str(seg.get("event_label", seg.get("behavior_id", seg.get("behavior", ""))) or ""),
                "behavior_id": str(seg.get("behavior_id", seg.get("behavior", "")) or ""),
                "target_instance_id": str(seg.get("target_instance_id", "") or ""),
                "intervals_frame": intervals_frame,
                "intervals_sec": intervals_sec,
                "start_sec": float(intervals_sec[0]["start_sec"]) if intervals_sec else 0.0,
                "end_sec": float(intervals_sec[-1]["end_sec"]) if intervals_sec else 0.0,
            }
        )
        for out_fid in matched_frames:
            frame_rows.append(
                {
                    "frame": int(out_fid),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                    "event_id": str(seg.get("event_id", seg.get("segment_id", "")) or ""),
                    "event_label": str(seg.get("event_label", seg.get("behavior_id", seg.get("behavior", ""))) or ""),
                    "behavior_id": str(seg.get("behavior_id", seg.get("behavior", "")) or ""),
                    "target_instance_id": str(seg.get("target_instance_id", "") or ""),
                }
            )
    return event_rows, frame_rows


def _build_sampled_indices(
    point_count: int,
    *,
    source_fps: float,
    target_fps: float,
    frame_stride: int,
    speedup: float,
) -> list[int]:
    n = int(point_count)
    if n <= 0:
        return []
    stride = max(1, int(frame_stride))
    idx = list(range(0, n, stride))
    if not idx:
        idx = [0]
    if idx[-1] != (n - 1):
        idx.append(n - 1)

    sampled = idx
    spd = max(1.0, float(speedup))
    src_fps = max(1e-6, float(source_fps))
    tgt_fps = max(1e-6, float(target_fps))
    duration_sec = float(max(0, n - 1)) / src_fps
    target_count = max(2, int(round(duration_sec * (tgt_fps / spd))) + 1)
    target_count = min(len(idx), max(2, target_count))
    if target_count < len(idx):
        pos = np.linspace(0, len(idx) - 1, num=target_count, endpoint=True).astype(np.int64)
        out: list[int] = []
        seen: set[int] = set()
        for p in pos:
            v = int(idx[int(p)])
            if v not in seen:
                out.append(v)
                seen.add(v)
        if out and out[-1] != (n - 1):
            out.append(n - 1)
        sampled = out if out else idx

    sanitized: list[int] = []
    last = -1
    for v in sampled:
        vv = max(0, min(n - 1, int(v)))
        if vv > last:
            sanitized.append(vv)
            last = vv

    if not sanitized:
        sanitized = [0]
    if sanitized[0] != 0:
        sanitized.insert(0, 0)
    if sanitized[-1] != (n - 1):
        sanitized.append(n - 1)
    _assert_strictly_increasing_indices(sanitized, name="sampled_indices")
    return sanitized


def _effective_pose_fps(source_fps: float, samples_per_segment: int, *, smoothing_applied: bool) -> float:
    base = max(1e-6, float(source_fps))
    if not smoothing_applied:
        return float(base)
    return float(base * max(1, int(samples_per_segment)))


def _assert_strictly_increasing_indices(indices: list[int], name: str = "indices") -> None:
    if not indices:
        return
    prev = int(indices[0])
    for i in range(1, len(indices)):
        cur = int(indices[i])
        if cur <= prev:
            raise RuntimeError(f"{name}_not_strictly_increasing: pos={i}, prev={prev}, cur={cur}")
        prev = cur


def _build_smoothed_forward_vectors(points_xyz: np.ndarray, smooth_window: int = 7) -> np.ndarray:
    n = int(points_xyz.shape[0]) if points_xyz.ndim == 2 else 0
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n == 1:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    raw = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if i == 0:
            diff = points_xyz[1] - points_xyz[0]
        elif i == (n - 1):
            diff = points_xyz[n - 1] - points_xyz[n - 2]
        else:
            diff = points_xyz[i + 1] - points_xyz[i - 1]
        norm = float(np.linalg.norm(diff))
        if norm < 1e-6:
            raw[i] = raw[i - 1] if i > 0 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            raw[i] = (diff / norm).astype(np.float32)

    win = max(1, int(smooth_window))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return raw

    radius = win // 2
    out = np.zeros_like(raw)
    for i in range(n):
        st = max(0, i - radius)
        ed = min(n, i + radius + 1)
        vec = np.mean(raw[st:ed], axis=0)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            out[i] = raw[i]
        else:
            out[i] = (vec / norm).astype(np.float32)

    return out.astype(np.float32)


def _resample_forward_vectors(vectors_xyz: np.ndarray, samples_per_segment: int) -> np.ndarray:
    arr = np.asarray(vectors_xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return arr.astype(np.float32)
    out = [arr[0]]
    k = max(1, int(samples_per_segment))
    for i in range(arr.shape[0] - 1):
        v0 = arr[i]
        v1 = arr[i + 1]
        for j in range(1, k + 1):
            t = float(j) / float(k)
            vec = ((1.0 - t) * v0 + t * v1).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            out.append((vec / norm).astype(np.float32) if norm > 1e-6 else out[-1])
    return np.asarray(out, dtype=np.float32)


def _smooth_unit_vectors(vectors: np.ndarray, smooth_window: int = 11) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = arr.copy()
    for i in range(out.shape[0]):
        norm = float(np.linalg.norm(out[i]))
        if norm < 1e-6:
            out[i] = out[i - 1] if i > 0 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            out[i] = (out[i] / norm).astype(np.float32)
    win = max(1, int(smooth_window))
    if win % 2 == 0:
        win += 1
    if win <= 1:
        return out.astype(np.float32)
    radius = win // 2
    smoothed = np.zeros_like(out)
    for i in range(out.shape[0]):
        st = max(0, i - radius)
        ed = min(out.shape[0], i + radius + 1)
        vec = np.mean(out[st:ed], axis=0)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            smoothed[i] = out[i]
        else:
            smoothed[i] = (vec / norm).astype(np.float32)
    return smoothed.astype(np.float32)


def _segment_for_source_idx(segments: list[dict[str, Any]] | None, src_idx: int) -> dict[str, Any] | None:
    for seg in list(segments or []):
        st = int(seg.get("waypoint_start_idx", -1) or -1)
        ed = int(seg.get("waypoint_end_idx", -1) or -1)
        if st <= int(src_idx) <= ed:
            return seg
    return None


def _video_trim_start_index(
    segments: list[dict[str, Any]] | None,
    *,
    total_points: int,
    raw_waypoints_xyz: np.ndarray | None = None,
    dense_waypoints_xyz: np.ndarray | None = None,
) -> int:
    rows = _segment_waypoint_ranges(
        list(segments or []),
        total_points=int(total_points),
        raw_waypoints_xyz=raw_waypoints_xyz,
        dense_waypoints_xyz=dense_waypoints_xyz,
    )
    if not rows:
        return 0
    first = rows[0]
    start_idx = max(0, int(first.get("waypoint_start_idx", 0) or 0))
    projected = bool(first.get("projected_start", False))
    if projected and start_idx > 0:
        return start_idx
    return max(0, start_idx)


def _camera_forward_for_segment(
    *,
    pos: np.ndarray,
    sampled_points: np.ndarray,
    out_fid: int,
    source_idx: int,
    segment: dict[str, Any] | None,
    segment_ranges: list[dict[str, Any]] | None,
    target_center_3d: list[float],
    target_center_lookup: dict[str, list[float]] | None = None,
    blend_window_points: int = 16,
) -> np.ndarray:
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-6:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return (arr / norm).astype(np.float32)

    def _normalize_camera_mode(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if text in {"track_target", "landmark_track"}:
            return "landmark_track"
        if text in {"velocity_aligned", "look_forward"}:
            return "look_forward"
        return "landmark_track"

    def _velocity_aligned_forward(gaze_pitch_deg: float) -> np.ndarray:
        if out_fid == 0 and sampled_points.shape[0] > 1:
            vec = sampled_points[1] - sampled_points[0]
        elif out_fid >= sampled_points.shape[0] - 1 and sampled_points.shape[0] > 1:
            vec = sampled_points[-1] - sampled_points[-2]
        else:
            vec = sampled_points[min(sampled_points.shape[0] - 1, out_fid + 1)] - sampled_points[max(0, out_fid - 1)]
        vec = _normalize(vec)
        xy_norm = float(np.linalg.norm(vec[:2]))
        if xy_norm < 1e-6:
            xy = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            xy = (vec[:2] / xy_norm).astype(np.float32)
        pitch = math.radians(gaze_pitch_deg)
        return _normalize(
            np.asarray(
                [
                    float(math.cos(pitch)) * float(xy[0]),
                    float(math.cos(pitch)) * float(xy[1]),
                    float(math.sin(pitch)),
                ],
                dtype=np.float32,
            )
        )

    def _base_forward(seg: dict[str, Any] | None, target_center: list[float]) -> np.ndarray:
        params = dict((seg or {}).get("params", {}) or {})
        camera_mode = _normalize_camera_mode(params.get("camera_mode", "landmark_track"))
        gaze_pitch_deg = float(params.get("gaze_pitch_deg", -15.0) or -15.0)
        if camera_mode == "landmark_track":
            target = np.asarray(target_center, dtype=np.float32)
            forward = target - pos
            forward = _normalize(forward)
            if abs(gaze_pitch_deg) > 1e-6:
                forward = forward.copy()
                forward[2] = np.clip(forward[2] + 0.25 * math.sin(math.radians(gaze_pitch_deg)), -0.98, 0.98)
                forward = _normalize(forward)
            return forward
        return _velocity_aligned_forward(gaze_pitch_deg)

    def _segment_mode(seg: dict[str, Any] | None) -> str:
        params = dict((seg or {}).get("params", {}) or {})
        return _normalize_camera_mode(params.get("camera_mode", "landmark_track"))

    current = _base_forward(segment, target_center_3d)
    rows = [dict(seg) for seg in list(segment_ranges or []) if isinstance(seg, dict)]
    if not rows:
        return current

    current_idx = None
    for idx, row in enumerate(rows):
        st = int(row.get("waypoint_start_idx", -1) or -1)
        ed = int(row.get("waypoint_end_idx", -1) or -1)
        if st <= int(source_idx) <= ed:
            current_idx = idx
            break
    if current_idx is None:
        return current

    def _target_center_for(seg: dict[str, Any]) -> list[float]:
        seg_target_id = str(seg.get("target_instance_id", "") or "")
        center = (target_center_lookup or {}).get(seg_target_id, None)
        if isinstance(center, list) and len(center) >= 3:
            return [float(center[0]), float(center[1]), float(center[2])]
        return list(target_center_3d)

    blended = current
    window = max(1, int(blend_window_points))
    row = rows[current_idx]
    start_idx = int(row.get("waypoint_start_idx", 0) or 0)
    end_idx = int(row.get("waypoint_end_idx", start_idx) or start_idx)
    current_mode = _segment_mode(row)
    current_target_id = str(row.get("target_instance_id", "") or "")
    if current_idx > 0 and int(source_idx) - start_idx < window:
        prev_row = rows[current_idx - 1]
        prev_target_id = str(prev_row.get("target_instance_id", "") or "")
        prev_mode = _segment_mode(prev_row)
        if (prev_target_id == current_target_id or not prev_target_id or not current_target_id) and prev_mode == current_mode:
            prev_forward = _base_forward(prev_row, _target_center_for(prev_row))
            alpha = max(0.0, min(1.0, float(int(source_idx) - start_idx) / float(window)))
            blended = _normalize((1.0 - alpha) * prev_forward + alpha * blended)
    if current_idx + 1 < len(rows) and end_idx - int(source_idx) < window:
        next_row = rows[current_idx + 1]
        next_target_id = str(next_row.get("target_instance_id", "") or "")
        next_mode = _segment_mode(next_row)
        if (next_target_id == current_target_id or not next_target_id or not current_target_id) and next_mode == current_mode:
            next_forward = _base_forward(next_row, _target_center_for(next_row))
            alpha = max(0.0, min(1.0, float(end_idx - int(source_idx)) / float(window)))
            blended = _normalize(alpha * blended + (1.0 - alpha) * next_forward)
    return blended


def _build_camera_forward_sequence(
    *,
    sampled_points: np.ndarray,
    sampled_idx: list[int],
    segment_ranges: list[dict[str, Any]] | None,
    target_center_3d: list[float],
    target_center_lookup: dict[str, list[float]] | None = None,
    blend_window_points: int = 16,
    smooth_window: int = 15,
) -> np.ndarray:
    n = int(sampled_points.shape[0]) if sampled_points.ndim == 2 else 0
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    raw = np.zeros((n, 3), dtype=np.float32)
    for out_fid in range(n):
        src_idx = int(sampled_idx[out_fid]) if out_fid < len(sampled_idx) else out_fid
        seg = _segment_for_source_idx(segment_ranges, src_idx)
        target_center_for_frame = list(target_center_3d)
        if isinstance(seg, dict):
            seg_target_id = str(seg.get("target_instance_id", "") or "")
            center = (target_center_lookup or {}).get(seg_target_id, None)
            if isinstance(center, list) and len(center) >= 3:
                target_center_for_frame = [float(center[0]), float(center[1]), float(center[2])]
        raw[out_fid] = _camera_forward_for_segment(
            pos=sampled_points[out_fid],
            sampled_points=sampled_points,
            out_fid=out_fid,
            source_idx=src_idx,
            segment=seg,
            segment_ranges=segment_ranges,
            target_center_3d=target_center_for_frame,
            target_center_lookup=target_center_lookup,
            blend_window_points=int(blend_window_points),
        )
    return _smooth_unit_vectors(raw, smooth_window=int(smooth_window))


def _analyze_final_task_visibility_metadata(
    *,
    config: dict[str, Any],
    scene_root: Path,
    scene_id: str,
    out_dir: Path,
    waypoints_xyz: np.ndarray,
    target_center_3d: list[float],
    target_bbox_list: list[float],
    mission_meta: dict[str, Any] | None,
    segments: list[dict[str, Any]] | None,
    source_pose_fps_override: float | None = None,
    waypoint_forwards: np.ndarray | None = None,
    raw_waypoints_xyz: np.ndarray | None = None,
) -> dict[str, Any]:
    if waypoints_xyz.size == 0:
        raise RuntimeError("empty waypoints for final task analysis")

    stage3_cfg = _stage3_cfg(config)
    source_pose_fps = float(source_pose_fps_override or (config.get("camera", {}) or {}).get("fps", 10.0) or 10.0)
    fps = float(stage3_cfg.get("final_video_fps", 5) or 5)
    out_w = int(stage3_cfg.get("final_video_width", 1280) or 1280)
    out_h = int(stage3_cfg.get("final_video_height", 720) or 720)
    frame_stride = max(1, int(stage3_cfg.get("final_video_frame_stride", 1) or 1))
    speedup = max(1.0, float(stage3_cfg.get("final_video_speedup", 1.0) or 1.0))
    mask_grid_n = max(4, int(stage3_cfg.get("final_visibility_grid_n", 8) or 8))
    min_visible_points = max(4, int(stage3_cfg.get("final_visibility_min_points", 16) or 16))
    min_visible_ratio = max(0.01, min(1.0, float(stage3_cfg.get("final_visibility_min_ratio", 0.08) or 0.08)))
    depth_margin_m = max(0.0, float(stage3_cfg.get("final_visibility_depth_margin_m", 0.2) or 0.2))
    obs_max_points = max(5000, int(stage3_cfg.get("final_visibility_obstacles_max_points", 70000) or 70000))
    camera_fov = float((config.get("camera", {}) or {}).get("fov", 72.0) or 72.0)

    final_dir = out_dir / "final_task"
    ensure_dir(final_dir)
    final_meta_path = final_dir / "task_data.json"
    final_index_map_path = final_dir / "frame_index_map.json"

    trim_start_idx = min(
        max(
            0,
            _video_trim_start_index(
                segments,
                total_points=int(waypoints_xyz.shape[0]),
                raw_waypoints_xyz=raw_waypoints_xyz,
                dense_waypoints_xyz=waypoints_xyz,
            ),
        ),
        max(0, int(waypoints_xyz.shape[0]) - 1),
    )
    sampled_idx = _build_sampled_indices(
        point_count=int(max(1, int(waypoints_xyz.shape[0]) - trim_start_idx)),
        source_fps=float(source_pose_fps),
        target_fps=float(fps),
        frame_stride=int(frame_stride),
        speedup=float(speedup),
    )
    sampled_idx = [int(trim_start_idx + idx) for idx in sampled_idx]
    _assert_strictly_increasing_indices(sampled_idx, name="final_sampled_indices_analysis")
    sampled_points = waypoints_xyz[np.asarray(sampled_idx, dtype=np.int64)].astype(np.float32)
    playback_duration_sec = float(max(0, len(sampled_idx) - 1)) / float(max(1.0, fps))
    source_duration_sec = float(max(0, sampled_idx[-1] - sampled_idx[0])) / float(max(1e-6, source_pose_fps)) if sampled_idx else 0.0
    normalized_segment_ranges = _segment_waypoint_ranges(
        list(segments or []),
        total_points=int(waypoints_xyz.shape[0]),
        raw_waypoints_xyz=raw_waypoints_xyz,
        dense_waypoints_xyz=waypoints_xyz,
    )

    target_center_lookup: dict[str, list[float]] = {}
    for row in list((mission_meta or {}).get("element_instances", []) or []):
        target_id = str(row.get("target_instance_id", "") or "")
        center_row = list(row.get("target_center_3d", []) or [])
        if target_id and len(center_row) >= 3:
            target_center_lookup[target_id] = [float(center_row[0]), float(center_row[1]), float(center_row[2])]

    camera_forward_window = max(5, int(stage3_cfg.get("final_camera_forward_smooth_window", max(9, int(round(float(fps) * 0.6))))) or max(9, int(round(float(fps) * 0.6))))
    if camera_forward_window % 2 == 0:
        camera_forward_window += 1
    transition_window = max(6, int(stage3_cfg.get("final_camera_transition_window_points", int(round(float(fps) * 0.4))) or int(round(float(fps) * 0.4))))
    if isinstance(waypoint_forwards, np.ndarray) and waypoint_forwards.ndim == 2 and waypoint_forwards.shape[0] == waypoints_xyz.shape[0]:
        camera_forwards = _smooth_unit_vectors(waypoint_forwards[np.asarray(sampled_idx, dtype=np.int64)].astype(np.float32), smooth_window=int(camera_forward_window))
    else:
        camera_forwards = _build_camera_forward_sequence(
            sampled_points=sampled_points,
            sampled_idx=sampled_idx,
            segment_ranges=normalized_segment_ranges,
            target_center_3d=target_center_3d,
            target_center_lookup=target_center_lookup,
            blend_window_points=int(transition_window),
            smooth_window=int(camera_forward_window),
        )

    obstacles_xyz, obstacles_bgr = _try_load_obstacles_semantic(
        scene_root=scene_root,
        scene_id=scene_id,
        max_points=max(200000, obs_max_points * 3),
        config=config,
    )
    obstacles_crop, _ = _crop_points_near_trajectory(
        obstacles_xyz=obstacles_xyz,
        obstacles_bgr=obstacles_bgr,
        path_xyz=waypoints_xyz,
    )
    if obstacles_crop.shape[0] > obs_max_points:
        obstacles_crop, _ = _sample_points(obstacles_crop, max_points=obs_max_points)
    obstacles_crop = obstacles_crop.astype(np.float32) if obstacles_crop.size > 0 else np.zeros((0, 3), dtype=np.float32)

    target_specs = _build_target_specs_from_mission_meta(
        mission_meta=mission_meta,
        fallback_target_center_3d=target_center_3d,
        fallback_target_bbox_list=target_bbox_list,
        mask_grid_n=int(mask_grid_n),
    )
    primary_target_id = next(iter(target_specs.keys()))
    target_surface_points = np.asarray(target_specs[primary_target_id]["surface_points_world"], dtype=np.float32)

    total = len(sampled_idx)
    bboxes_per_frame: list[dict[str, Any]] = []
    frame_bboxes_xyxy_norm: list[dict[str, Any]] = []
    frame_visibility_rows: list[dict[str, Any]] = []
    masks_per_frame: list[dict[str, Any]] = []
    visible_frames: list[int] = []

    for out_fid in range(total):
        src_idx = sampled_idx[out_fid]
        pos = waypoints_xyz[int(src_idx)].astype(np.float32)
        segment_meta = _segment_for_source_idx(normalized_segment_ranges, int(src_idx))
        seg_target_id = str(segment_meta.get("target_instance_id", "") or "") if isinstance(segment_meta, dict) else ""
        target_center_for_frame = list(target_center_3d)
        target_surface_points_for_frame = target_surface_points
        if isinstance(segment_meta, dict):
            target_spec = target_specs.get(seg_target_id, None)
            if isinstance(target_spec, dict):
                target_center_for_frame = list(target_spec.get("target_center_3d", target_center_3d) or target_center_3d)
                target_surface_points_for_frame = np.asarray(target_spec.get("surface_points_world", target_surface_points), dtype=np.float32)

        forward = camera_forwards[out_fid] if out_fid < camera_forwards.shape[0] else _camera_forward_for_segment(
            pos=pos,
            sampled_points=sampled_points,
            out_fid=out_fid,
            source_idx=int(src_idx),
            segment=segment_meta,
            segment_ranges=normalized_segment_ranges,
            target_center_3d=target_center_for_frame,
            target_center_lookup=target_center_lookup,
            blend_window_points=int(transition_window),
        )
        _, right, cam_up = _camera_axes_from_forward(forward)

        mask_img, vis_ratio, vis_count, target_inframe, bbox = _compute_target_visibility_and_mask(
            target_points_world=target_surface_points_for_frame,
            obstacles_xyz=obstacles_crop,
            eye=pos,
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=int(out_w),
            height=int(out_h),
            fov_deg=float(camera_fov),
            depth_margin_m=float(depth_margin_m),
            min_visible_points=int(min_visible_points),
            min_visible_ratio=float(min_visible_ratio),
        )
        vis_meta = {
            "visible_point_count": int(vis_count),
            "target_inframe_point_count": int(target_inframe),
            "visible_ratio": float(vis_ratio),
            "target_instance_id": str(seg_target_id or primary_target_id),
        }
        frame_visibility_rows.append(
            {
                "frame": int(out_fid),
                "time_sec": float(out_fid) / float(max(1.0, fps)),
                "target_instance_id": str(vis_meta.get("target_instance_id", "") or ""),
                "is_visible": bool(bbox is not None),
                "visible_ratio": float(vis_meta.get("visible_ratio", 0.0) or 0.0),
                "visible_point_count": int(vis_meta.get("visible_point_count", 0) or 0),
                "target_inframe_point_count": int(vis_meta.get("target_inframe_point_count", 0) or 0),
            }
        )
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            x1 = max(0, min(int(out_w - 1), x))
            y1 = max(0, min(int(out_h - 1), y))
            x2 = max(x1 + 1, min(int(out_w), x + w))
            y2 = max(y1 + 1, min(int(out_h), y + h))
            bbox_xyxy_norm = [
                float(x1) / float(max(1, int(out_w))),
                float(y1) / float(max(1, int(out_h))),
                float(x2) / float(max(1, int(out_w))),
                float(y2) / float(max(1, int(out_h))),
            ]
            bboxes_per_frame.append(
                {
                    "frame": int(out_fid),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                    "bbox_xywh": [x, y, w, h],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xyxy_norm": bbox_xyxy_norm,
                    "visible_ratio": float(vis_meta.get("visible_ratio", 0.0) or 0.0),
                    "visible_point_count": int(vis_meta.get("visible_point_count", 0) or 0),
                }
            )
            frame_bboxes_xyxy_norm.append(
                {
                    "frame": int(out_fid),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                    "bbox_xyxy_norm": bbox_xyxy_norm,
                }
            )
            visible_frames.append(int(out_fid))
        if isinstance(mask_img, np.ndarray) and mask_img.size > 0 and int(np.count_nonzero(mask_img)) > 0:
            masks_per_frame.append(
                {
                    "frame": int(out_fid),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                    "visible_ratio": float(vis_meta.get("visible_ratio", 0.0) or 0.0),
                    "visible_point_count": int(vis_meta.get("visible_point_count", 0) or 0),
                    "target_inframe_point_count": int(vis_meta.get("target_inframe_point_count", 0) or 0),
                }
            )

    write_json(
        final_index_map_path,
        {
            "fps": float(fps),
            "frame_count": int(total),
            "frames": [
                {
                    "out_frame": int(out_fid),
                    "source_idx": int(sampled_idx[out_fid]),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                }
                for out_fid in range(total)
            ],
        },
    )

    intervals_frame, intervals_sec = _frames_to_intervals(visible_frames, fps=float(fps))
    event_rows, frame_event_rows = _event_visibility_rows(
        segment_ranges=normalized_segment_ranges,
        sampled_idx=sampled_idx,
        fps=float(fps),
    )
    visible_duration_ratio = float(len(visible_frames)) / float(max(1, total))
    bbox_area_ratios: list[float] = []
    for row in frame_bboxes_xyxy_norm:
        bbox = list(row.get("bbox_xyxy_norm", []) or [])
        if len(bbox) == 4:
            bbox_area_ratios.append(max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1])))
    mean_visible_bbox_area_ratio = float(sum(bbox_area_ratios) / len(bbox_area_ratios)) if bbox_area_ratios else 0.0
    env_difficulty_score, env_difficulty_band = _compute_env_difficulty(
        duration_sec=float(max(0.0, len(sampled_idx) - 1) / max(1.0, fps)),
        visible_count=int(len(intervals_sec)),
        visible_duration_ratio=float(visible_duration_ratio),
        mean_visible_bbox_area_ratio=float(mean_visible_bbox_area_ratio),
    )
    self_track_meta = dict((mission_meta or {}).get("self_state", {}) or {})
    element_instances_meta = list((mission_meta or {}).get("element_instances", []) or [])
    target_presence_targets: dict[str, Any] = {}
    for target_id, spec in target_specs.items():
        frame_ids: list[int] = []
        target_bboxes: list[dict[str, Any]] = []
        for out_fid, vis_row in enumerate(frame_visibility_rows):
            if str((vis_row or {}).get("target_instance_id", "") or "") != str(target_id):
                continue
            bbox_row = next((row for row in frame_bboxes_xyxy_norm if int(row.get("frame", -1) or -1) == int(out_fid)), None)
            if bbox_row is None:
                continue
            frame_ids.append(int(out_fid))
            target_bboxes.append(dict(bbox_row))
        by_frame, by_sec = _frames_to_intervals(frame_ids, fps=float(fps))
        target_presence_targets[str(target_id)] = {
            "target_instance_id": str(target_id),
            "target_center_3d": list(spec.get("target_center_3d", []) or []),
            "target_bbox_list": list(spec.get("target_bbox_list", []) or []),
            "target_category": str(spec.get("target_category", "") or ""),
            "target_description": str(spec.get("target_description", "") or ""),
            "intervals_frame": by_frame,
            "intervals_sec": by_sec,
            "frame_bboxes_xyxy_norm": target_bboxes,
            "visible_frame_count": int(len(frame_ids)),
        }

    meta = {
        "task_schema_version": "stage3.v2",
        "scene_id": scene_id,
        "video": {
            "path": "",
            "path_marked": "",
            "path_web": "",
            "path_marked_web": "",
            "frames_manifest": "",
            "frame_index_map": str(final_index_map_path.as_posix()),
            "fps": float(fps),
            "source_pose_fps": float(source_pose_fps),
            "width": int(out_w),
            "height": int(out_h),
            "frame_count": int(len(sampled_idx)),
            "duration_sec": float(playback_duration_sec),
            "source_duration_sec": float(source_duration_sec),
            "bitrate": str(stage3_cfg.get("final_video_bitrate", "10M") or "10M"),
            "web_bitrate": str(stage3_cfg.get("web_video_bitrate", "2M") or "2M"),
            "speedup": float(speedup),
            "capture_parallel_workers": 0,
            "capture_vehicles": [],
            "generated_without_video": True,
            "clean_video_without_overlays": True,
            "overlay_show_frame_index": False,
            "overlay_show_pose_preview": False,
            "marked_overlay_show_frame_index": bool(stage3_cfg.get("final_overlay_show_frame_index", False)),
            "marked_overlay_show_pose_preview": bool(stage3_cfg.get("final_overlay_show_pose_preview", False)),
        },
        "target_presence": {
            "intervals_frame": intervals_frame,
            "intervals_sec": intervals_sec,
            "bboxes": bboxes_per_frame,
            "frame_visibility": frame_visibility_rows,
            "frame_bboxes_xyxy_norm": frame_bboxes_xyxy_norm,
            "keyframe_gt_dense": frame_bboxes_xyxy_norm,
            "masks": masks_per_frame,
            "mask_tensor_archive": "",
            "visible_frame_count": int(len(visible_frames)),
            "visibility_params": {
                "grid_n": int(mask_grid_n),
                "min_visible_points": int(min_visible_points),
                "min_visible_ratio": float(min_visible_ratio),
                "depth_margin_m": float(depth_margin_m),
                "obstacles_points": int(obstacles_crop.shape[0]),
                "generated_without_video": True,
            },
        },
        "target_presence_targets": target_presence_targets,
        "task_tracks": {
            "environmental_awareness": {
                "visible_count": int(len(intervals_sec)),
                "difficulty_score": float(env_difficulty_score),
                "difficulty_band": str(env_difficulty_band),
                "visible_duration_ratio": float(visible_duration_ratio),
                "mean_visible_bbox_area_ratio": float(mean_visible_bbox_area_ratio),
                "intervals_frame": intervals_frame,
                "intervals_sec": intervals_sec,
                "frame_visibility": frame_visibility_rows,
                "frame_bboxes_xyxy_norm": frame_bboxes_xyxy_norm,
                "keyframe_gt_dense": frame_bboxes_xyxy_norm,
                "visible_frame_count": int(len(visible_frames)),
                "targets": target_presence_targets,
            },
            "self_state_awareness": {
                "task_type": str((mission_meta or {}).get("task_type", "") or ""),
                "task_subtype": str((mission_meta or {}).get("task_subtype", "") or ""),
                "task_difficulty": str((mission_meta or {}).get("task_difficulty", "") or ""),
                "task_difficulty_score": float((mission_meta or {}).get("task_difficulty_score", 0.0) or 0.0),
                "set_instance": dict((mission_meta or {}).get("set_instance", {}) or {}),
                "element_instances": element_instances_meta,
                "landmark_order": list(self_track_meta.get("landmark_order", []) or []),
                "mode_sequence": list((mission_meta or {}).get("mode_sequence", []) or []),
                "event_sequence": list((mission_meta or {}).get("event_sequence", []) or []),
                "behavior_sequence": [str(row.get("behavior_id", row.get("behavior", "")) or "") for row in list(normalized_segment_ranges or [])],
                "behavior_intervals_sec": event_rows,
                "frame_behavior_labels": frame_event_rows,
            },
        },
    }
    write_json(final_meta_path, meta)
    return {
        "final_meta_path": str(final_meta_path.as_posix()),
        "frame_index_map_path": str(final_index_map_path.as_posix()),
        "meta": meta,
    }


def _generate_final_task_video_and_data(
    out_dir: Path,
    scene_root: Path,
    scene_id: str,
    config: dict[str, Any],
    waypoints_xyz: np.ndarray,
    target_center_3d: list[float],
    target_bbox_list: list[float],
    segments: list[dict[str, Any]] | None = None,
    mission_meta: dict[str, Any] | None = None,
    progress_cb: Any = None,
    shared_runtime: dict[str, Any] | None = None,
    source_pose_fps_override: float | None = None,
    waypoint_forwards: np.ndarray | None = None,
    raw_waypoints_xyz: np.ndarray | None = None,
) -> dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("opencv(cv2) not available for final task video generation")
    if waypoints_xyz.size == 0:
        raise RuntimeError("empty waypoints for final task video")

    stage3_cfg = _stage3_cfg(config)
    source_pose_fps = float(source_pose_fps_override or (config.get("camera", {}) or {}).get("fps", 10.0) or 10.0)
    fps = float(stage3_cfg.get("final_video_fps", 5) or 5)
    final_video_bitrate = str(stage3_cfg.get("final_video_bitrate", "10M") or "10M").strip()
    web_video_bitrate = str(stage3_cfg.get("web_video_bitrate", "2M") or "2M").strip()
    save_marked_video = bool(stage3_cfg.get("final_save_marked_video", False))
    out_w = int(stage3_cfg.get("final_video_width", 1280) or 1280)
    out_h = int(stage3_cfg.get("final_video_height", 720) or 720)
    frame_stride = max(1, int(stage3_cfg.get("final_video_frame_stride", 1) or 1))
    speedup = max(1.0, float(stage3_cfg.get("final_video_speedup", 1.0) or 1.0))
    pose_smooth_window = max(1, int(stage3_cfg.get("final_pose_smooth_window", 7) or 7))
    settle_sec = max(0.0, float(stage3_cfg.get("final_capture_settle_sec", 0.03) or 0.03))
    mask_grid_n = max(4, int(stage3_cfg.get("final_visibility_grid_n", 8) or 8))
    min_visible_points = max(4, int(stage3_cfg.get("final_visibility_min_points", 16) or 16))
    min_visible_ratio = max(0.01, min(1.0, float(stage3_cfg.get("final_visibility_min_ratio", 0.08) or 0.08)))
    depth_margin_m = max(0.0, float(stage3_cfg.get("final_visibility_depth_margin_m", 0.2) or 0.2))
    obs_max_points = max(5000, int(stage3_cfg.get("final_visibility_obstacles_max_points", 70000) or 70000))
    capture_w = int(stage3_cfg.get("final_capture_width", (config.get("camera", {}) or {}).get("width", 4096)) or 4096)
    capture_h = int(stage3_cfg.get("final_capture_height", (config.get("camera", {}) or {}).get("height", 3072)) or 3072)
    capture_workers = int(stage3_cfg.get("final_capture_parallel_workers", 0) or 0)
    if capture_workers <= 0:
        capture_workers = max(1, int((config.get("parallel", {}) or {}).get("workers", 1) or 1))
    postprocess_workers = int(stage3_cfg.get("final_postprocess_parallel_workers", 0) or 0)
    if postprocess_workers <= 0:
        cpu_total = max(1, int(os.cpu_count() or 1))
        capped = max(1, int(max(1, cpu_total * 0.5)))
        postprocess_workers = max(1, min(8, max(1, int(capped * 0.6))))
    camera_fov = float((config.get("camera", {}) or {}).get("fov", 72.0) or 72.0)
    show_frame_index = bool(stage3_cfg.get("final_overlay_show_frame_index", False))
    show_pose_preview = bool(stage3_cfg.get("final_overlay_show_pose_preview", False))
    image_cfg = build_image_compression_cfg(stage3_cfg)

    preview_dir = out_dir / "preview"
    final_dir = out_dir / "final_task"
    frames_dir = final_dir / "frames"
    ensure_dir(preview_dir)
    ensure_dir(frames_dir)

    final_video_path = final_dir / "task_rgb.mp4"
    final_video_marked_path = final_dir / "task_rgb_marked.mp4"
    final_video_web_path = final_dir / "task_rgb_web.mp4"
    final_video_marked_web_path = final_dir / "task_rgb_marked_web.mp4"
    final_meta_path = final_dir / "task_data.json"
    final_frames_manifest_path = final_dir / "frames_manifest.json"
    final_masks_tensor_path = final_dir / "target_visibility_masks.npz"

    writer = None
    writer_marked = None

    final_index_map_path = final_dir / "frame_index_map.json"
    existing_plan = _load_existing_final_task_plan(
        final_meta_path=final_meta_path,
        final_index_map_path=final_index_map_path,
        waypoints_xyz=waypoints_xyz,
        default_source_pose_fps=float(source_pose_fps),
        default_speedup=float(speedup),
    )
    if existing_plan is not None:
        fps = float(existing_plan["fps"])
        source_pose_fps = float(existing_plan["source_pose_fps"])
        speedup = float(existing_plan["speedup"])
        sampled_idx = [int(v) for v in list(existing_plan["sampled_idx"]) or []]
        playback_duration_sec = float(existing_plan["playback_duration_sec"])
        source_duration_sec = float(existing_plan["source_duration_sec"])
    else:
        trim_start_idx = min(
            max(
                0,
                _video_trim_start_index(
                    segments,
                    total_points=int(waypoints_xyz.shape[0]),
                    raw_waypoints_xyz=raw_waypoints_xyz,
                    dense_waypoints_xyz=waypoints_xyz,
                ),
            ),
            max(0, int(waypoints_xyz.shape[0]) - 1),
        )
        sampled_idx = _build_sampled_indices(
            point_count=int(max(1, int(waypoints_xyz.shape[0]) - trim_start_idx)),
            source_fps=float(source_pose_fps),
            target_fps=float(fps),
            frame_stride=int(frame_stride),
            speedup=float(speedup),
        )
        sampled_idx = [int(trim_start_idx + idx) for idx in sampled_idx]
        _assert_strictly_increasing_indices(sampled_idx, name="final_sampled_indices")
        playback_duration_sec = float(max(0, len(sampled_idx) - 1)) / float(max(1.0, fps))
        source_duration_sec = float(max(0, sampled_idx[-1] - sampled_idx[0])) / float(max(1e-6, source_pose_fps)) if sampled_idx else 0.0
    sampled_points = waypoints_xyz[np.asarray(sampled_idx, dtype=np.int64)].astype(np.float32)
    normalized_segment_ranges = _segment_waypoint_ranges(
        list(segments or []),
        total_points=int(waypoints_xyz.shape[0]),
        raw_waypoints_xyz=raw_waypoints_xyz,
        dense_waypoints_xyz=waypoints_xyz,
    )
    target_center_lookup: dict[str, list[float]] = {}
    for row in list((mission_meta or {}).get("element_instances", []) or []):
        target_id = str(row.get("target_instance_id", "") or "")
        center_row = list(row.get("target_center_3d", []) or [])
        if target_id and len(center_row) >= 3:
            target_center_lookup[target_id] = [float(center_row[0]), float(center_row[1]), float(center_row[2])]
    camera_forward_window = max(5, int(stage3_cfg.get("final_camera_forward_smooth_window", max(9, int(round(float(fps) * 0.6))))) or max(9, int(round(float(fps) * 0.6))))
    if camera_forward_window % 2 == 0:
        camera_forward_window += 1
    transition_window = max(6, int(stage3_cfg.get("final_camera_transition_window_points", int(round(float(fps) * 0.4))) or int(round(float(fps) * 0.4))))
    if isinstance(waypoint_forwards, np.ndarray) and waypoint_forwards.ndim == 2 and waypoint_forwards.shape[0] == waypoints_xyz.shape[0]:
        camera_forwards = _smooth_unit_vectors(waypoint_forwards[np.asarray(sampled_idx, dtype=np.int64)].astype(np.float32), smooth_window=int(camera_forward_window))
    else:
        camera_forwards = _build_camera_forward_sequence(
            sampled_points=sampled_points,
            sampled_idx=sampled_idx,
            segment_ranges=normalized_segment_ranges,
            target_center_3d=target_center_3d,
            target_center_lookup=target_center_lookup,
            blend_window_points=int(transition_window),
            smooth_window=int(camera_forward_window),
        )

    total = len(sampled_idx)
    capture_workers = max(1, min(int(capture_workers), int(total)))
    vehicles = _parse_stage3_capture_vehicles(config=config, stage3_cfg=stage3_cfg, worker_count=capture_workers)
    worker_segments = _split_contiguous_segments(total_count=total, worker_count=capture_workers)
    engine, _, base_bridge_cfg = _build_bridge_config_for_stage3(
        config,
        image_width_override=int(capture_w),
        image_height_override=int(capture_h),
    )
    configured_port = int(base_bridge_cfg.get("sim_port", 41471))
    runtime_port = int(configured_port)
    bootstrap_bridge: Any | None = None
    persistent_bridge: Any | None = None
    launched_by_bridge = False
    owns_runtime = True

    if isinstance(shared_runtime, dict) and engine == "airsim":
        runtime_port = int(shared_runtime.get("runtime_port", configured_port) or configured_port)
        configured_port = int(shared_runtime.get("configured_port", configured_port) or configured_port)
        bootstrap_bridge = shared_runtime.get("bootstrap_bridge", None)
        persistent_bridge = shared_runtime.get("persistent_bridge", None)
        launched_by_bridge = bool(shared_runtime.get("launched_by_bridge", False))
        owns_runtime = False
    elif engine == "airsim":
        runtime_port, bootstrap_bridge, launched_by_bridge, configured_port = prepare_airsim_runtime_unified(
            config=config,
            scene_id=scene_id,
            base_bridge_cfg=base_bridge_cfg,
            vehicle_name=str(vehicles[0] if vehicles else base_bridge_cfg.get("vehicle_name", "drone_1")),
            vehicle_names=[str(v) for v in vehicles],
        )

    startup_port_msg = format_unified_startup_ports_message(
        stage="stage3",
        engine=engine,
        configured_sim_port=int(configured_port),
        runtime_sim_port=int(runtime_port),
        launched_by_bridge=bool(launched_by_bridge),
    )
    print(startup_port_msg)
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage3",
        step="final_task",
        message=startup_port_msg,
        payload=build_unified_stage_event(
            stage="stage3",
            step="final_task",
            scene_id=scene_id,
            engine=engine,
            status="ready",
            extra={
                "configured_sim_port": int(configured_port),
                "runtime_sim_port": int(runtime_port),
                "launched_by_bridge": bool(launched_by_bridge),
                "output_dir": str(out_dir.as_posix()),
            },
        ),
    )
    if callable(progress_cb):
        progress_cb(0, max(1, total), startup_port_msg)

    base_bridge_cfg["sim_port"] = int(runtime_port)
    base_bridge_cfg["launch_sim"] = False
    base_bridge_cfg["connect_on_init"] = True
    base_bridge_cfg["auto_select_port_on_conflict"] = False

    writer = cv2.VideoWriter(
        str(final_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1.0, fps)),
        (int(out_w), int(out_h)),
    )
    if not writer.isOpened():
        raise RuntimeError("failed_to_open_final_video_writer")

    if save_marked_video:
        writer_marked = cv2.VideoWriter(
            str(final_video_marked_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(max(1.0, fps)),
            (int(out_w), int(out_h)),
        )
        if not writer_marked.isOpened():
            writer.release()
            writer = None
            raise RuntimeError("failed_to_open_final_video_marked_writer")

    frame_images: list[np.ndarray | None] = [None] * total
    frame_archive_images: list[np.ndarray | None] = [None] * total
    frame_images_marked: list[np.ndarray | None] = [None] * total
    frame_boxes: list[list[int] | None] = [None] * total
    frame_masks: list[np.ndarray | None] = [None] * total
    frame_visibility: list[dict[str, Any] | None] = [None] * total
    done_counter = {"done": 0}
    counter_lock = threading.Lock()

    obstacles_xyz, obstacles_bgr = _try_load_obstacles_semantic(
        scene_root=scene_root,
        scene_id=scene_id,
        max_points=max(200000, obs_max_points * 3),
        config=config,
    )
    obstacles_crop, obstacles_crop_bgr = _crop_points_near_trajectory(
        obstacles_xyz=obstacles_xyz,
        obstacles_bgr=obstacles_bgr,
        path_xyz=waypoints_xyz,
    )
    if obstacles_crop.shape[0] > obs_max_points:
        obstacles_crop, _ = _sample_points(obstacles_crop, max_points=obs_max_points)
    obstacles_crop = obstacles_crop.astype(np.float32) if obstacles_crop.size > 0 else np.zeros((0, 3), dtype=np.float32)
    obstacles_crop_bgr = (
        obstacles_crop_bgr.astype(np.uint8)
        if isinstance(obstacles_crop_bgr, np.ndarray) and obstacles_crop_bgr.size > 0
        else np.zeros((0, 3), dtype=np.uint8)
    )

    target_specs = _build_target_specs_from_mission_meta(
        mission_meta=mission_meta,
        fallback_target_center_3d=target_center_3d,
        fallback_target_bbox_list=target_bbox_list,
        mask_grid_n=int(mask_grid_n),
    )
    primary_target_id = next(iter(target_specs.keys()))
    target_surface_points = np.asarray(target_specs[primary_target_id]["surface_points_world"], dtype=np.float32)

    def _capture_segment(worker_id: int, segment: tuple[int, int], vehicle_name: str) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray, list[int] | None, np.ndarray, dict[str, Any]]]:
        st, ed = int(segment[0]), int(segment[1])
        local_frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, list[int] | None, np.ndarray, dict[str, Any]]] = []
        owns_bridge = True
        if persistent_bridge is not None and int(capture_workers) == 1:
            bridge = persistent_bridge
            owns_bridge = False
        else:
            local_cfg = dict(base_bridge_cfg)
            local_cfg["vehicle_name"] = str(vehicle_name)
            bridge = create_bridge(engine=engine, scene_id=scene_id, config=local_cfg)
        try:
            for out_fid in range(st, ed):
                src_idx = sampled_idx[out_fid]
                pos = waypoints_xyz[int(src_idx)].astype(np.float32)
                segment_meta = _segment_for_source_idx(normalized_segment_ranges, int(src_idx))
                seg_target_id = str(segment_meta.get("target_instance_id", "") or "") if isinstance(segment_meta, dict) else ""
                target_center_for_frame = list(target_center_3d)
                target_bbox_for_frame = list(target_bbox_list)
                target_surface_points_for_frame = target_surface_points
                if isinstance(segment_meta, dict):
                    target_spec = target_specs.get(seg_target_id, None)
                    if isinstance(target_spec, dict):
                        target_center_for_frame = list(target_spec.get("target_center_3d", target_center_3d) or target_center_3d)
                        target_bbox_for_frame = list(target_spec.get("target_bbox_list", target_bbox_list) or target_bbox_list)
                        target_surface_points_for_frame = np.asarray(target_spec.get("surface_points_world", target_surface_points), dtype=np.float32)
                forward = camera_forwards[out_fid] if out_fid < camera_forwards.shape[0] else _camera_forward_for_segment(
                    pos=pos,
                    sampled_points=sampled_points,
                    out_fid=out_fid,
                    source_idx=int(src_idx),
                    segment=segment_meta,
                    segment_ranges=normalized_segment_ranges,
                    target_center_3d=target_center_for_frame,
                    target_center_lookup=target_center_lookup,
                    blend_window_points=int(transition_window),
                )
                forward, right, cam_up = _camera_axes_from_forward(forward)
                yaw_deg, pitch_deg = _forward_to_yaw_pitch_deg(forward)

                bridge.set_uav_pose(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                    yaw=float(yaw_deg),
                    pitch=float(pitch_deg),
                    roll=0.0,
                    vehicle_or_actor=str(vehicle_name),
                )
                if settle_sec > 0:
                    time.sleep(settle_sec)
                rgb = bridge.capture_rgb()
                rgb_np = np.asarray(rgb) if rgb is not None else np.empty((0, 0, 3), dtype=np.uint8)
                if rgb_np.ndim == 2:
                    rgb_np = cv2.cvtColor(rgb_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                if rgb_np.ndim != 3 or rgb_np.shape[2] < 3:
                    archive_canvas = np.zeros((int(capture_h), int(capture_w), 3), dtype=np.uint8)
                    canvas = np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)
                else:
                    archive_canvas = _center_crop_to_aspect(
                        rgb_np[:, :, :3].copy(),
                        target_w=int(capture_w),
                        target_h=int(capture_h),
                    )
                    canvas = _center_crop_to_aspect(
                        archive_canvas,
                        target_w=int(out_w),
                        target_h=int(out_h),
                    )
                canvas_marked = canvas.copy()

                mask_img, vis_ratio, vis_count, target_inframe, bbox = _compute_target_visibility_and_mask(
                    target_points_world=target_surface_points_for_frame,
                    obstacles_xyz=obstacles_crop,
                    eye=pos,
                    right=right,
                    cam_up=cam_up,
                    forward=forward,
                    width=int(out_w),
                    height=int(out_h),
                    fov_deg=float(camera_fov),
                    depth_margin_m=float(depth_margin_m),
                    min_visible_points=int(min_visible_points),
                    min_visible_ratio=float(min_visible_ratio),
                )
                if bbox is not None:
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(canvas_marked, (x, y), (x + w - 1, y + h - 1), (20, 20, 240), 2, lineType=cv2.LINE_AA)
                    bbox = [x, y, w, h]

                if show_pose_preview:
                    preview_panel = _draw_scene_image(
                        obstacles_xyz=obstacles_crop,
                        obstacles_bgr=obstacles_crop_bgr,
                        path_xyz=sampled_points,
                        target_center_3d=target_center_3d,
                        target_bbox_list=target_bbox_list,
                        az_deg=35.0,
                        el_deg=24.0,
                        width=max(360, int(round(float(out_w) * 0.34))),
                        height=int(out_h),
                        progress_ratio=float(out_fid + 1) / float(max(1, total)),
                        current_pose_xyz=pos,
                        current_forward_xyz=forward,
                        camera_fov_deg=float(camera_fov),
                    )
                    canvas_marked = _overlay_right_preview_panel(canvas_marked, preview_panel, title="POSE PREVIEW")

                if show_frame_index:
                    canvas_marked = _overlay_frame_index(canvas_marked, frame_id=int(out_fid), total=int(total), src_idx=int(src_idx))
                vis_meta = {
                    "visible_point_count": int(vis_count),
                    "target_inframe_point_count": int(target_inframe),
                    "visible_ratio": float(vis_ratio),
                    "target_instance_id": str(seg_target_id or primary_target_id),
                }
                local_frames.append((int(out_fid), archive_canvas, canvas, canvas_marked, bbox, mask_img, vis_meta))

                with counter_lock:
                    done_counter["done"] += 1
                    done = int(done_counter["done"])
                if callable(progress_cb):
                    progress_cb(done, total, f"captured frame {done}/{total} (worker={worker_id})")
            return local_frames
        finally:
            if owns_bridge:
                try:
                    bridge.shutdown()
                except Exception:
                    pass

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=capture_workers) as executor:
            futures: dict[concurrent.futures.Future[list[tuple[int, np.ndarray, np.ndarray, np.ndarray, list[int] | None, np.ndarray, dict[str, Any]]]], int] = {}
            for worker_id, segment in enumerate(worker_segments):
                fut = executor.submit(_capture_segment, worker_id, segment, vehicles[worker_id])
                futures[fut] = int(worker_id)

            merged_count = 0
            for fut in concurrent.futures.as_completed(futures):
                worker_id = int(futures[fut])
                local_frames = fut.result()
                for out_fid, archive_canvas, canvas, canvas_marked, bbox, mask_img, vis_meta in local_frames:
                    if out_fid < 0 or out_fid >= total:
                        raise RuntimeError(f"frame_out_of_range: out_fid={out_fid}, total={total}, worker={worker_id}")
                    if frame_images[out_fid] is not None:
                        raise RuntimeError(f"duplicate_frame_capture: out_fid={out_fid}, worker={worker_id}")
                    frame_archive_images[out_fid] = archive_canvas
                    frame_images[out_fid] = canvas
                    frame_images_marked[out_fid] = canvas_marked
                    frame_boxes[out_fid] = bbox
                    frame_masks[out_fid] = mask_img
                    frame_visibility[out_fid] = vis_meta
                    merged_count += 1

            validate_complete_indices(
                [idx for idx, img in enumerate(frame_images) if img is not None],
                total_count=int(total),
                name="stage3_capture_frames",
            )

        bboxes_per_frame: list[dict[str, Any]] = []
        frame_bboxes_xyxy_norm: list[dict[str, Any]] = []
        frame_visibility_rows: list[dict[str, Any]] = []
        masks_per_frame: list[dict[str, Any]] = []
        mask_frame_ids: list[int] = []
        mask_tensors: list[np.ndarray] = []
        visible_frames: list[int] = []
        frame_paths: list[str] = []
        frame_write_jobs: list[tuple[Path, np.ndarray]] = []
        for out_fid in range(total):
            canvas = frame_images[out_fid]
            if canvas is None:
                canvas = np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)
            archive_canvas = frame_archive_images[out_fid]
            if archive_canvas is None:
                archive_canvas = np.zeros((int(capture_h), int(capture_w), 3), dtype=np.uint8)
            canvas_marked = frame_images_marked[out_fid]
            if canvas_marked is None:
                canvas_marked = canvas.copy()
            bbox = frame_boxes[out_fid]
            vis_meta = frame_visibility[out_fid] if isinstance(frame_visibility[out_fid], dict) else {}
            frame_visibility_rows.append(
                {
                    "frame": int(out_fid),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                    "target_instance_id": str(vis_meta.get("target_instance_id", "") or ""),
                    "is_visible": bool(bbox is not None),
                    "visible_ratio": float(vis_meta.get("visible_ratio", 0.0) or 0.0),
                    "visible_point_count": int(vis_meta.get("visible_point_count", 0) or 0),
                    "target_inframe_point_count": int(vis_meta.get("target_inframe_point_count", 0) or 0),
                }
            )
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                x1 = max(0, min(int(out_w - 1), x))
                y1 = max(0, min(int(out_h - 1), y))
                x2 = max(x1 + 1, min(int(out_w), x + w))
                y2 = max(y1 + 1, min(int(out_h), y + h))
                bbox_xyxy_norm = [
                    float(x1) / float(max(1, int(out_w))),
                    float(y1) / float(max(1, int(out_h))),
                    float(x2) / float(max(1, int(out_w))),
                    float(y2) / float(max(1, int(out_h))),
                ]
                bboxes_per_frame.append(
                    {
                        "frame": int(out_fid),
                        "time_sec": float(out_fid) / float(max(1.0, fps)),
                        "bbox_xywh": [x, y, w, h],
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_xyxy_norm": bbox_xyxy_norm,
                        "visible_ratio": float(vis_meta.get("visible_ratio", 0.0) or 0.0),
                        "visible_point_count": int(vis_meta.get("visible_point_count", 0) or 0),
                    }
                )
                frame_bboxes_xyxy_norm.append(
                    {
                        "frame": int(out_fid),
                        "time_sec": float(out_fid) / float(max(1.0, fps)),
                        "bbox_xyxy_norm": bbox_xyxy_norm,
                    }
                )
                visible_frames.append(int(out_fid))

            mask_img = frame_masks[out_fid]
            if isinstance(mask_img, np.ndarray) and mask_img.size > 0 and int(np.count_nonzero(mask_img)) > 0:
                mask_frame_ids.append(int(out_fid))
                mask_tensors.append((mask_img > 0).astype(np.uint8))
                masks_per_frame.append(
                    {
                        "frame": int(out_fid),
                        "time_sec": float(out_fid) / float(max(1.0, fps)),
                        "visible_ratio": float(vis_meta.get("visible_ratio", 0.0) or 0.0),
                        "visible_point_count": int(vis_meta.get("visible_point_count", 0) or 0),
                        "target_inframe_point_count": int(vis_meta.get("target_inframe_point_count", 0) or 0),
                    }
                )

            frame_path = preferred_output_path(
                frames_dir / f"frame_{out_fid:06d}.jpg",
                compress_enabled=bool(image_cfg.get("enabled", True)),
            )
            frame_paths.append(str(frame_path.as_posix()))
            frame_write_jobs.append((frame_path, archive_canvas))
            writer.write(canvas)
            if writer_marked is not None:
                writer_marked.write(canvas_marked)

        def _write_frame_png(job: tuple[Path, np.ndarray]) -> None:
            path_obj, img = job
            save_bgr_image(img, path_obj, cfg=image_cfg)

        if postprocess_workers <= 1:
            for job in frame_write_jobs:
                _write_frame_png(job)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(postprocess_workers, len(frame_write_jobs)))) as executor:
                futures = [executor.submit(_write_frame_png, job) for job in frame_write_jobs]
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if writer_marked is not None:
            try:
                writer_marked.release()
            except Exception:
                pass
        if bootstrap_bridge is not None and owns_runtime:
            try:
                bootstrap_bridge.shutdown()
            except Exception:
                pass

    def _postprocess_video(src: Path, web_dst: Path) -> Path:
        _make_mp4_web_compatible(src, bitrate=final_video_bitrate)
        return _export_web_mp4_variant(src, web_dst, bitrate=str(web_video_bitrate)) or src

    if postprocess_workers <= 1:
        final_web_variant = _postprocess_video(final_video_path, final_video_web_path)
        final_marked_web_variant = _postprocess_video(final_video_marked_path, final_video_marked_web_path) if save_marked_video else None
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_plain = executor.submit(_postprocess_video, final_video_path, final_video_web_path)
            final_web_variant = fut_plain.result()
            final_marked_web_variant = None
            if save_marked_video:
                fut_marked = executor.submit(_postprocess_video, final_video_marked_path, final_video_marked_web_path)
                final_marked_web_variant = fut_marked.result()
    if not save_marked_video:
        for obsolete in [final_video_marked_path, final_video_marked_web_path]:
            if obsolete.exists():
                obsolete.unlink(missing_ok=True)
    if mask_tensors:
        np.savez_compressed(
            final_masks_tensor_path,
            frames=np.asarray(mask_frame_ids, dtype=np.int32),
            masks=np.stack(mask_tensors, axis=0).astype(np.uint8),
            fps=np.asarray([float(fps)], dtype=np.float32),
            frame_count=np.asarray([int(total)], dtype=np.int32),
            height=np.asarray([int(out_h)], dtype=np.int32),
            width=np.asarray([int(out_w)], dtype=np.int32),
        )
    _write_frame_manifest(
        final_frames_manifest_path,
        frames=frame_paths,
        fps=float(fps),
        width=int(capture_w),
        height=int(capture_h),
    )
    write_json(
        final_index_map_path,
        {
            "fps": float(fps),
            "frame_count": int(total),
            "frames": [
                {
                    "out_frame": int(out_fid),
                    "source_idx": int(sampled_idx[out_fid]),
                    "time_sec": float(out_fid) / float(max(1.0, fps)),
                }
                for out_fid in range(total)
            ],
        },
    )

    intervals_frame, intervals_sec = _frames_to_intervals(visible_frames, fps=float(fps))
    segment_ranges = normalized_segment_ranges
    event_rows, frame_event_rows = _event_visibility_rows(
        segment_ranges=segment_ranges,
        sampled_idx=sampled_idx,
        fps=float(fps),
    )
    visible_duration_ratio = 0.0
    if total > 0:
        visible_duration_ratio = float(len(visible_frames)) / float(total)
    bbox_area_ratios: list[float] = []
    for row in frame_bboxes_xyxy_norm:
        bbox = list(row.get("bbox_xyxy_norm", []) or [])
        if len(bbox) == 4:
            bbox_area_ratios.append(max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1])))
    mean_visible_bbox_area_ratio = float(sum(bbox_area_ratios) / len(bbox_area_ratios)) if bbox_area_ratios else 0.0
    env_difficulty_score, env_difficulty_band = _compute_env_difficulty(
        duration_sec=float(max(0.0, len(sampled_idx) - 1) / max(1.0, fps)),
        visible_count=int(len(intervals_sec)),
        visible_duration_ratio=float(visible_duration_ratio),
        mean_visible_bbox_area_ratio=float(mean_visible_bbox_area_ratio),
    )
    self_track_meta = dict((mission_meta or {}).get("self_state", {}) or {})
    element_instances_meta = list((mission_meta or {}).get("element_instances", []) or [])
    target_presence_targets: dict[str, Any] = {}
    for target_id, spec in target_specs.items():
        frame_ids: list[int] = []
        target_bboxes: list[dict[str, Any]] = []
        for out_fid, vis_row in enumerate(frame_visibility_rows):
            if str((vis_row or {}).get("target_instance_id", "") or "") != str(target_id):
                continue
            bbox_row = next((row for row in frame_bboxes_xyxy_norm if int(row.get("frame", -1) or -1) == int(out_fid)), None)
            if bbox_row is None:
                continue
            frame_ids.append(int(out_fid))
            target_bboxes.append(dict(bbox_row))
        by_frame, by_sec = _frames_to_intervals(frame_ids, fps=float(fps))
        target_presence_targets[str(target_id)] = {
            "target_instance_id": str(target_id),
            "target_center_3d": list(spec.get("target_center_3d", []) or []),
            "target_bbox_list": list(spec.get("target_bbox_list", []) or []),
            "target_category": str(spec.get("target_category", "") or ""),
            "target_description": str(spec.get("target_description", "") or ""),
            "intervals_frame": by_frame,
            "intervals_sec": by_sec,
            "frame_bboxes_xyxy_norm": target_bboxes,
            "visible_frame_count": int(len(frame_ids)),
        }

    existing_meta = dict((existing_plan or {}).get("meta", {}) or {}) if isinstance((existing_plan or {}).get("meta", {}), dict) else {}
    if existing_meta:
        meta = dict(existing_meta)
        video_meta = dict(meta.get("video", {}) or {})
        video_meta.update(
            {
                "path": str(final_video_path.as_posix()),
                "path_marked": str(final_video_marked_path.as_posix()) if save_marked_video else "",
                "path_web": str((final_web_variant or final_video_path).as_posix()),
                "path_marked_web": str((final_marked_web_variant or final_video_marked_path).as_posix()) if save_marked_video and final_marked_web_variant else "",
                "frames_manifest": str(final_frames_manifest_path.as_posix()),
                "frame_index_map": str(final_index_map_path.as_posix()),
                "fps": float(fps),
                "source_pose_fps": float(source_pose_fps),
                "width": int(out_w),
                "height": int(out_h),
                "frame_width": int(capture_w),
                "frame_height": int(capture_h),
                "frame_count": int(len(sampled_idx)),
                "duration_sec": float(playback_duration_sec),
                "source_duration_sec": float(source_duration_sec),
                "bitrate": str(final_video_bitrate),
                "web_bitrate": str(web_video_bitrate),
                "speedup": float(speedup),
                "pose_smooth_window": int(pose_smooth_window),
                "capture_width": int(capture_w),
                "capture_height": int(capture_h),
                "capture_parallel_workers": int(capture_workers),
                "capture_vehicles": [str(v) for v in vehicles],
                "runtime_sim_port": int(runtime_port),
                "scene_launched_by_bridge": bool(launched_by_bridge),
                "clean_video_without_overlays": True,
                "overlay_show_frame_index": False,
                "overlay_show_pose_preview": False,
                "marked_overlay_show_frame_index": bool(show_frame_index) if save_marked_video else False,
                "marked_overlay_show_pose_preview": bool(show_pose_preview) if save_marked_video else False,
            }
        )
        video_meta.pop("generated_without_video", None)
        meta["video"] = video_meta
        target_presence = dict(meta.get("target_presence", {}) or {})
        visibility_params = dict(target_presence.get("visibility_params", {}) or {})
        if "generated_without_video" in visibility_params:
            visibility_params["generated_without_video"] = False
            target_presence["visibility_params"] = visibility_params
        meta["target_presence"] = target_presence
    else:
        meta = {
        "task_schema_version": "stage3.v2",
        "scene_id": scene_id,
        "video": {
            "path": str(final_video_path.as_posix()),
            "path_marked": str(final_video_marked_path.as_posix()) if save_marked_video else "",
            "path_web": str((final_web_variant or final_video_path).as_posix()),
            "path_marked_web": str((final_marked_web_variant or final_video_marked_path).as_posix()) if save_marked_video and final_marked_web_variant else "",
            "frames_manifest": str(final_frames_manifest_path.as_posix()),
            "frame_index_map": str(final_index_map_path.as_posix()),
            "fps": float(fps),
            "source_pose_fps": float(source_pose_fps),
            "width": int(out_w),
            "height": int(out_h),
            "frame_width": int(capture_w),
            "frame_height": int(capture_h),
            "frame_count": int(len(sampled_idx)),
            "duration_sec": float(playback_duration_sec),
            "source_duration_sec": float(source_duration_sec),
            "bitrate": str(final_video_bitrate),
            "web_bitrate": str(web_video_bitrate),
            "speedup": float(speedup),
            "pose_smooth_window": int(pose_smooth_window),
            "capture_width": int(capture_w),
            "capture_height": int(capture_h),
            "capture_parallel_workers": int(capture_workers),
            "capture_vehicles": [str(v) for v in vehicles],
            "runtime_sim_port": int(runtime_port),
            "scene_launched_by_bridge": bool(launched_by_bridge),
            "clean_video_without_overlays": True,
            "overlay_show_frame_index": False,
            "overlay_show_pose_preview": False,
            "marked_overlay_show_frame_index": bool(show_frame_index) if save_marked_video else False,
            "marked_overlay_show_pose_preview": bool(show_pose_preview) if save_marked_video else False,
        },
        "target_presence": {
            "intervals_frame": intervals_frame,
            "intervals_sec": intervals_sec,
            "bboxes": bboxes_per_frame,
            "frame_visibility": frame_visibility_rows,
            "frame_bboxes_xyxy_norm": frame_bboxes_xyxy_norm,
            "keyframe_gt_dense": frame_bboxes_xyxy_norm,
            "masks": masks_per_frame,
            "mask_tensor_archive": str(final_masks_tensor_path.as_posix()) if mask_tensors else "",
            "visible_frame_count": int(len(visible_frames)),
            "visibility_params": {
                "grid_n": int(mask_grid_n),
                "min_visible_points": int(min_visible_points),
                "min_visible_ratio": float(min_visible_ratio),
                "depth_margin_m": float(depth_margin_m),
                "obstacles_points": int(obstacles_crop.shape[0]),
            },
        },
        "target_presence_targets": target_presence_targets,
        "task_tracks": {
            "environmental_awareness": {
                "visible_count": int(len(intervals_sec)),
                "difficulty_score": float(env_difficulty_score),
                "difficulty_band": str(env_difficulty_band),
                "visible_duration_ratio": float(visible_duration_ratio),
                "mean_visible_bbox_area_ratio": float(mean_visible_bbox_area_ratio),
                "intervals_frame": intervals_frame,
                "intervals_sec": intervals_sec,
                "frame_visibility": frame_visibility_rows,
                "frame_bboxes_xyxy_norm": frame_bboxes_xyxy_norm,
                "keyframe_gt_dense": frame_bboxes_xyxy_norm,
                "visible_frame_count": int(len(visible_frames)),
                "targets": target_presence_targets,
            },
            "self_state_awareness": {
                "task_type": str((mission_meta or {}).get("task_type", "") or ""),
                "task_subtype": str((mission_meta or {}).get("task_subtype", "") or ""),
                "task_difficulty": str((mission_meta or {}).get("task_difficulty", "") or ""),
                "task_difficulty_score": float((mission_meta or {}).get("task_difficulty_score", 0.0) or 0.0),
                "set_instance": dict((mission_meta or {}).get("set_instance", {}) or {}),
                "element_instances": element_instances_meta,
                "landmark_order": list(self_track_meta.get("landmark_order", []) or []),
                "mode_sequence": list((mission_meta or {}).get("mode_sequence", []) or []),
                "event_sequence": list((mission_meta or {}).get("event_sequence", []) or []),
                "behavior_sequence": [str(row.get("behavior_id", row.get("behavior", "")) or "") for row in list(segment_ranges or [])],
                "behavior_intervals_sec": event_rows,
                "frame_behavior_labels": frame_event_rows,
                "keyframe_gt_dense": [
                    {
                        "frame": int(row.get("frame", 0) or 0),
                        "time_sec": float(row.get("time_sec", 0.0) or 0.0),
                        "event_id": str(row.get("event_id", "") or ""),
                        "event_label": str(row.get("event_label", "") or ""),
                        "behavior_id": str(row.get("behavior_id", "") or ""),
                    }
                    for row in frame_event_rows
                    if int(row.get("frame", 0) or 0) % max(1, int(round(float(fps)))) == 0
                ],
                "behavior_vocab": [
                    {
                        "behavior_id": str(row.get("behavior_id", "") or ""),
                        "event_label": str(row.get("event_label", "") or ""),
                    }
                    for row in list((mission_meta or {}).get("event_sequence", []) or [])
                ],
            },
        },
        }
    write_json(final_meta_path, meta)

    return {
        "final_video": str(final_video_path.as_posix()),
        "final_video_marked": str(final_video_marked_path.as_posix()),
        "final_video_web": str((final_web_variant or final_video_path).as_posix()),
        "final_video_marked_web": str((final_marked_web_variant or final_video_marked_path).as_posix()),
        "final_metadata": str(final_meta_path.as_posix()),
        "final_frames_dir": str(frames_dir.as_posix()),
        "final_frames_manifest": str(final_frames_manifest_path.as_posix()),
        "final_frame_index_map": str(final_index_map_path.as_posix()),
        "final_masks_tensor": str(final_masks_tensor_path.as_posix()) if mask_tensors else "",
        "final_summary": {
            "visible_frame_count": int(len(visible_frames)),
            "intervals_frame": intervals_frame,
            "intervals_sec": intervals_sec,
            "frame_count": int(len(sampled_idx)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAV-DualCog Stage3 trajectory generation")
    parser.add_argument("--mode", type=str, default="generate_mission", choices=["generate", "generate_mission", "generate_dataset", "run_experiment", "record_scene_videos", "web"])
    parser.add_argument("--behavior-sequence", type=str, default=None)
    parser.add_argument("--mission-type", type=str, default=None)
    parser.add_argument("--generation-kind", type=str, default="auto", choices=["auto", "atomic-only", "composite-driven"])
    parser.add_argument("--mission-mode", type=str, default="single-landmark", choices=["single-landmark", "multi-landmark"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20262)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--landmark-id", type=str, default=None, help="instance_id from Stage2 outputs")
    parser.add_argument("--traj-id", type=str, default=None)
    parser.add_argument("--trajectory-root", type=str, default="scene_data")
    parser.add_argument("--instances-json", type=str, default=None, help="optional override source json path")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points-per-behavior", type=int, default=None)
    parser.add_argument("--samples-per-segment", type=int, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--forms", type=str, default=",".join(STAGE3_DEFAULT_TASKS))
    parser.add_argument("--task-group", type=str, default="all", choices=["all", "self-state", "environmental"])
    parser.add_argument("--approved-only", action="store_true", default=False)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--traj-ids", type=str, default=None, help="comma-separated traj ids for single-scene batch recording")
    parser.add_argument("--record-parallel-workers", type=int, default=1, help="parallel task recorders for single-scene batch recording")
    parser.add_argument("--record-reuse-worker-connections", action="store_true", default=False)
    parser.add_argument("--rerender-existing", action="store_true", default=False, help="force rerender even if final task video already exists")
    parser.add_argument(
        "--ignore-waypoint-forwards",
        action="store_true",
        default=False,
        help="ignore legacy forwards.npy and rebuild camera forward from segment camera_mode/targets",
    )
    parser.add_argument("--provide-flight-description", action="store_true", default=False)
    parser.add_argument("--include-keyframes", action="store_true", default=False)
    return parser.parse_args()




def generate_single(args: argparse.Namespace, config: dict[str, Any], logger: StageLogger, include_preview: bool = False) -> dict[str, Any]:
    t_start = time.time()
    scene_id = _resolve_scene_id(args, config)
    layout = _resolve_stage3_layout(config=config, scene_id=scene_id)
    scene_root = layout["scene_root"]
    stage3_root = layout["missions_root"]
    task_cfg = config.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim")).lower().strip()
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage3",
        step="generate",
        message="generate_started",
        payload=build_unified_stage_event(
            stage="stage3",
            step="generate",
            scene_id=scene_id,
            engine=engine_name,
            status="started",
            extra={"output_root": str(stage3_root.as_posix())},
        ),
    )

    source_name, source_path, source_instances = _resolve_source_instances(scene_root=scene_root, scene_id=scene_id, args=args, config=config)
    normalized_instances = [_normalize_landmark_item(item) for item in source_instances]
    landmark_lookup = {str(item.get("instance_id", "") or ""): item for item in normalized_instances}
    selected_instance_ids_raw = getattr(args, "selected_instance_ids", None)
    selected_instance_ids = [str(x).strip() for x in list(selected_instance_ids_raw or []) if str(x).strip()]
    if not selected_instance_ids and args.landmark_id:
        selected_instance_ids = [str(args.landmark_id).strip()]
    chosen_raw = None
    if selected_instance_ids:
        chosen_raw = next((item for item in source_instances if str(item.get("instance_id", "") or "") == selected_instance_ids[0]), None)
    if chosen_raw is None:
        chosen_raw = _pick_landmark(source_instances, landmark_id=args.landmark_id) if args.landmark_id else _auto_pick_landmark(source_instances)
    chosen = _normalize_landmark_item(chosen_raw)

    traj_cfg = _stage3_cfg(config)
    behavior_shared_cfg = _load_stage3_behavior_shared()
    camera_cfg = config.get("camera", {}) or {}
    mission_mode = str(args.mission_mode or "single-landmark")
    auto_set_candidates = [str(x).strip() for x in list(getattr(args, "auto_set_candidates", []) or []) if str(x).strip()]
    auto_set_rule = str(getattr(args, "auto_set_rule", "heuristic") or "heuristic").strip().lower()
    set_profiles = getattr(args, "set_profiles", None)
    if not isinstance(set_profiles, dict):
        set_profiles = _load_stage3_behavior_defaults()
    secondary_landmarks: list[dict[str, Any]] = []
    landmark_set_map = dict(getattr(args, "landmark_set_map", {}) or {}) if isinstance(getattr(args, "landmark_set_map", {}), dict) else {}
    if mission_mode == "multi-landmark":
        if len(selected_instance_ids) >= 2:
            explicit_secondary = []
            for candidate_id in selected_instance_ids[1:]:
                found = landmark_lookup.get(candidate_id, None)
                if found is not None:
                    explicit_secondary.append(found)
            max_secondary = int(traj_cfg.get("multi_landmark_max_secondary", 2) or 2)
            secondary_landmarks = explicit_secondary[: max(1, max_secondary)] if explicit_secondary else []
        if not secondary_landmarks:
            secondary_landmarks = _select_secondary_landmarks(
                instances=normalized_instances,
                primary=chosen,
                radius_m=float(traj_cfg.get("multi_landmark_radius_m", 90.0) or 90.0),
                max_secondary=int(traj_cfg.get("multi_landmark_max_secondary", 2) or 2),
            )
        selected_landmarks = [chosen, *list(secondary_landmarks or [])]
        set_spec, resolved_landmark_set_map = _build_multi_landmark_composite_set(
            selected_landmarks=selected_landmarks,
            landmark_set_map=landmark_set_map,
            allowed_set_types=auto_set_candidates or _single_landmark_component_set_keys(),
            auto_rule=auto_set_rule,
            seed=int(args.seed),
            explicit_multi_set_key=str(args.mission_type or "").strip() or None,
        )
        landmark_set_map = dict(resolved_landmark_set_map)
    else:
        set_spec = _select_set_template(
            landmark=chosen,
            set_type=args.mission_type,
            mode=mission_mode,
            allowed_set_types=auto_set_candidates,
            auto_rule=auto_set_rule,
            seed=int(args.seed),
        )
    set_key = str(set_spec.get("set_key", "") or "")
    set_profile = _extract_set_profile(set_profiles, set_key)

    center_3d = list(chosen.get("center_3d", [0.0, 0.0, 0.0]))
    bbox_list = list(chosen.get("bbox_3d", [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0]))
    start_pos = _build_start_pos(center_3d=center_3d, bbox_list=bbox_list)
    obstacles_xyz, obstacles_bgr = _try_load_obstacles_semantic(scene_root=scene_root, scene_id=scene_id, config=config)
    keepout_boxes = _build_keepout_boxes(
        normalized_instances,
        margin_xy=max(0.2, float(traj_cfg.get("safety_distance", behavior_shared_cfg.get("safety_distance_m", 2.0)) or behavior_shared_cfg.get("safety_distance_m", 2.0) or 2.0) * 0.25),
        margin_z=max(0.1, float(traj_cfg.get("safety_distance", behavior_shared_cfg.get("safety_distance_m", 2.0)) or behavior_shared_cfg.get("safety_distance_m", 2.0) or 2.0) * 0.15),
    )

    generation_kind = _resolve_generation_kind(args, set_spec)
    if str(getattr(args, "generation_kind", "auto") or "auto").strip().lower() == "auto":
        prof_kind = str(set_profile.get("generation_kind", "") or "").strip().lower()
        if prof_kind in {"auto", "atomic-only", "composite-driven"}:
            generation_kind = prof_kind if prof_kind != "auto" else generation_kind
    explicit_sequence = [b.strip() for b in str(args.behavior_sequence or "").split(",") if b.strip()] if args.behavior_sequence else None
    if not explicit_sequence:
        prof_seq = [str(x).strip() for x in list(set_profile.get("behavior_sequence", []) or []) if str(x).strip()]
        if prof_seq:
            explicit_sequence = prof_seq
    allow_interleave_repeat = bool(getattr(args, "allow_interleave_repeat", False))
    max_total_elements = int(getattr(args, "max_total_elements", 0) or 0)
    if "allow_interleave_repeat" in set_profile:
        allow_interleave_repeat = bool(set_profile.get("allow_interleave_repeat"))
    if "max_total_elements" in set_profile:
        max_total_elements = max(0, int(set_profile.get("max_total_elements", 0) or 0))
    set_instance, element_instances, plan_rows = _build_element_instances(
        set_spec=set_spec,
        primary=chosen,
        secondary=secondary_landmarks,
        start_pos=start_pos,
        seed=int(args.seed),
        generation_kind=generation_kind,
        explicit_sequence=explicit_sequence,
        param_overrides=set_profile.get("element_param_overrides", getattr(args, "element_param_overrides", None)),
        auto_param_rules=set_profile.get("element_auto_rules", getattr(args, "element_auto_rules", None)),
        adaptive_sequential_params=bool(getattr(args, "adaptive_sequential_params", True)),
        allow_interleave_repeat=allow_interleave_repeat,
        max_total_elements=max_total_elements,
        safety_distance_m=float(traj_cfg.get("safety_distance", behavior_shared_cfg.get("safety_distance_m", 2.0)) or behavior_shared_cfg.get("safety_distance_m", 2.0) or 2.0),
        obstacles_xyz=obstacles_xyz,
        keepout_boxes=keepout_boxes,
        preview_points_per_element=int(set_spec.get("preview_points_per_element", 40) or 40),
    )
    behavior_sequence = [str(item.get("element_class", "") or "") for item in element_instances]
    invalid = [b for b in behavior_sequence if b not in BEHAVIOR_SET]
    if invalid:
        raise ValueError(f"unsupported atomic classes in sequence: {invalid}")

    mode_sequence = [
        {
            "mode_key": str(item.get("element_class", "") or ""),
            "mode_name": str(item.get("element_display_name", "") or ""),
            "target_instance_id": str(item.get("target_instance_id", "") or ""),
        }
        for item in element_instances
    ]
    event_sequence = [
        {
            "event_id": str(item.get("element_instance_id", "") or ""),
            "event_label": str(item.get("element_display_name", "") or item.get("element_class", "")),
            "mode_key": str(item.get("element_class", "") or ""),
            "mode_name": str(item.get("element_display_name", "") or ""),
            "behavior_id": str(item.get("element_class", "") or ""),
            "target_instance_id": str(item.get("target_instance_id", "") or ""),
            "target_class_name": _landmark_category(landmark_lookup.get(str(item.get("target_instance_id", "") or ""), chosen)),
            "target_description": _landmark_description(landmark_lookup.get(str(item.get("target_instance_id", "") or ""), chosen)),
            "description": str(item.get("description", "") or ""),
            "params": dict(item.get("params", {}) or {}),
            "element_instance_id": str(item.get("element_instance_id", "") or ""),
            "element_class": str(item.get("element_class", "") or ""),
        }
        for item in element_instances
    ]

    points_per_behavior = int(args.points_per_behavior or traj_cfg.get("points_per_behavior", 60) or 60)
    samples_per_segment = int(args.samples_per_segment or traj_cfg.get("samples_per_segment", 4) or 4)
    smooth_window = int(args.smooth_window or traj_cfg.get("smooth_window", 5) or 5)
    fps = float(camera_cfg.get("fps", 10.0) or 10.0)

    limits = {
        "v_max": float(traj_cfg.get("v_max", 8.0) or 8.0),
        "a_max": float(traj_cfg.get("a_max", 10.0) or 10.0),
        "yaw_rate_max": float(traj_cfg.get("yaw_rate_max", 120.0) or 120.0),
    }
    safety_distance = float(traj_cfg.get("safety_distance", behavior_shared_cfg.get("safety_distance_m", 2.0)) or behavior_shared_cfg.get("safety_distance_m", 2.0) or 2.0)

    if args.traj_id:
        traj_id = str(args.traj_id)
    else:
        set_slug = _safe_name(str((set_instance or {}).get("set_name", set_spec.get("display_name", "mission"))) or "mission")
        traj_id = f"{set_slug}_{int(time.time())}"

    out_dir = stage3_root / traj_id
    ensure_dir(out_dir)

    raw_points, raw_forwards, segments = compose_trajectory(
        primary_landmark=chosen,
        element_instances=element_instances,
        landmark_lookup=landmark_lookup,
        start_pos=start_pos,
        points_per_element=points_per_behavior,
        pose_fps=float(fps),
        obstacles_xyz=obstacles_xyz,
        keepout_boxes=keepout_boxes,
        safety_distance=float(safety_distance),
        repair_max_lift_step_m=float(traj_cfg.get("repair_max_lift_step_m", 2.0) or 2.0),
        repair_max_total_lift_m=float(traj_cfg.get("repair_max_total_lift_m", 24.0) or 24.0),
        keepout_margin_xy=max(0.2, float(safety_distance) * 0.25),
        keepout_margin_z=max(0.1, float(safety_distance) * 0.15),
    )
    for idx, seg in enumerate(segments):
        if idx < len(element_instances):
            elem = element_instances[idx]
            seg["behavior_id"] = str(elem.get("element_class", "") or "")
            seg["event_id"] = str(elem.get("element_instance_id", seg["segment_id"]) or seg["segment_id"])
            seg["event_label"] = str(elem.get("element_display_name", seg["behavior_id"]) or seg["behavior_id"])
            seg["mode_key"] = str(elem.get("element_class", "") or "")
            seg["mode_name"] = str(elem.get("element_display_name", seg["behavior_id"]) or seg["behavior_id"])
            seg["target_instance_id"] = str(elem.get("target_instance_id", "") or "")
            seg["target_class_name"] = _landmark_category(landmark_lookup.get(seg["target_instance_id"], chosen))
            seg["target_description"] = _landmark_description(landmark_lookup.get(seg["target_instance_id"], chosen))
            seg["params"] = dict(elem.get("params", {}) or {})

    smoothing_applied = bool(raw_points.shape[0] > 3)
    effective_pose_fps = _effective_pose_fps(float(fps), int(samples_per_segment), smoothing_applied=smoothing_applied)
    if smoothing_applied:
        smooth_points = smooth_trajectory(
            points=raw_points,
            samples_per_segment=samples_per_segment,
            smooth_window=smooth_window,
        )
        smooth_forwards = _smooth_unit_vectors(
            _resample_forward_vectors(raw_forwards, samples_per_segment=samples_per_segment),
            smooth_window=max(3, int(smooth_window)),
        )
    else:
        smooth_points = raw_points.astype(np.float32)
        smooth_forwards = raw_forwards.astype(np.float32)
    if smooth_forwards.shape[0] != smooth_points.shape[0]:
        raise RuntimeError(f"forward_point_count_mismatch: points={smooth_points.shape[0]} forwards={smooth_forwards.shape[0]}")
    repair_summary = _repair_summary_from_segments(segments, waypoint_count=int(smooth_points.shape[0]))

    poses = build_poses(points=smooth_points, fps=effective_pose_fps)

    check_kinematics = bool(_stage3_cfg(config).get("check_kinematics", False))
    if check_kinematics:
        constraints = check_constraints(poses=poses, limits=limits)
    else:
        constraints = {
            "feasible": True,
            "skipped": True,
            "reason": "kinematics_check_disabled",
            "violations": [],
        }
    collision = check_collision(
        points=smooth_points,
        obstacles_xyz=obstacles_xyz,
        safety_distance=safety_distance,
        keepout_boxes=keepout_boxes,
        keepout_margin_xy=max(0.2, float(safety_distance) * 0.25),
        keepout_margin_z=max(0.1, float(safety_distance) * 0.15),
    )

    waypoints_path = out_dir / "waypoints.npy"
    raw_path = out_dir / "composed_path_raw.npy"
    forwards_path = out_dir / "forwards.npy"
    poses_path = out_dir / "poses.csv"
    segments_path = out_dir / "composed_segments.json"
    report_path = out_dir / "constraint_report.json"

    np.save(waypoints_path, smooth_points.astype(np.float32))
    np.save(forwards_path, smooth_forwards.astype(np.float32))
    np.save(raw_path, raw_points.astype(np.float32))
    _write_poses_csv(poses_path, poses)

    duration_sec = float(max(0.0, len(poses) - 1) / max(1.0, effective_pose_fps))
    landmark_order = list((set_instance or {}).get("landmark_order", [str(chosen.get("instance_id", "") or "")]) or [str(chosen.get("instance_id", "") or "")])
    revisit_count = max(0, len(landmark_order) - len(set(landmark_order)))
    self_difficulty_score, self_difficulty_band = _compute_self_state_difficulty(
        duration_sec=duration_sec,
        landmark_count=len(set(landmark_order)),
        element_count=len(element_instances),
        revisit_count=int(revisit_count),
    )
    flight_description = _build_flight_description(
        set_spec=set_spec,
        set_instance=set_instance,
        element_instances=element_instances,
        primary=chosen,
        secondary=secondary_landmarks,
    )

    segments_payload = {
        "scene_id": scene_id,
        "traj_id": traj_id,
        "mode": mission_mode,
        "task_group": "dual_awareness",
        "task_family": "flight_mission",
        "task_type": str((set_instance or {}).get("set_name", set_spec.get("display_name", "single_atomic"))),
        "task_subtype": str((set_instance or {}).get("set_id", _safe_name(str(set_spec.get("display_name", "single_atomic"))))),
        "task_difficulty": str(self_difficulty_band),
        "task_difficulty_score": float(self_difficulty_score),
        "instance_id": str(chosen_raw.get("instance_id", "")),
        "secondary_instance_ids": [str(item.get("instance_id", "") or "") for item in secondary_landmarks],
        "set_class": dict(set_spec),
        "set_instance": dict(set_instance or {}),
        "element_instances": element_instances,
        "element_plan": plan_rows,
        "mode_sequence": mode_sequence,
        "event_sequence": event_sequence,
        "behavior_sequence": behavior_sequence,
        "mission": {
            "set_name": str((set_instance or {}).get("set_name", set_spec.get("display_name", "")) or ""),
            "set_id": str((set_instance or {}).get("set_id", _safe_name(str(set_spec.get("display_name", "")))) or ""),
            "sequence": behavior_sequence,
        },
        "source": {
            "type": source_name,
            "path": str(source_path.as_posix()),
        },
        "summary": {
            "segment_count": int(len(segments)),
            "raw_points": int(raw_points.shape[0]),
            "waypoints": int(smooth_points.shape[0]),
            "fps": float(effective_pose_fps),
            "source_pose_fps": float(effective_pose_fps),
            "base_pose_fps": float(fps),
            "samples_per_segment": int(samples_per_segment),
        },
        "segments": segments,
    }
    write_json(segments_path, segments_payload)

    report_payload = {
        "scene_id": scene_id,
        "traj_id": traj_id,
        "mode": mission_mode,
        "task_group": "dual_awareness",
        "task_family": "flight_mission",
        "task_type": str((set_instance or {}).get("set_name", set_spec.get("display_name", "single_atomic"))),
        "task_subtype": str((set_instance or {}).get("set_id", _safe_name(str(set_spec.get("display_name", "single_atomic"))))),
        "task_difficulty": str(self_difficulty_band),
        "task_difficulty_score": float(self_difficulty_score),
        "instance_id": str(chosen_raw.get("instance_id", "")),
        "landmark_class_id": chosen_raw.get("class_id", None),
        "landmark_class_name": _landmark_category(chosen),
        "landmark_subcategory": _landmark_subcategory(chosen),
        "landmark_description": _landmark_description(chosen),
        "landmark_center_3d": center_3d,
        "landmark_bbox_3d": chosen_raw.get("bbox_3d", {}),
        "set_class": dict(set_spec),
        "set_instance": dict(set_instance or {}),
        "landmark_set_map": dict(landmark_set_map),
        "element_instances": element_instances,
        "mode_sequence": mode_sequence,
        "event_sequence": event_sequence,
        "event_count": int(len(event_sequence)),
        "secondary_instance_ids": [str(item.get("instance_id", "") or "") for item in secondary_landmarks],
        "secondary_landmarks": [
            {
                "instance_id": str(item.get("instance_id", "") or ""),
                "class_name": _landmark_category(item),
                "center_3d": list(item.get("center_3d", []) or []),
                "bbox_3d": list(item.get("bbox_3d", []) or []),
            }
            for item in secondary_landmarks
        ],
        "flight_description": flight_description,
        "limits": {
            **limits,
            "safety_distance": safety_distance,
        },
        "checks": {
            "kinematics": constraints,
            "collision": collision,
            "repair": repair_summary,
        },
        "files": {
            "waypoints": str(waypoints_path.as_posix()),
            "forwards": str(forwards_path.as_posix()),
            "raw_path": str(raw_path.as_posix()),
            "poses": str(poses_path.as_posix()),
            "segments": str(segments_path.as_posix()),
        },
        "summary": {
            "segment_count": int(len(segments)),
            "raw_points": int(raw_points.shape[0]),
            "waypoints": int(smooth_points.shape[0]),
            "base_pose_fps": float(fps),
            "source_pose_fps": float(effective_pose_fps),
            "samples_per_segment": int(samples_per_segment),
            "landmark_set_map": dict(landmark_set_map),
            "selected_component_set_map": dict((set_spec or {}).get("selected_component_set_map", {}) or {}),
            "component_sequence": list((set_spec or {}).get("component_sequence", []) or []),
        },
        "runtime_sec": float(time.time() - t_start),
    }

    panorama_files: dict[str, str] = {}
    if bool(_stage3_cfg(config).get("preview_panorama_enabled", True)):
        try:
            panorama_files = _generate_panorama_images(
                out_dir=out_dir,
                obstacles_xyz=obstacles_xyz,
                obstacles_bgr=obstacles_bgr,
                path_xyz=smooth_points,
                target_center_3d=center_3d,
                target_bbox_list=bbox_list,
            )
        except Exception as exc:
            panorama_files = {}
            report_payload["preview_warning"] = f"panorama_generation_failed: {exc}"

    if panorama_files:
        report_payload.setdefault("files", {})
        report_payload["files"].update(panorama_files)
    write_json(report_path, report_payload)

    runtime_sec = float(time.time() - t_start)
    out = {
        "ok": True,
        "mode": "generate_traj",
        "generation_kind": generation_kind,
        "mission_mode": mission_mode,
        "scene_id": scene_id,
        "traj_id": traj_id,
        "instance_id": str(chosen_raw.get("instance_id", "")),
        "secondary_instance_ids": [str(item.get("instance_id", "") or "") for item in secondary_landmarks],
        "source": source_name,
        "source_path": str(source_path.as_posix()),
        "output_dir": str(out_dir.as_posix()),
        "files": {
            "waypoints": str(waypoints_path.as_posix()),
            "forwards": str(forwards_path.as_posix()),
            "raw_path": str(raw_path.as_posix()),
            "poses": str(poses_path.as_posix()),
            "segments": str(segments_path.as_posix()),
            "report": str(report_path.as_posix()),
            **panorama_files,
        },
        "summary": {
            "task_family": "flight_mission",
            "task_type": str((set_instance or {}).get("set_name", set_spec.get("display_name", "single_atomic"))),
            "task_subtype": str((set_instance or {}).get("set_id", _safe_name(str(set_spec.get("display_name", "single_atomic"))))),
            "task_difficulty": str(self_difficulty_band),
            "task_difficulty_score": float(self_difficulty_score),
            "set_name": str((set_instance or {}).get("set_name", set_spec.get("display_name", "")) or ""),
            "set_id": str((set_instance or {}).get("set_id", _safe_name(str(set_spec.get("display_name", "")))) or ""),
            "mode_sequence": mode_sequence,
            "event_sequence": event_sequence,
            "event_count": int(len(event_sequence)),
            "behavior_sequence": behavior_sequence,
            "element_sequence": behavior_sequence,
            "element_instances": element_instances,
            "set_instance": dict(set_instance or {}),
            "secondary_landmark_count": int(len(secondary_landmarks)),
            "secondary_instance_ids": [str(item.get("instance_id", "") or "") for item in secondary_landmarks],
            "landmark_set_map": dict(landmark_set_map),
            "segments": int(len(segments)),
            "raw_points": int(raw_points.shape[0]),
            "waypoints": int(smooth_points.shape[0]),
            "target_center_3d": center_3d,
            "target_bbox_3d_list": bbox_list,
            "flight_description": flight_description,
            "kinematics_checked": bool(check_kinematics),
            "kinematics_feasible": (bool(constraints.get("feasible", False)) if check_kinematics else None),
            "collision_free": bool(collision.get("collision_free", False)),
            "repair_repaired_segments": int(repair_summary.get("repaired_segments", 0) or 0),
            "repair_lifted_points": int(repair_summary.get("lifted_points", 0) or 0),
            "repair_lifted_fraction": float(repair_summary.get("lifted_fraction", 0.0) or 0.0),
            "repair_max_lift_m": float(repair_summary.get("max_lift_m", 0.0) or 0.0),
            "source_pose_fps": float(effective_pose_fps),
            "base_pose_fps": float(fps),
            "samples_per_segment": int(samples_per_segment),
            "runtime_sec": runtime_sec,
        },
    }
    if include_preview:
        out["preview"] = _sample_preview_points(smooth_points, max_points=240)

    logger.info(
        f"done scene={scene_id} traj={traj_id} instance={chosen_raw.get('instance_id', '')} "
        f"segments={len(segments)} waypoints={smooth_points.shape[0]} runtime={runtime_sec:.2f}s"
    )
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage3",
        step="generate",
        message="generate_finished",
        payload=build_unified_stage_event(
            stage="stage3",
            step="generate",
            scene_id=scene_id,
            engine=engine_name,
            status="finished",
            extra={"traj_id": str(traj_id), "waypoints": int(smooth_points.shape[0])},
        ),
    )
    return out


def _review_files(review_root: Path, scene_id: str) -> tuple[Path, Path]:
    review_dir = review_root
    ensure_dir(review_dir)
    return review_dir / "review_index.json", review_dir / "review_log.jsonl"


def _load_review_index(path: Path) -> dict[str, Any]:
    data = read_json_if_exists(path, default={})
    if not isinstance(data, dict):
        return {"scene_id": "", "items": {}}
    items = data.get("items", {})
    if not isinstance(items, dict):
        items = {}
    return {"scene_id": data.get("scene_id", ""), "items": items}


def _write_review_index(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _build_mission_history_entry(
    *,
    out: dict[str, Any],
    selected_instance_ids: list[str],
) -> dict[str, Any]:
    files = dict(out.get("files", {}) or {})
    summary = dict(out.get("summary", {}) or {})
    return {
        "traj_id": str(out.get("traj_id", "") or ""),
        "mission_mode": str(out.get("mission_mode", out.get("mode", "")) or ""),
        "generation_kind": str(out.get("generation_kind", "") or ""),
        "instance_id": str(out.get("instance_id", "") or ""),
        "landmark_instance_ids": [str(x) for x in list(selected_instance_ids or []) if str(x).strip()],
        "secondary_instance_ids": [str(x) for x in list(out.get("secondary_instance_ids", []) or []) if str(x).strip()],
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "traj_status": "video_ready",
        "files": files,
        "summary": summary,
    }


def _history_status_rank(status: str) -> int:
    order = {
        "pending": 0,
        "pano_ready": 1,
        "pano_confirmed": 2,
        "video_ready": 3,
        "video_confirmed": 4,
        "final_ready": 5,
        "pano_rejected": -1,
        "video_rejected": -1,
    }
    return int(order.get(str(status or "pending"), 0))


def _append_history_entry(rec: dict[str, Any], entry: dict[str, Any]) -> None:
    history = list(rec.get("mission_history", []) or []) if isinstance(rec.get("mission_history", []), list) else []
    traj_id = str(entry.get("traj_id", "") or "").strip()
    if traj_id:
        history = [row for row in history if str((row or {}).get("traj_id", "") or "") != traj_id]
    history.append(dict(entry))
    history.sort(key=lambda row: str(row.get("updated_at", row.get("created_at", "")) or ""), reverse=True)
    rec["mission_history"] = history[:80]


def _update_history_entry(
    rec: dict[str, Any],
    *,
    traj_id: str,
    updates: dict[str, Any],
) -> dict[str, Any]:
    history = list(rec.get("mission_history", []) or []) if isinstance(rec.get("mission_history", []), list) else []
    target_id = str(traj_id or "").strip()
    updated_row: dict[str, Any] | None = None
    for row in history:
        if not isinstance(row, dict):
            continue
        if str(row.get("traj_id", "") or "").strip() != target_id:
            continue
        row.update(dict(updates))
        row["updated_at"] = _utc_now_iso()
        updated_row = row
        break
    if updated_row is None and target_id:
        updated_row = {"traj_id": target_id, **dict(updates), "updated_at": _utc_now_iso()}
        history.append(updated_row)
    history.sort(key=lambda row: str((row or {}).get("updated_at", (row or {}).get("created_at", "")) or ""), reverse=True)
    rec["mission_history"] = history[:80]
    return updated_row or {}


def _select_representative_history_entry(rec: dict[str, Any]) -> dict[str, Any]:
    history = list(rec.get("mission_history", []) or []) if isinstance(rec.get("mission_history", []), list) else []
    rows = [row for row in history if isinstance(row, dict)]
    if not rows:
        return {}
    rows.sort(
        key=lambda row: (
            _history_status_rank(str(row.get("traj_status", "pending") or "pending")),
            str(row.get("updated_at", row.get("created_at", "")) or ""),
        ),
        reverse=True,
    )
    return dict(rows[0])


def _reset_instance_record_from_history(rec: dict[str, Any]) -> None:
    best = _select_representative_history_entry(rec)
    if best:
        rec["traj_id"] = str(best.get("traj_id", "") or "")
        rec["traj_status"] = str(best.get("traj_status", "pending") or "pending")
        rec["updated_at"] = str(best.get("updated_at", best.get("created_at", "")) or "")
        rec["files"] = dict(best.get("files", {}) or {})
        rec["summary"] = dict(best.get("summary", {}) or {})
        return
    rec["traj_id"] = ""
    rec["traj_status"] = "pending"
    rec["updated_at"] = _utc_now_iso()
    rec["files"] = {}
    rec["summary"] = {}


def _remove_mission_from_record(rec: dict[str, Any], *, traj_id: str) -> bool:
    history = list(rec.get("mission_history", []) or []) if isinstance(rec.get("mission_history", []), list) else []
    target = str(traj_id or "").strip()
    if not target:
        return False
    kept = [row for row in history if str((row or {}).get("traj_id", "") or "").strip() != target]
    changed = len(kept) != len(history)
    if changed:
        rec["mission_history"] = kept
        _reset_instance_record_from_history(rec)
    return changed


def _delete_mission_artifacts(missions_root: Path, *, traj_id: str) -> dict[str, Any]:
    target = str(traj_id or "").strip()
    mission_dir = missions_root / target
    if not target:
        return {"traj_id": target, "deleted": False, "reason": "missing_traj_id"}
    if not mission_dir.exists():
        return {"traj_id": target, "deleted": False, "reason": "not_found", "path": str(mission_dir.as_posix())}
    if not mission_dir.is_dir():
        return {"traj_id": target, "deleted": False, "reason": "not_a_directory", "path": str(mission_dir.as_posix())}
    try:
        shutil.rmtree(mission_dir)
    except Exception as exc:
        return {
            "traj_id": target,
            "deleted": False,
            "reason": "rmtree_failed",
            "error": str(exc),
            "path": str(mission_dir.as_posix()),
        }
    deleted = not mission_dir.exists()
    return {
        "traj_id": target,
        "deleted": bool(deleted),
        "reason": "deleted" if deleted else "still_exists_after_rmtree",
        "path": str(mission_dir.as_posix()),
    }


def _collect_mission_history(review_items: dict[str, Any], selected_instance_ids: list[str]) -> list[dict[str, Any]]:
    wanted = [str(x).strip() for x in list(selected_instance_ids or []) if str(x).strip()]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for instance_id, rec in list((review_items or {}).items()):
        if not isinstance(rec, dict):
            continue
        history = list(rec.get("mission_history", []) or []) if isinstance(rec.get("mission_history", []), list) else []
        if not history and str(rec.get("traj_id", "") or "").strip():
            history = [{
                "traj_id": str(rec.get("traj_id", "") or ""),
                "instance_id": str(instance_id),
                "landmark_instance_ids": [str(instance_id)],
                "created_at": str(rec.get("updated_at", "") or ""),
                "updated_at": str(rec.get("updated_at", "") or ""),
                "traj_status": str(rec.get("traj_status", "pending") or "pending"),
                "files": dict(rec.get("files", {}) or {}),
                "summary": dict(rec.get("summary", {}) or {}),
                "mission_mode": str((rec.get("summary", {}) or {}).get("mode", "") or ""),
            }]
        for entry in history:
            if not isinstance(entry, dict):
                continue
            traj_id = str(entry.get("traj_id", "") or "").strip()
            if not traj_id or traj_id in seen:
                continue
            related_ids = [str(x).strip() for x in list(entry.get("landmark_instance_ids", []) or []) if str(x).strip()]
            if wanted and not all(item in related_ids for item in wanted):
                continue
            seen.add(traj_id)
            rows.append(dict(entry))
    rows.sort(key=lambda row: str(row.get("created_at", "") or ""), reverse=True)
    return rows


def _build_instance_rows(instances: list[dict[str, Any]], review_items: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in instances:
        instance_id = str(item.get("instance_id", ""))
        review = review_items.get(instance_id, {}) if isinstance(review_items.get(instance_id, {}), dict) else {}
        decision_status = str(review.get("status", "pending") or "pending")
        best_history = _select_representative_history_entry(review)
        traj_status = str(best_history.get("traj_status", review.get("traj_status", "pending")) or "pending")
        list_status = decision_status if decision_status in {"valid", "invalid"} else traj_status
        rows.append(
            {
                "instance_id": instance_id,
                "class_name": str(item.get("class_name", "") or ""),
                "class_id": item.get("class_id", None),
                "point_count": int(item.get("point_count", 0) or 0),
                "center_3d": item.get("center_3d", None),
                "bbox_3d": item.get("bbox_3d", None),
                "status": list_status,
                "decision_status": decision_status,
                "traj_status": traj_status,
                "label": str(review.get("label", "") or ""),
                "note": str(review.get("note", "") or ""),
                "latest_traj_id": str(best_history.get("traj_id", review.get("traj_id", "")) or ""),
                "updated_at": str(best_history.get("updated_at", review.get("updated_at", "")) or ""),
                "latest_files": best_history.get("files", review.get("files", {})),
                "latest_summary": best_history.get("summary", review.get("summary", {})),
                "mission_history_count": int(len(list(review.get("mission_history", []) or []))) if isinstance(review.get("mission_history", []), list) else 0,
            }
        )
    rows.sort(key=lambda it: (str(it.get("class_name", "")), -int(it["point_count"]), it["instance_id"]))
    return rows


def _sample_points(points: np.ndarray, max_points: int, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32), (np.zeros((0, 3), dtype=np.uint8) if colors is not None else None)
    n = int(points.shape[0])
    if n <= max_points:
        return points.astype(np.float32), (colors.astype(np.uint8) if colors is not None else None)
    idx = np.linspace(0, n - 1, num=max_points, endpoint=True).astype(np.int64)
    return points[idx].astype(np.float32), (colors[idx].astype(np.uint8) if colors is not None else None)


def _crop_points_near_trajectory(
    obstacles_xyz: np.ndarray,
    obstacles_bgr: np.ndarray,
    path_xyz: np.ndarray,
    *,
    margin_xy_override_m: float | None = None,
    margin_z_low_override_m: float | None = None,
    margin_z_high_override_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if obstacles_xyz.size == 0 or path_xyz.size == 0:
        return obstacles_xyz, obstacles_bgr
    px = path_xyz[:, 0]
    py = path_xyz[:, 1]
    pz = path_xyz[:, 2]
    min_x, max_x = float(np.min(px)), float(np.max(px))
    min_y, max_y = float(np.min(py)), float(np.max(py))
    min_z, max_z = float(np.min(pz)), float(np.max(pz))
    diag_xy = float(math.sqrt(max(1e-6, (max_x - min_x) ** 2 + (max_y - min_y) ** 2)))
    margin_xy = float(margin_xy_override_m) if margin_xy_override_m is not None else max(3.0, min(14.0, diag_xy * 0.045))
    margin_z_low = float(margin_z_low_override_m) if margin_z_low_override_m is not None else max(2.0, min(8.0, diag_xy * 0.02))
    margin_z_high = float(margin_z_high_override_m) if margin_z_high_override_m is not None else max(4.0, min(10.0, diag_xy * 0.03))

    mask = (
        (obstacles_xyz[:, 0] >= (min_x - margin_xy))
        & (obstacles_xyz[:, 0] <= (max_x + margin_xy))
        & (obstacles_xyz[:, 1] >= (min_y - margin_xy))
        & (obstacles_xyz[:, 1] <= (max_y + margin_xy))
        & (obstacles_xyz[:, 2] >= (min_z - margin_z_low))
        & (obstacles_xyz[:, 2] <= (max_z + margin_z_high))
    )
    selected = int(np.count_nonzero(mask))
    if selected <= 0:
        return obstacles_xyz, obstacles_bgr
    return obstacles_xyz[mask], obstacles_bgr[mask]


def _target_bbox_corners_world(center_3d: list[float], bbox_list: list[float]) -> np.ndarray:
    cx, cy, cz = float(center_3d[0]), float(center_3d[1]), float(center_3d[2])
    sx = float(bbox_list[3]) if len(bbox_list) > 3 else 3.0
    sy = float(bbox_list[4]) if len(bbox_list) > 4 else 3.0
    sz = float(bbox_list[5]) if len(bbox_list) > 5 else 3.0
    yaw_deg = float(bbox_list[6]) if len(bbox_list) > 6 else 0.0
    hx, hy, hz = max(0.2, sx * 0.5), max(0.2, sy * 0.5), max(0.2, sz * 0.5)
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)

    corners_local = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    corners_world = corners_local @ rot.T
    corners_world[:, 0] += cx
    corners_world[:, 1] += cy
    corners_world[:, 2] += cz
    return corners_world.astype(np.float32)


def _project_oblique(points: np.ndarray, az_deg: float, el_deg: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    az = math.radians(float(az_deg))
    el = math.radians(float(el_deg))
    cz, sz = math.cos(az), math.sin(az)
    cx, sx = math.cos(el), math.sin(el)

    rot_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot = rot_x @ rot_z
    p = points @ rot.T
    return p[:, :2].astype(np.float32)


def _fit_to_canvas(xy: np.ndarray, width: int, height: int, pad: int = 20) -> np.ndarray:
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    xs = xy[:, 0]
    ys = xy[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
    px = (xs - min_x) * scale + pad
    py = height - ((ys - min_y) * scale + pad)
    out = np.stack([px, py], axis=1)
    return np.round(out).astype(np.int32)


def _draw_scene_image(
    obstacles_xyz: np.ndarray,
    obstacles_bgr: np.ndarray,
    path_xyz: np.ndarray,
    target_center_3d: list[float] | None,
    target_bbox_list: list[float] | None,
    az_deg: float,
    el_deg: float,
    width: int,
    height: int,
    progress_ratio: float = 1.0,
    current_pose_xyz: np.ndarray | None = None,
    current_forward_xyz: np.ndarray | None = None,
    camera_fov_deg: float = 72.0,
) -> np.ndarray:
    img = np.full((height, width, 3), 248, dtype=np.uint8)
    sampled_obs, sampled_obs_bgr = _sample_points(obstacles_xyz, max_points=26000, colors=obstacles_bgr)
    sampled_path, _ = _sample_points(path_xyz, max_points=3000)
    target_corners_world = None
    if target_center_3d is not None and target_bbox_list is not None:
        target_corners_world = _target_bbox_corners_world(target_center_3d, target_bbox_list)

    merged = sampled_path if sampled_obs.size == 0 else (sampled_obs if sampled_path.size == 0 else np.vstack([sampled_obs, sampled_path]))
    if target_corners_world is not None and target_corners_world.size > 0:
        merged = target_corners_world if merged.size == 0 else np.vstack([merged, target_corners_world])
    focus_world = sampled_path
    if target_corners_world is not None and target_corners_world.size > 0:
        focus_world = target_corners_world if focus_world.size == 0 else np.vstack([focus_world, target_corners_world])
    if current_pose_xyz is not None:
        pose_arr = np.asarray(current_pose_xyz, dtype=np.float32).reshape(1, 3)
        focus_world = pose_arr if focus_world.size == 0 else np.vstack([focus_world, pose_arr])
    focus_source = focus_world if focus_world.size > 0 else merged
    proj_focus = _project_oblique(focus_source, az_deg=az_deg, el_deg=el_deg)
    proj_all = _project_oblique(merged, az_deg=az_deg, el_deg=el_deg)
    if proj_all.size == 0:
        proj_all = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    if proj_focus.size == 0:
        proj_focus = proj_all
    xs = proj_focus[:, 0]
    ys = proj_focus[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    # Fit around trajectory corridor instead of the full obstacle crop.
    min_x -= span_x * 0.10
    max_x += span_x * 0.10
    min_y -= span_y * 0.10
    max_y += span_y * 0.10
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    pad = 28
    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)

    def _map_proj_points(xy: np.ndarray) -> np.ndarray:
        if xy.size == 0:
            return np.zeros((0, 2), dtype=np.int32)
        px = (xy[:, 0] - min_x) * scale + pad
        py = height - ((xy[:, 1] - min_y) * scale + pad)
        return np.round(np.stack([px, py], axis=1)).astype(np.int32)

    canvas_all = _map_proj_points(proj_all)

    obs_n = sampled_obs.shape[0]
    obs_pts = canvas_all[:obs_n] if obs_n > 0 else np.zeros((0, 2), dtype=np.int32)
    path_n = sampled_path.shape[0] if sampled_path.shape[0] > 0 else 0
    path_start = obs_n
    path_end = path_start + path_n
    path_pts = canvas_all[path_start:path_end] if path_n > 0 else np.zeros((0, 2), dtype=np.int32)
    box_pts = canvas_all[path_end:] if target_corners_world is not None and target_corners_world.size > 0 else np.zeros((0, 2), dtype=np.int32)

    if sampled_obs_bgr is None:
        sampled_obs_bgr = np.full((obs_pts.shape[0], 3), 185, dtype=np.uint8)
    for idx, p in enumerate(obs_pts):
        c = sampled_obs_bgr[idx] if idx < sampled_obs_bgr.shape[0] else np.array([185, 185, 185], dtype=np.uint8)
        cv2.circle(img, (int(p[0]), int(p[1])), 1, (int(c[0]), int(c[1]), int(c[2])), -1, lineType=cv2.LINE_AA)

    if path_pts.shape[0] >= 2:
        keep = int(max(2, min(path_pts.shape[0], round(path_pts.shape[0] * float(progress_ratio)))))
        sub = path_pts[:keep]
        cv2.polylines(img, [sub.reshape(-1, 1, 2)], False, (46, 111, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(sub[0]), 5, (35, 165, 45), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(sub[-1]), 5, (40, 40, 225), -1, lineType=cv2.LINE_AA)

    if box_pts.shape[0] >= 4:
        x0, y0 = int(np.min(box_pts[:, 0])), int(np.min(box_pts[:, 1]))
        x1, y1 = int(np.max(box_pts[:, 0])), int(np.max(box_pts[:, 1]))
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        if (x1 - x0) >= 6 and (y1 - y0) >= 6:
            cv2.rectangle(img, (x0, y0), (x1, y1), (20, 20, 240), 2, lineType=cv2.LINE_AA)
            cv2.putText(img, "TARGET", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 240), 2, cv2.LINE_AA)

    if (
        current_pose_xyz is not None
        and isinstance(current_pose_xyz, np.ndarray)
        and current_pose_xyz.size >= 3
        and current_forward_xyz is not None
        and isinstance(current_forward_xyz, np.ndarray)
        and current_forward_xyz.size >= 3
    ):
        pose = current_pose_xyz.astype(np.float32).reshape(3)
        fwd = current_forward_xyz.astype(np.float32).reshape(3)
        fwd_norm = float(np.linalg.norm(fwd))
        if fwd_norm >= 1e-6:
            fwd = (fwd / fwd_norm).astype(np.float32)
            diag_xy = 0.0
            if path_xyz.size > 0:
                dx = float(np.max(path_xyz[:, 0]) - np.min(path_xyz[:, 0]))
                dy = float(np.max(path_xyz[:, 1]) - np.min(path_xyz[:, 1]))
                diag_xy = float(math.sqrt(max(1e-6, dx * dx + dy * dy)))
            arrow_len = max(4.0, min(16.0, diag_xy * 0.08 if diag_xy > 0 else 8.0))
            half_fov = math.radians(float(max(20.0, min(140.0, camera_fov_deg))) * 0.5)
            c1, s1 = math.cos(half_fov), math.sin(half_fov)
            c2, s2 = math.cos(-half_fov), math.sin(-half_fov)
            fx, fy, fz = float(fwd[0]), float(fwd[1]), float(fwd[2])
            left_dir = np.asarray([fx * c1 - fy * s1, fx * s1 + fy * c1, fz], dtype=np.float32)
            right_dir = np.asarray([fx * c2 - fy * s2, fx * s2 + fy * c2, fz], dtype=np.float32)
            left_dir /= max(1e-6, float(np.linalg.norm(left_dir)))
            right_dir /= max(1e-6, float(np.linalg.norm(right_dir)))

            tip = pose + fwd * np.float32(arrow_len)
            left_tip = pose + left_dir * np.float32(arrow_len * 1.2)
            right_tip = pose + right_dir * np.float32(arrow_len * 1.2)
            proj_pose = _map_proj_points(_project_oblique(np.asarray([pose], dtype=np.float32), az_deg=az_deg, el_deg=el_deg))
            proj_tip = _map_proj_points(_project_oblique(np.asarray([tip], dtype=np.float32), az_deg=az_deg, el_deg=el_deg))
            proj_left = _map_proj_points(_project_oblique(np.asarray([left_tip], dtype=np.float32), az_deg=az_deg, el_deg=el_deg))
            proj_right = _map_proj_points(_project_oblique(np.asarray([right_tip], dtype=np.float32), az_deg=az_deg, el_deg=el_deg))
            if proj_pose.shape[0] > 0:
                p0 = tuple(int(v) for v in proj_pose[0])
                cv2.circle(img, p0, 6, (0, 185, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(img, "UAV", (p0[0] + 8, max(16, p0[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 220), 1, cv2.LINE_AA)
                if proj_tip.shape[0] > 0:
                    p1 = tuple(int(v) for v in proj_tip[0])
                    cv2.arrowedLine(img, p0, p1, (30, 130, 255), 2, line_type=cv2.LINE_AA, tipLength=0.28)
                if proj_left.shape[0] > 0 and proj_right.shape[0] > 0:
                    pl = tuple(int(v) for v in proj_left[0])
                    pr = tuple(int(v) for v in proj_right[0])
                    cv2.line(img, p0, pl, (0, 200, 200), 1, lineType=cv2.LINE_AA)
                    cv2.line(img, p0, pr, (0, 200, 200), 1, lineType=cv2.LINE_AA)
                    cv2.line(img, pl, pr, (0, 180, 180), 1, lineType=cv2.LINE_AA)

    return img


def _overlay_right_preview_panel(base_bgr: np.ndarray, preview_bgr: np.ndarray, title: str = "POSE PREVIEW") -> np.ndarray:
    if base_bgr.ndim != 3 or base_bgr.shape[2] < 3:
        return base_bgr
    h, w = int(base_bgr.shape[0]), int(base_bgr.shape[1])
    if h <= 0 or w <= 0:
        return base_bgr
    margin = 12
    panel_w = max(220, min(w - 2 * margin, int(round(float(w) * 0.34))))
    panel_h = max(160, min(h - 2 * margin, int(round(float(h) * 0.34))))
    if panel_w <= 0 or panel_h <= 0 or panel_w >= w or panel_h >= h:
        return base_bgr

    panel = preview_bgr
    if panel.ndim != 3 or panel.shape[2] < 3:
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    if panel.shape[0] != panel_h or panel.shape[1] != panel_w:
        panel = cv2.resize(panel[:, :, :3], (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    out = base_bgr.copy()
    x0 = w - panel_w - margin
    y0 = h - panel_h - margin
    x1 = x0 + panel_w
    y1 = y0 + panel_h
    roi = out[y0:y1, x0:x1, :]
    blended = cv2.addWeighted(roi, 0.20, panel, 0.80, 0.0)
    out[y0:y1, x0:x1, :] = blended

    cv2.rectangle(out, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), (32, 41, 56), 2, lineType=cv2.LINE_AA)
    title_y0 = max(0, y0 - 30)
    title_y1 = max(0, y0 - 6)
    cv2.rectangle(out, (x0, title_y0), (min(w - 1, x0 + panel_w - 1), title_y1), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(out, str(title), (x0 + 8, max(14, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 2, cv2.LINE_AA)
    return out


def _overlay_frame_index(base_bgr: np.ndarray, frame_id: int, total: int, src_idx: int) -> np.ndarray:
    if base_bgr.ndim != 3 or base_bgr.shape[2] < 3:
        return base_bgr
    out = base_bgr.copy()
    text = f"Frame {int(frame_id) + 1}/{int(max(1, total))}  src={int(src_idx)}"
    h, w = out.shape[:2]
    box_w = min(max(320, int(round(0.44 * w))), w - 16)
    x0, y0 = 8, 8
    x1, y1 = x0 + box_w, 42
    cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(out, text, (x0 + 10, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 2, cv2.LINE_AA)
    return out


def _generate_panorama_images(
    out_dir: Path,
    obstacles_xyz: np.ndarray,
    obstacles_bgr: np.ndarray,
    path_xyz: np.ndarray,
    target_center_3d: list[float] | None,
    target_bbox_list: list[float] | None,
) -> dict[str, str]:
    if cv2 is None:
        raise RuntimeError("opencv(cv2) not available for panorama generation")
    stage3_cfg = _stage3_cfg(config)
    image_cfg = build_image_compression_cfg(stage3_cfg)
    panorama_dir = out_dir / "preview"
    ensure_dir(panorama_dir)
    p1 = preferred_output_path(panorama_dir / "panorama_left.jpg", compress_enabled=bool(image_cfg.get("enabled", True)))
    p2 = preferred_output_path(panorama_dir / "panorama_right.jpg", compress_enabled=bool(image_cfg.get("enabled", True)))
    crop_xyz, crop_bgr = _crop_points_near_trajectory(obstacles_xyz=obstacles_xyz, obstacles_bgr=obstacles_bgr, path_xyz=path_xyz)
    img1 = _draw_scene_image(
        obstacles_xyz=crop_xyz,
        obstacles_bgr=crop_bgr,
        path_xyz=path_xyz,
        target_center_3d=target_center_3d,
        target_bbox_list=target_bbox_list,
        az_deg=35.0,
        el_deg=28.0,
        width=1280,
        height=720,
    )
    img2 = _draw_scene_image(
        obstacles_xyz=crop_xyz,
        obstacles_bgr=crop_bgr,
        path_xyz=path_xyz,
        target_center_3d=target_center_3d,
        target_bbox_list=target_bbox_list,
        az_deg=-35.0,
        el_deg=22.0,
        width=1280,
        height=720,
    )
    save_bgr_image(img1, p1, cfg=image_cfg)
    save_bgr_image(img2, p2, cfg=image_cfg)
    return {
        "panorama_left": str(p1.as_posix()),
        "panorama_right": str(p2.as_posix()),
    }


def _generate_trajectory_video(
    out_dir: Path,
    obstacles_xyz: np.ndarray,
    obstacles_bgr: np.ndarray,
    path_xyz: np.ndarray,
    target_center_3d: list[float] | None,
    target_bbox_list: list[float] | None,
    segments: list[dict[str, Any]] | None = None,
    source_pose_fps: float = 10.0,
    fps: float = 5.0,
    frame_stride: int = 1,
    speedup: float = 1.1,
    bitrate: str = "10M",
    web_bitrate: str = "2M",
    pose_smooth_window: int = 7,
    camera_fov_deg: float = 72.0,
) -> dict[str, str]:
    if cv2 is None:
        raise RuntimeError("opencv(cv2) not available for video generation")
    stage3_cfg = _stage3_cfg(config)
    image_cfg = build_image_compression_cfg(stage3_cfg)
    preview_dir = out_dir / "preview"
    ensure_dir(preview_dir)
    video_path = preview_dir / "trajectory_preview.mp4"
    video_web_path = preview_dir / "trajectory_preview_web.mp4"
    preview_frames_dir = preview_dir / "frames"
    preview_manifest_path = preview_dir / "trajectory_preview_frames.json"
    preview_index_map_path = preview_dir / "trajectory_preview_index_map.json"
    ensure_dir(preview_frames_dir)

    width, height = 1280, 720
    trim_start_idx = min(max(0, _video_trim_start_index(segments, total_points=int(path_xyz.shape[0]))), max(0, int(path_xyz.shape[0]) - 1))
    display_path_xyz = path_xyz[trim_start_idx:].astype(np.float32) if int(path_xyz.shape[0]) > 0 else path_xyz.astype(np.float32)
    sampled_idx = _build_sampled_indices(
        point_count=int(display_path_xyz.shape[0]),
        source_fps=float(source_pose_fps),
        target_fps=float(fps),
        frame_stride=int(frame_stride),
        speedup=float(speedup),
    )
    sampled_idx = [int(trim_start_idx + idx) for idx in sampled_idx]
    _assert_strictly_increasing_indices(sampled_idx, name="preview_sampled_indices")
    sampled_points = path_xyz[np.asarray(sampled_idx, dtype=np.int64)].astype(np.float32)
    smoothed_forwards = _build_smoothed_forward_vectors(sampled_points, smooth_window=int(pose_smooth_window))
    frame_count = max(1, int(len(sampled_idx)))
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1.0, fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("failed_to_open_video_writer")

    crop_xyz, crop_bgr = _crop_points_near_trajectory(obstacles_xyz=obstacles_xyz, obstacles_bgr=obstacles_bgr, path_xyz=display_path_xyz)

    frame_paths: list[str] = []
    for i, src_idx in enumerate(sampled_idx):
        r = float(int(src_idx) + 1) / float(max(1, int(path_xyz.shape[0])))
        curr_pose = sampled_points[i] if i < sampled_points.shape[0] else None
        curr_forward = smoothed_forwards[i] if i < smoothed_forwards.shape[0] else None
        left = _draw_scene_image(
            obstacles_xyz=crop_xyz,
            obstacles_bgr=crop_bgr,
            path_xyz=display_path_xyz,
            target_center_3d=target_center_3d,
            target_bbox_list=target_bbox_list,
            az_deg=35.0,
            el_deg=28.0,
            width=width // 2,
            height=height,
            progress_ratio=r,
            current_pose_xyz=curr_pose,
            current_forward_xyz=curr_forward,
            camera_fov_deg=float(camera_fov_deg),
        )
        right = _draw_scene_image(
            obstacles_xyz=crop_xyz,
            obstacles_bgr=crop_bgr,
            path_xyz=display_path_xyz,
            target_center_3d=target_center_3d,
            target_bbox_list=target_bbox_list,
            az_deg=-35.0,
            el_deg=22.0,
            width=width // 2,
            height=height,
            progress_ratio=r,
            current_pose_xyz=curr_pose,
            current_forward_xyz=curr_forward,
            camera_fov_deg=float(camera_fov_deg),
        )
        frame = np.concatenate([left, right], axis=1)
        cv2.putText(frame, f"Trajectory Preview {int(r*100)}%", (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        frame_path = preferred_output_path(
            preview_frames_dir / f"frame_{i:06d}.jpg",
            compress_enabled=bool(image_cfg.get("enabled", True)),
        )
        save_bgr_image(frame, frame_path, cfg=image_cfg)
        frame_paths.append(str(frame_path.as_posix()))
        writer.write(frame)
    writer.release()
    _make_mp4_web_compatible(video_path, bitrate=str(bitrate or "10M"))
    web_variant = _export_web_mp4_variant(video_path, video_web_path, bitrate=str(web_bitrate or "2M"))
    _write_frame_manifest(
        preview_manifest_path,
        frames=frame_paths,
        fps=float(fps),
        width=int(width),
        height=int(height),
    )
    write_json(
        preview_index_map_path,
        {
            "fps": float(fps),
            "frame_count": int(len(sampled_idx)),
            "frames": [
                {
                    "out_frame": int(i),
                    "source_idx": int(src_idx),
                    "time_sec": float(i) / float(max(1.0, fps)),
                }
                for i, src_idx in enumerate(sampled_idx)
            ],
        },
    )
    return {
        "video": str(video_path.as_posix()),
        "video_web": str((web_variant or video_path).as_posix()),
        "video_frames_manifest": str(preview_manifest_path.as_posix()),
        "video_frame_index_map": str(preview_index_map_path.as_posix()),
    }


def generate_preview_assets_for_mission(
    *,
    files: dict[str, Any],
    summary: dict[str, Any],
    config: dict[str, Any],
    scene_root: Path,
    scene_id: str,
) -> dict[str, Any]:
    out_files = dict(files or {})
    out_summary = dict(summary or {})
    waypoints_path = Path(str(out_files.get("waypoints", "") or ""))
    if not waypoints_path.exists():
        raise FileNotFoundError("waypoints file missing, regenerate trajectory first")
    waypoints = np.load(waypoints_path).astype(np.float32)
    obstacles_xyz, obstacles_bgr = _try_load_obstacles_semantic(scene_root=scene_root, scene_id=scene_id, config=config)
    target_center_3d = out_summary.get("target_center_3d", None)
    target_bbox_list = out_summary.get("target_bbox_3d_list", None)
    stage3_cfg = _stage3_cfg(config)
    preview_fps = float(stage3_cfg.get("final_video_fps", 5) or 5)
    preview_frame_stride = int(stage3_cfg.get("final_video_frame_stride", 1) or 1)
    preview_speedup = float(stage3_cfg.get("final_video_speedup", 1.0) or 1.0)
    preview_pose_smooth_window = int(stage3_cfg.get("preview_pose_smooth_window", stage3_cfg.get("final_pose_smooth_window", 7)) or 7)
    preview_bitrate = str(stage3_cfg.get("preview_video_bitrate", "10M") or "10M").strip()
    web_video_bitrate = str(stage3_cfg.get("web_video_bitrate", "2M") or "2M").strip()
    camera_fov = float((config.get("camera", {}) or {}).get("fov", 72.0) or 72.0)
    source_pose_fps = float(out_summary.get("source_pose_fps", (config.get("camera", {}) or {}).get("fps", 10.0)) or 10.0)
    if not bool(stage3_cfg.get("preview_video_enabled", False)):
        return out_files
    extra = _generate_trajectory_video(
        out_dir=waypoints_path.parent,
        obstacles_xyz=obstacles_xyz,
        obstacles_bgr=obstacles_bgr,
        path_xyz=waypoints,
        segments=list((read_json_if_exists(Path(str(out_files.get("segments", "") or "")), default={}) or {}).get("segments", []) or []),
        target_center_3d=target_center_3d,
        target_bbox_list=target_bbox_list,
        source_pose_fps=float(source_pose_fps),
        fps=float(preview_fps),
        frame_stride=int(preview_frame_stride),
        speedup=float(preview_speedup),
        bitrate=str(preview_bitrate),
        web_bitrate=str(web_video_bitrate),
        pose_smooth_window=int(preview_pose_smooth_window),
        camera_fov_deg=float(camera_fov),
    )
    out_files.update(extra)
    return out_files


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Stage3 Review</title>
    <style>
        :root {
            --bg: #0f1115;
            --surface: #171c25;
            --surface-2: #1e2531;
            --text: #f3f4f6;
            --muted: #a7b0c0;
            --line: #2d3748;
            --accent: #4f46e5;
            --accent-2: #2563eb;
            --ok: #16a34a;
            --bad: #dc2626;
            --code: #f59e0b;
            --list-hover: #273142;
            --list-active: #334155;
        }
        body[data-theme='light'] {
            --bg: #f5f7fb;
            --surface: #ffffff;
            --surface-2: #f8fafc;
            --text: #0f172a;
            --muted: #475569;
            --line: #dbe3ef;
            --accent: #4f46e5;
            --accent-2: #2563eb;
            --ok: #16a34a;
            --bad: #dc2626;
            --code: #d97706;
            --list-hover: #eef2ff;
            --list-active: #e0e7ff;
        }
        * { box-sizing: border-box; }
        body {
            font-family: Inter, Arial, sans-serif;
            margin: 0;
            background: var(--bg);
            color: var(--text);
            height: 100vh;
            overflow: hidden;
        }
        .banner {
            height: 56px;
            background: var(--surface);
            border-bottom: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 14px;
            font-weight: 700;
        }
        .banner-actions { display: flex; gap: 8px; }
        .footer {
            height: 34px;
            background: var(--surface);
            border-top: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 12px;
            font-size: 12px;
            color: var(--muted);
        }
        .main {
            height: calc(100vh - 90px);
            display: grid;
            grid-template-columns: 320px minmax(0, 1fr) 420px;
            gap: 10px;
            padding: 10px;
            min-height: 0;
        }
        .card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; min-height: 0; }
        .sub-title { margin: 0; padding: 10px; border-bottom: 1px solid var(--line); font-size: 13px; color: var(--muted); }
        .left-list { display: flex; flex-direction: column; height: 100%; }
        .left-tools { display: flex; gap: 8px; padding: 8px 10px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); background: var(--surface); }
        .left-tools > * { flex: 1 1 0; min-width: 0; }
        .left-tools input, .left-tools select {
            width: 100%;
            background: var(--surface-2);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 6px 7px;
            font-size: 12px;
        }
        .list-box { overflow-y: auto; flex: 1; border-top: 1px solid var(--line); background: var(--surface-2); }
        .list-item { padding: 10px; border-bottom: 1px solid var(--line); cursor: pointer; font-size: 13px; }
        .list-item.active { background: var(--list-active); }
        .list-item:hover { background: var(--list-hover); }
        .status-chip { display: inline-block; border-radius: 10px; padding: 1px 7px; font-size: 11px; margin-left: 6px; }
        .status-pending { color: #e2e8f0; background: #475569; }
        .status-valid { color: #dcfce7; background: #15803d; }
        .status-invalid { color: #fee2e2; background: #b91c1c; }
        .status-pano_ready { color: #dbeafe; background: #1d4ed8; }
        .status-pano_confirmed { color: #dcfce7; background: #166534; }
        .status-pano_rejected { color: #fee2e2; background: #991b1b; }
        .status-video_ready { color: #fef3c7; background: #92400e; }
        .status-video_confirmed { color: #dcfce7; background: #15803d; }
        .status-video_rejected { color: #fee2e2; background: #b91c1c; }
        .status-final_ready { color: #e9d5ff; background: #6b21a8; }
        .group-title { padding: 8px 10px; color: var(--accent); font-weight: 700; border-bottom: 1px solid var(--line); background: var(--surface); }
        .middle-pane { display: grid; grid-template-rows: auto minmax(0, 1fr); gap: 10px; min-height: 0; align-content: start; }
        .view-card { display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }
        .view-title { margin: 0; padding: 8px 10px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 12px; }
        .plot-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            padding: 8px;
            background: var(--surface-2);
            min-height: 0;
        }
        .plot-cell { border: 1px solid var(--line); border-radius: 8px; background: var(--surface); padding: 8px; }
        .preview-img { width: 100%; height: 220px; object-fit: contain; border: 1px solid var(--line); border-radius: 6px; background: #f8fafc; }
        .video-box { width: 100%; height: 280px; border: 1px solid var(--line); border-radius: 6px; background: #000; }
        .video-fallback { width: 100%; height: 280px; border: 1px solid var(--line); border-radius: 6px; background: #000; object-fit: contain; display: none; }
        .video-grid { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
        .hint { color: var(--muted); font-size: 12px; }
        .attrs { height: 100%; overflow-y: auto; padding: 10px; background: var(--surface-2); }
        .row { margin: 8px 0; display: grid; grid-template-columns: 148px minmax(0, 1fr); gap: 10px; align-items: center; }
        .row label { color: var(--muted); font-size: 13px; }
        .row input, .row select, textarea {
            width: 100%;
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 6px 8px;
            font-size: 13px;
        }
        textarea { min-height: 72px; }
        pre {
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--line);
            padding: 8px;
            border-radius: 6px;
            max-height: 260px;
            overflow: auto;
            font-size: 12px;
        }
        button {
            margin: 4px 4px 4px 0;
            padding: 7px 12px;
            border: 1px solid transparent;
            border-radius: 7px;
            cursor: pointer;
            background: var(--surface);
            color: var(--text);
        }
        .primary { background: var(--accent); color: #fff; }
        .keep { background: var(--ok); color: #fff; }
        .drop { background: var(--bad); color: #fff; }
        .toolbar { margin-top: 8px; }
        code { color: var(--code); }
    </style>
</head>
<body>
    <div class='banner'>
        <div>
            Stage3 Workbench - <code>mission generation</code>
            <span style="margin-left:14px; font-size:12px;">
                <a href="/" style="color:inherit; margin-right:8px;">Missions</a>
                <a href="/review" style="color:inherit; margin-right:8px;">Review</a>
                <a href="/generate" style="color:inherit; margin-right:8px;">Generate</a>
                <a href="/dataset" style="color:inherit; margin-right:8px;">Dataset</a>
                <a href="/experiments" style="color:inherit; margin-right:8px;">Experiments</a>
                <a href="/results" style="color:inherit; margin-right:8px;">Results</a>
                <a href="/metrics" style="color:inherit;">Metrics</a>
            </span>
        </div>
        <div class='banner-actions'>
            <button onclick='toggleTheme()'>切换主题</button>
        </div>
    </div>
    <div class='main'>
        <div class='card left-list'>
            <p class='sub-title'>Stage3 实例列表</p>
            <div class='left-tools'>
                <input id='q' placeholder='搜索 instance_id/class...' />
            </div>
            <div class='left-tools'>
                <select id='statusFilter'>
                    <option value='all'>全部状态</option>
                    <option value='pending'>待筛选</option>
                    <option value='pano_ready'>全景待确认</option>
                    <option value='pano_confirmed'>全景已确认</option>
                    <option value='video_ready'>视频待确认</option>
                    <option value='video_confirmed'>视频已确认</option>
                    <option value='final_ready'>最终数据已生成</option>
                    <option value='valid'>valid</option>
                    <option value='invalid'>invalid</option>
                </select>
            </div>
            <div class='list-box' id='instances-list'>Loading...</div>
        </div>

        <div class='middle-pane'>
            <div class='card view-card'>
                <p class='view-title'>第一步：点云中轨迹全景图（先确认）</p>
                <div class='plot-grid'>
                    <div class='plot-cell'>
                        <div class='hint'>Panorama Left</div>
                        <img id='pano-left' class='preview-img' alt='panorama_left' />
                    </div>
                    <div class='plot-cell'>
                        <div class='hint'>Panorama Right</div>
                        <img id='pano-right' class='preview-img' alt='panorama_right' />
                    </div>
                </div>
                <div class='card-pad'>
                    <div id='res-status' class='hint'>请选择实例后生成轨迹并先确认全景图</div>
                </div>
            </div>

            <div class='card view-card'>
                <p class='view-title'>第二步：预览视频 + 最终任务视频（右侧）</p>
                <div class='card-pad'>
                    <div class='video-grid'>
                        <div>
                            <div class='hint'>点云预览视频</div>
                            <video id='traj-video' class='video-box' controls></video>
                            <img id='traj-video-fallback' class='video-fallback' alt='traj-fallback' />
                        </div>
                        <div>
                            <div class='hint'>最终任务RGB视频 (720P@24FPS)</div>
                            <video id='final-video' class='video-box' controls></video>
                            <img id='final-video-fallback' class='video-fallback' alt='final-fallback' />
                        </div>
                    </div>
                    <div class='hint' style='margin-top:8px;'>目标地标时间区间（按帧 / 按秒）</div>
                    <pre id='final-info'>{}</pre>
                    <pre id='res-json'>{}</pre>
                </div>
            </div>
        </div>

        <div class='card'>
            <div class='attrs' id='controls' style='display:none;'>
                <div class='row'><label>instance_id</label><input id='lbl-id' disabled /></div>
                <div class='row'><label>class_name</label><input id='lbl-class' disabled /></div>
                <div class='row'><label>当前状态</label><input id='lbl-status' disabled /></div>
                <div class='row'><label>行为序列</label><input id='inp-behaviors' value='B5, B1, B10' /></div>
                <div class='toolbar'>
                    <button class='primary' onclick='generateTrajectory()'>生成轨迹</button>
                    <button onclick='confirmPanorama(true)'>全景通过</button>
                    <button onclick='confirmPanorama(false)'>全景驳回</button>
                    <button class='primary' onclick='generateVideo()'>生成视频</button>
                    <button onclick='confirmVideo(true)'>视频通过</button>
                    <button onclick='confirmVideo(false)'>视频驳回</button>
                    <button class='primary' onclick='generateFinalTask()'>生成最终任务数据</button>
                </div>

                <hr style='border-color: var(--line); border-style: solid; border-width: 1px 0 0 0;' />
                <div class='row'><label>标签(label)</label><input id='inp-label' placeholder='orbit_good / collision_risk' /></div>
                <div class='row'><label>备注(note)</label><input id='inp-note' placeholder='人工筛选备注' /></div>
                <div class='toolbar'>
                    <button class='keep' onclick="saveDecision('valid')">通过 valid</button>
                    <button class='drop' onclick="saveDecision('invalid')">驳回 invalid</button>
                    <button onclick="saveDecision('pending')">回退 pending</button>
                </div>
                <p id='save-status' class='hint'></p>
            </div>
        </div>
    </div>
    <div class='footer'>
        <div>实时写入：<code>review_index.json</code><code style='margin-left:12px'>review_log.jsonl</code></div>
        <div id='opStatus'>就绪</div>
    </div>

    <script>
        let currentInstance = null;
        let lastResult = null;
        let allInstances = [];
        let progressTimer = null;
        let progressPollMs = 3000;
        let frameFallbackTimers = {};
        const ENABLE_FRAME_FALLBACK = false;

        function stopFrameFallback(videoId){
            const fb = document.getElementById(`${videoId}-fallback`);
            const vd = document.getElementById(videoId);
            if(frameFallbackTimers[videoId]){
                clearInterval(frameFallbackTimers[videoId]);
                delete frameFallbackTimers[videoId];
            }
            if(fb){
                fb.style.display = 'none';
                fb.removeAttribute('src');
            }
            if(vd){
                vd.style.display = '';
            }
        }

        function startFrameFallback(videoId, manifestPath){
            if(!ENABLE_FRAME_FALLBACK) return;
            const fb = document.getElementById(`${videoId}-fallback`);
            const vd = document.getElementById(videoId);
            const manifestUrl = assetUrl(manifestPath || '');
            if(!fb || !vd || !manifestUrl) return;

            fetch(manifestUrl).then(r=>r.json()).then(meta=>{
                const framesRaw = Array.isArray(meta.frames) ? meta.frames : [];
                const frames = framesRaw.map(p => {
                    const raw = String(p || '');
                    if(!raw) return '';
                    if(raw.startsWith('/')) return assetUrl(raw);
                    return new URL(raw, manifestUrl).toString();
                }).filter(Boolean);
                if(!frames.length) return;
                const fps = Math.max(1, Number(meta.fps || 24));
                let idx = 0;
                vd.style.display = 'none';
                fb.style.display = 'block';
                fb.src = frames[0];
                if(frameFallbackTimers[videoId]) clearInterval(frameFallbackTimers[videoId]);
                frameFallbackTimers[videoId] = setInterval(()=>{
                    idx = (idx + 1) % frames.length;
                    fb.src = frames[idx];
                }, Math.max(20, Math.round(1000 / fps)));
            }).catch(()=>{});
        }

        function setVideoSource(id, path, fallbackManifestPath='') {
            const el = document.getElementById(id);
            if(!el) return;
            stopFrameFallback(id);
            const url = assetUrl(path || '');
            if(!url){
                el.removeAttribute('src');
                try { el.pause(); } catch(_e) {}
                try { el.load(); } catch(_e) {}
                if(ENABLE_FRAME_FALLBACK && fallbackManifestPath){
                    startFrameFallback(id, fallbackManifestPath);
                }
                return;
            }
            const prev = el.getAttribute('src') || '';
            if(prev !== url){
                el.setAttribute('src', url);
            }
            const fallbackOnce = () => {
                if(ENABLE_FRAME_FALLBACK && fallbackManifestPath){
                    startFrameFallback(id, fallbackManifestPath);
                }
            };
            el.onerror = null;
            try { el.load(); } catch(_e) {}
            if(ENABLE_FRAME_FALLBACK){
                setTimeout(()=>{
                    if(el.readyState < 2 && fallbackManifestPath){
                        fallbackOnce();
                    }
                }, 4500);
            }
        }

        function renderFinalIntervals(meta){
            const presence = (meta && meta.target_presence) ? meta.target_presence : {};
            const frameIntervals = Array.isArray(presence.intervals_frame) ? presence.intervals_frame : [];
            const secIntervals = Array.isArray(presence.intervals_sec) ? presence.intervals_sec : [];
            return {
                intervals_frame: frameIntervals,
                intervals_sec: secIntervals,
            };
        }

        function assetUrl(path){
            if(!path) return '';
            if(path.startsWith('/artifacts/')) return path;
            if(path.startsWith('/artifact_by_id/')) return path;
            if(path.startsWith('/')){
                const sceneTag = '/scene_data/';
                const si = path.indexOf(sceneTag);
                if(si >= 0){
                    const rel = path.slice(si + sceneTag.length).replace(/^\\/+/, '');
                    return `/artifacts/${rel}`;
                }
                const ti = path.indexOf('/trajectory/');
                if(ti >= 0){
                    const rel = path.slice(ti + 1).replace(/^\\/+/, '');
                    return `/artifacts/${rel}`;
                }
            }
            return path;
        }

        function toggleTheme() {
            const body = document.body;
            const next = body.dataset.theme === 'light' ? 'dark' : 'light';
            body.dataset.theme = next === 'dark' ? '' : 'light';
        }

        function loadInstances() {
            const q = document.getElementById('q').value || '';
            const status = document.getElementById('statusFilter').value || 'all';
            const url = `/api/instances?q=${encodeURIComponent(q)}&status=${encodeURIComponent(status)}`;
            fetch(url).then(r=>r.json()).then(data => {
                allInstances = data;
                renderInstanceList();
                if(currentInstance && currentInstance.instance_id){
                    const matched = allInstances.find(it => String(it.instance_id || '') === String(currentInstance.instance_id || ''));
                    if(matched){
                        const left = document.getElementById('instances-list');
                        const rows = left ? Array.from(left.querySelectorAll('.list-item')) : [];
                        const row = rows.find(el => (el.querySelector('b')?.innerText || '') === String(matched.instance_id || ''));
                        if(row){
                            selectInstance(matched, row);
                        }else{
                            currentInstance = matched;
                        }
                    }
                }
                const op = document.getElementById('opStatus');
                if(op) op.innerText = `已加载实例: ${allInstances.length}`;
            });
        }

        function loadProgress() {
            fetch('/api/progress').then(r=>r.json()).then(p => {
                const op = document.getElementById('opStatus');
                if(!op) return;
                if(p && p.active){
                    op.innerText = `[进度] ${p.message || '-'} (${p.done || 0}/${p.total || 0})`;
                    progressPollMs = 1000;
                } else {
                    progressPollMs = 10000;
                }
            }).catch(()=>{
                progressPollMs = 12000;
            }).finally(()=>{
                if(progressTimer) clearTimeout(progressTimer);
                progressTimer = setTimeout(loadProgress, progressPollMs);
            });
        }

        function renderInstanceList() {
            const container = document.getElementById('instances-list');
            container.innerHTML = '';
            let lastClass = '__none__';
            allInstances.forEach(inst => {
                const cls = `${inst.class_id ?? '-'}|${inst.class_name || '-'}`;
                if(cls !== lastClass){
                    const g = document.createElement('div');
                    g.className = 'group-title';
                    g.innerText = `[${inst.class_id ?? '-'}] ${inst.class_name || '-'}`;
                    container.appendChild(g);
                    lastClass = cls;
                }
                const div = document.createElement('div');
                div.className = 'list-item';
                const statusClass = `status-chip status-${inst.status || 'pending'}`;
                div.innerHTML = `
                    <div><b>${inst.instance_id}</b></div>
                    <div><span class="${statusClass}">${inst.status || 'pending'}</span> <span class='hint'>点数: ${inst.point_count || 0}</span></div>
                    <div class="hint">traj: ${inst.latest_traj_id || '-'} | label: ${inst.label || '-'}</div>
                `;
                div.onclick = () => selectInstance(inst, div);
                container.appendChild(div);
            });
        }

        document.getElementById('q').addEventListener('input', loadInstances);
        document.getElementById('statusFilter').addEventListener('change', loadInstances);
        loadInstances();
        loadProgress();

        function selectInstance(inst, element) {
            document.querySelectorAll('.list-item').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
            currentInstance = inst;
            document.getElementById('controls').style.display = 'block';
            document.getElementById('lbl-id').value = inst.instance_id;
            document.getElementById('lbl-class').value = inst.class_name || '';
            document.getElementById('lbl-status').value = inst.status || 'pending';
            document.getElementById('inp-label').value = inst.label || '';
            document.getElementById('inp-note').value = inst.note || '';
            document.getElementById('res-json').innerText = '{}';
            document.getElementById('final-info').innerText = '{}';
            document.getElementById('res-status').innerText = '已选择实例，先生成轨迹并确认全景，再生成视频。';
            const urls = inst.asset_urls || {};
            document.getElementById('pano-left').src = assetUrl(urls.panorama_left || '');
            document.getElementById('pano-right').src = assetUrl(urls.panorama_right || '');
            setVideoSource('traj-video', urls.video_web || urls.video || '', urls.video_frames_manifest || '');
            setVideoSource('final-video', urls.final_video_web || urls.final_video || '', urls.final_frames_manifest || '');
            if(urls.final_metadata){
                fetch(assetUrl(urls.final_metadata)).then(r=>r.json()).then(meta=>{
                    document.getElementById('final-info').innerText = JSON.stringify(renderFinalIntervals(meta), null, 2);
                }).catch(()=>{ document.getElementById('final-info').innerText = '{}'; });
            }
        }

        function generateTrajectory() {
            if(!currentInstance) return;
            document.getElementById('res-status').innerText = "生成中...";
            document.getElementById('save-status').innerText = "";
            document.getElementById('res-json').innerText = "Working...";
            
            const behaviors = document.getElementById('inp-behaviors').value;
            
            fetch('/api/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ instance_id: currentInstance.instance_id, behavior_sequence: behaviors })
            }).then(r=>r.json()).then(data => {
                lastResult = data;
                if(data.error) {
                    document.getElementById('res-status').innerText = "生成失败";
                    document.getElementById('res-json').innerText = JSON.stringify(data, null, 2);
                    const op = document.getElementById('opStatus');
                    if(op) op.innerText = '生成失败';
                } else {
                    const kin = data.summary.kinematics_checked === false ? "SKIP" : (data.summary.kinematics_feasible ? "Pass" : "FAIL");
                    const col = data.summary.collision_free ? "Pass" : "FAIL";
                    document.getElementById('res-status').innerText = `轨迹已生成。Kinematics: ${kin} | Collision: ${col}。请先确认全景图。`;
                    document.getElementById('res-json').innerText = JSON.stringify(data, null, 2);
                    const urls = data.asset_urls || {};
                    document.getElementById('pano-left').src = assetUrl(urls.panorama_left || '');
                    document.getElementById('pano-right').src = assetUrl(urls.panorama_right || '');
                    setVideoSource('traj-video', urls.video_web || urls.video || '', urls.video_frames_manifest || '');
                    const op = document.getElementById('opStatus');
                    if(op) op.innerText = `生成完成: ${data.traj_id || '-'}`;
                    loadInstances();
                }
            });
        }

        function confirmPanorama(approved) {
            if(!currentInstance) return;
            fetch('/api/confirm_pano', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({instance_id: currentInstance.instance_id, approved: !!approved})
            }).then(r=>r.json()).then(res => {
                document.getElementById('save-status').innerText = approved ? '全景已通过' : '全景已驳回';
                loadInstances();
            });
        }

        function generateVideo() {
            if(!currentInstance) return;
            fetch('/api/generate_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({instance_id: currentInstance.instance_id})
            }).then(r=>r.json()).then(data => {
                if(data.error){
                    document.getElementById('save-status').innerText = `视频生成失败: ${data.error}`;
                    return;
                }
                const urls = data.asset_urls || {};
                setVideoSource('traj-video', urls.video_web || urls.video || '', urls.video_frames_manifest || '');
                document.getElementById('save-status').innerText = '视频已生成，请确认';
                loadInstances();
            });
        }

        function confirmVideo(approved) {
            if(!currentInstance) return;
            fetch('/api/confirm_video', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({instance_id: currentInstance.instance_id, approved: !!approved})
            }).then(r=>r.json()).then(res => {
                document.getElementById('save-status').innerText = approved ? '视频已通过' : '视频已驳回';
                loadInstances();
            });
        }

        function generateFinalTask() {
            if(!currentInstance) return;
            document.getElementById('save-status').innerText = '最终任务数据生成中...';
            progressPollMs = 500;
            if(progressTimer) clearTimeout(progressTimer);
            loadProgress();
            fetch('/api/generate_final_task', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({instance_id: currentInstance.instance_id})
            }).then(r=>r.json()).then(data => {
                if(data.error){
                    document.getElementById('save-status').innerText = `最终数据生成失败: ${data.error}`;
                    return;
                }
                const urls = data.asset_urls || {};
                setVideoSource('final-video', urls.final_video_web || urls.final_video || '', urls.final_frames_manifest || '');
                if(urls.final_metadata){
                    fetch(assetUrl(urls.final_metadata)).then(r=>r.json()).then(meta=>{
                        document.getElementById('final-info').innerText = JSON.stringify(renderFinalIntervals(meta), null, 2);
                    }).catch(()=>{});
                }
                document.getElementById('save-status').innerText = '最终任务视频与数据已生成';
                loadInstances();
            });
        }
        
        function saveDecision(decision) {
            if(!lastResult || !currentInstance) return;
            const label = document.getElementById('inp-label').value || '';
            const note = document.getElementById('inp-note').value || '';
            fetch('/api/save_decision', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    instance_id: currentInstance.instance_id,
                    traj_id: lastResult.traj_id,
                    decision: decision,
                    label: label,
                    note: note,
                    files: lastResult.files,
                    summary: lastResult.summary
                })
            }).then(r=>r.json()).then(res => {
                document.getElementById('save-status').innerText = `已保存: ${decision}`;
                loadInstances();
                const op = document.getElementById('opStatus');
                if(op) op.innerText = `已写入筛选结果: ${decision}`;
            });
        }
    </script>
</body>
</html>
"""

def run_web(args: argparse.Namespace, config: dict[str, Any], logger: StageLogger) -> None:
    if Flask is None:
        raise ImportError("Flask is required for Stage3 web mode")

    app = Flask("stage3_web")
    scene_id = _resolve_scene_id(args, config)
    scene_root, trajectory_root, review_root = _resolve_stage3_dirs(config, scene_id)
    config_path = Path(str(args.config)).resolve()
    task_cfg = config.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim")).lower().strip()
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage3",
        step="web",
        message="web_started",
        payload=build_unified_stage_event(
            stage="stage3",
            step="web",
            scene_id=scene_id,
            engine=engine_name,
            status="started",
            extra={"trajectory_root": str(trajectory_root.as_posix())},
        ),
    )

    if args.instances_json:
        _, _, instances = _resolve_source_instances(scene_root=scene_root, scene_id=scene_id, args=args, config=config)
    else:
        review_dir_name = resolve_output_dir_name(config, key="stage2_review_dir", default="landmarks_review")
        valid_path = scene_root / review_dir_name / f"{scene_id}.valid_instances.json"
        valid_payload = read_json_if_exists(valid_path, default={})
        if not isinstance(valid_payload, dict):
            raise FileNotFoundError(f"invalid valid_instances payload: {valid_path}")
        instances = [it for it in list(valid_payload.get("valid_instances", []) or []) if isinstance(it, dict)]
        instances = [it for it in instances if str(it.get("annotation_status", "") or "").strip().lower() == "labeled"]
        instances = [it for it in instances if str(it.get("review_action", "") or "").strip().lower() in {"keep", ""}]
        instances = [it for it in instances if str(it.get("landmark_description", "") or it.get("description", "") or "").strip()]
        if not instances:
            raise FileNotFoundError(f"stage2 passed instances not found: {valid_path}")
    
    asset_alias_map: dict[str, str] = {}
    register_stage3_task_routes(
        app,
        default_config=config,
        scene_id=scene_id,
        engine=engine_name,
        config_path=config_path,
    )

    def _register_asset_alias(path: Path) -> str:
        resolved = str(path.resolve())
        token = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
        asset_alias_map[token] = resolved
        return token
        
    review_index_path, review_log_path = _review_files(review_root=review_root, scene_id=scene_id)
    progress_state: dict[str, Any] = {"active": False, "done": 0, "total": 0, "message": "idle"}

    def _asset_url(abs_path: str) -> str:
        if not abs_path:
            return ""
        try:
            p = Path(str(abs_path)).resolve()
            rel = p.relative_to(trajectory_root.resolve())
            return f"/artifacts/{rel.as_posix()}"
        except Exception:
            p = Path(str(abs_path)).resolve()
            if p.exists() and p.is_file():
                return f"/artifact_by_id/{_register_asset_alias(p)}"
            return ""

    def _build_asset_urls(files: dict[str, Any] | None) -> dict[str, str]:
        files = files if isinstance(files, dict) else {}
        return {
            "panorama_left": _asset_url(str(files.get("panorama_left", "") or "")),
            "panorama_right": _asset_url(str(files.get("panorama_right", "") or "")),
            "video_web": _asset_url(str(files.get("video_web", "") or "")),
            "video": _asset_url(str(files.get("video", "") or "")),
            "video_frames_manifest": _asset_url(str(files.get("video_frames_manifest", "") or "")),
            "video_frame_index_map": _asset_url(str(files.get("video_frame_index_map", "") or "")),
            "final_video_web": _asset_url(str(files.get("final_video_web", "") or "")),
            "final_video": _asset_url(str(files.get("final_video", "") or "")),
            "final_video_marked_web": _asset_url(str(files.get("final_video_marked_web", "") or "")),
            "final_video_marked": _asset_url(str(files.get("final_video_marked", "") or "")),
            "final_metadata": _asset_url(str(files.get("final_metadata", "") or "")),
            "final_frames_manifest": _asset_url(str(files.get("final_frames_manifest", "") or "")),
            "final_frame_index_map": _asset_url(str(files.get("final_frame_index_map", "") or "")),
        }

    def _generate_preview_assets(files: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
        return generate_preview_assets_for_mission(
            files=files,
            summary=summary,
            config=config,
            scene_root=scene_root,
            scene_id=scene_id,
        )

    def _upsert_item(instance_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        review_index = _load_review_index(review_index_path)
        if review_index.get("scene_id", "") in {"", None}:
            review_index["scene_id"] = scene_id
        items = review_index.get("items", {})
        if not isinstance(items, dict):
            items = {}
        item = items.get(instance_id, {})
        if not isinstance(item, dict):
            item = {}
        items[instance_id] = item
        review_index["items"] = items
        return review_index, items, item

    def _resolve_selected_instances(selected_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        normalized = [_normalize_landmark_item(item) for item in instances if isinstance(item, dict)]
        lookup = {str(item.get("instance_id", "") or ""): item for item in normalized}
        chosen = lookup.get(selected_ids[0], None) if selected_ids else None
        if chosen is None:
            chosen = _auto_pick_landmark(normalized)
        secondary = [lookup[item_id] for item_id in selected_ids[1:] if item_id in lookup]
        return chosen, secondary, lookup

    def _build_generation_schema(payload: dict[str, Any]) -> dict[str, Any]:
        selected_ids = [str(x).strip() for x in list(payload.get("instance_ids", []) or []) if str(x).strip()]
        instance_id = str(payload.get("instance_id", "") or "").strip()
        if not selected_ids and instance_id:
            selected_ids = [instance_id]
        chosen, secondary, _ = _resolve_selected_instances(selected_ids)
        mission_mode = str(payload.get("mission_mode", "single-landmark") or "single-landmark").strip()
        if len(selected_ids) > 1:
            mission_mode = "multi-landmark"
        mission_key = str(payload.get("mission_type", "") or "").strip() or None
        landmark_set_map = dict(payload.get("landmark_set_map", {}) or {}) if isinstance(payload.get("landmark_set_map", {}), dict) else {}
        auto_set_candidates = [str(x).strip() for x in list(payload.get("auto_set_candidates", []) or []) if str(x).strip()]
        auto_set_rule = str(payload.get("auto_set_rule", "heuristic") or "heuristic").strip().lower()
        set_profiles = payload.get("set_profiles", None)
        if not isinstance(set_profiles, dict):
            set_profiles = _load_stage3_behavior_defaults()
        generation_kind = _normalize_generation_kind(payload.get("generation_kind", "auto"))
        behavior_sequence = str(payload.get("behavior_sequence", "") or "").strip()
        explicit_sequence = [b.strip() for b in behavior_sequence.split(",") if b.strip()] if behavior_sequence else None
        if mission_mode == "multi-landmark":
            selected_landmarks = [chosen, *list(secondary or [])]
            set_spec, resolved_landmark_set_map = _build_multi_landmark_composite_set(
                selected_landmarks=selected_landmarks,
                landmark_set_map=landmark_set_map,
                allowed_set_types=auto_set_candidates or _single_landmark_component_set_keys(),
                auto_rule=auto_set_rule,
                seed=int(payload.get("seed", getattr(args, "seed", 42)) or getattr(args, "seed", 42)),
                explicit_multi_set_key=mission_key,
            )
            landmark_set_map = dict(resolved_landmark_set_map)
        else:
            set_spec = _select_set_template(
                landmark=chosen,
                set_type=mission_key,
                mode=mission_mode,
                allowed_set_types=auto_set_candidates,
                auto_rule=auto_set_rule,
                seed=int(payload.get("seed", getattr(args, "seed", 42)) or getattr(args, "seed", 42)),
            )
        set_key = str(set_spec.get("set_key", "") or "")
        set_profile = _extract_set_profile(set_profiles, set_key)
        if generation_kind == "auto":
            if explicit_sequence:
                generation_kind = "atomic-only"
            elif str(set_spec.get("scope", "single-landmark") or "single-landmark") == "multi-landmark":
                generation_kind = "composite-driven"
            elif str(set_spec.get("set_key", "") or "").startswith("atomic_"):
                generation_kind = "atomic-only"
            else:
                generation_kind = "composite-driven"
        prof_kind = str(set_profile.get("generation_kind", "") or "").strip().lower()
        if str(payload.get("generation_kind", "auto") or "auto").strip().lower() == "auto" and prof_kind in {"atomic-only", "composite-driven"}:
            generation_kind = prof_kind
        if not explicit_sequence:
            prof_seq = [str(x).strip() for x in list(set_profile.get("behavior_sequence", []) or []) if str(x).strip()]
            if prof_seq:
                explicit_sequence = prof_seq
        start_pos = _build_start_pos(
            center_3d=list(chosen.get("center_3d", [0.0, 0.0, 0.0])),
            bbox_list=list(chosen.get("bbox_3d", [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0])),
        )
        obstacles_xyz, _ = _try_load_obstacles_semantic(scene_root=scene_root, scene_id=scene_id, config=config)
        keepout_boxes = _build_keepout_boxes(
            [it for it in instances if isinstance(it, dict)],
            margin_xy=max(0.2, float(_load_stage3_behavior_shared().get("safety_distance_m", 2.0) or 2.0) * 0.25),
            margin_z=max(0.1, float(_load_stage3_behavior_shared().get("safety_distance_m", 2.0) or 2.0) * 0.15),
        )
        set_instance, element_instances, _ = _build_element_instances(
            set_spec=set_spec,
            primary=chosen,
            secondary=secondary,
            start_pos=start_pos,
            seed=int(payload.get("seed", getattr(args, "seed", 42)) or getattr(args, "seed", 42)),
            generation_kind=generation_kind,
            explicit_sequence=explicit_sequence,
            param_overrides=set_profile.get("element_param_overrides", payload.get("element_param_overrides", None)),
            auto_param_rules=set_profile.get("element_auto_rules", payload.get("element_auto_rules", None)),
            adaptive_sequential_params=bool(payload.get("adaptive_sequential_params", True)),
            allow_interleave_repeat=bool(set_profile.get("allow_interleave_repeat", payload.get("allow_interleave_repeat", False))),
            max_total_elements=int(set_profile.get("max_total_elements", payload.get("max_total_elements", 0)) or 0),
            safety_distance_m=float(_load_stage3_behavior_shared().get("safety_distance_m", 2.0) or 2.0),
            obstacles_xyz=obstacles_xyz,
            keepout_boxes=keepout_boxes,
            preview_points_per_element=int(set_spec.get("preview_points_per_element", 40) or 40),
        )
        step_rows = []
        for idx, row in enumerate(element_instances):
            spec = dict((ELEMENT_LIBRARY.get(str(row.get("element_class", "") or ""), {}) or {}).get("params", {}) or {})
            step_rows.append(
                {
                    "step_index": idx,
                    "element_class": str(row.get("element_class", "") or ""),
                    "element_display_name": str(row.get("element_display_name", "") or ""),
                    "target_instance_id": str(row.get("target_instance_id", "") or ""),
                    "params": dict(row.get("params", {}) or {}),
                    "param_specs": spec,
                }
            )
        return {
            "selected_instance_ids": selected_ids,
            "mission_mode": mission_mode,
            "generation_kind": generation_kind,
            "auto_set_rule": auto_set_rule,
            "auto_set_candidates": list(auto_set_candidates or _matching_set_keys_for_mode(mission_mode)),
            "set_spec": dict(set_spec),
            "set_instance": dict(set_instance or {}),
            "landmark_set_map": dict(landmark_set_map),
            "steps": step_rows,
        }

    @app.route("/")
    def index():
        if redirect is None:
            return jsonify({"error": "redirect unavailable"}), 500
        return redirect("/missions", code=302)

    @app.route("/artifacts/<path:relpath>")
    def serve_artifact(relpath: str):
        if send_file is None:
            return jsonify({"error": "send_file unavailable"}), 500
        root = trajectory_root.resolve()
        fp = (root / relpath).resolve()
        if root not in fp.parents or not fp.exists() or not fp.is_file():
            return jsonify({"error": "artifact_not_found"}), 404
        if fp.suffix.lower() == ".mp4":
            _ensure_mp4_web_playable(fp)
        return send_file(str(fp))

    @app.route("/artifact_by_id/<token>")
    def serve_artifact_by_id(token: str):
        if send_file is None:
            return jsonify({"error": "send_file unavailable"}), 500
        path_str = asset_alias_map.get(str(token), "")
        if not path_str:
            return jsonify({"error": "artifact_not_found"}), 404
        fp = Path(path_str).resolve()
        if not fp.exists() or not fp.is_file():
            return jsonify({"error": "artifact_not_found"}), 404
        if fp.suffix.lower() == ".mp4":
            _ensure_mp4_web_playable(fp)
        return send_file(str(fp))

    @app.route("/api/instances")
    def get_instances():
        q = str(request.args.get("q", "") or "").strip().lower()
        status_filter = str(request.args.get("status", "all") or "all").strip().lower()
        review_index = _load_review_index(review_index_path)
        rows = _build_instance_rows(instances=instances, review_items=review_index.get("items", {}))
        for row in rows:
            row["asset_urls"] = _build_asset_urls(row.get("latest_files", {}))
        if status_filter in {"pending", "valid", "invalid", "pano_ready", "pano_confirmed", "video_ready", "video_confirmed", "pano_rejected", "video_rejected", "final_ready"}:
            rows = [row for row in rows if row.get("status") == status_filter]
        if q:
            rows = [
                row
                for row in rows
                if q in str(row.get("instance_id", "")).lower()
                or q in str(row.get("class_name", "")).lower()
                or q in str(row.get("label", "")).lower()
                or q in str(row.get("traj_status", "")).lower()
            ]
        return jsonify(rows)

    @app.route("/api/stage3_mission_catalog")
    def get_stage3_mission_catalog():
        behavior_defaults = _load_stage3_behavior_defaults()
        mission_rows = []
        for key, spec in SET_LIBRARY.items():
            step_rows = [dict(x) for x in list(spec.get("element_steps", []) or []) if isinstance(x, dict)]
            sequence = [str(x) for x in list(spec.get("element_template", []) or [])]
            if step_rows:
                sequence = [str(row.get("element_class", "") or "") for row in step_rows if str(row.get("element_class", "") or "").strip()]
            mission_rows.append(
                {
                    "mission_key": key,
                    "mission_family": "flight_set",
                    "mission_type": str(spec.get("display_name", "") or ""),
                    "mission_subtype": str(key),
                    "service_scenario": str(spec.get("scope", "") or ""),
                    "sequence": sequence,
                    "element_steps": step_rows,
                    "mode_steps": [{"mode_key": str(x), "target": "primary"} for x in sequence],
                    "description": str(spec.get("description", "") or ""),
                    "generation_notes": f"scope={spec.get('scope', '')} allow_revisit={spec.get('allow_revisit', False)}",
                    "multi_landmark_component": bool(spec.get("multi_landmark_component", False)),
                }
            )
        behavior_rows = []
        for key in sorted(ELEMENT_LIBRARY):
            spec = dict(ELEMENT_LIBRARY.get(key, {}) or {})
            row = {"behavior_id": key}
            row.update(spec)
            behavior_rows.append(row)
        return jsonify(
            {
                "missions": mission_rows,
                "behaviors": behavior_rows,
                "families": sorted({row["mission_family"] for row in mission_rows if row["mission_family"]}),
                "service_scenarios": sorted({row["service_scenario"] for row in mission_rows if row["service_scenario"]}),
                "behavior_defaults": behavior_defaults,
                "behavior_defaults_path": str(COMMON_STAGE_CONFIG_PATH),
            }
        )

    @app.route("/api/stage3_behavior_defaults", methods=["POST"])
    def save_stage3_behavior_defaults():
        payload = request.json or {}
        set_key = str(payload.get("set_key", "") or "").strip()
        profile = dict(payload.get("profile", {}) or {}) if isinstance(payload.get("profile", {}), dict) else {}
        if not set_key:
            return jsonify({"error": "set_key_required"}), 400
        existing = _load_stage3_behavior_defaults()
        existing[set_key] = profile
        path = _write_stage3_behavior_defaults(existing)
        return jsonify({"ok": True, "config_path": str(path), "set_key": set_key})

    @app.route("/api/mission_generation_schema", methods=["POST"])
    def mission_generation_schema():
        data = request.json or {}
        try:
            return jsonify({"ok": True, "schema": _build_generation_schema(data)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/mission_history")
    def mission_history_api():
        selected_raw = str(request.args.get("instance_ids", "") or "").strip()
        selected_ids = [item.strip() for item in selected_raw.split(",") if item.strip()]
        review_index = _load_review_index(review_index_path)
        rows = _collect_mission_history(review_index.get("items", {}), selected_ids)
        for row in rows:
            row["asset_urls"] = _build_asset_urls(row.get("files", {}))
        return jsonify({"rows": rows})

    @app.route("/api/mission_delete", methods=["POST"])
    def mission_delete_api():
        payload = request.json or {}
        traj_id = str(payload.get("traj_id", "") or "").strip()
        if not traj_id:
            return jsonify({"error": "traj_id is required"}), 400
        delete_artifacts = bool(payload.get("delete_artifacts", True))
        review_index = _load_review_index(review_index_path)
        items = review_index.get("items", {})
        if not isinstance(items, dict):
            items = {}
        changed_any = False
        artifact_result: dict[str, Any] = {"traj_id": traj_id, "deleted": False, "reason": "skipped"}
        for instance_id, rec in list(items.items()):
            if not isinstance(rec, dict):
                continue
            if _remove_mission_from_record(rec, traj_id=traj_id):
                changed_any = True
                items[instance_id] = rec
        if delete_artifacts:
            artifact_result = _delete_mission_artifacts(trajectory_root, traj_id=traj_id)
        review_index["items"] = items
        _write_review_index(review_index_path, review_index)
        append_jsonl(
            review_log_path,
            {
                "event": "delete_mission",
                "scene_id": scene_id,
                "time": _utc_now_iso(),
                "traj_id": traj_id,
                "delete_artifacts": bool(delete_artifacts),
                "changed": bool(changed_any),
                "artifact_result": artifact_result,
            },
        )
        return jsonify({"ok": True, "traj_id": traj_id, "changed": bool(changed_any), "artifact_result": artifact_result})

    @app.route("/api/mission_clear", methods=["POST"])
    def mission_clear_api():
        payload = request.json or {}
        selected_ids = [str(x).strip() for x in list(payload.get("instance_ids", []) or []) if str(x).strip()]
        delete_artifacts = bool(payload.get("delete_artifacts", True))
        review_index = _load_review_index(review_index_path)
        items = review_index.get("items", {})
        if not isinstance(items, dict):
            items = {}
        target_instance_ids = set(selected_ids) if selected_ids else set(items.keys())
        deleted_traj_ids: set[str] = set()
        artifact_results: list[dict[str, Any]] = []
        for instance_id, rec in list(items.items()):
            if instance_id not in target_instance_ids or not isinstance(rec, dict):
                continue
            history = list(rec.get("mission_history", []) or []) if isinstance(rec.get("mission_history", []), list) else []
            for row in history:
                traj_id = str((row or {}).get("traj_id", "") or "").strip()
                if traj_id:
                    deleted_traj_ids.add(traj_id)
            rec["mission_history"] = []
            _reset_instance_record_from_history(rec)
            items[instance_id] = rec
        if delete_artifacts:
            for traj_id in deleted_traj_ids:
                artifact_results.append(_delete_mission_artifacts(trajectory_root, traj_id=traj_id))
        review_index["items"] = items
        _write_review_index(review_index_path, review_index)
        append_jsonl(
            review_log_path,
            {
                "event": "clear_missions",
                "scene_id": scene_id,
                "time": _utc_now_iso(),
                "instance_ids": sorted(target_instance_ids),
                "deleted_traj_ids": sorted(deleted_traj_ids),
                "delete_artifacts": bool(delete_artifacts),
                "artifact_results": artifact_results,
            },
        )
        return jsonify({"ok": True, "instance_ids": sorted(target_instance_ids), "deleted_traj_ids": sorted(deleted_traj_ids), "artifact_results": artifact_results})

    @app.route("/api/progress")
    def get_progress():
        return jsonify(progress_state)

    @app.route("/api/reviews")
    def list_reviews():
        review_index = _load_review_index(review_index_path)
        return jsonify(review_index)

    @app.route("/api/generate", methods=["POST"])
    def generate_api():
        data = request.json or {}
        selected_instance_ids = [str(x).strip() for x in list(data.get("instance_ids", []) or []) if str(x).strip()]
        instance_id = str(data.get("instance_id", "") or "").strip()
        if not selected_instance_ids and instance_id:
            selected_instance_ids = [instance_id]
        if not selected_instance_ids:
            return jsonify({"error": "instance_id or instance_ids is required"}), 400
        mission_mode = str(data.get("mission_mode", "single-landmark") or "single-landmark").strip()
        if len(selected_instance_ids) > 1:
            mission_mode = "multi-landmark"
        if mission_mode not in {"single-landmark", "multi-landmark"}:
            return jsonify({"error": "mission_mode must be single-landmark or multi-landmark"}), 400
        mission_key = str(data.get("mission_type", "") or "").strip() or None
        behavior_sequence = str(data.get("behavior_sequence", "") or "").strip()
        auto_set_candidates = [str(x).strip() for x in list(data.get("auto_set_candidates", []) or []) if str(x).strip()]
        auto_set_rule = str(data.get("auto_set_rule", "heuristic") or "heuristic").strip().lower()
        landmark_set_map = dict(data.get("landmark_set_map", {}) or {}) if isinstance(data.get("landmark_set_map", {}), dict) else {}
        generation_kind = _normalize_generation_kind(data.get("generation_kind", "auto"))
        mission_count = max(1, min(16, int(data.get("mission_count", 1) or 1)))
        element_param_overrides = data.get("element_param_overrides", None)
        element_auto_rules = data.get("element_auto_rules", None)
        set_profiles = data.get("set_profiles", None)
        adaptive_sequential_params = bool(data.get("adaptive_sequential_params", True))
        allow_interleave_repeat = bool(data.get("allow_interleave_repeat", False))
        max_total_elements = max(0, int(data.get("max_total_elements", 0) or 0))

        try:
            outputs: list[dict[str, Any]] = []
            now_iso = _utc_now_iso()
            for mission_idx in range(mission_count):
                run_args = _build_run_args(
                    args,
                    {
                        "landmark_id": selected_instance_ids[0],
                        "selected_instance_ids": list(selected_instance_ids),
                        "behavior_sequence": behavior_sequence,
                        "mission_type": mission_key,
                        "auto_set_candidates": auto_set_candidates,
                        "auto_set_rule": auto_set_rule,
                        "landmark_set_map": landmark_set_map,
                        "generation_kind": generation_kind,
                        "mission_mode": mission_mode,
                        "traj_id": f"traj_{selected_instance_ids[0]}_{int(time.time())}_{mission_idx + 1:02d}",
                        "seed": int(getattr(args, "seed", 42)) + mission_idx * 101,
                        "element_param_overrides": element_param_overrides,
                        "element_auto_rules": element_auto_rules,
                        "set_profiles": set_profiles,
                        "adaptive_sequential_params": adaptive_sequential_params,
                        "allow_interleave_repeat": allow_interleave_repeat,
                        "max_total_elements": max_total_elements,
                    },
                )
                out = generate_single(run_args, config, logger, include_preview=True)
                out_files = dict(out.get("files", {}) or {})
                out_summary = dict(out.get("summary", {}) or {})
                out_files = _generate_preview_assets(out_files, out_summary)
                out["files"] = out_files
                out["summary"] = out_summary
                out["asset_urls"] = _build_asset_urls(out_files)
                outputs.append(out)

                history_entry = _build_mission_history_entry(out=out, selected_instance_ids=selected_instance_ids)
                for related_id in selected_instance_ids:
                    review_index, _, rec = _upsert_item(related_id)
                    rec["instance_id"] = related_id
                    rec["traj_id"] = str(out.get("traj_id", "") or "")
                    rec["files"] = out_files
                    rec["summary"] = out_summary
                    rec["status"] = str(rec.get("status", "pending") or "pending")
                    rec["traj_status"] = "video_ready"
                    rec["updated_at"] = now_iso
                    _append_history_entry(rec, history_entry)
                    _write_review_index(review_index_path, review_index)
                append_jsonl(
                    review_log_path,
                    {
                        "event": "generate_traj",
                        "scene_id": scene_id,
                        "time": now_iso,
                        "instance_id": selected_instance_ids[0],
                        "selected_instance_ids": list(selected_instance_ids),
                        "traj_id": str(out.get("traj_id", "") or ""),
                        "traj_status": "video_ready",
                    },
                )
            response = outputs[0] if len(outputs) == 1 else {
                "ok": True,
                "generated_count": len(outputs),
                "selected_instance_ids": selected_instance_ids,
                "missions": outputs,
                "latest_traj_id": str(outputs[0].get("traj_id", "") or ""),
                "asset_urls": dict(outputs[0].get("asset_urls", {}) or {}),
                "files": dict(outputs[0].get("files", {}) or {}),
                "summary": dict(outputs[0].get("summary", {}) or {}),
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/confirm_pano", methods=["POST"])
    def confirm_pano_api():
        data = request.json or {}
        instance_id = str(data.get("instance_id", "") or "").strip()
        if not instance_id:
            return jsonify({"error": "instance_id is required"}), 400
        approved = bool(data.get("approved", False))
        traj_id = str(data.get("traj_id", "") or "").strip()
        review_index, _, rec = _upsert_item(instance_id)
        now_iso = _utc_now_iso()
        new_status = "pano_confirmed" if approved else "pano_rejected"
        if traj_id:
            _update_history_entry(rec, traj_id=traj_id, updates={"traj_status": new_status})
        rec["traj_status"] = new_status
        rec["updated_at"] = now_iso
        _write_review_index(review_index_path, review_index)
        append_jsonl(
            review_log_path,
            {
                "event": "confirm_pano",
                "scene_id": scene_id,
                    "time": now_iso,
                    "instance_id": instance_id,
                    "traj_id": traj_id,
                    "approved": approved,
                    "traj_status": new_status,
                },
            )
        return jsonify({"ok": True, "traj_status": new_status})

    @app.route("/api/generate_video", methods=["POST"])
    def generate_video_api():
        data = request.json or {}
        instance_id = str(data.get("instance_id", "") or "").strip()
        if not instance_id:
            return jsonify({"error": "instance_id is required"}), 400
        traj_id = str(data.get("traj_id", "") or "").strip()

        review_index, _, rec = _upsert_item(instance_id)
        active_history = {}
        if traj_id:
            active_history = _update_history_entry(rec, traj_id=traj_id, updates={})
        files = active_history.get("files", rec.get("files", {})) if isinstance(active_history.get("files", rec.get("files", {})), dict) else {}
        try:
            summary = active_history.get("summary", rec.get("summary", {})) if isinstance(active_history.get("summary", rec.get("summary", {})), dict) else {}
            files = _generate_preview_assets(dict(files), summary)
            if traj_id:
                _update_history_entry(rec, traj_id=traj_id, updates={"files": files, "summary": summary, "traj_status": "video_ready"})
            rec["files"] = files
            rec["traj_status"] = "video_ready"
            rec["updated_at"] = _utc_now_iso()
            _write_review_index(review_index_path, review_index)
            append_jsonl(
                review_log_path,
                {
                    "event": "generate_video",
                    "scene_id": scene_id,
                    "time": rec.get("updated_at", ""),
                    "instance_id": instance_id,
                    "traj_id": traj_id or rec.get("traj_id", ""),
                    "traj_status": rec.get("traj_status", "video_ready"),
                },
            )
            return jsonify({"ok": True, "traj_status": rec.get("traj_status", "video_ready"), "asset_urls": _build_asset_urls(files)})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/generate_final_task", methods=["POST"])
    def generate_final_task_api():
        data = request.json or {}
        instance_id = str(data.get("instance_id", "") or "").strip()
        if not instance_id:
            return jsonify({"error": "instance_id is required"}), 400
        traj_id = str(data.get("traj_id", "") or "").strip()

        review_index, _, rec = _upsert_item(instance_id)
        active_history = {}
        if traj_id:
            active_history = _update_history_entry(rec, traj_id=traj_id, updates={})
        files = active_history.get("files", rec.get("files", {})) if isinstance(active_history.get("files", rec.get("files", {})), dict) else {}
        waypoints_path = Path(str(files.get("waypoints", "") or ""))
        if not waypoints_path.exists():
            return jsonify({"error": "waypoints file missing, regenerate trajectory first"}), 400

        summary = active_history.get("summary", rec.get("summary", {})) if isinstance(active_history.get("summary", rec.get("summary", {})), dict) else {}
        target_center_3d = summary.get("target_center_3d", None)
        target_bbox_list = summary.get("target_bbox_3d_list", None)
        if not isinstance(target_center_3d, list) or len(target_center_3d) < 3 or not isinstance(target_bbox_list, list) or len(target_bbox_list) < 6:
            return jsonify({"error": "target geometry missing in summary, regenerate trajectory first"}), 400

        try:
            waypoints = np.load(waypoints_path).astype(np.float32)
            raw_waypoints_path = waypoints_path.parent / "composed_path_raw.npy"
            raw_waypoints = np.load(raw_waypoints_path).astype(np.float32) if raw_waypoints_path.exists() else None
            segments_meta = read_json_if_exists(Path(str(files.get("segments", "") or "")), default={})
            segments_rows = list(segments_meta.get("segments", []) or []) if isinstance(segments_meta, dict) else []
            mission_meta = {
                "task_type": str(summary.get("task_type", summary.get("mission_type", "")) or ""),
                "task_subtype": str(summary.get("task_subtype", summary.get("mission_subtype", "")) or ""),
                "task_difficulty": str(summary.get("task_difficulty", "") or ""),
                "task_difficulty_score": float(summary.get("task_difficulty_score", 0.0) or 0.0),
                "set_instance": dict(summary.get("set_instance", {}) or {}),
                "element_instances": list(summary.get("element_instances", []) or []),
                "self_state": {
                    "landmark_order": list((summary.get("set_instance", {}) or {}).get("landmark_order", []) or []),
                },
                "mode_sequence": list(summary.get("mode_sequence", []) or []),
                "event_sequence": list(summary.get("event_sequence", []) or []),
            }

            progress_state["active"] = True
            progress_state["done"] = 0
            progress_state["total"] = int(max(1, waypoints.shape[0]))
            progress_state["message"] = "start final task generation"

            def _on_progress(done: int, total: int, msg: str) -> None:
                progress_state["active"] = True
                progress_state["done"] = int(done)
                progress_state["total"] = int(total)
                progress_state["message"] = str(msg)

            final_out = _generate_final_task_video_and_data(
                out_dir=waypoints_path.parent,
                scene_root=scene_root,
                scene_id=scene_id,
                config=config,
                waypoints_xyz=waypoints,
                target_center_3d=[float(target_center_3d[0]), float(target_center_3d[1]), float(target_center_3d[2])],
                target_bbox_list=[float(v) for v in target_bbox_list],
                segments=segments_rows,
                mission_meta=mission_meta,
                progress_cb=_on_progress,
                source_pose_fps_override=float(summary.get("source_pose_fps", (config.get("camera", {}) or {}).get("fps", 10.0)) or 10.0),
                waypoint_forwards=np.load(waypoints_path.parent / "forwards.npy").astype(np.float32) if (waypoints_path.parent / "forwards.npy").exists() else None,
                raw_waypoints_xyz=raw_waypoints,
            )

            files.update(
                {
                    "final_video_web": final_out.get("final_video_web", ""),
                    "final_video": final_out.get("final_video", ""),
                    "final_video_marked_web": final_out.get("final_video_marked_web", ""),
                    "final_video_marked": final_out.get("final_video_marked", ""),
                    "final_metadata": final_out.get("final_metadata", ""),
                    "final_frames_dir": final_out.get("final_frames_dir", ""),
                    "final_frames_manifest": final_out.get("final_frames_manifest", ""),
                    "final_masks_tensor": final_out.get("final_masks_tensor", ""),
                }
            )
            if traj_id:
                _update_history_entry(rec, traj_id=traj_id, updates={"files": files, "summary": summary, "traj_status": "final_ready"})
            rec["files"] = files
            rec["final_summary"] = final_out.get("final_summary", {})
            rec["traj_status"] = "final_ready"
            rec["updated_at"] = _utc_now_iso()
            _write_review_index(review_index_path, review_index)
            append_jsonl(
                review_log_path,
                {
                    "event": "generate_final_task",
                    "scene_id": scene_id,
                    "time": rec.get("updated_at", ""),
                    "instance_id": instance_id,
                    "traj_id": traj_id or rec.get("traj_id", ""),
                    "traj_status": rec.get("traj_status", "final_ready"),
                    "final_summary": rec.get("final_summary", {}),
                },
            )
            progress_state["active"] = False
            progress_state["message"] = "final task generation completed"
            return jsonify(
                {
                    "ok": True,
                    "traj_status": rec.get("traj_status", "final_ready"),
                    "asset_urls": _build_asset_urls(files),
                    "final_summary": rec.get("final_summary", {}),
                }
            )
        except Exception as exc:
            progress_state["active"] = False
            progress_state["message"] = f"error: {exc}"
            try:
                final_dir = waypoints_path.parent / "final_task"
                for name in [
                    "task_rgb.mp4",
                    "task_rgb_marked.mp4",
                    "task_rgb_web.mp4",
                    "task_rgb_marked_web.mp4",
                    "task_rgb_720p.mp4",
                    "task_rgb_720p_marked.mp4",
                    "task_rgb_720p_web.mp4",
                    "task_rgb_720p_marked_web.mp4",
                    "task_data.json",
                    "frames_manifest.json",
                    "frame_index_map.json",
                ]:
                    fp = final_dir / name
                    if fp.exists():
                        fp.unlink()
            except Exception:
                pass
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/confirm_video", methods=["POST"])
    def confirm_video_api():
        data = request.json or {}
        instance_id = str(data.get("instance_id", "") or "").strip()
        if not instance_id:
            return jsonify({"error": "instance_id is required"}), 400
        approved = bool(data.get("approved", False))
        traj_id = str(data.get("traj_id", "") or "").strip()
        review_index, _, rec = _upsert_item(instance_id)
        now_iso = _utc_now_iso()
        new_status = "video_confirmed" if approved else "video_rejected"
        if traj_id:
            _update_history_entry(rec, traj_id=traj_id, updates={"traj_status": new_status})
        rec["traj_status"] = new_status
        rec["updated_at"] = now_iso
        _write_review_index(review_index_path, review_index)
        append_jsonl(
            review_log_path,
            {
                "event": "confirm_video",
                    "scene_id": scene_id,
                    "time": now_iso,
                    "instance_id": instance_id,
                    "traj_id": traj_id,
                    "approved": approved,
                    "traj_status": new_status,
                },
            )
        return jsonify({"ok": True, "traj_status": new_status})
            
    @app.route("/api/save_decision", methods=["POST"])
    def save_decision():
        data = request.json or {}
        instance_id = str(data.get("instance_id", "") or "").strip()
        if not instance_id:
            return jsonify({"error": "instance_id is required"}), 400

        status = str(data.get("decision", "pending") or "pending").strip().lower()
        if status not in {"pending", "valid", "invalid"}:
            return jsonify({"error": "decision must be one of pending|valid|invalid"}), 400

        review_index = _load_review_index(review_index_path)
        if review_index.get("scene_id", "") in {"", None}:
            review_index["scene_id"] = scene_id
        items = review_index.get("items", {})
        if not isinstance(items, dict):
            items = {}

        now_iso = _utc_now_iso()
        record = {
            "instance_id": instance_id,
            "status": status,
            "traj_status": str(data.get("traj_status", "") or "") or str(items.get(instance_id, {}).get("traj_status", "pending") or "pending"),
            "label": str(data.get("label", "") or "").strip(),
            "note": str(data.get("note", "") or "").strip(),
            "traj_id": str(data.get("traj_id", "") or "").strip(),
            "files": data.get("files", {}),
            "summary": data.get("summary", {}),
            "updated_at": now_iso,
        }
        items[instance_id] = record
        review_index["items"] = items
        _write_review_index(review_index_path, review_index)

        append_jsonl(
            review_log_path,
            {
                "event": "save_decision",
                "scene_id": scene_id,
                "time": now_iso,
                **record,
            },
        )
        return jsonify({"ok": True, "status": status})
        
    port = args.port or 20262
    host = str(args.host or "0.0.0.0")
    logger.info(f"Starting web server on {host}:{port}")
    app.run(host=host, port=port, threaded=True)


def generate_dataset_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    scene_id = _resolve_scene_id(args, config)
    task_cfg = config.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim")).lower().strip()
    forms = [str(x).strip() for x in str(args.forms or "").split(",") if str(x).strip()]
    task_group = str(args.task_group or "all").strip().lower()
    if task_group == "self-state":
        forms = [name for name in forms if name.startswith("self_")]
    elif task_group == "environmental":
        forms = [name for name in forms if name.startswith("env_") or name in {"count_only", "intervals", "intervals_plus_keyframes"}]
    out = generate_stage3_manifest(
        config=config,
        scene_id=scene_id,
        engine=engine_name,
        sample_count=max(1, int(args.sample_count or 1)),
        seed=int(args.seed),
        forms=forms,
        approved_only=bool(args.approved_only),
        mode=str(args.mission_mode or "single-landmark"),
    )
    return {
        "ok": True,
        "mode": "generate_dataset",
        "task_group": task_group,
        "manifest_path": _path_for_json(out["manifest_path"]),
        "summary": out["manifest"].get("summary", {}),
    }


def run_experiment_cli(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    scene_id = _resolve_scene_id(args, config)
    task_cfg = config.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim")).lower().strip()
    manifest_path = Path(str(args.manifest_path or "")).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_not_found: {manifest_path}")
    out = run_stage3_experiment_once(
        config=config,
        scene_id=scene_id,
        engine=engine_name,
        manifest_path=manifest_path,
        model=(str(args.model).strip() if args.model else None),
        limit=(int(args.limit) if int(args.limit or 0) > 0 else None),
        api_overrides={
            "provide_flight_description": bool(args.provide_flight_description),
            "include_keyframes": bool(args.include_keyframes),
        },
        cancel_event=None,
        progress_callback=None,
    )
    return {
        "ok": True,
        "mode": "run_experiment",
        "run_id": out["run_id"],
        "report_path": _path_for_json(out["report_path"]),
        "summary": out["report"].get("summary", {}),
    }


def record_scene_videos_cli(args: argparse.Namespace, config: dict[str, Any], progress_cb: Any = None, detail_log_cb: Any = None) -> dict[str, Any]:
    scene_id = _resolve_scene_id(args, config)
    layout = _resolve_stage3_layout(config=config, scene_id=scene_id)
    missions_root = layout["missions_root"]
    traj_ids = [str(x).strip() for x in str(getattr(args, "traj_ids", "") or "").split(",") if str(x).strip()]
    if not traj_ids:
        raise RuntimeError("traj_ids_required_for_record_scene_videos")

    rerender_existing = bool(getattr(args, "rerender_existing", False))
    ignore_waypoint_forwards = bool(getattr(args, "ignore_waypoint_forwards", False))
    inputs = []
    skipped_traj_ids: list[str] = []
    for traj_id in traj_ids:
        mission_dir = missions_root / traj_id
        if not mission_dir.exists():
            raise FileNotFoundError(f"mission_dir_not_found: {mission_dir}")
        if not rerender_existing and _stage3_render_is_complete(mission_dir):
            skipped_traj_ids.append(str(traj_id))
            continue
        row = _load_final_task_inputs_from_mission_dir(mission_dir)
        inputs.append(row)

    if callable(detail_log_cb) and skipped_traj_ids:
        detail_log_cb(f"resume render: skipped already completed missions={len(skipped_traj_ids)}")

    if not inputs:
        return {
            "ok": True,
            "mode": "record_scene_videos",
            "scene_id": scene_id,
            "traj_ids": traj_ids,
            "pending_traj_ids": [],
            "skipped_completed_traj_ids": skipped_traj_ids,
            "task_parallel": 0,
            "reuse_worker_connections": False,
            "vehicles": [],
            "runtime_port": None,
            "configured_port": None,
            "elapsed_sec": 0.0,
            "results": [],
        }

    first = inputs[0]
    mission_meta = dict(first.get("mission_meta", {}) or {})
    stage3_cfg = _stage3_cfg(config)
    requested_parallel = max(1, int(getattr(args, "record_parallel_workers", 1) or 1))
    task_parallel = max(1, min(len(inputs), int(requested_parallel)))
    inputs = _spread_stage3_render_inputs(inputs, worker_count=task_parallel)
    reuse_worker_connections = bool(getattr(args, "record_reuse_worker_connections", False) or stage3_cfg.get("record_reuse_worker_connections", False))
    vehicles = _parse_stage3_capture_vehicles(
        config=config,
        stage3_cfg=stage3_cfg,
        worker_count=max(1, int(task_parallel)),
    )
    engine, _, base_bridge_cfg = _build_bridge_config_for_stage3(
        config,
        image_width_override=int(stage3_cfg.get("final_capture_width", (config.get("camera", {}) or {}).get("width", 4096)) or 4096),
        image_height_override=int(stage3_cfg.get("final_capture_height", (config.get("camera", {}) or {}).get("height", 3072)) or 3072),
    )
    if engine != "airsim":
        raise RuntimeError("record_scene_videos_cli currently supports airsim only")
    ensure_single_airsim_process("stage3_render")
    runtime_port, bootstrap_bridge, launched_by_bridge, configured_port = prepare_airsim_runtime_unified(
        config=config,
        scene_id=scene_id,
        base_bridge_cfg=base_bridge_cfg,
        vehicle_name=str(vehicles[0] if vehicles else base_bridge_cfg.get("vehicle_name", "drone_1")),
        vehicle_names=[str(v) for v in vehicles],
    )
    shared_runtime = {
        "runtime_port": int(runtime_port),
        "configured_port": int(configured_port),
        "bootstrap_bridge": bootstrap_bridge,
        "launched_by_bridge": bool(launched_by_bridge),
    }
    results: list[dict[str, Any]] = []
    t0 = time.time()
    persistent_worker_bridges: list[Any] = []
    try:
        done_counter = {"done": 0}
        done_lock = threading.Lock()

        def _notify_done(traj_id: str, vehicle_name: str) -> None:
            if callable(progress_cb):
                with done_lock:
                    done_counter["done"] += 1
                    done = int(done_counter["done"])
                progress_cb(done, len(inputs), f"rendered {traj_id} on {vehicle_name}")

        def _rewrite_worker_detail(detail: str, *, worker_id: int, vehicle_name: str) -> str:
            text = str(detail or "").strip()
            if not text:
                return f"worker={worker_id} vehicle={vehicle_name}"
            text = text.replace("(worker=0)", f"(worker={worker_id}, vehicle={vehicle_name})")
            if "worker=" not in text:
                text = f"{text} (worker={worker_id}, vehicle={vehicle_name})"
            return text

        def _run_one(item: dict[str, Any], vehicle_name: str, worker_id: int) -> dict[str, Any]:
            mission_dir = Path(item["mission_dir"])
            local_config = copy.deepcopy(config)
            local_config.setdefault("parallel", {})["workers"] = 1
            local_config["parallel"]["bindings"] = [{"worker_id": 0, "vehicle": str(vehicle_name)}]
            local_config.setdefault("stage3", {})["final_capture_parallel_workers"] = 1
            local_config["stage3"]["final_capture_bindings"] = [{"worker_id": 0, "vehicle": str(vehicle_name)}]
            local_config.setdefault("engine_params", {}).setdefault("airsim", {})["vehicle_name"] = str(vehicle_name)
            out = _generate_final_task_video_and_data(
                out_dir=mission_dir,
                scene_root=layout["scene_root"],
                scene_id=scene_id,
                config=local_config,
                waypoints_xyz=item["waypoints"],
                target_center_3d=item["target_center_3d"],
                target_bbox_list=item["target_bbox_list"],
                segments=item["segments"],
                mission_meta=item["mission_meta"],
                progress_cb=(lambda done, total, detail: detail_log_cb(f"{mission_dir.name}: {_rewrite_worker_detail(detail, worker_id=worker_id, vehicle_name=str(vehicle_name))}") if callable(detail_log_cb) else None),
                shared_runtime=shared_runtime,
                source_pose_fps_override=float(item.get("source_pose_fps", 10.0) or 10.0),
                waypoint_forwards=None if ignore_waypoint_forwards else np.asarray(item.get("forwards"), dtype=np.float32),
                raw_waypoints_xyz=np.asarray(item.get("raw_waypoints"), dtype=np.float32) if item.get("raw_waypoints") is not None else None,
            )
            _notify_done(mission_dir.name, str(vehicle_name))
            return {
                "traj_id": mission_dir.name,
                "vehicle_name": str(vehicle_name),
                "final_video": str(out.get("final_video", "") or ""),
                "final_video_web": str(out.get("final_video_web", "") or ""),
                "final_summary": dict(out.get("final_summary", {}) or {}),
            }

        if reuse_worker_connections:
            for worker_id in range(task_parallel):
                local_cfg = dict(base_bridge_cfg)
                local_cfg["vehicle_name"] = str(vehicles[worker_id])
                local_cfg["sim_port"] = int(runtime_port)
                local_cfg["launch_sim"] = False
                local_cfg["connect_on_init"] = True
                local_cfg["auto_select_port_on_conflict"] = False
                persistent_worker_bridges.append(create_bridge(engine=engine, scene_id=scene_id, config=local_cfg))

            worker_item_groups: list[list[dict[str, Any]]] = [[] for _ in range(task_parallel)]
            for idx, item in enumerate(inputs):
                worker_item_groups[idx % task_parallel].append(item)

            def _run_many(worker_id: int, items_for_worker: list[dict[str, Any]]) -> list[dict[str, Any]]:
                worker_bridge = persistent_worker_bridges[worker_id]
                local_results: list[dict[str, Any]] = []
                for item in items_for_worker:
                    mission_dir = Path(item["mission_dir"])
                    local_config = copy.deepcopy(config)
                    local_config.setdefault("parallel", {})["workers"] = 1
                    local_config["parallel"]["bindings"] = [{"worker_id": 0, "vehicle": str(vehicles[worker_id])}]
                    local_config.setdefault("stage3", {})["final_capture_parallel_workers"] = 1
                    local_config["stage3"]["final_capture_bindings"] = [{"worker_id": 0, "vehicle": str(vehicles[worker_id])}]
                    local_config.setdefault("engine_params", {}).setdefault("airsim", {})["vehicle_name"] = str(vehicles[worker_id])
                    out = _generate_final_task_video_and_data(
                        out_dir=mission_dir,
                        scene_root=layout["scene_root"],
                        scene_id=scene_id,
                        config=local_config,
                        waypoints_xyz=item["waypoints"],
                        target_center_3d=item["target_center_3d"],
                        target_bbox_list=item["target_bbox_list"],
                        segments=item["segments"],
                        mission_meta=item["mission_meta"],
                        progress_cb=(lambda done, total, detail: detail_log_cb(f"{mission_dir.name}: {_rewrite_worker_detail(detail, worker_id=worker_id, vehicle_name=str(vehicles[worker_id]))}") if callable(detail_log_cb) else None),
                        shared_runtime={**shared_runtime, "persistent_bridge": worker_bridge},
                        source_pose_fps_override=float(item.get("source_pose_fps", 10.0) or 10.0),
                        waypoint_forwards=None if ignore_waypoint_forwards else np.asarray(item.get("forwards"), dtype=np.float32),
                        raw_waypoints_xyz=np.asarray(item.get("raw_waypoints"), dtype=np.float32) if item.get("raw_waypoints") is not None else None,
                    )
                    _notify_done(mission_dir.name, str(vehicles[worker_id]))
                    local_results.append(
                        {
                            "traj_id": mission_dir.name,
                            "vehicle_name": str(vehicles[worker_id]),
                            "final_video": str(out.get("final_video", "") or ""),
                            "final_video_web": str(out.get("final_video_web", "") or ""),
                            "final_summary": dict(out.get("final_summary", {}) or {}),
                        }
                    )
                return local_results

            if task_parallel <= 1:
                results.extend(_run_many(0, worker_item_groups[0]))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=task_parallel) as executor:
                    futures = [executor.submit(_run_many, worker_id, worker_item_groups[worker_id]) for worker_id in range(task_parallel)]
                    for fut in concurrent.futures.as_completed(futures):
                        results.extend(fut.result())
        elif task_parallel <= 1:
            for item in inputs:
                results.append(_run_one(item, vehicles[0], 0))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=task_parallel) as executor:
                futures = []
                for idx, item in enumerate(inputs):
                    vehicle_name = str(vehicles[idx % len(vehicles)])
                    worker_id = int(idx % len(vehicles))
                    futures.append(executor.submit(_run_one, item, vehicle_name, worker_id))
                for fut in concurrent.futures.as_completed(futures):
                    results.append(fut.result())
    finally:
        for bridge in persistent_worker_bridges:
            try:
                bridge.shutdown()
            except Exception:
                pass
        if bootstrap_bridge is not None:
            try:
                bootstrap_bridge.shutdown()
            except Exception:
                pass
    return {
        "ok": True,
        "mode": "record_scene_videos",
        "scene_id": scene_id,
        "traj_ids": traj_ids,
        "pending_traj_ids": [str(Path(str(item.get("mission_dir", "") or "")).name) for item in inputs],
        "skipped_completed_traj_ids": skipped_traj_ids,
        "task_parallel": int(task_parallel),
        "reuse_worker_connections": bool(reuse_worker_connections),
        "ignore_waypoint_forwards": bool(ignore_waypoint_forwards),
        "vehicles": [str(v) for v in vehicles[:task_parallel]],
        "runtime_port": int(runtime_port),
        "configured_port": int(configured_port),
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }


def _stage3_render_is_complete(mission_dir: Path) -> bool:
    final_dir = Path(mission_dir) / "final_task"
    meta_path = final_dir / "task_data.json"
    video_paths = [
        final_dir / "task_rgb.mp4",
        final_dir / "task_rgb_720p.mp4",
    ]
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    video_meta = dict((meta or {}).get("video", {}) or {}) if isinstance(meta, dict) else {}
    if bool(video_meta.get("generated_without_video", False)):
        return False
    first_existing = next((path for path in video_paths if path.exists()), None)
    if first_existing is None:
        return False
    if int(first_existing.stat().st_size) < 1024:
        return False
    return True


def _stage3_render_anchor_point(item: dict[str, Any]) -> np.ndarray | None:
    center = list(item.get("target_center_3d", []) or [])
    if len(center) >= 3:
        try:
            return np.asarray([float(center[0]), float(center[1]), float(center[2])], dtype=np.float32)
        except Exception:
            return None
    mission_meta = dict(item.get("mission_meta", {}) or {})
    center = list(mission_meta.get("target_center_3d", []) or [])
    if len(center) >= 3:
        try:
            return np.asarray([float(center[0]), float(center[1]), float(center[2])], dtype=np.float32)
        except Exception:
            return None
    return None


def _stage3_render_item_cost(item: dict[str, Any]) -> int:
    waypoints = np.asarray(item.get("waypoints", []), dtype=np.float32)
    if waypoints.ndim >= 2:
        return max(0, int(waypoints.shape[0]))
    if waypoints.ndim == 1 and waypoints.size > 0:
        return max(0, int(waypoints.size))
    segments = list(item.get("segments", []) or [])
    return max(0, len(segments))


def _stage3_render_landmark_id(item: dict[str, Any]) -> str:
    mission_meta = dict(item.get("mission_meta", {}) or {})
    for key in ("landmark_id", "instance_id"):
        value = str(item.get(key, "") or mission_meta.get(key, "") or "").strip()
        if value:
            return value
    mission_dir = Path(str(item.get("mission_dir", "") or ""))
    parts = mission_dir.name.split("_")
    if len(parts) >= 5:
        return f"{parts[3]}_{parts[4]}"
    return ""


def _spread_stage3_render_inputs(inputs: list[dict[str, Any]], *, worker_count: int) -> list[dict[str, Any]]:
    if worker_count <= 1 or len(inputs) <= 1:
        return list(inputs)
    remaining = sorted(
        [dict(item) for item in inputs],
        key=lambda row: str(Path(str(row.get("mission_dir", "") or "")).name),
    )
    if len(remaining) <= worker_count:
        return remaining

    def _traj_name(row: dict[str, Any]) -> str:
        return str(Path(str(row.get("mission_dir", "") or "")).name)

    scheduled: list[dict[str, Any]] = []
    while remaining:
        if len(remaining) <= worker_count:
            scheduled.extend(remaining)
            break

        batch: list[dict[str, Any]] = []
        anchors = [pt for pt in (_stage3_render_anchor_point(row) for row in remaining) if pt is not None]
        if anchors:
            centroid = np.mean(np.vstack(anchors), axis=0)
            def _seed_score(idx: int) -> float:
                anchor = _stage3_render_anchor_point(remaining[idx])
                base = anchor if anchor is not None else centroid
                return float(np.linalg.norm(base - centroid))
            seed_idx = max(
                range(len(remaining)),
                key=_seed_score,
            )
        else:
            seed_idx = 0
        batch.append(remaining.pop(seed_idx))
        target_cost = float(max(1, _stage3_render_item_cost(batch[0])))

        while len(batch) < worker_count and remaining:
            ranked_by_cost = sorted(
                range(len(remaining)),
                key=lambda idx: (
                    abs(float(_stage3_render_item_cost(remaining[idx])) - target_cost),
                    _traj_name(remaining[idx]),
                ),
            )
            candidate_pool = ranked_by_cost[: max(worker_count * 4, 16)]
            batch_anchors = [anchor for anchor in (_stage3_render_anchor_point(item) for item in batch) if anchor is not None]
            batch_landmark_ids = {
                landmark_id
                for landmark_id in (_stage3_render_landmark_id(item) for item in batch)
                if landmark_id
            }
            best_idx = candidate_pool[0] if candidate_pool else 0
            best_score = -1.0
            best_cost_gap = float("inf")
            best_same_landmark = True
            best_name = ""
            for idx in candidate_pool:
                row = remaining[idx]
                anchor = _stage3_render_anchor_point(row)
                if anchor is None or not batch_anchors:
                    dist_score = 0.0
                else:
                    dist_score = min(float(np.linalg.norm(anchor - other_anchor)) for other_anchor in batch_anchors)
                cost_gap = abs(float(_stage3_render_item_cost(row)) - target_cost) / target_cost
                landmark_id = _stage3_render_landmark_id(row)
                same_landmark = bool(landmark_id and landmark_id in batch_landmark_ids)
                score = dist_score / (1.0 + 6.0 * cost_gap)
                if same_landmark:
                    score *= 0.15
                name = _traj_name(row)
                if (
                    (not same_landmark and best_same_landmark)
                    or (
                        same_landmark == best_same_landmark
                        and (
                            score > best_score
                            or (abs(score - best_score) <= 1e-6 and cost_gap < best_cost_gap)
                            or (abs(score - best_score) <= 1e-6 and abs(cost_gap - best_cost_gap) <= 1e-6 and (not best_name or name < best_name))
                        )
                    )
                ):
                    best_idx = idx
                    best_score = score
                    best_cost_gap = cost_gap
                    best_same_landmark = same_landmark
                    best_name = name
            batch.append(remaining.pop(best_idx))

        scheduled.extend(batch)
    return scheduled

def main() -> None:
    args = parse_args()
    logger = StageLogger("stage3.main")
    cfg_path = Path(str(args.config)).resolve()
    config = load_yaml(cfg_path)
    
    if args.mode in {"generate", "generate_mission"}:
        out = generate_single(args, config, logger)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.mode == "generate_dataset":
        out = generate_dataset_cli(args, config)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.mode == "run_experiment":
        out = run_experiment_cli(args, config)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.mode == "record_scene_videos":
        out = record_scene_videos_cli(args, config)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.mode == "web":
        run_web(args, config, logger)


if __name__ == "__main__":
    main()
