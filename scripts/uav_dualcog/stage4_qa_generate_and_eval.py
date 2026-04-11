from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import mimetypes
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

import numpy as np

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    from flask import Flask, jsonify, request, send_file
except Exception:  # pragma: no cover
    Flask = None
    jsonify = None
    request = None
    send_file = None

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_RUNTIME_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "uav_dualcog" / "common_api_runtime.yaml"
COMMON_STAGE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "uav_dualcog" / "common_stage_configs.yaml"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from media_path_utils import resolve_existing_file_with_suffix_fallback
from api_common import (
    build_model_request_controls,
    compute_rate_limited_concurrency,
    detect_model_family,
    load_api_registry,
    pick_first_text,
    resolve_default_model,
    resolve_model_api_endpoint,
    should_inline_system_prompt_for_multimodal,
)
from pipeline_common import list_task_pipeline_tasks, resolve_task_pipeline_scene_root
from pipeline_common import append_unified_scene_log, build_unified_bridge_config, build_unified_stage_event
from pipeline_common import ensure_single_airsim_process, format_unified_startup_ports_message, prepare_airsim_runtime_unified
from image_compression_utils import compression_cfg as build_image_compression_cfg
from image_compression_utils import preferred_output_path, save_bgr_image, save_pil_image
from sim_bridge.factory import create_bridge
from stage2_landmark_label import _build_camera_pose_params, _camera_pose_for_yaw, _forward_to_yaw_pitch_deg
from prompt_templates import get_config_template, get_prompt_template, render_prompt_template

LABEL_SPACE_4WAY = ["Front", "Back", "Left", "Right"]
LABEL_SPACE_8WAY = [
    "Front",
    "Back",
    "Left",
    "Right",
    "Front-Left",
    "Front-Right",
    "Back-Left",
    "Back-Right",
]
OPTION_IDS = ["A", "B", "C", "D"]
TASK_SPECS: list[tuple[str, str]] = [
    ("Self-Aware", "self_where"),
    ("Self-Aware", "self_what"),
    ("Environment-Aware", "env_where"),
    ("Environment-Aware", "env_how"),
]
VIEW_KEY_TO_LABEL = {
    "front": "Front",
    "back": "Back",
    "left": "Left",
    "right": "Right",
    "front_left": "Front-Left",
    "front_right": "Front-Right",
    "back_left": "Back-Left",
    "back_right": "Back-Right",
}
VIEW_LABEL_TO_KEY = {value: key for key, value in VIEW_KEY_TO_LABEL.items()}
GLOBAL_SCENE_ID = "__all__"
GLOBAL_SCENE_LABEL = "ALL scenes"
ENV_POSITION_LABEL_SPACE_4WAY = ["Left", "Front", "Right", "Back"]
ENV_POSITION_LABEL_SPACE_8WAY = ["Front", "Front-Right", "Right", "Back-Right", "Back", "Back-Left", "Left", "Front-Left"]
OBJ_TO_OBS_LABEL = {
    "Front": "Front",
    "Back": "Back",
    "Left": "Right",
    "Right": "Left",
    "Front-Left": "Front-Right",
    "Front-Right": "Front-Left",
    "Back-Left": "Back-Right",
    "Back-Right": "Back-Left",
}
TASK_GROUPS = {
    "self_where": "self-aware",
    "self_what": "self-aware",
    "env_where": "environment-aware",
    "env_how": "environment-aware",
}
TASK_KIND = {
    "self_where": "single_choice_bbox",
    "self_what": "single_choice_bbox",
    "env_where": "single_choice_bbox",
    "env_how": "single_choice_bbox",
    "label_multiple_choice": "single_choice_bbox",
    "image_multiple_choice": "single_choice_bbox",
}
TASK_DISPLAY = {
    "self_where": "Self-1 / Where Am I",
    "self_what": "Self-2 / What Am I Doing",
    "env_where": "Env-1 / Where Is The Landmark",
    "env_how": "Env-2 / How Should I Move",
    "label_multiple_choice": "Image to Label",
    "image_multiple_choice": "Label to Image",
}
VIEW_ANGLE_DEG = {
    "Front": 0.0,
    "Front-Right": 45.0,
    "Right": 90.0,
    "Back-Right": 135.0,
    "Back": 180.0,
    "Back-Left": 225.0,
    "Left": 270.0,
    "Front-Left": 315.0,
}


def _now_ts() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "item"


def _is_global_scene_id(scene_id: str | None) -> bool:
    text = str(scene_id or "").strip().lower()
    return text in {GLOBAL_SCENE_ID, "all", "*"}


def _path_for_json(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(WORKSPACE_ROOT))
    except Exception:
        return str(p)


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


def resolve_output_dir_name(config: dict[str, Any], *, key: str, default: str) -> str:
    output_layout = config.get("output_layout", {}) or {}
    if key in output_layout:
        raw = str(output_layout.get(key, default)).strip()
        return raw or default
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


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"config_not_dict: {path}")
    return data


def _load_common_runtime_cfg() -> dict[str, Any]:
    if not COMMON_RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        return _load_yaml(COMMON_RUNTIME_CONFIG_PATH)
    except Exception:
        return {}


def _load_common_stage_cfg() -> dict[str, Any]:
    if not COMMON_STAGE_CONFIG_PATH.exists():
        return {}
    try:
        return _load_yaml(COMMON_STAGE_CONFIG_PATH)
    except Exception:
        return {}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_view_key(raw: Any) -> str | None:
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "frontleft": "front_left",
        "frontright": "front_right",
        "backleft": "back_left",
        "backright": "back_right",
    }
    text = aliases.get(text, text)
    return text if text in VIEW_KEY_TO_LABEL else None


def _view_key_to_label(key: str | None) -> str | None:
    if not key:
        return None
    return VIEW_KEY_TO_LABEL.get(key)


def _object_to_observer_label(obj_label: str) -> str:
    return OBJ_TO_OBS_LABEL[obj_label]


def _resolve_stage4_root(config: dict[str, Any], *, scene_root: Path) -> Path:
    dir_name = resolve_output_dir_name(config, key="stage4_qa_dir", default="image_tasks")
    task_cfg = config.get("task", {}) or {}
    scene_id = str(task_cfg.get("scene_id", "") or "").strip()
    engine = str(task_cfg.get("engine", "airsim") or "airsim").strip().lower()
    artifact_scene_root = resolve_task_pipeline_scene_root(
        config,
        scene_id=scene_id,
        engine=engine,
        workspace_root=WORKSPACE_ROOT,
    ) or scene_root
    return artifact_scene_root / dir_name


def _resolve_stage2_review_paths(config: dict[str, Any], *, scene_root: Path, scene_id: str) -> tuple[Path, Path]:
    review_dir = scene_root / resolve_output_dir_name(config, key="stage2_review_dir", default="landmarks_review")
    raw_dir = scene_root / resolve_output_dir_name(config, key="stage2_raw_dir", default="landmarks_raw")
    valid_instances_path = review_dir / f"{scene_id}.valid_instances.json"
    return raw_dir, valid_instances_path


def _resolve_stage2_image_path(scene_root: Path, raw_root: Path, rel_path: str) -> Path | None:
    raw = str(rel_path or "").strip()
    if not raw:
        return None
    return resolve_existing_file_with_suffix_fallback(
        raw,
        base_dirs=[raw_root, scene_root, WORKSPACE_ROOT],
    )


def _bbox_to_norm(bbox_xyxy: list[Any] | tuple[Any, ...] | None, image_size: list[Any] | tuple[Any, ...] | None) -> list[float] | None:
    if not isinstance(bbox_xyxy, (list, tuple)) or len(bbox_xyxy) != 4:
        return None
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return None
    try:
        w = float(image_size[0])
        h = float(image_size[1])
        if w <= 1.0 or h <= 1.0:
            return None
        x0, y0, x1, y1 = [float(v) for v in bbox_xyxy]
    except Exception:
        return None
    return [
        max(0.0, min(1.0, x0 / w)),
        max(0.0, min(1.0, y0 / h)),
        max(0.0, min(1.0, x1 / w)),
        max(0.0, min(1.0, y1 / h)),
    ]


def _draw_reference_bbox(
    *,
    source_image: Path,
    bbox_xyxy: list[float],
    output_path: Path,
    cfg: dict[str, Any] | None = None,
) -> Path:
    if Image is None or ImageDraw is None:
        raise ImportError("Pillow is required for bbox overlays")
    image_cfg = dict(cfg or build_image_compression_cfg({}))
    output_path = preferred_output_path(output_path, compress_enabled=bool(image_cfg.get("enabled", True)))
    _ensure_dir(output_path.parent)
    with Image.open(source_image) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        x0, y0, x1, y1 = bbox_xyxy
        line_w = max(4, int(round(min(w, h) * 0.008)))
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=line_w)
        save_pil_image(canvas, output_path, cfg=image_cfg)
    return output_path


def _stage4_overlay_worker(payload: dict[str, Any]) -> str:
    source_image = Path(str(payload.get("source_image", "") or ""))
    output_path = Path(str(payload.get("output_path", "") or ""))
    bbox_xyxy = [float(v) for v in list(payload.get("bbox_xyxy", []) or [])]
    cfg = dict(payload.get("cfg", {}) or {})
    _draw_reference_bbox(
        source_image=source_image,
        bbox_xyxy=bbox_xyxy,
        output_path=output_path,
        cfg=cfg,
    )
    return str(output_path.name)


def _stage4_cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict((_load_common_stage_cfg().get("stage4_qa_defaults", {}) or {}))
    if not merged:
        raise RuntimeError("missing_stage4_qa_defaults_in_common_stage_configs")
    merged.update(dict(config.get("stage4", {}) or {}))
    merged.update(dict(config.get("stage4_qa", {}) or {}))
    return merged


def _stage4_render_requests_root(stage4_root: Path) -> Path:
    return _ensure_dir(stage4_root / "render_requests")


def _stage4_render_requests_paths(stage4_root: Path, generation_id: str, scene_id: str) -> tuple[Path, Path]:
    root = _stage4_render_requests_root(stage4_root)
    return root / f"{generation_id}.json", root / f"{scene_id}.latest_render_requests.json"


def _discover_scene_catalog(config_path: Path | None = None) -> list[dict[str, Any]]:
    cfg_dir = (config_path.parent if config_path is not None else (WORKSPACE_ROOT / "configs" / "uav_dualcog")).resolve()
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(cfg_dir.glob("task_*.yaml")):
        try:
            cfg = _load_yaml(path)
        except Exception:
            continue
        task_cfg = cfg.get("task", {}) or {}
        stage4_cfg = _stage4_cfg(cfg)
        engine = str(task_cfg.get("engine", "airsim") or "airsim").strip().lower()
        scene_id = str(task_cfg.get("scene_id", "") or "").strip()
        if not scene_id:
            continue
        key = (engine, scene_id)
        if key in seen:
            continue
        seen.add(key)
        default_model = pick_first_text(
            stage4_cfg.get("default_model"),
            resolve_default_model(cfg, stage_name="stage4"),
        )
        default_api_source = ""
        default_api_base = ""
        if default_model:
            endpoint = resolve_model_api_endpoint(
                config=cfg,
                model=default_model,
                stage_name="stage4",
                stage_cfg=stage4_cfg,
                explicit_source=pick_first_text(stage4_cfg.get("api_source")),
                explicit_api_base=pick_first_text(stage4_cfg.get("api_base")),
                explicit_api_key=pick_first_text(stage4_cfg.get("api_key")),
            )
            default_api_source = str(endpoint.get("api_source", "") or "")
            default_api_base = str(endpoint.get("api_base", "") or "")
        items.append(
            {
                "engine": engine,
                "scene_id": scene_id,
                "config_path": str(path.resolve()),
                "display_name": f"{engine}:{scene_id}",
                "default_model": default_model,
                "default_api_source": default_api_source,
                "default_api_base": default_api_base,
            }
        )
    return sorted(items, key=lambda item: (item["engine"], item["scene_id"]))


def _load_scene_config_from_catalog(
    *,
    engine: str,
    scene_id: str,
    catalog: list[dict[str, Any]],
    fallback_config_path: Path,
) -> tuple[dict[str, Any], Path]:
    engine_norm = str(engine or "airsim").strip().lower()
    scene_text = str(scene_id or "").strip()
    for item in catalog:
        if item["engine"] == engine_norm and item["scene_id"] == scene_text:
            path = Path(item["config_path"]).resolve()
            return _load_yaml(path), path
    return _load_yaml(fallback_config_path), fallback_config_path.resolve()


def _filter_samples(
    samples: list[dict[str, Any]],
    *,
    view_definitions: list[str] | None = None,
    task_types: list[str] | None = None,
    difficulties: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    out = list(samples)
    if view_definitions:
        allow = {str(x) for x in view_definitions}
        out = [sample for sample in out if str(sample.get("view_definition", "")) in allow]
    if task_types:
        allow = {str(x) for x in task_types}
        out = [sample for sample in out if str(sample.get("task_type", "")) in allow]
    if difficulties:
        allow = {str(x) for x in difficulties}
        out = [sample for sample in out if str(sample.get("difficulty", "")) in allow]
    if limit is not None and limit > 0:
        out = out[: int(limit)]
    return out


def _load_valid_instances(config: dict[str, Any], *, scene_root: Path, scene_id: str) -> list[dict[str, Any]]:
    raw_root, valid_instances_path = _resolve_stage2_review_paths(config, scene_root=scene_root, scene_id=scene_id)
    if not valid_instances_path.exists():
        raise FileNotFoundError(f"missing valid instances json: {valid_instances_path}")
    payload = json.loads(valid_instances_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = list(payload.get("valid_instances", []) or [])
    elif isinstance(payload, list):
        items = list(payload)
    else:
        raise RuntimeError(f"invalid valid_instances payload: {valid_instances_path}")

    prepared: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("annotation_status", "") or "").strip().lower() != "labeled":
            continue
        if str(item.get("review_action", "") or "").strip().lower() not in {"keep", ""}:
            continue
        landmark_description = str(item.get("landmark_description", "") or item.get("description", "") or "").strip()
        if not landmark_description:
            continue
        rgb_views = []
        for view in list(item.get("rgb_views", []) or []):
            if not isinstance(view, dict):
                continue
            key = _normalize_view_key(view.get("view_direction", None))
            obj_label = _view_key_to_label(key)
            if obj_label is None:
                continue
            if bool(view.get("bbox_2d_valid", False)) is False:
                continue
            abs_path = _resolve_stage2_image_path(scene_root, raw_root, str(view.get("path", "") or ""))
            if abs_path is None:
                continue
            bbox_norm = _bbox_to_norm(view.get("bbox_2d_xyxy", None), view.get("bbox_2d_image_size", None))
            if bbox_norm is None:
                continue
            rgb_views.append(
                {
                    "path": str(abs_path),
                    "relative_path": str(view.get("path", "") or ""),
                    "object_label": obj_label,
                    "observer_label": _object_to_observer_label(obj_label),
                    "bbox_xyxy_px": [float(v) for v in view.get("bbox_2d_xyxy", [])],
                    "bbox_xyxy_norm": bbox_norm,
                    "bbox_2d_image_size": list(view.get("bbox_2d_image_size", []) or []),
                    "is_query_view": bool(view.get("is_query_view", False)),
                    "mode": str(view.get("mode", "") or "orbit"),
                    "source_view_direction": str(view.get("view_direction", "") or ""),
                    "yaw_deg": float(view.get("yaw_deg", 0.0) or 0.0),
                    "pitch_deg": float(view.get("pitch_deg", 0.0) or 0.0),
                    "pitch_offset_deg": float(view.get("pitch_offset_deg", 0.0) or 0.0),
                }
            )
        if not rgb_views:
            continue
        prepared.append(
            {
                "instance_id": str(item.get("instance_id", "") or ""),
                "class_name": str(item.get("landmark_category", "") or item.get("class_name", "") or "").strip(),
                "class_id": item.get("class_id", None),
                "landmark_description": landmark_description,
                "annotation_status": str(item.get("annotation_status", "") or ""),
                "review_action": str(item.get("review_action", "") or ""),
                "center_3d": list(item.get("center_3d", []) or []),
                "bbox_3d": item.get("bbox_3d", {}),
                "rgb_views": rgb_views,
            }
        )
    return prepared


def _label_space_for_difficulty(difficulty: str) -> list[str]:
    return LABEL_SPACE_4WAY if str(difficulty) == "4way" else LABEL_SPACE_8WAY


def _eligible_views_for_definition(views: list[dict[str, Any]], definition: str, difficulty: str) -> list[dict[str, Any]]:
    label_space = set(_label_space_for_difficulty(difficulty))
    label_key = "object_label" if definition == "Object-Centric View" else "observer_label"
    return [view for view in views if str(view.get(label_key, "")) in label_space and str(view.get("mode", "orbit")) == "orbit"]


def _task_kind(task_type: str) -> str:
    return str(TASK_KIND.get(str(task_type), "single_choice_bbox"))


def _entry_landmark_description(entry: dict[str, Any]) -> str:
    return (
        str(entry.get("landmark_description", "") or "").strip()
        or str(entry.get("class_name", "") or "").strip()
        or str(entry.get("instance_id", "") or "").strip()
        or "the landmark"
    )


def _rotate_observer_label(reference_label: str, delta_deg: int, direction: str, difficulty: str) -> str:
    allowed = {str(x) for x in _label_space_for_difficulty(difficulty)}
    ref_angle = _angle_for_label(reference_label)
    offset = float(delta_deg) if str(direction).lower() == "ccw" else -float(delta_deg)
    target_angle = (ref_angle + offset) % 360.0
    for label, angle in VIEW_ANGLE_DEG.items():
        if label in allowed and abs(float(angle) - target_angle) < 1e-6:
            return str(label)
    return ""


def _angle_for_label(label: str) -> float:
    return float(VIEW_ANGLE_DEG.get(str(label), 0.0))


def _cw_ccw_from_labels(reference_label: str, query_label: str) -> tuple[int, int]:
    ref = _angle_for_label(reference_label)
    query = _angle_for_label(query_label)
    ccw = int(round((query - ref) % 360.0))
    cw = int(round((ref - query) % 360.0))
    return cw, ccw


def _format_orbit_action(direction: str, angle_deg: int) -> str:
    dir_text = "clockwise" if str(direction) == "cw" else "counterclockwise"
    return render_prompt_template(
        get_config_template("behavior_templates", "stage4", "orbit_action"),
        {
            "angle_deg": int(angle_deg),
            "direction_text": dir_text,
        },
    ).strip()


def _bbox_dict_to_list(bbox_3d: Any) -> list[float]:
    if isinstance(bbox_3d, list):
        vals = [float(v) for v in bbox_3d]
        if len(vals) >= 7:
            return vals[:7]
    if isinstance(bbox_3d, dict):
        center = [0.0, 0.0, 0.0]
        size = list(bbox_3d.get("size", []) or [])
        if len(size) < 3:
            pmin = list(bbox_3d.get("min", []) or [])
            pmax = list(bbox_3d.get("max", []) or [])
            if len(pmin) >= 3 and len(pmax) >= 3:
                size = [float(pmax[i] - pmin[i]) for i in range(3)]
                center = [float((pmin[i] + pmax[i]) * 0.5) for i in range(3)]
        yaw = float(bbox_3d.get("yaw_deg", 0.0) or 0.0)
        if len(size) >= 3:
            return [float(center[0]), float(center[1]), float(center[2]), float(size[0]), float(size[1]), float(size[2]), yaw]
    return [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0]


def _target_bbox_corners_world(center_3d: list[float], bbox_list: list[float]) -> np.ndarray:
    cx, cy, cz = [float(center_3d[i]) for i in range(3)]
    sx = float(bbox_list[3]) if len(bbox_list) > 3 else 3.0
    sy = float(bbox_list[4]) if len(bbox_list) > 4 else 3.0
    sz = float(bbox_list[5]) if len(bbox_list) > 5 else 3.0
    yaw_deg = float(bbox_list[6]) if len(bbox_list) > 6 else 0.0
    hx, hy, hz = max(0.2, sx * 0.5), max(0.2, sy * 0.5), max(0.2, sz * 0.5)
    pts = np.asarray(
        [
            [-hx, -hy, -hz],
            [-hx, -hy, hz],
            [-hx, hy, -hz],
            [-hx, hy, hz],
            [hx, -hy, -hz],
            [hx, -hy, hz],
            [hx, hy, -hz],
            [hx, hy, hz],
        ],
        dtype=np.float32,
    )
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    out = pts @ rot.T
    out[:, 0] += cx
    out[:, 1] += cy
    out[:, 2] += cz
    return out


def _project_bbox_xyxy_norm(
    *,
    target_center_3d: list[float],
    target_bbox_list: list[float],
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    require_full_in_frame: bool = False,
) -> list[float] | None:
    corners = _target_bbox_corners_world(target_center_3d, target_bbox_list)
    rel = corners - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1e-3
    if require_full_in_frame and not np.all(valid):
        return None
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
    if require_full_in_frame and not np.all(in_frame):
        return None
    if not np.any(in_frame):
        return None
    px = px[in_frame]
    py = py[in_frame]
    x0, y0 = float(np.min(px)), float(np.min(py))
    x1, y1 = float(np.max(px)), float(np.max(py))
    if x1 - x0 < 4.0 or y1 - y0 < 4.0:
        return None
    return [
        max(0.0, min(1.0, x0 / float(width))),
        max(0.0, min(1.0, y0 / float(height))),
        max(0.0, min(1.0, x1 / float(width))),
        max(0.0, min(1.0, y1 / float(height))),
    ]


def _adjust_env_camera_for_full_bbox(
    *,
    target_center_3d: list[float],
    target_bbox_list: list[float],
    eye: np.ndarray,
    forward_base: np.ndarray,
    right_base: np.ndarray,
    cam_up_base: np.ndarray,
    desired_yaw_delta_deg: float,
    difficulty: str,
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float], str] | None:
    base = float(desired_yaw_delta_deg)
    candidate_offsets = [0.0, -8.0, 8.0, -15.0, 15.0, -22.0, 22.0, -30.0, 30.0]
    best_full: tuple[float, np.ndarray, np.ndarray, np.ndarray, list[float], str] | None = None
    best_full_score = float("inf")
    best_partial: tuple[float, np.ndarray, np.ndarray, np.ndarray, list[float], str] | None = None
    best_partial_score = -1.0
    for offset in candidate_offsets:
        yaw_delta_deg = base + float(offset)
        forward, right, cam_up = _rotate_camera_axes_yaw(
            forward=np.asarray(forward_base, dtype=np.float32),
            right=np.asarray(right_base, dtype=np.float32),
            cam_up=np.asarray(cam_up_base, dtype=np.float32),
            yaw_delta_deg=yaw_delta_deg,
        )
        bbox_full = _project_bbox_xyxy_norm(
            target_center_3d=target_center_3d,
            target_bbox_list=target_bbox_list,
            eye=np.asarray(eye, dtype=np.float32),
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=int(width),
            height=int(height),
            fov_deg=float(fov_deg),
            require_full_in_frame=True,
        )
        bbox_partial = _project_bbox_xyxy_norm(
            target_center_3d=target_center_3d,
            target_bbox_list=target_bbox_list,
            eye=np.asarray(eye, dtype=np.float32),
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=int(width),
            height=int(height),
            fov_deg=float(fov_deg),
            require_full_in_frame=False,
        )
        if bbox_full is None and bbox_partial is None:
            continue
        direction_label = _env_direction_label(
            target_center_3d=target_center_3d,
            eye=np.asarray(eye, dtype=np.float32),
            right=right,
            forward=forward,
            difficulty=str(difficulty),
        )
        if bbox_full is not None:
            score = abs(float(offset))
            if score < best_full_score:
                best_full_score = score
                best_full = (yaw_delta_deg, forward, right, cam_up, bbox_full, direction_label)
        if bbox_partial is not None:
            x0, y0, x1, y1 = [float(v) for v in list(bbox_partial)[:4]]
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            score = area - 0.001 * abs(float(offset))
            if score > best_partial_score:
                best_partial_score = score
                best_partial = (yaw_delta_deg, forward, right, cam_up, bbox_partial, direction_label)
    chosen = best_full or best_partial
    if chosen is None:
        return None
    yaw_delta_deg, forward, right, cam_up, bbox_norm, direction_label = chosen
    return forward, right, cam_up, float(yaw_delta_deg), list(bbox_norm), str(direction_label)


def _env_direction_label(
    *,
    target_center_3d: list[float],
    eye: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    difficulty: str,
) -> str:
    rel = np.asarray(target_center_3d, dtype=np.float32) - eye.astype(np.float32)
    x_cam = float(rel @ right)
    z_cam = float(rel @ forward)
    center_eps = max(1e-6, abs(z_cam) * 0.001)
    if z_cam >= 0.0:
        if abs(x_cam) <= center_eps:
            return "Front"
        if str(difficulty) == "4way":
            return "Right" if x_cam > 0.0 else "Left"
        return "Front-Right" if x_cam > 0.0 else "Front-Left"
    if abs(x_cam) <= center_eps:
        return "Back"
    if str(difficulty) == "4way":
        return "Right" if x_cam > 0.0 else "Left"
    return "Back-Right" if x_cam > 0.0 else "Back-Left"


def _rotate_camera_axes_yaw(
    *,
    forward: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    yaw_delta_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(float(yaw_delta_deg))
    c = float(math.cos(yaw))
    s = float(math.sin(yaw))
    rot = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    out_forward = rot @ np.asarray(forward, dtype=np.float32)
    out_right = rot @ np.asarray(right, dtype=np.float32)
    out_up = rot @ np.asarray(cam_up, dtype=np.float32)
    return out_forward, out_right, out_up


def _format_label_options(options: list[dict[str, str]]) -> str:
    return "\n".join(f"{opt['option_id']}. {opt['label']}" for opt in options)


def _sample_four_options(label_space: list[str], gold_label: str, rng: random.Random) -> tuple[list[dict[str, str]], str]:
    choices = [str(x) for x in list(label_space or []) if str(x).strip()]
    gold = str(gold_label or "").strip()
    if not gold:
        raise RuntimeError("missing_gold_label")
    if gold not in choices:
        choices.append(gold)
    distractors = [label for label in choices if label != gold]
    if len(distractors) < 3:
        raise RuntimeError(f"insufficient_distractors_for_label_space: gold={gold}")
    picked = [gold] + rng.sample(distractors, 3)
    rng.shuffle(picked)
    options = [{"option_id": opt_id, "label": label} for opt_id, label in zip(OPTION_IDS, picked)]
    answer_option_id = next(opt["option_id"] for opt in options if opt["label"] == gold)
    return options, answer_option_id


def _env_position_label_space_for_difficulty(difficulty: str) -> list[str]:
    return ENV_POSITION_LABEL_SPACE_4WAY if str(difficulty) == "4way" else ENV_POSITION_LABEL_SPACE_8WAY


def _build_prompt(sample: dict[str, Any]) -> str:
    task_type = str(sample["task_type"])
    if task_type == "self_where":
        return render_prompt_template(
            get_prompt_template("stage4", "self_where", "user"),
            {
                "options_text": _format_label_options(list(sample["label_options"])),
                "landmark_description": str(sample.get("landmark_description", "") or "the landmark"),
                "reference_object_view": str(sample.get("reference_object_view", "") or "Front"),
            },
        )
    if task_type == "self_what":
        return render_prompt_template(
            get_prompt_template("stage4", "self_what", "user"),
            {
                "landmark_description": str(sample.get("landmark_description", "") or "the landmark"),
                "reference_object_view": str(sample.get("reference_object_view", "") or "Front"),
                "behavior_instance": str(sample.get("behavior_instance", "") or "an orbit around the landmark"),
            },
        )
    if task_type == "env_where":
        return render_prompt_template(
            get_prompt_template("stage4", "env_where", "user"),
            {
                "options_text": _format_label_options(list(sample["label_options"])),
                "landmark_description": str(sample.get("landmark_description", "") or "the landmark"),
                "reference_object_view": str(sample.get("reference_object_view", "") or "Front"),
            },
        )
    if task_type == "env_how":
        return render_prompt_template(
            get_prompt_template("stage4", "env_how", "user"),
            {
                "options_text": _format_label_options(list(sample["label_options"])),
                "landmark_description": str(sample.get("landmark_description", "") or "the landmark"),
                "reference_object_view": str(sample.get("reference_object_view", "") or "Front"),
            },
        )
    definition = str(sample["view_definition"])
    if task_type == "label_multiple_choice":
        return render_prompt_template(
            get_prompt_template("stage4", "label_multiple_choice", "user"),
            {
                "definition": definition,
                "options_text": _format_label_options(list(sample["label_options"])),
            },
        )
    return render_prompt_template(
        get_prompt_template("stage4", "image_multiple_choice", "user"),
        {
            "target_view": sample["target_view"],
            "definition": definition,
        },
    )


def _definition_text(definition: str) -> str:
    if definition == "Object-Centric View":
        return "Object-Centric View means the visible side of the landmark in the landmark's own coordinate frame."
    return (
        "Observer-Centric View means the camera-side viewpoint category relative to the landmark, namely from which side the camera observes the landmark. "
        "Do not interpret Observer-Centric View as screen-space left/right position inside the image."
    )


def _build_system_prompt(sample: dict[str, Any]) -> str:
    task_type = str(sample["task_type"])
    definition = str(sample.get("view_definition", "") or "")
    definition_text = _definition_text(definition)
    if task_type == "self_where":
        return render_prompt_template(
            get_prompt_template("stage4", "self_where", "system"),
            {"landmark_description": str(sample.get("landmark_description", "") or "the landmark")},
        )
    if task_type == "self_what":
        return render_prompt_template(
            get_prompt_template("stage4", "self_what", "system"),
            {"landmark_description": str(sample.get("landmark_description", "") or "the landmark")},
        )
    if task_type == "env_where":
        return render_prompt_template(
            get_prompt_template("stage4", "env_where", "system"),
            {"landmark_description": str(sample.get("landmark_description", "") or "the landmark")},
        )
    if task_type == "env_how":
        return render_prompt_template(
            get_prompt_template("stage4", "env_how", "system"),
            {"landmark_description": str(sample.get("landmark_description", "") or "the landmark")},
        )
    if task_type == "label_multiple_choice":
        return render_prompt_template(
            get_prompt_template("stage4", "label_multiple_choice", "system"),
            {
                "definition": definition,
                "definition_text": definition_text,
            },
        )
    return render_prompt_template(
        get_prompt_template("stage4", "image_multiple_choice", "system"),
        {
            "definition": definition,
            "definition_text": definition_text,
            "target_view": sample["target_view"],
        },
    )


def _normalize_category_filters(values: list[Any] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in list(values or []):
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _filter_entries_by_categories(entries: list[dict[str, Any]], categories: list[str] | None) -> list[dict[str, Any]]:
    normalized = _normalize_category_filters(categories)
    if not normalized:
        return list(entries)
    allow = {item.lower() for item in normalized}
    return [entry for entry in entries if str(entry.get("class_name", "") or "").strip().lower() in allow]


def _filter_entries_by_landmark_ids(entries: list[dict[str, Any]], landmark_ids: list[str] | None) -> list[dict[str, Any]]:
    wanted = {str(x).strip() for x in list(landmark_ids or []) if str(x).strip()}
    if not wanted:
        return list(entries)
    return [entry for entry in entries if str(entry.get("instance_id", "") or "").strip() in wanted]


def _entry_has_diagonal_view(entry: dict[str, Any]) -> bool:
    diagonal = {"Front-Left", "Front-Right", "Back-Left", "Back-Right"}
    for view in list(entry.get("rgb_views", []) or []):
        if str(view.get("object_label", "") or "").strip() in diagonal:
            return True
    return False


def _sample_label_task(
    *,
    entry: dict[str, Any],
    definition: str,
    difficulty: str,
    rng: random.Random,
    reference_main_only: bool,
    stage4_root: Path,
    scene_id: str,
    engine: str,
    sample_idx: int,
) -> dict[str, Any] | None:
    views = _eligible_views_for_definition(list(entry["rgb_views"]), definition, difficulty)
    if not views:
        return None
    reference_pool = [v for v in views if v["is_query_view"]] if reference_main_only else list(views)
    if not reference_pool:
        reference_pool = list(views)
    reference_view = rng.choice(reference_pool)
    target_pool = [v for v in views if v["path"] != reference_view["path"]]
    if not target_pool:
        target_pool = list(views)
    if not target_pool:
        return None
    target_view = rng.choice(target_pool)
    gold_label = target_view["object_label"] if definition == "Object-Centric View" else target_view["observer_label"]
    label_space = list(_label_space_for_difficulty(difficulty))
    distractors = [label for label in label_space if label != gold_label]
    if len(distractors) < 3:
        return None
    picked = [gold_label] + rng.sample(distractors, 3)
    rng.shuffle(picked)
    label_options = [{"option_id": opt_id, "label": label} for opt_id, label in zip(OPTION_IDS, picked)]
    answer_option_id = next(opt["option_id"] for opt in label_options if opt["label"] == gold_label)

    stage4_img_cfg = build_image_compression_cfg(_stage4_cfg({}))
    overlay_dir = _ensure_dir(stage4_root / "assets" / "reference_bbox" / entry["instance_id"])
    sample_prefix = f"{scene_id}_{entry['instance_id']}_{_safe_name(definition)}_{difficulty}_{sample_idx:06d}"
    ref_overlay_path = _draw_reference_bbox(
        source_image=Path(reference_view["path"]),
        bbox_xyxy=list(reference_view["bbox_xyxy_px"]),
        output_path=overlay_dir / f"{sample_prefix}_ref.jpg",
        cfg=stage4_img_cfg,
    )
    view_labels = {
        "Object-Centric View": str(target_view["object_label"]),
        "Observer-Centric View": str(target_view["observer_label"]),
    }
    return {
        "sample_id": sample_prefix,
        "scene_id": scene_id,
        "engine": engine,
        "landmark_id": entry["instance_id"],
        "landmark_category": entry["class_name"] or f"class_{entry.get('class_id', 'unknown')}",
        "task_family": "qa",
        "view_definition": definition,
        "task_type": "label_multiple_choice",
        "difficulty": difficulty,
        "full_label_space": label_space,
        "label_options": label_options,
        "reference_image": _path_for_json(Path(reference_view["path"])),
        "reference_image_with_bbox": _path_for_json(ref_overlay_path),
        "reference_bbox_xyxy_norm": list(reference_view["bbox_xyxy_norm"]),
        "reference_view": reference_view["object_label"] if definition == "Object-Centric View" else reference_view["observer_label"],
        "target_image": _path_for_json(Path(target_view["path"])),
        "answer_option_id": answer_option_id,
        "answer_label": gold_label,
        "answer_bbox_xyxy_norm": list(target_view["bbox_xyxy_norm"]),
        "view_labels": view_labels,
        "metadata": {
            "reference_is_main_view": bool(reference_view["is_query_view"]),
            "source_view_direction": str(target_view["source_view_direction"]),
        },
    }


def _sample_image_task(
    *,
    entry: dict[str, Any],
    definition: str,
    difficulty: str,
    rng: random.Random,
    reference_main_only: bool,
    stage4_root: Path,
    scene_id: str,
    engine: str,
    sample_idx: int,
) -> dict[str, Any] | None:
    views = _eligible_views_for_definition(list(entry["rgb_views"]), definition, difficulty)
    if not views:
        return None
    reference_pool = [v for v in views if v["is_query_view"]] if reference_main_only else list(views)
    if not reference_pool:
        reference_pool = list(views)
    if not reference_pool:
        return None
    reference_view = rng.choice(reference_pool)
    label_key = "object_label" if definition == "Object-Centric View" else "observer_label"
    label_to_views: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in views:
        label_to_views[str(view[label_key])].append(view)
    if len(label_to_views) < 4:
        return None
    picked_labels = rng.sample(sorted(label_to_views.keys()), 4)
    target_view_label = rng.choice(picked_labels)
    options: list[dict[str, Any]] = []
    answer_option_id = ""
    answer_bbox_xyxy_norm: list[float] | None = None
    for opt_id, label in zip(OPTION_IDS, picked_labels):
        candidates = [v for v in label_to_views[label] if v["path"] != reference_view["path"]]
        if not candidates:
            candidates = list(label_to_views[label])
        chosen = rng.choice(candidates)
        option = {
            "option_id": opt_id,
            "image": _path_for_json(Path(chosen["path"])),
            "view_under_definition": label,
            "bbox_xyxy_norm": list(chosen["bbox_xyxy_norm"]),
        }
        options.append(option)
        if label == target_view_label:
            answer_option_id = opt_id
            answer_bbox_xyxy_norm = list(chosen["bbox_xyxy_norm"])
    if not answer_option_id or answer_bbox_xyxy_norm is None:
        return None

    stage4_img_cfg = build_image_compression_cfg(_stage4_cfg({}))
    overlay_dir = _ensure_dir(stage4_root / "assets" / "reference_bbox" / entry["instance_id"])
    sample_prefix = f"{scene_id}_{entry['instance_id']}_{_safe_name(definition)}_{difficulty}_{sample_idx:06d}"
    ref_overlay_path = _draw_reference_bbox(
        source_image=Path(reference_view["path"]),
        bbox_xyxy=list(reference_view["bbox_xyxy_px"]),
        output_path=overlay_dir / f"{sample_prefix}_ref.jpg",
        cfg=stage4_img_cfg,
    )
    return {
        "sample_id": sample_prefix,
        "scene_id": scene_id,
        "engine": engine,
        "landmark_id": entry["instance_id"],
        "landmark_category": entry["class_name"] or f"class_{entry.get('class_id', 'unknown')}",
        "task_family": "qa",
        "view_definition": definition,
        "task_type": "image_multiple_choice",
        "difficulty": difficulty,
        "target_label_space": list(_label_space_for_difficulty(difficulty)),
        "reference_image": _path_for_json(Path(reference_view["path"])),
        "reference_image_with_bbox": _path_for_json(ref_overlay_path),
        "reference_bbox_xyxy_norm": list(reference_view["bbox_xyxy_norm"]),
        "reference_view": reference_view["object_label"] if definition == "Object-Centric View" else reference_view["observer_label"],
        "target_view": target_view_label,
        "candidates": options,
        "answer_option_id": answer_option_id,
        "answer_bbox_xyxy_norm": answer_bbox_xyxy_norm,
        "reference_definition": definition,
        "target_definition": definition,
        "metadata": {
            "reference_is_main_view": bool(reference_view["is_query_view"]),
            "source_view_direction": str(reference_view["source_view_direction"]),
        },
    }


def _sample_self_where_pair(
    *,
    entry: dict[str, Any],
    difficulty: str,
    rng: random.Random,
    reference_main_only: bool,
    stage4_root: Path,
    scene_id: str,
    engine: str,
    sample_idx: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    views = _eligible_views_for_definition(list(entry["rgb_views"]), "Observer-Centric View", difficulty)
    if not views:
        return None, None
    reference_pool = [v for v in views if v["is_query_view"]] if reference_main_only else list(views)
    if not reference_pool:
        reference_pool = list(views)
    reference_view = rng.choice(reference_pool)
    target_pool = [v for v in views if v["path"] != reference_view["path"]]
    if not target_pool:
        return None, None
    target_view = rng.choice(target_pool)
    gold_label = str(target_view["object_label"])
    label_space = list(_label_space_for_difficulty(difficulty))
    distractors = [label for label in label_space if label != gold_label]
    if len(distractors) < 3:
        return None, None
    picked = [gold_label] + rng.sample(distractors, 3)
    rng.shuffle(picked)
    label_options = [{"option_id": opt_id, "label": label} for opt_id, label in zip(OPTION_IDS, picked)]
    answer_option_id = next(opt["option_id"] for opt in label_options if opt["label"] == gold_label)
    sample_prefix = f"{scene_id}_{entry['instance_id']}_self_shared_{difficulty}_{sample_idx:06d}"
    stage4_img_cfg = build_image_compression_cfg(_stage4_cfg(config))
    overlay_dir = _ensure_dir(stage4_root / "assets" / "reference_bbox" / entry["instance_id"])
    ref_overlay_path = _draw_reference_bbox(
        source_image=Path(reference_view["path"]),
        bbox_xyxy=list(reference_view["bbox_xyxy_px"]),
        output_path=overlay_dir / f"{sample_prefix}_ref.jpg",
        cfg=stage4_img_cfg,
    )
    landmark_description = _entry_landmark_description(entry)
    shared = {
        "scene_id": scene_id,
        "engine": engine,
        "landmark_id": entry["instance_id"],
        "landmark_category": entry["class_name"] or f"class_{entry.get('class_id', 'unknown')}",
        "landmark_description": landmark_description,
        "task_family": "qa",
        "task_group": "self-aware",
        "difficulty": difficulty,
        "reference_image": _path_for_json(Path(reference_view["path"])),
        "reference_image_with_bbox": _path_for_json(ref_overlay_path),
        "reference_bbox_xyxy_norm": list(reference_view["bbox_xyxy_norm"]),
        "reference_view": str(reference_view["observer_label"]),
        "reference_object_view": str(reference_view.get("object_label", "") or ""),
        "target_image": _path_for_json(Path(target_view["path"])),
        "answer_bbox_xyxy_norm": list(target_view["bbox_xyxy_norm"]),
        "metadata": {
            "reference_is_main_view": bool(reference_view["is_query_view"]),
            "reference_label": str(reference_view["observer_label"]),
            "query_label": str(target_view["observer_label"]),
        },
    }
    self_where = {
        **shared,
        "sample_id": f"{sample_prefix}_where",
        "view_definition": "Self-Aware",
        "task_type": "self_where",
        "reference_view": str(reference_view.get("object_label", "") or ""),
        "label_options": label_options,
        "answer_option_id": answer_option_id,
        "metadata": {
            "reference_is_main_view": bool(reference_view["is_query_view"]),
            "reference_label": str(reference_view.get("object_label", "") or ""),
            "query_label": str(target_view.get("object_label", "") or ""),
        },
    }
    # For self_what in 4way mode, reference/target must still come from upright directions,
    # but the four answer images may be any valid orbit views.
    all_orbit_views = [
        dict(v)
        for v in list(entry.get("rgb_views", []) or [])
        if isinstance(v, dict) and str(v.get("mode", "orbit") or "orbit") == "orbit" and str(v.get("path", "") or "").strip()
    ]
    distinct_views: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for view in all_orbit_views:
        path_text = str(view.get("path", "") or "")
        if path_text in seen_paths:
            continue
        seen_paths.add(path_text)
        distinct_views.append(view)
    if len(distinct_views) < 4:
        return self_where, None
    non_reference_views = [v for v in views if str(v.get("path", "") or "") != str(reference_view["path"])]
    if not non_reference_views:
        return self_where, None
    target_view = rng.choice(non_reference_views)
    candidate_pool = [dict(v) for v in distinct_views if str(v.get("path", "") or "").strip() != str(reference_view["path"])]
    picked_views = [dict(target_view)]
    for view in candidate_pool:
        if len(picked_views) >= 4:
            break
        path_text = str(view.get("path", "") or "")
        if any(str(row.get("path", "") or "") == path_text for row in picked_views):
            continue
        picked_views.append(dict(view))
    if len(picked_views) < 4:
        for view in distinct_views:
            if len(picked_views) >= 4:
                break
            path_text = str(view.get("path", "") or "")
            if any(str(row.get("path", "") or "") == path_text for row in picked_views):
                continue
            picked_views.append(dict(view))
    if len(picked_views) < 4:
        return self_where, None
    rng.shuffle(picked_views)
    candidate_meta = []
    for view in picked_views:
        label = str(view.get("observer_label", "") or "")
        cw_deg, ccw_deg = _cw_ccw_from_labels(str(reference_view["observer_label"]), label)
        if cw_deg <= 180 and (cw_deg < ccw_deg or cw_deg == 180):
            direction = "cw"
            delta_deg = int(cw_deg)
        else:
            direction = "ccw"
            delta_deg = int(ccw_deg)
        candidate_meta.append((label, direction, delta_deg))
    if not candidate_meta:
        return self_where, None
    target_label = str(target_view["observer_label"])
    target_meta = next((row for row in candidate_meta if row[0] == target_label), None)
    if target_meta is None:
        return self_where, None
    _, orbit_direction, orbit_delta_deg = target_meta
    options: list[dict[str, Any]] = []
    for opt_id, chosen in zip(OPTION_IDS, picked_views):
        options.append(
            {
                "option_id": opt_id,
                "image": _path_for_json(Path(chosen["path"])),
                "view_under_definition": str(chosen["observer_label"]),
                "bbox_xyxy_norm": list(chosen["bbox_xyxy_norm"]),
            }
        )
    answer_option_ids = [row["option_id"] for row in options if row["view_under_definition"] == target_label]
    self_what = {
        **shared,
        "sample_id": f"{sample_prefix}_what",
        "view_definition": "Self-Aware",
        "task_type": "self_what",
        "multi_select": False,
        "behavior_instance": _format_orbit_action(orbit_direction, orbit_delta_deg),
        "candidates": options,
        "answer_option_ids": answer_option_ids,
        "answer_option_id": answer_option_ids[0] if answer_option_ids else "",
        "answer_bbox_xyxy_norm": list(target_view["bbox_xyxy_norm"]),
    }
    return self_where, self_what


def _stage4_env_capture_workers(config: dict[str, Any]) -> int:
    stage4_cfg = _stage4_cfg(config)
    return max(1, int(stage4_cfg.get("env_capture_parallel_workers", 24) or 24))


def _stage4_overlay_workers(config: dict[str, Any]) -> int:
    stage4_cfg = _stage4_cfg(config)
    return max(1, int(stage4_cfg.get("overlay_parallel_workers", 8) or 8))


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


def _build_stage4_bridge_config(config: dict[str, Any], *, vehicle_name: str) -> tuple[str, dict[str, Any]]:
    task_cfg = config.get("task", {}) or {}
    stage4_cfg = _stage4_cfg(config)
    engine = str(task_cfg.get("engine", "airsim")).lower()
    bridge_cfg = build_unified_bridge_config(
        config=config,
        engine=engine,
        vehicle_name=vehicle_name,
        image_width=int(stage4_cfg.get("env_capture_width", stage4_cfg.get("api_upload_max_width", 640)) or 640),
        image_height=int(stage4_cfg.get("env_capture_height", stage4_cfg.get("api_upload_max_height", 480)) or 480),
        default_width=int(stage4_cfg.get("env_capture_width", stage4_cfg.get("api_upload_max_width", 640)) or 640),
        default_height=int(stage4_cfg.get("env_capture_height", stage4_cfg.get("api_upload_max_height", 480)) or 480),
        default_fov=72.0,
    )
    bridge_cfg["camera_capture_image_types"] = [0]
    return engine, bridge_cfg


def _capture_env_requests(
    *,
    config: dict[str, Any],
    scene_id: str,
    stage4_root: Path,
    requests: list[dict[str, Any]],
    progress_cb: Any | None = None,
) -> dict[str, dict[str, Any]]:
    if not requests:
        return {}
    if cv2 is None:
        raise RuntimeError("opencv(cv2) required for stage4 env capture")
    engine, base_bridge_cfg = _build_stage4_bridge_config(config, vehicle_name="drone_1")
    if engine != "airsim":
        raise RuntimeError("stage4 env capture currently supports airsim only")
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine)
    stage4_cfg = _stage4_cfg(config)
    ensure_single_airsim_process("stage4_env_capture")
    worker_count = min(len(requests), _stage4_env_capture_workers(config))
    vehicles = [f"drone_{idx+1}" for idx in range(worker_count)]
    capture_width = int(stage4_cfg.get("env_capture_width", stage4_cfg.get("api_upload_max_width", 640)) or 640)
    capture_height = int(stage4_cfg.get("env_capture_height", stage4_cfg.get("api_upload_max_height", 480)) or 480)
    prepare_msg = (
        f"[stage4][airsim] preparing runtime: requests={len(requests)} "
        f"workers={worker_count} image_size={capture_width}x{capture_height}"
    )
    print(prepare_msg)
    if callable(progress_cb):
        progress_cb(0, max(1, len(request_list) if 'request_list' in locals() else len(requests)), prepare_msg)
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage4",
        step="render_capture_prepare",
        message=prepare_msg,
        payload=build_unified_stage_event(
            stage="stage4",
            step="render_capture_prepare",
            scene_id=scene_id,
            engine=engine,
            status="starting",
            extra={
                "request_count": int(len(requests)),
                "worker_count": int(worker_count),
                "image_width": int(capture_width),
                "image_height": int(capture_height),
            },
        ),
    )
    runtime_port, bootstrap_bridge, launched_by_bridge, configured_port = prepare_airsim_runtime_unified(
        config=config,
        scene_id=scene_id,
        base_bridge_cfg=base_bridge_cfg,
        vehicle_name=str(vehicles[0]),
        vehicle_names=[str(v) for v in vehicles],
    )
    startup_port_msg = format_unified_startup_ports_message(
        stage="stage4",
        engine=engine,
        configured_sim_port=int(configured_port),
        runtime_sim_port=int(runtime_port),
        launched_by_bridge=bool(launched_by_bridge),
    )
    print(startup_port_msg)
    if callable(progress_cb):
        progress_cb(0, max(1, len(request_list) if 'request_list' in locals() else len(requests)), startup_port_msg)
    append_unified_scene_log(
        config=config,
        scene_root=scene_root,
        stage="stage4",
        step="render_capture_ready",
        message=startup_port_msg,
        payload=build_unified_stage_event(
            stage="stage4",
            step="render_capture_ready",
            scene_id=scene_id,
            engine=engine,
            status="ready",
            extra={
                "configured_sim_port": int(configured_port),
                "runtime_sim_port": int(runtime_port),
                "launched_by_bridge": bool(launched_by_bridge),
                "request_count": int(len(requests)),
                "worker_count": int(worker_count),
                "image_width": int(capture_width),
                "image_height": int(capture_height),
            },
        ),
    )
    results: dict[str, dict[str, Any]] = {}
    done_counter = {"done": 0}
    done_lock = threading.Lock()
    request_list = [dict(row) for row in list(requests or [])]
    worker_segments = _split_contiguous_segments(total_count=len(request_list), worker_count=worker_count)
    try:
        worker_ready_msg = (
            f"[stage4][airsim] scheduling worker bridges: workers={worker_count} "
            f"vehicles={','.join(vehicles)}"
        )
        print(worker_ready_msg)
        if callable(progress_cb):
            progress_cb(0, max(1, len(request_list)), worker_ready_msg)
        append_unified_scene_log(
            config=config,
            scene_root=scene_root,
            stage="stage4",
            step="render_capture_workers_ready",
            message=worker_ready_msg,
            payload=build_unified_stage_event(
                stage="stage4",
                step="render_capture_workers_ready",
                scene_id=scene_id,
                engine=engine,
                status="ready",
                extra={
                    "worker_count": int(worker_count),
                    "vehicles": [str(v) for v in vehicles],
                },
            ),
        )
        capture_start_msg = (
            f"[stage4][airsim] starting env capture: requests={len(requests)} "
            f"workers={worker_count}"
        )
        print(capture_start_msg)
        if callable(progress_cb):
            progress_cb(0, max(1, len(request_list)), capture_start_msg)
        append_unified_scene_log(
            config=config,
            scene_root=scene_root,
            stage="stage4",
            step="render_capture_start",
            message=capture_start_msg,
            payload=build_unified_stage_event(
                stage="stage4",
                step="render_capture_start",
                scene_id=scene_id,
                engine=engine,
                status="running",
                extra={
                    "request_count": int(len(requests)),
                    "worker_count": int(worker_count),
                },
            ),
        )

        def _worker(worker_id: int, segment: tuple[int, int]) -> list[dict[str, Any]]:
            st, ed = int(segment[0]), int(segment[1])
            local_cfg = dict(base_bridge_cfg)
            local_cfg["vehicle_name"] = str(vehicles[worker_id])
            local_cfg["sim_port"] = int(runtime_port)
            local_cfg["launch_sim"] = False
            local_cfg["connect_on_init"] = True
            local_cfg["auto_select_port_on_conflict"] = False
            bridge = create_bridge(engine=engine, scene_id=scene_id, config=local_cfg)
            worker_connect_msg = (
                f"[stage4][airsim] worker connected: worker={worker_id} "
                f"vehicle={vehicles[worker_id]} runtime_sim_port={runtime_port}"
            )
            print(worker_connect_msg)
            if callable(progress_cb):
                progress_cb(0, max(1, len(request_list)), worker_connect_msg)
            append_unified_scene_log(
                config=config,
                scene_root=scene_root,
                stage="stage4",
                step="render_capture_worker_connected",
                message=worker_connect_msg,
                payload=build_unified_stage_event(
                    stage="stage4",
                    step="render_capture_worker_connected",
                    scene_id=scene_id,
                    engine=engine,
                    status="ready",
                    extra={
                        "worker_id": int(worker_id),
                        "vehicle": str(vehicles[worker_id]),
                        "runtime_sim_port": int(runtime_port),
                    },
                ),
            )
            out_rows: list[dict[str, Any]] = []
            try:
                for item in request_list[st:ed]:
                    eye = np.asarray(item["eye"], dtype=np.float32)
                    forward = np.asarray(item["forward"], dtype=np.float32)
                    right = np.asarray(item["right"], dtype=np.float32)
                    cam_up = np.asarray(item["cam_up"], dtype=np.float32)
                    yaw_deg, pitch_deg = _forward_to_yaw_pitch_deg(forward)
                    bridge.set_uav_pose(
                        x=float(eye[0]),
                        y=float(eye[1]),
                        z=float(eye[2]),
                        yaw=float(yaw_deg),
                        pitch=float(pitch_deg),
                        roll=0.0,
                        vehicle_or_actor=str(vehicles[worker_id]),
                    )
                    settle_sec = max(0.0, float((stage4_cfg.get("env_capture_settle_sec", 0.03) or 0.03)))
                    if settle_sec > 0:
                        time.sleep(settle_sec)
                    rgb = bridge.capture_rgb()
                    rgb_np = np.asarray(rgb) if rgb is not None else np.empty((0, 0, 3), dtype=np.uint8)
                    if rgb_np.ndim == 2:
                        rgb_np = cv2.cvtColor(rgb_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    if rgb_np.ndim != 3 or rgb_np.shape[2] < 3:
                        continue
                    image = rgb_np[:, :, :3].copy()
                    h, w = image.shape[:2]
                    bbox_norm = _project_bbox_xyxy_norm(
                        target_center_3d=item["target_center_3d"],
                        target_bbox_list=item["target_bbox_list"],
                        eye=eye,
                        right=right,
                        cam_up=cam_up,
                        forward=forward,
                        width=int(w),
                        height=int(h),
                        fov_deg=float((config.get("camera", {}) or {}).get("fov", 72.0) or 72.0),
                    )
                    if bbox_norm is None:
                        cached_bbox = list(item.get("bbox_xyxy_norm", []) or [])
                        bbox_norm = cached_bbox if len(cached_bbox) == 4 else None
                    if bbox_norm is None:
                        continue
                    out_path = Path(item["output_path"])
                    _ensure_dir(out_path.parent)
                    stage4_img_cfg = build_image_compression_cfg(stage4_cfg)
                    out_path = save_bgr_image(image, out_path, cfg=stage4_img_cfg)
                    out_rows.append(
                        {
                            "request_id": str(item["request_id"]),
                            "image": _path_for_json(out_path),
                            "bbox_xyxy_norm": bbox_norm,
                            "direction_label": item["direction_label"],
                        }
                    )
                    if callable(progress_cb):
                        with done_lock:
                            done_counter["done"] += 1
                            done = int(done_counter["done"])
                        progress_cb(done, len(request_list), f"captured {item['request_id']}")
                return out_rows
            finally:
                try:
                    bridge.shutdown()
                except Exception:
                    pass

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_worker, worker_id, worker_segments[worker_id]) for worker_id in range(len(worker_segments))]
            for fut in as_completed(futures):
                for row in fut.result():
                    results[str(row["request_id"])] = dict(row)
    finally:
        try:
            bootstrap_bridge.shutdown()
        except Exception:
            pass
    return results


def _build_env_request(
    *,
    entry: dict[str, Any],
    difficulty: str,
    rng: random.Random,
    stage4_root: Path,
    scene_id: str,
    sample_idx: int,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    views = _eligible_views_for_definition(list(entry["rgb_views"]), "Object-Centric View", difficulty)
    if not views:
        return None
    main_views = [v for v in views if v["is_query_view"]]
    source_view = rng.choice(main_views if main_views else views)
    observed_pool = [v for v in views if str(v.get("path", "") or "") != str(source_view.get("path", "") or "")]
    if not observed_pool:
        observed_pool = list(views)
    if not observed_pool:
        return None
    observed_pool = list(observed_pool)
    rng.shuffle(observed_pool)
    center_3d = list(entry.get("center_3d", []) or [])
    bbox_list = _bbox_dict_to_list(entry.get("bbox_3d", {}))
    if len(center_3d) < 3 or len(bbox_list) < 7:
        return None
    target_points = _target_bbox_corners_world(center_3d, bbox_list)
    pose_params = _build_camera_pose_params(dict(config.get("stage2", {}) or {}))
    _, source_eye, source_forward, source_right, source_up = _camera_pose_for_yaw(
        target_points_xyz=target_points,
        yaw_deg=float(source_view.get("yaw_deg", 0.0) or 0.0),
        extra_pitch_deg=float(source_view.get("pitch_offset_deg", 0.0) or 0.0),
        pose_params=pose_params,
    )
    fov_deg = float((config.get("camera", {}) or {}).get("fov", 72.0) or 72.0)
    capture_width = int((_stage4_cfg(config).get("env_capture_width", (_stage4_cfg(config).get("api_upload_max_width", 640))) or 640))
    capture_height = int((_stage4_cfg(config).get("env_capture_height", (_stage4_cfg(config).get("api_upload_max_height", 480))) or 480))
    observed_view = None
    observed_eye = None
    observed_forward = None
    observed_right = None
    observed_up = None
    direction_label = ""
    yaw_delta_deg = 0.0
    bbox_norm = None
    for candidate_view in observed_pool:
        _, candidate_eye, candidate_forward_base, candidate_right_base, candidate_up_base = _camera_pose_for_yaw(
            target_points_xyz=target_points,
            yaw_deg=float(candidate_view.get("yaw_deg", 0.0) or 0.0),
            extra_pitch_deg=float(candidate_view.get("pitch_offset_deg", 0.0) or 0.0),
            pose_params=pose_params,
        )
        desired_yaw_delta_deg = float(rng.choice([30.0, 0.0, -30.0]))
        adjusted = _adjust_env_camera_for_full_bbox(
            target_center_3d=center_3d,
            target_bbox_list=bbox_list,
            eye=np.asarray(candidate_eye, dtype=np.float32),
            forward_base=np.asarray(candidate_forward_base, dtype=np.float32),
            right_base=np.asarray(candidate_right_base, dtype=np.float32),
            cam_up_base=np.asarray(candidate_up_base, dtype=np.float32),
            desired_yaw_delta_deg=desired_yaw_delta_deg,
            difficulty=str(difficulty),
            width=int(capture_width),
            height=int(capture_height),
            fov_deg=float(fov_deg),
        )
        if adjusted is None:
            continue
        observed_view = candidate_view
        observed_eye = np.asarray(candidate_eye, dtype=np.float32)
        observed_forward, observed_right, observed_up, yaw_delta_deg, bbox_norm, direction_label = adjusted
        break
    if observed_view is None or observed_eye is None or observed_forward is None or observed_right is None or observed_up is None:
        return None
    stage4_img_cfg = build_image_compression_cfg(_stage4_cfg(config))
    out_dir = _ensure_dir(stage4_root / "assets" / "env_observations" / str(entry["instance_id"]))
    sample_prefix = f"{scene_id}_{entry['instance_id']}_env_shared_{difficulty}_{sample_idx:06d}"
    overlay_dir = _ensure_dir(stage4_root / "assets" / "reference_bbox" / entry["instance_id"])
    ref_overlay_path = _draw_reference_bbox(
        source_image=Path(source_view["path"]),
        bbox_xyxy=list(source_view["bbox_xyxy_px"]),
        output_path=overlay_dir / f"{sample_prefix}_ref.jpg",
        cfg=stage4_img_cfg,
    )
    return {
        "request_id": sample_prefix,
        "output_path": str(preferred_output_path(out_dir / f"{sample_prefix}.jpg", compress_enabled=bool(stage4_img_cfg.get("enabled", True))).resolve()),
        "target_center_3d": center_3d,
        "target_bbox_list": bbox_list,
        "eye": observed_eye.tolist(),
        "forward": observed_forward.tolist(),
        "right": observed_right.tolist(),
        "cam_up": observed_up.tolist(),
        "direction_label": direction_label,
        "yaw_delta_deg": float(yaw_delta_deg),
        "reference_view": source_view,
        "observed_view": observed_view,
        "bbox_xyxy_norm": list(bbox_norm or []),
        "reference_overlay": str(ref_overlay_path.resolve()),
        "sample_prefix": sample_prefix,
    }


def _stage4_prepare_entry_worker(payload: dict[str, Any]) -> dict[str, Any]:
    entry_idx = int(payload.get("entry_idx", 0) or 0)
    entry = dict(payload.get("entry", {}) or {})
    difficulty_plan = list(payload.get("difficulty_plan", []) or [])
    selected_task_types = [str(x).strip() for x in list(payload.get("selected_task_types", []) or []) if str(x).strip()]
    reference_main_only = bool(payload.get("reference_main_only", True))
    stage4_root = Path(str(payload.get("stage4_root", "") or "")).resolve()
    scene_id = str(payload.get("scene_id", "") or "")
    engine = str(payload.get("engine", "") or "")
    config = dict(payload.get("config", {}) or {})
    seed = int(payload.get("seed", 7) or 7)
    local_rng = random.Random(int(seed) + entry_idx * 1009 + 17)
    bundle_rows: list[dict[str, Any]] = []
    env_req_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, Any]] = []
    local_sample_idx = 1 + entry_idx * 4
    for difficulty_idx, difficulty in enumerate(difficulty_plan):
        sample_idx = local_sample_idx + difficulty_idx * 2
        if "self_where" in selected_task_types or "self_what" in selected_task_types:
            self_where, self_what = _sample_self_where_pair(
                entry=entry,
                difficulty=str(difficulty),
                rng=local_rng,
                reference_main_only=reference_main_only,
                stage4_root=stage4_root,
                scene_id=scene_id,
                engine=engine,
                sample_idx=sample_idx,
                config=config,
            )
            if self_where is not None and "self_where" in selected_task_types:
                bundle_rows.append(self_where)
            if self_what is not None and "self_what" in selected_task_types:
                bundle_rows.append(self_what)
        if "env_where" in selected_task_types or "env_how" in selected_task_types:
            env_request = _build_env_request(
                entry=entry,
                difficulty=str(difficulty),
                rng=local_rng,
                stage4_root=stage4_root,
                scene_id=scene_id,
                sample_idx=sample_idx + 1,
                config=config,
            )
            if env_request is not None:
                env_req_rows.append(env_request)
                pending_rows.append(
                    {
                        "entry": entry,
                        "difficulty": str(difficulty),
                        "sample_idx": sample_idx + 1,
                        "request_id": str(env_request["request_id"]),
                        "selected_task_types": list(selected_task_types),
                    }
                )
    return {
        "entry_idx": int(entry_idx),
        "samples": bundle_rows,
        "env_requests": env_req_rows,
        "pending_env_pairs": pending_rows,
    }


def generate_manifest(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    sample_count: int,
    seed: int,
    reference_main_only: bool,
    difficulties: list[str],
    view_definitions: list[str] | None = None,
    task_types: list[str] | None = None,
    landmark_categories: list[str] | None = None,
    selected_landmark_ids: list[str] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    stage4_root = _ensure_dir(_resolve_stage4_root(config, scene_root=scene_root))
    manifests_root = _ensure_dir(stage4_root / "manifests")
    entries = _load_valid_instances(config, scene_root=scene_root, scene_id=scene_id)
    selected_categories = _normalize_category_filters(landmark_categories)
    entries = _filter_entries_by_categories(entries, selected_categories)
    entries = _filter_entries_by_landmark_ids(entries, selected_landmark_ids)
    if not entries:
        raise RuntimeError(f"no_valid_instances_for_stage4: scene_id={scene_id} categories={selected_categories or 'ALL'}")

    rng = random.Random(seed)
    allowed_task_types = {str(x) for x in list(task_types or [])}
    selected_task_types = [task_type for _, task_type in TASK_SPECS if (not allowed_task_types or task_type in allowed_task_types)]
    if not selected_task_types:
        raise RuntimeError("no_task_type_selected_for_stage4_generate")

    samples: list[dict[str, Any]] = []
    env_requests: list[dict[str, Any]] = []
    pending_env_pairs: list[dict[str, Any]] = []
    entry_order = list(entries)
    rng.shuffle(entry_order)
    stage4_cfg = _stage4_cfg(config)
    configured_prepare_workers = int(stage4_cfg.get("data_prepare_parallel_workers", 0) or 0)
    if configured_prepare_workers > 0:
        prepare_workers = max(1, min(len(entry_order), configured_prepare_workers))
    else:
        cpu_total = max(1, int(os.cpu_count() or 1))
        capped = max(1, int(max(1, cpu_total * 0.5)))
        prepare_workers = max(1, min(len(entry_order), max(1, int(capped * 0.6))))

    prepared_rows: list[dict[str, Any]] = []
    if prepare_workers <= 1:
        for entry_idx, entry in enumerate(entry_order):
            prepared_rows.append(
                _stage4_prepare_entry_worker(
                    {
                        "entry_idx": int(entry_idx),
                        "entry": entry,
                        "difficulty_plan": ["4way", "8way"] if _entry_has_diagonal_view(entry) else ["4way", "4way"],
                        "selected_task_types": selected_task_types,
                        "reference_main_only": bool(reference_main_only),
                        "stage4_root": str(stage4_root),
                        "scene_id": scene_id,
                        "engine": engine,
                        "config": config,
                        "seed": int(seed),
                    }
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=prepare_workers) as executor:
            futures = {
                executor.submit(
                    _stage4_prepare_entry_worker,
                    {
                        "entry_idx": int(entry_idx),
                        "entry": entry,
                        "difficulty_plan": ["4way", "8way"] if _entry_has_diagonal_view(entry) else ["4way", "4way"],
                        "selected_task_types": selected_task_types,
                        "reference_main_only": bool(reference_main_only),
                        "stage4_root": str(stage4_root),
                        "scene_id": scene_id,
                        "engine": engine,
                        "config": config,
                        "seed": int(seed),
                    },
                ): entry_idx
                for entry_idx, entry in enumerate(entry_order)
            }
            for fut in as_completed(futures):
                prepared_rows.append(fut.result())
    prepared_rows.sort(key=lambda row: int(row.get("entry_idx", 0)))
    for row in prepared_rows:
        for sample in list(row.get("samples", []) or []):
            if len(samples) < int(sample_count):
                samples.append(sample)
        env_requests.extend(list(row.get("env_requests", []) or []))
        pending_env_pairs.extend(list(row.get("pending_env_pairs", []) or []))
        if len(samples) >= int(sample_count) and not env_requests:
            break

    env_results = _capture_env_requests(config=config, scene_id=scene_id, stage4_root=stage4_root, requests=env_requests, progress_cb=progress_callback) if env_requests else {}
    for row in pending_env_pairs:
        payload = dict(env_results.get(str(row["request_id"]), {}) or {})
        if not payload:
            continue
        entry = dict(row["entry"])
        difficulty = str(row["difficulty"])
        sample_prefix = str(payload.get("request_id", row["request_id"]))
        direction_label = str(payload.get("direction_label", "Front"))
        bbox_xyxy_norm = list(payload.get("bbox_xyxy_norm", []) or [])
        if len(bbox_xyxy_norm) != 4:
            continue
        source_req = dict(next((req for req in env_requests if str(req["request_id"]) == str(row["request_id"])), {}) or {})
        reference_view = dict(source_req.get("reference_view", {}) or {})
        reference_image = _path_for_json(Path(str(reference_view.get("path", "") or ""))) if reference_view else ""
        reference_image_with_bbox = _path_for_json(Path(str(source_req.get("reference_overlay", "") or ""))) if source_req.get("reference_overlay") else reference_image
        reference_bbox_xyxy_norm = list(reference_view.get("bbox_xyxy_norm", []) or []) if reference_view else []
        reference_object_view = str(reference_view.get("object_label", "") or "") if reference_view else ""
        target_image = str(payload.get("image", "") or "")
        if "env_where" in selected_task_types:
            label_options, answer_option_id = _sample_four_options(
                _env_position_label_space_for_difficulty(difficulty),
                direction_label,
                rng,
            )
            samples.append(
                {
                    "sample_id": f"{sample_prefix}_where",
                    "scene_id": scene_id,
                    "engine": engine,
                    "landmark_id": entry["instance_id"],
                    "landmark_category": entry["class_name"] or f"class_{entry.get('class_id', 'unknown')}",
                    "landmark_description": _entry_landmark_description(entry),
                    "task_family": "qa",
                    "task_group": "environment-aware",
                    "view_definition": "Environment-Aware",
                    "task_type": "env_where",
                    "difficulty": difficulty,
                    "label_options": label_options,
                    "reference_image": reference_image,
                    "reference_image_with_bbox": reference_image_with_bbox,
                    "reference_bbox_xyxy_norm": reference_bbox_xyxy_norm,
                    "reference_object_view": reference_object_view,
                    "target_image": target_image,
                    "answer_option_id": answer_option_id,
                    "answer_bbox_xyxy_norm": bbox_xyxy_norm,
                }
            )
        if "env_how" in selected_task_types:
            label_options, answer_option_id = _sample_four_options(
                _env_position_label_space_for_difficulty(difficulty),
                direction_label,
                rng,
            )
            samples.append(
                {
                    "sample_id": f"{sample_prefix}_how",
                    "scene_id": scene_id,
                    "engine": engine,
                    "landmark_id": entry["instance_id"],
                    "landmark_category": entry["class_name"] or f"class_{entry.get('class_id', 'unknown')}",
                    "landmark_description": _entry_landmark_description(entry),
                    "task_family": "qa",
                    "task_group": "environment-aware",
                    "view_definition": "Environment-Aware",
                    "task_type": "env_how",
                    "difficulty": difficulty,
                    "label_options": label_options,
                    "reference_image": reference_image,
                    "reference_image_with_bbox": reference_image_with_bbox,
                    "reference_bbox_xyxy_norm": reference_bbox_xyxy_norm,
                    "reference_object_view": reference_object_view,
                    "target_image": target_image,
                    "answer_option_id": answer_option_id,
                    "answer_bbox_xyxy_norm": bbox_xyxy_norm,
                }
            )

    samples = samples[: int(sample_count)]
    for sample in samples:
        sample["prompt_text"] = _build_prompt(sample)
        sample["user_prompt"] = sample["prompt_text"]
        sample["system_prompt"] = _build_system_prompt(sample)

    if not samples:
        raise RuntimeError(f"stage4_generate_failed: scene_id={scene_id}")

    generation_id = f"{scene_id}_qa_manifest_{len(samples)}samples_{_now_ts()}"
    manifest = {
        "generation_id": generation_id,
        "generated_at": _iso_now(),
        "scene_id": scene_id,
        "engine": engine,
        "sample_count": len(samples),
        "reference_main_only": bool(reference_main_only),
        "seed": int(seed),
        "difficulties": list(difficulties),
        "view_definitions": list(view_definitions or []),
        "task_types": list(task_types or []),
        "selected_landmark_categories": list(selected_categories),
        "samples": samples,
    }
    render_requests_payload = {
        "generation_id": generation_id,
        "scene_id": scene_id,
        "engine": engine,
        "generated_at": _iso_now(),
        "env_requests": env_requests,
    }
    manifest_path = manifests_root / f"{generation_id}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = manifests_root / f"{scene_id}.latest_manifest.json"
    latest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    render_requests_path, latest_render_requests_path = _stage4_render_requests_paths(stage4_root, generation_id, scene_id)
    render_requests_path.write_text(json.dumps(render_requests_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_render_requests_path.write_text(json.dumps(render_requests_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest, "manifest_path": manifest_path, "latest_path": latest_path, "stage4_root": stage4_root}


def render_manifest_assets(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    manifest_path: Path,
    rerender_existing: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError(f"invalid_stage4_manifest: {manifest_path}")
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    stage4_root = _ensure_dir(_resolve_stage4_root(config, scene_root=scene_root))
    generation_id = str(manifest.get("generation_id", "") or "").strip()
    requests_path = None
    latest_requests = _stage4_render_requests_root(stage4_root) / f"{scene_id}.latest_render_requests.json"
    if generation_id:
        candidate = _stage4_render_requests_root(stage4_root) / f"{generation_id}.json"
        if candidate.exists():
            requests_path = candidate
    if requests_path is None and latest_requests.exists():
        requests_path = latest_requests
    requests_payload = json.loads(requests_path.read_text(encoding="utf-8")) if requests_path and requests_path.exists() else {}
    env_requests_all = [dict(item) for item in list((requests_payload or {}).get("env_requests", []) or []) if isinstance(item, dict)]
    env_requests_by_id = {str(item.get("request_id", "") or ""): item for item in env_requests_all if str(item.get("request_id", "") or "").strip()}

    overlay_jobs: dict[str, dict[str, Any]] = {}
    for sample in list(manifest.get("samples", []) or []):
        if not isinstance(sample, dict):
            continue
        ref_image = str(sample.get("reference_image", "") or "").strip()
        ref_overlay = str(sample.get("reference_image_with_bbox", "") or "").strip()
        bbox_norm = list(sample.get("reference_bbox_xyxy_norm", []) or [])
        if not ref_image or not ref_overlay or len(bbox_norm) != 4:
            continue
        src_path = (WORKSPACE_ROOT / ref_image).resolve() if not Path(ref_image).is_absolute() else Path(ref_image).resolve()
        out_path = (WORKSPACE_ROOT / ref_overlay).resolve() if not Path(ref_overlay).is_absolute() else Path(ref_overlay).resolve()
        if (not rerender_existing) and out_path.exists():
            continue
        try:
            with Image.open(src_path) as img:
                w, h = img.size
        except Exception:
            continue
        bbox_xyxy = [
            float(bbox_norm[0]) * float(w),
            float(bbox_norm[1]) * float(h),
            float(bbox_norm[2]) * float(w),
            float(bbox_norm[3]) * float(h),
        ]
        overlay_jobs[str(out_path)] = {
            "source_image": src_path,
            "bbox_xyxy": bbox_xyxy,
            "output_path": out_path,
        }

    env_requests_run_by_id: dict[str, dict[str, Any]] = {}
    for sample in list(manifest.get("samples", []) or []):
        if not isinstance(sample, dict) or str(sample.get("task_type", "") or "") not in {"env_where", "env_how"}:
            continue
        sample_id = str(sample.get("sample_id", "") or "").strip()
        request_id = sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
        req = dict(env_requests_by_id.get(request_id, {}) or {})
        if not req:
            continue
        out_path = Path(str(req.get("output_path", "") or "")).resolve()
        if (not rerender_existing) and out_path.exists():
            continue
        env_requests_run_by_id[str(request_id)] = req
    env_requests_run = list(env_requests_run_by_id.values())

    total_jobs = len(overlay_jobs) + len(env_requests_run)
    done_counter = {"done": 0}

    def _advance(detail: str) -> None:
        if callable(progress_callback):
            done_counter["done"] += 1
            progress_callback(int(done_counter["done"]), int(max(1, total_jobs)), str(detail))

    stage4_img_cfg = build_image_compression_cfg(_stage4_cfg(config))
    overlay_count = 0
    if overlay_jobs:
        overlay_rows = list(overlay_jobs.values())
        overlay_workers = max(1, min(len(overlay_rows), _stage4_overlay_workers(config)))
        overlay_payloads = [
            {
                "source_image": str(Path(row["source_image"]).resolve()),
                "output_path": str(Path(row["output_path"]).resolve()),
                "bbox_xyxy": [float(v) for v in list(row["bbox_xyxy"])],
                "cfg": dict(stage4_img_cfg),
            }
            for row in overlay_rows
        ]

        if overlay_workers <= 1:
            for payload in overlay_payloads:
                name = _stage4_overlay_worker(payload)
                overlay_count += 1
                _advance(f"overlay {name}")
        else:
            with ProcessPoolExecutor(max_workers=overlay_workers) as executor:
                futures = [executor.submit(_stage4_overlay_worker, payload) for payload in overlay_payloads]
                for fut in as_completed(futures):
                    name = str(fut.result() or "")
                    overlay_count += 1
                    _advance(f"overlay {name}")

    capture_count = 0
    if env_requests_run:
        def _capture_progress(done: int, total: int, detail: str) -> None:
            if callable(progress_callback):
                progress_callback(int(overlay_count + done), int(max(1, total_jobs)), str(detail))
        _capture_env_requests(
            config=config,
            scene_id=scene_id,
            stage4_root=stage4_root,
            requests=env_requests_run,
            progress_cb=_capture_progress,
        )
        capture_count = len(env_requests_run)

    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "render_requests_path": str(requests_path.resolve()) if requests_path is not None else None,
        "overlay_count": int(overlay_count),
        "env_capture_count": int(capture_count),
        "total_jobs": int(total_jobs),
    }


def _task_display_name(view_definition: str, task_type: str) -> str:
    if str(task_type) in TASK_DISPLAY:
        return TASK_DISPLAY[str(task_type)]
    short = "Image→Label" if task_type == "label_multiple_choice" else "Label→Image"
    prefix = "ObjView" if view_definition == "Object-Centric View" else "ObsView"
    return f"{prefix} / {short}"


def _task_combo_id(view_definition: str, task_type: str, difficulty: str) -> str:
    if str(task_type) in TASK_DISPLAY:
        return f"{str(task_type)}_{str(difficulty)}"
    view_key = "obj" if view_definition == "Object-Centric View" else "obs"
    task_key = "label" if task_type == "label_multiple_choice" else "image"
    return f"{view_key}_{task_key}_{difficulty}"


def _summarize_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    samples = list(payload.get("samples", []) or [])
    landmark_ids = sorted({str(sample.get("landmark_id", "") or "") for sample in samples if str(sample.get("landmark_id", "") or "").strip()})
    categories = sorted({str(sample.get("landmark_category", "") or "") for sample in samples if str(sample.get("landmark_category", "") or "").strip()})
    task_combo_counter: dict[str, int] = defaultdict(int)
    difficulty_counter: dict[str, int] = defaultdict(int)
    for sample in samples:
        combo_id = _task_combo_id(
            str(sample.get("view_definition", "") or ""),
            str(sample.get("task_type", "") or ""),
            str(sample.get("difficulty", "") or ""),
        )
        task_combo_counter[combo_id] += 1
        difficulty_counter[str(sample.get("difficulty", "") or "")] += 1
    return {
        "used_landmark_count": len(landmark_ids),
        "used_landmark_ids": landmark_ids,
        "used_category_count": len(categories),
        "used_categories": categories,
        "selected_landmark_categories": list(payload.get("selected_landmark_categories", []) or []),
        "task_combo_counts": dict(task_combo_counter),
        "difficulty_counts": dict(difficulty_counter),
    }


def _collect_scene_landmark_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total_valid_landmarks = len(entries)
    all_categories = sorted({str(entry.get("class_name", "") or "") for entry in entries if str(entry.get("class_name", "") or "").strip()})
    landmark_category_map = {
        str(entry.get("instance_id", "") or "").strip(): str(entry.get("class_name", "") or "").strip()
        for entry in entries
        if str(entry.get("instance_id", "") or "").strip()
    }
    result: dict[str, Any] = {
        "total_valid_landmarks": total_valid_landmarks,
        "total_category_count": len(all_categories),
        "categories": all_categories,
        "landmark_category_map": landmark_category_map,
        "by_difficulty": {},
    }
    for difficulty in ["4way", "8way"]:
        per_task: dict[str, Any] = {}
        all_task_eligible_ids: set[str] = set()
        eligible_view_counts: list[int] = []
        eligible_categories: set[str] = set()
        for definition, task_type in TASK_SPECS:
            combo_id = _task_combo_id(definition, task_type, difficulty)
            eligible_ids: list[str] = []
            category_set: set[str] = set()
            view_counts: list[int] = []
            distinct_label_counts: list[int] = []
            for entry in entries:
                rgb_views = list(entry.get("rgb_views", []) or [])
                if task_type == "self_where":
                    views = _eligible_views_for_definition(rgb_views, "Observer-Centric View", difficulty)
                    unique_paths = {str(view.get("path", "") or "") for view in views if str(view.get("path", "") or "").strip()}
                    distinct_labels = sorted({str(view.get("observer_label", "") or "") for view in views if str(view.get("observer_label", "") or "").strip()})
                    eligible = len(unique_paths) >= 2 and len(distinct_labels) >= 2
                elif task_type == "self_what":
                    views = _eligible_views_for_definition(rgb_views, "Observer-Centric View", difficulty)
                    unique_paths = {str(view.get("path", "") or "") for view in views if str(view.get("path", "") or "").strip()}
                    distinct_labels = sorted({str(view.get("observer_label", "") or "") for view in views if str(view.get("observer_label", "") or "").strip()})
                    eligible = len(unique_paths) >= 4 and len(distinct_labels) >= 4
                elif task_type in {"env_where", "env_how"}:
                    views = _eligible_views_for_definition(rgb_views, "Object-Centric View", difficulty)
                    unique_paths = {str(view.get("path", "") or "") for view in views if str(view.get("path", "") or "").strip()}
                    distinct_labels = sorted({str(view.get("object_label", "") or "") for view in views if str(view.get("object_label", "") or "").strip()})
                    eligible = len(unique_paths) >= 1 and len(center := list(entry.get("center_3d", []) or [])) >= 3 and bool(entry.get("bbox_3d", {}))
                else:
                    views = _eligible_views_for_definition(rgb_views, definition, difficulty)
                    unique_paths = {str(view.get("path", "") or "") for view in views if str(view.get("path", "") or "").strip()}
                    distinct_labels = sorted({str((view.get("object_label") if definition == "Object-Centric View" else view.get("observer_label")) or "") for view in views if str((view.get("object_label") if definition == "Object-Centric View" else view.get("observer_label")) or "").strip()})
                    if task_type == "label_multiple_choice":
                        eligible = len(unique_paths) >= 2
                    else:
                        eligible = len(distinct_labels) >= 4
                if not eligible:
                    continue
                entry_id = str(entry.get("instance_id", "") or "").strip()
                if not entry_id:
                    continue
                eligible_ids.append(entry_id)
                all_task_eligible_ids.add(entry_id)
                view_counts.append(len(views))
                distinct_label_counts.append(len(distinct_labels))
                category = str(entry.get("class_name", "") or "").strip()
                if category:
                    category_set.add(category)
                    eligible_categories.add(category)
            per_task[combo_id] = {
                "view_definition": definition,
                "task_type": task_type,
                "difficulty": difficulty,
                "display_name": _task_display_name(definition, task_type),
                "eligible_landmark_count": len(eligible_ids),
                "eligible_landmark_ids": sorted(eligible_ids),
                "eligible_category_count": len(category_set),
                "avg_view_count": (sum(view_counts) / len(view_counts)) if view_counts else 0.0,
                "avg_distinct_label_count": (sum(distinct_label_counts) / len(distinct_label_counts)) if distinct_label_counts else 0.0,
            }
        for entry in entries:
            entry_id = str(entry.get("instance_id", "") or "").strip()
            if entry_id not in all_task_eligible_ids:
                continue
            obj_views = _eligible_views_for_definition(list(entry.get("rgb_views", []) or []), "Object-Centric View", difficulty)
            eligible_view_counts.append(len(obj_views))
        result["by_difficulty"][difficulty] = {
            "eligible_all_task_landmark_count": len(all_task_eligible_ids),
            "eligible_all_task_landmark_ids": sorted(all_task_eligible_ids),
            "eligible_all_task_category_count": len(eligible_categories),
            "eligible_all_task_categories": sorted(eligible_categories),
            "avg_view_count_per_eligible_landmark": (sum(eligible_view_counts) / len(eligible_view_counts)) if eligible_view_counts else 0.0,
            "per_task": per_task,
        }
    return result


def _load_report_rows_from_report_path(report_path: Path) -> list[dict[str, Any]]:
    parsed_path = report_path.parent / "parsed_predictions.jsonl"
    rows: list[dict[str, Any]] = []
    if not parsed_path.exists():
        return rows
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def recompute_report_from_run_dir(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    run_dir: Path,
) -> dict[str, Any]:
    del config
    report_path = run_dir / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    parsed_rows = _load_report_rows_from_report_path(report_path)
    summary = _summarize_predictions(parsed_rows)
    updated = {
        "run_id": str(report_payload.get("run_id", run_dir.name) or run_dir.name),
        "generated_at": report_payload.get("generated_at", _iso_now()),
        "scene_id": str(report_payload.get("scene_id", scene_id) or scene_id),
        "engine": str(report_payload.get("engine", engine) or engine),
        "model": report_payload.get("model"),
        "manifest_path": report_payload.get("manifest_path"),
        "summary": summary,
        "api_overrides": dict(report_payload.get("api_overrides", {}) or {}),
    }
    report_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = run_dir.parent / f"{scene_id}.latest_report.json"
    if latest_path.exists():
        latest_payload = json.loads(latest_path.read_text(encoding="utf-8"))
        if str(latest_payload.get("run_id", "") or "") == str(updated["run_id"]):
            latest_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"run_id": str(updated["run_id"]), "report_path": report_path, "report": updated}


def _build_metrics_matrix(stage4_root: Path, scene_id: str, *, latest_only: bool, by_difficulty: bool) -> dict[str, Any]:
    reports = _list_reports(stage4_root, scene_id)
    return _build_metrics_matrix_from_reports(reports, latest_only=latest_only, by_difficulty=by_difficulty)


def _report_generated_sort_key(report: dict[str, Any]) -> tuple[str, str]:
    generated_at = str(report.get("generated_at", "") or "")
    path = str(report.get("path", "") or "")
    return generated_at, path


def _collect_stage4_rows_by_model(reports: list[dict[str, Any]], *, latest_only: bool) -> dict[str, list[dict[str, Any]]]:
    reports_sorted = sorted(reports, key=_report_generated_sort_key)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not latest_only:
        for report in reports_sorted:
            report_path = (WORKSPACE_ROOT / str(report.get("path", "") or "")).resolve()
            model = str(report.get("model", "") or "").strip()
            if not model or not report_path.exists():
                continue
            grouped[model].extend(_load_report_rows_from_report_path(report_path))
        return grouped

    latest_by_key: dict[tuple[str, str], tuple[tuple[str, str], dict[str, Any]]] = {}
    for report in reports_sorted:
        report_path = (WORKSPACE_ROOT / str(report.get("path", "") or "")).resolve()
        model = str(report.get("model", "") or "").strip()
        if not model or not report_path.exists():
            continue
        sort_key = _report_generated_sort_key(report)
        for row in _load_report_rows_from_report_path(report_path):
            sample_id = str(row.get("sample_id", "") or "").strip()
            if not sample_id:
                continue
            key = (model, sample_id)
            latest_by_key[key] = (sort_key, row)
    for (model, _sample_id), (_sort_key, row) in latest_by_key.items():
        grouped[model].append(row)
    return grouped


def _build_metrics_matrix_from_reports(reports: list[dict[str, Any]], *, latest_only: bool, by_difficulty: bool) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for view_definition, task_type in TASK_SPECS:
        if by_difficulty:
            for difficulty in ["4way", "8way"]:
                columns.append(
                    {
                        "combo_id": _task_combo_id(view_definition, task_type, difficulty),
                        "view_definition": view_definition,
                        "task_type": task_type,
                        "difficulty": difficulty,
                        "display_name": _task_display_name(view_definition, task_type),
                        "metrics": ["option_accuracy", "bbox_acc@50iou", "bbox_mean_iou", "count"],
                    }
                )
        else:
            columns.append(
                {
                    "combo_id": f"{view_definition}|{task_type}",
                    "view_definition": view_definition,
                    "task_type": task_type,
                    "difficulty": "ALL",
                    "display_name": _task_display_name(view_definition, task_type),
                    "metrics": ["option_accuracy", "bbox_acc@50iou", "bbox_mean_iou", "count"],
                }
            )
    combo_key = "combo_by_difficulty" if by_difficulty else "combo"
    aggregated: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    rows_by_model = _collect_stage4_rows_by_model(reports, latest_only=latest_only)
    for model, model_rows in rows_by_model.items():
        summary = _summarize_predictions(model_rows) if model_rows else {}
        grouped = dict((summary.get("grouped", {}) or {}))
        aggregated[model] = dict(grouped.get(combo_key, {}) or {})
    model_rows: list[dict[str, Any]] = []
    for model in sorted(aggregated.keys()):
        combos_payload: dict[str, Any] = {}
        for column in columns:
            combo_id = str(column["combo_id"])
            summary = dict(aggregated[model].get(combo_id, {}) or {})
            combos_payload[combo_id] = {
                "count": summary.get("count", 0),
                "option_accuracy": summary.get("option_accuracy", None),
                "bbox_acc@50iou": summary.get("bbox_acc@50iou", None),
                "bbox_mean_iou": summary.get("bbox_mean_iou", None),
            }
        model_rows.append({"model": model, "combos": combos_payload})
    return {"latest_only": bool(latest_only), "columns": columns, "rows": model_rows, "by_difficulty": bool(by_difficulty)}


def _stage4_metrics_matrix_csv_rows(matrix: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = ["model", "view_definition", "task_type", "difficulty", "count", "option_accuracy", "bbox_acc@50iou", "bbox_mean_iou"]
    rows: list[dict[str, Any]] = []
    columns = list(matrix.get("columns", []) or [])
    for row in list(matrix.get("rows", []) or []):
        model = str(row.get("model", "") or "")
        combos = dict(row.get("combos", {}) or {})
        for column in columns:
            combo_id = str(column.get("combo_id", "") or "")
            payload = dict(combos.get(combo_id, {}) or {})
            rows.append(
                {
                    "model": model,
                    "view_definition": str(column.get("view_definition", "") or ""),
                    "task_type": str(column.get("task_type", "") or ""),
                    "difficulty": str(column.get("difficulty", "") or ""),
                    "count": payload.get("count", 0),
                    "option_accuracy": payload.get("option_accuracy"),
                    "bbox_acc@50iou": payload.get("bbox_acc@50iou"),
                    "bbox_mean_iou": payload.get("bbox_mean_iou"),
                }
            )
    return fieldnames, rows


def _csv_text(fieldnames: list[str], rows: list[dict[str, Any]]) -> str:
    buff = io.StringIO()
    writer = csv.DictWriter(buff, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return buff.getvalue()


def _group_report_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in list(rows or []):
        key = f"{str(row.get('view_definition', '') or '')}|{str(row.get('task_type', '') or '')}|{str(row.get('difficulty', '') or '')}"
        grouped[key].append(row)
    return dict(grouped)


def _resolve_api_settings(
    config: dict[str, Any],
    *,
    override_model: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage4_cfg = _stage4_cfg(config)
    overrides = dict(overrides or {})

    def _pick_cfg(*keys: str) -> str:
        for key in keys:
            if key in stage4_cfg:
                value = stage4_cfg.get(key)
                text = str(value or "").strip()
                if text:
                    return text
        return ""

    model = str(
        override_model
        or _pick_cfg("default_model", "model")
        or resolve_default_model(config, stage_name="stage4")
        or (list(stage4_cfg.get("models", []) or [None])[0] or "")
    ).strip()
    if not model:
        raise RuntimeError("missing api_key/model for stage4 experiment")
    model_controls = build_model_request_controls(model)
    route_model = str(model_controls["base_model"] or model).strip()

    endpoint = resolve_model_api_endpoint(
        config=config,
        model=route_model,
        stage_name="stage4",
        stage_cfg=stage4_cfg,
        explicit_source=pick_first_text(
            overrides.get("api_source"),
            _pick_cfg("api_source"),
        ),
        explicit_api_base=pick_first_text(
            overrides.get("api_base"),
            _pick_cfg("api_base", "base_url"),
        ),
        explicit_api_key=pick_first_text(
            overrides.get("api_key"),
            _pick_cfg("api_key"),
        ),
    )
    api_key = str(endpoint.get("api_key", "") or "").strip()
    api_base = pick_first_text(endpoint.get("api_base"), "https://api.siliconflow.cn/v1")
    if not api_key:
        # Stage3 experiment may pass models that are not explicitly registered in api.models.
        # In that case, fall back to default model routes to recover api_base/api_key.
        for fallback_stage in ("stage3", "stage4"):
            fallback_model = str(resolve_default_model(config, stage_name=fallback_stage) or "").strip()
            if not fallback_model:
                continue
            fallback_ep = resolve_model_api_endpoint(
                config=config,
                model=fallback_model,
                stage_name=fallback_stage,
                stage_cfg=stage4_cfg,
                explicit_source=pick_first_text(overrides.get("api_source"), _pick_cfg("api_source")),
                explicit_api_base=pick_first_text(overrides.get("api_base"), _pick_cfg("api_base", "base_url")),
                explicit_api_key=pick_first_text(overrides.get("api_key"), _pick_cfg("api_key")),
            )
            fallback_key = str(fallback_ep.get("api_key", "") or "").strip()
            fallback_base = str(fallback_ep.get("api_base", "") or "").strip()
            if fallback_key:
                api_key = fallback_key
                if fallback_base:
                    api_base = fallback_base
                break
        if not api_key:
            # Last resort: use any configured API key from registry defaults/routes.
            try:
                registry = load_api_registry(config)
                model_routes = dict(registry.get("models", {}) or {})
                for _name, route in model_routes.items():
                    if not isinstance(route, dict):
                        continue
                    key_candidate = str(route.get("api_key", "") or "").strip()
                    base_candidate = str(route.get("api_base", route.get("base_url", "")) or "").strip()
                    if key_candidate:
                        api_key = key_candidate
                        if not api_base and base_candidate:
                            api_base = base_candidate
                        break
            except Exception:
                pass
        # If the only available API base is local, allow execution with a sentinel key.
        if api_base.startswith("http://localhost") or api_base.startswith("http://127.0.0.1"):
            api_key = "EMPTY"
        # Otherwise, still no api_key means we cannot talk to any endpoint.
        if not api_key:
            raise RuntimeError(
                "missing api_key/model for stage4 experiment "
                f"(model={model!r}, route_model={route_model!r}, api_base={api_base!r})"
            )

    request_model = str(endpoint.get("request_model", route_model) or route_model)
    rewrite_from = _pick_cfg("model_rewrite_from")
    rewrite_to = _pick_cfg("model_rewrite_to")
    if rewrite_from and rewrite_to and route_model == rewrite_from:
        request_model = rewrite_to
    request_model_override = pick_first_text(overrides.get("request_model"))
    if request_model_override:
        request_model = request_model_override
    upload_resize_enabled = bool(
        overrides.get(
            "upload_resize_enabled",
            stage4_cfg.get("api_upload_resize_enabled", True),
        )
    )
    upload_max_width = int(
        overrides.get(
            "upload_max_width",
            stage4_cfg.get("api_upload_max_width", 640),
        )
        or 640
    )
    upload_max_height = int(
        overrides.get(
            "upload_max_height",
            stage4_cfg.get("api_upload_max_height", 480),
        )
        or 480
    )
    upload_jpeg_quality = int(
        overrides.get(
            "upload_jpeg_quality",
            stage4_cfg.get("api_upload_jpeg_quality", 80),
        )
        or 80
    )
    timeout_s = float(overrides.get("timeout_s", stage4_cfg.get("timeout_s", 30.0)) or 30.0)
    temperature = float(overrides.get("temperature", stage4_cfg.get("temperature", 0.2)) or 0.2)
    max_tokens = int(overrides.get("max_tokens", stage4_cfg.get("max_tokens", 500)) or 500)
    request_retry_attempts = int(overrides.get("request_retry_attempts", stage4_cfg.get("request_retry_attempts", 3)) or 3)
    request_retry_backoff_sec = float(overrides.get("request_retry_backoff_sec", stage4_cfg.get("request_retry_backoff_sec", 2.0)) or 2.0)
    request_retry_forever = bool(overrides.get("request_retry_forever", stage4_cfg.get("request_retry_forever", False)))
    requested_concurrency = max(1, int(overrides.get("concurrency", 1) or 1))
    configured_rpm_limit = int(overrides.get("rpm_limit", endpoint.get("rpm_limit", 0)) or 0)
    configured_tpm_limit = int(overrides.get("tpm_limit", endpoint.get("tpm_limit", 0)) or 0)
    reserve_ratio = float(overrides.get("rate_limit_reserve_ratio", endpoint.get("rate_limit_reserve_ratio", 0.1)) or 0.1)
    estimated_tokens_per_request = int(overrides.get("estimated_tokens_per_request", endpoint.get("estimated_tokens_per_request", 0)) or 0)
    if estimated_tokens_per_request <= 0:
        estimated_tokens_per_request = max(1200, int(max_tokens) + 1000)
    rate_limit_cfg = compute_rate_limited_concurrency(
        requested_concurrency,
        rpm_limit=configured_rpm_limit,
        tpm_limit=configured_tpm_limit,
        estimated_tokens_per_request=estimated_tokens_per_request,
        reserve_ratio=reserve_ratio,
    )
    return {
        "api_key": api_key,
        "api_base": api_base,
        "api_source": str(endpoint.get("api_source", "") or ""),
        "model": str(model_controls["display_model"]),
        "base_model": route_model,
        "model_family": str(model_controls["family"]),
        "reasoning_mode": str(model_controls["mode"]),
        "assistant_prefill": str(model_controls.get("assistant_prefill", "") or ""),
        "system_prompt_prefix": str(model_controls.get("system_prompt_prefix", "") or ""),
        "system_prompt_as_blocks": bool(model_controls.get("system_prompt_as_blocks", False)),
        "request_model": request_model,
        "timeout_s": timeout_s,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_retry_attempts": request_retry_attempts,
        "request_retry_backoff_sec": request_retry_backoff_sec,
        "request_retry_forever": request_retry_forever,
        "upload_resize_enabled": upload_resize_enabled,
        "upload_max_width": upload_max_width,
        "upload_max_height": upload_max_height,
        "upload_jpeg_quality": upload_jpeg_quality,
        "prefix": str(stage4_cfg.get("prefix", "") or "").strip(),
        "request_extra_body": dict(model_controls.get("extra_body", {}) or {}),
        "configured_rpm_limit": int(rate_limit_cfg["configured_rpm_limit"]),
        "configured_tpm_limit": int(rate_limit_cfg["configured_tpm_limit"]),
        "rpm_limit": int(rate_limit_cfg["effective_rpm_limit"]),
        "tpm_limit": int(rate_limit_cfg["effective_tpm_limit"]),
        "requested_concurrency": int(rate_limit_cfg["requested_concurrency"]),
        "concurrency": int(rate_limit_cfg["effective_concurrency"]),
        "estimated_tokens_per_request": int(rate_limit_cfg["estimated_tokens_per_request"]),
        "rate_limit_reserve_ratio": float(rate_limit_cfg["reserve_ratio"]),
        "rate_limit_concurrency_applied": bool(rate_limit_cfg["rate_limit_concurrency_applied"]),
    }


def _prepare_upload_image(path: Path, api_cfg: dict[str, Any]) -> tuple[bytes, str]:
    if Image is None:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return path.read_bytes(), mime
    with Image.open(path) as img:
        canvas = img.convert("RGB")
        if api_cfg.get("upload_resize_enabled", True):
            max_w = max(64, int(api_cfg.get("upload_max_width", 640)))
            max_h = max(64, int(api_cfg.get("upload_max_height", 480)))
            scale = min(max_w / max(1, canvas.width), max_h / max(1, canvas.height), 1.0)
            if scale < 1.0:
                canvas = canvas.resize((max(1, int(round(canvas.width * scale))), max(1, int(round(canvas.height * scale)))), Image.Resampling.LANCZOS)
        buff = io.BytesIO()
        canvas.save(buff, format="JPEG", quality=max(60, min(95, int(api_cfg.get("upload_jpeg_quality", 85)))))
        return buff.getvalue(), "image/jpeg"


def _prepare_web_image(
    path: Path,
    *,
    resize_enabled: bool,
    max_width: int,
    max_height: int,
    jpeg_quality: int,
) -> tuple[bytes, str] | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            canvas = img.convert("RGB")
            if resize_enabled:
                max_w = max(64, int(max_width or 640))
                max_h = max(64, int(max_height or 480))
                scale = min(max_w / max(1, canvas.width), max_h / max(1, canvas.height), 1.0)
                if scale < 1.0:
                    canvas = canvas.resize(
                        (
                            max(1, int(round(canvas.width * scale))),
                            max(1, int(round(canvas.height * scale))),
                        ),
                        Image.Resampling.LANCZOS,
                    )
            buff = io.BytesIO()
            canvas.save(buff, format="JPEG", quality=max(55, min(95, int(jpeg_quality or 80))))
            return buff.getvalue(), "image/jpeg"
    except Exception:
        return None


def _build_openai_messages(sample: dict[str, Any], api_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    image_paths: list[Path] = []
    if sample["task_type"] in {"label_multiple_choice", "self_where", "env_where", "env_how"}:
        image_paths = [
            (WORKSPACE_ROOT / sample["reference_image_with_bbox"]).resolve(),
            (WORKSPACE_ROOT / sample["target_image"]).resolve(),
        ]
    elif sample["task_type"] == "self_what":
        image_paths = [(WORKSPACE_ROOT / sample["reference_image_with_bbox"]).resolve()]
        image_paths.extend((WORKSPACE_ROOT / cand["image"]).resolve() for cand in list(sample.get("candidates", []) or []))
    else:
        image_paths = [(WORKSPACE_ROOT / sample["reference_image_with_bbox"]).resolve()]
        image_paths.extend((WORKSPACE_ROOT / cand["image"]).resolve() for cand in list(sample["candidates"]))

    user_prompt_text = str(sample["prompt_text"])
    blocks: list[dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
    for image_path in image_paths:
        image_bytes, mime_type = _prepare_upload_image(image_path, api_cfg)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
    system_prompt = str(sample.get("system_prompt", "") or _build_system_prompt(sample))
    system_prefix = str(api_cfg.get("system_prompt_prefix", "") or "").strip()
    if system_prefix:
        system_prompt = f"{system_prefix}\n\nThen follow the task-specific instruction below.\n\n{system_prompt}".strip()
    route_model_name = str(api_cfg.get("base_model", "") or api_cfg.get("model", "") or "")
    if should_inline_system_prompt_for_multimodal(route_model_name):
        blocks[0] = {
            "type": "text",
            "text": f"{system_prompt}\n\n{user_prompt_text}".strip(),
        }
        messages = [{"role": "user", "content": blocks}]
    else:
        system_content: Any = system_prompt
        if bool(api_cfg.get("system_prompt_as_blocks", False)):
            system_content = [{"type": "text", "text": system_prompt}]
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": blocks},
        ]
    assistant_prefill = str(api_cfg.get("assistant_prefill", "") or "").strip()
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    return messages, [_path_for_json(p) for p in image_paths]


def _is_retryable_request_failure(error_text: str) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return True
    retry_markers = [
        "timeout",
        "timed out",
        "empty_model_response",
        "connection error",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporarily unavailable",
        "server disconnected",
        "read timeout",
        "remote protocol error",
        "502",
        "503",
        "504",
    ]
    fatal_markers = [
        "unsupported_model_mode_suffix",
        "model_not_found",
        "invalid_request_error",
        "badrequesterror",
        "authentication",
        "permission",
        "context_length",
        "does not exist",
        "not support",
        "unknown field",
        "unrecognized",
    ]
    if any(marker in text for marker in fatal_markers):
        return False
    return any(marker in text for marker in retry_markers)


def _messages_preview(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for message in list(messages or []):
        role = str(message.get("role", "") or "")
        content = message.get("content")
        if isinstance(content, str):
            preview.append({"role": role, "content": content})
            continue
        blocks = []
        for block in list(content or []):
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "") or "")
            if block_type == "text":
                blocks.append({"type": "text", "text": str(block.get("text", "") or "")})
            elif block_type == "image_url":
                blocks.append({"type": "image_url", "image_url": {"url": "<base64-image>"}})
            else:
                blocks.append({"type": block_type})
        preview.append({"role": role, "content": blocks})
    return preview


def _extract_response_text(resp: Any) -> tuple[str, dict[str, Any]]:
    raw_text = ""
    meta: dict[str, Any] = {}

    def _collect_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        chunks: list[str] = []
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text", None)
                    if isinstance(text, str):
                        chunks.append(text)
                        continue
                    if isinstance(text, dict):
                        nested = text.get("value", None)
                        if isinstance(nested, str):
                            chunks.append(nested)
                    nested_text = item.get("output_text", None)
                    if isinstance(nested_text, str):
                        chunks.append(nested_text)
                    content = item.get("content", None)
                    if isinstance(content, str):
                        chunks.append(content)
                    elif isinstance(content, list):
                        chunks.append(_collect_text(content))
                    continue
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    chunks.append(text_attr)
                    continue
                content_attr = getattr(item, "content", None)
                if content_attr is not None:
                    chunks.append(_collect_text(content_attr))
            return "\n".join([str(x).strip() for x in chunks if str(x).strip()])
        if isinstance(value, dict):
            for key in ("text", "output_text", "content"):
                if key in value:
                    nested = _collect_text(value.get(key))
                    if nested:
                        chunks.append(nested)
            return "\n".join([str(x).strip() for x in chunks if str(x).strip()])
        for attr in ("output_text", "text", "content"):
            nested = getattr(value, attr, None)
            if nested is not None:
                text = _collect_text(nested)
                if text:
                    chunks.append(text)
        return "\n".join([str(x).strip() for x in chunks if str(x).strip()])

    try:
        # Responses API / instant-style payloads.
        response_output_text = _collect_text(getattr(resp, "output_text", None))
        if response_output_text:
            raw_text = response_output_text
            meta["response_api_output_text"] = True

        if not raw_text:
            output_items = getattr(resp, "output", None)
            if output_items is not None:
                output_text = _collect_text(output_items)
                if output_text:
                    raw_text = output_text
                    meta["response_api_output"] = True

        # Chat-completions payloads, including reasoning models.
        if getattr(resp, "choices", None):
            choice0 = resp.choices[0]
            msg = getattr(choice0, "message", None)
            if msg is not None and not raw_text:
                content_text = _collect_text(getattr(msg, "content", None))
                reasoning_text = _collect_text(
                    getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                )
                raw_text = content_text or reasoning_text
            if not raw_text:
                raw_text = _collect_text(getattr(choice0, "text", None))
            meta["finish_reason"] = getattr(choice0, "finish_reason", None)

        meta["id"] = getattr(resp, "id", None)
        meta["model"] = getattr(resp, "model", None)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            if isinstance(usage, dict):
                meta["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            else:
                meta["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
    except Exception:
        pass
    return str(raw_text or "").strip(), meta


def _parse_option_ids(raw_text: str) -> tuple[list[str], bool]:
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw_text[start : end + 1])
            if isinstance(payload, dict):
                ids = [str(x).strip().upper() for x in list(payload.get("answer_option_ids", []) or []) if str(x).strip().upper() in {"A", "B", "C", "D"}]
                if ids:
                    seen = []
                    for item in ids:
                        if item not in seen:
                            seen.append(item)
                    return seen, True
                text = str(payload.get("answer_option_id", "") or "").strip().upper()
                if text in {"A", "B", "C", "D"}:
                    return [text], True
    except Exception:
        pass
    match = re.search(r"Option\s*:\s*([A-D])\b", raw_text, flags=re.IGNORECASE)
    if match:
        return [match.group(1).upper()], True
    matches = re.findall(r"\b([A-D])\b", raw_text)
    if matches:
        out = []
        for item in matches:
            up = str(item).upper()
            if up not in out:
                out.append(up)
        return out, True
    return [], False


def _parse_bbox(raw_text: str) -> tuple[list[float] | None, bool]:
    try:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw_text[start : end + 1])
            bbox = list(payload.get("bbox_xyxy_norm", []) or [])
            if len(bbox) == 4:
                coords = [max(0.0, min(1.0, float(v))) for v in bbox]
                x0, y0, x1, y1 = coords
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                return [x0, y0, x1, y1], True
    except Exception:
        pass
    match = re.search(
        r"BBox\s*:\s*[\[\(]?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        raw_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, False
    try:
        coords = [max(0.0, min(1.0, float(match.group(i)))) for i in range(1, 5)]
    except Exception:
        return None, False
    x0, y0, x1, y1 = coords
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1], True


def _bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0.0 else 0.0


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_ok = sum(1 for row in rows if bool(row.get("parse_ok", False)))
    option_correct = sum(1 for row in rows if bool(row.get("option_correct", False)))
    bbox_rows = [row for row in rows if row.get("gold_bbox_xyxy_norm") is not None]
    bbox_acc = sum(1 for row in bbox_rows if bool(row.get("bbox_acc@50iou", False)))
    bbox_ious = [float(row.get("bbox_iou", 0.0) or 0.0) for row in bbox_rows]
    latency_values = [float(row.get("latency_ms", 0.0) or 0.0) for row in rows if row.get("latency_ms") is not None]

    def _rate(n: int, d: int) -> float:
        return float(n) / float(d) if d else 0.0

    grouped: dict[str, Any] = {}
    for key in ["view_definition", "task_type", "difficulty"]:
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bucket[str(row.get(key, ""))].append(row)
        grouped[key] = {
            name: {
                "count": len(items),
                "option_accuracy": _rate(sum(1 for item in items if item.get("option_correct")), len(items)),
                "bbox_acc@50iou": _rate(sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None and item.get("bbox_acc@50iou")), sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None)),
                "bbox_mean_iou": (
                    sum(float(item.get("bbox_iou", 0.0) or 0.0) for item in items if item.get("gold_bbox_xyxy_norm") is not None)
                    / sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None)
                ) if any(item.get("gold_bbox_xyxy_norm") is not None for item in items) else 0.0,
            }
            for name, items in bucket.items()
        }
    combo_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combo_diff_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        view_definition = str(row.get("view_definition", "") or "")
        task_type = str(row.get("task_type", "") or "")
        difficulty = str(row.get("difficulty", "") or "")
        combo_bucket[f"{view_definition}|{task_type}"].append(row)
        combo_diff_bucket[_task_combo_id(view_definition, task_type, difficulty)].append(row)
    grouped["combo"] = {
        name: {
            "count": len(items),
            "option_accuracy": _rate(sum(1 for item in items if item.get("option_correct")), len(items)),
            "bbox_acc@50iou": _rate(sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None and item.get("bbox_acc@50iou")), sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None)),
            "bbox_mean_iou": (
                sum(float(item.get("bbox_iou", 0.0) or 0.0) for item in items if item.get("gold_bbox_xyxy_norm") is not None)
                / sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None)
            ) if any(item.get("gold_bbox_xyxy_norm") is not None for item in items) else 0.0,
        }
        for name, items in combo_bucket.items()
    }
    grouped["combo_by_difficulty"] = {
        name: {
            "count": len(items),
            "option_accuracy": _rate(sum(1 for item in items if item.get("option_correct")), len(items)),
            "bbox_acc@50iou": _rate(sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None and item.get("bbox_acc@50iou")), sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None)),
            "bbox_mean_iou": (
                sum(float(item.get("bbox_iou", 0.0) or 0.0) for item in items if item.get("gold_bbox_xyxy_norm") is not None)
                / sum(1 for item in items if item.get("gold_bbox_xyxy_norm") is not None)
            ) if any(item.get("gold_bbox_xyxy_norm") is not None for item in items) else 0.0,
        }
        for name, items in combo_diff_bucket.items()
    }
    return {
        "count": total,
        "parse_success_rate": _rate(parse_ok, total),
        "option_accuracy": _rate(option_correct, total),
        "bbox_acc@50iou": _rate(bbox_acc, len(bbox_rows)),
        "bbox_mean_iou": (sum(bbox_ious) / len(bbox_ious)) if bbox_ious else 0.0,
        "avg_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else None,
        "grouped": grouped,
    }


class CancelledExperimentError(RuntimeError):
    pass


class ApiRateLimiter:
    def __init__(self, *, rpm_limit: int = 0, tpm_limit: int = 0) -> None:
        self.rpm_limit = max(0, int(rpm_limit))
        self.tpm_limit = max(0, int(tpm_limit))
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._token_records: deque[tuple[float, int]] = deque()

    def _cleanup(self, now: float) -> None:
        while self._request_times and now - self._request_times[0] >= 60.0:
            self._request_times.popleft()
        while self._token_records and now - self._token_records[0][0] >= 60.0:
            self._token_records.popleft()

    def acquire(self, *, estimated_tokens: int, cancel_event: threading.Event | None = None) -> None:
        while True:
            now = time.monotonic()
            with self._lock:
                self._cleanup(now)
                wait_sec = 0.0
                if self.rpm_limit > 0 and len(self._request_times) >= self.rpm_limit:
                    wait_sec = max(wait_sec, 60.0 - (now - self._request_times[0]) + 0.01)
                if self.tpm_limit > 0:
                    current_tokens = sum(tokens for _, tokens in self._token_records)
                    if current_tokens + estimated_tokens > self.tpm_limit and self._token_records:
                        wait_sec = max(wait_sec, 60.0 - (now - self._token_records[0][0]) + 0.01)
                if wait_sec <= 0.0:
                    self._request_times.append(now)
                    if self.tpm_limit > 0:
                        self._token_records.append((now, max(1, int(estimated_tokens))))
                    return
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledExperimentError("experiment_cancelled")
            time.sleep(min(wait_sec, 0.5) if wait_sec > 0.0 else 0.05)


def _estimate_request_tokens(sample: dict[str, Any], api_cfg: dict[str, Any]) -> int:
    image_count = 2 if sample.get("task_type") in {"label_multiple_choice", "self_where", "env_where", "env_how"} else 5
    prompt_tokens = max(64, int(len(str(sample.get("prompt_text", ""))) / 4))
    image_tokens = int(image_count * 400)
    return prompt_tokens + image_tokens + int(api_cfg.get("max_tokens", 500))


def _run_single_sample_request(sample: dict[str, Any], *, api_cfg: dict[str, Any], cancel_event: threading.Event | None, limiter: ApiRateLimiter) -> tuple[dict[str, Any], dict[str, Any]]:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledExperimentError("experiment_cancelled")
    messages, upload_images = _build_openai_messages(sample, api_cfg)
    system_prompt = str(sample.get("system_prompt", "") or _build_system_prompt(sample))
    user_prompt = str(sample.get("user_prompt", "") or sample.get("prompt_text", "") or "")
    request_row = {
        "sample_id": sample["sample_id"],
        "model": api_cfg["model"],
        "request_model": api_cfg["request_model"],
        "api_source": api_cfg.get("api_source"),
        "api_base": api_cfg.get("api_base"),
        "reasoning_mode": api_cfg.get("reasoning_mode"),
        "task_type": sample["task_type"],
        "view_definition": sample["view_definition"],
        "assistant_prefill": api_cfg.get("assistant_prefill", ""),
        "system_prompt_prefix": api_cfg.get("system_prompt_prefix", ""),
        "system_prompt_as_blocks": bool(api_cfg.get("system_prompt_as_blocks", False)),
        "request_extra_body": dict(api_cfg.get("request_extra_body", {}) or {}),
        "messages_preview": _messages_preview(messages),
        "images": upload_images,
        "expected_option_id": sample["answer_option_id"],
        "expected_bbox_xyxy_norm": sample["answer_bbox_xyxy_norm"],
    }
    limiter.acquire(estimated_tokens=_estimate_request_tokens(sample, api_cfg), cancel_event=cancel_event)
    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["api_base"])
    raw_text = ""
    raw_meta: dict[str, Any] = {}
    request_status = "ok"
    retry_attempts = max(1, int(api_cfg.get("request_retry_attempts", 1) or 1))
    retry_backoff_sec = max(0.0, float(api_cfg.get("request_retry_backoff_sec", 0.0) or 0.0))
    retry_forever = bool(api_cfg.get("request_retry_forever", False))
    attempt_errors: list[str] = []
    latency_ms = 0.0
    attempt_idx = 0
    while True:
        started = time.perf_counter()
        should_retry = False
        try:
            kwargs: dict[str, Any] = {
                "model": api_cfg["request_model"],
                "messages": messages,
                "temperature": float(api_cfg["temperature"]),
                "timeout": float(api_cfg["timeout_s"]),
            }
            extra_body: dict[str, Any] = dict(api_cfg.get("request_extra_body", {}) or {})
            if api_cfg.get("prefix"):
                extra_body["prefix"] = api_cfg["prefix"]
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            latency_ms += (time.perf_counter() - started) * 1000.0
            raw_text, raw_meta = _extract_response_text(resp)
            if str(raw_text or "").strip():
                break
            request_status = "error"
            error_text = "empty_model_response"
            attempt_errors.append(error_text)
            raw_meta = {**dict(raw_meta or {}), "error": error_text}
            should_retry = _is_retryable_request_failure(error_text)
        except Exception as exc:
            latency_ms += (time.perf_counter() - started) * 1000.0
            request_status = "error"
            raw_text = str(exc)
            raw_meta = {"error": str(exc)}
            attempt_errors.append(str(exc))
            should_retry = _is_retryable_request_failure(str(exc))
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledExperimentError("experiment_cancelled")
        attempt_idx += 1
        if not should_retry:
            raise RuntimeError(str((raw_meta or {}).get("error", raw_text) or "non_retryable_request_error"))
        if retry_forever or attempt_idx < retry_attempts:
            time.sleep(retry_backoff_sec)
            continue
        break
    raw_meta = dict(raw_meta or {})
    raw_meta["request_retry_attempts"] = int(attempt_idx)
    raw_meta["request_retry_errors"] = list(attempt_errors)

    response_row = {
        "sample_id": sample["sample_id"],
        "request_status": request_status,
        "latency_ms": latency_ms,
        "raw_text": raw_text,
        "raw_response": raw_meta,
    }
    pred_option_ids, option_parse_ok = _parse_option_ids(raw_text)
    pred_option_id = pred_option_ids[0] if pred_option_ids else None
    expects_bbox = sample.get("answer_bbox_xyxy_norm") is not None
    if expects_bbox:
        pred_bbox, bbox_parse_ok = _parse_bbox(raw_text)
        bbox_iou = _bbox_iou(pred_bbox, list(sample["answer_bbox_xyxy_norm"]))
    else:
        pred_bbox, bbox_parse_ok = None, True
        bbox_iou = 0.0
    gold_option_ids = [str(x).strip() for x in list(sample.get("answer_option_ids", []) or []) if str(x).strip()]
    if not gold_option_ids:
        gold_option_ids = [str(sample["answer_option_id"])]
    option_exact = set(pred_option_ids) == set(gold_option_ids)
    parsed_row = {
        "sample_id": sample["sample_id"],
        "scene_id": sample["scene_id"],
        "engine": sample["engine"],
        "view_definition": sample["view_definition"],
        "task_type": sample["task_type"],
        "difficulty": sample["difficulty"],
        "gold_option_ids": gold_option_ids,
        "gold_option_id": sample["answer_option_id"],
        "gold_bbox_xyxy_norm": sample["answer_bbox_xyxy_norm"],
        "pred_option_ids": pred_option_ids,
        "pred_option_id": pred_option_id,
        "pred_bbox_xyxy_norm": pred_bbox,
        "parse_ok": bool(option_parse_ok and bbox_parse_ok),
        "option_correct": bool(option_exact),
        "bbox_iou": bbox_iou,
        "bbox_acc@50iou": bool(expects_bbox and bbox_iou >= 0.5),
        "latency_ms": latency_ms,
    }
    return request_row, {"response": response_row, "parsed": parsed_row}


def run_experiment_once(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    manifest_path: Path,
    model: str | None = None,
    samples: list[dict[str, Any]] | None = None,
    api_overrides: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai package is required for stage4 experiment")
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    stage4_root = _ensure_dir(_resolve_stage4_root(config, scene_root=scene_root))
    experiments_root = _ensure_dir(stage4_root / "experiments")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_samples = list(samples if samples is not None else (manifest.get("samples", []) or []))
    api_cfg = _resolve_api_settings(config, override_model=model, overrides=api_overrides)
    manifest_tag = _safe_name(Path(manifest_path).stem)[:48]
    model_tag = _safe_name(str(api_cfg["model"]))
    unique_experiment = bool((api_overrides or {}).get("unique_experiment", False))
    run_id = f"{scene_id}_{model_tag}_{manifest_tag}_unique" if unique_experiment else f"{scene_id}_{model_tag}_{_now_ts()}"
    run_dir = _ensure_dir(experiments_root / run_id)
    requests_path = run_dir / "requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    parsed_path = run_dir / "parsed_predictions.jsonl"
    requests_txt_path = run_dir / "requests.txt"
    responses_txt_path = run_dir / "responses.txt"
    parsed_rows: list[dict[str, Any]] = []
    completed_sample_ids: set[str] = set()
    if unique_experiment and responses_path.exists():
        latest_status_by_sample: dict[str, str] = {}
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "") or "").strip()
            if sample_id:
                latest_status_by_sample[sample_id] = str(row.get("request_status", "") or "").strip().lower()
        for sample_id, status in latest_status_by_sample.items():
            if status == "ok":
                completed_sample_ids.add(sample_id)
    selected_samples = [dict(sample, _sample_index=int(idx), _sample_ordinal=int(idx) + 1) for idx, sample in enumerate(selected_samples)]
    if unique_experiment:
        selected_samples = [sample for sample in selected_samples if str(sample.get("sample_id", "") or "").strip() not in completed_sample_ids]
    dynamic_estimated_tokens = max(
        [_estimate_request_tokens(sample, api_cfg) for sample in selected_samples] or [max(1200, int(api_cfg.get("max_tokens", 500) or 500) + 1000)]
    )
    rate_limit_cfg = compute_rate_limited_concurrency(
        int(api_cfg.get("requested_concurrency", api_cfg.get("concurrency", 1)) or 1),
        rpm_limit=int(api_cfg.get("configured_rpm_limit", api_cfg.get("rpm_limit", 0)) or 0),
        tpm_limit=int(api_cfg.get("configured_tpm_limit", api_cfg.get("tpm_limit", 0)) or 0),
        estimated_tokens_per_request=int(dynamic_estimated_tokens),
        reserve_ratio=float(api_cfg.get("rate_limit_reserve_ratio", 0.1) or 0.1),
    )
    api_cfg = dict(api_cfg)
    api_cfg["configured_rpm_limit"] = int(rate_limit_cfg["configured_rpm_limit"])
    api_cfg["configured_tpm_limit"] = int(rate_limit_cfg["configured_tpm_limit"])
    api_cfg["rpm_limit"] = int(rate_limit_cfg["effective_rpm_limit"])
    api_cfg["tpm_limit"] = int(rate_limit_cfg["effective_tpm_limit"])
    api_cfg["requested_concurrency"] = int(rate_limit_cfg["requested_concurrency"])
    api_cfg["concurrency"] = int(rate_limit_cfg["effective_concurrency"])
    api_cfg["estimated_tokens_per_request"] = int(rate_limit_cfg["estimated_tokens_per_request"])
    api_cfg["rate_limit_reserve_ratio"] = float(rate_limit_cfg["reserve_ratio"])
    api_cfg["rate_limit_concurrency_applied"] = bool(rate_limit_cfg["rate_limit_concurrency_applied"])
    limiter = ApiRateLimiter(rpm_limit=api_cfg.get("rpm_limit", 0), tpm_limit=api_cfg.get("tpm_limit", 0))

    file_mode = "a" if unique_experiment else "w"
    with requests_path.open(file_mode, encoding="utf-8") as req_fp, responses_path.open(file_mode, encoding="utf-8") as resp_fp, parsed_path.open(file_mode, encoding="utf-8") as pred_fp, requests_txt_path.open(file_mode, encoding="utf-8") as req_txt_fp, responses_txt_path.open(file_mode, encoding="utf-8") as resp_txt_fp:
        total = len(samples if samples is not None else (manifest.get("samples", []) or []))
        done = len(completed_sample_ids)
        write_lock = threading.Lock()
        failed_request_rows: list[dict[str, Any]] = []

        def _submit(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            return _run_single_sample_request(sample, api_cfg=api_cfg, cancel_event=cancel_event, limiter=limiter)

        with ThreadPoolExecutor(max_workers=max(1, int(api_cfg.get("concurrency", 1)))) as executor:
            future_to_sample = {executor.submit(_submit, sample): sample for sample in selected_samples}
            try:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    worker_id = int((done % max(1, int(api_cfg.get("concurrency", 1)))) + 1)
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledExperimentError("experiment_cancelled")
                    request_row: dict[str, Any]
                    payload: dict[str, Any]
                    try:
                        request_row, payload = future.result()
                    except CancelledExperimentError:
                        raise
                    except Exception as exc:
                        fallback_messages_preview: list[dict[str, Any]] = []
                        fallback_upload_images: list[str] = []
                        try:
                            fallback_messages, fallback_upload_images = _build_openai_messages(sample, api_cfg)
                            fallback_messages_preview = _messages_preview(fallback_messages)
                        except Exception:
                            fallback_messages_preview = []
                            fallback_upload_images = []
                        request_row = {
                            "sample_id": sample["sample_id"],
                            "model": api_cfg["model"],
                            "request_model": api_cfg["request_model"],
                            "api_source": api_cfg.get("api_source"),
                            "api_base": api_cfg.get("api_base"),
                            "reasoning_mode": api_cfg.get("reasoning_mode"),
                            "task_type": sample["task_type"],
                            "view_definition": sample["view_definition"],
                            "assistant_prefill": api_cfg.get("assistant_prefill", ""),
                            "system_prompt_prefix": api_cfg.get("system_prompt_prefix", ""),
                            "system_prompt_as_blocks": bool(api_cfg.get("system_prompt_as_blocks", False)),
                            "request_extra_body": dict(api_cfg.get("request_extra_body", {}) or {}),
                            "messages_preview": fallback_messages_preview,
                            "images": fallback_upload_images,
                            "expected_option_id": sample["answer_option_id"],
                            "expected_bbox_xyxy_norm": sample["answer_bbox_xyxy_norm"],
                        }
                        payload = {
                            "response": {
                                "sample_id": sample["sample_id"],
                                "request_status": "error",
                                "latency_ms": None,
                                "raw_text": str(exc),
                                "raw_response": {"error": str(exc)},
                            },
                            "parsed": {
                                "sample_id": sample["sample_id"],
                                "scene_id": sample["scene_id"],
                                "engine": sample["engine"],
                                "view_definition": sample["view_definition"],
                                "task_type": sample["task_type"],
                                "difficulty": sample["difficulty"],
                                "gold_option_ids": [str(x).strip() for x in list(sample.get("answer_option_ids", []) or []) if str(x).strip()],
                                "gold_option_id": sample["answer_option_id"],
                                "gold_bbox_xyxy_norm": sample["answer_bbox_xyxy_norm"],
                                "pred_option_id": None,
                                "pred_option_ids": [],
                                "pred_bbox_xyxy_norm": None,
                                "parse_ok": False,
                                "option_correct": False,
                                "bbox_iou": 0.0,
                                "bbox_acc@50iou": False,
                                "latency_ms": None,
                            },
                        }
                    done += 1
                    response_row = {
                        "run_id": run_id,
                        **payload["response"],
                    }
                    parsed_row = {
                        "run_id": run_id,
                        **payload["parsed"],
                    }
                    request_out = {
                        "run_id": run_id,
                        **request_row,
                    }
                    with write_lock:
                        req_fp.write(json.dumps(request_out, ensure_ascii=False) + "\n")
                        req_fp.flush()
                        resp_fp.write(json.dumps(response_row, ensure_ascii=False) + "\n")
                        resp_fp.flush()
                        pred_fp.write(json.dumps(parsed_row, ensure_ascii=False) + "\n")
                        pred_fp.flush()
                        usage = dict((response_row.get("raw_response", {}) or {}).get("usage", {}) or {}) if isinstance((response_row.get("raw_response", {}) or {}).get("usage", {}), dict) else {}
                        req_txt_fp.write(
                            f"[{_iso_now()}] sample_id={sample['sample_id']} model={api_cfg['model']} task={sample['task_type']} definition={sample['view_definition']}\n"
                            f"request_model={api_cfg['request_model']} api_source={api_cfg.get('api_source','')} reasoning_mode={api_cfg.get('reasoning_mode','')}\n"
                            f"system_prompt_prefix:\n{request_row.get('system_prompt_prefix','')}\n"
                            f"system_prompt_as_blocks={json.dumps(request_row.get('system_prompt_as_blocks', False), ensure_ascii=False)}\n"
                            f"assistant_prefill:\n{request_row.get('assistant_prefill','')}\n"
                            f"request_extra_body={json.dumps(request_row.get('request_extra_body', {}), ensure_ascii=False)}\n"
                            f"messages_preview={json.dumps(request_row.get('messages_preview', []), ensure_ascii=False)}\n"
                            f"images={json.dumps(request_row.get('images', []), ensure_ascii=False)}\n\n"
                        )
                        resp_txt_fp.write(
                            f"[{_iso_now()}] sample_id={sample['sample_id']} status={response_row['request_status']} latency_ms={response_row['latency_ms']} "
                            f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} total_tokens={usage.get('total_tokens')}\n"
                            f"raw_text:\n{response_row['raw_text']}\n"
                            f"raw_response={json.dumps(response_row.get('raw_response', {}), ensure_ascii=False)}\n\n"
                        )
                        req_txt_fp.flush()
                        resp_txt_fp.flush()
                    parsed_rows.append(parsed_row)
                    if str(response_row.get("request_status", "") or "") == "error":
                        failed_request_rows.append(
                            {
                                "sample_index": int(sample.get("_sample_index", 0) or 0),
                                "sample_ordinal": int(sample.get("_sample_ordinal", 1) or 1),
                                "sample_id": str(sample.get("sample_id", "") or ""),
                                "task_type": str(sample.get("task_type", "") or ""),
                                "difficulty": str(sample.get("difficulty", "") or ""),
                                "request_retry_attempts": int(((response_row.get("raw_response", {}) or {}).get("request_retry_attempts", api_cfg.get("request_retry_attempts", 0)) or 0)),
                                "request_retry_errors": list(((response_row.get("raw_response", {}) or {}).get("request_retry_errors", []) or [])),
                            }
                        )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "run_id": run_id,
                                "model": api_cfg["model"],
                                "completed": done,
                                "total": total,
                                "worker_id": worker_id,
                                "sample_id": sample["sample_id"],
                                "task_type": sample["task_type"],
                                "request_status": response_row["request_status"],
                                "parse_ok": bool(parsed_row.get("parse_ok", False)),
                                "latency_ms": response_row.get("latency_ms"),
                            }
                        )
            finally:
                if cancel_event is not None and cancel_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)

    summary = _summarize_predictions(parsed_rows)
    report = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "scene_id": scene_id,
        "engine": engine,
        "model": api_cfg["model"],
        "manifest_path": _path_for_json(manifest_path),
        "summary": summary,
        "api_overrides": dict(api_overrides or {}),
    }
    sample_meta_by_id = {
        str(sample.get("sample_id", "") or "").strip(): {
            "sample_index": int(sample.get("_sample_index", 0) or 0),
            "sample_ordinal": int(sample.get("_sample_ordinal", 1) or 1),
            "sample_id": str(sample.get("sample_id", "") or ""),
            "task_type": str(sample.get("task_type", "") or ""),
            "difficulty": str(sample.get("difficulty", "") or ""),
        }
        for sample in selected_samples
    }
    latest_response_by_sample: dict[str, dict[str, Any]] = {}
    for line in responses_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "") or "").strip()
        if sample_id:
            latest_response_by_sample[sample_id] = row
    failed_request_rows_final: list[dict[str, Any]] = []
    for sample_id, response_row in latest_response_by_sample.items():
        if str(response_row.get("request_status", "") or "") != "error":
            continue
        meta = dict(sample_meta_by_id.get(sample_id, {}))
        if not meta:
            meta = {
                "sample_index": 0,
                "sample_ordinal": 1,
                "sample_id": sample_id,
                "task_type": "",
                "difficulty": "",
            }
        raw_response = dict(response_row.get("raw_response", {}) or {}) if isinstance(response_row.get("raw_response", {}), dict) else {}
        meta["request_retry_attempts"] = int(raw_response.get("request_retry_attempts", api_cfg.get("request_retry_attempts", 0)) or 0)
        meta["request_retry_errors"] = list(raw_response.get("request_retry_errors", []) or [])
        failed_request_rows_final.append(meta)
    failed_request_rows_final.sort(key=lambda row: int(row.get("sample_index", 0) or 0))
    failed_indices_path = run_dir / "failed_request_indices.json"
    failed_indices_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": "stage4",
                "scene_id": scene_id,
                "engine": engine,
                "model": api_cfg["model"],
                "manifest_path": _path_for_json(manifest_path),
                "request_retry_attempts": int(api_cfg.get("request_retry_attempts", 0) or 0),
                "failed_count": len(failed_request_rows_final),
                "failed_sample_indices": [int(row.get("sample_index", 0) or 0) for row in failed_request_rows_final],
                "failed_samples": failed_request_rows_final,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = experiments_root / f"{scene_id}.latest_report.json"
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "report_path": report_path,
        "latest_report_path": latest_path,
        "report": report,
    }


def run_experiment(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    manifest_path: Path,
    override_model: str | None = None,
    limit: int | None = None,
    api_overrides: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    samples = _filter_samples(list(manifest.get("samples", []) or []), limit=limit)
    return run_experiment_once(
        config=config,
        scene_id=scene_id,
        engine=engine,
        manifest_path=Path(manifest_path),
        model=override_model,
        samples=samples,
        api_overrides=api_overrides,
        progress_callback=progress_callback,
    )


def _find_latest_json(path: Path, pattern: str) -> Path | None:
    candidates = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_latest_manifest(stage4_root: Path, scene_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    manifests_root = stage4_root / "manifests"
    latest = manifests_root / f"{scene_id}.latest_manifest.json"
    if latest.exists():
        return latest, json.loads(latest.read_text(encoding="utf-8"))
    fallback = _find_latest_json(manifests_root, f"{scene_id}_qa_manifest_*.json")
    if fallback is None:
        return None, None
    return fallback, json.loads(fallback.read_text(encoding="utf-8"))


def _load_latest_report(stage4_root: Path, scene_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    experiments_root = stage4_root / "experiments"
    latest = experiments_root / f"{scene_id}.latest_report.json"
    if latest.exists():
        return latest, json.loads(latest.read_text(encoding="utf-8"))
    fallback = _find_latest_json(experiments_root, "*/report.json")
    if fallback is None:
        return None, None
    return fallback, json.loads(fallback.read_text(encoding="utf-8"))


def _list_manifests(stage4_root: Path, scene_id: str) -> list[dict[str, Any]]:
    manifests_root = stage4_root / "manifests"
    rows: list[dict[str, Any]] = []
    for path in sorted(manifests_root.glob(f"{scene_id}_qa_manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "path": _path_for_json(path),
                "generation_id": payload.get("generation_id", path.stem),
                "generated_at": payload.get("generated_at"),
                "sample_count": payload.get("sample_count", 0),
                "reference_main_only": payload.get("reference_main_only", None),
                "difficulties": payload.get("difficulties", []),
                "view_definitions": payload.get("view_definitions", []),
                "task_types": payload.get("task_types", []),
                "summary": _summarize_manifest_payload(payload),
            }
        )
    return rows


def _list_reports(stage4_root: Path, scene_id: str) -> list[dict[str, Any]]:
    experiments_root = stage4_root / "experiments"
    rows: list[dict[str, Any]] = []
    for path in sorted(experiments_root.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("scene_id", "")) != str(scene_id):
            continue
        summary = dict(payload.get("summary", {}) or {})
        grouped = summary.get("grouped")
        if isinstance(grouped, dict) and grouped.get("combo"):
            pass
        else:
            report_rows = _load_report_rows_from_report_path(path)
            if report_rows:
                summary = _summarize_predictions(report_rows)
        rows.append(
            {
                "path": _path_for_json(path),
                "run_id": payload.get("run_id", path.parent.name),
                "generated_at": payload.get("generated_at"),
                "model": payload.get("model"),
                "manifest_path": payload.get("manifest_path"),
                "manifest_name": Path(str(payload.get("manifest_path", "") or "")).name or "-",
                "count": summary.get("count", 0),
                "parse_success_rate": summary.get("parse_success_rate"),
                "option_accuracy": summary.get("option_accuracy"),
                "bbox_acc@50iou": summary.get("bbox_acc@50iou"),
                "avg_latency_ms": summary.get("avg_latency_ms"),
                "summary": summary,
            }
        )
    return rows


def _resolve_stage4_task_pipeline_root(config: dict[str, Any], scene_id: str, engine: str, task_name: str) -> Path | None:
    task_name_value = str(task_name or "").strip()
    if not task_name_value:
        return None
    cfg = dict(config)
    cfg["task_pipeline"] = {"task_name": task_name_value, "root_dir": "task_pipeline_data"}
    source_scene_root = resolve_scene_root(cfg, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    root = _resolve_stage4_root(cfg, scene_root=source_scene_root)
    return root if root.exists() else None


def _stage4_roots(config: dict[str, Any], scene_id: str, engine: str, task_name: str | None, source_scene_root: Path) -> list[Path]:
    roots: list[Path] = []
    pipeline_root = _resolve_stage4_task_pipeline_root(config, scene_id, engine, str(task_name or ""))
    if pipeline_root is not None:
        roots.append(pipeline_root)
    roots.append(_resolve_stage4_root(config, scene_root=source_scene_root))
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _load_latest_manifest_multi(stage4_roots: list[Path], scene_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    for root in stage4_roots:
        path, payload = _load_latest_manifest(root, scene_id)
        if path is not None and payload is not None:
            return path, payload
    return None, None


def _load_latest_report_multi(stage4_roots: list[Path], scene_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    for root in stage4_roots:
        path, payload = _load_latest_report(root, scene_id)
        if path is not None and payload is not None:
            return path, payload
    return None, None


def _list_manifests_multi(stage4_roots: list[Path], scene_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in stage4_roots:
        for row in _list_manifests(root, scene_id):
            key = str(row.get("path", "") or "")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    return rows


def _list_reports_multi(stage4_roots: list[Path], scene_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in stage4_roots:
        for row in _list_reports(root, scene_id):
            key = str(row.get("path", "") or "")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    return rows


def _count_completed_prediction_rows(parsed_path: Path) -> int:
    responses_path = parsed_path.parent / "responses.jsonl"
    if responses_path.exists():
        latest_status_by_sample: dict[str, str] = {}
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "") or "").strip()
            if sample_id:
                latest_status_by_sample[sample_id] = str(row.get("request_status", "") or "").strip().lower()
        return sum(1 for status in latest_status_by_sample.values() if status == "ok")
    if not parsed_path.exists():
        return 0
    sample_ids: set[str] = set()
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id", "") or "").strip()
        if sample_id:
            sample_ids.add(sample_id)
    return len(sample_ids)


def _stage4_best_run_progress(stage4_roots: list[Path], scene_id: str, model: str, manifest_path: str) -> dict[str, Any]:
    model_tag = _safe_name(str(model or ""))
    manifest_tag = _safe_name(Path(str(manifest_path or "")).stem)[:48] if str(manifest_path or "").strip() else ""
    prefix = f"{scene_id}_{model_tag}_{manifest_tag}" if manifest_tag else f"{scene_id}_{model_tag}_"
    best_completed = 0
    best_run_dir = ""
    best_mtime = -1.0
    for root in stage4_roots:
        experiments_root = root / "experiments"
        if not experiments_root.exists():
            continue
        for run_dir in experiments_root.iterdir():
            if not run_dir.is_dir():
                continue
            if not str(run_dir.name).startswith(prefix):
                continue
            completed = _count_completed_prediction_rows(run_dir / "parsed_predictions.jsonl")
            try:
                mtime = float(run_dir.stat().st_mtime)
            except Exception:
                mtime = 0.0
            if completed > best_completed or (completed == best_completed and mtime > best_mtime):
                best_completed = completed
                best_run_dir = _path_for_json(run_dir)
                best_mtime = mtime
    return {"completed": int(best_completed), "run_dir": best_run_dir}


def _discover_stage4_models_from_runs(stage4_roots: list[Path], scene_id: str) -> set[str]:
    models: set[str] = set()
    for root in stage4_roots:
        experiments_root = root / "experiments"
        if not experiments_root.exists():
            continue
        for run_dir in experiments_root.iterdir():
            if not run_dir.is_dir():
                continue
            if not str(run_dir.name).startswith(f"{scene_id}_"):
                continue
            report_path = run_dir / "report.json"
            if report_path.exists():
                try:
                    payload = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    model = str(payload.get("model", "") or "").strip()
                    if model:
                        models.add(model)
                        continue
            requests_path = run_dir / "requests.jsonl"
            if requests_path.exists():
                for line in requests_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(row, dict):
                        continue
                    model = str(row.get("model", "") or "").strip()
                    if model:
                        models.add(model)
                        break
    return models


def _build_stage4_experiment_progress_matrix(*, catalog: list[dict[str, Any]], selected_engine: str, selected_scene_id: str, task_name: str | None, fallback_config_path: Path) -> dict[str, Any]:
    scene_items = _catalog_items_for_engine(catalog, selected_engine)
    if not _is_global_scene_id(selected_scene_id):
        scene_items = [item for item in scene_items if str(item.get("scene_id", "") or "").strip() == str(selected_scene_id or "").strip()]
    contexts: list[dict[str, Any]] = []
    known_models: set[str] = set()
    for item in scene_items:
        scene_value = str(item.get("scene_id", "") or "").strip()
        cfg, _ = _load_scene_config_from_catalog(
            engine=selected_engine,
            scene_id=scene_value,
            catalog=catalog,
            fallback_config_path=fallback_config_path,
        )
        scene_root = resolve_scene_root(cfg, scene_id=scene_value, engine=selected_engine, workspace_root=WORKSPACE_ROOT)
        if str(task_name or "").strip() and _resolve_stage4_task_pipeline_root(cfg, scene_value, selected_engine, str(task_name or "")) is None:
            continue
        roots = _stage4_roots(cfg, scene_value, selected_engine, str(task_name or ""), scene_root)
        latest_manifest_path, latest_manifest = _load_latest_manifest_multi(roots, scene_value)
        manifest_path = str(_path_for_json(latest_manifest_path) if latest_manifest_path else "")
        total_samples = int((latest_manifest or {}).get("sample_count", len((latest_manifest or {}).get("samples", []) or [])) or 0) if latest_manifest else 0
        reports = _list_reports_multi(roots, scene_value)
        known_models.update(str(row.get("model", "") or "").strip() for row in reports if str(row.get("model", "") or "").strip())
        known_models.update(_discover_stage4_models_from_runs(roots, scene_value))
        if not manifest_path and not reports and not any((root / 'experiments').exists() for root in roots):
            continue
        contexts.append({
            "scene_id": scene_value,
            "roots": roots,
            "manifest_path": manifest_path,
            "total_samples": total_samples,
        })
    scene_columns = [{"scene_id": ctx["scene_id"], "total_samples": int(ctx["total_samples"])} for ctx in contexts]
    rows: list[dict[str, Any]] = []
    overall_completed = 0
    overall_total = 0
    for model in sorted(m for m in known_models if m):
        scene_progress: dict[str, Any] = {}
        model_completed = 0
        model_total = 0
        for ctx in contexts:
            total = int(ctx.get("total_samples", 0) or 0)
            progress = _stage4_best_run_progress(ctx["roots"], ctx["scene_id"], model, str(ctx.get("manifest_path", "") or ""))
            completed = min(total, int(progress.get("completed", 0) or 0)) if total > 0 else int(progress.get("completed", 0) or 0)
            scene_progress[ctx["scene_id"]] = {
                "completed": completed,
                "total": total,
                "ratio": (float(completed) / float(total)) if total > 0 else None,
                "run_dir": str(progress.get("run_dir", "") or ""),
            }
            model_completed += completed
            model_total += total
        overall_completed += model_completed
        overall_total += model_total
        rows.append({
            "model": model,
            "scenes": scene_progress,
            "total_completed": model_completed,
            "total_samples": model_total,
            "total_ratio": (float(model_completed) / float(model_total)) if model_total > 0 else None,
        })
    return {
        "scene_id": GLOBAL_SCENE_ID if _is_global_scene_id(selected_scene_id) else str(selected_scene_id or ""),
        "scenes": scene_columns,
        "rows": rows,
        "overall_completed": overall_completed,
        "overall_total": overall_total,
        "overall_ratio": (float(overall_completed) / float(overall_total)) if overall_total > 0 else None,
    }


def _catalog_items_for_engine(catalog: list[dict[str, Any]], engine: str) -> list[dict[str, Any]]:
    engine_value = str(engine or "").strip().lower()
    return [dict(item) for item in list(catalog or []) if str(item.get("engine", "") or "").strip().lower() == engine_value]


def _aggregate_landmark_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_valid = 0
    counts: dict[str, int] = defaultdict(int)
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        total_valid += int(row.get("total_valid_landmarks", 0) or 0)
        for item in list(row.get("categories", []) or []):
            if isinstance(item, dict):
                label = str(item.get("category", "") or "").strip()
                count = int(item.get("count", 0) or 0)
            else:
                label = str(item or "").strip()
                count = 1 if label else 0
            if label:
                counts[label] += max(0, count)
    categories = [{"category": key, "count": counts[key]} for key in sorted(counts.keys())]
    return {
        "total_valid_landmarks": total_valid,
        "total_category_count": len(categories),
        "categories": categories,
        "by_difficulty": {},
    }


def _global_stage4_payload(
    *,
    catalog: list[dict[str, Any]],
    selected_engine: str,
    task_name: str | None,
    fallback_config_path: Path,
) -> dict[str, Any]:
    items = _catalog_items_for_engine(catalog, selected_engine)
    if not items:
        probe: dict[str, Any] = {}
        if fallback_config_path.exists():
            try:
                probe = _load_yaml(fallback_config_path)
            except Exception:
                probe = {}
        stage4_defaults = _stage4_cfg(probe)
        return {
            "scene_id": GLOBAL_SCENE_ID,
            "scene_label": GLOBAL_SCENE_LABEL,
            "engine": str(selected_engine),
            "config_path": _path_for_json(fallback_config_path) if fallback_config_path.exists() else None,
            "stage4_root": None,
            "latest_manifest_path": None,
            "latest_report_path": None,
            "sample_count": 0,
            "samples_preview": [],
            "latest_manifest_summary": None,
            "report_summary": None,
            "report_count": 0,
            "task_name": str(task_name or "").strip() or None,
            "known_models": [],
            "scene_landmark_stats": _aggregate_landmark_stats([]),
            "stage4_defaults": {
                "default_model": pick_first_text(
                    stage4_defaults.get("default_model"),
                    resolve_default_model(probe, stage_name="stage4"),
                ),
                "models": [str(item or "").strip() for item in list(stage4_defaults.get("models", []) or []) if str(item or "").strip()],
                "api_upload_resize_enabled": bool(stage4_defaults.get("api_upload_resize_enabled", True)),
                "api_upload_max_width": int(stage4_defaults.get("api_upload_max_width", 640) or 640),
                "api_upload_max_height": int(stage4_defaults.get("api_upload_max_height", 480) or 480),
                "api_upload_jpeg_quality": int(stage4_defaults.get("api_upload_jpeg_quality", 80) or 80),
                "web_image_resize_enabled": bool(stage4_defaults.get("web_image_resize_enabled", True)),
                "web_image_max_width": int(stage4_defaults.get("web_image_max_width", 640) or 640),
                "web_image_max_height": int(stage4_defaults.get("web_image_max_height", 480) or 480),
                "web_image_jpeg_quality": int(stage4_defaults.get("web_image_jpeg_quality", 80) or 80),
            },
            "global_manifests": [],
            "global_reports": [],
        }
    manifests: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    landmark_stats_rows: list[dict[str, Any]] = []
    known_models: set[str] = set()
    sample_count = 0
    default_cfg, default_cfg_path = _load_scene_config_from_catalog(
        engine=selected_engine,
        scene_id=items[0]["scene_id"],
        catalog=catalog,
        fallback_config_path=fallback_config_path,
    )
    stage4_defaults = _stage4_cfg(default_cfg)
    for item in items:
        scene_value = str(item.get("scene_id", "") or "").strip()
        cfg, cfg_path = _load_scene_config_from_catalog(
            engine=selected_engine,
            scene_id=scene_value,
            catalog=catalog,
            fallback_config_path=fallback_config_path,
        )
        scene_root = resolve_scene_root(cfg, scene_id=scene_value, engine=selected_engine, workspace_root=WORKSPACE_ROOT)
        roots = _stage4_roots(cfg, scene_value, selected_engine, str(task_name or ""), scene_root)
        latest_manifest_path, latest_manifest = _load_latest_manifest_multi(roots, scene_value)
        latest_report_path, latest_report = _load_latest_report_multi(roots, scene_value)
        if latest_manifest_path and latest_manifest:
            sample_count += int((latest_manifest or {}).get("sample_count", len((latest_manifest or {}).get("samples", []) or [])) or 0)
        for row in _list_manifests_multi(roots, scene_value):
            entry = dict(row)
            entry["scene_id"] = scene_value
            manifests.append(entry)
        for row in _list_reports_multi(roots, scene_value):
            entry = dict(row)
            entry["scene_id"] = scene_value
            reports.append(entry)
            if str(entry.get("model", "") or "").strip():
                known_models.add(str(entry.get("model", "") or "").strip())
        try:
            entries = _load_valid_instances(cfg, scene_root=scene_root, scene_id=scene_value)
            landmark_stats_rows.append(_collect_scene_landmark_stats(entries))
        except Exception:
            pass
        cfg_models = [str(m or "").strip() for m in list((_stage4_cfg(cfg).get("models", []) or [])) if str(m or "").strip()]
        known_models.update(cfg_models)
        default_model = pick_first_text(
            (_stage4_cfg(cfg).get("default_model")),
            resolve_default_model(cfg, stage_name="stage4"),
        )
        if default_model:
            known_models.add(default_model)
    manifests.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    reports.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    latest_manifest_path = str(manifests[0].get("path", "") or "") if manifests else None
    latest_report_path = str(reports[0].get("path", "") or "") if reports else None
    return {
        "scene_id": GLOBAL_SCENE_ID,
        "scene_label": GLOBAL_SCENE_LABEL,
        "engine": str(selected_engine),
        "config_path": _path_for_json(default_cfg_path),
        "stage4_root": None,
        "latest_manifest_path": latest_manifest_path,
        "latest_report_path": latest_report_path,
        "sample_count": sample_count,
        "samples_preview": [],
        "latest_manifest_summary": None,
        "report_summary": None,
        "report_count": len(reports),
        "task_name": str(task_name or "").strip() or None,
        "known_models": sorted(m for m in known_models if m),
        "scene_landmark_stats": _aggregate_landmark_stats(landmark_stats_rows),
        "stage4_defaults": {
            "default_model": pick_first_text(
                stage4_defaults.get("default_model"),
                resolve_default_model(default_cfg, stage_name="stage4"),
            ),
            "models": [str(item or "").strip() for item in list(stage4_defaults.get("models", []) or []) if str(item or "").strip()],
            "api_upload_resize_enabled": bool(stage4_defaults.get("api_upload_resize_enabled", True)),
            "api_upload_max_width": int(stage4_defaults.get("api_upload_max_width", 640) or 640),
            "api_upload_max_height": int(stage4_defaults.get("api_upload_max_height", 480) or 480),
            "api_upload_jpeg_quality": int(stage4_defaults.get("api_upload_jpeg_quality", 80) or 80),
            "web_image_resize_enabled": bool(stage4_defaults.get("web_image_resize_enabled", True)),
            "web_image_max_width": int(stage4_defaults.get("web_image_max_width", 640) or 640),
            "web_image_max_height": int(stage4_defaults.get("web_image_max_height", 480) or 480),
            "web_image_jpeg_quality": int(stage4_defaults.get("web_image_jpeg_quality", 80) or 80),
        },
        "global_manifests": manifests,
        "global_reports": reports,
    }


class ExperimentJobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, payload: dict[str, Any]) -> str:
        job_id = f"job_{_now_ts()}_{len(self._jobs)+1:03d}"
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": _iso_now(),
                "payload": payload,
                "progress": {"completed": 0, "total": 0, "current_model": None, "current_sample_id": None},
                "runs": [],
                "logs": [],
                "cancel_event": threading.Event(),
            }
        return job_id

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["logs"].append({"ts": _iso_now(), "message": str(message)})
            job["logs"] = job["logs"][-200:]

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in changes.items():
                job[key] = value

    def update_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["progress"].update(progress)

    def add_run(self, job_id: str, run_info: dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["runs"].append(run_info)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            cancel_event = job.get("cancel_event")
            if isinstance(cancel_event, threading.Event):
                cancel_event.set()
            if job.get("status") not in {"completed", "error", "cancelled"}:
                job["status"] = "cancel_requested"
            return True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            out = dict(job)
            out.pop("cancel_event", None)
            return out

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = []
            for job in self._jobs.values():
                row = dict(job)
                row.pop("cancel_event", None)
                rows.append(row)
            return sorted(rows, key=lambda item: item.get("created_at", ""), reverse=True)


def _make_web_app(default_config: dict[str, Any], *, scene_id: str, engine: str, config_path: Path) -> Any:
    if Flask is None:
        raise ImportError("Flask is required for stage4 web")
    app = Flask(__name__)
    catalog = _discover_scene_catalog(config_path)
    job_manager = ExperimentJobManager()
    default_engine = str(engine).strip().lower()
    default_scene_id = str(scene_id).strip()

    def _load_scene_context(selected_engine: str | None, selected_scene_id: str | None) -> tuple[dict[str, Any], Path, Path, Path]:
        engine_value = str(selected_engine or default_engine).strip().lower()
        scene_value = str(selected_scene_id or default_scene_id).strip()
        cfg, cfg_path = _load_scene_config_from_catalog(
            engine=engine_value,
            scene_id=scene_value,
            catalog=catalog,
            fallback_config_path=config_path,
        )
        scene_root = resolve_scene_root(cfg, scene_id=scene_value, engine=engine_value, workspace_root=WORKSPACE_ROOT)
        stage4_root = _ensure_dir(_resolve_stage4_root(cfg, scene_root=scene_root))
        return cfg, cfg_path, scene_root, stage4_root

    def _scene_state(selected_engine: str | None, selected_scene_id: str | None, task_name: str | None = None) -> dict[str, Any]:
        if _is_global_scene_id(selected_scene_id):
            return _global_stage4_payload(
                catalog=catalog,
                selected_engine=str(selected_engine or default_engine).strip().lower(),
                task_name=task_name,
                fallback_config_path=config_path,
            )
        cfg, cfg_path, scene_root, stage4_root = _load_scene_context(selected_engine, selected_scene_id)
        task_cfg = cfg.get("task", {}) or {}
        engine_value = str(task_cfg.get("engine", selected_engine or default_engine)).strip().lower()
        scene_value = str(task_cfg.get("scene_id", selected_scene_id or default_scene_id)).strip()
        roots = _stage4_roots(cfg, scene_value, engine_value, str(task_name or ""), scene_root)
        manifest_path, manifest = _load_latest_manifest_multi(roots, scene_value)
        report_path, report = _load_latest_report_multi(roots, scene_value)
        samples = list((manifest or {}).get("samples", []) or [])
        reports = _list_reports_multi(roots, scene_value)
        try:
            entries = _load_valid_instances(cfg, scene_root=scene_root, scene_id=scene_value)
            landmark_stats = _collect_scene_landmark_stats(entries)
        except Exception as exc:
            entries = []
            landmark_stats = {"error": str(exc), "total_valid_landmarks": 0, "total_category_count": 0, "categories": [], "by_difficulty": {}}
        stage4_defaults = _stage4_cfg(cfg)
        default_model = pick_first_text(
            stage4_defaults.get("default_model"),
            resolve_default_model(cfg, stage_name="stage4"),
            (cfg.get("stage2", {}) or {}).get("auto_label_model"),
        )
        cfg_models = [str(item or "").strip() for item in list(stage4_defaults.get("models", []) or []) if str(item or "").strip()]
        known_models = sorted({m for m in [default_model, *cfg_models, *(row.get("model", "") for row in reports)] if m})
        return {
            "scene_id": scene_value,
            "engine": engine_value,
            "config_path": _path_for_json(cfg_path),
            "stage4_root": _path_for_json(stage4_root),
            "latest_manifest_path": _path_for_json(manifest_path) if manifest_path else None,
            "latest_report_path": _path_for_json(report_path) if report_path else None,
            "sample_count": len(samples),
            "samples_preview": samples[: min(12, len(samples))],
            "latest_manifest_summary": _summarize_manifest_payload(manifest or {}) if manifest else None,
            "report_summary": (report or {}).get("summary", None),
            "report_count": len(reports),
            "task_name": str(task_name or "").strip() or None,
            "known_models": known_models,
            "scene_landmark_stats": landmark_stats,
            "stage4_defaults": {
                "default_model": default_model,
                "models": cfg_models,
                "api_upload_resize_enabled": bool(stage4_defaults.get("api_upload_resize_enabled", True)),
                "api_upload_max_width": int(stage4_defaults.get("api_upload_max_width", 640) or 640),
                "api_upload_max_height": int(stage4_defaults.get("api_upload_max_height", 480) or 480),
                "api_upload_jpeg_quality": int(stage4_defaults.get("api_upload_jpeg_quality", 80) or 80),
                "web_image_resize_enabled": bool(stage4_defaults.get("web_image_resize_enabled", True)),
                "web_image_max_width": int(stage4_defaults.get("web_image_max_width", 640) or 640),
                "web_image_max_height": int(stage4_defaults.get("web_image_max_height", 480) or 480),
                "web_image_jpeg_quality": int(stage4_defaults.get("web_image_jpeg_quality", 80) or 80),
            },
        }

    def _render_shell(active_page: str) -> str:
        nav_items = [
            ("generate", "任务生成"),
            ("dataset", "任务查看"),
            ("experiments", "实验执行"),
            ("results", "结果查看"),
            ("metrics", "指标汇总"),
        ]
        nav_html = "".join(
            f'<a class="nav-item {"active" if key == active_page else ""}" href="/{key}">{label}</a>'
            for key, label in nav_items
        )
        return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4 QA Workbench</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffdf9;
      --paper-soft: #fbf7f0;
      --paper-soft-2: #efe7d9;
      --paper-soft-3: #f7f2e9;
      --ink: #14212b;
      --muted: #6d7379;
      --line: #ded5c5;
      --line-soft: #eadfce;
      --brand: #114b5f;
      --accent: #d95d39;
      --good: #1b7f5c;
      --warn: #b76e00;
      --bad: #9b2226;
      --banner-bg: rgba(255,253,249,0.96);
      --footer-bg: rgba(255,253,249,0.96);
      --input-bg: #fff;
      --input-border: #cfc5b5;
      --nav-bg: #fff;
      --nav-active-bg: var(--brand);
      --nav-active-border: var(--brand);
      --nav-active-ink: #fff;
      --secondary-btn-bg: #eadfce;
      --secondary-btn-ink: var(--ink);
      --card-shadow: 0 14px 30px rgba(17,75,95,0.05);
      --thead-bg: #f6f0e4;
      --inline-code-bg: #f7f2e9;
      --inline-code-border: #e8dcc8;
      --thumb-border: #e5ded1;
    }}
    body[data-theme="dark"] {{
      --bg: #0f141d;
      --paper: #171c25;
      --paper-soft: #1b2230;
      --paper-soft-2: #20293a;
      --paper-soft-3: #151b24;
      --ink: #eef3f8;
      --muted: #9da9b6;
      --line: #2a3442;
      --line-soft: #344355;
      --brand: #55c1ff;
      --accent: #ffb347;
      --good: #36c48b;
      --warn: #ffb347;
      --bad: #ff8f70;
      --banner-bg: rgba(15,19,26,0.92);
      --footer-bg: rgba(15,19,26,0.94);
      --input-bg: #1e2530;
      --input-border: #2a3442;
      --nav-bg: #171c25;
      --nav-active-bg: #1f2835;
      --nav-active-border: #324052;
      --nav-active-ink: #eef3f8;
      --secondary-btn-bg: #202837;
      --secondary-btn-ink: #eef3f8;
      --card-shadow: 0 18px 32px rgba(0,0,0,0.18);
      --thead-bg: #151b24;
      --inline-code-bg: #111722;
      --inline-code-border: #2d3745;
      --thumb-border: #2d3745;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: linear-gradient(180deg, color-mix(in srgb, var(--bg) 85%, #ffffff 15%) 0%, var(--bg) 100%); color: var(--ink); font-family: "Helvetica Neue", Arial, sans-serif; }}
    .banner {{ position: sticky; top: 0; z-index: 20; padding: 18px 24px; background: var(--banner-bg); border-bottom: 1px solid var(--line); backdrop-filter: blur(10px); }}
    .banner-top {{ display:flex; align-items:center; justify-content:space-between; gap: 18px; }}
    .brand {{ font-size: 26px; font-weight: 700; letter-spacing: 0.02em; color: var(--brand); }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .selectors {{ display:flex; align-items:end; gap: 12px; flex-wrap: wrap; }}
    .selectors label, .form-grid label {{ display:flex; flex-direction:column; gap: 6px; font-size: 12px; color: var(--muted); }}
    select, input, textarea {{ padding: 10px 12px; border:1px solid var(--input-border); border-radius: 10px; background: var(--input-bg); color: var(--ink); font: inherit; }}
    select[multiple] {{ min-height: 110px; }}
    button {{ border: 0; border-radius: 10px; padding: 10px 14px; background: var(--brand); color: #fff; cursor:pointer; font-weight: 600; }}
    button.secondary {{ background: var(--secondary-btn-bg); color: var(--secondary-btn-ink); }}
    button.warn {{ background: var(--accent); }}
    .nav {{ display:flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }}
    .nav-item {{ text-decoration:none; color: var(--ink); padding: 9px 14px; border-radius: 999px; border:1px solid var(--line); background:var(--nav-bg); }}
    .nav-item.active {{ background: var(--nav-active-bg); color:var(--nav-active-ink); border-color: var(--nav-active-border); }}
    .page {{ padding: 26px 24px 170px; }}
    .footer {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 20; background: var(--footer-bg); border-top: 1px solid var(--line); backdrop-filter: blur(10px); }}
    .footer-shell {{ overflow: hidden; }}
    .footer-grid {{ display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 0; }}
    .footer-col {{ padding: 16px 24px; min-height: 108px; }}
    .footer-col + .footer-col {{ border-left: 1px solid var(--line); }}
    .footer-title {{ margin: 0 0 10px; font-size: 14px; letter-spacing: 0.02em; color: var(--brand); text-transform: uppercase; }}
    .footer-list {{ display:grid; gap: 8px; color: var(--muted); font-size: 13px; }}
    .footer-item strong {{ color: var(--ink); }}
    .grid {{ display:grid; grid-template-columns: 1.1fr 0.9fr; gap: 20px; }}
    .stack {{ display:grid; gap: 20px; }}
    .card {{ background: var(--paper); border:1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: var(--card-shadow); }}
    .card h2, .card h3 {{ margin: 0 0 12px; }}
    .card p {{ color: var(--muted); }}
    .form-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    .compact-field {{ max-width: none; }}
    .compact-field input {{ width: 100%; min-width: 0; padding-left: 10px; padding-right: 10px; text-align: left; }}
    .actions {{ display:flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line-soft); padding: 9px 8px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .pill {{ display:inline-block; padding: 4px 9px; border-radius:999px; background:var(--paper-soft-2); color: var(--ink); font-size: 12px; }}
    .mini-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }}
    .metric {{ background: var(--paper-soft); border:1px solid var(--line-soft); border-radius: 12px; padding: 12px; }}
    .metric .v {{ font-size: 24px; font-weight: 700; color: var(--brand); }}
    .bars {{ display:grid; gap: 10px; }}
    .bar-row {{ display:grid; grid-template-columns: 180px 1fr 72px; gap: 10px; align-items:center; }}
    .bar-track {{ background:var(--paper-soft-2); border-radius:999px; overflow:hidden; height: 12px; }}
    .bar-fill {{ background: linear-gradient(90deg, var(--brand), var(--accent)); height:100%; }}
    .sample-list {{ display:grid; gap: 12px; max-height: 760px; overflow:auto; }}
    .sample-card {{ border:1px solid var(--line-soft); border-radius: 12px; padding: 10px; background: var(--input-bg); }}
    .sample-card img {{ width: 100%; max-width: 320px; border-radius: 10px; border: 1px solid var(--thumb-border); }}
    .detail-grid {{ display:grid; gap: 12px; }}
    .summary-list {{ display:grid; gap: 8px; color: var(--muted); font-size: 13px; }}
    .summary-item strong {{ color: var(--ink); }}
    .inline-code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background:var(--inline-code-bg); border:1px solid var(--inline-code-border); border-radius: 8px; padding: 2px 6px; }}
    .compact-table th, .compact-table td {{ font-size: 12px; padding: 7px 6px; }}
    .thumb-row {{ display:flex; gap: 10px; flex-wrap: wrap; align-items:flex-start; }}
    .thumb-box {{ max-width: 220px; }}
    .thumb-box img {{ width: 100%; border-radius: 10px; border:1px solid var(--thumb-border); }}
    .option-grid {{ display:grid; gap: 8px; }}
    .option-card {{ border:1px solid var(--line-soft); border-radius: 10px; padding: 8px; background:var(--paper-soft); }}
    .choice-shell {{ display:grid; gap:8px; }}
    .choice-actions {{ display:flex; gap:8px; flex-wrap:wrap; }}
    .choice-actions button {{ padding:6px 10px; font-size:12px; border-radius:999px; }}
    .choice-panel {{ display:grid; grid-template-columns: 1.15fr 0.85fr; border:1px solid var(--line-soft); border-radius:14px; overflow:hidden; background:var(--paper-soft); min-height:220px; }}
    .choice-panel.no-summary {{ grid-template-columns: 1fr; }}
    .choice-left {{ display:grid; gap:8px; padding:10px; max-height:320px; overflow:auto; background:var(--paper); }}
    .choice-right {{ display:grid; align-content:start; gap:10px; padding:12px; border-left:1px solid var(--line-soft); background:var(--paper-soft-3); }}
    .choice-right .muted strong {{ color:var(--ink); }}
    .choice-selected-list {{ display:grid; gap:8px; max-height:170px; overflow:auto; }}
    .choice-selected-item {{ display:flex; gap:8px; align-items:flex-start; padding:8px 10px; border:1px solid var(--line-soft); border-radius:10px; background:var(--paper); font-size:12px; }}
    .choice-selected-empty {{ padding:10px; border:1px dashed var(--line-soft); border-radius:10px; color:var(--muted); font-size:12px; background:var(--paper); }}
    .choice-list {{ display:grid; gap:8px; }}
    .choice-list.cols-2 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .choice-list.cols-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .choice-item {{ display:flex; align-items:center; gap:8px; flex-wrap:nowrap; color:var(--ink); font-size:13px; padding:8px 10px; border:1px solid var(--line-soft); border-radius:10px; background:var(--paper-soft); }}
    .choice-item input {{ margin:0; flex:0 0 auto; }}
    .choice-item span {{ flex:1 1 auto; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .muted {{ color: var(--muted); }}
    .status-running {{ color: var(--good); }}
    .status-error, .status-cancelled {{ color: var(--bad); }}
    .status-queued, .status-cancel_requested {{ color: var(--warn); }}
    pre {{ white-space: pre-wrap; word-break: break-word; background:var(--inline-code-bg); border:1px solid var(--inline-code-border); padding: 12px; border-radius: 12px; max-height: 360px; overflow:auto; }}
    @media (max-width: 1100px) {{ .grid, .footer-grid, .choice-panel {{ grid-template-columns: 1fr; }} .choice-right {{ border-left:0; border-top:1px solid var(--line-soft); }} }}
  </style>
</head>
<body>
  <div class="banner">
    <div class="banner-top">
      <div>
        <div class="brand">UAV-DualCog Stage 4 QA Workbench</div>
        <div class="sub">Multi-scene task generation, experiment control, result browsing, and metric analysis</div>
      </div>
      <div class="selectors">
        <label>Engine
          <select id="global_engine"></select>
        </label>
        <label>Scene
          <select id="global_scene"></select>
        </label>
        <label>Task
          <select id="global_task_pipeline"></select>
        </label>
        <label>Web Img
          <input id="web_img_enabled" type="checkbox" checked onchange="renderPage()">
        </label>
        <label>W
          <input id="web_img_width" value="640" style="width:84px;" onchange="renderPage()">
        </label>
        <label>H
          <input id="web_img_height" value="480" style="width:84px;" onchange="renderPage()">
        </label>
        <label>Q
          <input id="web_img_quality" value="80" style="width:70px;" onchange="renderPage()">
        </label>
        <label>Theme
          <select id="theme_select">
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
        <button class="secondary" onclick="refreshAll()">Refresh</button>
      </div>
    </div>
    <div class="nav">{nav_html}</div>
  </div>
  <div class="page" id="app"></div>
  <div class="footer">
    <div class="footer-shell">
      <div class="footer-grid">
        <div class="footer-col">
        <div class="footer-title">当前工作区</div>
        <div class="footer-list" id="footer_left"></div>
        </div>
        <div class="footer-col">
        <div class="footer-title">操作提示</div>
        <div class="footer-list" id="footer_right"></div>
        </div>
      </div>
    </div>
  </div>
<script>
const ACTIVE_PAGE = {json.dumps(active_page)};
const DEFAULT_ENGINE = {json.dumps(default_engine)};
const DEFAULT_SCENE = {json.dumps(default_scene_id)};
const THEME_KEY = 'uav_dualcog_stage4_theme';
const TASK_KEY = 'uav_dualcog_stage4_task_pipeline';
const state = {{
  catalog: [],
  taskPipelines: [],
  current: null,
  manifests: [],
  reports: [],
  jobs: [],
  activeManifest: null,
  activeReport: null,
  activeManifestData: null,
  activeReportRows: [],
  metricsMatrix: null,
  activeManifestSampleId: '',
}};

function getSelectedEngine() {{
  return document.getElementById('global_engine')?.value || localStorage.getItem('stage4_engine') || DEFAULT_ENGINE;
}}
function getSelectedScene() {{
  return document.getElementById('global_scene')?.value || localStorage.getItem('stage4_scene') || DEFAULT_SCENE;
}}
function getSelectedTaskPipeline() {{
  return document.getElementById('global_task_pipeline')?.value || localStorage.getItem(TASK_KEY) || '';
}}
function globalQuery() {{
  const q = new URLSearchParams();
  q.set('engine', getSelectedEngine());
  q.set('scene_id', getSelectedScene());
  const taskName = getSelectedTaskPipeline();
  if(taskName) q.set('task_name', taskName);
  return q.toString();
}}
function esc(text) {{
  return String(text ?? '').replace(/[&<>"]/g, (ch)=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[ch]));
}}
function fmtPct(v) {{
  if(v === null || v === undefined || Number.isNaN(Number(v))) return '-';
  return `${{(Number(v)*100).toFixed(1)}}%`;
}}
function fmtMs(v) {{
  if(v === null || v === undefined || Number.isNaN(Number(v))) return '-';
  return `${{Number(v).toFixed(0)}} ms`;
}}
function fmtFloat(v, digits=3) {{
  if(v === null || v === undefined || Number.isNaN(Number(v))) return '-';
  return Number(v).toFixed(digits);
}}
function fmtBbox(bbox) {{
  if(!Array.isArray(bbox) || bbox.length !== 4) return '-';
  return bbox.map((v)=>Number(v).toFixed(3)).join(', ');
}}
function storageKey(name) {{
  return `stage4_${{getSelectedEngine()}}_${{getSelectedScene()}}_${{name}}`;
}}
function readStoredList(name, fallback=[]) {{
  try {{
    const raw = localStorage.getItem(storageKey(name));
    if(!raw) return [...fallback];
    const value = JSON.parse(raw);
    return Array.isArray(value) ? value : [...fallback];
  }} catch {{
    return [...fallback];
  }}
}}
function writeStoredList(name, values) {{
  localStorage.setItem(storageKey(name), JSON.stringify([...(values || [])]));
}}
function readStoredValue(name, fallback='') {{
  const raw = localStorage.getItem(storageKey(name));
  return raw === null || raw === undefined || raw === '' ? fallback : raw;
}}
function writeStoredValue(name, value) {{
  localStorage.setItem(storageKey(name), String(value ?? ''));
}}
function choiceValues(id) {{
  return [...document.querySelectorAll(`#${{id}} input:checked`)].map((el)=>el.value);
}}
function choiceValue(id, fallback='') {{
  return document.querySelector(`#${{id}} input:checked`)?.value || fallback;
}}
function renderChoiceSelectionSummary(id) {{
  const root = document.getElementById(id);
  if(!root) return;
  const box = document.getElementById(`${{id}}_selected`);
  if(!box) return;
  if(root.dataset.showSelected !== '1') {{
    box.innerHTML = '';
    return;
  }}
  const multiple = root.dataset.multiple === '1';
  const selectedInputs = [...root.querySelectorAll('input:checked')];
  const title = multiple ? `已选 ${{selectedInputs.length}} 项` : `当前选择 ${{selectedInputs.length ? '1' : '0'}} 项`;
  const items = selectedInputs.map((input)=> {{
    const text = input.closest('.choice-item')?.querySelector('span')?.textContent || input.value;
    return `<div class="choice-selected-item"><strong>${{esc(input.value)}}</strong><span>${{esc(text)}}</span></div>`;
  }}).join('');
  box.innerHTML = `
    <div class="muted"><strong>${{title}}</strong></div>
    <div class="choice-selected-list">
      ${{items || '<div class="choice-selected-empty">当前还没有选中项</div>'}}
    </div>`;
}}
function renderChoiceGroup(id, options, cfg={{}}) {{
  const multiple = cfg.multiple !== false;
  const cols = Math.max(1, Math.min(3, Number(cfg.cols || 1)));
  const storageName = cfg.storageName || id;
  const showSelected = !!cfg.showSelected;
  const showActions = cfg.showActions !== false && multiple;
  const fallbackValues = multiple ? (cfg.defaultValues || []) : [];
  const fallbackValue = multiple ? '' : (cfg.defaultValue || '');
  const selectedList = multiple ? readStoredList(storageName, fallbackValues) : [readStoredValue(storageName, fallbackValue)];
  const selectedSet = new Set((selectedList || []).map((x)=>String(x)));
  const items = (options || []).map((opt, idx)=>`
    <label class="choice-item">
      <input type="${{multiple ? 'checkbox' : 'radio'}}" name="${{esc(id)}}" value="${{esc(opt.value)}}" ${{selectedSet.has(String(opt.value)) || (!multiple && !selectedSet.size && idx === 0) ? 'checked' : ''}}>
      <span>${{esc(opt.label ?? opt.value)}}</span>
    </label>`).join('');
  const actions = showActions ? `
    <div class="choice-actions">
      <button type="button" class="secondary" data-choice-id="${{esc(id)}}" data-choice-action="all">全选</button>
      <button type="button" class="secondary" data-choice-id="${{esc(id)}}" data-choice-action="none">清空</button>
    </div>` : '';
  return `
    <div class="choice-shell">
      ${{actions}}
      <div class="choice-panel ${{showSelected ? '' : 'no-summary'}}">
        <div class="choice-left">
          <div class="muted"><strong>可选列表</strong></div>
          <div class="choice-list cols-${{cols}}" id="${{esc(id)}}" data-multiple="${{multiple ? '1' : '0'}}" data-storage-name="${{esc(storageName)}}" data-show-selected="${{showSelected ? '1' : '0'}}">
            ${{items || '<div class="muted">暂无可选项</div>'}}
          </div>
        </div>
        ${{showSelected ? `<div class="choice-right" id="${{esc(id)}}_selected"><div class="choice-selected-empty">当前还没有选中项</div></div>` : ''}}
      </div>
    </div>`;
}}
function bindChoiceGroup(id, cfg={{}}) {{
  const root = document.getElementById(id);
  if(!root) return;
  const multiple = root.dataset.multiple === '1';
  const storageName = root.dataset.storageName || cfg.storageName || id;
  const onChange = typeof cfg.onChange === 'function' ? cfg.onChange : null;
  root.querySelectorAll('input').forEach((input)=> {{
    input.onchange = ()=> {{
      if(multiple) writeStoredList(storageName, choiceValues(id));
      else writeStoredValue(storageName, choiceValue(id, cfg.defaultValue || ''));
      renderChoiceSelectionSummary(id);
      if(onChange) onChange();
    }};
  }});
  document.querySelectorAll(`button[data-choice-id="${{id}}"]`).forEach((btn)=> {{
    btn.onclick = ()=> {{
      const action = btn.dataset.choiceAction;
      const inputs = [...root.querySelectorAll('input')];
      if(action === 'all') inputs.forEach((el)=> {{ el.checked = true; }});
      if(action === 'none') inputs.forEach((el)=> {{ el.checked = false; }});
      writeStoredList(storageName, choiceValues(id));
      renderChoiceSelectionSummary(id);
      if(onChange) onChange();
    }};
  }});
  if(multiple) writeStoredList(storageName, choiceValues(id));
  else writeStoredValue(storageName, choiceValue(id, cfg.defaultValue || ''));
  renderChoiceSelectionSummary(id);
}}
function getSelectedCategories(prefix='gen_categories_group') {{
  return choiceValues(prefix);
}}
function buildImageSrc(path) {{
  const defaults = state.current?.stage4_defaults || {{}};
  const enabledEl = document.getElementById('web_img_enabled');
  const enabled = enabledEl ? enabledEl.checked : !!defaults.web_image_resize_enabled;
  const url = new URL('/api/file', window.location.origin);
  url.searchParams.set('path', path);
  if(enabled) {{
    const width = parseInt(document.getElementById('web_img_width')?.value || String(defaults.web_image_max_width || 640), 10);
    const height = parseInt(document.getElementById('web_img_height')?.value || String(defaults.web_image_max_height || 480), 10);
    const quality = parseInt(document.getElementById('web_img_quality')?.value || String(defaults.web_image_jpeg_quality || 80), 10);
    url.searchParams.set('resize', '1');
    url.searchParams.set('w', String(width));
    url.searchParams.set('h', String(height));
    url.searchParams.set('q', String(quality));
  }}
  return url.toString();
}}
function getTheme() {{
  const saved = localStorage.getItem(THEME_KEY);
  return saved === 'dark' || saved === 'light' ? saved : 'light';
}}
function applyTheme(theme) {{
  const normalized = theme === 'dark' ? 'dark' : 'light';
  document.body.dataset.theme = normalized;
  const sel = document.getElementById('theme_select');
  if(sel) sel.value = normalized;
  localStorage.setItem(THEME_KEY, normalized);
}}
function initTheme() {{
  applyTheme(getTheme());
  const sel = document.getElementById('theme_select');
  if(sel) sel.onchange = ()=>applyTheme(sel.value);
}}
function comboLabel(viewDefinition, taskType, difficulty) {{
  const taskMap = {{
    self_where: 'Self-1 / Where Am I',
    self_what: 'Self-2 / What Am I Doing',
    env_where: 'Env-1 / Where Is The Landmark',
    env_how: 'Env-2 / How Should I Move',
  }};
  if(taskMap[taskType]) return `${{taskMap[taskType]}} / ${{difficulty}}`;
  const head = viewDefinition === 'Object-Centric View' ? 'ObjView' : 'ObsView';
  const tail = taskType === 'label_multiple_choice' ? 'Image→Label' : 'Label→Image';
  return `${{head}} / ${{tail}} / ${{difficulty}}`;
}}
function getSelectedDifficulty(prefix, fallback='4way') {{
  return choiceValue(`${{prefix}}_difficulty_group`, readStoredValue(`${{prefix}}_difficulty`, fallback) || fallback);
}}
function getSceneDifficultyStats(difficulty) {{
  return state.current?.scene_landmark_stats?.by_difficulty?.[difficulty] || null;
}}
function summaryItemsHtml(items) {{
  return (items || []).map((item)=>`<div class="summary-item">${{item}}</div>`).join('') || '<div class="muted">暂无信息</div>';
}}
function renderManifestSummary(summary) {{
  if(!summary) return '<div class="muted">暂无 manifest 摘要</div>';
  const comboEntries = Object.entries(summary.task_combo_counts || {{}});
  const diffEntries = Object.entries(summary.difficulty_counts || {{}});
  return `
    <div class="summary-list">
      <div class="summary-item"><strong>使用地标数</strong>：${{esc(summary.used_landmark_count || 0)}}</div>
      <div class="summary-item"><strong>包含地标类别数</strong>：${{esc(summary.used_category_count || 0)}}</div>
      <div class="summary-item"><strong>类别筛选</strong>：${{(summary.selected_landmark_categories || []).length ? (summary.selected_landmark_categories || []).map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') : '全部类别'}}</div>
      <div class="summary-item"><strong>任务组合</strong>：${{comboEntries.length ? comboEntries.map(([k,v])=>`${{esc(k)}}=${{esc(v)}}`).join(' | ') : '暂无'}}</div>
      <div class="summary-item"><strong>难度分布</strong>：${{diffEntries.length ? diffEntries.map(([k,v])=>`${{esc(k)}}=${{esc(v)}}`).join(' | ') : '暂无'}}</div>
      <div class="summary-item"><strong>地标类别</strong>：${{(summary.used_categories || []).slice(0, 12).map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') || '暂无'}}</div>
    </div>
  `;
}}
function estimateAutoSampleCount(strategy, difficulty, perLandmark, perTaskLandmark) {{
  const stats = getSceneDifficultyStats(difficulty);
  if(!stats) return 0;
  const selectedViewDefs = choiceValues('gen_view_defs_group');
  const selectedTaskTypes = choiceValues('gen_task_types_group');
  const selectedCategories = new Set(getSelectedCategories().map((x)=>String(x).toLowerCase()));
  const categoryMap = state.current?.scene_landmark_stats?.landmark_category_map || {{}};
  const perTask = Object.values(stats.per_task || {{}}).filter((item)=>(
    (!selectedViewDefs.length || selectedViewDefs.includes(item.view_definition)) &&
    (!selectedTaskTypes.length || selectedTaskTypes.includes(item.task_type))
  ));
  if(strategy === 'per_landmark') {{
    if(!perTask.length) return 0;
    const idSets = perTask.map((item)=>new Set((item.eligible_landmark_ids || []).filter((id)=>(
      !selectedCategories.size || selectedCategories.has(String(categoryMap[id] || '').toLowerCase())
    ))));
    if(!idSets.length || !idSets[0].size) return 0;
    const intersection = [...idSets[0]].filter((id)=>idSets.every((set)=>set.has(id)));
    return Math.max(0, intersection.length * Number(perLandmark || 0));
  }}
  return perTask.reduce((acc, item)=>acc + (item.eligible_landmark_ids || []).filter((id)=>(
    !selectedCategories.size || selectedCategories.has(String(categoryMap[id] || '').toLowerCase())
  )).length * Number(perTaskLandmark || 0), 0);
}}
function optionHtml(options) {{
  return (options || []).map((item)=>`<span class="pill">${{esc(item.option_id)}}:${{esc(item.label || item.view_under_definition || '')}}</span>`).join(' ');
}}
async function api(url, opts) {{
  const resp = await fetch(url, opts);
  const data = await resp.json();
  if(!resp.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}}
async function loadCatalog() {{
  state.catalog = await api('/api/catalog');
  const taskData = await api('/api/task_pipeline_tasks');
  state.taskPipelines = Array.isArray(taskData?.tasks) ? taskData.tasks : [];
  const engines = [...new Set(state.catalog.map((item)=>item.engine))];
  const engineSel = document.getElementById('global_engine');
  engineSel.innerHTML = engines.map((name)=>`<option value="${{esc(name)}}" ${{name===getSelectedEngine()?'selected':''}}>${{esc(name)}}</option>`).join('');
  const taskSel = document.getElementById('global_task_pipeline');
  if(taskSel) {{
    const currentTask = getSelectedTaskPipeline();
    taskSel.innerHTML = ['<option value="">Scene-local outputs</option>', ...state.taskPipelines.map((name)=>`<option value="${{esc(name)}}" ${{String(name)===String(currentTask)?'selected':''}}>${{esc(name)}}</option>`)].join('');
    taskSel.onchange = async ()=> {{
      localStorage.setItem(TASK_KEY, taskSel.value || '');
      await refreshAll();
    }};
  }}
  await populateScenes();
  engineSel.onchange = async ()=> {{
    localStorage.setItem('stage4_engine', engineSel.value);
    await populateScenes();
    await refreshAll();
  }};
}}
async function populateScenes() {{
  const engine = getSelectedEngine();
  const scenes = state.catalog.filter((item)=>item.engine===engine).map((item)=>item.scene_id);
  const preferred = localStorage.getItem('stage4_scene') || {json.dumps(GLOBAL_SCENE_ID)};
  const sceneSel = document.getElementById('global_scene');
  const allOptions = [{json.dumps(GLOBAL_SCENE_ID)}, ...scenes];
  sceneSel.innerHTML = allOptions.map((scene)=>`<option value="${{esc(scene)}}" ${{scene===preferred?'selected':''}}>${{esc(scene === {json.dumps(GLOBAL_SCENE_ID)} ? {json.dumps(GLOBAL_SCENE_LABEL)} : scene)}}</option>`).join('');
  if(!allOptions.includes(sceneSel.value) && allOptions.length) sceneSel.value = allOptions[0];
  sceneSel.onchange = async ()=> {{
    localStorage.setItem('stage4_scene', sceneSel.value);
    await refreshAll();
  }};
}}
async function refreshAll() {{
  const engine = getSelectedEngine();
  const scene = getSelectedScene();
  const taskName = getSelectedTaskPipeline();
  localStorage.setItem('stage4_engine', engine);
  localStorage.setItem('stage4_scene', scene);
  localStorage.setItem(TASK_KEY, taskName);
  state.current = await api(`/api/state?${{globalQuery()}}`);
  state.manifests = await api(`/api/manifests?${{globalQuery()}}`);
  state.reports = await api(`/api/reports?${{globalQuery()}}`);
  state.jobs = await api('/api/jobs');
  state.metricsMatrix = null;
  if(!state.activeManifest && state.current.latest_manifest_path) state.activeManifest = state.current.latest_manifest_path;
  if(!state.activeReport && state.current.latest_report_path) state.activeReport = state.current.latest_report_path;
  const defaults = state.current.stage4_defaults || {{}};
  const enabledEl = document.getElementById('web_img_enabled');
  const widthEl = document.getElementById('web_img_width');
  const heightEl = document.getElementById('web_img_height');
  const qualityEl = document.getElementById('web_img_quality');
  if(enabledEl) enabledEl.checked = !!defaults.web_image_resize_enabled;
  if(widthEl) widthEl.value = String(defaults.web_image_max_width ?? 640);
  if(heightEl) heightEl.value = String(defaults.web_image_max_height ?? 480);
  if(qualityEl) qualityEl.value = String(defaults.web_image_jpeg_quality ?? 80);
  renderPage();
}}
function buildSummaryCards(summary) {{
  if(!summary) return '<div class="muted">暂无实验结果</div>';
  return `
    <div class="mini-grid">
      <div class="metric"><div class="muted">样本数</div><div class="v">${{esc(summary.count)}}</div></div>
      <div class="metric"><div class="muted">Parse Success</div><div class="v">${{fmtPct(summary.parse_success_rate)}}</div></div>
      <div class="metric"><div class="muted">Option Accuracy</div><div class="v">${{fmtPct(summary.option_accuracy)}}</div></div>
      <div class="metric"><div class="muted">BBox Acc@50IoU</div><div class="v">${{fmtPct(summary['bbox_acc@50iou'])}}</div></div>
      <div class="metric"><div class="muted">BBox Mean IoU</div><div class="v">${{Number(summary.bbox_mean_iou||0).toFixed(3)}}</div></div>
      <div class="metric"><div class="muted">Avg Latency</div><div class="v">${{fmtMs(summary.avg_latency_ms)}}</div></div>
    </div>
  `;
}}
function footerHtml(items) {{
  return (items || []).map((item)=>`<div class="footer-item">${{item}}</div>`).join('') || '<div class="muted">暂无内容</div>';
}}
function renderFooter() {{
  const current = state.current || {{}};
  const left = document.getElementById('footer_left');
  const right = document.getElementById('footer_right');
  if(!left || !right) return;
  let leftItems = [];
  let rightItems = [];
  if(ACTIVE_PAGE === 'generate') {{
    leftItems = [
      `<strong>任务数据生成</strong>：为当前场景生成 Stage 4-1 QA manifest。`,
      `<strong>当前场景</strong>：${{esc(current.engine || '')}} / ${{esc(current.scene_id || '')}}。`,
      `<strong>已有数据</strong>：manifest ${{esc(state.manifests.length)}} 份，report ${{esc(state.reports.length)}} 份。`,
      `<strong>默认采样</strong>：可按新四类任务、难度和主视角策略联合控制。`,
    ];
    rightItems = [
      `<strong>当前操作</strong>：设置生成方式、样本量、任务类型和难度。`,
      `先确认 Stage 2 已完成地标方向与主视角整理，再开始生成。`,
      `样本量建议先用 5 到 20 条快速检查，再扩大规模。`,
      `生成后可切到“任务数据查看”核对 Self/Env 配对图像、参考图 BBox 与选项是否合理。`,
    ];
  }} else if(ACTIVE_PAGE === 'dataset') {{
    leftItems = [
      `<strong>任务数据查看</strong>：浏览 manifest 列表与样本预览。`,
      `<strong>当前 Manifest</strong>：${{esc(state.activeManifest || current.latest_manifest_path || '未选择')}}。`,
      `<strong>预览内容</strong>：显示参考图、任务定义、候选项与样本元信息。`,
      `<strong>目标</strong>：确认四类任务与 4-way/8-way 采样是否符合预期。`,
    ];
    rightItems = [
      `<strong>当前操作</strong>：切换 manifest 并抽查样本。`,
      `优先检查参考图的目标地标 BBox 是否准确。`,
      `重点核对 Self-1/2 与 Env-1/2 是否共享同一张查询图或观测图。`,
      `如果采样偏斜，可返回生成页调整主视角策略或样本量。`,
    ];
  }} else if(ACTIVE_PAGE === 'experiments') {{
    const runningJobs = (state.jobs || []).filter((job)=>['queued','running','cancel_requested'].includes(job.status || '')).length;
    const latestJob = (state.jobs || [])[0] || null;
    leftItems = [
      `<strong>实验执行</strong>：选择 manifest、模型、分辨率、压缩率、并发和限频。`,
      `<strong>默认上传预处理</strong>：${{esc(current.stage4_defaults?.api_upload_max_width ?? 640)}} x ${{esc(current.stage4_defaults?.api_upload_max_height ?? 480)}}, JPEG ${{esc(current.stage4_defaults?.api_upload_jpeg_quality ?? 80)}}。`,
      `<strong>默认网页图片预览</strong>：${{esc(current.stage4_defaults?.web_image_max_width ?? 640)}} x ${{esc(current.stage4_defaults?.web_image_max_height ?? 480)}}, JPEG ${{esc(current.stage4_defaults?.web_image_jpeg_quality ?? 80)}}。`,
      `<strong>后台任务</strong>：当前共有 ${{esc((state.jobs || []).length)}} 个 job，可启动、刷新、取消。`,
      `<strong>运行中</strong>：${{esc(runningJobs)}} 个；最近任务：${{esc(latestJob?.job_id || '-')}} / ${{esc(latestJob?.status || '-')}}。`,
      `<strong>实验目标</strong>：输出 Option 预测与归一化 BBox 预测。`,
    ];
    rightItems = [
      `<strong>当前操作</strong>：启动任务、取消任务、查看当前样本进度。`,
      `为降低 524 风险，建议先用 480P 级别上传和较低并发。`,
      `多模型对比时，优先保持相同 manifest、相同任务筛选条件。`,
      `如果任务运行较久，可用刷新进度或查看任务详情定位卡点。`,
    ];
  }} else if(ACTIVE_PAGE === 'results') {{
    leftItems = [
      `<strong>实验结果查看</strong>：浏览每次 run 的总体结果与逐样本输出。`,
      `<strong>当前 Report</strong>：${{esc(state.activeReport || current.latest_report_path || '未选择')}}。`,
      `<strong>逐样本字段</strong>：pred/gold option、BBox@50、IoU。`,
      `<strong>用途</strong>：定位单个失败样本和解析错误。`,
    ];
    rightItems = [
      `<strong>当前操作</strong>：切换 report 并查看逐样本结果。`,
      `优先看 option 是否选对，再看 BBox 是否达到 Acc@50IoU。`,
      `如果 parse success 低，先检查提示词、输出格式或模型稳定性。`,
      `如需横向比较，建议切到“指标汇总”查看大表和分组图。`,
    ];
  }} else {{
    leftItems = [
      `<strong>实验指标汇总</strong>：查看总体 summary、分组条形图和实验大表。`,
      `<strong>核心指标</strong>：Option Accuracy、BBox Acc@50IoU、BBox Mean IoU、Avg Latency。`,
      `<strong>当前汇总</strong>：基于最近 report 自动加载。`,
      `<strong>分组分析</strong>：按参照系、任务类型、难度对比表现。`,
    ];
    rightItems = [
      `<strong>当前操作</strong>：切换 latest-only 统计并刷新矩阵。`,
      `建议先看总体指标，再看分组差异，最后回到结果页抽查样本。`,
      `如果某一类任务显著偏低，优先回查对应 prompt 和数据采样。`,
      `做论文表格时，可直接以这里的大表为整理基础。`,
    ];
  }}
  left.innerHTML = footerHtml(leftItems);
  right.innerHTML = footerHtml(rightItems);
}}
function renderGeneratePage() {{
  const current = state.current || {{}};
  const difficulty = getSelectedDifficulty('gen', '4way');
  const stats = getSceneDifficultyStats(difficulty) || {{}};
  const categoryOptions = (state.current?.scene_landmark_stats?.categories || []).map((name)=>({{value:name, label:name}}));
  const strategyOptions = [
    {{value:'manual', label:'手动指定样本量'}},
    {{value:'per_landmark', label:'自动：每个满足四类任务的地标生成固定数量'}},
    {{value:'per_task_landmark', label:'自动：每类任务的可用地标分别生成固定数量'}},
  ];
  const taskTypeOptions = [
    {{value:'self_where', label:'self_where'}},
    {{value:'self_what', label:'self_what'}},
    {{value:'env_where', label:'env_where'}},
    {{value:'env_how', label:'env_how'}},
  ];
  const difficultyOptions = [
    {{value:'4way', label:'4way'}},
    {{value:'8way', label:'8way'}},
  ];
  const perTaskRows = Object.values(stats.per_task || {{}}).map((item)=>`
    <tr>
      <td>${{esc(item.display_name)}}</td>
      <td>${{esc(item.eligible_landmark_count || 0)}}</td>
      <td>${{esc(item.eligible_category_count || 0)}}</td>
      <td>${{fmtFloat(item.avg_view_count || 0, 2)}}</td>
    </tr>`).join('');
  return `
    <div class="grid">
      <div class="card">
        <h2>任务数据生成</h2>
        <p>按场景生成 Stage 4-1 QA manifest，可按参照系、任务类型和难度进行采样控制。</p>
        <div class="form-grid">
          <label class="compact-field">样本量<input id="gen_sample_count" value="5"></label>
          <label class="compact-field">随机种子<input id="gen_seed" value="7"></label>
          <label>主视角参考<input id="gen_main_only" type="checkbox" checked></label>
          <label class="compact-field">每地标样本数<input id="gen_per_landmark" value="2" onchange="updateGenerateEstimator()"></label>
          <label class="compact-field">每类任务每地标样本数<input id="gen_per_task_landmark" value="1" onchange="updateGenerateEstimator()"></label>
        </div>
        <div class="form-grid" style="margin-top:12px;">
          <label>生成方式
            ${{renderChoiceGroup('gen_strategy_group', strategyOptions, {{ multiple:false, storageName:'gen_strategy', defaultValue:'manual', cols:1, showSelected:false }})}}
          </label>
          <label>任务类型
            ${{renderChoiceGroup('gen_task_types_group', taskTypeOptions, {{ multiple:true, storageName:'gen_task_types', defaultValues:['self_where','self_what','env_where','env_how'], cols:1, showSelected:false, showActions:false }})}}
          </label>
          <label>难度
            ${{renderChoiceGroup('gen_difficulty_group', difficultyOptions, {{ multiple:false, storageName:'gen_difficulty', defaultValue:'4way', cols:1, showSelected:false }})}}
          </label>
        </div>
        <div class="form-grid" style="margin-top:12px;">
          <label style="grid-column:1/-1;">地标类别
            ${{renderChoiceGroup('gen_categories_group', categoryOptions, {{ multiple:true, storageName:'gen_categories', defaultValues:[], cols:1, showSelected:true, showActions:true }})}}
          </label>
        </div>
        <div class="summary-list" style="margin-top:12px;" id="generate_estimator"></div>
        <div class="actions">
          <button onclick="runGenerate()">生成任务数据</button>
          <button class="secondary" onclick="loadManifestPreview()">加载最新 Manifest</button>
        </div>
        <pre id="generate_out"></pre>
      </div>
      <div class="card">
        <h2>场景摘要</h2>
        <div class="mini-grid">
          <div class="metric"><div class="muted">Scene</div><div class="v" style="font-size:20px;">${{esc(current.scene_id || '')}}</div></div>
          <div class="metric"><div class="muted">Engine</div><div class="v" style="font-size:20px;">${{esc(current.engine || '')}}</div></div>
          <div class="metric"><div class="muted">满足四类任务地标数</div><div class="v">${{esc(stats.eligible_all_task_landmark_count || 0)}}</div></div>
          <div class="metric"><div class="muted">满足四类任务类别数</div><div class="v">${{esc(stats.eligible_all_task_category_count || 0)}}</div></div>
          <div class="metric"><div class="muted">平均视图数</div><div class="v">${{fmtFloat(stats.avg_view_count_per_eligible_landmark || 0, 2)}}</div></div>
          <div class="metric"><div class="muted">Manifest 数量</div><div class="v">${{state.manifests.length}}</div></div>
        </div>
        <div class="detail-grid" style="margin-top:16px;">
          <div class="summary-list">
            <div class="summary-item"><strong>Stage 2 有效地标总数</strong>：${{esc(current.scene_landmark_stats?.total_valid_landmarks || 0)}}</div>
            <div class="summary-item"><strong>全部地标类别数</strong>：${{esc(current.scene_landmark_stats?.total_category_count || 0)}}</div>
            <div class="summary-item"><strong>可用于四类任务的类别</strong>：${{(stats.eligible_all_task_categories || []).slice(0, 16).map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') || '暂无'}}</div>
          </div>
          <div>
            <h3>地标任务可用性统计</h3>
            <table class="compact-table">
              <thead><tr><th>任务</th><th>地标数</th><th>类别数</th><th>平均视图</th></tr></thead>
              <tbody>${{perTaskRows || '<tr><td colspan="4" class="muted">暂无统计</td></tr>'}}</tbody>
            </table>
          </div>
          <div>
            <h3>最近实验摘要</h3>
            ${{buildSummaryCards(current.report_summary)}}
          </div>
        </div>
      </div>
    </div>
  `;
}}
function renderDatasetPage() {{
  const manifestRows = state.manifests.map((row)=>`
    <tr onclick="selectManifest('${{esc(row.path)}}')" style="cursor:pointer;">
      <td>${{esc(row.generation_id)}}</td>
      <td>${{esc(row.sample_count)}}</td>
      <td>${{esc(row.summary?.used_landmark_count || 0)}}</td>
      <td>${{esc(row.summary?.used_category_count || 0)}}</td>
      <td>${{esc((row.task_types||[]).join(','))}}</td>
      <td>${{esc((row.difficulties||[]).join(','))}}</td>
      <td>${{esc(row.generated_at || '')}}</td>
    </tr>`).join('');
  return `
    <div class="grid">
      <div class="card">
        <h2>任务数据查看</h2>
        <table>
          <thead><tr><th>Manifest</th><th>样本数</th><th>地标数</th><th>类别数</th><th>任务类型</th><th>难度</th><th>时间</th></tr></thead>
          <tbody>${{manifestRows || '<tr><td colspan="7" class="muted">暂无 manifest</td></tr>'}}</tbody>
        </table>
        <div class="section-title">任务列表</div>
        <div id="manifest_task_list" class="table-wrap"></div>
      </div>
      <div class="card">
        <h2>样本预览</h2>
        <div id="manifest_summary" class="summary-list" style="margin-bottom:12px;"></div>
        <div id="manifest_preview" class="sample-list"></div>
      </div>
    </div>
  `;
}}
function renderExperimentsPage() {{
  const current = state.current || {{}};
  const stage4Defaults = current.stage4_defaults || {{}};
  const defaultModel = stage4Defaults.default_model || '';
  const modelOptions = (current.known_models || []).map((m)=>({{value:m, label:m}}));
  const jobRows = (state.jobs || []).map((job)=>`
    <tr>
      <td>${{esc(job.job_id)}}</td>
      <td class="status-${{esc(job.status)}}">${{esc(job.status)}}</td>
      <td>${{esc(job.payload?.scene_id || '')}}</td>
      <td>${{esc(job.payload?.model || ((job.payload?.models || [])[0] || ''))}}</td>
      <td>${{esc(job.payload?.manifest_name || '-')}}</td>
      <td>${{esc(job.progress?.completed || 0)}} / ${{esc(job.progress?.total || 0)}}</td>
      <td>${{esc(job.progress?.current_sample_id || '')}}</td>
      <td><button class="secondary" onclick="inspectJob('${{esc(job.job_id)}}')">查看</button> <button class="warn" onclick="cancelJob('${{esc(job.job_id)}}')">取消</button></td>
    </tr>`).join('');
  return `
    <div class="grid">
      <div class="card">
        <h2>实验执行</h2>
        <div class="form-grid">
          <label>Manifest
            <select id="exp_manifest">${{state.manifests.map((row)=>`<option value="${{esc(row.path)}}" ${{row.path===state.activeManifest?'selected':''}}>${{esc(row.generation_id)}}</option>`).join('')}}</select>
          </label>
          <label class="compact-field">样本数<input id="exp_limit" value="5"></label>
          <label class="compact-field">并发<input id="exp_concurrency" value="1"></label>
          <label class="compact-field">RPM<input id="exp_rpm" value="0"></label>
          <label class="compact-field">TPM<input id="exp_tpm" value="0"></label>
          <label class="compact-field">宽<input id="exp_width" value="${{esc(stage4Defaults.api_upload_max_width ?? 640)}}"></label>
          <label class="compact-field">高<input id="exp_height" value="${{esc(stage4Defaults.api_upload_max_height ?? 480)}}"></label>
          <label class="compact-field">JPEG<input id="exp_quality" value="${{esc(stage4Defaults.api_upload_jpeg_quality ?? 80)}}"></label>
          <label class="compact-field">超时<input id="exp_timeout" value="30"></label>
        </div>
        <div class="form-grid" style="margin-top:12px;">
          <label style="grid-column:1/-1;">模型（可多选）
            ${{renderChoiceGroup('exp_models_group', modelOptions, {{ multiple:true, storageName:'exp_models', defaultValues: defaultModel ? [defaultModel] : [], cols:1, showSelected:true, showActions:true }})}}
          </label>
        </div>
        <div class="actions">
          <button onclick="startExperimentJob()">启动实验</button>
          <button class="secondary" onclick="refreshJobs()">刷新进度</button>
        </div>
        <div id="experiment_manifest_info" class="summary-list" style="margin-top:12px;"></div>
        <pre id="job_detail"></pre>
      </div>
      <div class="card">
        <h2>后台实验任务</h2>
        <table>
          <thead><tr><th>Job</th><th>状态</th><th>场景</th><th>模型</th><th>Manifest</th><th>进度</th><th>当前样本</th><th>操作</th></tr></thead>
          <tbody>${{jobRows || '<tr><td colspan="8" class="muted">暂无任务</td></tr>'}}</tbody>
        </table>
      </div>
    </div>
  `;
}}
function renderResultsPage() {{
  const rows = state.reports.map((row)=>`
    <tr onclick="selectReport('${{esc(row.path)}}')" style="cursor:pointer;">
      <td>${{esc(row.run_id)}}</td>
      <td>${{esc(row.model)}}</td>
      <td>${{esc(row.manifest_name || '-')}}</td>
      <td>${{esc(row.count)}}</td>
      <td>${{fmtPct(row.option_accuracy)}}</td>
      <td>${{fmtPct(row['bbox_acc@50iou'])}}</td>
      <td>${{fmtMs(row.avg_latency_ms)}}</td>
      <td>${{esc(row.generated_at || '')}}</td>
    </tr>`).join('');
  return `
    <div class="grid">
      <div class="card">
        <h2>实验结果查看</h2>
        <table>
          <thead><tr><th>Run</th><th>模型</th><th>Manifest</th><th>样本数</th><th>Opt</th><th>BBox</th><th>延迟</th><th>时间</th></tr></thead>
          <tbody>${{rows || '<tr><td colspan="8" class="muted">暂无 report</td></tr>'}}</tbody>
        </table>
      </div>
      <div class="card">
        <h2>逐样本结果</h2>
        <div id="report_meta" class="summary-list" style="margin-bottom:10px;"><div class="muted">选择左侧 report 查看</div></div>
        <div id="report_rows"><div class="muted">选择左侧 report 查看</div></div>
      </div>
    </div>
  `;
}}
function renderMetricsPage() {{
  return `
    <div class="stack">
      <div class="card">
        <h2>实验指标汇总</h2>
        <div class="actions">
          <label style="display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;">
            <input id="metrics_latest_only" type="checkbox" onchange="loadMetricsMatrix()">
            按单个样本最新结果汇总
          </label>
          <label style="display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px;">
            <input id="metrics_by_difficulty" type="checkbox" onchange="loadMetricsMatrix()">
            按难度区分
          </label>
          <button class="secondary" onclick="loadMetricsMatrix()">刷新指标大表</button>
          <button class="secondary" onclick="exportMetricsCsv()">导出 CSV</button>
        </div>
        <div id="metrics_report_summary_cards">${{buildSummaryCards(state.current?.report_summary)}}</div>
      </div>
      <div class="card">
        <h2>分组分析</h2>
        <div id="metrics_group_bars" class="stack"><div class="muted">加载中...</div></div>
      </div>
      <div class="card">
        <h2>实验大表</h2>
        <div id="metrics_matrix"><div class="muted">加载中...</div></div>
      </div>
      <div class="card">
        <h2>实验进度大表</h2>
        <div id="metrics_progress_summary" class="summary-list" style="margin-bottom:12px;"></div>
        <div id="metrics_progress_matrix"><div class="muted">加载中...</div></div>
      </div>
    </div>
  `;
}}
function renderPage() {{
  const holder = document.getElementById('app');
  if(ACTIVE_PAGE === 'generate') holder.innerHTML = renderGeneratePage();
  else if(ACTIVE_PAGE === 'dataset') holder.innerHTML = renderDatasetPage();
  else if(ACTIVE_PAGE === 'experiments') holder.innerHTML = renderExperimentsPage();
  else if(ACTIVE_PAGE === 'results') holder.innerHTML = renderResultsPage();
  else holder.innerHTML = renderMetricsPage();
  if(ACTIVE_PAGE === 'generate') {{
    bindChoiceGroup('gen_strategy_group', {{ multiple:false, storageName:'gen_strategy', defaultValue:'manual', onChange:updateGenerateEstimator }});
    bindChoiceGroup('gen_view_defs_group', {{ multiple:true, storageName:'gen_view_defs', onChange:updateGenerateEstimator }});
    bindChoiceGroup('gen_task_types_group', {{ multiple:true, storageName:'gen_task_types', onChange:updateGenerateEstimator }});
    bindChoiceGroup('gen_categories_group', {{ multiple:true, storageName:'gen_categories', onChange:updateGenerateEstimator }});
    bindChoiceGroup('gen_difficulty_group', {{ multiple:false, storageName:'gen_difficulty', defaultValue:'4way', onChange:updateGenerateEstimator }});
  }}
  if(ACTIVE_PAGE === 'experiments') {{
    bindChoiceGroup('exp_models_group', {{ multiple:true, storageName:'exp_models' }});
  }}
  renderFooter();
  if(ACTIVE_PAGE === 'generate') updateGenerateEstimator();
  if(ACTIVE_PAGE === 'dataset') loadManifestPreview();
  if(ACTIVE_PAGE === 'experiments') updateExperimentManifestInfo();
  if(ACTIVE_PAGE === 'results') loadReportRows();
  if(ACTIVE_PAGE === 'metrics') loadMetricsMatrix();
}}
async function runGenerate() {{
  const strategy = choiceValue('gen_strategy_group', readStoredValue('gen_strategy', 'manual') || 'manual');
  const difficulty = getSelectedDifficulty('gen', '4way');
  let sampleCount = parseInt(document.getElementById('gen_sample_count').value || '5', 10);
  if(strategy !== 'manual') {{
    sampleCount = estimateAutoSampleCount(
      strategy,
      difficulty,
      parseInt(document.getElementById('gen_per_landmark').value || '0', 10),
      parseInt(document.getElementById('gen_per_task_landmark').value || '0', 10),
    );
    document.getElementById('gen_sample_count').value = String(sampleCount);
  }}
  const payload = {{
    engine: getSelectedEngine(),
    scene_id: getSelectedScene(),
    task_name: getSelectedTaskPipeline(),
    sample_count: sampleCount,
    seed: parseInt(document.getElementById('gen_seed').value || '7', 10),
    reference_main_only: document.getElementById('gen_main_only').checked,
    view_definitions: [],
    task_types: choiceValues('gen_task_types_group'),
    landmark_categories: getSelectedCategories(),
    difficulties: [difficulty],
  }};
  const data = await api('/api/generate', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload) }});
  document.getElementById('generate_out').textContent = JSON.stringify(data, null, 2);
  state.activeManifest = data.manifest_path;
  await refreshAll();
}}
function updateGenerateEstimator() {{
  const holder = document.getElementById('generate_estimator');
  if(!holder) return;
  const strategy = choiceValue('gen_strategy_group', readStoredValue('gen_strategy', 'manual') || 'manual');
  const difficulty = getSelectedDifficulty('gen', '4way');
  const stats = getSceneDifficultyStats(difficulty) || {{}};
  const perLandmark = parseInt(document.getElementById('gen_per_landmark')?.value || '0', 10);
  const perTaskLandmark = parseInt(document.getElementById('gen_per_task_landmark')?.value || '0', 10);
  const estimated = estimateAutoSampleCount(strategy, difficulty, perLandmark, perTaskLandmark);
  const items = [
    `<strong>当前难度</strong>：${{esc(difficulty)}}`,
    `<strong>满足四类任务地标数</strong>：${{esc(stats.eligible_all_task_landmark_count || 0)}}`,
    `<strong>类别筛选</strong>：${{getSelectedCategories().length ? getSelectedCategories().map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') : '全部类别'}}`,
    `<strong>自动估算样本量</strong>：${{strategy === 'manual' ? '手动模式，不自动改写' : esc(estimated)}}`,
    `<strong>说明</strong>：${{strategy === 'per_task_landmark' ? '按四类任务各自可用地标分别生成。' : strategy === 'per_landmark' ? '按满足四类任务的地标统一生成。' : '直接使用上面的样本量输入框。'}}`,
  ];
  holder.innerHTML = summaryItemsHtml(items);
}}
async function loadManifestPreview() {{
  if(!state.activeManifest && state.manifests.length) state.activeManifest = state.manifests[0].path;
  const holder = document.getElementById('manifest_preview');
  const summaryHolder = document.getElementById('manifest_summary');
  const listHolder = document.getElementById('manifest_task_list');
  if(!holder) return;
  if(!state.activeManifest) {{
    holder.innerHTML = '<div class="muted">暂无 manifest</div>';
    if(summaryHolder) summaryHolder.innerHTML = '<div class="muted">暂无 manifest 摘要</div>';
    if(listHolder) listHolder.innerHTML = '<div class="muted">暂无样本</div>';
    return;
  }}
  const data = await api(`/api/manifest?path=${{encodeURIComponent(state.activeManifest)}}`);
  state.activeManifestData = data;
  state.activeManifestSampleId = state.activeManifestSampleId || String((data.samples || [])[0]?.sample_id || '');
  if(summaryHolder) summaryHolder.innerHTML = renderManifestSummary(data.summary || {{
    used_landmark_count: [...new Set((data.samples || []).map((x)=>x.landmark_id))].length,
    used_category_count: [...new Set((data.samples || []).map((x)=>x.landmark_category))].length,
    used_categories: [...new Set((data.samples || []).map((x)=>x.landmark_category))],
    selected_landmark_categories: data.selected_landmark_categories || [],
    task_combo_counts: {{}},
    difficulty_counts: {{}},
  }});
  if(listHolder) {{
    listHolder.innerHTML = `<table><thead><tr><th>sample_id</th><th>task</th><th>difficulty</th><th>landmark</th></tr></thead><tbody>${{
      (data.samples || []).map((sample)=>`<tr onclick="selectManifestSample('${{esc(sample.sample_id)}}')" style="cursor:pointer;${{String(state.activeManifestSampleId)===String(sample.sample_id)?'background:var(--paper-soft);':''}}"><td>${{esc(sample.sample_id)}}</td><td>${{esc(sample.task_type)}}</td><td>${{esc(sample.difficulty)}}</td><td>${{esc(sample.landmark_id)}}</td></tr>`).join('') || '<tr><td colspan="4" class="muted">暂无样本</td></tr>'
    }}</tbody></table>`;
  }}
  const sample = (data.samples || []).find((row)=>String(row.sample_id)===String(state.activeManifestSampleId)) || (data.samples || [])[0] || null;
  if(!sample) {{
    holder.innerHTML = '<div class="muted">暂无样本</div>';
    return;
  }}
  state.activeManifestSampleId = String(sample.sample_id || '');
  holder.innerHTML = `
    <div class="sample-card">
      <div><strong>${{esc(sample.sample_id)}}</strong></div>
      <div class="muted">${{esc(comboLabel(sample.view_definition, sample.task_type, sample.difficulty))}}</div>
      <div class="summary-list" style="margin:8px 0;">
        <div class="summary-item"><strong>地标</strong>：${{esc(sample.landmark_id)}} / ${{esc(sample.landmark_category)}}</div>
        <div class="summary-item"><strong>System Prompt</strong>：<pre>${{esc(sample.system_prompt || '')}}</pre></div>
        <div class="summary-item"><strong>User Prompt</strong>：<pre>${{esc(sample.user_prompt || sample.prompt_text || '')}}</pre></div>
        <div class="summary-item"><strong>选项</strong>：${{optionHtml(sample.label_options || sample.candidates || [])}}</div>
        <div class="summary-item"><strong>答案</strong>：${{esc((sample.answer_option_ids || [sample.answer_option_id || '-']).join(', '))}}</div>
        <div class="summary-item"><strong>答案 BBox</strong>：<span class="inline-code">${{fmtBbox(sample.answer_bbox_xyxy_norm)}}</span></div>
      </div>
      <div class="thumb-row">
        <div class="thumb-box">
          <div class="muted" style="margin-bottom:4px;">参考图</div>
          <img src="${{buildImageSrc(sample.reference_image_with_bbox)}}" />
        </div>
        ${{
          ['label_multiple_choice','self_where','env_where','env_how'].includes(sample.task_type)
            ? `<div class="thumb-box"><div class="muted" style="margin-bottom:4px;">目标图</div><img src="${{buildImageSrc(sample.target_image)}}" /></div>`
            : `<div class="option-grid">${{(sample.candidates || []).map((cand)=>`<div class="option-card"><div><strong>${{esc(cand.option_id)}}</strong> / ${{esc(cand.view_under_definition || '')}}</div><div class="muted">BBox: ${{fmtBbox(cand.bbox_xyxy_norm)}}</div><img src="${{buildImageSrc(cand.image)}}" style="margin-top:6px; max-width:180px;" /></div>`).join('')}}</div>`
        }}
      </div>
    </div>`;
}}
function selectManifest(path) {{
  state.activeManifest = path;
  loadManifestPreview();
}}
function selectManifestSample(sampleId) {{
  state.activeManifestSampleId = String(sampleId || '');
  loadManifestPreview();
}}
async function updateExperimentManifestInfo() {{
  const holder = document.getElementById('experiment_manifest_info');
  const select = document.getElementById('exp_manifest');
  if(!holder || !select || !select.value) return;
  const row = (state.manifests || []).find((item)=>item.path === select.value);
  if(row) {{
    holder.innerHTML = renderManifestSummary(row.summary) + `
      <div class="summary-list" style="margin-top:10px;">
        <div class="summary-item"><strong>参照系</strong>：${{(row.view_definitions || []).map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') || '由 manifest 决定'}}</div>
        <div class="summary-item"><strong>任务类型</strong>：${{(row.task_types || []).map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') || '由 manifest 决定'}}</div>
        <div class="summary-item"><strong>难度</strong>：${{(row.difficulties || []).map((x)=>`<span class="pill">${{esc(x)}}</span>`).join(' ') || '由 manifest 决定'}}</div>
      </div>`;
  }} else {{
    holder.innerHTML = '<div class="muted">暂无 manifest 参考信息</div>';
  }}
  select.onchange = ()=> {{
    state.activeManifest = select.value;
    updateExperimentManifestInfo();
  }};
}}
async function startExperimentJob() {{
  const selected = choiceValues('exp_models_group');
  const payload = {{
    engine: getSelectedEngine(),
    scene_id: getSelectedScene(),
    task_name: getSelectedTaskPipeline(),
    manifest_path: document.getElementById('exp_manifest').value,
    limit: parseInt(document.getElementById('exp_limit').value || '0', 10),
    concurrency: parseInt(document.getElementById('exp_concurrency').value || '1', 10),
    rpm_limit: parseInt(document.getElementById('exp_rpm').value || '0', 10),
    tpm_limit: parseInt(document.getElementById('exp_tpm').value || '0', 10),
    upload_max_width: parseInt(document.getElementById('exp_width').value || '640', 10),
    upload_max_height: parseInt(document.getElementById('exp_height').value || '480', 10),
    upload_jpeg_quality: parseInt(document.getElementById('exp_quality').value || '85', 10),
    timeout_s: parseFloat(document.getElementById('exp_timeout').value || '30'),
    models: [...new Set([...selected])],
  }};
  const data = await api('/api/jobs/start', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(payload) }});
  document.getElementById('job_detail').textContent = JSON.stringify(data, null, 2);
  await refreshJobs();
}}
async function refreshJobs() {{
  state.jobs = await api('/api/jobs');
  renderPage();
}}
async function inspectJob(jobId) {{
  const data = await api(`/api/jobs/${{encodeURIComponent(jobId)}}`);
  const el = document.getElementById('job_detail');
  if(el) el.textContent = JSON.stringify(data, null, 2);
}}
async function cancelJob(jobId) {{
  const data = await api(`/api/jobs/${{encodeURIComponent(jobId)}}/cancel`, {{ method:'POST' }});
  const el = document.getElementById('job_detail');
  if(el) el.textContent = JSON.stringify(data, null, 2);
  await refreshJobs();
}}
function selectReport(path) {{
  state.activeReport = path;
  loadReportRows();
}}
async function loadReportRows() {{
  const holder = document.getElementById('report_rows');
  const meta = document.getElementById('report_meta');
  if(!holder) return;
  if(!state.activeReport && state.reports.length) state.activeReport = state.reports[0].path;
  if(!state.activeReport) {{
    holder.innerHTML = '<div class="muted">暂无 report</div>';
    if(meta) meta.innerHTML = '<div class="muted">暂无 report</div>';
    return;
  }}
  const rows = await api(`/api/report_rows?path=${{encodeURIComponent(state.activeReport)}}`);
  state.activeReportRows = rows;
  const reportRow = (state.reports || []).find((row)=>row.path === state.activeReport);
  if(meta) {{
    meta.innerHTML = `
      <div class="summary-item"><strong>Run</strong>：${{esc(reportRow?.run_id || '-')}}</div>
      <div class="summary-item"><strong>模型</strong>：${{esc(reportRow?.model || '-')}}</div>
      <div class="summary-item"><strong>Manifest</strong>：${{esc(reportRow?.manifest_name || '-')}}</div>
    `;
  }}
  const grouped = {{}};
  rows.forEach((row)=>{{ const key = `${{row.view_definition}}|${{row.task_type}}|${{row.difficulty}}`; (grouped[key] ||= []).push(row); }});
  holder.innerHTML = Object.entries(grouped).map(([key, taskRows])=>`
    <div class="card" style="margin-bottom:12px;">
      <div class="summary-item"><strong>${{esc(key)}}</strong></div>
      <table>
        <thead><tr><th>sample_id</th><th>gold option</th><th>pred option</th><th>gold bbox</th><th>pred bbox</th><th>option</th><th>bbox@50</th><th>iou</th></tr></thead>
        <tbody>${{taskRows.map((row)=>`<tr>
          <td>${{esc(row.sample_id)}}</td>
          <td>${{esc(row.gold_option_id || '-')}}</td>
          <td>${{esc(row.pred_option_id || '-')}}</td>
          <td><span class="inline-code">${{fmtBbox(row.gold_bbox_xyxy_norm)}}</span></td>
          <td><span class="inline-code">${{fmtBbox(row.pred_bbox_xyxy_norm)}}</span></td>
          <td>${{esc(row.option_correct)}}</td>
          <td>${{esc(row['bbox_acc@50iou'])}}</td>
          <td>${{Number(row.bbox_iou||0).toFixed(3)}}</td>
        </tr>`).join('')}}</tbody>
      </table>
    </div>`).join('') || '<div class="muted">暂无逐样本结果</div>';
}}
async function loadMetricsMatrix() {{
  const holder = document.getElementById('metrics_matrix');
  if(!holder) return;
  const latestOnly = document.getElementById('metrics_latest_only')?.checked ? '1' : '0';
  const byDifficulty = document.getElementById('metrics_by_difficulty')?.checked ? '1' : '0';
  const mq = globalQuery();
  const rsEl = document.getElementById('metrics_report_summary_cards');
  const gbEl = document.getElementById('metrics_group_bars');
  const pmEl = document.getElementById('metrics_progress_matrix');
  const psEl = document.getElementById('metrics_progress_summary');
  holder.innerHTML = '<div class="muted">加载中...</div>';
  if(pmEl) pmEl.innerHTML = '<div class="muted">加载中...</div>';
  if(psEl) psEl.innerHTML = '';
  if(gbEl) gbEl.innerHTML = '<div class="muted">加载中...</div>';
  let reports = [];
  let data = {{ columns: [], rows: [] }};
  let progressData = {{ scenes: [], rows: [], overall_completed: 0, overall_total: 0, overall_ratio: null }};
  try {{
    [reports, data, progressData] = await Promise.all([
      api(`/api/reports?${{mq}}`).catch(()=>[]),
      api(`/api/metrics_matrix?${{mq}}&latest_only=${{latestOnly}}&by_difficulty=${{byDifficulty}}`).catch(()=>({{columns:[],rows:[]}})),
      api(`/api/experiment_progress_matrix?${{mq}}`).catch(()=>({{scenes:[],rows:[],overall_completed:0,overall_total:0,overall_ratio:null}})),
    ]);
  }} catch (e) {{
    holder.innerHTML = `<div class="warn">加载失败：${{esc(String((e && e.message) || e))}}</div>`;
    if(pmEl) pmEl.innerHTML = '';
    if(gbEl) gbEl.innerHTML = '';
    renderFooter();
    return;
  }}
  state.metricsMatrix = data;
  const latestReport = (reports || [])[0] || {{}};
  const latestSummary = latestReport.summary || {{}};
  if(rsEl) rsEl.innerHTML = buildSummaryCards(latestSummary);
  function barSection(title, payload) {{
    if(!payload || Object.keys(payload).length === 0) return `<div class="muted">${{title}}: 暂无数据</div>`;
    return `<div style="margin-bottom:18px;"><h3>${{title}}</h3><div class="bars">${{Object.entries(payload).map(([name, item])=>`
      <div class="bar-row">
        <div>${{esc(name)}}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${{Math.max(0, Math.min(100, Number(item.option_accuracy||0)*100))}}%"></div></div>
        <div>${{fmtPct(item.option_accuracy)}}</div>
      </div>`).join('')}}</div></div>`;
  }}
  if(gbEl) {{
    const grouped = latestSummary.grouped || {{}};
    gbEl.innerHTML = [
      barSection('按参照系 · Option', grouped.view_definition),
      barSection('按任务类型 · Option', grouped.task_type),
      barSection('按难度 · Option', grouped.difficulty),
    ].join('');
  }}
  const columns = data.columns || [];
  const rows = data.rows || [];
  const groups = [];
  for(const col of columns) {{
    const last = groups.length ? groups[groups.length - 1] : null;
    if(last && last.display_name === col.display_name && last.view_definition === col.view_definition && last.task_type === col.task_type) {{
      last.columns.push(col);
    }} else {{
      groups.push({{
        display_name: col.display_name,
        view_definition: col.view_definition,
        task_type: col.task_type,
        columns: [col],
      }});
    }}
  }}
  const topHead = groups.map((group)=>`<th colspan="${{group.columns.length * 4}}">${{esc(group.display_name)}}</th>`).join('');
  const diffHead = groups.map((group)=>group.columns.map((col)=>`<th colspan="4">${{esc(col.difficulty)}}</th>`).join('')).join('');
  const metricHead = columns.map(()=>`<th>Opt</th><th>BBox</th><th>IoU</th><th>N</th>`).join('');
  holder.innerHTML = `<table class="compact-table">
    <thead>
      <tr><th rowspan="3">Model</th>${{topHead || '<th>暂无列</th>'}}</tr>
      <tr>${{diffHead}}</tr>
      <tr>${{metricHead}}</tr>
    </thead>
    <tbody>${{rows.map((row)=>`<tr>
      <td>${{esc(row.model)}}</td>
      ${{
        columns.map((col)=>{{
          const cell = row.combos?.[col.combo_id] || {{}};
          return `<td>${{fmtPct(cell.option_accuracy)}}</td><td>${{fmtPct(cell['bbox_acc@50iou'])}}</td><td>${{fmtFloat(cell.bbox_mean_iou, 3)}}</td><td>${{esc(cell.count ?? 0)}}</td>`;
        }}).join('')
      }}
    </tr>`).join('') || `<tr><td colspan="${{1 + columns.length * 4}}" class="muted">暂无实验记录</td></tr>`}}</tbody>
  </table>`;
  const progressSummary = document.getElementById('metrics_progress_summary');
  if(progressSummary) {{
    progressSummary.innerHTML = `<div class="summary-item"><strong>总体进度</strong>：${{esc(progressData.overall_completed || 0)}} / ${{esc(progressData.overall_total || 0)}} (${{fmtPct(progressData.overall_ratio)}})</div>`;
  }}
  const progressHolder = document.getElementById('metrics_progress_matrix');
  if(progressHolder) {{
    const progressScenes = progressData.scenes || [];
    const progressRows = progressData.rows || [];
    const head = progressScenes.map((scene)=>`<th>${{esc(scene.scene_id)}}</th>`).join('');
    const totalCols = 2 + progressScenes.length + 1;
    progressHolder.innerHTML = `<table class="compact-table"><thead><tr><th>Model</th>${{head}}<th>汇总</th></tr></thead><tbody>${{
      progressRows.map((row)=>`<tr><td>${{esc(row.model)}}</td>${{progressScenes.map((scene)=>{{ const cell = row.scenes?.[scene.scene_id] || {{}}; return `<td>${{esc(cell.completed ?? 0)}} / ${{esc(cell.total ?? 0)}} (${{fmtPct(cell.ratio)}})</td>`; }}).join('')}}<td>${{esc(row.total_completed ?? 0)}} / ${{esc(row.total_samples ?? 0)}} (${{fmtPct(row.total_ratio)}})</td></tr>`).join('') || `<tr><td colspan="${{totalCols}}" class="muted">暂无实验进度</td></tr>`
    }}</tbody></table>`;
  }}
  renderFooter();
}}

function exportMetricsCsv() {{
  const latestOnly = document.getElementById('metrics_latest_only')?.checked ? '1' : '0';
  const byDifficulty = document.getElementById('metrics_by_difficulty')?.checked ? '1' : '0';
  const params = new URLSearchParams();
  params.set('engine', state.engine || '');
  params.set('scene_id', state.scene_id || '');
  if (state.taskName) params.set('task_name', state.taskName);
  params.set('latest_only', latestOnly);
  params.set('by_difficulty', byDifficulty);
  window.location.href = `/api/metrics_matrix_csv?${{params.toString()}}`;
}}
setInterval(()=>{{ if(ACTIVE_PAGE==='experiments') refreshJobs(); }}, 4000);
initTheme();
loadCatalog().then(refreshAll);
</script>
</body>
</html>
"""

    @app.get("/")
    @app.get("/generate")
    @app.get("/dataset")
    @app.get("/experiments")
    @app.get("/results")
    @app.get("/metrics")
    def page() -> Any:
        active = request.path.strip("/") or "generate"
        return _render_shell(active)

    @app.get("/api/catalog")
    def api_catalog() -> Any:
        return jsonify(catalog)

    @app.get("/api/task_pipeline_tasks")
    def api_task_pipeline_tasks() -> Any:
        return jsonify({"tasks": list_task_pipeline_tasks(workspace_root=WORKSPACE_ROOT)})

    @app.get("/api/state")
    def api_state() -> Any:
        selected_engine = str(request.args.get("engine", "") or default_engine)
        selected_scene_id = str(request.args.get("scene_id", "") or default_scene_id)
        return jsonify(_scene_state(selected_engine, selected_scene_id, request.args.get("task_name")))

    @app.get("/api/manifests")
    def api_manifests() -> Any:
        selected_engine = str(request.args.get("engine", "") or default_engine)
        selected_scene_id = str(request.args.get("scene_id", "") or default_scene_id)
        if _is_global_scene_id(selected_scene_id):
            payload = _global_stage4_payload(
                catalog=catalog,
                selected_engine=selected_engine,
                task_name=request.args.get("task_name"),
                fallback_config_path=config_path,
            )
            return jsonify(payload.get("global_manifests", []))
        cfg, _, scene_root, stage4_root = _load_scene_context(selected_engine, selected_scene_id)
        roots = _stage4_roots(cfg, selected_scene_id, selected_engine, request.args.get("task_name"), scene_root)
        return jsonify(_list_manifests_multi(roots, selected_scene_id))

    @app.get("/api/manifest")
    def api_manifest() -> Any:
        raw_path = str(request.args.get("path", "") or "").strip()
        if not raw_path:
            return jsonify({"error": "missing_path"}), 400
        path = (WORKSPACE_ROOT / raw_path).resolve()
        if not path.exists():
            return jsonify({"error": "manifest_not_found"}), 404
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            samples = []
            for row in list(payload.get("samples", []) or []):
                if not isinstance(row, dict):
                    continue
                sample = dict(row)
                sample["prompt_text"] = _build_prompt(sample)
                sample["user_prompt"] = str(sample.get("user_prompt", "") or sample.get("prompt_text", "") or "")
                sample["system_prompt"] = str(sample.get("system_prompt", "") or _build_system_prompt(sample))
                samples.append(sample)
            payload["samples"] = samples
            payload["summary"] = _summarize_manifest_payload(payload)
        return jsonify(payload)

    @app.post("/api/generate")
    def api_generate() -> Any:
        payload = request.get_json(silent=True) or {}
        selected_engine = str(payload.get("engine", default_engine) or default_engine)
        selected_scene_id = str(payload.get("scene_id", default_scene_id) or default_scene_id)
        if _is_global_scene_id(selected_scene_id):
            return jsonify({"error": "global_scene_mode_is_read_only"}), 400
        cfg, _, _, _ = _load_scene_context(selected_engine, selected_scene_id)
        task_name = str(payload.get("task_name", "") or "").strip()
        if task_name:
            cfg = dict(cfg)
            cfg["task_pipeline"] = {"task_name": task_name, "root_dir": "task_pipeline_data"}
        result = generate_manifest(
            config=cfg,
            scene_id=selected_scene_id,
            engine=selected_engine,
            sample_count=max(1, int(payload.get("sample_count", 5) or 5)),
            seed=int(payload.get("seed", 7) or 7),
            reference_main_only=bool(payload.get("reference_main_only", True)),
            difficulties=list(payload.get("difficulties", ["4way", "8way"]) or ["4way", "8way"]),
            view_definitions=list(payload.get("view_definitions", []) or []),
            task_types=list(payload.get("task_types", []) or []),
            landmark_categories=list(payload.get("landmark_categories", []) or []),
        )
        return jsonify(
            {
                "manifest_path": _path_for_json(Path(result["manifest_path"])),
                "sample_count": int(result["manifest"]["sample_count"]),
                "generation_id": result["manifest"]["generation_id"],
                "selected_landmark_categories": list(result["manifest"].get("selected_landmark_categories", []) or []),
            }
        )

    @app.get("/api/reports")
    def api_reports() -> Any:
        selected_engine = str(request.args.get("engine", "") or default_engine)
        selected_scene_id = str(request.args.get("scene_id", "") or default_scene_id)
        if _is_global_scene_id(selected_scene_id):
            payload = _global_stage4_payload(
                catalog=catalog,
                selected_engine=selected_engine,
                task_name=request.args.get("task_name"),
                fallback_config_path=config_path,
            )
            return jsonify(payload.get("global_reports", []))
        cfg, _, scene_root, stage4_root = _load_scene_context(selected_engine, selected_scene_id)
        roots = _stage4_roots(cfg, selected_scene_id, selected_engine, request.args.get("task_name"), scene_root)
        return jsonify(_list_reports_multi(roots, selected_scene_id))

    @app.get("/api/report")
    def api_report() -> Any:
        raw_path = str(request.args.get("path", "") or "").strip()
        if not raw_path:
            return jsonify({"error": "missing_path"}), 400
        path = (WORKSPACE_ROOT / raw_path).resolve()
        if not path.exists():
            return jsonify({"error": "report_not_found"}), 404
        payload = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(
            {
                **payload,
                "requests_txt_path": _path_for_json(path.parent / "requests.txt") if (path.parent / "requests.txt").exists() else "",
                "responses_txt_path": _path_for_json(path.parent / "responses.txt") if (path.parent / "responses.txt").exists() else "",
            }
        )

    @app.get("/api/report_rows")
    def api_report_rows() -> Any:
        raw_path = str(request.args.get("path", "") or "").strip()
        if not raw_path:
            return jsonify([])
        path = (WORKSPACE_ROOT / raw_path).resolve()
        if not path.exists():
            return jsonify([])
        return jsonify(_load_report_rows_from_report_path(path)[:500])

    @app.get("/api/metrics_matrix")
    def api_metrics_matrix() -> Any:
        selected_engine = str(request.args.get("engine", "") or default_engine)
        selected_scene_id = str(request.args.get("scene_id", "") or default_scene_id)
        latest_only_text = str(request.args.get("latest_only", "0") or "0").strip().lower()
        latest_only = latest_only_text in {"1", "true", "yes", "on"}
        by_difficulty = str(request.args.get("by_difficulty", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if _is_global_scene_id(selected_scene_id):
            payload = _global_stage4_payload(
                catalog=catalog,
                selected_engine=selected_engine,
                task_name=request.args.get("task_name"),
                fallback_config_path=config_path,
            )
            return jsonify(_build_metrics_matrix_from_reports(list(payload.get("global_reports", []) or []), latest_only=latest_only, by_difficulty=by_difficulty))
        cfg, _, scene_root, stage4_root = _load_scene_context(selected_engine, selected_scene_id)
        roots = _stage4_roots(cfg, selected_scene_id, selected_engine, request.args.get("task_name"), scene_root)
        reports = _list_reports_multi(roots, selected_scene_id)
        # Reuse existing builder by materializing merged root preference order.
        return jsonify(_build_metrics_matrix_from_reports(reports, latest_only=latest_only, by_difficulty=by_difficulty))

    @app.get("/api/metrics_matrix_csv")
    def api_metrics_matrix_csv() -> Any:
        selected_engine = str(request.args.get("engine", "") or default_engine)
        selected_scene_id = str(request.args.get("scene_id", "") or default_scene_id)
        latest_only_text = str(request.args.get("latest_only", "0") or "0").strip().lower()
        latest_only = latest_only_text in {"1", "true", "yes", "on"}
        by_difficulty = str(request.args.get("by_difficulty", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if _is_global_scene_id(selected_scene_id):
            payload = _global_stage4_payload(
                catalog=catalog,
                selected_engine=selected_engine,
                task_name=request.args.get("task_name"),
                fallback_config_path=config_path,
            )
            matrix = _build_metrics_matrix_from_reports(list(payload.get("global_reports", []) or []), latest_only=latest_only, by_difficulty=by_difficulty)
        else:
            cfg, _, scene_root, _stage4_root = _load_scene_context(selected_engine, selected_scene_id)
            roots = _stage4_roots(cfg, selected_scene_id, selected_engine, request.args.get("task_name"), scene_root)
            reports = _list_reports_multi(roots, selected_scene_id)
            matrix = _build_metrics_matrix_from_reports(reports, latest_only=latest_only, by_difficulty=by_difficulty)
        fieldnames, rows = _stage4_metrics_matrix_csv_rows(matrix)
        filename = f"stage4_metrics_{selected_scene_id or 'scene'}_{'latest' if latest_only else 'all'}_{'difficulty' if by_difficulty else 'summary'}.csv"
        return _csv_text(fieldnames, rows), 200, {
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": f'attachment; filename="{filename}"',
        }

    @app.get("/api/experiment_progress_matrix")
    def api_experiment_progress_matrix() -> Any:
        selected_engine = str(request.args.get("engine", "") or default_engine)
        selected_scene_id = str(request.args.get("scene_id", "") or default_scene_id)
        return jsonify(_build_stage4_experiment_progress_matrix(catalog=catalog, selected_engine=selected_engine, selected_scene_id=selected_scene_id, task_name=request.args.get("task_name"), fallback_config_path=config_path))

    def _job_worker(job_id: str) -> None:
        job = job_manager.get(job_id)
        if not job:
            return
        payload = dict(job.get("payload", {}) or {})
        cancel_event = job_manager._jobs[job_id]["cancel_event"]  # type: ignore[index]
        selected_engine = str(payload.get("engine", default_engine) or default_engine)
        selected_scene_id = str(payload.get("scene_id", default_scene_id) or default_scene_id)
        cfg, _, _, _ = _load_scene_context(selected_engine, selected_scene_id)
        manifest_path = (WORKSPACE_ROOT / str(payload.get("manifest_path", "") or "")).resolve()
        if not manifest_path.exists():
            job_manager.update(job_id, status="error", finished_at=_iso_now())
            job_manager.append_log(job_id, "manifest_not_found")
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = _filter_samples(
            list(manifest.get("samples", []) or []),
            view_definitions=list(payload.get("view_definitions", []) or []),
            task_types=list(payload.get("task_types", []) or []),
            difficulties=list(payload.get("difficulties", []) or []),
            limit=int(payload.get("limit", 0) or 0) if int(payload.get("limit", 0) or 0) > 0 else None,
        )
        model_name = str(payload.get("model", "") or "").strip()
        if not model_name:
            models = list(payload.get("models", []) or [])
            if models:
                model_name = str(models[0] or "").strip()
        if not model_name:
            try:
                model_name = str(_resolve_api_settings(cfg)["model"] or "").strip()
            except Exception:
                model_name = ""
        if not model_name:
            job_manager.update(job_id, status="error", finished_at=_iso_now())
            job_manager.append_log(job_id, "no_models_selected")
            return
        total = len(samples)
        job_manager.update(job_id, status="running", started_at=_iso_now())
        job_manager.update_progress(job_id, {"completed": 0, "total": total, "current_model": model_name, "current_sample_id": None})
        try:
            if cancel_event.is_set():
                raise CancelledExperimentError("cancelled")
            job_manager.append_log(job_id, f"start model={model_name}")
            overrides = {
                "upload_max_width": int(payload.get("upload_max_width", 640) or 640),
                "upload_max_height": int(payload.get("upload_max_height", 480) or 480),
                "upload_jpeg_quality": int(payload.get("upload_jpeg_quality", 85) or 85),
                "timeout_s": float(payload.get("timeout_s", 30) or 30),
                "rpm_limit": int(payload.get("rpm_limit", 0) or 0),
                "tpm_limit": int(payload.get("tpm_limit", 0) or 0),
                "concurrency": int(payload.get("concurrency", 1) or 1),
            }
            def _progress(info: dict[str, Any]) -> None:
                job_manager.update_progress(
                    job_id,
                    {
                        "completed": int(info.get("completed", 0)),
                        "total": total,
                        "current_model": model_name,
                        "current_sample_id": info.get("sample_id"),
                        "current_task_type": info.get("task_type"),
                        "request_status": info.get("request_status"),
                        "parse_ok": info.get("parse_ok"),
                        "latency_ms": info.get("latency_ms"),
                    },
                )
            result = run_experiment_once(
                config=cfg,
                scene_id=selected_scene_id,
                engine=selected_engine,
                manifest_path=manifest_path,
                model=model_name,
                samples=samples,
                api_overrides=overrides,
                cancel_event=cancel_event,
                progress_callback=_progress,
            )
            job_manager.add_run(
                job_id,
                {
                    "run_id": result["run_id"],
                    "report_path": _path_for_json(Path(result["report_path"])),
                    "model": result["report"]["model"],
                    "manifest_path": _path_for_json(manifest_path),
                    "manifest_name": manifest_path.name,
                    "summary": result["report"]["summary"],
                },
            )
            job_manager.append_log(job_id, f"finished model={model_name}")
            status = "cancelled" if cancel_event.is_set() else "completed"
            job_manager.update(job_id, status=status, finished_at=_iso_now())
        except CancelledExperimentError:
            job_manager.update(job_id, status="cancelled", finished_at=_iso_now())
            job_manager.append_log(job_id, "cancelled")
        except Exception as exc:
            job_manager.update(job_id, status="error", finished_at=_iso_now(), error=str(exc))
            job_manager.append_log(job_id, f"error: {exc}")

    @app.post("/api/jobs/start")
    def api_job_start() -> Any:
        payload = request.get_json(silent=True) or {}
        if _is_global_scene_id(payload.get("scene_id")):
            return jsonify({"error": "global_scene_mode_is_read_only"}), 400
        manifest_path = (WORKSPACE_ROOT / str(payload.get("manifest_path", "") or "")).resolve()
        manifest_name = manifest_path.name if manifest_path.exists() else str(payload.get("manifest_path", "") or "").strip()
        models = [str(x).strip() for x in list(payload.get("models", []) or []) if str(x).strip()]
        if not models:
            models = [""]
        jobs = []
        for model_name in models:
            job_payload = dict(payload)
            job_payload["manifest_name"] = manifest_name
            job_payload["model"] = model_name
            job_payload["models"] = [model_name] if model_name else []
            job_id = job_manager.create_job(job_payload)
            threading.Thread(target=_job_worker, args=(job_id,), daemon=True).start()
            job = job_manager.get(job_id)
            if job:
                jobs.append(job)
        return jsonify({"jobs": jobs, "job_ids": [job["job_id"] for job in jobs]})

    @app.get("/api/jobs")
    def api_jobs() -> Any:
        return jsonify(job_manager.list())

    @app.get("/api/jobs/<job_id>")
    def api_job_detail(job_id: str) -> Any:
        job = job_manager.get(job_id)
        if job is None:
            return jsonify({"error": "job_not_found"}), 404
        return jsonify(job)

    @app.post("/api/jobs/<job_id>/cancel")
    def api_job_cancel(job_id: str) -> Any:
        if not job_manager.cancel(job_id):
            return jsonify({"error": "job_not_found"}), 404
        return jsonify(job_manager.get(job_id))

    @app.get("/api/file")
    def api_file() -> Any:
        raw_path = str(request.args.get("path", "") or "").strip()
        if not raw_path:
            return jsonify({"error": "missing_path"}), 400
        path = (WORKSPACE_ROOT / raw_path).resolve()
        if not path.exists() or not path.is_file():
            return jsonify({"error": "file_not_found"}), 404
        resize_enabled = str(request.args.get("resize", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if resize_enabled:
            prepared = _prepare_web_image(
                path,
                resize_enabled=True,
                max_width=int(request.args.get("w", "640") or 640),
                max_height=int(request.args.get("h", "480") or 480),
                jpeg_quality=int(request.args.get("q", "80") or 80),
            )
            if prepared is not None:
                payload, mime_type = prepared
                return send_file(io.BytesIO(payload), mimetype=mime_type, download_name=f"{path.stem}.jpg")
        return send_file(path)

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4-1 QA generation and evaluation")
    parser.add_argument("--config", required=True, help="task config yaml")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--engine", default="airsim")
    parser.add_argument("--mode", default="all", choices=["generate", "experiment", "web", "all"])
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reference-main-only", action="store_true", default=True)
    parser.add_argument("--allow-non-main-reference", action="store_true", default=False)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--port", type=int, default=20264)
    parser.add_argument("--landmark-category", action="append", default=[], help="repeatable landmark category filter for generation")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_yaml(Path(args.config).resolve())
    scene_id = str(args.scene_id)
    engine = str(args.engine or "airsim").strip().lower()
    reference_main_only = bool(args.reference_main_only and not args.allow_non_main_reference)

    manifest_path: Path | None = Path(args.manifest_path).resolve() if str(args.manifest_path or "").strip() else None
    generated: dict[str, Any] | None = None
    if args.mode in {"generate", "all"}:
        generated = generate_manifest(
            config=config,
            scene_id=scene_id,
            engine=engine,
            sample_count=max(1, int(args.sample_count)),
            seed=int(args.seed),
            reference_main_only=reference_main_only,
            difficulties=["4way", "8way"],
            landmark_categories=list(args.landmark_category or []),
        )
        manifest_path = Path(generated["manifest_path"]).resolve()
        print(
            json.dumps(
                {
                    "mode": "generate",
                    "scene_id": scene_id,
                    "engine": engine,
                    "manifest_path": str(manifest_path),
                    "sample_count": int(generated["manifest"]["sample_count"]),
                },
                ensure_ascii=False,
            )
        )

    experiment_result: dict[str, Any] | None = None
    if args.mode in {"experiment", "all"}:
        if manifest_path is None:
            scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
            stage4_root = _resolve_stage4_root(config, scene_root=scene_root)
            manifest_path, _ = _load_latest_manifest(stage4_root, scene_id)
        if manifest_path is None:
            raise RuntimeError("missing_manifest_for_experiment")
        experiment_result = run_experiment(
            config=config,
            scene_id=scene_id,
            engine=engine,
            manifest_path=manifest_path,
            override_model=str(args.model or "").strip() or None,
            limit=int(args.limit) if int(args.limit) > 0 else None,
        )
        print(
            json.dumps(
                {
                    "mode": "experiment",
                    "scene_id": scene_id,
                    "engine": engine,
                    "run_id": experiment_result["run_id"],
                    "report_path": str(experiment_result["report_path"]),
                    "summary": experiment_result["report"]["summary"],
                },
                ensure_ascii=False,
            )
        )

    if args.mode == "web":
        app = _make_web_app(config, scene_id=scene_id, engine=engine, config_path=Path(args.config).resolve())
        print(json.dumps({"mode": "web", "scene_id": scene_id, "engine": engine, "port": int(args.port)}, ensure_ascii=False))
        app.run(host="0.0.0.0", port=int(args.port), debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
