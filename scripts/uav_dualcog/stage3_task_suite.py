from __future__ import annotations

import base64
import csv
import io
import json
import math
import random
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from flask import jsonify, redirect, render_template_string, request, send_file
except Exception:
    jsonify = None
    redirect = None
    render_template_string = None
    request = None
    send_file = None

try:
    import yaml
except Exception:
    yaml = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from pipeline_common import (
    list_task_pipeline_tasks,
    resolve_output_dir_name,
    resolve_scene_root,
    resolve_task_pipeline_scene_root,
)
from media_path_utils import resolve_existing_file_with_suffix_fallback
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
from api_common import (
    build_model_request_controls,
    compute_rate_limited_concurrency,
    max_data_uri_video_bytes_for_model,
    merge_common_stage_block,
    required_video_placeholder_for_model,
    resolve_default_model,
    should_inline_system_prompt_for_multimodal,
)
from image_compression_utils import compression_cfg as build_image_compression_cfg
from image_compression_utils import preferred_output_path, save_pil_image
from prompt_templates import get_config_template, get_prompt_template, render_prompt_template


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
COMMON_STAGE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "uav_dualcog" / "common_stage_configs.yaml"
GLOBAL_SCENE_ID = "__all__"
GLOBAL_SCENE_LABEL = "ALL scenes"
TASK_SPECS: dict[str, dict[str, Any]] = {
    "self_instance_recognition_joint": {"task_group": "self-state", "display_name": "联合行为识别", "response_kind": "choice_with_optional_intervals", "multi_select": True},
    "self_set_instance_recognition": {"task_group": "self-state", "display_name": "Composite 实例识别", "response_kind": "choice_with_optional_intervals", "multi_select": False},
    "self_element_instance_recognition": {"task_group": "self-state", "display_name": "Atomic 实例识别", "response_kind": "choice_with_optional_intervals", "multi_select": True},
    "self_composite_instance_recognition": {"task_group": "self-state", "display_name": "Composite 实例识别", "response_kind": "choice_with_optional_intervals", "multi_select": False},
    "self_atomic_instance_recognition": {"task_group": "self-state", "display_name": "Atomic 实例识别", "response_kind": "choice_with_optional_intervals", "multi_select": True},
    "env_visibility_reasoning": {"task_group": "environmental", "display_name": "环境感知", "response_kind": "count_and_intervals"},
}
MODE_CHOICES = ["single-landmark", "multi-landmark"]


def _is_global_scene_id(scene_id: str | None) -> bool:
    text = str(scene_id or "").strip().lower()
    return text in {GLOBAL_SCENE_ID, "all", "*"}
_BEHAVIOR_LABELS_EN: dict[str, str] = {
    "gradual_approach": "Gradual Approach",
    "gradual_depart": "Gradual Depart",
    "circular_orbit": "Circular Orbit",
    "spiral_orbit": "Spiral Orbit",
    "square_orbit": "Square Orbit",
    "triangular_orbit": "Triangular Orbit",
    "figure8_orbit": "Figure-8 Orbit",
    "sky_rise": "Sky Rise",
    "comet": "Comet",
}
_CAMERA_MODE_LABELS_EN: dict[str, str] = {
    "landmark_track": "Landmark-Track",
    "look_forward": "Look-Forward",
    "track_target": "Landmark-Track",
    "velocity_aligned": "Look-Forward",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_mp4_web_compatible(path: Path) -> bool:
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin is None or not path.exists() or path.suffix.lower() != ".mp4":
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


def _make_mp4_web_compatible(path: Path, bitrate: str | None = None) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None or not path.exists() or path.suffix.lower() != ".mp4":
        return
    tmp_path = path.with_name(f"{path.stem}.webtmp.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
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
        elif tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _ensure_mp4_web_playable(path: Path) -> None:
    if path.suffix.lower() != ".mp4" or not path.exists():
        return
    if _is_mp4_web_compatible(path):
        return
    target_bitrate = "1M" if str(path.name).endswith("_web.mp4") else ""
    _make_mp4_web_compatible(path, bitrate=target_bitrate)


def _compress_video_for_data_uri_limit(path: Path, *, target_bytes: int, duration_sec: float) -> Path:
    if not path.exists() or path.suffix.lower() != ".mp4" or target_bytes <= 0:
        return path
    if path.stat().st_size <= target_bytes:
        return path
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return path

    cache_path = path.with_name(f"{path.stem}.upload_10mb.mp4")
    if cache_path.exists() and cache_path.stat().st_size <= target_bytes:
        return cache_path

    safe_duration = max(1.0, float(duration_sec or 0.0))
    target_bitrate_kbps = max(220, min(1200, int((target_bytes * 8.0 / safe_duration) / 1000.0 * 0.88)))
    trial_widths = [None, 576, 512, 448, 384]
    trial_bitrates = [
        target_bitrate_kbps,
        int(target_bitrate_kbps * 0.85),
        int(target_bitrate_kbps * 0.72),
        420,
        320,
        260,
    ]

    for width in trial_widths:
        for bitrate_kbps in trial_bitrates:
            tmp_path = cache_path.with_suffix(".tmp.mp4")
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
                "-b:v",
                f"{max(180, int(bitrate_kbps))}k",
                "-maxrate",
                f"{max(180, int(bitrate_kbps))}k",
                "-bufsize",
                f"{max(360, int(bitrate_kbps) * 2)}k",
                "-an",
            ]
            if width is not None:
                cmd.extend(["-vf", f"scale={int(width)}:-2"])
            cmd.append(str(tmp_path))
            try:
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300, check=False)
                if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                    tmp_path.replace(cache_path)
                    if cache_path.stat().st_size <= target_bytes:
                        return cache_path
                elif tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                continue
    return cache_path if cache_path.exists() and cache_path.stat().st_size <= target_bytes else path


def _prepare_upload_video(path: Path, sample: dict[str, Any], api_cfg: dict[str, Any]) -> tuple[bytes, Path]:
    route_model_name = str(api_cfg.get("base_model", "") or api_cfg.get("model", "") or "")
    max_video_bytes = int(max_data_uri_video_bytes_for_model(route_model_name) or 0)
    upload_path = path
    if max_video_bytes > 0:
        fps = float(sample.get("fps", 0.0) or 0.0)
        frame_count = int(sample.get("frame_count", 0) or 0)
        duration_sec = max(0.0, float(frame_count - 1) / fps) if fps > 0.0 and frame_count > 0 else 0.0
        upload_path = _compress_video_for_data_uri_limit(path, target_bytes=max_video_bytes, duration_sec=duration_sec)
    return upload_path.read_bytes(), upload_path


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"invalid_yaml_root: {path}")
    return data


def _load_common_stage_cfg() -> dict[str, Any]:
    if not COMMON_STAGE_CONFIG_PATH.exists():
        return {}
    try:
        return _load_yaml(COMMON_STAGE_CONFIG_PATH)
    except Exception:
        return {}


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(dict(out.get(key, {}) or {}), value)
        else:
            out[key] = value
    return out


def _stage3_cfg(config: dict[str, Any]) -> dict[str, Any]:
    common_cfg = dict((_load_common_stage_cfg().get("stage3_runtime_defaults", {}) or {}))
    scene_cfg = dict(config.get("stage3", {}) or config.get("trajectory", {}) or {})
    return _deep_merge_dict(common_cfg, scene_cfg)


def _stage3_temporal_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return merge_common_stage_block(config, "stage3_temporal")


def _resolve_stage3_layout(config: dict[str, Any], scene_id: str, engine: str) -> dict[str, Path]:
    scene_root = resolve_scene_root(config, scene_id=scene_id, engine=engine, workspace_root=WORKSPACE_ROOT)
    artifact_scene_root = resolve_task_pipeline_scene_root(
        config,
        scene_id=scene_id,
        engine=engine,
        workspace_root=WORKSPACE_ROOT,
    ) or scene_root
    task_root_name = resolve_output_dir_name(config, key="stage3_task_root_dir", default="stage3_tasks")
    root = artifact_scene_root / task_root_name
    return {
        "scene_root": scene_root,
        "artifacts_scene_root": artifact_scene_root,
        "stage3_root": root,
        "missions_root": root / resolve_output_dir_name(config, key="stage3_mission_dir", default="missions"),
        "review_root": root / resolve_output_dir_name(config, key="stage3_review_dir", default="review"),
        "datasets_root": root / resolve_output_dir_name(config, key="stage3_dataset_dir", default="datasets"),
        "experiments_root": root / resolve_output_dir_name(config, key="stage3_experiment_dir", default="experiments"),
        "reports_root": root / resolve_output_dir_name(config, key="stage3_report_dir", default="reports"),
        "cache_root": root / resolve_output_dir_name(config, key="stage3_cache_dir", default="cache"),
    }


def _with_task_pipeline_cfg(cfg: dict[str, Any], task_name: str | None) -> dict[str, Any]:
    tn = str(task_name or "").strip()
    if not tn:
        return cfg
    out = dict(cfg)
    out["task_pipeline"] = {"task_name": tn, "root_dir": "task_pipeline_data"}
    return out


def _load_manifest_sample_count(path: str | Path) -> int:
    manifest_path = Path(str(path))
    if not manifest_path.exists():
        return 0
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    if not isinstance(payload, dict):
        return 0
    raw = payload.get("sample_count")
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    return int(len(list(payload.get("samples", []) or [])))


def _candidate_review_files(layout: dict[str, Path]) -> tuple[Path, Path]:
    review_root = _ensure_dir(layout["review_root"])
    return review_root / "candidate_review_index.json", review_root / "candidate_review_log.jsonl"


def _load_candidate_review_index(path: Path) -> dict[str, Any]:
    payload = _read_json(path, default={})
    items = payload.get("items", {}) if isinstance(payload, dict) else {}
    if not isinstance(items, dict):
        items = {}
    return {"items": items}


def _load_frames_manifest(manifest_path: Path) -> list[Path]:
    payload = _read_json(manifest_path, default={})
    if not isinstance(payload, dict):
        return []
    out: list[Path] = []
    for row in list(payload.get("frames", []) or []):
        p = Path(str(row))
        cand = p if p.is_absolute() else manifest_path.parent / p
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        if resolved.exists():
            out.append(resolved)
    return out


def _resolve_path_near(base_dir: Path, raw_path: Any) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    cand = Path(text)
    if not cand.is_absolute():
        cand = base_dir / cand
    try:
        resolved = cand.resolve()
    except Exception:
        return None
    return resolved


def _resolve_workspace_json_path(raw_path: Any) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return Path("")
    path = Path(text)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / text
    return path.resolve()


def _bbox_xywh_to_xyxy_norm(bbox_xywh: list[Any] | tuple[Any, ...], width: int, height: int) -> list[float] | None:
    if not isinstance(bbox_xywh, (list, tuple)) or len(bbox_xywh) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox_xywh]
    except Exception:
        return None
    if width <= 1 or height <= 1:
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(x1 + 1.0, min(float(width), x + w))
    y2 = max(y1 + 1.0, min(float(height), y + h))
    return [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]


def _interval_iou(pred: list[float], gold: list[float]) -> float:
    if len(pred) != 2 or len(gold) != 2:
        return 0.0
    ps, pe = float(pred[0]), float(pred[1])
    gs, ge = float(gold[0]), float(gold[1])
    if pe < ps:
        ps, pe = pe, ps
    if ge < gs:
        gs, ge = ge, gs
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 1e-6 else 0.0


def _match_interval_metrics(gold: list[list[float]], pred: list[list[float]], threshold: float) -> tuple[int, int, int]:
    used_pred: set[int] = set()
    tp = 0
    for g in gold:
        best_idx = None
        best_iou = 0.0
        for idx, p in enumerate(pred):
            if idx in used_pred:
                continue
            tiou = _interval_iou(p, g)
            if tiou > best_iou:
                best_iou = tiou
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            used_pred.add(best_idx)
            tp += 1
    fp = max(0, len(pred) - tp)
    fn = max(0, len(gold) - tp)
    return tp, fp, fn


def _bbox_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def _fallback_behavior_intervals_from_segments(segments_payload: dict[str, Any], *, frame_count: int, fps: float) -> list[dict[str, Any]]:
    segments = list(segments_payload.get("segments", []) or [])
    if not segments or frame_count <= 0:
        return []
    raw_counts = [max(1, int(seg.get("num_points", 1) or 1)) for seg in segments]
    total_raw = max(1, sum(raw_counts))
    out = []
    cursor = 0
    for idx, (seg, count) in enumerate(zip(segments, raw_counts)):
        span = max(1, int(round(float(count) / float(total_raw) * float(frame_count))))
        start_frame = int(cursor)
        end_frame = int(frame_count - 1 if idx == len(raw_counts) - 1 else min(frame_count - 1, start_frame + span - 1))
        out.append(
            {
                "event_id": str(seg.get("event_id", seg.get("segment_id", f"seg_{idx:02d}")) or f"seg_{idx:02d}"),
                "event_label": str(seg.get("event_label", seg.get("behavior_id", seg.get("behavior", ""))) or ""),
                "behavior_id": str(seg.get("behavior_id", seg.get("behavior", "")) or ""),
                "intervals_sec": [{"start_sec": float(start_frame) / float(max(1.0, fps)), "end_sec": float(end_frame) / float(max(1.0, fps))}],
            }
        )
        cursor = end_frame + 1
    return out


def _difficulty_band(visible_count: int) -> str:
    if visible_count <= 1:
        return "1"
    if visible_count <= 3:
        return "2-3"
    if visible_count <= 5:
        return "4-5"
    return "6+"


def _stage3_include_overview_image(config: dict[str, Any]) -> bool:
    stage3_cfg = _stage3_cfg(config)
    return str(stage3_cfg.get("dataset_include_overview_image", False)).strip().lower() in {"1", "true", "yes", "on"}


def _stage3_include_keyframe_board_image(config: dict[str, Any]) -> bool:
    stage3_cfg = _stage3_cfg(config)
    return str(stage3_cfg.get("dataset_include_keyframe_board_image", False)).strip().lower() in {"1", "true", "yes", "on"}


def _landmark_category_text(item: dict[str, Any]) -> str:
    return str(item.get("landmark_category", "") or item.get("class_name", "") or "").strip()


def _landmark_subcategory_text(item: dict[str, Any]) -> str:
    return str(item.get("landmark_subcategory", "") or "").strip()


def _landmark_description_text(item: dict[str, Any]) -> str:
    text = str(item.get("landmark_description", "") or "").strip()
    if text:
        return text
    text = str(item.get("description", "") or "").strip()
    if text:
        return text
    return str(item.get("instance_id", "") or "").strip() or "landmark"


def _whole_second_keyframes(candidate: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    dense = list(candidate.get("keyframe_gt_dense", []) or [])
    by_frame = {int(row.get("frame", -1)): row for row in dense if isinstance(row, dict)}
    fps = float(candidate.get("fps", 24.0) or 24.0)
    frame_count = int(candidate.get("frame_count", 0) or 0)
    eval_cfg = dict((_stage3_cfg(config).get("keyframe_eval", {}) or {}))
    fallback = bool(eval_cfg.get("include_endpoint_fallback", True))
    max_per_interval = int(eval_cfg.get("max_frames_per_interval", 8) or 8)
    out: list[dict[str, Any]] = []
    for interval in list(candidate.get("visible_intervals_sec", []) or []):
        if not isinstance(interval, dict):
            continue
        start_sec = float(interval.get("start_sec", 0.0) or 0.0)
        end_sec = float(interval.get("end_sec", 0.0) or 0.0)
        selected_times = []
        sec = int(math.ceil(start_sec))
        while sec <= int(math.floor(end_sec)):
            selected_times.append(float(sec))
            sec += 1
        if not selected_times and fallback:
            selected_times.append((start_sec + end_sec) * 0.5)
        if max_per_interval > 0:
            selected_times = selected_times[:max_per_interval]
        for t in selected_times:
            frame_id = max(0, min(frame_count - 1, int(round(t * fps)))) if frame_count > 0 else 0
            row = by_frame.get(frame_id)
            if row is None and by_frame:
                nearest = min(by_frame.keys(), key=lambda fid: abs(fid - frame_id))
                row = by_frame.get(nearest)
            if row is None:
                continue
            out.append(
                {
                    "frame": int(row.get("frame", frame_id) or frame_id),
                    "time_sec": float(row.get("time_sec", t) or t),
                    "bbox_xyxy_norm": list(row.get("bbox_xyxy_norm", []) or []),
                }
            )
    dedup: dict[int, dict[str, Any]] = {}
    for row in out:
        dedup[int(row["frame"])] = row
    return [dedup[key] for key in sorted(dedup.keys())]


def _load_font(size: int) -> Any:
    if ImageFont is None:
        return None
    for name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_text(draw: Any, xy: tuple[int, int], text: str, *, fill: tuple[int, int, int], font: Any) -> None:
    try:
        draw.text(xy, text, fill=fill, font=font)
    except Exception:
        draw.text(xy, text, fill=fill)


def _fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        canvas = image.convert("RGB")
        canvas.thumbnail(size, Image.Resampling.LANCZOS)
        out = Image.new("RGB", size, (246, 242, 235))
        ox = (size[0] - canvas.width) // 2
        oy = (size[1] - canvas.height) // 2
        out.paste(canvas, (ox, oy))
        return out


def _build_storyboard_image(frame_paths: list[Path], out_path: Path, title: str, *, max_cells: int = 8) -> Path:
    if Image is None or ImageDraw is None:
        raise ImportError("Pillow is required")
    _ensure_dir(out_path.parent)
    if not frame_paths:
        raise RuntimeError("no_frames_for_storyboard")
    idxs = [int(round(v)) for v in np.linspace(0, len(frame_paths) - 1, num=min(max_cells, len(frame_paths)), endpoint=True)]
    selected = []
    seen: set[int] = set()
    for idx in idxs:
        idx = max(0, min(len(frame_paths) - 1, int(idx)))
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(idx)
    cols = min(4, max(1, len(selected)))
    rows = int(math.ceil(float(len(selected)) / float(cols)))
    cell_w, cell_h = 280, 180
    pad, title_h = 16, 36
    canvas = Image.new("RGB", (pad * 2 + cols * cell_w + (cols - 1) * pad, title_h + rows * (cell_h + pad) + pad), (250, 246, 238))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(18)
    _draw_text(draw, (pad, 10), title, fill=(20, 34, 45), font=font_title)
    for pos, idx in enumerate(selected):
        row = pos // cols
        col = pos % cols
        x0 = pad + col * (cell_w + pad)
        y0 = title_h + row * (cell_h + pad)
        thumb = _fit_image(frame_paths[idx], (cell_w, cell_h))
        canvas.paste(thumb, (x0, y0))
    image_cfg = build_image_compression_cfg(_load_common_stage_cfg().get("stage3_runtime_defaults", {}) or {})
    out_path = preferred_output_path(out_path, compress_enabled=bool(image_cfg.get("enabled", True)))
    save_pil_image(canvas, out_path, cfg=image_cfg)
    return out_path


def _build_keyframe_board_image(frame_paths: list[Path], keyframes: list[dict[str, Any]], out_path: Path, *, fps: float) -> Path:
    if Image is None or ImageDraw is None:
        raise ImportError("Pillow is required")
    _ensure_dir(out_path.parent)
    if not frame_paths or not keyframes:
        raise RuntimeError("no_keyframes_for_board")
    cols = min(4, max(1, len(keyframes)))
    rows = int(math.ceil(float(len(keyframes)) / float(cols)))
    cell_w, cell_h = 280, 170
    pad, title_h = 16, 36
    canvas = Image.new("RGB", (pad * 2 + cols * cell_w + (cols - 1) * pad, title_h + rows * (cell_h + 34 + pad) + pad), (250, 246, 238))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(18)
    font_small = _load_font(13)
    _draw_text(draw, (pad, 10), "Whole-second keyframe evaluation view", fill=(20, 34, 45), font=font_title)
    for pos, row in enumerate(keyframes):
        frame_id = int(row.get("frame", 0) or 0)
        if frame_id < 0 or frame_id >= len(frame_paths):
            continue
        r = pos // cols
        c = pos % cols
        x0 = pad + c * (cell_w + pad)
        y0 = title_h + r * (cell_h + 34 + pad)
        thumb = _fit_image(frame_paths[frame_id], (cell_w, cell_h))
        canvas.paste(thumb, (x0, y0 + 24))
        bbox = list(row.get("bbox_xyxy_norm", []) or [])
        if len(bbox) == 4:
            overlay = ImageDraw.Draw(canvas)
            bx1 = x0 + int(float(bbox[0]) * cell_w)
            by1 = y0 + 24 + int(float(bbox[1]) * cell_h)
            bx2 = x0 + int(float(bbox[2]) * cell_w)
            by2 = y0 + 24 + int(float(bbox[3]) * cell_h)
            overlay.rectangle([bx1, by1, bx2, by2], outline=(220, 62, 54), width=4)
        _draw_text(draw, (x0 + 4, y0), f"Frame {frame_id} | {float(row.get('time_sec', frame_id / max(1.0, fps))):.2f}s", fill=(70, 78, 84), font=font_small)
    image_cfg = build_image_compression_cfg(_load_common_stage_cfg().get("stage3_runtime_defaults", {}) or {})
    out_path = preferred_output_path(out_path, compress_enabled=bool(image_cfg.get("enabled", True)))
    save_pil_image(canvas, out_path, cfg=image_cfg)
    return out_path


def _resolve_ref_bbox_xyxy_and_norm(view: dict[str, Any]) -> tuple[list[int] | None, list[float] | None]:
    bbox_xyxy = None
    if isinstance(view.get("bbox_xyxy_px"), list) and len(view.get("bbox_xyxy_px", [])) == 4:
        bbox_xyxy = [int(v) for v in list(view.get("bbox_xyxy_px", []))]
    elif isinstance(view.get("bbox_2d_xyxy"), list) and len(view.get("bbox_2d_xyxy", [])) == 4:
        bbox_xyxy = [int(v) for v in list(view.get("bbox_2d_xyxy", []))]
    bbox_norm = None
    if isinstance(view.get("bbox_xyxy_norm"), list) and len(view.get("bbox_xyxy_norm", [])) == 4:
        bbox_norm = [float(v) for v in list(view.get("bbox_xyxy_norm", []))]
    elif bbox_xyxy is not None:
        image_size = view.get("bbox_2d_image_size", None) or view.get("image_size", None)
        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            bbox_norm = _bbox_xywh_to_xyxy_norm(
                [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]],
                int(image_size[0]),
                int(image_size[1]),
            )
    return bbox_xyxy, bbox_norm


def _resolve_reference_view_asset(
    *,
    config: dict[str, Any],
    layout: dict[str, Path],
    scene_id: str,
    instance_entry: dict[str, Any],
    assets_root: Path,
    mission_id: str,
) -> dict[str, Any] | None:
    views = list(instance_entry.get("rgb_views", []) or [])
    ref_view = next((view for view in views if bool(view.get("is_query_view", False))), None)
    if ref_view is None:
        ref_view = views[0] if views else None
    if ref_view is None:
        return None
    review_dir_name = resolve_output_dir_name(config, key="stage2_raw_dir", default="landmarks_raw")
    ref_path = resolve_existing_file_with_suffix_fallback(
        str(ref_view.get("path", "") or ""),
        base_dirs=[
            layout["scene_root"] / review_dir_name,
            layout["scene_root"],
        ],
    )
    if ref_path is None:
        return None
    ref_bbox_xyxy, ref_bbox_norm = _resolve_ref_bbox_xyxy_and_norm(ref_view)
    if ref_bbox_xyxy is None:
        return None
    instance_id = str(instance_entry.get("instance_id", "") or "").strip()
    image_cfg = build_image_compression_cfg(_stage3_cfg(config))
    ref_overlay = preferred_output_path(
        assets_root / "reference_bbox" / instance_id / f"{scene_id}_{mission_id}_ref.jpg",
        compress_enabled=bool(image_cfg.get("enabled", True)),
    )
    if not ref_overlay.exists():
        ref_overlay = _draw_reference_bbox(source_image=ref_path, bbox_xyxy=ref_bbox_xyxy, output_path=ref_overlay, cfg=image_cfg)
    return {
        "instance_id": instance_id,
        "reference_image": _path_for_json(ref_path),
        "reference_image_with_bbox": _path_for_json(ref_overlay),
        "reference_bbox_xyxy_norm": list(ref_bbox_norm or []),
    }


def _discover_candidates(
    config: dict[str, Any],
    *,
    scene_id: str,
    engine: str,
    require_final_task: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    layout = _resolve_stage3_layout(config=config, scene_id=scene_id, engine=engine)
    candidates: list[dict[str, Any]] = []
    art_root = layout["artifacts_scene_root"]
    base_root = layout["scene_root"]
    try:
        roots = [art_root, base_root] if art_root.resolve() != base_root.resolve() else [base_root]
    except Exception:
        roots = [art_root, base_root] if str(art_root) != str(base_root) else [base_root]
    stage2_entries: list[dict[str, Any]] | None = None
    last_missing: FileNotFoundError | None = None
    for root in roots:
        try:
            stage2_entries = _load_valid_instances(config, scene_root=root, scene_id=scene_id)
            break
        except FileNotFoundError as exc:
            last_missing = exc
            continue
    if stage2_entries is None:
        if last_missing is not None:
            raise last_missing
        stage2_entries = []
    stage2_map = {str(item.get("instance_id", "") or ""): item for item in stage2_entries}
    review_index_path, _ = _candidate_review_files(layout)
    review_items = _load_candidate_review_index(review_index_path).get("items", {})
    cache_root = _ensure_dir(layout["cache_root"])
    assets_root = _ensure_dir(cache_root / "assets")
    mission_dirs: list[Path] = sorted(layout["missions_root"].glob("*")) if layout["missions_root"].exists() else []

    def _load_one_candidate(mission_dir: Path) -> dict[str, Any] | None:
        if not mission_dir.is_dir():
            return None
        final_meta_path = mission_dir / "final_task" / "task_data.json"
        constraint_path = mission_dir / "constraint_report.json"
        segments_path = mission_dir / "composed_segments.json"
        if not constraint_path.exists():
            return None
        has_final_task = final_meta_path.exists()
        if require_final_task and not has_final_task:
            return None
        final_meta = _read_json(final_meta_path, default={}) if has_final_task else {}
        constraint = _read_json(constraint_path, default={})
        segments = _read_json(segments_path, default={})
        if not isinstance(constraint, dict):
            return None
        if has_final_task and not isinstance(final_meta, dict):
            return None
        instance_id = str(constraint.get("instance_id", "") or "").strip()
        stage2_entry = stage2_map.get(instance_id)
        if not stage2_entry:
            return None
        secondary_instance_ids = [str(x).strip() for x in list(constraint.get("secondary_instance_ids", []) or []) if str(x).strip()]
        if not secondary_instance_ids:
            secondary_instance_ids = [
                str((row or {}).get("instance_id", "") or "").strip()
                for row in list(constraint.get("secondary_landmarks", []) or [])
                if str((row or {}).get("instance_id", "") or "").strip()
            ]
        all_landmark_ids = [instance_id, *secondary_instance_ids]
        landmark_entries = [stage2_map[item_id] for item_id in all_landmark_ids if item_id in stage2_map]
        video = dict(final_meta.get("video", {}) or {}) if has_final_task else {}
        generated_without_video = bool(video.get("generated_without_video", False)) if has_final_task else False
        presence = dict(final_meta.get("target_presence", {}) or {}) if has_final_task else {}
        target_presence_targets = dict(final_meta.get("target_presence_targets", {}) or {}) if has_final_task else {}
        frame_manifest_path = _resolve_path_near(final_meta_path.parent, video.get("frames_manifest", ""))
        frame_paths = _load_frames_manifest(frame_manifest_path) if frame_manifest_path and frame_manifest_path.exists() else []
        frame_count = int(video.get("frame_count", len(frame_paths)) or len(frame_paths))
        fps = float(video.get("fps", 24.0) or 24.0)
        width = int(video.get("width", 640) or 640)
        height = int(video.get("height", 480) or 480)
        intervals_sec = list(presence.get("intervals_sec", []) or [])
        dense = list(presence.get("keyframe_gt_dense", []) or [])
        if not dense:
            dense = []
            for row in list(presence.get("bboxes", []) or []):
                if not isinstance(row, dict):
                    continue
                bbox_xywh = list(row.get("bbox_xywh", []) or [])
                bbox_norm = list(row.get("bbox_xyxy_norm", []) or []) or _bbox_xywh_to_xyxy_norm(bbox_xywh, width, height)
                if not bbox_norm:
                    continue
                dense.append(
                    {
                        "frame": int(row.get("frame", 0) or 0),
                        "time_sec": float(row.get("time_sec", 0.0) or 0.0),
                        "bbox_xyxy_norm": bbox_norm,
                    }
                )
        if not dense:
            dense_by_frame = {}
            for row in list(presence.get("frame_view_annotations", []) or []):
                if not isinstance(row, dict):
                    continue
                frame_id = int(row.get("frame", -1) or -1)
                bbox_norm = list(row.get("bbox_xyxy_norm", []) or [])
                if frame_id < 0 or len(bbox_norm) != 4:
                    continue
                dense_by_frame[frame_id] = {
                    "frame": frame_id,
                    "time_sec": float(frame_id / max(1.0, fps)),
                    "bbox_xyxy_norm": [float(v) for v in bbox_norm],
                }
            dense = [dense_by_frame[key] for key in sorted(dense_by_frame.keys())]
        if frame_count <= 0 and dense:
            frame_count = max(int(row.get("frame", 0) or 0) for row in dense) + 1
        tracks = dict(final_meta.get("task_tracks", {}) or {}) if has_final_task else {}
        env_track = dict(tracks.get("environmental_awareness", {}) or {})
        self_track = dict(tracks.get("self_state_awareness", {}) or {})
        visible_count = int(env_track.get("visible_count", len(intervals_sec)) or len(intervals_sec))
        difficulty_band = str(env_track.get("difficulty_band", "") or self_track.get("task_difficulty", "") or "") or _difficulty_band(visible_count)
        difficulty_score = float(env_track.get("difficulty_score", 0.0) or self_track.get("task_difficulty_score", 0.0) or 0.0)
        review = review_items.get(mission_dir.name, {}) if isinstance(review_items.get(mission_dir.name, {}), dict) else {}
        ref_asset = _resolve_reference_view_asset(
            config=config,
            layout=layout,
            scene_id=scene_id,
            instance_entry=stage2_entry,
            assets_root=assets_root,
            mission_id=mission_dir.name,
        )
        if ref_asset is None:
            return None
        reference_assets = {}
        for entry in landmark_entries:
            asset = _resolve_reference_view_asset(
                config=config,
                layout=layout,
                scene_id=scene_id,
                instance_entry=entry,
                assets_root=assets_root,
                mission_id=mission_dir.name,
            )
            if asset is None:
                continue
            reference_assets[str(asset["instance_id"])] = asset
        image_cfg = build_image_compression_cfg(_stage3_cfg(config))
        overview = preferred_output_path(
            assets_root / "overview" / f"{mission_dir.name}.jpg",
            compress_enabled=bool(image_cfg.get("enabled", True)),
        )
        if _stage3_include_overview_image(config) and frame_paths and not overview.exists():
            _build_storyboard_image(frame_paths=frame_paths, out_path=overview, title=f"Mission overview | {mission_dir.name}")
        include_keyframe_board = _stage3_include_keyframe_board_image(config)
        keyframes_whole_second = _whole_second_keyframes(
            {
                "keyframe_gt_dense": dense,
                "visible_intervals_sec": intervals_sec,
                "fps": fps,
                "frame_count": frame_count,
            },
            config=config,
        )
        keyframe_board = preferred_output_path(
            assets_root / "keyframes" / f"{mission_dir.name}.jpg",
            compress_enabled=bool(image_cfg.get("enabled", True)),
        )
        if include_keyframe_board and frame_paths and keyframes_whole_second and not keyframe_board.exists():
            _build_keyframe_board_image(frame_paths=frame_paths, keyframes=keyframes_whole_second, out_path=keyframe_board, fps=fps)
        set_class = dict(constraint.get("set_class", {}) or {})
        set_instance = dict(constraint.get("set_instance", {}) or self_track.get("set_instance", {}) or {})
        element_instances = list(constraint.get("element_instances", []) or self_track.get("element_instances", []) or [])
        mission_family = "flight_mission"
        mission_type = str(set_instance.get("set_name", set_class.get("display_name", "single_atomic")) or "single_atomic")
        mission_subtype = str(set_instance.get("set_id", "single_atomic") or "single_atomic")
        service_scenario = str(set_class.get("scope", constraint.get("mode", "single-landmark")) or "single-landmark")
        low_level_sequence = [str(item.get("element_class", "") or "") for item in element_instances if str(item.get("element_class", "")).strip()]
        event_sequence = list(self_track.get("event_sequence", []) or [])
        mode_sequence = list(self_track.get("mode_sequence", []) or [])
        behavior_intervals_sec = list(self_track.get("behavior_intervals_sec", []) or [])
        if not behavior_intervals_sec and isinstance(segments, dict):
            behavior_intervals_sec = _fallback_behavior_intervals_from_segments(segments, frame_count=frame_count, fps=float(fps))
        frame_behavior_labels = list(self_track.get("frame_behavior_labels", []) or [])
        self_keyframes = list(self_track.get("keyframe_gt_dense", []) or [])
        task_type_label = str(self_track.get("task_type", mission_type) or mission_type)
        task_subtype_label = str(self_track.get("task_subtype", mission_subtype) or mission_subtype)
        task_group = str(final_meta.get("task_group", constraint.get("task_group", "dual_awareness")) or "dual_awareness")
        flight_description = str(constraint.get("flight_description", "") or "").strip()
        if not flight_description:
            low_level_desc = ", ".join(low_level_sequence) if low_level_sequence else "no elements"
            flight_description = f"Stage 3 mission around landmark {instance_id}, composed from {low_level_desc}."
        video_path = _resolve_path_near(final_meta_path.parent, video.get("path", ""))
        if video_path is None or not video_path.exists():
            for cand_name in ["task_rgb.mp4", "task_rgb_web.mp4", "task_rgb_720p.mp4", "task_rgb_720p_web.mp4"]:
                cand = final_meta_path.parent / cand_name
                if cand.exists():
                    video_path = cand.resolve()
                    break
        video_web_path = _resolve_path_near(final_meta_path.parent, video.get("path_web", "")) or video_path
        summary_payload = dict((constraint.get("summary", {}) or {}) if isinstance(constraint.get("summary", {}), dict) else {})
        fallback_landmark_set_map = dict(summary_payload.get("landmark_set_map", {}) or {})
        if not fallback_landmark_set_map:
            fallback_landmark_set_map = dict((set_class.get("selected_component_set_map", {}) or {}))
        return {
            "traj_id": mission_dir.name,
            "mission_id": mission_dir.name,
            "scene_id": scene_id,
            "engine": engine,
            "mode": str(constraint.get("mode", "single-landmark") or "single-landmark"),
            "landmark_id": instance_id,
            "landmark_ids": list(all_landmark_ids),
            "landmark_category": _landmark_category_text(stage2_entry),
            "landmark_subcategory": _landmark_subcategory_text(stage2_entry),
            "landmark_description": _landmark_description_text(stage2_entry),
            "landmark_descriptions": {
                str(item.get("instance_id", "") or ""): _landmark_description_text(item)
                for item in landmark_entries
                if str(item.get("instance_id", "") or "").strip()
            },
            "reference_image": str(ref_asset["reference_image"]),
            "reference_image_with_bbox": str(ref_asset["reference_image_with_bbox"]),
            "reference_bbox_xyxy_norm": list(ref_asset["reference_bbox_xyxy_norm"]),
            "reference_images_with_bbox": {
                key: str(value.get("reference_image_with_bbox", "") or "")
                for key, value in reference_assets.items()
            },
            "traj_dir": _path_for_json(mission_dir),
            "final_meta_path": _path_for_json(final_meta_path) if has_final_task else "",
            "video_path": _path_for_json(video_path) if video_path and video_path.exists() else "",
            "video_web_path": _path_for_json(video_web_path) if video_web_path and video_web_path.exists() else "",
            "frame_manifest_path": _path_for_json(frame_manifest_path) if frame_manifest_path and frame_manifest_path.exists() else None,
            "overview_image": _path_for_json(overview) if overview.exists() and _stage3_include_overview_image(config) else None,
            "keyframe_board_image": _path_for_json(keyframe_board) if keyframe_board.exists() and include_keyframe_board else None,
            "frame_count": frame_count,
            "fps": fps,
            "video_width": width,
            "video_height": height,
            "task_group": task_group,
            "task_type_label": task_type_label,
            "task_subtype_label": task_subtype_label,
            "mission_family": mission_family,
            "mission_type": mission_type,
            "mission_subtype": mission_subtype,
            "service_scenario": service_scenario,
            "set_class": set_class,
            "set_instance": set_instance,
            "set_id": str(set_instance.get("set_id", "") or ""),
            "set_name": str(set_instance.get("set_name", "") or ""),
            "landmark_set_map": fallback_landmark_set_map if fallback_landmark_set_map else dict(constraint.get("landmark_set_map", {}) or {}),
            "element_instances": element_instances,
            "mode_sequence": mode_sequence,
            "event_sequence": event_sequence,
            "behavior_intervals_sec": behavior_intervals_sec,
            "frame_behavior_labels": frame_behavior_labels,
            "self_state_keyframe_gt_dense": self_keyframes,
            "low_level_sequence": low_level_sequence,
            "flight_description": flight_description,
            "visible_count": visible_count,
            "visible_intervals_sec": intervals_sec,
            "target_presence_targets": target_presence_targets,
            "keyframe_gt_dense": dense,
            "difficulty_band": difficulty_band,
            "difficulty_score": difficulty_score,
            "render_status": "ready" if has_final_task and not generated_without_video and video_path and video_path.exists() else "pending",
            "review_status": str(review.get("status", "pending") or "pending"),
            "review_note": str(review.get("note", "") or ""),
            "review_updated_at": str(review.get("updated_at", "") or ""),
        }

    if len(mission_dirs) <= 1:
        for mission_dir in mission_dirs:
            row = _load_one_candidate(mission_dir)
            if row is not None:
                candidates.append(row)
    else:
        worker_count = max(1, min(len(mission_dirs), 12))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_load_one_candidate, mission_dir): mission_dir for mission_dir in mission_dirs}
            for fut in as_completed(futures):
                row = fut.result()
                if row is not None:
                    candidates.append(row)
        candidates.sort(key=lambda row: str(row.get("traj_id", "") or ""))
    return candidates, layout


def _build_manifest_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    missions = {str(s.get("mission_id", "") or "") for s in samples if str(s.get("mission_id", "") or "").strip()}
    landmarks = {str(s.get("landmark_id", "") or "") for s in samples if str(s.get("landmark_id", "") or "").strip()}
    categories = sorted({str(s.get("landmark_category", "") or "") for s in samples if str(s.get("landmark_category", "") or "").strip()})
    return {
        "sample_count": len(samples),
        "mission_count": len(missions),
        "landmark_count": len(landmarks),
        "category_count": len(categories),
        "categories": categories,
        "task_groups": sorted({str(s.get("task_group", "") or "") for s in samples if str(s.get("task_group", "") or "").strip()}),
        "task_names": sorted({str(s.get("task_name", s.get("form", "")) or "") for s in samples if str(s.get("task_name", s.get("form", "")) or "").strip()}),
        "forms": sorted({str(s.get("form", "") or "") for s in samples if str(s.get("form", "") or "").strip()}),
        "difficulty_bands": sorted({str(s.get("difficulty_band", "") or "") for s in samples if str(s.get("difficulty_band", "") or "").strip()}),
        "render_statuses": sorted({str(s.get("render_status", "") or "") for s in samples if str(s.get("render_status", "") or "").strip()}),
    }


def _normalize_requested_task_names(forms: list[str]) -> list[str]:
    alias = {
        "count_only": "env_visibility_reasoning",
        "intervals": "env_visibility_reasoning",
        "intervals_plus_keyframes": "env_visibility_reasoning",
        "env_visible_count": "env_visibility_reasoning",
        "env_visible_intervals": "env_visibility_reasoning",
        "env_keyframe_bbox": "env_visibility_reasoning",
        "self_element_instance_localization": "self_element_instance_recognition",
    }
    out = []
    for name in forms:
        normalized = alias.get(str(name), str(name))
        if normalized in TASK_SPECS and normalized not in out:
            out.append(normalized)
    return out


def _behavior_name_en(behavior_key: str) -> str:
    key = str(behavior_key or "").strip()
    if key in _BEHAVIOR_LABELS_EN:
        return _BEHAVIOR_LABELS_EN[key]
    return " ".join(part.capitalize() for part in key.replace("-", "_").split("_") if part) or "Behavior"


def _camera_mode_name_en(camera_mode: str) -> str:
    key = str(camera_mode or "").strip()
    return _CAMERA_MODE_LABELS_EN.get(key, " ".join(part.capitalize() for part in key.replace("-", "_").split("_") if part) or "Unknown")


def _format_angle_text(value: Any) -> str:
    try:
        return f"{int(round(float(value)))}°"
    except Exception:
        return ""


def _direction_text_en(direction: str) -> str:
    key = str(direction or "").strip().lower()
    if key == "cw":
        return "clockwise"
    if key == "ccw":
        return "counterclockwise"
    return ""


def _viewing_text_en(camera_mode: str) -> str:
    key = str(camera_mode or "").strip().lower()
    if key in {"track_target", "landmark_track"}:
        return "while keeping the landmark in view"
    if key in {"velocity_aligned", "look_forward"}:
        return "while looking forward"
    return ""


def _element_option_label(row: dict[str, Any]) -> str:
    behavior_key = str(row.get("element_class", row.get("behavior_id", "")) or "").strip()
    params = dict(row.get("params", {}) or {})
    camera_mode = str(params.get("camera_mode", "") or "").strip()
    view_text = _viewing_text_en(camera_mode)
    direction_text = _direction_text_en(str(params.get("direction", "") or ""))
    angle_text = _format_angle_text(params.get("arc_deg")) if "arc_deg" in params else ""
    travel_text = f"{int(round(float(params.get('travel_distance_m', 0) or 0)))} meters" if "travel_distance_m" in params else ""
    rise_text = f"and rising {int(round(float(params.get('rise_m', 0) or 0)))} meters" if "rise_m" in params else ""
    edge_text = ""
    if behavior_key == "square_orbit":
        edge_text = f"with {int(params.get('edge_count', 4) or 4)} corners" if "edge_count" in params else ""
    elif behavior_key == "triangular_orbit":
        edge_text = f"with {int(params.get('edge_count', 3) or 3)} corners" if "edge_count" in params else ""
    cycle_text = f"over {int(params.get('cycles', 1) or 1)} full loop{'s' if int(params.get('cycles', 1) or 1) > 1 else ''}" if behavior_key == "figure8_orbit" else ""
    width_text = f"{int(round(float(params.get('effective_scan_width_m', 0) or 0)))} meters" if behavior_key == "surface_mapping" else ""
    height_text = f"{int(round(float(params.get('effective_scan_height_m', 0) or 0)))} meters" if behavior_key == "surface_mapping" else ""
    lane_text = f"using {int(params.get('lane_count', 0) or 0)} back-and-forth passes" if behavior_key == "surface_mapping" else ""
    template_key = behavior_key if behavior_key else "default"
    try:
        template = get_config_template("behavior_templates", "stage3", "atomic", template_key)
    except Exception:
        template = get_config_template("behavior_templates", "stage3", "atomic", "default")
    return render_prompt_template(
        template,
        {
            "travel_text": travel_text,
            "view_text": view_text,
            "rise_text": rise_text,
            "angle_text": angle_text,
            "direction_text": direction_text,
            "edge_text": edge_text,
            "cycle_text": cycle_text,
            "width_text": width_text,
            "height_text": height_text,
            "lane_text": lane_text,
            "behavior_name": _behavior_name_en(behavior_key).lower(),
        },
    ).replace("  ", " ").strip()


def _set_option_label(sample: dict[str, Any]) -> str:
    set_id = str(sample.get("set_id", "") or ((sample.get("set_instance", {}) or {}).get("set_id", "")) or "").strip()
    if not set_id:
        return ""
    if set_id == "surface_mapping":
        return render_prompt_template(get_config_template("behavior_templates", "stage3", "composite", "surface_mapping"), {})
    if set_id.startswith("atomic_"):
        element_rows = list(sample.get("element_instances", []) or [])
        if element_rows:
            return render_prompt_template(
                get_config_template("behavior_templates", "stage3", "composite", "atomic_wrapper"),
                {"element_label": _element_option_label(element_rows[0])},
            ).strip()
        return render_prompt_template(
            get_config_template("behavior_templates", "stage3", "composite", "atomic_wrapper"),
            {"element_label": _behavior_name_en(set_id.replace('atomic_', '', 1)).lower()},
        ).strip()
    parts = set_id.split("_")
    family_map = {
        "circular": "a circular inspection flight",
        "spiral": "a spiral inspection flight",
        "square": "a square-orbit inspection flight",
        "triangular": "a triangular-orbit inspection flight",
        "multi": "a multi-landmark survey flight",
    }
    family = family_map.get(parts[0], "an inspection mission")
    variant = "that first looks forward and then tracks the landmark"
    orbit_angles = []
    for row in list(sample.get("element_instances", []) or []):
        params = dict(row.get("params", {}) or {})
        if "arc_deg" in params:
            angle_text = _format_angle_text(params.get("arc_deg"))
            if angle_text:
                orbit_angles.append(angle_text)
    angle_suffix = ""
    if orbit_angles:
        angle_suffix = "including " + ", ".join(orbit_angles) + " orbit segments"
    extras = [x for x in [variant, angle_suffix] if x]
    template_key = parts[0] if parts else "default"
    try:
        template = get_config_template("behavior_templates", "stage3", "composite", template_key)
    except Exception:
        template = get_config_template("behavior_templates", "stage3", "composite", "default")
    return render_prompt_template(
        template,
        {
            "family": family,
            "variant": variant,
            "angle_suffix": angle_suffix,
            "extras": " ".join(extras).strip(),
        },
    ).strip()


def _format_choice_options_for_prompt(options: list[dict[str, Any]]) -> str:
    return "\n".join(f"{row['option_id']}. {row['label']}" for row in list(options or []))


def _landmark_alias_map(sample: dict[str, Any]) -> dict[str, str]:
    ids = [str(x).strip() for x in list(sample.get("landmark_ids", []) or []) if str(x).strip()]
    if not ids:
        single_id = str(sample.get("landmark_id", "") or "").strip()
        if single_id:
            ids = [single_id]
    return {landmark_id: f"Landmark {idx + 1}" for idx, landmark_id in enumerate(ids)}


def _landmark_context_rows(sample: dict[str, Any]) -> list[dict[str, str]]:
    alias_map = _landmark_alias_map(sample)
    descriptions = dict(sample.get("landmark_descriptions", {}) or {})
    rows: list[dict[str, str]] = []
    for landmark_id, alias in alias_map.items():
        desc = str(descriptions.get(landmark_id, "") or "").strip()
        if not desc and len(alias_map) == 1:
            desc = str(sample.get("landmark_description", "") or "").strip()
        if not desc:
            desc = str(landmark_id)
        rows.append(
            {
                "landmark_id": str(landmark_id),
                "alias": str(alias),
                "description": str(desc),
            }
        )
    return rows


def _landmark_context_lines(sample: dict[str, Any]) -> list[str]:
    return [f"{row['description']} ({row['alias']})" for row in _landmark_context_rows(sample)]


def _landmark_alias_block(sample: dict[str, Any]) -> str:
    rows = _landmark_context_rows(sample)
    if len(rows) <= 1:
        return ""
    body = "\n".join(f"- {row['description']} ({row['alias']})" for row in rows)
    return f"Target landmarks:\n{body}\n\n"


def _landmark_alias_summary(sample: dict[str, Any]) -> str:
    return "; ".join(_landmark_context_lines(sample))


def _landmark_alias_bullets(sample: dict[str, Any]) -> str:
    return "\n".join(f"- {row['description']} ({row['alias']})" for row in _landmark_context_rows(sample))


def _landmark_alias_for_target(sample: dict[str, Any], target_instance_id: str) -> str:
    alias_map = _landmark_alias_map(sample)
    return str(alias_map.get(str(target_instance_id or "").strip(), str(target_instance_id or "").strip() or "Target")).strip()


def _element_option_label_for_sample(row: dict[str, Any], sample: dict[str, Any]) -> str:
    base = _element_option_label(row)
    target_id = str(row.get("target_instance_id", "") or "").strip()
    alias = _landmark_alias_for_target(sample, target_id)
    if alias and len(_landmark_alias_map(sample)) > 1:
        return f"{alias}: {base}"
    return base


def _set_component_label_for_sample(sample: dict[str, Any], target_instance_id: str) -> str:
    alias = _landmark_alias_for_target(sample, target_instance_id)
    set_map = dict(sample.get("landmark_set_map", {}) or {})
    set_key = str(set_map.get(target_instance_id, "") or "").strip()
    if not set_key:
        return ""
    rows = [row for row in list(sample.get("element_instances", []) or []) if str(row.get("target_instance_id", "") or "").strip() == str(target_instance_id).strip()]
    temp_sample = {
        "set_id": set_key,
        "set_instance": {"set_id": set_key},
        "element_instances": rows,
    }
    base = _set_option_label(temp_sample)
    if alias and len(_landmark_alias_map(sample)) > 1:
        return f"{alias}: {base}"
    return base


def _element_label_interval_map(sample: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    element_rows = list(sample.get("element_instances", []) or [])
    behavior_rows = list(sample.get("behavior_intervals_sec", []) or [])
    for idx, row in enumerate(element_rows):
        label = _element_option_label(row)
        if not label:
            continue
        intervals = []
        if idx < len(behavior_rows) and isinstance(behavior_rows[idx], dict):
            intervals = list(behavior_rows[idx].get("intervals_sec", []) or [])
        out.setdefault(label, [])
        out[label].extend(list(intervals or []))
    return out


def _sample_choice_options(correct_label: str, universe: list[str], rng: random.Random, *, count: int = 4) -> list[dict[str, Any]]:
    pool = [str(x) for x in universe if str(x).strip() and str(x) != str(correct_label)]
    rng.shuffle(pool)
    chosen = [str(correct_label)] + pool[: max(0, count - 1)]
    rng.shuffle(chosen)
    return [{"option_id": chr(ord("A") + idx), "label": label} for idx, label in enumerate(chosen)]


def _build_joint_choice_labels(
    *,
    composite_labels: list[str],
    atomic_labels: list[str],
    universe: list[str],
    rng: random.Random,
    max_count: int = 8,
) -> list[str]:
    # Composite labels are mandatory for any mission that actually has a composite layer.
    # We pin them first, then keep atomic gold labels, then fill remaining slots with distractors.
    pinned: list[str] = []
    for label in [*list(composite_labels or []), *list(atomic_labels or [])]:
        text = str(label or "").strip()
        if text and text not in pinned:
            pinned.append(text)
    distractors = [str(label).strip() for label in list(universe or []) if str(label).strip() and str(label).strip() not in pinned]
    rng.shuffle(distractors)
    chosen = pinned + distractors[: max(0, int(max_count) - len(pinned))]
    if len(chosen) > int(max_count):
        chosen = chosen[: int(max_count)]
    if composite_labels:
        for label in composite_labels:
            text = str(label or "").strip()
            if text and text not in chosen:
                if len(chosen) < int(max_count):
                    chosen.append(text)
                else:
                    # Replace the latest non-composite distractor if the list is full.
                    for idx in range(len(chosen) - 1, -1, -1):
                        if str(chosen[idx]) not in {str(x).strip() for x in composite_labels}:
                            chosen[idx] = text
                            break
    rng.shuffle(chosen)
    return chosen


def _base_manifest_sample(candidate: dict[str, Any], *, scene_id: str, engine: str, sample_id: str, mode: str, task_name: str) -> dict[str, Any]:
    spec = TASK_SPECS[task_name]
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "engine": engine,
        "mode": str(candidate.get("mode", mode) or mode),
        "manifest_mode": str(mode or ""),
        "form": task_name,
        "task_group": spec["task_group"],
        "task_name": task_name,
        "task_display_name": spec["display_name"],
        "response_kind": spec["response_kind"],
        "mission_id": candidate["mission_id"],
        "mission_family": candidate["mission_family"],
        "service_scenario": candidate["service_scenario"],
        "set_class": dict(candidate.get("set_class", {}) or {}),
        "set_instance": dict(candidate.get("set_instance", {}) or {}),
        "set_id": str((candidate.get("set_instance", {}) or {}).get("set_id", "") or ""),
        "set_name": str((candidate.get("set_instance", {}) or {}).get("set_name", "") or ""),
        "element_instances": list(candidate.get("element_instances", []) or []),
        "element_instance_ids": [str(row.get("element_instance_id", "") or "") for row in list(candidate.get("element_instances", []) or []) if str(row.get("element_instance_id", "") or "").strip()],
        "element_sequence": [str(row.get("element_class", "") or "") for row in list(candidate.get("element_instances", []) or []) if str(row.get("element_class", "") or "").strip()],
        "task_type_label": candidate.get("task_type_label", candidate.get("set_name", "")),
        "task_subtype_label": candidate.get("task_subtype_label", candidate.get("set_id", "")),
        "mode_sequence": list(candidate.get("mode_sequence", []) or []),
        "event_sequence": list(candidate.get("event_sequence", []) or []),
        "behavior_intervals_sec": list(candidate.get("behavior_intervals_sec", []) or []),
        "frame_behavior_labels": list(candidate.get("frame_behavior_labels", []) or []),
        "low_level_sequence": candidate["low_level_sequence"],
        "landmark_id": candidate["landmark_id"],
        "landmark_ids": list(candidate.get("landmark_ids", []) or []),
        "landmark_category": candidate["landmark_category"],
        "landmark_subcategory": str(candidate.get("landmark_subcategory", "") or ""),
        "landmark_description": str(candidate.get("landmark_description", "") or ""),
        "landmark_descriptions": dict(candidate.get("landmark_descriptions", {}) or {}),
        "reference_images_with_bbox": dict(candidate.get("reference_images_with_bbox", {}) or {}),
        "landmark_set_map": dict(candidate.get("landmark_set_map", {}) or {}),
        "reference_image_with_bbox": candidate["reference_image_with_bbox"],
        "reference_bbox_xyxy_norm": candidate["reference_bbox_xyxy_norm"],
        "overview_image": candidate["overview_image"],
        "keyframe_board_image": candidate["keyframe_board_image"],
        "video_web_path": candidate["video_web_path"],
        "video_path": candidate["video_path"],
        "fps": float(candidate["fps"]),
        "frame_count": int(candidate["frame_count"]),
        "video_width": int(candidate["video_width"]),
        "video_height": int(candidate["video_height"]),
        "flight_description": candidate["flight_description"],
        "visible_count": int(candidate["visible_count"]),
        "visible_intervals_sec": list(candidate["visible_intervals_sec"]),
        "target_presence_targets": dict(candidate.get("target_presence_targets", {}) or {}),
        "keyframe_gt_dense": list(candidate["keyframe_gt_dense"]),
        "self_state_keyframe_gt_dense": list(candidate.get("self_state_keyframe_gt_dense", []) or []),
        "difficulty_score": float(candidate.get("difficulty_score", 0.0) or 0.0),
        "difficulty_band": candidate["difficulty_band"],
        "candidate_path": candidate["final_meta_path"],
    }


def generate_manifest(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    sample_count: int,
    seed: int,
    forms: list[str],
    approved_only: bool,
    mode: str = "single-landmark",
    include_temporal_localization: bool = False,
    require_final_task: bool = True,
    update_latest: bool = True,
) -> dict[str, Any]:
    candidates, layout = _discover_candidates(config=config, scene_id=scene_id, engine=engine, require_final_task=require_final_task)
    if str(mode or "single-landmark") == "all":
        eligible = list(candidates)
    else:
        eligible = [row for row in candidates if str(row.get("mode", "single-landmark")) == mode]
    if approved_only:
        eligible = [row for row in eligible if str(row.get("review_status", "pending")) == "approved"]
    if not eligible:
        raise RuntimeError("no_stage3_candidates_available")
    allowed_tasks = _normalize_requested_task_names(forms)
    if not allowed_tasks:
        raise RuntimeError("no_stage3_task_types_selected")
    rng = random.Random(int(seed))
    experiment_defaults = (_stage3_cfg(config).get("experiment_defaults", {}) or {})
    manifest_provide_flight_description = bool(experiment_defaults.get("provide_flight_description", True))
    manifest_include_keyframes = bool(experiment_defaults.get("include_keyframes", False))
    samples: list[dict[str, Any]] = []
    max_attempts = max(int(sample_count) * max(4, len(allowed_tasks)), 16)
    attempt = 0
    while len(samples) < int(sample_count) and attempt < max_attempts:
        form = allowed_tasks[attempt % len(allowed_tasks)]
        candidate_round = attempt // max(1, len(allowed_tasks))
        if str(mode or "single-landmark") == "all":
            candidate = eligible[candidate_round % len(eligible)] if candidate_round < len(eligible) else rng.choice(eligible)
        else:
            candidate = eligible[candidate_round % len(eligible)] if candidate_round < len(eligible) else rng.choice(eligible)
        sample_id = f"{scene_id}_{candidate['mission_id']}_{form}_{len(samples) + 1:06d}"
        sample = _base_manifest_sample(candidate, scene_id=scene_id, engine=engine, sample_id=sample_id, mode=mode, task_name=form)
        sample["include_keyframes"] = manifest_include_keyframes
        if form == "self_instance_recognition_joint":
            alias_map = _landmark_alias_map(sample)
            set_label = _set_option_label(sample) if len(alias_map) <= 1 else ""
            element_labels = [_element_option_label_for_sample(row, sample) for row in list(sample.get("element_instances", []) or [])]
            element_labels = [label for label in element_labels if str(label).strip()]
            if not element_labels:
                attempt += 1
                continue
            is_atomic_set = str(sample.get("set_id", "") or "").startswith("atomic_")
            gold_labels = list(element_labels)
            if len(alias_map) > 1:
                component_labels = [_set_component_label_for_sample(sample, target_id) for target_id in alias_map.keys()]
                component_labels = [label for label in component_labels if str(label).strip()]
                gold_labels = [*component_labels, *gold_labels]
            elif set_label and not is_atomic_set:
                gold_labels = [set_label, *gold_labels]
            if len(alias_map) > 1:
                eligible_universe = [item for item in eligible if len(_landmark_alias_map(item)) > 1]
                universe = sorted(
                    {label for item in eligible_universe for target_id in _landmark_alias_map(item).keys() for label in [_set_component_label_for_sample(item, target_id)] if str(label).strip()}
                    | {label for item in eligible_universe for row in list(item.get("element_instances", []) or []) for label in [_element_option_label_for_sample(row, item)] if str(label).strip()}
                )
            else:
                eligible_universe = [item for item in eligible if len(_landmark_alias_map(item)) <= 1]
                universe = sorted(
                    {label for item in eligible_universe for label in [_set_option_label(item)] if str(label).strip() and not str(item.get("set_id", "") or "").startswith("atomic_")}
                    | {label for item in eligible_universe for row in list(item.get("element_instances", []) or []) for label in [_element_option_label(row)] if str(label).strip()}
                )
            universe = [label for label in universe if str(label).strip()]
            composite_gold_labels = component_labels if len(alias_map) > 1 else ([set_label] if set_label and not is_atomic_set else [])
            chosen = _build_joint_choice_labels(
                composite_labels=composite_gold_labels,
                atomic_labels=element_labels,
                universe=universe,
                rng=rng,
                max_count=8,
            )
            options = [{"option_id": chr(ord("A") + idx), "label": label} for idx, label in enumerate(chosen)]
            sample["choice_options"] = options
            sample["multi_select"] = True
            behavior_rows = list(sample.get("behavior_intervals_sec", []) or [])
            event_map = _element_label_interval_map(sample)
            full_interval = []
            if behavior_rows:
                starts = [float(it.get("start_sec", 0.0) or 0.0) for row in behavior_rows for it in list(row.get("intervals_sec", []) or []) if isinstance(it, dict)]
                ends = [float(it.get("end_sec", 0.0) or 0.0) for row in behavior_rows for it in list(row.get("intervals_sec", []) or []) if isinstance(it, dict)]
                if starts and ends:
                    full_interval = [[min(starts), max(ends)]]
            answer_items = []
            for row in options:
                label = str(row["label"])
                if label not in gold_labels:
                    continue
                if len(alias_map) <= 1 and set_label and label == set_label and not is_atomic_set:
                    intervals = full_interval if include_temporal_localization else []
                elif len(alias_map) > 1 and any(label == _set_component_label_for_sample(sample, target_id) for target_id in alias_map.keys()):
                    target_id = next((tid for tid in alias_map.keys() if label == _set_component_label_for_sample(sample, tid)), "")
                    target_rows = [it for it in list(sample.get("behavior_intervals_sec", []) or []) if str(it.get("target_instance_id", "") or "").strip() == str(target_id).strip()]
                    starts = [float(it.get("start_sec", 0.0) or 0.0) for row2 in target_rows for it in list(row2.get("intervals_sec", []) or []) if isinstance(it, dict)]
                    ends = [float(it.get("end_sec", 0.0) or 0.0) for row2 in target_rows for it in list(row2.get("intervals_sec", []) or []) if isinstance(it, dict)]
                    intervals = [[min(starts), max(ends)]] if (include_temporal_localization and starts and ends) else []
                else:
                    intervals = event_map.get(label, []) if include_temporal_localization else []
                answer_items.append({"option_id": row["option_id"], "label": label, "intervals_sec": list(intervals or [])})
            sample["answer_option_ids"] = [row["option_id"] for row in answer_items]
            sample["answer_items"] = answer_items
        elif form == "self_set_instance_recognition":
            alias_map = _landmark_alias_map(sample)
            if len(alias_map) > 1:
                correct_labels = [_set_component_label_for_sample(sample, target_id) for target_id in alias_map.keys()]
                correct_labels = [label for label in correct_labels if str(label).strip()]
                if not correct_labels:
                    attempt += 1
                    continue
                universe = sorted({label for item in eligible for target_id in _landmark_alias_map(item).keys() for label in [_set_component_label_for_sample(item, target_id)] if str(label).strip()})
                distractors = [label for label in universe if label not in correct_labels]
                rng.shuffle(distractors)
                chosen = correct_labels + distractors[: max(0, 8 - len(correct_labels))]
                chosen = chosen[:8]
                rng.shuffle(chosen)
                options = [{"option_id": chr(ord("A") + idx), "label": label} for idx, label in enumerate(chosen)]
                sample["choice_options"] = options
                sample["multi_select"] = True
                answer_items = []
                for row in options:
                    label = str(row["label"])
                    if label not in correct_labels:
                        continue
                    target_id = next((tid for tid in alias_map.keys() if label == _set_component_label_for_sample(sample, tid)), "")
                    target_rows = [it for it in list(sample.get("behavior_intervals_sec", []) or []) if str(it.get("target_instance_id", "") or "").strip() == str(target_id).strip()]
                    starts = [float(it.get("start_sec", 0.0) or 0.0) for row2 in target_rows for it in list(row2.get("intervals_sec", []) or []) if isinstance(it, dict)]
                    ends = [float(it.get("end_sec", 0.0) or 0.0) for row2 in target_rows for it in list(row2.get("intervals_sec", []) or []) if isinstance(it, dict)]
                    intervals = [[min(starts), max(ends)]] if (include_temporal_localization and starts and ends) else []
                    answer_items.append({"option_id": row["option_id"], "label": label, "intervals_sec": intervals})
                sample["answer_items"] = answer_items
                sample["answer_option_ids"] = [row["option_id"] for row in answer_items]
                sample["answer_option_id"] = sample["answer_option_ids"][0] if sample["answer_option_ids"] else ""
                sample["multi_select"] = True
                sample["include_temporal_localization"] = bool(include_temporal_localization)
                sample["prompt_text"] = _build_prompt(
                    sample,
                    provide_flight_description=manifest_provide_flight_description,
                    include_keyframes=manifest_include_keyframes,
                )
                sample["user_prompt"] = sample["prompt_text"]
                sample["system_prompt"] = _build_system_prompt(sample)
                samples.append(sample)
                attempt += 1
                continue
            correct = _set_option_label(sample)
            if not correct:
                attempt += 1
                continue
            eligible_universe = [item for item in eligible if len(_landmark_alias_map(item)) <= 1]
            universe = sorted({label for item in eligible_universe for label in [_set_option_label(item)] if str(label).strip()})
            options = _sample_choice_options(correct, universe, rng, count=8)
            sample["choice_options"] = options
            sample["answer_option_id"] = next((row["option_id"] for row in options if row["label"] == correct), "A")
            behavior_rows = list(sample.get("behavior_intervals_sec", []) or [])
            full_interval = []
            if behavior_rows:
                starts = [float(it.get("start_sec", 0.0) or 0.0) for row in behavior_rows for it in list(row.get("intervals_sec", []) or []) if isinstance(it, dict)]
                ends = [float(it.get("end_sec", 0.0) or 0.0) for row in behavior_rows for it in list(row.get("intervals_sec", []) or []) if isinstance(it, dict)]
                if starts and ends:
                    full_interval = [[min(starts), max(ends)]]
            sample["answer_items"] = [{"option_id": sample["answer_option_id"], "label": correct, "intervals_sec": full_interval if include_temporal_localization else []}]
        elif form == "self_element_instance_recognition":
            labels = [_element_option_label(row) for row in list(sample.get("element_instances", []) or [])]
            labels = [label for label in labels if str(label).strip()]
            if not labels:
                attempt += 1
                continue
            eligible_universe = [item for item in eligible if len(_landmark_alias_map(item)) <= 1]
            universe = sorted({label for item in eligible_universe for row in list(item.get("element_instances", []) or []) for label in [_element_option_label(row)] if str(label).strip()})
            distractors = [label for label in universe if label not in labels]
            rng.shuffle(distractors)
            chosen = labels + distractors[: max(0, 8 - len(labels))]
            chosen = chosen[:8]
            rng.shuffle(chosen)
            options = [{"option_id": chr(ord("A") + idx), "label": label} for idx, label in enumerate(chosen)]
            sample["choice_options"] = options
            sample["multi_select"] = True
            event_map = _element_label_interval_map(sample)
            answer_items = []
            for row in options:
                label = str(row["label"])
                if label not in labels:
                    continue
                answer_items.append({"option_id": row["option_id"], "label": label, "intervals_sec": event_map.get(label, []) if include_temporal_localization else []})
            sample["answer_option_ids"] = [row["option_id"] for row in answer_items]
            sample["answer_items"] = answer_items
        elif form == "env_visibility_reasoning":
            if len([str(x).strip() for x in list(sample.get("landmark_ids", []) or []) if str(x).strip()]) > 1:
                alias_map = _landmark_alias_map(sample)
                target_rows = []
                target_presence_targets = dict(sample.get("target_presence_targets", {}) or {})
                for target_id, alias in alias_map.items():
                    target_payload = dict(target_presence_targets.get(target_id, {}) or {})
                    target_rows.append(
                        {
                            "target_id": alias,
                            "instance_id": target_id,
                            "description": str((sample.get("landmark_descriptions", {}) or {}).get(target_id, "") or ""),
                            "visible_count": int(target_payload.get("visible_frame_count", 0) or 0),
                            "visible_intervals_sec": list(target_payload.get("intervals_sec", []) or []),
                            "keyframes": list(target_payload.get("frame_bboxes_xyxy_norm", []) or []),
                        }
                    )
                sample["gold_environment_answer"] = {"targets": target_rows}
            else:
                sample["gold_environment_answer"] = {
                    "visible_count": int(sample["visible_count"]),
                    "visible_intervals_sec": list(sample["visible_intervals_sec"]),
                }
            sample["keyframe_eval_view_whole_second"] = _whole_second_keyframes(
                {
                    "keyframe_gt_dense": list(candidate.get("keyframe_gt_dense", []) or []),
                    "visible_intervals_sec": list(candidate.get("visible_intervals_sec", []) or []),
                    "fps": float(candidate.get("fps", 24.0) or 24.0),
                    "frame_count": int(candidate.get("frame_count", 0) or 0),
                },
                config=config,
            )
        sample["include_temporal_localization"] = bool(include_temporal_localization and str(form).startswith("self_"))
        sample["prompt_text"] = _build_prompt(
            sample,
            provide_flight_description=manifest_provide_flight_description,
            include_keyframes=manifest_include_keyframes,
        )
        sample["user_prompt"] = sample["prompt_text"]
        sample["system_prompt"] = _build_system_prompt(sample)
        samples.append(sample)
        attempt += 1
    manifests_root = _ensure_dir(layout["datasets_root"])
    task_tag = _safe_name("_".join(sorted(allowed_tasks))[:80] or "tasks")
    generation_id = f"{scene_id}_{mode.replace('-', '_')}_{task_tag}_{len(samples)}samples_{_now_ts()}"
    manifest = {
        "generation_id": generation_id,
        "generated_at": _iso_now(),
        "scene_id": scene_id,
        "engine": engine,
        "mode": mode,
        "sample_count": len(samples),
        "seed": int(seed),
        "forms": allowed_tasks,
        "include_temporal_localization": bool(include_temporal_localization),
        "task_group": "all",
        "approved_only": bool(approved_only),
        "require_final_task": bool(require_final_task),
        "update_latest": bool(update_latest),
        "samples": samples,
        "summary": _build_manifest_summary(samples),
    }
    manifest_path = manifests_root / f"{generation_id}.json"
    _write_json(manifest_path, manifest)
    if update_latest:
        _write_json(manifests_root / f"{scene_id}.latest_manifest.json", manifest)
    return {"manifest_path": manifest_path, "manifest": manifest}


def _build_prompt(sample: dict[str, Any], *, provide_flight_description: bool, include_keyframes: bool) -> str:
    del provide_flight_description, include_keyframes
    task_name = str(sample.get("task_name", sample.get("form", "")) or "")
    options_text = _format_choice_options_for_prompt(list(sample.get("choice_options", []) or []))
    alias_lines = _landmark_context_lines(sample)
    alias_block = _landmark_alias_block(sample)
    if task_name == "self_instance_recognition_joint":
        template_key = _joint_prompt_template_key(sample)
        return render_prompt_template(
            get_prompt_template("stage3", template_key, "user"),
            {
                "alias_block": alias_block,
                "options_text": options_text,
            },
        )
    elif task_name == "self_set_instance_recognition":
        key = "self_set_instance_recognition_multi" if bool(sample.get("multi_select", False)) else "self_set_instance_recognition_single"
        return render_prompt_template(
            get_prompt_template("stage3", key, "user"),
            {
                "alias_block": alias_block,
                "options_text": options_text,
            },
        )
    elif task_name == "self_element_instance_recognition":
        return render_prompt_template(
            get_prompt_template("stage3", "self_element_instance_recognition", "user"),
            {
                "alias_block": alias_block,
                "options_text": options_text,
            },
        )
    elif task_name == "env_visibility_reasoning":
        desc = str(sample.get("landmark_description", "") or "").strip() or "the query landmark"
        alias_lines = _landmark_context_lines(sample)
        if len(alias_lines) > 1:
            return render_prompt_template(
                get_prompt_template("stage3", "env_visibility_reasoning_multi", "user"),
                {
                    "alias_summary": _landmark_alias_summary(sample),
                    "alias_bullets": _landmark_alias_bullets(sample),
                },
            )
        return render_prompt_template(
            get_prompt_template("stage3", "env_visibility_reasoning_single", "user"),
            {
                "landmark_description": desc,
            },
        )
    return "Question:\nPlease answer the task using the provided UAV flight video."


def _build_system_prompt(sample: dict[str, Any]) -> str:
    task_group = str(sample.get("task_group", "") or "")
    task_name = str(sample.get("task_name", sample.get("form", "")) or "")
    if task_name == "self_instance_recognition_joint":
        return get_prompt_template("stage3", _joint_prompt_template_key(sample), "system")
    if task_name == "self_set_instance_recognition":
        return get_prompt_template("stage3", "self_set_instance_recognition", "system")
    if task_name == "self_element_instance_recognition":
        return get_prompt_template("stage3", "self_element_instance_recognition", "system")
    if task_group == "environmental":
        alias_lines = _landmark_context_lines(sample)
        if len(alias_lines) > 1:
            return get_prompt_template("stage3", "env_visibility_reasoning_multi", "system")
        return get_prompt_template("stage3", "env_visibility_reasoning_single", "system")
    if task_group == "self-state":
        return (
            "You are evaluating first-person UAV flight-behavior understanding from a UAV first-person flight video.\n"
            "Return exactly one valid JSON object."
        )
    return (
        "You are evaluating first-person UAV flight-behavior understanding.\n"
        "Return exactly one valid JSON object."
    )


def _joint_prompt_template_key(sample: dict[str, Any]) -> str:
    set_id = str(sample.get("set_id", "") or ((sample.get("set_instance", {}) or {}).get("set_id", "")) or "").strip()
    if set_id.startswith("atomic_"):
        return "self_instance_recognition_joint_atomic"
    return "self_instance_recognition_joint_composite"


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _normalize_interval_rows(rows: Any) -> list[list[float]]:
    out: list[list[float]] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            try:
                s = float(row.get("start_sec", row.get("start", 0.0)) or 0.0)
                e = float(row.get("end_sec", row.get("end", 0.0)) or 0.0)
            except Exception:
                continue
            if e < s:
                s, e = e, s
            out.append([s, e])
            continue
        if isinstance(row, (list, tuple)) and len(row) == 2:
            try:
                s = float(row[0])
                e = float(row[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            out.append([s, e])
    return out


def _normalize_answer_items(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        option_id = str(row.get("option_id", "") or "").strip()
        label = str(row.get("label", "") or "").strip()
        intervals = _normalize_interval_rows(row.get("intervals_sec", []))
        out.append(
            {
                "option_id": option_id,
                "label": label,
                "intervals_sec": intervals,
            }
        )
    return out


def _flatten_answer_item_intervals(rows: Any) -> list[list[float]]:
    out: list[list[float]] = []
    for row in _normalize_answer_items(rows):
        out.extend(_normalize_interval_rows(row.get("intervals_sec", [])))
    return out


def _partition_joint_labels(sample: dict[str, Any]) -> tuple[list[str], list[str]]:
    alias_map = _landmark_alias_map(sample)
    set_label = _set_option_label(sample) if len(alias_map) <= 1 else ""
    is_atomic_set = str(sample.get("set_id", "") or "").startswith("atomic_")
    composite_labels: list[str] = []
    if len(alias_map) > 1:
        composite_labels = [label for label in [_set_component_label_for_sample(sample, target_id) for target_id in alias_map.keys()] if str(label).strip()]
    elif set_label and not is_atomic_set:
        composite_labels = [set_label]
    atomic_labels = [label for label in [_element_option_label_for_sample(row, sample) for row in list(sample.get("element_instances", []) or [])] if str(label).strip()]
    return composite_labels, atomic_labels


def _filter_answer_items_by_labels(rows: Any, labels: list[str]) -> list[dict[str, Any]]:
    label_set = {str(label) for label in list(labels or []) if str(label).strip()}
    if not label_set:
        return []
    return [row for row in _normalize_answer_items(rows) if str(row.get("label", "") or "") in label_set]


def _looks_like_composite_answer_label(label: str) -> bool:
    text = str(label or "").strip().lower()
    if not text:
        return False
    keywords = [
        "inspection",
        "survey",
        "mission",
        "multi-landmark",
        "surface mapping",
    ]
    return any(token in text for token in keywords)


def _derive_joint_level_payloads(
    *,
    sample: dict[str, Any],
    gold_answer_items: list[dict[str, Any]],
    pred_answer_items: list[dict[str, Any]],
) -> dict[str, Any]:
    composite_labels, atomic_labels = _partition_joint_labels(sample)
    gold_composite_answer_items = _filter_answer_items_by_labels(gold_answer_items, composite_labels)
    gold_atomic_answer_items = _filter_answer_items_by_labels(gold_answer_items, atomic_labels)
    pred_composite_answer_items = _filter_answer_items_by_labels(pred_answer_items, composite_labels)
    pred_atomic_answer_items = _filter_answer_items_by_labels(pred_answer_items, atomic_labels)

    if not gold_composite_answer_items:
        gold_composite_answer_items = [
            row for row in _normalize_answer_items(gold_answer_items) if _looks_like_composite_answer_label(str(row.get("label", "") or ""))
        ]
    if not gold_atomic_answer_items:
        gold_atomic_answer_items = [
            row for row in _normalize_answer_items(gold_answer_items) if not _looks_like_composite_answer_label(str(row.get("label", "") or ""))
        ]
    if not pred_composite_answer_items:
        pred_composite_answer_items = [
            row for row in _normalize_answer_items(pred_answer_items) if _looks_like_composite_answer_label(str(row.get("label", "") or ""))
        ]
    if not pred_atomic_answer_items:
        pred_atomic_answer_items = [
            row for row in _normalize_answer_items(pred_answer_items) if not _looks_like_composite_answer_label(str(row.get("label", "") or ""))
        ]

    def _option_ids(rows: list[dict[str, Any]]) -> list[str]:
        return [str(row.get("option_id", "") or "") for row in rows if str(row.get("option_id", "") or "").strip()]

    def _mean_tiou(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> float | None:
        gold_pairs = _flatten_answer_item_intervals(gold_rows)
        pred_pairs = _flatten_answer_item_intervals(pred_rows)
        if not gold_pairs:
            return None
        return sum(
            max((_interval_iou(pred_pair, gold_pair) for pred_pair in pred_pairs), default=0.0)
            for gold_pair in gold_pairs
        ) / float(len(gold_pairs))

    return {
        "joint_composite_answer_items": gold_composite_answer_items,
        "joint_atomic_answer_items": gold_atomic_answer_items,
        "pred_joint_composite_answer_items": pred_composite_answer_items,
        "pred_joint_atomic_answer_items": pred_atomic_answer_items,
        "joint_composite_option_ids": _option_ids(gold_composite_answer_items),
        "joint_atomic_option_ids": _option_ids(gold_atomic_answer_items),
        "pred_joint_composite_option_ids": _option_ids(pred_composite_answer_items),
        "pred_joint_atomic_option_ids": _option_ids(pred_atomic_answer_items),
        "joint_composite_temporal_mean_tiou": _mean_tiou(gold_composite_answer_items, pred_composite_answer_items),
        "joint_atomic_temporal_mean_tiou": _mean_tiou(gold_atomic_answer_items, pred_atomic_answer_items),
    }


def _joint_rows_to_level_rows(rows: list[dict[str, Any]], *, level: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    level_key = "composite" if str(level) == "composite" else "atomic"
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        clone = dict(row)
        if level_key == "composite":
            clone["task_name"] = "self_composite_instance_recognition"
            clone["form"] = "self_composite_instance_recognition"
            clone["task_display_name"] = "Composite 实例识别"
            clone["gold_option_ids"] = list(row.get("joint_composite_option_ids", []) or [])
            clone["pred_option_ids"] = list(row.get("pred_joint_composite_option_ids", []) or [])
            clone["gold_option_id"] = clone["gold_option_ids"][0] if clone["gold_option_ids"] else ""
            clone["pred_option_id"] = clone["pred_option_ids"][0] if clone["pred_option_ids"] else ""
            clone["gold_answer_items"] = _normalize_answer_items(row.get("joint_composite_answer_items", []))
            clone["pred_answer_items"] = _normalize_answer_items(row.get("pred_joint_composite_answer_items", []))
            clone["gold_choice_intervals_sec"] = _flatten_answer_item_intervals(clone["gold_answer_items"])
            clone["pred_choice_intervals_sec"] = _flatten_answer_item_intervals(clone["pred_answer_items"])
            clone["self_temporal_mean_tiou"] = row.get("joint_composite_temporal_mean_tiou")
        else:
            clone["task_name"] = "self_atomic_instance_recognition"
            clone["form"] = "self_atomic_instance_recognition"
            clone["task_display_name"] = "Atomic 实例识别"
            clone["gold_option_ids"] = list(row.get("joint_atomic_option_ids", []) or [])
            clone["pred_option_ids"] = list(row.get("pred_joint_atomic_option_ids", []) or [])
            clone["gold_option_id"] = clone["gold_option_ids"][0] if clone["gold_option_ids"] else ""
            clone["pred_option_id"] = clone["pred_option_ids"][0] if clone["pred_option_ids"] else ""
            clone["gold_answer_items"] = _normalize_answer_items(row.get("joint_atomic_answer_items", []))
            clone["pred_answer_items"] = _normalize_answer_items(row.get("pred_joint_atomic_answer_items", []))
            clone["gold_choice_intervals_sec"] = _flatten_answer_item_intervals(clone["gold_answer_items"])
            clone["pred_choice_intervals_sec"] = _flatten_answer_item_intervals(clone["pred_answer_items"])
            clone["self_temporal_mean_tiou"] = row.get("joint_atomic_temporal_mean_tiou")
        if clone["gold_option_ids"] or clone["pred_option_ids"]:
            out.append(clone)
    return out


def _rows_for_display(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(str(row.get("task_name", "")) == "self_instance_recognition_joint" for row in rows):
        return rows
    joint_rows = [row for row in rows if str(row.get("task_name", "")) == "self_instance_recognition_joint"]
    env_rows = [row for row in rows if str(row.get("task_group", "")) == "environmental"]
    display_rows = []
    display_rows.extend(_joint_rows_to_level_rows(joint_rows, level="composite"))
    display_rows.extend(_joint_rows_to_level_rows(joint_rows, level="atomic"))
    display_rows.extend(env_rows)
    return display_rows


def _normalize_keyframe_rows(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        bbox = list(row.get("bbox_xyxy_norm", []) or [])
        if len(bbox) != 4:
            continue
        try:
            t = float(row.get("time_sec", 0.0) or 0.0)
            bbox = [max(0.0, min(1.0, float(v))) for v in bbox]
        except Exception:
            continue
        out.append({"time_sec": t, "bbox_xyxy_norm": bbox})
    return out


def _normalize_choice_list(rows: Any) -> list[str]:
    out: list[str] = []
    if isinstance(rows, str) and rows.strip():
        return [rows.strip()]
    for row in list(rows or []):
        text = str(row or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _build_openai_messages(sample: dict[str, Any], api_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    video_path_raw = str(sample.get("video_web_path", "") or sample.get("video_path", "") or "").strip()
    if not video_path_raw:
        raise FileNotFoundError("stage3_sample_video_missing")
    video_path = Path(video_path_raw).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"stage3_sample_video_not_found: {video_path_raw}")
    image_paths: list[Path] = []
    for key in ["reference_image_with_bbox", "overview_image", "keyframe_board_image"]:
        raw = str(sample.get(key, "") or "").strip()
        if not raw or raw.lower() == "none":
            continue
        path = Path(raw).resolve()
        if path.exists():
            image_paths.append(path)
    user_prompt_text = str(sample["prompt_text"])
    route_model_name = str(api_cfg.get("base_model", "") or api_cfg.get("model", "") or "")
    video_placeholder = str(required_video_placeholder_for_model(route_model_name) or "").strip()
    if video_placeholder:
        stripped_prompt = user_prompt_text.lstrip()
        if not stripped_prompt.startswith(video_placeholder):
            user_prompt_text = f"{video_placeholder}\n{user_prompt_text}".strip()
    blocks: list[dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
    video_bytes, upload_video_path = _prepare_upload_video(video_path, sample, api_cfg)
    uploads: list[str] = [_path_for_json(upload_video_path)]
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    blocks.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}})
    for path in image_paths:
        image_bytes, mime_type = _prepare_upload_image(path, api_cfg)
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"}})
        uploads.append(_path_for_json(path))
    system_prompt = str(sample.get("system_prompt", "") or "").strip() or _build_system_prompt(sample)
    system_prefix = str(api_cfg.get("system_prompt_prefix", "") or "").strip()
    if system_prefix:
        system_prompt = f"{system_prefix}\n\nThen follow the task-specific instruction below.\n\n{system_prompt}".strip()
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
    return messages, uploads


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
            elif block_type == "video_url":
                blocks.append({"type": "video_url", "video_url": {"url": "<base64-video>"}})
            else:
                blocks.append({"type": block_type})
        preview.append({"role": role, "content": blocks})
    return preview


def _estimate_request_tokens(sample: dict[str, Any], include_keyframes: bool) -> int:
    base = max(120, int(len(str(sample.get("prompt_text", ""))) / 4))
    video_tokens = 2400
    image_tokens = 450 * (3 if include_keyframes and sample.get("keyframe_board_image") else 2)
    return base + video_tokens + image_tokens + 500


def _resolve_api_cfg(config: dict[str, Any], *, override_model: str | None, overrides: dict[str, Any] | None) -> dict[str, Any]:
    temporal_cfg = _stage3_temporal_cfg(config)
    stage3_defaults = dict((_stage3_cfg(config).get("experiment_defaults", {}) or {}))
    override_payload = dict(overrides or {})
    default_api_source = str(stage3_defaults.get("api_source", temporal_cfg.get("api_source", "")) or "").strip()
    default_api_base = str(
        stage3_defaults.get("api_base", stage3_defaults.get("base_url", temporal_cfg.get("api_base", temporal_cfg.get("base_url", ""))))
        or ""
    ).strip()
    default_api_key = str(stage3_defaults.get("api_key", temporal_cfg.get("api_key", "")) or "").strip()
    merged = {
        "upload_resize_enabled": bool(temporal_cfg.get("api_upload_resize_enabled", True)),
        "upload_max_width": int(temporal_cfg.get("api_upload_max_width", 640) or 640),
        "upload_max_height": int(temporal_cfg.get("api_upload_max_height", 480) or 480),
        "upload_jpeg_quality": int(temporal_cfg.get("api_upload_jpeg_quality", 80) or 80),
        "concurrency": int(stage3_defaults.get("concurrency", temporal_cfg.get("experiment_concurrency", 1)) or 1),
        "temperature": 0.0,
        "max_tokens": 600,
        "timeout_s": float(stage3_defaults.get("timeout_s", temporal_cfg.get("timeout_s", 600.0)) or 600.0),
        "request_retry_attempts": int(stage3_defaults.get("request_retry_attempts", 3) or 3),
        "request_retry_backoff_sec": float(stage3_defaults.get("request_retry_backoff_sec", 2.0) or 2.0),
        "request_retry_forever": bool(stage3_defaults.get("request_retry_forever", False)),
        "api_source": default_api_source,
        "api_base": default_api_base,
        "api_key": default_api_key,
    }
    default_rpm_limit = stage3_defaults.get("rpm_limit", temporal_cfg.get("api_rpm_limit"))
    default_tpm_limit = stage3_defaults.get("tpm_limit", temporal_cfg.get("api_tpm_limit"))
    if default_rpm_limit is not None and int(default_rpm_limit or 0) > 0:
        merged["rpm_limit"] = int(default_rpm_limit)
    if default_tpm_limit is not None and int(default_tpm_limit or 0) > 0:
        merged["tpm_limit"] = int(default_tpm_limit)
    for key, value in override_payload.items():
        if value is not None:
            merged[key] = value
    resolved = _resolve_api_settings(config, override_model=override_model, overrides=merged)
    model_controls = build_model_request_controls(str(resolved.get("model", "") or override_model or ""))
    resolved["model"] = str(model_controls["display_model"])
    resolved["base_model"] = str(model_controls["base_model"])
    resolved["model_family"] = str(model_controls["family"])
    resolved["reasoning_mode"] = str(model_controls["mode"])
    resolved["assistant_prefill"] = str(model_controls.get("assistant_prefill", "") or "")
    requested_concurrency = max(1, int(merged.get("concurrency", 1) or 1))
    configured_rpm_limit = int(
        override_payload.get(
            "rpm_limit",
            resolved.get("configured_rpm_limit", resolved.get("rpm_limit", 0)),
        )
        or 0
    )
    configured_tpm_limit = int(
        override_payload.get(
            "tpm_limit",
            resolved.get("configured_tpm_limit", resolved.get("tpm_limit", 0)),
        )
        or 0
    )
    reserve_ratio = float(override_payload.get("rate_limit_reserve_ratio", resolved.get("rate_limit_reserve_ratio", 0.1)) or 0.1)
    estimated_tokens = int(override_payload.get("estimated_tokens_per_request", resolved.get("estimated_tokens_per_request", 0)) or 0)
    if estimated_tokens <= 0:
        estimated_tokens = max(3200, int(merged.get("max_tokens", 600) or 600) + 3600)
    rate_limit_cfg = compute_rate_limited_concurrency(
        requested_concurrency,
        rpm_limit=configured_rpm_limit,
        tpm_limit=configured_tpm_limit,
        estimated_tokens_per_request=estimated_tokens,
        reserve_ratio=reserve_ratio,
    )
    resolved["configured_rpm_limit"] = int(rate_limit_cfg["configured_rpm_limit"])
    resolved["configured_tpm_limit"] = int(rate_limit_cfg["configured_tpm_limit"])
    resolved["rpm_limit"] = int(rate_limit_cfg["effective_rpm_limit"])
    resolved["tpm_limit"] = int(rate_limit_cfg["effective_tpm_limit"])
    resolved["requested_concurrency"] = int(rate_limit_cfg["requested_concurrency"])
    resolved["concurrency"] = int(rate_limit_cfg["effective_concurrency"])
    resolved["estimated_tokens_per_request"] = int(rate_limit_cfg["estimated_tokens_per_request"])
    resolved["rate_limit_reserve_ratio"] = float(rate_limit_cfg["reserve_ratio"])
    resolved["rate_limit_concurrency_applied"] = bool(rate_limit_cfg["rate_limit_concurrency_applied"])
    request_extra_body = dict(model_controls.get("extra_body", {}) or {})
    if request_extra_body:
        existing_extra = dict(resolved.get("request_extra_body", {}) or {})
        existing_extra.update(request_extra_body)
        resolved["request_extra_body"] = existing_extra
    if str(override_payload.get("request_model", "") or "").strip():
        resolved["request_model"] = str(override_payload["request_model"]).strip()
    return resolved


def _is_retryable_request_failure(error_text: str) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return True
    retry_markers = [
        "timeout",
        "timed out",
        "empty_model_response",
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


def _run_single_sample_request(sample: dict[str, Any], *, api_cfg: dict[str, Any], limiter: ApiRateLimiter, cancel_event: threading.Event | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledExperimentError("experiment_cancelled")
    messages, uploads = _build_openai_messages(sample, api_cfg)
    system_prompt = str(sample.get("system_prompt", "") or "").strip() or _build_system_prompt(sample)
    user_prompt = str(sample.get("user_prompt", "") or sample.get("prompt_text", "") or "").strip()
    include_keyframes = bool(sample.get("include_keyframes", False))
    limiter.acquire(estimated_tokens=_estimate_request_tokens(sample, include_keyframes), cancel_event=cancel_event)
    client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["api_base"])
    started = time.perf_counter()
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
        attempt_started = time.perf_counter()
        should_retry = False
        try:
            kwargs: dict[str, Any] = {
                "model": api_cfg["request_model"],
                "messages": messages,
                "temperature": float(api_cfg["temperature"]),
                "timeout": float(api_cfg["timeout_s"]),
            }
            extra_body: dict[str, Any] = dict(api_cfg.get("request_extra_body", {}) or {})
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            latency_ms += (time.perf_counter() - attempt_started) * 1000.0
            raw_text, raw_meta = _extract_response_text(resp)
            if str(raw_text or "").strip():
                break
            request_status = "error"
            error_text = "empty_model_response"
            attempt_errors.append(error_text)
            raw_meta = {**dict(raw_meta or {}), "error": error_text}
            should_retry = _is_retryable_request_failure(error_text)
        except Exception as exc:
            latency_ms += (time.perf_counter() - attempt_started) * 1000.0
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
    payload = _extract_json_object(raw_text)
    task_name = str(sample.get("task_name", sample.get("form", "")) or "")
    pred_count = -999
    pred_count_raw = payload.get("visible_count", None)
    if str(pred_count_raw).strip():
        try:
            pred_count = int(pred_count_raw)
            if pred_count < 0:
                pred_count = -999
        except Exception:
            pred_count = -999
    pred_intervals = _normalize_interval_rows(payload.get("visible_intervals_sec", payload.get("intervals_sec", [])))
    pred_keyframes = _normalize_keyframe_rows(payload.get("keyframes", []))
    pred_targets = []
    for row in list(payload.get("targets", []) or []):
        if not isinstance(row, dict):
            continue
        pred_targets.append(
            {
                "target_id": str(row.get("target_id", "") or ""),
                "visible_count": int(row.get("visible_count", 0) or 0),
                "visible_intervals_sec": _normalize_interval_rows(row.get("visible_intervals_sec", row.get("intervals_sec", []))),
                "keyframes": _normalize_keyframe_rows(row.get("keyframes", [])),
            }
        )
    option_label_map = {
        str(row.get("option_id", "") or ""): str(row.get("label", "") or "")
        for row in list(sample.get("choice_options", []) or [])
        if str(row.get("option_id", "") or "").strip()
    }
    pred_answer_items = []
    for row in list(payload.get("answers", []) or []):
        if not isinstance(row, dict):
            continue
        opt_id = str(row.get("option_id", "") or "").strip()
        if not opt_id:
            continue
        pred_answer_items.append(
            {
                "option_id": opt_id,
                "label": option_label_map.get(opt_id, ""),
                "intervals_sec": _normalize_interval_rows(row.get("intervals_sec", [])),
            }
        )
    if not pred_answer_items:
        pred_option_id = str(payload.get("answer_option_id", "") or "").strip()
        pred_option_ids = _normalize_choice_list(payload.get("answer_option_ids", []))
        if pred_option_id:
            pred_answer_items = [{"option_id": pred_option_id, "label": option_label_map.get(pred_option_id, ""), "intervals_sec": []}]
        elif pred_option_ids:
            pred_answer_items = [{"option_id": x, "label": option_label_map.get(x, ""), "intervals_sec": []} for x in pred_option_ids]
    pred_answer_items = _normalize_answer_items(pred_answer_items)
    pred_option_ids = [str(row.get("option_id", "") or "") for row in pred_answer_items if str(row.get("option_id", "") or "").strip()]
    pred_option_id = pred_option_ids[0] if pred_option_ids else ""
    pred_choice_intervals_flat = [pair for row in pred_answer_items for pair in _normalize_interval_rows(row.get("intervals_sec", []))]
    request_row = {
        "sample_id": sample["sample_id"],
        "model": api_cfg["model"],
        "request_model": api_cfg["request_model"],
        "api_source": api_cfg.get("api_source"),
        "api_base": api_cfg.get("api_base"),
        "reasoning_mode": api_cfg.get("reasoning_mode"),
        "form": sample["form"],
        "assistant_prefill": api_cfg.get("assistant_prefill", ""),
        "system_prompt_prefix": api_cfg.get("system_prompt_prefix", ""),
        "system_prompt_as_blocks": bool(api_cfg.get("system_prompt_as_blocks", False)),
        "request_extra_body": dict(api_cfg.get("request_extra_body", {}) or {}),
        "messages_preview": _messages_preview(messages),
        "inputs": uploads,
    }
    usage = dict(raw_meta.get("usage", {}) or {}) if isinstance(raw_meta.get("usage", {}), dict) else {}
    response_row = {
        "sample_id": sample["sample_id"],
        "request_status": request_status,
        "latency_ms": latency_ms,
        "raw_text": raw_text,
        "raw_response": raw_meta,
        "usage": usage,
    }
    gold_answer_items = _normalize_answer_items(sample.get("answer_items", []))
    gold_choice_intervals_flat = [pair for row in gold_answer_items for pair in _normalize_interval_rows(row.get("intervals_sec", []))]
    self_temporal_mean_tiou = None
    if bool(sample.get("include_temporal_localization", False)) and gold_choice_intervals_flat:
        self_temporal_mean_tiou = sum(
            max((_interval_iou(pred_pair, gold_pair) for pred_pair in pred_choice_intervals_flat), default=0.0)
            for gold_pair in gold_choice_intervals_flat
        ) / float(len(gold_choice_intervals_flat))
    joint_level_payloads = {}
    if task_name == "self_instance_recognition_joint":
        joint_level_payloads = _derive_joint_level_payloads(
            sample=sample,
            gold_answer_items=gold_answer_items,
            pred_answer_items=pred_answer_items,
        )
    parsed_row = {
        "sample_id": sample["sample_id"],
        "scene_id": sample["scene_id"],
        "engine": sample["engine"],
        "mode": sample["mode"],
        "task_group": sample.get("task_group"),
        "task_name": sample.get("task_name", sample.get("form")),
        "task_display_name": sample.get("task_display_name", ""),
        "set_id": sample.get("set_id", ""),
        "set_name": sample.get("set_name", ""),
        "element_instance_ids": list(sample.get("element_instance_ids", []) or []),
        "mission_type": sample.get("mission_type", sample.get("set_name", "")),
        "mission_subtype": sample.get("mission_subtype", sample.get("set_id", "")),
        "service_scenario": sample.get("service_scenario", sample.get("mode", "")),
        "landmark_category": sample["landmark_category"],
        "difficulty_band": sample["difficulty_band"],
        "form": sample["form"],
        "gold_option_id": sample.get("answer_option_id", ""),
        "pred_option_id": pred_option_id,
        "gold_option_ids": list(sample.get("answer_option_ids", []) or []),
        "pred_option_ids": pred_option_ids,
        "gold_answer_items": gold_answer_items,
        "pred_answer_items": pred_answer_items,
        "gold_choice_intervals_sec": gold_choice_intervals_flat,
        "pred_choice_intervals_sec": pred_choice_intervals_flat,
        "self_temporal_mean_tiou": self_temporal_mean_tiou,
        "include_temporal_localization": bool(sample.get("include_temporal_localization", False)),
        "gold_visible_count": int(sample["visible_count"]),
        "pred_visible_count": pred_count,
        "gold_visible_intervals_sec": list(sample["visible_intervals_sec"]),
        "pred_visible_intervals_sec": pred_intervals,
        "gold_environment_answer": sample.get("gold_environment_answer"),
        "pred_environment_answer": {"targets": pred_targets} if pred_targets else None,
        "gold_keyframes_eval": list(sample.get("keyframe_eval_view_whole_second", []) or [])
        if bool(sample.get("include_keyframes", False))
        else [],
        "pred_keyframes": pred_keyframes if bool(sample.get("include_keyframes", False)) else [],
        "include_keyframes": bool(sample.get("include_keyframes", False)),
        "parse_ok": bool(request_status == "ok" and isinstance(payload, dict) and payload),
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        **joint_level_payloads,
    }
    return request_row, {"response": response_row, "parsed": parsed_row}


def _score_keyframes(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    if not gold:
        return {"bbox_acc@50iou": None, "bbox_mean_iou": None, "keyframe_hit_rate": None}
    hits = 0
    ious: list[float] = []
    for row in gold:
        gt_t = float(row.get("time_sec", 0.0) or 0.0)
        best = None
        best_dt = None
        for pr in pred:
            dt = abs(float(pr.get("time_sec", 0.0) or 0.0) - gt_t)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = pr
        if best is None or (best_dt is not None and best_dt > 0.51):
            ious.append(0.0)
            continue
        iou = _bbox_iou(list(row.get("bbox_xyxy_norm", []) or []), list(best.get("bbox_xyxy_norm", []) or []))
        ious.append(iou)
        if iou >= 0.5:
            hits += 1
    return {
        "bbox_acc@50iou": float(hits) / float(len(gold)),
        "bbox_mean_iou": sum(ious) / float(len(ious)) if ious else 0.0,
        "keyframe_hit_rate": float(sum(1 for v in ious if v > 0.0)) / float(len(gold)),
    }


def _summarize_predictions(rows: list[dict[str, Any]], *, include_grouped: bool = True) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"count": 0}
    env_rows = [row for row in rows if str(row.get("task_group", "")) == "environmental"]
    self_rows = [row for row in rows if str(row.get("task_group", "")) == "self-state"]
    def _valid_pred_visible_count(row: dict[str, Any]) -> int | None:
        if not bool(row.get("parse_ok", False)):
            return None
        try:
            value = int(row.get("pred_visible_count", -999))
        except Exception:
            return None
        if value < 0:
            return None
        return value

    count_exact = 0
    count_within1 = 0
    mae_values: list[float] = []
    for row in env_rows:
        pred_count = _valid_pred_visible_count(row)
        gold_count = int(row.get("gold_visible_count", -1))
        if pred_count is None:
            continue
        if pred_count == gold_count:
            count_exact += 1
        if abs(pred_count - gold_count) <= 1:
            count_within1 += 1
        mae_values.append(abs(pred_count - gold_count))
    parse_ok = sum(1 for row in rows if bool(row.get("parse_ok", False)))
    interval_p30 = interval_r30 = interval_p50 = interval_r50 = 0.0
    best_tious: list[float] = []
    count_consistency = 0
    bbox_accs = []
    bbox_ious = []
    bbox_hits = []
    for row in env_rows:
        gold = list(row.get("gold_visible_intervals_sec", []) or [])
        pred = list(row.get("pred_visible_intervals_sec", []) or [])
        gold_pairs = [[float(x.get("start_sec", 0.0) or 0.0), float(x.get("end_sec", 0.0) or 0.0)] for x in gold if isinstance(x, dict)]
        pred_pairs = _normalize_interval_rows(pred)
        tp30, fp30, fn30 = _match_interval_metrics(gold_pairs, pred_pairs, 0.3)
        tp50, fp50, fn50 = _match_interval_metrics(gold_pairs, pred_pairs, 0.5)
        interval_p30 += float(tp30) / float(max(1, tp30 + fp30))
        interval_r30 += float(tp30) / float(max(1, tp30 + fn30))
        interval_p50 += float(tp50) / float(max(1, tp50 + fp50))
        interval_r50 += float(tp50) / float(max(1, tp50 + fn50))
        if gold_pairs:
            best_tious.append(sum(max((_interval_iou(p, g) for p in pred_pairs), default=0.0) for g in gold_pairs) / float(len(gold_pairs)))
        count_consistency += 1 if len(pred_pairs) == len(gold_pairs) else 0
        if row.get("include_keyframes"):
            key_metrics = _score_keyframes(list(row.get("gold_keyframes_eval", []) or []), list(row.get("pred_keyframes", []) or []))
            if key_metrics["bbox_acc@50iou"] is not None:
                bbox_accs.append(float(key_metrics["bbox_acc@50iou"]))
                bbox_ious.append(float(key_metrics["bbox_mean_iou"]))
                bbox_hits.append(float(key_metrics["keyframe_hit_rate"]))
    def _avg(values: list[float]) -> float | None:
        return (sum(values) / float(len(values))) if values else None
    env_total = len(env_rows)
    f1_30 = (2.0 * _avg([interval_p30 / env_total]) * _avg([interval_r30 / env_total]) / max(1e-6, (_avg([interval_p30 / env_total]) or 0) + (_avg([interval_r30 / env_total]) or 0))) if env_total else None
    f1_50 = (2.0 * _avg([interval_p50 / env_total]) * _avg([interval_r50 / env_total]) / max(1e-6, (_avg([interval_p50 / env_total]) or 0) + (_avg([interval_r50 / env_total]) or 0))) if env_total else None
    self_set_rows = [row for row in self_rows if str(row.get("task_name", "")) in {"self_set_instance_recognition", "self_composite_instance_recognition"}]
    self_joint_rows = [row for row in self_rows if str(row.get("task_name", "")) == "self_instance_recognition_joint"]
    self_element_choice_rows = [row for row in self_rows if str(row.get("task_name", "")) in {"self_element_instance_recognition", "self_atomic_instance_recognition"}]
    if self_joint_rows:
        self_set_rows = _joint_rows_to_level_rows(self_joint_rows, level="composite")
        self_element_choice_rows = _joint_rows_to_level_rows(self_joint_rows, level="atomic")
    self_set_acc = (
        sum(1 for row in self_set_rows if str(row.get("pred_option_id", "")) == str(row.get("gold_option_id", ""))) / float(len(self_set_rows))
        if self_set_rows
        else None
    )
    if self_joint_rows or self_element_choice_rows:
        precs = []
        recs = []
        for row in [*self_joint_rows, *self_element_choice_rows]:
            gold = set(str(x) for x in list(row.get("gold_option_ids", []) or []) if str(x).strip())
            pred = set(str(x) for x in list(row.get("pred_option_ids", []) or []) if str(x).strip())
            if not gold and not pred:
                precs.append(1.0)
                recs.append(1.0)
                continue
            tp = len(gold & pred)
            precs.append(float(tp) / float(max(1, len(pred))))
            recs.append(float(tp) / float(max(1, len(gold))))
        self_element_precision = _avg(precs)
        self_element_recall = _avg(recs)
        self_element_f1 = (
            2.0 * float(self_element_precision or 0.0) * float(self_element_recall or 0.0) / max(1e-6, float(self_element_precision or 0.0) + float(self_element_recall or 0.0))
            if self_element_precision is not None and self_element_recall is not None
            else None
        )
    else:
        self_element_precision = self_element_recall = self_element_f1 = None
    self_loc_rows = [row for row in self_rows if bool(row.get("include_temporal_localization", False))]
    if self_loc_rows:
        loc_ps = []
        loc_rs = []
        loc_tious = []
        for row in self_loc_rows:
            gold = _normalize_interval_rows(row.get("gold_choice_intervals_sec", []))
            pred = _normalize_interval_rows(row.get("pred_choice_intervals_sec", []))
            tp, fp, fn = _match_interval_metrics(gold, pred, 0.5)
            loc_ps.append(float(tp) / float(max(1, tp + fp)))
            loc_rs.append(float(tp) / float(max(1, tp + fn)))
            if gold:
                loc_tious.append(
                    sum(max((_interval_iou(pred_pair, gold_pair) for pred_pair in pred), default=0.0) for gold_pair in gold)
                    / float(len(gold))
                )
        self_loc_precision = _avg(loc_ps)
        self_loc_recall = _avg(loc_rs)
        self_loc_f1 = (
            2.0 * float(self_loc_precision or 0.0) * float(self_loc_recall or 0.0) / max(1e-6, float(self_loc_precision or 0.0) + float(self_loc_recall or 0.0))
            if self_loc_precision is not None and self_loc_recall is not None
            else None
        )
        self_loc_tiou = _avg(loc_tious)
    else:
        self_loc_precision = self_loc_recall = self_loc_f1 = self_loc_tiou = None
    out = {
        "count": total,
        "parse_success_rate": float(parse_ok) / float(total),
        "environmental_count": len(env_rows),
        "self_state_count": len(self_rows),
        "count_exact_acc": float(count_exact) / float(env_total) if env_total else None,
        "count_within1_acc": float(count_within1) / float(env_total) if env_total else None,
        "count_mae": _avg([float(v) for v in mae_values]) if mae_values else None,
        "segment_precision@0.3": interval_p30 / float(env_total) if env_total else None,
        "segment_recall@0.3": interval_r30 / float(env_total) if env_total else None,
        "segment_f1@0.3": f1_30,
        "segment_precision@0.5": interval_p50 / float(env_total) if env_total else None,
        "segment_recall@0.5": interval_r50 / float(env_total) if env_total else None,
        "segment_f1@0.5": f1_50,
        "mean_best_tIoU": _avg(best_tious),
        "count_consistency": float(count_consistency) / float(env_total) if env_total else None,
        "bbox_acc@50iou": _avg(bbox_accs),
        "bbox_mean_iou": _avg(bbox_ious),
        "keyframe_hit_rate": _avg(bbox_hits),
        "set_instance_acc": self_set_acc,
        "element_instance_precision": self_element_precision,
        "element_instance_recall": self_element_recall,
        "element_instance_f1": self_element_f1,
        "self_temporal_loc_precision@0.5": self_loc_precision,
        "self_temporal_loc_recall@0.5": self_loc_recall,
        "self_temporal_loc_f1@0.5": self_loc_f1,
        "self_temporal_loc_mean_tIoU": self_loc_tiou,
    }
    if include_grouped:
        display_rows = [*env_rows, *self_set_rows, *self_element_choice_rows]

        def _main_metric_key(form_name: str) -> str:
            if str(form_name) in {"self_set_instance_recognition", "self_composite_instance_recognition"}:
                return "set_instance_acc"
            if str(form_name) in {"self_element_instance_recognition", "self_atomic_instance_recognition"}:
                return "element_instance_f1"
            # Environmental count tasks only score exact visible-count matches.
            return "count_exact_acc"

        grouped_payload: dict[str, Any] = {}
        for key in ["mode", "form", "difficulty_band"]:
            bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in display_rows:
                bucket[str(row.get(key, "") or "")].append(row)
            grouped_payload[key] = {}
            for name, items in bucket.items():
                summary = _summarize_predictions(items, include_grouped=False)
                form_name = str(items[0].get("form", "") or "")
                grouped_payload[key][name] = {
                    "count": summary.get("count", 0),
                    "main_metric": summary.get(_main_metric_key(form_name)),
                    "count_within1_acc": summary.get("count_within1_acc"),
                    "segment_f1@0.5": summary.get("segment_f1@0.5"),
                    "mean_best_tIoU": summary.get("mean_best_tIoU"),
                    "bbox_acc@50iou": summary.get("bbox_acc@50iou"),
                    "set_instance_acc": summary.get("set_instance_acc"),
                    "element_instance_f1": summary.get("element_instance_f1"),
                    "self_temporal_loc_f1@0.5": summary.get("self_temporal_loc_f1@0.5"),
                    "self_temporal_loc_mean_tIoU": summary.get("self_temporal_loc_mean_tIoU"),
                }
        combo_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        combo_diff_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in display_rows:
            mode_name = str(row.get("mode", "") or "")
            form_name = str(row.get("form", "") or "")
            diff_name = str(row.get("difficulty_band", "") or "")
            combo_bucket[f"{mode_name}|{form_name}"].append(row)
            combo_diff_bucket[f"{mode_name}|{form_name}|{diff_name}"].append(row)
        grouped_payload["combo"] = {}
        for name, items in combo_bucket.items():
            summary = _summarize_predictions(items, include_grouped=False)
            form_name = str(items[0].get("form", "") or "")
            grouped_payload["combo"][name] = {
                "count": summary.get("count", 0),
                "main_metric": summary.get(_main_metric_key(form_name)),
                "count_within1_acc": summary.get("count_within1_acc"),
                "segment_f1@0.5": summary.get("segment_f1@0.5"),
                "mean_best_tIoU": summary.get("mean_best_tIoU"),
                "bbox_acc@50iou": summary.get("bbox_acc@50iou"),
                "set_instance_acc": summary.get("set_instance_acc"),
                "element_instance_f1": summary.get("element_instance_f1"),
                "self_temporal_loc_f1@0.5": summary.get("self_temporal_loc_f1@0.5"),
                "self_temporal_loc_mean_tIoU": summary.get("self_temporal_loc_mean_tIoU"),
            }
        grouped_payload["combo_by_difficulty"] = {}
        for name, items in combo_diff_bucket.items():
            summary = _summarize_predictions(items, include_grouped=False)
            form_name = str(items[0].get("form", "") or "")
            grouped_payload["combo_by_difficulty"][name] = {
                "count": summary.get("count", 0),
                "main_metric": summary.get(_main_metric_key(form_name)),
                "count_within1_acc": summary.get("count_within1_acc"),
                "segment_f1@0.5": summary.get("segment_f1@0.5"),
                "mean_best_tIoU": summary.get("mean_best_tIoU"),
                "bbox_acc@50iou": summary.get("bbox_acc@50iou"),
                "set_instance_acc": summary.get("set_instance_acc"),
                "element_instance_f1": summary.get("element_instance_f1"),
                "self_temporal_loc_f1@0.5": summary.get("self_temporal_loc_f1@0.5"),
                "self_temporal_loc_mean_tIoU": summary.get("self_temporal_loc_mean_tIoU"),
            }
        out["grouped"] = grouped_payload
    return out

def run_experiment_once(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    manifest_path: Path,
    model: str | None,
    limit: int | None,
    api_overrides: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai package is required")
    manifest = _read_json(manifest_path, default={})
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest_load_failed")
    layout = _resolve_stage3_layout(config=config, scene_id=scene_id, engine=engine)
    experiments_root = _ensure_dir(layout["experiments_root"])
    samples = list(manifest.get("samples", []) or [])
    if limit and int(limit) > 0:
        samples = samples[: int(limit)]
    api_cfg = _resolve_api_cfg(config, override_model=model, overrides=api_overrides)
    manifest_tag = _safe_name(Path(manifest_path).stem)[:48]
    unique_experiment = bool((api_overrides or {}).get("unique_experiment", False))
    run_id = f"{scene_id}_{_safe_name(api_cfg['model'])}_{manifest_tag}_unique" if unique_experiment else f"{scene_id}_{_safe_name(api_cfg['model'])}_{manifest_tag}_{_now_ts()}"
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
    keyframe_cache_root = _ensure_dir(layout["cache_root"] / "keyframe_eval_view" / run_id)
    include_keyframes_for_prompt = bool((api_overrides or {}).get("include_keyframes", (_stage3_cfg(config).get("experiment_defaults", {}) or {}).get("include_keyframes", False)))
    samples_runtime: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        if unique_experiment and str(sample.get("sample_id", "") or "").strip() in completed_sample_ids:
            continue
        sample = dict(sample)
        sample["_sample_index"] = int(sample_index)
        sample["_sample_ordinal"] = int(sample_index) + 1
        sample["include_keyframes"] = include_keyframes_for_prompt
        keyframes_whole_second = _whole_second_keyframes(
            {
                "keyframe_gt_dense": list(sample.get("keyframe_gt_dense", []) or []),
                "visible_intervals_sec": list(sample.get("visible_intervals_sec", []) or []),
                "fps": float(sample.get("fps", 24.0) or 24.0),
                "frame_count": int(sample.get("frame_count", 0) or 0),
            },
            config=config,
        )
        sample["keyframe_eval_view_whole_second"] = keyframes_whole_second
        sample["prompt_text"] = _build_prompt(
            sample,
            provide_flight_description=bool((api_overrides or {}).get("provide_flight_description", (_stage3_cfg(config).get("experiment_defaults", {}) or {}).get("provide_flight_description", True))),
            include_keyframes=include_keyframes_for_prompt,
        )
        cache_path = keyframe_cache_root / f"{sample['sample_id']}.json"
        _write_json(cache_path, {"whole_second_keyframes": keyframes_whole_second})
        samples_runtime.append(sample)
    dynamic_estimated_tokens = max(
        [_estimate_request_tokens(sample, include_keyframes_for_prompt) for sample in samples_runtime] or [max(3200, int(api_cfg.get("max_tokens", 600) or 600) + 3600)]
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
        total = len(samples)
        done = len(completed_sample_ids)
        write_lock = threading.Lock()
        failed_request_rows: list[dict[str, Any]] = []
        if unique_experiment and not samples_runtime:
            pass
        with ThreadPoolExecutor(max_workers=max(1, int(api_cfg.get("concurrency", 1)))) as executor:
            future_to_sample = {
                executor.submit(_run_single_sample_request, sample, api_cfg=api_cfg, limiter=limiter, cancel_event=cancel_event): sample
                for sample in samples_runtime
            }
            try:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    worker_id = int((done % max(1, int(api_cfg.get("concurrency", 1)))) + 1)
                    if cancel_event is not None and cancel_event.is_set():
                        raise CancelledExperimentError("experiment_cancelled")
                    try:
                        request_row, payload = future.result()
                    except CancelledExperimentError:
                        raise
                    except Exception as exc:
                        request_row = {
                            "sample_id": sample["sample_id"],
                            "model": api_cfg["model"],
                            "request_model": api_cfg["request_model"],
                            "api_source": api_cfg.get("api_source"),
                            "api_base": api_cfg.get("api_base"),
                            "reasoning_mode": api_cfg.get("reasoning_mode"),
                            "form": sample["form"],
                            "assistant_prefill": api_cfg.get("assistant_prefill", ""),
                            "system_prompt_prefix": api_cfg.get("system_prompt_prefix", ""),
                            "system_prompt_as_blocks": bool(api_cfg.get("system_prompt_as_blocks", False)),
                            "request_extra_body": dict(api_cfg.get("request_extra_body", {}) or {}),
                            "messages_preview": [],
                            "inputs": [],
                        }
                        payload = {
                            "response": {"sample_id": sample["sample_id"], "request_status": "error", "latency_ms": None, "raw_text": str(exc), "raw_response": {"error": str(exc)}},
                            "parsed": {
                                "sample_id": sample["sample_id"],
                                "scene_id": sample["scene_id"],
                                "engine": sample["engine"],
                                "mode": sample["mode"],
                                "task_group": sample.get("task_group"),
                                "task_name": sample.get("task_name", sample.get("form")),
                                "mission_type": sample.get("mission_type", sample.get("set_name", "")),
                                "mission_subtype": sample.get("mission_subtype", sample.get("set_id", "")),
                                "service_scenario": sample.get("service_scenario", sample.get("mode", "")),
                                "landmark_category": sample["landmark_category"],
                                "difficulty_band": sample["difficulty_band"],
                                "form": sample["form"],
                                "gold_visible_count": int(sample["visible_count"]),
                                "pred_visible_count": -999,
                                "gold_visible_intervals_sec": list(sample["visible_intervals_sec"]),
                                "pred_visible_intervals_sec": [],
                                "gold_keyframes_eval": list(sample.get("keyframe_eval_view_whole_second", []) or [])
                                if bool(sample.get("include_keyframes", False))
                                else [],
                                "pred_keyframes": [],
                                "include_keyframes": bool(sample.get("include_keyframes", False)),
                                "parse_ok": False,
                                "latency_ms": None,
                            },
                        }
                    done += 1
                    response_row = {"run_id": run_id, **payload["response"]}
                    parsed_row = {"run_id": run_id, **payload["parsed"]}
                    with write_lock:
                        req_fp.write(json.dumps({"run_id": run_id, **request_row}, ensure_ascii=False) + "\n")
                        resp_fp.write(json.dumps(response_row, ensure_ascii=False) + "\n")
                        pred_fp.write(json.dumps(parsed_row, ensure_ascii=False) + "\n")
                        req_fp.flush()
                        resp_fp.flush()
                        pred_fp.flush()
                        req_txt_fp.write(
                            f"[{_iso_now()}] sample_id={sample['sample_id']} form={sample['form']} model={api_cfg['model']}\n"
                            f"request_model={api_cfg['request_model']} api_source={api_cfg.get('api_source','')} reasoning_mode={api_cfg.get('reasoning_mode','')}\n"
                            f"system_prompt_prefix:\n{request_row.get('system_prompt_prefix','')}\n"
                            f"system_prompt_as_blocks={json.dumps(request_row.get('system_prompt_as_blocks', False), ensure_ascii=False)}\n"
                            f"assistant_prefill:\n{request_row.get('assistant_prefill','')}\n"
                            f"request_extra_body={json.dumps(request_row.get('request_extra_body', {}), ensure_ascii=False)}\n"
                            f"messages_preview={json.dumps(request_row.get('messages_preview', []), ensure_ascii=False)}\n"
                            f"inputs={json.dumps(request_row.get('inputs', []), ensure_ascii=False)}\n\n"
                        )
                        resp_txt_fp.write(
                            f"[{_iso_now()}] sample_id={sample['sample_id']} status={response_row['request_status']} latency_ms={response_row['latency_ms']} "
                            f"prompt_tokens={parsed_row.get('prompt_tokens')} completion_tokens={parsed_row.get('completion_tokens')} total_tokens={parsed_row.get('total_tokens')}\n"
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
                                "form": str(sample.get("form", "") or ""),
                                "task_name": str(sample.get("task_name", sample.get("form", "")) or ""),
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
                                "form": sample["form"],
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
            "form": str(sample.get("form", "") or ""),
            "task_name": str(sample.get("task_name", sample.get("form", "")) or ""),
        }
        for sample in samples_runtime
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
                "form": "",
                "task_name": "",
            }
        raw_response = dict(response_row.get("raw_response", {}) or {}) if isinstance(response_row.get("raw_response", {}), dict) else {}
        meta["request_retry_attempts"] = int(raw_response.get("request_retry_attempts", api_cfg.get("request_retry_attempts", 0)) or 0)
        meta["request_retry_errors"] = list(raw_response.get("request_retry_errors", []) or [])
        failed_request_rows_final.append(meta)
    failed_request_rows_final.sort(key=lambda row: int(row.get("sample_index", 0) or 0))
    failed_indices_path = run_dir / "failed_request_indices.json"
    _write_json(
        failed_indices_path,
        {
            "run_id": run_id,
            "stage": "stage3",
            "scene_id": scene_id,
            "engine": engine,
            "model": api_cfg["model"],
            "manifest_path": _path_for_json(manifest_path),
            "request_retry_attempts": int(api_cfg.get("request_retry_attempts", 0) or 0),
            "failed_count": len(failed_request_rows_final),
            "failed_sample_indices": [int(row.get("sample_index", 0) or 0) for row in failed_request_rows_final],
            "failed_samples": failed_request_rows_final,
        },
    )
    report_path = run_dir / "report.json"
    _write_json(report_path, report)
    latest_path = experiments_root / f"{scene_id}.latest_report.json"
    _write_json(latest_path, report)
    return {"run_id": run_id, "report_path": report_path, "report": report}


def _list_manifests(layout: dict[str, Path], scene_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(layout["datasets_root"].glob(f"{scene_id}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _read_json(path, default={})
        if not isinstance(payload, dict) or "samples" not in payload:
            continue
        rows.append({"path": _path_for_json(path), "generated_at": payload.get("generated_at"), "summary": payload.get("summary", _build_manifest_summary(list(payload.get("samples", []) or [])))})
    return rows


def _list_reports(layout: dict[str, Path], scene_id: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(layout["experiments_root"].glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = _read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("scene_id", "")) != str(scene_id):
            continue
        manifest_path = str(payload.get("manifest_path", "") or "")
        summary = dict(payload.get("summary", {}) or {})
        if "grouped" not in summary:
            if isinstance(summary, dict) and summary.get("count") is not None:
                pass
            else:
                report_rows = _load_report_rows(path)
                if report_rows:
                    summary = _summarize_predictions(report_rows)
        rows.append({
            "path": _path_for_json(path),
            "generated_at": payload.get("generated_at"),
            "model": payload.get("model"),
            "run_id": payload.get("run_id", path.parent.name),
            "manifest_path": manifest_path,
            "manifest_name": Path(manifest_path).name if manifest_path else "-",
            "summary": summary,
        })
    return rows


def _resolve_stage3_task_pipeline_layout(config: dict[str, Any], scene_id: str, engine: str, task_name: str) -> dict[str, Path] | None:
    task_name_value = str(task_name or "").strip()
    if not task_name_value:
        return None
    cfg = dict(config)
    cfg["task_pipeline"] = {"task_name": task_name_value, "root_dir": "task_pipeline_data"}
    layout = _resolve_stage3_layout(cfg, scene_id, engine)
    return layout if layout["stage3_root"].exists() else None


def _stage3_layout_candidates(config: dict[str, Any], scene_id: str, engine: str, task_name: str | None) -> list[dict[str, Path]]:
    layouts: list[dict[str, Path]] = []
    pipeline_layout = _resolve_stage3_task_pipeline_layout(config, scene_id, engine, str(task_name or ""))
    if pipeline_layout is not None:
        layouts.append(pipeline_layout)
    layouts.append(_resolve_stage3_layout(config, scene_id, engine))
    seen: set[str] = set()
    unique: list[dict[str, Path]] = []
    for layout in layouts:
        key = str(layout["stage3_root"].resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(layout)
    return unique


def _list_manifests_multi(layouts: list[dict[str, Path]], scene_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layout in layouts:
        for row in _list_manifests(layout, scene_id):
            key = str(row.get("path", "") or "")
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    return rows


def _list_reports_multi(layouts: list[dict[str, Path]], scene_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layout in layouts:
        for row in _list_reports(layout, scene_id):
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


def _stage3_best_run_progress(layouts: list[dict[str, Path]], scene_id: str, model: str, manifest_path: str) -> dict[str, Any]:
    model_tag = _safe_name(str(model or ""))
    manifest_tag = _safe_name(Path(str(manifest_path or "")).stem)[:48] if str(manifest_path or "").strip() else ""
    prefix = f"{scene_id}_{model_tag}_{manifest_tag}" if manifest_tag else f"{scene_id}_{model_tag}_"
    best_completed = 0
    best_run_dir = ""
    best_mtime = -1.0
    for layout in layouts:
        experiments_root = Path(layout.get("experiments_root", ""))
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


def _discover_stage3_models_from_runs(layouts: list[dict[str, Path]], scene_id: str) -> set[str]:
    models: set[str] = set()
    for layout in layouts:
        experiments_root = Path(layout.get("experiments_root", ""))
        if not experiments_root.exists():
            continue
        for run_dir in experiments_root.iterdir():
            if not run_dir.is_dir():
                continue
            if not str(run_dir.name).startswith(f"{scene_id}_"):
                continue
            report_path = run_dir / "report.json"
            if report_path.exists():
                payload = _read_json(report_path, default={})
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


def _build_stage3_experiment_progress_matrix(*, catalog: list[dict[str, Any]], selected_engine: str, selected_scene_id: str, task_name: str | None, fallback_config_path: Path) -> dict[str, Any]:
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
        if str(task_name or "").strip() and _resolve_stage3_task_pipeline_layout(cfg, scene_value, selected_engine, str(task_name or "")) is None:
            continue
        layouts = _stage3_layout_candidates(cfg, scene_value, selected_engine, str(task_name or ""))
        if not layouts:
            continue
        latest_manifest_path = layouts[0]["datasets_root"] / f"{scene_value}.latest_manifest.json"
        manifest_path = str(_path_for_json(latest_manifest_path) if latest_manifest_path.exists() else "")
        total_samples = _load_manifest_sample_count(latest_manifest_path) if latest_manifest_path.exists() else 0
        reports = _list_reports_multi(layouts, scene_value)
        known_models.update(str(row.get("model", "") or "").strip() for row in reports if str(row.get("model", "") or "").strip())
        known_models.update(_discover_stage3_models_from_runs(layouts, scene_value))
        if not manifest_path and not reports and not any(Path(layout.get("experiments_root", "")).exists() for layout in layouts):
            continue
        contexts.append({
            "scene_id": scene_value,
            "layouts": layouts,
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
            progress = _stage3_best_run_progress(ctx["layouts"], ctx["scene_id"], model, str(ctx.get("manifest_path", "") or ""))
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


def _global_stage3_reports(
    *,
    catalog: list[dict[str, Any]],
    selected_engine: str,
    task_name: str | None,
    fallback_config_path: Path,
) -> list[dict[str, Any]]:
    reports_all: list[dict[str, Any]] = []
    items = _catalog_items_for_engine(catalog, selected_engine)
    for item in items:
        scene_value = str(item.get("scene_id", "") or "").strip()
        if not scene_value:
            continue
        cfg, _ = _load_scene_config_from_catalog(
            engine=selected_engine,
            scene_id=scene_value,
            catalog=catalog,
            fallback_config_path=fallback_config_path,
        )
        if str(task_name or "").strip() and _resolve_stage3_task_pipeline_layout(cfg, scene_value, selected_engine, str(task_name or "")) is None:
            continue
        layouts = _stage3_layout_candidates(cfg, scene_value, selected_engine, str(task_name or ""))
        if not layouts:
            continue
        reports_all.extend({**row, "scene_id": scene_value} for row in _list_reports_multi(layouts, scene_value))
    reports_all.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    return reports_all


def _catalog_items_for_engine(catalog: list[dict[str, Any]], engine: str) -> list[dict[str, Any]]:
    engine_value = str(engine or "").strip().lower()
    return [dict(item) for item in list(catalog or []) if str(item.get("engine", "") or "").strip().lower() == engine_value]


def _global_stage3_payload(
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
        stage3_cfg = _stage3_cfg(probe)
        ui_defaults = {
            "auto_pick_single_count": 1,
            "auto_pick_multi_count": max(2, int(stage3_cfg.get("multi_landmark_max_secondary", 2) or 2) + 1),
            "auto_pick_min_points": int(stage3_cfg.get("auto_pick_min_points", 500) or 500),
            "mission_count": int(stage3_cfg.get("mission_count_default", 1) or 1),
            "auto_set_rule": str(stage3_cfg.get("auto_set_rule_default", "heuristic") or "heuristic"),
            "allow_interleave_repeat": bool(stage3_cfg.get("allow_interleave_repeat_default", False)),
            "max_total_elements": int(stage3_cfg.get("max_total_elements_default", 0) or 0),
        }
        return {
            "candidate_stats": {
                "candidate_count": 0,
                "approved_count": 0,
                "category_count": 0,
                "avg_visible_count": 0.0,
                "single_landmark_count": 0,
                "multi_landmark_count": 0,
                "mission_families": [],
            },
            "latest_manifest_path": None,
            "latest_report_path": None,
            "ui_defaults": ui_defaults,
            "task_name": str(task_name or "").strip() or None,
            "global_candidates": [],
            "global_manifests": [],
            "global_reports": [],
        }
    candidates_all: list[dict[str, Any]] = []
    manifests_all: list[dict[str, Any]] = []
    reports_all: list[dict[str, Any]] = []
    default_cfg, _ = _load_scene_config_from_catalog(
        engine=selected_engine,
        scene_id=items[0]["scene_id"],
        catalog=catalog,
        fallback_config_path=fallback_config_path,
    )
    stage3_cfg = _stage3_cfg(default_cfg)
    for item in items:
        scene_value = str(item.get("scene_id", "") or "").strip()
        cfg, _ = _load_scene_config_from_catalog(
            engine=selected_engine,
            scene_id=scene_value,
            catalog=catalog,
            fallback_config_path=fallback_config_path,
        )
        cfg = _with_task_pipeline_cfg(cfg, task_name)
        layouts = _stage3_layout_candidates(cfg, scene_value, selected_engine, str(task_name or ""))
        try:
            rows, _ = _discover_candidates(cfg, scene_id=scene_value, engine=selected_engine)
        except Exception:
            rows = []
        candidates_all.extend(rows)
        manifests_all.extend({**row, "scene_id": scene_value} for row in _list_manifests_multi(layouts, scene_value))
        reports_all.extend({**row, "scene_id": scene_value} for row in _list_reports_multi(layouts, scene_value))
    manifests_all.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    reports_all.sort(key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    stats = {
        "candidate_count": len(candidates_all),
        "approved_count": sum(1 for row in candidates_all if str(row.get("review_status", "")) == "approved"),
        "category_count": len({str(row.get("landmark_category", "") or "") for row in candidates_all if str(row.get("landmark_category", "") or "").strip()}),
        "avg_visible_count": (sum(int(row.get("visible_count", 0) or 0) for row in candidates_all) / float(len(candidates_all))) if candidates_all else 0.0,
        "single_landmark_count": sum(1 for row in candidates_all if str(row.get("mode", "single-landmark")) == "single-landmark"),
        "multi_landmark_count": sum(1 for row in candidates_all if str(row.get("mode", "single-landmark")) == "multi-landmark"),
        "mission_families": sorted({str(row.get("set_name", "") or "") for row in candidates_all if str(row.get("set_name", "") or "").strip()}),
    }
    ui_defaults = {
        "auto_pick_single_count": 1,
        "auto_pick_multi_count": max(2, int(stage3_cfg.get("multi_landmark_max_secondary", 2) or 2) + 1),
        "auto_pick_min_points": int(stage3_cfg.get("auto_pick_min_points", 500) or 500),
        "mission_count": int(stage3_cfg.get("mission_count_default", 1) or 1),
        "auto_set_rule": str(stage3_cfg.get("auto_set_rule_default", "heuristic") or "heuristic"),
        "allow_interleave_repeat": bool(stage3_cfg.get("allow_interleave_repeat_default", False)),
        "max_total_elements": int(stage3_cfg.get("max_total_elements_default", 0) or 0),
    }
    return {
        "candidate_stats": stats,
        "latest_manifest_path": str(manifests_all[0].get("path", "") or "") if manifests_all else None,
        "latest_report_path": str(reports_all[0].get("path", "") or "") if reports_all else None,
        "ui_defaults": ui_defaults,
        "task_name": str(task_name or "").strip() or None,
        "global_candidates": candidates_all,
        "global_manifests": manifests_all,
        "global_reports": reports_all,
    }


def recompute_report_from_run_dir(
    *,
    config: dict[str, Any],
    scene_id: str,
    engine: str,
    run_dir: Path,
) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    report_payload = _read_json(report_path, default={})
    if not isinstance(report_payload, dict):
        raise RuntimeError(f"invalid_stage3_report: {report_path}")
    parsed_rows = _load_report_rows(report_path)
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
    _write_json(report_path, updated)
    layout = _resolve_stage3_layout(config=config, scene_id=scene_id, engine=engine)
    latest_path = layout["experiments_root"] / f"{scene_id}.latest_report.json"
    if latest_path.exists():
        latest_payload = _read_json(latest_path, default={})
        latest_run_id = str((latest_payload or {}).get("run_id", "") or "")
        if latest_run_id == str(updated["run_id"]):
            _write_json(latest_path, updated)
    return {"run_id": str(updated["run_id"]), "report_path": report_path, "report": updated}


def _load_report_rows(report_path: Path) -> list[dict[str, Any]]:
    report_payload = _read_json(report_path, default={})
    api_overrides = dict((report_payload or {}).get("api_overrides", {}) or {})
    sample_lookup: dict[str, dict[str, Any]] = {}
    manifest_path_raw = str((report_payload or {}).get("manifest_path", "") or "") if isinstance(report_payload, dict) else ""
    if manifest_path_raw:
        manifest_path = _resolve_workspace_json_path(manifest_path_raw)
        manifest_payload = _read_json(manifest_path, default={})
        if isinstance(manifest_payload, dict):
            for sample in list(manifest_payload.get("samples", []) or []):
                if not isinstance(sample, dict):
                    continue
                sample_id = str(sample.get("sample_id", "") or "").strip()
                if sample_id:
                    sample_lookup[sample_id] = dict(sample)
    parsed_path = report_path.parent / "parsed_predictions.jsonl"
    rows = []
    if not parsed_path.exists():
        return rows
    for line in parsed_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row["gold_answer_items"] = _normalize_answer_items(row.get("gold_answer_items", []))
            row["pred_answer_items"] = _normalize_answer_items(row.get("pred_answer_items", []))
            row["gold_choice_intervals_sec"] = _normalize_interval_rows(row.get("gold_choice_intervals_sec", []))
            row["pred_choice_intervals_sec"] = _normalize_interval_rows(row.get("pred_choice_intervals_sec", []))
            sample = sample_lookup.get(str(row.get("sample_id", "") or "").strip(), {})
            if sample:
                if row.get("gold_answer_items") == []:
                    row["gold_answer_items"] = _normalize_answer_items(sample.get("answer_items", []))
                    row["gold_choice_intervals_sec"] = _flatten_answer_item_intervals(row["gold_answer_items"])
                option_label_map = {
                    str(item.get("option_id", "") or ""): str(item.get("label", "") or "")
                    for item in list(sample.get("choice_options", []) or [])
                    if str(item.get("option_id", "") or "").strip()
                }
                if row.get("pred_answer_items"):
                    patched = []
                    for item in list(row.get("pred_answer_items", []) or []):
                        item = dict(item)
                        if not str(item.get("label", "") or "").strip():
                            item["label"] = option_label_map.get(str(item.get("option_id", "") or ""), "")
                        patched.append(item)
                    row["pred_answer_items"] = _normalize_answer_items(patched)
                elif row.get("pred_option_ids"):
                    row["pred_answer_items"] = _normalize_answer_items(
                        [
                            {
                                "option_id": opt_id,
                                "label": option_label_map.get(str(opt_id), ""),
                                "intervals_sec": list(row.get("pred_choice_intervals_sec", []) or []),
                            }
                            for opt_id in list(row.get("pred_option_ids", []) or [])
                        ]
                    )
            if row.get("self_temporal_mean_tiou", None) is None and row.get("gold_choice_intervals_sec"):
                gold_pairs = _normalize_interval_rows(row.get("gold_choice_intervals_sec", []))
                pred_pairs = _normalize_interval_rows(row.get("pred_choice_intervals_sec", []))
                if gold_pairs:
                    row["self_temporal_mean_tiou"] = sum(
                        max((_interval_iou(pred_pair, gold_pair) for pred_pair in pred_pairs), default=0.0)
                        for gold_pair in gold_pairs
                    ) / float(len(gold_pairs))
            if str(row.get("task_name", row.get("form", "")) or "") == "self_instance_recognition_joint" and sample:
                row.update(
                    _derive_joint_level_payloads(
                        sample=sample,
                        gold_answer_items=_normalize_answer_items(row.get("gold_answer_items", [])),
                        pred_answer_items=_normalize_answer_items(row.get("pred_answer_items", [])),
                    )
                )
            if "include_keyframes" not in row:
                if "include_keyframes" in api_overrides:
                    row["include_keyframes"] = bool(api_overrides["include_keyframes"])
                elif sample:
                    row["include_keyframes"] = bool(sample.get("include_keyframes", False))
                else:
                    row["include_keyframes"] = False
            rows.append(row)
        except Exception:
            continue
    return rows


def _build_metrics_matrix(layout: dict[str, Path], scene_id: str, *, latest_only: bool, by_difficulty: bool) -> dict[str, Any]:
    reports = sorted(_list_reports(layout, scene_id), key=lambda row: str(row.get("generated_at", "") or ""), reverse=True)
    return _build_metrics_matrix_from_reports(reports, scene_id, latest_only=latest_only, by_difficulty=by_difficulty)


def _report_generated_sort_key(report: dict[str, Any]) -> tuple[str, str]:
    generated_at = str(report.get("generated_at", "") or "")
    path = str(report.get("path", "") or "")
    return generated_at, path


def _collect_stage3_rows_by_model(reports: list[dict[str, Any]], *, latest_only: bool) -> dict[str, list[dict[str, Any]]]:
    reports_sorted = sorted(reports, key=_report_generated_sort_key)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not latest_only:
        for report in reports_sorted:
            model = str(report.get("model", "") or "")
            report_path = Path(str(report.get("path", "") or "")).resolve()
            if not model or not report_path.exists():
                continue
            grouped[model].extend(_load_report_rows(report_path))
        return grouped

    latest_by_key: dict[tuple[str, str], tuple[tuple[str, str], dict[str, Any]]] = {}
    for report in reports_sorted:
        model = str(report.get("model", "") or "")
        report_path = Path(str(report.get("path", "") or "")).resolve()
        if not model or not report_path.exists():
            continue
        sort_key = _report_generated_sort_key(report)
        for row in _load_report_rows(report_path):
            sample_id = str(row.get("sample_id", "") or "").strip()
            if not sample_id:
                continue
            latest_by_key[(model, sample_id)] = (sort_key, row)
    for (model, _sample_id), (_sort_key, row) in latest_by_key.items():
        grouped[model].append(row)
    return grouped


def _build_metrics_matrix_from_reports(reports: list[dict[str, Any]], scene_id: str, *, latest_only: bool, by_difficulty: bool) -> dict[str, Any]:
    metric_forms = [
        "self_composite_instance_recognition",
        "self_atomic_instance_recognition",
        "env_visibility_reasoning",
    ]

    def _metrics_for_form(form: str) -> list[str]:
        text = str(form or "")
        if text == "env_visibility_reasoning":
            return ["main_metric", "segment_f1@0.5", "mean_best_tIoU"]
        return ["main_metric", "self_temporal_loc_f1@0.5", "self_temporal_loc_mean_tIoU"]

    columns = []
    for mode in MODE_CHOICES:
        for form in metric_forms:
            if by_difficulty:
                for difficulty in ["easy", "medium", "hard", "1", "2-3", "4-5", "6+"]:
                    columns.append({"combo_id": f"{mode}|{form}|{difficulty}", "mode": mode, "form": form, "difficulty": difficulty, "metrics": _metrics_for_form(form)})
            else:
                columns.append({"combo_id": f"{mode}|{form}", "mode": mode, "form": form, "difficulty": "ALL", "metrics": _metrics_for_form(form)})
    aggregated: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    combo_key = "combo_by_difficulty" if by_difficulty else "combo"
    rows_by_model = _collect_stage3_rows_by_model(reports, latest_only=latest_only)
    for model, rows in rows_by_model.items():
        summary = _summarize_predictions(rows) if rows else {}
        grouped_summary = dict(summary.get("grouped", {}) or {}) if isinstance(summary, dict) else {}
        aggregated[model] = dict(grouped_summary.get(combo_key, {}) or {})
    matrix_rows = []
    for model in sorted(aggregated.keys()):
        combo_payload: dict[str, Any] = {}
        for column in columns:
            summary = dict(aggregated[model].get(column["combo_id"], {}) or {})
            combo_payload[column["combo_id"]] = {
                "count": summary.get("count", 0),
                "count_exact_acc": summary.get("main_metric"),
                "main_metric": summary.get("main_metric"),
                "count_within1_acc": summary.get("count_within1_acc"),
                "segment_f1@0.5": summary.get("segment_f1@0.5"),
                "mean_best_tIoU": summary.get("mean_best_tIoU"),
                "bbox_acc@50iou": summary.get("bbox_acc@50iou"),
                "self_temporal_loc_f1@0.5": summary.get("self_temporal_loc_f1@0.5"),
                "self_temporal_loc_mean_tIoU": summary.get("self_temporal_loc_mean_tIoU"),
                "metrics": list(column.get("metrics", []) or []),
            }
        matrix_rows.append({"model": model, "combos": combo_payload})
    return {"columns": columns, "rows": matrix_rows, "latest_only": bool(latest_only), "by_difficulty": bool(by_difficulty)}


def _stage3_metrics_matrix_csv_rows(matrix: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = [
        "model",
        "mode",
        "form",
        "difficulty",
        "count",
        "main_metric",
        "count_within1_acc",
        "segment_f1@0.5",
        "mean_best_tIoU",
        "bbox_acc@50iou",
        "self_temporal_loc_f1@0.5",
        "self_temporal_loc_mean_tIoU",
    ]
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
                    "mode": str(column.get("mode", "") or ""),
                    "form": str(column.get("form", "") or ""),
                    "difficulty": str(column.get("difficulty", "") or ""),
                    "count": payload.get("count", 0),
                    "main_metric": payload.get("main_metric"),
                    "count_within1_acc": payload.get("count_within1_acc"),
                    "segment_f1@0.5": payload.get("segment_f1@0.5"),
                    "mean_best_tIoU": payload.get("mean_best_tIoU"),
                    "bbox_acc@50iou": payload.get("bbox_acc@50iou"),
                    "self_temporal_loc_f1@0.5": payload.get("self_temporal_loc_f1@0.5"),
                    "self_temporal_loc_mean_tIoU": payload.get("self_temporal_loc_mean_tIoU"),
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


def register_stage3_task_routes(
    app: Any,
    *,
    default_config: dict[str, Any],
    scene_id: str,
    engine: str,
    config_path: Path,
) -> None:
    if jsonify is None:
        raise ImportError("Flask is required")
    catalog = _discover_scene_catalog(config_path)
    job_manager = ExperimentJobManager()
    default_engine = str(engine).strip().lower()
    default_scene_id = str(scene_id).strip()

    def _load_scene_context(selected_engine: str | None, selected_scene_id: str | None) -> tuple[dict[str, Any], dict[str, Path]]:
        engine_value = str(selected_engine or default_engine).strip().lower()
        scene_value = str(selected_scene_id or default_scene_id).strip()
        cfg, _ = _load_scene_config_from_catalog(engine=engine_value, scene_id=scene_value, catalog=catalog, fallback_config_path=config_path)
        layout = _resolve_stage3_layout(cfg, scene_value, engine_value)
        return cfg, layout

    def _load_layouts(selected_engine: str | None, selected_scene_id: str | None, task_name: str | None) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Path]]]:
        engine_value = str(selected_engine or default_engine).strip().lower()
        scene_value = str(selected_scene_id or default_scene_id).strip()
        cfg, layout = _load_scene_context(engine_value, scene_value)
        layouts = _stage3_layout_candidates(cfg, scene_value, engine_value, str(task_name or ""))
        return cfg, layout, layouts

    def _artifact_response(raw_path: str) -> Any:
        path = Path(str(raw_path or "")).resolve()
        workspace = WORKSPACE_ROOT.resolve()
        if not str(path).startswith(str(workspace)):
            return jsonify({"error": "path_outside_workspace"}), 403
        if not path.exists() or not path.is_file():
            return jsonify({"error": "artifact_not_found"}), 404
        if path.suffix.lower() == ".mp4":
            _ensure_mp4_web_playable(path)
        resize_enabled = str(request.args.get("resize", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        if resize_enabled and Image is not None:
            suffix = path.suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                max_w = max(64, int(request.args.get("w", "854") or 854))
                max_h = max(64, int(request.args.get("h", "480") or 480))
                quality = max(40, min(95, int(request.args.get("q", "80") or 80)))
                try:
                    with Image.open(path) as image:
                        canvas = image.convert("RGB")
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
                        canvas.save(buff, format="JPEG", quality=quality)
                        buff.seek(0)
                        return send_file(buff, mimetype="image/jpeg", download_name=f"{path.stem}_preview.jpg")
                except Exception:
                    pass
        return send_file(path)

    def _shell(active_page: str) -> str:
        nav_items = [
            ("behavior_library", "行为库"),
            ("missions", "任务生成"),
            ("review", "候选复核"),
            ("generate", "数据生成"),
            ("dataset", "任务查看"),
            ("experiments", "实验执行"),
            ("results", "结果查看"),
            ("metrics", "指标汇总"),
        ]
        nav_html = "".join(f'<a class="nav-item {"active" if key == active_page else ""}" href="/{key}">{label}</a>' for key, label in nav_items)
        template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 3 工作台</title>
  <style>
    :root {
      --body-grad-1: #0c1016; --body-grad-2:#131923; --body-grad-3:#0f141d;
      --panel:#171c25; --panel-2:#1e2530; --panel-soft:#1b2230; --ink:#eef3f8; --muted:#9da9b6;
      --line:#2a3442; --brand:#55c1ff; --accent:#ffb347; --banner-bg:rgba(15,19,26,0.92); --footer-bg:rgba(15,19,26,0.94);
      --input-bg:var(--panel-2); --input-border:var(--line); --nav-active-bg:#1f2835; --nav-active-border:#324052; --nav-active-ink:#eef3f8;
      --secondary-btn-bg:#202837; --secondary-btn-ink:#eef3f8; --warn-btn-bg:#5f2f26; --warn-btn-border:#8a4337; --warn-btn-ink:#fff3ef;
      --preview-bg:#0d1118; --shadow:0 18px 32px rgba(0,0,0,0.18); --thead-bg:#151b24;
    }
    body[data-theme="light"] {
      --body-grad-1:#f6f1e8; --body-grad-2:#efe7da; --body-grad-3:#f4efe6;
      --panel:#fffdf9; --panel-2:#ffffff; --panel-soft:#fbf7f0; --ink:#14212b; --muted:#6d7379;
      --line:#ded5c5; --brand:#114b5f; --accent:#d95d39; --banner-bg:rgba(255,253,249,0.96); --footer-bg:rgba(255,253,249,0.96);
      --input-bg:#ffffff; --input-border:#cfc5b5; --nav-active-bg:#114b5f; --nav-active-border:#114b5f; --nav-active-ink:#fff;
      --secondary-btn-bg:#eadfce; --secondary-btn-ink:#14212b; --warn-btn-bg:#d95d39; --warn-btn-border:#bf4f2f; --warn-btn-ink:#fff;
      --preview-bg:#f4efe6; --shadow:0 18px 32px rgba(17,75,95,0.08); --thead-bg:#f6f0e4;
    }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; font-family:"Segoe UI","PingFang SC",sans-serif; background:linear-gradient(180deg,var(--body-grad-1),var(--body-grad-2) 38%,var(--body-grad-3)); color:var(--ink); }
    .banner { position:sticky; top:0; z-index:30; display:flex; justify-content:space-between; align-items:center; gap:18px; padding:14px 22px; background:var(--banner-bg); backdrop-filter:blur(10px); border-bottom:1px solid var(--line); }
    .banner .left { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
    .brand { font-size:18px; font-weight:700; }
    .nav { display:flex; gap:8px; flex-wrap:wrap; }
    .nav-item { padding:8px 12px; border-radius:999px; color:var(--muted); border:1px solid transparent; text-decoration:none; }
    .nav-item.active { color:var(--nav-active-ink); background:var(--nav-active-bg); border-color:var(--nav-active-border); }
    .selectors { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
    .selectors label { display:inline-flex; align-items:center; gap:8px; padding:8px 12px; border:1px solid var(--line); border-radius:12px; background:var(--panel-soft); color:var(--muted); }
    select,input,button,textarea { background:var(--input-bg); color:var(--ink); border:1px solid var(--input-border); border-radius:10px; padding:9px 11px; font:inherit; }
    button { cursor:pointer; }
    button.primary { background:linear-gradient(135deg,#1f5eff,#3ca8ff); border:none; color:#fff; }
    button.secondary { background:var(--secondary-btn-bg); color:var(--secondary-btn-ink); }
    button.warn { background:var(--warn-btn-bg); border-color:var(--warn-btn-border); color:var(--warn-btn-ink); }
    .page { padding:24px 24px 120px; width:100%; max-width:none; margin:0; min-height:calc(100vh - 160px); }
    .grid-2 { display:grid; grid-template-columns:1.1fr 0.9fr; gap:18px; }
    .card { background:color-mix(in srgb, var(--panel) 92%, transparent); border:1px solid var(--line); border-radius:18px; padding:18px; box-shadow:var(--shadow); }
    .toolbar { display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; margin-bottom:14px; }
    .toolbar > input,
    .toolbar > select { flex:1 1 220px; min-width:0; }
    .toolbar > button { flex:0 0 auto; }
    .toolbar label { display:inline-flex; align-items:center; gap:8px; flex:1 1 210px; min-width:0; min-height:44px; padding:8px 12px; border:1px solid var(--line); border-radius:12px; background:var(--panel-soft); color:var(--muted); }
    .toolbar label input[type="checkbox"],
    .toolbar label input[type="radio"] { width:auto; flex:0 0 auto; margin:0; }
    .toolbar label input:not([type="checkbox"]):not([type="radio"]),
    .toolbar label select { flex:1 1 auto; min-width:92px; width:auto; }
    .stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px,1fr)); gap:12px; }
    .stat { background:var(--panel-soft); border:1px solid var(--line); border-radius:14px; padding:12px 14px; }
    .stat .value { font-size:22px; font-weight:700; margin-top:6px; }
    .muted { color:var(--muted); }
    .table-wrap { max-height:min(72vh, 860px); overflow:auto; border:1px solid var(--line); border-radius:14px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th,td { border-bottom:1px solid var(--line); padding:9px 8px; text-align:left; vertical-align:top; }
    th { background:var(--thead-bg); position:sticky; top:0; }
    .preview-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(250px,1fr)); gap:14px; }
    img.preview, video.preview { width:100%; border-radius:14px; border:1px solid var(--line); background:var(--preview-bg); }
    .pill-row { display:flex; gap:8px; flex-wrap:wrap; }
    .pill { display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; border:1px solid var(--line); background:var(--panel-soft); color:var(--ink); font-size:12px; }
    .section-title { margin: 10px 0 8px; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }
    .split-card { display:grid; gap:12px; }
    .stack { display:grid; gap:18px; }
    .bars { display:grid; gap:10px; }
    .bar-row { display:grid; grid-template-columns: 180px 1fr 72px; gap:10px; align-items:center; }
    .bar-track { background:var(--panel-soft); border:1px solid var(--line); border-radius:999px; overflow:hidden; height:12px; }
    .bar-fill { background:linear-gradient(90deg,var(--brand),var(--accent)); height:100%; }
    .metric-band { height:14px; border-radius:999px; overflow:hidden; background:var(--panel-soft); border:1px solid var(--line); }
    .metric-band > div { height:100%; background:linear-gradient(90deg,#2dd4bf,#60a5fa,#f59e0b,#ef4444); }
    .chip-good { color:#d1fae5; background:#065f46; }
    .chip-mid { color:#fef3c7; background:#92400e; }
    .chip-bad { color:#fee2e2; background:#991b1b; }
    .mini-kv { display:grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap:10px; }
    .mini-card { background:var(--panel-soft); border:1px solid var(--line); border-radius:12px; padding:10px; }
    .kv { display:grid; grid-template-columns: 180px 1fr; gap:8px 14px; font-size:13px; }
    .kv .k { color:var(--muted); }
    .empty { padding:18px; color:var(--muted); }
    footer { position:fixed; left:0; right:0; bottom:0; z-index:20; display:grid; grid-template-columns:1fr 1px 1fr; background:var(--footer-bg); backdrop-filter:blur(10px); border-top:1px solid var(--line); }
    footer .col { padding:12px 22px; font-size:12px; color:var(--muted); line-height:1.6; }
    footer .sep { background:var(--line); }
  </style>
</head>
<body>
  <div class="banner">
    <div class="left">
      <div class="brand">Stage 3 飞行任务工作台</div>
      <div class="nav">__NAV_HTML__</div>
    </div>
    <div class="selectors">
      <label>引擎 <select id="engine_select" onchange="switchScene()"></select></label>
      <label>场景 <select id="scene_select" onchange="switchScene()"></select></label>
      <label>任务 <select id="task_pipeline_select" onchange="refreshAll()"></select></label>
      <label>皮肤 <select id="theme_select"><option value="light">亮色</option><option value="dark">暗色</option></select></label>
    </div>
  </div>
  <div class="page">
    <div id="page_behavior_library" style="display:__DISPLAY_BEHAVIOR_LIBRARY__">
      <div class="grid-2">
        <div class="card">
          <h1>Behavior Library</h1>
          <div class="muted">查看当前 Stage 3 的 Composite class、Atomic class、参数范围、默认值和生成说明。</div>
          <div id="behavior_stats" class="stats" style="margin-top:14px;"></div>
          <div id="set_table" class="table-wrap" style="margin-top:14px;"></div>
        </div>
        <div class="card">
          <h2>定义详情</h2>
          <div id="behavior_detail" class="empty">点击左侧 Composite class 后，这里会显示 Atomic 组成、参数范围和生成说明。</div>
        </div>
      </div>
    </div>
    <div id="page_missions" style="display:__DISPLAY_MISSIONS__">
      <div class="grid-2">
        <div class="card">
          <h1>任务生成</h1>
          <div class="muted">默认以 Composite class 或 Atomic instance 为生成单位。系统会自动实例化参数、连接轨迹，并生成 Preview 视频和 Task Video。</div>
          <div class="toolbar" style="margin-top:14px;">
            <input id="mission_query" placeholder="按 instance_id / class / label 搜索">
            <select id="mission_status" onchange="loadMissionInstances()">
              <option value="all">全部状态</option>
              <option value="pending">待生成</option>
              <option value="pano_ready">全景已完成</option>
              <option value="pano_confirmed">全景已确认</option>
              <option value="video_ready">Preview 已完成</option>
              <option value="video_confirmed">Preview 已确认</option>
              <option value="final_ready">Task Video 已完成</option>
              <option value="valid">通过</option>
              <option value="invalid">未通过</option>
            </select>
            <button class="secondary" onclick="loadMissionInstances()">刷新</button>
          </div>
          <div class="section-title">自动选择地标</div>
          <div class="toolbar">
            <label>选择模式 <select id="mission_auto_pick_mode"><option value="single">single</option><option value="multi">multi</option></select></label>
            <label>数量 <input id="mission_auto_pick_count" type="number" value="1" min="1" max="12"></label>
            <label>最少点数 <input id="mission_auto_pick_min_points" type="number" value="500" min="0"></label>
          </div>
          <div class="toolbar">
            <label><input id="mission_auto_pick_diverse" type="checkbox"> 类别尽量不同</label>
            <button class="secondary" onclick="autoSelectMissionLandmarks()">自动选择</button>
            <button class="secondary" onclick="clearMissionSelection()">清空选择</button>
          </div>
          <div id="mission_stats" class="stats"></div>
          <div class="section-title">地标类型选择</div>
          <div id="mission_category_filter_table" class="table-wrap" style="margin-top:10px;"></div>
          <div class="section-title">Instance 列表</div>
          <div id="mission_table" class="table-wrap" style="margin-top:14px;"></div>
          <div class="section-title">所选地标</div>
          <div id="mission_selected_summary" class="summary-list">请先选择地标。</div>
          <div class="section-title">任务列表</div>
          <div class="toolbar">
            <button class="primary" onclick="openNewMission()">新增任务</button>
            <button class="secondary" onclick="loadMissionHistory()">刷新任务列表</button>
            <button class="warn" onclick="clearMissionHistory()">清空当前任务列表</button>
          </div>
          <div id="mission_history_table" class="table-wrap" style="margin-top:10px;"></div>
        </div>
        <div class="card">
          <h2>任务详情</h2>
          <div class="section-title">任务范围</div>
          <div class="toolbar">
            <label>任务模式 <input id="mission_mode_display" value="single-landmark" readonly></label>
            <label>生成数量 <input id="mission_count" type="number" value="1" min="1" max="16"></label>
          </div>
          <div class="section-title">Composite 配置</div>
          <div class="toolbar">
            <label><input id="mission_auto" type="checkbox" checked onchange="refreshMissionTemplateSelect()"> 自动选择 Composite</label>
            <label>Composite 模板 <select id="mission_type_select"></select></label>
            <label>选择规则 <select id="mission_auto_set_rule"><option value="heuristic">heuristic</option><option value="random">random</option><option value="round_robin">round_robin</option></select></label>
            <label>可用 Composite <select id="mission_auto_set_candidates" multiple></select></label>
          </div>
          <div class="section-title">轨迹拼接与序列</div>
          <div class="toolbar">
            <label>生成路径 <select id="mission_generation_kind"><option value="auto">auto</option><option value="atomic-only">atomic-only</option><option value="composite-driven">composite-driven</option></select></label>
            <label>Atomic 序列（可选覆盖） <input id="mission_behavior_override" placeholder="例如 gradual_approach,circular_orbit,gradual_depart"></label>
            <label><input id="mission_adaptive_params" type="checkbox" checked> 逐段自适应参数</label>
          </div>
          <div class="section-title" data-block="multi-rule">多地标重复规则</div>
          <div class="toolbar" data-block="multi-rule">
            <label><input id="mission_allow_interleave_repeat" type="checkbox"> 允许交叉重复出现</label>
            <label>单地标 Composite 总 atomic 数 <input id="mission_max_total_elements" type="number" value="0" min="0"></label>
          </div>
          <div id="mission_template_summary" class="muted" style="margin-bottom:10px;">请先选择一个目标地标。默认会根据地标尺度、局部空间结构和障碍布局自动选择 Composite class。</div>
          <div class="section-title">Atomic 参数</div>
          <div class="toolbar">
            <label><input id="mission_set_params_auto" type="checkbox" checked> Composite 中各 Atomic 参数全部自动设置</label>
          </div>
          <div id="mission_step_editor" class="empty" style="margin-bottom:12px;">这里会显示当前任务的逐段参数配置。</div>
          <div class="section-title">媒体显示</div>
          <div class="toolbar">
            <label><input id="mission_media_compress" type="checkbox" checked> 图片压缩显示（默认 480P）</label>
            <label>宽 <input id="mission_media_w" type="number" value="640" min="64"></label>
            <label>高 <input id="mission_media_h" type="number" value="480" min="64"></label>
            <label>JPEG <input id="mission_media_q" type="number" value="80" min="40" max="95"></label>
            <label><input id="mission_video_prefer_web" type="checkbox" checked> 视频优先压缩版（默认 1M）</label>
          </div>
          <div class="toolbar">
            <button class="primary" onclick="generateMission()">生成任务（含全景与 Preview）</button>
            <button class="secondary" onclick="generateMissionVideo()">重新生成 Preview</button>
            <button class="secondary" onclick="confirmMissionVideo(true)">Preview 通过</button>
            <button class="warn" onclick="confirmMissionVideo(false)">Preview 驳回</button>
            <button class="primary" onclick="generateMissionFinalTask()">生成 Task Video</button>
          </div>
          <div id="mission_detail" class="empty">选择一个地标后，这里会显示其几何信息、实例化结果、保存位置和推荐 Composite 模板。</div>
        </div>
      </div>
    </div>
    <div id="page_review" style="display:__DISPLAY_REVIEW__">
      <div class="grid-2">
        <div class="card">
          <h1>候选复核</h1>
          <div class="muted">先筛选生成好的任务视频，再将通过的样本转为时序定位任务数据。</div>
          <div class="toolbar" style="margin-top:14px;">
            <input id="candidate_query" placeholder="按 traj_id / category / set 搜索">
            <select id="candidate_status" onchange="loadCandidates()"><option value="all">全部</option><option value="approved">已通过</option><option value="pending">待定</option><option value="rejected">已驳回</option></select>
            <button class="secondary" onclick="loadCandidates()">刷新</button>
          </div>
          <div id="candidate_stats" class="stats"></div>
          <div id="candidate_table" class="table-wrap" style="margin-top:14px;"></div>
        </div>
        <div class="card">
          <h2>候选详情</h2>
          <div id="candidate_detail" class="empty">选择一条候选任务后，这里会显示参考图、可选总览图和整秒关键帧看板。</div>
        </div>
      </div>
    </div>
    <div id="page_generate" style="display:__DISPLAY_GENERATE__">
      <div class="grid-2">
        <div class="card">
          <h1>任务数据生成</h1>
          <div class="toolbar">
            <label>模式 <select id="gen_mode"><option value="single-landmark">single-landmark</option><option value="multi-landmark">multi-landmark</option></select></label>
            <label>样本数 <input id="gen_sample_count" type="number" value="24" min="1" style="width:100px;"></label>
            <label>随机种子 <input id="gen_seed" type="number" value="7" style="width:100px;"></label>
            <label><input id="gen_approved_only" type="checkbox" checked> 仅使用已通过候选</label>
          </div>
          <div class="muted" style="margin:8px 0 6px;">Self-State Awareness</div>
          <div class="toolbar">
            <label><input class="gen_form" type="checkbox" value="self_instance_recognition_joint" checked> 联合实例识别</label>
            <label><input class="gen_form" type="checkbox" value="self_set_instance_recognition" checked> Composite 实例识别</label>
            <label><input class="gen_form" type="checkbox" value="self_element_instance_recognition" checked> Atomic 实例识别</label>
            <label><input id="gen_include_temporal_localization" type="checkbox" checked> 加入时序定位</label>
          </div>
          <div class="muted" style="margin:8px 0 6px;">Environmental Awareness</div>
          <div class="toolbar">
            <label><input class="gen_form" type="checkbox" value="env_visibility_reasoning" checked> 环境感知</label>
          </div>
          <button class="primary" onclick="generateManifest()">生成 Manifest</button>
          <div id="generate_feedback" class="muted" style="margin-top:12px;"></div>
        </div>
        <div class="card">
          <h2>生成参考</h2>
          <div id="generate_reference" class="empty">这里会显示候选统计和最新 manifest 概览。</div>
        </div>
      </div>
    </div>
    <div id="page_dataset" style="display:__DISPLAY_DATASET__">
      <div class="grid-2">
        <div class="card">
          <h1>任务数据查看</h1>
          <div class="toolbar"><select id="manifest_select" onchange="loadManifestDetail()"></select><button class="secondary" onclick="loadManifestList()">刷新</button></div>
          <div id="manifest_summary" class="empty">请选择一个 manifest。</div>
          <div class="section-title">任务列表</div>
          <div id="manifest_sample_list" class="table-wrap"></div>
        </div>
        <div class="card">
          <h2>样本预览</h2>
          <div id="manifest_samples" class="empty">这里会展示样本的完整内容。</div>
        </div>
      </div>
    </div>
    <div id="page_experiments" style="display:__DISPLAY_EXPERIMENTS__">
      <div class="grid-2">
        <div class="card">
          <h1>实验执行</h1>
          <div class="toolbar">
            <select id="exp_manifest_select" onchange="loadExperimentManifestSummary()"></select>
            <input id="exp_models" placeholder="多个模型用逗号分隔；留空则用默认模型" style="min-width:260px;">
            <label>数量限制 <input id="exp_limit" type="number" min="1" style="width:90px;"></label>
          </div>
          <div class="toolbar">
            <label>上传宽 <input id="exp_upload_w" type="number" value="640" style="width:90px;"></label>
            <label>上传高 <input id="exp_upload_h" type="number" value="480" style="width:90px;"></label>
            <label>JPEG <input id="exp_upload_q" type="number" value="80" style="width:90px;"></label>
            <label>并发 <input id="exp_concurrency" type="number" value="1" style="width:90px;"></label>
            <label>RPM <input id="exp_rpm" type="number" value="0" style="width:90px;"></label>
            <label>TPM <input id="exp_tpm" type="number" value="0" style="width:110px;"></label>
          </div>
          <div class="toolbar">
            <label><input id="exp_flight_description" type="checkbox" checked> 提供飞行任务描述</label>
            <label><input id="exp_include_keyframes" type="checkbox"> 开启关键帧评测</label>
            <button class="primary" onclick="startExperiment()">开始实验</button>
          </div>
          <div id="experiment_manifest_summary" class="muted"></div>
        </div>
        <div class="card">
          <h2>实验任务</h2>
          <div id="job_table" class="table-wrap"></div>
        </div>
      </div>
    </div>
    <div id="page_results" style="display:__DISPLAY_RESULTS__">
      <div class="grid-2">
        <div class="card">
          <h1>实验结果查看</h1>
          <div class="toolbar"><select id="report_select" onchange="loadReportDetail()"></select><button class="secondary" onclick="loadReportList()">刷新</button></div>
          <div id="report_summary" class="empty">请选择一个实验报告。</div>
        </div>
        <div class="card">
          <h2>逐样本详情</h2>
          <div id="report_rows" class="table-wrap"></div>
        </div>
      </div>
    </div>
    <div id="page_metrics" style="display:__DISPLAY_METRICS__">
      <div class="stack">
        <div class="card">
          <h1>实验指标汇总</h1>
          <div class="toolbar"><label><input id="metrics_latest_only" type="checkbox" onchange="loadMetricsMatrix()"> 按单个样本最新结果汇总</label><label><input id="metrics_by_difficulty" type="checkbox" onchange="loadMetricsMatrix()"> 按难度区分</label><button class="secondary" onclick="loadMetricsMatrix()">刷新</button><button class="secondary" onclick="exportStage3MetricsCsv()">导出 CSV</button></div>
          <div id="metrics_summary_cards" class="stats" style="margin-bottom:14px;"></div>
        </div>
        <div class="card">
          <h2>分组分析</h2>
          <div id="metrics_group_bars"></div>
        </div>
        <div class="card">
          <h2>实验大表</h2>
          <div id="metrics_matrix" class="table-wrap"><div class="muted">加载中...</div></div>
        </div>
        <div class="card">
          <h2>实验进度大表</h2>
          <div id="metrics_progress_summary" class="summary-list" style="margin-bottom:12px;"></div>
          <div id="metrics_progress_matrix" class="table-wrap"><div class="muted">加载中...</div></div>
        </div>
      </div>
    </div>
  </div>
  <footer>
    <div class="col" id="footer_left">Stage 3 将 Composite / Atomic 生成、候选复核、数据构建和时序推理实验统一到同一个工作台中。</div>
    <div class="sep"></div>
    <div class="col" id="footer_right">建议先在“任务生成”页生成并检查实例，再到“候选复核”筛选样本，之后生成 manifest 并执行实验。</div>
  </footer>
<script>
const state = {
  catalog: [],
  currentPage: __ACTIVE_PAGE_JSON__,
  missionCatalog: {missions: [], behaviors: []},
  currentMissionInstance: null,
  currentMissionResult: null,
  currentMissionHistory: null,
  currentMissionSchema: null,
  missionDetailMode: 'blank',
  lastAppliedSetKey: '',
  missionBehaviorOverrideTouched: false,
  selectedMissionInstanceIds: [],
  missionRows: [],
  missionHistoryRows: [],
  uiDefaults: {},
  activeManifestSampleId: '',
};
const DEFAULT_ENGINE = __DEFAULT_ENGINE_JSON__;
const DEFAULT_SCENE = __DEFAULT_SCENE_JSON__;
const TASK_KEY = 'uav_dualcog_stage3_task_pipeline';
const THEME_KEY = 'uav_dualcog_stage3_theme';
function esc(v) { return String(v ?? '').replace(/[&<>\"']/g, (c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c])); }
async function api(url, options) { const resp = await fetch(url, options); const data = await resp.json(); if(!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`); return data; }
function fmtPct(v) { return (v===null || v===undefined || Number.isNaN(Number(v))) ? '-' : `${(Number(v)*100).toFixed(1)}%`; }
function fmtFloat(v,n=3) { return (v===null || v===undefined || Number.isNaN(Number(v))) ? '-' : Number(v).toFixed(n); }
function footerHtml(items) { return (items || []).map((item)=>`<div>${item}</div>`).join('') || '<div class="muted">暂无内容</div>'; }
function readGlobalSetProfiles() {
  return (state.missionCatalog && state.missionCatalog.behavior_defaults && typeof state.missionCatalog.behavior_defaults === 'object')
    ? state.missionCatalog.behavior_defaults
    : {};
}
function getSetProfileByKey(setKey) {
  const key = String(setKey || '').trim();
  const profiles = readGlobalSetProfiles();
  return key && profiles[key] && typeof profiles[key] === 'object' ? profiles[key] : {};
}
function singleLandmarkInspectionSets() {
  return (state.missionCatalog.missions || []).filter((row)=>String(row.service_scenario || '') === 'single-landmark' && !!row.multi_landmark_component);
}
function currentSchemaSetKey() {
  return String(state.currentMissionSchema?.set_spec?.set_key || state.currentMissionSchema?.set_instance?.set_id || '').trim();
}
function currentSetProfile() {
  return getSetProfileByKey(currentSchemaSetKey());
}
function selectedMissionIds() { return [...new Set((state.selectedMissionInstanceIds || []).map((x)=>String(x || '').trim()).filter(Boolean))]; }
function missionMediaUrl(rawPath) {
  const path = String(rawPath || '').trim();
  if(!path || path === '-') return '';
  if(path.startsWith('/artifacts/') || path.startsWith('/artifact?') || path.startsWith('http://') || path.startsWith('https://')) return path;
  return `/artifact?path=${encodeURIComponent(path)}`;
}
function renderVideoPlayer(primaryUrl, secondaryUrl) {
  const url = missionMediaUrl(primaryUrl || secondaryUrl || '');
  if(!url) return '<div class="empty">暂无视频</div>';
  return `<video class="preview" controls preload="metadata" playsinline src="${esc(url)}">当前浏览器无法播放该视频。</video>`;
}
function activeMissionContext() {
  const row = state.currentMissionInstance || {};
  const hist = state.currentMissionHistory || null;
  return hist || { traj_id: row.latest_traj_id || '', asset_urls: row.asset_urls || {}, files: row.latest_files || {}, summary: row.latest_summary || {} };
}
function missionImagePreviewUrl(rawPath) {
  const path = String(rawPath || '').trim();
  if(!path || path === '-') return '';
  const enable = !!document.getElementById('mission_media_compress')?.checked;
  if(!enable) return missionMediaUrl(path);
  const w = Number(document.getElementById('mission_media_w')?.value || 640) || 640;
  const h = Number(document.getElementById('mission_media_h')?.value || 480) || 480;
  const q = Number(document.getElementById('mission_media_q')?.value || 80) || 80;
  return `/artifact?path=${encodeURIComponent(path)}&resize=1&w=${encodeURIComponent(w)}&h=${encodeURIComponent(h)}&q=${encodeURIComponent(q)}`;
}
function renderMissionEmpty(message='点击左侧“新增任务”或选择已有 mission 后，这里才显示内容。') {
  const box = document.getElementById('mission_detail');
  if(box) box.innerHTML = `<div class="empty">${esc(message)}</div>`;
}
function preferredMissionVideo(webPath, rawPath) {
  const preferWeb = !!document.getElementById('mission_video_prefer_web')?.checked;
  const web = String(webPath || '').trim();
  const raw = String(rawPath || '').trim();
  return preferWeb ? (web || raw) : (raw || web);
}
function collectScopeParamOverrides(scopeClass) {
  const rows = {};
  document.querySelectorAll(`.${scopeClass} .mission-param-card`).forEach((card)=>{
    const checkbox = card.querySelector('.mission-auto-enabled');
    if(checkbox && checkbox.checked) return;
    const input = card.querySelector('.mission-manual-input');
    if(!input || input.disabled) return;
    const stepIndex = String(input.dataset.stepIndex || '');
    const paramKey = String(input.dataset.paramKey || '');
    const raw = String(input.value || '').trim();
    if(!stepIndex || !paramKey || raw === '') return;
    if(!rows[stepIndex]) rows[stepIndex] = {};
    rows[stepIndex][paramKey] = raw;
  });
  return rows;
}
function collectScopeAutoRules(scopeClass) {
  const rows = {};
  document.querySelectorAll(`.${scopeClass} .mission-auto-enabled`).forEach((checkbox)=>{
    const stepIndex = String(checkbox.dataset.stepIndex || '');
    const paramKey = String(checkbox.dataset.paramKey || '');
    if(!stepIndex || !paramKey || !checkbox.checked || checkbox.disabled) return;
    const root = checkbox.closest('.mission-param-card');
    if(!root) return;
    if(!rows[stepIndex]) rows[stepIndex] = {};
    rows[stepIndex][paramKey] = {
      enabled: true,
      min: Number(root.querySelector('.mission-auto-min')?.value || 0),
      max: Number(root.querySelector('.mission-auto-max')?.value || 0),
      step: Number(root.querySelector('.mission-auto-step')?.value || 0),
      method: String(root.querySelector('.mission-auto-method')?.value || 'random'),
      mean: Number(root.querySelector('.mission-auto-mean')?.value || 0),
      std: Number(root.querySelector('.mission-auto-std')?.value || 0),
    };
  });
  return rows;
}
function collectMissionParamOverrides() {
  if(document.getElementById('mission_set_params_auto')?.checked) return {};
  return collectScopeParamOverrides('mission-scope');
}
function collectMissionAutoRules() {
  if(document.getElementById('mission_set_params_auto')?.checked) return {};
  return collectScopeAutoRules('mission-scope');
}
function bindMissionParamEditor() {
  bindScopeParamEditor('mission-scope', null);
}
function updateScopeParamDependencyStates(scopeClass) {
  const forceAuto = scopeClass === 'mission-scope' && !!document.getElementById('mission_set_params_auto')?.checked;
  const stepMap = new Map();
  document.querySelectorAll(`.${scopeClass} .mission-param-card`).forEach((card)=>{
    const stepIndex = String(card.dataset.stepIndex || '');
    if(!stepMap.has(stepIndex)) stepMap.set(stepIndex, []);
    stepMap.get(stepIndex).push(card);
  });
  for(const [stepIndex, cards] of stepMap.entries()) {
    const cameraCard = cards.find((card)=>String(card.dataset.paramKey || '') === 'camera_mode');
    const cameraInput = cameraCard?.querySelector('.mission-manual-input');
    const cameraMode = String(cameraInput?.value || 'landmark_track');
    for(const card of cards) {
      const paramKey = String(card.dataset.paramKey || '');
      if(!['gaze_pitch_deg', 'yaw_offset_deg'].includes(paramKey)) continue;
      const autoBox = card.querySelector('.mission-auto-enabled');
      const manualInput = card.querySelector('.mission-manual-input');
      const auxInputs = card.querySelectorAll('.mission-auto-min, .mission-auto-max, .mission-auto-step, .mission-auto-method, .mission-auto-mean, .mission-auto-std');
      const lock = cameraMode === 'landmark_track';
      if(autoBox) {
        if(forceAuto) autoBox.checked = true;
        if(lock) autoBox.checked = true;
        autoBox.disabled = lock || forceAuto;
      }
      if(manualInput) manualInput.disabled = lock || forceAuto || !!autoBox?.checked;
      auxInputs.forEach((el)=>{ el.disabled = lock || forceAuto; });
      const tip = card.querySelector('.mission-param-tip');
      if(tip) tip.textContent = lock ? 'landmark_track 下自动跟随' : (forceAuto ? '当前按全局默认值自动生成' : '');
    }
  }
}
function bindScopeParamEditor(scopeClass, onChange) {
  document.querySelectorAll(`.${scopeClass} .mission-auto-enabled`).forEach((checkbox)=>{
    const root = checkbox.closest('.mission-param-card');
    const input = root?.querySelector('.mission-manual-input');
    const toggle = () => {
      if(input) input.disabled = !!checkbox.checked;
      updateScopeParamDependencyStates(scopeClass);
      if(onChange) onChange();
    };
    checkbox.addEventListener('change', toggle);
    toggle();
  });
  document.querySelectorAll(`.${scopeClass} .mission-manual-input, .${scopeClass} .mission-auto-min, .${scopeClass} .mission-auto-max, .${scopeClass} .mission-auto-step, .${scopeClass} .mission-auto-method, .${scopeClass} .mission-auto-mean, .${scopeClass} .mission-auto-std`).forEach((el)=>{
    el.addEventListener('change', ()=>{ updateScopeParamDependencyStates(scopeClass); if(onChange) onChange(); });
    el.addEventListener('input', ()=>{ updateScopeParamDependencyStates(scopeClass); if(onChange) onChange(); });
  });
  updateScopeParamDependencyStates(scopeClass);
}
function parseCsvList(text) {
  return String(text || '').split(',').map((x)=>String(x || '').trim()).filter(Boolean);
}
function selectedMissionCategoryFilters() {
  return [...document.querySelectorAll('.mission-category-filter:checked')].map((el)=>String(el.value || '').trim()).filter(Boolean);
}
function collectLandmarkSetMap() {
  const out = {};
  document.querySelectorAll('.mission-landmark-set-select').forEach((el)=>{
    const id = String(el.dataset.instanceId || '').trim();
    const val = String(el.value || '').trim();
    if(id && val) out[id] = val;
  });
  return out;
}
function setAllMissionCategoryFilters(checked) {
  document.querySelectorAll('.mission-category-filter').forEach((el)=>{ el.checked = !!checked; });
  if(state.currentPage === 'missions') refreshMissionSelectionUi();
}
function renderMissionCategoryFilterTable() {
  const holder = document.getElementById('mission_category_filter_table');
  if(!holder) return;
  const categories = [...new Set((state.missionRows || []).map((row)=>String(row.class_name || '').trim()).filter(Boolean))].sort();
  const selected = new Set(selectedMissionCategoryFilters().length ? selectedMissionCategoryFilters() : categories);
  holder.innerHTML = `<div class="choice-actions" style="margin-bottom:8px;">
    <button type="button" class="secondary" onclick="setAllMissionCategoryFilters(true);">全选</button>
    <button type="button" class="secondary" onclick="setAllMissionCategoryFilters(false);">清空</button>
  </div>
  <table class="compact-table"><thead><tr><th>选</th><th>类别</th><th>地标数</th></tr></thead><tbody>${
    categories.map((name)=>{
      const count = (state.missionRows || []).filter((row)=>String(row.class_name || '').trim() === name).length;
      const checked = selected.has(name) ? 'checked' : '';
      return `<tr><td><input type="checkbox" class="mission-category-filter" value="${esc(name)}" ${checked}></td><td>${esc(name)}</td><td>${count}</td></tr>`;
    }).join('') || '<tr><td colspan="3" class="empty">暂无类别</td></tr>'
  }</tbody></table>`;
  holder.querySelectorAll('.mission-category-filter').forEach((el)=>el.addEventListener('change', ()=>{ if(state.currentPage === 'missions') refreshMissionSelectionUi(); }));
}
function clearMissionSelection() {
  state.selectedMissionInstanceIds = [];
  state.currentMissionHistory = null;
  state.missionDetailMode = 'blank';
  state.missionBehaviorOverrideTouched = false;
  renderMissionEmpty('请先选择地标，然后点击左侧“新增任务”或选择已有 mission。');
  refreshMissionSelectionUi();
}
function syncAutoPickCountByMode() {
  const mode = String(document.getElementById('mission_auto_pick_mode')?.value || 'single');
  const el = document.getElementById('mission_auto_pick_count');
  if(!el) return;
  const defaults = state.uiDefaults || {};
  if(mode === 'multi') el.value = String(defaults.auto_pick_multi_count ?? 3);
  else el.value = String(defaults.auto_pick_single_count ?? 1);
}
function autoSelectMissionLandmarks() {
  const mode = String(document.getElementById('mission_auto_pick_mode')?.value || 'single');
  const count = Math.max(1, Number(document.getElementById('mission_auto_pick_count')?.value || (mode === 'multi' ? 3 : 1)));
  const minPoints = Math.max(0, Number(document.getElementById('mission_auto_pick_min_points')?.value || 0));
  const classes = selectedMissionCategoryFilters().map((x)=>x.toLowerCase());
  const diverse = !!document.getElementById('mission_auto_pick_diverse')?.checked;
  let rows = [...(state.missionRows || [])].filter((row)=>Number(row.point_count || 0) >= minPoints);
  if(classes.length) rows = rows.filter((row)=>classes.includes(String(row.class_name || '').trim().toLowerCase()));
  rows.sort((a,b)=>Number(b.point_count || 0) - Number(a.point_count || 0));
  const selected = [];
  const usedClasses = new Set();
  for(const row of rows) {
    const cls = String(row.class_name || '').trim().toLowerCase();
    if(diverse && cls && usedClasses.has(cls)) continue;
    selected.push(String(row.instance_id));
    if(cls) usedClasses.add(cls);
    if(selected.length >= count) break;
  }
  if(selected.length < count && diverse) {
    for(const row of rows) {
      const id = String(row.instance_id);
      if(selected.includes(id)) continue;
      selected.push(id);
      if(selected.length >= count) break;
    }
  }
  state.selectedMissionInstanceIds = mode === 'single' ? selected.slice(0,1) : selected.slice(0, count);
  if(state.selectedMissionInstanceIds.length) {
    const active = (state.missionRows || []).find((row)=>String(row.instance_id) === String(state.selectedMissionInstanceIds[0]));
    if(active) state.currentMissionInstance = active;
  }
  state.currentMissionHistory = null;
  state.missionDetailMode = 'blank';
  state.missionBehaviorOverrideTouched = false;
  renderMissionEmpty('已自动选择地标。点击左侧“新增任务”开始配置，或选择已有 mission 查看。');
  refreshMissionSelectionUi();
}
function saveGlobalSetProfile(setKey) {
  const key = String(setKey || '').trim();
  if(!key) return;
  const profile = {
    generation_kind: String(document.getElementById('library_generation_kind')?.value || 'auto'),
    behavior_sequence: parseCsvList(document.getElementById('library_behavior_sequence')?.value || ''),
    allow_interleave_repeat: !!document.getElementById('library_allow_interleave_repeat')?.checked,
    max_total_elements: Number(document.getElementById('library_max_total_elements')?.value || 0) || 0,
    element_param_overrides: collectScopeParamOverrides('library-scope'),
    element_auto_rules: collectScopeAutoRules('library-scope'),
  };
  api('/api/stage3_behavior_defaults', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ set_key: key, profile }),
  }).then(async ()=>{
    await loadMissionCatalog();
    if(state.currentPage === 'behavior_library') showBehaviorDetail(key);
    if(state.currentPage === 'missions') refreshMissionSelectionUi();
  });
}
function renderFooter() {
  const left = document.getElementById('footer_left');
  const right = document.getElementById('footer_right');
  if(!left || !right) return;
  const current = state.currentMissionInstance || {};
  const result = state.currentMissionResult || {};
  let leftItems = [];
  let rightItems = [];
  if(state.currentPage === 'behavior_library') {
    leftItems = [
      `<strong>当前页面</strong>：Behavior Library`,
      `<strong>Composite class</strong>：${esc((state.missionCatalog.missions || []).length)} 个`,
      `<strong>Atomic class</strong>：${esc((state.missionCatalog.behaviors || []).length)} 个`,
    ];
    rightItems = [
      `先查看 Composite 结构，再对照 Atomic 参数范围。`,
      `切到“任务生成”页时，可直接复用这里选中的结构理解。`,
    ];
  } else if(state.currentPage === 'missions') {
    leftItems = [
      `<strong>当前页面</strong>：任务生成`,
      `<strong>当前实例</strong>：${esc(current.instance_id || '-')}`,
      `<strong>当前 Composite</strong>：${esc(result.summary?.set_id || current.latest_summary?.set_id || '-')}`,
      `<strong>后台状态</strong>：${esc(current.traj_status || current.status || 'pending')}`,
    ];
    rightItems = [
      `先生成 Preview，再确认后生成 Task Video。`,
      `如果指定 Composite 模板，建议同时检查 Atomic 序列是否符合预期。`,
    ];
  } else if(state.currentPage === 'review') {
    leftItems = [
      `<strong>当前页面</strong>：候选复核`,
      `<strong>候选实例</strong>：${esc(current.instance_id || '-')}`,
      `<strong>复核状态</strong>：${esc(current.review_status || '-')}`,
    ];
    rightItems = [
      `重点核对参考图、Task Video 和关键帧明细是否一致。`,
      `通过后再去生成 manifest，避免把坏样本带进实验。`,
    ];
  } else if(state.currentPage === 'generate') {
    leftItems = [
      `<strong>当前页面</strong>：数据生成`,
      `<strong>候选数</strong>：生成参考区会显示当前可用候选统计。`,
      `<strong>任务组</strong>：支持 Self-State 和 Environmental 两类任务。`,
    ];
    rightItems = [
      `先小样本生成检查 schema，再扩大样本量。`,
      `建议分别为 self-state 和 environmental 生成独立 manifest。`,
    ];
  } else if(state.currentPage === 'dataset') {
    leftItems = [
      `<strong>当前页面</strong>：任务查看`,
      `<strong>当前 Manifest</strong>：${esc(document.getElementById('manifest_select')?.value || '-')}`,
    ];
    rightItems = [
      `优先检查样本字段、地标描述、区间与选项是否匹配。`,
      `如果样本结构不理想，回到生成页调整任务类型组合。`,
    ];
  } else if(state.currentPage === 'experiments') {
    const jobRows = document.querySelectorAll('#job_table tbody tr').length;
    leftItems = [
      `<strong>当前页面</strong>：实验执行`,
      `<strong>当前 Manifest</strong>：${esc(document.getElementById('exp_manifest_select')?.value || '-')}`,
      `<strong>任务数</strong>：${esc(jobRows)}`,
    ];
    rightItems = [
      `本页支持后台任务轮询和取消。`,
      `建议保持相同 manifest 条件再比较不同模型。`,
    ];
  } else if(state.currentPage === 'results') {
    leftItems = [
      `<strong>当前页面</strong>：结果查看`,
      `<strong>当前 Report</strong>：${esc(document.getElementById('report_select')?.value || '-')}`,
    ];
    rightItems = [
      `先看 summary，再下钻到逐样本结果。`,
      `自我状态任务优先看 Composite / Atomic 识别与定位，环境任务优先看 count / interval。`,
    ];
  } else if(state.currentPage === 'metrics') {
    leftItems = [
      `<strong>当前页面</strong>：指标汇总`,
      `<strong>统计方式</strong>：${document.getElementById('metrics_latest_only')?.checked ? 'latest only' : 'all runs'}`,
    ];
    rightItems = [
      `先看顶部摘要卡，再看矩阵。`,
      `如果某类任务明显偏低，回到结果页定位对应样本。`,
    ];
  }
  left.innerHTML = footerHtml(leftItems);
  right.innerHTML = footerHtml(rightItems);
}
function applyTheme(theme) { const t = theme === 'light' ? 'light' : 'dark'; document.body.dataset.theme = t; document.getElementById('theme_select').value = t; localStorage.setItem(THEME_KEY, t); }
function initTheme() { applyTheme(localStorage.getItem(THEME_KEY) || 'light'); document.getElementById('theme_select').onchange = ()=>applyTheme(document.getElementById('theme_select').value); }
function selectedEngine() { return document.getElementById('engine_select')?.value || DEFAULT_ENGINE; }
function selectedScene() { return document.getElementById('scene_select')?.value || DEFAULT_SCENE; }
function selectedTaskPipeline() { return document.getElementById('task_pipeline_select')?.value || localStorage.getItem(TASK_KEY) || ''; }
async function loadCatalog() {
  const data = await api('/api/catalog');
  state.catalog = data || [];
  const engines = [...new Set(state.catalog.map((item)=>item.engine))].sort();
  const eSel = document.getElementById('engine_select');
  eSel.innerHTML = engines.map((eng)=>`<option value="${esc(eng)}" ${eng===DEFAULT_ENGINE?'selected':''}>${esc(eng)}</option>`).join('');
  updateSceneSelect();
  await loadTaskPipelineOptions();
}
function updateSceneSelect() {
  const rows = state.catalog.filter((item)=>item.engine === selectedEngine());
  const preferred = localStorage.getItem('stage3_scene') || __GLOBAL_SCENE__;
  const opts = [{scene_id: __GLOBAL_SCENE__, label: __GLOBAL_SCENE_LABEL__}, ...rows.map((item)=>({scene_id:item.scene_id, label:item.scene_id}))];
  const target = opts.some((item)=>item.scene_id === preferred) ? preferred : (opts[0]?.scene_id || '');
  document.getElementById('scene_select').innerHTML = opts.map((item)=>`<option value="${esc(item.scene_id)}" ${item.scene_id===target?'selected':''}>${esc(item.label)}</option>`).join('');
}
function switchScene() { localStorage.setItem('stage3_scene', selectedScene()); updateSceneSelect(); refreshAll(); }
async function loadTaskPipelineOptions() {
  const data = await api('/api/task_pipeline_tasks');
  const tasks = Array.isArray(data?.tasks) ? data.tasks : [];
  const sel = document.getElementById('task_pipeline_select');
  if(!sel) return;
  const current = selectedTaskPipeline();
  sel.innerHTML = ['<option value="">场景内默认产物</option>', ...tasks.map((name)=>`<option value="${esc(name)}" ${String(name)===String(current)?'selected':''}>${esc(name)}</option>`)].join('');
  sel.onchange = ()=>{ localStorage.setItem(TASK_KEY, sel.value || ''); refreshAll(); };
}
function globalQuery() {
  const q = new URLSearchParams();
  q.set('engine', selectedEngine());
  q.set('scene_id', selectedScene());
  const taskName = selectedTaskPipeline();
  if(taskName) q.set('task_name', taskName);
  return q.toString();
}
function renderIntervals(intervals) {
  return (intervals || []).map((it)=>`[${Number(it.start_sec ?? it[0] ?? 0).toFixed(2)}, ${Number(it.end_sec ?? it[1] ?? 0).toFixed(2)}]`).join('<br>') || '-';
}
async function loadMissionCatalog() {
  const data = await api('/api/stage3_mission_catalog');
  state.missionCatalog = data || {missions: [], behaviors: []};
  refreshMissionTemplateSelect();
}
function showBehaviorDetail(setKey) {
  const sets = state.missionCatalog.missions || [];
  const behaviors = state.missionCatalog.behaviors || [];
  const row = sets.find((item)=>String(item.mission_key) === String(setKey)) || sets[0] || null;
  const byId = Object.fromEntries(behaviors.map((item)=>[String(item.behavior_id), item]));
  const detail = document.getElementById('behavior_detail');
  if(!detail) return;
  if(!row) {
    detail.innerHTML = '<div class="empty">暂无行为库定义</div>';
    return;
  }
  const globalProfile = getSetProfileByKey(String(row.mission_key || ''));
  const stepRows = Array.isArray(row.element_steps) && row.element_steps.length
    ? row.element_steps.map((step, idx)=>({step_index: idx, element_class: String(step.element_class || ''), params: {...(step.params || {})}, auto_rules: {...(step.auto_rules || {})}}))
    : (row.sequence || []).map((bid, idx)=>({step_index: idx, element_class: String(bid || ''), params: {}, auto_rules: {}}));
  const parts = stepRows.map((step)=>{
    const bid = String(step.element_class || '');
    const spec = byId[String(bid)] || {};
    const stepCameraMode = String((step.params || {}).camera_mode || spec.camera_mode_default || '-');
    const stepParamText = Object.keys(step.params || {}).length
      ? Object.entries(step.params || {}).map(([k,v])=>`${k}: ${v}`).join('<br>')
      : '-';
    const params = Object.entries(spec.params || {}).map(([key, meta])=>`${key}: ${meta.min ?? '-'} ~ ${meta.max ?? '-'} / step ${meta.step ?? '-'} / 手动默认 ${meta.default ?? '-'}`).join('<br>');
    return `<div class="card" style="margin-bottom:12px;"><div class="kv">
      <div class="k">step</div><div>${step.step_index + 1}</div>
      <div class="k">element</div><div>${esc(spec.display_name || bid)} / ${esc(bid)}</div>
      <div class="k">说明</div><div>${esc(spec.description || '-')}</div>
      <div class="k">本步视线</div><div>${esc(stepCameraMode)}</div>
      <div class="k">本步参数覆盖</div><div>${stepParamText}</div>
      <div class="k">原子默认视线</div><div>${esc(spec.camera_mode_default || '-')}</div>
      <div class="k">参数</div><div>${params || '-'}</div>
    </div></div>`;
  }).join('');
  const globalEditor = (row.sequence || []).map((bid, idx)=>{
    const spec = byId[String(bid)] || {};
    const params = spec.params || {};
    return `<div class="card library-scope" style="margin-bottom:12px;padding:10px;">
      <div class="summary-item"><strong>Step ${idx + 1}</strong>：${esc(spec.display_name || bid)} / ${esc(bid)}</div>
      ${Object.entries(params).map(([paramKey, meta])=>{
        const manualValue = globalProfile?.element_param_overrides?.[String(idx)]?.[paramKey];
        const autoRule = globalProfile?.element_auto_rules?.[String(idx)]?.[paramKey];
        const currentValue = manualValue ?? meta.default ?? '';
        const autoChecked = autoRule ? true : (manualValue !== undefined ? false : true);
        const methodValue = String(autoRule?.method || 'random');
        const minVal = autoRule?.min ?? meta.min ?? '';
        const maxVal = autoRule?.max ?? meta.max ?? '';
        const stepVal = autoRule?.step ?? meta.step ?? '';
        const meanVal = autoRule?.mean ?? meta.auto_center ?? currentValue;
        const stdVal = autoRule?.std ?? meta.auto_std ?? ((meta.step ?? '') ? Number(meta.step) * 2 : '');
        const choiceOptions = Array.isArray(meta.choices) && meta.choices.length
          ? `<select class="mission-manual-input" data-step-index="${idx}" data-param-key="${esc(paramKey)}">${meta.choices.map((choice)=>`<option value="${esc(choice)}" ${String(currentValue)===String(choice)?'selected':''}>${esc(choice)}</option>`).join('')}</select>`
          : `<input class="mission-manual-input" data-step-index="${idx}" data-param-key="${esc(paramKey)}" value="${esc(currentValue)}">`;
        const disabledAuto = paramKey === 'gaze_pitch_deg' || paramKey === 'yaw_offset_deg' ? 'disabled' : '';
        const fixedTip = disabledAuto ? `<span class="muted">landmark_track 下自动跟随</span>` : '';
        return `<div class="mission-param-card card" data-step-index="${idx}" data-param-key="${esc(paramKey)}" style="margin:8px 0;padding:10px;">
          <div><strong>${esc(paramKey)}</strong> <span class="muted">${esc(meta.label || '')}</span> ${fixedTip}</div>
          <div class="toolbar" style="margin-top:8px;">
            <label>手动值 ${choiceOptions}</label>
            <label><input type="checkbox" class="mission-auto-enabled" data-step-index="${idx}" data-param-key="${esc(paramKey)}" ${autoChecked ? 'checked' : ''} ${disabledAuto}> 自动</label>
            <label>min <input class="mission-auto-min" value="${esc(minVal)}" ${disabledAuto}></label>
            <label>max <input class="mission-auto-max" value="${esc(maxVal)}" ${disabledAuto}></label>
            <label>step <input class="mission-auto-step" value="${esc(stepVal)}" ${disabledAuto}></label>
          </div>
          <div class="toolbar">
            <label>生成方法 <select class="mission-auto-method" ${disabledAuto}><option value="random" ${methodValue==='random'?'selected':''}>random</option><option value="normal" ${methodValue==='normal'?'selected':''}>normal</option></select></label>
            <label>mean <input class="mission-auto-mean" value="${esc(meanVal)}" ${disabledAuto}></label>
            <label>std <input class="mission-auto-std" value="${esc(stdVal)}" ${disabledAuto}></label>
            <span class="muted">手动默认值=${esc(meta.default ?? '-')}</span>
          </div>
        </div>`;
      }).join('') || '<div class="muted">该 element 没有可配置参数。</div>'}
    </div>`;
  }).join('');
  detail.innerHTML = `
    <div class="kv">
      <div class="k">set</div><div>${esc(row.mission_type || '-')} / ${esc(row.mission_subtype || '-')}</div>
      <div class="k">scope</div><div>${esc(row.service_scenario || '-')}</div>
      <div class="k">说明</div><div>${esc(row.description || '-')}</div>
      <div class="k">生成说明</div><div>${esc(row.generation_notes || '-')}</div>
    </div>
    <div style="margin-top:12px;">${parts || '<div class="empty">该 Set 没有 element</div>'}</div>
    <div class="section-title">全局默认值</div>
    <div class="toolbar">
      <label>生成路径 <select id="library_generation_kind"><option value="auto" ${String(globalProfile.generation_kind || 'auto')==='auto'?'selected':''}>auto</option><option value="atomic-only" ${String(globalProfile.generation_kind || '')==='atomic-only'?'selected':''}>atomic-only</option><option value="composite-driven" ${String(globalProfile.generation_kind || '')==='composite-driven'?'selected':''}>composite-driven</option></select></label>
      <label>Atomic 序列 <input id="library_behavior_sequence" value="${esc((globalProfile.behavior_sequence || []).join(','))}" placeholder="留空使用 Composite 默认序列"></label>
      <label><input id="library_allow_interleave_repeat" type="checkbox" ${globalProfile.allow_interleave_repeat ? 'checked' : ''}> 允许交叉重复</label>
      <label>最多 atomic <input id="library_max_total_elements" type="number" value="${esc(globalProfile.max_total_elements ?? 0)}" min="0"></label>
    </div>
    ${globalEditor || '<div class="empty">暂无可编辑参数</div>'}
    <div class="toolbar" style="margin-top:12px;">
      <button class="primary" onclick="saveGlobalSetProfile('${esc(row.mission_key)}')">保存为全局默认值</button>
    </div>`;
  bindScopeParamEditor('library-scope', ()=>{});
}
async function loadBehaviorLibrary() {
  const data = await api('/api/stage3_mission_catalog');
  state.missionCatalog = data || {missions: [], behaviors: []};
  const sets = state.missionCatalog.missions || [];
  const behaviors = state.missionCatalog.behaviors || [];
  const stats = document.getElementById('behavior_stats');
  if(stats) {
    stats.innerHTML = `
      <div class="stat"><div class="muted">Composite class 数</div><div class="value">${sets.length}</div></div>
      <div class="stat"><div class="muted">Atomic class 数</div><div class="value">${behaviors.length}</div></div>
      <div class="stat"><div class="muted">多地标 Composite</div><div class="value">${sets.filter((row)=>String(row.service_scenario || '').includes('multi')).length}</div></div>
      <div class="stat"><div class="muted">家族数</div><div class="value">${(state.missionCatalog.families || []).length}</div></div>`;
  }
  const table = document.getElementById('set_table');
  if(table) {
    table.innerHTML = `<table><thead><tr><th>set_id</th><th>set_name</th><th>scope</th><th>elements</th></tr></thead><tbody>${
      sets.map((row)=>`<tr onclick="showBehaviorDetail('${esc(row.mission_key)}')" style="cursor:pointer;"><td>${esc(row.mission_subtype)}</td><td>${esc(row.mission_type)}</td><td>${esc(row.service_scenario || '-')}</td><td>${esc((row.sequence || []).join(', ') || '-')}</td></tr>`).join('') || '<tr><td colspan="4" class="empty">暂无行为库</td></tr>'
    }</tbody></table>`;
  }
  showBehaviorDetail(String(sets[0]?.mission_key || ''));
  renderFooter();
}
function refreshMissionTemplateSelect() {
  const sel = document.getElementById('mission_type_select');
  const autoSel = document.getElementById('mission_auto_set_candidates');
  const autoRuleSel = document.getElementById('mission_auto_set_rule');
  if(!sel) return;
  const mode = selectedMissionIds().length > 1 ? 'multi-landmark' : 'single-landmark';
  const modeDisplay = document.getElementById('mission_mode_display');
  if(modeDisplay) modeDisplay.value = mode;
  const autoEnabled = document.getElementById('mission_auto')?.checked ?? true;
  const previousValue = sel.dataset.manualValue || sel.value || '';
  const previousAutoValues = autoSel ? [...autoSel.selectedOptions].map((opt)=>String(opt.value || '')) : [];
  const behaviorMap = Object.fromEntries((state.missionCatalog.behaviors || []).map((row)=>[row.behavior_id, row]));
  const missions = (state.missionCatalog.missions || []).filter((row)=> {
    if(mode === 'multi-landmark') return !!row.multi_landmark_component;
    return String(row.service_scenario || '') === 'single-landmark';
  });
  sel.disabled = autoEnabled;
  if(mode === 'multi-landmark') sel.disabled = true;
  sel.innerHTML = `<option value="">自动</option>` + missions.map((row)=>`<option value="${esc(row.mission_key)}">${esc(row.mission_subtype)} | ${esc(row.mission_type)}</option>`).join('');
  if(autoSel) {
    autoSel.innerHTML = missions.map((row)=>`<option value="${esc(row.mission_key)}">${esc(row.mission_subtype)} | ${esc(row.mission_type)}</option>`).join('');
    const desired = previousAutoValues.length ? previousAutoValues : missions.map((row)=>String(row.mission_key));
    [...autoSel.options].forEach((opt)=>{ opt.selected = desired.includes(String(opt.value)); });
    autoSel.disabled = !autoEnabled;
  }
  if(autoRuleSel) autoRuleSel.disabled = !autoEnabled;
  if(!autoEnabled) {
    if(previousValue && missions.some((row)=>row.mission_key === previousValue)) {
      sel.value = previousValue;
    } else if(missions.length) {
      sel.value = missions[0].mission_key;
    }
    sel.dataset.manualValue = sel.value || '';
  } else {
    sel.value = '';
  }
  const info = document.getElementById('mission_template_summary');
  if(info) {
    if(mode === 'multi-landmark') {
      info.innerHTML = '当前为多地标模式。请在左侧“所选地标”区域为每个地标单独指定单地标巡检 Set，或保留“自动选择”让系统逐个地标自动决定。';
    } else if(autoEnabled) {
      info.innerHTML = '当前启用自动 Composite 选择。系统会根据地标尺度、语义类别、局部自由空间和障碍分布自动选择 Composite class。';
    } else {
      const chosen = missions.find((row)=>row.mission_key === sel.value) || missions[0];
      if(chosen) {
        const lowLevel = (chosen.sequence || []).map((bid)=> {
          const spec = behaviorMap[bid] || {};
          return `<div><b>${esc(bid)}</b> ${esc(spec.display_name || '')} <span class="muted">(${esc(spec.description || '-')})</span></div>`;
        }).join('');
        info.innerHTML = `<b>${esc(chosen.mission_type)}</b> / ${esc(chosen.mission_subtype)}<br>${esc(chosen.description || '-')}<br><span class="muted">${esc(chosen.generation_notes || '')}</span><div style="margin-top:8px;">${lowLevel}</div>`;
      } else {
        info.innerHTML = '当前模式下没有可用的 Composite class。';
      }
    }
  }
  sel.onchange = ()=>{
    const nextValue = sel.value || '';
    sel.dataset.manualValue = nextValue;
    const behaviorOverrideEl = document.getElementById('mission_behavior_override');
    if(behaviorOverrideEl) behaviorOverrideEl.value = '';
    state.missionBehaviorOverrideTouched = false;
    state.lastAppliedSetKey = '';
    state.currentMissionSchema = null;
    refreshMissionTemplateSelect();
  };
  if(state.currentPage === 'missions') {
    setTimeout(()=>{ refreshMissionSchema(); }, 0);
  }
}
async function loadMissionInstances() {
  const q = document.getElementById('mission_query')?.value || '';
  const status = document.getElementById('mission_status')?.value || 'all';
  const rows = await api(`/api/instances?q=${encodeURIComponent(q)}&status=${encodeURIComponent(status)}`);
  const stats = {
    total: rows.length,
    ready: rows.filter((row)=>String(row.traj_status || '').includes('ready')).length,
    finalReady: rows.filter((row)=>String(row.traj_status || '') === 'final_ready').length,
    categories: [...new Set(rows.map((row)=>String(row.class_name || '').trim()).filter(Boolean))].length,
  };
  document.getElementById('mission_stats').innerHTML = `
    <div class="stat"><div class="muted">地标数量</div><div class="value">${stats.total}</div></div>
    <div class="stat"><div class="muted">已完成阶段</div><div class="value">${stats.ready}</div></div>
    <div class="stat"><div class="muted">Task Video 完成</div><div class="value">${stats.finalReady}</div></div>
    <div class="stat"><div class="muted">类别数</div><div class="value">${stats.categories}</div></div>`;
  document.getElementById('mission_table').innerHTML = `<table><thead><tr><th>选</th><th>instance</th><th>class</th><th>point_count</th><th>missions</th></tr></thead><tbody>${
    rows.map((row)=>{
      const checked = selectedMissionIds().includes(String(row.instance_id)) ? 'checked' : '';
      const active = String(state.currentMissionInstance?.instance_id || '') === String(row.instance_id) ? ' style="background:var(--list-hover);cursor:pointer;"' : ' style="cursor:pointer;"';
      return `<tr${active} onclick="selectMissionInstance('${esc(row.instance_id)}')">
        <td onclick="event.stopPropagation();"><input type="checkbox" ${checked} onchange="toggleMissionSelection('${esc(row.instance_id)}', this.checked)"></td>
        <td>${esc(row.instance_id)}</td><td>${esc(row.class_name || '-')}</td><td>${esc(row.point_count || 0)}</td><td>${esc(row.mission_history_count || 0)} missions</td>
      </tr>`;
    }).join('') || '<tr><td colspan="5" class="empty">暂无地标记录</td></tr>'
  }</tbody></table>`;
  state.missionRows = rows;
  renderMissionCategoryFilterTable();
  if(state.currentMissionInstance?.instance_id) {
    const latest = rows.find((row)=>String(row.instance_id) === String(state.currentMissionInstance.instance_id));
    if(latest) state.currentMissionInstance = latest;
  } else if(rows.length) {
    state.currentMissionInstance = rows[0];
  }
  if(state.currentMissionInstance && state.missionDetailMode !== 'blank') {
    renderMissionDetail(state.currentMissionInstance, state.currentMissionHistory);
  } else {
    renderMissionEmpty('请先选择地标，然后点击左侧“新增任务”或选择已有 mission。');
  }
  state.lastAppliedSetKey = '';
  await refreshMissionSelectionUi();
  renderFooter();
}
function toggleMissionSelection(instanceId, checked) {
  const ids = selectedMissionIds();
  const next = checked ? [...ids, String(instanceId)] : ids.filter((id)=>id !== String(instanceId));
  state.selectedMissionInstanceIds = [...new Set(next)];
  if(!state.currentMissionInstance || String(state.currentMissionInstance.instance_id || '') !== String(instanceId)) {
    const row = (state.missionRows || []).find((it)=>String(it.instance_id) === String(instanceId));
    if(row) state.currentMissionInstance = row;
  }
  state.currentMissionHistory = null;
  state.missionDetailMode = 'blank';
  state.lastAppliedSetKey = '';
  renderMissionEmpty('地标已更新。点击左侧“新增任务”开始配置，或选择已有 mission 查看。');
  refreshMissionSelectionUi();
}
async function refreshMissionSelectionUi() {
  const ids = selectedMissionIds();
  const summary = document.getElementById('mission_selected_summary');
  if(summary) {
    const rows = ids.map((id)=>{
      const item = (state.missionRows || []).find((row)=>String(row.instance_id) === id) || {};
      const currentSet = String((state.currentMissionSchema?.landmark_set_map || {})[id] || '');
      const options = singleLandmarkInspectionSets().map((row)=>`<option value="${esc(row.mission_key)}" ${currentSet===String(row.mission_key)?'selected':''}>${esc(row.mission_type)}</option>`).join('');
      return `<div class="summary-item"><strong>${esc(id)}</strong> / ${esc(item.class_name || '-')}<span style="margin-left:10px;">单地标巡检 Set</span><select class="mission-landmark-set-select" data-instance-id="${esc(id)}"><option value="">自动选择</option>${options}</select></div>`;
    });
    summary.innerHTML = ids.length ? rows.join('') : '请先选择地标。';
    summary.querySelectorAll('.mission-landmark-set-select').forEach((el)=>el.addEventListener('change', ()=>refreshMissionSelectionUi()));
  }
  const modeDisplay = document.getElementById('mission_mode_display');
  if(modeDisplay) modeDisplay.value = ids.length > 1 ? 'multi-landmark' : 'single-landmark';
  const multiRuleTitle = document.querySelector('.section-title[data-block="multi-rule"]');
  const multiRuleToolbar = document.querySelector('.toolbar[data-block="multi-rule"]');
  if(multiRuleTitle) multiRuleTitle.style.display = ids.length > 1 ? '' : 'none';
  if(multiRuleToolbar) multiRuleToolbar.style.display = ids.length > 1 ? '' : 'none';
  const autoPickMode = document.getElementById('mission_auto_pick_mode');
  if(autoPickMode && ids.length > 1) autoPickMode.value = 'multi';
  refreshMissionTemplateSelect();
  await loadMissionHistory();
  await refreshMissionSchema();
  renderFooter();
}
async function loadMissionHistory() {
  const ids = selectedMissionIds();
  const query = ids.join(',');
  const data = await api(`/api/mission_history?instance_ids=${encodeURIComponent(query)}`);
  state.missionHistoryRows = data.rows || [];
  document.getElementById('mission_history_table').innerHTML = `<table><thead><tr><th>traj_id</th><th>mode</th><th>set</th><th>landmarks</th><th>status</th><th>time</th><th>op</th></tr></thead><tbody>${
    (state.missionHistoryRows || []).map((row)=>{
      const active = String(state.currentMissionHistory?.traj_id || '') === String(row.traj_id || '') ? ' style="background:var(--list-hover);cursor:pointer;"' : ' style="cursor:pointer;"';
      const sum = row.summary || {};
      return `<tr${active} onclick="selectMissionHistory('${esc(row.traj_id)}')">
        <td>${esc(row.traj_id || '-')}</td>
        <td>${esc(row.mission_mode || '-')}</td>
        <td>${esc(sum.set_id || sum.mission_subtype || '-')}</td>
        <td>${esc((row.landmark_instance_ids || []).join(', ') || '-')}</td>
        <td>${esc(row.traj_status || '-')}</td>
        <td>${esc(row.updated_at || row.created_at || '-')}</td>
        <td onclick="event.stopPropagation();"><button class="warn" onclick="deleteMission('${esc(row.traj_id)}')">删除</button></td>
      </tr>`;
    }).join('') || '<tr><td colspan="7" class="empty">当前范围下还没有 mission</td></tr>'
  }</tbody></table>`;
  if(state.currentMissionHistory) {
    const matched = (state.missionHistoryRows || []).find((row)=>String(row.traj_id || '') === String(state.currentMissionHistory.traj_id || ''));
    state.currentMissionHistory = matched || null;
  }
}
async function deleteMission(trajId) {
  await api('/api/mission_delete', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({traj_id: trajId, delete_artifacts: true})});
  if(String(state.currentMissionHistory?.traj_id || '') === String(trajId)) {
    state.currentMissionHistory = null;
    state.missionDetailMode = 'blank';
  }
  await loadMissionInstances();
  await loadMissionHistory();
  if(state.currentMissionInstance) {
    state.lastAppliedSetKey = '';
    await refreshMissionSchema();
    if(state.missionDetailMode === 'blank') renderMissionEmpty('点击左侧“新增任务”或选择已有 mission 后，这里才显示内容。');
    else renderMissionDetail(state.currentMissionInstance, state.currentMissionHistory);
  }
}
async function clearMissionHistory() {
  const ids = selectedMissionIds();
  await api('/api/mission_clear', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({instance_ids: ids, delete_artifacts: true})});
  state.currentMissionHistory = null;
  await loadMissionInstances();
  await loadMissionHistory();
  if(state.currentMissionInstance) {
    state.lastAppliedSetKey = '';
    await refreshMissionSchema();
    state.missionDetailMode = 'blank';
    renderMissionEmpty('任务列表已清空。点击左侧“新增任务”开始新的配置。');
  }
}
function openNewMission() {
  if(!state.currentMissionInstance) {
    renderMissionEmpty('请先选择一个地标，再点击“新增任务”。');
    return;
  }
  state.currentMissionHistory = null;
  state.missionDetailMode = 'create';
  renderMissionDetail(state.currentMissionInstance, null);
}
function selectMissionHistory(trajId) {
  const row = (state.missionHistoryRows || []).find((it)=>String(it.traj_id || '') === String(trajId));
  if(!row) return;
  state.currentMissionHistory = row;
  state.missionDetailMode = 'history';
  renderMissionDetail(state.currentMissionInstance, row);
}
async function refreshMissionSchema() {
  const ids = selectedMissionIds();
  const activeId = String(state.currentMissionInstance?.instance_id || '');
  const payload = {
    instance_id: activeId,
    instance_ids: ids.length ? ids : (activeId ? [activeId] : []),
    mission_mode: ids.length > 1 ? 'multi-landmark' : 'single-landmark',
    mission_type: (document.getElementById('mission_auto')?.checked ?? true) ? null : (document.getElementById('mission_type_select')?.value || null),
    auto_set_rule: document.getElementById('mission_auto_set_rule')?.value || 'heuristic',
    auto_set_candidates: [...(document.getElementById('mission_auto_set_candidates')?.selectedOptions || [])].map((opt)=>String(opt.value || '')),
    landmark_set_map: collectLandmarkSetMap(),
    generation_kind: document.getElementById('mission_generation_kind')?.value || 'auto',
    behavior_sequence: state.missionBehaviorOverrideTouched ? (document.getElementById('mission_behavior_override')?.value || '') : '',
    set_params_auto: document.getElementById('mission_set_params_auto')?.checked ?? true,
    allow_interleave_repeat: document.getElementById('mission_allow_interleave_repeat')?.checked ?? false,
    max_total_elements: Number(document.getElementById('mission_max_total_elements')?.value || 0) || 0,
    set_profiles: readGlobalSetProfiles(),
    seed: 42,
  };
  const box = document.getElementById('mission_step_editor');
  if(!payload.instance_ids.length || !box) {
    if(box) box.innerHTML = '这里会显示当前任务的逐段参数配置。';
    return;
  }
  try {
    const data = await api('/api/mission_generation_schema', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const schema = data.schema || {};
    state.currentMissionSchema = schema;
    const schemaSetKey = String(schema.set_spec?.set_key || '');
    const storedProfile = getSetProfileByKey(schemaSetKey);
    if(schemaSetKey && state.lastAppliedSetKey !== schemaSetKey) {
      state.lastAppliedSetKey = schemaSetKey;
      let changed = false;
      const generationKindEl = document.getElementById('mission_generation_kind');
      const behaviorOverrideEl = document.getElementById('mission_behavior_override');
      const allowRepeatEl = document.getElementById('mission_allow_interleave_repeat');
      const maxTotalEl = document.getElementById('mission_max_total_elements');
      const nextGenerationKind = String(storedProfile.generation_kind || 'auto');
      const nextBehaviorSeq = Array.isArray(storedProfile.behavior_sequence) ? storedProfile.behavior_sequence.join(',') : '';
      const nextAllowRepeat = 'allow_interleave_repeat' in storedProfile ? !!storedProfile.allow_interleave_repeat : !!(state.uiDefaults?.allow_interleave_repeat);
      const nextMaxTotal = 'max_total_elements' in storedProfile ? Number(storedProfile.max_total_elements || 0) : Number(state.uiDefaults?.max_total_elements || 0);
      if(generationKindEl && generationKindEl.value !== nextGenerationKind) { generationKindEl.value = nextGenerationKind; changed = true; }
      if(behaviorOverrideEl && behaviorOverrideEl.value !== nextBehaviorSeq) { behaviorOverrideEl.value = nextBehaviorSeq; state.missionBehaviorOverrideTouched = false; changed = true; }
      if(allowRepeatEl && allowRepeatEl.checked !== nextAllowRepeat) { allowRepeatEl.checked = nextAllowRepeat; changed = true; }
      if(maxTotalEl && String(maxTotalEl.value) !== String(nextMaxTotal)) { maxTotalEl.value = String(nextMaxTotal); changed = true; }
      if(changed) {
        setTimeout(()=>refreshMissionSchema(), 0);
        return;
      }
    }
    const steps = schema.steps || [];
    const forceAuto = !!payload.set_params_auto;
    const header = `<div class="summary-list" style="margin-bottom:10px;">
      <div class="summary-item"><strong>当前生成模式</strong>：${esc(schema.mission_mode || '-')} / ${esc(schema.generation_kind || '-')}</div>
      <div class="summary-item"><strong>当前 Composite</strong>：${esc(schema.set_spec?.set_key || schema.set_instance?.set_id || '-')} / ${esc(schema.set_spec?.display_name || schema.set_instance?.set_name || '-')}</div>
      <div class="summary-item"><strong>Auto Composite 规则</strong>：${esc(schema.auto_set_rule || '-')}；可用 Composite：${esc((schema.auto_set_candidates || []).join(', ') || '-')}</div>
    </div>`;
    box.innerHTML = steps.map((step)=>{
      const specs = step.param_specs || {};
      const current = step.params || {};
      if(forceAuto) {
        const summaryRows = Object.entries(specs).map(([paramKey, spec])=>{
          const currentValue = current[paramKey] ?? spec.default ?? '';
          const autoRule = storedProfile?.element_auto_rules?.[String(step.step_index)]?.[paramKey];
          const manualValue = storedProfile?.element_param_overrides?.[String(step.step_index)]?.[paramKey];
          const modeText = autoRule
            ? String(autoRule.method || 'random')
            : (manualValue !== undefined ? 'global-manual-default' : (Array.isArray(spec.choices) && spec.choices.length ? 'choice-default' : 'global-default'));
          const rangeText = (spec.min !== undefined && spec.min !== null && spec.max !== undefined && spec.max !== null) ? `[${spec.min}, ${spec.max}]` : '-';
          const stepText = spec.step !== undefined && spec.step !== null ? `${spec.step}` : '-';
          return `<div class="summary-item"><strong>${esc(paramKey)}</strong>：value=${esc(String(currentValue))}；range=${esc(rangeText)}；step=${esc(stepText)}；mode=${esc(modeText)}</div>`;
        }).join('');
        return `<div class="card" style="margin-bottom:10px;padding:10px;">
          <div class="summary-item"><strong>Step ${Number(step.step_index) + 1}</strong>：${esc(step.element_display_name || step.element_class || '-')} -> ${esc(step.target_instance_id || '-')}</div>
          <div class="muted" style="margin-top:6px;">按全局默认值自动生成</div>
          <div class="summary-list" style="margin-top:8px; font-size:12px; line-height:1.8;">${summaryRows || '<div class="summary-item">-</div>'}</div>
        </div>`;
      }
      const rows = Object.entries(specs).map(([paramKey, spec])=>{
        const manualValue = forceAuto ? undefined : storedProfile?.element_param_overrides?.[String(step.step_index)]?.[paramKey];
        const autoRule = forceAuto ? undefined : storedProfile?.element_auto_rules?.[String(step.step_index)]?.[paramKey];
        const currentValue = manualValue ?? current[paramKey] ?? spec.default ?? '';
        const stepVal = spec.step ?? '';
        const minVal = spec.min ?? '';
        const maxVal = spec.max ?? '';
        const defaultStd = stepVal ? Number(stepVal) * 2 : '';
        const autoChecked = forceAuto ? true : (autoRule ? true : (manualValue !== undefined ? false : true));
        const methodValue = String(autoRule?.method || 'random');
        const autoMin = autoRule?.min ?? minVal;
        const autoMax = autoRule?.max ?? maxVal;
        const autoStep = autoRule?.step ?? stepVal;
        const autoMean = autoRule?.mean ?? currentValue;
        const autoStd = autoRule?.std ?? defaultStd;
        const manualControl = Array.isArray(spec.choices) && spec.choices.length
          ? `<select class="mission-manual-input" data-step-index="${esc(step.step_index)}" data-param-key="${esc(paramKey)}">${spec.choices.map((choice)=>`<option value="${esc(choice)}" ${String(currentValue)===String(choice)?'selected':''}>${esc(choice)}</option>`).join('')}</select>`
          : `<input class="mission-manual-input" data-step-index="${esc(step.step_index)}" data-param-key="${esc(paramKey)}" value="${esc(currentValue)}">`;
        const disabledAuto = paramKey === 'gaze_pitch_deg' || paramKey === 'yaw_offset_deg' ? 'disabled' : '';
        const fixedTip = disabledAuto ? `<span class="muted">landmark_track 下自动跟随</span>` : '';
        return `<div class="mission-param-card card" data-step-index="${esc(step.step_index)}" data-param-key="${esc(paramKey)}" style="margin:8px 0;padding:10px;">
          <div><strong>${esc(paramKey)}</strong> <span class="muted">${esc(spec.label || '')}</span> ${fixedTip}</div>
          <div class="toolbar" style="margin-top:8px;">
            <label>手动值 ${manualControl}</label>
            <label><input type="checkbox" class="mission-auto-enabled" data-step-index="${esc(step.step_index)}" data-param-key="${esc(paramKey)}" ${autoChecked ? 'checked' : ''} ${disabledAuto}> 自动</label>
            <label>min <input class="mission-auto-min" value="${esc(autoMin)}" ${disabledAuto}></label>
            <label>max <input class="mission-auto-max" value="${esc(autoMax)}" ${disabledAuto}></label>
            <label>step <input class="mission-auto-step" value="${esc(autoStep)}" ${disabledAuto}></label>
          </div>
          <div class="toolbar">
            <label>生成方法 <select class="mission-auto-method" ${disabledAuto}><option value="random" ${methodValue==='random'?'selected':''}>random</option><option value="normal" ${methodValue==='normal'?'selected':''}>normal</option></select></label>
            <label>mean <input class="mission-auto-mean" value="${esc(autoMean)}" ${disabledAuto}></label>
            <label>std <input class="mission-auto-std" value="${esc(autoStd)}" ${disabledAuto}></label>
            <span class="muted">手动默认值=${esc(spec.default ?? '-')}</span><span class="muted mission-param-tip"></span>
          </div>
        </div>`;
      }).join('');
      return `<div class="card" style="margin-bottom:12px;">
        <div class="summary-item"><strong>Step ${Number(step.step_index) + 1}</strong>：${esc(step.element_display_name || step.element_class || '-')} -> ${esc(step.target_instance_id || '-')}</div>
        ${rows || '<div class="muted">该 element 没有可配置参数。</div>'}
      </div>`;
    }).join('') || '<div class="empty">当前任务没有逐段参数。</div>';
    box.innerHTML = header + box.innerHTML;
    bindMissionParamEditor();
    if(state.currentMissionInstance) {
      renderMissionDetail(state.currentMissionInstance, state.currentMissionHistory);
    }
  } catch(err) {
    box.innerHTML = `<div class="empty">参数配置加载失败：${esc(err.message || err)}</div>`;
  }
}
function selectMissionInstance(instanceId) {
  const row = (state.missionRows || []).find((it)=>String(it.instance_id) === String(instanceId));
  if(row) {
    state.currentMissionInstance = row;
    state.currentMissionHistory = null;
    state.missionDetailMode = 'blank';
    state.lastAppliedSetKey = '';
    if(!selectedMissionIds().includes(String(instanceId))) {
      state.selectedMissionInstanceIds = [String(instanceId)];
    }
    renderMissionEmpty('已选中地标。点击左侧“新增任务”开始配置，或选择已有 mission 查看。');
    refreshMissionSelectionUi();
  }
}
function renderMissionDetail(row, historyRow=null) {
  if(!row) return;
  state.currentMissionInstance = row;
  state.currentMissionHistory = historyRow || state.currentMissionHistory || null;
  if(historyRow) state.missionDetailMode = 'history';
  else if(state.missionDetailMode !== 'history') state.missionDetailMode = 'create';
  const latestMission = { asset_urls: row.asset_urls || {}, files: row.latest_files || {}, summary: row.latest_summary || {}, traj_id: row.latest_traj_id || '' };
  let activeMission = historyRow || state.currentMissionHistory || latestMission;
  const hasFinalVideo = !!((activeMission.files || {}).final_video || (activeMission.files || {}).final_video_web || (activeMission.asset_urls || {}).final_video || (activeMission.asset_urls || {}).final_video_web || (activeMission.files || {}).final_video_marked || (activeMission.asset_urls || {}).final_video_marked);
  if(String(row.traj_status || '') === 'final_ready' && !hasFinalVideo) {
    activeMission = latestMission;
    state.currentMissionHistory = null;
  }
  const urls = activeMission.asset_urls || {};
  const files = activeMission.files || {};
  const missionMode = selectedMissionIds().length > 1 ? 'multi-landmark' : 'single-landmark';
  const geometry = row.bbox_3d || [];
  const summary = activeMission.summary || row.latest_summary || {};
  const currentSchema = state.currentMissionSchema || {};
  const currentSteps = Array.isArray(currentSchema.steps) ? currentSchema.steps : [];
  const currentSetName = String(currentSchema.set_spec?.display_name || currentSchema.set_instance?.set_name || '');
  const currentSetKey = String(currentSchema.set_spec?.set_key || currentSchema.set_instance?.set_id || '');
  const landmarkSetMapText = Object.entries(currentSchema.landmark_set_map || {}).map(([k,v])=>`${k} -> ${v}`).join(' | ');
  const multiSetRows = Object.entries(currentSchema.landmark_set_map || {}).map(([k,v])=>`<div class="summary-item"><strong>${esc(k)}</strong>：${esc(v)}</div>`).join('');
  const showMultiSetList = missionMode === 'multi-landmark' && !document.getElementById('mission_allow_interleave_repeat')?.checked;
  const previewVideoUrl = preferredMissionVideo(urls.video_web || files.video_web || '', urls.video || files.video || '');
  const instanceMarkedUrl = preferredMissionVideo(urls.final_video_marked_web || files.final_video_marked_web || '', urls.final_video_marked || files.final_video_marked || '');
  const instancePlainUrl = preferredMissionVideo(urls.final_video_web || files.final_video_web || '', urls.final_video || files.final_video || '');
  const preferWebVideo = !!document.getElementById('mission_video_prefer_web')?.checked;
  const previewVideoPath = preferWebVideo ? (files.video_web || files.video || '-') : (files.video || files.video_web || '-');
  const instanceMarkedPath = preferWebVideo ? (files.final_video_marked_web || files.final_video_marked || '-') : (files.final_video_marked || files.final_video_marked_web || '-');
  const instancePlainPath = preferWebVideo ? (files.final_video_web || files.final_video || '-') : (files.final_video || files.final_video_web || '-');
  const panoramaLeftPath = files.panorama_left || '-';
  const panoramaRightPath = files.panorama_right || '-';
  const metadataPath = files.final_metadata || '-';
  document.getElementById('mission_detail').innerHTML = `
    <div class="section-title">当前配置预览</div>
    <div class="kv">
      <div class="k">当前 Composite</div><div>${esc(currentSetName || '-')} / ${esc(currentSetKey || '-')}</div>
      <div class="k">当前模式</div><div>${esc(missionMode)}</div>
      <div class="k">当前 Atomic 序列</div><div>${esc(currentSteps.map((it)=>it.element_display_name || it.element_class || '-').join(', ') || '-')}</div>
      <div class="k">地标映射</div><div>${esc(landmarkSetMapText || JSON.stringify(currentSchema.landmark_set_map || {}))}</div>
    </div>
    ${showMultiSetList ? `<div class="section-title">多地标巡检 Composite 列表</div><div class="summary-list">${multiSetRows || '<div class="summary-item">-</div>'}</div>` : ''}
    <div class="section-title">已保存任务</div>
    <div class="kv">
      <div class="k">instance</div><div>${esc(row.instance_id)}</div>
      <div class="k">类别</div><div>${esc(row.class_name || row.landmark_category || '-')}</div>
      <div class="k">当前状态</div><div>${esc(row.traj_status || row.status || 'pending')}</div>
      <div class="k">点云数量</div><div>${esc(row.point_count || '-')}</div>
      <div class="k">当前 Composite</div><div>${esc(summary.set_name || summary.mission_type || '-')} / ${esc(summary.set_id || summary.mission_subtype || '-')}</div>
      <div class="k">任务形态</div><div>${esc(summary.task_family || '-')}</div>
      <div class="k">Atomic 序列</div><div>${esc((summary.element_sequence || summary.behavior_sequence || []).join(', ') || '-')}</div>
      <div class="k">bbox_3d</div><div>${esc(JSON.stringify(geometry || []))}</div>
      <div class="k">任务模式</div><div>${esc(missionMode)}</div>
      <div class="k">当前 traj</div><div>${esc(activeMission.traj_id || row.latest_traj_id || '-')}</div>
      <div class="k">备注</div><div>${esc(row.note || '-')}</div>
    </div>
    <div class="section-title">已保存任务 Element Instances</div>
    <div class="pill-row">
      ${((summary.element_instances || []).map((it)=>`<span class="pill">${esc(it.element_display_name || it.element_class || '-')} / ${esc(it.target_instance_id || '-')}</span>`).join('')) || '<span class="muted">暂无实例</span>'}
    </div>
    <div class="section-title">预览与视频</div>
    <div class="preview-grid" style="margin-top:12px;">
      <div><div class="muted">左全景</div><img class="preview" src="${esc(missionImagePreviewUrl(panoramaLeftPath))}"><div class="muted" style="margin-top:6px;word-break:break-all;">${esc(panoramaLeftPath)}</div></div>
      <div><div class="muted">右全景</div><img class="preview" src="${esc(missionImagePreviewUrl(panoramaRightPath))}"><div class="muted" style="margin-top:6px;word-break:break-all;">${esc(panoramaRightPath)}</div></div>
      <div><div class="muted">预览视频</div>${renderVideoPlayer(previewVideoUrl, '')}<div class="muted" style="margin-top:6px;word-break:break-all;">${esc(previewVideoPath)}</div></div>
      <div><div class="muted">实例视频（检查版）</div>${renderVideoPlayer(instanceMarkedUrl, '')}<div class="muted" style="margin-top:6px;word-break:break-all;">${esc(instanceMarkedPath)}</div></div>
      <div><div class="muted">实例视频（纯画面）</div>${renderVideoPlayer(instancePlainUrl, '')}<div class="muted" style="margin-top:6px;word-break:break-all;">${esc(instancePlainPath)}</div></div>
    </div>
    <div class="section-title">Task 元数据</div>
    <div class="toolbar" style="margin-top:12px;">
      <button class="secondary" onclick="loadMissionFinalMeta()">加载 Task 元数据</button>
    </div>
    <div id="mission_final_meta" class="muted">Task 元数据位置：${esc(metadataPath)}</div>`;
  renderFooter();
}
async function loadMissionFinalMeta() {
  const row = state.currentMissionInstance || {};
  const activeMission = state.currentMissionHistory || { asset_urls: row.asset_urls || {} };
  const meta = activeMission.asset_urls?.final_metadata || row.asset_urls?.final_metadata || '';
  const box = document.getElementById('mission_final_meta');
  if(!box) return;
  if(!meta) { box.innerHTML = '当前还没有 Task 元数据。'; return; }
  const data = await api(meta);
  const presence = data.target_presence || {};
  const selfState = data.task_tracks?.self_state_awareness || {};
  box.innerHTML = `<div class="kv">
    <div class="k">可见区间</div><div>${esc(JSON.stringify(presence.intervals_sec || []))}</div>
    <div class="k">区间数量</div><div>${esc((presence.intervals_sec || []).length)}</div>
    <div class="k">Composite 实例</div><div>${esc(selfState.set_instance?.set_name || '-')}</div>
  </div>`;
}
async function generateMission() {
  if(!state.currentMissionInstance) return;
  state.missionDetailMode = 'create';
  const autoEnabled = document.getElementById('mission_auto')?.checked ?? true;
  const ids = selectedMissionIds();
  const body = {
    instance_id: state.currentMissionInstance.instance_id,
    instance_ids: ids.length ? ids : [state.currentMissionInstance.instance_id],
    mission_mode: ids.length > 1 ? 'multi-landmark' : 'single-landmark',
    mission_type: autoEnabled ? null : (document.getElementById('mission_type_select')?.value || null),
    auto_set_rule: document.getElementById('mission_auto_set_rule')?.value || 'heuristic',
    auto_set_candidates: [...(document.getElementById('mission_auto_set_candidates')?.selectedOptions || [])].map((opt)=>String(opt.value || '')),
    landmark_set_map: collectLandmarkSetMap(),
    generation_kind: document.getElementById('mission_generation_kind')?.value || 'auto',
    behavior_sequence: state.missionBehaviorOverrideTouched ? (document.getElementById('mission_behavior_override')?.value || '') : '',
    set_params_auto: document.getElementById('mission_set_params_auto')?.checked ?? true,
    mission_count: Number(document.getElementById('mission_count')?.value || 1) || 1,
    adaptive_sequential_params: document.getElementById('mission_adaptive_params')?.checked ?? true,
    allow_interleave_repeat: document.getElementById('mission_allow_interleave_repeat')?.checked ?? false,
    max_total_elements: Number(document.getElementById('mission_max_total_elements')?.value || 0) || 0,
    element_param_overrides: collectMissionParamOverrides(),
    element_auto_rules: collectMissionAutoRules(),
    set_profiles: readGlobalSetProfiles(),
  };
  const data = await api('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  state.currentMissionResult = data;
  state.currentMissionHistory = null;
  await loadMissionInstances();
  await loadMissionHistory();
  if(state.currentMissionInstance) renderMissionDetail(state.currentMissionInstance, null);
}
async function confirmMissionPanorama(approved) {
  if(!state.currentMissionInstance) return;
  await api('/api/confirm_pano', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({instance_id: state.currentMissionInstance.instance_id, approved: !!approved})});
  await loadMissionInstances();
  await loadMissionHistory();
  selectMissionInstance(state.currentMissionInstance.instance_id);
}
async function generateMissionVideo() {
  if(!state.currentMissionInstance) return;
  state.missionDetailMode = state.currentMissionHistory ? 'history' : 'create';
  const active = activeMissionContext();
  await api('/api/generate_video', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({instance_id: state.currentMissionInstance.instance_id, traj_id: active.traj_id || ''})});
  await loadMissionInstances();
  await loadMissionHistory();
  selectMissionInstance(state.currentMissionInstance.instance_id);
}
async function confirmMissionVideo(approved) {
  if(!state.currentMissionInstance) return;
  state.missionDetailMode = state.currentMissionHistory ? 'history' : 'create';
  const active = activeMissionContext();
  await api('/api/confirm_video', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({instance_id: state.currentMissionInstance.instance_id, traj_id: active.traj_id || '', approved: !!approved})});
  await loadMissionInstances();
  await loadMissionHistory();
  selectMissionInstance(state.currentMissionInstance.instance_id);
}
async function generateMissionFinalTask() {
  if(!state.currentMissionInstance) return;
  state.missionDetailMode = state.currentMissionHistory ? 'history' : 'create';
  const active = activeMissionContext();
  await api('/api/generate_final_task', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({instance_id: state.currentMissionInstance.instance_id, traj_id: active.traj_id || ''})});
  await loadMissionInstances();
  await loadMissionHistory();
  selectMissionInstance(state.currentMissionInstance.instance_id);
}
async function refreshState() {
  const data = await api(`/api/stage3_task_state?${globalQuery()}`);
  const stats = data.candidate_stats || {};
  state.uiDefaults = data.ui_defaults || {};
  const defaults = state.uiDefaults || {};
  const autoPickMode = document.getElementById('mission_auto_pick_mode');
  const autoPickCount = document.getElementById('mission_auto_pick_count');
  const autoPickMinPoints = document.getElementById('mission_auto_pick_min_points');
  const missionCount = document.getElementById('mission_count');
  const autoSetRule = document.getElementById('mission_auto_set_rule');
  const allowRepeat = document.getElementById('mission_allow_interleave_repeat');
  const maxTotal = document.getElementById('mission_max_total_elements');
  if(autoPickMode && !autoPickMode.dataset.initialized) {
    autoPickMode.value = 'single';
    autoPickMode.dataset.initialized = '1';
  }
  if(autoPickCount && !autoPickCount.dataset.initialized) {
    autoPickCount.value = String(defaults.auto_pick_single_count ?? 1);
    autoPickCount.dataset.initialized = '1';
  }
  if(autoPickMinPoints && !autoPickMinPoints.dataset.initialized) {
    autoPickMinPoints.value = String(defaults.auto_pick_min_points ?? 500);
    autoPickMinPoints.dataset.initialized = '1';
  }
  if(missionCount && !missionCount.dataset.initialized) {
    missionCount.value = String(defaults.mission_count ?? 1);
    missionCount.dataset.initialized = '1';
  }
  if(autoSetRule && !autoSetRule.dataset.initialized) {
    autoSetRule.value = String(defaults.auto_set_rule ?? 'heuristic');
    autoSetRule.dataset.initialized = '1';
  }
  if(allowRepeat && !allowRepeat.dataset.initialized) {
    allowRepeat.checked = !!defaults.allow_interleave_repeat;
    allowRepeat.dataset.initialized = '1';
  }
  if(maxTotal && !maxTotal.dataset.initialized) {
    maxTotal.value = String(defaults.max_total_elements ?? 0);
    maxTotal.dataset.initialized = '1';
  }
  const box = document.getElementById('candidate_stats');
  if(box) box.innerHTML = `
    <div class="stat"><div class="muted">候选数</div><div class="value">${stats.candidate_count ?? 0}</div></div>
    <div class="stat"><div class="muted">已通过</div><div class="value">${stats.approved_count ?? 0}</div></div>
    <div class="stat"><div class="muted">类别数</div><div class="value">${stats.category_count ?? 0}</div></div>
    <div class="stat"><div class="muted">平均可见次数</div><div class="value">${fmtFloat(stats.avg_visible_count, 2)}</div></div>`;
  const genRef = document.getElementById('generate_reference');
  if(genRef) genRef.innerHTML = `<div class="kv">
    <div class="k">候选数</div><div>${stats.candidate_count ?? 0}</div>
    <div class="k">可用于单地标</div><div>${stats.single_landmark_count ?? 0}</div>
    <div class="k">可用于多地标</div><div>${stats.multi_landmark_count ?? 0}</div>
    <div class="k">Composite 类别</div><div>${esc((stats.mission_families || []).join(', ') || '-')}</div>
    <div class="k">最新 manifest</div><div>${esc(data.latest_manifest_path || '-')}</div>
  </div>`;
}
async function loadCandidates() {
  const q = document.getElementById('candidate_query')?.value || '';
  const status = document.getElementById('candidate_status')?.value || 'all';
  const data = await api(`/api/stage3_candidates?${globalQuery()}&q=${encodeURIComponent(q)}&status=${encodeURIComponent(status)}`);
  const rows = data.rows || [];
  document.getElementById('candidate_table').innerHTML = `<table><thead><tr><th>traj_id</th><th>mode</th><th>set</th><th>landmark</th><th>visible count</th><th>difficulty</th><th>status</th></tr></thead><tbody>${
    rows.map((row)=>`<tr onclick="loadCandidateDetail('${esc(row.traj_id)}')" style="cursor:pointer;">
      <td>${esc(row.traj_id)}</td><td>${esc(row.mode)}</td><td>${esc(row.set_id || row.mission_subtype || row.mission_type || '-')}</td><td>${esc(row.landmark_category)} / ${esc(row.landmark_id)}</td><td>${esc(row.visible_count)}</td><td>${esc(row.difficulty_band)}</td><td>${esc(row.review_status)}</td>
    </tr>`).join('') || '<tr><td colspan="7" class="empty">暂无候选任务</td></tr>'
  }</tbody></table>`;
}
async function loadCandidateDetail(trajId) {
  const data = await api(`/api/stage3_candidate?${globalQuery()}&traj_id=${encodeURIComponent(trajId)}`);
  const row = data.candidate || {};
  let keyframeHtml = '<div class="empty">暂无关键帧</div>';
  let keyframeRows = [];
  try {
    const taskMeta = row.final_meta_path ? await fetch(`/artifact?path=${encodeURIComponent(row.final_meta_path)}`).then(r=>r.json()) : null;
    keyframeRows = (((taskMeta || {}).task_tracks || {}).environmental_awareness || {}).keyframe_gt_dense || [];
  } catch(_err) {
    keyframeRows = [];
  }
  if(Array.isArray(keyframeRows) && keyframeRows.length) {
    const finalManifest = (((await fetch(`/artifact?path=${encodeURIComponent((row.final_meta_path || '').replace('task_data.json','frames_manifest.json'))}`).then(r=>r.json()).catch(()=>({}))) || {}).frames || []);
    keyframeHtml = `<div class="preview-grid">${keyframeRows.slice(0, 16).map((kf)=>{
      const frame = Number(kf.frame ?? 0);
      const rel = finalManifest[frame] || '';
      const baseDir = String(row.final_meta_path || '').replace('task_data.json', '');
      const framePath = rel ? `${baseDir}/${rel}` : '';
      return `<div><div class="muted">frame ${esc(frame)} / ${esc(Number(kf.time_sec ?? 0).toFixed(2))}s</div><img class="preview" src="/artifact?path=${encodeURIComponent(framePath)}"></div>`;
    }).join('')}</div>`;
  }
  document.getElementById('candidate_detail').innerHTML = `
    <div class="toolbar">
      <button class="secondary" onclick="reviewCandidate('${esc(trajId)}','approved')">通过</button>
      <button class="warn" onclick="reviewCandidate('${esc(trajId)}','rejected')">驳回</button>
      <button class="secondary" onclick="reviewCandidate('${esc(trajId)}','pending')">重置</button>
    </div>
    <div class="kv">
      <div class="k">Composite</div><div>${esc(row.set_name || row.mission_type || '-')} / ${esc(row.set_id || row.mission_subtype || '-')}</div>
      <div class="k">Atomic 序列</div><div>${esc((row.element_sequence || row.low_level_sequence || []).join(', ') || '-')}</div>
      <div class="k">地标描述</div><div>${esc(row.landmark_description || '-')}</div>
      <div class="k">飞行描述</div><div>${esc(row.flight_description || '-')}</div>
    </div>
    <div class="section-title">任务摘要</div>
    <div class="mini-kv">
      <div class="mini-card"><div class="muted">可见次数</div><div>${esc(row.visible_count ?? '-')}</div></div>
      <div class="mini-card"><div class="muted">难度</div><div>${esc(row.difficulty_band || '-')}</div></div>
      <div class="mini-card"><div class="muted">Atomic 数</div><div>${esc((row.element_instances || []).length)}</div></div>
      <div class="mini-card"><div class="muted">地标类别</div><div>${esc(row.landmark_category || '-')}</div></div>
    </div>
    <div class="preview-grid" style="margin-top:12px;">
      <div><div class="muted">参考图</div><img class="preview" src="/artifact?path=${encodeURIComponent(row.reference_image_with_bbox || '')}"></div>
      <div><div class="muted">总览图（可选）</div><img class="preview" src="/artifact?path=${encodeURIComponent(row.overview_image || '')}"></div>
      <div><div class="muted">整秒关键帧看板</div><img class="preview" src="/artifact?path=${encodeURIComponent(row.keyframe_board_image || '')}"></div>
    </div>
    <div style="margin-top:14px;">
      <div class="muted" style="margin-bottom:8px;">整秒关键帧明细</div>
      ${keyframeHtml}
    </div>
    <div class="muted" style="margin-top:12px;">当前复核状态：${esc(row.review_status || 'pending')}</div>`;
  renderFooter();
}
async function reviewCandidate(trajId, status) {
  await api('/api/stage3_review_candidate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({engine:selectedEngine(), scene_id:selectedScene(), task_name:selectedTaskPipeline(), traj_id:trajId, status})});
  await loadCandidates();
  await loadCandidateDetail(trajId);
}
async function generateManifest() {
  const forms = [...document.querySelectorAll('.gen_form:checked')].map((el)=>el.value);
  const body = {
    engine: selectedEngine(),
    scene_id: selectedScene(),
    task_name: selectedTaskPipeline(),
    mode: document.getElementById('gen_mode').value,
    sample_count: Number(document.getElementById('gen_sample_count').value || 24),
    seed: Number(document.getElementById('gen_seed').value || 7),
    approved_only: document.getElementById('gen_approved_only').checked,
    forms,
    include_temporal_localization: document.getElementById('gen_include_temporal_localization').checked,
  };
  const data = await api('/api/stage3_generate_manifest', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  document.getElementById('generate_feedback').innerText = `Manifest 已生成：${data.manifest_path}`;
  await loadManifestList();
}
async function loadManifestList() {
  const data = await api(`/api/stage3_manifests?${globalQuery()}`);
  const selA = document.getElementById('manifest_select');
  const selB = document.getElementById('exp_manifest_select');
  const opts = (data || []).map((row)=>`<option value="${esc(row.path)}">${esc(row.generated_at || row.path)} | n=${row.summary?.sample_count ?? 0}</option>`).join('');
  if(selA) selA.innerHTML = opts;
  if(selB) selB.innerHTML = opts;
  if((data || []).length) {
    await loadManifestDetail();
    await loadExperimentManifestSummary();
  }
}
async function loadManifestDetail() {
  const path = document.getElementById('manifest_select')?.value;
  if(!path) return;
  const data = await api(`/api/stage3_manifest?path=${encodeURIComponent(path)}`);
  const s = data.summary || {};
  state.activeManifestSampleId = state.activeManifestSampleId || String((data.samples || [])[0]?.sample_id || '');
  document.getElementById('manifest_summary').innerHTML = `<div class="kv">
    <div class="k">样本数</div><div>${s.sample_count ?? 0}</div>
    <div class="k">轨迹数</div><div>${s.mission_count ?? 0}</div>
    <div class="k">地标数</div><div>${s.landmark_count ?? 0}</div>
    <div class="k">类别</div><div>${esc((s.categories || []).join(', ') || '-')}</div>
    <div class="k">任务组</div><div>${esc((s.task_groups || []).join(', ') || '-')}</div>
    <div class="k">任务类型</div><div>${esc((s.task_names || []).join(', ') || '-')}</div>
    <div class="k">难度</div><div>${esc((s.difficulty_bands || []).join(', ') || '-')}</div>
  </div>`;
  const rows = (data.samples || []);
  document.getElementById('manifest_sample_list').innerHTML = `<table><thead><tr><th>sample_id</th><th>task</th><th>difficulty</th><th>landmark</th></tr></thead><tbody>${
    rows.map((row)=>`<tr onclick="selectManifestSample('${esc(row.sample_id)}')" style="cursor:pointer;${String(state.activeManifestSampleId)===String(row.sample_id)?'background:var(--panel-soft);':''}"><td>${esc(row.sample_id)}</td><td>${esc(row.task_display_name || row.form)}</td><td>${esc(row.difficulty_band)}</td><td>${esc(row.landmark_id)}</td></tr>`).join('') || '<tr><td colspan="4" class="empty">暂无样本</td></tr>'
  }</tbody></table>`;
  renderManifestSample(rows.find((row)=>String(row.sample_id)===String(state.activeManifestSampleId)) || rows[0] || null);
  renderFooter();
}
function renderManifestSample(row) {
  const holder = document.getElementById('manifest_samples');
  if(!holder) return;
  if(!row) {
    holder.innerHTML = '<div class="empty">暂无样本</div>';
    return;
  }
  state.activeManifestSampleId = String(row.sample_id || '');
  const answerItems = Array.isArray(row.answer_items) ? row.answer_items : [];
  const choiceOptions = Array.isArray(row.choice_options) ? row.choice_options : [];
  const answerText = answerItems.length
    ? answerItems.map((it)=>`<div style="margin-bottom:8px;"><strong>${esc(it.option_id || '-')}</strong> / ${esc(it.label || '-')}<div class="muted">区间：${renderIntervals(it.intervals_sec || [])}</div></div>`).join('')
    : (row.gold_environment_answer ? `<pre style="white-space:pre-wrap;">${esc(JSON.stringify(row.gold_environment_answer, null, 2))}</pre>` : '-');
  const optionText = choiceOptions.length
    ? `<div class="option-grid">${choiceOptions.map((it)=>`<div class="option-card"><div><strong>${esc(it.option_id || '-')}</strong></div><div>${esc(it.label || '-')}</div></div>`).join('')}</div>`
    : '<div class="muted">该任务没有离散选项。</div>';
  const behaviorText = (row.behavior_intervals_sec || []).map((it)=>`<div style="margin-bottom:8px;"><strong>${esc(it.event_label || it.behavior_id || '-')}</strong><div class="muted">区间：${renderIntervals(it.intervals_sec || [])}</div></div>`).join('') || '-';
  const envAnswerText = row.task_group === 'environmental'
    ? `<div class="summary-item"><strong>可见次数</strong>：${esc(row.visible_count ?? 0)}</div>
       <div class="summary-item"><strong>可见时间区间</strong>：${renderIntervals(row.visible_intervals_sec || [])}</div>`
    : '';
  const referenceImg = row.reference_image_with_bbox ? `<div class="thumb-box"><div class="muted" style="margin-bottom:4px;">查询地标参考图</div><img class="preview" src="/artifact?path=${encodeURIComponent(row.reference_image_with_bbox)}"></div>` : '';
  const overviewImg = row.overview_image ? `<div class="thumb-box"><div class="muted" style="margin-bottom:4px;">总览图</div><img class="preview" src="/artifact?path=${encodeURIComponent(row.overview_image)}"></div>` : '';
  const keyframeImg = row.keyframe_board_image ? `<div class="thumb-box"><div class="muted" style="margin-bottom:4px;">关键帧看板</div><img class="preview" src="/artifact?path=${encodeURIComponent(row.keyframe_board_image)}"></div>` : '';
  const videoPlayer = renderVideoPlayer(row.video_web_path || row.video_path, row.video_path || row.video_web_path);
    holder.innerHTML = `
    <div class="card" style="margin-bottom:12px;">
      <h3 style="margin-top:0;">样本总体信息</h3>
      <div class="kv">
        <div class="k">sample_id</div><div>${esc(row.sample_id)}</div>
        <div class="k">任务类型</div><div>${esc(row.task_group)} / ${esc(row.task_display_name || row.form)} / ${esc(row.difficulty_band)}</div>
        <div class="k">mission_id</div><div>${esc(row.mission_id || '-')}</div>
        <div class="k">Composite / Task</div><div>${esc(row.set_id || '-')} / ${esc(row.task_type_label || '-')}</div>
        <div class="k">查询地标</div><div>${esc(row.landmark_id)} / ${esc(row.landmark_category || '-')} / ${esc(row.landmark_subcategory || '-')}</div>
        <div class="k">地标描述</div><div>${esc(row.landmark_description || '-')}</div>
        <div class="k">飞行描述</div><div>${esc(row.flight_description || '-')}</div>
        <div class="k">视频路径</div><div style="word-break:break-all;">${esc(row.video_web_path || row.video_path || '-')}</div>
        <div class="k">候选来源</div><div style="word-break:break-all;">${esc(row.candidate_path || '-')}</div>
        <div class="k">视频规格</div><div>${esc(row.video_width || '-')} x ${esc(row.video_height || '-')} / ${esc(row.fps || '-')} FPS / ${esc(row.frame_count || '-')} frames</div>
      </div>
      <div class="thumb-row" style="margin-top:12px;">
        ${referenceImg}
        ${overviewImg}
        ${keyframeImg}
      </div>
      <div style="margin-top:14px;">
        <div class="muted" style="margin-bottom:6px;">样本视频</div>
        ${videoPlayer}
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0;">QA 展示</h3>
      <div class="summary-list" style="margin:8px 0 14px;">
        <div class="summary-item"><strong>System Prompt</strong>：<pre style="white-space:pre-wrap;">${esc(row.system_prompt || '')}</pre></div>
        <div class="summary-item"><strong>User Prompt</strong>：<pre style="white-space:pre-wrap;">${esc(row.user_prompt || row.prompt_text || '')}</pre></div>
        <div class="summary-item"><strong>选项</strong>：${optionText}</div>
        <div class="summary-item"><strong>标准答案</strong>：${answerText}</div>
        ${envAnswerText}
        <div class="summary-item"><strong>行为时间区间</strong>：${behaviorText}</div>
      </div>
    </div>`;
}
function selectManifestSample(sampleId) {
  state.activeManifestSampleId = String(sampleId || '');
  loadManifestDetail();
}
async function loadExperimentManifestSummary() {
  const path = document.getElementById('exp_manifest_select')?.value;
  if(!path) return;
  const data = await api(`/api/stage3_manifest?path=${encodeURIComponent(path)}`);
  const s = data.summary || {};
  document.getElementById('experiment_manifest_summary').innerText = `样本数=${s.sample_count ?? 0}；轨迹数=${s.mission_count ?? 0}；地标数=${s.landmark_count ?? 0}；任务组=${(s.task_groups || []).join(', ') || '-'}；任务类型=${(s.task_names || []).join(', ') || '-'}；难度=${(s.difficulty_bands || []).join(', ') || '-'}`;
}
async function startExperiment() {
  const body = {
    engine: selectedEngine(),
    scene_id: selectedScene(),
    task_name: selectedTaskPipeline(),
    manifest_path: document.getElementById('exp_manifest_select').value,
    models: (document.getElementById('exp_models').value || '').split(',').map((x)=>x.trim()).filter(Boolean),
    limit: Number(document.getElementById('exp_limit').value || 0) || null,
    upload_max_width: Number(document.getElementById('exp_upload_w').value || 640),
    upload_max_height: Number(document.getElementById('exp_upload_h').value || 480),
    upload_jpeg_quality: Number(document.getElementById('exp_upload_q').value || 80),
    concurrency: Number(document.getElementById('exp_concurrency').value || 1),
    rpm_limit: Number(document.getElementById('exp_rpm').value || 0),
    tpm_limit: Number(document.getElementById('exp_tpm').value || 0),
    provide_flight_description: document.getElementById('exp_flight_description').checked,
    include_keyframes: document.getElementById('exp_include_keyframes').checked,
  };
  await api('/api/stage3_start_experiment', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  await refreshJobs();
}
async function refreshJobs() {
  const rows = await api('/api/stage3_jobs');
  document.getElementById('job_table').innerHTML = `<table><thead><tr><th>job_id</th><th>status</th><th>model</th><th>manifest</th><th>progress</th><th>runs</th><th>op</th></tr></thead><tbody>${
    (rows || []).map((row)=>`<tr><td>${esc(row.job_id)}</td><td>${esc(row.status)}</td><td>${esc(row.payload?.model || ((row.payload?.models || [])[0] || '-'))}</td><td>${esc(row.payload?.manifest_name || '-')}</td><td>${esc(row.progress?.completed ?? 0)} / ${esc(row.progress?.total ?? 0)}<br><span class="muted">${esc(row.progress?.sample_id || '')}</span></td><td>${esc((row.runs || []).length)}</td><td>${['queued','running','cancel_requested'].includes(row.status) ? `<button class="warn" onclick="cancelJob('${esc(row.job_id)}')">取消</button>` : ''}</td></tr>`).join('') || '<tr><td colspan="7" class="empty">暂无任务</td></tr>'
  }</tbody></table>`;
  renderFooter();
}
async function cancelJob(jobId) { await api('/api/stage3_cancel_job', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({job_id:jobId})}); await refreshJobs(); }
async function loadReportList() {
  const data = await api(`/api/stage3_reports?${globalQuery()}`);
  const sel = document.getElementById('report_select');
  sel.innerHTML = (data || []).map((row)=>`<option value="${esc(row.path)}">${esc(row.generated_at || row.path)} | ${esc(row.model || '-')} | ${esc(row.manifest_name || '-')}</option>`).join('');
  if((data || []).length) await loadReportDetail();
}
async function loadReportDetail() {
  const path = document.getElementById('report_select')?.value;
  if(!path) return;
  const data = await api(`/api/stage3_report?path=${encodeURIComponent(path)}`);
  const s = data.summary || {};
  document.getElementById('report_summary').innerHTML = `<div class="kv">
    <div class="k">模型</div><div>${esc(data.model || '-')}</div>
    <div class="k">Manifest</div><div>${esc(data.manifest_name || '-')}</div>
    <div class="k">自我状态样本数</div><div>${esc(s.self_state_count ?? 0)}</div>
    <div class="k">环境观察样本数</div><div>${esc(s.environmental_count ?? 0)}</div>
    <div class="k">次数完全正确率</div><div>${fmtPct(s.count_exact_acc)}</div>
    <div class="k">Composite 实例识别准确率</div><div>${fmtPct(s.set_instance_acc)}</div>
    <div class="k">Atomic 实例识别 F1</div><div>${fmtPct(s.element_instance_f1)}</div>
    <div class="k">自我感知时序定位 F1@0.5</div><div>${fmtPct(s['self_temporal_loc_f1@0.5'])}</div>
    <div class="k">自我感知时序定位 mean tIoU</div><div>${fmtPct(s['self_temporal_loc_mean_tIoU'])}</div>
    <div class="k">次数 ±1 正确率</div><div>${fmtPct(s.count_within1_acc)}</div>
    <div class="k">区间 F1@0.5</div><div>${fmtPct(s['segment_f1@0.5'])}</div>
    <div class="k">环境区间 mean tIoU</div><div>${fmtPct(s['mean_best_tIoU'])}</div>
  </div>`;
  const rows = data.rows || [];
  const byTask = {};
  rows.forEach((row)=>{ const key = String(row.task_name || row.form || 'unknown'); (byTask[key] ||= []).push(row); });
  function answerItemsHtml(items) {
    const rows = Array.isArray(items) ? items : [];
    if(!rows.length) return '<div class="muted">-</div>';
    return rows.map((item)=>`
      <div class="mini-card" style="margin-bottom:8px;">
        <div><strong>${esc(item.option_id || '-')}</strong> / ${esc(item.label || '-')}</div>
        <div class="muted">区间：${renderIntervals(item.intervals_sec || [])}</div>
      </div>`).join('');
  }
  document.getElementById('report_rows').innerHTML = Object.entries(byTask).map(([task, taskRows])=>{
    const isEnv = task === 'env_visibility_reasoning';
    const cards = taskRows.map((row)=>{
      if(isEnv) {
        const kf = !!row.include_keyframes;
        return `<div class="card" style="margin-bottom:12px;">
          <div class="kv">
            <div class="k">sample_id</div><div>${esc(row.sample_id)}</div>
            <div class="k">difficulty</div><div>${esc(row.difficulty_band)}</div>
            <div class="k">gold count</div><div>${esc(row.gold_visible_count)}</div>
            <div class="k">pred count</div><div>${esc(row.pred_visible_count)}</div>
            <div class="k">gold intervals</div><div>${renderIntervals(row.gold_visible_intervals_sec || [])}</div>
            <div class="k">pred intervals</div><div>${renderIntervals(row.pred_visible_intervals_sec || [])}</div>
            ${kf ? `<div class="k">BBox Acc@50IoU</div><div>${fmtPct(row['bbox_acc@50iou'])}</div>` : ''}
          </div>
        </div>`;
      }
      return `<div class="card" style="margin-bottom:12px;">
        <div class="kv">
          <div class="k">sample_id</div><div>${esc(row.sample_id)}</div>
          <div class="k">difficulty</div><div>${esc(row.difficulty_band)}</div>
          <div class="k">gold options</div><div>${esc((row.gold_option_ids || [row.gold_option_id || '-']).join(','))}</div>
          <div class="k">pred options</div><div>${esc((row.pred_option_ids || [row.pred_option_id || '-']).join(','))}</div>
          <div class="k">tIoU</div><div>${fmtPct(row.self_temporal_mean_tiou)}</div>
        </div>
        <div class="grid-2" style="margin-top:12px;">
          <div>
            <div class="section-title">Gold</div>
            ${answerItemsHtml(row.gold_answer_items || [])}
          </div>
          <div>
            <div class="section-title">Pred</div>
            ${answerItemsHtml(row.pred_answer_items || [])}
          </div>
        </div>
      </div>`;
    }).join('');
    return `<div style="margin-bottom:16px;"><div class="section-title">${esc(task)}</div>${cards}</div>`;
  }).join('') || '<div class="empty">暂无结果</div>';
  renderFooter();
}
async function loadMetricsMatrix() {
  const latestOnly = document.getElementById('metrics_latest_only').checked ? '1' : '0';
  const byDifficulty = document.getElementById('metrics_by_difficulty').checked ? '1' : '0';
  const mq = globalQuery();
  const mmEl = document.getElementById('metrics_matrix');
  const pmEl = document.getElementById('metrics_progress_matrix');
  if(mmEl) mmEl.innerHTML = '<div class="muted">加载中...</div>';
  if(pmEl) pmEl.innerHTML = '<div class="muted">加载中...</div>';
  const psEl = document.getElementById('metrics_progress_summary');
  if(psEl) psEl.innerHTML = '';
  let reports = [];
  let data = { columns: [], rows: [] };
  let progressData = { scenes: [], rows: [], overall_completed: 0, overall_total: 0, overall_ratio: null };
  try {
    [reports, data, progressData] = await Promise.all([
      api(`/api/stage3_reports?${mq}`).catch(()=>[]),
      api(`/api/stage3_metrics_matrix?${mq}&latest_only=${latestOnly}&by_difficulty=${byDifficulty}`).catch(()=>({columns:[],rows:[]})),
      api(`/api/stage3_experiment_progress_matrix?${mq}`).catch(()=>({scenes:[],rows:[],overall_completed:0,overall_total:0,overall_ratio:null})),
    ]);
  } catch (e) {
    const msg = esc(String((e && e.message) || e));
    if(mmEl) mmEl.innerHTML = `<div class="warn">加载失败：${msg}</div>`;
    if(pmEl) pmEl.innerHTML = '';
    renderFooter();
    return;
  }
  const latestReport = (reports || [])[0] || {};
  const latestSummary = latestReport.summary || {};
  const columns = data.columns || [];
  const rows = data.rows || [];
  const summaryCards = document.getElementById('metrics_summary_cards');
  if(summaryCards) {
    const totalModels = rows.length;
    const totalCombos = columns.length;
    const filled = rows.reduce((acc, row)=>acc + Object.values(row.combos || {}).filter((cell)=>Number(cell.count || 0) > 0).length, 0);
    summaryCards.innerHTML = `
      <div class="stat"><div class="muted">模型数</div><div class="value">${totalModels}</div></div>
      <div class="stat"><div class="muted">组合数</div><div class="value">${totalCombos}</div></div>
      <div class="stat"><div class="muted">已填充单元</div><div class="value">${filled}</div></div>
      <div class="stat"><div class="muted">Parse Success</div><div class="value">${fmtPct(latestSummary.parse_success_rate)}</div></div>
      <div class="stat"><div class="muted">Composite</div><div class="value">${fmtPct(latestSummary.set_instance_acc)}</div></div>
      <div class="stat"><div class="muted">Atomic F1</div><div class="value">${fmtPct(latestSummary.element_instance_f1)}</div></div>
      <div class="stat"><div class="muted">Count Exact</div><div class="value">${fmtPct(latestSummary.count_exact_acc)}</div></div>
      <div class="stat"><div class="muted">Segment F1@0.5</div><div class="value">${fmtPct(latestSummary['segment_f1@0.5'])}</div></div>
      <div class="stat"><div class="muted">Env tIoU</div><div class="value">${fmtPct(latestSummary['mean_best_tIoU'])}</div></div>
      <div class="stat"><div class="muted">Self tIoU</div><div class="value">${fmtPct(latestSummary['self_temporal_loc_mean_tIoU'])}</div></div>
      <div class="stat"><div class="muted">统计方式</div><div class="value">${latestOnly === '1' ? 'latest only' : 'all runs'}</div></div>`;
  }
  const grouped = latestSummary.grouped || {};
  const groupHolder = document.getElementById('metrics_group_bars');
  function barSection(title, payload) {
    if(!payload || Object.keys(payload).length === 0) return `<div class="muted">${title}: 暂无数据</div>`;
    return `<div style="margin-bottom:18px;"><h3>${title}</h3><div class="bars">${Object.entries(payload).map(([name, item])=>`
      <div class="bar-row">
        <div>${esc(name)}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, Number(item.main_metric || 0) * 100))}%"></div></div>
        <div>${fmtPct(item.main_metric)}</div>
      </div>`).join('')}</div></div>`;
  }
  if(groupHolder) {
    groupHolder.innerHTML = [
      barSection('按任务类型主指标', grouped.form),
      barSection('按模式主指标', grouped.mode),
      barSection('按难度主指标', grouped.difficulty_band),
    ].join('');
  }
  const groups = [];
  for(const col of columns) {
    const last = groups.length ? groups[groups.length - 1] : null;
    const metricCount = Array.isArray(col.metrics) && col.metrics.length ? col.metrics.length : 0;
    if(last && last.mode === col.mode && last.form === col.form) last.columns.push(col); else groups.push({mode:col.mode, form:col.form, columns:[col], metricCount});
  }
  const top = groups.map((g)=>`<th colspan="${g.columns.reduce((acc,col)=>acc + ((col.metrics || []).length || 0),0)}">${esc(g.mode)} / ${esc(g.form)}</th>`).join('');
  const mid = groups.map((g)=>g.columns.map((c)=>`<th colspan="${(c.metrics || []).length || 0}">${esc(c.difficulty)}</th>`).join('')).join('');
  const metricLabels = {main_metric:'Main','segment_f1@0.5':'SegF1@0.5','mean_best_tIoU':'tIoU','self_temporal_loc_f1@0.5':'SelfF1@0.5','self_temporal_loc_mean_tIoU':'tIoU','bbox_acc@50iou':'BBox'};
  const bottom = columns.map((col)=>(col.metrics || []).map((key)=>`<th>${esc(metricLabels[key] || key)}</th>`).join('')).join('');
  document.getElementById('metrics_matrix').innerHTML = `<table><thead><tr><th rowspan="3">模型</th>${top}</tr><tr>${mid}</tr><tr>${bottom}</tr></thead><tbody>${
    rows.map((row)=>`<tr><td>${esc(row.model)}</td>${columns.map((col)=>{ const cell = row.combos?.[col.combo_id] || {}; return (col.metrics || []).map((key)=>`<td>${fmtPct(cell[key])}</td>`).join(''); }).join('')}</tr>`).join('') || `<tr><td colspan="${1 + columns.reduce((acc,col)=>acc + ((col.metrics || []).length || 0),0)}" class="empty">暂无实验结果</td></tr>`
  }</tbody></table>`;
  const progressSummary = document.getElementById('metrics_progress_summary');
  if(progressSummary) {
    progressSummary.innerHTML = `<div class="summary-item"><strong>总体进度</strong>：${esc(progressData.overall_completed || 0)} / ${esc(progressData.overall_total || 0)} (${fmtPct(progressData.overall_ratio)})</div>`;
  }
  const progressHolder = document.getElementById('metrics_progress_matrix');
  if(progressHolder) {
    const progressScenes = progressData.scenes || [];
    const progressRows = progressData.rows || [];
    const head = progressScenes.map((scene)=>`<th>${esc(scene.scene_id)}</th>`).join('');
    const totalCols = 2 + progressScenes.length + 1;
    progressHolder.innerHTML = `<table><thead><tr><th>模型</th>${head}<th>汇总</th></tr></thead><tbody>${
      progressRows.map((row)=>`<tr><td>${esc(row.model)}</td>${progressScenes.map((scene)=>{ const cell = row.scenes?.[scene.scene_id] || {}; return `<td>${esc(cell.completed ?? 0)} / ${esc(cell.total ?? 0)} (${fmtPct(cell.ratio)})</td>`; }).join('')}<td>${esc(row.total_completed ?? 0)} / ${esc(row.total_samples ?? 0)} (${fmtPct(row.total_ratio)})</td></tr>`).join('') || `<tr><td colspan="${totalCols}" class="empty">暂无实验进度</td></tr>`
    }</tbody></table>`;
  }
  renderFooter();
}

function exportStage3MetricsCsv() {
  const latestOnly = document.getElementById('metrics_latest_only').checked ? '1' : '0';
  const byDifficulty = document.getElementById('metrics_by_difficulty').checked ? '1' : '0';
  const mq = globalQuery();
  window.location.href = `/api/stage3_metrics_matrix_csv?${{mq}}&latest_only=${{latestOnly}}&by_difficulty=${{byDifficulty}}`;
}
async function refreshAll() {
  await refreshState();
  if(state.currentPage === 'behavior_library') await loadBehaviorLibrary();
  if(state.currentPage === 'missions') { await loadMissionCatalog(); await loadMissionInstances(); }
  if(state.currentPage === 'review') await loadCandidates();
  if(state.currentPage === 'generate') await refreshState();
  if(state.currentPage === 'dataset') await loadManifestList();
  if(state.currentPage === 'experiments') { await loadManifestList(); await refreshJobs(); }
  if(state.currentPage === 'results') await loadReportList();
  if(state.currentPage === 'metrics') await loadMetricsMatrix();
  renderFooter();
}
setInterval(()=>{ if(state.currentPage === 'experiments') refreshJobs(); }, 4000);
['mission_mode','mission_generation_kind','mission_behavior_override','mission_count','mission_adaptive_params','mission_auto_set_rule','mission_auto_set_candidates','mission_allow_interleave_repeat','mission_max_total_elements','mission_auto_pick_mode','mission_auto_pick_count','mission_auto_pick_min_points','mission_auto_pick_diverse','mission_set_params_auto','mission_media_compress','mission_media_w','mission_media_h','mission_media_q','mission_video_prefer_web'].forEach((id)=>{
  const el = document.getElementById(id);
  if(!el) return;
  el.addEventListener('change', ()=>{
    if(id === 'mission_auto_pick_mode') syncAutoPickCountByMode();
    if(id === 'mission_behavior_override') state.missionBehaviorOverrideTouched = true;
    if(id === 'mission_set_params_auto') updateScopeParamDependencyStates('mission-scope');
    if(['mission_media_compress','mission_media_w','mission_media_h','mission_media_q','mission_video_prefer_web'].includes(id) && state.currentMissionInstance) {
      renderMissionDetail(state.currentMissionInstance, state.currentMissionHistory);
    }
    if(state.currentPage === 'missions') refreshMissionSelectionUi();
  });
  if(el.tagName === 'INPUT') el.addEventListener('input', ()=>{
    if(id === 'mission_behavior_override') state.missionBehaviorOverrideTouched = true;
    if(state.currentPage === 'missions' && id === 'mission_behavior_override') refreshMissionSelectionUi();
  });
});
initTheme();
loadCatalog().then(refreshAll);
</script>
</body>
</html>
"""
        return (
            template
            .replace("__DISPLAY_BEHAVIOR_LIBRARY__", "block" if active_page == "behavior_library" else "none")
            .replace("__NAV_HTML__", nav_html)
            .replace("__DISPLAY_MISSIONS__", "block" if active_page == "missions" else "none")
            .replace("__DISPLAY_REVIEW__", "block" if active_page == "review" else "none")
            .replace("__DISPLAY_GENERATE__", "block" if active_page == "generate" else "none")
            .replace("__DISPLAY_DATASET__", "block" if active_page == "dataset" else "none")
            .replace("__DISPLAY_EXPERIMENTS__", "block" if active_page == "experiments" else "none")
            .replace("__DISPLAY_RESULTS__", "block" if active_page == "results" else "none")
            .replace("__DISPLAY_METRICS__", "block" if active_page == "metrics" else "none")
            .replace("__ACTIVE_PAGE_JSON__", json.dumps(active_page))
            .replace("__DEFAULT_ENGINE_JSON__", json.dumps(default_engine))
            .replace("__DEFAULT_SCENE_JSON__", json.dumps(default_scene_id))
            .replace("__GLOBAL_SCENE__", json.dumps(GLOBAL_SCENE_ID))
            .replace("__GLOBAL_SCENE_LABEL__", json.dumps(GLOBAL_SCENE_LABEL))
        )

    @app.get("/behavior_library")
    @app.get("/missions")
    @app.get("/review")
    @app.get("/generate")
    @app.get("/dataset")
    @app.get("/experiments")
    @app.get("/results")
    @app.get("/metrics")
    def _page() -> Any:
        active = request.path.strip("/") or "missions"
        return _shell(active)

    @app.get("/artifact")
    def _artifact() -> Any:
        raw_path = str(request.args.get("path", "") or "").strip()
        return _artifact_response(raw_path)

    @app.get("/api/catalog")
    def _api_catalog() -> Any:
        return jsonify(catalog)

    @app.get("/api/task_pipeline_tasks")
    def _api_task_pipeline_tasks() -> Any:
        return jsonify({"tasks": list_task_pipeline_tasks(workspace_root=WORKSPACE_ROOT)})

    @app.get("/api/stage3_task_state")
    def _api_stage3_task_state() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        if _is_global_scene_id(scene_value):
            return jsonify(_global_stage3_payload(catalog=catalog, selected_engine=engine_value, task_name=task_name, fallback_config_path=config_path))
        cfg, layout, layouts = _load_layouts(engine_value, scene_value, task_name)
        cfg = _with_task_pipeline_cfg(cfg, task_name)
        candidates, _ = _discover_candidates(cfg, scene_id=str(request.args.get("scene_id", "") or default_scene_id), engine=str(request.args.get("engine", "") or default_engine))
        stage3_cfg = _stage3_cfg(cfg)
        stats = {
            "candidate_count": len(candidates),
            "approved_count": sum(1 for row in candidates if str(row.get("review_status", "")) == "approved"),
            "category_count": len({str(row.get("landmark_category", "") or "") for row in candidates if str(row.get("landmark_category", "") or "").strip()}),
            "avg_visible_count": (sum(int(row.get("visible_count", 0) or 0) for row in candidates) / float(len(candidates))) if candidates else 0.0,
            "single_landmark_count": sum(1 for row in candidates if str(row.get("mode", "single-landmark")) == "single-landmark"),
            "multi_landmark_count": sum(1 for row in candidates if str(row.get("mode", "single-landmark")) == "multi-landmark"),
            "mission_families": sorted({str(row.get("set_name", "") or "") for row in candidates if str(row.get("set_name", "") or "").strip()}),
        }
        latest_manifest = None
        for item in layouts:
            candidate = item["datasets_root"] / f"{str(request.args.get('scene_id', '') or default_scene_id)}.latest_manifest.json"
            if candidate.exists():
                latest_manifest = candidate
                break
        ui_defaults = {
            "auto_pick_single_count": 1,
            "auto_pick_multi_count": max(2, int(stage3_cfg.get("multi_landmark_max_secondary", 2) or 2) + 1),
            "auto_pick_min_points": int(stage3_cfg.get("auto_pick_min_points", 500) or 500),
            "mission_count": int(stage3_cfg.get("mission_count_default", 1) or 1),
            "auto_set_rule": str(stage3_cfg.get("auto_set_rule_default", "heuristic") or "heuristic"),
            "allow_interleave_repeat": bool(stage3_cfg.get("allow_interleave_repeat_default", False)),
            "max_total_elements": int(stage3_cfg.get("max_total_elements_default", 0) or 0),
        }
        return jsonify({"candidate_stats": stats, "latest_manifest_path": _path_for_json(latest_manifest) if latest_manifest and latest_manifest.exists() else None, "ui_defaults": ui_defaults, "task_name": task_name or None})

    @app.get("/api/stage3_candidates")
    def _api_stage3_candidates() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        if _is_global_scene_id(scene_value):
            payload = _global_stage3_payload(catalog=catalog, selected_engine=engine_value, task_name=task_name, fallback_config_path=config_path)
            rows = list(payload.get("global_candidates", []) or [])
            q = str(request.args.get("q", "") or "").strip().lower()
            status = str(request.args.get("status", "all") or "all")
            if status != "all":
                rows = [row for row in rows if str(row.get("review_status", "")) == status]
            if q:
                rows = [row for row in rows if q in str(row.get("traj_id", "")).lower() or q in str(row.get("landmark_category", "")).lower() or q in str(row.get("set_id", "")).lower() or q in str(row.get("set_name", "")).lower() or q in str(row.get("scene_id", "")).lower()]
            return jsonify({"rows": rows})
        cfg, _ = _load_scene_context(engine_value, scene_value)
        cfg = _with_task_pipeline_cfg(cfg, task_name)
        q = str(request.args.get("q", "") or "").strip().lower()
        status = str(request.args.get("status", "all") or "all")
        rows, _ = _discover_candidates(cfg, scene_id=scene_value, engine=engine_value)
        if status != "all":
            rows = [row for row in rows if str(row.get("review_status", "")) == status]
        if q:
            rows = [row for row in rows if q in str(row.get("traj_id", "")).lower() or q in str(row.get("landmark_category", "")).lower() or q in str(row.get("set_id", "")).lower() or q in str(row.get("set_name", "")).lower()]
        return jsonify({"rows": rows})

    @app.get("/api/stage3_candidate")
    def _api_stage3_candidate() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        traj_id = str(request.args.get("traj_id", "") or "").strip()
        if _is_global_scene_id(scene_value):
            payload = _global_stage3_payload(catalog=catalog, selected_engine=engine_value, task_name=request.args.get("task_name"), fallback_config_path=config_path)
            row = next((row for row in list(payload.get("global_candidates", []) or []) if row["traj_id"] == traj_id), None)
            if row is None:
                return jsonify({"error": "candidate_not_found"}), 404
            return jsonify({"candidate": row})
        cfg, _ = _load_scene_context(engine_value, scene_value)
        cfg = _with_task_pipeline_cfg(cfg, task_name)
        rows, _ = _discover_candidates(cfg, scene_id=scene_value, engine=engine_value)
        row = next((row for row in rows if row["traj_id"] == traj_id), None)
        if row is None:
            return jsonify({"error": "candidate_not_found"}), 404
        return jsonify({"candidate": row})

    @app.post("/api/stage3_review_candidate")
    def _api_stage3_review_candidate() -> Any:
        payload = request.json or {}
        engine_value = str(payload.get("engine", "") or default_engine)
        scene_value = str(payload.get("scene_id", "") or default_scene_id)
        if _is_global_scene_id(scene_value):
            return jsonify({"error": "global_scene_mode_is_read_only"}), 400
        status = str(payload.get("status", "pending") or "pending")
        traj_id = str(payload.get("traj_id", "") or "").strip()
        cfg, layout = _load_scene_context(engine_value, scene_value)
        review_index_path, review_log_path = _candidate_review_files(layout)
        review_index = _load_candidate_review_index(review_index_path)
        items = dict(review_index.get("items", {}) or {})
        items[traj_id] = {"traj_id": traj_id, "status": status, "updated_at": _iso_now()}
        review_index["items"] = items
        _write_json(review_index_path, review_index)
        _append_jsonl(review_log_path, {"event": "review_candidate", "traj_id": traj_id, "status": status, "scene_id": scene_value, "engine": engine_value, "updated_at": _iso_now()})
        return jsonify({"ok": True})

    @app.post("/api/stage3_generate_manifest")
    def _api_stage3_generate_manifest() -> Any:
        payload = request.json or {}
        engine_value = str(payload.get("engine", "") or default_engine)
        scene_value = str(payload.get("scene_id", "") or default_scene_id)
        if _is_global_scene_id(scene_value):
            return jsonify({"error": "global_scene_mode_is_read_only"}), 400
        cfg, _ = _load_scene_context(engine_value, scene_value)
        task_name = str(payload.get("task_name", "") or "").strip()
        if task_name:
            cfg = dict(cfg)
            cfg["task_pipeline"] = {"task_name": task_name, "root_dir": "task_pipeline_data"}
        out = generate_manifest(
            config=cfg,
            scene_id=scene_value,
            engine=engine_value,
            sample_count=max(1, int(payload.get("sample_count", 24) or 24)),
            seed=int(payload.get("seed", 7) or 7),
            forms=[str(x) for x in list(payload.get("forms", []))],
            approved_only=bool(payload.get("approved_only", True)),
            mode=str(payload.get("mode", "single-landmark") or "single-landmark"),
            include_temporal_localization=bool(payload.get("include_temporal_localization", False)),
        )
        return jsonify({"ok": True, "manifest_path": _path_for_json(out["manifest_path"]), "summary": out["manifest"]["summary"]})

    @app.get("/api/stage3_manifests")
    def _api_stage3_manifests() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        if _is_global_scene_id(scene_value):
            payload = _global_stage3_payload(catalog=catalog, selected_engine=engine_value, task_name=task_name, fallback_config_path=config_path)
            return jsonify(list(payload.get("global_manifests", []) or []))
        cfg, _, layouts = _load_layouts(engine_value, scene_value, task_name)
        return jsonify(_list_manifests_multi(layouts, scene_value))

    @app.get("/api/stage3_manifest")
    def _api_stage3_manifest() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        cfg, _ = _load_scene_context(engine_value, scene_value)
        path = _resolve_workspace_json_path(request.args.get("path", ""))
        payload = _read_json(path, default={})
        if not isinstance(payload, dict):
            return jsonify({"error": "manifest_not_found"}), 404
        samples = []
        experiment_defaults = (_stage3_cfg(cfg).get("experiment_defaults", {}) or {})
        provide_flight_description = bool(experiment_defaults.get("provide_flight_description", True))
        include_keyframes = bool(experiment_defaults.get("include_keyframes", False))
        for row in list(payload.get("samples", []) or []):
            if not isinstance(row, dict):
                continue
            sample = dict(row)
            if not str(sample.get("prompt_text", "") or "").strip():
                sample["prompt_text"] = _build_prompt(
                    sample,
                    provide_flight_description=provide_flight_description,
                    include_keyframes=include_keyframes,
                )
            sample["user_prompt"] = str(sample.get("user_prompt", "") or sample.get("prompt_text", "") or "")
            sample["system_prompt"] = str(sample.get("system_prompt", "") or _build_system_prompt(sample))
            samples.append(sample)
        return jsonify({"path": _path_for_json(path), "summary": payload.get("summary", _build_manifest_summary(samples)), "samples": samples})

    @app.post("/api/stage3_start_experiment")
    def _api_stage3_start_experiment() -> Any:
        payload = request.json or {}
        engine_value = str(payload.get("engine", "") or default_engine)
        scene_value = str(payload.get("scene_id", "") or default_scene_id)
        if _is_global_scene_id(scene_value):
            return jsonify({"error": "global_scene_mode_is_read_only"}), 400
        manifest_path = _resolve_workspace_json_path(payload.get("manifest_path", ""))
        if not manifest_path.exists():
            return jsonify({"error": "manifest_not_found"}), 404
        models = [str(x).strip() for x in list(payload.get("models", [])) if str(x).strip()]
        manifest_json_path = _path_for_json(manifest_path)
        manifest_name = manifest_path.name

        def _worker(job_id: str) -> None:
            try:
                job_manager.update(job_id, status="running")
                cfg, _ = _load_scene_context(engine_value, scene_value)
                job = job_manager.get(job_id) or {}
                job_payload = dict(job.get("payload", {}) or {})
                model_name = str(job_payload.get("model", "") or "").strip()
                if not model_name:
                    model_name = resolve_default_model(cfg, stage_name="stage3")
                if not model_name:
                    raise RuntimeError("no_model_selected")
                manifest = _read_json(manifest_path, default={})
                samples = list(manifest.get("samples", []) or [])
                limit = int(payload.get("limit", 0) or 0)
                total = min(limit, len(samples)) if limit > 0 else len(samples)
                job_manager.update_progress(job_id, {"completed": 0, "total": total, "sample_id": None})
                cancel_event = job_manager._jobs[job_id]["cancel_event"]
                if cancel_event.is_set():
                    raise CancelledExperimentError("experiment_cancelled")
                def _progress(p: dict[str, Any]) -> None:
                    job_manager.update_progress(
                        job_id,
                        {
                            "completed": int(p.get("completed", 0) or 0),
                            "total": total,
                            "sample_id": p.get("sample_id"),
                            "model": p.get("model"),
                            "form": p.get("form"),
                            "request_status": p.get("request_status"),
                            "parse_ok": p.get("parse_ok"),
                            "latency_ms": p.get("latency_ms"),
                        },
                    )
                out = run_experiment_once(
                    config=cfg,
                    scene_id=scene_value,
                    engine=engine_value,
                    manifest_path=manifest_path,
                    model=model_name,
                    limit=limit if limit > 0 else None,
                    api_overrides={
                        "upload_max_width": payload.get("upload_max_width"),
                        "upload_max_height": payload.get("upload_max_height"),
                        "upload_jpeg_quality": payload.get("upload_jpeg_quality"),
                        "concurrency": payload.get("concurrency"),
                        "rpm_limit": payload.get("rpm_limit"),
                        "tpm_limit": payload.get("tpm_limit"),
                        "provide_flight_description": payload.get("provide_flight_description"),
                        "include_keyframes": payload.get("include_keyframes"),
                    },
                    cancel_event=cancel_event,
                    progress_callback=_progress,
                )
                job_manager.add_run(job_id, {"model": model_name, "run_id": out["run_id"], "report_path": _path_for_json(out["report_path"]), "manifest_path": manifest_json_path, "manifest_name": manifest_name})
                job_manager.update(job_id, status="cancelled" if cancel_event.is_set() else "completed")
            except CancelledExperimentError:
                job_manager.update(job_id, status="cancelled")
            except Exception as exc:
                job_manager.update(job_id, status="error", error=str(exc))
        jobs = []
        for model_name in (models or [""]):
            job_payload = dict(payload)
            job_payload["manifest_path"] = manifest_json_path
            job_payload["manifest_name"] = manifest_name
            job_payload["model"] = model_name
            job_payload["models"] = [model_name] if model_name else []
            job_id = job_manager.create_job(job_payload)
            threading.Thread(target=_worker, args=(job_id,), daemon=True).start()
            job = job_manager.get(job_id)
            if job:
                jobs.append(job)
        return jsonify({"ok": True, "jobs": jobs, "job_ids": [job["job_id"] for job in jobs]})

    @app.get("/api/stage3_jobs")
    def _api_stage3_jobs() -> Any:
        return jsonify(job_manager.list())

    @app.post("/api/stage3_cancel_job")
    def _api_stage3_cancel_job() -> Any:
        payload = request.json or {}
        job_id = str(payload.get("job_id", "") or "").strip()
        ok = job_manager.cancel(job_id)
        if not ok:
            return jsonify({"error": "job_not_found"}), 404
        return jsonify({"ok": True})

    @app.get("/api/stage3_reports")
    def _api_stage3_reports() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        if _is_global_scene_id(scene_value):
            reports = _global_stage3_reports(catalog=catalog, selected_engine=engine_value, task_name=task_name, fallback_config_path=config_path)
            return jsonify(reports)
        cfg, _, layouts = _load_layouts(engine_value, scene_value, task_name)
        return jsonify(_list_reports_multi(layouts, scene_value))

    @app.get("/api/stage3_report")
    def _api_stage3_report() -> Any:
        path = _resolve_workspace_json_path(request.args.get("path", ""))
        payload = _read_json(path, default={})
        if not isinstance(payload, dict):
            return jsonify({"error": "report_not_found"}), 404
        rows = _load_report_rows(path)
        manifest_path = str(payload.get("manifest_path", "") or "")
        summary = dict(payload.get("summary", {}) or {})
        if not summary and rows:
            summary = _summarize_predictions(rows)
        return jsonify({
            "model": payload.get("model"),
            "manifest_path": manifest_path,
            "manifest_name": Path(manifest_path).name if manifest_path else "-",
            "summary": summary,
            "rows": _rows_for_display(rows),
            "requests_txt_path": _path_for_json(path.parent / "requests.txt") if (path.parent / "requests.txt").exists() else "",
            "responses_txt_path": _path_for_json(path.parent / "responses.txt") if (path.parent / "responses.txt").exists() else "",
        })

    @app.get("/api/stage3_metrics_matrix")
    def _api_stage3_metrics_matrix() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        latest_only = str(request.args.get("latest_only", "0") or "0").strip() not in {"0", "false", "False"}
        by_difficulty = str(request.args.get("by_difficulty", "0") or "0").strip() in {"1", "true", "True"}
        if _is_global_scene_id(scene_value):
            reports = _global_stage3_reports(catalog=catalog, selected_engine=engine_value, task_name=task_name, fallback_config_path=config_path)
            return jsonify(_build_metrics_matrix_from_reports(reports, scene_value, latest_only=latest_only, by_difficulty=by_difficulty))
        cfg, _, layouts = _load_layouts(engine_value, scene_value, task_name)
        reports = _list_reports_multi(layouts, scene_value)
        return jsonify(_build_metrics_matrix_from_reports(reports, scene_value, latest_only=latest_only, by_difficulty=by_difficulty))

    @app.get("/api/stage3_metrics_matrix_csv")
    def _api_stage3_metrics_matrix_csv() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        latest_only = str(request.args.get("latest_only", "0") or "0").strip() not in {"0", "false", "False"}
        by_difficulty = str(request.args.get("by_difficulty", "0") or "0").strip() in {"1", "true", "True"}
        if _is_global_scene_id(scene_value):
            reports = _global_stage3_reports(catalog=catalog, selected_engine=engine_value, task_name=task_name, fallback_config_path=config_path)
            matrix = _build_metrics_matrix_from_reports(reports, scene_value, latest_only=latest_only, by_difficulty=by_difficulty)
        else:
            cfg, _, layouts = _load_layouts(engine_value, scene_value, task_name)
            reports = _list_reports_multi(layouts, scene_value)
            matrix = _build_metrics_matrix_from_reports(reports, scene_value, latest_only=latest_only, by_difficulty=by_difficulty)
        fieldnames, rows = _stage3_metrics_matrix_csv_rows(matrix)
        filename = f"stage3_metrics_{scene_value or 'scene'}_{'latest' if latest_only else 'all'}_{'difficulty' if by_difficulty else 'summary'}.csv"
        return _csv_text(fieldnames, rows), 200, {
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": f'attachment; filename=\"{filename}\"',
        }

    @app.get("/api/stage3_experiment_progress_matrix")
    def _api_stage3_experiment_progress_matrix() -> Any:
        engine_value = str(request.args.get("engine", "") or default_engine)
        scene_value = str(request.args.get("scene_id", "") or default_scene_id)
        task_name = str(request.args.get("task_name", "") or "").strip()
        return jsonify(_build_stage3_experiment_progress_matrix(catalog=catalog, selected_engine=engine_value, selected_scene_id=scene_value, task_name=task_name, fallback_config_path=config_path))
