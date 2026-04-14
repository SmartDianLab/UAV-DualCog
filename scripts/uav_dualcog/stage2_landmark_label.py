#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import copy
import yaml
try:
    from flask import Flask, Response, jsonify, request, send_file
except Exception:
    Flask = None
    Response = None
    jsonify = None
    request = None
    send_file = None

from progress_utils import ProgressBar, StageLogger

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sim_bridge.factory import create_bridge
from pipeline_common import (
    build_unified_bridge_config,
    format_unified_startup_ports_message,
    prepare_airsim_runtime_unified,
    resolve_output_dir_name,
    resolve_scene_artifact_path,
    resolve_scene_root,
)
from api_common import (
    pick_first_text,
    resolve_default_model,
    resolve_model_api_endpoint,
)
from media_path_utils import resolve_existing_file_with_suffix_fallback


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("config yaml root must be a mapping")
    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@dataclass
class PcdData:
    fields: list[str]
    data: np.ndarray


@dataclass
class WorkerBinding:
    worker_id: int
    vehicle: str


def _parse_pcd_header(path: Path) -> tuple[list[str], int]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    fields: list[str] = []
    data_line = -1
    for idx, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if u.startswith("FIELDS "):
            fields = s.split()[1:]
        elif u.startswith("DATA "):
            if "ASCII" not in u:
                raise ValueError(f"Only ASCII PCD supported: {path}")
            data_line = idx + 1
            break
    if data_line < 0 or not fields:
        raise ValueError(f"Invalid PCD header: {path}")
    return fields, data_line


def _load_pcd_ascii(path: Path) -> PcdData:
    fields, data_line = _parse_pcd_header(path)
    field_count = len(fields)

    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for idx, line in enumerate(file):
            if idx < data_line:
                continue
            parts = line.strip().split()
            if not parts:
                continue
            for token in parts:
                try:
                    values.append(float(token))
                except Exception:
                    continue

    if not values:
        return PcdData(fields=fields, data=np.empty((0, field_count), dtype=np.float64))

    usable = (len(values) // field_count) * field_count
    if usable <= 0:
        return PcdData(fields=fields, data=np.empty((0, field_count), dtype=np.float64))

    arr = np.asarray(values[:usable], dtype=np.float64).reshape(-1, field_count)
    return PcdData(fields=fields, data=arr)


def _build_class_name_map(config: dict[str, Any]) -> dict[int, str]:
    stage2_cfg = config.get("stage2", {}) or {}
    mapping_raw = stage2_cfg.get("class_id_to_name", {}) or {}
    mapping: dict[int, str] = {}
    if isinstance(mapping_raw, dict):
        for key, value in mapping_raw.items():
            try:
                mapping[int(key)] = str(value)
            except Exception:
                continue
    return mapping


def parse_bindings(config: dict[str, Any], worker_count: int) -> list[WorkerBinding]:
    parallel = config.get("parallel", {}) or {}
    raw_bindings = parallel.get("bindings", []) or []
    bindings: list[WorkerBinding] = []
    if raw_bindings:
        for item in raw_bindings:
            worker_id = int(item.get("worker_id"))
            vehicle = str(item.get("vehicle"))
            bindings.append(WorkerBinding(worker_id=worker_id, vehicle=vehicle))
    else:
        for worker_id in range(worker_count):
            bindings.append(WorkerBinding(worker_id=worker_id, vehicle=f"drone_{worker_id + 1}"))

    if len(bindings) < worker_count:
        next_id = len(bindings)
        for worker_id in range(next_id, worker_count):
            bindings.append(WorkerBinding(worker_id=worker_id, vehicle=f"drone_{worker_id + 1}"))

    return sorted(bindings, key=lambda item: item.worker_id)


def normalize_airsim_vehicle_name(name: str, fallback_index: int) -> str:
    candidate = str(name).strip()
    if not candidate:
        candidate = f"drone_{fallback_index + 1}"
    candidate_low = candidate.lower()
    if candidate_low.startswith("drone"):
        suffix = candidate_low.replace("drone", "", 1).replace("_", "").replace("-", "")
        if suffix.isdigit():
            return f"drone_{int(suffix)}"
    return candidate


def _resolve_output_dir_name(config: dict[str, Any], key: str, default: str) -> str:
    return resolve_output_dir_name(config, key=key, default=default)


def _resolve_scene_root(config: dict[str, Any], scene_id: str) -> Path:
    task_cfg = config.get("task", {}) or {}
    engine_name = str(task_cfg.get("engine", "airsim") or "airsim").strip().lower()
    return resolve_scene_root(
        config,
        scene_id=scene_id,
        engine=engine_name,
        workspace_root=WORKSPACE_ROOT,
    )


def _build_bridge_config(
    config: dict[str, Any],
    vehicle_name: str,
    image_size_override: int | None = None,
    fov_override: float | None = None,
) -> dict[str, Any]:
    task_cfg = config.get("task", {}) or {}
    engine = str(task_cfg.get("engine", "airsim")).lower()
    engine_cfg = (config.get("engine_params", {}) or {}).get(engine, {}) or {}
    camera_cfg = config.get("camera", {}) or {}
    image_width = int(image_size_override) if image_size_override is not None else int(camera_cfg.get("width", 3840))
    image_height = int(image_size_override) if image_size_override is not None else int(camera_cfg.get("height", 2160))
    fov = float(fov_override) if fov_override is not None else float(camera_cfg.get("fov", 90.0))
    bridge_cfg: dict[str, Any] = {
        "sim_ip": str(engine_cfg.get("sim_ip", "127.0.0.1")),
        "sim_port": int(engine_cfg.get("sim_port", 41471)),
        "connect_on_init": bool(engine_cfg.get("connect_on_init", True)),
        "launch_sim": bool(engine_cfg.get("launch_sim", False)),
        "vehicle_name": str(engine_cfg.get("vehicle_name", vehicle_name)),
        "camera_name": str(engine_cfg.get("camera_name", "front_0")),
        "connect_timeout_sec": float(engine_cfg.get("connect_timeout_sec", 30)),
        "capture_retries": int(engine_cfg.get("capture_retries", 2)),
        "tick_after_set_pose": bool(engine_cfg.get("tick_after_set_pose", True)),
        "strict_vehicle_name": bool(engine_cfg.get("strict_vehicle_name", False)),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "fov": float(fov),
    }
    for key in [
        "camera_id",
        "attach_sensors",
        "headless",
        "auto_select_port_on_conflict",
        "launch_ready_timeout_sec",
        "launch_ready_check_interval_sec",
        "connect_retry_interval_sec",
        "launch_extra_args",
        "vehicle_names",
    ]:
        if key in engine_cfg:
            bridge_cfg[key] = engine_cfg[key]
    return bridge_cfg


def _build_view_yaws_deg(view_count: int) -> list[float]:
    count = max(1, int(view_count))
    if count == 4:
        return [45.0, -45.0, 135.0, -135.0]
    angles = np.linspace(0.0, 360.0, num=count, endpoint=False)
    return [float(v) for v in angles]


def _build_view_specs(
    view_yaws_deg: list[float],
    add_birdseye_view: bool,
    yaw_offset_deg: float = 0.0,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for yaw_deg in view_yaws_deg:
        specs.append({"mode": "orbit", "yaw_deg": float(yaw_deg + yaw_offset_deg)})
    if add_birdseye_view:
        specs.append({"mode": "topdown", "yaw_deg": 0.0})
    return specs


def _normalize_deg(deg: float) -> float:
    out = float(deg)
    while out <= -180.0:
        out += 360.0
    while out > 180.0:
        out -= 360.0
    return out


def _yaw_token(yaw_deg: float) -> str:
    yaw_i = int(round(_normalize_deg(yaw_deg)))
    sign = "p" if yaw_i >= 0 else "n"
    return f"{sign}{abs(yaw_i):03d}"


def _effective_square_fov_deg(fov_deg: float, source_width: int, source_height: int) -> float:
    w = max(1, int(source_width))
    h = max(1, int(source_height))
    if w == h:
        return float(fov_deg)
    crop_w = min(w, h)
    ratio = float(crop_w) / float(w)
    half = math.radians(float(fov_deg)) * 0.5
    return float(math.degrees(2.0 * math.atan(math.tan(half) * ratio)))


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return image
    h, w = int(image.shape[0]), int(image.shape[1])
    if h <= 0 or w <= 0 or h == w:
        return image
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]


def _build_candidate_side_view_specs(bbox_yaw_deg: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = float(bbox_yaw_deg)
    cardinal = [
        {"mode": "orbit", "offset_deg": 0.0, "label": "front"},
        {"mode": "orbit", "offset_deg": 180.0, "label": "back"},
        {"mode": "orbit", "offset_deg": 90.0, "label": "left"},
        {"mode": "orbit", "offset_deg": -90.0, "label": "right"},
    ]
    diagonal = [
        {"mode": "orbit", "offset_deg": 45.0, "label": "front_left"},
        {"mode": "orbit", "offset_deg": -45.0, "label": "front_right"},
        {"mode": "orbit", "offset_deg": 135.0, "label": "back_left"},
        {"mode": "orbit", "offset_deg": -135.0, "label": "back_right"},
    ]
    for item in cardinal + diagonal:
        item["yaw_deg"] = _normalize_deg(base + float(item["offset_deg"]))
    return cardinal, diagonal


def _build_side_view_pitch_offsets_deg(stage2_cfg: dict[str, Any]) -> list[float]:
    raw = stage2_cfg.get("collect_side_view_pitch_offsets_deg", [0.0, 8.0, 16.0, 24.0])
    values: list[float] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            try:
                values.append(max(0.0, float(item)))
            except Exception:
                continue
    elif raw is not None:
        try:
            values.append(max(0.0, float(raw)))
        except Exception:
            pass
    if not values:
        values = [0.0, 8.0, 16.0, 24.0]
    values = sorted(set(float(v) for v in values))
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


VIEW_DIRECTION_RING = [
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
]
VIEW_DIRECTION_SET = set(VIEW_DIRECTION_RING)


def _normalize_view_direction(direction: str | None) -> str | None:
    if direction is None:
        return None
    value = str(direction or "").strip().lower()
    return value if value in VIEW_DIRECTION_SET else None


def _assign_view_directions(
    base_direction: str,
    selected_view_index: int,
    view_count: int,
) -> list[str]:
    if view_count <= 0:
        return []
    base_idx = VIEW_DIRECTION_RING.index(base_direction)
    # 4 views -> step 2 (auto-fill the remaining 3 directions), 8 views -> step 1
    step = max(1, int(round(8.0 / float(view_count))))
    out: list[str] = []
    for i in range(view_count):
        rel = (i - selected_view_index) % view_count
        out.append(VIEW_DIRECTION_RING[(base_idx + rel * step) % 8])
    return out


def _assign_view_directions_by_yaw(
    base_direction: str,
    selected_view_index: int,
    views: list[Any],
) -> list[str] | None:
    view_count = len(views)
    if view_count <= 0 or selected_view_index < 0 or selected_view_index >= view_count:
        return None

    yaws: list[float] = []
    for view in views:
        if not isinstance(view, dict):
            return None
        yaw_raw = view.get("yaw_deg", None)
        if yaw_raw is None:
            return None
        try:
            yaws.append(float(yaw_raw))
        except Exception:
            return None

    yaw_selected = yaws[selected_view_index]
    base_idx = VIEW_DIRECTION_RING.index(base_direction)
    out: list[str] = []
    for yaw in yaws:
        delta = _normalize_deg(float(yaw) - float(yaw_selected))
        # Yaw is positive counter-clockwise in ENU; the direction ring increases clockwise, so negate to align.
        step_45 = int(round((-delta) / 45.0))
        out.append(VIEW_DIRECTION_RING[(base_idx + step_45) % 8])
    return out


def _resolve_image_path(path_str: str) -> Path | None:
    raw = str(path_str or "").strip()
    if not raw:
        return None
    resolved = resolve_existing_file_with_suffix_fallback(
        raw,
        base_dirs=[Path.cwd(), WORKSPACE_ROOT],
    )
    if resolved is not None:
        return resolved

    # Common relative path: rgb_views/<instance_id>/view_xx.png; actual root is scene_data/<scene>/landmarks_raw/
    scene_data_roots = [Path.cwd() / "scene_data", WORKSPACE_ROOT / "scene_data"]
    for scene_data_root in scene_data_roots:
        if not scene_data_root.exists():
            continue
        try:
            for landmarks_raw_root in scene_data_root.glob("*/landmarks_raw"):
                resolved = resolve_existing_file_with_suffix_fallback(raw, base_dirs=[landmarks_raw_root])
                if resolved is not None:
                    return resolved
        except Exception:
            continue
    return None


def _pick_vlm_views(item: dict[str, Any], max_views: int) -> list[dict[str, Any]]:
    max_views = max(1, int(max_views))
    views = [v for v in list(item.get("rgb_views", []) or []) if isinstance(v, dict)]
    if not views:
        return []

    def _dir(view: dict[str, Any]) -> str | None:
        return _normalize_view_direction(view.get("view_direction", view.get("label", None)))

    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    query_view = next((v for v in views if bool(v.get("is_query_view", False))), None)
    if isinstance(query_view, dict):
        selected.append(query_view)
        used_ids.add(id(query_view))

    # Priority order: front/back/left/right > front-left/front-right/back-left/back-right > others
    primary = ["front", "back", "left", "right"]
    secondary = ["front_left", "front_right", "back_left", "back_right"]
    for target in primary:
        match = next((v for v in views if id(v) not in used_ids and _dir(v) == target), None)
        if isinstance(match, dict):
            selected.append(match)
            used_ids.add(id(match))
        if len(selected) >= max_views:
            return selected[:max_views]

    for target in secondary:
        match = next((v for v in views if id(v) not in used_ids and _dir(v) == target), None)
        if isinstance(match, dict):
            selected.append(match)
            used_ids.add(id(match))
        if len(selected) >= max_views:
            return selected[:max_views]

    for target in VIEW_DIRECTION_RING:
        match = next((v for v in views if id(v) not in used_ids and _dir(v) == target), None)
        if isinstance(match, dict):
            selected.append(match)
            used_ids.add(id(match))
        if len(selected) >= max_views:
            return selected[:max_views]

    ordered_rest = sorted(
        [v for v in views if id(v) not in used_ids],
        key=lambda it: int(it.get("view_id", 10**9)) if isinstance(it, dict) else 10**9,
    )
    for view in ordered_rest:
        selected.append(view)
        if len(selected) >= max_views:
            break
    return selected[:max_views]


def _extract_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
                continue
            if isinstance(part, dict):
                text_val = part.get("text", None)
                if isinstance(text_val, str) and text_val.strip():
                    chunks.append(text_val)
                    continue
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                chunks.append(text_attr)
        return "\n".join(chunks).strip()
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(content)


def _parse_vlm_json_text(raw: str) -> dict[str, Any]:
    raw = str(raw or "").strip()
    if not raw:
        raise RuntimeError("empty_vlm_response")

    def _extract_balanced_json_snippets(text: str, opener: str, closer: str) -> list[str]:
        snippets: list[str] = []
        n = len(text)
        for start in range(n):
            if text[start] != opener:
                continue
            depth = 0
            in_str = False
            escape = False
            for end in range(start, n):
                ch = text[end]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue

                if ch == '"':
                    in_str = True
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : end + 1].strip()
                        if candidate:
                            snippets.append(candidate)
                        break
        return snippets

    candidates: list[str] = []
    m_json_block = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if m_json_block:
        candidates.append(m_json_block.group(1).strip())

    m_any_block = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if m_any_block:
        candidates.append(m_any_block.group(1).strip())

    candidates.append(raw)

    for snippet in _extract_balanced_json_snippets(raw, "{", "}"):
        candidates.append(snippet)
    for snippet in _extract_balanced_json_snippets(raw, "[", "]"):
        candidates.append(snippet)

    m_obj = re.search(r"\{[\s\S]*\}", raw)
    if m_obj:
        candidates.append(m_obj.group(0).strip())

    m_arr = re.search(r"\[[\s\S]*\]", raw)
    if m_arr:
        candidates.append(m_arr.group(0).strip())

    tried: set[str] = set()
    for text in candidates:
        if not text or text in tried:
            continue
        tried.add(text)
        try:
            parsed = json.loads(text)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]

    repair_text = raw
    repair_text = re.sub(r"```json", "", repair_text, flags=re.IGNORECASE)
    repair_text = repair_text.replace("```", "").strip()
    repair_text = repair_text.replace('\\"', '"').replace("\\'", "'")

    def _pick_str(key: str) -> str | None:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*"([^\"]*)"',
            rf"'{re.escape(key)}'\s*:\s*'([^']*)'",
            rf'"{re.escape(key)}"\s*:\s*\'([^\']*)\'',
            rf"'{re.escape(key)}'\s*:\s*\"([^\"]*)\"",
            rf"{re.escape(key)}\s*:\s*\"([^\"]*)\"",
            rf"{re.escape(key)}\s*:\s*'([^']*)'",
        ]
        for ptn in patterns:
            m = re.search(ptn, repair_text)
            if m:
                return str(m.group(1)).strip()
        return None

    def _pick_num(key: str) -> float | None:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*([-+]?\d+(?:\.\d+)?)',
            rf"'{re.escape(key)}'\s*:\s*([-+]?\d+(?:\.\d+)?)",
            rf"{re.escape(key)}\s*:\s*([-+]?\d+(?:\.\d+)?)",
        ]
        for ptn in patterns:
            m = re.search(ptn, repair_text)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    continue
        return None

    category = _pick_str("category") or _pick_str("landmark_type")
    subcategory = _pick_str("subcategory") or _pick_str("landmark_name")
    description = _pick_str("description") or _pick_str("landmark_description")
    confidence = _pick_num("confidence")

    if category or subcategory or description or confidence is not None:
        return {
            "category": category or "other",
            "subcategory": subcategory or "unknown",
            "description": description or "",
            "confidence": float(confidence if confidence is not None else 0.6),
        }

    snippet = raw[:500].replace("\n", "\\n")
    raise RuntimeError(f"vlm_response_not_json: {snippet}")


def _validate_vlm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    category = str(payload.get("category", payload.get("landmark_type", "")) or "").strip()
    subcategory = str(payload.get("subcategory", payload.get("landmark_name", "")) or "").strip()
    description = str(payload.get("description", payload.get("landmark_description", "")) or "").strip()
    confidence_raw = payload.get("confidence", None)
    if not category or not subcategory or not description:
        raise RuntimeError("vlm_missing_required_fields")
    try:
        confidence = float(confidence_raw)
    except Exception:
        raise RuntimeError("vlm_invalid_confidence")
    confidence = max(0.0, min(1.0, confidence))
    return {
        "category": category,
        "subcategory": subcategory,
        "description": description,
        "confidence": confidence,
    }


class _AutoLabelRateLimiter:
    def __init__(self, rpm_limit: int, tpm_limit: int) -> None:
        self.rpm_limit = max(0, int(rpm_limit))
        self.tpm_limit = max(0, int(tpm_limit))
        self._lock = threading.Lock()
        self._request_times: deque[float] = deque()
        self._token_events: deque[dict[str, float]] = deque()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_events and float(self._token_events[0].get("ts", 0.0)) <= cutoff:
            self._token_events.popleft()

    def _token_sum_locked(self) -> float:
        return float(sum(max(0.0, float(ev.get("tokens", 0.0))) for ev in self._token_events))

    def _token_wait_locked(self, now: float, request_tokens: float) -> float:
        if self.tpm_limit <= 0:
            return 0.0
        used = self._token_sum_locked()
        if used + request_tokens <= float(self.tpm_limit):
            return 0.0
        remain = used
        for ev in self._token_events:
            remain -= max(0.0, float(ev.get("tokens", 0.0)))
            if remain + request_tokens <= float(self.tpm_limit):
                ts = float(ev.get("ts", now))
                return max(0.0, 60.0 - (now - ts))
        first_ts = float(self._token_events[0].get("ts", now)) if self._token_events else now
        return max(0.0, 60.0 - (now - first_ts))

    def _rpm_wait_locked(self, now: float) -> float:
        if self.rpm_limit <= 0:
            return 0.0
        if len(self._request_times) < self.rpm_limit:
            return 0.0
        return max(0.0, 60.0 - (now - self._request_times[0]))

    def acquire(self, estimated_tokens: int) -> dict[str, float]:
        req_tokens = max(1.0, float(estimated_tokens))
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune_locked(now)
                wait_rpm = self._rpm_wait_locked(now)
                wait_tpm = self._token_wait_locked(now, req_tokens)
                wait_s = max(wait_rpm, wait_tpm)
                if wait_s <= 1e-3:
                    event = {"ts": now, "tokens": req_tokens}
                    self._request_times.append(now)
                    self._token_events.append(event)
                    return event
            time.sleep(min(max(wait_s, 0.02), 5.0))

    def finalize(self, reservation: dict[str, float], actual_total_tokens: int | None) -> None:
        if actual_total_tokens is None:
            return
        actual = max(1.0, float(actual_total_tokens))
        with self._lock:
            reservation["tokens"] = actual


_AUTO_LABEL_LIMITER_LOCK = threading.Lock()
_AUTO_LABEL_LIMITER: _AutoLabelRateLimiter | None = None
_AUTO_LABEL_LIMITER_SIGNATURE: tuple[int, int] | None = None


def _safe_int_cfg(stage2_cfg: dict[str, Any], default: int, *keys: str) -> int:
    for key in keys:
        if key not in stage2_cfg:
            continue
        try:
            return int(float(stage2_cfg.get(key)))
        except Exception:
            continue
    return int(default)


def _safe_float_cfg(stage2_cfg: dict[str, Any], default: float, *keys: str) -> float:
    for key in keys:
        if key not in stage2_cfg:
            continue
        try:
            return float(stage2_cfg.get(key))
        except Exception:
            continue
    return float(default)


def _get_auto_label_rate_limiter(stage2_cfg: dict[str, Any]) -> _AutoLabelRateLimiter:
    per_api_rpm = _safe_int_cfg(
        stage2_cfg,
        1000,
        "auto_label_api_rpm_per_api",
        "auto_label_rpm_per_api",
        "auto_label_rpm",
    )
    per_api_tpm = _safe_int_cfg(
        stage2_cfg,
        80000,
        "auto_label_api_tpm_per_api",
        "auto_label_tpm_per_api",
        "auto_label_tpm",
    )
    aggregate_count = _safe_int_cfg(
        stage2_cfg,
        1,
        "auto_label_api_aggregate_count",
        "auto_label_aggregate_count",
    )
    reserve_ratio = _safe_float_cfg(stage2_cfg, 0.10, "auto_label_token_reserve_ratio")
    reserve_ratio = min(max(reserve_ratio, 0.0), 0.5)
    eff_rpm = max(1, int(max(1, per_api_rpm) * max(1, aggregate_count)))
    eff_tpm = max(1, int(max(1, per_api_tpm) * max(1, aggregate_count) * (1.0 - reserve_ratio)))

    signature = (eff_rpm, eff_tpm)
    global _AUTO_LABEL_LIMITER, _AUTO_LABEL_LIMITER_SIGNATURE
    with _AUTO_LABEL_LIMITER_LOCK:
        if _AUTO_LABEL_LIMITER is None or _AUTO_LABEL_LIMITER_SIGNATURE != signature:
            _AUTO_LABEL_LIMITER = _AutoLabelRateLimiter(rpm_limit=eff_rpm, tpm_limit=eff_tpm)
            _AUTO_LABEL_LIMITER_SIGNATURE = signature
    return _AUTO_LABEL_LIMITER


def _estimate_vlm_request_tokens(
    class_name: str,
    image_paths: list[Path],
    view_infos: list[dict[str, Any]],
    stage2_cfg: dict[str, Any],
    max_tokens: int,
) -> int:
    info_chars = len(class_name)
    for info in view_infos:
        info_chars += len(str(info.get("view_direction", "") or ""))
        info_chars += len(str(info.get("label", "") or ""))
        info_chars += 8
    prompt_token_est = max(32, int(info_chars / 3.2) + 420)
    image_token_est = _safe_int_cfg(stage2_cfg, 1200, "auto_label_image_token_estimate")
    total = prompt_token_est + len(image_paths) * max(1, image_token_est) + max(1, int(max_tokens))
    return max(1, int(total))


def _prepare_auto_label_upload_image(
    image_path: Path,
    stage2_cfg: dict[str, Any],
) -> tuple[bytes, str]:
    resize_enabled = str(stage2_cfg.get("auto_label_upload_resize_enabled", True)).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    # Upload compression for red-box images, default 480P (configurable)
    max_width = max(1, _safe_int_cfg(stage2_cfg, 640, "auto_label_bbox_upload_max_width", "auto_label_upload_max_width"))
    max_height = max(1, _safe_int_cfg(stage2_cfg, 480, "auto_label_bbox_upload_max_height", "auto_label_upload_max_height"))
    jpeg_quality = int(np.clip(_safe_int_cfg(stage2_cfg, 85, "auto_label_upload_jpeg_quality"), 50, 100))

    try:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError("cv2_read_failed")
        h, w = image_bgr.shape[:2]
        if h <= 0 or w <= 0:
            raise RuntimeError("invalid_image_shape")

        out = image_bgr
        # New: decide whether to draw the red box and direction label
        add_bbox = str(stage2_cfg.get("auto_label_image_bbox", True)).strip().lower() not in {"0", "false", "no", "off"}
        debug_save_bbox_img = str(stage2_cfg.get("auto_label_debug_save_bbox_img", False)).strip().lower() in {"1", "true", "yes", "on"}
        debug_bbox_dir = _resolve_auto_label_debug_bbox_dir(stage2_cfg, image_path=image_path)
        if add_bbox and stage2_cfg.get("_auto_label_bbox_info") is not None:
            bbox_info = stage2_cfg["_auto_label_bbox_info"]
            out = draw_bbox_and_direction(out, bbox_info)
            # Debug: save boxed image
            if debug_save_bbox_img and debug_bbox_dir:
                import os
                os.makedirs(debug_bbox_dir, exist_ok=True)
                # Filename includes original name and direction
                fname = os.path.basename(str(image_path))
                direction = bbox_info.get("direction", "")
                debug_path = os.path.join(debug_bbox_dir, f"debug_{direction}_{fname}")
                cv2.imwrite(debug_path, out)

        if resize_enabled:
            scale = min(float(max_width) / float(w), float(max_height) / float(h), 1.0)
            if scale < 0.999:
                new_w = max(1, int(round(float(w) * scale)))
                new_h = max(1, int(round(float(h) * scale)))
                out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError("cv2_encode_failed")
        return bytes(enc.tobytes()), "image/jpeg"
    except Exception:
        raw = image_path.read_bytes()
        ext = image_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            return raw, "image/jpeg"
        if ext == ".webp":
            return raw, "image/webp"
        return raw, "image/png"


def _resolve_auto_label_debug_bbox_dir(stage2_cfg: dict[str, Any], image_path: Path | None = None) -> str | None:
    review_root_value = stage2_cfg.get("_review_root")
    scene_root_value = stage2_cfg.get("_scene_root")
    scene_id = str(stage2_cfg.get("_scene_id", "") or "").strip()

    debug_root: Path | None = None
    if review_root_value:
        debug_root = Path(str(review_root_value)) / "auto_label_debug"
    else:
        raw_value = str(stage2_cfg.get("auto_label_debug_bbox_dir", "") or "").strip()
        if raw_value:
            if scene_id:
                raw_value = raw_value.replace("${scene_id}", scene_id)
            debug_root = Path(raw_value)
            if not debug_root.is_absolute():
                if scene_root_value:
                    debug_root = Path(str(scene_root_value)) / debug_root
                else:
                    debug_root = Path.cwd() / debug_root
    if debug_root is None:
        return None
    if image_path is not None:
        parent_name = image_path.parent.name.strip()
        if parent_name:
            debug_root = debug_root / parent_name
    return str(debug_root)

# New: helper to draw red box and direction label on images
def draw_bbox_and_direction(image: np.ndarray, bbox_info: dict) -> np.ndarray:
    img = copy.deepcopy(image)
    # bbox_info: {"bbox": [x0, y0, x1, y1], "direction": "front"}
    bbox = bbox_info.get("bbox")
    direction = bbox_info.get("direction", "")
    img_h, img_w = img.shape[:2]
    short_side = max(1, min(int(img_h), int(img_w)))
    scale = max(1.0, float(short_side) / 480.0)
    box_thickness = max(4, int(round(4.0 * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(1.0, 0.95 * scale)
    text_thickness = max(3, int(round(2.5 * scale)))
    margin = max(10, int(round(12.0 * scale)))
    text_pad_x = max(10, int(round(12.0 * scale)))
    text_pad_y = max(8, int(round(10.0 * scale)))
    # 1) Red box
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = map(int, bbox)
        x0 = int(np.clip(x0, 0, max(0, img_w - 1)))
        y0 = int(np.clip(y0, 0, max(0, img_h - 1)))
        x1 = int(np.clip(x1, 0, max(0, img_w - 1)))
        y1 = int(np.clip(y1, 0, max(0, img_h - 1)))
        cv2.rectangle(img, (x0, y0), (x1, y1), (40, 40, 255), box_thickness, cv2.LINE_AA)
    # 2) Direction label at top-left (configurable)
    show_face_label = bbox_info.get("show_face_label")
    if show_face_label is None:
        show_face_label = True
    if show_face_label:
        face_label = bbox_info.get("face_label")
        if not face_label:
            # Default format
            face_label = f"Visible Side: {direction} (Landmark-centric)"
        face_label = " ".join(str(face_label).split())
        max_label_width = max(120, int(round(float(img_w) * 0.5)))
        max_text_width = max(40, max_label_width - text_pad_x * 2)
        fit_font_scale = float(font_scale)
        min_font_scale = max(0.45, float(font_scale) * 0.45)
        text_size, baseline = cv2.getTextSize(face_label, font, fit_font_scale, text_thickness)
        while text_size[0] > max_text_width and fit_font_scale > min_font_scale:
            fit_font_scale = max(min_font_scale, fit_font_scale * 0.92)
            text_size, baseline = cv2.getTextSize(face_label, font, fit_font_scale, text_thickness)
        bg_x0 = margin
        bg_y0 = margin
        bg_x1 = min(img_w - 1, bg_x0 + min(max_label_width, text_size[0] + text_pad_x * 2))
        bg_y1 = min(img_h - 1, bg_y0 + text_size[1] + baseline + text_pad_y * 2)
        overlay = img.copy()
        cv2.rectangle(overlay, (bg_x0, bg_y0), (bg_x1, bg_y1), (20, 20, 180), -1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.74, img, 0.26, 0.0)
        cv2.rectangle(
            img,
            (bg_x0, bg_y0),
            (bg_x1, bg_y1),
            (40, 40, 255),
            max(2, text_thickness - 1),
            cv2.LINE_AA,
        )
        tx = bg_x0 + text_pad_x
        ty = bg_y0 + text_pad_y + text_size[1]
        cv2.putText(img, face_label, (tx, ty), font, fit_font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return img


def _call_vlm_with_class_and_views(
    class_name: str,
    image_paths: list[Path],
    stage2_cfg: dict[str, Any],
    view_infos: list[dict[str, Any]] | None = None,
    item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not available")

    def _pick_cfg(*keys: str) -> str:
        for key in keys:
            value = stage2_cfg.get(key, None)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _pick_bool(default: bool, *keys: str) -> bool:
        for key in keys:
            if key not in stage2_cfg:
                continue
            value = stage2_cfg.get(key)
            if isinstance(value, bool):
                return value
            text = str(value or "").strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    full_config = stage2_cfg.get("_full_config", {})
    if not isinstance(full_config, dict):
        full_config = {}

    explicit_api_key = _pick_cfg("api_key")
    explicit_api_base = _pick_cfg("api_base", "base_url")
    explicit_api_source = _pick_cfg("api_source")

    model = _pick_cfg("model")
    if not model:
        model = resolve_default_model(full_config, stage_name="stage2")
    response_prefix = _pick_cfg("prefix")
    if not model:
        raise RuntimeError("missing model for auto label")

    endpoint = resolve_model_api_endpoint(
        config=full_config,
        model=model,
        stage_name="stage2",
        stage_cfg=stage2_cfg,
        explicit_source=explicit_api_source,
        explicit_api_base=explicit_api_base,
        explicit_api_key=explicit_api_key,
    )
    api_key = str(endpoint.get("api_key", "") or "").strip()
    api_base = pick_first_text(endpoint.get("api_base"), "https://api.siliconflow.cn/v1")
    if not api_key:
        if api_base.startswith("http://localhost") or api_base.startswith("http://127.0.0.1"):
            api_key = "EMPTY"
        else:
            raise RuntimeError("missing api_key/model for auto label")

    request_model = str(endpoint.get("request_model", model) or model)
    force_disable_thinking = False
    rewrite_from = _pick_cfg("model_rewrite_from")
    rewrite_to = _pick_cfg("model_rewrite_to")
    if rewrite_from and rewrite_to and model == rewrite_from:
        request_model = rewrite_to
    if _pick_bool(False, "force_disable_thinking", "disable_thinking"):
        force_disable_thinking = True

    client = OpenAI(api_key=api_key, base_url=api_base)
    image_blocks: list[dict[str, Any]] = []
    # New: set bbox and direction info for each image
    for idx, p in enumerate(image_paths):
        bbox_info = None
        if view_infos and idx < len(view_infos):
            info = view_infos[idx]
            # Expect bbox (xyxy) and direction fields in view_infos
            bbox = info.get("bbox")
            direction = info.get("view_direction", "")
            if bbox:
                bbox_info = {"bbox": bbox, "direction": direction}
        # Temporarily write into stage2_cfg for image processing
        if bbox_info:
            stage2_cfg["_auto_label_bbox_info"] = bbox_info
        else:
            stage2_cfg.pop("_auto_label_bbox_info", None)
        image_bytes, mime_type = _prepare_auto_label_upload_image(p, stage2_cfg)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
    # Clear temporary fields
    stage2_cfg.pop("_auto_label_bbox_info", None)

    view_infos = list(view_infos or [])
    view_lines: list[str] = []
    for idx, info in enumerate(view_infos):
        direction = str(info.get("view_direction", "") or "unknown")
        is_main = bool(info.get("is_query_view", False))
        label = str(info.get("label", "") or "")
        view_lines.append(
            f"- image_{idx+1}: direction={direction}, is_main_view={str(is_main).lower()}, label={label or '-'}"
        )
    if not view_lines:
        view_lines = ["- image_i: direction=unknown, is_main_view=false"]
    metadata_text = "\n".join(view_lines)

    # 2.4.5 Prompt Template
    # system prompt
    system_prompt = (
        "You are an aerial landmark recognition expert. Only use image evidence. "
    )
            # "Do not rely on external world knowledge guesses."
    # user prompt
    user_prompt = (
        f"- class_name: {class_name or '(empty)'} (user-filled weak hint from Step 2, optional)\n"
        f"- images: up to 4 views (front/back/left/right preferred, some directions may be missing)\n"
        "- Each uploaded image has a red bounding box marking the landmark, and the side label (e.g., 'Visible Side: Front (Landmark-centric)') at the top-left corner shows the visible side of the landmark in the object-centric (landmark-centric) frame.\n"
        "category candidates (must choose from list):\n"
        "[building, vehicle, public_facility, urban_landscape,transport_infrastructure, industrial_infrastructure, vegetation, other]\n\n"
        "subcategory requirements:\n"
            "- flexible generic subtype, not a proper noun or brand name\n"
            "- must stay category-consistent\n"
            "- examples below are illustrative only; use other common descriptive terms or phrases if they better match the landmark\n"
            "  • building: low-rise building, mid-rise building, high-rise building, warehouse, pagoda, chapel, factory shed, rural farmhouse, glass skyscraper, brick schoolhouse\n"
            "  • vehicle: sedan car, delivery van, city bus, cargo truck, motorcycle, construction excavator\n"
            "  • public_facility: street lamp post, bus shelter, public bench, trash bin, fire hydrant, antenna\n"
            "  • urban_landscape: sculpture, plaza, fountain, city square, urban garden, landscape installation, signboard, billboard, advertising board, wayfinding sign, landmark signage\n"
            "  • transport_infrastructure: arch bridge, overpass, railway track, tunnel entrance, roundabout, pedestrian crosswalk, subway station entrance, traffic island, railway platform\n"
            "  • industrial_infrastructure: shipping container stack, oil storage tank, grain silo, crane tower, pipeline\n"
            "  • vegetation: deciduous tree, coniferous tree, palm tree, shrub cluster, hedge row, grassy lawn\n"
            "  • other: temporary tent, rubble pile, playground slide, inflatable archway, construction barrier\n\n"
        "description constraints (`description`):\n"
        "- one noun phrase, <= 20 words\n"
        "- must include: subcategory, color, shape/texture\n"
        "- may include surrounding relation, visible text/pattern cues to distinguish the landmark\n"
        "- positive examples:\n"
        "  - dark red middle-rise building with white neon light featuring the word \"HOTEL\" on the top\n"
        "  - gray pagoda-like tower with layered roof edges beside roadside trees\n"
        "  - white arch bridge with curved span above a narrow river channel\n"
        "  - dark stone obelisk with sharp top in open paved square\n\n"
        "Output JSON (no extra explanation text):\n"
        "{\n"
        "  \"category\": \"building|vehicle|public_facility|urban_landscape|transport_infrastructure|industrial_infrastructure|vegetation|other\",\n"
        "  \"subcategory\": \"...\",\n"
        "  \"description\": \"...\",\n"
        "  \"confidence\": 0.0\n"
        "}"
    )
    # Data prep: write instance_id/class_id/class_name/center_3d/bbox_3d into stage2_cfg temporary fields
    if 'instance_id' in stage2_cfg:
        pass
    else:
        stage2_cfg['instance_id'] = class_name
    if item is not None:
        if 'class_id' not in stage2_cfg and 'class_id' in item:
            stage2_cfg['class_id'] = item['class_id']
        if 'center_3d' not in stage2_cfg and 'center_3d' in item:
            stage2_cfg['center_3d'] = item['center_3d']
        if 'bbox_3d' not in stage2_cfg and 'bbox_3d' in item:
            stage2_cfg['bbox_3d'] = item['bbox_3d']
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Backward-compatible API call structure
    text_block = {"type": "text", "text": user_prompt}

    def _extract_text_and_meta(resp_obj: Any) -> tuple[str, str, int | None]:
        if not resp_obj or not getattr(resp_obj, "choices", None):
            return "", "choices=0", None
        choice0 = resp_obj.choices[0]
        message = getattr(choice0, "message", None)
        candidates: list[Any] = []
        if message is not None:
            candidates.append(getattr(message, "content", None))
            candidates.append(getattr(message, "refusal", None))
            candidates.append(getattr(message, "reasoning_content", None))
        candidates.append(getattr(choice0, "text", None))

        text = ""
        for candidate in candidates:
            parsed = _extract_message_text(candidate).strip()
            if parsed:
                text = parsed
                break

        finish_reason = str(getattr(choice0, "finish_reason", "") or "")
        usage = getattr(resp_obj, "usage", None)
        usage_text = ""
        total_tokens_int: int | None = None
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            usage_text = f"prompt={prompt_tokens},completion={completion_tokens},total={total_tokens}"
            try:
                if total_tokens is not None:
                    total_tokens_int = int(total_tokens)
            except Exception:
                total_tokens_int = None
        meta = f"finish_reason={finish_reason or 'unknown'}"
        if usage_text:
            meta = f"{meta}; usage[{usage_text}]"
        return text, meta, total_tokens_int

    base_messages = [
        {
            "role": "system",
            "content": "You are an aerial landmark recognition expert. Return exactly one JSON object only. "
            "Do not return empty text, Markdown code blocks, or any extra explanations. /no_think",
        },
        {"role": "user", "content": [text_block, *image_blocks]},
    ]

    last_error = ""
    last_meta = ""
    retry_interval_s = float(stage2_cfg.get("auto_label_retry_interval_s", 1.0) or 1.0)
    use_prefix = bool(response_prefix)
    attempt = 0
    while True:
        attempt += 1
        messages = list(base_messages)
        if attempt >= 2:
            messages = messages + [
                {
                    "role": "user",
                    "content": "Your previous answer was empty or invalid. Return one valid JSON object now with keys: "
                    "category, subcategory, description, confidence.",
                }
            ]
        raw_text = ""
        meta = ""
        try:
            extra_body: dict[str, Any] = {}
            if force_disable_thinking:
                extra_body["enable_thinking"] = False
            if use_prefix and response_prefix:
                extra_body["prefix"] = response_prefix
            max_tokens = int(stage2_cfg.get("auto_label_max_tokens", 500) or 500)
            limiter = _get_auto_label_rate_limiter(stage2_cfg)
            estimated_tokens = _estimate_vlm_request_tokens(
                class_name=class_name,
                image_paths=image_paths,
                view_infos=view_infos,
                stage2_cfg=stage2_cfg,
                max_tokens=max_tokens,
            )
            reservation = limiter.acquire(estimated_tokens=estimated_tokens)
            request_kwargs: dict[str, Any] = {
                "model": request_model,
                "messages": messages,
                "temperature": float(stage2_cfg.get("auto_label_temperature", 0.2) or 0.2) if attempt == 1 else 0.0,
                "max_tokens": max_tokens,
                "timeout": float(stage2_cfg.get("auto_label_timeout_s", 30) or 30),
            }
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(
                **request_kwargs,
            )
            raw_text, meta, usage_total_tokens = _extract_text_and_meta(resp)
            limiter.finalize(reservation=reservation, actual_total_tokens=usage_total_tokens)
            last_meta = f"attempt={attempt}; {meta}"
            parsed = _parse_vlm_json_text(raw_text)
            return _validate_vlm_payload(parsed)
        except Exception as exc:
            last_error = str(exc)
            if use_prefix and "Prefix is not supported for this model" in last_error:
                use_prefix = False
                continue
            if "vlm_response_not_json" in last_error and raw_text:
                print(
                    "[stage2.auto_label][ERROR] raw_response_begin "
                    f"attempt={attempt} class_name={class_name or '-'}"
                )
                print(raw_text)
                print("[stage2.auto_label][ERROR] raw_response_end")
            print(
                "[stage2.auto_label][WARN] retry "
                f"attempt={attempt} class_name={class_name or '-'} error={last_error}"
            )
            if retry_interval_s > 0:
                time.sleep(retry_interval_s)


def _build_auto_label_fields(item: dict[str, Any], scope: str, stage2_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    stage2_cfg = stage2_cfg or {}
    class_name = str(item.get("class_name", "") or "").strip()
    class_id = item.get("class_id", None)
    instance_id = str(item.get("instance_id", "") or "")
    label_name = class_name if class_name else (f"class_{class_id}" if class_id is not None else "unknown")

    max_views = int(stage2_cfg.get("auto_label_max_views", 4) or 4)
    picked_views = _pick_vlm_views(item=item, max_views=max_views)
    image_paths: list[Path] = []
    view_infos: list[dict[str, Any]] = []
    view_directions: list[str] = []
    for view in picked_views:
        path_obj = _resolve_image_path(str(view.get("path", "") or ""))
        if path_obj is not None:
            image_paths.append(path_obj)
            # Try to read bbox (xyxy)
            bbox = None
            if "bbox_2d_xyxy" in view and view.get("bbox_2d_valid", False):
                bbox = view["bbox_2d_xyxy"]
            view_infos.append(
                {
                    "view_direction": _normalize_view_direction(view.get("view_direction", view.get("label", None))) or "unknown",
                    "is_query_view": bool(view.get("is_query_view", False)),
                    "label": str(view.get("label", "") or ""),
                    "bbox": bbox,
                }
            )
        direction = _normalize_view_direction(view.get("view_direction", view.get("label", None)))
        if direction:
            view_directions.append(direction)

    if not image_paths:
        raise RuntimeError(
            f"auto_label_no_resolved_images: scope={scope} instance_id={instance_id} class_id={class_id}"
        )

    vlm_result = _call_vlm_with_class_and_views(
        class_name=class_name,
        image_paths=image_paths,
        stage2_cfg=stage2_cfg,
        view_infos=view_infos,
        item=item,
    )

    label_name = str(vlm_result.get("subcategory", label_name) or label_name)
    confidence = float(vlm_result.get("confidence", 0.1) or 0.1)
    confidence = max(0.0, min(1.0, confidence))
    landmark_type = str(vlm_result.get("category", "other") or "other")
    landmark_desc = str(vlm_result.get("description", "") or "").strip()
    reason = f"auto_label_{scope}:vlm:{landmark_type}"
    comment = landmark_desc if landmark_desc else f"generated_by_vlm_{scope}"

    landmark_type = str(landmark_type or "").strip()
    landmark_desc = str(landmark_desc or "").strip()

    conf_threshold = float(stage2_cfg.get("auto_label_conf_threshold", 0.6) or 0.6)
    if confidence >= conf_threshold:
        annotation_status = "labeled"
        landmark_category = landmark_type
        landmark_subcategory = label_name
        landmark_description = landmark_desc
        landmark_decision = "auto_approved"
        landmark_note = f"auto approved by {scope}"
    elif confidence > 0.0:
        annotation_status = "pending_review"
        landmark_category = None
        landmark_subcategory = None
        landmark_description = None
        landmark_decision = None
        landmark_note = None
    else:
        annotation_status = "failed"

    views_text = ",".join(view_directions) if view_directions else None

    print(
        "[stage2.auto_label][INFO] "
        f"scope={scope} instance_id={instance_id or '-'} class_id={class_id} class_name={class_name or '-'} "
        f"category={landmark_type} subcategory={label_name} confidence={float(confidence):.3f} "
        f"status={annotation_status} reason={reason} views={views_text or '-'} "
        f"desc={landmark_desc[:120]}"
    )

    return {
        "auto_label_category": landmark_type,
        "auto_label_subcategory": label_name,
        "auto_label_description": landmark_desc,
        "auto_label_name": label_name,
        "auto_label_confidence": float(confidence),
        "auto_label_landmark_type": landmark_type,
        "auto_label_landmark_description": landmark_desc,
        "auto_label_reason": reason,
        "auto_label_views": views_text,
        "auto_label_comment": comment,
        "landmark_category": landmark_category,
        "landmark_subcategory": landmark_subcategory,
        "landmark_description": landmark_description,
        "landmark_decision": landmark_decision,
        "landmark_note": landmark_note,
        "annotation_status": annotation_status,
    }


def _build_clear_auto_label_fields() -> dict[str, Any]:
    return {
        "auto_label_category": None,
        "auto_label_subcategory": None,
        "auto_label_description": None,
        "auto_label_name": None,
        "auto_label_confidence": None,
        "auto_label_landmark_type": None,
        "auto_label_landmark_description": None,
        "auto_label_reason": None,
        "auto_label_views": None,
        "auto_label_comment": None,
        "landmark_category": None,
        "landmark_subcategory": None,
        "landmark_description": None,
        "landmark_decision": None,
        "landmark_note": None,
        "annotation_status": "unlabeled",
    }


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _has_auto_label_payload(item: dict[str, Any]) -> bool:
    keys = [
        "auto_label_category",
        "auto_label_subcategory",
        "auto_label_description",
        "auto_label_name",
        "auto_label_landmark_type",
        "auto_label_landmark_description",
        "auto_label_reason",
        "auto_label_views",
        "auto_label_comment",
    ]
    for key in keys:
        if str(item.get(key, "") or "").strip():
            return True
    return item.get("auto_label_confidence", None) is not None


def _has_final_label_payload(item: dict[str, Any]) -> bool:
    keys = ["landmark_category", "landmark_subcategory", "landmark_description", "landmark_decision"]
    for key in keys:
        if str(item.get(key, "") or "").strip():
            return True
    return False


def _normalize_annotation_payload(item: dict[str, Any], stage2_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    stage2_cfg = stage2_cfg or {}
    out = dict(item)

    if not str(out.get("auto_label_subcategory", "") or "").strip():
        out["auto_label_subcategory"] = str(out.get("auto_label_name", "") or "").strip() or None
    if not str(out.get("auto_label_category", "") or "").strip():
        out["auto_label_category"] = str(out.get("auto_label_landmark_type", "") or "").strip() or None
    if not str(out.get("auto_label_description", "") or "").strip():
        out["auto_label_description"] = str(out.get("auto_label_landmark_description", "") or "").strip() or None
    if not str(out.get("auto_label_name", "") or "").strip():
        out["auto_label_name"] = str(out.get("auto_label_subcategory", "") or "").strip() or None
    if not str(out.get("auto_label_landmark_type", "") or "").strip():
        out["auto_label_landmark_type"] = str(out.get("auto_label_category", "") or "").strip() or None
    if not str(out.get("auto_label_landmark_description", "") or "").strip():
        out["auto_label_landmark_description"] = str(out.get("auto_label_description", "") or "").strip() or None

    if not str(out.get("landmark_category", "") or "").strip():
        out["landmark_category"] = str(out.get("auto_label_category", "") or "").strip() or None
    if not str(out.get("landmark_subcategory", "") or "").strip():
        out["landmark_subcategory"] = str(out.get("auto_label_subcategory", "") or out.get("auto_label_name", "") or "").strip() or None
    if not str(out.get("landmark_description", "") or "").strip():
        out["landmark_description"] = str(out.get("auto_label_description", "") or "").strip() or None

    auto_conf = _parse_optional_float(out.get("auto_label_confidence", None))
    if auto_conf is not None:
        auto_conf = max(0.0, min(1.0, float(auto_conf)))
        out["auto_label_confidence"] = auto_conf

    conf_threshold = float(stage2_cfg.get("auto_label_conf_threshold", 0.6) or 0.6)
    has_auto_payload = _has_auto_label_payload(out)
    has_final_payload = _has_final_label_payload(out)

    landmark_decision = str(out.get("landmark_decision", "") or "").strip().lower()

    if landmark_decision == "manual":
        annotation_status = "labeled"
    elif has_final_payload:
        annotation_status = "labeled"
    elif has_auto_payload:
        if auto_conf is None:
            annotation_status = "pending_review"
        elif auto_conf >= conf_threshold:
            annotation_status = "labeled"
        elif auto_conf > 0.0:
            annotation_status = "pending_review"
        else:
            annotation_status = "failed"
    else:
        annotation_status = "unlabeled"

    out["annotation_status"] = annotation_status
    return out


def _select_view_specs_for_instance(
    target_points_xyz: np.ndarray,
    context_points_xyz: np.ndarray,
    context_instance_ids: np.ndarray,
    target_instance_id: int,
    bbox_yaw_deg: float,
    pose_params: dict[str, float] | None,
    camera_fov_deg: float,
    image_size: int,
    side_view_count: int,
    side_view_min_keep: int,
    add_birdseye_view: bool,
    min_visible_ratio: float,
    min_visible_points: int,
    occlusion_neighbor_px: int,
    occlusion_depth_margin_m: float,
    occlusion_search_radius_scale: float,
    occlusion_search_radius_min_m: float,
    side_view_pitch_offsets_deg: list[float] | None = None,
) -> list[dict[str, Any]] | None:
    wanted = max(1, min(8, int(side_view_count)))
    min_keep = max(1, min(wanted, int(side_view_min_keep)))
    cardinals, diagonals = _build_candidate_side_view_specs(bbox_yaw_deg=bbox_yaw_deg)
    pitch_offsets = [max(0.0, float(v)) for v in (side_view_pitch_offsets_deg or [0.0])]
    if not pitch_offsets:
        pitch_offsets = [0.0]
    if 0.0 not in pitch_offsets:
        pitch_offsets.insert(0, 0.0)
    target_center = np.mean(target_points_xyz, axis=0).astype(np.float32)
    cam_distance = _camera_distance_for_target(target_points_xyz=target_points_xyz, pose_params=pose_params)
    search_radius = max(float(occlusion_search_radius_min_m), float(cam_distance) * float(occlusion_search_radius_scale))
    neighbor_bboxes = _build_neighbor_yaw_obbs_from_context(
        context_points_xyz=context_points_xyz,
        context_instance_ids=context_instance_ids,
        target_instance_id=target_instance_id,
        target_center=target_center,
        search_radius_xy=search_radius,
    )

    def _is_good(spec: dict[str, Any]) -> bool:
        _, eye, forward, right, cam_up = _camera_pose_for_yaw(
            target_points_xyz=target_points_xyz,
            yaw_deg=float(spec["yaw_deg"]),
            extra_pitch_deg=float(spec.get("pitch_offset_deg", 0.0) or 0.0),
            pose_params=pose_params,
        )
        if _is_view_occluded_by_neighbor_bboxes(
            eye=eye,
            target_center=target_center,
            neighbor_bboxes=neighbor_bboxes,
            depth_margin_m=occlusion_depth_margin_m,
        ):
            return False
        visible, total, ratio = _compute_target_visible_stats(
            points_xyz=target_points_xyz,
            context_points_xyz=context_points_xyz,
            context_instance_ids=context_instance_ids,
            target_instance_id=target_instance_id,
            eye=eye,
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=image_size,
            height=image_size,
            fov_deg=camera_fov_deg,
            occlusion_neighbor_px=occlusion_neighbor_px,
            occlusion_depth_margin_m=occlusion_depth_margin_m,
        )
        return visible >= int(min_visible_points) and ratio >= float(min_visible_ratio)

    def _pick_best_variant(spec: dict[str, Any]) -> dict[str, Any] | None:
        for pitch_offset_deg in pitch_offsets:
            candidate = dict(spec)
            candidate["pitch_offset_deg"] = float(pitch_offset_deg)
            candidate["view_direction"] = str(spec.get("label", "") or "")
            if _is_good(candidate):
                return candidate
        return None

    selected: list[dict[str, Any]] = []
    for spec in cardinals:
        picked = _pick_best_variant(spec)
        if picked is not None:
            selected.append(picked)
    if len(selected) < wanted:
        for spec in diagonals:
            picked = _pick_best_variant(spec)
            if picked is not None:
                selected.append(picked)
            if len(selected) >= wanted:
                break
    if len(selected) < min_keep:
        return None

    out: list[dict[str, Any]] = []
    for idx, spec in enumerate(selected[:wanted]):
        out.append(
            {
                "mode": "orbit",
                "view_id": int(idx),
                "yaw_deg": float(spec["yaw_deg"]),
                "label": str(spec["label"]),
                "view_direction": str(spec.get("view_direction", spec.get("label", "")) or ""),
                "pitch_offset_deg": float(spec.get("pitch_offset_deg", 0.0) or 0.0),
            }
        )


    if add_birdseye_view:
        out.append(
            {
                "mode": "topdown",
                "view_id": int(len(out)),
                "yaw_deg": float(_normalize_deg(bbox_yaw_deg)),
                "label": "topdown",
                "view_direction": "topdown",
                "pitch_offset_deg": 0.0,
            }
        )
    return out


def _ray_hit_yaw_obb(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box_center: np.ndarray,
    box_half_size: np.ndarray,
    cos_yaw: float,
    sin_yaw: float,
    t_max: float,
) -> float | None:
    rel_o = ray_origin - box_center
    ox = float(cos_yaw * rel_o[0] + sin_yaw * rel_o[1])
    oy = float(-sin_yaw * rel_o[0] + cos_yaw * rel_o[1])
    oz = float(rel_o[2])
    dx = float(cos_yaw * ray_dir[0] + sin_yaw * ray_dir[1])
    dy = float(-sin_yaw * ray_dir[0] + cos_yaw * ray_dir[1])
    dz = float(ray_dir[2])

    local_origin = np.array([ox, oy, oz], dtype=np.float32)
    local_dir = np.array([dx, dy, dz], dtype=np.float32)
    box_min = -box_half_size
    box_max = box_half_size

    t_near = 0.0
    t_far = float(t_max)
    eps = 1e-6
    for axis in range(3):
        ro = float(local_origin[axis])
        rd = float(local_dir[axis])
        bmin = float(box_min[axis])
        bmax = float(box_max[axis])
        if abs(rd) < eps:
            if ro < bmin or ro > bmax:
                return None
            continue
        t1 = (bmin - ro) / rd
        t2 = (bmax - ro) / rd
        t_axis_near = min(t1, t2)
        t_axis_far = max(t1, t2)
        t_near = max(t_near, t_axis_near)
        t_far = min(t_far, t_axis_far)
        if t_far < t_near:
            return None
    if t_far < 0.0:
        return None
    return float(max(0.0, t_near))


def _build_neighbor_yaw_obbs_from_context(
    context_points_xyz: np.ndarray,
    context_instance_ids: np.ndarray,
    target_instance_id: int,
    target_center: np.ndarray,
    search_radius_xy: float,
) -> list[tuple[int, np.ndarray, np.ndarray, float, float]]:
    if context_points_xyz.shape[0] == 0:
        return []
    dx = context_points_xyz[:, 0] - float(target_center[0])
    dy = context_points_xyz[:, 1] - float(target_center[1])
    near_mask = (dx * dx + dy * dy) <= float(search_radius_xy * search_radius_xy)
    if not np.any(near_mask):
        return []
    pts = context_points_xyz[near_mask]
    ids = context_instance_ids[near_mask]
    if pts.shape[0] == 0:
        return []

    neighbors: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
    unique_ids = np.unique(ids)
    for inst_id_raw in unique_ids.tolist():
        inst_id = int(inst_id_raw)
        if inst_id == int(target_instance_id):
            continue
        inst_mask = ids == inst_id
        if int(np.count_nonzero(inst_mask)) < 8:
            continue
        inst_pts = pts[inst_mask]

        xy = inst_pts[:, :2].astype(np.float32)
        xy_center = np.mean(xy, axis=0, keepdims=True)
        xy_zero = xy - xy_center
        cov = (xy_zero.T @ xy_zero) / float(max(1, xy_zero.shape[0]))
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        major = eig_vecs[:, int(np.argmax(eig_vals))]
        yaw = float(math.atan2(float(major[1]), float(major[0])))
        cos_y = float(math.cos(yaw))
        sin_y = float(math.sin(yaw))

        rel = inst_pts - np.mean(inst_pts, axis=0, keepdims=True)
        lx = cos_y * rel[:, 0] + sin_y * rel[:, 1]
        ly = -sin_y * rel[:, 0] + cos_y * rel[:, 1]
        lz = rel[:, 2]
        local = np.stack([lx, ly, lz], axis=1)
        local_min = np.min(local, axis=0)
        local_max = np.max(local, axis=0)
        local_center = 0.5 * (local_min + local_max)
        half_size = 0.5 * (local_max - local_min)
        half_size = np.maximum(half_size, np.array([0.2, 0.2, 0.2], dtype=np.float32)).astype(np.float32)

        mean_world = np.mean(inst_pts, axis=0).astype(np.float32)
        center_offset_world = np.array(
            [
                cos_y * float(local_center[0]) - sin_y * float(local_center[1]),
                sin_y * float(local_center[0]) + cos_y * float(local_center[1]),
                float(local_center[2]),
            ],
            dtype=np.float32,
        )
        obb_center = mean_world + center_offset_world
        neighbors.append((inst_id, obb_center, half_size, cos_y, sin_y))
    return neighbors


def _is_view_occluded_by_neighbor_bboxes(
    eye: np.ndarray,
    target_center: np.ndarray,
    neighbor_bboxes: list[tuple[int, np.ndarray, np.ndarray, float, float]],
    depth_margin_m: float,
) -> bool:
    if not neighbor_bboxes:
        return False
    ray_vec = target_center - eye
    t_target = float(np.linalg.norm(ray_vec))
    if t_target < 1e-6:
        return False
    ray_dir = ray_vec / t_target
    margin = float(max(0.0, depth_margin_m))
    for _, obb_center, obb_half_size, cos_y, sin_y in neighbor_bboxes:
        t_hit = _ray_hit_yaw_obb(
            ray_origin=eye,
            ray_dir=ray_dir,
            box_center=obb_center,
            box_half_size=obb_half_size,
            cos_yaw=cos_y,
            sin_yaw=sin_y,
            t_max=t_target,
        )
        if t_hit is not None and t_hit < (t_target - margin):
            return True
    return False


def _camera_distance_for_target(target_points_xyz: np.ndarray, pose_params: dict[str, float] | None = None) -> float:
    params = pose_params or {}
    pmin = np.min(target_points_xyz, axis=0)
    pmax = np.max(target_points_xyz, axis=0)
    size = pmax - pmin
    target_diag_xy = float(np.linalg.norm(size[:2]))
    target_diag_3d = float(np.linalg.norm(size))
    target_h = float(max(1e-3, size[2]))
    distance_scale = float(params.get("distance_scale", 1.55))
    distance_min_m = float(params.get("distance_min_m", 9.0))
    camera_fov_deg = float(params.get("camera_fov_deg", 90.0) or 90.0)
    half_fov = math.radians(float(np.clip(camera_fov_deg, 10.0, 170.0))) * 0.5
    tan_half = max(1e-3, math.tan(half_fov))
    sin_half = max(1e-3, math.sin(half_fov))

    half_diag_xy = 0.5 * target_diag_xy
    half_h = 0.5 * target_h
    bbox_radius = 0.5 * max(target_diag_3d, 1.0)
    fit_xy = half_diag_xy / tan_half
    fit_z = half_h / tan_half
    fit_sphere = bbox_radius / sin_half
    fit_distance = max(1.0, fit_xy, fit_z, fit_sphere)
    return float(max(distance_min_m, fit_distance * distance_scale))


def _build_camera_pose_params(stage2_cfg: dict[str, Any]) -> dict[str, float]:
    return {
        "distance_scale": float(stage2_cfg.get("collect_camera_distance_scale", 1.55) or 1.55),
        "distance_min_m": float(stage2_cfg.get("collect_camera_distance_min_m", 9.0) or 9.0),
        "distance_max_m": 0.0,
        "height_base_m": float(stage2_cfg.get("collect_camera_height_base_m", 3.0) or 3.0),
        "height_scale": float(stage2_cfg.get("collect_camera_height_scale", 0.10) or 0.10),
        "height_obj_scale": float(stage2_cfg.get("collect_camera_height_obj_scale", 0.35) or 0.35),
        "height_max_m": float(stage2_cfg.get("collect_camera_height_max_m", 18.0) or 18.0),
        "look_at_z_ratio": float(stage2_cfg.get("collect_camera_look_at_z_ratio", 0.55) or 0.55),
    }


def _camera_pose_for_yaw(
    target_points_xyz: np.ndarray,
    yaw_deg: float,
    extra_pitch_deg: float = 0.0,
    pose_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = pose_params or {}
    center = np.mean(target_points_xyz, axis=0).astype(np.float32)
    pmin = np.min(target_points_xyz, axis=0)
    pmax = np.max(target_points_xyz, axis=0)
    size = pmax - pmin
    target_diag_xy = float(np.linalg.norm(size[:2]))
    target_h = float(max(1e-3, size[2]))

    height_base_m = float(params.get("height_base_m", 3.0))
    height_scale = float(params.get("height_scale", 0.10))
    height_obj_scale = float(params.get("height_obj_scale", 0.35))
    height_max_m = float(params.get("height_max_m", 18.0))
    look_at_z_ratio = float(params.get("look_at_z_ratio", 0.55))

    yaw = math.radians(float(yaw_deg))
    cam_distance = _camera_distance_for_target(target_points_xyz=target_points_xyz, pose_params=params)
    cam_height = min(height_max_m, max(1.8, height_base_m + target_h * height_obj_scale + cam_distance * height_scale))
    forward_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    eye = center - forward_xy * cam_distance + np.array([0.0, 0.0, cam_height], dtype=np.float32)

    look_target = center.copy()
    look_target[2] = float(pmin[2] + target_h * np.clip(look_at_z_ratio, 0.1, 0.95))

    if abs(float(extra_pitch_deg)) > 1e-3:
        rel = eye - look_target
        orbit_radius = float(np.linalg.norm(rel) + 1e-6)
        base_horiz = float(np.linalg.norm(rel[:2]))
        base_elev_deg = float(math.degrees(math.atan2(float(rel[2]), max(1e-6, base_horiz))))
        desired_elev_deg = float(np.clip(base_elev_deg + float(extra_pitch_deg), 5.0, 80.0))
        desired_elev = math.radians(desired_elev_deg)
        horiz = orbit_radius * math.cos(desired_elev)
        vert = orbit_radius * math.sin(desired_elev)
        eye = look_target - forward_xy * horiz + np.array([0.0, 0.0, vert], dtype=np.float32)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward = look_target - eye
    forward /= float(np.linalg.norm(forward) + 1e-6)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right /= right_norm
    cam_up = np.cross(right, forward)
    cam_up /= float(np.linalg.norm(cam_up) + 1e-6)
    return center, eye, forward, right, cam_up


def _camera_pose_topdown(
    target_points_xyz: np.ndarray,
    yaw_deg: float = 0.0,
    pose_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Topdown camera pose with consistent object orientation alignment.
    
    Design principle: 
    - Camera is positioned directly above target looking straight down
    - The camera's coordinate system is aligned such that:
      * forward points downward (-Z in world)
      * cam_up (V-axis in image) points toward the object's heading (yaw_deg)
      * right (U-axis in image) rotates accordingly
    
    This ensures that when yaw_deg=0, the object's front direction points upward in the image,
    providing visual consistency with orbit views captured at the same yaw.
    
    Transformation:
    - World coordinates: X(East), Y(North), Z(Up) in ENU
    - Object heading: yaw_deg measured from East (X-axis) counterclockwise
    - Image coords: U(right), V(up), where V aligns with object's forward direction
    """
    center = np.mean(target_points_xyz, axis=0).astype(np.float32)
    cam_distance = _camera_distance_for_target(target_points_xyz=target_points_xyz, pose_params=pose_params)
    eye = center + np.array([0.0, 0.0, cam_distance], dtype=np.float32)
    look_target = center.copy()
    forward = look_target - eye
    forward /= float(np.linalg.norm(forward) + 1e-6)  # forward = [0, 0, -1]

    # cam_up should point in the direction of object's forward (heading)
    # yaw_deg=0 -> forward direction = [1, 0, 0] (East)
    # yaw_deg=90 -> forward direction = [0, 1, 0] (North)
    yaw = math.radians(float(yaw_deg))
    cam_up = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
    cam_up /= float(np.linalg.norm(cam_up) + 1e-6)
    
    # Compute right = forward × cam_up (left-handed: forward to up)
    # Since forward = [0,0,-1], right = [sin(yaw), -cos(yaw), 0]
    right = np.cross(forward, cam_up)
    right /= float(np.linalg.norm(right) + 1e-6)
    
    return center, eye, forward, right, cam_up


def _project_camera_uv(
    points_xyz: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rel = points_xyz - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1.0
    if not np.any(valid):
        empty_ids = np.empty((0,), dtype=np.int64) if ids is not None else None
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), empty_ids
    u = (x_cam[valid] / z_cam[valid]).astype(np.float32)
    v = (y_cam[valid] / z_cam[valid]).astype(np.float32)
    out_ids = ids[valid] if ids is not None else None
    return u, v, out_ids


def _adaptive_uv_to_pixels(
    u_ctx: np.ndarray,
    v_ctx: np.ndarray,
    u_tar: np.ndarray,
    v_tar: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if u_tar.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )
    if u_ctx.size >= 32:
        u_min, u_max = np.quantile(u_ctx, [0.05, 0.95]).astype(np.float32)
        v_min, v_max = np.quantile(v_ctx, [0.05, 0.95]).astype(np.float32)
    elif u_ctx.size > 0:
        u_min, u_max = float(np.min(u_ctx)), float(np.max(u_ctx))
        v_min, v_max = float(np.min(v_ctx)), float(np.max(v_ctx))
    else:
        u_min, u_max = float(np.min(u_tar)), float(np.max(u_tar))
        v_min, v_max = float(np.min(v_tar)), float(np.max(v_tar))

    u_min = min(float(u_min), float(np.min(u_tar)))
    u_max = max(float(u_max), float(np.max(u_tar)))
    v_min = min(float(v_min), float(np.min(v_tar)))
    v_max = max(float(v_max), float(np.max(v_tar)))

    span_u = max(1e-6, float(u_max - u_min))
    span_v = max(1e-6, float(v_max - v_min))
    pad = 0.06
    u_min -= span_u * pad
    u_max += span_u * pad
    v_min -= span_v * pad
    v_max += span_v * pad

    px_ctx = ((u_ctx - u_min) / max(1e-6, u_max - u_min) * (image_size - 1)).astype(np.int32)
    py_ctx = ((v_max - v_ctx) / max(1e-6, v_max - v_min) * (image_size - 1)).astype(np.int32)
    px_tar = ((u_tar - u_min) / max(1e-6, u_max - u_min) * (image_size - 1)).astype(np.int32)
    py_tar = ((v_max - v_tar) / max(1e-6, v_max - v_min) * (image_size - 1)).astype(np.int32)
    px_ctx = np.clip(px_ctx, 0, image_size - 1)
    py_ctx = np.clip(py_ctx, 0, image_size - 1)
    px_tar = np.clip(px_tar, 0, image_size - 1)
    py_tar = np.clip(py_tar, 0, image_size - 1)
    return px_ctx, py_ctx, px_tar, py_tar


def _draw_target_bbox(canvas: np.ndarray, px: np.ndarray, py: np.ndarray) -> None:
    if px.size == 0 or py.size == 0 or px.size != py.size:
        return
    x0 = int(np.min(px))
    x1 = int(np.max(px))
    y0 = int(np.min(py))
    y1 = int(np.max(py))
    if x1 > x0 and y1 > y0:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 40, 255), 2, cv2.LINE_AA)


def _bbox_2d_from_pixels(px: np.ndarray, py: np.ndarray, width: int, height: int) -> dict[str, Any]:
    out = {
        "bbox_2d_xyxy": None,
        "bbox_2d_xywh": None,
        "bbox_2d_image_size": [int(width), int(height)],
        "bbox_2d_valid": False,
    }
    if px.size == 0 or py.size == 0 or px.size != py.size:
        return out
    x0 = int(np.min(px))
    x1 = int(np.max(px))
    y0 = int(np.min(py))
    y1 = int(np.max(py))
    if x1 <= x0 or y1 <= y0:
        return out
    w = int(x1 - x0 + 1)
    h = int(y1 - y0 + 1)
    out["bbox_2d_xyxy"] = [x0, y0, x1, y1]
    out["bbox_2d_xywh"] = [x0, y0, w, h]
    out["bbox_2d_valid"] = True
    return out


def _forward_to_yaw_pitch_deg(forward: np.ndarray) -> tuple[float, float]:
    """
    Convert a forward vector (in ENU coordinates) to yaw/pitch angles for AirSim (in NED coordinates).
    
    The forward vector is computed in ENU: f = look_target - eye (both in ENU).
    To match AirSim's NED convention for yaw, we transform the horizontal components,
    but pitch (elevation angle) works the same way in both coordinate systems.
    
    In ENU: yaw is measured from East (X) axis
    In NED: yaw should be measured from North (X) axis
    Transform: yaw_ned = atan2(-fy_enu, fx_enu)
    """
    f = np.asarray(forward, dtype=np.float32).reshape(3)
    
    # For yaw: transform from ENU to NED convention
    # In ENU: yaw = atan2(fy, fx)
    # In NED: yaw = atan2(-fy_enu, fx_enu) because Y_ned = -Y_enu
    yaw_deg = float(math.degrees(math.atan2(-float(f[1]), float(f[0]))))
    
    # For pitch: elevation angle works the same way
    # pitch = atan2(fz, sqrt(fx^2 + fy^2))
    horiz = float(np.linalg.norm(f[:2]))
    pitch_deg = float(math.degrees(math.atan2(float(f[2]), max(1e-6, horiz))))
    
    return yaw_deg, pitch_deg


def _project_points_pinhole(
    points_xyz: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rel = points_xyz - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1e-3
    if not np.any(valid):
        empty_ids = np.empty((0,), dtype=np.int64) if ids is not None else None
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), empty_ids

    x_cam = x_cam[valid]
    y_cam = y_cam[valid]
    z_cam = z_cam[valid]
    in_ids = ids[valid] if ids is not None else None

    f = 0.5 * float(width) / max(1e-6, math.tan(math.radians(float(fov_deg)) * 0.5))
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    px = (x_cam / z_cam) * f + cx
    py = cy - (y_cam / z_cam) * f
    in_frame = (px >= 0.0) & (px < float(width)) & (py >= 0.0) & (py < float(height))
    if not np.any(in_frame):
        empty_ids = np.empty((0,), dtype=np.int64) if ids is not None else None
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), empty_ids

    out_ids = in_ids[in_frame] if in_ids is not None else None
    return px[in_frame].astype(np.int32), py[in_frame].astype(np.int32), out_ids


def _project_points_pinhole_with_depth(
    points_xyz: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    rel = points_xyz - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1e-3
    if not np.any(valid):
        empty_ids = np.empty((0,), dtype=np.int64) if ids is not None else None
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            empty_ids,
        )

    x_cam = x_cam[valid]
    y_cam = y_cam[valid]
    z_cam = z_cam[valid]
    in_ids = ids[valid] if ids is not None else None

    f = 0.5 * float(width) / max(1e-6, math.tan(math.radians(float(fov_deg)) * 0.5))
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    px = (x_cam / z_cam) * f + cx
    py = cy - (y_cam / z_cam) * f
    in_frame = (px >= 0.0) & (px < float(width)) & (py >= 0.0) & (py < float(height))
    if not np.any(in_frame):
        empty_ids = np.empty((0,), dtype=np.int64) if ids is not None else None
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            empty_ids,
        )

    out_ids = in_ids[in_frame] if in_ids is not None else None
    return (
        px[in_frame].astype(np.int32),
        py[in_frame].astype(np.int32),
        z_cam[in_frame].astype(np.float32),
        out_ids,
    )


def _compute_target_visible_stats(
    points_xyz: np.ndarray,
    context_points_xyz: np.ndarray,
    context_instance_ids: np.ndarray,
    target_instance_id: int,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
    occlusion_neighbor_px: int,
    occlusion_depth_margin_m: float,
) -> tuple[int, int, float]:
    """Compute visible-point statistics for the target instance under a given view.

    This function only estimates the visible-point ratio; instance-level occlusion is handled by neighbor BBox ray checks.
    """
    tx, ty, tz, _ = _project_points_pinhole_with_depth(
        points_xyz=points_xyz,
        eye=eye,
        right=right,
        cam_up=cam_up,
        forward=forward,
        width=width,
        height=height,
        fov_deg=fov_deg,
        ids=None,
    )
    total = int(points_xyz.shape[0])
    if tx.size == 0:
        return 0, total, 0.0

    cx, cy, cz, cids = _project_points_pinhole_with_depth(
        points_xyz=context_points_xyz,
        eye=eye,
        right=right,
        cam_up=cam_up,
        forward=forward,
        width=width,
        height=height,
        fov_deg=fov_deg,
        ids=context_instance_ids,
    )
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    if cx.size > 0:
        occ_mask = np.ones((cx.shape[0],), dtype=bool)
        if cids is not None and cids.size == cx.shape[0]:
            occ_mask = cids != int(target_instance_id)
        if np.any(occ_mask):
            np.minimum.at(depth_map, (cy[occ_mask], cx[occ_mask]), cz[occ_mask])

    neighbor = max(0, int(occlusion_neighbor_px))
    margin = float(max(0.0, occlusion_depth_margin_m))
    visible_count = 0
    for px, py, pz in zip(tx.tolist(), ty.tolist(), tz.tolist()):
        x0 = max(0, int(px) - neighbor)
        x1 = min(width, int(px) + neighbor + 1)
        y0 = max(0, int(py) - neighbor)
        y1 = min(height, int(py) + neighbor + 1)
        local_min = float(np.min(depth_map[y0:y1, x0:x1]))
        if not np.isfinite(local_min) or local_min >= float(pz) - margin:
            visible_count += 1

    ratio = float(visible_count) / float(max(1, total))
    return int(visible_count), total, ratio


def _project_target_bbox_pinhole(
    target_points_xyz: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    cam_up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    rel = target_points_xyz - eye[None, :]
    x_cam = rel @ right
    y_cam = rel @ cam_up
    z_cam = rel @ forward
    valid = z_cam > 1e-3
    if not np.any(valid):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
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
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    return px[in_frame].astype(np.int32), py[in_frame].astype(np.int32)


def _capture_rgb_views_from_sim(
    scene_id: str,
    config: dict[str, Any],
    worker_binding: WorkerBinding,
    target_points_xyz: np.ndarray,
    view_specs: list[dict[str, Any]],
    output_width: int,
    output_height: int,
    out_dir: Path,
    pose_settle_sec: float,
    sim_camera_fov_deg: float,
    project_camera_fov_deg: float,
    square_crop: bool,
    draw_bbox_on_image: bool = False,
    pose_params: dict[str, float] | None = None,
    bridge: Any | None = None,
) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    output_width = max(64, int(output_width))
    output_height = max(64, int(output_height))

    bridge_created_here = bridge is None
    if bridge is None:
        bridge_cfg = _build_bridge_config(config=config, vehicle_name=worker_binding.vehicle)
        task_cfg = config.get("task", {}) or {}
        engine = str(task_cfg.get("engine", "airsim")).lower()
        bridge = create_bridge(engine=engine, scene_id=scene_id, config=bridge_cfg)

    views: list[dict[str, Any]] = []
    try:
        if bridge_created_here and hasattr(bridge, "set_camera_fov"):
            try:
                bridge.set_camera_fov(
                    fov_deg=float(sim_camera_fov_deg),
                    vehicle_or_actor=worker_binding.vehicle,
                )
            except Exception:
                pass

        for vid, view_spec in enumerate(view_specs):
            view_id = int(view_spec.get("view_id", vid))
            mode = str(view_spec.get("mode", "orbit"))
            label = str(view_spec.get("label", f"view_{view_id:02d}"))
            view_direction = str(view_spec.get("view_direction", label) or label)
            yaw_deg_view = float(view_spec.get("yaw_deg", 0.0))
            pitch_offset_deg = float(view_spec.get("pitch_offset_deg", 0.0) or 0.0)
            if mode == "topdown":
                center, eye, forward, right, cam_up = _camera_pose_topdown(
                    target_points_xyz=target_points_xyz,
                    yaw_deg=yaw_deg_view,
                    pose_params=pose_params,
                )
            else:
                yaw_deg_preset = yaw_deg_view
                center, eye, forward, right, cam_up = _camera_pose_for_yaw(
                    target_points_xyz=target_points_xyz,
                    yaw_deg=yaw_deg_preset,
                    extra_pitch_deg=pitch_offset_deg,
                    pose_params=pose_params,
                )
            yaw_deg, pitch_deg = _forward_to_yaw_pitch_deg(forward)
            roll_deg = 0.0
            if mode == "topdown":
                roll_deg = -float(yaw_deg_view)
            bridge.set_uav_pose(
                x=float(eye[0]),
                y=float(eye[1]),
                z=float(eye[2]),
                yaw=yaw_deg,
                pitch=pitch_deg,
                roll=roll_deg,
                vehicle_or_actor=worker_binding.vehicle,
            )
            if pose_settle_sec > 0:
                time.sleep(pose_settle_sec)
            rgb = bridge.capture_rgb()
            rgb_np = np.asarray(rgb) if rgb is not None else np.empty((0, 0, 3), dtype=np.uint8)
            if rgb_np.ndim == 2:
                rgb_np = cv2.cvtColor(rgb_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if rgb_np.ndim != 3 or rgb_np.shape[2] < 3:
                canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            else:
                canvas = rgb_np[:, :, :3].copy()
                if square_crop:
                    canvas = _center_crop_square(canvas)
                if canvas.shape[1] != output_width or canvas.shape[0] != output_height:
                    canvas = cv2.resize(canvas, (output_width, output_height), interpolation=cv2.INTER_AREA)

            bbox_px, bbox_py = _project_target_bbox_pinhole(
                target_points_xyz=target_points_xyz,
                eye=eye,
                right=right,
                cam_up=cam_up,
                forward=forward,
                width=int(canvas.shape[1]),
                height=int(canvas.shape[0]),
                fov_deg=project_camera_fov_deg,
            )
            bbox_meta = _bbox_2d_from_pixels(
                px=bbox_px,
                py=bbox_py,
                width=int(canvas.shape[1]),
                height=int(canvas.shape[0]),
            )
            if draw_bbox_on_image:
                _draw_target_bbox(canvas, bbox_px, bbox_py)

            if mode == "topdown":
                out_name = f"view_{view_id:02d}_yaw_{_yaw_token(yaw_deg_view)}.png"
            else:
                out_name = f"view_{view_id:02d}_yaw_{_yaw_token(yaw_deg_view)}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), canvas)
            views.append(
                {
                    "view_id": int(view_id),
                    "path": out_name,
                    "mode": mode,
                    "label": label,
                    "view_direction": view_direction,
                    "yaw_deg": float(yaw_deg_view),
                    "pitch_deg": float(pitch_deg),
                    "pitch_offset_deg": float(pitch_offset_deg),
                    "bbox_2d_xyxy": bbox_meta.get("bbox_2d_xyxy", None),
                    "bbox_2d_xywh": bbox_meta.get("bbox_2d_xywh", None),
                    "bbox_2d_image_size": bbox_meta.get("bbox_2d_image_size", None),
                    "bbox_2d_valid": bool(bbox_meta.get("bbox_2d_valid", False)),
                }
            )
    finally:
        if bridge_created_here:
            try:
                bridge.shutdown()
            except Exception:
                pass
    return views


def _resolve_input_pcd(scene_id: str, args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.pcd:
        return Path(args.pcd)
    scene_root = _resolve_scene_root(config=config, scene_id=scene_id)
    stage1_dir_name = _resolve_output_dir_name(config, key="stage1_dir", default="pcd_map")
    pcd_map_dir = scene_root / stage1_dir_name
    candidates = [
        pcd_map_dir / f"{scene_id}.pcd",
        pcd_map_dir / f"{scene_id}.semantic_lidar.pcd",
        pcd_map_dir / f"{scene_id}.semantic_lidar.ply",
        pcd_map_dir / f"{scene_id}.raw.pcd",
    ]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        return existing[0]
    tried = ", ".join(p.as_posix() for p in candidates)
    raise FileNotFoundError(f"No stage2 input point cloud found, tried: {tried}")


def _render_side_view(
    target_points_xyz: np.ndarray,
    context_points_xyz: np.ndarray,
    context_instance_ids: np.ndarray,
    out_path: Path,
    title: str,
    lower_row_range_scale: float = 1.8,
) -> None:
    panel_h = 320
    panel_w = 480
    margin = 24
    height = panel_h * 2 + margin
    width = panel_w * 2
    image = np.full((height, width, 3), 18, dtype=np.uint8)

    cx = context_points_xyz[:, 0]
    cy = context_points_xyz[:, 1]
    cz = context_points_xyz[:, 2]

    tx = target_points_xyz[:, 0]
    ty = target_points_xyz[:, 1]
    tz = target_points_xyz[:, 2]

    x_min_t, x_max_t = float(np.min(tx)), float(np.max(tx))
    y_min_t, y_max_t = float(np.min(ty)), float(np.max(ty))
    z_min_t, z_max_t = float(np.min(tz)), float(np.max(tz))

    def _expand_range(vmin: float, vmax: float, scale: float) -> tuple[float, float]:
        center = 0.5 * (vmin + vmax)
        half = max(1e-3, 0.5 * (vmax - vmin) * max(1.0, scale))
        return center - half, center + half

    def _project(
        v0: np.ndarray,
        v1: np.ndarray,
        min0: float,
        max0: float,
        min1: float,
        max1: float,
        x_offset: int,
        y_offset: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        eps = 1e-6
        span0 = max(eps, max0 - min0)
        span1 = max(eps, max1 - min1)
        draw_w = panel_w - 40
        draw_h = panel_h - 40
        scale = min(draw_w / span0, draw_h / span1)

        width_used = span0 * scale
        height_used = span1 * scale
        base_x = x_offset + 20 + 0.5 * (draw_w - width_used)
        base_y = y_offset + 20 + 0.5 * (draw_h - height_used)

        px = ((v0 - min0) * scale + base_x).astype(np.int32)
        py = ((max1 - v1) * scale + base_y).astype(np.int32)
        return np.clip(px, 0, width - 1), np.clip(py, 0, height - 1)

    def _color_from_instance_id(instance_id: int) -> tuple[int, int, int]:
        hue = (int(instance_id) * 37) % 180
        hsv = np.array([[[hue, 210, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def _project_drone_view(
        context_xyz: np.ndarray,
        context_ids: np.ndarray,
        target_xyz: np.ndarray,
        yaw_deg: float,
        x_offset: int,
        y_offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target_center = np.mean(target_xyz, axis=0).astype(np.float32)
        target_min = np.min(target_xyz, axis=0)
        target_max = np.max(target_xyz, axis=0)
        target_size = target_max - target_min
        target_diag_xy = float(np.linalg.norm(target_size[:2]))
        target_height = float(max(1e-3, target_size[2]))

        scale = max(1.0, float(lower_row_range_scale))
        scale_eff = min(2.0, math.sqrt(scale))
        cam_distance = max(24.0, target_diag_xy * (2.6 + 0.85 * scale_eff))
        cam_height = max(7.0, target_height * 1.2 + cam_distance * 0.28)
        fov_deg = 62.0
        fov_tan = math.tan(math.radians(fov_deg) * 0.5)

        yaw = math.radians(yaw_deg)
        forward_xy = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float32)
        eye = target_center - forward_xy * cam_distance + np.array([0.0, 0.0, cam_height], dtype=np.float32)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        forward = target_center - eye
        forward /= float(np.linalg.norm(forward) + 1e-6)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm
        cam_up = np.cross(right, forward)
        cam_up /= float(np.linalg.norm(cam_up) + 1e-6)

        def _camera_project(points_xyz: np.ndarray, ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
            rel = points_xyz - eye[None, :]
            x_cam = rel @ right
            y_cam = rel @ cam_up
            z_cam = rel @ forward
            valid = z_cam > 1.0
            if not np.any(valid):
                return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), (np.empty((0,), dtype=np.int64) if ids is not None else None)
            u = x_cam[valid] / z_cam[valid]
            v = y_cam[valid] / z_cam[valid]
            fov_mask = (np.abs(u) <= fov_tan) & (np.abs(v) <= fov_tan)
            if not np.any(fov_mask):
                return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), (np.empty((0,), dtype=np.int64) if ids is not None else None)
            u = u[fov_mask]
            v = v[fov_mask]
            id_out = ids[valid][fov_mask] if ids is not None else None
            return u.astype(np.float32), v.astype(np.float32), id_out

        cu, cv, cid = _camera_project(context_xyz, context_ids)
        tu, tv, _ = _camera_project(target_xyz)
        if tu.size == 0:
            return (
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )
        if cu.size == 0:
            cu, cv = tu, tv
            cid = np.zeros((tu.size,), dtype=np.int64)

        if cu.size >= 32:
            u_min, u_max = np.quantile(cu, [0.05, 0.95]).astype(np.float32)
            v_min, v_max = np.quantile(cv, [0.05, 0.95]).astype(np.float32)
        else:
            u_min, u_max = float(np.min(cu)), float(np.max(cu))
            v_min, v_max = float(np.min(cv)), float(np.max(cv))

        u_min = min(float(u_min), float(np.min(tu)))
        u_max = max(float(u_max), float(np.max(tu)))
        v_min = min(float(v_min), float(np.min(tv)))
        v_max = max(float(v_max), float(np.max(tv)))

        u_min, u_max = _expand_range(u_min, u_max, 1.06)
        v_min, v_max = _expand_range(v_min, v_max, 1.06)

        cpx, cpy = _project(cu, cv, u_min, u_max, v_min, v_max, x_offset, y_offset)
        tpx, tpy = _project(tu, tv, u_min, u_max, v_min, v_max, x_offset, y_offset)
        cid_arr = cid if cid is not None else np.zeros((cu.size,), dtype=np.int64)
        return cpx, cpy, cid_arr.astype(np.int64), tpx, tpy

    top_left_ctx = _project(tx, tz, x_min_t, x_max_t, z_min_t, z_max_t, 0, 0)
    top_left_tar = _project(tx, tz, x_min_t, x_max_t, z_min_t, z_max_t, 0, 0)
    top_right_ctx = _project(ty, tz, y_min_t, y_max_t, z_min_t, z_max_t, panel_w, 0)
    top_right_tar = _project(ty, tz, y_min_t, y_max_t, z_min_t, z_max_t, panel_w, 0)
    bottom_left = _project_drone_view(
        context_points_xyz,
        context_instance_ids,
        target_points_xyz,
        yaw_deg=45.0,
        x_offset=0,
        y_offset=panel_h + margin,
    )
    bottom_right = _project_drone_view(
        context_points_xyz,
        context_instance_ids,
        target_points_xyz,
        yaw_deg=-45.0,
        x_offset=panel_w,
        y_offset=panel_h + margin,
    )

    context_draw_views = [top_left_ctx, top_right_ctx, (bottom_left[0], bottom_left[1]), (bottom_right[0], bottom_right[1])]
    target_draw_views = [top_left_tar, top_right_tar, (bottom_left[3], bottom_left[4]), (bottom_right[3], bottom_right[4])]
    lower_context_ids = [bottom_left[2], bottom_right[2]]

    target_color = (40, 210, 255)
    for i, (ctx, tar) in enumerate(zip(context_draw_views, target_draw_views)):
        cx, cy = ctx
        txp, typ = tar
        if i < 2:
            image[cy, cx] = (95, 95, 95)
        else:
            ids = lower_context_ids[i - 2]
            if cx.size > 0 and ids.size == cx.size:
                for inst_id in np.unique(ids):
                    mask = ids == inst_id
                    color = _color_from_instance_id(int(inst_id))
                    image[cy[mask], cx[mask]] = color
            else:
                image[cy, cx] = (95, 95, 95)
        if txp.size > 0 and typ.size > 0 and txp.size == typ.size:
            image[typ, txp] = target_color
            x0 = int(np.min(txp))
            x1 = int(np.max(txp))
            y0 = int(np.min(typ))
            y1 = int(np.max(typ))
            if x1 > x0 and y1 > y0:
                cv2.rectangle(image, (x0, y0), (x1, y1), (40, 40, 255), 2, cv2.LINE_AA)
    cv2.line(image, (panel_w, 0), (panel_w, height - 1), (90, 90, 90), 2)
    cv2.line(image, (0, panel_h + margin // 2), (width - 1, panel_h + margin // 2), (90, 90, 90), 2)

    cv2.putText(image, "X-Z", (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "Y-Z", (panel_w + 18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "DroneView(+45)", (18, panel_h + margin + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "DroneView(-45)", (panel_w + 18, panel_h + margin + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image, "Red bbox = target", (panel_w + 220, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(image, title[:120], (18, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    ensure_dir(out_path.parent)
    cv2.imwrite(str(out_path), image)


def _render_pointcloud_views(
    target_points_xyz: np.ndarray,
    context_points_xyz: np.ndarray,
    context_instance_ids: np.ndarray,
    out_dir: Path,
    view_specs: list[dict[str, Any]],
    image_size: int = 320,
    pose_params: dict[str, float] | None = None,
    camera_fov_deg: float = 90.0,
) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    out: list[dict[str, Any]] = []
    for idx, view_spec in enumerate(view_specs):
        view_id = int(view_spec.get("view_id", idx))
        mode = str(view_spec.get("mode", "orbit"))
        label = str(view_spec.get("label", f"view_{view_id:02d}"))
        view_direction = str(view_spec.get("view_direction", label) or label)
        yaw_deg_view = float(view_spec.get("yaw_deg", 0.0))
        pitch_offset_deg = float(view_spec.get("pitch_offset_deg", 0.0) or 0.0)
        if mode == "topdown":
            _, eye, forward, right, cam_up = _camera_pose_topdown(
                target_points_xyz=target_points_xyz,
                yaw_deg=yaw_deg_view,
                pose_params=pose_params,
            )
        else:
            yaw_deg = yaw_deg_view
            _, eye, forward, right, cam_up = _camera_pose_for_yaw(
                target_points_xyz=target_points_xyz,
                yaw_deg=yaw_deg,
                extra_pitch_deg=pitch_offset_deg,
                pose_params=pose_params,
            )
        _, pitch_deg_capture = _forward_to_yaw_pitch_deg(forward)
        px_ctx, py_ctx, ids_ctx = _project_points_pinhole(
            points_xyz=context_points_xyz,
            eye=eye,
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=image_size,
            height=image_size,
            fov_deg=camera_fov_deg,
            ids=context_instance_ids,
        )
        px_tar, py_tar, _ = _project_points_pinhole(
            points_xyz=target_points_xyz,
            eye=eye,
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=image_size,
            height=image_size,
            fov_deg=camera_fov_deg,
            ids=None,
        )
        canvas = np.full((image_size, image_size, 3), 18, dtype=np.uint8)
        if px_ctx.size > 0 and ids_ctx is not None and ids_ctx.size == px_ctx.size:
            uniq_ids = np.unique(ids_ctx)
            for inst_id in uniq_ids:
                hue = (int(inst_id) * 37) % 180
                hsv = np.array([[[hue, 210, 255]]], dtype=np.uint8)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                mask = ids_ctx == inst_id
                canvas[py_ctx[mask], px_ctx[mask]] = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        elif px_ctx.size > 0:
            canvas[py_ctx, px_ctx] = (95, 95, 95)

        if px_tar.size > 0:
            canvas[py_tar, px_tar] = (40, 210, 255)
            _draw_target_bbox(canvas, px_tar, py_tar)
        cv2.putText(
            canvas,
            f"PCD-View {view_id}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        if mode == "topdown":
            name = f"pcd_view_{view_id:02d}_yaw_{_yaw_token(yaw_deg_view)}.png"
        else:
            name = f"pcd_view_{view_id:02d}_yaw_{_yaw_token(yaw_deg_view)}.png"
        out_path = out_dir / name
        cv2.imwrite(str(out_path), canvas)
        out.append(
            {
                "view_id": int(view_id),
                "path": name,
                "mode": mode,
                "label": label,
                "view_direction": view_direction,
                "yaw_deg": float(yaw_deg_view),
                "pitch_deg": float(pitch_deg_capture),
                "pitch_offset_deg": float(pitch_offset_deg),
            }
        )
    return out


def _render_rgb_views(
    target_points_xyz: np.ndarray,
    context_points_xyz: np.ndarray,
    out_dir: Path,
    view_specs: list[dict[str, Any]],
    image_size: int = 320,
    pose_params: dict[str, float] | None = None,
    camera_fov_deg: float = 90.0,
    draw_bbox_on_image: bool = False,
) -> list[dict[str, Any]]:
    ensure_dir(out_dir)
    points = context_points_xyz.astype(np.float32)

    views: list[dict[str, Any]] = []
    for vid, view_spec in enumerate(view_specs):
        view_id = int(view_spec.get("view_id", vid))
        mode = str(view_spec.get("mode", "orbit"))
        label = str(view_spec.get("label", f"view_{view_id:02d}"))
        yaw_deg_view = float(view_spec.get("yaw_deg", 0.0))
        if mode == "topdown":
            _, eye, forward, right, cam_up = _camera_pose_topdown(
                target_points_xyz=target_points_xyz,
                yaw_deg=yaw_deg_view,
                pose_params=pose_params,
            )
        else:
            yaw_deg = yaw_deg_view
            _, eye, forward, right, cam_up = _camera_pose_for_yaw(
                target_points_xyz=target_points_xyz,
                yaw_deg=yaw_deg,
                pose_params=pose_params,
            )
        px_ctx, py_ctx, _ = _project_points_pinhole(
            points_xyz=points,
            eye=eye,
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=image_size,
            height=image_size,
            fov_deg=camera_fov_deg,
            ids=None,
        )
        px_tar, py_tar, _ = _project_points_pinhole(
            points_xyz=target_points_xyz,
            eye=eye,
            right=right,
            cam_up=cam_up,
            forward=forward,
            width=image_size,
            height=image_size,
            fov_deg=camera_fov_deg,
            ids=None,
        )
        if px_tar.size == 0:
            canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            canvas = np.full((image_size, image_size, 3), 16, dtype=np.uint8)
            if px_ctx.size > 0:
                base_color = np.array([110, 180, 210], dtype=np.uint8)
                bgr = np.repeat(base_color[None, :], px_ctx.shape[0], axis=0)
                canvas[py_ctx, px_ctx] = bgr
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    xx = np.clip(px_ctx + dx, 0, image_size - 1)
                    yy = np.clip(py_ctx + dy, 0, image_size - 1)
                    canvas[yy, xx] = bgr

            if px_tar.size > 0:
                canvas[py_tar, px_tar] = (40, 210, 255)
                if draw_bbox_on_image:
                    _draw_target_bbox(canvas, px_tar, py_tar)

        bbox_meta = _bbox_2d_from_pixels(
            px=px_tar,
            py=py_tar,
            width=int(image_size),
            height=int(image_size),
        )

        cv2.putText(canvas, f"RGB-View {view_id}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if mode == "topdown":
            out_name = f"view_{view_id:02d}_yaw_{_yaw_token(yaw_deg_view)}.png"
        else:
            out_name = f"view_{view_id:02d}_yaw_{_yaw_token(yaw_deg_view)}.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), canvas)
        views.append(
            {
                "view_id": int(view_id),
                "path": out_name,
                "mode": mode,
                "label": label,
                "yaw_deg": float(yaw_deg_view),
                "bbox_2d_xyxy": bbox_meta.get("bbox_2d_xyxy", None),
                "bbox_2d_xywh": bbox_meta.get("bbox_2d_xywh", None),
                "bbox_2d_image_size": bbox_meta.get("bbox_2d_image_size", None),
                "bbox_2d_valid": bool(bbox_meta.get("bbox_2d_valid", False)),
            }
        )

    return views


def _as_int_col(data: np.ndarray, index: int) -> np.ndarray:
    return np.asarray(np.rint(data[:, index]), dtype=np.int64)


def collect_instances(scene_id: str, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    logger = StageLogger("stage2.collect_instances")
    t_start = time.time()
    logger.info("phase=read_pcd start")
    required = ["x", "y", "z", "class_id", "instance_id"]
    input_pcd = _resolve_input_pcd(scene_id, args, config)
    if not input_pcd.exists():
        raise FileNotFoundError(f"pcd not found: {input_pcd}")

    candidate_paths = [input_pcd]
    if not args.pcd:
        scene_root = _resolve_scene_root(config=config, scene_id=scene_id)
        stage1_dir_name = _resolve_output_dir_name(config, key="stage1_dir", default="pcd_map")
        pcd_map_dir = scene_root / stage1_dir_name
        fallback_paths = [
            pcd_map_dir / f"{scene_id}.semantic_lidar.pcd",
            pcd_map_dir / f"{scene_id}.pcd",
            pcd_map_dir / f"{scene_id}.raw.pcd",
            pcd_map_dir / f"{scene_id}.semantic_lidar.ply",
        ]
        for path in fallback_paths:
            if path.exists() and path not in candidate_paths:
                candidate_paths.append(path)

    pcd: PcdData | None = None
    input_pcd_used: Path | None = None
    field_index: dict[str, int] = {}
    for candidate in candidate_paths:
        if candidate.suffix.lower() != ".pcd":
            continue
        parsed = _load_pcd_ascii(candidate)
        parsed_index = {name: idx for idx, name in enumerate(parsed.fields)}
        if all(key in parsed_index for key in required):
            pcd = parsed
            input_pcd_used = candidate
            field_index = parsed_index
            break

    if pcd is None or input_pcd_used is None:
        if args.pcd:
            raise ValueError(f"PCD missing required fields {required}: {input_pcd}")
        tried = ", ".join(path.as_posix() for path in candidate_paths)
        raise ValueError(f"No valid segmented PCD found with fields {required}. tried: {tried}")

    input_pcd = input_pcd_used
    fields = pcd.fields
    arr = pcd.data
    logger.info(f"source_pcd={input_pcd.as_posix()} fields={fields}")

    xyz = np.stack(
        [
            arr[:, field_index["x"]],
            arr[:, field_index["y"]],
            arr[:, field_index["z"]],
        ],
        axis=1,
    ).astype(np.float32)
    class_ids = _as_int_col(arr, field_index["class_id"])
    instance_ids = _as_int_col(arr, field_index["instance_id"])

    stage2_cfg = config.get("stage2", {}) or {}
    min_points = int(args.min_points if args.min_points is not None else stage2_cfg.get("collect_min_points", 50))
    exclude_class_ids = set(int(v) for v in (stage2_cfg.get("collect_exclude_class_ids", [0]) or [0]))
    exclude_nonpositive_instance = bool(stage2_cfg.get("collect_exclude_nonpositive_instance", True))
    ground_percentile = stage2_cfg.get("pcd_ground_percentile", None)
    ground_margin_m = float(stage2_cfg.get("pcd_ground_margin_m", 0.0) or 0.0)
    max_instances = int(stage2_cfg.get("pcd_max_instances", 0) or 0)
    preview_bg_max_points = int(stage2_cfg.get("collect_preview_bg_max_points", 300000) or 300000)
    preview_context_radius_scale = float(stage2_cfg.get("collect_preview_context_radius_scale", 8.0) or 8.0)
    preview_context_radius_min_m = float(stage2_cfg.get("collect_preview_context_radius_min_m", 25.0) or 25.0)
    preview_context_height_scale = float(stage2_cfg.get("collect_preview_context_height_scale", 6.0) or 6.0)
    preview_context_height_min_m = float(stage2_cfg.get("collect_preview_context_height_min_m", 15.0) or 15.0)
    class_name_map = _build_class_name_map(config)
    enable_rgb_views = bool(stage2_cfg.get("collect_enable_rgb_views", True))
    # collect_rgb_views_count counts orbit side views only.
    # Bird's-eye, when enabled, is an extra optional view and must not consume this quota.
    rgb_views_count = int(stage2_cfg.get("collect_rgb_views_count", 8) or 8)
    rgb_views_count = max(1, min(8, rgb_views_count))
    side_view_count = max(1, min(8, rgb_views_count))
    side_view_min_keep = int(stage2_cfg.get("collect_side_view_min_keep", 4) or 4)
    side_view_min_keep = max(1, min(8, side_view_min_keep))
    add_birdseye_view = bool(stage2_cfg.get("collect_add_birdseye_view", False))
    side_view_min_visible_ratio = float(stage2_cfg.get("collect_side_view_min_visible_ratio", 0.02) or 0.02)
    side_view_min_visible_points = int(stage2_cfg.get("collect_side_view_min_visible_points", 80) or 80)
    side_view_occlusion_neighbor_px = int(stage2_cfg.get("collect_side_view_occlusion_neighbor_px", 2) or 2)
    side_view_occlusion_depth_margin_m = float(stage2_cfg.get("collect_side_view_occlusion_depth_margin_m", 0.8) or 0.8)
    side_view_occlusion_search_radius_scale = float(
        stage2_cfg.get("collect_side_view_occlusion_search_radius_scale", 1.5) or 1.5
    )
    side_view_occlusion_search_radius_min_m = float(
        stage2_cfg.get("collect_side_view_occlusion_search_radius_min_m", 12.0) or 12.0
    )
    camera_pose_params = _build_camera_pose_params(stage2_cfg)
    camera_cfg = config.get("camera", {}) or {}
    sim_camera_fov_deg = float(camera_cfg.get("fov", 90.0) or 90.0)
    camera_width = int(camera_cfg.get("width", 3840) or 3840)
    camera_height = int(camera_cfg.get("height", 2160) or 2160)
    view_image_size = int(
        stage2_cfg.get("collect_view_image_size", stage2_cfg.get("collect_rgb_image_size", 320)) or 320
    )
    view_image_size = max(96, min(2048, view_image_size))
    view_image_width = int(stage2_cfg.get("collect_view_image_width", camera_width) or camera_width)
    view_image_height = int(stage2_cfg.get("collect_view_image_height", camera_height) or camera_height)
    view_image_width = max(96, min(8192, view_image_width))
    view_image_height = max(96, min(8192, view_image_height))
    rgb_square_crop = bool(stage2_cfg.get("collect_rgb_square_crop", True))
    use_square_capture_in_airsim = bool(stage2_cfg.get("collect_use_square_capture_in_airsim", True))
    rgb_source = str(stage2_cfg.get("collect_rgb_source", "sim")).strip().lower()
    if rgb_source not in {"sim", "pointcloud"}:
        rgb_source = "sim"
    if rgb_source == "sim" and use_square_capture_in_airsim:
        rgb_square_crop = False
        project_camera_fov_deg = sim_camera_fov_deg
    elif rgb_source == "sim" and rgb_square_crop:
        project_camera_fov_deg = _effective_square_fov_deg(
            fov_deg=sim_camera_fov_deg,
            source_width=camera_width,
            source_height=camera_height,
        )
    else:
        project_camera_fov_deg = sim_camera_fov_deg
    camera_pose_params["camera_fov_deg"] = float(project_camera_fov_deg)
    side_view_pitch_offsets_deg = _build_side_view_pitch_offsets_deg(stage2_cfg)
    rgb_pose_settle_sec = max(0.0, float(stage2_cfg.get("collect_rgb_pose_settle_sec", 0.03) or 0.03))
    rgb_draw_bbox_on_image = bool(stage2_cfg.get("collect_rgb_draw_bbox_on_image", False))
    collect_parallel_workers = int(stage2_cfg.get("collect_parallel_workers", 0) or 0)
    if collect_parallel_workers <= 0:
        collect_parallel_workers = max(1, int((config.get("parallel", {}) or {}).get("workers", 2) or 2))
    collect_log_every = max(1, int(stage2_cfg.get("collect_progress_log_every", 25) or 25))
    collect_pair_limit = max(0, int(stage2_cfg.get("collect_pair_limit", 0) or 0))

    valid_mask = np.ones((xyz.shape[0],), dtype=bool)
    if exclude_class_ids:
        valid_mask &= ~np.isin(class_ids, np.asarray(sorted(exclude_class_ids), dtype=np.int64))
    if exclude_nonpositive_instance:
        valid_mask &= instance_ids > 0

    ground_z_threshold = None
    if ground_percentile is not None and xyz.shape[0] > 0:
        p = float(ground_percentile)
        p = max(0.0, min(100.0, p))
        base_ground_z = float(np.percentile(xyz[:, 2], p))
        ground_z_threshold = base_ground_z + ground_margin_m
        valid_mask &= xyz[:, 2] > ground_z_threshold

    xyz = xyz[valid_mask]
    class_ids = class_ids[valid_mask]
    instance_ids = instance_ids[valid_mask]

    pair = np.stack([class_ids, instance_ids], axis=1)
    unique_pair, inverse = np.unique(pair, axis=0, return_inverse=True)
    if collect_pair_limit > 0 and unique_pair.shape[0] > collect_pair_limit:
        unique_pair = unique_pair[:collect_pair_limit]
    logger.info("phase=read_pcd done")
    logger.info(
        f"input_points={arr.shape[0]} valid_points={xyz.shape[0]} "
        f"candidate_pairs={unique_pair.shape[0]} min_points={min_points}"
    )
    logger.info(
        f"phase=collect_render start pointcloud_preview=on rgb_views={'on' if enable_rgb_views else 'off'} "
        f"rgb_source={rgb_source} side_view_count={side_view_count} workers={collect_parallel_workers} "
        f"pair_limit={collect_pair_limit} cam_dist_scale={camera_pose_params['distance_scale']:.2f} "
        f"cam_h_scale={camera_pose_params['height_scale']:.2f} pitch_offsets={side_view_pitch_offsets_deg} birdseye={add_birdseye_view} "
        f"fov_sim={sim_camera_fov_deg:.2f} fov_proj={project_camera_fov_deg:.2f} square_capture={use_square_capture_in_airsim} "
        f"rgb_bbox_baked={rgb_draw_bbox_on_image}"
    )

    scene_root = _resolve_scene_root(config=config, scene_id=scene_id)
    stage2_raw_dir_name = _resolve_output_dir_name(config, key="stage2_raw_dir", default="landmarks_raw")
    landmarks_raw_root = scene_root / stage2_raw_dir_name
    pcd_views_root = landmarks_raw_root / "pcd_views"
    rgb_views_root = landmarks_raw_root / "rgb_views"
    ensure_dir(pcd_views_root)
    ensure_dir(rgb_views_root)

    instances: list[dict[str, Any]] = []
    image_index: list[dict[str, Any]] = []
    dropped_small = 0
    dropped_view_insufficient = 0
    rendered_pointcloud = 0
    rendered_rgb_instances = 0

    if xyz.shape[0] > preview_bg_max_points:
        rng = np.random.default_rng(20260320)
        bg_idx = rng.choice(xyz.shape[0], size=preview_bg_max_points, replace=False)
        preview_bg_points = xyz[bg_idx]
        preview_bg_instance_ids = instance_ids[bg_idx]
    else:
        preview_bg_points = xyz
        preview_bg_instance_ids = instance_ids

    progress = ProgressBar(total=int(unique_pair.shape[0]), label="stage2.collect")
    progress.update(0, detail="start")
    rgb_progress = ProgressBar(total=int(unique_pair.shape[0]), label="stage2.rgb") if enable_rgb_views else None
    if rgb_progress is not None:
        rgb_progress.update(0, detail="start")

    worker_bindings = parse_bindings(config=config, worker_count=collect_parallel_workers)
    task_cfg_engine = config.get("task", {}) or {}
    if str(task_cfg_engine.get("engine", "airsim")).lower() == "airsim":
        for idx, binding in enumerate(worker_bindings):
            binding.vehicle = normalize_airsim_vehicle_name(binding.vehicle, idx)
    logger.info(
        "worker_vehicle_map="
        + ", ".join([f"{int(binding.worker_id)}->{str(binding.vehicle)}" for binding in worker_bindings])
    )
    worker_vehicle_names = [str(binding.vehicle) for binding in worker_bindings]

    bridge_cache: dict[int, Any] = {}
    bridge_locks: dict[int, threading.Lock] = {}
    bridge_fov_initialized_workers: set[int] = set()
    vehicle_capture_locks: dict[str, threading.Lock] = {}
    fallback_vehicle_lock_warnings: set[str] = set()
    bridge_cache_lock = threading.Lock()
    engine_name = str(task_cfg_engine.get("engine", "airsim")).lower()
    rgb_executor: concurrent.futures.ThreadPoolExecutor | None = None
    bootstrap_bridge: Any | None = None
    runtime_port = None

    if engine_name == "airsim" and worker_bindings:
        first_vehicle = str(worker_bindings[0].vehicle)
        airsim_cfg = ((config.get("engine_params", {}) or {}).get("airsim", {}) or {})
        base_bridge_cfg = build_unified_bridge_config(
            config,
            engine="airsim",
            vehicle_name=first_vehicle,
            sim_port=int(airsim_cfg.get("sim_port", 41471)),
            image_width=(view_image_size if use_square_capture_in_airsim else view_image_width),
            image_height=(view_image_size if use_square_capture_in_airsim else view_image_height),
            fov=sim_camera_fov_deg,
            vehicle_names=worker_vehicle_names,
        )
        base_bridge_cfg["headless"] = bool(airsim_cfg.get("headless", True))
        base_bridge_cfg["launch_sim"] = True
        base_bridge_cfg["connect_on_init"] = True
        base_bridge_cfg["auto_select_port_on_conflict"] = True
        base_bridge_cfg["strict_vehicle_name"] = True

        launch_config = {
            "engine_params": {
                "airsim": {
                    "launch_sim": True,
                }
            }
        }

        runtime_port, bootstrap_bridge, launched_by_bridge, configured_port = prepare_airsim_runtime_unified(
            config=launch_config,
            scene_id=scene_id,
            base_bridge_cfg=base_bridge_cfg,
            vehicle_name=first_vehicle,
            vehicle_names=worker_vehicle_names,
        )
        logger.info(
            format_unified_startup_ports_message(
                stage="stage2.collect_instances",
                engine="airsim",
                configured_sim_port=int(configured_port),
                runtime_sim_port=int(runtime_port),
                launched_by_bridge=bool(launched_by_bridge),
            )
        )

    def _get_shared_bridge(binding: WorkerBinding) -> Any:
        def _init_worker_bridge_fov(worker_id: int, bridge_obj: Any) -> None:
            if worker_id in bridge_fov_initialized_workers:
                return
            if not hasattr(bridge_obj, "set_camera_fov"):
                bridge_fov_initialized_workers.add(worker_id)
                return
            try:
                bridge_obj.set_camera_fov(
                    fov_deg=float(sim_camera_fov_deg),
                    vehicle_or_actor=binding.vehicle,
                )
            except Exception:
                pass
            bridge_fov_initialized_workers.add(worker_id)

        with bridge_cache_lock:
            cached = bridge_cache.get(binding.worker_id)
            if cached is not None:
                _init_worker_bridge_fov(binding.worker_id, cached)
                return cached
            size_override = view_image_size if (rgb_source == "sim" and use_square_capture_in_airsim) else None
            bridge_cfg = _build_bridge_config(
                config=config,
                vehicle_name=binding.vehicle,
                image_size_override=size_override,
                fov_override=sim_camera_fov_deg,
            )
            if engine_name == "airsim":
                if runtime_port is not None:
                    bridge_cfg["sim_port"] = int(runtime_port)
                bridge_cfg["launch_sim"] = False
                bridge_cfg["connect_on_init"] = True
                bridge_cfg["auto_select_port_on_conflict"] = False
                bridge_cfg["strict_vehicle_name"] = True
                bridge_cfg["vehicle_names"] = list(worker_vehicle_names)
            bridge_obj = create_bridge(engine=engine_name, scene_id=scene_id, config=bridge_cfg)
            bridge_cache[binding.worker_id] = bridge_obj
            bridge_locks.setdefault(binding.worker_id, threading.Lock())
            _init_worker_bridge_fov(binding.worker_id, bridge_obj)
            return bridge_obj

    def _get_vehicle_capture_lock(binding: WorkerBinding, bridge_obj: Any | None = None) -> threading.Lock:
        requested_key = normalize_airsim_vehicle_name(binding.vehicle, binding.worker_id)
        resolved_vehicle_name = ""
        if bridge_obj is not None:
            try:
                resolved_vehicle_name = str(getattr(bridge_obj, "vehicle_name", "") or "").strip()
            except Exception:
                resolved_vehicle_name = ""
        if resolved_vehicle_name:
            vehicle_key = normalize_airsim_vehicle_name(resolved_vehicle_name, binding.worker_id)
        else:
            vehicle_key = requested_key
            warn_key = f"{binding.worker_id}:{requested_key}->{vehicle_key}"
            if warn_key not in fallback_vehicle_lock_warnings:
                fallback_vehicle_lock_warnings.add(warn_key)
                logger.warn(
                    f"vehicle_fallback_lock worker={binding.worker_id} requested={requested_key} lock_key={vehicle_key}"
                )
        with bridge_cache_lock:
            lock = vehicle_capture_locks.get(vehicle_key)
            if lock is None:
                lock = threading.Lock()
                vehicle_capture_locks[vehicle_key] = lock
            return lock

    def _process_pair(idx: int, pair_value: np.ndarray) -> dict[str, Any]:
        t_inst = time.time()
        thread_name = threading.current_thread().name
        mask = inverse == idx
        points = xyz[mask]
        if points.shape[0] < min_points:
            return {"status": "dropped_small"}

        class_id = int(pair_value[0])
        instance_id = int(pair_value[1])
        center = np.mean(points, axis=0)
        pmin = np.min(points, axis=0)
        pmax = np.max(points, axis=0)
        size = pmax - pmin
        yaw_deg = 0.0
        if points.shape[0] >= 3:
            xy = points[:, :2] - np.mean(points[:, :2], axis=0, keepdims=True)
            cov = xy.T @ xy / max(1, xy.shape[0])
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            major = eig_vecs[:, int(np.argmax(eig_vals))]
            yaw_deg = float(math.degrees(math.atan2(float(major[1]), float(major[0]))))

        instance_key = f"{class_id}_{instance_id}"
        pcd_instance_dir = pcd_views_root / instance_key

        d_box = float(np.linalg.norm(size))
        context_radius_xy = max(preview_context_radius_min_m, d_box * preview_context_radius_scale)
        context_height = max(preview_context_height_min_m, float(size[2]) * preview_context_height_scale)
        context_mask = (
            (np.abs(preview_bg_points[:, 0] - float(center[0])) <= context_radius_xy)
            & (np.abs(preview_bg_points[:, 1] - float(center[1])) <= context_radius_xy)
            & (np.abs(preview_bg_points[:, 2] - float(center[2])) <= context_height)
        )
        context_points = preview_bg_points[context_mask]
        context_point_instance_ids = preview_bg_instance_ids[context_mask]
        if context_points.shape[0] == 0:
            context_points = points
            context_point_instance_ids = np.full((points.shape[0],), instance_id, dtype=np.int64)

        if idx < 5 or (idx + 1) % collect_log_every == 0:
            logger.info(
                f"task_start idx={idx + 1}/{int(unique_pair.shape[0])} instance={instance_key} "
                f"points={int(points.shape[0])} thread={thread_name}"
            )

        instance_view_specs = _select_view_specs_for_instance(
            target_points_xyz=points,
            context_points_xyz=context_points,
            context_instance_ids=context_point_instance_ids,
            target_instance_id=instance_id,
            bbox_yaw_deg=yaw_deg,
            pose_params=camera_pose_params,
            camera_fov_deg=project_camera_fov_deg,
            image_size=view_image_size,
            side_view_count=side_view_count,
            side_view_min_keep=side_view_min_keep,
            add_birdseye_view=add_birdseye_view,
            min_visible_ratio=side_view_min_visible_ratio,
            min_visible_points=side_view_min_visible_points,
            occlusion_neighbor_px=side_view_occlusion_neighbor_px,
            occlusion_depth_margin_m=side_view_occlusion_depth_margin_m,
            occlusion_search_radius_scale=side_view_occlusion_search_radius_scale,
            occlusion_search_radius_min_m=side_view_occlusion_search_radius_min_m,
            side_view_pitch_offsets_deg=side_view_pitch_offsets_deg,
        )
        if not instance_view_specs:
            return {"status": "dropped_view_insufficient", "instance_key": instance_key}

        def _pcd_task() -> list[dict[str, Any]]:
            return _render_pointcloud_views(
                target_points_xyz=points,
                context_points_xyz=context_points,
                context_instance_ids=context_point_instance_ids,
                out_dir=pcd_instance_dir,
                view_specs=instance_view_specs,
                image_size=view_image_size,
                pose_params=camera_pose_params,
                camera_fov_deg=project_camera_fov_deg,
            )

        def _rgb_task() -> list[dict[str, Any]]:
            if not enable_rgb_views:
                return []
            rgb_instance_dir = rgb_views_root / instance_key
            if rgb_source == "sim":
                binding = worker_bindings[idx % len(worker_bindings)]
                shared_bridge = _get_shared_bridge(binding)
                lock = bridge_locks.setdefault(binding.worker_id, threading.Lock())
                vehicle_lock = _get_vehicle_capture_lock(binding, shared_bridge)
                with vehicle_lock, lock:
                    return _capture_rgb_views_from_sim(
                        scene_id=scene_id,
                        config=config,
                        worker_binding=binding,
                        target_points_xyz=points,
                        view_specs=instance_view_specs,
                        output_width=view_image_width,
                        output_height=view_image_height,
                        out_dir=rgb_instance_dir,
                        pose_settle_sec=rgb_pose_settle_sec,
                        sim_camera_fov_deg=sim_camera_fov_deg,
                        project_camera_fov_deg=project_camera_fov_deg,
                        square_crop=rgb_square_crop,
                        draw_bbox_on_image=rgb_draw_bbox_on_image,
                        pose_params=camera_pose_params,
                        bridge=shared_bridge,
                    )
            return _render_rgb_views(
                target_points_xyz=points,
                context_points_xyz=context_points,
                out_dir=rgb_instance_dir,
                view_specs=instance_view_specs,
                image_size=view_image_size,
                pose_params=camera_pose_params,
                camera_fov_deg=project_camera_fov_deg,
                draw_bbox_on_image=rgb_draw_bbox_on_image,
            )

        pcd_views = _pcd_task()
        rgb_future: concurrent.futures.Future[list[dict[str, Any]]] | None = None
        if enable_rgb_views:
            if rgb_executor is None:
                raise RuntimeError("rgb executor is not initialized")
            rgb_future = rgb_executor.submit(_rgb_task)
        rgb_views: list[dict[str, Any]] = []

        pcd_views = sorted(
            list(pcd_views),
            key=lambda item: int(item.get("view_id", 10**9)) if isinstance(item, dict) else 10**9,
        )
        for view_meta in pcd_views:
            view_path = str(view_meta.get("path", ""))
            view_meta["path"] = f"pcd_views/{instance_key}/{view_path}"

        image_rel = ""

        if idx < 5 or (idx + 1) % collect_log_every == 0:
            logger.info(
                f"task_done_pcd idx={idx + 1}/{int(unique_pair.shape[0])} instance={instance_key} "
                f"pcd_views={len(pcd_views)} rgb_queued={bool(rgb_future is not None)} elapsed={time.time() - t_inst:.2f}s"
            )

        item = {
            "instance_id": instance_key,
            "class_id": class_id,
            "class_name": "",
            "instance_numeric_id": instance_id,
            "center_3d": [float(center[0]), float(center[1]), float(center[2])],
            "bbox_3d": {
                "min": [float(pmin[0]), float(pmin[1]), float(pmin[2])],
                "max": [float(pmax[0]), float(pmax[1]), float(pmax[2])],
                "size": [float(size[0]), float(size[1]), float(size[2])],
                "yaw_deg": yaw_deg,
            },
            "point_count": int(points.shape[0]),
            "preview_image": image_rel,
            "pcd_views": pcd_views,
            "rgb_views": rgb_views,
        }
        return {
            "status": "kept",
            "item": item,
            "image_index": {"instance_id": instance_key, "path": image_rel},
            "has_rgb": bool(len(rgb_views) > 0),
            "instance_key": instance_key,
            "rgb_future": rgb_future,
        }
    futures: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    rgb_futures: dict[concurrent.futures.Future[list[dict[str, Any]]], dict[str, Any]] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=collect_parallel_workers) as executor:
            rgb_workers = max(1, int(stage2_cfg.get("collect_rgb_parallel_workers", collect_parallel_workers) or collect_parallel_workers))
            rgb_workers = max(1, min(64, rgb_workers))
            rgb_executor = concurrent.futures.ThreadPoolExecutor(max_workers=rgb_workers)
            for idx, pair_value in enumerate(unique_pair):
                fut = executor.submit(_process_pair, idx, pair_value)
                futures[fut] = idx

            processed = 0
            total_pairs = int(unique_pair.shape[0])
            for fut in concurrent.futures.as_completed(futures):
                processed += 1
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.warn(f"render_failed idx={futures[fut]} error={exc}")
                    dropped_small += 1
                    progress.advance(detail=f"kept={len(instances)} dropped={dropped_small}")
                    continue

                status = str(result.get("status", ""))
                if status == "dropped_small":
                    dropped_small += 1
                    progress.advance(detail=f"kept={len(instances)} dropped={dropped_small}")
                    if rgb_progress is not None:
                        rgb_progress.advance(detail=f"skip_small={dropped_small}")
                elif status == "dropped_view_insufficient":
                    dropped_view_insufficient += 1
                    progress.advance(
                        detail=(
                            f"kept={len(instances)} dropped={dropped_small + dropped_view_insufficient} "
                            f"dropped_view={dropped_view_insufficient}"
                        )
                    )
                    if rgb_progress is not None:
                        rgb_progress.advance(detail=f"skip_view={dropped_view_insufficient}")
                elif status == "kept":
                    item_ref = dict(result["item"])
                    instances.append(item_ref)
                    image_index.append(dict(result["image_index"]))
                    rendered_pointcloud += 1
                    rgb_future_obj = result.get("rgb_future", None)
                    if isinstance(rgb_future_obj, concurrent.futures.Future):
                        rgb_futures[rgb_future_obj] = {
                            "item": item_ref,
                            "instance_key": str(result.get("instance_key", "")),
                        }
                    elif bool(result.get("has_rgb", False)):
                        rendered_rgb_instances += 1
                        if rgb_progress is not None:
                            rgb_progress.advance(detail=f"rgb_done={rendered_rgb_instances} pending={len(rgb_futures)}")
                    instance_key = str(result.get("instance_key", ""))
                    progress.advance(
                        detail=(
                            f"kept={len(instances)} dropped={dropped_small} "
                            f"pcd={rendered_pointcloud} rgb={rendered_rgb_instances} pending_rgb={len(rgb_futures)} last={instance_key}"
                        )
                    )

                if processed % collect_log_every == 0 or processed == total_pairs:
                    elapsed = max(1e-6, time.time() - t_start)
                    speed = processed / elapsed
                    remain = max(0, total_pairs - processed)
                    eta_sec = remain / max(1e-6, speed)
                    logger.info(
                        f"progress={processed}/{total_pairs} kept={len(instances)} dropped={dropped_small} "
                        f"pcd={rendered_pointcloud} rgb={rendered_rgb_instances} pending_rgb={len(rgb_futures)} speed={speed:.2f} pair/s eta={eta_sec:.1f}s"
                    )

            if rgb_futures:
                for rgb_fut in concurrent.futures.as_completed(list(rgb_futures.keys())):
                    info = rgb_futures[rgb_fut]
                    item_ref = info["item"]
                    instance_key = str(info.get("instance_key", ""))
                    try:
                        rgb_views = rgb_fut.result()
                    except Exception as exc:
                        logger.warn(f"rgb_render_failed instance={instance_key} error={exc}")
                        rgb_views = []
                    rgb_views = sorted(
                        list(rgb_views),
                        key=lambda item: int(item.get("view_id", 10**9)) if isinstance(item, dict) else 10**9,
                    )
                    for view_meta in rgb_views:
                        view_path = str(view_meta.get("path", ""))
                        view_meta["path"] = f"rgb_views/{instance_key}/{view_path}"
                    item_ref["rgb_views"] = rgb_views
                    if rgb_views:
                        rendered_rgb_instances += 1
                    if rgb_progress is not None:
                        rgb_progress.advance(detail=f"rgb_done={rendered_rgb_instances} pending={max(0, len(rgb_futures)-1)}")
                rgb_futures.clear()
    finally:
        try:
            if rgb_executor is not None:
                rgb_executor.shutdown(wait=True)
        except Exception:
            pass
        for bridge_obj in list(bridge_cache.values()):
            try:
                bridge_obj.shutdown()
            except Exception:
                pass
        if bootstrap_bridge is not None:
            try:
                bootstrap_bridge.shutdown()
            except Exception:
                pass

    progress.finish(detail=f"done kept={len(instances)} dropped={dropped_small + dropped_view_insufficient}")
    if rgb_progress is not None:
        rgb_progress.finish(detail=f"done rgb={rendered_rgb_instances}")
    logger.info(
        f"phase=collect_render done rendered_pointcloud={rendered_pointcloud} rendered_rgb_instances={rendered_rgb_instances} "
        f"dropped_small={dropped_small} dropped_view_insufficient={dropped_view_insufficient}"
    )

    instances.sort(key=lambda v: v["point_count"], reverse=True)
    if max_instances > 0 and len(instances) > max_instances:
        instances = instances[:max_instances]
        kept_ids = {it["instance_id"] for it in instances}
        image_index = [it for it in image_index if str(it.get("instance_id", "")) in kept_ids]

    instances_json_path = resolve_scene_artifact_path(landmarks_raw_root, scene_id, ".instances.json")
    bundle_json_path = resolve_scene_artifact_path(landmarks_raw_root, scene_id, ".landmarks_bundle.json")

    instances_payload = {
        "scene_id": scene_id,
        "source_pcd": str(input_pcd.as_posix()),
        "coordinate": "ENU",
        "unit": "meter",
        "filters": {
            "collect_min_points": min_points,
            "collect_exclude_class_ids": sorted(list(exclude_class_ids)),
            "collect_exclude_nonpositive_instance": exclude_nonpositive_instance,
            "collect_enable_rgb_views": enable_rgb_views,
            "collect_rgb_views_count": rgb_views_count,
            "collect_side_view_min_keep": side_view_min_keep,
            "collect_add_birdseye_view": add_birdseye_view,
            "collect_view_image_size": view_image_size,
            "collect_rgb_image_size": view_image_size,
            "collect_view_image_width": view_image_width,
            "collect_view_image_height": view_image_height,
            "collect_rgb_source": rgb_source,
            "collect_rgb_pose_settle_sec": rgb_pose_settle_sec,
            "collect_rgb_parallel_workers": int(stage2_cfg.get("collect_rgb_parallel_workers", collect_parallel_workers) or collect_parallel_workers),
            "collect_rgb_square_crop": rgb_square_crop,
            "collect_rgb_draw_bbox_on_image": rgb_draw_bbox_on_image,
            "collect_use_square_capture_in_airsim": use_square_capture_in_airsim,
            "collect_side_view_min_visible_ratio": side_view_min_visible_ratio,
            "collect_side_view_min_visible_points": side_view_min_visible_points,
            "collect_side_view_occlusion_neighbor_px": side_view_occlusion_neighbor_px,
            "collect_side_view_occlusion_depth_margin_m": side_view_occlusion_depth_margin_m,
            "collect_side_view_occlusion_search_radius_scale": side_view_occlusion_search_radius_scale,
            "collect_side_view_occlusion_search_radius_min_m": side_view_occlusion_search_radius_min_m,
            "collect_side_view_pitch_offsets_deg": [float(v) for v in side_view_pitch_offsets_deg],
            "collect_parallel_workers": collect_parallel_workers,
            "collect_progress_log_every": collect_log_every,
            "collect_pair_limit": collect_pair_limit,
            "collect_camera_pose": camera_pose_params,
            "pcd_ground_percentile": ground_percentile,
            "pcd_ground_margin_m": ground_margin_m,
            "pcd_ground_z_threshold": ground_z_threshold,
            "pcd_max_instances": max_instances,
        },
        "summary": {
            "input_points": int(arr.shape[0]),
            "valid_points": int(xyz.shape[0]),
            "candidate_pairs": int(unique_pair.shape[0]),
            "dropped_small_instances": int(dropped_small),
            "dropped_view_insufficient": int(dropped_view_insufficient),
            "instances": int(len(instances)),
            "runtime_sec": float(time.time() - t_start),
        },
        "instances": instances,
    }

    bundle_payload = {
        "scene_id": scene_id,
        "instances_json": str(instances_json_path.as_posix()),
        "image_index": image_index,
        "instances": instances,
    }

    write_json(instances_json_path, instances_payload)
    write_json(bundle_json_path, bundle_payload)

    runtime_sec = max(1e-6, float(time.time() - t_start))
    out = {
        "ok": True,
        "mode": "collect_instances",
        "scene_id": scene_id,
        "source_pcd": str(input_pcd.as_posix()),
        "instances_json": str(instances_json_path.as_posix()),
        "bundle_json": str(bundle_json_path.as_posix()),
        "instances": len(instances),
        "dropped_small_instances": dropped_small,
        "dropped_view_insufficient": dropped_view_insufficient,
        "min_points": min_points,
        "runtime_sec": runtime_sec,
        "instances_per_sec": float(len(instances) / runtime_sec),
        "stats": {
            "valid_points": int(xyz.shape[0]),
            "candidate_pairs": int(unique_pair.shape[0]),
            "preview_bg_points": int(preview_bg_points.shape[0]),
            "rendered_pointcloud": int(rendered_pointcloud),
            "rendered_rgb_instances": int(rendered_rgb_instances),
            "view_spec": {
                "pointcloud": {
                    "image_width": int(view_image_size),
                    "image_height": int(view_image_size),
                    "image_size": int(view_image_size),
                    "format": "png",
                    "bbox_overlay_baked": True,
                },
                "rgb": {
                    "source": rgb_source,
                    "image_width": int(view_image_width),
                    "image_height": int(view_image_height),
                    "image_size": int(view_image_size),
                    "format": "png",
                    "bbox_overlay_baked": bool(rgb_draw_bbox_on_image),
                    "bbox_json_fields": [
                        "bbox_2d_xyxy",
                        "bbox_2d_xywh",
                        "bbox_2d_image_size",
                        "bbox_2d_valid",
                    ],
                },
            },
        },
    }
    logger.info(
        f"done instances={len(instances)} dropped={dropped_small} "
        f"runtime={runtime_sec:.1f}s speed={len(instances)/runtime_sec:.2f} inst/s"
    )
    return out


def _resolve_review_decision(item: Any, default_action: str) -> tuple[str, str, str]:
    action = default_action
    note = ""
    source = "default"
    if isinstance(item, str):
        action = item.strip().lower()
        source = "decision_file"
    elif isinstance(item, dict):
        action = str(item.get("action", default_action)).strip().lower()
        note = str(item.get("note", "") or "")
        source = "decision_file"
    if action not in {"keep", "drop"}:
        action = default_action
    return action, note, source


def _build_review_outputs(
    scene_id: str,
    instances: list[dict[str, Any]],
    decision_map_raw: dict[str, Any],
    default_action: str,
    instance_overrides: dict[str, dict[str, Any]] | None = None,
    stage2_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid_instances: list[dict[str, Any]] = []
    dropped_instances: list[dict[str, Any]] = []
    log_buffer: list[dict[str, Any]] = []

    for item in instances:
        if not isinstance(item, dict):
            continue
        instance_id = str(item.get("instance_id", ""))
        decision_obj = decision_map_raw.get(instance_id, None)
        if decision_obj is None:
            action = "undecided"
            note = ""
            source = "none"
        else:
            action, note, source = _resolve_review_decision(decision_obj, default_action)

        reviewed = dict(item)
        if instance_overrides and instance_id in instance_overrides and isinstance(instance_overrides[instance_id], dict):
            reviewed.update(instance_overrides[instance_id])
        reviewed = _normalize_annotation_payload(reviewed, stage2_cfg=stage2_cfg)
        reviewed["review_action"] = action
        if note:
            reviewed["review_note"] = note

        if action == "keep" and str(reviewed.get("annotation_status", "") or "").strip().lower() == "labeled":
            valid_instances.append(reviewed)
        elif action == "drop":
            dropped_instances.append(reviewed)

        log_buffer.append(
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "instance_reviewed",
                "instance_id": instance_id,
                "action": action,
                "source": source,
                "note": note,
            }
        )

    return valid_instances, dropped_instances, log_buffer


def review_instances(scene_id: str, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    logger = StageLogger("stage2.review_instances")
    t_start = time.time()

    stage2_cfg = config.get("stage2", {}) or {}
    scene_root = _resolve_scene_root(config=config, scene_id=scene_id)
    raw_root = scene_root / _resolve_output_dir_name(config, key="stage2_raw_dir", default="landmarks_raw")
    review_root = scene_root / _resolve_output_dir_name(config, key="stage2_review_dir", default="landmarks_review")
    ensure_dir(review_root)

    instances_json_path = resolve_scene_artifact_path(raw_root, scene_id, ".instances.json")
    valid_instances_path = resolve_scene_artifact_path(review_root, scene_id, ".valid_instances.json")
    review_log_path = resolve_scene_artifact_path(review_root, scene_id, ".review_log.jsonl")

    raw_payload = read_json_if_exists(instances_json_path, default={})
    if not isinstance(raw_payload, dict):
        raise ValueError(f"invalid instances json: {instances_json_path}")
    instances = list(raw_payload.get("instances", []) or [])
    if len(instances) == 0:
        raise FileNotFoundError(
            f"no instances found for review: {instances_json_path}. Run --mode collect_instances first."
        )

    decisions_path = Path(str(args.review_decisions)) if args.review_decisions else None
    decisions_payload = {}
    if decisions_path is None:
        cfg_decision = stage2_cfg.get("review_decisions_json", None)
        if cfg_decision:
            decisions_path = Path(str(cfg_decision))
    if decisions_path is not None:
        decisions_payload = read_json_if_exists(decisions_path, default={})
        if not isinstance(decisions_payload, dict):
            decisions_payload = {}

    decision_map_raw = decisions_payload.get("instance_decisions", decisions_payload)
    if not isinstance(decision_map_raw, dict):
        decision_map_raw = {}

    default_action = str(stage2_cfg.get("review_default_action", "keep")).strip().lower()
    if default_action not in {"keep", "drop"}:
        default_action = "keep"

    valid_instances, dropped_instances, log_buffer = _build_review_outputs(
        scene_id=scene_id,
        instances=instances,
        decision_map_raw=decision_map_raw,
        default_action=default_action,
        stage2_cfg=stage2_cfg,
    )

    payload = {
        "scene_id": scene_id,
        "source_instances_json": str(instances_json_path.as_posix()),
        "decision_file": str(decisions_path.as_posix()) if decisions_path else None,
        "summary": {
            "instances_total": int(len(instances)),
            "valid_instances": int(len(valid_instances)),
            "dropped_instances": int(len(dropped_instances)),
            "default_action": default_action,
            "runtime_sec": float(time.time() - t_start),
        },
        "valid_instances": valid_instances,
    }
    write_json(valid_instances_path, payload)

    append_jsonl(
        review_log_path,
        {
            "ts": time.time(),
            "scene_id": scene_id,
            "event": "review_session_start",
            "instances_total": len(instances),
            "default_action": default_action,
            "decision_file": str(decisions_path.as_posix()) if decisions_path else None,
        },
    )
    for event in log_buffer:
        append_jsonl(review_log_path, event)
    append_jsonl(
        review_log_path,
        {
            "ts": time.time(),
            "scene_id": scene_id,
            "event": "review_session_end",
            "valid_instances": len(valid_instances),
            "dropped_instances": len(dropped_instances),
            "runtime_sec": float(time.time() - t_start),
        },
    )

    runtime_sec = max(1e-6, float(time.time() - t_start))
    logger.info(
        f"done total={len(instances)} valid={len(valid_instances)} dropped={len(dropped_instances)} "
        f"runtime={runtime_sec:.1f}s"
    )
    return {
        "ok": True,
        "mode": "review_instances",
        "scene_id": scene_id,
        "source_instances_json": str(instances_json_path.as_posix()),
        "valid_instances_json": str(valid_instances_path.as_posix()),
        "review_log_jsonl": str(review_log_path.as_posix()),
        "instances_total": int(len(instances)),
        "valid_instances": int(len(valid_instances)),
        "dropped_instances": int(len(dropped_instances)),
        "runtime_sec": runtime_sec,
    }


def review_instances_web(scene_id: str, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if Flask is None:
        raise ImportError("Flask is required for review_instances_web mode")

    logger = StageLogger("stage2.review_instances_web")
    stage2_cfg = dict(config.get("stage2", {}) or {})
    stage2_cfg["_full_config"] = config
    scene_root = _resolve_scene_root(config=config, scene_id=scene_id)
    raw_root = scene_root / _resolve_output_dir_name(config, key="stage2_raw_dir", default="landmarks_raw")
    raw_root_candidates = [raw_root.resolve()]
    review_root = scene_root / _resolve_output_dir_name(config, key="stage2_review_dir", default="landmarks_review")
    ensure_dir(review_root)
    stage2_cfg["_scene_id"] = scene_id
    stage2_cfg["_scene_root"] = str(scene_root)
    stage2_cfg["_review_root"] = str(review_root)

    instances_json_path = resolve_scene_artifact_path(raw_root, scene_id, ".instances.json")
    images_root = raw_root / "images"
    valid_instances_path = resolve_scene_artifact_path(review_root, scene_id, ".valid_instances.json")
    review_log_path = resolve_scene_artifact_path(review_root, scene_id, ".review_log.jsonl")
    web_state_path = resolve_scene_artifact_path(review_root, scene_id, ".web_review_state.json")

    raw_payload = read_json_if_exists(instances_json_path, default={})
    if not isinstance(raw_payload, dict):
        raise ValueError(f"invalid instances json: {instances_json_path}")
    instances = list(raw_payload.get("instances", []) or [])
    if len(instances) == 0:
        raise FileNotFoundError(
            f"no instances found for review: {instances_json_path}. Run --mode collect_instances first."
        )

    default_action = str(stage2_cfg.get("review_default_action", "keep")).strip().lower()
    if default_action not in {"keep", "drop"}:
        default_action = "keep"

    decisions_path = Path(str(args.review_decisions)) if args.review_decisions else None
    if decisions_path is None:
        cfg_decision = stage2_cfg.get("review_decisions_json", None)
        if cfg_decision:
            decisions_path = Path(str(cfg_decision))

    decision_map_raw: dict[str, Any] = {}
    if decisions_path is not None:
        decisions_payload = read_json_if_exists(decisions_path, default={})
        if isinstance(decisions_payload, dict):
            decision_map_raw = decisions_payload.get("instance_decisions", decisions_payload)
            if not isinstance(decision_map_raw, dict):
                decision_map_raw = {}

    state_payload = read_json_if_exists(web_state_path, default={})
    instance_overrides: dict[str, dict[str, Any]] = {}
    current_index = -1
    if isinstance(state_payload, dict):
        state_decisions = state_payload.get("instance_decisions", {})
        if isinstance(state_decisions, dict):
            decision_map_raw.update(state_decisions)
        state_overrides = state_payload.get("instance_overrides", {})
        if isinstance(state_overrides, dict):
            instance_overrides = {
                str(k): dict(v)
                for k, v in state_overrides.items()
                if isinstance(v, dict)
            }
        try:
            current_index = int(state_payload.get("current_index", -1))
        except Exception:
            current_index = -1

    if current_index >= 0:
        current_index = max(0, min(len(instances) - 1, current_index))
    elif len(instances) > 0:
        current_index = 0

    auto_label_keys = [
        "auto_label_category",
        "auto_label_subcategory",
        "auto_label_description",
        "auto_label_name",
        "auto_label_confidence",
        "auto_label_landmark_type",
        "auto_label_landmark_description",
        "auto_label_reason",
        "auto_label_views",
        "auto_label_comment",
        "annotation_status",
        "landmark_category",
        "landmark_subcategory",
        "landmark_description",
        "landmark_decision",
        "landmark_note",
    ]
    state_lock = threading.RLock()
    auto_label_job: dict[str, Any] = {
        "job_id": 0,
        "running": False,
        "cancel_requested": False,
        "scope": None,
        "class_id": None,
        "class_name": None,
        "total": 0,
        "processed": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        "current_instance_id": None,
        "message": "Ready",
        "last_error": None,
        "started_at": None,
        "finished_at": None,
        "thread": None,
    }

    def _with_auto_placeholders(item: dict[str, Any]) -> dict[str, Any]:
        out = _normalize_annotation_payload(item, stage2_cfg=stage2_cfg)
        for key in auto_label_keys:
            out.setdefault(key, None)
        out.setdefault("review_note", "")
        return out

    def _sync_annotation_status_override(instance_id: str) -> bool:
        base_item = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(base_item, dict):
            return False
        effective = dict(base_item)
        override = instance_overrides.get(instance_id, None)
        if isinstance(override, dict):
            effective.update(override)
        normalized = _normalize_annotation_payload(effective, stage2_cfg=stage2_cfg)

        changed = False
        for key in ["annotation_status", "auto_label_confidence"]:
            normalized_value = normalized.get(key, None)
            override_value = override.get(key, None) if isinstance(override, dict) else None
            effective_value = effective.get(key, None)
            if normalized_value != effective_value or override_value != normalized_value:
                if instance_id not in instance_overrides:
                    instance_overrides[instance_id] = {}
                instance_overrides[instance_id][key] = normalized_value
                changed = True
        return changed

    def _refresh_annotation_status_overrides() -> int:
        changed = 0
        for item in instances:
            if not isinstance(item, dict):
                continue
            if _sync_annotation_status_override(str(item.get("instance_id", ""))):
                changed += 1
        return changed

    def _export_auto_label_task() -> dict[str, Any]:
        with state_lock:
            return {
                "job_id": int(auto_label_job.get("job_id", 0) or 0),
                "running": bool(auto_label_job.get("running", False)),
                "cancel_requested": bool(auto_label_job.get("cancel_requested", False)),
                "scope": auto_label_job.get("scope", None),
                "class_id": auto_label_job.get("class_id", None),
                "class_name": auto_label_job.get("class_name", None),
                "total": int(auto_label_job.get("total", 0) or 0),
                "processed": int(auto_label_job.get("processed", 0) or 0),
                "updated": int(auto_label_job.get("updated", 0) or 0),
                "skipped": int(auto_label_job.get("skipped", 0) or 0),
                "failed": int(auto_label_job.get("failed", 0) or 0),
                "current_instance_id": auto_label_job.get("current_instance_id", None),
                "message": str(auto_label_job.get("message", "") or ""),
                "last_error": auto_label_job.get("last_error", None),
                "started_at": auto_label_job.get("started_at", None),
                "finished_at": auto_label_job.get("finished_at", None),
            }

    def _calc_counts() -> dict[str, int]:
        keep_count = 0
        drop_count = 0
        undecided_count = 0
        for item in instances:
            instance_id = str(item.get("instance_id", ""))
            decision_obj = decision_map_raw.get(instance_id, None)
            if decision_obj is None:
                undecided_count += 1
                continue
            action, _, _ = _resolve_review_decision(decision_obj, default_action)
            if action == "keep":
                keep_count += 1
            else:
                drop_count += 1
        return {
            "total": int(len(instances)),
            "keep": int(keep_count),
            "drop": int(drop_count),
            "undecided": int(undecided_count),
        }

    def _write_snapshot(trigger: str) -> tuple[int, int]:
        with state_lock:
            _refresh_annotation_status_overrides()
            valid_instances, dropped_instances, _ = _build_review_outputs(
                scene_id=scene_id,
                instances=instances,
                decision_map_raw=decision_map_raw,
                default_action=default_action,
                instance_overrides=instance_overrides,
                stage2_cfg=stage2_cfg,
            )
            payload = {
                "scene_id": scene_id,
                "source_instances_json": str(instances_json_path.as_posix()),
                "decision_file": str(decisions_path.as_posix()) if decisions_path else None,
                "summary": {
                    "instances_total": int(len(instances)),
                    "valid_instances": int(len(valid_instances)),
                    "dropped_instances": int(len(dropped_instances)),
                    "default_action": default_action,
                    "last_trigger": trigger,
                    "updated_at": float(time.time()),
                },
                "valid_instances": valid_instances,
            }
            write_json(valid_instances_path, payload)
            write_json(
                web_state_path,
                {
                    "scene_id": scene_id,
                    "current_index": int(current_index),
                    "updated_at": float(time.time()),
                    "instance_decisions": decision_map_raw,
                    "instance_overrides": instance_overrides,
                },
            )
            return int(len(valid_instances)), int(len(dropped_instances))

    def _view_caption_from_meta(view: dict[str, Any], index: int) -> str:
        label = str(view.get("label", "") or "").strip().lower()
        yaw = float(view.get("yaw_deg", 0.0) or 0.0)
        pitch = float(view.get("pitch_deg", 0.0) or 0.0)
        yaw_txt = f"{yaw:+.0f}°"
        pitch_txt = f", top-down {max(0.0, -pitch):.0f}°"
        if label == "topdown":
            return f"Top-down ({yaw_txt})"
        return f"Side View {index + 1} ({yaw_txt}{pitch_txt})"

    def _get_effective_item_by_instance_id(instance_id: str) -> dict[str, Any] | None:
        base_item = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(base_item, dict):
            return None
        out = dict(base_item)
        if instance_id in instance_overrides and isinstance(instance_overrides[instance_id], dict):
            out.update(instance_overrides[instance_id])
        return _with_auto_placeholders(out)

    def _build_item(index: int) -> dict[str, Any]:
        base_item = _get_effective_item_by_instance_id(str(instances[index].get("instance_id", "")))
        if not isinstance(base_item, dict):
            base_item = _with_auto_placeholders(dict(instances[index]))
        item = dict(base_item)
        instance_id = str(item.get("instance_id", ""))
        decision_obj = decision_map_raw.get(instance_id, None)
        action, note, _ = _resolve_review_decision(decision_obj, default_action)
        item["review_action"] = action
        item["review_note"] = note
        preview_image = str(item.get("preview_image", ""))
        pcd_urls: list[str] = []
        pcd_captions: list[str] = []
        pcd_views = sorted(
            list(item.get("pcd_views", []) or []),
            key=lambda entry: int(entry.get("view_id", 10**9)) if isinstance(entry, dict) else 10**9,
        )
        for i, view in enumerate(pcd_views):
            if not isinstance(view, dict):
                continue
            p = str(view.get("path", "") or "")
            if p:
                pcd_urls.append(f"/raw/{p.lstrip('/')}")
                pcd_captions.append(_view_caption_from_meta(view, i))
        if not pcd_urls and preview_image:
            pcd_urls = [f"/raw/{preview_image.lstrip('/')}"]
            pcd_captions = ["Preview"]

        rgb_urls: list[str] = []
        rgb_captions: list[str] = []
        rgb_views_meta: list[dict[str, Any]] = []
        rgb_views = sorted(
            list(item.get("rgb_views", []) or []),
            key=lambda entry: int(entry.get("view_id", 10**9)) if isinstance(entry, dict) else 10**9,
        )
        include_topdown_slot = bool(stage2_cfg.get("collect_add_birdseye_view", False))
        for i, view in enumerate(rgb_views):
            if not isinstance(view, dict):
                continue
            p = str(view.get("path", "") or "")
            view_direction = _normalize_view_direction(view.get("view_direction", view.get("label", None)))
            view_mode = str(view.get("mode", "") or "").strip().lower()
            is_topdown = view_mode == "topdown" or str(view.get("label", "") or "").strip().lower() == "topdown"
            is_valid = bool(view.get("is_valid", True)) and bool(p)
            if is_topdown:
                include_topdown_slot = True
            if p:
                rgb_urls.append(f"/raw/{p.lstrip('/')}")
                rgb_captions.append(_view_caption_from_meta(view, i))
            rgb_views_meta.append(
                {
                    "view_index": int(i),
                    "url": f"/raw/{p.lstrip('/')}" if p else "",
                    "caption": _view_caption_from_meta(view, i),
                    "view_direction": view_direction,
                    "is_query_view": bool(view.get("is_query_view", False)),
                    "is_valid": bool(is_valid),
                    "is_occluded": not bool(p),
                    "mode": "topdown" if is_topdown else "orbit",
                    "bbox_2d_xyxy": view.get("bbox_2d_xyxy", None),
                    "bbox_2d_xywh": view.get("bbox_2d_xywh", None),
                    "bbox_2d_image_size": view.get("bbox_2d_image_size", None),
                    "bbox_2d_valid": bool(view.get("bbox_2d_valid", False)),
                }
            )

        rgb_slots: list[dict[str, Any]] = []
        used_indices: set[int] = set()
        for direction in VIEW_DIRECTION_RING:
            matched = next(
                (
                    v
                    for v in rgb_views_meta
                    if int(v.get("view_index", -1)) not in used_indices
                    and str(v.get("mode", "orbit")) != "topdown"
                    and _normalize_view_direction(v.get("view_direction", None)) == direction
                ),
                None,
            )
            if matched is not None:
                used_indices.add(int(matched.get("view_index", -1)))
                slot = dict(matched)
                slot["slot_key"] = direction
                slot["slot_label"] = direction
                rgb_slots.append(slot)
            else:
                rgb_slots.append(
                    {
                        "slot_key": direction,
                        "slot_label": direction,
                        "view_index": None,
                        "url": "",
                        "caption": f"{direction} (occluded)",
                        "view_direction": direction,
                        "is_query_view": False,
                        "is_valid": False,
                        "is_occluded": True,
                        "mode": "orbit",
                        "bbox_2d_xyxy": None,
                        "bbox_2d_xywh": None,
                        "bbox_2d_image_size": None,
                        "bbox_2d_valid": False,
                    }
                )

        if include_topdown_slot:
            topdown_view = next((v for v in rgb_views_meta if str(v.get("mode", "")) == "topdown"), None)
            if topdown_view is not None:
                slot = dict(topdown_view)
                slot["slot_key"] = "topdown"
                slot["slot_label"] = "topdown"
                rgb_slots.append(slot)
            else:
                rgb_slots.append(
                    {
                        "slot_key": "topdown",
                        "slot_label": "topdown",
                        "view_index": None,
                        "url": "",
                        "caption": "topdown (occluded)",
                        "view_direction": None,
                        "is_query_view": False,
                        "is_valid": False,
                        "is_occluded": True,
                        "mode": "topdown",
                        "bbox_2d_xyxy": None,
                        "bbox_2d_xywh": None,
                        "bbox_2d_image_size": None,
                        "bbox_2d_valid": False,
                    }
                )
        return {
            "index": int(index),
            "item": item,
            "pcd_urls": pcd_urls,
            "pcd_captions": pcd_captions,
            "pcd_reason": str(item.get("pcd_reason", "") or ""),
            "rgb_urls": rgb_urls,
            "rgb_captions": rgb_captions,
            "rgb_views": rgb_views_meta,
            "rgb_slots": rgb_slots,
            "rgb_reason": str(item.get("rgb_reason", "") or ""),
        }

    def _build_list_items() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, item in enumerate(instances):
            instance_id = str(item.get("instance_id", ""))
            effective_item = _get_effective_item_by_instance_id(instance_id)
            if not isinstance(effective_item, dict):
                effective_item = _with_auto_placeholders(dict(item))
            class_name = str(effective_item.get("class_name", ""))
            class_id = effective_item.get("class_id", "")
            annotation_status = str(effective_item.get("annotation_status", "") or "")
            auto_label_name = str(effective_item.get("auto_label_name", "") or "")
            auto_label_confidence = effective_item.get("auto_label_confidence", None)
            decision_obj = decision_map_raw.get(instance_id, None)
            if decision_obj is None:
                action = "undecided"
            else:
                action, _, _ = _resolve_review_decision(decision_obj, default_action)
            rows.append(
                {
                    "index": i,
                    "instance_id": instance_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "action": action,
                    "annotation_status": annotation_status,
                    "auto_label_name": auto_label_name,
                    "auto_label_confidence": auto_label_confidence,
                }
            )
        rows.sort(key=lambda x: (int(x["class_id"]) if str(x["class_id"]).isdigit() else x["class_id"], x["instance_id"]))
        return rows

    def _has_query_view(item: dict[str, Any], view_type: str = "rgb", require_valid: bool = True) -> bool:
        views = list(item.get(f"{view_type}_views", []) or [])
        for view in views:
            if not isinstance(view, dict):
                continue
            if not bool(view.get("is_query_view", False)):
                continue
            if not require_valid:
                return True
            path_text = str(view.get("path", "") or "").strip()
            is_valid = bool(view.get("is_valid", True))
            if path_text and is_valid:
                return True
        return False

    def _next_index_by_class_order(current_idx: int) -> int:
        ordered_indices = [int(r.get("index", -1)) for r in _build_list_items() if int(r.get("index", -1)) >= 0]
        if not ordered_indices:
            return current_idx
        try:
            pos = ordered_indices.index(int(current_idx))
        except ValueError:
            return ordered_indices[0]
        if pos + 1 < len(ordered_indices):
            return ordered_indices[pos + 1]
        return ordered_indices[-1]

    def _item_has_existing_auto_label(item: dict[str, Any]) -> bool:
        return _has_auto_label_payload(item)

    def _normalize_class_id_for_match(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() in {"none", "null", "nan"}:
            return None
        try:
            num = float(text)
            if math.isfinite(num) and abs(num - round(num)) < 1e-9:
                return str(int(round(num)))
        except Exception:
            pass
        return text

    def _class_id_matches(item: dict[str, Any], normalized_class_id: str | None) -> bool:
        if normalized_class_id is None:
            return False
        item_class_id = _normalize_class_id_for_match(item.get("class_id", None))
        return item_class_id == normalized_class_id

    def _prepare_auto_label_targets(scope: str, *, instance_id: str = "", class_id_raw: Any = None, class_name_raw: str = "") -> tuple[list[str], dict[str, int]]:
        targets: list[str] = []
        stats = {"eligible": 0, "skipped": 0, "skipped_not_keep": 0, "skipped_existing": 0}
        normalized_class_id = _normalize_class_id_for_match(class_id_raw)

        for item in instances:
            if not isinstance(item, dict):
                continue
            current_instance_id = str(item.get("instance_id", ""))
            if scope == "single" and current_instance_id != instance_id:
                continue

            effective_item = _get_effective_item_by_instance_id(current_instance_id)
            if not isinstance(effective_item, dict):
                continue

            if scope == "class":
                # Class-scope auto labeling must match by class_id only.
                if not _class_id_matches(effective_item, normalized_class_id):
                    continue

            decision_obj = decision_map_raw.get(current_instance_id, None)
            action, _, _ = _resolve_review_decision(decision_obj, default_action)
            if action != "keep":
                stats["skipped_not_keep"] += 1
                stats["skipped"] += 1
                continue

            if scope == "all" and _item_has_existing_auto_label(effective_item):
                stats["skipped_existing"] += 1
                stats["skipped"] += 1
                continue

            targets.append(current_instance_id)
            stats["eligible"] += 1

            if scope == "single":
                break

        return targets, stats

    def _set_auto_label_job_state(**kwargs: Any) -> None:
        with state_lock:
            auto_label_job.update(kwargs)

    def _finalize_auto_label_job(job_id: int, *, message: str, last_error: str | None = None) -> None:
        with state_lock:
            if int(auto_label_job.get("job_id", 0) or 0) != int(job_id):
                return
            auto_label_job["running"] = False
            auto_label_job["message"] = str(message or "")
            auto_label_job["last_error"] = last_error
            auto_label_job["current_instance_id"] = None
            auto_label_job["finished_at"] = float(time.time())
            auto_label_job["thread"] = None

    def _run_auto_label_job(job_id: int, scope: str, target_instance_ids: list[str], *, class_id_raw: Any = None, class_name_raw: str = "") -> None:
        updated = 0
        skipped = 0
        failed = 0
        total = len(target_instance_ids)

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_auto_label_job_start",
                "job_id": job_id,
                "scope": scope,
                "class_id": class_id_raw,
                "class_name": class_name_raw,
                "total": total,
            },
        )

        try:
            for idx, current_instance_id in enumerate(target_instance_ids, start=1):
                with state_lock:
                    if int(auto_label_job.get("job_id", 0) or 0) != int(job_id):
                        return
                    cancel_requested = bool(auto_label_job.get("cancel_requested", False))
                    auto_label_job["current_instance_id"] = current_instance_id
                    auto_label_job["message"] = f"Auto-labeling: {scope} {idx}/{total} {current_instance_id}"
                if cancel_requested:
                    append_jsonl(
                        review_log_path,
                        {
                            "ts": time.time(),
                            "scene_id": scene_id,
                            "event": "web_auto_label_job_cancelled",
                            "job_id": job_id,
                            "scope": scope,
                            "processed": idx - 1,
                            "updated": updated,
                            "skipped": skipped,
                            "failed": failed,
                        },
                    )
                    _write_snapshot(trigger=f"auto_label_{scope}_cancelled")
                    _finalize_auto_label_job(
                        job_id,
                        message=f"Auto-labeling canceled: {scope} processed={idx - 1}/{total} updated={updated} skipped={skipped} failed={failed}",
                    )
                    return

                effective_item = _get_effective_item_by_instance_id(current_instance_id)
                if not isinstance(effective_item, dict):
                    skipped += 1
                    _set_auto_label_job_state(processed=idx, skipped=skipped)
                    continue

                decision_obj = decision_map_raw.get(current_instance_id, None)
                action, _, _ = _resolve_review_decision(decision_obj, default_action)
                if action != "keep":
                    skipped += 1
                    _set_auto_label_job_state(processed=idx, skipped=skipped)
                    continue

                if scope == "all" and _item_has_existing_auto_label(effective_item):
                    skipped += 1
                    _set_auto_label_job_state(processed=idx, skipped=skipped)
                    continue

                try:
                    fields = _build_auto_label_fields(effective_item, scope=scope, stage2_cfg=stage2_cfg)
                except Exception as exc:
                    failed += 1
                    _set_auto_label_job_state(processed=idx, failed=failed, last_error=str(exc))
                    append_jsonl(
                        review_log_path,
                        {
                            "ts": time.time(),
                            "scene_id": scene_id,
                            "event": "web_auto_label_item_failed",
                            "job_id": job_id,
                            "scope": scope,
                            "instance_id": current_instance_id,
                            "error": str(exc),
                        },
                    )
                    continue

                with state_lock:
                    if current_instance_id not in instance_overrides:
                        instance_overrides[current_instance_id] = {}
                    instance_overrides[current_instance_id].update(fields)
                    _sync_annotation_status_override(current_instance_id)
                    updated += 1
                    auto_label_job["processed"] = idx
                    auto_label_job["updated"] = updated
                    auto_label_job["skipped"] = skipped
                    auto_label_job["failed"] = failed
                    auto_label_job["last_error"] = None
                    auto_label_job["message"] = f"Auto-labeling: {scope} {idx}/{total} {current_instance_id}"
                append_jsonl(
                    review_log_path,
                    {
                        "ts": time.time(),
                        "scene_id": scene_id,
                        "event": "web_auto_label_item_done",
                        "job_id": job_id,
                        "scope": scope,
                        "instance_id": current_instance_id,
                        "fields": fields,
                    },
                )
                _write_snapshot(trigger=f"auto_label_{scope}_progress")

            append_jsonl(
                review_log_path,
                {
                    "ts": time.time(),
                    "scene_id": scene_id,
                    "event": "web_auto_label_job_end",
                    "job_id": job_id,
                    "scope": scope,
                    "class_id": class_id_raw,
                    "class_name": class_name_raw,
                    "total": total,
                    "updated": updated,
                    "skipped": skipped,
                    "failed": failed,
                },
            )
            _write_snapshot(trigger=f"auto_label_{scope}")
            _finalize_auto_label_job(
                job_id,
                message=f"Auto-labeling finished: {scope} total={total} updated={updated} skipped={skipped} failed={failed}",
                last_error=None if failed <= 0 else f"failed={failed}",
            )
        except Exception as exc:
            append_jsonl(
                review_log_path,
                {
                    "ts": time.time(),
                    "scene_id": scene_id,
                    "event": "web_auto_label_job_crashed",
                    "job_id": job_id,
                    "scope": scope,
                    "error": str(exc),
                },
            )
            _write_snapshot(trigger=f"auto_label_{scope}_error")
            _finalize_auto_label_job(job_id, message=f"Auto-labeling aborted: {scope}", last_error=str(exc))

    def _start_auto_label_job(scope: str, target_instance_ids: list[str], *, class_id_raw: Any = None, class_name_raw: str = "") -> tuple[bool, dict[str, Any] | None, str | None]:
        with state_lock:
            if bool(auto_label_job.get("running", False)):
                return False, _export_auto_label_task(), "auto_label_job_running"
            next_job_id = int(auto_label_job.get("job_id", 0) or 0) + 1
            auto_label_job.update(
                {
                    "job_id": next_job_id,
                    "running": True,
                    "cancel_requested": False,
                    "scope": scope,
                    "class_id": class_id_raw,
                    "class_name": class_name_raw,
                    "total": int(len(target_instance_ids)),
                    "processed": 0,
                    "updated": 0,
                    "skipped": 0,
                    "failed": 0,
                    "current_instance_id": None,
                    "message": f"Auto-labeling started: {scope}",
                    "last_error": None,
                    "started_at": float(time.time()),
                    "finished_at": None,
                    "thread": None,
                }
            )
            thread = threading.Thread(
                target=_run_auto_label_job,
                args=(next_job_id, scope, list(target_instance_ids)),
                kwargs={"class_id_raw": class_id_raw, "class_name_raw": class_name_raw},
                daemon=True,
            )
            auto_label_job["thread"] = thread
            task_state = _export_auto_label_task()
        thread.start()
        return True, task_state, None

    _refresh_annotation_status_overrides()

    append_jsonl(
        review_log_path,
        {
            "ts": time.time(),
            "scene_id": scene_id,
            "event": "web_review_session_start",
            "instances_total": len(instances),
            "default_action": default_action,
            "decision_file": str(decisions_path.as_posix()) if decisions_path else None,
        },
    )
    _write_snapshot(trigger="init")

    app = Flask(__name__)

    def _cfg_bool(default: bool, key: str) -> bool:
        if key not in stage2_cfg:
            return bool(default)
        value = stage2_cfg.get(key)
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    web_img_compress_default = _cfg_bool(True, "review_web_image_compress_enabled")
    web_img_max_width_default = max(1, int(stage2_cfg.get("review_web_image_max_width", 854) or 854))
    web_img_max_height_default = max(1, int(stage2_cfg.get("review_web_image_max_height", 480) or 480))
    web_img_quality_default = int(np.clip(int(stage2_cfg.get("review_web_image_jpeg_quality", 80) or 80), 40, 100))
    web_show_bbox_default = _cfg_bool(False, "review_web_show_bbox_enabled")

    @app.get("/")
    def index_page() -> str:
        return """<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <title>Stage2 Review</title>
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
        body { font-family: Inter, Arial, sans-serif; margin: 0; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }
        .banner {
            height: 56px; background: var(--surface); border-bottom: 1px solid var(--line);
            display: flex; align-items: center; justify-content: space-between; padding: 0 14px; font-weight: 700;
        }
        .banner-actions { display: flex; align-items: center; gap: 8px; }
        .footer {
            height: 34px; background: var(--surface); border-top: 1px solid var(--line);
            display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px;
            align-items: center; padding: 0 12px; font-size: 12px; color: var(--muted);
        }
        .footer-pane { display: flex; align-items: center; gap: 8px; min-width: 0; }
        .footer-pane.right { justify-content: flex-end; }
        .footer-label { color: var(--muted); flex: 0 0 auto; }
        .op-status { color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
        .op-status.success { color: #16a34a; }
        .op-status.error { color: #dc2626; }
        .op-status.info { color: var(--accent-2); }
        .main {
            height: calc(100vh - 90px);
            display: grid;
            grid-template-columns: 300px 1fr 460px;
            gap: 10px;
            padding: 10px;
            min-height: 0;
        }
        .card { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; min-height: 0; }
        .card-pad { padding: 10px; }
        .left-list { display: flex; flex-direction: column; height: 100%; }
        .left-tools { display: flex; gap: 8px; padding: 8px 10px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
        .left-tools select { width: 100%; background: var(--surface-2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 5px 7px; font-size: 12px; }
        .list-box { overflow-y: auto; flex: 1; border-top: 1px solid var(--line); background: var(--surface-2); }
        .list-item { padding: 10px; border-bottom: 1px solid var(--line); cursor: pointer; font-size: 13px; }
        .list-item.active { background: var(--list-active); }
        .list-item:hover { background: var(--list-hover); }
        .status-chip { display: inline-block; border-radius: 10px; padding: 1px 7px; font-size: 11px; margin-left: 6px; }
        .status-keep { color: #dcfce7; background: #15803d; }
        .status-drop { color: #fee2e2; background: #b91c1c; }
        .status-undecided { color: #e2e8f0; background: #475569; }
        .status-labeled { color: #dcfce7; background: #166534; }
        .status-pending_review { color: #fef3c7; background: #92400e; }
        .status-failed { color: #fee2e2; background: #991b1b; }
        .status-unlabeled { color: #dbeafe; background: #1e40af; }
        .middle-pane { display: grid; grid-template-rows: repeat(3, minmax(0, 1fr)); gap: 10px; height: 100%; min-height: 0; overflow: hidden; }
        .view-card { display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }
        .view-title { margin: 0; padding: 8px 10px; border-bottom: 1px solid var(--line); color: var(--muted); font-size: 12px; }
        .top-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            grid-auto-rows: minmax(0, 1fr);
            gap: 8px;
            min-height: 0;
            padding: 8px;
            background: var(--surface-2);
        }
        .rgb-side-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            grid-auto-rows: minmax(0, 1fr);
            gap: 8px;
            min-height: 0;
            padding: 8px;
            background: var(--surface-2);
        }
        .view-reason { margin: 0; padding: 0 10px 8px 10px; font-size: 12px; color: var(--muted); }
        .grid-cell { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #00000010; display: grid; grid-template-rows: minmax(0, 1fr) auto; min-height: 0; }
        .grid-cell img { width: 100%; height: 100%; max-height: 100%; object-fit: contain; border-radius: 6px; background: #00000018; }
        .grid-media { position: relative; width: 100%; height: 100%; min-height: 0; }
        .bbox-overlay {
            position: absolute;
            border: 2px solid #ef4444;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.45) inset;
            pointer-events: none;
            display: none;
        }
        .cell-caption { font-size: 11px; color: var(--muted); text-align: center; padding: 4px 2px; border-top: 1px solid var(--line); background: var(--surface); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .cell-tools { display: flex; gap: 6px; padding: 6px; border-top: 1px solid var(--line); background: var(--surface); }
        .cell-tools select { flex: 1; background: var(--surface-2); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 3px 5px; font-size: 11px; }
        .cell-tools button { margin: 0; padding: 4px 8px; font-size: 11px; border: 1px solid var(--line); border-radius: 6px; }
        .occluded-box {
            width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;
            color: #fecaca; font-size: 16px; font-weight: 700; background: repeating-linear-gradient(
                45deg,
                rgba(127, 29, 29, 0.20),
                rgba(127, 29, 29, 0.20) 12px,
                rgba(239, 68, 68, 0.16) 12px,
                rgba(239, 68, 68, 0.16) 24px
            );
        }
        .empty-tip { color: var(--muted); font-size: 13px; }
        .attrs { height: 100%; overflow-y: auto; padding: 10px; background: var(--surface-2); }
        .row { margin: 7px 0; display: grid; grid-template-columns: 170px 1fr; gap: 8px; align-items: center; }
        .row.textarea-row { align-items: start; }
        .row label { color: var(--muted); font-size: 13px; }
        .row input, .row select, textarea {
            width: 100%; background: var(--surface); color: var(--text); border: 1px solid var(--line); border-radius: 6px;
            padding: 6px 8px; font-size: 13px;
        }
        textarea { min-height: 72px; margin-top: 8px; }
        #f_auto_label_landmark_description { min-height: 96px; resize: vertical; overflow: hidden; margin-top: 0; }
        button {
            margin: 4px 4px 4px 0; padding: 7px 12px; border: 1px solid transparent; border-radius: 7px;
            cursor: pointer; background: var(--surface); color: var(--text);
        }
        button:hover { filter: brightness(1.05); }
        .keep { background: var(--ok); color: #fff; }
        .drop { background: var(--bad); color: #fff; }
        .nav { background: var(--accent-2); color: #fff; }
        .primary { background: var(--accent); color: #fff; }
        .toolbar { margin-top: 8px; }
        .sub-title { margin: 0; padding: 10px; border-bottom: 1px solid var(--line); font-size: 13px; color: var(--muted); }
        code { color: var(--code); }
  </style>
</head>
<body>
        <div class='banner'>
            <div>Stage 2 Step 2 Review Workspace - <code id='scene' style='margin-left:8px'></code></div>
            <div class='banner-actions'>
                <label style='display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);'>
                    <input id='webCompressToggle' type='checkbox' checked onchange='toggleWebCompress(this.checked)' />
                    <span id='webCompressLabel'>Compressed View (480P)</span>
                </label>
                <label style='display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);'>
                    <input id='webBboxToggle' type='checkbox' onchange='toggleWebShowBbox(this.checked)' />
                    <span>Show BBox</span>
                </label>
                <button id='themeToggle' onclick='toggleTheme()'>Toggle Theme</button>
            </div>
        </div>
    <div class='main'>
            <div class='card left-list'>
                <p class='sub-title' id='progress'></p>
                <div class='left-tools'>
                    <select id='statusFilter' onchange='setStatusFilter(this.value)'>
                        <option value='all'>All Status</option>
                        <option value='undecided'>Unreviewed</option>
                        <option value='keep'>keep</option>
                        <option value='drop'>drop</option>
                    </select>
                </div>
                <div class='list-box' id='landmarkList'></div>
            </div>

            <div class='middle-pane'>
                <div class='card view-card'>
                    <p class='view-title'>Row 1: Point-cloud Main View / Point-cloud Top-down / RGB Top-down</p>
                    <div class='top-grid' id='topGrid'></div>
                    <p class='view-reason' id='topReason'></p>
                </div>
                <div class='card view-card'>
                    <p class='view-title'>Row 2: RGB Side Views (front → back_right)</p>
                    <div class='rgb-side-grid' id='rgbGridRow1'></div>
                    <p class='view-reason' id='rgbReasonRow1'></p>
                </div>
                <div class='card view-card'>
                    <p class='view-title'>Row 3: RGB Side Views (back → front_left)</p>
                    <div class='rgb-side-grid' id='rgbGridRow2'></div>
                    <p class='view-reason' id='rgbReasonRow2'></p>
                </div>
            </div>

            <div class='card'>
                <div class='attrs'>
                <div class='row'><label>instance_id</label><input id='f_instance_id' disabled /></div>
                <div class='row'><label>class_id</label><input id='f_class_id' /></div>
                <div class='row'><label>class_name</label><input id='f_class_name' /></div>
                <div class='row'><label></label><button onclick='syncClassName()'>Sync to Same-Class Landmarks</button></div>
                <div class='row'><label>point_count</label><input id='f_point_count' /></div>
                <div class='row'><label>review_action</label><select id='f_review_action'><option>keep</option><option>drop</option></select></div>
                <div class='row'><label>review_note</label><input id='f_review_note' /></div>
                <hr/>
                <div class='row'><label>auto_label_category</label><input id='f_auto_label_landmark_type' /></div>
                <div class='row'><label>auto_label_subcategory</label><input id='f_auto_label_name' /></div>
                <div class='row textarea-row'><label>auto_label_description</label><textarea id='f_auto_label_landmark_description' oninput='autoGrowTextarea(this.id)'></textarea></div>
                <div class='row'><label>auto_label_confidence</label><input id='f_auto_label_confidence' /></div>
                <hr/>
                <div class='row'><label>landmark_category</label><select id='f_landmark_category'>
                    <option value="">--Select--</option>
                    <option value="building">building</option>
                    <option value="vehicle">vehicle</option>
                    <option value="public_facility">public_facility</option>
                    <option value="transport_infrastructure">transport_infrastructure</option>
                    <option value="industrial_infrastructure">industrial_infrastructure</option>
                    <option value="vegetation">vegetation</option>
                    <option value="urban_landscape">urban_landscape</option>
                    <option value="other">other</option>
                </select></div>
                <div class='row'><label>landmark_subcategory</label><input id='f_landmark_subcategory' /></div>
                <div class='row textarea-row'><label>landmark_description</label><textarea id='f_landmark_description' oninput='autoGrowTextarea(this.id)'></textarea></div>
                <div class='row'><label>landmark_note</label><input id='f_landmark_note' /></div>
                <div class='row'><label>Step 2 Review</label><div><button class='keep' onclick="decide('keep')">Keep</button><button class='drop' onclick="decide('drop')">Drop</button><button onclick="decide('clear')">Clear</button></div></div>
                <div class='row'><label>Auto Labeling</label><div><button onclick='autoLabelSingle()'>Current</button><button onclick='autoLabelClass()'>By Class</button><button onclick='autoLabelAll()'>Global</button><button id='autoLabelCancelBtn' onclick='cancelAutoLabelTask()' disabled>Cancel Task</button></div></div>
                <div class='row'><label>Review Actions</label><div><button class='primary' onclick='approveAutoLabel()'>Accept Auto Label</button><button class='primary' onclick='saveManualReview()'>Save Manual Revision</button></div></div>
                <div class='row'><label>Clear Auto Labels</label><div><button onclick='clearAutoLabelSingle()'>Current</button><button onclick='clearAutoLabelClass()'>Class</button><button onclick='clearAutoLabelAll()'>Global</button></div></div>
                                <div class='toolbar'>
                    <button class='nav' onclick='move(-1)'>Previous</button>
                    <button class='nav' onclick='move(1)'>Next</button>
                    <button class='primary' onclick='saveItem()'>Save All Fields</button>
                    <button onclick='refreshState()'>Refresh</button>
                </div>
                </div>
      </div>
    </div>
    <div class='footer'>
        <div class='footer-pane'>
            <span class='footer-label'>Actions</span>
            <div id='opStatusLeft' class='op-status info'>Ready</div>
        </div>
        <div class='footer-pane right'>
            <span class='footer-label'>Annotation</span>
            <div id='opStatusRight' class='op-status info'>Auto Label: Ready</div>
        </div>
    </div>

<script>
let state = null;
let listRows = [];
let statusFilter = 'all';
let lastCurrentIndex = -1;
let autoLabelPollHandle = null;
let lastAutoLabelTaskKey = '';
const directionOptions = ['front','front_right','right','back_right','back','back_left','left','front_left'];
const webDisplay = {
    compressEnabled: true,
    maxWidth: 854,
    maxHeight: 480,
    jpegQuality: 80,
    showBbox: false,
    initialized: false,
};
function v(x){ return (x===null || x===undefined) ? '' : String(x); }
function parseMaybeNumber(s){ if(s==='') return null; const n=Number(s); return Number.isFinite(n)?n:s; }

function _toBool(v, d){
    if(v === true || v === false) return v;
    const t = String(v ?? '').trim().toLowerCase();
    if(['1','true','yes','on'].includes(t)) return true;
    if(['0','false','no','off'].includes(t)) return false;
    return !!d;
}

function _readBoolQueryParam(name){
    try{
        const params = new URLSearchParams(window.location.search || '');
        if(!params.has(name)) return null;
        const raw = String(params.get(name) || '').trim().toLowerCase();
        if(['1','true','yes','on'].includes(raw)) return true;
        if(['0','false','no','off'].includes(raw)) return false;
        return null;
    }catch(_e){
        return null;
    }
}

function applyWebDisplaySettings(cfg){
    if(!cfg) return;
    webDisplay.compressEnabled = _toBool(cfg.compress_enabled, true);
    webDisplay.maxWidth = Math.max(1, Number(cfg.max_width) || 854);
    webDisplay.maxHeight = Math.max(1, Number(cfg.max_height) || 480);
    webDisplay.jpegQuality = Math.min(100, Math.max(40, Number(cfg.jpeg_quality) || 80));
    const queryShowBbox = _readBoolQueryParam('show_bbox');
    const cfgShowBbox = _toBool(cfg.show_bbox, false);
    webDisplay.showBbox = (queryShowBbox === null) ? cfgShowBbox : queryShowBbox;
    const toggle = document.getElementById('webCompressToggle');
    if(toggle){ toggle.checked = !!webDisplay.compressEnabled; }
    const label = document.getElementById('webCompressLabel');
    if(label){ label.textContent = `Compressed View (${webDisplay.maxHeight}P)`; }
    const bboxToggle = document.getElementById('webBboxToggle');
    if(bboxToggle){ bboxToggle.checked = !!webDisplay.showBbox; }
}

function displayImageUrl(url){
    const s = String(url || '').trim();
    if(!s) return '';
    if(!(s.startsWith('/raw/') || s.startsWith('http://') || s.startsWith('https://'))){
        return s;
    }
    try{
        const u = new URL(s, window.location.origin);
        if(webDisplay.compressEnabled){
            u.searchParams.set('compress', '1');
            u.searchParams.set('max_w', String(webDisplay.maxWidth));
            u.searchParams.set('max_h', String(webDisplay.maxHeight));
            u.searchParams.set('quality', String(webDisplay.jpegQuality));
        }else{
            u.searchParams.set('compress', '0');
        }
        return `${u.pathname}${u.search}`;
    }catch(_e){
        return s;
    }
}

function toggleWebCompress(enabled){
    webDisplay.compressEnabled = !!enabled;
    setStatus(`Web image compression: ${webDisplay.compressEnabled ? 'On' : 'Off'}`, 'info');
    refreshState();
}

function toggleWebShowBbox(enabled){
    webDisplay.showBbox = !!enabled;
    setStatus(`BBox display: ${webDisplay.showBbox ? 'On' : 'Off'}`, 'info');
    refreshState();
}

function autoGrowTextarea(id){
    const el = document.getElementById(id);
    if(!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.max(el.scrollHeight, 96)}px`;
}

function setStatus(message, level){
    const el = document.getElementById('opStatusLeft');
    if(!el) return;
    const lv = level || 'info';
    el.className = `op-status ${lv}`;
    el.textContent = message || 'Ready';
}

function setAutoLabelStatus(message, level){
    const el = document.getElementById('opStatusRight');
    if(!el) return;
    const lv = level || 'info';
    el.className = `op-status ${lv}`;
    el.textContent = message || 'Auto Label: Ready';
}

function autoLabelTaskKey(task){
    const t = task || {};
    return JSON.stringify([
        Number(t.job_id || 0),
        !!t.running,
        !!t.cancel_requested,
        Number(t.processed || 0),
        Number(t.updated || 0),
        Number(t.skipped || 0),
        Number(t.failed || 0),
        String(t.current_instance_id || ''),
        String(t.message || ''),
        t.finished_at || null,
        t.last_error || null,
    ]);
}

function autoLabelTaskText(task){
    const t = task || {};
    const scope = String(t.scope || '-');
    const total = Number(t.total || 0);
    const processed = Number(t.processed || 0);
    const updated = Number(t.updated || 0);
    const skipped = Number(t.skipped || 0);
    const failed = Number(t.failed || 0);
    const instanceId = String(t.current_instance_id || '');
    if(t.running){
        return `Auto-labeling: ${scope} ${processed}/${total} updated=${updated} skipped=${skipped} failed=${failed}${instanceId ? ` | ${instanceId}` : ''}`;
    }
    if(t.message){
        return String(t.message);
    }
    if(t.last_error){
        return `Auto-labeling failed: ${t.last_error}`;
    }
    return 'Ready';
}

function syncAutoLabelTaskUi(task){
    const t = task || {};
    const active = !!(t.running || t.cancel_requested);
    const btn = document.getElementById('autoLabelCancelBtn');
    if(btn){ btn.disabled = !(t.running && !t.cancel_requested); }
    if(active && !autoLabelPollHandle){
        autoLabelPollHandle = window.setInterval(()=>{
            refreshState().catch((err)=>{
                console.error(err);
                setAutoLabelStatus(`Failed to refresh auto-label status: ${err}`, 'error');
            });
        }, 2000);
    }else if(!active && autoLabelPollHandle){
        window.clearInterval(autoLabelPollHandle);
        autoLabelPollHandle = null;
    }
    const key = autoLabelTaskKey(t);
    if(active){
        setAutoLabelStatus(autoLabelTaskText(t), 'info');
        lastAutoLabelTaskKey = key;
        return active;
    }
    if(key !== lastAutoLabelTaskKey){
        if(Number(t.job_id || 0) > 0){
            setAutoLabelStatus(autoLabelTaskText(t), t.last_error ? 'error' : 'success');
        }
        lastAutoLabelTaskKey = key;
    }
    return active;
}

function setStatusFilter(v){
    statusFilter = (v || 'all');
    setStatus(`Filter changed to: ${statusFilter}`, 'info');
    refreshList();
}

function statusClass(action){
    if(action === 'keep') return 'status-keep';
    if(action === 'drop') return 'status-drop';
    return 'status-undecided';
}

function statusText(action){
    if(action === 'keep') return 'keep';
    if(action === 'drop') return 'drop';
    return 'undecided';
}

function annotationClass(status){
    if(status === 'labeled') return 'status-labeled';
    if(status === 'pending_review') return 'status-pending_review';
    if(status === 'failed') return 'status-failed';
    return 'status-unlabeled';
}

function annotationText(status){
    if(status === 'labeled') return 'labeled';
    if(status === 'pending_review') return 'pending_review';
    if(status === 'failed') return 'failed';
    return 'unlabeled';
}

function initTheme(){
    const stored = localStorage.getItem('stage2_theme');
    const autoDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = stored || (autoDark ? 'dark' : 'light');
    document.body.setAttribute('data-theme', theme);
}

function toggleTheme(){
    const cur = document.body.getAttribute('data-theme') || 'dark';
    const next = cur === 'dark' ? 'light' : 'dark';
    document.body.setAttribute('data-theme', next);
    localStorage.setItem('stage2_theme', next);
}

async function refreshList(){
    const res = await fetch('/api/list');
    const data = await res.json();
    listRows = data.items || [];
    const filtered = statusFilter === 'all' ? listRows : listRows.filter((row)=>String(row.action||'undecided')===statusFilter);
    const box = document.getElementById('landmarkList');
    box.innerHTML = '';
    // Render grouped by class_name + class_id; index within each group
    let lastClass = null;
    let lastClassId = null;
    let classCounter = 1;
    for(const row of filtered){
        if(row.class_name !== lastClass || row.class_id !== lastClassId){
            const groupTitle = document.createElement('div');
            groupTitle.className = 'list-item';
            groupTitle.style.background = 'var(--surface)';
            groupTitle.style.fontWeight = 'bold';
            groupTitle.style.color = 'var(--accent)';
            groupTitle.innerText = `[${row.class_id}] ${row.class_name || row.class_id}`;
            box.appendChild(groupTitle);
            lastClass = row.class_name;
            lastClassId = row.class_id;
            classCounter = 1;
        }
        const div = document.createElement('div');
        const action = statusText(row.action);
        const ann = annotationText(String(row.annotation_status || ''));
        const autoLabelCategory = v(row.auto_label_category || row.auto_label_landmark_type);
        const autoLabelName = v(row.auto_label_subcategory || row.auto_label_name);
        const autoLabelConfRaw = row.auto_label_confidence;
        const autoLabelConf = (autoLabelConfRaw === null || autoLabelConfRaw === undefined || autoLabelConfRaw === '') ? '' : Number(autoLabelConfRaw);
        const autoLabelText = (autoLabelCategory || autoLabelName)
            ? `${autoLabelCategory || '-'} / ${autoLabelName || '-'}${Number.isFinite(autoLabelConf) ? ` / ${(autoLabelConf*100).toFixed(1)}%` : ''}`
            : '';
        div.className = 'list-item' + (state && state.current.index===row.index ? ' active' : '');
        div.innerHTML = `[${classCounter}] ${row.instance_id} <span class="status-chip ${statusClass(action)}">${action}</span><span class="status-chip ${annotationClass(ann)}">${ann}</span>${autoLabelText ? `<span class="status-chip" title="auto_label">${autoLabelText}</span>` : ''}`;
        div.onclick = async ()=>{
            setStatus(`Switching to landmark ${row.instance_id} ...`, 'info');
            await fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:row.index})});
            await refreshState();
            setStatus(`Switched to ${row.instance_id}`, 'success');
        };
        box.appendChild(div);
        classCounter++;
    }
}

function bindFields(item){
    document.getElementById('f_instance_id').value=v(item.instance_id);
    document.getElementById('f_class_id').value=v(item.class_id);
    document.getElementById('f_class_name').value=v(item.class_name);
    document.getElementById('f_point_count').value=v(item.point_count);
    document.getElementById('f_review_action').value=v(item.review_action||'keep');
    document.getElementById('f_review_note').value=v(item.review_note);
    document.getElementById('f_auto_label_name').value=v(item.auto_label_subcategory || item.auto_label_name);
    document.getElementById('f_auto_label_confidence').value=v(item.auto_label_confidence);
    document.getElementById('f_auto_label_landmark_type').value=v(item.auto_label_category || item.auto_label_landmark_type);
    document.getElementById('f_auto_label_landmark_description').value=v(item.auto_label_description || item.auto_label_landmark_description);
    autoGrowTextarea('f_auto_label_landmark_description');
    document.getElementById('f_landmark_category').value=v(item.landmark_category);
    document.getElementById('f_landmark_subcategory').value=v(item.landmark_subcategory);
    document.getElementById('f_landmark_description').value=v(item.landmark_description);
    autoGrowTextarea('f_landmark_description');
    document.getElementById('f_landmark_note').value=v(item.landmark_note);
}

function clearFields(){
    bindFields({});
    document.getElementById('f_review_action').value='keep';
}

function renderViewGrid(gridId, urls, captions, reasonId, reasonText){
    const grid = document.getElementById(gridId);
    const arr = Array.isArray(urls) ? urls : [];
    const capArr = Array.isArray(captions) ? captions : [];
    const count = arr.length;
    const cols = count <= 0 ? 1 : (count <= 5 ? count : 4);
    grid.style.setProperty('--cols', String(cols));
    grid.innerHTML = '';
    if(count === 0){
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        const cap = document.createElement('div');
        cap.className = 'cell-caption';
        cap.textContent = 'No available view';
        cell.appendChild(cap);
        grid.appendChild(cell);
    }else{
        for(let i=0;i<count;i++){
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            const img = document.createElement('img');
            img.alt = `${gridId}_${i}`;
            img.src = displayImageUrl(arr[i] || '');
            img.style.opacity = arr[i] ? '1' : '0.2';
            const cap = document.createElement('div');
            cap.className = 'cell-caption';
            cap.textContent = capArr[i] || `View ${i+1}`;
            cell.appendChild(img);
            cell.appendChild(cap);
            grid.appendChild(cell);
        }
    }
    const reasonEl = document.getElementById(reasonId);
    if(reasonEl){
        reasonEl.textContent = reasonText || '';
    }
}

function createSimpleCell(slot){
    const cell = document.createElement('div');
    cell.className = 'grid-cell';
    if(slot && slot.url){
        const img = document.createElement('img');
        img.alt = 'top_view';
        img.src = displayImageUrl(String(slot.url || ''));
        cell.appendChild(img);
    }else{
        const occ = document.createElement('div');
        occ.className = 'occluded-box';
        occ.textContent = 'Occluded';
        cell.appendChild(occ);
    }
    const cap = document.createElement('div');
    cap.className = 'cell-caption';
    cap.textContent = (slot && slot.caption) ? slot.caption : 'No available view';
    cell.appendChild(cap);
    return cell;
}

function bindPcd(urls, captions, reason, rgbSlots){
    const arr = Array.isArray(urls) ? urls : [];
    const capArr = Array.isArray(captions) ? captions : [];
    const slots = Array.isArray(rgbSlots) ? rgbSlots : [];

    let pcdMain = null;
    let pcdTop = null;
    for(let i=0;i<arr.length;i++){
        const slot = {url: arr[i], caption: capArr[i] || `Point-cloud View ${i+1}`};
        const c = String(slot.caption || '');
        if(!pcdMain && c.indexOf('Top-down') < 0) pcdMain = slot;
        if(!pcdTop && c.indexOf('Top-down') >= 0) pcdTop = slot;
    }
    if(!pcdMain && arr.length > 0){
        pcdMain = {url: arr[0], caption: capArr[0] || 'Point-cloud Main View'};
    }
    if(!pcdTop && arr.length > 1){
        pcdTop = {url: arr[1], caption: capArr[1] || 'Point-cloud Top-down'};
    }

    const rgbTopdown = slots.find((s)=>s && String(s.mode || '') === 'topdown') || null;
    const rgbTop = rgbTopdown ? {
        url: rgbTopdown.url || '',
        caption: `RGB Top-down${rgbTopdown.is_valid ? ' | Valid' : ' | Invalid'}`,
    } : null;

    const topGrid = document.getElementById('topGrid');
    topGrid.innerHTML = '';
    topGrid.appendChild(createSimpleCell(pcdMain ? {url: pcdMain.url, caption: `Point-cloud Main View | ${pcdMain.caption}`} : null));
    topGrid.appendChild(createSimpleCell(pcdTop ? {url: pcdTop.url, caption: `Point-cloud Top-down | ${pcdTop.caption}`} : null));
    topGrid.appendChild(createSimpleCell(rgbTop ? {url: rgbTop.url, caption: rgbTop.caption} : {url:'', caption:'RGB Top-down (occluded)'}));

    const reasonEl = document.getElementById('topReason');
    if(reasonEl){
        reasonEl.textContent = reason || '';
    }
}

async function setRgbDirection(viewIndex, direction, markMain){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    setStatus(`Updating view direction: ${direction} ...`, 'info');
    const res = await fetch('/api/set_view_direction', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
            instance_id: item.instance_id,
            view_index: Number(viewIndex),
            direction: String(direction || ''),
            view_type: 'rgb',
            mark_main: !!markMain,
        }),
    });
    const data = await res.json();
    if(!data.ok){ setStatus('Failed to update direction', 'error'); alert('Update failed: ' + (data.error || 'unknown')); return; }
    await refreshState();
    setStatus('View direction updated', 'success');
}

async function setRgbValidity(viewIndex, isValid){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    setStatus(`Updating RGB validity: view_index=${viewIndex} ...`, 'info');
    const res = await fetch('/api/set_rgb_view_validity', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
            instance_id: item.instance_id,
            view_index: Number(viewIndex),
            is_valid: !!isValid,
            view_type: 'rgb',
        }),
    });
    const data = await res.json();
    if(!data.ok){ setStatus('Failed to update RGB validity', 'error'); alert('Update failed: ' + (data.error || 'unknown')); return; }
    await refreshState();
    setStatus('RGB validity updated', 'success');
}

function _renderRgbCell(slot, i){
    const realViewIndex = Number(slot.view_index ?? -1);
    const hasRealImage = !!slot.url;
    const cell = document.createElement('div');
    cell.className = 'grid-cell';
    if(hasRealImage){
        const media = document.createElement('div');
        media.className = 'grid-media';
        const img = document.createElement('img');
        img.alt = `rgbGrid_${i}`;
        img.src = displayImageUrl(String(slot.url || ''));
        img.style.opacity = slot.is_valid ? '1' : '0.45';
        media.appendChild(img);

        const bbox = Array.isArray(slot.bbox_2d_xyxy) ? slot.bbox_2d_xyxy : null;
        const bboxSize = Array.isArray(slot.bbox_2d_image_size) ? slot.bbox_2d_image_size : null;
        const bboxValid = !!slot.bbox_2d_valid && Array.isArray(bbox) && bbox.length >= 4;
        if(webDisplay.showBbox && bboxValid){
            const srcW = Math.max(1, Number((bboxSize && bboxSize[0]) || 1));
            const srcH = Math.max(1, Number((bboxSize && bboxSize[1]) || 1));
            const x0 = Math.max(0, Number(bbox[0]) || 0);
            const y0 = Math.max(0, Number(bbox[1]) || 0);
            const x1 = Math.max(x0 + 1, Number(bbox[2]) || (x0 + 1));
            const y1 = Math.max(y0 + 1, Number(bbox[3]) || (y0 + 1));
            const overlay = document.createElement('div');
            overlay.className = 'bbox-overlay';
            const updateOverlay = ()=>{
                const rectW = Math.max(1, media.clientWidth || 1);
                const rectH = Math.max(1, media.clientHeight || 1);
                const scale = Math.min(rectW / srcW, rectH / srcH);
                const drawW = srcW * scale;
                const drawH = srcH * scale;
                const offX = (rectW - drawW) * 0.5;
                const offY = (rectH - drawH) * 0.5;
                const bx = offX + (x0 / srcW) * drawW;
                const by = offY + (y0 / srcH) * drawH;
                const bw = Math.max(1, ((x1 - x0) / srcW) * drawW);
                const bh = Math.max(1, ((y1 - y0) / srcH) * drawH);
                overlay.style.left = `${bx}px`;
                overlay.style.top = `${by}px`;
                overlay.style.width = `${bw}px`;
                overlay.style.height = `${bh}px`;
                overlay.style.display = 'block';
            };
            img.onload = ()=>{ updateOverlay(); };
            updateOverlay();
            media.appendChild(overlay);
        }

        cell.appendChild(media);
    }else{
        const occ = document.createElement('div');
        occ.className = 'occluded-box';
        occ.textContent = 'Occluded';
        cell.appendChild(occ);
    }

    const cap = document.createElement('div');
    cap.className = 'cell-caption';
    const d = slot.view_direction ? ` | Direction: ${slot.view_direction}` : '';
    const q = slot.is_query_view ? ' | Main View' : '';
    const iv = slot.is_valid ? ' | Valid' : ' | Invalid';
    cap.textContent = `${slot.caption || `View ${i+1}`}${d}${q}${iv}`;

    const tools = document.createElement('div');
    tools.className = 'cell-tools';
    const sel = document.createElement('select');
    for(const opt of directionOptions){
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        sel.appendChild(o);
    }
    sel.value = String(slot.view_direction || 'front');
    sel.disabled = !hasRealImage;
    sel.onchange = async ()=>{ await setRgbDirection(realViewIndex, sel.value, false); };

    const btn = document.createElement('button');
    if(slot.is_query_view){
        btn.textContent = 'Main View ✓';
        btn.style.background = 'var(--accent)';
        btn.style.color = '#fff';
        btn.style.border = '2px solid var(--accent)';
    }else{
        btn.textContent = 'Set as Main View';
    }
    btn.disabled = !hasRealImage;
    btn.onclick = async ()=>{ await setRgbDirection(realViewIndex, sel.value, true); };

    const validSel = document.createElement('select');
    const optValid = document.createElement('option');
    optValid.value = 'valid';
    optValid.textContent = 'Valid';
    const optInvalid = document.createElement('option');
    optInvalid.value = 'invalid';
    optInvalid.textContent = 'Invalid';
    validSel.appendChild(optValid);
    validSel.appendChild(optInvalid);
    validSel.value = slot.is_valid ? 'valid' : 'invalid';
    validSel.disabled = !hasRealImage;
    validSel.onchange = async ()=>{ await setRgbValidity(realViewIndex, validSel.value === 'valid'); };

    tools.appendChild(sel);
    tools.appendChild(btn);
    tools.appendChild(validSel);

    cell.appendChild(cap);
    cell.appendChild(tools);
    return cell;
}

function bindRgb(slots, reason){
    const arr = Array.isArray(slots) ? slots : [];
    const sideSlots = directionOptions.map((direction)=>{
        const found = arr.find((s)=>s && String(s.mode || 'orbit') !== 'topdown' && String(s.slot_key || s.view_direction || '') === direction);
        if(found) return found;
        return {
            slot_key: direction,
            caption: `${direction} (occluded)`,
            view_index: null,
            url: '',
            view_direction: direction,
            is_query_view: false,
            is_valid: false,
            is_occluded: true,
            mode: 'orbit',
        };
    });

    const row1 = document.getElementById('rgbGridRow1');
    const row2 = document.getElementById('rgbGridRow2');
    row1.innerHTML = '';
    row2.innerHTML = '';

    for(let i=0;i<4;i++){
        row1.appendChild(_renderRgbCell(sideSlots[i], i));
    }
    for(let i=4;i<8;i++){
        row2.appendChild(_renderRgbCell(sideSlots[i], i));
    }

    const reason1 = document.getElementById('rgbReasonRow1');
    const reason2 = document.getElementById('rgbReasonRow2');
    if(reason1){ reason1.textContent = reason || ''; }
    if(reason2){ reason2.textContent = 'Direction order: front → front_right → right → back_right → back → back_left → left → front_left'; }
}

async function refreshState() {
  const res = await fetch('/api/state');
  state = await res.json();
    if(!webDisplay.initialized){
            applyWebDisplaySettings(state.web_display || {});
            webDisplay.initialized = true;
    }
  document.getElementById('scene').innerText = state.scene_id;
    const cur = state.current || {};
    const hasSel = Number(cur.index) >= 0;
    document.getElementById('progress').innerText = hasSel
        ? `Progress ${cur.index + 1}/${state.counts.total} | keep=${state.counts.keep} drop=${state.counts.drop} undecided=${state.counts.undecided}`
        : `No landmark selected | total=${state.counts.total} keep=${state.counts.keep} drop=${state.counts.drop} undecided=${state.counts.undecided}`;
  const item = state.current.item || {};
    const currentIndex = hasSel ? Number(cur.index) : -1;
    if(hasSel){
        bindFields(item);
        bindPcd(state.current.pcd_urls || [], state.current.pcd_captions || [], state.current.pcd_reason || '', state.current.rgb_slots || state.current.rgb_views || []);
        bindRgb(state.current.rgb_slots || state.current.rgb_views || [], state.current.rgb_reason || '');
    }else{
        clearFields();
        bindPcd([], [], '');
        bindRgb([], '');
    }
    if(currentIndex >= 0 && currentIndex !== lastCurrentIndex){
        setStatus(`Current landmark: ${item.instance_id || '-'} (${currentIndex + 1}/${state.counts.total})`, 'info');
    }
    lastCurrentIndex = currentIndex;
    await refreshList();
    syncAutoLabelTaskUi(state.auto_label_task || {});
}

async function autoLabelSingle(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    setAutoLabelStatus(`Starting auto-label job: ${item.instance_id || '-'}`, 'info');
    const res = await fetch('/api/auto_label_single', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance_id:item.instance_id})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to start auto-label job', 'error'); alert('Failed to start auto-label job: ' + (data.error || 'unknown')); return; }
    await refreshState();
}

async function autoLabelClass(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const classId = document.getElementById('f_class_id').value;
    const className = document.getElementById('f_class_name').value;
    setAutoLabelStatus(`Starting class-level auto-label job: class_id=${classId || '-'} ...`, 'info');
    const res = await fetch('/api/auto_label_class', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({class_id:classId, class_name:className})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to start class-level auto-label job', 'error'); alert('Failed to start class-level auto-label job: ' + (data.error || 'unknown')); return; }
    await refreshState();
}

async function autoLabelAll(){
    setAutoLabelStatus('Starting global auto-label job...', 'info');
    const res = await fetch('/api/auto_label_all', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to start global auto-label job', 'error'); alert('Failed to start global auto-label job: ' + (data.error || 'unknown')); return; }
    await refreshState();
}

async function cancelAutoLabelTask(){
    setAutoLabelStatus('Sending cancel request...', 'info');
    const res = await fetch('/api/auto_label_cancel', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to cancel auto-label job', 'error'); alert('Failed to cancel auto-label job: ' + (data.error || 'unknown')); return; }
    await refreshState();
}

async function clearAutoLabelSingle(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    if(!confirm(`Clear auto labels for current landmark ${item.instance_id || ''}?`)) return;
    setAutoLabelStatus(`Clearing current auto label: ${item.instance_id || '-'} ...`, 'info');
    const res = await fetch('/api/clear_auto_label_single', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance_id:item.instance_id})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to clear current auto label', 'error'); alert('Failed to clear current auto label: ' + (data.error || 'unknown')); return; }
    await refreshState();
    setAutoLabelStatus(`Cleared current auto label: ${item.instance_id || '-'}`, 'success');
}

async function clearAutoLabelClass(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const classId = document.getElementById('f_class_id').value;
    const className = document.getElementById('f_class_name').value;
    if(!confirm(`Clear class-level auto labels? class_id=${classId || '-'} class_name=${className || '-'}`)) return;
    setAutoLabelStatus(`Clearing class-level auto labels: class_id=${classId || '-'} ...`, 'info');
    const res = await fetch('/api/clear_auto_label_class', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({class_id:classId, class_name:className})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to clear class-level auto labels', 'error'); alert('Failed to clear class-level auto labels: ' + (data.error || 'unknown')); return; }
    alert(`Class-level auto labels cleared. Updated ${data.updated} landmarks.`);
    await refreshState();
    setAutoLabelStatus(`Class-level auto labels cleared. Updated ${data.updated} landmarks.`, 'success');
}

async function clearAutoLabelAll(){
    if(!confirm('Clear all auto labels globally? This will remove auto-label fields from every landmark.')) return;
    setAutoLabelStatus('Clearing global auto labels...', 'info');
    const res = await fetch('/api/clear_auto_label_all', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({})
    });
    const data = await res.json();
    if(!data.ok){ setAutoLabelStatus('Failed to clear global auto labels', 'error'); alert('Failed to clear global auto labels: ' + (data.error || 'unknown')); return; }
    alert(`Global auto labels cleared. Updated ${data.updated} landmarks.`);
    await refreshState();
    setAutoLabelStatus(`Global auto labels cleared. Updated ${data.updated} landmarks.`, 'success');
}

async function decide(action) {
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
  const item = state.current.item || {};
    const rgbViews = state.current.rgb_slots || state.current.rgb_views || [];
    const hasMainView = Array.isArray(rgbViews) && rgbViews.some((v)=>v && v.is_query_view && !!v.url && !!v.is_valid);
        if(action === 'keep' && !hasMainView){
                alert('Please set a main view before marking Keep.');
                return;
        }
    const note = document.getElementById('f_review_note').value || '';
    setStatus(`Applying action: ${action} ...`, 'info');
    const resp = await fetch('/api/decide', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      instance_id: item.instance_id,
      action: action,
            note: note,
      goto_next: true
    })
  });
    const data = await resp.json();
        if(!data.ok){ setStatus(`Action failed: ${action}`, 'error'); alert('Action failed: ' + (data.error || 'unknown')); return; }
  await refreshState();
        setStatus(`Action completed: ${action}`, 'success');
}

async function move(delta) {
    setStatus(delta < 0 ? 'Switching to previous item...' : 'Switching to next item...', 'info');
  await fetch('/api/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({delta: delta})
  });
  await refreshState();
    setStatus(delta < 0 ? 'Switched to previous item' : 'Switched to next item', 'success');
}

async function saveItem(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    setStatus(`Saving fields: ${item.instance_id || '-'}`, 'info');
    let patch = {
        class_id: parseMaybeNumber(document.getElementById('f_class_id').value),
        class_name: document.getElementById('f_class_name').value,
        point_count: parseMaybeNumber(document.getElementById('f_point_count').value),
        review_note: document.getElementById('f_review_note').value,
        auto_label_subcategory: document.getElementById('f_auto_label_name').value || null,
        auto_label_name: document.getElementById('f_auto_label_name').value || null,
        auto_label_confidence: parseMaybeNumber(document.getElementById('f_auto_label_confidence').value),
        auto_label_category: document.getElementById('f_auto_label_landmark_type').value || null,
        auto_label_landmark_type: document.getElementById('f_auto_label_landmark_type').value || null,
        auto_label_description: document.getElementById('f_auto_label_landmark_description').value || null,
        auto_label_landmark_description: document.getElementById('f_auto_label_landmark_description').value || null,
        landmark_category: document.getElementById('f_landmark_category').value || null,
        landmark_subcategory: document.getElementById('f_landmark_subcategory').value || null,
        landmark_description: document.getElementById('f_landmark_description').value || null,
        landmark_note: document.getElementById('f_landmark_note').value || null,
    };
    const action = document.getElementById('f_review_action').value;
    await fetch('/api/decide', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance_id:item.instance_id, action:action, note: patch.review_note, goto_next:false})
    });

    await fetch('/api/update_item', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance_id:item.instance_id, fields:patch})
    });
    await refreshState();
    setStatus(`Fields saved: ${item.instance_id || '-'}`, 'success');
}

async function approveAutoLabel(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    const manualNote = document.getElementById('f_landmark_note').value || '';
    const patch = {
        landmark_category: document.getElementById('f_auto_label_landmark_type').value || null,
        landmark_subcategory: document.getElementById('f_auto_label_name').value || null,
        landmark_description: document.getElementById('f_auto_label_landmark_description').value || null,
        landmark_decision: 'approved',
        landmark_note: manualNote || 'approved auto label',
        annotation_status: 'labeled',
    };
    setStatus(`Reviewing auto label: ${item.instance_id || '-'}`, 'info');
    await fetch('/api/update_item', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance_id:item.instance_id, fields:patch})
    });
    await refreshState();
    setStatus(`Auto label reviewed: ${item.instance_id || '-'}`, 'success');
}

async function saveManualReview(){
    if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
    const item = state.current.item || {};
    const landmarkCategory = document.getElementById('f_landmark_category').value || null;
    const landmarkSubcategory = document.getElementById('f_landmark_subcategory').value || null;
    const landmarkDescription = document.getElementById('f_landmark_description').value || null;
    const rawNote = document.getElementById('f_landmark_note').value || '';
    const normalizedRawNote = String(rawNote || '').trim().toLowerCase();
    const manualNote = (!rawNote || normalizedRawNote === 'approved auto label')
        ? 'manual correction'
        : `manual correction: ${rawNote}`;
    const patch = {
        landmark_category: landmarkCategory,
        landmark_subcategory: landmarkSubcategory,
        landmark_description: landmarkDescription,
        landmark_decision: 'manual',
        landmark_note: manualNote,
        annotation_status: 'labeled',
    };
    setStatus(`Saving manual revision: ${item.instance_id || '-'}`, 'info');
    await fetch('/api/update_item', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance_id:item.instance_id, fields:patch})
    });
    await refreshState();
    setStatus(`Manual revision saved: ${item.instance_id || '-'}`, 'success');
}

async function syncClassName(){
        if(!state || !state.current || state.current.index < 0){ alert('Please select a landmark from the left panel first.'); return; }
        const item = state.current.item || {};
        const className = document.getElementById('f_class_name').value || '';
        setStatus(`Syncing class name: ${className || '-'} ...`, 'info');
        const res = await fetch('/api/sync_class_name', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify({instance_id:item.instance_id, class_name:className})
        });
        const data = await res.json();
        if(!data.ok){ setStatus('Failed to sync class name', 'error'); alert('Sync failed: ' + (data.error || 'unknown')); return; }
        alert(`Synced ${data.affected} landmarks.`);
        await refreshState();
        setStatus(`Class name synced. Updated ${data.affected} items.`, 'success');
}

initTheme();
refreshState();
</script>
</body>
</html>"""

    @app.get("/raw/<path:filename>")
    def preview_image(filename: str):
        for root in raw_root_candidates:
            fp = (root / filename).resolve()
            if root in fp.parents and fp.exists() and fp.is_file():
                compress_enabled = web_img_compress_default
                max_w = web_img_max_width_default
                max_h = web_img_max_height_default
                quality = web_img_quality_default
                if request is not None:
                    compress_raw = str(request.args.get("compress", "1" if web_img_compress_default else "0") or "").strip().lower()
                    if compress_raw in {"0", "false", "no", "off"}:
                        compress_enabled = False
                    elif compress_raw in {"1", "true", "yes", "on"}:
                        compress_enabled = True
                    try:
                        max_w = max(1, int(request.args.get("max_w", max_w) or max_w))
                    except Exception:
                        max_w = web_img_max_width_default
                    try:
                        max_h = max(1, int(request.args.get("max_h", max_h) or max_h))
                    except Exception:
                        max_h = web_img_max_height_default
                    try:
                        quality = int(np.clip(int(request.args.get("quality", quality) or quality), 40, 100))
                    except Exception:
                        quality = web_img_quality_default

                if compress_enabled and Response is not None:
                    try:
                        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                        if img is not None:
                            h, w = img.shape[:2]
                            if h > 0 and w > 0:
                                scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
                                if scale < 0.999:
                                    new_w = max(1, int(round(float(w) * scale)))
                                    new_h = max(1, int(round(float(h) * scale)))
                                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                            if ok:
                                return Response(enc.tobytes(), mimetype="image/jpeg")
                    except Exception:
                        pass
                return send_file(str(fp))
        return jsonify({"ok": False, "error": "image_not_found", "filename": filename}), 404

    @app.get("/api/list")
    def api_list():
        return jsonify({"ok": True, "items": _build_list_items()})

    @app.get("/api/state")
    def api_state():
        nonlocal current_index
        if current_index < 0:
            return jsonify(
                {
                    "ok": True,
                    "scene_id": scene_id,
                    "counts": _calc_counts(),
                    "web_display": {
                        "compress_enabled": bool(web_img_compress_default),
                        "max_width": int(web_img_max_width_default),
                        "max_height": int(web_img_max_height_default),
                        "jpeg_quality": int(web_img_quality_default),
                        "show_bbox": bool(web_show_bbox_default),
                    },
                    "auto_label_task": _export_auto_label_task(),
                    "current": {"index": -1, "item": {}, "pcd_urls": [], "rgb_urls": []},
                }
            )
        current_index = max(0, min(len(instances) - 1, current_index))
        return jsonify(
            {
                "ok": True,
                "scene_id": scene_id,
                "counts": _calc_counts(),
                "web_display": {
                    "compress_enabled": bool(web_img_compress_default),
                    "max_width": int(web_img_max_width_default),
                    "max_height": int(web_img_max_height_default),
                    "jpeg_quality": int(web_img_quality_default),
                    "show_bbox": bool(web_show_bbox_default),
                },
                "auto_label_task": _export_auto_label_task(),
                "current": _build_item(current_index),
            }
        )

    @app.post("/api/move")
    def api_move():
        nonlocal current_index
        payload = request.get_json(silent=True) or {}
        delta = int(payload.get("delta", 0) or 0)
        if current_index < 0:
            current_index = 0
        else:
            current_index = max(0, min(len(instances) - 1, current_index + delta))
        _write_snapshot(trigger="move")
        return jsonify({"ok": True, "current_index": current_index})

    @app.post("/api/select")
    def api_select():
        nonlocal current_index
        payload = request.get_json(silent=True) or {}
        if "index" in payload:
            idx = int(payload.get("index", 0) or 0)
            current_index = max(0, min(len(instances) - 1, idx))
        else:
            instance_id = str(payload.get("instance_id", "") or "")
            found = next((i for i, v in enumerate(instances) if str(v.get("instance_id", "")) == instance_id), None)
            if found is not None:
                current_index = int(found)
        _write_snapshot(trigger="select")
        return jsonify({"ok": True, "current_index": current_index})

    @app.post("/api/sync_class_name")
    def api_sync_class_name():
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        class_name_new = str(payload.get("class_name", "") or "").strip()
        if not instance_id or not class_name_new:
            return jsonify({"ok": False, "error": "missing instance_id or class_name"}), 400

        base_item = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(base_item, dict):
            return jsonify({"ok": False, "error": "instance not found"}), 404

        class_id_val = base_item.get("class_id", None)
        if class_id_val is None:
            return jsonify({"ok": False, "error": "instance class_id missing"}), 400

        affected = 0
        for item in instances:
            if item.get("class_id", None) != class_id_val:
                continue
            iid = str(item.get("instance_id", ""))
            if iid not in instance_overrides:
                instance_overrides[iid] = {}
            instance_overrides[iid]["class_name"] = class_name_new
            affected += 1

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_sync_class_name",
                "class_id": class_id_val,
                "class_name": class_name_new,
                "affected": affected,
                "trigger_instance_id": instance_id,
            },
        )
        _write_snapshot(trigger="sync_class_name")
        return jsonify({"ok": True, "affected": affected, "class_id": class_id_val, "class_name": class_name_new})

    @app.post("/api/decide")
    def api_decide():
        nonlocal current_index
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        action = str(payload.get("action", "")).strip().lower()
        note = str(payload.get("note", "") or "")
        goto_next = bool(payload.get("goto_next", False))

        if action not in {"keep", "drop", "clear"}:
            return jsonify({"ok": False, "error": "invalid action"}), 400
        if not instance_id:
            return jsonify({"ok": False, "error": "missing instance_id"}), 400

        if action == "keep":
            effective_item = _get_effective_item_by_instance_id(instance_id)
            if effective_item is None:
                return jsonify({"ok": False, "error": "instance not found"}), 404
            if not _has_query_view(effective_item, view_type="rgb", require_valid=True):
                return jsonify({"ok": False, "error": "keep_requires_main_view"}), 400

        decided_index = next((i for i, v in enumerate(instances) if str(v.get("instance_id", "")) == instance_id), None)
        if decided_index is not None:
            current_index = int(decided_index)

        if action == "clear":
            decision_map_raw.pop(instance_id, None)
            final_action = default_action
        else:
            decision_map_raw[instance_id] = {"action": action, "note": note}
            final_action = action

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_instance_decided",
                "instance_id": instance_id,
                "action": final_action,
                "note": note,
            },
        )

        _write_snapshot(trigger="decide")
        if goto_next:
            current_index = _next_index_by_class_order(current_index)
        return jsonify({"ok": True, "current_index": current_index})

    @app.post("/api/set_view_direction")
    def api_set_view_direction():
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        view_index = int(payload.get("view_index", 0))
        direction = _normalize_view_direction(payload.get("direction", None))
        view_type = str(payload.get("view_type", "rgb") or "").strip().lower()
        mark_main = bool(payload.get("mark_main", True))

        if not instance_id or direction is None:
            return jsonify({"ok": False, "error": "invalid instance_id or direction"}), 400

        base_item = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(base_item, dict):
            return jsonify({"ok": False, "error": "instance not found"}), 404

        views_key = f"{view_type}_views"
        views = list(base_item.get(views_key, []) or [])

        if view_index < 0 or view_index >= len(views):
            return jsonify({"ok": False, "error": "invalid view_index"}), 400

        if instance_id not in instance_overrides:
            instance_overrides[instance_id] = {}

        # Preserve the views structure in overrides
        if views_key not in instance_overrides[instance_id]:
            instance_overrides[instance_id][views_key] = [dict(v) if isinstance(v, dict) else v for v in views]

        override_views = instance_overrides[instance_id][views_key]
        while len(override_views) < len(views):
            j = len(override_views)
            base_view = views[j] if j < len(views) and isinstance(views[j], dict) else {}
            override_views.append(dict(base_view) if isinstance(base_view, dict) else {})

        assigned_directions = _assign_view_directions_by_yaw(
            base_direction=direction,
            selected_view_index=view_index,
            views=override_views,
        )
        if assigned_directions is None:
            assigned_directions = _assign_view_directions(
                base_direction=direction,
                selected_view_index=view_index,
                view_count=len(override_views),
            )

        for i, auto_direction in enumerate(assigned_directions):
            if not isinstance(override_views[i], dict):
                override_views[i] = {}
            override_views[i]["view_direction"] = auto_direction
            override_views[i]["view_direction_auto"] = True
            if mark_main:
                override_views[i]["is_query_view"] = bool(i == view_index)
        
        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_set_view_direction",
                "instance_id": instance_id,
                "view_type": view_type,
                "view_index": view_index,
                "direction": direction,
                "mark_main": mark_main,
                "assigned_directions": assigned_directions,
            },
        )
        _write_snapshot(trigger="set_view_direction")
        return jsonify({
            "ok": True,
            "direction": direction,
            "assigned_directions": assigned_directions,
            "main_view_index": view_index if mark_main else None,
        })

    @app.post("/api/set_rgb_view_validity")
    def api_set_rgb_view_validity():
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        view_index = int(payload.get("view_index", -1))
        is_valid = bool(payload.get("is_valid", True))
        view_type = str(payload.get("view_type", "rgb") or "").strip().lower()

        if not instance_id or view_index < 0:
            return jsonify({"ok": False, "error": "invalid instance_id or view_index"}), 400

        base_item = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(base_item, dict):
            return jsonify({"ok": False, "error": "instance not found"}), 404

        views_key = f"{view_type}_views"
        base_views = list(base_item.get(views_key, []) or [])
        if view_index >= len(base_views):
            return jsonify({"ok": False, "error": "invalid view_index"}), 400

        if instance_id not in instance_overrides:
            instance_overrides[instance_id] = {}
        if views_key not in instance_overrides[instance_id]:
            instance_overrides[instance_id][views_key] = [dict(v) if isinstance(v, dict) else {} for v in base_views]

        override_views = instance_overrides[instance_id][views_key]
        while len(override_views) < len(base_views):
            j = len(override_views)
            base_view = base_views[j] if j < len(base_views) and isinstance(base_views[j], dict) else {}
            override_views.append(dict(base_view) if isinstance(base_view, dict) else {})

        if not isinstance(override_views[view_index], dict):
            override_views[view_index] = {}

        path_text = str(
            override_views[view_index].get(
                "path",
                base_views[view_index].get("path", "") if isinstance(base_views[view_index], dict) else "",
            )
            or ""
        ).strip()
        effective_valid = bool(is_valid and bool(path_text))
        override_views[view_index]["is_valid"] = effective_valid

        if not effective_valid and bool(override_views[view_index].get("is_query_view", False)):
            override_views[view_index]["is_query_view"] = False

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_set_rgb_view_validity",
                "instance_id": instance_id,
                "view_type": view_type,
                "view_index": view_index,
                "is_valid": effective_valid,
            },
        )
        _write_snapshot(trigger="set_rgb_view_validity")
        return jsonify({"ok": True, "instance_id": instance_id, "view_index": view_index, "is_valid": effective_valid})

    @app.post("/api/auto_label_single")
    def api_auto_label_single():
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        if not instance_id:
            return jsonify({"ok": False, "error": "missing instance_id"}), 400

        found = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(found, dict):
            return jsonify({"ok": False, "error": "instance not found"}), 404

        targets, stats = _prepare_auto_label_targets("single", instance_id=instance_id)
        if len(targets) <= 0:
            return jsonify({"ok": False, "error": "only_keep_instances_can_be_auto_labeled"}), 400

        started, task_state, error = _start_auto_label_job("single", targets)
        if not started:
            return jsonify({"ok": False, "error": error or "auto_label_job_running", "task": task_state}), 409
        return jsonify({"ok": True, "started": True, "task": task_state, "eligible": stats.get("eligible", 0)})

    @app.post("/api/auto_label_class")
    def api_auto_label_class():
        payload = request.get_json(silent=True) or {}
        class_id_raw = payload.get("class_id", None)
        class_id_normalized = _normalize_class_id_for_match(class_id_raw)
        class_name_raw = str(payload.get("class_name", "") or "").strip()
        if class_id_normalized is None:
            return jsonify({"ok": False, "error": "missing class_id"}), 400

        targets, stats = _prepare_auto_label_targets("class", class_id_raw=class_id_normalized, class_name_raw=class_name_raw)
        if len(targets) <= 0:
            return jsonify({
                "ok": False,
                "error": "no_keep_instances_matched",
                "eligible": int(stats.get("eligible", 0)),
                "skipped_not_keep": int(stats.get("skipped_not_keep", 0)),
                "class_id": class_id_normalized,
            }), 400

        started, task_state, error = _start_auto_label_job("class", targets, class_id_raw=class_id_normalized, class_name_raw=class_name_raw)
        if not started:
            return jsonify({"ok": False, "error": error or "auto_label_job_running", "task": task_state}), 409
        return jsonify(
            {
                "ok": True,
                "started": True,
                "task": task_state,
                "class_id": class_id_normalized,
                "eligible": int(stats.get("eligible", 0)),
                "skipped_not_keep": int(stats.get("skipped_not_keep", 0)),
            }
        )

    @app.post("/api/auto_label_all")
    def api_auto_label_all():
        targets, stats = _prepare_auto_label_targets("all")
        if len(targets) <= 0:
            return jsonify(
                {
                    "ok": False,
                    "error": "no_eligible_instances",
                    "eligible": int(stats.get("eligible", 0)),
                    "skipped": int(stats.get("skipped", 0)),
                    "skipped_not_keep": int(stats.get("skipped_not_keep", 0)),
                    "skipped_existing": int(stats.get("skipped_existing", 0)),
                }
            ), 400

        started, task_state, error = _start_auto_label_job("all", targets)
        if not started:
            return jsonify({"ok": False, "error": error or "auto_label_job_running", "task": task_state}), 409
        return jsonify(
            {
                "ok": True,
                "started": True,
                "task": task_state,
                "eligible": int(stats.get("eligible", 0)),
                "skipped": int(stats.get("skipped", 0)),
                "skipped_not_keep": int(stats.get("skipped_not_keep", 0)),
                "skipped_existing": int(stats.get("skipped_existing", 0)),
            }
        )

    @app.post("/api/auto_label_cancel")
    def api_auto_label_cancel():
        with state_lock:
            if not bool(auto_label_job.get("running", False)):
                return jsonify({"ok": False, "error": "no_running_auto_label_job", "task": _export_auto_label_task()}), 400
            auto_label_job["cancel_requested"] = True
            auto_label_job["message"] = "Auto-label cancel request sent"
        return jsonify({"ok": True, "task": _export_auto_label_task()})

    @app.post("/api/clear_auto_label_single")
    def api_clear_auto_label_single():
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        if not instance_id:
            return jsonify({"ok": False, "error": "missing instance_id"}), 400

        found = next((it for it in instances if str(it.get("instance_id", "")) == instance_id), None)
        if not isinstance(found, dict):
            return jsonify({"ok": False, "error": "instance not found"}), 404

        fields = _build_clear_auto_label_fields()
        if instance_id not in instance_overrides:
            instance_overrides[instance_id] = {}
        instance_overrides[instance_id].update(fields)

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_clear_auto_label_single",
                "instance_id": instance_id,
                "fields": fields,
            },
        )
        _write_snapshot(trigger="clear_auto_label_single")
        return jsonify({"ok": True, "updated": 1, "instance_id": instance_id})

    @app.post("/api/clear_auto_label_class")
    def api_clear_auto_label_class():
        payload = request.get_json(silent=True) or {}
        class_id_raw = payload.get("class_id", None)
        class_id_normalized = _normalize_class_id_for_match(class_id_raw)
        class_name_raw = str(payload.get("class_name", "") or "").strip()
        if class_id_normalized is None:
            return jsonify({"ok": False, "error": "missing class_id"}), 400

        fields = _build_clear_auto_label_fields()
        updated = 0
        for item in instances:
            if not isinstance(item, dict):
                continue
            instance_id = str(item.get("instance_id", ""))
            effective_item = dict(item)
            if instance_id in instance_overrides and isinstance(instance_overrides[instance_id], dict):
                effective_item.update(instance_overrides[instance_id])

            if not _class_id_matches(effective_item, class_id_normalized):
                continue

            if instance_id not in instance_overrides:
                instance_overrides[instance_id] = {}
            instance_overrides[instance_id].update(fields)
            updated += 1

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_clear_auto_label_class",
                "class_id": class_id_normalized,
                "class_name": class_name_raw,
                "updated": updated,
            },
        )
        _write_snapshot(trigger="clear_auto_label_class")
        return jsonify({"ok": True, "updated": int(updated), "class_id": class_id_normalized})

    @app.post("/api/clear_auto_label_all")
    def api_clear_auto_label_all():
        fields = _build_clear_auto_label_fields()
        updated = 0
        for item in instances:
            if not isinstance(item, dict):
                continue
            instance_id = str(item.get("instance_id", ""))
            if instance_id not in instance_overrides:
                instance_overrides[instance_id] = {}
            instance_overrides[instance_id].update(fields)
            updated += 1

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_clear_auto_label_all",
                "updated": updated,
            },
        )
        _write_snapshot(trigger="clear_auto_label_all")
        return jsonify({"ok": True, "updated": int(updated)})

    @app.post("/api/update_item")
    def api_update_item():
        payload = request.get_json(silent=True) or {}
        instance_id = str(payload.get("instance_id", "") or "")
        fields = payload.get("fields", {})
        if not instance_id:
            return jsonify({"ok": False, "error": "missing instance_id"}), 400
        if not isinstance(fields, dict):
            return jsonify({"ok": False, "error": "fields must be object"}), 400

        clean_fields = dict(fields)
        clean_fields.pop("instance_id", None)
        if instance_id not in instance_overrides:
            instance_overrides[instance_id] = {}
        instance_overrides[instance_id].update(clean_fields)
        _sync_annotation_status_override(instance_id)

        append_jsonl(
            review_log_path,
            {
                "ts": time.time(),
                "scene_id": scene_id,
                "event": "web_instance_updated",
                "instance_id": instance_id,
                "fields": clean_fields,
            },
        )
        _write_snapshot(trigger="update_item")
        return jsonify({"ok": True})

    host = str(args.host or stage2_cfg.get("review_web_host", "127.0.0.1"))
    port = int(args.port if args.port is not None else stage2_cfg.get("review_web_port", 8765))
    logger.info(
        f"web review server started scene={scene_id} url=http://{host}:{port} total={len(instances)}"
    )
    app.run(host=host, port=port, debug=False, use_reloader=False)
    return {
        "ok": True,
        "mode": "review_instances_web",
        "scene_id": scene_id,
        "valid_instances_json": str(valid_instances_path.as_posix()),
        "review_log_jsonl": str(review_log_path.as_posix()),
    }


def auto_label(scene_id: str, args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    logger = StageLogger("stage2.auto_label")
    t_start = time.time()

    stage2_cfg = dict(config.get("stage2", {}) or {})
    stage2_cfg["_full_config"] = config
    scene_root = _resolve_scene_root(config=config, scene_id=scene_id)
    raw_root = scene_root / _resolve_output_dir_name(config, key="stage2_raw_dir", default="landmarks_raw")
    review_root = scene_root / _resolve_output_dir_name(config, key="stage2_review_dir", default="landmarks_review")
    auto_root = scene_root / _resolve_output_dir_name(config, key="stage2_auto_dir", default="landmarks_auto")
    landmarks_root = scene_root / _resolve_output_dir_name(config, key="stage2_landmarks_dir", default="landmarks")
    ensure_dir(review_root)
    ensure_dir(auto_root)
    ensure_dir(landmarks_root)
    stage2_cfg["_scene_id"] = scene_id
    stage2_cfg["_scene_root"] = str(scene_root)
    stage2_cfg["_review_root"] = str(review_root)

    instances_json_path = resolve_scene_artifact_path(raw_root, scene_id, ".instances.json")
    valid_instances_path = resolve_scene_artifact_path(review_root, scene_id, ".valid_instances.json")
    auto_label_dump_path = resolve_scene_artifact_path(auto_root, scene_id, ".auto_label.json")
    landmarks_json_path = resolve_scene_artifact_path(landmarks_root, scene_id, ".json")

    source_payload = read_json_if_exists(valid_instances_path, default={})
    source_name = "valid_instances"
    source_instances = list(source_payload.get("valid_instances", []) or []) if isinstance(source_payload, dict) else []
    if len(source_instances) == 0:
        raise FileNotFoundError(
            f"no keep instances found for auto_label. expected step2 keep list at: {valid_instances_path}"
        )

    sample_size = int(getattr(args, "auto_label_sample_size", 0) or 0)
    if sample_size > 0 and sample_size < len(source_instances):
        sample_seed = int(getattr(args, "auto_label_sample_seed", 42) or 42)
        rng = random.Random(sample_seed)
        source_instances = rng.sample(source_instances, sample_size)

    labeled_instances: list[dict[str, Any]] = []
    for item in source_instances:
        if not isinstance(item, dict):
            continue
        labeled = dict(item)
        labeled.update(_build_auto_label_fields(labeled, scope="cli_all", stage2_cfg=stage2_cfg))
        labeled_instances.append(labeled)

    out_payload = {
        "scene_id": scene_id,
        "source": source_name,
        "source_path": str(valid_instances_path.as_posix()),
        "summary": {
            "instances_total": int(len(source_instances)),
            "auto_labeled": int(len(labeled_instances)),
            "runtime_sec": float(time.time() - t_start),
        },
        "landmarks": labeled_instances,
    }
    write_json(auto_label_dump_path, out_payload)
    write_json(landmarks_json_path, out_payload)

    runtime_sec = max(1e-6, float(time.time() - t_start))
    logger.info(
        f"done source={source_name} total={len(source_instances)} auto_labeled={len(labeled_instances)} runtime={runtime_sec:.1f}s"
    )
    return {
        "ok": True,
        "mode": "auto_label",
        "scene_id": scene_id,
        "source": source_name,
        "instances_total": int(len(source_instances)),
        "auto_labeled": int(len(labeled_instances)),
        "auto_label_json": str(auto_label_dump_path.as_posix()),
        "landmarks_json": str(landmarks_json_path.as_posix()),
        "runtime_sec": runtime_sec,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UAV-DualCog Stage2 landmark labeling")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["collect_instances", "review_instances", "review_instances_web", "auto_label", "all"],
    )
    parser.add_argument("--pcd", type=str, default=None)
    parser.add_argument("--min-points", type=int, default=None)
    parser.add_argument("--review-decisions", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--auto-label-sample-size", type=int, default=0)
    parser.add_argument("--auto-label-sample-seed", type=int, default=42)
    parser.add_argument(
        "--auto-label-debug-save-bbox-img",
        action="store_true",
        help="Save debug images with red bbox and direction during auto_label (to landmarks_review/auto_label_debug/)"
    )
    return parser.parse_args()


def _resolve_mode(args: argparse.Namespace, config: dict[str, Any]) -> str:
    if args.mode:
        return str(args.mode)
    stage2_cfg = config.get("stage2", {}) or {}
    mode = str(stage2_cfg.get("mode", "collect_instances"))
    alias = {
        "surface_attr": "collect_instances",
        "landmarks": "collect_instances",
    }
    return alias.get(mode, mode)


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    task_cfg = config.get("task", {}) or {}
    scene_id = str(args.scene_id or task_cfg.get("scene_id", "env_airsim_16"))
    mode = _resolve_mode(args, config)

    if mode == "collect_instances":
        result = collect_instances(scene_id=scene_id, args=args, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if mode == "review_instances":
        result = review_instances(scene_id=scene_id, args=args, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if mode == "review_instances_web":
        review_instances_web(scene_id=scene_id, args=args, config=config)
        return

    if mode == "all":
        out1 = collect_instances(scene_id=scene_id, args=args, config=config)
        out2 = review_instances(scene_id=scene_id, args=args, config=config)
        print(json.dumps({"ok": True, "mode": "all", "scene_id": scene_id, "step1": out1, "step2": out2}, ensure_ascii=False, indent=2))
        return

    if mode == "auto_label":
        result = auto_label(scene_id=scene_id, args=args, config=config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
