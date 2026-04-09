#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
COMMON_STAGE_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "flightmvstg" / "common_stage_configs.yaml"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage3_generate_traj import (
    _analyze_final_task_visibility_metadata,
    _load_final_task_inputs_from_mission_dir,
    generate_preview_assets_for_mission,
    generate_single,
    load_yaml,
    record_scene_videos_cli,
)
from stage3_task_suite import generate_manifest as generate_stage3_manifest
from stage3_task_suite import _resolve_stage3_layout as resolve_stage3_layout
from stage3_task_suite import _stage3_best_run_progress as stage3_best_run_progress
from stage3_task_suite import recompute_report_from_run_dir as recompute_stage3_report_from_run_dir
from stage3_task_suite import run_experiment_once as run_stage3_experiment_once
from stage4_qa_generate_and_eval import generate_manifest as generate_stage4_manifest
from stage4_qa_generate_and_eval import render_manifest_assets as render_stage4_manifest_assets
from stage4_qa_generate_and_eval import _resolve_stage4_root as resolve_stage4_root
from stage4_qa_generate_and_eval import _stage4_best_run_progress as stage4_best_run_progress
from stage4_qa_generate_and_eval import recompute_report_from_run_dir as recompute_stage4_report_from_run_dir
from stage4_qa_generate_and_eval import run_experiment as run_stage4_experiment
from stage4_qa_generate_and_eval import _safe_name
from trajectory.behaviors import ELEMENT_LIBRARY, SET_LIBRARY
from progress_utils import ProgressBar, StageLogger
from pipeline_common import (
    cleanup_airsim_processes,
    resolve_task_pipeline_base_dir,
    resolve_task_pipeline_scene_root,
    resolve_task_pipeline_task_root,
)


TASK_PIPELINE_META_DIRNAME = "task_pipeline"


class PipelineLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_only(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _emit(self, level: str, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"[{ts}][task_pipeline][{level}] {message}"
        print(line, flush=True)
        self._append_only(line)

    def info(self, message: str) -> None:
        self._emit("INFO", message)

    def warn(self, message: str) -> None:
        self._emit("WARN", message)

    def error(self, message: str) -> None:
        self._emit("ERROR", message)

    def mirror_progress(self, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self._append_only(f"[{ts}][task_pipeline][INFO] {message}")


def _format_experiment_detail(
    *,
    model_name: str,
    model_width: int,
    unit_label: str,
    unit_value: str,
    request_status: str,
    parse_ok: bool,
    latency_ms: Any = None,
) -> str:
    detail = (
        f"model={str(model_name):<{model_width}} "
        f"{unit_label}={unit_value} "
        f"status={str(request_status):<5} "
        f"parse_ok={str(bool(parse_ok)):<5}"
    )
    if latency_ms is not None:
        detail += f" latency_ms={round(float(latency_ms), 1):>7}"
    return detail


def _resolve_experiment_manifest_path_for_scene(scene_cfg_path: Path, spec: dict[str, Any]) -> Path | None:
    scene_cfg = load_yaml(scene_cfg_path)
    merged_cfg = _task_pipeline_scene_task_cfg(scene_cfg, spec)
    stage_mode = str(spec.get("stage", "both") or "both").strip().lower()
    if stage_mode == "stage3":
        path = _resolve_latest_stage3_manifest_path(merged_cfg)
        return path if path.exists() else None
    if stage_mode == "stage4":
        path = _resolve_latest_stage4_manifest_path(merged_cfg)
        return path if path.exists() else None
    return None


def _remaining_experiment_samples_for_scene_model(scene_cfg_path: Path, spec: dict[str, Any], model_name: str) -> int:
    scene_cfg = load_yaml(scene_cfg_path)
    merged_cfg = _task_pipeline_scene_task_cfg(scene_cfg, spec)
    stage_mode = str(spec.get("stage", "both") or "both").strip().lower()
    engine, scene_id = _scene_identity(merged_cfg)
    remaining_total = 0
    if stage_mode in {"stage3", "both"}:
        manifest_path = _resolve_latest_stage3_manifest_path(merged_cfg)
        if manifest_path.exists():
            total = _load_manifest_sample_count(manifest_path)
            layouts = [resolve_stage3_layout(merged_cfg, scene_id=scene_id, engine=engine)]
            progress = stage3_best_run_progress(layouts, scene_id, model_name, str(manifest_path))
            completed = min(total, int(progress.get("completed", 0) or 0))
            remaining_total += max(0, total - completed)
    if stage_mode in {"stage4", "both"}:
        manifest_path = _resolve_latest_stage4_manifest_path(merged_cfg)
        if manifest_path.exists():
            total = _load_manifest_sample_count(manifest_path)
            scene_root = _resolve_artifact_scene_root(merged_cfg)
            stage4_root = resolve_stage4_root(merged_cfg, scene_root=scene_root)
            progress = stage4_best_run_progress([stage4_root], scene_id, model_name, str(manifest_path))
            completed = min(total, int(progress.get("completed", 0) or 0))
            remaining_total += max(0, total - completed)
    return int(remaining_total)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _expand_scene_config_paths(values: Any) -> list[Path]:
    if isinstance(values, str):
        raw_items = [values]
    else:
        raw_items = [str(x).strip() for x in list(values or []) if str(x).strip()]
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_items:
        text = str(raw).strip()
        if not text:
            continue
        if text.lower() == "all":
            matches = sorted((WORKSPACE_ROOT / "configs" / "flightmvstg").glob("task_airsim_env_*.yaml"))
        elif "*" in text or "?" in text or "[" in text:
            pattern = Path(text)
            matches = sorted((WORKSPACE_ROOT / pattern).parent.glob(pattern.name)) if not pattern.is_absolute() else sorted(pattern.parent.glob(pattern.name))
        else:
            matches = [Path(text)]
        for path in matches:
            resolved = path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            out.append(resolved)
    return out


def _config_path_for_scene_id(scene_id: str, *, engine: str = "airsim") -> Path:
    return (WORKSPACE_ROOT / "configs" / "flightmvstg" / f"task_{engine}_{scene_id}.yaml").resolve()


def _resolve_scene_root(cfg: dict[str, Any]) -> Path:
    task_cfg = cfg.get("task", {}) or {}
    base_dir = Path(str(task_cfg.get("base_dir", "scene_data")))
    if not base_dir.is_absolute():
        base_dir = (WORKSPACE_ROOT / base_dir).resolve()
    scene_dir_name = str(task_cfg.get("scene_dir_name", "") or "").strip()
    if scene_dir_name:
        return base_dir / scene_dir_name
    engine = str(task_cfg.get("engine", "airsim") or "airsim").strip().lower()
    scene_id = str(task_cfg.get("scene_id", "") or "").strip()
    return base_dir / f"{engine}_{scene_id}"


def _task_pipeline_name(spec: dict[str, Any]) -> str:
    return str(spec.get("task_name", "") or spec.get("pipeline_name", "") or "default_task").strip()


def _task_pipeline_task_root(spec: dict[str, Any]) -> Path:
    cfg = {"task_pipeline": {"task_name": _task_pipeline_name(spec), "root_dir": str(spec.get("task_pipeline_root_dir", "task_pipeline_data") or "task_pipeline_data")}}
    root = resolve_task_pipeline_task_root(cfg, workspace_root=WORKSPACE_ROOT)
    if root is None:
        raise RuntimeError("task_pipeline_task_root_unresolved")
    return root


def _task_pipeline_meta_root(spec: dict[str, Any]) -> Path:
    return _task_pipeline_task_root(spec) / TASK_PIPELINE_META_DIRNAME


def _task_pipeline_scene_task_cfg(scene_cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(scene_cfg)
    cfg.setdefault("task_pipeline", {})
    cfg["task_pipeline"]["task_name"] = _task_pipeline_name(spec)
    cfg["task_pipeline"]["root_dir"] = str(spec.get("task_pipeline_root_dir", "task_pipeline_data") or "task_pipeline_data")
    for section_name in ("stage3", "stage4"):
        section_override = dict(spec.get(section_name, {}) or {})
        if section_override:
            cfg[section_name] = _deep_merge(dict(cfg.get(section_name, {}) or {}), section_override)
    return cfg


def _format_stage3_render_terminal_detail(*, scene_id: str, done: int, total: int, detail: str) -> str:
    text = str(detail or "").strip()
    latest_traj = "-"
    if text.startswith("rendered "):
        parts = text.split()
        if len(parts) >= 2:
            latest_traj = parts[1]
    elif text:
        latest_traj = text
    return f"scene={scene_id} scene_tasks={int(done)}/{int(total)} latest={latest_traj}"


def _scene_identity(scene_cfg: dict[str, Any]) -> tuple[str, str]:
    task = scene_cfg.get("task", {}) or {}
    engine = str(task.get("engine", "airsim") or "airsim").strip().lower()
    scene_id = str(task.get("scene_id", "") or "").strip()
    return engine, scene_id


def _count_or_all(value: Any, *, fallback: int, total_available: int | None = None) -> int:
    text = str(value).strip().lower() if value is not None else ""
    if text == "all":
        return int(total_available if total_available is not None else fallback)
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _load_valid_instances(scene_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scene_root = _resolve_scene_root(scene_cfg)
    scene_id = str((scene_cfg.get("task", {}) or {}).get("scene_id", "") or "").strip()
    review_dir = scene_root / str((scene_cfg.get("output_layout", {}) or {}).get("stage2_review_dir", "landmarks_review"))
    payload = json.loads((review_dir / f"{scene_id}.valid_instances.json").read_text(encoding="utf-8"))
    rows = [it for it in list(payload.get("valid_instances", []) or []) if isinstance(it, dict)]
    kept = [it for it in rows if str(it.get("review_action", "") or "").strip().lower() in {"keep", ""}]
    kept = [it for it in kept if str(it.get("annotation_status", "") or "").strip().lower() == "labeled"]
    return [
        it
        for it in kept
        if str(it.get("landmark_description", "") or it.get("description", "") or "").strip()
    ]


def _resolve_global_landmark_list_dir(spec: dict[str, Any]) -> Path:
    out = _task_pipeline_meta_root(spec) / "landmark_lists"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_global_landmark_list(
    *,
    spec: dict[str, Any],
    artifact_name: str,
    seed: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    out_dir = _resolve_global_landmark_list_dir(spec)
    payload = {
        "artifact_name": str(artifact_name),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "seed": int(seed),
        "selected_count": int(len(entries)),
        "selected_landmarks": entries,
    }
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = out_dir / f"{artifact_name}.{len(entries)}items.{ts}.json"
    latest_path = out_dir / f"{artifact_name}.latest.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(out_path), "latest_path": str(latest_path), "payload": payload}


def _build_global_list_entries(
    *,
    config_path: Path,
    scene_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    engine, scene_id = _scene_identity(scene_cfg)
    out = []
    for row in list(rows or []):
        instance_id = str(row.get("instance_id", "") or "").strip()
        if not instance_id:
            continue
        out.append(
            {
                "engine": engine,
                "scene_id": scene_id,
                "config_path": str(config_path.resolve()),
                "instance_id": instance_id,
            }
        )
    return out


def _load_common_stage_cfg() -> dict[str, Any]:
    if not COMMON_STAGE_CONFIG_PATH.exists():
        return {}
    try:
        return load_yaml(COMMON_STAGE_CONFIG_PATH)
    except Exception:
        return {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out.get(key, {}) or {}), value)
        else:
            out[key] = value
    return out


def _uniform_sample_by_subcategory(entries: list[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entries:
        key = str(row.get("landmark_subcategory") or row.get("class_name") or "unknown")
        groups[key].append(row)
    buckets = {k: list(v) for k, v in groups.items()}
    for rows in buckets.values():
        rng.shuffle(rows)
    ordered_keys = sorted(buckets.keys())
    selected: list[dict[str, Any]] = []
    while len(selected) < int(count):
        progressed = False
        for key in ordered_keys:
            if not buckets[key]:
                continue
            selected.append(buckets[key].pop())
            progressed = True
            if len(selected) >= int(count):
                break
        if not progressed:
            break
    return selected


def _single_landmark_set_keys() -> list[str]:
    return [
        key
        for key, spec in SET_LIBRARY.items()
        if str(spec.get("scope", "")) == "single-landmark" and not str(key).startswith("atomic_")
    ]


def _single_landmark_component_set_keys() -> list[str]:
    return [key for key, spec in SET_LIBRARY.items() if str(spec.get("scope", "")) == "single-landmark" and bool(spec.get("multi_landmark_component", False))]


def _single_landmark_element_keys() -> list[str]:
    return sorted(ELEMENT_LIBRARY.keys())


def _resolve_artifact_scene_root(scene_cfg: dict[str, Any]) -> Path:
    engine, scene_id = _scene_identity(scene_cfg)
    return resolve_task_pipeline_scene_root(
        scene_cfg,
        scene_id=scene_id,
        engine=engine,
        workspace_root=WORKSPACE_ROOT,
    ) or _resolve_scene_root(scene_cfg)


def _resolve_stage3_missions_root(scene_cfg: dict[str, Any]) -> Path:
    scene_root = _resolve_artifact_scene_root(scene_cfg)
    output_layout = scene_cfg.get("output_layout", {}) or {}
    stage3_root = scene_root / str(output_layout.get("stage3_task_root_dir", "stage3_tasks"))
    return stage3_root / str(output_layout.get("stage3_mission_dir", "missions"))


def _discover_stage3_traj_ids(scene_cfg: dict[str, Any]) -> list[str]:
    missions_root = _resolve_stage3_missions_root(scene_cfg)
    if not missions_root.exists():
        return []
    return sorted(path.name for path in missions_root.iterdir() if path.is_dir())


def _stage3_single_family_from_report(report: dict[str, Any]) -> str | None:
    mode = str(report.get("mode", "") or "")
    if mode != "single-landmark":
        return None
    set_instance = dict(report.get("set_instance", {}) or {})
    set_id = str(set_instance.get("set_id", "") or "")
    if not set_id:
        return None
    return "atomic" if set_id.startswith("atomic_") else "composite"


def _discover_stage3_single_missions(scene_cfg: dict[str, Any]) -> dict[str, Any]:
    missions_root = _resolve_stage3_missions_root(scene_cfg)
    by_family: dict[str, list[dict[str, Any]]] = {"atomic": [], "composite": []}
    by_landmark_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not missions_root.exists():
        return {"by_family": by_family, "by_landmark_family": by_landmark_family}
    for mission_dir in sorted(path for path in missions_root.iterdir() if path.is_dir()):
        report_path = mission_dir / "constraint_report.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        family = _stage3_single_family_from_report(report)
        if family is None:
            continue
        landmark_id = str(report.get("instance_id", "") or "")
        set_instance = dict(report.get("set_instance", {}) or {})
        row = {
            "traj_id": mission_dir.name,
            "landmark_id": landmark_id,
            "family": family,
            "set_id": str(set_instance.get("set_id", "") or ""),
            "set_name": str(set_instance.get("set_name", "") or ""),
            "needs_render": _needs_stage3_render(scene_cfg, mission_dir.name, rerender_existing=False),
        }
        by_family[family].append(row)
        by_landmark_family.setdefault((landmark_id, family), []).append(row)
    return {"by_family": by_family, "by_landmark_family": by_landmark_family}


def _load_stage3_failed_landmark_families(failed_records_path: Path, *, scene_id: str) -> dict[str, set[str]]:
    failed: dict[str, set[str]] = {"atomic": set(), "composite": set()}
    if not failed_records_path.exists():
        return failed
    for line in failed_records_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if str(row.get("scene_id", "") or "") != str(scene_id):
            continue
        if str(row.get("stage", "") or "") != "stage3":
            continue
        traj_id = str(row.get("traj_id", "") or "")
        family = "composite" if "_composite_" in traj_id else "atomic"
        landmark_id = str(row.get("landmark_id", "") or "")
        if landmark_id:
            failed[family].add(landmark_id)
    return failed


def _pick_stage3_family_keys(
    all_keys: list[str],
    *,
    requested_per_landmark: int,
    offset: int,
    existing_keys: set[str],
) -> list[str]:
    if requested_per_landmark <= 0 or not all_keys:
        return []
    fresh = [key for key in all_keys if key not in existing_keys]
    source = fresh if fresh else list(all_keys)
    picked = _cycle_pick(source, count=requested_per_landmark, offset=offset)
    if len(picked) < requested_per_landmark:
        picked.extend(_cycle_pick(all_keys, count=requested_per_landmark - len(picked), offset=offset + len(picked)))
    return list(picked[:requested_per_landmark])


def _needs_stage3_render(scene_cfg: dict[str, Any], traj_id: str, *, rerender_existing: bool) -> bool:
    if rerender_existing:
        return True
    mission_dir = _resolve_stage3_missions_root(scene_cfg) / str(traj_id)
    final_dir = mission_dir / "final_task"
    meta_path = final_dir / "task_data.json"
    video_paths = [
        final_dir / "task_rgb.mp4",
        final_dir / "task_rgb_720p.mp4",
    ]
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    video_meta = dict((meta or {}).get("video", {}) or {}) if isinstance(meta, dict) else {}
    if bool(video_meta.get("generated_without_video", False)):
        return True
    if not any(path.exists() for path in video_paths):
        return True
    first_existing = next((path for path in video_paths if path.exists()), None)
    if first_existing is None:
        return True
    if int(first_existing.stat().st_size) < 1024:
        return True
    return False


def _clear_stage_outputs(scene_cfg: dict[str, Any], *, clear_stage3: bool, clear_stage4: bool) -> list[str]:
    scene_root = _resolve_artifact_scene_root(scene_cfg)
    removed = []
    output_layout = scene_cfg.get("output_layout", {}) or {}
    if clear_stage3:
        root = scene_root / str(output_layout.get("stage3_task_root_dir", "stage3_tasks"))
        if root.exists():
            shutil.rmtree(root)
            removed.append(str(root))
    if clear_stage4:
        root = scene_root / str(output_layout.get("stage4_qa_dir", "qa"))
        if root.exists():
            shutil.rmtree(root)
            removed.append(str(root))
    return removed


def _resolve_latest_stage3_manifest_path(scene_cfg: dict[str, Any]) -> Path:
    scene_root = _resolve_artifact_scene_root(scene_cfg)
    output_layout = scene_cfg.get("output_layout", {}) or {}
    stage3_root = scene_root / str(output_layout.get("stage3_task_root_dir", "stage3_tasks"))
    dataset_dir = stage3_root / str(output_layout.get("stage3_dataset_dir", "datasets"))
    scene_id = str((scene_cfg.get("task", {}) or {}).get("scene_id", "") or "").strip()
    return dataset_dir / f"{scene_id}.latest_manifest.json"


def _resolve_latest_stage4_manifest_path(scene_cfg: dict[str, Any]) -> Path:
    scene_root = _resolve_artifact_scene_root(scene_cfg)
    output_layout = scene_cfg.get("output_layout", {}) or {}
    qa_root = scene_root / str(output_layout.get("stage4_qa_dir", "qa"))
    manifest_dir = qa_root / "manifests"
    scene_id = str((scene_cfg.get("task", {}) or {}).get("scene_id", "") or "").strip()
    return manifest_dir / f"{scene_id}.latest_manifest.json"


def _stage4_manifest_is_complete(scene_cfg: dict[str, Any], *, expected_sample_count: int) -> bool:
    manifest_path = _resolve_latest_stage4_manifest_path(scene_cfg)
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    samples = list(payload.get("samples", []) or []) if isinstance(payload, dict) else []
    if len(samples) < max(1, int(expected_sample_count or 0)):
        return False
    for row in samples:
        if not isinstance(row, dict):
            return False
        ref_path_raw = str(row.get("reference_image_with_bbox", "") or row.get("reference_image", "") or "").strip()
        if not ref_path_raw:
            return False
        ref_path = Path(ref_path_raw)
        if not ref_path.is_absolute():
            ref_path = (WORKSPACE_ROOT / ref_path).resolve()
        if not ref_path.exists():
            return False
        target_image_raw = str(row.get("target_image", "") or "").strip()
        if target_image_raw:
            target_path = Path(target_image_raw)
            if not target_path.is_absolute():
                target_path = (WORKSPACE_ROOT / target_path).resolve()
            if not target_path.exists():
                return False
        for cand in list(row.get("candidates", []) or []):
            if not isinstance(cand, dict):
                return False
            image_raw = str(cand.get("image", "") or "").strip()
            if not image_raw:
                return False
            image_path = Path(image_raw)
            if not image_path.is_absolute():
                image_path = (WORKSPACE_ROOT / image_path).resolve()
            if not image_path.exists():
                return False
    return True


def _resolve_stage3_selection_dir(scene_cfg: dict[str, Any]) -> Path:
    scene_root = _resolve_artifact_scene_root(scene_cfg)
    output_layout = scene_cfg.get("output_layout", {}) or {}
    stage3_root = scene_root / str(output_layout.get("stage3_task_root_dir", "stage3_tasks"))
    return stage3_root / "selections"


def _stage3_manifest_is_complete(scene_cfg: dict[str, Any], *, expected_sample_count: int) -> bool:
    manifest_path = _resolve_latest_stage3_manifest_path(scene_cfg)
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    samples = list(payload.get("samples", []) or []) if isinstance(payload, dict) else []
    if len(samples) < max(1, int(expected_sample_count or 0)):
        return False
    for row in samples:
        if not isinstance(row, dict):
            return False
        video_path_raw = str(row.get("video_web_path", "") or row.get("video_path", "") or "").strip()
        if not video_path_raw:
            return False
        video_path = Path(video_path_raw)
        if not video_path.is_absolute():
            video_path = (WORKSPACE_ROOT / video_path).resolve()
        if not video_path.exists():
            return False
    return True


def _resolve_stage4_selection_dir(scene_cfg: dict[str, Any]) -> Path:
    scene_root = _resolve_artifact_scene_root(scene_cfg)
    output_layout = scene_cfg.get("output_layout", {}) or {}
    qa_root = scene_root / str(output_layout.get("stage4_qa_dir", "qa"))
    return qa_root / "selections"


def _write_selection_artifact(
    *,
    output_dir: Path,
    scene_id: str,
    artifact_name: str,
    seed: int,
    requested_count: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_id": str(scene_id),
        "artifact_name": str(artifact_name),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "seed": int(seed),
        "requested_count": int(requested_count),
        "selected_count": int(len(entries)),
        "selected_landmark_ids": [str(row.get("instance_id", "") or "") for row in entries if str(row.get("instance_id", "") or "").strip()],
        "selected_landmarks": entries,
    }
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = output_dir / f"{scene_id}.{artifact_name}.{len(entries)}items.{ts}.json"
    latest_path = output_dir / f"{scene_id}.{artifact_name}.latest.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(out_path), "latest_path": str(latest_path), "payload": payload}


def _write_pipeline_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _json_safe_copy(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            if callable(item):
                continue
            out[str(key)] = _json_safe_copy(item)
        return out
    if isinstance(value, list):
        return [_json_safe_copy(item) for item in value if not callable(item)]
    if isinstance(value, tuple):
        return [_json_safe_copy(item) for item in value if not callable(item)]
    if isinstance(value, set):
        return [_json_safe_copy(item) for item in value if not callable(item)]
    if callable(value):
        return None
    return value


def _task_pipeline_run_root(spec: dict[str, Any], run_id: str) -> Path:
    return _task_pipeline_meta_root(spec) / "runs" / run_id


def _selection_entries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "instance_id": str(row.get("instance_id", "") or ""),
            }
        )
    return out


def _load_manifest_sample_count(path: str | Path) -> int:
    manifest_path = Path(str(path))
    if not manifest_path.exists():
        return 0
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    return int(len(list(payload.get("samples", []) or [])))


def _stage4_expected_sample_count(scene_cfg: dict[str, Any], spec: dict[str, Any], valid_instances: list[dict[str, Any]] | None = None) -> int:
    stage4_spec = dict(spec.get("stage4", {}) or {})
    stage4_per_landmark = dict(stage4_spec.get("per_landmark", {}) or {})
    task_types = list(stage4_spec.get("task_types", stage4_per_landmark.get("task_types", ["self_where", "self_what", "env_where", "env_how"])) or ["self_where", "self_what", "env_where", "env_how"])
    qa_samples_per_difficulty = max(1, int(stage4_per_landmark.get("qa_samples_per_difficulty", 1) or 1))
    latest_stage4_selection = _resolve_stage4_selection_dir(scene_cfg) / f"{_scene_identity(scene_cfg)[1]}.stage4_landmarks.latest.json"
    stage4_landmarks = []
    if latest_stage4_selection.exists():
        try:
            stage4_landmarks = list((json.loads(latest_stage4_selection.read_text(encoding='utf-8')) or {}).get('selected_landmarks', []) or [])
        except Exception:
            stage4_landmarks = []
    if not stage4_landmarks:
        rows = list(valid_instances or _load_valid_instances(scene_cfg) or [])
        stage4_landmark_count = _count_or_all(stage4_spec.get('landmark_count', len(rows)), fallback=len(rows), total_available=len(rows))
        stage4_landmarks = rows[:stage4_landmark_count]
    landmark_ids = [str(row.get('instance_id', '') or '') for row in stage4_landmarks if str(row.get('instance_id', '') or '').strip()]
    per_landmark_tasks = max(1, int(len(task_types)))
    per_landmark_difficulty_samples = 2 * qa_samples_per_difficulty
    return max(1, int(stage4_spec.get('sample_count', max(1, len(landmark_ids) * per_landmark_tasks * per_landmark_difficulty_samples)) or max(1, len(landmark_ids) * per_landmark_tasks * per_landmark_difficulty_samples)))


def _stage3_expected_sample_count(scene_cfg: dict[str, Any], spec: dict[str, Any], valid_instances: list[dict[str, Any]] | None = None) -> int:
    global_stage3_defaults = dict((_load_common_stage_cfg().get('stage3', {}) or {}))
    stage3_spec = dict(spec.get('stage3', {}) or {})
    stage3_single = dict(stage3_spec.get('single_landmark', {}) or {})
    sample_count = _count_or_all(spec.get('landmark_count', 50), fallback=50, total_available=len(valid_instances or []))
    single_atomic_classes_per_landmark = max(0, int(stage3_single.get('atomic_classes_per_landmark', 1) or 1))
    single_composite_classes_per_landmark = max(0, int(stage3_single.get('composite_classes_per_landmark', 1) or 1))
    single_instances_per_class = max(1, int(stage3_single.get('instances_per_class', 1) or 1))
    self_state_forms = [str(x).strip() for x in list(stage3_single.get('self_state_forms', ['self_set_instance_recognition', 'self_element_instance_recognition'])) if str(x).strip()]
    environmental_forms = [str(x).strip() for x in list(stage3_single.get('environmental_forms', ['env_visibility_reasoning'])) if str(x).strip()]
    qa_samples_per_task = max(1, int(stage3_single.get('qa_samples_per_task', 1) or 1))
    stage3_forms = list(stage3_spec.get('forms', [*self_state_forms, *environmental_forms]) or [*self_state_forms, *environmental_forms])
    pair_instances_per_group = max(1, int(stage3_spec.get('pair_instances_per_group', 1) or 1))
    triple_instances_per_group = max(1, int(stage3_spec.get('triple_instances_per_group', 1) or 1))
    latest_stage3_selection = _resolve_stage3_selection_dir(scene_cfg) / f"{_scene_identity(scene_cfg)[1]}.stage3_landmarks.latest.json"
    sampled_landmarks = []
    if latest_stage3_selection.exists():
        try:
            sampled_landmarks = list((json.loads(latest_stage3_selection.read_text(encoding='utf-8')) or {}).get('selected_landmarks', []) or [])
        except Exception:
            sampled_landmarks = []
    if not sampled_landmarks:
        rows = list(valid_instances or _load_valid_instances(scene_cfg) or [])
        stage3_landmark_count = _count_or_all(stage3_spec.get('landmark_count', sample_count), fallback=sample_count, total_available=len(rows))
        sampled_landmarks = rows[:stage3_landmark_count]
    latest_pairs = _resolve_stage3_selection_dir(scene_cfg) / f"{_scene_identity(scene_cfg)[1]}.stage3_pairs.latest.json"
    pair_groups = []
    if latest_pairs.exists():
        try:
            pair_groups = list((json.loads(latest_pairs.read_text(encoding='utf-8')) or {}).get('selected_landmarks', []) or [])
        except Exception:
            pair_groups = []
    latest_triples = _resolve_stage3_selection_dir(scene_cfg) / f"{_scene_identity(scene_cfg)[1]}.stage3_triples.latest.json"
    triple_groups = []
    if latest_triples.exists():
        try:
            triple_groups = list((json.loads(latest_triples.read_text(encoding='utf-8')) or {}).get('selected_landmarks', []) or [])
        except Exception:
            triple_groups = []
    manifest_basis_count = (
        len(sampled_landmarks) * (max(0, single_atomic_classes_per_landmark) + max(0, single_composite_classes_per_landmark)) * max(1, single_instances_per_class)
        + max(0, len(pair_groups)) * pair_instances_per_group
        + max(0, len(triple_groups)) * triple_instances_per_group
    )
    manifest_basis_count = max(1, int(manifest_basis_count or 1))
    return max(1, int(stage3_spec.get('sample_count', manifest_basis_count * len(stage3_forms) * qa_samples_per_task) or 1))


def _experiment_scene_readiness(scene_cfg_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    scene_cfg = load_yaml(scene_cfg_path)
    merged_cfg = _task_pipeline_scene_task_cfg(scene_cfg, spec)
    stage_mode = str(spec.get('stage', 'both') or 'both').strip().lower()
    engine, scene_id = _scene_identity(merged_cfg)
    valid_instances = None
    try:
        valid_instances = _load_valid_instances(merged_cfg)
    except Exception:
        valid_instances = None
    if stage_mode == 'stage3':
        expected = _stage3_expected_sample_count(merged_cfg, spec, valid_instances=valid_instances)
        ready = _stage3_manifest_is_complete(merged_cfg, expected_sample_count=expected)
        return {'scene_id': scene_id, 'engine': engine, 'ready': bool(ready), 'expected_sample_count': int(expected), 'stage': 'stage3'}
    if stage_mode == 'stage4':
        expected = _stage4_expected_sample_count(merged_cfg, spec, valid_instances=valid_instances)
        ready = _stage4_manifest_is_complete(merged_cfg, expected_sample_count=expected)
        return {'scene_id': scene_id, 'engine': engine, 'ready': bool(ready), 'expected_sample_count': int(expected), 'stage': 'stage4'}
    stage3_expected = _stage3_expected_sample_count(merged_cfg, spec, valid_instances=valid_instances)
    stage4_expected = _stage4_expected_sample_count(merged_cfg, spec, valid_instances=valid_instances)
    stage3_ready = _stage3_manifest_is_complete(merged_cfg, expected_sample_count=stage3_expected)
    stage4_ready = _stage4_manifest_is_complete(merged_cfg, expected_sample_count=stage4_expected)
    return {'scene_id': scene_id, 'engine': engine, 'ready': bool(stage3_ready and stage4_ready), 'stage3_ready': bool(stage3_ready), 'stage4_ready': bool(stage4_ready), 'stage3_expected_sample_count': int(stage3_expected), 'stage4_expected_sample_count': int(stage4_expected), 'stage': 'both'}


def _center_xy(row: dict[str, Any]) -> tuple[float, float] | None:
    center = list(row.get("center_3d", []) or [])
    if len(center) < 2:
        return None
    try:
        return float(center[0]), float(center[1])
    except Exception:
        return None


def _sample_landmark_groups(
    entries: list[dict[str, Any]],
    *,
    group_size: int,
    count: int,
    max_range_m: float,
    seed: int,
) -> list[list[dict[str, Any]]]:
    if group_size <= 1 or count <= 0:
        return []
    rng = random.Random(int(seed))
    rows = [row for row in list(entries or []) if _center_xy(row) is not None and str(row.get("instance_id", "") or "").strip()]
    rng.shuffle(rows)
    chosen: list[list[dict[str, Any]]] = []
    used_ids: set[tuple[str, ...]] = set()
    for anchor in rows:
        anchor_xy = _center_xy(anchor)
        if anchor_xy is None:
            continue
        nearby = []
        for cand in rows:
            if str(cand.get("instance_id", "") or "") == str(anchor.get("instance_id", "") or ""):
                continue
            cand_xy = _center_xy(cand)
            if cand_xy is None:
                continue
            dx = float(cand_xy[0] - anchor_xy[0])
            dy = float(cand_xy[1] - anchor_xy[1])
            if (dx * dx + dy * dy) ** 0.5 <= float(max_range_m):
                nearby.append(cand)
        rng.shuffle(nearby)
        if len(nearby) < group_size - 1:
            continue
        group = [anchor, *nearby[: group_size - 1]]
        key = tuple(sorted(str(item.get("instance_id", "") or "") for item in group))
        if key in used_ids:
            continue
        used_ids.add(key)
        chosen.append(group)
        if len(chosen) >= int(count):
            break
    return chosen


def _read_landmark_list_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    return [dict(row) for row in list(payload.get("selected_landmarks", []) or []) if isinstance(row, dict)]


def _resolve_ref_rows(
    valid_rows: list[dict[str, Any]],
    refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {str(row.get("instance_id", "") or ""): row for row in list(valid_rows or []) if str(row.get("instance_id", "") or "").strip()}
    out: list[dict[str, Any]] = []
    for ref in list(refs or []):
        if not isinstance(ref, dict):
            continue
        if "center_3d" in ref or "point_count" in ref or "class_name" in ref:
            out.append(dict(ref))
            continue
        instance_id = str(ref.get("instance_id", "") or "").strip()
        row = lookup.get(instance_id)
        if row is not None:
            out.append(dict(row))
    return out


def _build_stage3_args(config_path: Path, scene_cfg: dict[str, Any], *, landmark_id: str, mission_type: str | None, behavior_sequence: str | None, traj_id: str, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        mode="generate_mission",
        config=str(config_path),
        scene_id=str((scene_cfg.get("task", {}) or {}).get("scene_id", "") or ""),
        landmark_id=str(landmark_id),
        selected_instance_ids=[str(landmark_id)],
        mission_type=mission_type,
        behavior_sequence=behavior_sequence,
        generation_kind="atomic-only" if behavior_sequence else "composite-driven",
        mission_mode="single-landmark",
        traj_id=traj_id,
        trajectory_root="scene_data",
        instances_json=None,
        seed=int(seed),
        points_per_behavior=None,
        samples_per_segment=None,
        smooth_window=None,
        sample_count=32,
        forms="",
        task_group="all",
        approved_only=False,
        manifest_path=None,
        limit=0,
        model=None,
        provide_flight_description=False,
        include_keyframes=False,
        host="0.0.0.0",
        port=20262,
        element_param_overrides=None,
        element_auto_rules=None,
        set_profiles=None,
        adaptive_sequential_params=True,
        allow_interleave_repeat=False,
        max_total_elements=0,
    )


def _cycle_pick(items: list[str], *, count: int, offset: int) -> list[str]:
    if not items or count <= 0:
        return []
    out: list[str] = []
    n = len(items)
    for idx in range(count):
        out.append(str(items[(offset + idx) % n]))
    return out


def _auto_parallel_workers(
    requested: Any,
    *,
    job_count: int,
    reserve_cpu: int = 1,
    cpu_fraction: float = 1.0,
    load_factor: float = 1.0,
    hard_cap: int | None = None,
) -> int:
    total_jobs = max(1, int(job_count))
    try:
        requested_int = int(requested or 0)
    except Exception:
        requested_int = 0
    if requested_int > 0:
        return max(1, min(total_jobs, requested_int))
    cpu_total = max(1, int(os.cpu_count() or 1))
    capped = max(1, int(math.floor(float(cpu_total) * max(0.1, min(1.0, float(cpu_fraction))))))
    usable = max(1, min(capped, cpu_total - max(0, int(reserve_cpu))))
    scaled = max(1, int(math.floor(float(usable) * max(0.1, min(1.0, float(load_factor))))))
    if hard_cap is not None and int(hard_cap) > 0:
        scaled = min(int(hard_cap), scaled)
    return max(1, min(total_jobs, scaled))


def _stage3_generate_job_worker(payload: dict[str, Any]) -> dict[str, Any]:
    scene_cfg = dict(payload.get("scene_cfg", {}) or {})
    scene_root = Path(str(payload.get("scene_root", "") or "")).resolve()
    scene_id = str(payload.get("scene_id", "") or "")
    valid_instances = list(payload.get("valid_instances", []) or [])
    component_set_keys = [str(x).strip() for x in list(payload.get("component_set_keys", []) or []) if str(x).strip()]
    max_repair_lift_m = float(payload.get("max_repair_lift_m", 20.0) or 20.0)
    max_repair_fraction = float(payload.get("max_repair_fraction", 0.4) or 0.4)
    generation_retry_limit = max(1, int(payload.get("generation_retry_limit", 8) or 8))
    job = dict(payload.get("job", {}) or {})
    base_run_args = copy.deepcopy(job.get("run_args"))
    if base_run_args is None:
        raise RuntimeError("missing_run_args")
    log_text = str(job.get("log", "") or "")
    last_exc: Exception | None = None
    is_multi = str(getattr(base_run_args, "mission_mode", "") or "") == "multi-landmark"
    group_kind = str(job.get("group_kind", "") or "")
    group_size = 2 if group_kind == "pair" else 3 if group_kind == "triple" else 0
    group_max_range_m = float(job.get("group_max_range_m", 100.0) or 100.0)
    current_ids = [str(x).strip() for x in list(getattr(base_run_args, "selected_instance_ids", []) or []) if str(x).strip()]
    current_key = tuple(sorted(current_ids))
    alt_attempt = 0
    worker_logger = StageLogger("task_pipeline.worker")
    for attempt_idx in range(generation_retry_limit):
        alt_attempt += 1
        run_args = copy.deepcopy(base_run_args)
        swapped_group = False
        if is_multi and attempt_idx > 0 and group_size >= 2:
            sampled_groups = _sample_landmark_groups(
                valid_instances,
                group_size=group_size,
                count=8,
                max_range_m=float(group_max_range_m),
                seed=int(getattr(base_run_args, "seed", 7) or 7) + alt_attempt * 997,
            )
            for group in sampled_groups:
                ids = [str(item.get("instance_id", "") or "") for item in group if str(item.get("instance_id", "") or "").strip()]
                if len(ids) != group_size:
                    continue
                if tuple(sorted(ids)) == current_key:
                    continue
                run_args.selected_instance_ids = list(ids)
                run_args.landmark_id = str(ids[0])
                landmark_set_map = {}
                for local_idx, landmark_id in enumerate(ids):
                    if component_set_keys:
                        landmark_set_map[landmark_id] = component_set_keys[(attempt_idx + local_idx) % len(component_set_keys)]
                run_args.landmark_set_map = landmark_set_map
                swapped_group = True
                break
        run_args.seed = int(getattr(base_run_args, "seed", 7) or 7) + alt_attempt * 1009
        try:
            out = generate_single(run_args, scene_cfg, worker_logger, include_preview=True)
            summary = dict(out.get("summary", {}) or {})
            if (
                not bool(summary.get("collision_free", False))
                or float(summary.get("repair_max_lift_m", 0.0) or 0.0) > max_repair_lift_m
                or float(summary.get("repair_lifted_fraction", 0.0) or 0.0) > max_repair_fraction
            ):
                last_exc = RuntimeError(
                    f"collision_free={summary.get('collision_free')} "
                    f"repair_max_lift_m={summary.get('repair_max_lift_m')} "
                    f"repair_lifted_fraction={summary.get('repair_lifted_fraction')}"
                )
                continue
            out["files"] = generate_preview_assets_for_mission(
                files=dict(out.get("files", {}) or {}),
                summary=summary,
                config=scene_cfg,
                scene_root=scene_root,
                scene_id=scene_id,
            )
            return {
                "traj_id": str(out.get("traj_id", job.get("traj_id", "")) or job.get("traj_id", "")),
                "summary": summary,
                "attempt": int(attempt_idx + 1),
                "swapped_group": bool(swapped_group),
                "log": log_text,
            }
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"stage3_generation_failed_after_retries: {job.get('traj_id', '')}")


def _stage3_analysis_job_worker(payload: dict[str, Any]) -> str:
    scene_cfg = dict(payload.get("scene_cfg", {}) or {})
    scene_root = Path(str(payload.get("scene_root", "") or "")).resolve()
    scene_id = str(payload.get("scene_id", "") or "")
    mission_dir = Path(str(payload.get("mission_dir", "") or "")).resolve()
    if not mission_dir.exists():
        return f"missing::{mission_dir.name}"
    row = _load_final_task_inputs_from_mission_dir(mission_dir)
    _analyze_final_task_visibility_metadata(
        config=scene_cfg,
        scene_root=scene_root,
        scene_id=scene_id,
        out_dir=mission_dir,
        waypoints_xyz=np.asarray(row.get("waypoints"), dtype=np.float32),
        target_center_3d=list(row.get("target_center_3d", []) or []),
        target_bbox_list=list(row.get("target_bbox_list", []) or []),
        mission_meta=dict(row.get("mission_meta", {}) or {}),
        segments=list(row.get("segments", []) or []),
        source_pose_fps_override=float(row.get("source_pose_fps", 10.0) or 10.0),
        waypoint_forwards=np.asarray(row.get("forwards"), dtype=np.float32),
    )
    return f"ok::{mission_dir.name}"


def run_batch(config_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_id = f"{_task_pipeline_name(spec)}_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
    run_root = _task_pipeline_run_root(spec, run_id)
    pipeline_logger = PipelineLogger(run_root / "pipeline.log")
    logger = StageLogger("task_pipeline")
    batch_t0 = time.time()

    common_defaults = dict((_load_common_stage_cfg().get("task_pipeline_defaults", {}) or {}))
    spec = _deep_merge(common_defaults, dict(spec or {}))
    global_stage3_defaults = dict((spec.get("stage3", {}) or {}))
    spec.setdefault("task_name", _task_pipeline_name(spec))
    phase_mode_raw = str(spec.get("phase", "both") or "both").strip().lower()
    pipeline_logger.info(f"task={_task_pipeline_name(spec)} stage={spec.get('stage','both')} phase={phase_mode_raw} config={config_path}")
    _write_pipeline_json(
        run_root / "request.json",
        {
            "run_id": run_id,
            "task_name": _task_pipeline_name(spec),
            "config_path": str(config_path),
            "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            "spec": _json_safe_copy(spec),
        },
    )

    global_list_spec = dict(spec.get("landmark_list", {}) or {})
    if global_list_spec and phase_mode_raw == "selection":
        pipeline_logger.info("phase selection: building reusable cross-scene landmark list")
        scene_configs = _expand_scene_config_paths(global_list_spec.get("scene_configs", []))
        if not scene_configs:
            scene_ids = global_list_spec.get("scene_ids", [])
            engine_name = str(global_list_spec.get("engine", "airsim") or "airsim").strip().lower()
            if isinstance(scene_ids, str) and str(scene_ids).strip().lower() == "all":
                scene_configs = sorted((WORKSPACE_ROOT / "configs" / "flightmvstg").glob(f"task_{engine_name}_*.yaml"))
            else:
                scene_configs = [_config_path_for_scene_id(str(scene_id).strip(), engine=engine_name) for scene_id in list(scene_ids or []) if str(scene_id).strip()]
        if not scene_configs:
            scene_configs = [config_path.resolve()]
        per_scene_counts = {str(k): v for k, v in dict(global_list_spec.get("scene_landmark_counts", {}) or {}).items()}
        default_count = global_list_spec.get("landmark_count", spec.get("landmark_count", 50))
        global_count = global_list_spec.get("global_landmark_count", None)
        seed = int(global_list_spec.get("seed", spec.get("seed", 7)) or spec.get("seed", 7))
        artifact_name = str(global_list_spec.get("artifact_name", "stage34_landmark_list") or "stage34_landmark_list").strip()
        selected_entries: list[dict[str, Any]] = []
        assigned_counts: dict[str, Any] = {}
        if global_count is not None and not per_scene_counts:
            total = int(global_count) if str(global_count).strip().lower() != "all" else "all"
            if total == "all":
                assigned_counts = {str(_scene_identity(load_yaml(path))[1]): "all" for path in scene_configs}
            else:
                base = int(total) // max(1, len(scene_configs))
                rem = int(total) % max(1, len(scene_configs))
                for idx, scene_cfg_path in enumerate(scene_configs):
                    scene_id_local = _scene_identity(load_yaml(scene_cfg_path))[1]
                    assigned_counts[scene_id_local] = base + (1 if idx < rem else 0)
        selection_bar = ProgressBar(total=len(scene_configs), label="task_pipeline.selection", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
        selection_bar.update(0, detail="starting scene landmark selection")
        for idx, scene_cfg_path in enumerate(scene_configs):
            scene_cfg_local = load_yaml(scene_cfg_path)
            engine_local, scene_id_local = _scene_identity(scene_cfg_local)
            valid_rows = _load_valid_instances(scene_cfg_local)
            requested = per_scene_counts.get(scene_id_local, assigned_counts.get(scene_id_local, default_count))
            if str(requested).strip().lower() == "all":
                sampled = list(valid_rows)
            else:
                pick_count = int(requested or default_count)
                sampled = _uniform_sample_by_subcategory(valid_rows, pick_count, seed=seed + idx * 101)
            selected_entries.extend(_build_global_list_entries(config_path=scene_cfg_path, scene_cfg=scene_cfg_local, rows=sampled))
            selection_bar.advance(1, detail=f"{scene_id_local}: selected {len(sampled)} landmarks")
        selection_bar.finish(detail=f"selected total {len(selected_entries)} landmarks")
        written = _write_global_landmark_list(spec=spec, artifact_name=artifact_name, seed=seed, entries=selected_entries)
        out = {
            "ok": True,
            "run_id": run_id,
            "task_name": _task_pipeline_name(spec),
            "stage_mode": str(spec.get("stage", "both") or "both"),
            "phase_mode": str(spec.get("phase", "selection") or "selection"),
            "global_landmark_list_path": written["path"],
            "global_landmark_list_latest_path": written["latest_path"],
            "selected_count": int(len(selected_entries)),
        }
        _write_pipeline_json(run_root / "summary.json", out)
        _write_pipeline_json(_task_pipeline_meta_root(spec) / "latest_run.json", {"run_id": run_id, "summary_path": str((run_root / 'summary.json').resolve())})
        pipeline_logger.info(f"selection completed: selected_count={len(selected_entries)} latest={written['latest_path']}")
        return out

    scene_cfg = load_yaml(config_path)
    scene_cfg = _task_pipeline_scene_task_cfg(scene_cfg, spec)
    scene_id = str((scene_cfg.get("task", {}) or {}).get("scene_id", "") or "")
    engine = str((scene_cfg.get("task", {}) or {}).get("engine", "airsim") or "airsim")
    scene_root = _resolve_scene_root(scene_cfg)
    artifact_scene_root = _resolve_artifact_scene_root(scene_cfg)
    stage_mode_raw = str(spec.get("stage", "both") or "both").strip().lower()
    if stage_mode_raw not in {"both", "stage3", "stage4"}:
        raise ValueError(f"unsupported_stage_mode: {stage_mode_raw}")
    if phase_mode_raw not in {"both", "selection", "data", "render", "data_render", "experiment", "analyze"}:
        raise ValueError(f"unsupported_phase_mode: {phase_mode_raw}")
    run_stage3 = stage_mode_raw in {"both", "stage3"}
    run_stage4 = stage_mode_raw in {"both", "stage4"}
    run_selection = phase_mode_raw in {"both", "selection", "data", "data_render"}
    run_data = phase_mode_raw in {"both", "data", "data_render"}
    run_render = phase_mode_raw in {"both", "render", "data_render"}
    run_experiment = phase_mode_raw in {"both", "experiment"}
    run_analyze = phase_mode_raw in {"analyze"}

    landmark_list_path = Path(str(spec.get("landmark_list_path", "") or "")).resolve() if str(spec.get("landmark_list_path", "") or "").strip() else None
    reused_landmark_entries: list[dict[str, Any]] = []
    if landmark_list_path is not None and landmark_list_path.exists():
        reused_landmark_entries = _read_landmark_list_entries(landmark_list_path)

    removed = _clear_stage_outputs(
        scene_cfg,
        clear_stage3=bool(spec.get("clear_existing_stage3", False)) and run_stage3,
        clear_stage4=bool(spec.get("clear_existing_stage4", False)) and run_stage4,
    )

    valid_instances = _load_valid_instances(scene_cfg) if ((run_stage3 or run_stage4) and run_selection) else []
    sample_count = int(spec.get("landmark_count", 50) or 50)
    sampled_landmarks: list[dict[str, Any]] = []
    stage3_selection_out: dict[str, Any] | None = None
    stage4_selection_out: dict[str, Any] | None = None
    stage3_pair_selection_out: dict[str, Any] | None = None
    stage3_triple_selection_out: dict[str, Any] | None = None
    if run_stage3 and run_selection:
        stage3_spec = dict(spec.get("stage3", {}) or {})
        stage3_single = dict(stage3_spec.get("single_landmark", {}) or {})
        stage3_landmark_count = _count_or_all(stage3_single.get("landmark_count", stage3_spec.get("landmark_count", sample_count)), fallback=sample_count, total_available=len(valid_instances))
        if reused_landmark_entries:
            allowed_ids = {
                str(row.get("instance_id", "") or "")
                for row in reused_landmark_entries
                if str(row.get("scene_id", "") or "") == str(scene_id)
                and str(row.get("engine", "") or "") == str(engine)
            }
            sampled_landmarks = [row for row in valid_instances if str(row.get("instance_id", "") or "") in allowed_ids]
        else:
            sampled_landmarks = _uniform_sample_by_subcategory(valid_instances, stage3_landmark_count, seed=int(spec.get("seed", 7) or 7))
        stage3_selection_out = _write_selection_artifact(
            output_dir=_resolve_stage3_selection_dir(scene_cfg),
            scene_id=scene_id,
            artifact_name="stage3_landmarks",
            seed=int(spec.get("seed", 7) or 7),
            requested_count=stage3_landmark_count,
            entries=_selection_entries(sampled_landmarks),
        )
        pair_scene_counts = {str(k): v for k, v in dict(stage3_spec.get("pair_scene_counts", {}) or {}).items()}
        pair_count = int(pair_scene_counts.get(scene_id, stage3_spec.get("pair_count", 0)) or 0)
        if pair_count > 0:
            pair_groups = _sample_landmark_groups(valid_instances, group_size=2, count=pair_count, max_range_m=float(stage3_spec.get("pair_max_range_m", 50.0) or 50.0), seed=int(spec.get("seed", 7) or 7) + 201)
            stage3_pair_selection_out = _write_selection_artifact(
                output_dir=_resolve_stage3_selection_dir(scene_cfg),
                scene_id=scene_id,
                artifact_name="stage3_pairs",
                seed=int(spec.get("seed", 7) or 7) + 201,
                requested_count=pair_count,
                entries=[{"instance_id": "+".join(str(item.get("instance_id", "") or "") for item in group), "group_size": 2, "members": _selection_entries(group)} for group in pair_groups],
            )
        triple_scene_counts = {str(k): v for k, v in dict(stage3_spec.get("triple_scene_counts", {}) or {}).items()}
        triple_count = int(triple_scene_counts.get(scene_id, stage3_spec.get("triple_count", 0)) or 0)
        if triple_count > 0:
            triple_groups = _sample_landmark_groups(valid_instances, group_size=3, count=triple_count, max_range_m=float(stage3_spec.get("triple_max_range_m", 100.0) or 100.0), seed=int(spec.get("seed", 7) or 7) + 301)
            stage3_triple_selection_out = _write_selection_artifact(
                output_dir=_resolve_stage3_selection_dir(scene_cfg),
                scene_id=scene_id,
                artifact_name="stage3_triples",
                seed=int(spec.get("seed", 7) or 7) + 301,
                requested_count=triple_count,
                entries=[{"instance_id": "+".join(str(item.get("instance_id", "") or "") for item in group), "group_size": 3, "members": _selection_entries(group)} for group in triple_groups],
            )

    stage3_spec = dict(spec.get("stage3", {}) or {})
    stage3_single = dict(stage3_spec.get("single_landmark", {}) or {})
    single_atomic_classes_per_landmark = _count_or_all(
        stage3_single.get("atomic_classes_per_landmark", stage3_single.get("element_classes_per_landmark", 1)),
        fallback=1,
        total_available=len(_single_landmark_element_keys()),
    )
    single_composite_classes_per_landmark = _count_or_all(
        stage3_single.get("composite_classes_per_landmark", stage3_single.get("set_classes_per_landmark", 1)),
        fallback=1,
        total_available=len(_single_landmark_set_keys()),
    )
    single_instances_per_class = max(1, int(stage3_single.get("instances_per_class", 1) or 1))
    self_state_forms = [str(x).strip() for x in list(stage3_single.get("self_state_forms", ["self_set_instance_recognition", "self_element_instance_recognition"])) if str(x).strip()]
    environmental_forms = [str(x).strip() for x in list(stage3_single.get("environmental_forms", ["env_visibility_reasoning"])) if str(x).strip()]
    qa_samples_per_task = max(1, int(stage3_single.get("qa_samples_per_task", 1) or 1))
    stage3_task_parallel = int(global_stage3_defaults.get("record_parallel_workers", 24) or 24)
    stage3_reuse_worker_connections = bool(global_stage3_defaults.get("record_reuse_worker_connections", True))
    stage3_forms = list(stage3_spec.get("forms", [*self_state_forms, *environmental_forms]) or [*self_state_forms, *environmental_forms])
    include_temporal_localization = bool(stage3_spec.get("include_temporal_localization", True))
    stage3_rerender_existing = bool(stage3_spec.get("rerender_existing", False))
    pair_instances_per_group = max(1, int(stage3_spec.get("pair_instances_per_group", 1) or 1))
    triple_instances_per_group = max(1, int(stage3_spec.get("triple_instances_per_group", 1) or 1))

    element_keys = _single_landmark_element_keys() if (run_stage3 and run_data) else []
    set_keys = _single_landmark_set_keys() if (run_stage3 and run_data) else []
    generated_traj_ids: list[str] = []
    planned_reuse_render_traj_ids: list[str] = []
    render_traj_ids: list[str] = []
    pair_groups: list[dict[str, Any]] = []
    triple_groups: list[dict[str, Any]] = []
    stage3_manifest: dict[str, Any] | None = None
    stage3_experiment: dict[str, Any] | None = None
    stage3_experiments: list[dict[str, Any]] = []
    record_out: dict[str, Any] | None = None
    stage4_manifest: dict[str, Any] | None = None
    stage4_render: dict[str, Any] | None = None
    stage4_experiment: dict[str, Any] | None = None
    stage4_experiments: list[dict[str, Any]] = []
    stage3_generate_elapsed = 0.0
    stage3_record_elapsed = 0.0
    stage3_manifest_elapsed = 0.0
    stage3_experiment_elapsed = 0.0
    stage3_analyze_elapsed = 0.0
    stage4_manifest_elapsed = 0.0
    stage4_render_elapsed = 0.0
    stage4_experiment_elapsed = 0.0
    stage4_analyze_elapsed = 0.0
    stage3_recomputed_reports: list[dict[str, Any]] = []
    stage4_recomputed_reports: list[dict[str, Any]] = []
    failed_records_path = _task_pipeline_meta_root(spec) / "failed_landmarks.jsonl"
    run_failed_records_path = run_root / "failed_landmarks.jsonl"

    experiment_model = str(spec.get("experiment_model", "Qwen/Qwen3.5-9B") or "Qwen/Qwen3.5-9B")
    experiment_models = [str(x).strip() for x in list(spec.get("experiment_models", []) or []) if str(x).strip()]
    if not experiment_models:
        experiment_models = [experiment_model]
    experiment_model_parallelism = int(spec.get("experiment_model_parallelism", 0) or 0)
    experiment_model_workers = _auto_parallel_workers(
        experiment_model_parallelism,
        job_count=max(1, len(experiment_models)),
        cpu_fraction=0.5,
        load_factor=1.0,
    )
    global_experiment_progress_callback = spec.get("_global_experiment_progress_callback")
    experiment_overrides = {
        "concurrency": max(1, int(spec.get("experiment_concurrency", 4) or 4)),
        "unique_experiment": bool(spec.get("unique_experiment", False)),
    }
    for _key in ["timeout_s", "rpm_limit", "tpm_limit", "request_retry_attempts", "request_retry_backoff_sec", "request_retry_forever"]:
        if spec.get(_key) is not None:
            experiment_overrides[_key] = spec.get(_key)
    if run_stage3 and run_data:
        pipeline_logger.info("stage3 data: preparing single-scene mission generation")
        if stage3_selection_out is not None:
            sampled_landmarks = list(stage3_selection_out["payload"].get("selected_landmarks", []) or [])
        elif not sampled_landmarks:
            latest_stage3_selection = _resolve_stage3_selection_dir(scene_cfg) / f"{scene_id}.stage3_landmarks.latest.json"
            if latest_stage3_selection.exists():
                sampled_landmarks = list((json.loads(latest_stage3_selection.read_text(encoding="utf-8")) or {}).get("selected_landmarks", []) or [])
            else:
                stage3_landmark_count = int(stage3_spec.get("landmark_count", sample_count) or sample_count)
                sampled_landmarks = _uniform_sample_by_subcategory(valid_instances, stage3_landmark_count, seed=int(spec.get("seed", 7) or 7))
        sampled_landmarks = _resolve_ref_rows(valid_instances, sampled_landmarks)
        pipeline_logger.info(f"stage3 data: selected_landmarks={len(sampled_landmarks)} atomic_per_landmark={single_atomic_classes_per_landmark} composite_per_landmark={single_composite_classes_per_landmark}")
        selected_landmark_ids = [str(row.get("instance_id", "") or "") for row in sampled_landmarks if str(row.get("instance_id", "") or "").strip()]
        stage3_target_atomic = len(selected_landmark_ids) * max(0, single_atomic_classes_per_landmark) * max(1, single_instances_per_class)
        stage3_target_composite = len(selected_landmark_ids) * max(0, single_composite_classes_per_landmark) * max(1, single_instances_per_class)
        existing_single = _discover_stage3_single_missions(scene_cfg)
        existing_by_landmark_family = dict(existing_single.get("by_landmark_family", {}) or {})
        planned_reuse_render_traj_ids = sorted(
            {
                str(row.get("traj_id", "") or "")
                for family in ("atomic", "composite")
                for row in list(existing_single.get("by_family", {}).get(family, []) or [])
                if bool(row.get("needs_render", False))
            }
        )
        historical_failed = _load_stage3_failed_landmark_families(failed_records_path, scene_id=scene_id)
        planned_family_counts: dict[tuple[str, str], int] = {}
        planned_family_keys: dict[tuple[str, str], set[str]] = {}
        existing_family_totals = {
            "atomic": len(list(existing_single.get("by_family", {}).get("atomic", []) or [])),
            "composite": len(list(existing_single.get("by_family", {}).get("composite", []) or [])),
        }
        for family in ("atomic", "composite"):
            for landmark_id in selected_landmark_ids:
                rows = list(existing_by_landmark_family.get((landmark_id, family), []) or [])
                planned_family_counts[(landmark_id, family)] = len(rows)
                planned_family_keys[(landmark_id, family)] = {str(row.get("set_id", "") or "") for row in rows if str(row.get("set_id", "") or "").strip()}
        pipeline_logger.info(
            f"stage3 data: existing_single atomic={existing_family_totals['atomic']} composite={existing_family_totals['composite']} "
            f"target_atomic={stage3_target_atomic} target_composite={stage3_target_composite}"
        )
        if planned_reuse_render_traj_ids:
            pipeline_logger.info(f"stage3 data: reusable unrendered missions={len(planned_reuse_render_traj_ids)}")
        component_set_keys = _single_landmark_component_set_keys()
        pair_groups = []
        triple_groups = []
        if stage3_pair_selection_out is not None:
            pair_groups = list(stage3_pair_selection_out["payload"].get("selected_landmarks", []) or [])
        else:
            latest_pairs = _resolve_stage3_selection_dir(scene_cfg) / f"{scene_id}.stage3_pairs.latest.json"
            if latest_pairs.exists():
                pair_groups = list((json.loads(latest_pairs.read_text(encoding="utf-8")) or {}).get("selected_landmarks", []) or [])
        if stage3_triple_selection_out is not None:
            triple_groups = list(stage3_triple_selection_out["payload"].get("selected_landmarks", []) or [])
        else:
            latest_triples = _resolve_stage3_selection_dir(scene_cfg) / f"{scene_id}.stage3_triples.latest.json"
            if latest_triples.exists():
                triple_groups = list((json.loads(latest_triples.read_text(encoding="utf-8")) or {}).get("selected_landmarks", []) or [])
        stage3_generate_t0 = time.time()
        generation_retry_limit = max(1, int(stage3_spec.get("generation_retry_limit", 8) or 8))
        max_repair_lift_m = float(stage3_spec.get("generation_max_repair_lift_m", 6.0) or 6.0)
        max_repair_fraction = float(stage3_spec.get("generation_max_repair_fraction", 0.2) or 0.2)
        total_stage3_jobs = 0
        generation_jobs: list[dict[str, Any]] = []
        supplemental_jobs_by_family = {"atomic": 0, "composite": 0}
        for idx, landmark in enumerate(sampled_landmarks):
            instance_id = str(landmark.get("instance_id", "") or "")
            if not instance_id:
                continue
            atomic_key = (instance_id, "atomic")
            composite_key = (instance_id, "composite")
            atomic_missing = max(0, max(0, single_atomic_classes_per_landmark) * max(1, single_instances_per_class) - int(planned_family_counts.get(atomic_key, 0)))
            composite_missing = max(0, max(0, single_composite_classes_per_landmark) * max(1, single_instances_per_class) - int(planned_family_counts.get(composite_key, 0)))
            if atomic_missing > 0 and instance_id not in historical_failed["atomic"]:
                chosen_elements = _pick_stage3_family_keys(
                    [f"atomic_{element_key}" for element_key in element_keys],
                    requested_per_landmark=atomic_missing,
                    offset=idx,
                    existing_keys=set(planned_family_keys.get(atomic_key, set())),
                )
                for local_idx, mission_type in enumerate(chosen_elements):
                    traj_id = f"batch_{scene_id}_{instance_id}_atomic_{idx+1:04d}_fill_{local_idx+1:02d}"
                    generation_jobs.append(
                        {
                            "traj_id": traj_id,
                            "run_args": _build_stage3_args(
                                config_path=config_path,
                                scene_cfg=scene_cfg,
                                landmark_id=instance_id,
                                mission_type=mission_type,
                                behavior_sequence=None,
                                traj_id=traj_id,
                                seed=int(spec.get('seed', 7) or 7) + idx * 101 + local_idx,
                            ),
                            "log": f"stage3 mission queue: traj_id={traj_id} landmark_id={instance_id} mission_type={mission_type}",
                        }
                    )
                    planned_family_counts[atomic_key] = int(planned_family_counts.get(atomic_key, 0)) + 1
                    planned_family_keys.setdefault(atomic_key, set()).add(str(mission_type))
            if composite_missing > 0 and instance_id not in historical_failed["composite"]:
                chosen_sets = _pick_stage3_family_keys(
                    set_keys,
                    requested_per_landmark=composite_missing,
                    offset=idx,
                    existing_keys=set(planned_family_keys.get(composite_key, set())),
                )
                for local_idx, mission_type in enumerate(chosen_sets):
                    traj_id = f"batch_{scene_id}_{instance_id}_composite_{idx+1:04d}_fill_{local_idx+1:02d}"
                    generation_jobs.append(
                        {
                            "traj_id": traj_id,
                            "run_args": _build_stage3_args(
                                config_path=config_path,
                                scene_cfg=scene_cfg,
                                landmark_id=instance_id,
                                mission_type=mission_type,
                                behavior_sequence=None,
                                traj_id=traj_id,
                                seed=int(spec.get('seed', 7) or 7) + idx * 101 + 1000 + local_idx,
                            ),
                            "log": f"stage3 mission queue: traj_id={traj_id} landmark_id={instance_id} mission_type={mission_type}",
                        }
                    )
                    planned_family_counts[composite_key] = int(planned_family_counts.get(composite_key, 0)) + 1
                    planned_family_keys.setdefault(composite_key, set()).add(str(mission_type))

        family_targets = {"atomic": stage3_target_atomic, "composite": stage3_target_composite}
        family_key_sources = {"atomic": [f"atomic_{element_key}" for element_key in element_keys], "composite": list(set_keys)}
        valid_landmark_ids = [str(row.get("instance_id", "") or "") for row in valid_instances if str(row.get("instance_id", "") or "").strip()]
        for family in ("atomic", "composite"):
            current_total = sum(int(planned_family_counts.get((landmark_id, family), 0)) for landmark_id in valid_landmark_ids)
            deficit = max(0, int(family_targets[family]) - current_total)
            if deficit <= 0:
                continue
            eligible_ids = [landmark_id for landmark_id in valid_landmark_ids if landmark_id not in historical_failed[family]]
            if not eligible_ids:
                pipeline_logger.warn(f"stage3 data: no eligible landmarks available for supplemental {family} generation")
                continue
            for extra_idx in range(deficit):
                chosen_landmark_id = min(
                    eligible_ids,
                    key=lambda landmark_id: (
                        int(planned_family_counts.get((landmark_id, family), 0)),
                        0 if landmark_id in selected_landmark_ids else 1,
                        landmark_id,
                    ),
                )
                existing_keys = set(planned_family_keys.get((chosen_landmark_id, family), set()))
                mission_type = _pick_stage3_family_keys(
                    family_key_sources[family],
                    requested_per_landmark=1,
                    offset=extra_idx,
                    existing_keys=existing_keys,
                )[0]
                traj_id = f"batch_{scene_id}_{chosen_landmark_id}_{family}_topup_{extra_idx+1:04d}"
                generation_jobs.append(
                    {
                        "traj_id": traj_id,
                        "run_args": _build_stage3_args(
                            config_path=config_path,
                            scene_cfg=scene_cfg,
                            landmark_id=chosen_landmark_id,
                            mission_type=mission_type,
                            behavior_sequence=None,
                            traj_id=traj_id,
                            seed=int(spec.get('seed', 7) or 7) + 200000 + extra_idx + (0 if family == 'atomic' else 50000),
                        ),
                        "log": f"stage3 supplemental queue: traj_id={traj_id} landmark_id={chosen_landmark_id} mission_type={mission_type}",
                    }
                )
                planned_family_counts[(chosen_landmark_id, family)] = int(planned_family_counts.get((chosen_landmark_id, family), 0)) + 1
                planned_family_keys.setdefault((chosen_landmark_id, family), set()).add(str(mission_type))
                supplemental_jobs_by_family[family] += 1

        total_stage3_jobs = len(generation_jobs) + max(0, len(pair_groups)) * pair_instances_per_group + max(0, len(triple_groups)) * triple_instances_per_group
        generate_parallel_workers = _auto_parallel_workers(
            global_stage3_defaults.get("data_generate_parallel_workers", 0),
            job_count=total_stage3_jobs,
            cpu_fraction=0.5,
            load_factor=0.6,
        )
        stage3_gen_bar = ProgressBar(total=max(1, total_stage3_jobs), label="task_pipeline.stage3.data.generate", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
        stage3_gen_bar.update(0, detail="building missions")
        generation_running = {"count": 0}
        generation_lock = threading.Lock()
        if supplemental_jobs_by_family["atomic"] or supplemental_jobs_by_family["composite"]:
            pipeline_logger.info(
                f"stage3 data: supplemental jobs atomic={supplemental_jobs_by_family['atomic']} composite={supplemental_jobs_by_family['composite']}"
            )
        pair_set_type = str(stage3_spec.get("pair_set_type", "") or "").strip()
        triple_set_type = str(stage3_spec.get("triple_set_type", "") or "").strip()
        all_groups = [("pair", row) for row in pair_groups] + [("triple", row) for row in triple_groups]
        for group_idx, (group_kind, group_row) in enumerate(all_groups):
            members = list(group_row.get("members", []) or []) if isinstance(group_row, dict) else []
            member_ids = [str(item.get("instance_id", "") or "") for item in members if str(item.get("instance_id", "") or "").strip()]
            if len(member_ids) < 2:
                continue
            instances_per_group = pair_instances_per_group if group_kind == "pair" else triple_instances_per_group
            for inst_idx in range(instances_per_group):
                landmark_set_map = {}
                for local_idx, landmark_id in enumerate(member_ids):
                    if component_set_keys:
                        landmark_set_map[landmark_id] = component_set_keys[(group_idx + inst_idx + local_idx) % len(component_set_keys)]
                traj_id = f"batch_{scene_id}_{member_ids[0]}_multi_{group_idx+1:04d}_{inst_idx+1:02d}"
                multi_mission_type = pair_set_type if group_kind == "pair" and pair_set_type else (triple_set_type if group_kind == "triple" and triple_set_type else "multi_landmark_composite_inspection")
                run_args = argparse.Namespace(
                    mode="generate_mission",
                    config=str(config_path),
                    scene_id=scene_id,
                    landmark_id=member_ids[0],
                    selected_instance_ids=list(member_ids),
                    mission_type=multi_mission_type,
                    behavior_sequence=None,
                    generation_kind="composite-driven",
                    mission_mode="multi-landmark",
                    traj_id=traj_id,
                    trajectory_root="scene_data",
                    instances_json=None,
                    seed=int(spec.get("seed", 7) or 7) + 500 + group_idx * 17 + inst_idx,
                    points_per_behavior=None,
                    samples_per_segment=None,
                    smooth_window=None,
                    sample_count=32,
                    forms="",
                    task_group="all",
                    approved_only=False,
                    manifest_path=None,
                    limit=0,
                    model=None,
                    provide_flight_description=False,
                    include_keyframes=False,
                    host="0.0.0.0",
                    port=20262,
                    element_param_overrides=None,
                    element_auto_rules=None,
                    set_profiles=None,
                    adaptive_sequential_params=True,
                    allow_interleave_repeat=False,
                    max_total_elements=0,
                    landmark_set_map=landmark_set_map,
                )
                generation_jobs.append(
                    {
                        "traj_id": traj_id,
                        "run_args": run_args,
                        "log": f"stage3 mission generate: traj_id={traj_id} landmark_id={member_ids[0]} mission_type={multi_mission_type}",
                        "group_kind": group_kind,
                        "group_max_range_m": float(stage3_spec.get("pair_max_range_m", 50.0) if group_kind == "pair" else stage3_spec.get("triple_max_range_m", 100.0)),
                    }
                )

        pipeline_logger.info(
            f"stage3 data.generate queue: total_jobs={len(generation_jobs)} workers={generate_parallel_workers}"
        )

        if generate_parallel_workers <= 1:
            for job in generation_jobs:
                try:
                    worker_out = _stage3_generate_job_worker(
                        {
                            "job": job,
                            "scene_cfg": scene_cfg,
                            "scene_root": str(scene_root),
                            "scene_id": scene_id,
                            "valid_instances": valid_instances,
                            "component_set_keys": component_set_keys,
                            "generation_retry_limit": generation_retry_limit,
                            "max_repair_lift_m": max_repair_lift_m,
                            "max_repair_fraction": max_repair_fraction,
                        }
                    )
                    traj_id = str(worker_out.get("traj_id", job["traj_id"]) or job["traj_id"])
                    generated_traj_ids.append(traj_id)
                    stage3_gen_bar.advance(1, detail=f"generated {traj_id}")
                except Exception as exc:
                    run_args = job.get("run_args")
                    mission_mode = str(getattr(run_args, "mission_mode", "") or "")
                    landmark_id = str(getattr(run_args, "landmark_id", "") or "")
                    if mission_mode == "single-landmark" and landmark_id:
                        record = {
                            "run_id": run_id,
                            "task_name": _task_pipeline_name(spec),
                            "scene_id": scene_id,
                            "stage": "stage3",
                            "phase": "data.generate",
                            "landmark_id": landmark_id,
                            "traj_id": str(job.get("traj_id", "") or ""),
                            "reason": str(exc),
                            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                        }
                        _append_jsonl(failed_records_path, record)
                        _append_jsonl(run_failed_records_path, record)
                        pipeline_logger.warn(f"stage3 mission generation skipped: landmark_id={landmark_id} traj_id={job.get('traj_id','')} error={exc}")
                        stage3_gen_bar.advance(1, detail=f"skipped {job.get('traj_id','')}")
                        continue
                    pipeline_logger.error(f"stage3 mission generation failed: traj_id={job.get('traj_id','')} error={exc}")
                    raise
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=generate_parallel_workers) as executor:
                futures = {
                    executor.submit(
                        _stage3_generate_job_worker,
                        {
                            "job": job,
                            "scene_cfg": scene_cfg,
                            "scene_root": str(scene_root),
                            "scene_id": scene_id,
                            "valid_instances": valid_instances,
                            "component_set_keys": component_set_keys,
                            "generation_retry_limit": generation_retry_limit,
                            "max_repair_lift_m": max_repair_lift_m,
                            "max_repair_fraction": max_repair_fraction,
                        },
                    ): job
                    for job in generation_jobs
                }
                for fut in concurrent.futures.as_completed(futures):
                    job = futures[fut]
                    traj_id = str(job["traj_id"])
                    try:
                        worker_out = dict(fut.result() or {})
                        generated_traj_ids.append(str(worker_out.get("traj_id", traj_id) or traj_id))
                        stage3_gen_bar.advance(1, detail=f"generated {traj_id}")
                    except Exception as exc:
                        run_args = job.get("run_args")
                        mission_mode = str(getattr(run_args, "mission_mode", "") or "")
                        landmark_id = str(getattr(run_args, "landmark_id", "") or "")
                        if mission_mode == "single-landmark" and landmark_id:
                            record = {
                                "run_id": run_id,
                                "task_name": _task_pipeline_name(spec),
                                "scene_id": scene_id,
                                "stage": "stage3",
                                "phase": "data.generate",
                                "landmark_id": landmark_id,
                                "traj_id": traj_id,
                                "reason": str(exc),
                                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                            }
                            _append_jsonl(failed_records_path, record)
                            _append_jsonl(run_failed_records_path, record)
                            pipeline_logger.warn(f"stage3 mission generation skipped: landmark_id={landmark_id} traj_id={traj_id} error={exc}")
                            stage3_gen_bar.advance(1, detail=f"skipped {traj_id}")
                            continue
                        pipeline_logger.error(f"stage3 mission generation failed: traj_id={traj_id} error={exc}")
                        raise
        stage3_gen_bar.finish(detail=f"generated {len(generated_traj_ids)} missions")
        stage3_generate_elapsed = float(time.time() - stage3_generate_t0)
        pipeline_logger.info(f"stage3 mission generation completed: traj_count={len(generated_traj_ids)} elapsed_sec={stage3_generate_elapsed:.2f}")
        analysis_t0 = time.time()
        analyze_parallel_workers = _auto_parallel_workers(
            global_stage3_defaults.get("data_analyze_parallel_workers", 0),
            job_count=len(generated_traj_ids),
            cpu_fraction=0.5,
            load_factor=0.8,
        )
        if generated_traj_ids:
            stage3_analysis_bar = ProgressBar(total=max(1, len(generated_traj_ids)), label="task_pipeline.stage3.data.analyze", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
            stage3_analysis_bar.update(0, detail="computing visibility metadata")
            if analyze_parallel_workers <= 1:
                for traj_id in generated_traj_ids:
                    try:
                        status = _stage3_analysis_job_worker(
                            {
                                "scene_cfg": scene_cfg,
                                "scene_root": str(scene_root),
                                "scene_id": scene_id,
                                "mission_dir": str(_resolve_stage3_missions_root(scene_cfg) / str(traj_id)),
                            }
                        )
                        if status.startswith("missing::"):
                            stage3_analysis_bar.advance(1, detail=f"missing mission dir for {traj_id}")
                        else:
                            stage3_analysis_bar.advance(1, detail=f"analyzed {traj_id}")
                    except Exception as exc:
                        pipeline_logger.warn(f"stage3 analysis skipped for {traj_id}: {exc}")
                        stage3_analysis_bar.advance(1, detail=f"skipped {traj_id}")
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=analyze_parallel_workers) as executor:
                    futures = {
                        executor.submit(
                            _stage3_analysis_job_worker,
                            {
                                "scene_cfg": scene_cfg,
                                "scene_root": str(scene_root),
                                "scene_id": scene_id,
                                "mission_dir": str(_resolve_stage3_missions_root(scene_cfg) / str(traj_id)),
                            },
                        ): traj_id
                        for traj_id in generated_traj_ids
                    }
                    for fut in concurrent.futures.as_completed(futures):
                        traj_id = str(futures[fut])
                        try:
                            status = str(fut.result() or "")
                            if status.startswith("missing::"):
                                stage3_analysis_bar.advance(1, detail=f"missing mission dir for {traj_id}")
                            else:
                                stage3_analysis_bar.advance(1, detail=f"analyzed {traj_id}")
                        except Exception as exc:
                            pipeline_logger.warn(f"stage3 analysis skipped for {traj_id}: {exc}")
                            stage3_analysis_bar.advance(1, detail=f"skipped {traj_id}")
            stage3_analysis_bar.finish(detail=f"processed {len(generated_traj_ids)} missions")
            pipeline_logger.info(f"stage3 analysis completed: elapsed_sec={time.time() - analysis_t0:.2f}")
            stage3_manifest_t0 = time.time()
            pipeline_logger.info("stage3 manifest: building placeholder manifest before render")
            stage3_manifest = generate_stage3_manifest(
                config=scene_cfg,
                scene_id=scene_id,
                engine=engine,
                sample_count=max(1, int(stage3_spec.get("sample_count", len(generated_traj_ids) * len(stage3_forms) * qa_samples_per_task) or 1)),
                seed=int(spec.get("seed", 7) or 7),
                forms=stage3_forms,
                approved_only=False,
                mode=str(stage3_spec.get("manifest_mode", "all") or "all"),
                include_temporal_localization=include_temporal_localization,
                require_final_task=False,
                update_latest=False,
            )
            stage3_manifest_elapsed = float(time.time() - stage3_manifest_t0)
            pipeline_logger.info(f"stage3 placeholder manifest completed: manifest_path={stage3_manifest['manifest_path']}")
        else:
            pipeline_logger.info("stage3 data: no new missions generated; skip placeholder manifest rebuild")

    if run_stage3 and run_render:
        render_seed_ids = list(dict.fromkeys([*generated_traj_ids, *planned_reuse_render_traj_ids]))
        render_traj_ids = render_seed_ids or _discover_stage3_traj_ids(scene_cfg)
        render_traj_ids = [traj_id for traj_id in render_traj_ids if _needs_stage3_render(scene_cfg, traj_id, rerender_existing=stage3_rerender_existing)]
        if render_traj_ids:
            stage3_render_bar = ProgressBar(total=max(1, len(render_traj_ids)), label="task_pipeline.stage3.render", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
            stage3_render_bar.update(0, detail="starting task-video rendering")
            stage3_record_t0 = time.time()
            pre_cleanup = cleanup_airsim_processes("task_pipeline_stage3_render_before")
            pipeline_logger.info(
                f"stage3 render: pre-cleanup found={pre_cleanup['found_count']} terminated={pre_cleanup['terminated_count']} "
                f"killed={pre_cleanup['killed_count']} remaining={pre_cleanup['remaining_count']}"
            )
            try:
                pipeline_logger.info(f"stage3 render: starting multi-task scene recording traj_count={len(render_traj_ids)} workers={stage3_task_parallel} reuse_connections={stage3_reuse_worker_connections}")
                record_args = argparse.Namespace(
                    mode="record_scene_videos",
                    config=str(config_path),
                    scene_id=scene_id,
                    traj_ids=",".join(render_traj_ids),
                    record_parallel_workers=stage3_task_parallel,
                    record_reuse_worker_connections=stage3_reuse_worker_connections,
                    rerender_existing=stage3_rerender_existing,
                )
                def _render_progress(done: int, total: int, detail: str) -> None:
                    stage3_render_bar.update(
                        done,
                        detail=_format_stage3_render_terminal_detail(
                            scene_id=scene_id,
                            done=done,
                            total=total,
                            detail=detail,
                        ),
                    )
                record_out = record_scene_videos_cli(
                    record_args,
                    scene_cfg,
                    progress_cb=_render_progress,
                    detail_log_cb=pipeline_logger.mirror_progress,
                )
                stage3_record_elapsed = float(time.time() - stage3_record_t0)
                stage3_render_bar.finish(detail=f"rendered {len(render_traj_ids)} missions")
                pipeline_logger.info(f"stage3 render completed: elapsed_sec={stage3_record_elapsed:.2f}")
            finally:
                post_cleanup = cleanup_airsim_processes("task_pipeline_stage3_render_after")
                pipeline_logger.info(
                    f"stage3 render: post-cleanup found={post_cleanup['found_count']} terminated={post_cleanup['terminated_count']} "
                    f"killed={post_cleanup['killed_count']} remaining={post_cleanup['remaining_count']}"
                )
        else:
            pipeline_logger.info("stage3 render: no missions require rendering")

        stage3_manifest_t0 = time.time()
        pipeline_logger.info("stage3 manifest: building dataset manifest after render")
        manifest_landmarks = list(sampled_landmarks)
        if not manifest_landmarks:
            latest_stage3_selection = _resolve_stage3_selection_dir(scene_cfg) / f"{scene_id}.stage3_landmarks.latest.json"
            if latest_stage3_selection.exists():
                manifest_landmarks = list((json.loads(latest_stage3_selection.read_text(encoding="utf-8")) or {}).get("selected_landmarks", []) or [])
                manifest_landmarks = _resolve_ref_rows(valid_instances or _load_valid_instances(scene_cfg), manifest_landmarks)
        manifest_pair_groups = list(pair_groups)
        if not manifest_pair_groups:
            latest_pairs = _resolve_stage3_selection_dir(scene_cfg) / f"{scene_id}.stage3_pairs.latest.json"
            if latest_pairs.exists():
                manifest_pair_groups = list((json.loads(latest_pairs.read_text(encoding="utf-8")) or {}).get("selected_landmarks", []) or [])
        manifest_triple_groups = list(triple_groups)
        if not manifest_triple_groups:
            latest_triples = _resolve_stage3_selection_dir(scene_cfg) / f"{scene_id}.stage3_triples.latest.json"
            if latest_triples.exists():
                manifest_triple_groups = list((json.loads(latest_triples.read_text(encoding="utf-8")) or {}).get("selected_landmarks", []) or [])
        manifest_basis_count = (
            len(manifest_landmarks) * (max(0, single_atomic_classes_per_landmark) + max(0, single_composite_classes_per_landmark)) * max(1, single_instances_per_class)
            + max(0, len(manifest_pair_groups)) * pair_instances_per_group
            + max(0, len(manifest_triple_groups)) * triple_instances_per_group
        )
        manifest_basis_count = max(1, int(manifest_basis_count or 1))
        if not render_traj_ids and _stage3_manifest_is_complete(scene_cfg, expected_sample_count=max(1, int(stage3_spec.get("sample_count", manifest_basis_count * len(stage3_forms) * qa_samples_per_task) or 1))):
            pipeline_logger.info("stage3 render: all scene videos and final manifest already complete; skip manifest rebuild")
            stage3_manifest = {"manifest_path": _resolve_latest_stage3_manifest_path(scene_cfg)}
        else:
            stage3_manifest = generate_stage3_manifest(
                config=scene_cfg,
                scene_id=scene_id,
                engine=engine,
                sample_count=max(1, int(stage3_spec.get("sample_count", manifest_basis_count * len(stage3_forms) * qa_samples_per_task) or 1)),
                seed=int(spec.get("seed", 7) or 7),
                forms=stage3_forms,
                approved_only=False,
                mode=str(stage3_spec.get("manifest_mode", "all") or "all"),
                include_temporal_localization=include_temporal_localization,
                require_final_task=True,
            )
            stage3_manifest_elapsed = float(time.time() - stage3_manifest_t0)
            pipeline_logger.info(f"stage3 manifest completed: manifest_path={stage3_manifest['manifest_path']}")

    if run_stage3 and run_experiment:
        manifest_path = None
        if stage3_manifest is not None:
            manifest_path = stage3_manifest["manifest_path"]
        else:
            manifest_path = str(_resolve_latest_stage3_manifest_path(scene_cfg))
        if not Path(str(manifest_path)).exists():
            raise FileNotFoundError(f"stage3_manifest_not_found_for_experiment: {manifest_path}")
        pipeline_logger.info(
            f"stage3 experiment: manifest={manifest_path} models={experiment_models} "
            f"model_workers={experiment_model_workers} request_concurrency={experiment_overrides.get('concurrency', 1)}"
        )
        stage3_experiment_t0 = time.time()
        stage3_manifest_total = max(1, _load_manifest_sample_count(manifest_path))
        stage3_remaining_by_model: dict[str, int] = {}
        for model_name in experiment_models:
            layouts = [resolve_stage3_layout(scene_cfg, scene_id=scene_id, engine=engine)]
            progress = stage3_best_run_progress(layouts, scene_id, model_name, str(manifest_path))
            completed = min(stage3_manifest_total, int(progress.get("completed", 0) or 0))
            stage3_remaining_by_model[str(model_name)] = max(0, stage3_manifest_total - completed)
        stage3_progress_total = max(1, sum(stage3_remaining_by_model.values()))
        stage3_total_label = "task_pipeline.stage3.experiment.total"
        stage3_model_label = "task_pipeline.stage3.experiment.model"
        stage3_label_width = max(len(stage3_total_label), len(stage3_model_label))
        stage3_model_width = max(len(str(x)) for x in experiment_models) if experiment_models else 8
        stage3_exp_bar = ProgressBar(total=stage3_progress_total, label=stage3_total_label, label_width=stage3_label_width, min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
        if any(count > 0 for count in stage3_remaining_by_model.values()):
            stage3_exp_bar.update(0, detail=f"starting model jobs remaining_samples={sum(stage3_remaining_by_model.values())}")
        else:
            stage3_exp_bar.update(0, detail="all model samples already complete")
        stage3_model_bars = {
            str(model_name): ProgressBar(
                total=max(1, int(stage3_remaining_by_model.get(str(model_name), 0) or 0)),
                label=stage3_model_label,
                label_width=stage3_label_width,
                min_interval_sec=0.2,
                emit_fn=pipeline_logger.mirror_progress,
            )
            for model_name in experiment_models
        }
        stage3_progress_state = {"done": 0, "by_model": {str(model_name): 0 for model_name in experiment_models}}
        stage3_progress_lock = threading.Lock()

        for model_name, bar in stage3_model_bars.items():
            remaining = int(stage3_remaining_by_model.get(model_name, 0) or 0)
            bar.update(0, detail="waiting" if remaining > 0 else "already complete")

        def _stage3_progress(payload: dict[str, Any]) -> None:
            sample_id = str(payload.get("sample_id", "") or "")
            form = str(payload.get("form", "") or "")
            model_name = str(payload.get("model", "") or "")
            worker_id = int(payload.get("worker_id", 0) or 0)
            request_status = str(payload.get("request_status", "") or "")
            parse_ok = bool(payload.get("parse_ok", False))
            latency_ms = payload.get("latency_ms", None)
            with stage3_progress_lock:
                stage3_progress_state["done"] += 1
                done = int(stage3_progress_state["done"])
                stage3_progress_state["by_model"][model_name] = int(stage3_progress_state["by_model"].get(model_name, 0) or 0) + 1
                model_done = int(stage3_progress_state["by_model"][model_name])
            stage3_exp_bar.update(done, detail=f"progress {done}/{stage3_progress_total}")
            model_bar = stage3_model_bars.get(model_name)
            if model_bar is not None:
                model_bar.update(
                    model_done,
                    detail=_format_experiment_detail(
                        model_name=model_name,
                        model_width=stage3_model_width,
                        unit_label="sample",
                        unit_value=sample_id,
                        request_status=request_status,
                        parse_ok=parse_ok,
                        latency_ms=latency_ms,
                    )
                    + (f" worker={worker_id}" if worker_id > 0 else "")
                    + f" form={form}",
                )
            if callable(global_experiment_progress_callback):
                try:
                    global_experiment_progress_callback(payload)
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=experiment_model_workers) as executor:
            futures = [
                executor.submit(
                    run_stage3_experiment_once,
                    config=scene_cfg,
                    scene_id=scene_id,
                    engine=engine,
                    manifest_path=Path(str(manifest_path)),
                    model=model_name,
                    limit=None,
                    api_overrides=experiment_overrides,
                    cancel_event=None,
                    progress_callback=_stage3_progress,
                )
                for model_name in experiment_models
            ]
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                stage3_experiments.append(result)
                finished_model = str((result or {}).get("report", {}).get("model", "") or "")
                if finished_model in stage3_model_bars:
                    stage3_model_bars[finished_model].finish(detail="completed")
                pipeline_logger.info(f"stage3 experiment: finished model {len(stage3_experiments)}/{len(experiment_models)}")
        for model_name, bar in stage3_model_bars.items():
            model_total = int(getattr(bar, "total", 0) or 0)
            if stage3_progress_state["by_model"].get(model_name, 0) < model_total:
                bar.finish(detail="completed" if model_total > 0 else "already complete")
        if any(count > 0 for count in stage3_remaining_by_model.values()):
            stage3_exp_bar.finish(detail=f"completed {len(stage3_experiments)} model runs remaining_samples={sum(stage3_remaining_by_model.values())}")
        else:
            stage3_exp_bar.finish(detail="all model samples already complete")
        stage3_experiments.sort(key=lambda row: str(row.get("run_id", "")))
        stage3_experiment = stage3_experiments[0] if stage3_experiments else None
        stage3_experiment_elapsed = float(time.time() - stage3_experiment_t0)
        pipeline_logger.info(f"stage3 experiment completed: report_count={len(stage3_experiments)} elapsed_sec={stage3_experiment_elapsed:.2f}")

    stage4_spec = dict(spec.get("stage4", {}) or {})
    stage4_per_landmark = dict(stage4_spec.get("per_landmark", {}) or {})
    stage4_task_types = list(stage4_spec.get("task_types", stage4_per_landmark.get("task_types", ["self_where", "self_what", "env_where", "env_how"])) or ["self_where", "self_what", "env_where", "env_how"])
    stage4_qa_samples_per_difficulty = max(1, int(stage4_per_landmark.get("qa_samples_per_difficulty", 1) or 1))
    stage4_rerender_existing = bool(stage4_spec.get("rerender_existing", False))
    stage4_landmarks: list[dict[str, Any]] = []
    if run_stage4 and run_selection:
        pipeline_logger.info("stage4 selection: sampling landmarks for image QA")
        stage4_sel_bar = ProgressBar(total=1, label="task_pipeline.stage4.selection", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
        stage4_sel_bar.update(0, detail="sampling landmark subset")
        stage4_landmark_count = _count_or_all(stage4_spec.get("landmark_count", sample_count), fallback=sample_count, total_available=len(valid_instances))
        if reused_landmark_entries:
            allowed_ids = {
                str(row.get("instance_id", "") or "")
                for row in reused_landmark_entries
                if str(row.get("scene_id", "") or "") == str(scene_id)
                and str(row.get("engine", "") or "") == str(engine)
            }
            stage4_landmarks = [row for row in valid_instances if str(row.get("instance_id", "") or "") in allowed_ids]
        else:
            stage4_landmarks = _uniform_sample_by_subcategory(valid_instances, stage4_landmark_count, seed=int(spec.get("seed", 7) or 7) + 101)
        stage4_selection_out = _write_selection_artifact(
            output_dir=_resolve_stage4_selection_dir(scene_cfg),
            scene_id=scene_id,
            artifact_name="stage4_landmarks",
            seed=int(spec.get("seed", 7) or 7) + 101,
            requested_count=stage4_landmark_count,
            entries=_selection_entries(stage4_landmarks),
        )
        stage4_sel_bar.finish(detail=f"selected {len(stage4_landmarks)} landmarks")
        pipeline_logger.info(f"stage4 selection completed: selected_landmarks={len(stage4_landmarks)}")
    if run_stage4 and run_data:
        pipeline_logger.info("stage4 data: building manifest from selected landmarks")
        if stage4_selection_out is not None:
            stage4_landmarks = list(stage4_selection_out["payload"].get("selected_landmarks", []) or [])
        else:
            latest_stage4_selection = _resolve_stage4_selection_dir(scene_cfg) / f"{scene_id}.stage4_landmarks.latest.json"
            if latest_stage4_selection.exists():
                stage4_landmarks = list((json.loads(latest_stage4_selection.read_text(encoding='utf-8')) or {}).get("selected_landmarks", []) or [])
            else:
                stage4_landmark_count = int(stage4_spec.get("landmark_count", sample_count) or sample_count)
                stage4_landmarks = _uniform_sample_by_subcategory(valid_instances, stage4_landmark_count, seed=int(spec.get("seed", 7) or 7) + 101)
        stage4_landmarks = _resolve_ref_rows(valid_instances, stage4_landmarks)
        stage4_landmark_ids = [str(row.get("instance_id", "") or "") for row in stage4_landmarks if str(row.get("instance_id", "") or "").strip()]
        per_landmark_tasks = max(1, int(len(stage4_task_types)))
        per_landmark_difficulty_samples = 2 * stage4_qa_samples_per_difficulty
        stage4_sample_count = max(1, int(stage4_spec.get("sample_count", max(1, len(stage4_landmark_ids) * per_landmark_tasks * per_landmark_difficulty_samples)) or max(1, len(stage4_landmark_ids) * per_landmark_tasks * per_landmark_difficulty_samples)))
        if _stage4_manifest_is_complete(scene_cfg, expected_sample_count=stage4_sample_count):
            latest_stage4_manifest_path = _resolve_latest_stage4_manifest_path(scene_cfg)
            stage4_manifest = {
                "manifest_path": latest_stage4_manifest_path,
                "manifest": json.loads(latest_stage4_manifest_path.read_text(encoding="utf-8")),
            }
            pipeline_logger.info("stage4 data: latest manifest and assets already complete; skip regeneration")
        else:
            stage4_data_bar = ProgressBar(total=max(1, len(stage4_landmark_ids)), label="task_pipeline.stage4.data", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
            stage4_data_bar.update(0, detail=f"preparing manifest for {len(stage4_landmark_ids)} landmarks")
            stage4_manifest_t0 = time.time()
            pre_cleanup = cleanup_airsim_processes("task_pipeline_stage4_data_before")
            pipeline_logger.info(
                f"stage4 data: pre-cleanup found={pre_cleanup['found_count']} terminated={pre_cleanup['terminated_count']} "
                f"killed={pre_cleanup['killed_count']} remaining={pre_cleanup['remaining_count']}"
            )
            try:
                stage4_manifest = generate_stage4_manifest(
                    config=scene_cfg,
                    scene_id=scene_id,
                    engine=engine,
                    sample_count=stage4_sample_count,
                    seed=int(spec.get("seed", 7) or 7),
                    reference_main_only=True,
                    difficulties=list(stage4_spec.get("difficulties", ["4way", "8way"]) or ["4way", "8way"]),
                    task_types=stage4_task_types,
                    landmark_categories=[],
                    selected_landmark_ids=stage4_landmark_ids,
                    progress_callback=lambda done, total, detail: stage4_data_bar.update(done, detail=detail),
                )
                stage4_manifest_elapsed = float(time.time() - stage4_manifest_t0)
                stage4_data_bar.finish(detail=f"generated manifest with sample_count={stage4_manifest['manifest'].get('sample_count', 0)}")
                pipeline_logger.info(f"stage4 manifest completed: manifest_path={stage4_manifest['manifest_path']}")
            finally:
                post_cleanup = cleanup_airsim_processes("task_pipeline_stage4_data_after")
                pipeline_logger.info(
                    f"stage4 data: post-cleanup found={post_cleanup['found_count']} terminated={post_cleanup['terminated_count']} "
                    f"killed={post_cleanup['killed_count']} remaining={post_cleanup['remaining_count']}"
                )

    if run_stage4 and run_render:
        manifest_path = None
        if stage4_manifest is not None:
            manifest_path = stage4_manifest["manifest_path"]
        else:
            manifest_path = _resolve_latest_stage4_manifest_path(scene_cfg)
        manifest_path = Path(str(manifest_path)).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"stage4_manifest_not_found_for_render: {manifest_path}")
        if stage4_manifest is None:
            stage4_manifest = {
                "manifest_path": manifest_path,
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
        pipeline_logger.info(f"stage4 render: rerendering assets from manifest={manifest_path}")
        stage4_render_t0 = time.time()
        stage4_render_bar = ProgressBar(total=1, label="task_pipeline.stage4.render", min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
        stage4_render_bar.update(0, detail=f"scene={scene_id} preparing render jobs")
        pre_cleanup = cleanup_airsim_processes("task_pipeline_stage4_render_before")
        pipeline_logger.info(
            f"stage4 render: pre-cleanup found={pre_cleanup['found_count']} terminated={pre_cleanup['terminated_count']} "
            f"killed={pre_cleanup['killed_count']} remaining={pre_cleanup['remaining_count']}"
        )
        try:
            stage4_render = render_stage4_manifest_assets(
                config=scene_cfg,
                scene_id=scene_id,
                engine=engine,
                manifest_path=manifest_path,
                rerender_existing=stage4_rerender_existing,
                progress_callback=lambda done, total, detail: stage4_render_bar.update(done, detail=f"scene={scene_id} {detail}"),
            )
            stage4_render_elapsed = float(time.time() - stage4_render_t0)
            stage4_render_bar.finish(
                detail=(
                    f"scene={scene_id} overlays={int((stage4_render or {}).get('overlay_count', 0) or 0)} "
                    f"env_caps={int((stage4_render or {}).get('env_capture_count', 0) or 0)}"
                )
            )
            pipeline_logger.info(
                f"stage4 render completed: overlays={int((stage4_render or {}).get('overlay_count', 0) or 0)} "
                f"env_captures={int((stage4_render or {}).get('env_capture_count', 0) or 0)} "
                f"elapsed_sec={stage4_render_elapsed:.2f}"
            )
        finally:
            post_cleanup = cleanup_airsim_processes("task_pipeline_stage4_render_after")
            pipeline_logger.info(
                f"stage4 render: post-cleanup found={post_cleanup['found_count']} terminated={post_cleanup['terminated_count']} "
                f"killed={post_cleanup['killed_count']} remaining={post_cleanup['remaining_count']}"
            )
        if not stage4_landmarks:
            latest_stage4_selection = _resolve_stage4_selection_dir(scene_cfg) / f"{scene_id}.stage4_landmarks.latest.json"
            if latest_stage4_selection.exists():
                stage4_landmarks = list((json.loads(latest_stage4_selection.read_text(encoding='utf-8')) or {}).get("selected_landmarks", []) or [])
                stage4_landmarks = _resolve_ref_rows(valid_instances, stage4_landmarks)
        selected_stage4_ids = {str(row.get("instance_id", "") or "").strip() for row in list(stage4_landmarks or []) if str(row.get("instance_id", "") or "").strip()}
        used_stage4_ids = {
            str(row.get("landmark_id", "") or "").strip()
            for row in list((stage4_manifest.get("manifest", {}) or {}).get("samples", []) or [])
            if str(row.get("landmark_id", "") or "").strip()
        }
        for landmark_id in sorted(selected_stage4_ids - used_stage4_ids):
            record = {
                "run_id": run_id,
                "task_name": _task_pipeline_name(spec),
                "scene_id": scene_id,
                "stage": "stage4",
                "phase": "data",
                "landmark_id": landmark_id,
                "reason": "no_eligible_stage4_samples",
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            }
            _append_jsonl(failed_records_path, record)
            _append_jsonl(run_failed_records_path, record)
    if run_stage4 and run_experiment:
        manifest_path = None
        if stage4_manifest is not None:
            manifest_path = stage4_manifest["manifest_path"]
        else:
            manifest_path = str(_resolve_latest_stage4_manifest_path(scene_cfg))
        if not Path(str(manifest_path)).exists():
            raise FileNotFoundError(f"stage4_manifest_not_found_for_experiment: {manifest_path}")
        pipeline_logger.info(
            f"stage4 experiment: manifest={manifest_path} models={experiment_models} "
            f"model_workers={experiment_model_workers} request_concurrency={experiment_overrides.get('concurrency', 1)}"
        )
        stage4_experiment_t0 = time.time()
        stage4_manifest_total = max(1, _load_manifest_sample_count(manifest_path))
        stage4_remaining_by_model: dict[str, int] = {}
        for model_name in experiment_models:
            scene_root = _resolve_artifact_scene_root(scene_cfg)
            stage4_root = resolve_stage4_root(scene_cfg, scene_root=scene_root)
            progress = stage4_best_run_progress([stage4_root], scene_id, model_name, str(manifest_path))
            completed = min(stage4_manifest_total, int(progress.get("completed", 0) or 0))
            stage4_remaining_by_model[str(model_name)] = max(0, stage4_manifest_total - completed)
        stage4_progress_total = max(1, sum(stage4_remaining_by_model.values()))
        stage4_total_label = "task_pipeline.stage4.experiment.total"
        stage4_model_label = "task_pipeline.stage4.experiment.model"
        stage4_label_width = max(len(stage4_total_label), len(stage4_model_label))
        stage4_model_width = max(len(str(x)) for x in experiment_models) if experiment_models else 8
        stage4_exp_bar = ProgressBar(total=stage4_progress_total, label=stage4_total_label, label_width=stage4_label_width, min_interval_sec=0.2, emit_fn=pipeline_logger.mirror_progress)
        if any(count > 0 for count in stage4_remaining_by_model.values()):
            stage4_exp_bar.update(0, detail=f"starting model jobs remaining_samples={sum(stage4_remaining_by_model.values())}")
        else:
            stage4_exp_bar.update(0, detail="all model samples already complete")
        stage4_model_bars = {
            str(model_name): ProgressBar(
                total=max(1, int(stage4_remaining_by_model.get(str(model_name), 0) or 0)),
                label=stage4_model_label,
                label_width=stage4_label_width,
                min_interval_sec=0.2,
                emit_fn=pipeline_logger.mirror_progress,
            )
            for model_name in experiment_models
        }
        stage4_progress_state = {"done": 0, "by_model": {str(model_name): 0 for model_name in experiment_models}}
        stage4_progress_lock = threading.Lock()

        for model_name, bar in stage4_model_bars.items():
            remaining = int(stage4_remaining_by_model.get(model_name, 0) or 0)
            bar.update(0, detail="waiting" if remaining > 0 else "already complete")

        def _stage4_progress(payload: dict[str, Any]) -> None:
            sample_id = str(payload.get("sample_id", "") or "")
            task_type = str(payload.get("task_type", "") or "")
            model_name = str(payload.get("model", "") or "")
            worker_id = int(payload.get("worker_id", 0) or 0)
            request_status = str(payload.get("request_status", "") or "")
            parse_ok = bool(payload.get("parse_ok", False))
            latency_ms = payload.get("latency_ms", None)
            with stage4_progress_lock:
                stage4_progress_state["done"] += 1
                done = int(stage4_progress_state["done"])
                stage4_progress_state["by_model"][model_name] = int(stage4_progress_state["by_model"].get(model_name, 0) or 0) + 1
                model_done = int(stage4_progress_state["by_model"][model_name])
            stage4_exp_bar.update(done, detail=f"progress {done}/{stage4_progress_total}")
            model_bar = stage4_model_bars.get(model_name)
            if model_bar is not None:
                model_bar.update(
                    model_done,
                    detail=_format_experiment_detail(
                        model_name=model_name,
                        model_width=stage4_model_width,
                        unit_label="sample",
                        unit_value=sample_id,
                        request_status=request_status,
                        parse_ok=parse_ok,
                        latency_ms=latency_ms,
                    )
                    + (f" worker={worker_id}" if worker_id > 0 else "")
                    + f" task={task_type}",
                )
            if callable(global_experiment_progress_callback):
                try:
                    global_experiment_progress_callback(payload)
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=experiment_model_workers) as executor:
            futures = [
                executor.submit(
                    run_stage4_experiment,
                    config=scene_cfg,
                    scene_id=scene_id,
                    engine=engine,
                    manifest_path=Path(str(manifest_path)),
                    override_model=model_name,
                    limit=None,
                    api_overrides=experiment_overrides,
                    progress_callback=_stage4_progress,
                )
                for model_name in experiment_models
            ]
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                stage4_experiments.append(result)
                finished_model = str((result or {}).get("report", {}).get("model", "") or "")
                if finished_model in stage4_model_bars:
                    stage4_model_bars[finished_model].finish(detail="completed")
                pipeline_logger.info(f"stage4 experiment: finished model {len(stage4_experiments)}/{len(experiment_models)}")
        for model_name, bar in stage4_model_bars.items():
            model_total = int(getattr(bar, "total", 0) or 0)
            if stage4_progress_state["by_model"].get(model_name, 0) < model_total:
                bar.finish(detail="completed" if model_total > 0 else "already complete")
        if any(count > 0 for count in stage4_remaining_by_model.values()):
            stage4_exp_bar.finish(detail=f"completed {len(stage4_experiments)} model runs remaining_samples={sum(stage4_remaining_by_model.values())}")
        else:
            stage4_exp_bar.finish(detail="all model samples already complete")
        stage4_experiments.sort(key=lambda row: str(row.get("run_id", "")))
        stage4_experiment = stage4_experiments[0] if stage4_experiments else None
        stage4_experiment_elapsed = float(time.time() - stage4_experiment_t0)
        pipeline_logger.info(f"stage4 experiment completed: report_count={len(stage4_experiments)} elapsed_sec={stage4_experiment_elapsed:.2f}")

    if run_stage3 and run_analyze:
        stage3_analyze_t0 = time.time()
        layout = resolve_stage3_layout(scene_cfg, scene_id=scene_id, engine=engine)
        report_paths = sorted(layout["experiments_root"].glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        pipeline_logger.info(f"stage3 analyze: recomputing reports count={len(report_paths)}")
        for report_path in report_paths:
            stage3_recomputed_reports.append(
                recompute_stage3_report_from_run_dir(
                    config=scene_cfg,
                    scene_id=scene_id,
                    engine=engine,
                    run_dir=report_path.parent,
                )
            )
        stage3_analyze_elapsed = float(time.time() - stage3_analyze_t0)
        pipeline_logger.info(f"stage3 analyze completed: report_count={len(stage3_recomputed_reports)} elapsed_sec={stage3_analyze_elapsed:.2f}")

    if run_stage4 and run_analyze:
        stage4_analyze_t0 = time.time()
        stage4_root = resolve_stage4_root(scene_cfg, scene_root=scene_root)
        report_paths = sorted((stage4_root / "experiments").glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        pipeline_logger.info(f"stage4 analyze: recomputing reports count={len(report_paths)}")
        for report_path in report_paths:
            stage4_recomputed_reports.append(
                recompute_stage4_report_from_run_dir(
                    config=scene_cfg,
                    scene_id=scene_id,
                    engine=engine,
                    run_dir=report_path.parent,
                )
            )
        stage4_analyze_elapsed = float(time.time() - stage4_analyze_t0)
        pipeline_logger.info(f"stage4 analyze completed: report_count={len(stage4_recomputed_reports)} elapsed_sec={stage4_analyze_elapsed:.2f}")
    total_elapsed = float(time.time() - batch_t0)

    out = {
        "ok": True,
        "run_id": run_id,
        "task_name": _task_pipeline_name(spec),
        "stage_mode": stage_mode_raw,
        "phase_mode": phase_mode_raw,
        "scene_id": scene_id,
        "engine": engine,
        "source_scene_root": str(scene_root),
        "artifact_scene_root": str(artifact_scene_root),
        "removed_roots": removed,
        "sampled_landmark_count": len(sampled_landmarks),
        "generated_traj_count": len(generated_traj_ids),
        "rendered_traj_count": len(render_traj_ids),
        "timing": {
            "total_elapsed_sec": round(total_elapsed, 2),
            "stage3_generate_sec": round(stage3_generate_elapsed, 2),
            "stage3_record_sec": round(stage3_record_elapsed, 2),
            "stage3_manifest_sec": round(stage3_manifest_elapsed, 2),
            "stage3_experiment_sec": round(stage3_experiment_elapsed, 2),
            "stage3_analyze_sec": round(stage3_analyze_elapsed, 2),
            "stage4_manifest_sec": round(stage4_manifest_elapsed, 2),
            "stage4_render_sec": round(stage4_render_elapsed, 2),
            "stage4_experiment_sec": round(stage4_experiment_elapsed, 2),
            "stage4_analyze_sec": round(stage4_analyze_elapsed, 2),
        },
        "experiment_model_parallel_workers": int(experiment_model_workers),
        "experiment_request_concurrency": int(experiment_overrides.get("concurrency", 1) or 1),
    }
    if stage3_selection_out is not None:
        out["stage3_selection_path"] = stage3_selection_out["path"]
        out["stage3_selection_latest_path"] = stage3_selection_out["latest_path"]
    if landmark_list_path is not None and landmark_list_path.exists():
        out["landmark_list_path"] = str(landmark_list_path)
    if stage4_selection_out is not None:
        out["stage4_selection_path"] = stage4_selection_out["path"]
        out["stage4_selection_latest_path"] = stage4_selection_out["latest_path"]
    if stage3_pair_selection_out is not None:
        out["stage3_pair_selection_path"] = stage3_pair_selection_out["path"]
        out["stage3_pair_selection_latest_path"] = stage3_pair_selection_out["latest_path"]
    if stage3_triple_selection_out is not None:
        out["stage3_triple_selection_path"] = stage3_triple_selection_out["path"]
        out["stage3_triple_selection_latest_path"] = stage3_triple_selection_out["latest_path"]
    if record_out is not None:
        out["record_scene_videos"] = record_out
    if stage3_manifest is not None:
        out["stage3_manifest_path"] = str(stage3_manifest["manifest_path"])
    if stage3_experiment is not None:
        out["stage3_experiment_report"] = str(stage3_experiment["report_path"])
        out["stage3_experiment_reports"] = [str(item["report_path"]) for item in stage3_experiments]
    if stage4_manifest is not None:
        out["stage4_manifest_path"] = str(stage4_manifest["manifest_path"])
    if stage4_render is not None:
        out["stage4_render"] = stage4_render
    if stage4_experiment is not None:
        out["stage4_experiment_report"] = str(stage4_experiment["report_path"])
        out["stage4_experiment_reports"] = [str(item["report_path"]) for item in stage4_experiments]
    if stage3_recomputed_reports:
        out["stage3_recomputed_reports"] = [str(item["report_path"]) for item in stage3_recomputed_reports]
    if stage4_recomputed_reports:
        out["stage4_recomputed_reports"] = [str(item["report_path"]) for item in stage4_recomputed_reports]
    _write_pipeline_json(run_root / "summary.json", out)
    _write_pipeline_json(_task_pipeline_meta_root(spec) / "latest_run.json", {"run_id": run_id, "summary_path": str((run_root / 'summary.json').resolve())})
    pipeline_logger.info(
        "completed: "
        + ", ".join(
            [
                f"stage3_traj={len(generated_traj_ids)}",
                f"stage3_manifest={'yes' if stage3_manifest is not None else 'no'}",
                f"stage3_reports={len(stage3_experiments)}",
                f"stage3_analyze_reports={len(stage3_recomputed_reports)}",
                f"stage4_manifest={'yes' if stage4_manifest is not None else 'no'}",
                f"stage4_reports={len(stage4_experiments)}",
                f"stage4_analyze_reports={len(stage4_recomputed_reports)}",
                f"elapsed_sec={total_elapsed:.2f}",
            ]
        )
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task pipeline runner")
    parser.add_argument("--config", required=False)
    parser.add_argument("--spec", required=False)
    parser.add_argument("--spec-json", required=False)
    parser.add_argument("--stage", choices=["both", "stage3", "stage4"], default="both")
    parser.add_argument("--phase", choices=["both", "selection", "data", "render", "experiment", "analyze"], default="both")
    parser.add_argument("--clear-stage3", action="store_true")
    parser.add_argument("--clear-stage4", action="store_true")
    parser.add_argument("--landmark-count", type=int, default=50)
    parser.add_argument("--task-name", required=False)
    parser.add_argument("--experiment-models", nargs="+", required=False, help="Override experiment models; supports repeated values or comma-separated items")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    argv = list(sys.argv[1:])
    cli_overrides = {
        "config": "--config" in argv,
        "stage": "--stage" in argv,
        "phase": "--phase" in argv,
        "landmark_count": "--landmark-count" in argv,
        "clear_stage3": "--clear-stage3" in argv,
        "clear_stage4": "--clear-stage4" in argv,
    }
    spec_path_raw = str(args.spec or args.spec_json or "").strip()
    payload = {}
    if spec_path_raw:
        spec_path = Path(spec_path_raw).resolve()
        if spec_path.suffix.lower() in {".yaml", ".yml"}:
            payload = load_yaml(spec_path)
        else:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = {
        "stage": str(args.stage),
        "phase": str(args.phase),
        "seed": 7,
        "clear_existing_stage3": bool(args.clear_stage3),
        "clear_existing_stage4": bool(args.clear_stage4),
        "landmark_count": int(args.landmark_count),
        "task_name": str(args.task_name or payload.get("task_name", "") or "default_task"),
        "task_pipeline_root_dir": "task_pipeline_data",
        "experiment_model": "Qwen/Qwen3.5-9B",
        "experiment_concurrency": 4,
        "experiment_model_parallelism": 0,
        "stage3": {
            "record_parallel_workers": 24,
            "record_reuse_worker_connections": True,
            "forms": [
                "self_instance_recognition_joint",
                "env_visibility_reasoning",
            ],
            "include_temporal_localization": True,
        },
        "stage4": {},
    }
    spec = _deep_merge(spec, payload)
    if cli_overrides["stage"]:
        spec["stage"] = str(args.stage)
    if cli_overrides["phase"]:
        spec["phase"] = str(args.phase)
    if cli_overrides["config"]:
        spec["config_path"] = str(args.config)
    if cli_overrides["landmark_count"]:
        spec["landmark_count"] = int(args.landmark_count)
    if str(args.task_name or "").strip():
        spec["task_name"] = str(args.task_name).strip()
    cli_experiment_models: list[str] = []
    for raw in list(getattr(args, "experiment_models", None) or []):
        for item in str(raw or "").split(','):
            model_name = str(item or "").strip()
            if model_name:
                cli_experiment_models.append(model_name)
    if cli_experiment_models:
        spec["experiment_models"] = cli_experiment_models
        spec["experiment_model"] = cli_experiment_models[0]
    if cli_overrides["clear_stage3"]:
        spec["clear_existing_stage3"] = bool(args.clear_stage3)
    if cli_overrides["clear_stage4"]:
        spec["clear_existing_stage4"] = bool(args.clear_stage4)
    phase_mode = str(spec.get("phase", args.phase) or args.phase).strip().lower()
    scene_config_targets = _expand_scene_config_paths(spec.get("scene_configs", []))
    if not scene_config_targets:
        scene_ids = spec.get("scene_ids", [])
        engine_name = str(spec.get("engine", "airsim") or "airsim").strip().lower()
        if isinstance(scene_ids, str) and str(scene_ids).strip().lower() == "all":
            scene_config_targets = sorted((WORKSPACE_ROOT / "configs" / "flightmvstg").glob(f"task_{engine_name}_*.yaml"))
        else:
            scene_config_targets = [_config_path_for_scene_id(str(scene_id).strip(), engine=engine_name) for scene_id in list(scene_ids or []) if str(scene_id).strip()]
    config_path_raw = str(args.config or spec.get("config_path", "") or "").strip()
    if not config_path_raw:
        top_scene_id = str(spec.get("scene_id", "") or "").strip()
        top_engine = str(spec.get("engine", "airsim") or "airsim").strip().lower()
        if top_scene_id:
            config_path_raw = str(_config_path_for_scene_id(top_scene_id, engine=top_engine))
    if not config_path_raw:
        scene_configs = [str(x).strip() for x in list((spec.get("landmark_list", {}) or {}).get("scene_configs", []) or []) if str(x).strip()]
        if scene_configs:
            config_path_raw = scene_configs[0]
    if not config_path_raw:
        landmark_scene_ids = (spec.get("landmark_list", {}) or {}).get("scene_ids", [])
        landmark_engine = str((spec.get("landmark_list", {}) or {}).get("engine", spec.get("engine", "airsim")) or "airsim").strip().lower()
        if isinstance(landmark_scene_ids, str) and str(landmark_scene_ids).strip().lower() != "all":
            config_path_raw = str(_config_path_for_scene_id(str(landmark_scene_ids).strip(), engine=landmark_engine))
        elif isinstance(landmark_scene_ids, (list, tuple)) and landmark_scene_ids:
            config_path_raw = str(_config_path_for_scene_id(str(landmark_scene_ids[0]).strip(), engine=landmark_engine))
    if not config_path_raw:
        raise SystemExit("--config or spec config_path/landmark_list.scene_configs/scene_ids is required")
    config_path_resolved = Path(config_path_raw).resolve()
    if phase_mode == "selection":
        out = run_batch(config_path_resolved, spec)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    if phase_mode == "both" and (spec.get("landmark_list", {}) or {}) and not str(spec.get("landmark_list_path", "") or "").strip():
        selection_spec = copy.deepcopy(spec)
        selection_spec["phase"] = "selection"
        selection_out = run_batch(config_path_resolved, selection_spec)
        latest_list = str(selection_out.get("global_landmark_list_latest_path", "") or "").strip()
        if latest_list:
            spec["landmark_list_path"] = latest_list
    if scene_config_targets:
        outputs = []
        run_stage = str(spec.get("stage", args.stage) or args.stage).strip().lower()
        run_phase = str(spec.get("phase", args.phase) or args.phase).strip().lower()
        if run_phase == "experiment":
            ready_scene_targets: list[Path] = []
            skipped_scene_targets: list[tuple[Path, dict[str, Any]]] = []
            for scene_cfg_path in scene_config_targets:
                readiness = _experiment_scene_readiness(scene_cfg_path.resolve(), spec)
                if bool(readiness.get("ready", False)):
                    ready_scene_targets.append(scene_cfg_path)
                else:
                    skipped_scene_targets.append((scene_cfg_path, readiness))
            scene_config_targets = ready_scene_targets
            if skipped_scene_targets:
                for scene_cfg_path, readiness in skipped_scene_targets:
                    print(f"[task_pipeline] skip incomplete experiment scene={readiness.get('scene_id', Path(scene_cfg_path).stem)} stage={readiness.get('stage')} readiness={json.dumps(readiness, ensure_ascii=False)}")
            if not scene_config_targets:
                print("[task_pipeline] no complete scenes available for experiment; skipped all incomplete scenes")
                print(json.dumps([], ensure_ascii=False, indent=2))
                return
        show_multi_scene_experiment_progress = run_phase in {"experiment", "both"} and run_stage in {"stage3", "stage4", "both"}
        experiment_models = [str(x).strip() for x in list(spec.get("experiment_models", []) or []) if str(x).strip()]
        if not experiment_models:
            experiment_models = [str(spec.get("experiment_model", "Qwen/Qwen3.5-9B") or "Qwen/Qwen3.5-9B")]
        experiment_model_parallelism = int(spec.get("experiment_model_parallelism", 0) or 0)
        experiment_model_workers = _auto_parallel_workers(
            experiment_model_parallelism,
            job_count=max(1, len(experiment_models)),
            cpu_fraction=0.5,
            load_factor=1.0,
        )
        global_model_bars: dict[str, ProgressBar] = {}
        global_model_progress: dict[str, int] = {}
        global_model_total = 0
        global_model_width = max(len(str(x)) for x in experiment_models) if experiment_models else 8
        global_label_width = len("task_pipeline.experiment.model")
        if show_multi_scene_experiment_progress:
            remaining_by_model: dict[str, int] = {}
            for model_name in experiment_models:
                remaining_total = 0
                for scene_cfg_path in scene_config_targets:
                    remaining_total += _remaining_experiment_samples_for_scene_model(scene_cfg_path.resolve(), spec, model_name)
                remaining_by_model[model_name] = int(remaining_total)
            global_model_bars = {
                model_name: ProgressBar(
                    total=max(1, int(remaining_by_model.get(model_name, 0) or 0)),
                    label="task_pipeline.experiment.model",
                    label_width=global_label_width,
                    min_interval_sec=0.2,
                )
                for model_name in experiment_models
            }
            global_model_progress = {model_name: 0 for model_name in experiment_models}
            for model_name, bar in global_model_bars.items():
                remaining = int(remaining_by_model.get(model_name, 0) or 0)
                bar.update(0, detail="waiting" if remaining > 0 else "already complete")

            def _global_experiment_progress(payload: dict[str, Any]) -> None:
                model_name = str(payload.get("model", "") or "")
                if model_name not in global_model_bars:
                    return
                global_model_progress[model_name] = int(global_model_progress.get(model_name, 0) or 0) + 1
                model_done = int(global_model_progress[model_name])
                sample_id = str(payload.get("sample_id", "") or "")
                worker_id = int(payload.get("worker_id", 0) or 0)
                request_status = str(payload.get("request_status", "") or "")
                parse_ok = bool(payload.get("parse_ok", False))
                latency_ms = payload.get("latency_ms", None)
                unit_suffix = ""
                if str(payload.get("form", "") or "").strip():
                    unit_suffix = f" form={payload.get('form')}"
                elif str(payload.get("task_type", "") or "").strip():
                    unit_suffix = f" task={payload.get('task_type')}"
                global_model_bars[model_name].update(
                    model_done,
                    detail=_format_experiment_detail(
                        model_name=model_name,
                        model_width=global_model_width,
                        unit_label="sample",
                        unit_value=sample_id,
                        request_status=request_status,
                        parse_ok=parse_ok,
                        latency_ms=latency_ms,
                    )
                    + (f" worker={worker_id}" if worker_id > 0 else "")
                    + unit_suffix,
                )
        else:
            _global_experiment_progress = None

        if run_phase == "experiment":
            scene_job_plan: list[tuple[Path, str]] = []
            for model_name in experiment_models:
                for scene_cfg_path in scene_config_targets:
                    remaining = _remaining_experiment_samples_for_scene_model(scene_cfg_path.resolve(), spec, model_name)
                    if remaining > 0:
                        scene_job_plan.append((scene_cfg_path, model_name))
            total_scene_jobs = len(scene_job_plan)
            multi_scene_exp_bar = ProgressBar(
                total=max(1, total_scene_jobs),
                label="task_pipeline.experiment.scenes",
                min_interval_sec=0.2,
            ) if show_multi_scene_experiment_progress else None
            if multi_scene_exp_bar is not None:
                if total_scene_jobs > 0:
                    multi_scene_exp_bar.update(0, detail=f"starting 0/{total_scene_jobs} scene-model jobs")
                else:
                    multi_scene_exp_bar.update(0, detail="all scene-model jobs already complete")

            scene_job_lock = threading.Lock()

            def _run_model_across_scenes(model_name: str) -> list[dict[str, Any]]:
                local_outputs: list[dict[str, Any]] = []
                for scene_cfg_path in scene_config_targets:
                    remaining = _remaining_experiment_samples_for_scene_model(scene_cfg_path.resolve(), spec, model_name)
                    if remaining <= 0:
                        continue
                    local_spec = dict(spec)
                    local_spec["config_path"] = str(scene_cfg_path)
                    local_spec["experiment_models"] = [model_name]
                    local_spec["experiment_model"] = model_name
                    if show_multi_scene_experiment_progress and _global_experiment_progress is not None:
                        local_spec["_global_experiment_progress_callback"] = _global_experiment_progress
                    result = run_batch(scene_cfg_path.resolve(), local_spec)
                    local_outputs.append(result)
                    if multi_scene_exp_bar is not None:
                        with scene_job_lock:
                            outputs.append(result)
                            done = len(outputs)
                        scene_cfg_name = Path(str(scene_cfg_path)).name
                        scene_name = str((result or {}).get("scene_id", "") or scene_cfg_name)
                        multi_scene_exp_bar.update(done, detail=f"completed {done}/{total_scene_jobs} jobs model={model_name} scene={scene_name}")
                    else:
                        with scene_job_lock:
                            outputs.append(result)
                return local_outputs

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(len(experiment_models), experiment_model_workers))) as executor:
                futures = {executor.submit(_run_model_across_scenes, model_name): model_name for model_name in experiment_models}
                for fut in concurrent.futures.as_completed(futures):
                    fut.result()
            if multi_scene_exp_bar is not None:
                if total_scene_jobs > 0:
                    multi_scene_exp_bar.finish(detail=f"completed {len(outputs)}/{total_scene_jobs} scene-model jobs")
                else:
                    multi_scene_exp_bar.finish(detail="all scene-model jobs already complete")
        else:
            show_multi_scene_progress = len(scene_config_targets) > 1
            multi_scene_bar = ProgressBar(
                total=len(scene_config_targets),
                label="task_pipeline.scenes",
                min_interval_sec=0.2,
            ) if show_multi_scene_progress else None
            if multi_scene_bar is not None:
                multi_scene_bar.update(0, detail=f"total_scenes={len(scene_config_targets)} phase={run_phase}")
            for scene_cfg_path in scene_config_targets:
                local_spec = dict(spec)
                local_spec["config_path"] = str(scene_cfg_path)
                if show_multi_scene_experiment_progress and _global_experiment_progress is not None:
                    local_spec["_global_experiment_progress_callback"] = _global_experiment_progress
                result = run_batch(scene_cfg_path.resolve(), local_spec)
                outputs.append(result)
                if multi_scene_bar is not None:
                    done = len(outputs)
                    scene_cfg_name = Path(str(scene_cfg_path)).name
                    scene_name = str((result or {}).get("scene_id", "") or scene_cfg_name)
                    rendered_count = int((result or {}).get("rendered_traj_count", 0) or 0)
                    multi_scene_bar.update(
                        done,
                        detail=f"scenes={done}/{len(scene_config_targets)} current={scene_name} scene_tasks={rendered_count}",
                    )
            if multi_scene_bar is not None:
                multi_scene_bar.finish(detail=f"scenes={len(outputs)}/{len(scene_config_targets)} phase={run_phase}")
        for model_name, bar in global_model_bars.items():
            model_total = int(getattr(bar, "total", 0) or 0)
            if int(global_model_progress.get(model_name, 0) or 0) < model_total:
                bar.finish(detail="completed" if model_total > 0 else "already complete")
        print(json.dumps({"ok": True, "phase_mode": phase_mode, "scene_count": len(outputs), "results": outputs}, ensure_ascii=False, indent=2))
        return
    out = run_batch(config_path_resolved, spec)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
