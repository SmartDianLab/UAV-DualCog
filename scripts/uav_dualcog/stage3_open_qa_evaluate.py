from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any
import types

from api_common import compute_rate_limited_concurrency
from prompt_templates_stage3_open import (
    get_stage3_open_prompt_template,
    render_stage3_open_prompt_template,
)
if "sim_bridge.factory" not in sys.modules:
    sim_bridge_pkg = types.ModuleType("sim_bridge")
    sim_bridge_factory = types.ModuleType("sim_bridge.factory")

    def _missing_sim_bridge_create_bridge(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("sim_bridge_is_not_available_in_this_environment")

    sim_bridge_factory.create_bridge = _missing_sim_bridge_create_bridge  # type: ignore[attr-defined]
    sim_bridge_pkg.factory = sim_bridge_factory  # type: ignore[attr-defined]
    sys.modules["sim_bridge"] = sim_bridge_pkg
    sys.modules["sim_bridge.factory"] = sim_bridge_factory

from stage3_task_suite import (
    ApiRateLimiter,
    CancelledExperimentError,
    OpenAI,
    _build_openai_messages,
    _count_completed_prediction_rows,
    _extract_response_text,
    _interval_iou,
    _iso_now,
    _messages_preview,
    _now_ts,
    _normalize_interval_rows,
    _path_for_json,
    _prepare_upload_video,
    _resolve_api_cfg,
    _resolve_stage3_layout,
    _safe_name,
    _stage3_cfg,
    _match_interval_metrics,
)
from stage4_open_qa_evaluate import (
    _chat_completion_with_retry,
    _extract_first_object_from_text,
    _extract_json_objects,
    _parse_loose_object,
    _strip_code_fences,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _normalize_text(text: str) -> str:
    out = str(text or "").strip().lower()
    out = out.replace("_", " ").replace("/", " ").replace("\\", " ")
    out = re.sub(r"[-–—]+", "-", out)
    out = re.sub(r"[^a-z0-9\-\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _extract_first_array_from_text(raw_text: str) -> list[Any] | None:
    text = _strip_code_fences(str(raw_text or "").strip())
    if not text:
        return None
    for candidate in (text, text[text.find("[") : text.rfind("]") + 1] if "[" in text and "]" in text else ""):
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            if isinstance(payload, list):
                return payload
        except Exception:
            pass
        try:
            payload = ast.literal_eval(candidate)
            if isinstance(payload, list):
                return payload
        except Exception:
            pass
    return None


def _normalize_open_answer_item(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        if isinstance(row, str) and row.strip():
            return {"answer_text": row.strip(), "intervals_sec": []}
        return None
    text_value = row.get("answer_text", row.get("answer", row.get("final_answer", row.get("prediction", row.get("text", "")))))
    answer_text = str(text_value or "").strip()
    if not answer_text and row.get("intervals_sec") is None and row.get("intervals") is None:
        return None
    intervals = _normalize_interval_rows(
        row.get(
            "intervals_sec",
            row.get("intervals", row.get("time_intervals_sec", row.get("visible_intervals_sec", []))),
        )
    )
    return {"answer_text": answer_text, "intervals_sec": intervals}


def _normalize_open_answer_items(rows: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(rows, dict):
        if "answer_items" in rows:
            return _normalize_open_answer_items(rows.get("answer_items"))
        if "answers" in rows:
            return _normalize_open_answer_items(rows.get("answers"))
        single = _normalize_open_answer_item(rows)
        if single is not None:
            out.append(single)
        return out
    for row in list(rows or []):
        item = _normalize_open_answer_item(row)
        if item is not None:
            out.append(item)
    return out


def _parse_self_instance_answer(raw_text: str) -> tuple[list[dict[str, Any]], bool, Any]:
    array_payload = _extract_first_array_from_text(raw_text)
    if array_payload is not None:
        items = _normalize_open_answer_items(array_payload)
        if items:
            return items, True, array_payload

    obj = _extract_first_object_from_text(raw_text)
    if obj is not None:
        if "answer_items" in obj or "answers" in obj:
            items = _normalize_open_answer_items(obj.get("answer_items", obj.get("answers")))
            if items:
                return items, True, obj
        single = _normalize_open_answer_item(obj)
        if single is not None and single.get("answer_text"):
            return [single], True, obj

    line = _strip_code_fences(str(raw_text or "").strip())
    if not line:
        return [], False, None
    match = re.search(r"(?:answer_text|answer|final_answer|prediction)\s*[:=：]\s*([^\n,}]+)", line, flags=re.IGNORECASE)
    if match:
        return [{"answer_text": match.group(1).strip().strip("\"'"), "intervals_sec": []}], True, None
    first_line = next((seg.strip() for seg in line.splitlines() if seg.strip()), "")
    if first_line:
        return [{"answer_text": first_line.strip("\"'"), "intervals_sec": []}], True, None
    return [], False, None


def _parse_visibility_answer(raw_text: str) -> tuple[int, list[list[float]], bool, Any]:
    payload = _extract_first_object_from_text(raw_text)
    if payload is None:
        line = _strip_code_fences(str(raw_text or "").strip())
        if not line:
            return -999, [], False, None
        payload = _parse_loose_object(line)
        if payload is None:
            return -999, [], False, None
    answer_text = payload.get("answer_text", payload.get("answer", payload.get("final_answer", payload.get("prediction", payload.get("count", payload.get("visible_count"))))))
    count = -999
    try:
        count = int(str(answer_text).strip())
    except Exception:
        try:
            count = int(payload.get("visible_count", -999))
        except Exception:
            count = -999
    intervals = _normalize_interval_rows(payload.get("visible_intervals_sec", payload.get("intervals_sec", payload.get("intervals", []))))
    return count, intervals, bool(count >= 0 and intervals is not None), payload


def _item_best_tiou(gold_intervals: list[list[float]], pred_intervals: list[list[float]]) -> float:
    if not gold_intervals:
        return 0.0
    if not pred_intervals:
        return 0.0
    return sum(max((_interval_iou(pred, gold) for pred in pred_intervals), default=0.0) for gold in gold_intervals) / float(len(gold_intervals))


def _item_interval_match(gold_intervals: list[list[float]], pred_intervals: list[list[float]], threshold: float) -> bool:
    return _item_best_tiou(gold_intervals, pred_intervals) >= float(threshold)


def _judge_self_items(
    *,
    gold_items: list[dict[str, Any]],
    pred_items: list[dict[str, Any]],
    judge_client: Any,
    judge_cfg: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cache_key = json.dumps(
        {
            "gold": gold_items,
            "pred": pred_items,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    if cache_key in cache:
        return dict(cache[cache_key])

    gold_lines = []
    for idx, item in enumerate(gold_items):
        gold_lines.append(
            f"{idx}. answer_text={json.dumps(str(item.get('answer_text', '') or ''), ensure_ascii=False)} intervals_sec={json.dumps(list(item.get('intervals_sec', []) or []), ensure_ascii=False)}"
        )
    pred_lines = []
    for idx, item in enumerate(pred_items):
        pred_lines.append(
            f"{idx}. answer_text={json.dumps(str(item.get('answer_text', '') or ''), ensure_ascii=False)} intervals_sec={json.dumps(list(item.get('intervals_sec', []) or []), ensure_ascii=False)}"
        )
    template_system = get_stage3_open_prompt_template("stage3_open_judge", "self_instance_recognition_joint", "system")
    template_user = get_stage3_open_prompt_template("stage3_open_judge", "self_instance_recognition_joint", "user")
    system_prompt = render_stage3_open_prompt_template(template_system, {})
    user_prompt = render_stage3_open_prompt_template(
        template_user,
        {
            "gold_items_json": json.dumps(gold_lines, ensure_ascii=False, indent=2),
            "pred_items_json": json.dumps(pred_lines, ensure_ascii=False, indent=2),
        },
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    raw_text, raw_meta, latency_ms = _chat_completion_with_retry(
        client=judge_client,
        api_cfg=judge_cfg,
        messages=messages,
        max_tokens=240,
        temperature=0.0,
    )
    matches: list[dict[str, Any]] = []
    parsed_ok = False
    payload = _extract_first_object_from_text(raw_text)
    if payload is None:
        payload = _parse_loose_object(raw_text)
    if isinstance(payload, dict) and isinstance(payload.get("matches"), list):
        parsed_ok = True
        for row in list(payload.get("matches", []) or []):
            if not isinstance(row, dict):
                continue
            try:
                pred_index = int(row.get("pred_index", -1))
            except Exception:
                pred_index = -1
            try:
                gold_index = int(row.get("gold_index", -1))
            except Exception:
                gold_index = -1
            matches.append(
                {
                    "pred_index": pred_index,
                    "gold_index": gold_index,
                    "semantic_match": bool(row.get("semantic_match", gold_index >= 0)),
                    "reason": str(row.get("reason", "") or ""),
                }
            )
    if not matches and pred_items:
        # Fallback to a conservative one-to-one greedy alignment on lexical overlap.
        gold_norms = [_normalize_text(str(item.get("answer_text", "") or "")) for item in gold_items]
        used_gold: set[int] = set()
        for pred_index, pred in enumerate(pred_items):
            pred_norm = _normalize_text(str(pred.get("answer_text", "") or ""))
            best_gold = -1
            best_score = 0
            for gold_index, gold_norm in enumerate(gold_norms):
                if gold_index in used_gold:
                    continue
                if not gold_norm or not pred_norm:
                    continue
                score = 0
                if pred_norm == gold_norm:
                    score = 3
                elif pred_norm in gold_norm or gold_norm in pred_norm:
                    score = 2
                elif any(tok in gold_norm.split() for tok in pred_norm.split() if len(tok) > 2):
                    score = 1
                if score > best_score:
                    best_score = score
                    best_gold = gold_index
            semantic_match = best_gold >= 0 and best_score > 0
            if semantic_match:
                used_gold.add(best_gold)
            matches.append(
                {
                    "pred_index": pred_index,
                    "gold_index": best_gold if semantic_match else -1,
                    "semantic_match": bool(semantic_match),
                    "reason": "fallback lexical match" if semantic_match else "no lexical match",
                }
            )

    result = {
        "judge_parsed_ok": bool(parsed_ok),
        "judge_matches": matches,
        "judge_overall_match": bool(
            len([m for m in matches if int(m.get("gold_index", -1)) >= 0 and bool(m.get("semantic_match", False))]) == len(gold_items)
            and len(pred_items) == len(matches)
        ),
        "judge_raw_text": raw_text,
        "judge_raw_response": raw_meta,
        "judge_latency_ms": float(latency_ms),
    }
    cache[cache_key] = dict(result)
    return result


def _stage3_open_root_from_manifest(manifest_path: Path) -> Path:
    path = manifest_path.resolve()
    if path.parent.name == "manifests":
        return path.parent.parent.parent
    if path.parent.name == "datasets":
        return path.parent.parent
    return path.parent


def _stage3_open_best_run_progress(stage3_root: Path, scene_id: str, model: str, manifest_path: str) -> dict[str, Any]:
    experiments_root = (Path(stage3_root).resolve() / "experiments_open").resolve()
    if not experiments_root.exists():
        return {"completed": 0, "run_dir": ""}
    model_tag = _safe_name(str(model or ""))
    manifest_tag = _safe_name(Path(str(manifest_path or "")).stem)[:48] if str(manifest_path or "").strip() else ""
    prefix = f"{scene_id}_{model_tag}_{manifest_tag}" if manifest_tag else f"{scene_id}_{model_tag}_"
    best_completed = 0
    best_run_dir = ""
    best_mtime = -1.0
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


def _form_main_metric(row: dict[str, Any]) -> float | None:
    form = str(row.get("form", "") or "")
    if form == "env_visibility_reasoning":
        try:
            return float(row.get("count_exact", 0.0) or 0.0)
        except Exception:
            return 0.0
    if form == "self_instance_recognition_joint":
        try:
            return float(row.get("set_instance_acc", 0.0) or 0.0)
        except Exception:
            return 0.0
    return None


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"count": 0}
    parse_ok = sum(1 for row in rows if bool(row.get("parse_ok", False)))
    answer_parse_ok = sum(1 for row in rows if bool(row.get("answer_parse_ok", False)))
    count_rows = [row for row in rows if str(row.get("form", "")) == "env_visibility_reasoning"]
    self_rows = [row for row in rows if str(row.get("form", "")) == "self_instance_recognition_joint"]
    env_rows = count_rows
    self_count = len(self_rows)
    env_count = len(env_rows)
    count_exact = sum(1 for row in env_rows if bool(row.get("count_exact", False)))
    count_within1 = sum(1 for row in env_rows if bool(row.get("count_within1", False)))
    count_mae_values = [float(row.get("count_abs_error", 0.0) or 0.0) for row in env_rows]
    count_consistency = sum(1 for row in rows if bool(row.get("count_consistency", False)))
    segment_precision_03 = [float(row.get("segment_precision@0.3", 0.0) or 0.0) for row in env_rows]
    segment_recall_03 = [float(row.get("segment_recall@0.3", 0.0) or 0.0) for row in env_rows]
    segment_precision_05 = [float(row.get("segment_precision@0.5", 0.0) or 0.0) for row in env_rows]
    segment_recall_05 = [float(row.get("segment_recall@0.5", 0.0) or 0.0) for row in env_rows]
    best_tious = [float(row.get("mean_best_tIoU", 0.0) or 0.0) for row in env_rows]
    set_instance = [bool(row.get("set_instance_acc", False)) for row in self_rows]
    element_precisions = [float(row.get("element_instance_precision", 0.0) or 0.0) for row in self_rows]
    element_recalls = [float(row.get("element_instance_recall", 0.0) or 0.0) for row in self_rows]
    element_f1s = [float(row.get("element_instance_f1", 0.0) or 0.0) for row in self_rows]
    self_loc_precisions = [float(row.get("self_temporal_loc_precision@0.5", 0.0) or 0.0) for row in self_rows]
    self_loc_recalls = [float(row.get("self_temporal_loc_recall@0.5", 0.0) or 0.0) for row in self_rows]
    self_loc_f1s = [float(row.get("self_temporal_loc_f1@0.5", 0.0) or 0.0) for row in self_rows]
    self_loc_tious = [float(row.get("self_temporal_loc_mean_tIoU", 0.0) or 0.0) for row in self_rows]
    return {
        "count": total,
        "parse_success_rate": float(parse_ok) / float(total),
        "answer_parse_success_rate": float(answer_parse_ok) / float(total),
        "environmental_count": env_count,
        "self_state_count": self_count,
        "count_exact_acc": float(count_exact) / float(env_count) if env_count else None,
        "count_within1_acc": float(count_within1) / float(env_count) if env_count else None,
        "count_mae": (sum(count_mae_values) / float(len(count_mae_values))) if count_mae_values else None,
        "segment_precision@0.3": (sum(segment_precision_03) / float(len(segment_precision_03))) if segment_precision_03 else None,
        "segment_recall@0.3": (sum(segment_recall_03) / float(len(segment_recall_03))) if segment_recall_03 else None,
        "segment_f1@0.3": (
            2.0
            * ((sum(segment_precision_03) / float(len(segment_precision_03))) if segment_precision_03 else 0.0)
            * ((sum(segment_recall_03) / float(len(segment_recall_03))) if segment_recall_03 else 0.0)
            / max(
                1e-6,
                ((sum(segment_precision_03) / float(len(segment_precision_03))) if segment_precision_03 else 0.0)
                + ((sum(segment_recall_03) / float(len(segment_recall_03))) if segment_recall_03 else 0.0),
            )
            if env_count
            else None
        ),
        "segment_precision@0.5": (sum(segment_precision_05) / float(len(segment_precision_05))) if segment_precision_05 else None,
        "segment_recall@0.5": (sum(segment_recall_05) / float(len(segment_recall_05))) if segment_recall_05 else None,
        "segment_f1@0.5": (
            2.0
            * ((sum(segment_precision_05) / float(len(segment_precision_05))) if segment_precision_05 else 0.0)
            * ((sum(segment_recall_05) / float(len(segment_recall_05))) if segment_recall_05 else 0.0)
            / max(
                1e-6,
                ((sum(segment_precision_05) / float(len(segment_precision_05))) if segment_precision_05 else 0.0)
                + ((sum(segment_recall_05) / float(len(segment_recall_05))) if segment_recall_05 else 0.0),
            )
            if env_count
            else None
        ),
        "mean_best_tIoU": (sum(best_tious) / float(len(best_tious))) if best_tious else None,
        "count_consistency": float(count_consistency) / float(total),
        "set_instance_acc": (sum(1 for v in set_instance if v) / float(len(set_instance))) if set_instance else None,
        "element_instance_precision": (sum(element_precisions) / float(len(element_precisions))) if element_precisions else None,
        "element_instance_recall": (sum(element_recalls) / float(len(element_recalls))) if element_recalls else None,
        "element_instance_f1": (sum(element_f1s) / float(len(element_f1s))) if element_f1s else None,
        "self_temporal_loc_precision@0.5": (sum(self_loc_precisions) / float(len(self_loc_precisions))) if self_loc_precisions else None,
        "self_temporal_loc_recall@0.5": (sum(self_loc_recalls) / float(len(self_loc_recalls))) if self_loc_recalls else None,
        "self_temporal_loc_f1@0.5": (sum(self_loc_f1s) / float(len(self_loc_f1s))) if self_loc_f1s else None,
        "self_temporal_loc_mean_tIoU": (sum(self_loc_tious) / float(len(self_loc_tious))) if self_loc_tious else None,
    }


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for key in ("mode", "form", "difficulty_band"):
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bucket[str(row.get(key, ""))].append(row)
        grouped[key] = {
            name: {
                **_aggregate_metric_rows(items),
                "main_metric": _form_main_metric(items[0]) if items else None,
            }
            for name, items in bucket.items()
        }
    combo_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combo_diff_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        form = str(row.get("form", "") or "")
        difficulty = str(row.get("difficulty_band", "") or "")
        mode = str(row.get("mode", "") or "")
        combo_bucket[mode + "|" + form].append(row)
        combo_diff_bucket[mode + "|" + form + "|" + difficulty].append(row)
    grouped["combo"] = {name: _aggregate_metric_rows(items) for name, items in combo_bucket.items()}
    grouped["combo_by_difficulty"] = {name: _aggregate_metric_rows(items) for name, items in combo_diff_bucket.items()}
    out = _aggregate_metric_rows(rows)
    out["grouped"] = grouped
    return out


def _resolve_manifest_path(manifest_path: Path | None, *, scene_id: str, task_name: str, task_pipeline_root: str) -> Path:
    if manifest_path is not None:
        return Path(manifest_path).resolve()
    root = Path(task_pipeline_root)
    if not root.is_absolute():
        root = (WORKSPACE_ROOT / root).resolve()
    manifest_dir = root / task_name / f"airsim_{scene_id}" / "stage3_tasks" / "datasets"
    prefer_open = manifest_dir / f"{scene_id}.latest_manifest.open.json"
    fallback = manifest_dir / f"{scene_id}.latest_manifest.json"
    if prefer_open.exists():
        return prefer_open
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"manifest_not_found: {prefer_open} / {fallback}")


def run_open_qa_experiment(
    *,
    config: dict[str, Any],
    manifest_path: Path,
    override_model: str | None,
    judge_model: str | None,
    enable_judge: bool,
    limit: int | None,
    api_overrides: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai package is required for stage3 open QA experiment")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scene_id = str(manifest.get("scene_id", "") or "")
    engine = str(manifest.get("engine", "airsim") or "airsim").strip().lower()
    samples = list(manifest.get("samples", []) or [])
    if limit is not None and int(limit) > 0:
        samples = samples[: int(limit)]
    if not samples:
        raise RuntimeError("no_samples_for_stage3_open_qa_experiment")

    api_cfg = _resolve_api_cfg(config, override_model=override_model, overrides=api_overrides)
    eval_client = OpenAI(api_key=api_cfg["api_key"], base_url=api_cfg["api_base"])

    judge_cfg: dict[str, Any] | None = None
    judge_client: Any | None = None
    if enable_judge:
        judge_cfg = _resolve_api_cfg(
            config,
            override_model=str(judge_model or "Qwen/Qwen3.5-4B"),
            overrides={
                **dict(api_overrides or {}),
                "temperature": 0.0,
                "max_tokens": 240,
            },
        )
        judge_client = OpenAI(api_key=judge_cfg["api_key"], base_url=judge_cfg["api_base"])

    stage3_root = _stage3_open_root_from_manifest(Path(manifest_path))
    experiments_root = (stage3_root / "experiments_open").resolve()
    experiments_root.mkdir(parents=True, exist_ok=True)

    manifest_tag = _safe_name(Path(manifest_path).stem)[:48]
    model_tag = _safe_name(str(api_cfg["model"]))
    unique_experiment = _as_bool((api_overrides or {}).get("unique_experiment", config.get("unique_experiment", False)), False)
    run_id = f"{scene_id}_{model_tag}_{manifest_tag}_unique" if unique_experiment else f"{scene_id}_{model_tag}_{manifest_tag}_{_now_ts()}"
    run_dir = (experiments_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    requests_path = run_dir / "requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    parsed_path = run_dir / "parsed_predictions.jsonl"
    requests_txt_path = run_dir / "requests.txt"
    responses_txt_path = run_dir / "responses.txt"
    judge_txt_path = run_dir / "judge.txt"
    failed_indices_path = run_dir / "failed_request_indices.json"

    completed_sample_ids: set[str] = set()
    latest_response_by_sample: dict[str, dict[str, Any]] = {}
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
                latest_response_by_sample[sample_id] = row
                latest_status_by_sample[sample_id] = str(row.get("request_status", "") or "").strip().lower()
        completed_sample_ids = {sample_id for sample_id, status in latest_status_by_sample.items() if status == "ok"}

    parsed_rows: list[dict[str, Any]] = []
    if unique_experiment and parsed_path.exists():
        for line in parsed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                parsed_rows.append(row)

    sample_meta_by_id: dict[str, dict[str, Any]] = {}
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", "") or "").strip()
        if not sample_id:
            continue
        sample_meta_by_id[sample_id] = {
            "sample_index": int(sample_index),
            "sample_ordinal": int(sample_index + 1),
            "sample_id": sample_id,
            "form": str(sample.get("form", "") or ""),
            "task_name": str(sample.get("task_name", sample.get("form", "")) or ""),
            "mode": str(sample.get("mode", "") or ""),
            "difficulty_band": str(sample.get("difficulty_band", "") or ""),
        }

    pending_samples = list(samples)
    if unique_experiment:
        pending_samples = [sample for sample in pending_samples if str(sample.get("sample_id", "") or "").strip() not in completed_sample_ids]
    retry_connection_error_sample_ids = {
        sample_id
        for sample_id, response_row in latest_response_by_sample.items()
        if str(response_row.get("request_status", "") or "").strip().lower() == "error"
        and (
            "connection error" in str(response_row.get("raw_text", "") or "").lower()
            or "connection error" in json.dumps(response_row.get("raw_response", {}), ensure_ascii=False).lower()
        )
    }
    if unique_experiment and retry_connection_error_sample_ids:
        print(
            f"[{_iso_now()}] scene={scene_id} resume_retry_connection_error_samples={len(retry_connection_error_sample_ids)}",
            flush=True,
        )

    file_mode = "a" if unique_experiment else "w"
    judge_cache: dict[str, dict[str, Any]] = {}
    with requests_path.open(file_mode, encoding="utf-8") as req_fp, responses_path.open(file_mode, encoding="utf-8") as resp_fp, parsed_path.open(file_mode, encoding="utf-8") as pred_fp, requests_txt_path.open(file_mode, encoding="utf-8") as req_txt_fp, responses_txt_path.open(file_mode, encoding="utf-8") as resp_txt_fp, judge_txt_path.open(file_mode, encoding="utf-8") as judge_txt_fp:
        total_samples = len(pending_samples)
        for index, sample in enumerate(pending_samples, start=1):
            sample_id = str(sample.get("sample_id", "") or "").strip()
            sample_meta = dict(sample_meta_by_id.get(sample_id, {}) or {})
            print(
                f"[{_iso_now()}] scene={scene_id} progress={index}/{total_samples} sample_id={sample_id} model={api_cfg['model']}",
                flush=True,
            )

            messages, upload_paths = _build_openai_messages(sample, api_cfg)
            request_row = {
                "run_id": run_id,
                "sample_index": sample_meta.get("sample_index"),
                "sample_ordinal": sample_meta.get("sample_ordinal"),
                "sample_id": sample_id,
                "model": api_cfg["model"],
                "request_model": api_cfg["request_model"],
                "api_source": api_cfg.get("api_source", ""),
                "api_base": api_cfg.get("api_base", ""),
                "reasoning_mode": api_cfg.get("reasoning_mode", ""),
                "form": sample.get("form"),
                "task_name": sample.get("task_name", sample.get("form")),
                "task_group": sample.get("task_group"),
                "mode": sample.get("mode"),
                "difficulty_band": sample.get("difficulty_band"),
                "assistant_prefill": api_cfg.get("assistant_prefill", ""),
                "system_prompt_prefix": api_cfg.get("system_prompt_prefix", ""),
                "system_prompt_as_blocks": bool(api_cfg.get("system_prompt_as_blocks", False)),
                "request_extra_body": dict(api_cfg.get("request_extra_body", {}) or {}),
                "expected_answer_text": sample.get("answer_text"),
                "expected_answer_items": sample.get("answer_items"),
                "expected_visible_count": sample.get("visible_count"),
                "expected_visible_intervals_sec": sample.get("visible_intervals_sec"),
                "messages_preview": _messages_preview(messages),
                "inputs": upload_paths,
                "qa_format": sample.get("qa_format"),
                "response_kind": sample.get("response_kind"),
                "manifest_mode": sample.get("manifest_mode"),
            }

            request_status = "ok"
            raw_text = ""
            raw_meta: dict[str, Any] = {}
            latency_ms: float | None = None
            try:
                raw_text, raw_meta, latency_ms_val = _chat_completion_with_retry(
                    client=eval_client,
                    api_cfg=api_cfg,
                    messages=messages,
                    max_tokens=int(api_cfg.get("max_tokens", 600) or 600),
                    temperature=float(api_cfg.get("temperature", 0.0) or 0.0),
                )
                latency_ms = float(latency_ms_val)
            except Exception as exc:
                request_status = "error"
                raw_text = str(exc)
                raw_meta = {"error": str(exc)}
                latency_ms = None

            response_row = {
                "run_id": run_id,
                "sample_index": sample_meta.get("sample_index"),
                "sample_ordinal": sample_meta.get("sample_ordinal"),
                "sample_id": sample_id,
                "request_status": request_status,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "raw_response": raw_meta,
                "usage": dict(raw_meta.get("usage", {}) or {}) if isinstance(raw_meta.get("usage", {}), dict) else {},
            }

            form = str(sample.get("form", "") or "")
            gold_answer_items = _normalize_open_answer_items(sample.get("answer_items", []))
            gold_visible_count = int(sample.get("visible_count", -999) or -999)
            gold_visible_intervals = _normalize_interval_rows(sample.get("visible_intervals_sec", []))
            answer_parse_ok = False
            parsed_payload: Any = None
            behavior_judge_meta: dict[str, Any] | None = None
            behavior_judge_used = False
            parsed_row: dict[str, Any]
            request_ok = str(request_status).strip().lower() == "ok"

            if form == "self_instance_recognition_joint":
                pred_answer_items: list[dict[str, Any]] = []
                if request_ok:
                    pred_answer_items, answer_parse_ok, parsed_payload = _parse_self_instance_answer(raw_text)
                gold_count = len(gold_answer_items)
                pred_count = len(pred_answer_items)
                matched_pred_to_gold = [-1 for _ in pred_answer_items]
                matched_gold_to_pred = [-1 for _ in gold_answer_items]
                if request_ok and pred_answer_items and gold_answer_items and judge_client is not None and judge_cfg is not None:
                    behavior_judge_used = True
                    try:
                        behavior_judge_meta = _judge_self_items(
                            gold_items=gold_answer_items,
                            pred_items=pred_answer_items,
                            judge_client=judge_client,
                            judge_cfg=judge_cfg,
                            cache=judge_cache,
                        )
                    except Exception as exc:
                        behavior_judge_meta = {
                            "judge_error": str(exc),
                            "judge_parsed_ok": False,
                            "judge_matches": [],
                            "judge_raw_text": "",
                            "judge_raw_response": {"error": str(exc)},
                            "judge_latency_ms": None,
                        }

                if behavior_judge_meta and isinstance(behavior_judge_meta.get("judge_matches"), list):
                    used_gold: set[int] = set()
                    for match_row in list(behavior_judge_meta.get("judge_matches", []) or []):
                        try:
                            pred_idx = int(match_row.get("pred_index", -1))
                        except Exception:
                            pred_idx = -1
                        try:
                            gold_idx = int(match_row.get("gold_index", -1))
                        except Exception:
                            gold_idx = -1
                        semantic_match = bool(match_row.get("semantic_match", False))
                        if not semantic_match or pred_idx < 0 or pred_idx >= len(pred_answer_items) or gold_idx < 0 or gold_idx >= len(gold_answer_items):
                            continue
                        if gold_idx in used_gold:
                            continue
                        matched_pred_to_gold[pred_idx] = gold_idx
                        matched_gold_to_pred[gold_idx] = pred_idx
                        used_gold.add(gold_idx)

                matched_pred_count = sum(1 for idx in matched_pred_to_gold if idx >= 0)
                matched_gold_count = sum(1 for idx in matched_gold_to_pred if idx >= 0)
                behavior_precision = (matched_pred_count / float(pred_count)) if pred_count > 0 else (1.0 if gold_count == 0 else 0.0)
                behavior_recall = (matched_gold_count / float(gold_count)) if gold_count > 0 else (1.0 if pred_count == 0 else 0.0)
                behavior_f1 = (
                    2.0 * behavior_precision * behavior_recall / max(1e-6, behavior_precision + behavior_recall)
                    if (pred_count > 0 or gold_count > 0)
                    else 1.0
                )
                set_instance_acc = bool(pred_count == gold_count and matched_pred_count == pred_count and matched_gold_count == gold_count and gold_count > 0 or (pred_count == 0 and gold_count == 0))

                per_gold_tious: list[float] = []
                interval_tp = 0
                for gold_idx, gold_item in enumerate(gold_answer_items):
                    pred_idx = matched_gold_to_pred[gold_idx] if gold_idx < len(matched_gold_to_pred) else -1
                    gold_intervals = _normalize_interval_rows(gold_item.get("intervals_sec", []))
                    if pred_idx < 0 or pred_idx >= len(pred_answer_items):
                        per_gold_tious.append(0.0)
                        continue
                    pred_intervals = _normalize_interval_rows(pred_answer_items[pred_idx].get("intervals_sec", []))
                    tiou = _item_best_tiou(gold_intervals, pred_intervals)
                    per_gold_tious.append(tiou)
                    if tiou >= 0.5:
                        interval_tp += 1
                self_temporal_loc_precision = (interval_tp / float(pred_count)) if pred_count > 0 else (1.0 if gold_count == 0 else 0.0)
                self_temporal_loc_recall = (interval_tp / float(gold_count)) if gold_count > 0 else (1.0 if pred_count == 0 else 0.0)
                self_temporal_loc_f1 = (
                    2.0
                    * self_temporal_loc_precision
                    * self_temporal_loc_recall
                    / max(1e-6, self_temporal_loc_precision + self_temporal_loc_recall)
                    if (pred_count > 0 or gold_count > 0)
                    else 1.0
                )
                self_temporal_loc_mean_tiou = _mean(per_gold_tious)
                parsed_row = {
                    "run_id": run_id,
                    "sample_index": sample_meta.get("sample_index"),
                    "sample_ordinal": sample_meta.get("sample_ordinal"),
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "engine": engine,
                    "mode": sample.get("mode"),
                    "task_group": sample.get("task_group"),
                    "task_name": sample.get("task_name", form),
                    "task_display_name": sample.get("task_display_name", ""),
                    "set_id": sample.get("set_id", ""),
                    "set_name": sample.get("set_name", ""),
                    "mission_type": sample.get("mission_type", sample.get("set_name", "")),
                    "mission_subtype": sample.get("mission_subtype", sample.get("set_id", "")),
                    "service_scenario": sample.get("service_scenario", sample.get("mode", "")),
                    "landmark_category": sample.get("landmark_category", ""),
                    "difficulty_band": sample.get("difficulty_band", ""),
                    "form": form,
                    "gold_answer_items": gold_answer_items,
                    "pred_answer_items": pred_answer_items,
                    "pred_item_count": pred_count,
                    "gold_item_count": gold_count,
                    "matched_pred_to_gold": matched_pred_to_gold,
                    "matched_gold_to_pred": matched_gold_to_pred,
                    "behavior_judge_used": bool(behavior_judge_used),
                    "behavior_judge_meta": behavior_judge_meta,
                    "answer_parse_ok": bool(answer_parse_ok and request_ok),
                    "parse_ok": bool(answer_parse_ok and request_ok),
                    "behavior_match_precision": behavior_precision,
                    "behavior_match_recall": behavior_recall,
                    "behavior_match_f1": behavior_f1,
                    "set_instance_acc": bool(set_instance_acc),
                    "element_instance_precision": behavior_precision,
                    "element_instance_recall": behavior_recall,
                    "element_instance_f1": behavior_f1,
                    "self_temporal_loc_precision@0.5": self_temporal_loc_precision,
                    "self_temporal_loc_recall@0.5": self_temporal_loc_recall,
                    "self_temporal_loc_f1@0.5": self_temporal_loc_f1,
                    "self_temporal_loc_mean_tIoU": self_temporal_loc_mean_tiou,
                    "count_consistency": bool(pred_count == gold_count),
                    "latency_ms": latency_ms,
                    "prompt_tokens": response_row["usage"].get("prompt_tokens"),
                    "completion_tokens": response_row["usage"].get("completion_tokens"),
                    "total_tokens": response_row["usage"].get("total_tokens"),
                }
            else:
                pred_visible_count, pred_visible_intervals = -999, []
                if request_ok:
                    pred_visible_count, pred_visible_intervals, answer_parse_ok, parsed_payload = _parse_visibility_answer(raw_text)
                count_abs_error = abs(int(pred_visible_count) - int(gold_visible_count)) if pred_visible_count >= 0 else abs(int(gold_visible_count))
                count_exact = bool(pred_visible_count == gold_visible_count and pred_visible_count >= 0)
                count_within1 = bool(pred_visible_count >= 0 and abs(int(pred_visible_count) - int(gold_visible_count)) <= 1)
                tp_03, fp_03, fn_03 = _match_interval_metrics(gold_visible_intervals, pred_visible_intervals, 0.3)
                tp_05, fp_05, fn_05 = _match_interval_metrics(gold_visible_intervals, pred_visible_intervals, 0.5)
                per_gold_tious = [max((_interval_iou(pred, gold) for pred in pred_visible_intervals), default=0.0) for gold in gold_visible_intervals]
                mean_best_tiou = _mean(per_gold_tious)
                seg_precision_03 = float(tp_03) / float(max(1, tp_03 + fp_03))
                seg_recall_03 = float(tp_03) / float(max(1, tp_03 + fn_03))
                seg_f1_03 = 2.0 * seg_precision_03 * seg_recall_03 / max(1e-6, seg_precision_03 + seg_recall_03)
                seg_precision_05 = float(tp_05) / float(max(1, tp_05 + fp_05))
                seg_recall_05 = float(tp_05) / float(max(1, tp_05 + fn_05))
                seg_f1_05 = 2.0 * seg_precision_05 * seg_recall_05 / max(1e-6, seg_precision_05 + seg_recall_05)
                parsed_row = {
                    "run_id": run_id,
                    "sample_index": sample_meta.get("sample_index"),
                    "sample_ordinal": sample_meta.get("sample_ordinal"),
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "engine": engine,
                    "mode": sample.get("mode"),
                    "task_group": sample.get("task_group"),
                    "task_name": sample.get("task_name", form),
                    "task_display_name": sample.get("task_display_name", ""),
                    "set_id": sample.get("set_id", ""),
                    "set_name": sample.get("set_name", ""),
                    "mission_type": sample.get("mission_type", sample.get("set_name", "")),
                    "mission_subtype": sample.get("mission_subtype", sample.get("set_id", "")),
                    "service_scenario": sample.get("service_scenario", sample.get("mode", "")),
                    "landmark_category": sample.get("landmark_category", ""),
                    "difficulty_band": sample.get("difficulty_band", ""),
                    "form": form,
                    "gold_visible_count": gold_visible_count,
                    "pred_visible_count": pred_visible_count,
                    "gold_visible_intervals_sec": gold_visible_intervals,
                    "pred_visible_intervals_sec": pred_visible_intervals,
                    "answer_parse_ok": bool(answer_parse_ok and request_ok),
                    "parse_ok": bool(answer_parse_ok and request_ok),
                    "count_exact": bool(count_exact),
                    "count_within1": bool(count_within1),
                    "count_abs_error": int(count_abs_error),
                    "segment_precision@0.3": seg_precision_03,
                    "segment_recall@0.3": seg_recall_03,
                    "segment_f1@0.3": seg_f1_03,
                    "segment_precision@0.5": seg_precision_05,
                    "segment_recall@0.5": seg_recall_05,
                    "segment_f1@0.5": seg_f1_05,
                    "mean_best_tIoU": mean_best_tiou,
                    "count_consistency": bool(len(pred_visible_intervals) == len(gold_visible_intervals)),
                    "latency_ms": latency_ms,
                    "prompt_tokens": response_row["usage"].get("prompt_tokens"),
                    "completion_tokens": response_row["usage"].get("completion_tokens"),
                    "total_tokens": response_row["usage"].get("total_tokens"),
                }

            parsed_rows.append(parsed_row)

            req_fp.write(json.dumps({"run_id": run_id, **request_row}, ensure_ascii=False) + "\n")
            resp_fp.write(json.dumps(response_row, ensure_ascii=False) + "\n")
            pred_fp.write(json.dumps(parsed_row, ensure_ascii=False) + "\n")
            req_fp.flush()
            resp_fp.flush()
            pred_fp.flush()

            req_txt_fp.write(
                f"[{_iso_now()}] sample_id={sample_id} form={sample.get('form')} model={api_cfg['model']}\n"
                f"request_model={api_cfg['request_model']} api_source={api_cfg.get('api_source','')} reasoning_mode={api_cfg.get('reasoning_mode','')}\n"
                f"expected_answer_text={json.dumps(sample.get('answer_text'), ensure_ascii=False)}\n"
                f"expected_answer_items={json.dumps(sample.get('answer_items'), ensure_ascii=False)}\n"
                f"expected_visible_count={json.dumps(sample.get('visible_count'), ensure_ascii=False)}\n"
                f"expected_visible_intervals_sec={json.dumps(sample.get('visible_intervals_sec'), ensure_ascii=False)}\n"
                f"messages_preview={json.dumps(request_row.get('messages_preview', []), ensure_ascii=False)}\n"
                f"inputs={json.dumps(request_row.get('inputs', []), ensure_ascii=False)}\n\n"
            )
            resp_txt_fp.write(
                f"[{_iso_now()}] sample_id={sample_id} status={request_status} latency_ms={latency_ms} "
                f"prompt_tokens={response_row['usage'].get('prompt_tokens')} completion_tokens={response_row['usage'].get('completion_tokens')} total_tokens={response_row['usage'].get('total_tokens')}\n"
                f"raw_text:\n{raw_text}\n"
                f"raw_response={json.dumps(raw_meta, ensure_ascii=False)}\n\n"
            )
            judge_txt_fp.write(
                f"[{_iso_now()}] sample_id={sample_id} form={sample.get('form')}\n"
                f"gold_answer_items={json.dumps(sample.get('answer_items'), ensure_ascii=False)}\n"
                f"pred_answer_items={json.dumps(parsed_row.get('pred_answer_items', []), ensure_ascii=False)}\n"
                f"gold_visible_count={json.dumps(sample.get('visible_count'), ensure_ascii=False)} pred_visible_count={json.dumps(parsed_row.get('pred_visible_count'), ensure_ascii=False)}\n"
                f"gold_visible_intervals_sec={json.dumps(sample.get('visible_intervals_sec'), ensure_ascii=False)} pred_visible_intervals_sec={json.dumps(parsed_row.get('pred_visible_intervals_sec', []), ensure_ascii=False)}\n"
                f"behavior_judge_used={json.dumps(parsed_row.get('behavior_judge_used', False), ensure_ascii=False)} behavior_judge_meta={json.dumps(parsed_row.get('behavior_judge_meta', None), ensure_ascii=False)}\n"
                f"parse_ok={json.dumps(parsed_row.get('parse_ok', False), ensure_ascii=False)}\n\n"
            )
            req_txt_fp.flush()
            resp_txt_fp.flush()
            judge_txt_fp.flush()

            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "run_id": run_id,
                            "model": api_cfg["model"],
                            "completed": len(parsed_rows),
                            "total": total_samples,
                            "sample_id": sample_id,
                            "form": sample.get("form"),
                            "task_name": sample.get("task_name", sample.get("form")),
                            "request_status": request_status,
                            "parse_ok": bool(parsed_row.get("parse_ok", False)),
                            "latency_ms": latency_ms,
                        }
                    )
                except Exception:
                    pass

    latest_response_by_sample: dict[str, dict[str, Any]] = {}
    if responses_path.exists():
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                sample_id = str(row.get("sample_id", "") or "").strip()
                if sample_id:
                    latest_response_by_sample[sample_id] = row
    failed_request_rows_final: list[dict[str, Any]] = []
    for sample_id, response_row in latest_response_by_sample.items():
        if str(response_row.get("request_status", "") or "").strip().lower() != "error":
            continue
        meta = dict(sample_meta_by_id.get(sample_id, {}) or {})
        if not meta:
            meta = {
                "sample_index": 0,
                "sample_ordinal": 1,
                "sample_id": sample_id,
                "form": "",
                "task_name": "",
                "mode": "",
                "difficulty_band": "",
            }
        raw_response = dict(response_row.get("raw_response", {}) or {}) if isinstance(response_row.get("raw_response", {}), dict) else {}
        meta["request_retry_attempts"] = int(raw_response.get("request_retry_attempts", api_cfg.get("request_retry_attempts", 0)) or 0)
        meta["request_retry_errors"] = list(raw_response.get("request_retry_errors", []) or [])
        failed_request_rows_final.append(meta)
    failed_request_rows_final.sort(key=lambda row: int(row.get("sample_index", 0) or 0))
    failed_indices_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": "stage3",
                "scene_id": scene_id,
                "engine": engine,
                "model": api_cfg["model"],
                "manifest_path": _path_for_json(Path(manifest_path)),
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

    summary = _summarize_predictions(parsed_rows)
    report = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "scene_id": scene_id,
        "engine": engine,
        "model": api_cfg["model"],
        "request_model": api_cfg["request_model"],
        "manifest_path": _path_for_json(Path(manifest_path)),
        "mode": "open_video_qa_eval",
        "judge_model": str(judge_model or "") if enable_judge else None,
        "judge_enabled": bool(enable_judge),
        "judge_cache_size": len(judge_cache),
        "unique_experiment": bool(unique_experiment),
        "resumed_completed_sample_count": len(completed_sample_ids),
        "resumed_connection_error_sample_count": len(retry_connection_error_sample_ids),
        "run_sample_count": len(pending_samples),
        "total_manifest_sample_count": len(samples),
        "summary": summary,
        "api_overrides": dict(api_overrides or {}),
        "resumed_connection_error_sample_ids": sorted(retry_connection_error_sample_ids),
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = experiments_root / f"{scene_id}.latest_report.json"
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "report_path": report_path,
        "latest_report_path": latest_path,
        "requests_path": requests_path,
        "responses_path": responses_path,
        "parsed_path": parsed_path,
        "failed_indices_path": failed_indices_path,
        "requests_txt_path": requests_txt_path,
        "responses_txt_path": responses_txt_path,
        "judge_txt_path": judge_txt_path,
        "report": report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage3 open QA video evaluation with answer_text / answer_items scoring")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--engine", default="airsim")
    parser.add_argument("--task-name", default="UAV-DualCog-V1-Open")
    parser.add_argument("--model", default="")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--disable-judge", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    manifest_path = _resolve_manifest_path(
        Path(args.manifest_path).resolve() if str(args.manifest_path or "").strip() else None,
        scene_id=str(args.scene_id or "").strip(),
        task_name=str(args.task_name or "").strip(),
        task_pipeline_root=str(config.get("task_pipeline_root_dir", "task_pipeline_data") or "task_pipeline_data"),
    )
    result = run_open_qa_experiment(
        config=config,
        manifest_path=manifest_path,
        override_model=str(args.model or "").strip() or None,
        judge_model=str(args.judge_model or "").strip() or None,
        enable_judge=not bool(args.disable_judge),
        limit=int(args.limit or 0) or None,
        api_overrides={
            "unique_experiment": bool(config.get("unique_experiment", False)),
            "concurrency": int(config.get("experiment_concurrency", 16) or 16),
        },
    )
    print(json.dumps({"report_path": str(result["report_path"]), "run_dir": str(result["run_dir"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
