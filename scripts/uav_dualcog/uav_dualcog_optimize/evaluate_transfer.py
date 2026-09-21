#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uav_dualcog.uav_dualcog_optimize.common import ensure_dir, resolve_path


def _run(cmd: List[str], dry_run: bool) -> int:
    print("[evaluate_transfer] cmd:", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _parse_scene_ids(scene_ids_raw: str) -> List[str]:
    items = [x.strip() for x in scene_ids_raw.split(",") if x.strip()]
    out: List[str] = []
    for it in items:
        if it.startswith("env_"):
            out.append(it)
        else:
            out.append(f"env_{it}")
    return out


def _parse_iso_ts(ts: str) -> float:
    if not ts:
        return float("-inf")
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return float("-inf")


def _model_match(report_model: str, target_model: str, mode: str) -> bool:
    a = str(report_model or "").strip()
    b = str(target_model or "").strip()
    if mode == "contains":
        return b in a
    return a == b


def _collect_latest_report_for_scene(
    bench_root: Path,
    scene_id: str,
    model: str,
    mode: str,
    stage: str,
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    rel = "qa/experiments" if stage == "stage4" else "stage3_tasks/experiments"
    scene_dir = bench_root / f"airsim_{scene_id}"
    exp_dir = scene_dir / rel
    if not exp_dir.exists():
        return None
    best_path: Optional[Path] = None
    best_obj: Optional[Dict[str, Any]] = None
    best_ts = float("-inf")
    for p in exp_dir.glob("*/report.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _model_match(str(obj.get("model", "")), model, mode):
            continue
        ts = _parse_iso_ts(str(obj.get("generated_at", "")))
        if ts >= best_ts:
            best_ts = ts
            best_path = p
            best_obj = obj
    if best_path is None or best_obj is None:
        return None
    return best_path, best_obj


def _wavg(rows: List[Tuple[float, float]]) -> Optional[float]:
    if not rows:
        return None
    wsum = sum(w for w, _ in rows if w > 0)
    if wsum <= 0:
        return None
    return sum(w * v for w, v in rows if w > 0) / wsum


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _aggregate_stage4(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["parse_success_rate", "option_accuracy", "bbox_acc@50iou", "bbox_mean_iou", "avg_latency_ms"]
    grouped_views = ["Self-Aware", "Environment-Aware"]
    grouped_tasks = ["self_where", "self_what", "env_where", "env_how"]
    out: Dict[str, Any] = {"scene_count": len(reports), "overall": {}, "grouped": {"view_definition": {}, "task_type": {}}}
    for k in keys:
        rows: List[Tuple[float, float]] = []
        for rp in reports:
            s = rp.get("summary", {}) or {}
            c = _safe_float(s.get("count"))
            v = _safe_float(s.get(k))
            if c and v is not None:
                rows.append((c, v))
        out["overall"][k] = _wavg(rows)
    for gv in grouped_views:
        node: Dict[str, Any] = {}
        for mk in ("option_accuracy", "bbox_acc@50iou", "bbox_mean_iou"):
            rows = []
            for rp in reports:
                s = rp.get("summary", {}) or {}
                g = ((s.get("grouped") or {}).get("view_definition") or {}).get(gv) or {}
                c = _safe_float(g.get("count"))
                v = _safe_float(g.get(mk))
                if c and v is not None:
                    rows.append((c, v))
            node[mk] = _wavg(rows)
        out["grouped"]["view_definition"][gv] = node
    for gt in grouped_tasks:
        node = {}
        for mk in ("option_accuracy", "bbox_acc@50iou", "bbox_mean_iou"):
            rows = []
            for rp in reports:
                s = rp.get("summary", {}) or {}
                g = ((s.get("grouped") or {}).get("task_type") or {}).get(gt) or {}
                c = _safe_float(g.get("count"))
                v = _safe_float(g.get(mk))
                if c and v is not None:
                    rows.append((c, v))
            node[mk] = _wavg(rows)
        out["grouped"]["task_type"][gt] = node
    return out


def _aggregate_stage3(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "parse_success_rate",
        "count_exact_acc",
        "count_within1_acc",
        "count_mae",
        "segment_f1@0.5",
        "mean_best_tIoU",
        "self_temporal_loc_f1@0.5",
        "self_temporal_loc_mean_tIoU",
        "set_instance_acc",
        "element_instance_f1",
        "avg_latency_ms",
    ]
    grouped_forms = ["self_instance_recognition_joint", "env_visibility_reasoning"]
    out: Dict[str, Any] = {"scene_count": len(reports), "overall": {}, "grouped": {"form": {}}}
    for k in keys:
        rows: List[Tuple[float, float]] = []
        for rp in reports:
            s = rp.get("summary", {}) or {}
            c = _safe_float(s.get("count"))
            v = _safe_float(s.get(k))
            if c and v is not None:
                rows.append((c, v))
        out["overall"][k] = _wavg(rows)
    for form in grouped_forms:
        node: Dict[str, Any] = {}
        for mk in ("main_metric", "segment_f1@0.5", "mean_best_tIoU", "self_temporal_loc_f1@0.5", "set_instance_acc"):
            rows = []
            for rp in reports:
                s = rp.get("summary", {}) or {}
                g = ((s.get("grouped") or {}).get("form") or {}).get(form) or {}
                c = _safe_float(g.get("count"))
                v = _safe_float(g.get(mk))
                if c and v is not None:
                    rows.append((c, v))
            node[mk] = _wavg(rows)
        out["grouped"]["form"][form] = node
    return out


def _derive_dual_scores(agg: Dict[str, Any]) -> Dict[str, Any]:
    img = (agg.get("stage4") or {}).get("overall") or {}
    vid = (agg.get("stage3") or {}).get("overall") or {}
    # Lightweight composed scores for model-comparison convenience only.
    image_dual = None
    if img.get("option_accuracy") is not None and img.get("bbox_acc@50iou") is not None and img.get("bbox_mean_iou") is not None:
        image_dual = 0.5 * img["option_accuracy"] + 0.25 * img["bbox_acc@50iou"] + 0.25 * img["bbox_mean_iou"]
    video_dual = None
    if vid.get("segment_f1@0.5") is not None and vid.get("mean_best_tIoU") is not None and vid.get("count_exact_acc") is not None:
        video_dual = 0.4 * vid["segment_f1@0.5"] + 0.3 * vid["mean_best_tIoU"] + 0.3 * vid["count_exact_acc"]
    gap = None
    if image_dual is not None and video_dual is not None:
        gap = image_dual - video_dual
    return {
        "image_dual_score": image_dual,
        "video_dual_score": video_dual,
        "cross_media_gap_image_minus_video": gap,
        "notes": "Composed dual scores are heuristic summary views for optimization comparison; keep task-level metrics as the primary evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage4 + Stage3 experiments and aggregate cross-scene transfer summary.")
    parser.add_argument("--spec", default="configs/uav_dualcog/task_pipeline/task_pipeline_uav_dualcog_v1.yaml")
    parser.add_argument("--model", required=True, help="Model alias used in task_pipeline experiment phase.")
    parser.add_argument("--model-match", choices=["exact", "contains"], default="exact")
    parser.add_argument("--benchmark-root", default="task_pipeline_data/UAV-DualCog-V1")
    parser.add_argument("--scene-ids", default="7,8,9,10,11,13,16,17,20,21,23,24")
    parser.add_argument("--output-dir", default="task_pipeline_data/UAV-DualCog-V1/optimization_eval")
    parser.add_argument("--run-experiment", action="store_true", help="Actually trigger task_pipeline experiments.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = resolve_path(args.spec)
    benchmark_root = resolve_path(args.benchmark_root)
    out_dir = resolve_path(args.output_dir)
    ensure_dir(out_dir)
    scene_ids = _parse_scene_ids(args.scene_ids)

    if args.run_experiment:
        for stage in ("stage4", "stage3"):
            cmd = [
                args.python_bin,
                "scripts/uav_dualcog/task_pipeline.py",
                "--spec",
                str(spec),
                "--stage",
                stage,
                "--phase",
                "experiment",
                "--experiment-models",
                args.model,
            ]
            rc = _run(cmd, dry_run=args.dry_run)
            if rc != 0:
                raise RuntimeError(f"task_pipeline experiment failed stage={stage}, rc={rc}")

    picked_stage4: List[Dict[str, Any]] = []
    picked_stage3: List[Dict[str, Any]] = []
    picked_paths: Dict[str, Dict[str, str]] = {"stage4": {}, "stage3": {}}
    missing: Dict[str, List[str]] = {"stage4": [], "stage3": []}
    for sid in scene_ids:
        p4 = _collect_latest_report_for_scene(benchmark_root, sid, args.model, args.model_match, "stage4")
        if p4 is None:
            missing["stage4"].append(sid)
        else:
            picked_paths["stage4"][sid] = str(p4[0])
            picked_stage4.append(p4[1])
        p3 = _collect_latest_report_for_scene(benchmark_root, sid, args.model, args.model_match, "stage3")
        if p3 is None:
            missing["stage3"].append(sid)
        else:
            picked_paths["stage3"][sid] = str(p3[0])
            picked_stage3.append(p3[1])

    agg = {
        "stage4": _aggregate_stage4(picked_stage4),
        "stage3": _aggregate_stage3(picked_stage3),
    }
    derived = _derive_dual_scores(agg)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "spec": str(spec),
        "model": args.model,
        "model_match": args.model_match,
        "scene_ids_requested": scene_ids,
        "missing_scenes": missing,
        "picked_reports": picked_paths,
        "aggregated": agg,
        "derived": derived,
        "notes": [
            "Each scene contributes the latest report.json matching the target model.",
            "Aggregation is weighted by per-scene sample count.",
            "Use this output for optimization-stage model comparison and ablation write-up.",
        ],
    }
    out_path = out_dir / f"transfer_eval_{args.model.replace('/', '_')}.latest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[evaluate_transfer] wrote {out_path}")


if __name__ == "__main__":
    main()
