#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow direct execution: python scripts/uav_dualcog/uav_dualcog_optimize/*.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uav_dualcog.uav_dualcog_optimize.common import (
    check_manifest_quality,
    deterministic_subsample,
    ensure_dir,
    find_latest_stage4_manifests,
    format_bbox_norm_json,
    load_json,
    normalize_xyxy_norm,
    resolve_path,
    xyxy_norm_to_1000,
)


def _reference_image_for_training(sample: Dict[str, Any]) -> str:
    return str(sample.get("reference_image_with_bbox") or sample.get("reference_image") or "").strip()


def _image_paths_for_sample(sample: Dict[str, Any]) -> List[str]:
    task_type = str(sample.get("task_type", "")).strip()
    image_paths: List[str] = []
    ref = _reference_image_for_training(sample)
    if task_type in {"self_where", "env_where", "env_how", "label_multiple_choice"}:
        image_paths = [ref, str(sample.get("target_image", "")).strip()]
    elif task_type == "self_what":
        image_paths = [ref]
        image_paths.extend(str(c.get("image", "")).strip() for c in list(sample.get("candidates", []) or []))
    else:
        image_paths = [ref]
    return [p for p in image_paths if p]


def _build_response_json(sample: Dict[str, Any]) -> str:
    answer = str(sample.get("answer_option_id", "")).strip()
    bbox_norm = normalize_xyxy_norm(sample.get("answer_bbox_xyxy_norm", [0, 0, 0, 0]))
    return (
        "{"
        f"\"answer_option_id\":\"{answer}\","
        f"\"bbox_xyxy_norm\":{format_bbox_norm_json(bbox_norm)}"
        "}"
    )


def _build_prompt_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    system_prompt = str(sample.get("system_prompt", "") or "").strip()
    user_prompt = str(sample.get("user_prompt", "") or sample.get("prompt_text", "") or "").strip()
    image_paths = _image_paths_for_sample(sample)
    image_note = (
        f"\n\nThe images are provided in order from Image 1 to Image {len(image_paths)}. "
        "Use the visual evidence in these images directly."
    )
    user_prompt = (user_prompt + image_note).strip()
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _build_completion_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"role": "assistant", "content": _build_response_json(sample)}]


def _to_record(sample: Dict[str, Any], scene_id: str) -> Dict[str, Any]:
    image_paths = _image_paths_for_sample(sample)
    bbox_norm = normalize_xyxy_norm(sample.get("answer_bbox_xyxy_norm", [0, 0, 0, 0]))
    return {
        "id": str(sample.get("sample_id", "")),
        "scene_id": scene_id,
        "task_type": str(sample.get("task_type", "")),
        "task_group": str(sample.get("task_group", "")),
        "difficulty": str(sample.get("difficulty", "")),
        "prompt": _build_prompt_messages(sample),
        "completion": _build_completion_messages(sample),
        "images": image_paths,
        "response_text": _build_response_json(sample),
        "answer_option_id": str(sample.get("answer_option_id", "")),
        "answer_bbox_xyxy_norm": bbox_norm,
        "answer_bbox_xyxy_1000": xyxy_norm_to_1000(bbox_norm),
        "reference_image": str(sample.get("reference_image", "")),
        "reference_image_with_bbox": str(sample.get("reference_image_with_bbox", "")),
        "target_image": str(sample.get("target_image", "")),
        "system_prompt": str(sample.get("system_prompt", "") or ""),
        "user_prompt": str(sample.get("user_prompt", "") or sample.get("prompt_text", "") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare UAV-DualCog-Train-V1 stage4 manifests into multimodal SFT/GRPO jsonl data."
    )
    parser.add_argument(
        "--train-root",
        default="task_pipeline_data/UAV-DualCog-Train-V1",
        help="Root dir containing airsim_env_*/qa/manifests/*.latest_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm",
        help="Output dir for generated jsonl and integrity report.",
    )
    parser.add_argument("--max-per-task", type=int, default=0, help="Optional per (task_type,difficulty) cap per scene for smoke runs.")
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    train_root = resolve_path(args.train_root)
    out_dir = resolve_path(args.output_dir)
    ensure_dir(out_dir)
    manifests = find_latest_stage4_manifests(train_root)
    if not manifests:
        raise FileNotFoundError(f"No latest stage4 manifests under {train_root}")

    all_records: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    global_counter = Counter()
    for m in manifests:
        quality = check_manifest_quality(m)
        d = load_json(m)
        scene_id = str(d.get("scene_id", "unknown"))
        samples = list(d.get("samples", []) or [])
        if args.max_per_task > 0:
            samples = deterministic_subsample(samples, args.max_per_task, args.seed)
        rows = [_to_record(s, scene_id) for s in samples]
        all_records.extend(rows)
        td_counter = Counter((r["task_type"], r["difficulty"]) for r in rows)
        for k, v in td_counter.items():
            global_counter[k] += v
        report_rows.append(
            {
                "scene_id": scene_id,
                "manifest_path": str(m),
                "sample_count_after_sampling": len(rows),
                "sample_count_manifest": quality.sample_count,
                "task_difficulty_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(td_counter.items())},
                "missing_file_count": quality.missing_file_count,
                "bad_bbox_count": quality.bad_bbox_count,
                "missing_answer_option_count": quality.option_missing_count,
            }
        )

    bucketed: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_records:
        bucketed[f"{r['task_type']}::{r['difficulty']}"].append(r)
    train_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    for key in sorted(bucketed.keys()):
        arr = sorted(bucketed[key], key=lambda x: x["id"])
        for i, item in enumerate(arr):
            if i % 10 == 0:
                valid_rows.append(item)
            else:
                train_rows.append(item)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_path = out_dir / f"uav_dualcog_train_all_{len(all_records)}_{ts}.jsonl"
    train_path = out_dir / f"uav_dualcog_train_{len(train_rows)}_{ts}.jsonl"
    valid_path = out_dir / f"uav_dualcog_valid_{len(valid_rows)}_{ts}.jsonl"
    all_latest = out_dir / "uav_dualcog_train_all.latest.jsonl"
    train_latest = out_dir / "uav_dualcog_train.latest.jsonl"
    valid_latest = out_dir / "uav_dualcog_valid.latest.jsonl"

    def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump_jsonl(all_path, all_records)
    dump_jsonl(train_path, train_rows)
    dump_jsonl(valid_path, valid_rows)
    dump_jsonl(all_latest, all_records)
    dump_jsonl(train_latest, train_rows)
    dump_jsonl(valid_latest, valid_rows)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "train_root": str(train_root),
        "manifests": [str(m) for m in manifests],
        "max_per_task": args.max_per_task,
        "seed": args.seed,
        "all_count": len(all_records),
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "global_task_difficulty_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(global_counter.items())},
        "scene_checks": report_rows,
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "valid": str(valid_path),
            "all_latest": str(all_latest),
            "train_latest": str(train_latest),
            "valid_latest": str(valid_latest),
        },
        "record_contract": {
            "prompt": "Conversational messages for the benchmark system/user prompt.",
            "completion": "Assistant JSON response with answer_option_id and bbox_xyxy_norm.",
            "images": "Ordered image paths aligned with the benchmark task input.",
        },
    }
    report_path = out_dir / "dataset_integrity_and_prepare_report.latest.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[prepare_train_dataset] done")
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))
    print(f"[prepare_train_dataset] report: {report_path}")


if __name__ == "__main__":
    main()
