#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TASKS = {
    "self_where",
    "self_what",
    "env_where",
    "env_how",
    "flight_behavior",
    "landmark_visibility",
}

IMAGE_TASKS = {"self_where", "self_what", "env_where", "env_how"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_asset(path: Path, payload: dict[str, Any], value: str, *, use_asset_root: bool = True) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if candidate.is_absolute():
        return candidate.resolve()
    root = path.parent
    if use_asset_root and str(payload.get("asset_root", "") or "").strip():
        asset_root = Path(os.path.expandvars(os.path.expanduser(str(payload["asset_root"]))))
        root = asset_root if asset_root.is_absolute() else (path.parent / asset_root)
    return (root / candidate).resolve()


def validate_manifest(path: Path, *, require_final_landmarks: bool = False) -> list[str]:
    errors: list[str] = []
    payload = _load_json(path)
    for key in ("scene_id", "gs_checkpoint", "geometry_path", "T_world_to_gs", "camera_intrinsics", "landmarks"):
        if key not in payload:
            errors.append(f"manifest missing {key}")
    transform = np.asarray(payload.get("T_world_to_gs", []), dtype=float)
    if transform.shape != (4, 4):
        errors.append("T_world_to_gs must be 4x4")
    landmarks = payload.get("landmarks", []) or []
    target = int(payload.get("landmark_count_target", 20) or 20)
    candidate_target = int(payload.get("candidate_landmark_count_target", 100) or 100)
    if candidate_target < target:
        errors.append("candidate_landmark_count_target must be at least landmark_count_target")
    if require_final_landmarks and len(landmarks) != target:
        errors.append(f"expected {target} final landmarks, found {len(landmarks)}")
    if not require_final_landmarks and len(landmarks) not in {0, target}:
        errors.append(f"landmarks must be empty before review or contain {target} final items, found {len(landmarks)}")
    ids = [str(item.get("landmark_id", "")) for item in landmarks]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        errors.append("landmark_id values must be unique and non-empty")
    for item in landmarks:
        bbox_payload = item.get("bbox_3d_world", item.get("bbox_3d", []))
        if isinstance(bbox_payload, dict):
            bbox_payload = [bbox_payload.get("min", []), bbox_payload.get("max", [])]
        bbox = np.asarray(bbox_payload, dtype=float)
        if bbox.shape != (2, 3) or np.any(bbox[1] <= bbox[0]):
            errors.append(f"invalid 3D bbox for {item.get('landmark_id', '<unknown>')}")
        if len(item.get("reference_views", []) or []) < 4:
            errors.append(f"fewer than four reference views for {item.get('landmark_id', '<unknown>')}")
    for asset_key in ("gs_checkpoint", "geometry_path"):
        value = payload.get(asset_key)
        if value and not _resolve_manifest_asset(path, payload, str(value)).exists():
            errors.append(f"asset not found: {asset_key}={value}")
    backend = str(payload.get("renderer_backend", "standard_3dgs_ply") or "standard_3dgs_ply")
    if backend == "citygaussian_v2_checkpoint":
        value = str(payload.get("citygaussian_root", "") or "")
        if not value:
            errors.append("citygaussian_root is required for CityGaussianV2 checkpoints")
        elif not _resolve_manifest_asset(path, payload, value, use_asset_root=False).exists():
            errors.append(f"asset not found: citygaussian_root={value}")
    return errors


def validate_dataset(root: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for jsonl in root.rglob("*.jsonl"):
        for line_number, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{jsonl}:{line_number}: {exc}")
                continue
            rows.append(row)
            task = str(row.get("task_type", row.get("task", "")))
            counts[task] += 1
            if task and task not in TASKS:
                errors.append(f"unknown task type {task} in {jsonl}:{line_number}")
            bbox = row.get("bbox_xyxy_norm") or row.get("bbox")
            if bbox is not None:
                values = np.asarray(bbox, dtype=float)
                if values.shape != (4,) or np.any(values < 0) or np.any(values > 1) or values[2] <= values[0] or values[3] <= values[1]:
                    errors.append(f"invalid normalized bbox in {jsonl}:{line_number}")
            intervals = row.get("temporal_intervals") or row.get("intervals")
            if intervals is not None:
                for interval in intervals:
                    if len(interval) != 2 or float(interval[1]) <= float(interval[0]):
                        errors.append(f"invalid interval in {jsonl}:{line_number}")
            for key in ("image_path", "video_path"):
                asset = row.get(key)
                if asset:
                    asset_path = (jsonl.parent / str(asset)).resolve()
                    if not asset_path.exists():
                        errors.append(f"missing asset {asset_path}")
                    elif key == "image_path" and cv2.imread(str(asset_path)) is None:
                        errors.append(f"unreadable image {asset_path}")
    summary = {"samples": len(rows), "task_counts": dict(sorted(counts.items())), "errors": len(errors)}
    return errors, summary


def _valid_normalized_bbox(value: Any) -> bool:
    try:
        bbox = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(
        bbox.shape == (4,)
        and np.all(np.isfinite(bbox))
        and np.all(bbox >= 0)
        and np.all(bbox <= 1)
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def validate_stage4_manifest(
    path: Path,
    *,
    workspace: Path,
    expected_samples: int = 160,
    expected_landmarks: int = 20,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    payload = _load_json(path)
    rows = payload.get("samples", []) if isinstance(payload, dict) else []
    counts = Counter(str(row.get("task_type", "")) for row in rows)
    difficulty_counts = Counter(
        (str(row.get("task_type", "")), str(row.get("difficulty", ""))) for row in rows
    )
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    landmark_ids = {str(row.get("landmark_id", "")) for row in rows if row.get("landmark_id")}
    if len(rows) != expected_samples:
        errors.append(f"expected {expected_samples} Stage 4 samples, found {len(rows)}")
    if len(landmark_ids) != expected_landmarks:
        errors.append(f"expected {expected_landmarks} Stage 4 landmarks, found {len(landmark_ids)}")
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        errors.append("Stage 4 sample_id values must be unique and non-empty")
    unexpected = set(counts) - IMAGE_TASKS
    if unexpected:
        errors.append(f"unexpected Stage 4 task types: {sorted(unexpected)}")
    per_task_expected = expected_samples // len(IMAGE_TASKS)
    for task in sorted(IMAGE_TASKS):
        if counts[task] != per_task_expected:
            errors.append(f"expected {per_task_expected} samples for {task}, found {counts[task]}")

    image_cache: dict[Path, dict[str, float | int]] = {}
    for index, row in enumerate(rows, 1):
        prefix = f"sample {index} ({row.get('sample_id', '<missing>')})"
        if not _valid_normalized_bbox(row.get("answer_bbox_xyxy_norm")):
            errors.append(f"{prefix}: invalid answer_bbox_xyxy_norm")
        if not str(row.get("answer_option_id", "")).strip():
            errors.append(f"{prefix}: missing answer_option_id")
        if not str(row.get("user_prompt", row.get("prompt_text", ""))).strip():
            errors.append(f"{prefix}: missing prompt")
        for key in ("reference_image", "reference_image_with_bbox", "target_image"):
            value = str(row.get(key, "") or "")
            asset = Path(value)
            if not asset.is_absolute():
                asset = workspace / asset
            asset = asset.resolve()
            if not asset.exists():
                errors.append(f"{prefix}: missing {key} {asset}")
                continue
            if asset not in image_cache:
                image = cv2.imread(str(asset))
                if image is None:
                    errors.append(f"{prefix}: unreadable {key} {asset}")
                    continue
                image_cache[asset] = {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "mean": float(image.mean()),
                    "std": float(image.std()),
                    "black_ratio": float(np.all(image < 5, axis=2).mean()),
                }
                quality = image_cache[asset]
                if quality["std"] < 5.0 or quality["black_ratio"] > 0.35:
                    errors.append(f"{prefix}: failed image quality gate for {key} {asset}")

    summary = {
        "samples": len(rows),
        "landmarks": len(landmark_ids),
        "task_counts": dict(sorted(counts.items())),
        "difficulty_counts": {
            f"{task}:{difficulty}": count
            for (task, difficulty), count in sorted(difficulty_counts.items())
        },
        "unique_images_checked": len(image_cache),
        "minimum_image_std": round(min((float(row["std"]) for row in image_cache.values()), default=0.0), 4),
        "maximum_black_ratio": round(max((float(row["black_ratio"]) for row in image_cache.values()), default=0.0), 6),
        "errors": len(errors),
    }
    return errors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the UAV-DualCog real-scene 3DGS pilot")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--stage4-manifest", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--expected-stage4-samples", type=int, default=160)
    parser.add_argument("--expected-landmarks", type=int, default=20)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-final-landmarks", action="store_true")
    args = parser.parse_args()
    if not args.manifest and not args.dataset_root and not args.stage4_manifest:
        parser.error("provide --manifest, --dataset-root, or --stage4-manifest")
    errors: list[str] = []
    summary: dict[str, Any] = {}
    if args.manifest:
        manifest_errors = validate_manifest(args.manifest, require_final_landmarks=args.require_final_landmarks)
        errors.extend(manifest_errors)
        summary.update({"manifest": str(args.manifest), "manifest_errors": len(manifest_errors)})
    if args.dataset_root:
        dataset_errors, dataset_summary = validate_dataset(args.dataset_root)
        errors.extend(dataset_errors)
        summary["dataset"] = dataset_summary
    if args.stage4_manifest:
        stage4_errors, stage4_summary = validate_stage4_manifest(
            args.stage4_manifest,
            workspace=args.workspace.resolve(),
            expected_samples=args.expected_stage4_samples,
            expected_landmarks=args.expected_landmarks,
        )
        errors.extend(stage4_errors)
        summary["stage4_manifest"] = str(args.stage4_manifest)
        summary["stage4"] = stage4_summary
    summary["ok"] = not errors
    summary["errors"] = errors
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
