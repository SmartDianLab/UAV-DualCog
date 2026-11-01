#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


TASK_TITLES = {
    "self_where": "Self-relative Position Reasoning",
    "self_what": "Action-conditioned Future Observation Prediction",
    "env_where": "Landmark-relative Position Reasoning",
    "env_how": "Goal-conditioned Action Decision",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _samples(path: Path) -> list[dict[str, Any]]:
    data = _load(path)
    if isinstance(data, list):
        return [dict(row) for row in data if isinstance(row, dict)]
    return [dict(row) for row in data.get("samples", []) if isinstance(row, dict)]


def _latest_stage4_manifest(root: Path) -> Path:
    matches = [path for path in root.rglob("*.json") if "manifests" in path.parts and "qa" in path.parts]
    matches = sorted(matches, key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No Stage 4 manifest found under {root}")
    return matches[-1]


def _resolve(path_text: str, workspace: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else workspace / path


def _copy_image(source: Path, media_root: Path, name: str) -> str:
    suffix = source.suffix.lower() if source.suffix else ".jpg"
    target = media_root / f"{name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return f"/media/real-world/{target.name}"


def _fmt_bbox(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 4:
        return ""
    return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"


def _image_quality(path: Path) -> float:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            gray.thumbnail((320, 240))
            histogram = gray.histogram()
            total = max(1, sum(histogram))
            dark_ratio = sum(histogram[:24]) / total
            bright_ratio = sum(histogram[250:]) / total
            contrast = float(ImageStat.Stat(gray).stddev[0])
            return contrast - 80.0 * dark_ratio - 30.0 * bright_ratio
    except (OSError, ValueError):
        return float("-inf")


def _example_quality(sample: dict[str, Any], workspace: Path) -> float:
    image_paths = [
        str(sample.get("reference_image_with_bbox") or sample.get("reference_image") or ""),
        str(sample.get("target_image") or sample.get("query_image") or sample.get("observation_image") or ""),
    ]
    qualities = [_image_quality(_resolve(path, workspace)) for path in image_paths if path]
    if len(qualities) != 2:
        return float("-inf")
    bbox = sample.get("answer_bbox_xyxy_norm", [])
    bbox_area = 0.0
    if isinstance(bbox, list) and len(bbox) == 4:
        bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    return min(qualities) + 20.0 * sqrt(bbox_area)


def _image_example(sample: dict[str, Any], workspace: Path, media_root: Path, scene_id: str) -> dict[str, Any]:
    task_id = str(sample.get("task_type", ""))
    sample_id = str(sample.get("sample_id", ""))
    panels: list[dict[str, Any]] = []
    ref_text = str(sample.get("reference_image_with_bbox") or sample.get("reference_image") or "")
    if ref_text:
        url = _copy_image(_resolve(ref_text, workspace), media_root, f"{scene_id}_{task_id}_reference")
        panels.append({"kind": "image", "label": "Reference landmark view", "url": url})
    target_text = str(sample.get("target_image") or sample.get("query_image") or sample.get("observation_image") or "")
    if target_text:
        url = _copy_image(_resolve(target_text, workspace), media_root, f"{scene_id}_{task_id}_query")
        overlay = {"label": "Ground-truth landmark", "color": "#16a085", "bbox": sample.get("answer_bbox_xyxy_norm", [])}
        panels.append({"kind": "image", "label": "Query observation", "url": url, "overlays": [overlay]})
    answer = sample.get("answer_option_id") or sample.get("answer_option_ids") or sample.get("gold_answer") or ""
    if isinstance(answer, list):
        answer = ", ".join(str(item) for item in answer)
    return {
        "example_id": sample_id,
        "scene_id": str(sample.get("scene_id", scene_id)),
        "difficulty": str(sample.get("difficulty", "")),
        "prompt_excerpt": str(sample.get("user_prompt") or sample.get("prompt_text") or sample.get("question") or ""),
        "input_summary": str(sample.get("landmark_description", "")),
        "task_id": task_id,
        "task_modality": "image",
        "gold_answer": str(answer),
        "gold_bbox": _fmt_bbox(sample.get("answer_bbox_xyxy_norm")),
        "predictions": [],
        "media_panels": panels,
    }


def _scene_payload(spec: dict[str, Any], workspace: Path, media_root: Path) -> dict[str, Any]:
    scene_id = str(spec["scene_id"])
    pilot_root = _resolve(str(spec["pilot_root"]), workspace)
    stage4_path = _latest_stage4_manifest(pilot_root)
    stage4 = _samples(stage4_path)
    candidates = _load(_resolve(str(spec["candidate_file"]), workspace))
    selection = _load(_resolve(str(spec["selection_file"]), workspace))
    valid_data = _load(_resolve(str(spec["valid_instances"]), workspace))
    valid_rows = valid_data.get("valid_instances", valid_data.get("instances", [])) if isinstance(valid_data, dict) else valid_data

    by_stage4: dict[str, dict[str, Any]] = {}
    for sample in stage4:
        task_id = str(sample.get("task_type", ""))
        if task_id in TASK_TITLES and (
            task_id not in by_stage4
            or _example_quality(sample, workspace) > _example_quality(by_stage4[task_id], workspace)
        ):
            by_stage4[task_id] = sample
    groups = [
        {
            "task_id": task_id,
            "task_title": TASK_TITLES[task_id],
            "examples": [_image_example(by_stage4[task_id], workspace, media_root, scene_id)],
        }
        for task_id in ("self_where", "self_what", "env_where", "env_how")
        if task_id in by_stage4
    ]
    candidate_rows = candidates.get("candidates", candidates) if isinstance(candidates, dict) else candidates
    selection_rows = selection.get("selections", selection.get("valid_instances", []))
    selected_ids = {str(row.get("instance_id", "")) for row in selection_rows}
    selected_candidates = [row for row in candidate_rows if str(row.get("instance_id", "")) in selected_ids]
    candidate_grids = {str(row.get("scene_grid_id", "")) for row in candidate_rows if row.get("scene_grid_id")}
    selected_grids = {str(row.get("scene_grid_id", "")) for row in selected_candidates if row.get("scene_grid_id")}
    task_counts = {
        task_id: sum(1 for row in stage4 if str(row.get("task_type", "")) == task_id)
        for task_id in ("self_where", "self_what", "env_where", "env_how")
    }
    return {
        "scene_id": scene_id,
        "scene": str(spec.get("scene", scene_id)),
        "checkpoint": str(spec.get("checkpoint", "")),
        "status": "image_task_generation_complete_model_evaluation_pending",
        "statistics": {
            "registered_views": int(spec.get("registered_views", 0)),
            "sampled_views": int(spec.get("sampled_views", 200)),
            "landmark_candidates": len(candidate_rows),
            "temporary_landmarks": len(selected_ids),
            "covered_candidate_grids": len(candidate_grids),
            "covered_temporary_grids": len(selected_grids),
            "registered_landmark_views": sum(len(row.get("rgb_views", [])) for row in valid_rows),
            "image_tasks": len(stage4),
        },
        "task_counts": task_counts,
        "audit": {
            "stage4_manifest": str(stage4_path.relative_to(workspace)),
            "selected_landmark_ids": sorted(selected_ids),
            "video_tasks": "deferred_after_free_viewpoint_render_quality_gate",
            "model_evaluation_status": "not_started",
        },
        "examples": {"groups": groups},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GauUscene pilots to the supplementary website.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if args.media_root.exists():
        shutil.rmtree(args.media_root)
    args.media_root.mkdir(parents=True, exist_ok=True)
    config = _load(_resolve(str(args.scene_config), workspace))
    scenes = [_scene_payload(dict(spec), workspace, args.media_root) for spec in config.get("scenes", [])]
    groups_by_task: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        for group in scene["examples"]["groups"]:
            target = groups_by_task.setdefault(
                group["task_id"],
                {"task_id": group["task_id"], "task_title": group["task_title"], "examples": []},
            )
            target["examples"].extend(group["examples"])
    aggregate = {
        key: sum(int(scene["statistics"].get(key, 0)) for scene in scenes)
        for key in (
            "registered_views",
            "sampled_views",
            "landmark_candidates",
            "temporary_landmarks",
            "covered_candidate_grids",
            "covered_temporary_grids",
            "registered_landmark_views",
            "image_tasks",
        )
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene": "GauUscene V2.1 direct-load CityGaussianV2 scenes",
        "status": "image_task_generation_complete_model_evaluation_pending",
        "statistics": aggregate,
        "scenes": scenes,
        "examples": {"groups": list(groups_by_task.values())},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["statistics"], ensure_ascii=False, indent=2))
    print(json.dumps({scene["scene_id"]: scene["task_counts"] for scene in scenes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
