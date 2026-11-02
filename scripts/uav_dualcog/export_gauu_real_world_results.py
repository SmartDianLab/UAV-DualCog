#!/usr/bin/env python3
"""Export the completed GauUscene pilot, examples, and scene gallery for the website."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageStat


LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


TASK_TITLES = {
    "self_where": "Landmark-relative Position Reasoning",
    "self_what": "Action-conditioned Future Observation Prediction",
    "env_where": "Self-relative Position Reasoning",
    "env_how": "Goal-conditioned Action Decision",
}
SCENE_NAMES = {"gauu_upper_campus": "Upper Campus", "gauu_smbu": "SMBU", "gauu_lfls": "LFLS"}
SCENE_REGISTERED = {"gauu_upper_campus": 358, "gauu_smbu": 415, "gauu_lfls": 553}
EXAMPLE_MODELS = (
    "gemini38",
    "gpt6astra",
    "gpt56sol",
    "muse13",
    "qwen37plus",
    "glm53",
    "kimi26",
    "mimo25",
    "qwen38_27b",
    "qwen36_27b",
)
FIGURE_MODELS = ("gemini38", "glm53", "gpt55", "mimo25")
DISPLAY_EXCLUDED_MODELS = {"Claude Opus 4.6", "Claude Opus 4.7", "Claude Opus 4.8"}
COLORS = ("#2563eb", "#d97706", "#0f766e", "#dc2626", "#7c3aed", "#64748b", "#0891b2", "#be123c", "#4d7c0f", "#a16207")
SEMANTIC_COLORS = {
    "building": "#4f83cc",
    "industrial_infrastructure": "#c96f73",
    "transport_infrastructure": "#7669ae",
    "urban_landscape": "#e09f3e",
    "transportation": "#b56576",
    "vegetation": "#5a9f68",
    "water": "#52a7c7",
    "other": "#8b78b7",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def quality(path: Path) -> float:
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            gray.thumbnail((320, 240))
            stat = ImageStat.Stat(gray)
            hist = gray.histogram()
            total = max(1, sum(hist))
            clipping = (sum(hist[:12]) + sum(hist[244:])) / total
            return float(stat.stddev[0]) - 80 * clipping
    except OSError:
        return -1e9


def bbox_area(sample: dict[str, Any]) -> float:
    box = sample.get("answer_bbox_xyxy_norm") or []
    if not isinstance(box, list) or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def example_score(sample: dict[str, Any], workspace: Path) -> float:
    paths = [sample.get("reference_image_with_bbox") or sample.get("reference_image"), sample.get("target_image")]
    if not all(paths):
        return -1e9
    area = bbox_area(sample)
    return min(quality(resolve(workspace, str(path))) for path in paths) + 25 * min(area, 0.2)


def copy_image(workspace: Path, media_root: Path, value: str, name: str) -> str:
    source = resolve(workspace, value)
    suffix = source.suffix.lower() or ".jpg"
    target = media_root / f"{name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return f"/media/real-world/{target.name}"


def bbox_text(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return "-"
    return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"


def prediction_payload(record: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    semantic = bool(record.get("semantic"))
    evidence = bool(record.get("evidence"))
    return {
        "model": model["display"],
        "answer": record.get("pred_option_id") or "-",
        "answer_option_id": record.get("pred_option_id") or "-",
        "bbox": bbox_text(record.get("pred_bbox")),
        "status": "correct" if semantic and evidence else "incorrect",
        "option_status": "correct" if semantic else "incorrect",
        "bbox_status": "correct" if evidence else "incorrect",
        "bbox_metric": f"IoU = {float(record.get('iou', 0.0)):.3f}",
    }


def compact_model_names(items: list[dict[str, Any]], limit: int = 3) -> str:
    names = [str(item.get("model", "") or "").strip() for item in items]
    names = [name for name in names if name]
    if not names:
        return "none"
    shown = names[:limit]
    if len(shown) == 1:
        text = shown[0]
    elif len(shown) == 2:
        text = f"{shown[0]} and {shown[1]}"
    else:
        text = ", ".join(shown[:-1]) + f", and {shown[-1]}"
    if len(names) > limit:
        text += f" (+{len(names) - limit} more)"
    return text


def prediction_summary(sample: dict[str, Any], predictions: list[dict[str, Any]]) -> str:
    total = len(predictions)
    semantic_ok = [item for item in predictions if item.get("option_status") == "correct"]
    evidence_ok = [item for item in predictions if item.get("bbox_status") == "correct"]
    both = [item for item in semantic_ok if item in evidence_ok]
    semantic_only = [item for item in semantic_ok if item not in evidence_ok]
    evidence_only = [item for item in evidence_ok if item not in semantic_ok]
    neither = total - len(both) - len(semantic_only) - len(evidence_only)
    task = str(sample.get("task_type", ""))
    task_title = TASK_TITLES.get(task, task)
    landmark = str(sample.get("landmark_description", "") or "the target landmark").strip()
    target = f"GT option {sample.get('answer_option_id', '-')} and BBox {bbox_text(sample.get('answer_bbox_xyxy_norm'))}"
    if task in {"self_where", "env_where"}:
        diagnosis = "The split outcomes show that reference-frame reasoning and localization of the same reconstructed landmark do not yet reliably coincide."
    else:
        diagnosis = "The split outcomes show that action-observation reasoning and localization of the same reconstructed landmark do not yet reliably coincide."
    return (
        f"For the {landmark} in this {task_title} question, the target is {target}. Across {total} available outputs, "
        f"option prediction succeeds for {len(semantic_ok)}/{total} and landmark localization at IoU >= 0.5 for "
        f"{len(evidence_ok)}/{total}; both succeed for {len(both)}/{total} ({compact_model_names(both)}). "
        f"Semantic-only cases are {compact_model_names(semantic_only)}, evidence-only cases are "
        f"{compact_model_names(evidence_only)}, and {neither}/{total} fail both. {diagnosis}"
    )


def sample_example(sample: dict[str, Any], analysis: dict[str, Any], workspace: Path, media_root: Path) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    task = str(sample["task_type"])
    scene = str(sample["scene_id"])
    records = analysis["diagnostic_records"].get(sample_id, {})
    models = {row["key"]: row for row in analysis["models"]}
    model_keys = [
        key
        for key in EXAMPLE_MODELS
        if key in models and key in records and records[key].get("pred_option_id")
    ]
    predictions = [prediction_payload(records[key], models[key]) for key in model_keys]
    ref = str(sample.get("reference_image_with_bbox") or sample.get("reference_image") or "")
    target = str(sample.get("target_image") or "")
    overlays = [{"label": "Ground truth", "color": "#16a34a", "bbox": sample.get("answer_bbox_xyxy_norm", [])}]
    for key, color in zip(model_keys, COLORS):
        record = records.get(key, {})
        if record.get("pred_bbox"):
            overlays.append({"label": models[key]["display"], "color": color, "line_style": "dashed", "bbox": record["pred_bbox"]})
    return {
        "example_id": sample_id,
        "scene_id": scene,
        "difficulty": str(sample.get("difficulty", "")),
        "prompt_excerpt": str(sample.get("user_prompt") or sample.get("prompt_text") or ""),
        "input_summary": str(sample.get("landmark_description", "")),
        "task_id": task,
        "task_modality": "image",
        "gold_answer": str(sample.get("answer_option_id", "")),
        "gold_bbox": bbox_text(sample.get("answer_bbox_xyxy_norm")),
        "prediction_analysis": prediction_summary(sample, predictions),
        "predictions": predictions,
        "media_panels": [
            {"kind": "image", "label": "Reference landmark view", "url": copy_image(workspace, media_root, ref, f"{scene}_{task}_{sample_id}_reference")},
            {
                "kind": "image",
                "label": "Query observation",
                "url": copy_image(workspace, media_root, target, f"{scene}_{task}_{sample_id}_query"),
                "overlays": overlays,
            },
        ],
        "answer_media_panels": [],
    }


def make_panorama(samples: list[dict[str, Any]], workspace: Path, output: Path) -> None:
    candidates: list[tuple[float, Path]] = []
    used = set()
    for sample in samples:
        value = str(sample.get("target_image") or "")
        if not value or value in used:
            continue
        used.add(value)
        path = resolve(workspace, value)
        candidates.append((quality(path), path))
    selected = [path for _, path in sorted(candidates, reverse=True)[:4]]
    tiles = []
    for path in selected:
        with Image.open(path) as image:
            tiles.append(ImageOps.fit(image.convert("RGB"), (600, 360), method=LANCZOS))
    canvas = Image.new("RGB", (600 * len(tiles), 360), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, (600 * index, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def render_semantic_pointcloud(
    workspace: Path,
    scene_id: str,
    media_root: Path,
    *,
    view_mode: str,
    target_points: int = 260_000,
) -> dict[str, str] | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    config_path = workspace / "configs" / "flightmvstg" / "3dgs" / f"{scene_id}_scene.json"
    review_path = workspace / "scene_data" / f"gsplat_{scene_id}" / "landmarks_review" / f"{scene_id}.valid_instances.json"
    if not config_path.is_file() or not review_path.is_file():
        return None
    config = load(config_path)
    asset_root = (config_path.parent / str(config["asset_root"])).resolve()
    geometry_path = asset_root / str(config["geometry_path"])
    if not geometry_path.is_file():
        return None

    points_mm = np.load(geometry_path, mmap_mode="r")
    stride = max(1, int(np.ceil(len(points_mm) / float(target_points))))
    points = np.asarray(points_mm[::stride], dtype=np.float32)
    colors = np.tile(np.asarray([[0.76, 0.79, 0.82]], dtype=np.float32), (len(points), 1))
    review = load(review_path)
    landmarks = list(review.get("valid_instances", []) or [])
    active_categories: set[str] = set()
    for landmark in landmarks:
        bbox = landmark.get("bbox_3d", {}) or {}
        lower = np.asarray(bbox.get("min", []), dtype=np.float32)
        upper = np.asarray(bbox.get("max", []), dtype=np.float32)
        if lower.shape != (3,) or upper.shape != (3,):
            continue
        category = str(landmark.get("landmark_category", "other") or "other")
        color_hex = SEMANTIC_COLORS.get(category, SEMANTIC_COLORS["other"])
        color = np.asarray([int(color_hex[index:index + 2], 16) / 255.0 for index in (1, 3, 5)], dtype=np.float32)
        mask = np.all((points >= lower) & (points <= upper), axis=1)
        if np.any(mask):
            colors[mask] = color
            active_categories.add(category)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if view_mode == "top":
        px, py = x, y
        label = "Landmark-Semantic Point Cloud - Top-Down"
    else:
        azimuth = np.deg2rad(35.0)
        elevation = np.deg2rad(28.0)
        x_rot = x * np.cos(azimuth) - y * np.sin(azimuth)
        y_rot = x * np.sin(azimuth) + y * np.cos(azimuth)
        px = x_rot
        py = z * np.cos(elevation) - y_rot * np.sin(elevation)
        label = "Landmark-Semantic Point Cloud - Oblique"

    px = px - np.mean(px)
    py = py - np.mean(py)
    x_range = max(float(np.ptp(px)), 1e-6)
    y_range = max(float(np.ptp(py)), 1e-6)
    fig_w = 7.6
    fig_h = max(4.8, fig_w * y_range / x_range * 0.92)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.scatter(px, py, s=0.22, c=colors, linewidths=0, alpha=0.95, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=SEMANTIC_COLORS.get(category, SEMANTIC_COLORS["other"]), markeredgecolor="none", label=category.replace("_", " ").title())
        for category in sorted(active_categories)
    ]
    if legend_items:
        ax.legend(handles=legend_items, loc="upper right", frameon=True, framealpha=0.9, fontsize=7, ncol=2)
    fig.subplots_adjust(0, 0, 1, 1)
    target = media_root / f"{scene_id}_semantic_{view_mode}.png"
    fig.savefig(target, dpi=180, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return {"label": label, "url": f"/media/real-world/{target.name}"}


def scene_gallery(
    scene_id: str,
    samples: list[dict[str, Any]],
    workspace: Path,
    media_root: Path,
    overview_root: Path | None,
    selected_landmarks: list[str],
) -> dict[str, Any]:
    landmarks = sorted({str(sample.get("landmark_id")) for sample in samples})
    best = sorted(
        (sample for sample in samples if str(sample.get("landmark_id")) in selected_landmarks),
        key=lambda sample: example_score(sample, workspace),
        reverse=True,
    )
    snapshots = []
    for landmark_id in selected_landmarks:
        candidate = next((sample for sample in best if str(sample.get("landmark_id")) == landmark_id), None)
        if candidate is None:
            continue
        value = str(candidate.get("target_image") or "")
        snapshots.append({
            "label": f"Selected landmark {landmark_id}",
            "url": copy_image(workspace, media_root, value, f"{scene_id}_{landmark_id}_scene"),
        })
    panorama = media_root / f"{scene_id}_panorama.jpg"
    make_panorama(best, workspace, panorama)
    overview_images = [{"label": "Registered RGB Panorama", "url": f"/media/real-world/{panorama.name}"}]
    for view_mode in ("top", "oblique"):
        semantic_view = render_semantic_pointcloud(workspace, scene_id, media_root, view_mode=view_mode)
        if semantic_view:
            overview_images.append(semantic_view)
    return {
        "scene_id": scene_id,
        "label": SCENE_NAMES[scene_id],
        "images": [*overview_images, *snapshots[:1]],
        "boundary": "Registered reconstruction coverage",
        "area_m2": 0,
        "raw_landmark_candidates": 100,
        "reviewed_valid_landmarks": len(landmarks),
    }


def select_figure_case(samples: dict[str, dict[str, Any]], analysis: dict[str, Any], workspace: Path) -> dict[str, Any]:
    models = {row["key"]: row for row in analysis["models"]}
    ranked = []
    for sample_id, sample in samples.items():
        if sample.get("task_type") not in {"self_where", "env_where"}:
            continue
        area = bbox_area(sample)
        if not 0.025 <= area <= 0.25:
            continue
        records = analysis["diagnostic_records"].get(sample_id, {})
        states = {(bool(records.get(key, {}).get("semantic")), bool(records.get(key, {}).get("evidence"))) for key in FIGURE_MODELS}
        if len(states) < 3:
            continue
        score = 30 * len(states) + example_score(sample, workspace)
        ranked.append((score, sample_id, sample, records))
    if not ranked:
        raise RuntimeError("No diverse figure case found")
    _, sample_id, sample, records = max(ranked)
    return {
        "sample_id": sample_id,
        "task": sample,
        "predictions": {key: {"model": models[key]["display"], **records[key]} for key in FIGURE_MODELS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--figure-media-root", type=Path, required=True)
    parser.add_argument("--scene-overview-root", type=Path)
    parser.add_argument("--landmark-selection-json", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    root = resolve(workspace, str(args.root))
    analysis = load(resolve(workspace, str(args.analysis)))
    selection_payload = load(resolve(workspace, str(args.landmark_selection_json)))
    selected_by_scene = selection_payload.get("UAV-DualCog-3DGS-RealScene-400", selection_payload)
    args.media_root.mkdir(parents=True, exist_ok=True)
    args.figure_media_root.mkdir(parents=True, exist_ok=True)
    samples: dict[str, dict[str, Any]] = {}
    scene_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in sorted(root.glob("*/qa/manifests/*.latest_manifest.json")):
        for sample in load(manifest)["samples"]:
            samples[str(sample["sample_id"])] = sample
            scene_samples[str(sample["scene_id"])].append(sample)

    groups = []
    for task, title in TASK_TITLES.items():
        examples = []
        for scene_id, rows in sorted(scene_samples.items()):
            selected = {str(item) for item in selected_by_scene.get(scene_id, [])}
            candidates = [row for row in rows if row.get("task_type") == task and str(row.get("landmark_id")) in selected]
            if not candidates:
                raise RuntimeError(f"No {task} example for selected landmarks in {scene_id}")
            examples.append(sample_example(max(candidates, key=lambda row: example_score(row, workspace)), analysis, workspace, args.media_root))
        groups.append({"task_id": task, "task_title": title, "examples": examples})

    overview_root = resolve(workspace, str(args.scene_overview_root)) if args.scene_overview_root else None
    galleries = [
        scene_gallery(scene_id, rows, workspace, args.media_root, overview_root, [str(item) for item in selected_by_scene.get(scene_id, [])])
        for scene_id, rows in sorted(scene_samples.items())
    ]
    scene_rows = []
    for scene_id, rows in sorted(scene_samples.items()):
        scene_rows.append({
            "scene_id": scene_id, "scene": SCENE_NAMES[scene_id],
            "statistics": {"registered_views": SCENE_REGISTERED[scene_id], "landmarks": len({row['landmark_id'] for row in rows}), "image_tasks": len(rows)},
            "task_counts": {task: sum(row.get("task_type") == task for row in rows) for task in TASK_TITLES},
        })
    web_models = []
    for row in analysis["models"]:
        if row["display"] in DISPLAY_EXCLUDED_MODELS:
            continue
        web_models.append({
            "model": row["display"], "group": "Proprietary" if row["group"] == "proprietary" else "Open-weights", "thinking": row["thinking"],
            "tasks": {task: {key: row["tasks"][task][key] for key in ("semantic_acc", "evidence_acc", "mean_iou", "valid_count", "count")} for task in TASK_TITLES},
            "position": {"semantic": row["position"]["dual_semantic_acc"], "evidence": row["position"]["joint_localization_acc"], "pairs": row["position"]["pair_count"]},
            "action_observation": {"semantic": row["action_observation"]["dual_semantic_acc"], "evidence": row["action_observation"]["joint_localization_acc"], "pairs": row["action_observation"]["pair_count"]},
            "overall": row["overall"], "marginal": row["marginal"],
        })
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "evaluation_complete",
        "statistics": {"registered_views": sum(SCENE_REGISTERED.values()), "sampled_views": 600, "landmark_candidates": 300, "reviewed_landmarks": 30, "image_tasks": 400, "models": len(web_models)},
        "scenes": scene_rows, "scene_gallery": galleries, "examples": {"groups": groups}, "results": {"models": web_models},
        "protocol": analysis["scope"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_ids = {str(item) for values in selected_by_scene.values() for item in values}
    selected_samples = {sample_id: sample for sample_id, sample in samples.items() if str(sample.get("landmark_id")) in selected_ids}
    figure = select_figure_case(selected_samples, analysis, workspace)
    task = figure["task"]
    figure["media"] = {
        "reference": copy_image(workspace, args.figure_media_root, str(task.get("reference_image_with_bbox") or task.get("reference_image")), "real_to_sim_reference"),
        "query": copy_image(workspace, args.figure_media_root, str(task.get("target_image")), "real_to_sim_query"),
    }
    args.figure_output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_output.write_text(json.dumps(figure, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"statistics": payload["statistics"], "figure_sample": figure["sample_id"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
