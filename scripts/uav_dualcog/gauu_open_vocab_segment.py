#!/usr/bin/env python3
"""Detect and segment Stage 2 landmark candidates in rendered GauUscene views."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

try:
    from .landmark_taxonomy import LANDMARK_TAXONOMY
except ImportError:
    from landmark_taxonomy import LANDMARK_TAXONOMY


# These prompts retain the project taxonomy while prioritizing objects that are
# stable and observable from aerial campus views.
PROMPT_LABELS: dict[str, dict[str, str]] = {
    "building": {
        "building": "low-rise building", "low-rise building": "low-rise building",
        "mid-rise building": "mid-rise building", "high-rise building": "high-rise building",
        "warehouse": "warehouse", "factory shed": "factory shed",
        "glass building": "glass skyscraper", "school building": "brick schoolhouse",
    },
    "vehicle": {
        "car": "sedan car", "van": "delivery van", "bus": "city bus",
        "truck": "cargo truck", "excavator": "construction excavator",
    },
    "public_facility": {
        "street lamp": "street lamp post", "bus shelter": "bus shelter",
        "bench": "public bench", "antenna": "antenna",
    },
    "urban_landscape": {
        "plaza": "plaza", "fountain": "fountain", "city square": "city square",
        "urban garden": "urban garden", "sculpture": "sculpture",
        "signboard": "signboard", "billboard": "billboard", "landmark sign": "landmark signage",
    },
    "transport_infrastructure": {
        "bridge": "arch bridge", "overpass": "overpass", "roundabout": "roundabout",
        "crosswalk": "pedestrian crosswalk", "traffic island": "traffic island",
        "tunnel entrance": "tunnel entrance", "railway track": "railway track",
    },
    "industrial_infrastructure": {
        "shipping container": "shipping container stack", "storage tank": "oil storage tank",
        "silo": "grain silo", "tower crane": "crane tower", "pipeline": "pipeline",
    },
    "vegetation": {
        "large tree": "deciduous tree", "palm tree": "palm tree",
        "shrub cluster": "shrub cluster", "hedge": "hedge row", "grassy lawn": "grassy lawn",
    },
    "other": {
        "tent": "temporary tent", "rubble pile": "rubble pile",
        "playground slide": "playground slide", "construction barrier": "construction barrier",
    },
}


def _canonical_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


LABEL_DETAILS = {
    _canonical_text(label): (category, subcategory)
    for category, labels in PROMPT_LABELS.items()
    for label, subcategory in labels.items()
}


def _category_for_label(label: str) -> tuple[str, str] | None:
    text = _canonical_text(label)
    matches = [candidate for candidate in LABEL_DETAILS if candidate in text or text in candidate]
    if not matches:
        return None
    return LABEL_DETAILS[max(matches, key=len)]


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    overlap_min = np.maximum(left[:2], right[:2])
    overlap_max = np.minimum(left[2:], right[2:])
    wh = np.maximum(0.0, overlap_max - overlap_min)
    intersection = float(wh[0] * wh[1])
    left_area = float(np.prod(np.maximum(0.0, left[2:] - left[:2])))
    right_area = float(np.prod(np.maximum(0.0, right[2:] - right[:2])))
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _nms(rows: list[dict[str, Any]], threshold: float, max_count: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: -float(item["score"])):
        box = np.asarray(row["bbox_2d_xyxy"], dtype=np.float32)
        if any(_box_iou(box, np.asarray(item["bbox_2d_xyxy"], dtype=np.float32)) >= threshold for item in kept):
            continue
        kept.append(row)
        if len(kept) >= max_count:
            break
    return kept


def _resolve_image(view: dict[str, Any], manifest_path: Path) -> Path:
    value = str(view.get("image_path") or view.get("rgb_path") or "")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"rendered RGB view does not exist: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--masks-root", required=True, type=Path)
    parser.add_argument("--detector", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--segmenter", default="facebook/sam-vit-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.26)
    parser.add_argument("--text-threshold", type=float, default=0.22)
    parser.add_argument("--nms-threshold", type=float, default=0.72)
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.40)
    parser.add_argument("--max-detections", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = args.render_manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = list(payload.get("views", []))
    if args.limit > 0:
        views = views[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.masks_root.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line).get("view_id", "")))

    detector_processor = AutoProcessor.from_pretrained(args.detector)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).to(args.device).eval()
    sam_processor = SamProcessor.from_pretrained(args.segmenter)
    sam = SamModel.from_pretrained(args.segmenter).to(args.device).eval()
    prompt = ". ".join(label for labels in PROMPT_LABELS.values() for label in labels) + "."

    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8") as handle:
        for view_index, view in enumerate(views, 1):
            view_id = str(view["view_id"])
            if view_id in completed:
                continue
            image_path = _resolve_image(view, manifest_path)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            inputs = detector_processor(images=image, text=prompt, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                outputs = detector(**inputs)
            result = detector_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[(height, width)],
            )[0]
            labels = result.get("text_labels", result.get("labels", []))
            detections: list[dict[str, Any]] = []
            for score, box, label in zip(result["scores"].cpu(), result["boxes"].cpu(), labels):
                mapping = _category_for_label(str(label))
                if mapping is None:
                    continue
                xyxy = box.numpy().astype(float)
                area_ratio = float(max(0.0, xyxy[2] - xyxy[0]) * max(0.0, xyxy[3] - xyxy[1]) / (width * height))
                if not args.min_area_ratio <= area_ratio <= args.max_area_ratio:
                    continue
                category, subcategory = mapping
                detections.append(
                    {
                        "score": float(score),
                        "bbox_2d_xyxy": xyxy.tolist(),
                        "landmark_category": category,
                        "landmark_subcategory": subcategory,
                        "detector_label": str(label),
                    }
                )
            detections = _nms(detections, args.nms_threshold, args.max_detections)

            if detections:
                boxes = [item["bbox_2d_xyxy"] for item in detections]
                sam_inputs = sam_processor(images=image, input_boxes=[boxes], return_tensors="pt").to(args.device)
                original_sizes = sam_inputs["original_sizes"].cpu()
                reshaped_sizes = sam_inputs["reshaped_input_sizes"].cpu()
                with torch.inference_mode():
                    sam_outputs = sam(**{key: value for key, value in sam_inputs.items() if key not in {"original_sizes", "reshaped_input_sizes"}})
                masks = sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(), original_sizes, reshaped_sizes
                )[0]
                quality = sam_outputs.iou_scores.cpu()[0]
                for detection_index, detection in enumerate(detections):
                    best = int(torch.argmax(quality[detection_index]).item())
                    mask = masks[detection_index, best].numpy().astype(bool)
                    mask_path = args.masks_root.resolve() / view_id / f"mask_{detection_index:03d}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
                    detection.update(
                        {
                            "detection_id": f"{view_id}_{detection_index:03d}",
                            "mask_path": str(mask_path),
                            "mask_score": float(quality[detection_index, best]),
                            "is_static": detection["landmark_category"] != "vehicle",
                            "stability_confidence": 0.9 if detection["landmark_category"] != "vehicle" else 0.5,
                        }
                    )
            row = {
                "view_id": view_id,
                "scene_grid_id": view.get("scene_grid_id", "grid_unknown"),
                "image_path": str(image_path),
                "depth_path": str(view.get("depth_path", "")),
                "detections": detections,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{view_index}/{len(views)}] view={view_id} detections={len(detections)}", flush=True)


if __name__ == "__main__":
    main()
