#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .landmark_taxonomy import (
        CAMPUS_CATEGORY_PRIORITY,
        LANDMARK_TAXONOMY,
        STABILITY_SENSITIVE_CATEGORIES,
        normalize_category,
    )
except ImportError:
    from landmark_taxonomy import (
        CAMPUS_CATEGORY_PRIORITY,
        LANDMARK_TAXONOMY,
        STABILITY_SENSITIVE_CATEGORIES,
        normalize_category,
    )


QUALITY_WEIGHTS = {
    "view_coverage": 0.18,
    "detection_confidence": 0.12,
    "cross_view_consistency": 0.18,
    "projection_quality": 0.12,
    "visibility_quality": 0.10,
    "geometry_quality": 0.15,
    "stability_confidence": 0.15,
}

REVIEW_VIEW_SLOTS = (
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def _first_number(row: dict[str, Any], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _center(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("center_3d", row.get("center_world_m"))
    try:
        center = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return center[:3] if center.shape == (3,) and np.all(np.isfinite(center)) else None


def _bbox(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("bbox_3d", row.get("bbox_3d_world"))
    if isinstance(value, dict):
        value = [value.get("min"), value.get("max")]
    try:
        bbox = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if bbox.shape != (2, 3) or not np.all(np.isfinite(bbox)) or np.any(bbox[1] <= bbox[0]):
        return None
    return bbox


def _view_count(row: dict[str, Any]) -> int:
    direct = row.get("valid_view_count", row.get("view_count"))
    if direct is not None:
        try:
            return max(0, int(direct))
        except (TypeError, ValueError):
            pass
    for key in ("valid_viewpoints", "reference_views", "views", "view_images"):
        value = row.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _subcategory(row: dict[str, Any]) -> str:
    return str(
        row.get("landmark_subcategory")
        or row.get("subcategory")
        or row.get("class_name")
        or "unknown"
    ).strip()


def _category(row: dict[str, Any]) -> str:
    return normalize_category(row.get("landmark_category", row.get("category")))


def _quality_components(row: dict[str, Any], *, view_target: int) -> dict[str, float]:
    view_coverage = _clamp01(_view_count(row) / max(1, int(view_target)))
    detection = _clamp01(_first_number(row, ("detection_confidence", "confidence", "score"), 0.5))
    consistency = _clamp01(_first_number(row, ("cross_view_consistency", "mask_consistency"), 0.5))
    mean_area = _first_number(row, ("mean_projection_area", "mean_bbox_area", "visible_ratio"), 0.0)
    projection = _clamp01(mean_area / 0.05) if mean_area <= 1.0 else _clamp01(mean_area / 50000.0)
    occlusion = _clamp01(_first_number(row, ("mean_occlusion_ratio", "occlusion_ratio"), 0.5))
    visibility = _clamp01(_first_number(row, ("mean_visible_ratio",), 1.0 - occlusion))
    geometry = _clamp01(_first_number(row, ("geometry_compactness", "geometry_quality"), 0.5))
    stable_default = 1.0 if bool(row.get("is_static", False)) else 0.5
    stability = _clamp01(_first_number(row, ("stability_confidence",), stable_default))
    return {
        "view_coverage": view_coverage,
        "detection_confidence": detection,
        "cross_view_consistency": consistency,
        "projection_quality": projection,
        "visibility_quality": visibility,
        "geometry_quality": geometry,
        "stability_confidence": stability,
    }


def score_candidate(row: dict[str, Any], *, view_target: int = 8) -> tuple[float, dict[str, float]]:
    components = _quality_components(row, view_target=view_target)
    score = sum(QUALITY_WEIGHTS[key] * components[key] for key in QUALITY_WEIGHTS)
    return float(score), components


def _grid_bounds(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    centers: list[np.ndarray] = []
    for row in rows:
        center = _center(row)
        if center is not None:
            centers.append(center)
    if not centers:
        raise ValueError("no valid 3D centers are available for scene-grid assignment")
    points = np.asarray(centers, dtype=float)
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    if math.isclose(float(x_min), float(x_max)):
        x_max = x_min + 1.0
    if math.isclose(float(y_min), float(y_max)):
        y_max = y_min + 1.0
    return float(x_min), float(x_max), float(y_min), float(y_max)


def assign_scene_grid(
    row: dict[str, Any],
    *,
    bounds: tuple[float, float, float, float],
    rows: int,
    cols: int,
) -> str:
    center = _center(row)
    if center is None:
        return "grid_unknown"
    x_min, x_max, y_min, y_max = bounds
    col = min(cols - 1, max(0, int((float(center[0]) - x_min) / (x_max - x_min) * cols)))
    row_idx = min(rows - 1, max(0, int((float(center[1]) - y_min) / (y_max - y_min) * rows)))
    return f"grid_{row_idx:02d}_{col:02d}"


def _candidate_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("instance_id") or row.get("landmark_id") or row.get("candidate_id") or f"candidate_{index:04d}")


def _eligibility_reasons(
    row: dict[str, Any],
    *,
    min_views: int,
    min_quality: float,
    min_stability: float,
    score: float,
) -> list[str]:
    reasons: list[str] = []
    category = _category(row)
    if category not in LANDMARK_TAXONOMY:
        reasons.append("unsupported_category")
    if _center(row) is None:
        reasons.append("invalid_center_3d")
    if _bbox(row) is None:
        reasons.append("invalid_bbox_3d")
    if _view_count(row) < int(min_views):
        reasons.append("insufficient_views")
    if score < float(min_quality):
        reasons.append("quality_below_threshold")
    stability = _quality_components(row, view_target=max(min_views, 8))["stability_confidence"]
    if category in STABILITY_SENSITIVE_CATEGORIES and stability < float(min_stability):
        reasons.append("unstable_navigation_landmark")
    return reasons


def _bbox_iou_3d(left: np.ndarray, right: np.ndarray) -> float:
    overlap_min = np.maximum(left[0], right[0])
    overlap_max = np.minimum(left[1], right[1])
    overlap = np.maximum(0.0, overlap_max - overlap_min)
    intersection = float(np.prod(overlap))
    if intersection <= 0.0:
        return 0.0
    left_volume = float(np.prod(left[1] - left[0]))
    right_volume = float(np.prod(right[1] - right[0]))
    return intersection / max(left_volume + right_volume - intersection, 1e-9)


def _deduplicate(
    rows: list[dict[str, Any]],
    *,
    distance_m: float,
    bbox_iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda item: (-float(item["quality_score"]), item["instance_id"]))
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in ordered:
        center = _center(row)
        bbox = _bbox(row)
        duplicate_of = ""
        for existing in kept:
            if _category(row) != _category(existing):
                continue
            existing_center = _center(existing)
            existing_bbox = _bbox(existing)
            near = center is not None and existing_center is not None and float(np.linalg.norm(center - existing_center)) <= distance_m
            overlap = bbox is not None and existing_bbox is not None and _bbox_iou_3d(bbox, existing_bbox) >= bbox_iou_threshold
            if near or overlap:
                duplicate_of = str(existing["instance_id"])
                break
        if duplicate_of:
            duplicate = dict(row)
            duplicate["candidate_status"] = "rejected"
            duplicate["rejection_reasons"] = [f"duplicate_of:{duplicate_of}"]
            rejected.append(duplicate)
        else:
            kept.append(row)
    return kept, rejected


def _category_rank(category: str) -> int:
    try:
        return CAMPUS_CATEGORY_PRIORITY.index(category)
    except ValueError:
        return len(CAMPUS_CATEGORY_PRIORITY)


def build_shortlist(
    source_rows: list[dict[str, Any]],
    *,
    candidate_count: int = 100,
    grid_rows: int = 6,
    grid_cols: int = 6,
    min_views: int = 4,
    min_quality: float = 0.35,
    min_stability: float = 0.65,
    dedup_distance_m: float = 3.0,
    bbox_iou_threshold: float = 0.45,
    scene_id: str = "gauu_upper_campus",
) -> dict[str, Any]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    bounds = _grid_bounds(source_rows)
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        row = dict(source)
        row["instance_id"] = _candidate_id(row, index)
        row["landmark_category"] = _category(row)
        row["landmark_subcategory"] = _subcategory(row)
        source_grid = str(row.get("scene_grid_id", "") or "")
        row["scene_grid_id"] = (
            source_grid
            if source_grid.startswith("grid_") and source_grid != "grid_unknown"
            else assign_scene_grid(row, bounds=bounds, rows=grid_rows, cols=grid_cols)
        )
        score, components = score_candidate(row)
        row["quality_score"] = score
        row["quality_scores"] = components
        reasons = _eligibility_reasons(
            row,
            min_views=min_views,
            min_quality=min_quality,
            min_stability=min_stability,
            score=score,
        )
        if reasons:
            row["candidate_status"] = "rejected"
            row["rejection_reasons"] = reasons
            rejected.append(row)
        else:
            row["candidate_status"] = "eligible"
            row["rejection_reasons"] = []
            eligible.append(row)

    deduplicated, duplicates = _deduplicate(
        eligible,
        distance_m=float(dedup_distance_m),
        bbox_iou_threshold=float(bbox_iou_threshold),
    )
    rejected.extend(duplicates)

    selected: list[dict[str, Any]] = []
    remaining = list(deduplicated)
    grid_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    while remaining and len(selected) < int(candidate_count):
        def adjusted(item: dict[str, Any]) -> tuple[float, int, str]:
            grid = str(item["scene_grid_id"])
            category = str(item["landmark_category"])
            coverage_bonus = 0.08 / (1.0 + grid_counts[grid])
            diversity_bonus = 0.06 / (1.0 + category_counts[category])
            return (
                float(item["quality_score"]) + coverage_bonus + diversity_bonus,
                -_category_rank(category),
                str(item["instance_id"]),
            )

        best = max(remaining, key=adjusted)
        remaining.remove(best)
        selected.append(best)
        grid_counts[str(best["scene_grid_id"])] += 1
        category_counts[str(best["landmark_category"])] += 1

    for rank, row in enumerate(selected, 1):
        row["candidate_rank"] = rank
        row["candidate_status"] = "shortlisted"
        row.setdefault("review_status", "pending")
    for row in remaining:
        row["candidate_status"] = "rejected"
        row["rejection_reasons"] = ["outside_candidate_budget"]
        rejected.append(row)

    return {
        "scene_id": str(scene_id),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "candidate_target": int(candidate_count),
        "source_count": len(source_rows),
        "eligible_after_dedup": len(deduplicated),
        "shortlisted_count": len(selected),
        "scene_grid": {
            "rows": int(grid_rows),
            "cols": int(grid_cols),
            "bounds_xy_m": list(bounds),
            "covered_grid_count": len(grid_counts),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "candidates": selected,
        "rejected": rejected,
    }


def finalize_reviewed_candidates(payload: dict[str, Any], *, final_count: int = 20) -> dict[str, Any]:
    candidates = [dict(item) for item in payload.get("candidates", []) if isinstance(item, dict)]
    accepted: list[dict[str, Any]] = []
    for row in candidates:
        action = str(row.get("review_action", "") or "").strip().lower()
        status = str(row.get("review_status", "") or "").strip().lower()
        if action not in {"keep", "approved"} and status not in {"keep", "approved"}:
            continue
        category = _category(row)
        subcategory = _subcategory(row)
        description = str(row.get("landmark_description") or row.get("description") or "").strip()
        if category not in LANDMARK_TAXONOMY or not subcategory or subcategory == "unknown" or not description:
            raise ValueError(f"approved candidate {row.get('instance_id')} lacks a complete category or description")
        center = _center(row)
        bbox = _bbox(row)
        if center is None or bbox is None:
            raise ValueError(f"approved candidate {row.get('instance_id')} lacks valid 3D geometry")
        bbox_size = bbox[1] - bbox[0]
        item = dict(row)
        is_temporary = bool(payload.get("temporary_preselection", False))
        item.update(
            {
                "center_3d": [float(value) for value in center],
                "bbox_3d": {
                    "min": [float(value) for value in bbox[0]],
                    "max": [float(value) for value in bbox[1]],
                    "size": [float(value) for value in bbox_size],
                    "yaw_deg": float(row.get("bbox_yaw_deg", 0.0) or 0.0),
                },
                "landmark_category": category,
                "landmark_subcategory": subcategory,
                "landmark_description": description,
                "class_name": subcategory,
                "review_action": "keep",
                "review_status": "approved",
                "annotation_status": "labeled",
                "landmark_decision": "automatic_temporary" if is_temporary else "manual",
            }
        )
        accepted.append(item)
    if len(accepted) != int(final_count):
        raise ValueError(f"expected exactly {final_count} approved landmarks, found {len(accepted)}")
    accepted.sort(key=lambda item: int(item.get("candidate_rank", 10**9)))
    category_counts = Counter(str(item["landmark_category"]) for item in accepted)
    subcategories = sorted({str(item["landmark_subcategory"]) for item in accepted})
    grids = sorted({str(item.get("scene_grid_id", "grid_unknown")) for item in accepted})
    warnings: list[str] = []
    if len(category_counts) < 4:
        warnings.append("fewer_than_four_coarse_categories")
    if len(subcategories) < 10:
        warnings.append("fewer_than_ten_subcategories")
    return {
        "scene_id": str(payload.get("scene_id") or "gauu_upper_campus"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "selection_protocol": (
            "automatic_temporary_preselection_pending_manual_review"
            if bool(payload.get("temporary_preselection", False))
            else "manual_review_from_full_scene_candidate_shortlist"
        ),
        "temporary_selection": bool(payload.get("temporary_preselection", False)),
        "candidate_source_count": len(candidates),
        "selection_audit": {
            "category_counts": dict(sorted(category_counts.items())),
            "subcategory_count": len(subcategories),
            "covered_grid_count": len(grids),
            "warnings": warnings,
        },
        "valid_instances": accepted,
    }


def preselect_temporary_candidates(
    payload: dict[str, Any],
    *,
    final_count: int = 20,
    selections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Choose a deterministic, diverse pilot set while preserving manual-review provenance."""
    candidates = [dict(item) for item in payload.get("candidates", []) if isinstance(item, dict)]
    if len(candidates) < int(final_count):
        raise ValueError(f"temporary preselection needs {final_count} candidates, found {len(candidates)}")
    requested = [dict(item) for item in list(selections or []) if isinstance(item, dict)]
    if requested:
        candidate_by_id = {str(item.get("instance_id", "")): item for item in candidates}
        selected = []
        for selection in requested:
            instance_id = str(selection.get("instance_id", "") or "")
            if instance_id not in candidate_by_id:
                raise ValueError(f"selected candidate is absent from shortlist: {instance_id}")
            item = dict(candidate_by_id[instance_id])
            item.update({key: value for key, value in selection.items() if key != "instance_id"})
            selected.append(item)
        if len(selected) != int(final_count):
            raise ValueError(f"selection file must contain exactly {final_count} entries, found {len(selected)}")
        candidates_for_audit = candidates
    else:
        high_coverage = [
            item
            for item in candidates
            if _view_count(item) >= 4 and int(item.get("visual_valid_view_count", 4) or 0) >= 4
        ]
        if len(high_coverage) >= int(final_count):
            candidates = high_coverage
        candidates_for_audit = candidates
        selected = []

    remaining = list(candidates)
    category_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    grid_counts: Counter[str] = Counter()
    while not requested and remaining and len(selected) < int(final_count):
        def adjusted(item: dict[str, Any]) -> tuple[int, int, int, float, str]:
            category = _category(item)
            subcategory = _subcategory(item)
            grid = str(item.get("scene_grid_id", "grid_unknown"))
            score = float(item.get("quality_score", 0.0) or 0.0)
            return (
                int(grid_counts[grid] == 0),
                int(subcategory_counts[subcategory] == 0),
                int(category_counts[category] == 0),
                score,
                str(item.get("instance_id", "")),
            )

        best = max(remaining, key=adjusted)
        remaining.remove(best)
        category = _category(best)
        subcategory = _subcategory(best)
        grid = str(best.get("scene_grid_id", "grid_unknown"))
        selected.append(best)
        category_counts[category] += 1
        subcategory_counts[subcategory] += 1
        grid_counts[grid] += 1

    for best in selected:
        category = _category(best)
        subcategory = _subcategory(best)
        best["review_action"] = "keep"
        best["review_status"] = "approved"
        best["temporary_preselection"] = True
        best["landmark_description"] = str(
            best.get("landmark_description")
            or best.get("description")
            or f"a {subcategory} visible from multiple aerial viewpoints"
        )

    temporary_payload = dict(payload)
    temporary_payload["candidates"] = selected
    temporary_payload["temporary_preselection"] = True
    result = finalize_reviewed_candidates(temporary_payload, final_count=final_count)
    result["candidate_source_count"] = len(payload.get("candidates", []))
    result["selection_audit"]["source_candidate_count"] = len(candidates_for_audit)
    result["selection_audit"]["selection_seed"] = (
        "visual_pre_review_selection_file" if requested else "deterministic_quality_category_subcategory_grid"
    )
    return result


def prepare_stage2_review_bundle(
    payload: dict[str, Any],
    *,
    raw_root: Path,
    scene_id: str = "gauu_upper_campus",
    max_views: int = 8,
    min_views: int = 3,
) -> Path:
    """Adapt shortlisted real-scene candidates to the existing Stage 2 review UI."""
    candidates = [dict(item) for item in payload.get("candidates", []) if isinstance(item, dict)]
    if not candidates:
        raise ValueError("candidate shortlist is empty")
    raw_root = raw_root.expanduser().resolve()
    instances: list[dict[str, Any]] = []
    for candidate in candidates:
        instance_id = str(candidate.get("instance_id", "") or "")
        if not instance_id:
            raise ValueError("candidate is missing instance_id")
        bbox = _bbox(candidate)
        if bbox is None:
            raise ValueError(f"candidate {instance_id} has invalid 3D geometry")
        members = [dict(item) for item in candidate.get("view_instances", []) if isinstance(item, dict)]
        visual_audit = {
            str(item.get("view_id", "")): bool(item.get("passed", False))
            for item in candidate.get("visual_quality_audit", [])
            if isinstance(item, dict)
        }
        visually_valid_members = [
            member for member in members if visual_audit.get(str(member.get("view_id", "")), False)
        ]
        if len(visually_valid_members) >= int(min_views):
            members = visually_valid_members
        members.sort(key=lambda item: (-float(item.get("projection_area", 0.0) or 0.0), str(item.get("view_id", ""))))
        rgb_views: list[dict[str, Any]] = []
        for index, member in enumerate(members[: max(1, int(max_views))]):
            source = Path(str(member.get("image_path", "") or "")).expanduser()
            if not source.is_file():
                continue
            suffix = source.suffix.lower() or ".jpg"
            target = raw_root / "rgb_views" / instance_id / f"view_{index:02d}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
            bbox_2d = member.get("bbox_2d_xyxy")
            image_size = member.get("bbox_2d_image_size")
            rgb_views.append(
                {
                    "view_id": index,
                    "source_view_id": str(member.get("view_id", "")),
                    "path": target.relative_to(raw_root).as_posix(),
                    "view_direction": REVIEW_VIEW_SLOTS[index % len(REVIEW_VIEW_SLOTS)],
                    "direction_reference": "review_slot_only_not_landmark_heading",
                    "is_valid": True,
                    "bbox_2d_xyxy": bbox_2d,
                    "bbox_2d_image_size": image_size,
                    "bbox_2d_valid": bool(bbox_2d and image_size),
                }
            )
        if len(rgb_views) < int(min_views):
            raise ValueError(f"candidate {instance_id} has fewer than {min_views} available review images")
        bbox_size = bbox[1] - bbox[0]
        item = dict(candidate)
        item.update(
            {
                "class_id": int(candidate.get("candidate_rank", len(instances) + 1)),
                "class_name": _subcategory(candidate),
                "center_3d": [float(value) for value in _center(candidate)],
                "bbox_3d": {
                    "min": [float(value) for value in bbox[0]],
                    "max": [float(value) for value in bbox[1]],
                    "size": [float(value) for value in bbox_size],
                    "yaw_deg": float(candidate.get("bbox_yaw_deg", 0.0) or 0.0),
                },
                "point_count": int(sum(int(member.get("point_count", 0) or 0) for member in members)),
                "rgb_views": rgb_views,
                "pcd_views": [],
                "has_rgb": True,
                "annotation_status": "labeled",
                "landmark_category": _category(candidate),
                "landmark_subcategory": _subcategory(candidate),
                "landmark_description": str(
                    candidate.get("landmark_description")
                    or candidate.get("description")
                    or f"a {_subcategory(candidate)} visible from multiple aerial viewpoints"
                ),
            }
        )
        instances.append(item)
    output = raw_root / f"{scene_id}.instances.json"
    _write_json(
        output,
        {
            "scene_id": scene_id,
            "coordinate": "EPSG-aligned ENU",
            "unit": "meter",
            "source": "GauUscene V2.1 and CityGaussianV2",
            "summary": {"instances": len(instances), "review_adapter": "gauu_candidate_shortlist"},
            "instances": instances,
        },
    )
    return output


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [dict(row) for row in rows if isinstance(row, dict)], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)], {}
    if not isinstance(payload, dict):
        raise ValueError("candidate input must be a JSON object, JSON array, or JSONL file")
    for key in ("candidates", "instances", "valid_instances"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(row) for row in value if isinstance(row, dict)], payload
    raise ValueError("candidate input does not contain candidates, instances, or valid_instances")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and finalize full-scene GauU landmark candidates")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shortlist = subparsers.add_parser("shortlist", help="rank raw 3D instances into a spatially diverse candidate pool")
    shortlist.add_argument("--input", type=Path, required=True)
    shortlist.add_argument("--output", type=Path, required=True)
    shortlist.add_argument("--candidate-count", type=int, default=100)
    shortlist.add_argument("--grid-rows", type=int, default=6)
    shortlist.add_argument("--grid-cols", type=int, default=6)
    shortlist.add_argument("--min-views", type=int, default=4)
    shortlist.add_argument("--min-quality", type=float, default=0.35)
    shortlist.add_argument("--min-stability", type=float, default=0.65)
    shortlist.add_argument("--dedup-distance-m", type=float, default=3.0)
    shortlist.add_argument("--bbox-iou-threshold", type=float, default=0.45)
    shortlist.add_argument("--scene-id", default="")

    finalize = subparsers.add_parser("finalize", help="export exactly 20 manually approved landmarks for Stage 3/4")
    finalize.add_argument("--input", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--final-count", type=int, default=20)

    preselect = subparsers.add_parser("preselect", help="export a diverse temporary set pending Stage 2 manual review")
    preselect.add_argument("--input", type=Path, required=True)
    preselect.add_argument("--output", type=Path, required=True)
    preselect.add_argument("--final-count", type=int, default=20)
    preselect.add_argument("--selection-file", type=Path)

    review = subparsers.add_parser("prepare-review", help="adapt shortlisted candidates to the existing Stage 2 web UI")
    review.add_argument("--input", type=Path, required=True)
    review.add_argument("--raw-root", type=Path, required=True)
    review.add_argument("--scene-id", default="gauu_upper_campus")
    review.add_argument("--max-views", type=int, default=8)
    review.add_argument("--min-views", type=int, default=3)

    args = parser.parse_args()
    rows, source_payload = _read_rows(args.input)
    if args.command == "shortlist":
        output = build_shortlist(
            rows,
            candidate_count=args.candidate_count,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            min_views=args.min_views,
            min_quality=args.min_quality,
            min_stability=args.min_stability,
            dedup_distance_m=args.dedup_distance_m,
            bbox_iou_threshold=args.bbox_iou_threshold,
            scene_id=args.scene_id or str(source_payload.get("scene_id") or "gauu_upper_campus"),
        )
    elif args.command == "finalize":
        payload = dict(source_payload)
        payload["candidates"] = rows
        output = finalize_reviewed_candidates(payload, final_count=args.final_count)
    elif args.command == "preselect":
        payload = dict(source_payload)
        payload["candidates"] = rows
        selections = None
        if args.selection_file:
            selection_payload = json.loads(args.selection_file.read_text(encoding="utf-8"))
            selections = list(selection_payload.get("selections", []) or []) if isinstance(selection_payload, dict) else list(selection_payload or [])
        output = preselect_temporary_candidates(payload, final_count=args.final_count, selections=selections)
    else:
        payload = dict(source_payload)
        payload["candidates"] = rows
        output_path = prepare_stage2_review_bundle(
            payload,
            raw_root=args.raw_root,
            scene_id=args.scene_id,
            max_views=args.max_views,
            min_views=args.min_views,
        )
        print(json.dumps({"scene_id": args.scene_id, "instances": len(rows), "output": str(output_path)}, ensure_ascii=False, indent=2))
        return
    _write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if not isinstance(value, list)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
