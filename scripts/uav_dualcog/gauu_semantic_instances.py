#!/usr/bin/env python3
"""Lift per-view landmark masks to 3D and fuse them into auditable candidates."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .landmark_taxonomy import LANDMARK_TAXONOMY, normalize_category
except ImportError:
    from landmark_taxonomy import LANDMARK_TAXONOMY, normalize_category


def _load_array(path: Path, preferred_key: str) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    if path.suffix.lower() == ".npz":
        with np.load(path) as payload:
            key = preferred_key if preferred_key in payload else payload.files[0]
            return np.asarray(payload[key])
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(f"reading {path.suffix} masks requires OpenCV") from exc
    array = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise ValueError(f"cannot read array image: {path}")
    return array


def lift_mask_to_world(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: dict[str, float],
    camera_to_world: np.ndarray,
    *,
    sample_step: int = 4,
    min_depth_m: float = 0.5,
    max_depth_m: float = 300.0,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    mask = np.asarray(mask).squeeze() > 0
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"depth and mask must have the same 2D shape, got {depth.shape} and {mask.shape}")
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    rows, cols = np.nonzero(mask)
    if sample_step > 1 and len(rows):
        rows, cols = rows[::sample_step], cols[::sample_step]
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if not len(z):
        return np.empty((0, 3), dtype=np.float32)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    camera_points = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z, np.ones_like(z)))
    return (camera_points @ c2w.T)[:, :3].astype(np.float32)


def _robust_bounds(points: np.ndarray, lower: float = 0.02, upper: float = 0.98) -> tuple[np.ndarray, np.ndarray]:
    return np.quantile(points, lower, axis=0), np.quantile(points, upper, axis=0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_view_instances(
    view_manifest: dict[str, Any],
    detections: list[dict[str, Any]],
    *,
    base_dir: Path,
    sample_step: int = 4,
    min_points: int = 50,
) -> list[dict[str, Any]]:
    views = {str(item["view_id"]): item for item in view_manifest.get("views", [])}
    output: list[dict[str, Any]] = []
    for row in detections:
        view_id = str(row.get("view_id", ""))
        view = views.get(view_id)
        if view is None:
            continue
        depth_path = base_dir / str(row.get("depth_path") or view.get("depth_path") or "")
        if not depth_path.is_file():
            raise FileNotFoundError(f"missing depth for view {view_id}: {depth_path}")
        depth = _load_array(depth_path, "depth")
        for index, detection in enumerate(row.get("detections", []) or []):
            if not isinstance(detection, dict):
                continue
            category = normalize_category(detection.get("landmark_category", detection.get("category")))
            if category not in LANDMARK_TAXONOMY:
                continue
            mask_path = base_dir / str(detection.get("mask_path", ""))
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing mask for view {view_id}: {mask_path}")
            mask = _load_array(mask_path, "mask")
            points = lift_mask_to_world(
                depth,
                mask,
                dict(view["intrinsics"]),
                np.asarray(view["camera_to_world"]),
                sample_step=sample_step,
            )
            if len(points) < int(min_points):
                continue
            bbox_min, bbox_max = _robust_bounds(points)
            center = np.median(points, axis=0)
            mask_pixels = int(np.count_nonzero(mask))
            bbox_2d = detection.get("bbox_2d_xyxy", detection.get("bbox_xyxy", detection.get("bbox")))
            if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
                bbox_2d = None
            output.append(
                {
                    "view_instance_id": str(detection.get("detection_id") or f"{view_id}_{index:03d}"),
                    "view_id": view_id,
                    "scene_grid_id": str(view.get("scene_grid_id", "grid_unknown")),
                    "landmark_category": category,
                    "landmark_subcategory": str(detection.get("landmark_subcategory") or detection.get("label") or "unknown"),
                    "landmark_description": str(detection.get("landmark_description") or ""),
                    "detection_confidence": float(detection.get("score", detection.get("confidence", 0.5)) or 0.5),
                    "is_static": bool(detection.get("is_static", category != "vehicle")),
                    "stability_confidence": float(detection.get("stability_confidence", 0.9 if category != "vehicle" else 0.5) or 0.0),
                    "center_3d": center.tolist(),
                    "bbox_3d": [bbox_min.tolist(), bbox_max.tolist()],
                    "point_count": int(len(points)),
                    "projection_area": float(mask_pixels / max(1, mask.size)),
                    "bbox_2d_xyxy": list(bbox_2d) if bbox_2d is not None else None,
                    "bbox_2d_image_size": [int(mask.shape[1]), int(mask.shape[0])],
                    "camera_center_world_m": list(view.get("center_world_m", [])),
                    "mask_path": str(mask_path),
                    "depth_path": str(depth_path),
                    "image_path": str(row.get("image_path") or view.get("image_path") or ""),
                }
            )
    return output


def fuse_view_instances(
    view_instances: list[dict[str, Any]],
    *,
    center_distance_m: float = 12.0,
    instance_prefix: str = "gauu_upper_candidate",
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    ordered = sorted(view_instances, key=lambda row: (-float(row.get("detection_confidence", 0.0)), str(row["view_instance_id"])))
    for row in ordered:
        center = np.asarray(row["center_3d"], dtype=np.float64)
        category = str(row["landmark_category"])
        subcategory = str(row.get("landmark_subcategory", "unknown")).strip().lower()
        match = None
        for cluster in clusters:
            if cluster["landmark_category"] != category:
                continue
            cluster_subcategory = str(cluster.get("landmark_subcategory", "unknown")).strip().lower()
            if subcategory != "unknown" and cluster_subcategory != "unknown" and subcategory != cluster_subcategory:
                continue
            if float(np.linalg.norm(center - np.asarray(cluster["center_3d"]))) <= float(center_distance_m):
                match = cluster
                break
        if match is None:
            match = {
                "landmark_category": category,
                "landmark_subcategory": str(row.get("landmark_subcategory", "unknown")),
                "landmark_description": str(row.get("landmark_description", "")),
                "members": [],
                "center_3d": center.tolist(),
            }
            clusters.append(match)
        match["members"].append(row)
        centers = np.asarray([member["center_3d"] for member in match["members"]], dtype=np.float64)
        match["center_3d"] = np.median(centers, axis=0).tolist()

    output: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters):
        members = list(cluster["members"])
        cluster_fields = {key: value for key, value in cluster.items() if key != "members"}
        bbox_min = np.min(np.asarray([member["bbox_3d"][0] for member in members]), axis=0)
        bbox_max = np.max(np.asarray([member["bbox_3d"][1] for member in members]), axis=0)
        views = sorted({str(member["view_id"]) for member in members})
        grids = sorted({str(member.get("scene_grid_id", "grid_unknown")) for member in members})
        confidences = [float(member["detection_confidence"]) for member in members]
        projection_areas = [float(member["projection_area"]) for member in members]
        output.append(
            {
                "instance_id": f"{instance_prefix}_{index:04d}",
                **cluster_fields,
                "bbox_3d": [bbox_min.tolist(), bbox_max.tolist()],
                "scene_grid_id": grids[0] if len(grids) == 1 else "cross_grid",
                "source_grid_ids": grids,
                "valid_view_count": len(views),
                "valid_viewpoints": views,
                "view_instances": members,
                "detection_confidence": float(np.mean(confidences)),
                "cross_view_consistency": min(1.0, len(views) / 8.0),
                "mean_projection_area": float(np.mean(projection_areas)),
                "mean_visible_ratio": 1.0,
                "geometry_compactness": 0.8,
                "stability_confidence": float(np.mean([float(member.get("stability_confidence", 0.5)) for member in members])),
                "is_static": all(bool(member.get("is_static", False)) for member in members),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Lift and fuse full-scene GauU semantic landmark masks")
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-step", type=int, default=4)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--fusion-distance-m", type=float, default=12.0)
    parser.add_argument("--scene-id", default="gauu_upper_campus")
    parser.add_argument("--instance-prefix", default="")
    args = parser.parse_args()

    view_manifest = json.loads(args.view_manifest.read_text(encoding="utf-8"))
    detections = _read_jsonl(args.detections) if args.detections.suffix.lower() == ".jsonl" else json.loads(args.detections.read_text(encoding="utf-8"))
    if isinstance(detections, dict):
        detections = list(detections.get("views", detections.get("detections", [])) or [])
    view_instances = build_view_instances(
        view_manifest,
        list(detections),
        base_dir=args.base_dir.expanduser().resolve(),
        sample_step=args.sample_step,
        min_points=args.min_points,
    )
    candidates = fuse_view_instances(
        view_instances,
        center_distance_m=args.fusion_distance_m,
        instance_prefix=args.instance_prefix.strip() or f"{args.scene_id}_candidate",
    )
    payload = {
        "scene_id": str(args.scene_id),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "source_view_count": len(view_manifest.get("views", [])),
        "view_instance_count": len(view_instances),
        "instance_count": len(candidates),
        "instances": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "instances"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
