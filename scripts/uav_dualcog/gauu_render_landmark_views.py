#!/usr/bin/env python3
"""Render audited eight-direction views for selected GauUscene landmarks."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sim_bridge.factory import create_bridge

try:
    from .stage2_landmark_label import (
        WorkerBinding,
        _build_bridge_config,
        _build_camera_pose_params,
        _build_candidate_side_view_specs,
        _capture_rgb_views_from_sim,
    )
except ImportError:
    from stage2_landmark_label import (
        WorkerBinding,
        _build_bridge_config,
        _build_camera_pose_params,
        _build_candidate_side_view_specs,
        _capture_rgb_views_from_sim,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _read_instances(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("selection payload must be a JSON object or array")
    for key in ("valid_instances", "instances", "candidates"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return payload, [dict(item) for item in rows if isinstance(item, dict)]
    raise ValueError("selection payload has no instances")


def _bbox_arrays(item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    bbox = item.get("bbox_3d")
    if isinstance(bbox, dict):
        lower, upper = bbox.get("min"), bbox.get("max")
    else:
        lower, upper = (bbox or [None, None])[:2]
    pmin = np.asarray(lower, dtype=np.float32)
    pmax = np.asarray(upper, dtype=np.float32)
    if pmin.shape != (3,) or pmax.shape != (3,) or not np.all(pmax > pmin):
        raise ValueError(f"invalid bbox_3d for {item.get('instance_id')}")
    return pmin, pmax


def _bbox_support_points(pmin: np.ndarray, pmax: np.ndarray) -> np.ndarray:
    corners = np.asarray(list(itertools.product(*zip(pmin.tolist(), pmax.tolist()))), dtype=np.float32)
    center = (pmin + pmax) * 0.5
    face_centers = np.asarray(
        [
            [pmin[0], center[1], center[2]], [pmax[0], center[1], center[2]],
            [center[0], pmin[1], center[2]], [center[0], pmax[1], center[2]],
            [center[0], center[1], pmin[2]], [center[0], center[1], pmax[2]],
        ],
        dtype=np.float32,
    )
    return np.vstack((corners, face_centers, center[None, :]))


def _canonical_yaw_deg(item: dict[str, Any], pmin: np.ndarray, pmax: np.ndarray) -> float:
    explicit = item.get("bbox_yaw_deg")
    if explicit is None and isinstance(item.get("bbox_3d"), dict):
        explicit = item["bbox_3d"].get("yaw_deg")
    if explicit is not None:
        return float(explicit) % 360.0
    size = pmax - pmin
    return 90.0 if float(size[1]) > float(size[0]) else 0.0


VIEW_OFFSETS = (
    ("front", 0.0),
    ("front_left", 45.0),
    ("left", 90.0),
    ("back_left", 135.0),
    ("back", 180.0),
    ("back_right", -135.0),
    ("right", -90.0),
    ("front_right", -45.0),
)


def _normalize_angle(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _source_direction(camera_center: np.ndarray, landmark_center: np.ndarray, yaw_deg: float) -> tuple[str, float]:
    camera_azimuth = math.degrees(
        math.atan2(float(camera_center[1] - landmark_center[1]), float(camera_center[0] - landmark_center[0]))
    )
    view_yaw = _normalize_angle(camera_azimuth - 180.0)
    offset = _normalize_angle(view_yaw - yaw_deg)
    label, _ = min(VIEW_OFFSETS, key=lambda item: abs(_normalize_angle(offset - item[1])))
    return label, view_yaw


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _prepare_source_views(
    item: dict[str, Any],
    *,
    view_manifest: dict[str, dict[str, Any]],
    raw_root: Path,
    max_views: int,
    yaw_deg: float,
) -> list[dict[str, Any]]:
    instance_id = str(item["instance_id"])
    center = np.asarray(item["center_3d"], dtype=np.float32)
    candidates: list[dict[str, Any]] = []
    for member in item.get("view_instances", []) or []:
        if not isinstance(member, dict):
            continue
        view_id = str(member.get("view_id", "") or "")
        manifest_view = view_manifest.get(view_id)
        source = Path(str(member.get("image_path", "") or "")).expanduser()
        bbox = member.get("bbox_2d_xyxy")
        image_size = member.get("bbox_2d_image_size")
        if manifest_view is None or not source.is_file() or not bbox or not image_size:
            continue
        c2w = np.asarray(manifest_view.get("camera_to_world"), dtype=np.float64)
        if c2w.shape != (4, 4):
            continue
        camera_center = c2w[:3, 3]
        direction, view_yaw = _source_direction(camera_center, center, yaw_deg)
        quality = float(member.get("detection_confidence", 0.0) or 0.0) * max(
            1e-6, float(member.get("projection_area", 0.0) or 0.0)
        )
        candidates.append(
            {
                "member": member,
                "source": source,
                "camera_center": camera_center,
                "view_direction": direction,
                "yaw_deg": view_yaw,
                "quality": quality,
            }
        )
    if len(candidates) < 4:
        raise RuntimeError(f"{instance_id} has fewer than four usable registered views")

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    used_directions: set[str] = set()
    while remaining and len(selected) < max(4, int(max_views)):
        best = max(
            remaining,
            key=lambda row: (int(row["view_direction"] not in used_directions), row["quality"]),
        )
        remaining.remove(best)
        selected.append(best)
        used_directions.add(str(best["view_direction"]))

    output: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        member = row["member"]
        source = row["source"]
        suffix = source.suffix.lower() or ".jpg"
        target = raw_root / "rgb_source_views" / instance_id / f"view_{index:02d}{suffix}"
        _link_or_copy(source, target)
        output.append(
            {
                "view_id": index,
                "source_view_id": str(member.get("view_id", "")),
                "path": target.relative_to(raw_root).as_posix(),
                "mode": "orbit",
                "label": str(row["view_direction"]),
                "view_direction": str(row["view_direction"]),
                "yaw_deg": float(row["yaw_deg"]),
                "pitch_deg": 0.0,
                "pitch_offset_deg": 0.0,
                "camera_center_world_m": [float(value) for value in row["camera_center"]],
                "bbox_2d_xyxy": list(member["bbox_2d_xyxy"]),
                "bbox_2d_image_size": list(member["bbox_2d_image_size"]),
                "bbox_2d_valid": True,
                "is_valid": True,
                "is_query_view": index == 0,
                "direction_reference": "bbox_major_axis_in_epsg_aligned_world_frame",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--valid-output", type=Path, required=True)
    parser.add_argument("--instances-output", type=Path)
    parser.add_argument("--render-manifest", type=Path)
    parser.add_argument("--view-source", choices=("registered", "orbit"), default="registered")
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = _read_yaml(args.config.expanduser().resolve())
    source_payload, instances = _read_instances(args.selection.expanduser().resolve())
    if args.limit > 0:
        instances = instances[: args.limit]
    if not instances:
        raise ValueError("no selected landmarks to render")

    task_cfg = config.get("task", {}) or {}
    scene_id = str(task_cfg.get("scene_id", source_payload.get("scene_id", "gauu_upper_campus")))
    engine = str(task_cfg.get("engine", "gsplat")).lower()
    stage2_cfg = config.get("stage2", {}) or {}
    camera_cfg = config.get("camera", {}) or {}
    width = int(stage2_cfg.get("collect_view_image_width", camera_cfg.get("width", 1280)))
    height = int(stage2_cfg.get("collect_view_image_height", camera_cfg.get("height", 720)))
    fov = float(camera_cfg.get("fov", 72.0))
    pose_params = _build_camera_pose_params(stage2_cfg)
    pose_params["camera_fov_deg"] = fov

    source_views: dict[str, dict[str, Any]] = {}
    bridge = None
    if args.view_source == "registered":
        if args.render_manifest is None:
            raise ValueError("--render-manifest is required for registered views")
        manifest_payload = json.loads(args.render_manifest.expanduser().resolve().read_text(encoding="utf-8"))
        source_views = {str(view["view_id"]): dict(view) for view in manifest_payload.get("views", [])}
    else:
        bridge_cfg = _build_bridge_config(config, vehicle_name="virtual_uav")
        bridge_cfg.update(dict((config.get("engine_params", {}) or {}).get(engine, {}) or {}))
        bridge_cfg.update({"image_width": width, "image_height": height, "fov": fov})
        bridge = create_bridge(engine=engine, scene_id=scene_id, config=bridge_cfg)

    raw_root = args.raw_root.expanduser().resolve()
    rendered: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(instances, 1):
            instance_id = str(item.get("instance_id", "") or "")
            if not instance_id:
                raise ValueError("selected landmark is missing instance_id")
            pmin, pmax = _bbox_arrays(item)
            yaw_deg = _canonical_yaw_deg(item, pmin, pmax)
            if args.view_source == "registered":
                views = _prepare_source_views(
                    item,
                    view_manifest=source_views,
                    raw_root=raw_root,
                    max_views=args.max_views,
                    yaw_deg=yaw_deg,
                )
            else:
                cardinals, diagonals = _build_candidate_side_view_specs(yaw_deg)
                view_specs = cardinals + diagonals
                target_points = _bbox_support_points(pmin, pmax)
                view_dir = raw_root / "rgb_orbit_views" / instance_id
                views = _capture_rgb_views_from_sim(
                    scene_id=scene_id,
                    config=config,
                    worker_binding=WorkerBinding(worker_id=0, vehicle="virtual_uav"),
                    target_points_xyz=target_points,
                    view_specs=view_specs,
                    output_width=width,
                    output_height=height,
                    out_dir=view_dir,
                    pose_settle_sec=0.0,
                    sim_camera_fov_deg=fov,
                    project_camera_fov_deg=fov,
                    square_crop=False,
                    draw_bbox_on_image=False,
                    pose_params=pose_params,
                    bridge=bridge,
                )
                if len(views) != 8 or not all(bool(view.get("bbox_2d_valid", False)) for view in views):
                    raise RuntimeError(f"{instance_id} produced invalid orbit views: {len(views)}")
                for view in views:
                    view["path"] = (view_dir / str(view["path"])).relative_to(raw_root).as_posix()
                    view["is_query_view"] = str(view.get("view_direction", "")) == "front"
                    view["direction_reference"] = "bbox_major_axis_in_epsg_aligned_world_frame"
            enriched = dict(item)
            bbox_size = pmax - pmin
            enriched.update(
                {
                    "bbox_3d": {
                        "min": pmin.tolist(),
                        "max": pmax.tolist(),
                        "size": bbox_size.tolist(),
                        "yaw_deg": yaw_deg,
                    },
                    "bbox_yaw_deg": yaw_deg,
                    "rgb_views": views,
                    "has_rgb": True,
                    "annotation_status": "labeled",
                    "review_action": "keep",
                    "review_status": "temporary_preselection_pending_user_confirmation",
                    "temporary_selection": True,
                }
            )
            rendered.append(enriched)
            print(f"[{index}/{len(instances)}] {instance_id}: rendered {len(views)} views", flush=True)
    finally:
        if bridge is not None:
            bridge.shutdown()

    audit = dict(source_payload.get("selection_audit", {}) or {})
    output_payload = {
        "scene_id": scene_id,
        "selection_protocol": source_payload.get(
            "selection_protocol", "temporary_preselection_pending_manual_review"
        ),
        "temporary_selection": True,
        "selection_audit": audit,
        "view_generation": {
            "engine": engine,
            "view_source": args.view_source,
            "maximum_view_count_per_landmark": int(args.max_views),
            "camera_resolution": [width, height],
            "camera_fov_deg": fov,
            "direction_reference": "bbox_major_axis_in_epsg_aligned_world_frame",
            "bbox_source": (
                "open_vocabulary_mask_bounds_on_registered_views"
                if args.view_source == "registered"
                else "projection_of_fused_3d_landmark_bounds"
            ),
        },
        "valid_instances": rendered,
    }
    args.valid_output.parent.mkdir(parents=True, exist_ok=True)
    args.valid_output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.instances_output:
        raw_payload = dict(output_payload)
        raw_payload["instances"] = raw_payload.pop("valid_instances")
        args.instances_output.parent.mkdir(parents=True, exist_ok=True)
        args.instances_output.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scene_id": scene_id, "landmarks": len(rendered), "views": sum(len(x["rgb_views"]) for x in rendered)}, indent=2))


if __name__ == "__main__":
    main()
