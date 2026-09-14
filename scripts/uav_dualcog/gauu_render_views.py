#!/usr/bin/env python3
"""Render sampled GauUscene views while preserving COLMAP camera geometry."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def build_render_job(view: dict[str, Any], output_root: Path, *, max_width: int = 1600) -> dict[str, Any]:
    view_id = str(view["view_id"])
    intrinsics = dict(view["intrinsics"])
    source_width = int(view["width"])
    source_height = int(view["height"])
    scale = min(1.0, float(max_width) / source_width) if max_width > 0 else 1.0
    camera = {
        "width": int(round(source_width * scale)),
        "height": int(round(source_height * scale)),
        "fx": float(intrinsics["fx"]) * scale,
        "fy": float(intrinsics["fy"]) * scale,
        "cx": float(intrinsics["cx"]) * scale,
        "cy": float(intrinsics["cy"]) * scale,
    }
    view_root = output_root / view_id
    return {
        "view_id": view_id,
        "scene_grid_id": str(view.get("scene_grid_id", "grid_unknown")),
        "image_name": str(view.get("image_name", "")),
        "camera": camera,
        "intrinsics": {
            "fx": camera["fx"],
            "fy": camera["fy"],
            "cx": camera["cx"],
            "cy": camera["cy"],
        },
        "camera_to_world": view["camera_to_world"],
        "rgb_path": str(view_root / "rgb.jpg"),
        "image_path": str(view_root / "rgb.jpg"),
        "depth_path": str(view_root / "depth.npz"),
    }


def _request_render(render_url: str, job: dict[str, Any], mode: str, timeout: float) -> bytes:
    response = requests.post(
        render_url,
        json={
            "camera": job["camera"],
            "camera_to_world": job["camera_to_world"],
            "mode": mode,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.content


def render_job(render_url: str, job: dict[str, Any], *, timeout: float = 240.0) -> dict[str, Any]:
    rgb_path = Path(job["rgb_path"])
    depth_path = Path(job["depth_path"])
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_path.write_bytes(_request_render(render_url, job, "rgb", timeout))
    depth_path.write_bytes(_request_render(render_url, job, "depth", timeout))
    return {**job, "render_status": "complete"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render full-scene GauU sampled views")
    parser.add_argument("--view-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--render-url", default="http://127.0.0.1:18080/render")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=1600, help="Scale COLMAP resolution and intrinsics proportionally")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.view_manifest.read_text(encoding="utf-8"))
    views = list(source.get("views", []))
    if args.limit > 0:
        views = views[: args.limit]
    output_root = args.output_root.expanduser().resolve()
    jobs = [build_render_job(view, output_root, max_width=args.max_width) for view in views]
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if args.dry_run:
        results = [{**job, "render_status": "planned"} for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {pool.submit(render_job, args.render_url, job, timeout=args.timeout): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({"view_id": job["view_id"], "error": str(exc)})
        results.sort(key=lambda row: row["view_id"])

    payload = {
        "scene_id": str(source.get("scene_id", "gauu_upper_campus")),
        "source_view_manifest": str(args.view_manifest),
        "render_url": args.render_url,
        "dry_run": bool(args.dry_run),
        "requested_view_count": len(jobs),
        "completed_view_count": len(results) if not args.dry_run else 0,
        "failed_view_count": len(errors),
        "views": results,
        "errors": errors,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"views", "errors"}}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
