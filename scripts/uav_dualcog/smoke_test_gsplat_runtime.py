#!/usr/bin/env python3
"""GPU smoke test for the headless gsplat HTTP rendering path."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

try:
    from .gsplat_render_service import create_app
except ImportError:
    from gsplat_render_service import create_app


def _write_scene(root: Path) -> Path:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    points = np.zeros(9, dtype=dtype)
    index = 0
    for y in (-0.35, 0.0, 0.35):
        for x in (-0.35, 0.0, 0.35):
            points[index]["x"], points[index]["y"], points[index]["z"] = x, y, 3.0
            points[index]["opacity"] = 5.0
            points[index]["scale_0"] = points[index]["scale_1"] = points[index]["scale_2"] = -1.7
            points[index]["rot_0"] = 1.0
            points[index]["red"], points[index]["green"], points[index]["blue"] = 230, 80, 40
            index += 1
    splat_path = root / "synthetic_gaussians.ply"
    PlyData([PlyElement.describe(points, "vertex")], text=False).write(str(splat_path))
    geometry_path = root / "geometry.npy"
    np.save(geometry_path, np.column_stack([points[axis] for axis in ("x", "y", "z")]).astype(np.float32))
    manifest = {
        "scene_id": "synthetic_gsplat_smoke",
        "renderer_backend": "standard_3dgs_ply",
        "gs_checkpoint": splat_path.name,
        "geometry_path": geometry_path.name,
        "T_world_to_gs": np.eye(4).tolist(),
    }
    manifest_path = root / "scene.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test gsplat rendering on the selected CUDA device")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        manifest = _write_scene(Path(directory))
        app = create_app(str(manifest), device=args.device)
        client = app.test_client()
        health = client.get("/health")
        if health.status_code != 200 or not health.get_json().get("ok"):
            raise RuntimeError(f"health check failed: {health.status_code} {health.data!r}")
        request = {
            "camera": {"width": 128, "height": 96, "fx": 100.0, "fy": 100.0, "cx": 64.0, "cy": 48.0},
            "camera_to_world": np.eye(4).tolist(),
        }
        rgb_response = client.post("/render", json={**request, "mode": "rgb"})
        depth_response = client.post("/render", json={**request, "mode": "depth"})
        if rgb_response.status_code != 200 or depth_response.status_code != 200:
            raise RuntimeError(
                f"render failed: rgb={rgb_response.status_code} depth={depth_response.status_code}; "
                f"rgb_body={rgb_response.data[:300]!r} depth_body={depth_response.data[:300]!r}"
            )
        rgb = cv2.imdecode(np.frombuffer(rgb_response.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape != (96, 128, 3) or int(rgb.max()) == 0:
            raise RuntimeError(f"invalid RGB render: {None if rgb is None else (rgb.shape, int(rgb.max()))}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), rgb)
        print(json.dumps({"ok": True, "device": args.device, "rgb_shape": list(rgb.shape), "rgb_max": int(rgb.max())}))


if __name__ == "__main__":
    main()
