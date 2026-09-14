#!/usr/bin/env python3
"""Build a spatially balanced full-scene view manifest from GauU COLMAP poses."""
from __future__ import annotations

import argparse
import json
import math
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read(handle: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize("<" + fmt)
    data = handle.read(size)
    if len(data) != size:
        raise EOFError("unexpected end of COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def _qvec_to_rotation(qvec: list[float] | tuple[float, ...]) -> np.ndarray:
    qw, qx, qy, qz = [float(value) for value in qvec]
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _camera_intrinsics(model_name: str, params: list[float]) -> dict[str, float]:
    if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    else:
        fx, fy, cx, cy = [float(value) for value in params[:4]]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _record(image_id: int, camera: dict[str, Any], qvec: list[float], tvec: list[float], name: str) -> dict[str, Any]:
    rotation_world_to_camera = _qvec_to_rotation(qvec)
    translation = np.asarray(tvec, dtype=np.float64)
    rotation_camera_to_world = rotation_world_to_camera.T
    center = -rotation_camera_to_world @ translation
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation_camera_to_world
    c2w[:3, 3] = center
    intrinsics = _camera_intrinsics(str(camera["model"]), list(camera["params"]))
    return {
        "view_id": int(image_id),
        "image_name": str(name),
        "camera_id": int(camera["camera_id"]),
        "camera_model": str(camera["model"]),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "intrinsics": intrinsics,
        "distortion_params": [float(value) for value in list(camera["params"])[4:]],
        "center_world_m": [float(value) for value in center],
        "camera_to_world": [[float(value) for value in row] for row in c2w],
    }


def _read_cameras_binary(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        (count,) = _read(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _read(handle, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"unsupported COLMAP camera model id: {model_id}")
            model_name, param_count = CAMERA_MODELS[model_id]
            params = list(_read(handle, "d" * param_count))
            cameras[int(camera_id)] = {
                "camera_id": int(camera_id),
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_images_binary(path: Path, cameras: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        (count,) = _read(handle, "Q")
        for _ in range(count):
            values = _read(handle, "idddddddi")
            image_id = int(values[0])
            qvec = [float(value) for value in values[1:5]]
            tvec = [float(value) for value in values[5:8]]
            camera_id = int(values[8])
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if not char:
                    raise EOFError("unterminated image name in images.bin")
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")
            (point_count,) = _read(handle, "Q")
            handle.seek(int(point_count) * struct.calcsize("<ddq"), 1)
            records.append(_record(image_id, cameras[camera_id], qvec, tvec, name))
    return records


def _read_cameras_text(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        camera_id = int(fields[0])
        cameras[camera_id] = {
            "camera_id": camera_id,
            "model": fields[1],
            "width": int(fields[2]),
            "height": int(fields[3]),
            "params": [float(value) for value in fields[4:]],
        }
    return cameras


def _read_images_text(path: Path, cameras: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    records: list[dict[str, Any]] = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        if len(fields) < 10:
            continue
        image_id = int(fields[0])
        qvec = [float(value) for value in fields[1:5]]
        tvec = [float(value) for value in fields[5:8]]
        camera_id = int(fields[8])
        records.append(_record(image_id, cameras[camera_id], qvec, tvec, " ".join(fields[9:])))
    return records


def find_sparse_model(root: Path) -> Path:
    root = root.expanduser().resolve()
    candidates = [root, root / "sparse", root / "sparse" / "0", root / "0"]
    for candidate in candidates:
        if (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file():
            return candidate
        if (candidate / "cameras.txt").is_file() and (candidate / "images.txt").is_file():
            return candidate
    for cameras_path in root.rglob("cameras.bin"):
        if (cameras_path.parent / "images.bin").is_file():
            return cameras_path.parent
    for cameras_path in root.rglob("cameras.txt"):
        if (cameras_path.parent / "images.txt").is_file():
            return cameras_path.parent
    raise FileNotFoundError(f"no COLMAP sparse model found under {root}")


def load_colmap_views(colmap_root: Path) -> tuple[Path, list[dict[str, Any]]]:
    sparse = find_sparse_model(colmap_root)
    if (sparse / "cameras.bin").is_file():
        cameras = _read_cameras_binary(sparse / "cameras.bin")
        records = _read_images_binary(sparse / "images.bin", cameras)
    else:
        cameras = _read_cameras_text(sparse / "cameras.txt")
        records = _read_images_text(sparse / "images.txt", cameras)
    if not records:
        raise ValueError(f"COLMAP model contains no registered images: {sparse}")
    return sparse, records


def assign_grid(records: list[dict[str, Any]], rows: int, cols: int) -> tuple[list[dict[str, Any]], list[float]]:
    centers = np.asarray([record["center_world_m"] for record in records], dtype=np.float64)
    x_min, y_min = np.min(centers[:, :2], axis=0)
    x_max, y_max = np.max(centers[:, :2], axis=0)
    x_max = float(x_max) if not math.isclose(float(x_min), float(x_max)) else float(x_min + 1.0)
    y_max = float(y_max) if not math.isclose(float(y_min), float(y_max)) else float(y_min + 1.0)
    bounds = [float(x_min), x_max, float(y_min), y_max]
    output: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        x, y = [float(value) for value in record["center_world_m"][:2]]
        col = min(cols - 1, max(0, int((x - bounds[0]) / (bounds[1] - bounds[0]) * cols)))
        row = min(rows - 1, max(0, int((y - bounds[2]) / (bounds[3] - bounds[2]) * rows)))
        record["scene_grid_id"] = f"grid_{row:02d}_{col:02d}"
        output.append(record)
    return output, bounds


def sample_balanced_views(
    records: list[dict[str, Any]],
    target_count: int,
    rows: int = 6,
    cols: int = 6,
    *,
    scene_id: str = "gauu_upper_campus",
) -> dict[str, Any]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    assigned, bounds = assign_grid(records, rows=rows, cols=cols)
    by_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sorted(assigned, key=lambda item: (str(item["scene_grid_id"]), int(item["view_id"]))):
        by_grid[str(record["scene_grid_id"])].append(record)
    selected: list[dict[str, Any]] = []
    offsets = {grid: 0 for grid in by_grid}
    grid_ids = sorted(by_grid)
    while len(selected) < min(target_count, len(assigned)):
        progressed = False
        for grid_id in grid_ids:
            index = offsets[grid_id]
            rows_in_grid = by_grid[grid_id]
            if index >= len(rows_in_grid):
                continue
            selected.append(rows_in_grid[index])
            offsets[grid_id] += 1
            progressed = True
            if len(selected) >= min(target_count, len(assigned)):
                break
        if not progressed:
            break
    return {
        "scene_id": str(scene_id),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "sampling_protocol": "round_robin_over_full_scene_camera_center_grid",
        "registered_view_count": len(assigned),
        "selected_view_count": len(selected),
        "scene_grid": {
            "rows": int(rows),
            "cols": int(cols),
            "bounds_xy_m": bounds,
            "occupied_grid_count": len(by_grid),
            "selected_grid_count": len({record["scene_grid_id"] for record in selected}),
        },
        "views": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample spatially balanced full-scene GauU COLMAP views")
    parser.add_argument("--colmap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--target-count", type=int, default=200)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--scene-id", default="gauu_upper_campus")
    args = parser.parse_args()

    sparse, records = load_colmap_views(args.colmap_root)
    if args.image_root:
        image_root = args.image_root.expanduser().resolve()
        for record in records:
            image_path = image_root / str(record["image_name"])
            record["image_path"] = str(image_path)
            record["image_exists"] = image_path.is_file()
    payload = sample_balanced_views(
        records,
        target_count=args.target_count,
        rows=args.grid_rows,
        cols=args.grid_cols,
        scene_id=args.scene_id,
    )
    payload["colmap_sparse_model"] = str(sparse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "views"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
