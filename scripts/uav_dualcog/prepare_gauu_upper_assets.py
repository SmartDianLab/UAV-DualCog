#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_MODEL_ARCHIVE_SIZE = 9_404_314_078
EXPECTED_MODEL_ARCHIVE_SHA256 = "fe4fecc4bb6ee80e6c75c24f06dd57f67a31a9c3e45af6d9614aba2dd45e7c8d"
EXPECTED_CHECKPOINT_SUFFIX = "UPPER_CAMPUS/checkpoints/epoch=48-step=30000.ckpt"
EXPECTED_LIDAR_SIZE = 1_413_283_738
EXPECTED_LIDAR_SHA256 = "273e7918cb80bb7508151b51115baac70cfd06cae8379cd39a02373408f12cdc"


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        chunk = handle.read(chunk_size)
        while chunk:
            digest.update(chunk)
            chunk = handle.read(chunk_size)
    return digest.hexdigest()


def inspect_model_archive(path: Path, *, verify_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    archive_error = None
    names: list[str] = []
    bad_member = None
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            bad_member = archive.testzip() if bool(verify_hash) else None
    except zipfile.BadZipFile as exc:
        archive_error = str(exc)
    checkpoint_names = [name for name in names if name.lower().endswith(".ckpt")]
    return {
        "path": str(path.resolve()),
        "size": int(size),
        "expected_size": EXPECTED_MODEL_ARCHIVE_SIZE,
        "size_matches": int(size) == EXPECTED_MODEL_ARCHIVE_SIZE,
        "sha256": sha256_file(path) if verify_hash else None,
        "expected_sha256": EXPECTED_MODEL_ARCHIVE_SHA256,
        "checkpoint_names": checkpoint_names,
        "expected_checkpoint_present": EXPECTED_CHECKPOINT_SUFFIX in names,
        "archive_valid": archive_error is None and bad_member is None,
        "archive_error": archive_error,
        "zip_test_error": bad_member,
    }


def inspect_lidar(path: Path, *, verify_hash: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    digest = sha256_file(path) if verify_hash else None
    return {
        "path": str(path.resolve()),
        "size": int(size),
        "expected_size": EXPECTED_LIDAR_SIZE,
        "size_matches": int(size) == EXPECTED_LIDAR_SIZE,
        "sha256": digest,
        "expected_sha256": EXPECTED_LIDAR_SHA256,
        "hash_matches": digest == EXPECTED_LIDAR_SHA256 if digest is not None else None,
    }


def safe_extract_model_archive(path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(root)
    checkpoint = root / EXPECTED_CHECKPOINT_SUFFIX
    if not checkpoint.is_file():
        raise FileNotFoundError(f"expected checkpoint missing after extraction: {checkpoint}")
    return checkpoint


def convert_lidar_to_voxel_array(
    input_path: Path,
    output_path: Path,
    *,
    voxel_size: float = 0.5,
    chunk_size: int = 2_000_000,
) -> dict[str, Any]:
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("LiDAR conversion requires plyfile") from exc
    vertex = PlyData.read(str(input_path), mmap=True)["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("LiDAR PLY must contain x, y, and z properties")
    reduced_chunks: list[np.ndarray] = []
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for start in range(0, len(vertex), int(chunk_size)):
        block = vertex[start : start + int(chunk_size)]
        xyz = np.column_stack([block[axis] for axis in ("x", "y", "z")]).astype(np.float32)
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        if len(xyz) == 0:
            continue
        bounds_min = np.minimum(bounds_min, np.min(xyz, axis=0))
        bounds_max = np.maximum(bounds_max, np.max(xyz, axis=0))
        voxels = np.floor(xyz / float(voxel_size)).astype(np.int64)
        _, indices = np.unique(voxels, axis=0, return_index=True)
        reduced_chunks.append(xyz[np.sort(indices)])
    if not reduced_chunks:
        raise ValueError("LiDAR PLY contains no finite points")
    merged = np.concatenate(reduced_chunks, axis=0)
    merged_voxels = np.floor(merged / float(voxel_size)).astype(np.int64)
    _, indices = np.unique(merged_voxels, axis=0, return_index=True)
    reduced = merged[np.sort(indices)].astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, reduced)
    return {
        "input_path": str(input_path.resolve()),
        "output_path": str(output_path.resolve()),
        "input_points": int(len(vertex)),
        "output_points": int(len(reduced)),
        "voxel_size_m": float(voxel_size),
        "bounds_min_m": [float(value) for value in bounds_min],
        "bounds_max_m": [float(value) for value in bounds_max],
    }


def inspect_asset_root(asset_root: Path, citygaussian_root: Path) -> dict[str, Any]:
    archive = asset_root / "citygaussian_v2" / "UPPER_CAMPUS.zip"
    extracted_root = asset_root / "citygaussian_v2" / "extracted" / "UPPER_CAMPUS"
    checkpoint = extracted_root / "checkpoints" / "epoch=48-step=30000.ckpt"
    gauu = asset_root / "gauu_v2_1"
    required = {
        "model_archive": archive,
        "model_checkpoint": checkpoint,
        "colmap_metrics": gauu / "colmap-metrics",
        "lidar_ply": gauu / "Lidar.ply",
        "lidar_voxel_npy": gauu / "lidar_world_voxel_0p5m.npy",
        "citygaussian_code": citygaussian_root / "internal" / "utils" / "gaussian_model_loader.py",
    }
    present = {key: path.exists() for key, path in required.items()}
    return {
        "asset_root": str(asset_root.resolve()),
        "citygaussian_root": str(citygaussian_root.resolve()),
        "present": present,
        "paths": {key: str(path.resolve()) for key, path in required.items()},
        "ready_for_render": present["model_checkpoint"] and present["lidar_voxel_npy"] and present["citygaussian_code"],
        "ready_for_metric_alignment": present["colmap_metrics"] and present["lidar_ply"],
    }


def _write_report(path: Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GauUscene V2.1 Upper Campus assets for UAV-DualCog")
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--citygaussian-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-archive-hash", action="store_true")
    parser.add_argument("--verify-lidar-hash", action="store_true")
    parser.add_argument("--extract-model", action="store_true")
    parser.add_argument("--convert-lidar", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.5)
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    archive = asset_root / "citygaussian_v2" / "UPPER_CAMPUS.zip"
    actions: dict[str, Any] = {}
    if archive.exists():
        actions["model_archive"] = inspect_model_archive(archive, verify_hash=args.verify_archive_hash)
    lidar = asset_root / "gauu_v2_1" / "Lidar.ply"
    if lidar.exists():
        actions["lidar"] = inspect_lidar(lidar, verify_hash=args.verify_lidar_hash)
    if args.extract_model:
        destination = asset_root / "citygaussian_v2" / "extracted"
        actions["extracted_checkpoint"] = str(safe_extract_model_archive(archive, destination))
    if args.convert_lidar:
        gauu = asset_root / "gauu_v2_1"
        actions["lidar_conversion"] = convert_lidar_to_voxel_array(
            gauu / "Lidar.ply",
            gauu / f"lidar_world_voxel_{str(args.voxel_size).replace('.', 'p')}m.npy",
            voxel_size=args.voxel_size,
        )
        if args.voxel_size == 0.5:
            generated = Path(actions["lidar_conversion"]["output_path"])
            canonical = gauu / "lidar_world_voxel_0p5m.npy"
            if generated != canonical:
                shutil.move(str(generated), str(canonical))
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "actions": actions,
        "status": inspect_asset_root(asset_root, args.citygaussian_root.expanduser().resolve()),
    }
    _write_report(args.report, output)


if __name__ == "__main__":
    main()
