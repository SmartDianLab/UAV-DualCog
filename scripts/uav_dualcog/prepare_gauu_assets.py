#!/usr/bin/env python3
"""Validate and prepare released CityGaussianV2/GauUscene scene assets."""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path
from typing import Any

try:
    from .prepare_gauu_upper_assets import convert_lidar_to_voxel_array, sha256_file
except ImportError:
    from prepare_gauu_upper_assets import convert_lidar_to_voxel_array, sha256_file


SCENE_SPECS: dict[str, dict[str, Any]] = {
    "lfls": {
        "archive": "LFLS.zip",
        "archive_size": 8_025_931_066,
        "archive_sha256": "2fc0fb213b9f66379445832d77504da87987abda0c35a80a158048db2b1f2fa1",
        "extracted_dir": "LFLS",
    },
    "smbu": {
        "archive": "SMBU.zip",
        "archive_size": 7_715_544_640,
        "archive_sha256": "1eaf8cdd7bc00d4a82dbe8cbbd8a5fb4a8e957ec4596d2ad08fdc147261c9c8e",
        "extracted_dir": "SMBU",
    },
    "upper": {
        "archive": "UPPER_CAMPUS.zip",
        "archive_size": 9_404_314_078,
        "archive_sha256": "fe4fecc4bb6ee80e6c75c24f06dd57f67a31a9c3e45af6d9614aba2dd45e7c8d",
        "extracted_dir": "UPPER_CAMPUS",
    },
}


def _find_one(root: Path, names: tuple[str, ...]) -> Path:
    lowered = {name.lower() for name in names}
    matches = sorted(path for path in root.rglob("*") if path.is_file() and path.name.lower() in lowered)
    if not matches:
        raise FileNotFoundError(f"none of {names} found under {root}")
    return matches[0]


def find_colmap_root(root: Path) -> Path:
    for candidate in sorted(path for path in root.rglob("*") if path.is_dir()):
        names = {path.name.lower() for path in candidate.iterdir() if path.is_file()}
        if {"cameras.bin", "images.bin"}.issubset(names) or {"cameras.txt", "images.txt"}.issubset(names):
            return candidate
    raise FileNotFoundError(f"COLMAP sparse model not found under {root}")


def inspect_archive(path: Path, spec: dict[str, Any], *, verify_hash: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "expected_size": int(spec["archive_size"]),
    }
    result["size_matches"] = result["size"] == result["expected_size"]
    with zipfile.ZipFile(path) as archive:
        checkpoints = sorted(name for name in archive.namelist() if name.lower().endswith(".ckpt"))
        result["checkpoint_members"] = checkpoints
        result["archive_valid"] = archive.testzip() is None if verify_hash else bool(checkpoints)
    if verify_hash:
        result["sha256"] = sha256_file(path)
        result["expected_sha256"] = str(spec["archive_sha256"])
        result["hash_matches"] = result["sha256"] == result["expected_sha256"]
    return result


def extract_archive(path: Path, output_root: Path) -> Path:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(root)
    checkpoints = sorted(root.rglob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found after extracting {path}")
    return checkpoints[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENE_SPECS), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--verify-archive-hash", action="store_true")
    parser.add_argument("--extract-model", action="store_true")
    parser.add_argument("--convert-lidar", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.asset_root.expanduser().resolve()
    spec = SCENE_SPECS[args.scene]
    archive = root / "citygaussian_v2" / str(spec["archive"])
    gauu_root = root / "gauu_v2_1"
    lidar = _find_one(gauu_root, ("Lidar.ply", "LiDAR.ply"))
    colmap = find_colmap_root(gauu_root)
    report: dict[str, Any] = {
        "scene": args.scene,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "archive": inspect_archive(archive, spec, verify_hash=args.verify_archive_hash),
        "lidar": {"path": str(lidar), "size": lidar.stat().st_size},
        "colmap_sparse_model": str(colmap),
    }
    if args.verify_archive_hash and not report["archive"].get("hash_matches", False):
        raise ValueError(f"archive hash mismatch: {archive}")
    if args.extract_model:
        report["checkpoint"] = str(extract_archive(archive, root / "citygaussian_v2" / "extracted"))
    if args.convert_lidar:
        output = gauu_root / f"lidar_world_voxel_{str(args.voxel_size).replace('.', 'p')}m.npy"
        report["lidar_conversion"] = convert_lidar_to_voxel_array(
            lidar,
            output,
            voxel_size=args.voxel_size,
        )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
