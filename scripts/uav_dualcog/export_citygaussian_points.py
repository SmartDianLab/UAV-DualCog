#!/usr/bin/env python3
"""Export filtered Gaussian centers from an official CityGaussian checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _tensor_value(model: object, name: str):
    value = getattr(model, name, None)
    return value() if callable(value) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--citygaussian-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opacity-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    citygaussian_root = args.citygaussian_root.resolve()
    if not (citygaussian_root / "internal").is_dir():
        raise FileNotFoundError(f"invalid CityGaussian root: {citygaussian_root}")
    sys.path.insert(0, str(citygaussian_root))

    import torch
    from internal.utils.gaussian_model_loader import GaussianModelLoader

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = GaussianModelLoader.search_load_file(str(args.checkpoint_root.resolve()))
    model, _, _ = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(
        str(checkpoint),
        device=device,
        eval_mode=True,
        pre_activate=True,
    )
    model.freeze()

    xyz = _tensor_value(model, "get_xyz")
    opacity = _tensor_value(model, "get_opacity")
    if xyz is None:
        raise AttributeError("CityGaussian model does not expose get_xyz")
    if opacity is None:
        raise AttributeError("CityGaussian model does not expose get_opacity")

    xyz_np = xyz.detach().float().cpu().numpy()
    opacity_np = opacity.detach().float().cpu().numpy().reshape(-1)
    keep = np.isfinite(xyz_np).all(axis=1) & np.isfinite(opacity_np)
    keep &= opacity_np > args.opacity_threshold
    points = np.asarray(xyz_np[keep, :3], dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, points)
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "device": str(device),
        "opacity_threshold": args.opacity_threshold,
        "gaussian_count": int(len(xyz_np)),
        "exported_count": int(len(points)),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
