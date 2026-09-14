#!/usr/bin/env python3
"""Audit registered-view image quality and update GauU candidate ranking fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _view_quality(path: Path) -> dict[str, float | bool]:
    image = np.asarray(Image.open(path).convert("RGB").resize((320, 213)), dtype=np.float32)
    luminance = image.mean(axis=2)
    # 3DGS edge artifacts are often dark green rather than exactly black.
    near_black_ratio = float(np.mean(luminance < 24.0))
    white_ratio = float(np.mean(luminance > 247.0))
    contrast = float(np.std(luminance))
    gradient = float(
        0.5 * np.mean(np.abs(np.diff(luminance, axis=0)))
        + 0.5 * np.mean(np.abs(np.diff(luminance, axis=1)))
    )
    score = (
        0.45 * max(0.0, 1.0 - min(1.0, near_black_ratio / 0.20))
        + 0.15 * max(0.0, 1.0 - min(1.0, white_ratio / 0.35))
        + 0.25 * min(1.0, contrast / 55.0)
        + 0.15 * min(1.0, gradient / 18.0)
    )
    passed = near_black_ratio < 0.08 and white_ratio < 0.20 and contrast > 25.0 and gradient > 5.0
    return {
        "score": float(score),
        "passed": bool(passed),
        "near_black_ratio": near_black_ratio,
        "white_ratio": white_ratio,
        "contrast": contrast,
        "gradient": gradient,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [dict(row) for row in payload.get("candidates", [])]
    for row in rows:
        audits: list[dict] = []
        for member in row.get("view_instances", []) or []:
            path = Path(str(member.get("image_path", "")))
            path = path if path.is_absolute() else args.base_dir / path
            if not path.is_file():
                continue
            audit = _view_quality(path)
            audits.append({"view_id": str(member.get("view_id", "")), **audit})
        valid = [audit for audit in audits if bool(audit["passed"])]
        best = sorted((float(audit["score"]) for audit in valid), reverse=True)[:4]
        visual_score = float(np.mean(best)) if best else 0.0
        geometry_score = float(row.get("geometry_quality_score", row.get("quality_score", 0.0)) or 0.0)
        row["visual_quality_score"] = visual_score
        row["visual_valid_view_count"] = len(valid)
        row["visual_quality_audit"] = audits
        row["geometry_quality_score"] = geometry_score
        row["quality_score"] = 0.5 * geometry_score + 0.5 * visual_score
    payload["candidates"] = rows
    payload["visual_quality_protocol"] = {
        "minimum_valid_views_for_preselection": 4,
        "near_black_luminance_threshold": 24.0,
        "near_black_ratio_max": 0.08,
        "white_ratio_max": 0.20,
        "contrast_min": 25.0,
        "gradient_min": 5.0,
        "combined_score": "0.5 * geometry_quality + 0.5 * mean(top-4 valid image quality)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidates": len(rows),
        "eligible_with_four_visual_views": sum(int(row.get("visual_valid_view_count", 0)) >= 4 for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
