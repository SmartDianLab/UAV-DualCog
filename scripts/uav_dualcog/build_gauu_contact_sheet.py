#!/usr/bin/env python3
"""Build a labeled contact sheet for GauU landmark review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    for key in ("valid_instances", "instances", "candidates"):
        if isinstance(payload.get(key), list):
            return [dict(row) for row in payload[key]]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=260)
    args = parser.parse_args()

    rows = _rows(args.input)
    tiles: list[Image.Image] = []
    font = ImageFont.load_default()
    for row in rows:
        views = list(row.get("rgb_views", []) or [])
        if not views:
            continue
        image_path = args.image_root / str(views[0].get("path", ""))
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((args.tile_width, args.tile_height - 42), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (args.tile_width, args.tile_height), "white")
        tile.paste(image, ((args.tile_width - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        label = f"{row.get('instance_id', '')} | {row.get('landmark_subcategory', '')}"
        score = f"quality={float(row.get('quality_score', 0.0)):.3f} | views={len(views)}"
        draw.text((6, args.tile_height - 38), label[:58], fill="black", font=font)
        draw.text((6, args.tile_height - 22), score, fill="#444444", font=font)
        tiles.append(tile)
    if not tiles:
        raise ValueError("no landmark view images found")

    columns = max(1, args.columns)
    rows_count = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * args.tile_width, rows_count * args.tile_height), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * args.tile_width, (index // columns) * args.tile_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=92)
    print(json.dumps({"landmarks": len(tiles), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
