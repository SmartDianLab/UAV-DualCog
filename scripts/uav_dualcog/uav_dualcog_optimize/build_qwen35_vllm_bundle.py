#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Set

from safetensors import safe_open

# Allow direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uav_dualcog.uav_dualcog_optimize.common import ensure_dir, resolve_path


BASE_META_FILES = [
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def _load_merged_keys(merged_weight_path: Path) -> Set[str]:
    with safe_open(str(merged_weight_path), framework="np") as f:
        return set(f.keys())


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_bundle(base_model_dir: Path, merged_model_dir: Path, output_dir: Path) -> Dict[str, object]:
    base_index_path = base_model_dir / "model.safetensors.index.json"
    merged_weight_path = merged_model_dir / "model.safetensors"
    if not base_index_path.exists():
        raise FileNotFoundError(base_index_path)
    if not merged_weight_path.exists():
        raise FileNotFoundError(merged_weight_path)

    ensure_dir(output_dir)

    base_index = json.load(base_index_path.open("r", encoding="utf-8"))
    base_weight_map: Dict[str, str] = dict(base_index.get("weight_map", {}) or {})
    merged_keys = _load_merged_keys(merged_weight_path)
    base_keys = set(base_weight_map.keys())
    missing_keys = sorted(base_keys - merged_keys)
    extra_keys = sorted(merged_keys - base_keys)
    if extra_keys:
        raise ValueError(f"Merged model has keys absent from base model: {extra_keys[:20]}")

    # Copy metadata/tokenizer/processor files from the original multimodal package.
    for rel in BASE_META_FILES:
        src = base_model_dir / rel
        if src.exists():
            shutil.copy2(src, output_dir / rel)

    # Copy merged language-model weights into the bundle.
    shutil.copy2(merged_weight_path, output_dir / "model.safetensors")

    # Reuse original shard files for weights not present in merged export
    # (notably visual tower + mtp branch).
    referenced_base_files = sorted({base_weight_map[k] for k in missing_keys})
    for rel in referenced_base_files:
        _symlink_or_copy(base_model_dir / rel, output_dir / rel)

    # Build a hybrid index: prefer merged weights where available, fallback to base shard map.
    hybrid_weight_map: Dict[str, str] = {}
    for key in sorted(base_keys):
        hybrid_weight_map[key] = "model.safetensors" if key in merged_keys else base_weight_map[key]

    hybrid_index = {
        "metadata": base_index.get("metadata", {}),
        "weight_map": hybrid_weight_map,
    }
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(hybrid_index, f, ensure_ascii=False, indent=2)

    summary = {
        "base_model_dir": str(base_model_dir),
        "merged_model_dir": str(merged_model_dir),
        "output_dir": str(output_dir),
        "base_total_keys": len(base_keys),
        "merged_total_keys": len(merged_keys),
        "hybrid_total_keys": len(hybrid_weight_map),
        "missing_keys_filled_from_base": len(missing_keys),
        "missing_visual_or_vision_keys": sum(1 for k in missing_keys if ("visual" in k.lower() or "vision" in k.lower())),
        "missing_other_keys": [k for k in missing_keys if ("visual" not in k.lower() and "vision" not in k.lower())],
        "base_files_linked_or_copied": referenced_base_files,
    }
    with (output_dir / "bundle_build_report.latest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a vLLM-servable Qwen3.5 multimodal bundle from a language-only merged checkpoint.")
    parser.add_argument("--base-model-dir", default="vllm/models/qwen3_5-4b")
    parser.add_argument("--merged-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_model_dir = resolve_path(args.base_model_dir)
    merged_model_dir = resolve_path(args.merged_model_dir)
    output_dir = resolve_path(args.output_dir)
    summary = build_bundle(base_model_dir, merged_model_dir, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
