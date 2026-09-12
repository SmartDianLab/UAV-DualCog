#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_yaml_root: {path}")
    return payload


def _classify_models(payload: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    models = dict((payload.get("api", {}) or {}).get("models", {}) or {})
    local_rows: list[dict[str, str]] = []
    api_rows: list[dict[str, str]] = []
    for name, cfg in models.items():
        source = str((cfg or {}).get("api_source", "") or "").strip()
        api_base = str((cfg or {}).get("api_base", "") or "").strip()
        row = {
            "model": str(name),
            "api_source": source or "-",
            "api_base": api_base or "-",
            "mode": "local" if source.lower() == "local" else "api",
        }
        if row["mode"] == "local":
            local_rows.append(row)
        else:
            api_rows.append(row)
    local_rows.sort(key=lambda item: item["model"].lower())
    api_rows.sort(key=lambda item: item["model"].lower())
    return local_rows, api_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate model-routing checks without real LLM calls.")
    parser.add_argument(
        "--config",
        default="configs/uav_dualcog/common_api_runtime.yaml",
        help="Path to common_api_runtime.yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of plain text summary.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"config_not_found: {cfg_path}")
    payload = _load_yaml(cfg_path)
    local_rows, api_rows = _classify_models(payload)
    result = {
        "config": str(cfg_path),
        "local_model_count": len(local_rows),
        "api_model_count": len(api_rows),
        "local_models": local_rows,
        "api_models": api_rows,
        "note": "This is a dry routing check only. No model inference requests were sent.",
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"[mock_api_runtime_check] config={result['config']}")
    print(f"  local models: {result['local_model_count']}")
    for row in local_rows:
        print(f"    - {row['model']} [{row['api_source']}] -> {row['api_base']}")
    print(f"  api models: {result['api_model_count']}")
    for row in api_rows:
        print(f"    - {row['model']} [{row['api_source']}] -> {row['api_base']}")
    print(f"  note: {result['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
