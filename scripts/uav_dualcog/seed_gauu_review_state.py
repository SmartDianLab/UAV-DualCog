#!/usr/bin/env python3
"""Seed Stage 2 review decisions from a temporary GauU landmark selection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _rows(payload: object, *keys: str) -> list[dict]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in keys:
            if isinstance(payload.get(key), list):
                return [dict(row) for row in payload[key] if isinstance(row, dict)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    instance_payload = json.loads(args.instances.read_text(encoding="utf-8"))
    selection_payload = json.loads(args.selection.read_text(encoding="utf-8"))
    instances = _rows(instance_payload, "instances", "candidates", "valid_instances")
    selections = _rows(selection_payload, "selections", "valid_instances", "candidates")
    selected_ids = {str(row.get("instance_id", "")) for row in selections}
    if len(selected_ids) != 20:
        raise ValueError(f"expected 20 temporary selections, found {len(selected_ids)}")
    instance_ids = {str(row.get("instance_id", "")) for row in instances}
    missing = sorted(selected_ids - instance_ids)
    if missing:
        raise ValueError(f"selected candidates missing from review bundle: {missing}")

    note = "Temporary automatic preselection; pending manual confirmation."
    state = {
        "scene_id": args.scene_id,
        "current_index": 0,
        "updated_at": time.time(),
        "instance_decisions": {
            instance_id: {"action": "keep" if instance_id in selected_ids else "drop", "note": note}
            for instance_id in sorted(instance_ids)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scene_id": args.scene_id, "instances": len(instance_ids), "keep": 20, "drop": len(instance_ids) - 20}, indent=2))


if __name__ == "__main__":
    main()
