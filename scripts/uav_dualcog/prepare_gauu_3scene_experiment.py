from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "task_pipeline_data"
    / "UAV-DualCog-3DGS-Pilot"
    / "final_400"
    / "gauu_3scene_30landmark_8way_400.json"
)
DEFAULT_SCENES = ("gauu_upper_campus", "gauu_smbu", "gauu_lfls")
TASK_TYPES = ("self_where", "self_what", "env_where", "env_how")


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest_not_object: {path}")
    samples = list(payload.get("samples", []) or [])
    if not samples:
        raise RuntimeError(f"manifest_has_no_samples: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish the audited three-scene GauUscene manifest for one multi-scene experiment run."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task-name", default="UAV-DualCog-3DGS-RealScene-400")
    parser.add_argument(
        "--task-pipeline-root",
        type=Path,
        default=WORKSPACE_ROOT / "task_pipeline_data",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = _read_manifest(manifest_path)
    samples = [dict(row) for row in list(manifest.get("samples", []) or []) if isinstance(row, dict)]

    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        scene_id = str(row.get("scene_id", "") or "").strip()
        if not scene_id:
            raise RuntimeError(f"sample_without_scene_id: {row.get('sample_id', '<unknown>')}")
        scene_rows[scene_id].append(row)

    unexpected_scenes = sorted(set(scene_rows) - set(DEFAULT_SCENES))
    missing_scenes = sorted(set(DEFAULT_SCENES) - set(scene_rows))
    if unexpected_scenes or missing_scenes:
        raise RuntimeError(
            f"scene_set_mismatch: missing={missing_scenes} unexpected={unexpected_scenes}"
        )

    task_counts = Counter(str(row.get("task_type", "") or "") for row in samples)
    if len(samples) != 400 or any(task_counts[task_type] != 100 for task_type in TASK_TYPES):
        raise RuntimeError(
            f"unexpected_benchmark_distribution: total={len(samples)} tasks={dict(task_counts)}"
        )

    output_root = args.task_pipeline_root.resolve() / str(args.task_name)
    published: dict[str, dict[str, Any]] = {}
    for scene_id in DEFAULT_SCENES:
        rows = scene_rows[scene_id]
        scene_payload = {
            **{key: value for key, value in manifest.items() if key != "samples"},
            "generation_id": f"{scene_id}_balanced_8way_selected",
            "scene_id": scene_id,
            "sample_count": len(rows),
            "source_combined_manifest": str(manifest_path),
            "samples": rows,
        }
        manifest_dir = output_root / f"gsplat_{scene_id}" / "qa" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        selected_path = manifest_dir / f"{scene_id}.balanced_8way_selected.json"
        latest_path = manifest_dir / f"{scene_id}.latest_manifest.json"
        text = json.dumps(scene_payload, ensure_ascii=False, indent=2)
        selected_path.write_text(text, encoding="utf-8")
        latest_path.write_text(text, encoding="utf-8")
        published[scene_id] = {
            "sample_count": len(rows),
            "task_counts": dict(
                sorted(Counter(str(row.get("task_type", "") or "") for row in rows).items())
            ),
            "manifest": str(latest_path),
        }

    print(
        json.dumps(
            {
                "task_name": str(args.task_name),
                "combined_sample_count": len(samples),
                "combined_task_counts": dict(sorted(task_counts.items())),
                "published": published,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
