from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from stage4_qa_generate_and_eval import generate_manifest


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TASK_TYPES = ("self_where", "self_what", "env_where", "env_how")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"config_not_dict: {path}")
    return payload


def _scene_info(config_path: Path) -> tuple[dict[str, Any], str, str, Path]:
    config = _read_yaml(config_path)
    task = dict(config.get("task", {}) or {})
    layout = dict(config.get("output_layout", {}) or {})
    scene_id = str(task.get("scene_id", "") or "")
    engine = str(task.get("engine", "gsplat") or "gsplat")
    scene_root = WORKSPACE_ROOT / str(task.get("base_dir", "scene_data")) / str(task.get("scene_dir_name", scene_id))
    valid_path = scene_root / str(layout.get("stage2_review_dir", "landmarks_review")) / f"{scene_id}.valid_instances.json"
    return config, scene_id, engine, valid_path


def _pair_prefix(sample_id: str, task_type: str) -> str:
    suffix = {"self_where": "_where", "self_what": "_what", "env_where": "_where", "env_how": "_how"}[task_type]
    return sample_id[: -len(suffix)] if sample_id.endswith(suffix) else sample_id


def _pair_signature(rows: dict[str, dict[str, Any]]) -> str:
    compact = []
    for task_type in sorted(rows):
        row = rows[task_type]
        compact.append(
            {
                "task_type": task_type,
                "reference_image": row.get("reference_image"),
                "target_image": row.get("target_image"),
                "candidates": row.get("candidates"),
                "label_options": row.get("label_options"),
                "answer_option_id": row.get("answer_option_id"),
                "behavior_instance": row.get("behavior_instance"),
            }
        )
    return hashlib.sha1(json.dumps(compact, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _collect_pairs(samples: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    expected = ("self_where", "self_what") if family == "self" else ("env_where", "env_how")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for sample in samples:
        task_type = str(sample.get("task_type", "") or "")
        if task_type not in expected or str(sample.get("difficulty", "")) != "8way":
            continue
        prefix = _pair_prefix(str(sample.get("sample_id", "") or ""), task_type)
        grouped[prefix][task_type] = sample
    pairs: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, str]] = set()
    for prefix, rows in grouped.items():
        if any(task not in rows for task in expected):
            continue
        landmark_id = str(rows[expected[0]].get("landmark_id", "") or "")
        signature = _pair_signature(rows)
        signature_key = (landmark_id, signature)
        if signature_key in seen_signatures:
            continue
        seen_signatures.add(signature_key)
        pairs.append(
            {
                "prefix": prefix,
                "scene_id": str(rows[expected[0]].get("scene_id", "") or ""),
                "landmark_id": landmark_id,
                "rows": rows,
                "signature": signature,
            }
        )
    return pairs


def _balanced_select(pairs: list[dict[str, Any]], count: int, *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_landmark: dict[str, deque[dict[str, Any]]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["landmark_id"])].append(pair)
    for landmark_id, rows in grouped.items():
        rng.shuffle(rows)
        by_landmark[landmark_id] = deque(rows)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    while len(selected) < count:
        available = [landmark_id for landmark_id, queue in by_landmark.items() if queue]
        if not available:
            break
        min_count = min(counts[landmark_id] for landmark_id in available)
        least_used = sorted(landmark_id for landmark_id in available if counts[landmark_id] == min_count)
        landmark_id = rng.choice(least_used)
        selected.append(by_landmark[landmark_id].popleft())
        counts[landmark_id] += 1
    if len(selected) != count:
        raise RuntimeError(f"insufficient_unique_{count}_pairs: available={len(selected)} requested={count}")
    return selected


def _valid_landmark_ids(valid_path: Path) -> list[str]:
    payload = json.loads(valid_path.read_text(encoding="utf-8"))
    rows = list(payload.get("valid_instances", []) or []) if isinstance(payload, dict) else list(payload or [])
    return sorted(str(row.get("instance_id", "") or "") for row in rows if str(row.get("instance_id", "") or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced 400-sample, 8-way GauUscene image benchmark.")
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--pairs-per-family", type=int, default=100)
    parser.add_argument("--candidates-per-landmark", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "task_pipeline_data" / "UAV-DualCog-3DGS-Pilot" / "final_400",
    )
    args = parser.parse_args()

    all_candidates: list[dict[str, Any]] = []
    all_landmarks: list[str] = []
    source_manifests: list[str] = []
    for config_index, config_path in enumerate(args.config):
        config, scene_id, engine, valid_path = _scene_info(config_path.resolve())
        all_landmarks.extend(_valid_landmark_ids(valid_path))
        result = generate_manifest(
            config=config,
            scene_id=scene_id,
            engine=engine,
            sample_count=1_000_000,
            seed=int(args.seed) + config_index * 1009,
            reference_main_only=False,
            difficulties=["8way"],
            task_types=list(TASK_TYPES),
            qa_samples_per_difficulty=max(1, int(args.candidates_per_landmark)),
        )
        manifest = dict(result["manifest"])
        samples = [dict(row) for row in list(manifest.get("samples", []) or [])]
        if any(str(row.get("difficulty", "")) != "8way" for row in samples):
            raise RuntimeError(f"non_8way_candidate_generated: {scene_id}")
        all_candidates.extend(samples)
        source_manifests.append(str(Path(result["manifest_path"]).resolve()))
        print(f"[{scene_id}] candidate_samples={len(samples)}", flush=True)

    self_pairs = _collect_pairs(all_candidates, "self")
    env_pairs = _collect_pairs(all_candidates, "env")
    selected_self = _balanced_select(self_pairs, int(args.pairs_per_family), seed=int(args.seed) + 1)
    selected_env = _balanced_select(env_pairs, int(args.pairs_per_family), seed=int(args.seed) + 2)

    selected_samples: list[dict[str, Any]] = []
    for task_type in TASK_TYPES:
        source_pairs = selected_self if task_type.startswith("self_") else selected_env
        selected_samples.extend(dict(pair["rows"][task_type]) for pair in source_pairs)

    task_counts = Counter(str(row.get("task_type", "")) for row in selected_samples)
    expected_count = int(args.pairs_per_family)
    if any(task_counts[task_type] != expected_count for task_type in TASK_TYPES):
        raise RuntimeError(f"unexpected_task_counts: {dict(task_counts)}")

    landmark_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selected_samples:
        task_type = str(row.get("task_type", ""))
        landmark_task_counts[str(row.get("landmark_id", ""))][task_type] += 1
        scene_task_counts[str(row.get("scene_id", ""))][task_type] += 1

    eligible_self = sorted({str(pair["landmark_id"]) for pair in self_pairs})
    eligible_env = sorted({str(pair["landmark_id"]) for pair in env_pairs})
    qwen_label_counts = Counter(
        str((row.get("metadata", {}) or {}).get("query_position_label", "") or "")
        for row in selected_samples
        if str(row.get("task_type", "")).startswith("env_")
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generation_id": "gauu_3scene_30landmark_8way_400",
        "generated_at": generated_at,
        "scene_id": "gauu_3scene_real_reconstruction",
        "engine": "gsplat",
        "sample_count": len(selected_samples),
        "seed": int(args.seed),
        "difficulties": ["8way"],
        "task_types": list(TASK_TYPES),
        "source_manifests": source_manifests,
        "samples": selected_samples,
    }
    source_manifest_by_scene: dict[str, Path] = {}
    for source_path_text in source_manifests:
        source_path = Path(source_path_text)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_scene_id = str(source_payload.get("scene_id", "") or "")
        if source_scene_id:
            source_manifest_by_scene[source_scene_id] = source_path
    review_manifests: dict[str, str] = {}
    for source_scene_id, source_path in sorted(source_manifest_by_scene.items()):
        scene_samples = [row for row in selected_samples if str(row.get("scene_id", "") or "") == source_scene_id]
        scene_payload = {
            **{key: value for key, value in manifest.items() if key != "samples"},
            "generation_id": f"{source_scene_id}_balanced_8way_selected",
            "scene_id": source_scene_id,
            "sample_count": len(scene_samples),
            "samples": scene_samples,
        }
        review_path = source_path.parent / f"{source_scene_id}.balanced_8way_selected.json"
        review_text = json.dumps(scene_payload, ensure_ascii=False, indent=2)
        review_path.write_text(review_text, encoding="utf-8")
        (source_path.parent / f"{source_scene_id}.latest_manifest.json").write_text(review_text, encoding="utf-8")
        review_manifests[source_scene_id] = str(review_path.resolve())
    audit = {
        "generated_at": generated_at,
        "sample_count": len(selected_samples),
        "task_counts": dict(sorted(task_counts.items())),
        "scene_task_counts": {key: dict(sorted(value.items())) for key, value in sorted(scene_task_counts.items())},
        "landmark_task_counts": {key: dict(sorted(value.items())) for key, value in sorted(landmark_task_counts.items())},
        "valid_landmarks": sorted(set(all_landmarks)),
        "valid_landmark_count": len(set(all_landmarks)),
        "eligible_landmarks": {"self_pair": eligible_self, "env_pair": eligible_env},
        "ineligible_landmarks": {
            "self_pair": sorted(set(all_landmarks) - set(eligible_self)),
            "env_pair": sorted(set(all_landmarks) - set(eligible_env)),
        },
        "candidate_pair_counts": {"self": len(self_pairs), "env": len(env_pairs)},
        "selected_pair_counts": {"self": len(selected_self), "env": len(selected_env)},
        "qwen_query_position_counts": dict(sorted(qwen_label_counts.items())),
        "review_manifests": review_manifests,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gauu_3scene_30landmark_8way_400.json"
    latest_path = output_dir / "latest_manifest.json"
    audit_path = output_dir / "gauu_3scene_30landmark_8way_400.audit.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "audit": str(audit_path), **audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
