from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image


TASK_TYPE_DISPLAY: Dict[str, str] = {
    "self_where": "Landmark-Relative Position Reasoning",
    "self_what": "Future Observation Prediction",
    "env_where": "Self-Relative Position Reasoning",
    "env_how": "Landmark-Driven Action Decision",
}

MULTIMODAL_SUPPORT_FILES = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (project_root_from_here() / p).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_vllm_cuda_lib_priority() -> Dict[str, Any]:
    """
    Ensure the vllm-conda packaged CUDA libs are searched before system CUDA libs.
    This prevents torch loading an incompatible /usr/local/cuda/lib64/libnvJitLink.so.12.
    """
    py = Path(os.environ.get("CONDA_PREFIX", "")) / "lib" / f"python{os.sys.version_info.major}.{os.sys.version_info.minor}" / "site-packages"
    nvjit = py / "nvidia" / "nvjitlink" / "lib"
    cusparse = py / "nvidia" / "cusparse" / "lib"
    extra = [str(p) for p in (nvjit, cusparse) if p.exists()]
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if extra:
        parts = [p for p in old_ld.split(":") if p]
        for p in reversed(extra):
            if p in parts:
                parts.remove(p)
            parts.insert(0, p)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)
    return {
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "added_lib_paths": extra,
        "ld_library_path_head": os.environ.get("LD_LIBRARY_PATH", "").split(":")[:6],
    }


def normalize_qwen35_merged_config(model_dir: Path) -> Dict[str, Any]:
    """
    Normalize merged Qwen3.5 config for vLLM serving compatibility.
    Some export paths write model_type as `qwen3_5_text`, while certain
    vLLM versions expect `qwen3_5`.
    """
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {"changed": False, "reason": "config_not_found", "path": str(cfg_path)}
    with cfg_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    before = str(obj.get("model_type", ""))
    changed = False
    if before == "qwen3_5_text":
        obj["model_type"] = "qwen3_5"
        changed = True
    # Keep architecture explicit and text-causal for the merged checkpoints.
    if obj.get("architectures") != ["Qwen3_5ForCausalLM"]:
        obj["architectures"] = ["Qwen3_5ForCausalLM"]
        changed = True
    if changed:
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    return {
        "changed": changed,
        "path": str(cfg_path),
        "model_type_before": before,
        "model_type_after": str(obj.get("model_type", "")),
    }


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xyxy_norm_to_1000(b: Iterable[float]) -> List[int]:
    vals = [int(round(clamp(float(x), 0.0, 1.0) * 1000.0)) for x in b]
    if len(vals) != 4:
        raise ValueError(f"bbox must contain 4 numbers, got {vals}")
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def normalize_xyxy_norm(b: Iterable[float]) -> List[float]:
    vals = [clamp(float(x), 0.0, 1.0) for x in b]
    if len(vals) != 4:
        raise ValueError(f"bbox must contain 4 numbers, got {vals}")
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def format_bbox_norm_json(b: Iterable[float], digits: int = 6) -> str:
    vals = normalize_xyxy_norm(b)
    return "[" + ", ".join(f"{v:.{digits}f}" for v in vals) + "]"


def load_rgb_image(path_like: str | Path, max_side: int | None = None) -> Image.Image:
    path = resolve_path(str(path_like))
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if max_side is not None and max_side > 0:
            w, h = rgb.size
            scale = min(float(max_side) / float(max(w, h)), 1.0)
            if scale < 1.0:
                rgb = rgb.resize(
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    Image.Resampling.LANCZOS,
                )
        return rgb


def copy_multimodal_support_files(base_model_dir: str | Path, output_dir: str | Path) -> List[str]:
    src_dir = resolve_path(str(base_model_dir))
    dst_dir = resolve_path(str(output_dir))
    ensure_dir(dst_dir)
    copied: List[str] = []
    for name in MULTIMODAL_SUPPORT_FILES:
        src = src_dir / name
        if not src.exists():
            continue
        shutil.copy2(src, dst_dir / name)
        copied.append(name)
    return copied


def find_latest_stage4_manifests(train_root: Path) -> List[Path]:
    out: List[Path] = []
    for scene_dir in sorted(train_root.glob("airsim_env_*")):
        m = scene_dir / "qa" / "manifests" / f"{scene_dir.name.replace('airsim_', '')}.latest_manifest.json"
        if m.exists():
            out.append(m)
    return out


@dataclass
class ManifestCheckResult:
    scene_id: str
    sample_count: int
    task_difficulty_counts: Dict[str, int]
    missing_file_count: int
    bad_bbox_count: int
    option_missing_count: int


def _td_key(task_type: str, difficulty: str) -> str:
    return f"{task_type}:{difficulty}"


def check_manifest_quality(manifest_path: Path) -> ManifestCheckResult:
    d = load_json(manifest_path)
    scene_id = str(d.get("scene_id", "unknown"))
    samples = list(d.get("samples", []) or [])
    td_counts: Dict[str, int] = {}
    missing_files = 0
    bad_bbox = 0
    option_missing = 0
    for s in samples:
        task_type = str(s.get("task_type", "unknown"))
        difficulty = str(s.get("difficulty", "unknown"))
        key = _td_key(task_type, difficulty)
        td_counts[key] = td_counts.get(key, 0) + 1
        for pkey in ("reference_image", "reference_image_with_bbox", "target_image"):
            p = s.get(pkey)
            if not p:
                missing_files += 1
                continue
            rp = Path(str(p))
            if not rp.is_absolute():
                rp = (project_root_from_here() / rp).resolve()
            if not rp.exists():
                missing_files += 1
        bbox = s.get("answer_bbox_xyxy_norm")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            bad_bbox += 1
        answer = str(s.get("answer_option_id", "")).strip()
        if not answer:
            option_missing += 1
    return ManifestCheckResult(
        scene_id=scene_id,
        sample_count=len(samples),
        task_difficulty_counts=dict(sorted(td_counts.items())),
        missing_file_count=missing_files,
        bad_bbox_count=bad_bbox,
        option_missing_count=option_missing,
    )


def deterministic_subsample(items: List[Dict[str, Any]], max_per_task: int, seed: int) -> List[Dict[str, Any]]:
    if max_per_task <= 0:
        return items
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for s in items:
        k = (str(s.get("task_type", "")), str(s.get("difficulty", "")))
        grouped.setdefault(k, []).append(s)
    out: List[Dict[str, Any]] = []
    for k in sorted(grouped.keys()):
        arr = grouped[k]
        if len(arr) <= max_per_task:
            out.extend(arr)
            continue
        idx = list(range(len(arr)))
        rng.shuffle(idx)
        picked = sorted(idx[:max_per_task])
        out.extend(arr[i] for i in picked)
    return out
