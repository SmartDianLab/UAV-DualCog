from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image, ImageDraw


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
POSITION_TO_BENCHMARK = {
    "left": "Left",
    "front-left": "Front-Left",
    "center": "Front",
    "front-right": "Front-Right",
    "right": "Right",
}
SYSTEM_PROMPT = (
    "You classify the screen-space position of one marked aerial landmark. "
    "Return exactly one JSON object and no Markdown. /no_think"
)
USER_PROMPT = """The target landmark is enclosed by a red rectangle. Classify its horizontal position in the image into exactly one of five ordered zones:
- left: far-left peripheral region
- front-left: between far-left and image center
- center: central region
- front-right: between image center and far-right
- right: far-right peripheral region

Use the visible target and the red rectangle together. Return:
{"position": "left|front-left|center|front-right|right", "confidence": 0.0, "reason": "brief reason"}
"""


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"config_not_dict: {path}")
    return payload


def _resolve_scene_paths(config: dict[str, Any]) -> tuple[str, Path, Path, Path]:
    task = dict(config.get("task", {}) or {})
    layout = dict(config.get("output_layout", {}) or {})
    scene_id = str(task.get("scene_id", "") or "").strip()
    scene_dir = str(task.get("scene_dir_name", "") or scene_id).strip()
    scene_root = WORKSPACE_ROOT / str(task.get("base_dir", "scene_data")) / scene_dir
    review_root = scene_root / str(layout.get("stage2_review_dir", "landmarks_review"))
    raw_root = scene_root / str(layout.get("stage2_raw_dir", "landmarks_raw"))
    valid_path = review_root / f"{scene_id}.valid_instances.json"
    output_path = review_root / f"{scene_id}.query_position_annotations.json"
    return scene_id, raw_root, valid_path, output_path


def _resolve_image(raw_root: Path, scene_root: Path, relative_path: str) -> Path:
    raw = Path(relative_path)
    candidates = [raw] if raw.is_absolute() else [raw_root / raw, scene_root / raw, WORKSPACE_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"view image not found: {relative_path}")


def _marked_image_data_url(image_path: Path, bbox: list[Any]) -> str:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        x0, y0, x1, y1 = [float(value) for value in bbox]
        width = max(4, int(round(max(image.size) / 240)))
        draw.rectangle((x0, y0, x1, y1), outline=(230, 45, 45), width=width)
        max_side = 1280
        if max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_response(text: str) -> tuple[str, float, str]:
    cleaned = str(text or "").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"response_not_json: {cleaned[:160]}")
    payload = json.loads(match.group(0))
    position = str(payload.get("position", "") or "").strip().lower().replace("_", "-")
    if position not in POSITION_TO_BENCHMARK:
        raise ValueError(f"invalid_position: {position}")
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0)))
    reason = str(payload.get("reason", "") or "").strip()
    return position, confidence, reason


def _request_annotation(
    *,
    api_base: str,
    api_key: str,
    model: str,
    image_data_url: str,
    timeout: float,
    retries: int,
) -> tuple[str, float, str, str]:
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 160,
        "enable_thinking": False,
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        request = Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw_text = str(payload["choices"][0]["message"].get("content", "") or "")
            position, confidence, reason = _parse_response(raw_text)
            return position, confidence, reason, raw_text
        except (HTTPError, URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(4.0, 0.75 * attempt))
    raise RuntimeError(last_error or "query_position_annotation_failed")


def _load_existing(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("annotations", []) or []) if isinstance(payload, dict) else []
    return {
        (str(row.get("instance_id", "") or ""), str(row.get("relative_path", "") or "")): dict(row)
        for row in rows
        if isinstance(row, dict)
    }


def annotate_scene(
    config_path: Path,
    *,
    api_base: str,
    api_key: str,
    model: str,
    workers: int,
    timeout: float,
    retries: int,
    resume: bool,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    scene_id, raw_root, valid_path, output_path = _resolve_scene_paths(config)
    scene_root = valid_path.parents[1]
    payload = json.loads(valid_path.read_text(encoding="utf-8"))
    instances = list(payload.get("valid_instances", []) or [])
    existing = _load_existing(output_path) if resume else {}
    jobs: list[dict[str, Any]] = []
    for item in instances:
        if str(item.get("review_action", "") or "").lower() not in {"", "keep"}:
            continue
        instance_id = str(item.get("instance_id", "") or "")
        for view in list(item.get("rgb_views", []) or []):
            if not isinstance(view, dict) or not bool(view.get("is_valid", False)) or not bool(view.get("bbox_2d_valid", False)):
                continue
            relative_path = str(view.get("path", "") or "").replace("\\", "/")
            key = (instance_id, relative_path)
            if key in existing:
                continue
            jobs.append(
                {
                    "instance_id": instance_id,
                    "view_id": view.get("view_id", None),
                    "source_view_id": view.get("source_view_id", None),
                    "relative_path": relative_path,
                    "image_path": _resolve_image(raw_root, scene_root, relative_path),
                    "bbox_2d_xyxy": list(view.get("bbox_2d_xyxy", []) or []),
                    "bbox_2d_image_size": list(view.get("bbox_2d_image_size", []) or []),
                    "reviewed_view_direction": str(view.get("view_direction", "") or ""),
                }
            )

    def run(job: dict[str, Any]) -> dict[str, Any]:
        image_url = _marked_image_data_url(Path(job["image_path"]), list(job["bbox_2d_xyxy"]))
        position, confidence, reason, raw_text = _request_annotation(
            api_base=api_base,
            api_key=api_key,
            model=model,
            image_data_url=image_url,
            timeout=timeout,
            retries=retries,
        )
        return {
            "instance_id": job["instance_id"],
            "view_id": job["view_id"],
            "source_view_id": job["source_view_id"],
            "relative_path": job["relative_path"],
            "bbox_2d_xyxy": job["bbox_2d_xyxy"],
            "bbox_2d_image_size": job["bbox_2d_image_size"],
            "reviewed_view_direction": job["reviewed_view_direction"],
            "position_label": position,
            "benchmark_label": POSITION_TO_BENCHMARK[position],
            "confidence": confidence,
            "reason": reason,
            "source": f"vlm:{model}",
            "raw_response": raw_text,
        }

    completed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(run, job): job for job in jobs}
        for index, future in enumerate(as_completed(future_map), start=1):
            job = future_map[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append({"instance_id": str(job["instance_id"]), "relative_path": str(job["relative_path"]), "error": str(exc)})
            if index % 10 == 0 or index == len(jobs):
                print(f"[{scene_id}] annotated={index - len(errors)}/{len(jobs)} errors={len(errors)}", flush=True)

    merged = dict(existing)
    for row in completed:
        merged[(str(row["instance_id"]), str(row["relative_path"]))] = row
    annotations = sorted(merged.values(), key=lambda row: (str(row.get("instance_id", "")), str(row.get("relative_path", ""))))
    output = {
        "schema_version": "1.0",
        "scene_id": scene_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt_version": "screen_position_5way_v1",
        "label_mapping": POSITION_TO_BENCHMARK,
        "valid_landmark_count": len(instances),
        "annotation_count": len(annotations),
        "errors": errors,
        "annotations": annotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"scene_id": scene_id, "output": str(output_path), "annotations": len(annotations), "new": len(completed), "errors": len(errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate reviewed GauUscene views with five-way landmark image positions.")
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--api-base", default="http://127.0.0.1:40900/v1")
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    summaries = [
        annotate_scene(
            config_path.resolve(),
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            resume=not args.no_resume,
        )
        for config_path in args.config
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    if any(row["errors"] for row in summaries):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
