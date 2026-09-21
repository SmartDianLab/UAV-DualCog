from __future__ import annotations

import argparse
import ast
import base64
import io
import ipaddress
import json
import mimetypes
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import httpx
except Exception:
    httpx = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from PIL import Image
except Exception:
    Image = None

from api_common import (
    build_model_request_controls,
    compute_rate_limited_concurrency,
    load_api_registry,
    model_disallows_temperature,
    pick_first_text,
    resolve_default_model,
    resolve_model_api_endpoint,
    should_inline_system_prompt_for_multimodal,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
OPTION_TO_IMAGE = {"A": "Image 2", "B": "Image 3", "C": "Image 4", "D": "Image 5"}
TASK_SPECS = [
    ("Self-Aware", "self_where"),
    ("Self-Aware", "self_what"),
    ("Environment-Aware", "env_where"),
    ("Environment-Aware", "env_how"),
]


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _safe_name(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return out.strip("_") or "unnamed"


def _path_for_json(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT))
    except Exception:
        return str(path.resolve())


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("pyyaml_required_for_loading_config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid_config_yaml: {path}")
    return payload


def _should_bypass_proxy(api_base: str) -> bool:
    host = str(urlparse(str(api_base or "")).hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(ip.is_private or ip.is_loopback or ip.is_link_local)


def _build_openai_client(*, api_key: str, api_base: str) -> Any:
    # Keep the OpenAI client construction aligned with the original stage4 evaluator.
    # Some local/private deployments rely on the environment's proxy / transport setup.
    return OpenAI(api_key=api_key, base_url=api_base)


def _filter_samples(samples: list[dict[str, Any]], *, limit: int | None = None) -> list[dict[str, Any]]:
    out = list(samples)
    if limit is not None and int(limit) > 0:
        out = out[: int(limit)]
    return out


def _normalize_text(text: str) -> str:
    out = str(text or "").strip().lower()
    out = out.replace("_", " ").replace("/", " ").replace("\\", " ")
    out = re.sub(r"[-–—]+", "-", out)
    out = re.sub(r"[^a-z0-9\-\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _direction_aliases() -> dict[str, str]:
    raw = {
        "Front": ["front", "ahead", "forward", "in front", "straight ahead"],
        "Back": ["back", "behind", "rear", "backward"],
        "Left": ["left", "left side", "to the left"],
        "Right": ["right", "right side", "to the right"],
        "Front-Left": ["front-left", "front left", "forward left", "ahead left"],
        "Front-Right": ["front-right", "front right", "forward right", "ahead right"],
        "Back-Left": ["back-left", "back left", "rear left", "behind left"],
        "Back-Right": ["back-right", "back right", "rear right", "behind right"],
    }
    out: dict[str, str] = {}
    for canonical, aliases in raw.items():
        for alias in aliases:
            out[_normalize_text(alias)] = canonical
    return out


_DIRECTION_ALIAS_MAP = _direction_aliases()


def _normalize_direction(text: str) -> str | None:
    norm = _normalize_text(text)
    if not norm:
        return None
    if norm in _DIRECTION_ALIAS_MAP:
        return _DIRECTION_ALIAS_MAP[norm]
    for alias in sorted(_DIRECTION_ALIAS_MAP.keys(), key=len, reverse=True):
        if re.search(rf"(^|\s){re.escape(alias)}($|\s)", norm):
            return _DIRECTION_ALIAS_MAP[alias]

    tokens = set(norm.split())
    has_front = "front" in tokens or "forward" in tokens or "ahead" in tokens
    has_back = "back" in tokens or "behind" in tokens or "rear" in tokens
    has_left = "left" in tokens
    has_right = "right" in tokens

    if has_front and has_left:
        return "Front-Left"
    if has_front and has_right:
        return "Front-Right"
    if has_back and has_left:
        return "Back-Left"
    if has_back and has_right:
        return "Back-Right"
    if has_front:
        return "Front"
    if has_back:
        return "Back"
    if has_left:
        return "Left"
    if has_right:
        return "Right"
    return None


def _normalize_self_what(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in OPTION_TO_IMAGE:
        return OPTION_TO_IMAGE[upper]

    norm = _normalize_text(raw)
    if norm in {"a", "b", "c", "d"}:
        return OPTION_TO_IMAGE[norm.upper()]

    match = re.search(r"\bimage\s*([2-5])\b", norm)
    if match:
        return f"Image {match.group(1)}"
    match = re.search(r"\bimg\s*([2-5])\b", norm)
    if match:
        return f"Image {match.group(1)}"

    words_to_num = {"second": "2", "third": "3", "fourth": "4", "fifth": "5"}
    for word, num in words_to_num.items():
        if re.search(rf"\b{word}\b", norm):
            return f"Image {num}"

    lone_num = re.search(r"\b([2-5])\b", norm)
    if lone_num:
        return f"Image {lone_num.group(1)}"
    return None


def _normalize_answer(task_type: str, answer_text: str) -> str | None:
    task = str(task_type or "").strip()
    if task == "self_what":
        return _normalize_self_what(answer_text)
    if task in {"self_where", "env_where", "env_how"}:
        return _normalize_direction(answer_text)
    return None


def _task_combo_id(view_definition: str, task_type: str, difficulty: str) -> str:
    return f"{str(view_definition)}|{str(task_type)}|{str(difficulty)}"


def _task_display_name(view_definition: str, task_type: str) -> str:
    if task_type == "self_where":
        return "Self-Where"
    if task_type == "self_what":
        return "Self-What"
    if task_type == "env_where":
        return "Env-Where"
    if task_type == "env_how":
        return "Env-How"
    return f"{view_definition}-{task_type}"


def _rate(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _resolve_api_settings(
    config: dict[str, Any],
    *,
    override_model: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stage4_cfg = dict(config.get("stage4", {}) or {})
    overrides = dict(overrides or {})

    def _pick_cfg(*keys: str) -> str:
        for key in keys:
            if key in stage4_cfg:
                text = str(stage4_cfg.get(key, "") or "").strip()
                if text:
                    return text
        return ""

    model = str(
        override_model
        or _pick_cfg("default_model", "model")
        or resolve_default_model(config, stage_name="stage4")
        or (list(stage4_cfg.get("models", []) or [None])[0] or "")
    ).strip()
    if not model:
        raise RuntimeError("missing api_key/model for stage4 experiment")

    model_controls = build_model_request_controls(model)
    route_model = str(model_controls["base_model"] or model).strip()
    endpoint = resolve_model_api_endpoint(
        config=config,
        model=route_model,
        stage_name="stage4",
        stage_cfg=stage4_cfg,
        explicit_source=pick_first_text(overrides.get("api_source"), _pick_cfg("api_source")),
        explicit_api_base=pick_first_text(overrides.get("api_base"), _pick_cfg("api_base", "base_url")),
        explicit_api_key=pick_first_text(overrides.get("api_key"), _pick_cfg("api_key")),
    )
    model_controls = build_model_request_controls(model, api_source=str(endpoint.get("api_source", "") or ""))

    api_key = str(endpoint.get("api_key", "") or "").strip()
    api_base = pick_first_text(endpoint.get("api_base"), "https://api.siliconflow.cn/v1")

    if not api_key:
        for fallback_stage in ("stage3", "stage4"):
            fallback_model = str(resolve_default_model(config, stage_name=fallback_stage) or "").strip()
            if not fallback_model:
                continue
            fallback_ep = resolve_model_api_endpoint(
                config=config,
                model=fallback_model,
                stage_name=fallback_stage,
                stage_cfg=stage4_cfg,
                explicit_source=pick_first_text(overrides.get("api_source"), _pick_cfg("api_source")),
                explicit_api_base=pick_first_text(overrides.get("api_base"), _pick_cfg("api_base", "base_url")),
                explicit_api_key=pick_first_text(overrides.get("api_key"), _pick_cfg("api_key")),
            )
            fallback_key = str(fallback_ep.get("api_key", "") or "").strip()
            fallback_base = str(fallback_ep.get("api_base", "") or "").strip()
            if fallback_key:
                api_key = fallback_key
                if fallback_base:
                    api_base = fallback_base
                break

        if not api_key:
            try:
                registry = load_api_registry(config)
                for route in dict(registry.get("models", {}) or {}).values():
                    if not isinstance(route, dict):
                        continue
                    key_candidate = str(route.get("api_key", "") or "").strip()
                    base_candidate = str(route.get("api_base", route.get("base_url", "")) or "").strip()
                    if key_candidate:
                        api_key = key_candidate
                        if not api_base and base_candidate:
                            api_base = base_candidate
                        break
            except Exception:
                pass

        if str(api_base).startswith("http://localhost") or str(api_base).startswith("http://127.0.0.1"):
            api_key = "EMPTY"

        if not api_key:
            raise RuntimeError(
                "missing api_key/model for stage4 experiment "
                f"(model={model!r}, route_model={route_model!r}, api_base={api_base!r})"
            )

    request_model = str(endpoint.get("request_model", route_model) or route_model)
    request_model_override = pick_first_text(overrides.get("request_model"))
    if request_model_override:
        request_model = request_model_override

    upload_resize_enabled = bool(overrides.get("upload_resize_enabled", stage4_cfg.get("api_upload_resize_enabled", True)))
    upload_max_width = int(overrides.get("upload_max_width", stage4_cfg.get("api_upload_max_width", 640)) or 640)
    upload_max_height = int(overrides.get("upload_max_height", stage4_cfg.get("api_upload_max_height", 480)) or 480)
    upload_jpeg_quality = int(overrides.get("upload_jpeg_quality", stage4_cfg.get("api_upload_jpeg_quality", 80)) or 80)
    timeout_s = float(overrides.get("timeout_s", stage4_cfg.get("timeout_s", 30.0)) or 30.0)
    temperature = float(overrides.get("temperature", stage4_cfg.get("temperature", 0.2)) or 0.2)
    max_tokens = int(overrides.get("max_tokens", stage4_cfg.get("max_tokens", 500)) or 500)
    request_retry_attempts = int(overrides.get("request_retry_attempts", stage4_cfg.get("request_retry_attempts", 3)) or 3)
    request_retry_backoff_sec = float(overrides.get("request_retry_backoff_sec", stage4_cfg.get("request_retry_backoff_sec", 2.0)) or 2.0)
    request_retry_forever = bool(overrides.get("request_retry_forever", stage4_cfg.get("request_retry_forever", False)))

    requested_concurrency = max(1, int(overrides.get("concurrency", 1) or 1))
    configured_rpm_limit = int(overrides.get("rpm_limit", endpoint.get("rpm_limit", 0)) or 0)
    configured_tpm_limit = int(overrides.get("tpm_limit", endpoint.get("tpm_limit", 0)) or 0)
    reserve_ratio = float(overrides.get("rate_limit_reserve_ratio", endpoint.get("rate_limit_reserve_ratio", 0.1)) or 0.1)
    estimated_tokens = int(overrides.get("estimated_tokens_per_request", endpoint.get("estimated_tokens_per_request", 0)) or 0)
    if estimated_tokens <= 0:
        estimated_tokens = max(1200, int(max_tokens) + 1000)

    rate_limit_cfg = compute_rate_limited_concurrency(
        requested_concurrency,
        rpm_limit=configured_rpm_limit,
        tpm_limit=configured_tpm_limit,
        estimated_tokens_per_request=estimated_tokens,
        reserve_ratio=reserve_ratio,
    )

    return {
        "api_key": api_key,
        "api_base": api_base,
        "api_source": str(endpoint.get("api_source", "") or ""),
        "model": str(model_controls["display_model"]),
        "base_model": route_model,
        "reasoning_mode": str(model_controls["mode"]),
        "assistant_prefill": str(model_controls.get("assistant_prefill", "") or ""),
        "system_prompt_prefix": str(model_controls.get("system_prompt_prefix", "") or ""),
        "system_prompt_as_blocks": bool(model_controls.get("system_prompt_as_blocks", False)),
        "request_model": request_model,
        "timeout_s": timeout_s,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "request_retry_attempts": request_retry_attempts,
        "request_retry_backoff_sec": request_retry_backoff_sec,
        "request_retry_forever": request_retry_forever,
        "upload_resize_enabled": upload_resize_enabled,
        "upload_max_width": upload_max_width,
        "upload_max_height": upload_max_height,
        "upload_jpeg_quality": upload_jpeg_quality,
        "prefix": str(stage4_cfg.get("prefix", "") or "").strip(),
        "request_extra_body": dict(model_controls.get("extra_body", {}) or {}),
        "configured_rpm_limit": int(rate_limit_cfg["configured_rpm_limit"]),
        "configured_tpm_limit": int(rate_limit_cfg["configured_tpm_limit"]),
        "rpm_limit": int(rate_limit_cfg["effective_rpm_limit"]),
        "tpm_limit": int(rate_limit_cfg["effective_tpm_limit"]),
        "requested_concurrency": int(rate_limit_cfg["requested_concurrency"]),
        "concurrency": int(rate_limit_cfg["effective_concurrency"]),
        "estimated_tokens_per_request": int(rate_limit_cfg["estimated_tokens_per_request"]),
        "rate_limit_reserve_ratio": float(rate_limit_cfg["reserve_ratio"]),
        "rate_limit_concurrency_applied": bool(rate_limit_cfg["rate_limit_concurrency_applied"]),
    }


def _prepare_upload_image(path: Path, api_cfg: dict[str, Any]) -> tuple[bytes, str]:
    if Image is None:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return path.read_bytes(), mime

    with Image.open(path) as img:
        canvas = img.convert("RGB")
        if api_cfg.get("upload_resize_enabled", True):
            max_w = max(64, int(api_cfg.get("upload_max_width", 640) or 640))
            max_h = max(64, int(api_cfg.get("upload_max_height", 480) or 480))
            scale = min(max_w / max(1, canvas.width), max_h / max(1, canvas.height), 1.0)
            if scale < 1.0:
                canvas = canvas.resize(
                    (max(1, int(round(canvas.width * scale))), max(1, int(round(canvas.height * scale)))),
                    Image.Resampling.LANCZOS,
                )
        buff = io.BytesIO()
        canvas.save(buff, format="JPEG", quality=max(60, min(95, int(api_cfg.get("upload_jpeg_quality", 85) or 85))))
        return buff.getvalue(), "image/jpeg"


def _as_abs_workspace_path(raw_path: str) -> Path:
    p = Path(str(raw_path or "").strip())
    if p.is_absolute():
        return p.resolve()
    return (WORKSPACE_ROOT / p).resolve()


def _sanitize_open_user_prompt_text(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    # Remove explicit multi-choice option blocks if legacy prompts leak into open QA.
    out = re.sub(
        r"\n\s*Options:\s*(?:\n\s*[A-D][\.\)]\s*[^\n]*)+",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\bSelect one option\b", "Answer with one or a few words", out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _sanitize_open_system_prompt_text(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    out = re.sub(r"^\s*.*option_id.*$\n?", "", out, flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"^\s*.*answer_option_id.*$\n?", "", out, flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"^\s*.*Use exactly one option.*$\n?", "", out, flags=re.IGNORECASE | re.MULTILINE)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _build_openai_messages(sample: dict[str, Any], api_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    task_type = str(sample.get("task_type", "") or "")
    image_paths: list[Path] = []
    if task_type in {"self_where", "env_where", "env_how", "label_multiple_choice"}:
        image_paths = [
            _as_abs_workspace_path(str(sample.get("reference_image_with_bbox", "") or sample.get("reference_image", ""))),
            _as_abs_workspace_path(str(sample.get("target_image", "") or "")),
        ]
    elif task_type == "self_what":
        image_paths = [_as_abs_workspace_path(str(sample.get("reference_image_with_bbox", "") or sample.get("reference_image", "")))]
        for cand in list(sample.get("candidates", []) or []):
            if isinstance(cand, dict) and str(cand.get("image", "") or "").strip():
                image_paths.append(_as_abs_workspace_path(str(cand.get("image", ""))))
    else:
        image_paths = [_as_abs_workspace_path(str(sample.get("reference_image_with_bbox", "") or sample.get("reference_image", "")))]
        for cand in list(sample.get("candidates", []) or []):
            if isinstance(cand, dict) and str(cand.get("image", "") or "").strip():
                image_paths.append(_as_abs_workspace_path(str(cand.get("image", ""))))

    # For open-QA evaluation, always prefer converted prompt_text over legacy user_prompt.
    user_prompt_text = str(sample.get("prompt_text", "") or sample.get("user_prompt", "") or "").strip()
    user_prompt_text = _sanitize_open_user_prompt_text(user_prompt_text)
    blocks: list[dict[str, Any]] = [{"type": "text", "text": user_prompt_text}]
    for image_path in image_paths:
        image_bytes, mime_type = _prepare_upload_image(image_path, api_cfg)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})

    system_prompt = str(sample.get("system_prompt", "") or "").strip()
    system_prompt = _sanitize_open_system_prompt_text(system_prompt)
    system_prefix = str(api_cfg.get("system_prompt_prefix", "") or "").strip()
    if system_prefix:
        system_prompt = f"{system_prefix}\n\nThen follow the task-specific instruction below.\n\n{system_prompt}".strip()

    route_model_name = str(api_cfg.get("base_model", "") or api_cfg.get("model", "") or "")
    if should_inline_system_prompt_for_multimodal(route_model_name):
        blocks[0] = {"type": "text", "text": f"{system_prompt}\n\n{user_prompt_text}".strip()}
        messages = [{"role": "user", "content": blocks}]
    else:
        system_content: Any = system_prompt
        if bool(api_cfg.get("system_prompt_as_blocks", False)):
            system_content = [{"type": "text", "text": system_prompt}]
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": blocks},
        ]

    assistant_prefill = str(api_cfg.get("assistant_prefill", "") or "").strip()
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})

    return messages, [_path_for_json(path) for path in image_paths]


def _messages_for_log(messages: list[dict[str, Any]], image_paths: list[str]) -> list[dict[str, Any]]:
    """Return a log-friendly copy: keep text, replace image data URLs with path hints."""
    image_idx = 0
    out: list[dict[str, Any]] = []
    for message in list(messages or []):
        role = str(message.get("role", "") or "")
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": content})
            continue
        blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "") or "")
            if block_type == "text":
                blocks.append({"type": "text", "text": str(block.get("text", "") or "")})
                continue
            if block_type == "image_url":
                path_hint = image_paths[image_idx] if image_idx < len(image_paths) else "<unknown-image>"
                image_idx += 1
                blocks.append({"type": "image_url", "image_url": {"url": f"<image:{path_hint}>"}})
                continue
            blocks.append({"type": block_type})
        out.append({"role": role, "content": blocks})
    return out


def _collect_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    chunks: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                for key in ("text", "output_text", "content"):
                    if key in item:
                        chunks.append(_collect_text(item.get(key)))
                continue
            for attr in ("text", "content", "output_text"):
                nested = getattr(item, attr, None)
                if nested is not None:
                    chunks.append(_collect_text(nested))
        return "\n".join([str(x).strip() for x in chunks if str(x).strip()])
    if isinstance(value, dict):
        for key in ("text", "output_text", "content"):
            if key in value:
                chunks.append(_collect_text(value.get(key)))
        return "\n".join([str(x).strip() for x in chunks if str(x).strip()])
    for attr in ("output_text", "text", "content"):
        nested = getattr(value, attr, None)
        if nested is not None:
            chunks.append(_collect_text(nested))
    return "\n".join([str(x).strip() for x in chunks if str(x).strip()])


def _get_obj_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _extract_response_text(resp: Any) -> tuple[str, dict[str, Any]]:
    raw_text = ""
    meta: dict[str, Any] = {}
    try:
        top_output = _collect_text(_get_obj_field(resp, "output_text"))
        if top_output:
            raw_text = top_output
            meta["response_api_output_text"] = True

        if not raw_text:
            output_items = _get_obj_field(resp, "output")
            if output_items is not None:
                out_text = _collect_text(output_items)
                if out_text:
                    raw_text = out_text
                    meta["response_api_output"] = True

        choices = _get_obj_field(resp, "choices")
        if choices:
            choice0 = choices[0]
            msg = _get_obj_field(choice0, "message")
            if msg is not None and not raw_text:
                raw_text = _collect_text(_get_obj_field(msg, "content"))
            if msg is not None and not raw_text:
                raw_text = _collect_text(_get_obj_field(msg, "reasoning"))
            if not raw_text:
                raw_text = _collect_text(_get_obj_field(choice0, "text"))
            meta["finish_reason"] = _get_obj_field(choice0, "finish_reason")

        if not raw_text:
            raw_text = _collect_text(_get_obj_field(resp, "content"))

        meta["id"] = _get_obj_field(resp, "id")
        meta["model"] = _get_obj_field(resp, "model")
        usage = _get_obj_field(resp, "usage")
        if usage is not None:
            if isinstance(usage, dict):
                meta["usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                    "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                    "total_tokens": usage.get("total_tokens"),
                }
            else:
                meta["usage"] = {
                    "prompt_tokens": _get_obj_field(usage, "prompt_tokens") if _get_obj_field(usage, "prompt_tokens") is not None else _get_obj_field(usage, "input_tokens"),
                    "completion_tokens": _get_obj_field(usage, "completion_tokens") if _get_obj_field(usage, "completion_tokens") is not None else _get_obj_field(usage, "output_tokens"),
                    "total_tokens": _get_obj_field(usage, "total_tokens"),
                }
    except Exception:
        pass
    return str(raw_text or "").strip(), meta


def _extract_json_objects(raw_text: str) -> list[dict[str, Any]]:
    text = str(raw_text or "")
    if not text:
        return []
    out: list[dict[str, Any]] = []
    stack = 0
    start = -1
    for idx, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = idx
            stack += 1
        elif ch == "}":
            if stack <= 0:
                continue
            stack -= 1
            if stack == 0 and start >= 0:
                candidate = text[start : idx + 1]
                try:
                    payload = json.loads(candidate)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    out.append(payload)
    return out


def _strip_code_fences(text: str) -> str:
    out = str(text or "").strip()
    if not out:
        return out
    out = re.sub(r"^```(?:json|JSON)?\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*```$", "", out)
    return out.strip()


def _parse_loose_object(candidate: str) -> dict[str, Any] | None:
    text = _strip_code_fences(candidate)
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    try:
        payload = ast.literal_eval(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _extract_first_object_from_text(raw_text: str) -> dict[str, Any] | None:
    for payload in _extract_json_objects(raw_text):
        if isinstance(payload, dict):
            return payload
    text = _strip_code_fences(raw_text)
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return _parse_loose_object(text[start : end + 1])
    return _parse_loose_object(text)


def _coerce_bbox_from_value(value: Any) -> list[float] | None:
    if value is None:
        return None

    if isinstance(value, dict):
        keys = {str(key).lower(): key for key in value.keys()}
        aliases = [
            ("x1", "y1", "x2", "y2"),
            ("left", "top", "right", "bottom"),
            ("xmin", "ymin", "xmax", "ymax"),
        ]
        for alias in aliases:
            if all(name in keys for name in alias):
                try:
                    coords = [float(value[keys[name]]) for name in alias]
                except Exception:
                    return None
                return _coerce_bbox_from_value(coords)
        for key in (
            "bbox_xyxy_norm",
            "bbox_xyxy",
            "bbox_norm",
            "bbox",
            "box",
            "pred_bbox_xyxy_norm",
            "answer_bbox_xyxy_norm",
        ):
            if key in keys:
                return _coerce_bbox_from_value(value[keys[key]])
        return None

    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) == 1 and isinstance(items[0], (list, tuple, dict, str)):
            nested = _coerce_bbox_from_value(items[0])
            if nested is not None:
                return nested
        if len(items) == 2 and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in items):
            try:
                x0, y0 = float(items[0][0]), float(items[0][1])
                x1, y1 = float(items[1][0]), float(items[1][1])
            except Exception:
                return None
            return _coerce_bbox_from_value([x0, y0, x1, y1])
        if len(items) == 4:
            try:
                coords = [float(v) for v in items]
            except Exception:
                return None
            x0, y0, x1, y1 = coords
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            return [
                max(0.0, min(1.0, x0)),
                max(0.0, min(1.0, y0)),
                max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y1)),
            ]
        return None

    if isinstance(value, str):
        text = _strip_code_fences(value)
        if not text:
            return None
        try:
            payload = json.loads(text)
            return _coerce_bbox_from_value(payload)
        except Exception:
            pass
        try:
            payload = ast.literal_eval(text)
            return _coerce_bbox_from_value(payload)
        except Exception:
            return None

    return None


def _parse_answer_text(raw_text: str) -> tuple[str | None, bool]:
    for payload in _extract_json_objects(raw_text):
        for key in ("answer_text", "answer", "final_answer", "prediction"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), True
            if value is not None and not isinstance(value, (dict, list)):
                text = str(value).strip()
                if text:
                    return text, True
        option_id = str(payload.get("answer_option_id", "") or "").strip().upper()
        if option_id in OPTION_TO_IMAGE:
            return option_id, True

    obj = _extract_first_object_from_text(raw_text)
    if obj is not None:
        for key in ("answer_text", "answer", "final_answer", "prediction"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), True
            if value is not None and not isinstance(value, (dict, list)):
                text = str(value).strip()
                if text:
                    return text, True
        option_id = str(obj.get("answer_option_id", "") or obj.get("option_id", "") or "").strip().upper()
        if option_id in OPTION_TO_IMAGE:
            return option_id, True

    line = _strip_code_fences(str(raw_text or "").strip())
    if not line:
        return None, False

    match = re.search(r"(?:answer_text|answer|final_answer|prediction)\s*[:=：]\s*([^\n,}]+)", line, flags=re.IGNORECASE)
    if match:
        return str(match.group(1)).strip().strip("\"'"), True

    match = re.search(r"(?:option_id|answer_option_id)\s*[:=：]\s*([A-D])\b", line, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper(), True

    first_line = next((seg.strip() for seg in line.splitlines() if seg.strip()), "")
    if first_line:
        return first_line.strip("\"'"), True
    return None, False


def _parse_bbox(raw_text: str) -> tuple[list[float] | None, bool]:
    for payload in _extract_json_objects(raw_text):
        bbox = _coerce_bbox_from_value(
            payload.get("bbox_xyxy_norm")
            or payload.get("bbox_xyxy")
            or payload.get("bbox_norm")
            or payload.get("bbox")
            or payload.get("box")
            or payload.get("pred_bbox_xyxy_norm")
            or payload.get("answer_bbox_xyxy_norm")
        )
        if bbox is not None:
            return bbox, True

    obj = _extract_first_object_from_text(raw_text)
    if obj is not None:
        bbox = _coerce_bbox_from_value(obj)
        if bbox is not None:
            return bbox, True

    match = re.search(
        r"(?:BBox|bbox|bbox_xyxy_norm|bbox_xyxy|bbox_norm|box)\s*[:=：]\s*[\[\(]?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
        raw_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, False
    try:
        coords = [max(0.0, min(1.0, float(match.group(i)))) for i in range(1, 5)]
    except Exception:
        return None, False
    x0, y0, x1, y1 = coords
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1], True


def _bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0.0 else 0.0


def _is_retryable_request_failure(error_text: str) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return True
    retry_markers = [
        "timeout",
        "timed out",
        "empty_model_response",
        "too many requests",
        "rate limit",
        "limitation",
        "429",
        "connection error",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporarily unavailable",
        "server disconnected",
        "read timeout",
        "remote protocol error",
        "502",
        "503",
        "504",
    ]
    fatal_markers = [
        "unsupported_model_mode_suffix",
        "model_not_found",
        "invalid_request_error",
        "badrequesterror",
        "authentication",
        "permission",
        "context_length",
        "does not exist",
        "not support",
        "unknown field",
        "unrecognized",
    ]
    if any(marker in text for marker in fatal_markers):
        return False
    return any(marker in text for marker in retry_markers)


def _is_rate_limited_request_failure(error_text: str) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in ["429", "too many requests", "rate limit", "limitation"])


def _chat_completion_with_retry(
    *,
    client: Any,
    api_cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, dict[str, Any], float]:
    raw_text = ""
    raw_meta: dict[str, Any] = {}
    retries = max(1, int(api_cfg.get("request_retry_attempts", 1) or 1))
    retry_backoff = max(0.0, float(api_cfg.get("request_retry_backoff_sec", 0.0) or 0.0))
    retry_forever = bool(api_cfg.get("request_retry_forever", False))
    attempt = 0
    latency_ms = 0.0
    errors: list[str] = []

    while True:
        started = time.perf_counter()
        should_retry = False
        try:
            kwargs: dict[str, Any] = {
                "model": api_cfg["request_model"],
                "messages": messages,
                "timeout": float(api_cfg.get("timeout_s", 30.0) or 30.0),
            }
            if not model_disallows_temperature(str(api_cfg.get("request_model", "") or "")):
                kwargs["temperature"] = float(api_cfg.get("temperature", 0.0) if temperature is None else temperature)
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            extra_body = dict(api_cfg.get("request_extra_body", {}) or {})
            if api_cfg.get("prefix"):
                extra_body["prefix"] = api_cfg["prefix"]
            if extra_body:
                kwargs["extra_body"] = extra_body

            resp = client.chat.completions.create(**kwargs)
            latency_ms += (time.perf_counter() - started) * 1000.0
            raw_text, raw_meta = _extract_response_text(resp)
            if str(raw_text or "").strip():
                raw_meta = dict(raw_meta or {})
                raw_meta["request_retry_attempts"] = int(attempt)
                raw_meta["request_retry_errors"] = list(errors)
                return raw_text, raw_meta, latency_ms

            err = "empty_model_response"
            errors.append(err)
            raw_meta = {**dict(raw_meta or {}), "error": err}
            should_retry = _is_retryable_request_failure(err)
        except Exception as exc:
            latency_ms += (time.perf_counter() - started) * 1000.0
            err = str(exc)
            errors.append(err)
            raw_meta = {"error": err}
            should_retry = _is_retryable_request_failure(err)

        attempt += 1
        if not should_retry:
            raise RuntimeError(str((raw_meta or {}).get("error", "non_retryable_request_error")))
        if retry_forever or attempt < retries:
            err = str((raw_meta or {}).get("error", "") or "")
            base = max(0.25, retry_backoff if retry_backoff > 0 else 0.5)
            if _is_rate_limited_request_failure(err):
                base = max(base, 1.0)
            sleep_sec = min(60.0, base * (2 ** max(0, attempt - 1)))
            sleep_sec = sleep_sec * (1.0 + random.uniform(0.0, 0.2))
            time.sleep(sleep_sec)
            continue
        raise RuntimeError(str((raw_meta or {}).get("error", "request_failed")))


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_ok = sum(1 for row in rows if bool(row.get("parse_ok", False)))
    answer_parse_ok = sum(1 for row in rows if bool(row.get("answer_parse_ok", False)))
    answer_correct = sum(1 for row in rows if bool(row.get("answer_correct", False)))

    rule_rows = [row for row in rows if row.get("answer_rule_correct") is not None]
    rule_correct = sum(1 for row in rule_rows if bool(row.get("answer_rule_correct", False)))

    judge_rows = [row for row in rows if bool(row.get("answer_judge_used", False))]
    judge_correct = sum(1 for row in judge_rows if bool(row.get("answer_judge_correct", False)))

    bbox_rows = [row for row in rows if row.get("gold_bbox_xyxy_norm") is not None]
    bbox_acc = sum(1 for row in bbox_rows if bool(row.get("bbox_acc@50iou", False)))
    bbox_ious = [float(row.get("bbox_iou", 0.0) or 0.0) for row in bbox_rows]

    latency_values = [float(row.get("latency_ms", 0.0) or 0.0) for row in rows if row.get("latency_ms") is not None]
    return {
        "count": total,
        "parse_success_rate": _rate(parse_ok, total),
        "answer_parse_success_rate": _rate(answer_parse_ok, total),
        "answer_accuracy": _rate(answer_correct, total),
        "answer_rule_coverage": _rate(len(rule_rows), total),
        "answer_accuracy_rule_only": _rate(rule_correct, len(rule_rows)),
        "answer_judge_coverage": _rate(len(judge_rows), total),
        "answer_accuracy_judge_only": _rate(judge_correct, len(judge_rows)),
        "bbox_acc@50iou": _rate(bbox_acc, len(bbox_rows)),
        "bbox_mean_iou": (sum(bbox_ious) / len(bbox_ious)) if bbox_ious else 0.0,
        "avg_latency_ms": (sum(latency_values) / len(latency_values)) if latency_values else None,
    }


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for key in ("view_definition", "task_type", "difficulty"):
        bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bucket[str(row.get(key, ""))].append(row)
        grouped[key] = dict((name, _aggregate_metric_rows(items)) for name, items in bucket.items())

    combo_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    combo_diff_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        view_definition = str(row.get("view_definition", "") or "")
        task_type = str(row.get("task_type", "") or "")
        difficulty = str(row.get("difficulty", "") or "")
        combo_bucket[f"{view_definition}|{task_type}"].append(row)
        combo_diff_bucket[_task_combo_id(view_definition, task_type, difficulty)].append(row)

    grouped["combo"] = dict((name, _aggregate_metric_rows(items)) for name, items in combo_bucket.items())
    grouped["combo_by_difficulty"] = dict((name, _aggregate_metric_rows(items)) for name, items in combo_diff_bucket.items())

    out = _aggregate_metric_rows(rows)
    out["grouped"] = grouped
    return out


def _judge_answer_with_model(
    *,
    task_type: str,
    gold_answer_text: str,
    pred_answer_text: str,
    judge_client: Any,
    judge_cfg: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    cache_key = json.dumps(
        {"task_type": str(task_type), "gold": str(gold_answer_text), "pred": str(pred_answer_text)},
        ensure_ascii=False,
        sort_keys=True,
    )
    if cache_key in cache:
        return dict(cache[cache_key])

    system_prompt = (
        "You are a strict answer evaluator for UAV visual QA. "
        "Decide whether predicted answer semantically matches gold answer. "
        "Return exactly one JSON object: "
        '{"is_correct":true,"gold_normalized":"...","pred_normalized":"...","reason":"..."}'
    )
    user_prompt = (
        f"task_type: {task_type}\n"
        f"gold_answer_text: {gold_answer_text}\n"
        f"pred_answer_text: {pred_answer_text}\n\n"
        "Rules:\n"
        "1. For self_what, treat Image 2/3/4/5 and A/B/C/D mapping as equivalent.\n"
        "2. For directional tasks, allow lexical variants (behind=Back, forward=Front).\n"
        "3. If answer is ambiguous or missing, return false."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    raw_text, raw_meta, latency_ms = _chat_completion_with_retry(
        client=judge_client,
        api_cfg=judge_cfg,
        messages=messages,
        max_tokens=160,
        temperature=0.0,
    )

    parsed_ok = False
    is_correct = False
    for obj in _extract_json_objects(raw_text):
        if "is_correct" in obj:
            parsed_ok = True
            val = obj.get("is_correct")
            if isinstance(val, bool):
                is_correct = bool(val)
            else:
                is_correct = str(val).strip().lower() in {"true", "1", "yes"}
            break

    result = {
        "judge_parsed_ok": bool(parsed_ok),
        "judge_is_correct": bool(is_correct),
        "judge_raw_text": raw_text,
        "judge_raw_response": raw_meta,
        "judge_latency_ms": float(latency_ms),
    }
    cache[cache_key] = dict(result)
    return result


def _resolve_manifest_path(args: argparse.Namespace) -> Path:
    if str(args.manifest_path or "").strip():
        return Path(str(args.manifest_path)).resolve()
    if not str(args.scene_id or "").strip():
        raise RuntimeError("missing --scene-id when --manifest-path is not provided")

    engine = str(args.engine or "airsim").strip().lower()
    scene_id = str(args.scene_id).strip()
    task_name = str(args.task_name or "UAV-DualCog-V1-Open").strip()

    root = Path(str(args.task_pipeline_root or "task_pipeline_data"))
    if not root.is_absolute():
        root = (WORKSPACE_ROOT / root).resolve()

    manifest_dir = root / task_name / f"{engine}_{scene_id}" / "qa" / "manifests"
    prefer_open = manifest_dir / f"{scene_id}.latest_manifest.open.json"
    fallback = manifest_dir / f"{scene_id}.latest_manifest.json"
    if prefer_open.exists():
        return prefer_open
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"manifest_not_found: {prefer_open} / {fallback}")


def _stage4_root_from_manifest(manifest_path: Path) -> Path:
    path = manifest_path.resolve()
    if path.parent.name == "manifests":
        return path.parent.parent
    return path.parent


def run_open_qa_experiment(
    *,
    config: dict[str, Any],
    manifest_path: Path,
    override_model: str | None,
    judge_model: str | None,
    enable_judge: bool,
    limit: int | None,
    api_overrides: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    if OpenAI is None:
        raise ImportError("openai package is required for open QA experiment")

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    scene_id = str(manifest.get("scene_id", "") or "")
    engine = str(manifest.get("engine", "airsim") or "airsim").strip().lower()
    samples = _filter_samples(list(manifest.get("samples", []) or []), limit=limit)
    if not samples:
        raise RuntimeError("no_samples_for_open_qa_experiment")

    api_cfg = _resolve_api_settings(config, override_model=override_model, overrides=api_overrides)
    eval_client = _build_openai_client(api_key=api_cfg["api_key"], api_base=api_cfg["api_base"])

    judge_cfg: dict[str, Any] | None = None
    judge_client: Any | None = None
    if enable_judge:
        judge_cfg = _resolve_api_settings(
            config,
            override_model=str(judge_model or "Qwen/Qwen3.5-9B"),
            overrides={
                **dict(api_overrides or {}),
                "temperature": 0.0,
                "max_tokens": 160,
            },
        )
        judge_client = _build_openai_client(api_key=judge_cfg["api_key"], api_base=judge_cfg["api_base"])

    stage4_root = _stage4_root_from_manifest(Path(manifest_path))
    experiments_root = (stage4_root / "experiments_open").resolve()
    experiments_root.mkdir(parents=True, exist_ok=True)

    manifest_tag = _safe_name(Path(manifest_path).stem)[:48]
    model_tag = _safe_name(str(api_cfg["model"]))
    unique_experiment = _as_bool((api_overrides or {}).get("unique_experiment", config.get("unique_experiment", False)), False)
    run_id = f"{scene_id}_{model_tag}_{manifest_tag}_unique" if unique_experiment else f"{scene_id}_{model_tag}_{manifest_tag}_{_now_ts()}"
    run_dir = (experiments_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    requests_path = run_dir / "requests.jsonl"
    responses_path = run_dir / "responses.jsonl"
    parsed_path = run_dir / "parsed_predictions.jsonl"
    requests_txt_path = run_dir / "requests.txt"
    responses_txt_path = run_dir / "responses.txt"
    judge_txt_path = run_dir / "judge.txt"
    failed_indices_path = run_dir / "failed_request_indices.json"
    judge_cache: dict[str, dict[str, Any]] = {}
    parsed_rows: list[dict[str, Any]] = []
    completed_sample_ids: set[str] = set()
    sample_meta_by_id: dict[str, dict[str, Any]] = {}
    for sample_index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id", "") or "").strip()
        if not sample_id:
            continue
        sample_meta_by_id[sample_id] = {
            "sample_index": int(sample_index),
            "sample_ordinal": int(sample_index + 1),
            "sample_id": sample_id,
            "task_type": sample.get("task_type", ""),
            "difficulty": sample.get("difficulty", ""),
            "view_definition": sample.get("view_definition", ""),
        }
    if unique_experiment and responses_path.exists():
        latest_status_by_sample: dict[str, str] = {}
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "") or "").strip()
            if sample_id:
                latest_status_by_sample[sample_id] = str(row.get("request_status", "") or "").strip().lower()
        for sample_id, status in latest_status_by_sample.items():
            if status == "ok":
                completed_sample_ids.add(sample_id)
    if unique_experiment and parsed_path.exists():
        for line in parsed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                parsed_rows.append(row)

    pending_samples = list(samples)
    if unique_experiment:
        pending_samples = [sample for sample in pending_samples if str(sample.get("sample_id", "") or "").strip() not in completed_sample_ids]

    file_mode = "a" if unique_experiment else "w"
    with requests_path.open(file_mode, encoding="utf-8") as req_fp, responses_path.open(file_mode, encoding="utf-8") as resp_fp, parsed_path.open(file_mode, encoding="utf-8") as pred_fp, requests_txt_path.open(file_mode, encoding="utf-8") as req_txt_fp, responses_txt_path.open(file_mode, encoding="utf-8") as resp_txt_fp, judge_txt_path.open(file_mode, encoding="utf-8") as judge_txt_fp:
        total_samples = len(pending_samples)
        for index, sample in enumerate(pending_samples, start=1):
            sample_id = str(sample.get("sample_id", "") or "").strip()
            sample_meta = dict(sample_meta_by_id.get(sample_id, {}) or {})
            print(
                f"[{_iso_now()}] scene={scene_id} progress={index}/{total_samples} "
                f"sample_id={sample_id} model={api_cfg['model']}",
                flush=True,
            )
            messages, upload_images = _build_openai_messages(sample, api_cfg)
            request_row = {
                "run_id": run_id,
                "sample_index": sample_meta.get("sample_index"),
                "sample_ordinal": sample_meta.get("sample_ordinal"),
                "sample_id": sample.get("sample_id"),
                "model": api_cfg["model"],
                "request_model": api_cfg["request_model"],
                "api_source": api_cfg.get("api_source", ""),
                "api_base": api_cfg.get("api_base", ""),
                "reasoning_mode": api_cfg.get("reasoning_mode", ""),
                "task_type": sample.get("task_type"),
                "difficulty": sample.get("difficulty"),
                "view_definition": sample.get("view_definition"),
                "assistant_prefill": api_cfg.get("assistant_prefill", ""),
                "system_prompt_prefix": api_cfg.get("system_prompt_prefix", ""),
                "system_prompt_as_blocks": bool(api_cfg.get("system_prompt_as_blocks", False)),
                "request_extra_body": api_cfg.get("request_extra_body", {}),
                "expected_answer_text": sample.get("answer_text"),
                "expected_bbox_xyxy_norm": sample.get("answer_bbox_xyxy_norm"),
                "messages_preview": _messages_for_log(messages, upload_images),
                "images": upload_images,
            }

            request_status = "ok"
            raw_text = ""
            raw_meta: dict[str, Any] = {}
            latency_ms: float | None = None
            try:
                raw_text, raw_meta, latency_ms_val = _chat_completion_with_retry(
                    client=eval_client,
                    api_cfg=api_cfg,
                    messages=messages,
                    max_tokens=int(api_cfg.get("max_tokens", 500) or 500),
                    temperature=float(api_cfg.get("temperature", 0.2) or 0.2),
                )
                latency_ms = float(latency_ms_val)
            except Exception as exc:
                request_status = "error"
                raw_text = str(exc)
                raw_meta = {"error": str(exc)}
                latency_ms = None

            response_row = {
                "run_id": run_id,
                "sample_index": sample_meta.get("sample_index"),
                "sample_ordinal": sample_meta.get("sample_ordinal"),
                "sample_id": sample.get("sample_id"),
                "request_status": request_status,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "raw_response": raw_meta,
                "usage": dict(raw_meta.get("usage", {}) or {}) if isinstance(raw_meta.get("usage", {}), dict) else {},
            }

            pred_answer_text, answer_parse_ok = _parse_answer_text(raw_text)
            pred_bbox, bbox_parse_ok = _parse_bbox(raw_text)
            gold_bbox = sample.get("answer_bbox_xyxy_norm")
            bbox_iou = _bbox_iou(pred_bbox, list(gold_bbox)) if (pred_bbox is not None and gold_bbox is not None) else 0.0
            bbox_acc = bool(gold_bbox is not None and bbox_iou >= 0.5)

            task_type = str(sample.get("task_type", "") or "")
            gold_answer_text = str(sample.get("answer_text", "") or "")
            gold_norm = _normalize_answer(task_type, gold_answer_text)
            pred_norm = _normalize_answer(task_type, str(pred_answer_text or ""))

            rule_correct: bool | None = None
            if gold_norm and pred_norm:
                rule_correct = bool(gold_norm == pred_norm)

            answer_correct = False
            judge_used = False
            judge_correct: bool | None = None
            judge_meta: dict[str, Any] | None = None
            score_source = "none"

            if rule_correct is not None:
                answer_correct = bool(rule_correct)
                score_source = "rule"
            elif enable_judge and judge_client is not None and judge_cfg is not None and pred_answer_text:
                judge_used = True
                try:
                    judge_meta = _judge_answer_with_model(
                        task_type=task_type,
                        gold_answer_text=gold_answer_text,
                        pred_answer_text=str(pred_answer_text),
                        judge_client=judge_client,
                        judge_cfg=judge_cfg,
                        cache=judge_cache,
                    )
                    judge_correct = bool(judge_meta.get("judge_is_correct", False))
                    answer_correct = bool(judge_correct)
                    score_source = "judge"
                except Exception as exc:
                    judge_meta = {
                        "judge_error": str(exc),
                        "judge_is_correct": False,
                        "judge_parsed_ok": False,
                    }
                    judge_correct = None
                    answer_correct = False
                    score_source = "judge_error"

            parsed_row = {
                "run_id": run_id,
                "sample_index": sample_meta.get("sample_index"),
                "sample_ordinal": sample_meta.get("sample_ordinal"),
                "sample_id": sample.get("sample_id"),
                "scene_id": sample.get("scene_id", scene_id),
                "engine": sample.get("engine", engine),
                "view_definition": sample.get("view_definition"),
                "task_type": task_type,
                "difficulty": sample.get("difficulty"),
                "gold_answer_text": gold_answer_text,
                "gold_answer_norm": gold_norm,
                "pred_answer_text": pred_answer_text,
                "pred_answer_norm": pred_norm,
                "answer_parse_ok": bool(answer_parse_ok),
                "bbox_parse_ok": bool(bbox_parse_ok),
                "parse_ok": bool(answer_parse_ok and bbox_parse_ok),
                "answer_rule_correct": rule_correct,
                "answer_judge_used": bool(judge_used),
                "answer_judge_correct": judge_correct,
                "answer_correct": bool(answer_correct),
                "answer_score_source": score_source,
                "gold_bbox_xyxy_norm": gold_bbox,
                "pred_bbox_xyxy_norm": pred_bbox,
                "bbox_iou": float(bbox_iou),
                "bbox_acc@50iou": bool(bbox_acc),
                "latency_ms": latency_ms,
                "judge_meta": judge_meta,
            }
            parsed_rows.append(parsed_row)

            req_fp.write(json.dumps(request_row, ensure_ascii=False) + "\n")
            resp_fp.write(json.dumps(response_row, ensure_ascii=False) + "\n")
            pred_fp.write(json.dumps(parsed_row, ensure_ascii=False) + "\n")
            req_fp.flush()
            resp_fp.flush()
            pred_fp.flush()

            usage = dict((response_row.get("raw_response", {}) or {}).get("usage", {}) or {}) if isinstance((response_row.get("raw_response", {}) or {}).get("usage", {}), dict) else {}
            req_txt_fp.write(
                f"[{_iso_now()}] sample_id={sample.get('sample_id')} model={api_cfg['model']} task={sample.get('task_type')} "
                f"difficulty={sample.get('difficulty')} definition={sample.get('view_definition')}\n"
                f"request_model={api_cfg.get('request_model')} api_source={api_cfg.get('api_source','')} api_base={api_cfg.get('api_base','')} "
                f"reasoning_mode={api_cfg.get('reasoning_mode','')}\n"
                f"expected_answer_text={json.dumps(sample.get('answer_text'), ensure_ascii=False)}\n"
                f"expected_bbox_xyxy_norm={json.dumps(sample.get('answer_bbox_xyxy_norm'), ensure_ascii=False)}\n"
                f"messages_preview={json.dumps(request_row.get('messages_preview', []), ensure_ascii=False)}\n"
                f"images={json.dumps(upload_images, ensure_ascii=False)}\n\n"
            )
            resp_txt_fp.write(
                f"[{_iso_now()}] sample_id={sample.get('sample_id')} status={request_status} latency_ms={latency_ms} "
                f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')} total_tokens={usage.get('total_tokens')}\n"
                f"raw_text:\n{raw_text}\n"
                f"raw_response={json.dumps(raw_meta, ensure_ascii=False)}\n\n"
            )
            judge_txt_fp.write(
                f"[{_iso_now()}] sample_id={sample.get('sample_id')} answer_score_source={score_source} "
                f"answer_rule_correct={json.dumps(rule_correct, ensure_ascii=False)} answer_correct={answer_correct}\n"
                f"gold_answer_text={json.dumps(gold_answer_text, ensure_ascii=False)} pred_answer_text={json.dumps(pred_answer_text, ensure_ascii=False)}\n"
                f"gold_answer_norm={json.dumps(gold_norm, ensure_ascii=False)} pred_answer_norm={json.dumps(pred_norm, ensure_ascii=False)}\n"
                f"judge_used={judge_used} judge_correct={json.dumps(judge_correct, ensure_ascii=False)}\n"
                f"judge_meta={json.dumps(judge_meta, ensure_ascii=False)}\n"
                f"bbox_parse_ok={bbox_parse_ok} bbox_iou={bbox_iou:.6f} bbox_acc@50iou={bbox_acc}\n\n"
            )
            req_txt_fp.flush()
            resp_txt_fp.flush()
            judge_txt_fp.flush()

            if callable(progress_callback):
                try:
                    progress_callback(
                        {
                            "run_id": run_id,
                            "model": api_cfg["model"],
                            "completed": len(parsed_rows),
                            "total": total_samples,
                            "sample_id": sample.get("sample_id"),
                            "task_type": sample.get("task_type"),
                            "difficulty": sample.get("difficulty"),
                            "view_definition": sample.get("view_definition"),
                            "request_status": request_status,
                            "parse_ok": bool(answer_parse_ok and bbox_parse_ok),
                            "latency_ms": latency_ms,
                        }
                    )
                except Exception:
                    pass

    latest_response_by_sample: dict[str, dict[str, Any]] = {}
    if responses_path.exists():
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("sample_id", "") or "").strip()
            if sample_id:
                latest_response_by_sample[sample_id] = row
    failed_request_rows_final: list[dict[str, Any]] = []
    for sample_id, response_row in latest_response_by_sample.items():
        if str(response_row.get("request_status", "") or "").strip().lower() != "error":
            continue
        meta = dict(sample_meta_by_id.get(sample_id, {}) or {})
        if not meta:
            meta = {
                "sample_index": 0,
                "sample_ordinal": 1,
                "sample_id": sample_id,
                "task_type": "",
                "difficulty": "",
                "view_definition": "",
            }
        raw_response = response_row.get("raw_response", {})
        raw_response = dict(raw_response) if isinstance(raw_response, dict) else {}
        meta["request_retry_attempts"] = int(raw_response.get("request_retry_attempts", api_cfg.get("request_retry_attempts", 0)) or 0)
        meta["request_retry_errors"] = list(raw_response.get("request_retry_errors", []) or [])
        failed_request_rows_final.append(meta)
    failed_request_rows_final.sort(key=lambda row: int(row.get("sample_index", 0) or 0))

    failed_indices_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "stage": "stage4",
                "scene_id": scene_id,
                "engine": engine,
                "model": api_cfg["model"],
                "manifest_path": _path_for_json(Path(manifest_path)),
                "request_retry_attempts": int(api_cfg.get("request_retry_attempts", 0) or 0),
                "failed_count": len(failed_request_rows_final),
                "failed_sample_indices": [int(row.get("sample_index", 0) or 0) for row in failed_request_rows_final],
                "failed_samples": failed_request_rows_final,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = _summarize_predictions(parsed_rows)
    report = {
        "run_id": run_id,
        "generated_at": _iso_now(),
        "scene_id": scene_id,
        "engine": engine,
        "model": api_cfg["model"],
        "request_model": api_cfg["request_model"],
        "manifest_path": _path_for_json(Path(manifest_path)),
        "mode": "open_fill_blank_eval",
        "judge_model": str(judge_model or "") if enable_judge else None,
        "judge_enabled": bool(enable_judge),
        "judge_cache_size": len(judge_cache),
        "unique_experiment": bool(unique_experiment),
        "resumed_completed_sample_count": len(completed_sample_ids),
        "run_sample_count": len(pending_samples),
        "total_manifest_sample_count": len(samples),
        "summary": summary,
        "api_overrides": dict(api_overrides or {}),
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path = experiments_root / f"{scene_id}.latest_report.json"
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "report_path": report_path,
        "latest_report_path": latest_path,
        "requests_path": requests_path,
        "responses_path": responses_path,
        "parsed_path": parsed_path,
        "failed_indices_path": failed_indices_path,
        "requests_txt_path": requests_txt_path,
        "responses_txt_path": responses_txt_path,
        "judge_txt_path": judge_txt_path,
        "report": report,
    }


def _build_metrics_matrix_from_report(report: dict[str, Any], *, by_difficulty: bool) -> dict[str, Any]:
    columns: list[dict[str, Any]] = []
    for view_definition, task_type in TASK_SPECS:
        if by_difficulty:
            for difficulty in ["4way", "8way"]:
                columns.append(
                    {
                        "combo_id": _task_combo_id(view_definition, task_type, difficulty),
                        "view_definition": view_definition,
                        "task_type": task_type,
                        "difficulty": difficulty,
                        "display_name": _task_display_name(view_definition, task_type),
                        "metrics": ["answer_accuracy", "bbox_acc@50iou", "bbox_mean_iou", "count"],
                    }
                )
        else:
            columns.append(
                {
                    "combo_id": f"{view_definition}|{task_type}",
                    "view_definition": view_definition,
                    "task_type": task_type,
                    "difficulty": "ALL",
                    "display_name": _task_display_name(view_definition, task_type),
                    "metrics": ["answer_accuracy", "bbox_acc@50iou", "bbox_mean_iou", "count"],
                }
            )

    combo_key = "combo_by_difficulty" if by_difficulty else "combo"
    grouped = dict((report.get("summary", {}) or {}).get("grouped", {}) or {})
    payload = dict(grouped.get(combo_key, {}) or {})

    combos: dict[str, Any] = {}
    for column in columns:
        combo_id = str(column["combo_id"])
        data = dict(payload.get(combo_id, {}) or {})
        combos[combo_id] = {
            "count": data.get("count", 0),
            "answer_accuracy": data.get("answer_accuracy", None),
            "bbox_acc@50iou": data.get("bbox_acc@50iou", None),
            "bbox_mean_iou": data.get("bbox_mean_iou", None),
        }
    return {"columns": columns, "rows": [{"model": report.get("model"), "combos": combos}], "by_difficulty": bool(by_difficulty)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage4 open-fill-blank evaluation with answer_text scoring")
    parser.add_argument("--config", required=True, help="task config yaml")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--task-pipeline-root", default="task_pipeline_data")
    parser.add_argument("--task-name", default="UAV-DualCog-V1-Open")
    parser.add_argument("--scene-id", default="")
    parser.add_argument("--engine", default="airsim")
    parser.add_argument("--model", default="", help="evaluated model name")
    parser.add_argument("--judge-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--disable-judge", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1, help="reserved for api overrides compatibility")
    parser.add_argument("--rpm-limit", type=int, default=0)
    parser.add_argument("--tpm-limit", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--request-retry-attempts", type=int, default=3)
    parser.add_argument("--request-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--unique-experiment", dest="unique_experiment", action="store_true", default=None)
    parser.add_argument("--no-unique-experiment", dest="unique_experiment", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_yaml(Path(args.config).resolve())
    manifest_path = _resolve_manifest_path(args)
    overrides = {
        "concurrency": max(1, int(args.concurrency or 1)),
        "rpm_limit": int(args.rpm_limit or 0),
        "tpm_limit": int(args.tpm_limit or 0),
        "timeout_s": float(args.timeout_s or 30.0),
        "request_retry_attempts": max(1, int(args.request_retry_attempts or 1)),
        "request_retry_backoff_sec": max(0.0, float(args.request_retry_backoff_sec or 0.0)),
    }
    if args.unique_experiment is not None:
        overrides["unique_experiment"] = bool(args.unique_experiment)

    result = run_open_qa_experiment(
        config=config,
        manifest_path=manifest_path,
        override_model=str(args.model or "").strip() or None,
        judge_model=str(args.judge_model or "").strip() or None,
        enable_judge=not bool(args.disable_judge),
        limit=int(args.limit) if int(args.limit) > 0 else None,
        api_overrides=overrides,
    )
    report = dict(result["report"])
    metrics_summary = _build_metrics_matrix_from_report(report, by_difficulty=False)

    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "run_dir": _path_for_json(Path(result["run_dir"])),
                "manifest_path": _path_for_json(Path(manifest_path)),
                "report_path": _path_for_json(Path(result["report_path"])),
                "latest_report_path": _path_for_json(Path(result["latest_report_path"])),
                "parsed_path": _path_for_json(Path(result["parsed_path"])),
                "failed_indices_path": _path_for_json(Path(result["failed_indices_path"])),
                "summary": report.get("summary", {}),
                "metrics_matrix": metrics_summary,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
