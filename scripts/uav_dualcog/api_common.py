from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception:
    yaml = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
COMMON_RUNTIME_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "uav_dualcog" / "common_api_runtime.yaml"
_ENV_TOKEN_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_MODEL_MODE_PATTERN = re.compile(r"^(?P<base>.+?)-(?P<mode>Instant|Thinking|Reasoning)$")
INTERNVL_THINKING_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out.get(key, {}), value)
        else:
            out[key] = value
    return out


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def pick_first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def detect_model_family(model_name: str) -> str:
    text = str(model_name or "").strip().lower()
    if "intern-s1" in text:
        return "interns1"
    if "internvl" in text:
        return "internvl"
    # "Pro/" sourced models should follow Qwen-style instant/thinking toggles.
    if text.startswith("pro/") or text.startswith("pro_"):
        return "qwen"
    # Local VLMs that follow Qwen-style chat_template_kwargs enable_thinking switch.
    if "vst-7b-rl" in text or "spacethinker" in text or "spaceom" in text or "spacer" in text or "vst-7b-sft" in text or "livasr" in text or "vilasr" in text:
        return "qwen"
    if "qwen" in text:
        return "qwen"
    if "claude" in text:
        return "claude"
    if "x-ai/" in text or text.startswith("x-ai") or "grok" in text:
        return "xai"
    if "gemini" in text:
        return "gemini"
    if "google/" in text or text.startswith("google") or "google_" in text:
        return "google"
    if "kimi-k2.5" in text or "moonshotai/kimi" in text:
        return "kimi"
    if "glm-4.6v" in text or "z-ai/glm" in text:
        return "glm"
    if "mimo-v2-omni" in text or "xiaomi/mimo" in text:
        return "xiaomi"
    return ""


def should_inline_system_prompt_for_multimodal(model_name: str) -> bool:
    text = str(model_name or "").strip().lower()
    # SenseNova InternVL deployment currently fails on `system` + multimodal list content.
    return "sensenova-si-1.2-internvl3-8b" in text


def required_video_placeholder_for_model(model_name: str) -> str:
    text = str(model_name or "").strip().lower()
    # SenseNova InternVL video serving expects an explicit <video> placeholder
    # in the prompt text to match one video multimodal item.
    if "sensenova-si-1.2-internvl3-8b" in text:
        return "<video>"
    return ""


def max_data_uri_video_bytes_for_model(model_name: str) -> int:
    text = str(model_name or "").strip().lower()
    # DashScope qwen3.5-plus rejects data-uri video items above 10 MiB.
    if "qwen3.5-plus" in text:
        return 9_500_000
    return 0


def parse_experiment_model_name(model_name: str) -> dict[str, str]:
    raw = str(model_name or "").strip()
    if not raw:
        raise RuntimeError("missing experiment model")
    matched = _MODEL_MODE_PATTERN.match(raw)
    if not matched:
        family = detect_model_family(raw)
        return {
            "display_model": raw,
            "base_model": raw,
            "mode": "default",
            "family": family,
            "suffix": "",
        }
    base_model = str(matched.group("base") or "").strip()
    suffix = str(matched.group("mode") or "").strip()
    mode = "thinking" if suffix.lower() in {"thinking", "reasoning"} else "instant"
    family = detect_model_family(base_model)
    if family not in {"qwen", "internvl", "interns1", "claude", "xai", "gemini", "google", "kimi", "glm", "xiaomi", "mimo"}:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    return {
        "display_model": raw,
        "base_model": base_model,
        "mode": mode,
        "family": family,
        "suffix": suffix,
    }


def build_model_request_controls(model_name: str) -> dict[str, Any]:
    parsed = parse_experiment_model_name(model_name)
    extra_body: dict[str, Any] = {}
    assistant_prefill = ""
    system_prompt_prefix = ""
    system_prompt_as_blocks = False
    base_model_text = str(parsed.get("base_model", "") or "").strip().lower()
    is_mimo_omni = "mimo-v2-omni" in base_model_text
    is_qwen36_plus = "qwen3.6-plus" in base_model_text
    is_qwen35_plus = "qwen3.5-plus" in base_model_text
    is_qwen35_flash = "qwen3.5-flash" in base_model_text
    is_kimi_k25 = "kimi-k2.5" in base_model_text
    if parsed["mode"] == "instant":
        if is_qwen36_plus or is_qwen35_plus or is_qwen35_flash:
            extra_body["enable_thinking"] = False
        elif is_kimi_k25:
            extra_body["enable_thinking"] = False
        elif parsed["family"] == "qwen":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif parsed["family"] == "internvl":
            assistant_prefill = "<think>\n</think>\n"
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
        elif parsed["family"] == "interns1":
            extra_body["thinking_mode"] = False
        elif parsed["family"] == "glm":
            extra_body["thinking"] = {"type": "disabled"}
        elif is_mimo_omni:
            extra_body["thinking"] = {"type": "disabled"}
        elif parsed["family"] in {"claude", "xai", "gemini", "google", "kimi", "glm", "xiaomi", "mimo"}:
            extra_body["reasoning"] = {"enabled": False}
    elif parsed["mode"] == "thinking":
        if parsed["family"] == "internvl":
            system_prompt_prefix = INTERNVL_THINKING_SYSTEM_PROMPT
            system_prompt_as_blocks = True
        elif is_qwen36_plus or is_qwen35_plus or is_qwen35_flash:
            extra_body["enable_thinking"] = True
            extra_body["thinking_budget"] = 4000
        elif is_kimi_k25:
            extra_body["enable_thinking"] = True
        elif is_mimo_omni:
            pass
        elif parsed["family"] in {"claude", "xai", "gemini", "google", "kimi", "xiaomi", "mimo"}:
            extra_body["reasoning"] = {"enabled": True}
    return {
        **parsed,
        "assistant_prefill": assistant_prefill,
        "system_prompt_prefix": system_prompt_prefix,
        "system_prompt_as_blocks": system_prompt_as_blocks,
        "extra_body": extra_body,
    }


def compute_rate_limited_concurrency(
    requested_concurrency: int,
    *,
    rpm_limit: int = 0,
    tpm_limit: int = 0,
    estimated_tokens_per_request: int = 0,
    reserve_ratio: float = 0.1,
) -> dict[str, Any]:
    requested = max(1, _safe_int(requested_concurrency, 1))
    reserve = min(max(_safe_float(reserve_ratio, 0.1), 0.0), 0.5)
    effective_rpm = max(0, int(math.floor(max(0, _safe_int(rpm_limit, 0)) * (1.0 - reserve))))
    effective_tpm = max(0, int(math.floor(max(0, _safe_int(tpm_limit, 0)) * (1.0 - reserve))))
    estimated_tokens = max(0, _safe_int(estimated_tokens_per_request, 0))
    caps: list[int] = []
    if effective_rpm > 0:
        caps.append(max(1, effective_rpm // 60))
    if effective_tpm > 0 and estimated_tokens > 0:
        caps.append(max(1, effective_tpm // max(1, estimated_tokens) // 60))
    applied = bool(caps)
    effective_concurrency = max(1, min([requested, *caps])) if caps else requested
    return {
        "requested_concurrency": requested,
        "effective_concurrency": effective_concurrency,
        "configured_rpm_limit": max(0, _safe_int(rpm_limit, 0)),
        "configured_tpm_limit": max(0, _safe_int(tpm_limit, 0)),
        "effective_rpm_limit": effective_rpm,
        "effective_tpm_limit": effective_tpm,
        "estimated_tokens_per_request": estimated_tokens,
        "reserve_ratio": reserve,
        "rate_limit_concurrency_applied": applied,
    }


def resolve_env_token(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    matched = _ENV_TOKEN_PATTERN.match(text)
    if not matched:
        return text
    return str(os.environ.get(matched.group(1), "") or "").strip()


def load_common_runtime_cfg() -> dict[str, Any]:
    if yaml is None or not COMMON_RUNTIME_CONFIG_PATH.exists():
        return {}
    try:
        payload = yaml.safe_load(COMMON_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def merge_common_stage_block(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    common_cfg = _as_mapping(load_common_runtime_cfg().get(key, {}))
    scene_cfg = _as_mapping(config.get(key, {}))
    return _deep_merge_dict(common_cfg, scene_cfg)


def load_api_registry(config: Mapping[str, Any]) -> dict[str, Any]:
    common_api = _as_mapping(load_common_runtime_cfg().get("api", {}))
    scene_api = _as_mapping(config.get("api", {}))
    merged = _deep_merge_dict(common_api, scene_api)
    models = _as_mapping(merged.get("models", {}))
    default_models = _as_mapping(merged.get("default_models", {}))
    return {
        "models": models,
        "default_models": default_models,
    }


def resolve_default_model(config: Mapping[str, Any], *, stage_name: str) -> str:
    registry = load_api_registry(config)
    default_models = _as_mapping(registry.get("default_models", {}))
    return pick_first_text(default_models.get(stage_name), default_models.get("default"))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_model_route(route_value: Any) -> dict[str, Any]:
    if isinstance(route_value, dict):
        return {
            "api_source": pick_first_text(route_value.get("api_source"), route_value.get("source")),
            "api_base": pick_first_text(route_value.get("api_base"), route_value.get("base_url")),
            "api_key": pick_first_text(route_value.get("api_key"), route_value.get("key")),
            "request_model": pick_first_text(route_value.get("request_model")),
            "rpm_limit": _safe_int(route_value.get("rpm_limit"), 0),
            "tpm_limit": _safe_int(route_value.get("tpm_limit"), 0),
            "estimated_tokens_per_request": _safe_int(route_value.get("estimated_tokens_per_request"), 0),
            "rate_limit_reserve_ratio": _safe_float(route_value.get("rate_limit_reserve_ratio"), 0.1),
        }
    return {}


def _candidate_model_route_keys(model_name: str) -> list[str]:
    raw = str(model_name or "").strip()
    out: list[str] = []
    if raw:
        out.append(raw)

    low = raw.lower()
    if low.startswith("qwen/"):
        alt = raw.split("/", 1)[1].strip()
        if alt and alt not in out:
            out.append(alt)
    elif low.startswith("qwen"):
        alt = f"Qwen/{raw}"
        if alt not in out:
            out.append(alt)

    if "internvl" in low:
        if low.startswith("opengvlab/"):
            alt = raw.split("/", 1)[1].strip()
            if alt and alt not in out:
                out.append(alt)
        else:
            alt = f"OpenGVLab/{raw}"
            if alt not in out:
                out.append(alt)

        variant_pairs = []
        for item in list(out):
            variant_pairs.append(item)
            variant_pairs.append(item.replace("InternVL3_5", "InternVL3.5"))
            variant_pairs.append(item.replace("InternVL3.5", "InternVL3_5"))
            variant_pairs.append(item.replace("internvl3_5", "internvl3.5"))
            variant_pairs.append(item.replace("internvl3.5", "internvl3_5"))
        normalized = []
        for item in variant_pairs:
            if item and item not in normalized:
                normalized.append(item)
        out = normalized
    return out


def resolve_model_api_endpoint(
    *,
    config: Mapping[str, Any],
    model: str,
    stage_name: str,
    stage_cfg: Mapping[str, Any] | None = None,
    explicit_source: str = "",
    explicit_api_base: str = "",
    explicit_api_key: str = "",
) -> dict[str, str]:
    parsed_model = parse_experiment_model_name(str(model or "").strip()) if str(model or "").strip() else {"base_model": ""}
    model_name = str(parsed_model.get("base_model", model) or model or "").strip()
    if not model_name:
        raise RuntimeError(f"missing model for {stage_name}")

    registry = load_api_registry(config)
    model_routes = _as_mapping(registry.get("models", {}))
    # Some experiment model names may include suffixes that are not present in registry keys.
    # Example: "Qwen/Qwen3.5-4B-Instruct" should route to "Qwen/Qwen3.5-4B".
    model_name_for_route = model_name
    route_value = model_routes.get(model_name_for_route)
    if not route_value:
        lower = model_name_for_route.lower()
        instruct_suffix = "-instruct"
        if lower.endswith(instruct_suffix):
            alt = model_name_for_route[: -len(instruct_suffix)]
            if alt in model_routes:
                model_name_for_route = alt
                route_value = model_routes.get(model_name_for_route)

    route = _normalize_model_route(route_value)
    stage_cfg = _as_mapping(stage_cfg or {})
    api_base = resolve_env_token(
        pick_first_text(
            explicit_api_base,
            stage_cfg.get("api_base"),
            route.get("api_base"),
        )
    )
    api_key = resolve_env_token(
        pick_first_text(
            explicit_api_key,
            stage_cfg.get("api_key"),
            route.get("api_key"),
        )
    )
    request_model = pick_first_text(route.get("request_model"), model_name_for_route)
    return {
        "model": model_name_for_route,
        "request_model": request_model,
        "api_source": pick_first_text(explicit_source, stage_cfg.get("api_source"), route.get("api_source")),
        "api_base": api_base,
        "api_key": api_key,
        "rpm_limit": _safe_int(route.get("rpm_limit"), 0),
        "tpm_limit": _safe_int(route.get("tpm_limit"), 0),
        "estimated_tokens_per_request": _safe_int(route.get("estimated_tokens_per_request"), 0),
        "rate_limit_reserve_ratio": _safe_float(route.get("rate_limit_reserve_ratio"), 0.1),
    }
