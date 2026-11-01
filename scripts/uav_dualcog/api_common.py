from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import requests
except Exception:
    requests = None

try:
    import yaml
except Exception:
    yaml = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
COMMON_RUNTIME_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "flightmvstg" / "common_api_runtime.yaml"
_ENV_TOKEN_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_MODEL_CONTROL_TOKENS = {
    "batch",
    "instant",
    "thinking",
    "reasoning",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}
_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}
_QWEN3_MULTIMODAL_VARIANT_PATTERN = re.compile(
    r"^(?P<stem>(?:Qwen/)?Qwen3-(?:VL|Omni)-.+?)-"
    r"(?P<variant>Instruct|Instant|Thinking|Reasoning)"
    r"(?P<tail>(?:-(?:Batch|Minimal|Low|Medium|High|XHigh|Max))*)$",
    re.IGNORECASE,
)
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
    if text.startswith("gpt-") or text.startswith("openai/gpt") or text.startswith("chatgpt-"):
        return "openai"
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
    if "kimi-k2.5" in text or "kimi-k2.6" in text or "moonshotai/kimi" in text:
        return "kimi"
    if "deepseek" in text:
        return "deepseek"
    if "glm-4.5v" in text or "glm-4.6v" in text or "z-ai/glm" in text or text.startswith("zai-org/glm"):
        return "glm"
    if text.startswith("meta/"):
        return "meta"
    if "mimo-v2-omni" in text or "xiaomi/mimo" in text or "mimo" in text:
        return "xiaomi"
    return ""


def should_inline_system_prompt_for_multimodal(model_name: str) -> bool:
    text = str(model_name or "").strip().lower()
    # SenseNova InternVL deployment currently fails on `system` + multimodal list content.
    return "sensenova-si-1.2-internvl3-8b" in text


def model_disallows_temperature(model_name: str) -> bool:
    text = str(model_name or "").strip().lower()
    # Claude 4.7+ compatible gateways reject `temperature` with:
    # "`temperature` is deprecated for this model."
    matched = re.search(r"\bclaude-(?:opus|sonnet|haiku)-4-(\d+)\b", text)
    return bool(matched and int(matched.group(1)) >= 7)


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
    if "qwen3.5-plus" in text or "qwen3.7-plus" in text:
        return 9_500_000
    return 0


def _parse_qwen3_multimodal_variant(raw: str) -> dict[str, Any] | None:
    matched = _QWEN3_MULTIMODAL_VARIANT_PATTERN.match(raw)
    if not matched:
        return None

    variant = str(matched.group("variant") or "").strip().lower()
    tail_tokens = [token for token in str(matched.group("tail") or "").split("-") if token]
    normalized_tokens = [token.lower() for token in tail_tokens]
    effort_tokens = [token for token in normalized_tokens if token in _REASONING_EFFORTS]
    batch_mode = "batch" in normalized_tokens
    if len(effort_tokens) > 1 or normalized_tokens.count("batch") > 1:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")

    stem = str(matched.group("stem") or "").strip()
    if variant in {"instruct", "instant"}:
        if effort_tokens:
            raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
        request_model = f"{stem}-Instruct"
        mode = "instant"
        thinking_effort = ""
    else:
        request_model = f"{stem}-Thinking"
        mode = "thinking"
        thinking_effort = effort_tokens[0] if effort_tokens else "low"

    return {
        "display_model": raw,
        "base_model": request_model,
        "mode": mode,
        "family": "qwen",
        "suffix": "-".join([str(matched.group("variant") or "").strip(), *tail_tokens]),
        "thinking_effort": thinking_effort,
        "batch_mode": batch_mode,
        "fixed_model_variant": True,
    }


def parse_experiment_model_name(model_name: str) -> dict[str, Any]:
    raw = str(model_name or "").strip()
    if not raw:
        raise RuntimeError("missing experiment model")
    fixed_variant = _parse_qwen3_multimodal_variant(raw)
    if fixed_variant is not None:
        return fixed_variant
    parts = raw.split("-")
    suffix_tokens: list[str] = []
    while len(parts) > 1 and parts[-1].strip().lower() in _MODEL_CONTROL_TOKENS:
        suffix_tokens.append(parts.pop().strip())
    suffix_tokens.reverse()
    control_tokens = [token.lower() for token in suffix_tokens]
    if not any(token in {"batch", "instant", "thinking", "reasoning"} for token in control_tokens):
        family = detect_model_family(raw)
        return {
            "display_model": raw,
            "base_model": raw,
            "mode": "default",
            "family": family,
            "suffix": "",
            "thinking_effort": "",
            "batch_mode": False,
            "fixed_model_variant": False,
        }
    base_model = "-".join(parts).strip()
    if not base_model:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    batch_mode = "batch" in control_tokens
    if control_tokens.count("batch") > 1:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    mode_tokens = [token for token in control_tokens if token in {"instant", "thinking", "reasoning"}]
    if len(mode_tokens) > 1:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    effort_tokens = [token for token in control_tokens if token in _REASONING_EFFORTS]
    if len(effort_tokens) > 1:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    mode_token = mode_tokens[0] if mode_tokens else ""
    mode = "thinking" if mode_token in {"thinking", "reasoning"} else ("instant" if mode_token == "instant" else "default")
    if effort_tokens and mode != "thinking":
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    unknown_or_misordered = [
        token for token in control_tokens
        if token not in {"batch", "instant", "thinking", "reasoning"} and token not in _REASONING_EFFORTS
    ]
    if unknown_or_misordered:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    family = detect_model_family(base_model)
    if mode != "default" and family not in {"qwen", "internvl", "interns1", "claude", "xai", "gemini", "google", "kimi", "deepseek", "glm", "xiaomi", "mimo", "openai", "meta"}:
        raise RuntimeError(f"unsupported_model_mode_suffix: {raw}")
    thinking_effort = str((effort_tokens[0] if effort_tokens else "low") if mode == "thinking" else "").strip().lower()
    return {
        "display_model": raw,
        "base_model": base_model,
        "mode": mode,
        "family": family,
        "suffix": "-".join(suffix_tokens),
        "thinking_effort": thinking_effort,
        "batch_mode": batch_mode,
        "fixed_model_variant": False,
    }


def build_model_request_controls(model_name: str, *, api_source: str = "") -> dict[str, Any]:
    parsed = parse_experiment_model_name(model_name)
    extra_body: dict[str, Any] = {}
    request_overrides: dict[str, Any] = {}
    assistant_prefill = ""
    system_prompt_prefix = ""
    system_prompt_as_blocks = False
    base_model_text = str(parsed.get("base_model", "") or "").strip().lower()
    api_source_text = str(api_source or "").strip().lower()
    thinking_effort = str(parsed.get("thinking_effort", "") or "").strip().lower()
    budget_by_effort = {"low": 1024, "medium": 2048, "high": 4096}
    is_qwen37_plus = "qwen3.7-plus" in base_model_text
    is_qwen36_plus = "qwen3.6-plus" in base_model_text
    is_qwen35_plus = "qwen3.5-plus" in base_model_text
    is_qwen35_flash = "qwen3.5-flash" in base_model_text
    is_kimi_k25 = "kimi-k2.5" in base_model_text
    is_qwen3_multimodal_fixed_variant = bool(parsed.get("fixed_model_variant", False))
    is_gemini_reasoning_effort_compat = (
        parsed["family"] in {"gemini", "google"}
        and api_source_text in {"chatchat.api", "zhongzhuanapi"}
    )
    is_openrouter = api_source_text == "openrouter"
    is_siliconflow = api_source_text == "siliconflow"
    is_openrouter_gemini_flash_35_to_38 = bool(
        (is_openrouter or base_model_text.startswith("google/"))
        and re.search(r"(?:^|/)gemini-3\.[5-8]-flash(?:$|[-:])", base_model_text)
    )
    is_claude_official_thinking_compat = parsed["family"] == "claude" and api_source_text in {"chatchat.api", "fastapi"}
    is_claude_official_instant_compat = parsed["family"] == "claude" and api_source_text in {"zhongzhuan.win", "fastapi"}
    is_gpt55_zhongzhuan = base_model_text == "gpt-5.5" and api_source_text == "zhongzhuan.win"
    if parsed["mode"] == "instant":
        if is_openrouter_gemini_flash_35_to_38:
            raise RuntimeError(
                f"openrouter_mandatory_reasoning_model_does_not_support_instant: {parsed['display_model']}"
            )
        if is_qwen3_multimodal_fixed_variant:
            pass
        elif is_qwen37_plus or is_qwen36_plus or is_qwen35_plus or is_qwen35_flash:
            extra_body["enable_thinking"] = False
        elif is_kimi_k25:
            extra_body["enable_thinking"] = False
        elif parsed["family"] == "qwen" and is_siliconflow:
            extra_body["enable_thinking"] = False
        elif parsed["family"] == "qwen":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif parsed["family"] == "internvl":
            assistant_prefill = "<think>\n</think>\n"
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False
        elif parsed["family"] == "interns1":
            extra_body["thinking_mode"] = False
        elif parsed["family"] == "glm" and is_siliconflow:
            extra_body["enable_thinking"] = False
        elif parsed["family"] in {"glm", "xiaomi", "mimo"}:
            extra_body["thinking"] = {"type": "disabled"}
        elif is_claude_official_instant_compat:
            # Anthropic's official extended-thinking API disables thinking by
            # omitting the thinking block; our zhongzhuan gateway also accepts
            # an explicit disabled switch, which is the most reliable Instant
            # mapping we have verified for Claude on this route.
            extra_body["thinking"] = {"type": "disabled"}
        elif is_claude_official_thinking_compat:
            # Anthropic extended thinking is enabled explicitly; for instant mode
            # we avoid sending a provider-specific override so plain generation
            # remains compatible with third-party OpenAI-compatible gateways.
            pass
        elif is_gemini_reasoning_effort_compat:
            # Google Gemini OpenAI-compatible API uses reasoning_effort instead of
            # a nested reasoning.enabled flag.
            extra_body["reasoning_effort"] = "low"
        elif parsed["family"] == "openai":
            extra_body["reasoning"] = {"effort": "none"}
        elif parsed["family"] in {"claude", "xai", "gemini", "google", "kimi"}:
            extra_body["reasoning"] = {"enabled": False}
    elif parsed["mode"] == "thinking":
        if is_openrouter_gemini_flash_35_to_38:
            if thinking_effort not in {"minimal", "low", "medium", "high"}:
                raise RuntimeError(
                    f"unsupported_openrouter_gemini_reasoning_effort: {thinking_effort}"
                )
            extra_body["reasoning"] = {"effort": thinking_effort or "medium", "exclude": False}
        if is_qwen3_multimodal_fixed_variant:
            pass
        elif parsed["family"] == "internvl":
            system_prompt_prefix = INTERNVL_THINKING_SYSTEM_PROMPT
            system_prompt_as_blocks = True
        elif is_qwen37_plus or is_qwen36_plus or is_qwen35_plus or is_qwen35_flash:
            extra_body["enable_thinking"] = True
            extra_body["thinking_budget"] = 4000
        elif is_kimi_k25:
            extra_body["enable_thinking"] = True
        elif parsed["family"] == "qwen" and is_siliconflow:
            extra_body["enable_thinking"] = True
        elif parsed["family"] == "qwen":
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        elif parsed["family"] == "interns1":
            extra_body["thinking_mode"] = True
        elif parsed["family"] == "glm" and is_siliconflow:
            extra_body["enable_thinking"] = True
        elif parsed["family"] in {"xiaomi", "mimo"}:
            extra_body["thinking"] = {"type": "enabled"}
        elif is_claude_official_thinking_compat:
            # Anthropic extended thinking uses a `thinking` object rather than
            # OpenAI-style `reasoning.enabled`, and requires budget_tokens to be
            # >= 1024 and < max_tokens.
            budget_tokens = int(budget_by_effort.get(thinking_effort, 1024))
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            request_overrides["max_tokens"] = budget_tokens * 2
        elif is_openrouter_gemini_flash_35_to_38:
            pass
        elif is_gemini_reasoning_effort_compat:
            # Google official OpenAI-compatibility guidance uses reasoning_effort.
            extra_body["reasoning_effort"] = thinking_effort or "low"
        elif is_gpt55_zhongzhuan:
            extra_body["reasoning"] = {"effort": thinking_effort or "low"}
        elif parsed["family"] == "openai":
            extra_body["reasoning"] = {"effort": thinking_effort or "low"}
        elif parsed["family"] in {"claude", "xai", "gemini", "google", "kimi"}:
            extra_body["reasoning"] = {"enabled": True}
    if is_openrouter and parsed["mode"] == "thinking" and not is_qwen3_multimodal_fixed_variant:
        reasoning_preferences = dict(extra_body.get("reasoning", {}) or {})
        reasoning_preferences.pop("enabled", None)
        reasoning_preferences["effort"] = thinking_effort or "low"
        reasoning_preferences["exclude"] = False
        extra_body["reasoning"] = reasoning_preferences
    if is_openrouter:
        provider_preferences = dict(extra_body.get("provider", {}) or {})
        provider_preferences["sort"] = "price"
        extra_body["provider"] = provider_preferences
    return {
        **parsed,
        "assistant_prefill": assistant_prefill,
        "system_prompt_prefix": system_prompt_prefix,
        "system_prompt_as_blocks": system_prompt_as_blocks,
        "extra_body": extra_body,
        "request_overrides": request_overrides,
    }


def should_use_responses_reasoning_stream(api_source: str, model_family: str, reasoning_mode: str) -> bool:
    # These OpenAI-compatible gateways have been verified to expose `/responses`
    # streaming for GPT-family reasoning requests. Other gateways may still
    # require chat.completions with provider-specific `extra_body` switches.
    return (
        str(api_source or "").strip().lower() in {"krill-ai", "maa"}
        and str(model_family or "").strip().lower() == "openai"
        and str(reasoning_mode or "").strip().lower() == "thinking"
    )


def _normalize_usage_dict(usage: Any) -> dict[str, Any]:
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        total_tokens = usage.get("total_tokens")
        details = usage.get("completion_tokens_details", usage.get("output_tokens_details", {}))
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if prompt_tokens is None:
            prompt_tokens = getattr(usage, "input_tokens", None)
        if completion_tokens is None:
            completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        if details is None:
            details = getattr(usage, "output_tokens_details", None)
    details = details if isinstance(details, dict) else {}
    out = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    if details.get("reasoning_tokens") is not None:
        out["reasoning_tokens"] = details.get("reasoning_tokens")
    return out


def _extract_reasoning_from_responses_output(output_items: Any) -> dict[str, Any]:
    summary_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    output_chunks: list[str] = []
    encrypted_content = ""
    for item in list(output_items or []):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "") or "").strip().lower()
        if item_type == "reasoning":
            for summary in list(item.get("summary", []) or []):
                if not isinstance(summary, dict):
                    continue
                if str(summary.get("type", "") or "").strip().lower() == "summary_text":
                    text = str(summary.get("text", "") or "").strip()
                    if text:
                        summary_chunks.append(text)
            for content in list(item.get("content", []) or []):
                if not isinstance(content, dict):
                    continue
                if str(content.get("type", "") or "").strip().lower() == "reasoning_text":
                    text = str(content.get("text", "") or "").strip()
                    if text:
                        reasoning_chunks.append(text)
            encrypted = str(item.get("encrypted_content", "") or "").strip()
            if encrypted and not encrypted_content:
                encrypted_content = encrypted
        elif item_type == "message":
            for content in list(item.get("content", []) or []):
                if not isinstance(content, dict):
                    continue
                text = str(content.get("text", "") or "").strip()
                if text:
                    output_chunks.append(text)
    return {
        "summary_text": "\n".join(summary_chunks).strip(),
        "reasoning_text": "\n".join(reasoning_chunks).strip(),
        "output_text": "\n".join(output_chunks).strip(),
        "encrypted_content": encrypted_content,
    }


def _responses_content_from_chat_message_content(content: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(content, str):
        text = str(content or "").strip()
        if text:
            items.append({"type": "input_text", "text": text})
        return items
    for block in list(content or []):
        if isinstance(block, str):
            text = str(block or "").strip()
            if text:
                items.append({"type": "input_text", "text": text})
            continue
        if not isinstance(block, Mapping):
            continue
        block_type = str(block.get("type", "") or "").strip().lower()
        if block_type in {"text", "input_text"}:
            text = str(block.get("text", "") or "").strip()
            if text:
                items.append({"type": "input_text", "text": text})
        elif block_type in {"image_url", "input_image"}:
            image_url = block.get("image_url")
            if isinstance(image_url, Mapping):
                url = str(image_url.get("url", "") or "").strip()
            else:
                url = str(image_url or "").strip()
            if url:
                items.append({"type": "input_image", "image_url": url})
    return items


def _convert_chat_messages_to_responses_payload(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions_parts: list[str] = []
    input_messages: list[dict[str, Any]] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "") or "").strip().lower()
        content = message.get("content")
        if role == "system":
            system_blocks = _responses_content_from_chat_message_content(content)
            system_text = "\n".join(
                str(item.get("text", "") or "").strip()
                for item in system_blocks
                if str(item.get("type", "") or "").strip().lower() == "input_text"
            ).strip()
            if system_text:
                instructions_parts.append(system_text)
            continue
        converted = _responses_content_from_chat_message_content(content)
        if converted:
            input_messages.append({"role": role or "user", "content": converted})
    return "\n\n".join(part for part in instructions_parts if part).strip(), input_messages


def run_responses_reasoning_stream_request(api_cfg: Mapping[str, Any], messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], float]:
    if requests is None:
        raise RuntimeError("responses_stream_requires_requests")
    api_base = str(api_cfg.get("api_base", "") or "").rstrip("/")
    api_key = str(api_cfg.get("api_key", "") or "")
    if not api_base or not api_key:
        raise RuntimeError("missing_api_base_or_key_for_responses_stream")

    url = f"{api_base}/responses"
    reasoning_cfg = dict(((api_cfg.get("request_extra_body", {}) or {}).get("reasoning", {}) or {}))
    if not str(reasoning_cfg.get("effort", "") or "").strip():
        reasoning_cfg["effort"] = "high"
    reasoning_cfg["summary"] = str(reasoning_cfg.get("summary", "") or "auto")

    instructions_text, input_messages = _convert_chat_messages_to_responses_payload(messages)
    payload: dict[str, Any] = {
        "model": str(api_cfg.get("request_model", "") or ""),
        "input": input_messages,
        "reasoning": reasoning_cfg,
        "include": ["reasoning.encrypted_content"],
        "stream": True,
    }
    if instructions_text:
        payload["instructions"] = instructions_text
    temperature = api_cfg.get("temperature", None)
    if temperature is not None and not model_disallows_temperature(str(api_cfg.get("request_model", "") or api_cfg.get("model", "") or "")):
        payload["temperature"] = float(temperature)
    max_tokens = api_cfg.get("max_tokens", None)
    if max_tokens is not None:
        payload["max_output_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout_s = float(api_cfg.get("timeout_s", 30.0) or 30.0)
    latency_started = time.perf_counter()
    last_error: Exception | None = None
    trust_env_options = [True, False]
    for trust_env in trust_env_options:
        session = requests.Session()
        session.trust_env = trust_env
        try:
            event_types: list[str] = []
            summary_delta_chunks: list[str] = []
            summary_done_chunks: list[str] = []
            reasoning_delta_chunks: list[str] = []
            reasoning_done_chunks: list[str] = []
            output_delta_chunks: list[str] = []
            output_done_chunks: list[str] = []
            completed_response: dict[str, Any] = {}
            with session.post(url, headers=headers, json=payload, timeout=timeout_s, stream=True) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data: "):
                        continue
                    data = raw_line[6:]
                    if data == "[DONE]":
                        event_types.append("done")
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    event_type = str(obj.get("type", "") or "")
                    if event_type:
                        event_types.append(event_type)
                    if event_type == "response.reasoning_summary_text.delta":
                        delta = str(obj.get("delta", "") or "")
                        if delta:
                            summary_delta_chunks.append(delta)
                    elif event_type == "response.reasoning_summary_text.done":
                        text = str(obj.get("text", "") or "")
                        if text:
                            summary_done_chunks.append(text)
                    elif event_type == "response.reasoning_text.delta":
                        delta = str(obj.get("delta", "") or "")
                        if delta:
                            reasoning_delta_chunks.append(delta)
                    elif event_type == "response.reasoning_text.done":
                        text = str(obj.get("text", "") or "")
                        if text:
                            reasoning_done_chunks.append(text)
                    elif event_type == "response.output_text.delta":
                        delta = str(obj.get("delta", "") or "")
                        if delta:
                            output_delta_chunks.append(delta)
                    elif event_type == "response.output_text.done":
                        text = str(obj.get("text", "") or "")
                        if text:
                            output_done_chunks.append(text)
                    elif event_type == "response.completed":
                        completed_response = dict(obj.get("response", {}) or {})

            extracted = _extract_reasoning_from_responses_output(completed_response.get("output"))
            summary_text = "\n".join(summary_done_chunks).strip() or "".join(summary_delta_chunks).strip() or str(extracted.get("summary_text", "") or "").strip()
            reasoning_text = "\n".join(reasoning_done_chunks).strip() or "".join(reasoning_delta_chunks).strip() or str(extracted.get("reasoning_text", "") or "").strip()
            output_text = "\n".join(output_done_chunks).strip() or "".join(output_delta_chunks).strip() or str(completed_response.get("output_text", "") or "").strip() or str(extracted.get("output_text", "") or "").strip()

            meta: dict[str, Any] = {
                "id": completed_response.get("id"),
                "model": completed_response.get("model") or api_cfg.get("request_model"),
                "response_api_transport": "responses_stream",
                "response_api_reasoning": completed_response.get("reasoning"),
                "responses_stream_event_types": list(event_types),
                "responses_stream_trust_env": trust_env,
                "responses_request_reasoning": reasoning_cfg,
                "responses_request_include": list(payload.get("include", []) or []),
            }
            if completed_response.get("usage") is not None:
                meta["usage"] = _normalize_usage_dict(completed_response.get("usage"))
            if summary_text:
                meta["reasoning_summary_text"] = summary_text
            if reasoning_text:
                meta["reasoning_text"] = reasoning_text
                meta["thinking_text"] = reasoning_text
            elif summary_text:
                meta["reasoning_text"] = summary_text
                meta["thinking_text"] = summary_text
            encrypted_content = str(extracted.get("encrypted_content", "") or "").strip()
            if encrypted_content:
                meta["reasoning_encrypted_content"] = encrypted_content
            latency_ms = (time.perf_counter() - latency_started) * 1000.0
            return output_text, meta, latency_ms
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if trust_env and ("unexpected eof" in error_text or "ssl" in error_text):
                continue
            break
    raise RuntimeError(str(last_error or "responses_stream_request_failed"))


def compute_rate_limited_concurrency(
    requested_concurrency: int,
    *,
    rpm_limit: int = 0,
    tpm_limit: int = 0,
    estimated_tokens_per_request: int = 0,
    reserve_ratio: float = 0.1,
    avg_request_latency_sec: float = 0.0,
) -> dict[str, Any]:
    requested = max(1, _safe_int(requested_concurrency, 1))
    reserve = min(max(_safe_float(reserve_ratio, 0.1), 0.0), 0.5)
    effective_rpm = max(0, int(math.floor(max(0, _safe_int(rpm_limit, 0)) * (1.0 - reserve))))
    effective_tpm = max(0, int(math.floor(max(0, _safe_int(tpm_limit, 0)) * (1.0 - reserve))))
    estimated_tokens = max(0, _safe_int(estimated_tokens_per_request, 0))
    avg_latency = max(0.0, _safe_float(avg_request_latency_sec, 0.0))
    caps: list[int] = []
    if effective_rpm > 0:
        if avg_latency > 0.0:
            caps.append(max(1, int(math.floor((effective_rpm * avg_latency) / 60.0))))
        else:
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
        "avg_request_latency_sec": avg_latency,
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
            "batch_api_source": pick_first_text(route_value.get("batch_api_source")),
            "batch_api_base": pick_first_text(route_value.get("batch_api_base")),
            "batch_api_key": pick_first_text(route_value.get("batch_api_key")),
            "batch_request_model": pick_first_text(route_value.get("batch_request_model")),
            "batch_requests_per_job": _safe_int(
                route_value.get("batch_requests_per_job", route_value.get("batch_file_requests_per_job")), 0
            ),
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
        for candidate in _candidate_model_route_keys(model_name_for_route):
            if candidate in model_routes:
                model_name_for_route = candidate
                route_value = model_routes.get(model_name_for_route)
                break
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
    batch_mode = bool(parsed_model.get("batch_mode", False))
    api_base = resolve_env_token(
        pick_first_text(
            explicit_api_base,
            route.get("batch_api_base") if batch_mode else "",
            stage_cfg.get("api_base"),
            route.get("api_base"),
        )
    )
    api_key = pick_first_text(
        resolve_env_token(explicit_api_key),
        resolve_env_token(route.get("batch_api_key") if batch_mode else ""),
        resolve_env_token(stage_cfg.get("api_key")),
        resolve_env_token(route.get("api_key")),
    )
    request_model = pick_first_text(
        route.get("batch_request_model") if batch_mode else "",
        route.get("request_model"),
        model_name_for_route,
    )
    return {
        "model": model_name_for_route,
        "request_model": request_model,
        "api_source": pick_first_text(
            explicit_source,
            route.get("batch_api_source") if batch_mode else "",
            stage_cfg.get("api_source"),
            route.get("api_source"),
        ),
        "api_base": api_base,
        "api_key": api_key,
        "rpm_limit": _safe_int(route.get("rpm_limit"), 0),
        "tpm_limit": _safe_int(route.get("tpm_limit"), 0),
        "estimated_tokens_per_request": _safe_int(route.get("estimated_tokens_per_request"), 0),
        "rate_limit_reserve_ratio": _safe_float(route.get("rate_limit_reserve_ratio"), 0.1),
        "batch_mode": batch_mode,
        "batch_requests_per_job": _safe_int(route.get("batch_requests_per_job"), 0),
    }
