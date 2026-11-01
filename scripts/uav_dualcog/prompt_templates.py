from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:
    yaml = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROMPT_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "prompts" / "flightmvstg_prompts.yaml"
_TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


@lru_cache(maxsize=1)
def load_prompt_config() -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("prompt_config_yaml_unavailable")
    if not PROMPT_CONFIG_PATH.exists():
        raise RuntimeError(f"prompt_config_missing: {PROMPT_CONFIG_PATH}")
    payload = yaml.safe_load(PROMPT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError("prompt_config_invalid")
    return payload


def get_prompt_template(stage: str, key: str, section: str) -> str:
    payload = load_prompt_config()
    stage_block = payload.get(str(stage), {})
    if not isinstance(stage_block, dict):
        raise RuntimeError(f"prompt_stage_missing: {stage}")
    task_block = stage_block.get(str(key), {})
    if not isinstance(task_block, dict):
        raise RuntimeError(f"prompt_task_missing: {stage}.{key}")
    template = task_block.get(str(section), None)
    if not isinstance(template, str) or not template.strip():
        raise RuntimeError(f"prompt_template_missing: {stage}.{key}.{section}")
    return template


def get_config_template(*path_parts: str) -> str:
    payload: Any = load_prompt_config()
    joined = ".".join(str(x) for x in path_parts)
    for part in path_parts:
        if not isinstance(payload, dict):
            raise RuntimeError(f"prompt_config_path_missing: {joined}")
        payload = payload.get(str(part))
    if not isinstance(payload, str) or not payload.strip():
        raise RuntimeError(f"prompt_config_template_missing: {joined}")
    return payload


def render_prompt_template(template: str, variables: dict[str, Any] | None = None) -> str:
    values = {str(k): "" if v is None else str(v) for k, v in dict(variables or {}).items()}

    def _replace(match: re.Match[str]) -> str:
        return values.get(match.group(1), "")

    rendered = _TOKEN_RE.sub(_replace, str(template))
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip()
