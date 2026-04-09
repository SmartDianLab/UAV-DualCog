from __future__ import annotations

from pathlib import Path
from typing import Iterable


IMAGE_SUFFIX_PRIORITY = (".jpg", ".jpeg", ".png")


def resolve_existing_file_with_suffix_fallback(
    raw_path: str | Path,
    *,
    base_dirs: Iterable[Path] = (),
    suffix_priority: Iterable[str] = IMAGE_SUFFIX_PRIORITY,
) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidates: list[Path] = []
    candidates.append(path)
    for base_dir in base_dirs:
        candidates.append(Path(base_dir) / path)
    seen: set[str] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            continue
        marker = str(resolved)
        if marker in seen:
            continue
        seen.add(marker)
        if resolved.exists() and resolved.is_file():
            return resolved
        for suffix in suffix_priority:
            alt = resolved.with_suffix(str(suffix))
            if alt.exists() and alt.is_file():
                return alt
    return None


def rewrite_relative_suffix_if_needed(
    raw_path: str,
    *,
    base_dir: Path,
    suffix_priority: Iterable[str] = IMAGE_SUFFIX_PRIORITY,
) -> str | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    current = resolve_existing_file_with_suffix_fallback(raw, base_dirs=[base_dir], suffix_priority=suffix_priority)
    if current is None:
        return None
    try:
        candidate = (Path(base_dir) / raw).resolve()
    except Exception:
        candidate = None
    if candidate is not None and current == candidate:
        return None
    raw_obj = Path(raw)
    return raw_obj.with_suffix(current.suffix).as_posix()
