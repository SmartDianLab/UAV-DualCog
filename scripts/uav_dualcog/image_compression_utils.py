from __future__ import annotations

import io
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

import numpy as np


DEFAULT_IMAGE_TARGET_MAX_BYTES = 1024 * 1024


def compression_cfg(
    stage_cfg: dict[str, Any] | None = None,
    *,
    enabled_default: bool = True,
    target_max_bytes_default: int = DEFAULT_IMAGE_TARGET_MAX_BYTES,
    quality_min_default: int = 45,
    quality_max_default: int = 95,
    resize_step_default: float = 0.85,
    min_side_default: int = 720,
) -> dict[str, Any]:
    cfg = dict(stage_cfg or {})
    return {
        "enabled": bool(cfg.get("image_compress_enabled", enabled_default)),
        "target_max_bytes": max(0, int(cfg.get("image_target_max_bytes", target_max_bytes_default) or target_max_bytes_default)),
        "quality_min": max(20, min(95, int(cfg.get("image_jpeg_quality_min", quality_min_default) or quality_min_default))),
        "quality_max": max(20, min(95, int(cfg.get("image_jpeg_quality_max", quality_max_default) or quality_max_default))),
        "resize_step": max(0.5, min(0.98, float(cfg.get("image_resize_step", resize_step_default) or resize_step_default))),
        "min_side": max(128, int(cfg.get("image_min_side", min_side_default) or min_side_default)),
    }


def preferred_output_path(output_path: Path, *, compress_enabled: bool) -> Path:
    path = Path(output_path)
    if not compress_enabled:
        return path
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return path
    return path.with_suffix(".jpg")


def _ensure_pillow() -> None:
    if Image is None:
        raise ImportError("Pillow is required for image compression")


def _encode_jpeg(img: Any, *, quality: int) -> bytes:
    buff = io.BytesIO()
    img.save(buff, format="JPEG", quality=int(quality), optimize=True)
    return buff.getvalue()


def compress_pil_image_to_target(
    image: Any,
    *,
    target_max_bytes: int,
    quality_min: int,
    quality_max: int,
    resize_step: float,
    min_side: int,
) -> bytes:
    _ensure_pillow()
    canvas = image.convert("RGB")
    low = int(quality_min)
    high = int(quality_max)
    best_payload: bytes | None = None
    while low <= high:
        mid = (low + high) // 2
        payload = _encode_jpeg(canvas, quality=mid)
        if len(payload) <= int(target_max_bytes):
            best_payload = payload
            low = mid + 1
        else:
            high = mid - 1

    if best_payload is not None:
        return best_payload
    return _encode_jpeg(canvas, quality=int(quality_min))


def save_pil_image(
    image: Any,
    output_path: Path,
    *,
    cfg: dict[str, Any],
) -> Path:
    _ensure_pillow()
    output_path = preferred_output_path(output_path, compress_enabled=bool(cfg.get("enabled", True)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not bool(cfg.get("enabled", True)):
        image.convert("RGB").save(output_path)
        return output_path
    payload = compress_pil_image_to_target(
        image,
        target_max_bytes=int(cfg.get("target_max_bytes", DEFAULT_IMAGE_TARGET_MAX_BYTES)),
        quality_min=int(cfg.get("quality_min", 45)),
        quality_max=int(cfg.get("quality_max", 95)),
        resize_step=float(cfg.get("resize_step", 0.85)),
        min_side=int(cfg.get("min_side", 720)),
    )
    output_path.write_bytes(payload)
    return output_path


def save_bgr_image(
    image_bgr: np.ndarray,
    output_path: Path,
    *,
    cfg: dict[str, Any],
) -> Path:
    _ensure_pillow()
    rgb = image_bgr[:, :, ::-1] if image_bgr.ndim == 3 and image_bgr.shape[2] >= 3 else image_bgr
    pil_img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    return save_pil_image(pil_img, output_path, cfg=cfg)


def compress_existing_image_file(
    src_path: Path,
    *,
    cfg: dict[str, Any],
    dst_path: Path | None = None,
) -> Path:
    _ensure_pillow()
    src = Path(src_path)
    out = preferred_output_path(dst_path or src, compress_enabled=bool(cfg.get("enabled", True)))
    with Image.open(src) as image:
        written = save_pil_image(image, out, cfg=cfg)
    return written
