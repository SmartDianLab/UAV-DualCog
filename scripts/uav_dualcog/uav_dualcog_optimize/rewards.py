from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


_ANSWER_RE = re.compile(r"(?:Answer|Option)\s*:\s*([A-Za-z])", re.IGNORECASE)
_BBOX_NORM_RE = re.compile(
    r"(?:Bounding\s*Box|BBox)\s*:\s*\[\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\]",
    re.IGNORECASE,
)


def _extract_json_payload(text: str) -> Optional[dict]:
    raw = str(text or "")
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw[start : end + 1])
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def _normalize_xyxy(vals: Iterable[float]) -> List[float]:
    out = [float(v) for v in vals]
    if len(out) != 4:
        raise ValueError(f"bbox must contain 4 numbers, got {out}")
    x1, y1, x2, y2 = out
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _normalize_bbox_scale(vals: Iterable[float]) -> List[float]:
    bbox = _normalize_xyxy(vals)
    max_abs = max(abs(v) for v in bbox)
    if max_abs > 1.5:
        bbox = [max(0.0, min(1.0, v / 1000.0)) for v in bbox]
    else:
        bbox = [max(0.0, min(1.0, v)) for v in bbox]
    return _normalize_xyxy(bbox)


def parse_answer(text: str) -> Optional[str]:
    payload = _extract_json_payload(text)
    if payload is not None:
        answer = str(payload.get("answer_option_id", "") or "").strip().upper()
        if answer in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            return answer
    m = _ANSWER_RE.search(text or "")
    if not m:
        return None
    return m.group(1).upper()


def parse_bbox_norm(text: str) -> Optional[List[float]]:
    payload = _extract_json_payload(text)
    if payload is not None:
        bbox = list(payload.get("bbox_xyxy_norm", []) or [])
        if len(bbox) == 4:
            try:
                return _normalize_bbox_scale(bbox)
            except Exception:
                pass
    m = _BBOX_NORM_RE.search(text or "")
    if not m:
        return None
    vals = [float(m.group(i)) for i in range(1, 5)]
    return _normalize_bbox_scale(vals)


def reward_format(text: str) -> float:
    return 1.0 if (parse_answer(text) is not None and parse_bbox_norm(text) is not None) else 0.0


def reward_mcq(pred_text: str, gt_answer: str) -> float:
    pa = parse_answer(pred_text)
    if pa is None:
        return 0.0
    return 1.0 if pa == str(gt_answer).strip().upper() else 0.0


def _area_xyxy(b: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in b]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _inter_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def giou_norm(pred_bbox: Iterable[float], gt_bbox: Iterable[float]) -> float:
    pa = _area_xyxy(pred_bbox)
    ga = _area_xyxy(gt_bbox)
    if pa <= 0.0 or ga <= 0.0:
        return -1.0
    inter = _inter_xyxy(pred_bbox, gt_bbox)
    union = pa + ga - inter
    if union <= 0.0:
        return -1.0
    iou = inter / union

    px1, py1, px2, py2 = [float(v) for v in pred_bbox]
    gx1, gy1, gx2, gy2 = [float(v) for v in gt_bbox]
    cx1, cy1 = min(px1, gx1), min(py1, gy1)
    cx2, cy2 = max(px2, gx2), max(py2, gy2)
    ca = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if ca <= 0.0:
        return iou
    return iou - (ca - union) / ca


def reward_bbox(pred_text: str, gt_bbox_norm: Iterable[float]) -> float:
    pb = parse_bbox_norm(pred_text)
    if pb is None:
        return 0.0
    g = giou_norm(pb, gt_bbox_norm)
    return max(0.0, min(1.0, (g + 1.0) * 0.5))


@dataclass
class RewardWeights:
    w_format: float = 1.0
    w_mcq: float = 1.0
    w_bbox: float = 0.0


def total_reward(
    pred_text: str,
    gt_answer: str,
    gt_bbox_norm: Iterable[float],
    weights: RewardWeights,
) -> Tuple[float, dict]:
    rf = reward_format(pred_text)
    rm = reward_mcq(pred_text, gt_answer)
    rb = reward_bbox(pred_text, gt_bbox_norm)
    total = weights.w_format * rf + weights.w_mcq * rm + weights.w_bbox * rb
    return total, {"format": rf, "mcq": rm, "bbox": rb}
