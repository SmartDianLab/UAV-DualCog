#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uav_dualcog.uav_dualcog_optimize.common import ensure_dir, resolve_path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100.0:.2f}%"


def _fmt_num(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _get(d: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    try:
        return None if cur is None else float(cur)
    except Exception:
        return None


def _delta(now: Optional[float], ref: Optional[float]) -> Optional[float]:
    if now is None or ref is None:
        return None
    return now - ref


def _render_markdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# UAV-DualCog Optimization Stage Comparison")
    lines.append("")
    lines.append(f"- Generated at: `{datetime.now().isoformat()}`")
    lines.append("")
    lines.append("## 1) Branch-Level Core Metrics")
    lines.append("")
    lines.append("| Stage | Image Option Acc | Image BBox Acc@0.5 | Image mIoU | Video Segment F1@0.5 | Video mTIoU | Video Count Exact |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['stage']} | {_fmt_pct(r['img_option'])} | {_fmt_pct(r['img_bbox_acc'])} | {_fmt_pct(r['img_miou'])} | {_fmt_pct(r['vid_f1'])} | {_fmt_pct(r['vid_mtiou'])} | {_fmt_pct(r['vid_count_exact'])} |"
        )
    lines.append("")
    lines.append("## 2) Derived Dual-Cognition Scores")
    lines.append("")
    lines.append("| Stage | Image Dual Score | Video Dual Score | Gap (Image - Video) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['stage']} | {_fmt_num(r['image_dual_score'], 4)} | {_fmt_num(r['video_dual_score'], 4)} | {_fmt_num(r['cross_gap'], 4)} |"
        )
    lines.append("")
    lines.append("## 3) Delta vs Base")
    lines.append("")
    lines.append("| Stage | Δ Image Option | Δ Image BBox@0.5 | Δ Image mIoU | Δ Video F1@0.5 | Δ Video mTIoU | Δ Video Count Exact |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows[1:]:
        lines.append(
            f"| {r['stage']} | {_fmt_pct(r['d_img_option'])} | {_fmt_pct(r['d_img_bbox_acc'])} | {_fmt_pct(r['d_img_miou'])} | {_fmt_pct(r['d_vid_f1'])} | {_fmt_pct(r['d_vid_mtiou'])} | {_fmt_pct(r['d_vid_count_exact'])} |"
        )
    lines.append("")
    lines.append("## 4) Interpretation Checklist")
    lines.append("")
    lines.append("- Report both **semantic gains** and **evidence gains**; do not claim improvement from semantics alone.")
    lines.append("- For SFT, prioritize improvements in answer stability and parser success.")
    lines.append("- For joint GRPO, verify whether bbox- and interval-related metrics improve without harming semantic scores.")
    lines.append("- Keep migration discussion explicit: image-trained optimization should be analyzed against video-task transfer, not only image in-domain metrics.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base/SFT/joint-GRPO stage results from evaluate_transfer outputs.")
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--sft-json", required=True)
    parser.add_argument("--grpo-json", required=True)
    parser.add_argument("--output-dir", default="task_pipeline_data/UAV-DualCog-V1/optimization_eval")
    args = parser.parse_args()

    out_dir = resolve_path(args.output_dir)
    ensure_dir(out_dir)
    files = [
        ("Base", resolve_path(args.base_json)),
        ("SFT", resolve_path(args.sft_json)),
        ("GRPO", resolve_path(args.grpo_json)),
    ]
    loaded: List[Dict[str, Any]] = []
    for name, p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        loaded.append({"stage": name, "obj": _load_json(p), "path": str(p)})

    rows: List[Dict[str, Any]] = []
    for item in loaded:
        obj = item["obj"]
        stage = item["stage"]
        agg = obj.get("aggregated", {})
        drv = obj.get("derived", {})
        row = {
            "stage": stage,
            "img_option": _get(agg, ["stage4", "overall", "option_accuracy"]),
            "img_bbox_acc": _get(agg, ["stage4", "overall", "bbox_acc@50iou"]),
            "img_miou": _get(agg, ["stage4", "overall", "bbox_mean_iou"]),
            "vid_f1": _get(agg, ["stage3", "overall", "segment_f1@0.5"]),
            "vid_mtiou": _get(agg, ["stage3", "overall", "mean_best_tIoU"]),
            "vid_count_exact": _get(agg, ["stage3", "overall", "count_exact_acc"]),
            "image_dual_score": _get(drv, ["image_dual_score"]),
            "video_dual_score": _get(drv, ["video_dual_score"]),
            "cross_gap": _get(drv, ["cross_media_gap_image_minus_video"]),
            "source_json": item["path"],
        }
        rows.append(row)

    base = rows[0]
    for r in rows:
        r["d_img_option"] = _delta(r["img_option"], base["img_option"])
        r["d_img_bbox_acc"] = _delta(r["img_bbox_acc"], base["img_bbox_acc"])
        r["d_img_miou"] = _delta(r["img_miou"], base["img_miou"])
        r["d_vid_f1"] = _delta(r["vid_f1"], base["vid_f1"])
        r["d_vid_mtiou"] = _delta(r["vid_mtiou"], base["vid_mtiou"])
        r["d_vid_count_exact"] = _delta(r["vid_count_exact"], base["vid_count_exact"])

    json_out = out_dir / "optimization_stage_compare.latest.json"
    md_out = out_dir / "optimization_stage_compare.latest.md"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    _render_markdown(rows, md_out)
    print(f"[analyze_stage_results] wrote {json_out}")
    print(f"[analyze_stage_results] wrote {md_out}")


if __name__ == "__main__":
    main()
