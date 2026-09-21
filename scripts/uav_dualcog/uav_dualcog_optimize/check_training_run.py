#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

# Allow direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uav_dualcog.uav_dualcog_optimize.common import resolve_path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(x: Optional[float], nd: int = 6) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _find_checkpoint_steps(stage_dir: Path) -> List[int]:
    out: List[int] = []
    for d in stage_dir.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", d.name)
        if not m:
            continue
        out.append(int(m.group(1)))
    out.sort()
    return out


def _tail_mean(vals: List[float], k: int = 10) -> Optional[float]:
    if not vals:
        return None
    return float(mean(vals[-k:]))


def _extract_trainer_tail(stage_dir: Path, last_step: int) -> Dict[str, Any]:
    p = stage_dir / f"checkpoint-{last_step}" / "trainer_state.json"
    if not p.exists():
        return {"available": False}
    obj = _load_json(p)
    log = list(obj.get("log_history", []) or [])
    # Keep entries that are train-step logs.
    log_steps = [x for x in log if isinstance(x, dict) and ("step" in x)]
    rewards = [float(x["reward"]) for x in log_steps if x.get("reward") is not None]
    reward_std = [float(x["reward_std"]) for x in log_steps if x.get("reward_std") is not None]
    losses = [float(x["loss"]) for x in log_steps if x.get("loss") is not None]
    step_time = [float(x["step_time"]) for x in log_steps if x.get("step_time") is not None]
    eval_loss = [float(x["eval_loss"]) for x in log_steps if x.get("eval_loss") is not None]
    out: Dict[str, Any] = {
        "available": True,
        "global_step": int(obj.get("global_step", 0) or 0),
        "num_log_rows": len(log_steps),
        "reward_last": rewards[-1] if rewards else None,
        "reward_tail10_mean": _tail_mean(rewards, 10),
        "reward_std_tail10_mean": _tail_mean(reward_std, 10),
        "loss_tail10_mean": _tail_mean(losses, 10),
        "step_time_tail10_mean": _tail_mean(step_time, 10),
        "eval_loss_last": eval_loss[-1] if eval_loss else None,
    }
    return out


def _stage_summary(run_dir: Path, stage_name: str, stage_rel: str, metrics_file: str, plan_file: str) -> Dict[str, Any]:
    stage_dir = run_dir / "checkpoints" / stage_rel
    logs_dir = run_dir / "logs"
    merged_dir = stage_dir / "merged_model"
    adapter_dir = stage_dir / "adapter"
    metrics_path = stage_dir / metrics_file
    plan_path = stage_dir / plan_file
    checkpoints = _find_checkpoint_steps(stage_dir) if stage_dir.exists() else []
    last_step = checkpoints[-1] if checkpoints else 0

    out: Dict[str, Any] = {
        "stage_name": stage_name,
        "stage_dir": str(stage_dir),
        "exists": stage_dir.exists(),
        "plan_exists": plan_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "adapter_exists": adapter_dir.exists(),
        "merged_model_exists": merged_dir.exists(),
        "checkpoint_count": len(checkpoints),
        "last_checkpoint_step": last_step,
        "log_exists": (logs_dir / f"{stage_rel}.log").exists(),
    }
    if plan_path.exists():
        out["plan"] = _load_json(plan_path)
    if metrics_path.exists():
        out["metrics"] = _load_json(metrics_path)
    if last_step > 0:
        out["trainer_tail"] = _extract_trainer_tail(stage_dir, last_step)
    else:
        out["trainer_tail"] = {"available": False}
    return out


def _render_md(data: Dict[str, Any]) -> str:
    meta = data.get("meta", {})
    stages = data.get("stages", {})
    lines: List[str] = []
    lines.append(f"# UAV-DualCog Training Run Integrity Report")
    lines.append("")
    lines.append(f"- Generated at: `{meta.get('generated_at')}`")
    lines.append(f"- Run dir: `{meta.get('run_dir')}`")
    lines.append(f"- Overall complete: `{meta.get('overall_complete')}`")
    lines.append("")
    for k in ("sft", "grpo"):
        st = stages.get(k, {})
        lines.append(f"## {st.get('stage_name', k)}")
        lines.append("")
        lines.append(f"- Stage dir exists: `{st.get('exists')}`")
        lines.append(f"- Plan exists: `{st.get('plan_exists')}`")
        lines.append(f"- Metrics exists: `{st.get('metrics_exists')}`")
        lines.append(f"- Adapter exists: `{st.get('adapter_exists')}`")
        lines.append(f"- Merged model exists: `{st.get('merged_model_exists')}`")
        lines.append(f"- Checkpoints: `{st.get('checkpoint_count')}` (last step: `{st.get('last_checkpoint_step')}`)")
        m = st.get("metrics", {}) or {}
        if m:
            lines.append("- Core metrics:")
            lines.append(f"  - train_runtime(s): `{_fmt_float(m.get('train_runtime'), 4)}`")
            lines.append(f"  - train_loss: `{_fmt_float(m.get('train_loss'), 8)}`")
            lines.append(f"  - epoch: `{_fmt_float(m.get('epoch'), 4)}`")
        t = st.get("trainer_tail", {}) or {}
        if t.get("available"):
            lines.append("- Tail summary (last checkpoint trainer_state):")
            lines.append(f"  - reward_last: `{_fmt_float(t.get('reward_last'), 6)}`")
            lines.append(f"  - reward_tail10_mean: `{_fmt_float(t.get('reward_tail10_mean'), 6)}`")
            lines.append(f"  - reward_std_tail10_mean: `{_fmt_float(t.get('reward_std_tail10_mean'), 6)}`")
            lines.append(f"  - loss_tail10_mean: `{_fmt_float(t.get('loss_tail10_mean'), 6)}`")
            lines.append(f"  - step_time_tail10_mean(s): `{_fmt_float(t.get('step_time_tail10_mean'), 4)}`")
            lines.append(f"  - eval_loss_last: `{_fmt_float(t.get('eval_loss_last'), 6)}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check completeness and key metrics of a two-stage optimization training run.")
    parser.add_argument(
        "--run-dir",
        default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm/training_runs/uav_dualcog_full_20260519_004609",
    )
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/analysis")
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    out_dir = resolve_path(args.output_dir) if args.output_dir else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = {
        "sft": _stage_summary(
            run_dir,
            "Stage 1/2 - SFT",
            "01_sft_full",
            "sft_train_metrics.latest.json",
            "sft_plan.latest.json",
        ),
        "grpo": _stage_summary(
            run_dir,
            "Stage 2/2 - Joint GRPO",
            "02_grpo_joint",
            "grpo_metrics.latest.json",
            "grpo_plan.latest.json",
        ),
    }

    def _ok(st: Dict[str, Any]) -> bool:
        return bool(
            st.get("exists")
            and st.get("plan_exists")
            and st.get("metrics_exists")
            and st.get("adapter_exists")
            and st.get("merged_model_exists")
            and int(st.get("checkpoint_count", 0)) > 0
        )

    overall_complete = all(_ok(stages[k]) for k in ("sft", "grpo"))
    data: Dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "overall_complete": overall_complete,
        },
        "stages": stages,
    }

    json_path = out_dir / "training_integrity.latest.json"
    md_path = out_dir / "training_integrity.latest.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    md_path.write_text(_render_md(data), encoding="utf-8")
    print(f"[check_training_run] wrote {json_path}")
    print(f"[check_training_run] wrote {md_path}")


if __name__ == "__main__":
    main()
