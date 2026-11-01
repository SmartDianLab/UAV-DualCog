#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow direct execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.uav_dualcog.uav_dualcog_optimize.common import (
    copy_multimodal_support_files,
    ensure_dir,
    ensure_vllm_cuda_lib_priority,
    load_rgb_image,
    resolve_path,
)
from scripts.uav_dualcog.uav_dualcog_optimize.rewards import RewardWeights, total_reward


def _load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and i + 1 >= limit:
                break
    return rows


def _safe_import_train_stack() -> bool:
    try:
        ensure_vllm_cuda_lib_priority()
        import transformers  # noqa: F401
        import torch  # noqa: F401
        import trl  # noqa: F401
        import peft  # noqa: F401
        return True
    except Exception:
        return False


class _MultimodalGRPODataset:
    def __init__(self, rows: List[Dict[str, Any]], max_image_side: int) -> None:
        self.rows = rows
        self.max_image_side = max_image_side

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "prompt": copy.deepcopy(list(row.get("prompt", []) or [])),
            "images": [load_rgb_image(p, max_side=self.max_image_side) for p in list(row.get("images", []) or [])],
            "answer_option_id": str(row.get("answer_option_id", "")),
            "answer_bbox_xyxy_norm": list(row.get("answer_bbox_xyxy_norm", [0, 0, 0, 0])),
            "sample_id": str(row.get("id", "")),
            "task_type": str(row.get("task_type", "")),
            "difficulty": str(row.get("difficulty", "")),
        }


def _weights_from_args(args: argparse.Namespace) -> RewardWeights:
    return RewardWeights(w_format=args.w_format, w_mcq=args.w_mcq, w_bbox=args.w_bbox)


def _completion_to_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        if isinstance(c.get("content"), str):
            return str(c.get("content", ""))
        if isinstance(c.get("text"), str):
            return str(c.get("text", ""))
        return str(c)
    if isinstance(c, list):
        parts = [_completion_to_text(x) for x in c]
        return "\n".join([p for p in parts if p]).strip()
    return str(c)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal joint GRPO for UAV-DualCog optimization.")
    parser.add_argument("--train-jsonl", default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm/uav_dualcog_train.latest.jsonl")
    parser.add_argument("--sft-checkpoint", required=True, help="Checkpoint path from SFT stage.")
    parser.add_argument("--output-dir", default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm/checkpoints/grpo_joint")
    parser.add_argument("--model-name", default="vllm/models/qwen3_5-4b", help="Base model used to load adapter-only checkpoints.")
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--max-image-side", type=int, default=480, help="Resize each training image so its long side does not exceed this value. 480 is the recommended default; 720 is optional but significantly heavier.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-merged", action="store_true", help="Export merged full model for deployment.")
    parser.add_argument("--w-format", type=float, default=1.0)
    parser.add_argument("--w-mcq", type=float, default=1.0)
    parser.add_argument("--w-bbox", type=float, default=1.0)
    parser.add_argument("--smoke-limit", type=int, default=0, help="Read only first N rows for smoke run; 0 means full dataset.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_path = resolve_path(args.train_jsonl)
    ckpt_path = resolve_path(args.sft_checkpoint)
    out_dir = resolve_path(args.output_dir)
    ensure_dir(out_dir)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    rows = _load_jsonl(train_path, limit=args.smoke_limit if args.smoke_limit > 0 else 0)
    reward_w = _weights_from_args(args)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry-run" if args.dry_run else "train",
        "multimodal": True,
        "stage": "joint_grpo",
        "model_name": args.model_name,
        "sft_checkpoint": str(ckpt_path),
        "train_jsonl": str(train_path),
        "rows_loaded": len(rows),
        "hyperparams": {
            "group_size": args.group_size,
            "max_completion_length": args.max_completion_length,
            "max_image_side": args.max_image_side,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
        },
        "reward_weights": {
            "format": reward_w.w_format,
            "mcq": reward_w.w_mcq,
            "bbox": reward_w.w_bbox,
        },
    }
    with (out_dir / "grpo_plan.latest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        print("[run_grpo_light] dry-run complete. plan saved.")
        return

    if not _safe_import_train_stack():
        raise RuntimeError("TRL multimodal training stack missing.")

    ensure_vllm_cuda_lib_priority()
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    use_cuda = torch.cuda.is_available()
    base_model_path = str(resolve_path(args.model_name)) if Path(args.model_name).exists() else args.model_name
    model_ref: str = str(ckpt_path)

    # If checkpoint is adapter-only, merge once to a local full multimodal model folder for GRPO loading.
    if (ckpt_path / "adapter_config.json").exists():
        merged_src = out_dir / "_sft_merged_source"
        if not (merged_src / "config.json").exists():
            base = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
            )
            peft_model = PeftModel.from_pretrained(base, str(ckpt_path))
            merged = peft_model.merge_and_unload()
            ensure_dir(merged_src)
            merged.save_pretrained(str(merged_src))
            proc_tmp = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
            proc_tmp.save_pretrained(str(merged_src))
            copy_multimodal_support_files(base_model_path, merged_src)
        model_ref = str(merged_src)

    train_ds = _MultimodalGRPODataset(rows, max_image_side=args.max_image_side)
    processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_ref,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    model.config.use_cache = False

    def reward_func(prompts, completions, answer_option_id=None, answer_bbox_xyxy_norm=None, **kwargs):
        out = []
        for comp, gt_a, gt_b in zip(completions, answer_option_id, answer_bbox_xyxy_norm):
            pred_text = _completion_to_text(comp)
            score, _ = total_reward(pred_text, str(gt_a), gt_b, reward_w)
            out.append(float(score))
        return out

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1") or 1))
    generation_batch_size = max(args.group_size * world_size, args.group_size)

    grpo_kwargs = dict(
        output_dir=str(out_dir),
        learning_rate=args.learning_rate,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=max(1, args.logging_steps),
        save_steps=max(1, args.save_steps),
        save_total_limit=None,
        report_to="none",
        num_generations=max(2, args.group_size),
        generation_batch_size=generation_batch_size,
        max_completion_length=args.max_completion_length,
        bf16=bool(use_cuda),
        fp16=False,
        use_vllm=False,
        use_cpu=not use_cuda,
        remove_unused_columns=False,
        log_completions=True,
        num_completions_to_print=1,
        chat_template_kwargs={"enable_thinking": False},
    )
    grpo_sig = inspect.signature(GRPOConfig.__init__)
    if "max_prompt_length" in grpo_sig.parameters:
        grpo_kwargs["max_prompt_length"] = None
    if "ddp_find_unused_parameters" in grpo_sig.parameters:
        grpo_kwargs["ddp_find_unused_parameters"] = False

    grpo_args = GRPOConfig(**grpo_kwargs)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_args,
        train_dataset=train_ds,
        processing_class=processor,
        peft_config=lora_cfg,
    )
    train_out = trainer.train()

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    copy_multimodal_support_files(model_ref, adapter_dir)

    with (out_dir / "grpo_metrics.latest.json").open("w", encoding="utf-8") as f:
        json.dump(dict(train_out.metrics or {}), f, ensure_ascii=False, indent=2)

    if args.save_merged:
        base = AutoModelForImageTextToText.from_pretrained(
            model_ref,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = peft_model.merge_and_unload()
        merged_dir = out_dir / "merged_model"
        ensure_dir(merged_dir)
        merged.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        copy_multimodal_support_files(model_ref, merged_dir)
    print("[run_grpo_light] multimodal training complete.")


if __name__ == "__main__":
    main()
