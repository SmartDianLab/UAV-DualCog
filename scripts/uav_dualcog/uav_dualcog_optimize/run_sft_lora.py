#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except Exception:
        return False


class _MultimodalPromptCompletionDataset:
    def __init__(self, rows: List[Dict[str, Any]], max_image_side: int) -> None:
        self.rows = rows
        self.max_image_side = max_image_side

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        return {
            "prompt": copy.deepcopy(list(row.get("prompt", []) or [])),
            "completion": copy.deepcopy(list(row.get("completion", []) or [])),
            "images": [load_rgb_image(p, max_side=self.max_image_side) for p in list(row.get("images", []) or [])],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal LoRA SFT for UAV-DualCog optimization.")
    parser.add_argument("--train-jsonl", default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm/uav_dualcog_train.latest.jsonl")
    parser.add_argument("--valid-jsonl", default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm/uav_dualcog_valid.latest.jsonl")
    parser.add_argument("--output-dir", default="task_pipeline_data/UAV-DualCog-Train-V1/optimization_vllm/checkpoints/sft")
    parser.add_argument("--model-name", default="vllm/models/qwen3_5-4b")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override training steps; -1 means epoch-based.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-image-side", type=int, default=480, help="Resize each training image so its long side does not exceed this value. 480 is the recommended default; 720 is optional but significantly heavier.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-merged", action="store_true", help="Also export merged full model for deployment.")
    parser.add_argument("--smoke-limit", type=int, default=0, help="Read only first N rows for smoke run; 0 means full dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write plan only, without training.")
    args = parser.parse_args()

    train_path = resolve_path(args.train_jsonl)
    valid_path = resolve_path(args.valid_jsonl)
    out_dir = resolve_path(args.output_dir)
    ensure_dir(out_dir)
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not valid_path.exists():
        raise FileNotFoundError(valid_path)

    train_rows = _load_jsonl(train_path, limit=args.smoke_limit if args.smoke_limit > 0 else 0)
    valid_rows = _load_jsonl(valid_path, limit=max(8, args.smoke_limit // 4) if args.smoke_limit > 0 else 0)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "mode": "dry-run" if args.dry_run else "train",
        "multimodal": True,
        "model_name": args.model_name,
        "train_jsonl": str(train_path),
        "valid_jsonl": str(valid_path),
        "train_rows_loaded": len(train_rows),
        "valid_rows_loaded": len(valid_rows),
        "hyperparams": {
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_seq_length": args.max_seq_length,
            "max_image_side": args.max_image_side,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    }
    with (out_dir / "sft_plan.latest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        print("[run_sft_lora] dry-run complete. plan saved.")
        return

    if not _safe_import_train_stack():
        raise RuntimeError(
            "Training stack missing. Please install transformers/peft/trl/accelerate in the current environment."
        )

    ensure_vllm_cuda_lib_priority()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    model_path = str(resolve_path(args.model_name)) if Path(args.model_name).exists() else args.model_name
    use_cuda = torch.cuda.is_available()

    train_ds = _MultimodalPromptCompletionDataset(train_rows, max_image_side=args.max_image_side)
    valid_ds = _MultimodalPromptCompletionDataset(valid_rows, max_image_side=args.max_image_side)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if getattr(processor.tokenizer, "pad_token", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=max(1, args.logging_steps),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=max(1, args.logging_steps),
        save_steps=max(1, args.save_steps),
        save_total_limit=None,
        bf16=bool(use_cuda),
        fp16=False,
        gradient_checkpointing=bool(use_cuda),
        remove_unused_columns=False,
        report_to="none",
        use_cpu=not use_cuda,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        # Avoid per-step autograd graph traversal for unused-parameter detection in DDP.
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=processor,
    )

    train_out = trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    copy_multimodal_support_files(model_path, adapter_dir)

    metrics = dict(train_out.metrics or {})
    with (out_dir / "sft_train_metrics.latest.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.save_merged:
        merged_dir = out_dir / "merged_model"
        ensure_dir(merged_dir)
        base = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        )
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        copy_multimodal_support_files(model_path, merged_dir)

    print("[run_sft_lora] multimodal training complete.")


if __name__ == "__main__":
    main()
