#!/usr/bin/env python3
"""
Stage 2: DPO/SimPO 偏好对齐

在 Stage 1 SFT 基础上, 使用偏好对进一步优化报告质量和事实准确性。
默认使用 SimPO (无需参考模型, 节省 VRAM)。

使用方式:
  python training/stage2_dpo.py \
    --adapter /root/autodl-tmp/outputs/stage1_sft/lora_adapter \
    --data_dir /root/autodl-tmp/data/dpo \
    --output /root/autodl-tmp/outputs/stage2_dpo

预计: ~1.5-2h on RTX 5090, VRAM ~15GB
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import PeftModel, LoraConfig
from unsloth import FastVisionModel

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: DPO/SimPO 训练")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Stage 1 LoRA adapter 路径")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_dir", type=str,
                        default="/root/autodl-tmp/data/dpo")
    parser.add_argument("--output", type=str,
                        default="/root/autodl-tmp/outputs/stage2_dpo")
    parser.add_argument("--method", type=str, default="simpo",
                        choices=["simpo", "dpo"],
                        help="偏好优化方法: simpo (无需参考模型) / dpo")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta (simpo 中该值映射为 gamma)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print(f"  Stage 2: {args.method.upper()} 偏好对齐")
    print("=" * 60)
    print(f"  Adapter: {args.adapter}")
    print(f"  方法: {args.method}")
    print(f"  Beta: {args.beta}")

    # ---- 1. 加载 DPO 数据 ----
    train_file = os.path.join(args.data_dir, "dpo_train.jsonl")
    if not os.path.exists(train_file):
        print(f"[WARN] DPO 偏好对数据不存在: {train_file}")
        print("  请先运行: python data/dpo_dataset_builder.py")
        print("  或使用简化的偏好对构建方式: 基于 SFT 数据自动生成")
        return

    dataset = load_dataset("json", data_files={"train": train_file})
    print(f"[INFO] 偏好对: {len(dataset['train'])}")

    # DPO 数据格式: {prompt, chosen, rejected, [image]}
    # 检查格式
    sample = dataset["train"][0]
    required = ["prompt", "chosen", "rejected"]
    missing = [k for k in required if k not in sample]
    if missing:
        print(f"[ERROR] 数据缺少字段: {missing}")
        return

    # ---- 2. 加载基础模型 + Stage 1 adapter ----
    print("[INFO] 加载基础模型...")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # 加载 Stage 1 adapter
    print(f"[INFO] 加载 Stage 1 adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    # 为 DPO 添加 LoRA adapter
    # DPO 需要一个 reference model (除非用 SimPO)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    )

    tokenizer.padding_side = "left"  # DPO 需要 left padding

    # ---- 3. 训练参数 ----
    if args.method == "simpo":
        # SimPO: 不需要参考模型
        training_args = DPOConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="wandb",
            run_name="stage2_simpo",
            # SimPO 特定参数
            loss_type="simpo",
            simpo_gamma=args.beta,     # SimPO 的 margin 参数
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            # 禁用参考模型
            precompute_ref_log_probs=False,
        )
        ref_model = None
    else:
        # 标准 DPO: 需要参考模型 (Stage 1 adapter 作为参考)
        training_args = DPOConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            beta=args.beta,
            bf16=True,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="wandb",
            run_name="stage2_dpo",
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
        )
        # 加载参考模型
        print("[INFO] 加载参考模型 (Stage 1 adapter 作为参考)...")
        ref_model, _ = FastVisionModel.from_pretrained(
            args.model_path,
            load_in_4bit=True,
        )
        ref_model = PeftModel.from_pretrained(ref_model, args.adapter)

    # ---- 4. 数据预处理 ----
    def format_dpo_sample(example):
        """格式化 DPO 样本: {prompt, chosen, rejected} 格式"""
        # prompt 部分: user message + image
        prompt_messages = example.get("prompt_messages", [])

        if prompt_messages:
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: 直接用 prompt 字段
            prompt_text = example["prompt"]

        return {
            "prompt": prompt_text,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    # Apply formatting
    # 注意: 对于 VLM DPO, 图片需要在 prompt 中保留
    # 使用 preprocessing_num_workers 加速
    print("[INFO] 格式化数据...")
    dataset["train"] = dataset["train"].map(format_dpo_sample, remove_columns=dataset["train"].column_names)

    # ---- 5. Trainer ----
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
    )

    # ---- 6. 开训 ----
    print("\n[INFO] 开始 DPO 训练...")
    print(f"  VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

    trainer.train()

    # ---- 7. 保存 ----
    adapter_path = os.path.join(args.output, "lora_adapter")
    trainer.save_model(adapter_path)
    print(f"\n[DONE] LoRA adapter 已保存: {adapter_path}")

    print("\n下一步:")
    print(f"  python training/stage3_grpo.py --adapter {adapter_path}")


if __name__ == "__main__":
    main()
