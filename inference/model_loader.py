#!/usr/bin/env python3
"""模型加载 & QLoRA 合并工具"""

import os
import torch
from peft import PeftModel
from unsloth import FastVisionModel


def load_model_with_adapter(model_path: str, adapter_path: str = None, load_in_4bit: bool = True):
    """加载模型 + 可选 LoRA adapter"""
    model, tokenizer = FastVisionModel.from_pretrained(
        model_path, load_in_4bit=load_in_4bit,
    )
    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def merge_and_save(model_path: str, adapter_path: str, output_path: str):
    """合并 LoRA adapter 并保存为完整模型"""
    model, tokenizer = load_model_with_adapter(model_path, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[DONE] 合并模型已保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    merge_and_save(args.model, args.adapter, args.output)
