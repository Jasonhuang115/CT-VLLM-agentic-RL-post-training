#!/usr/bin/env python3
"""
Stage 1: SFT with FastVisionModel + Manual Training Loop

P2 (PLAN.md): 解冻 vision connector + ViT 最后 N 层, 分组 LR
"""

import os, sys, json, argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from unsloth import FastVisionModel
from tqdm import tqdm


def extract_image_paths(messages: list) -> list:
    """Extract image paths from user messages."""
    paths = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    p = item.get("image", "")
                    if p and os.path.exists(p):
                        paths.append(p)
    return paths


def unfreeze_vision_p2(model, vit_layers: int = 2):
    """P2: 解冻 vision projector + ViT 最后 N 层. 返回 vit_param_names 集合."""
    vit_params = set()

    # 1. 打印所有 visual 相关参数名 (首次运行诊断用)
    visual_names = [n for n, _ in model.named_parameters() if "visual" in n.lower()]
    print(f"[VISION] Found {len(visual_names)} visual parameters")

    # 2. 解冻 vision-language projector/merger
    merger_names = [n for n in visual_names if "merger" in n.lower()]
    if not merger_names:
        # Qwen2-VL 可能用其他名字, fallback: 所有 visual 中非 blocks 的参数
        merger_names = [n for n in visual_names if "block" not in n.lower()]
    for n, p in model.named_parameters():
        if any(m in n for m in merger_names[:3]):  # 前3个作为模式匹配
            p.requires_grad = True
            vit_params.add(n)
    print(f"[VISION] Projector/merger unfrozen: {len([n for n in vit_params if 'merger' in n.lower() or 'block' not in n.lower()])} params")

    # 3. 解冻 ViT 最后 N 层
    block_names = sorted([n for n in visual_names if "block" in n.lower() or "layer" in n.lower()])
    if block_names:
        # 提取层号
        import re
        layer_ids = set()
        for bn in block_names:
            m = re.search(r'(?:blocks?|layers?)\.(\d+)', bn)
            if m:
                layer_ids.add(int(m.group(1)))
        if layer_ids:
            last_n = sorted(layer_ids)[-vit_layers:]
            print(f"[VISION] ViT has {len(layer_ids)} layers, unfreezing last {vit_layers}: {last_n}")
            for n, p in model.named_parameters():
                for lid in last_n:
                    if f".{lid}." in n or f"blocks.{lid}" in n or f"layers.{lid}" in n:
                        p.requires_grad = True
                        vit_params.add(n)
                        break

    vit_count = len(vit_params)
    vit_params_total = sum(model.get_parameter(n).numel() for n in vit_params if model.get_parameter(n) is not None)
    print(f"[VISION] Total unfrozen: {vit_count} params, {vit_params_total/1e6:.1f}M")
    return vit_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/sft")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage1_sft_v3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    # P2: Vision unfreeze
    parser.add_argument("--unfreeze_vision", type=int, default=1,
                        help="1=unfreeze projector+ViT last N layers, 0=all frozen (old behavior)")
    parser.add_argument("--unfreeze_vit_layers", type=int, default=2,
                        help="Number of ViT layers to unfreeze (default 2)")
    parser.add_argument("--vit_lr_ratio", type=float, default=0.1,
                        help="ViT LR = lr * vit_lr_ratio (default 0.1)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("  Stage 1: SFT (FastVisionModel)")
    print("=" * 60)

    # ── 1. Load data ──
    train_file = os.path.join(args.data_dir, "sft_train.jsonl")
    val_file = os.path.join(args.data_dir, "sft_val.jsonl")
    ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    print(f"[DATA] train={len(ds['train'])}, val={len(ds['validation'])}")

    # ── 2. Load model ──
    print("[MODEL] Loading FastVisionModel …")
    model, tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05)

    # P2: 解冻 vision projector + ViT 最后 N 层
    vit_param_names = set()
    if args.unfreeze_vision:
        vit_param_names = unfreeze_vision_p2(model, args.unfreeze_vit_layers)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(1 for n, p in model.named_parameters() if p.requires_grad and n not in vit_param_names)
    n_vit = sum(1 for n, p in model.named_parameters() if p.requires_grad and n in vit_param_names)
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,} ({n_lora} LoRA + {n_vit} ViT)")

    # ── 3. Optimizer (P2: 分组 LR) ──
    total_batches = len(ds["train"])
    steps_per_epoch = max(total_batches // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in vit_param_names]
    vit_params = [p for n, p in model.named_parameters() if p.requires_grad and n in vit_param_names]

    if vit_params:
        opt = torch.optim.AdamW([
            {"params": lora_params, "lr": args.lr},
            {"params": vit_params, "lr": args.lr * args.vit_lr_ratio},
        ], weight_decay=0.01)
        print(f"[OPTIM] Grouped LR — LoRA: {args.lr:.1e}, ViT: {args.lr * args.vit_lr_ratio:.1e}")
    else:
        opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    print(f"[TRAIN] ~{steps_per_epoch} steps/epoch, {total_steps} total\n")

    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss_sum, epoch_n = 0.0, 0
        indices = torch.randperm(len(ds["train"])).tolist()
        pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")

        accum_loss = 0.0
        for start in pbar:
            batch_idx = indices[start:start + args.batch_size]
            batch_losses = []

            for idx in batch_idx:
                s = ds["train"][int(idx)]
                msgs = s["messages"]

                # Convert to text via chat template
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

                # Count vision placeholders and get image paths
                n_vision = txt.count('<|vision_start|>')
                img_paths = extract_image_paths(msgs)

                if n_vision == 0 or not img_paths:
                    continue

                # Qwen VL dynamic-res: 1 image → N vision blocks. Repeat paths to match.
                images_for_tok = []
                for p in img_paths:
                    images_for_tok.append(p)
                # If placeholder count > image count, repeat the images
                while len(images_for_tok) < n_vision:
                    images_for_tok = images_for_tok + images_for_tok
                images_for_tok = images_for_tok[:n_vision]

                enc = tok(text=[txt], images=images_for_tok, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}

                # ── Response-only loss mask ──
                # Build prompt-only encoding to find the prompt/response boundary
                prompt_txt = tok.apply_chat_template(
                    [m for m in msgs if m["role"] != "assistant"],
                    tokenize=False, add_generation_prompt=True)
                prompt_enc = tok(text=[prompt_txt], images=images_for_tok, return_tensors="pt")
                prompt_len = prompt_enc.input_ids.shape[1]

                labels = enc["input_ids"].clone()
                # Mask prompt tokens (including image placeholders)
                labels[:, :prompt_len] = -100

                # Use model's internal loss (memory-efficient fused kernel), convert sum→mean
                out = model(**enc, labels=labels)
                loss_sum = out.loss  # sum reduction over non-masked tokens
                n_tokens = (labels[:, 1:] != -100).sum().item()
                loss = loss_sum / max(n_tokens, 1)

                batch_losses.append(loss)

            if not batch_losses:
                continue

            # Track raw per-sample loss BEFORE grad_accum division
            loss_raw = torch.stack(batch_losses).mean()
            accum_loss += loss_raw.item()

            loss = loss_raw / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0 or start + args.batch_size >= len(indices):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

                global_step += 1
                avg_loss = accum_loss / args.grad_accum  # true per-sample mean CE
                epoch_loss_sum += avg_loss
                epoch_n += 1

                pbar.set_postfix(loss=f"{avg_loss:.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        avg_l = epoch_loss_sum / max(epoch_n, 1)
        print(f"[EPOCH {epoch+1}] loss={avg_l:.4f}")

    # ── Save ──
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)

    cfg = {"stage": "sft_v3", "epochs": args.epochs, "lr": args.lr}
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[DONE] → {adapter_out}")


if __name__ == "__main__":
    main()
