#!/usr/bin/env python3
"""
Stage 1: SFT with FastVisionModel + Manual Training Loop
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
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,}")

    # ── 3. Optimizer ──
    total_batches = len(ds["train"])
    steps_per_epoch = max(total_batches // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
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
