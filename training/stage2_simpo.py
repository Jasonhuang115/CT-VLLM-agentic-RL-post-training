#!/usr/bin/env python3
"""
Stage 2: Image-Conditioned SimPO Training

关键修正 (v3): 用 VLM tokenizer 做完整 [CT图像 + prompt + report] 前向，
只在 assistant 回复部分计算 log-prob。模型"看着"CT 图像判断报告优劣。

之前 v2 用 bare_tok 算无条件 P(text)，acc≈0.50 随机。原因是模型没看到 CT。
"""

import os, sys, json, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 文本提取
# ═══════════════════════════════════════════════════════════════

def extract_text(content):
    """从 DPO 数据格式提取纯文本 (chosen/rejected 字段)"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            t = extract_text(item)
            if t and isinstance(t, str):
                texts.append(t)
        return " ".join(texts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return extract_text(content["content"])
        if "type" in content and content["type"] == "text" and "text" in content:
            return str(content["text"])
    return ""


def extract_image_paths(messages: list) -> list:
    """从 user messages 中提取所有图像路径"""
    paths = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    p = item.get("image", "")
                    if p and os.path.exists(p):
                        paths.append(p)
    return paths


# ═══════════════════════════════════════════════════════════════
# Response-only log-prob
# ═══════════════════════════════════════════════════════════════

def response_log_prob(logits, input_ids, labels_mask):
    """
    只计算 response token 的 log-prob。
    labels_mask: 1=response token, 0=ignore (prompt/image/padding)
    """
    shift_logits = logits[:, :-1, :].contiguous()      # [1, L-1, V]
    shift_labels = input_ids[:, 1:].contiguous()        # [1, L-1]
    shift_mask = labels_mask[:, 1:].contiguous()        # [1, L-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * shift_mask.float()
    return token_lp.sum(-1) / shift_mask.sum(-1).clamp(min=1)


# ═══════════════════════════════════════════════════════════════
# Image-conditioned SimPO loss
# ═══════════════════════════════════════════════════════════════

def simpo_loss_vlm(model, vlm_tok, batch, beta, gamma, device):
    """
    图像条件化 SimPO loss。

    流程:
      1. 从 prompt messages 提取 CT 图像路径
      2. 构建 prompt+chosen 和 prompt+rejected 的完整 messages
      3. VLM tokenizer 处理 (文本 + 图像)
      4. 前向传播 → logits
      5. 用 labels mask 只对 response 部分算 log-prob
      6. SimPO: -log σ(β·(π_c - π_r) - γ)
    """
    losses = []
    accs = []

    for i in range(len(batch["prompt"])):
        prompt_msgs = batch["prompt"][i]
        chosen_text = extract_text(batch["chosen"][i])
        rejected_text = extract_text(batch["rejected"][i])

        image_paths = extract_image_paths(prompt_msgs)
        if not image_paths:
            continue  # 没有图像，跳过这个样本

        # ── 构建完整 messages ──
        assistant_chosen = {
            "role": "assistant",
            "content": [{"type": "text", "text": chosen_text}]
        }
        assistant_rejected = {
            "role": "assistant",
            "content": [{"type": "text", "text": rejected_text}]
        }

        full_chosen = prompt_msgs + [assistant_chosen]
        full_rejected = prompt_msgs + [assistant_rejected]

        # Qwen VL dynamic-res: 1 img → N vision blocks. Repeat paths to match.
        def _match_images(text, img_paths):
            n = text.count('<|vision_start|>')
            imgs = list(img_paths)
            while len(imgs) < n:
                imgs = imgs + img_paths
            return imgs[:n]

        # ── Tokenize: prompt only (用于确定 response 起始位置) ──
        prompt_text = vlm_tok.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        prompt_enc = vlm_tok(
            text=[prompt_text], images=_match_images(prompt_text, image_paths),
            return_tensors="pt")
        prompt_len = prompt_enc.input_ids.shape[1]

        # ── Tokenize: prompt + chosen ──
        text_c = vlm_tok.apply_chat_template(
            full_chosen, tokenize=False, add_generation_prompt=False)
        enc_c = vlm_tok(
            text=[text_c], images=_match_images(text_c, image_paths),
            return_tensors="pt")
        enc_c = {k: v.to(device) for k, v in enc_c.items()}

        # ── Tokenize: prompt + rejected ──
        text_r = vlm_tok.apply_chat_template(
            full_rejected, tokenize=False, add_generation_prompt=False)
        enc_r = vlm_tok(
            text=[text_r], images=_match_images(text_r, image_paths),
            return_tensors="pt")
        enc_r = {k: v.to(device) for k, v in enc_r.items()}

        # ── Labels mask: 只对 response 部分算 log-prob ──
        labels_c = enc_c["input_ids"].clone()
        labels_c[0, :prompt_len] = -100
        mask_c = (labels_c != -100).long()
        mask_c[0, :prompt_len] = 0
        mask_c[0, prompt_len:] = 1

        labels_r = enc_r["input_ids"].clone()
        labels_r[0, :prompt_len] = -100
        mask_r = (labels_r != -100).long()
        mask_r[0, :prompt_len] = 0
        mask_r[0, prompt_len:] = 1

        # ── Forward ──
        out_c = model(
            input_ids=enc_c["input_ids"],
            attention_mask=enc_c.get("attention_mask"),
            pixel_values=enc_c["pixel_values"],
            image_grid_thw=enc_c["image_grid_thw"],
        )
        out_r = model(
            input_ids=enc_r["input_ids"],
            attention_mask=enc_r.get("attention_mask"),
            pixel_values=enc_r["pixel_values"],
            image_grid_thw=enc_r["image_grid_thw"],
        )

        # ── Response-only log-prob ──
        lp_c = response_log_prob(out_c.logits, enc_c["input_ids"], mask_c)
        lp_r = response_log_prob(out_r.logits, enc_r["input_ids"], mask_r)

        # ── SimPO loss ──
        ratio = beta * (lp_c - lp_r) - gamma
        loss = -F.logsigmoid(ratio)
        acc = (lp_c > lp_r).float()

        losses.append(loss)
        accs.append(acc)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.5, device=device)

    return torch.stack(losses).mean(), torch.stack(accs).mean()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage1_sft_v2/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/dpo_v2")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage2_simpo_v3")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)   # VLM forward 佔显存大
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="限制样本数 (0=全部, 用于快速验证)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("  Stage 2: Image-Conditioned SimPO (v3)")
    print("=" * 60)
    print(f"  Model:    {args.model_path}")
    print(f"  Adapter:  {args.adapter}")
    print(f"  Data:     {args.data_dir}")
    print(f"  Beta:     {args.beta}  Gamma: {args.gamma}")
    print(f"  Batch:    {args.batch_size} × {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  Epochs:   {args.epochs}")

    # ── 1. 加载数据 ──
    train_file = os.path.join(args.data_dir, "dpo_train.jsonl")
    val_file = os.path.join(args.data_dir, "dpo_val.jsonl")
    if not os.path.exists(val_file):
        val_file = train_file

    ds = load_dataset("json", data_files={"train": train_file, "validation": val_file})
    if args.max_samples > 0:
        ds["train"] = ds["train"].select(range(min(args.max_samples, len(ds["train"]))))
    print(f"[DATA] train={len(ds['train'])}, val={len(ds['validation'])}")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=lambda batch: batch)

    # ── 2. 加载模型 + VLM tokenizer ──
    print("[MODEL] Loading FastVisionModel (4-bit QLoRA) …")
    model, vlm_tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"[TOKENIZER] VLM tokenizer (image-conditioned)")

    # ── 3. Optimizer ──
    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # ── 4. 训练 ──
    print(f"\n[TRAIN] {total_steps} steps ({steps_per_epoch}/epoch)\n")
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_n = 0

        for bidx, samples in enumerate(pbar):
            batch = {
                "prompt": [s["prompt"] for s in samples],
                "chosen": [s["chosen"] for s in samples],
                "rejected": [s["rejected"] for s in samples],
            }

            # 检查是否有图像（有些样本可能缺 PNG）
            has_images = any(extract_image_paths(p) for p in batch["prompt"])
            if not has_images:
                continue

            loss, acc = simpo_loss_vlm(model, vlm_tok, batch,
                                       beta=args.beta, gamma=args.gamma, device=device)
            if loss.item() == 0.0:
                continue

            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (bidx + 1) % args.grad_accum == 0 or (bidx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

                global_step += 1
                epoch_loss += accum_loss * args.grad_accum
                epoch_acc += acc.item()
                epoch_n += 1

                pbar.set_postfix(loss=f"{accum_loss * args.grad_accum:.4f}",
                                 acc=f"{acc.item():.3f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        avg_l = epoch_loss / max(epoch_n, 1)
        avg_a = epoch_acc / max(epoch_n, 1)
        print(f"[EPOCH {epoch+1}] loss={avg_l:.4f}  acc={avg_a:.4f}")

    # ── 5. 保存 ──
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    vlm_tok.save_pretrained(adapter_out)

    cfg = {"stage": "simpo_v3", "beta": args.beta, "gamma": args.gamma,
           "epochs": args.epochs, "lr": args.lr,
           "batch_size": args.batch_size, "grad_accum": args.grad_accum}
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[DONE]  →  {adapter_out}")
    print(f"Next: python training/stage3_grpo.py --adapter {adapter_out}")


if __name__ == "__main__":
    main()
