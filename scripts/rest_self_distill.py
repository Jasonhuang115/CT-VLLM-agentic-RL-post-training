#!/usr/bin/env python3
"""
P4: ReST 自蒸馏 (PLAN.md Phase 4)

Generate → Filter → SFT 循环:
  1. 用当前模型对每个 CT 生成 4 条报告
  2. composite_reward 打分
  3. 取 top-50% 高分报告
  4. 原始数据 + 高分报告 → 1 epoch 保守 SFT

使用方式:
  python scripts/rest_self_distill.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter /root/autodl-tmp/outputs/stage1_full_v2/lora_adapter \
    --data_dir /root/autodl-tmp/data/sft_full_v2 \
    --output /root/autodl-tmp/outputs/stage1_rest_v1
"""

import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from reward_functions import composite_reward


def extract_image_paths(messages: list) -> list:
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


def match_images(text: str, img_paths: list) -> list:
    n = text.count("<|vision_start|>")
    imgs = list(img_paths)
    while len(imgs) < n:
        imgs = imgs + img_paths
    return imgs[:n]


def build_reward_gt(metadata: dict) -> dict:
    d = metadata.get("diameter_mm", 10)
    return {
        "long_diameter_mm": d,
        "short_diameter_mm": d * 0.8,
        "malignancy_level": metadata.get("malignancy", 3),
        "characteristics": {
            "texture": metadata.get("texture", 5),
            "malignancy": metadata.get("malignancy", 3),
            "margin": metadata.get("margin", 3),
            "spiculation": metadata.get("spiculation", 1),
            "calcification": metadata.get("calcification", 6),
            "lobulation": metadata.get("lobulation", 1),
            "sphericity": metadata.get("sphericity", 3),
            "subtlety": metadata.get("subtlety", 3),
            "internalStructure": metadata.get("internalStructure", 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="ReST Self-Distillation")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage1_full_v2/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/sft_full_v2")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage1_rest_v1")
    parser.add_argument("--n_generations", type=int, default=4,
                        help="每个样本生成几条报告")
    parser.add_argument("--top_frac", type=float, default=0.5,
                        help="保留 top-N 高分报告的比例")
    parser.add_argument("--min_reward", type=float, default=0.3,
                        help="最低 reward 阈值，低于此值不保留")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("  P4: ReST 自蒸馏")
    print("=" * 60)
    print(f"  生成数/样本: {args.n_generations}")
    print(f"  保留比例: {args.top_frac:.0%}")
    print(f"  最低 reward: {args.min_reward}")

    # ── 1. Load data ──
    train_file = os.path.join(args.data_dir, "sft_train.jsonl")
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    total = len(ds)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, total)))
    print(f"[DATA] {len(ds)} 条训练数据 (from {total})")

    # ── 2. Load model ──
    print("[MODEL] Loading …")
    model, tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    if os.path.exists(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  Loaded adapter: {args.adapter}")
    model.eval()
    # Enable LoRA for generation but disable grads
    for n, p in model.named_parameters():
        p.requires_grad = False

    # ── 3. Generate & Score ──
    print("[GENERATE] 生成候选报告 …")
    candidates = []  # (data_idx, completion_text, reward, original_sample)

    pbar = tqdm(range(len(ds)), desc="生成+打分")
    for idx in pbar:
        s = ds[int(idx)]
        msgs = s["messages"]
        metadata = s.get("metadata", {})
        img_paths = extract_image_paths(msgs)
        if not img_paths:
            continue

        user_msgs = [m for m in msgs if m["role"] != "assistant"]
        prompt_text = tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)

        gt = build_reward_gt(metadata)

        for gen_i in range(args.n_generations):
            with torch.no_grad():
                gen_enc = tok(text=[prompt_text], images=match_images(prompt_text, img_paths),
                              return_tensors="pt")
                gen_enc = {k: v.to(device) for k, v in gen_enc.items()}
                gen_ids = model.generate(**gen_enc, max_new_tokens=args.max_new_tokens,
                                         do_sample=True, temperature=args.temperature,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new_ids = gen_ids[0, gen_enc["input_ids"].shape[1]:]
            completion = tok.decode(new_ids, skip_special_tokens=True)
            reward = composite_reward(completion, gt)

            candidates.append({
                "idx": idx,
                "completion": completion,
                "reward": float(reward),
                "sample": s,
            })

        if candidates:
            recent = [c["reward"] for c in candidates[-args.n_generations:]]
            pbar.set_postfix(r_mean=f"{sum(recent)/len(recent):.3f}", r_max=f"{max(recent):.3f}")

    # ── 4. Filter top-k ──
    print(f"\n[FILTER] 总候选: {len(candidates)}")
    candidates.sort(key=lambda x: x["reward"], reverse=True)

    # Per-sample top-k: 每个样本保留最好的 ceil(top_frac * n_generations) 条
    n_keep_per_sample = max(1, int(args.top_frac * args.n_generations))
    by_sample = {}
    for c in candidates:
        sidx = c["idx"]
        if sidx not in by_sample:
            by_sample[sidx] = []
        by_sample[sidx].append(c)

    kept = []
    for sidx, cands in by_sample.items():
        for c in cands[:n_keep_per_sample]:
            if c["reward"] >= args.min_reward:
                kept.append(c)

    rewards = [c["reward"] for c in kept]
    print(f"  保留: {len(kept)} 条 (min={min(rewards):.3f}, med={sorted(rewards)[len(rewards)//2]:.3f}, max={max(rewards):.3f})")

    # ── 5. Build augmented data ──
    print("[DATA] 构建增强数据集 …")
    new_samples = []
    for c in kept:
        s = c["sample"]
        new_samples.append({
            "messages": s["messages"][:-1] + [
                {"role": "assistant", "content": [{"type": "text", "text": c["completion"]}]}
            ],
            "metadata": {**s.get("metadata", {}),
                         "source": "rest_distilled",
                         "rest_reward": c["reward"]},
        })

    # 原始数据 + 蒸馏数据
    original_samples = []
    for idx in range(len(ds)):
        s = ds[int(idx)]
        original_samples.append({
            "messages": s["messages"],
            "metadata": {**s.get("metadata", {}), "source": "original"},
        })

    augmented = original_samples + new_samples
    random.shuffle(augmented)

    augmented_path = os.path.join(args.output, "rest_augmented.jsonl")
    with open(augmented_path, "w") as f:
        for s in augmented:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  增强数据集: {len(augmented)} 条 → {augmented_path}")
    print(f"    原始: {len(original_samples)}, 蒸馏: {len(new_samples)}")

    # Save stats
    stats = {
        "n_original": len(original_samples),
        "n_distilled": len(new_samples),
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "reward_min": float(min(rewards)),
        "reward_median": float(sorted(rewards)[len(rewards)//2]),
        "reward_max": float(max(rewards)),
    }
    with open(os.path.join(args.output, "rest_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ── 6. Conservative SFT on augmented data ──
    print("\n[TRAIN] 保守 SFT (lr=1e-4, 1 epoch) …")

    # Reload model in train mode
    model, tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    if os.path.exists(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable: {sum(p.numel() for p in trainable):,}")

    steps_per_epoch = max(len(augmented) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(augmented)
        pbar = tqdm(range(0, len(augmented), 1), desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss_sum, epoch_n = 0.0, 0
        accum_loss = 0.0
        batch_losses = []

        for i, start in enumerate(pbar):
            if start >= len(augmented):
                break
            s = augmented[start]
            msgs = s["messages"]
            img_paths = extract_image_paths(msgs)
            if not img_paths:
                continue

            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            n_vision = txt.count("<|vision_start|>")
            if n_vision == 0:
                continue

            # Match images to placeholders
            imgs = list(img_paths)
            while len(imgs) < n_vision:
                imgs = imgs + img_paths
            imgs = imgs[:n_vision]

            enc = tok(text=[txt], images=imgs, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            # Response-only mask
            prompt_txt = tok.apply_chat_template(
                [m for m in msgs if m["role"] != "assistant"],
                tokenize=False, add_generation_prompt=True)
            prompt_enc = tok(text=[prompt_txt], images=imgs, return_tensors="pt")
            prompt_len = prompt_enc.input_ids.shape[1]

            labels = enc["input_ids"].clone()
            labels[:, :prompt_len] = -100

            out = model(**enc, labels=labels)
            loss_sum = out.loss
            n_tokens = (labels[:, 1:] != -100).sum().item()
            loss = loss_sum / max(n_tokens, 1)

            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()
            batch_losses.append(loss.item() * args.grad_accum)

            if (i + 1) % args.grad_accum == 0 or start + 1 >= len(augmented):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

                global_step += 1
                avg_loss = sum(batch_losses) / len(batch_losses)
                epoch_loss_sum += avg_loss
                epoch_n += 1
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                accum_loss = 0.0
                batch_losses = []

        avg_l = epoch_loss_sum / max(epoch_n, 1)
        print(f"[EPOCH {epoch+1}] loss={avg_l:.4f}")

    # ── 7. Save ──
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)

    cfg = {"stage": "rest_v1", "n_original": stats["n_original"],
           "n_distilled": stats["n_distilled"],
           "reward_median": stats["reward_median"]}
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[DONE] → {adapter_out}")
    print(f"  Reward median: {stats['reward_median']:.3f}")
    print(f"  Augmented data: {len(augmented)} 条")


if __name__ == "__main__":
    main()
