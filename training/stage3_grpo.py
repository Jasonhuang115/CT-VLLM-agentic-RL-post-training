#!/usr/bin/env python3
"""
Stage 3: Image-Conditioned GRPO (REINFORCE with reward)

Fixes v2 bug: images were discarded during prompt building (build_prompt).
Now uses same image-conditioned pattern as stage1_sft.py and stage2_simpo.py.

Uses manual REINFORCE loop (not TRL GRPOTrainer) to guarantee VLM compatibility.
"""

import os, sys, json, argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_functions import composite_reward


def extract_image_paths(messages: list) -> list:
    paths = []
    for m in messages:
        if m.get("role") != "user": continue
        content = m.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    p = item.get("image", "")
                    if p and os.path.exists(p):
                        paths.append(p)
    return paths


def match_images(text: str, img_paths: list) -> list:
    """Repeat image paths to match Qwen dynamic-res placeholder count."""
    n = text.count('<|vision_start|>')
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage2_simpo_v3/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/sft")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage3_grpo_v3")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_beta", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("  Stage 3: Image-Conditioned GRPO (v3)")
    print("=" * 60)

    # ── 1. Load data ──
    train_file = os.path.join(args.data_dir, "sft_train.jsonl")
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    total = len(ds)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, total)))
    print(f"[DATA] {len(ds)} samples (from {total})")

    # ── 2. Load model ──
    print("[MODEL] Loading FastVisionModel …")
    model, tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = PeftModel.from_pretrained(model, args.adapter)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,}")

    # ── 3. Optimizer ──
    steps_per_epoch = max(len(ds) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"[TRAIN] ~{steps_per_epoch} steps/epoch\n")

    global_step = 0

    for epoch in range(args.epochs):
        indices = torch.randperm(len(ds)).tolist()
        pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_r, epoch_loss, epoch_n = 0.0, 0.0, 0

        for start in pbar:
            batch_rewards, batch_losses = [], []

            for idx in indices[start:start + args.batch_size]:
                s = ds[int(idx)]
                msgs = s["messages"]
                metadata = s.get("metadata", {})
                img_paths = extract_image_paths(msgs)
                if not img_paths: continue

                # Build prompt (user messages only, with images)
                user_msgs = [m for m in msgs if m["role"] == "user"]
                prompt_text = tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)

                # ── Generate ──
                with torch.no_grad():
                    gen_enc = tok(text=[prompt_text], images=match_images(prompt_text, img_paths),
                                  return_tensors="pt")
                    gen_enc = {k: v.to(device) for k, v in gen_enc.items()}
                    gen_ids = model.generate(**gen_enc, max_new_tokens=args.max_new_tokens,
                                             do_sample=True, temperature=args.temperature,
                                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
                new_ids = gen_ids[0, gen_enc["input_ids"].shape[1]:]
                completion = tok.decode(new_ids, skip_special_tokens=True)

                # ── Reward ──
                reward_gt = build_reward_gt(metadata)
                reward = composite_reward(completion, reward_gt)
                batch_rewards.append(reward)

                # ── Full forward for log-prob ──
                full_msgs = msgs + [{"role": "assistant", "content": completion}]
                full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
                full_enc = tok(text=[full_text], images=match_images(full_text, img_paths),
                              return_tensors="pt")
                full_enc = {k: v.to(device) for k, v in full_enc.items()}

                out = model(**full_enc)
                logits = out.logits  # [1, L, V]

                # Find response boundary
                prompt_full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompt_enc = tok(text=[prompt_full_text],
                                images=match_images(prompt_full_text, img_paths), return_tensors="pt")
                prompt_len = prompt_enc.input_ids.shape[1]

                # Log-prob of response only
                shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
                shift_labels = full_enc["input_ids"][:, prompt_len:].contiguous()
                token_lp = F.log_softmax(shift_logits, dim=-1)
                token_lp = token_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                gen_log_prob = token_lp.sum()

                # ── REINFORCE loss ──
                pg_loss = -reward * gen_log_prob

                # KL regularization (simple L2 penalty on log-prob)
                kl = args.kl_beta * (gen_log_prob ** 2)

                batch_losses.append(pg_loss + kl)

            if not batch_losses: continue

            loss = torch.stack(batch_losses).mean() / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0 or start + args.batch_size >= len(indices):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

                avg_r = sum(batch_rewards) / len(batch_rewards)
                epoch_r += avg_r; epoch_loss += loss.item() * args.grad_accum; epoch_n += 1
                pbar.set_postfix(reward=f"{avg_r:.3f}", loss=f"{loss.item() * args.grad_accum:.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        print(f"[EPOCH {epoch+1}] avg_reward={epoch_r/max(epoch_n,1):.4f}  loss={epoch_loss/max(epoch_n,1):.4f}")

    # ── Save ──
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)
    json.dump({"stage": "grpo_v3"}, open(os.path.join(args.output, "training_config.json"), "w"), indent=2)
    print(f"\n[DONE] → {adapter_out}")


if __name__ == "__main__":
    main()
