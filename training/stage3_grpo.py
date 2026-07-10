#!/usr/bin/env python3
"""
Stage 3: True GRPO (Group Relative Policy Optimization)

每组 G 条生成结果内部计算相对优势，KL 惩罚使用 frozen 参考模型。

与旧版差异:
  - G=4 条/样本 → 组内归一化优势 (不是单样本 REINFORCE)
  - KL 用参考模型: KL(π_θ || π_ref) 无偏估计
  - 图像条件化生成 + 完整 log-prob 前向 (v3)

参考: DeepSeekMath (2024), MedFact-R1 (2025)
"""

import os, sys, json, argparse, math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_functions import composite_reward
from wandb_utils import get_logger


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


def response_log_prob(logits, input_ids, prompt_len):
    """只计算 response token 的 log-prob (sum, not mean)"""
    shift_logits = logits[:, prompt_len-1:-1, :].contiguous()  # [1, L_resp, V]
    shift_labels = input_ids[:, prompt_len:].contiguous()        # [1, L_resp]
    token_lp = F.log_softmax(shift_logits, dim=-1)
    token_lp = token_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(-1)  # scalar


@torch.no_grad()
def generate_g_responses(model, tok, prompt_text, img_paths, device,
                          n_generations, max_new_tokens, temperature):
    """批量生成 G 条回复 (共享 prompt KV-cache)"""
    matched = match_images(prompt_text, img_paths)
    enc = tok(text=[prompt_text], images=matched, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    gen_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature,
        num_return_sequences=n_generations,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

    completions = []
    for gi in range(n_generations):
        new_ids = gen_ids[gi, prompt_len:]
        completions.append(tok.decode(new_ids, skip_special_tokens=True))

    return completions, prompt_text, img_paths


@torch.no_grad()
def compute_log_prob(model, tok, prompt_msgs, completion_text, img_paths, device):
    """计算 completion 在给定 prompt+image 下的 log-prob (sum)"""
    assistant_msg = {"role": "assistant",
                     "content": [{"type": "text", "text": completion_text}]}
    full_msgs = prompt_msgs + [assistant_msg]

    full_text = tok.apply_chat_template(full_msgs, tokenize=False,
                                         add_generation_prompt=False)
    matched = match_images(full_text, img_paths)
    full_enc = tok(text=[full_text], images=matched, return_tensors="pt")
    full_enc = {k: v.to(device) for k, v in full_enc.items()}

    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                           add_generation_prompt=True)
    prompt_enc = tok(text=[prompt_text],
                      images=match_images(prompt_text, img_paths),
                      return_tensors="pt")
    prompt_len = prompt_enc.input_ids.shape[1]

    out = model(**full_enc)
    return response_log_prob(out.logits, full_enc["input_ids"], prompt_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter",
                        default="/root/autodl-tmp/outputs/stage2_dpo/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/sft_nohint")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage3_grpo")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="极小学习率，GRPO 方差大需要保守")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--n_generations", type=int, default=4,
                        help="每组生成 G 条 (真 GRPO 至少 4)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_beta", type=float, default=0.04,
                        help="KL 惩罚权重 (DeepSeek 默认 0.04)")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--no_4bit", action="store_true",
                        help="禁用 4-bit 量化 (BF16, 推荐)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    use_4bit = not args.no_4bit

    print("=" * 60)
    print("  Stage 3: True GRPO (G={})".format(args.n_generations))
    print("=" * 60)
    print(f"  KL beta: {args.kl_beta}  LR: {args.lr:.1e}")
    print(f"  4-bit: {use_4bit}")

    wb_logger = get_logger("grpo", args.output, config=args)

    # ── 1. Load data ──
    train_file = os.path.join(args.data_dir, "sft_train.jsonl")
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    total = len(ds)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, total)))
    print(f"[DATA] {len(ds)} samples (from {total})")

    # ── 2. Load training model ──
    print(f"[MODEL] Loading training model (4bit={use_4bit}) …")
    model, tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=use_4bit,
        use_gradient_checkpointing="unsloth")
    if os.path.exists(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  Loaded adapter: {args.adapter}")
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,}")

    # ── 3. Load reference model (frozen) ──
    print(f"[MODEL] Loading reference model (frozen) …")
    ref_model, _ = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=use_4bit,
        use_gradient_checkpointing="unsloth")
    if os.path.exists(args.adapter):
        ref_model = PeftModel.from_pretrained(ref_model, args.adapter)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"[MODEL] Reference model frozen")

    # ── 4. Optimizer ──
    steps_per_epoch = max(len(ds) // (args.grad_accum * args.n_generations), 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"[TRAIN] ~{steps_per_epoch} steps/epoch, {total_steps} total\n")

    global_step = 0

    for epoch in range(args.epochs):
        indices = torch.randperm(len(ds)).tolist()
        pbar = tqdm(range(0, len(indices), args.batch_size),
                     desc=f"GRPO Epoch {epoch+1}/{args.epochs}")

        epoch_r, epoch_loss, epoch_n = 0.0, 0.0, 0
        accum_r, batch_count = 0.0, 0

        for start in pbar:
            batch_indices = indices[start:start + args.batch_size]
            group_losses = []
            group_rewards = []

            for idx in batch_indices:
                s = ds[int(idx)]
                msgs = s["messages"]
                metadata = s.get("metadata", {})
                img_paths = extract_image_paths(msgs)
                if not img_paths:
                    continue

                user_msgs = [m for m in msgs if m["role"] == "user"]
                prompt_text = tok.apply_chat_template(
                    user_msgs, tokenize=False, add_generation_prompt=True)

                # (a) 生成 G 条回复 (共享 KV-cache)
                completions, _, _ = generate_g_responses(
                    model, tok, prompt_text, img_paths, device,
                    args.n_generations, args.max_new_tokens, args.temperature)

                # (b) 打分
                reward_gt = build_reward_gt(metadata)
                rewards = torch.tensor(
                    [composite_reward(c, reward_gt) for c in completions],
                    device=device)

                if rewards.std() < 1e-6:
                    continue  # 全一样 → 无优势信号，跳过

                # (c) 组内归一化优势
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                # (d) 逐条计算 log-prob + GRPO loss
                for gi, completion in enumerate(completions):
                    adv = advantages[gi].detach()

                    # Log-prob under current (training) model
                    log_p = compute_log_prob(model, tok, user_msgs, completion,
                                              img_paths, device)

                    # Log-prob under reference (frozen) model
                    log_p_ref = compute_log_prob(ref_model, tok, user_msgs,
                                                  completion, img_paths, device)

                    # (e) GRPO loss = PG + KL penalty
                    pg_loss = -adv * log_p

                    # KL(π_θ || π_ref) unbiased estimator (DeepSeek k1)
                    log_ratio = log_p_ref - log_p
                    kl = torch.exp(log_ratio) - log_ratio - 1.0

                    group_losses.append(pg_loss + args.kl_beta * kl)
                    group_rewards.append(rewards[gi].item())

            if not group_losses:
                continue

            loss = torch.stack(group_losses).mean() / args.grad_accum
            loss.backward()

            avg_r = sum(group_rewards) / len(group_rewards)
            accum_r += avg_r
            batch_count += 1

            if (global_step + 1) % args.grad_accum == 0 or \
               start + args.batch_size >= len(indices):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1

                avg_r_batch = accum_r / max(batch_count, 1)
                epoch_r += avg_r_batch
                epoch_loss += loss.item() * args.grad_accum
                epoch_n += 1

                pbar.set_postfix(
                    reward=f"{avg_r_batch:.3f}",
                    loss=f"{loss.item() * args.grad_accum:.4f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}")
                wb_logger.log({
                    "grpo/reward_mean": avg_r_batch,
                    "grpo/loss": loss.item() * args.grad_accum,
                    "grpo/kl_beta": args.kl_beta,
                    "grpo/lr": sched.get_last_lr()[0],
                }, step=global_step)
                accum_r = 0.0
                batch_count = 0

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        avg_epoch_r = epoch_r / max(epoch_n, 1)
        avg_epoch_l = epoch_loss / max(epoch_n, 1)
        print(f"[EPOCH {epoch+1}] reward={avg_epoch_r:.4f}  loss={avg_epoch_l:.4f}")

    # ── Save ──
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)
    json.dump({"stage": "grpo_true", "n_generations": args.n_generations,
               "kl_beta": args.kl_beta, "lr": args.lr, "epochs": args.epochs},
              open(os.path.join(args.output, "training_config.json"), "w"), indent=2)
    print(f"\n[DONE] → {adapter_out}")

    wb_logger.finish()


if __name__ == "__main__":
    main()
