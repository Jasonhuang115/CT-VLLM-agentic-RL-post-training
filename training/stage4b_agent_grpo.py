#!/usr/bin/env python3
"""
Stage 4b: Agentic GRPO (Tool-Use Policy Optimization) — FIXED v3

Image-conditioned REINFORCE for tool-use trajectories.
Same pattern as stage3_grpo.py but with agent-specific reward.
"""

import os, sys, json, argparse, random
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
    n = text.count('<|vision_start|>')
    imgs = list(img_paths)
    while len(imgs) < n:
        imgs = imgs + img_paths
    return imgs[:n]


TOOLS_DEF = """你可以使用以下诊断工具:
- analyze_size(diameter_mm, location): 分析结节大小和位置的临床意义
- assess_margin(margin_score, spiculation_score): 评估边界特征和毛刺程度
- classify_density(texture_score): 根据LIDC纹理评分判断密度类型
- calc_malignancy_risk(scores_dict): 综合多特征计算恶性风险评分
- lung_rads_classify(malignancy, diameter_mm, texture): 给出Lung-RADS分级
- search_guidelines(query): 检索ACR Lung-RADS临床指南

格式: Thought → Action → Action Input → Observation
最后以 "Final Answer:" 开头给出完整诊断报告。"""


def agent_reward(completion: str, metadata: dict) -> float:
    """Composite reward: report quality + tool usage bonus."""
    # Extract final answer
    final = completion.split("Final Answer:")[-1] if "Final Answer:" in completion else completion
    d = metadata.get("diameter_mm", 10)
    gt = {
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
    score = composite_reward(final, gt)

    # Tool usage bonuses
    if "Action:" in completion:
        score += 0.05
    tool_count = completion.count("Action:")
    if tool_count >= 2:
        score += 0.05
    # Penalize no tool usage
    if tool_count == 0:
        score -= 0.05

    return max(0.0, min(1.0, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage4a_agent_sft_v3/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/sft")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage4b_agent_grpo_v3")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--kl_beta", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("  Stage 4b: Agentic GRPO (v3 — Image-Conditioned)")
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
    if os.path.exists(args.adapter):
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
        epoch_r, epoch_loss, epoch_tools, epoch_n = 0.0, 0.0, 0.0, 0

        for start in pbar:
            batch_rewards, batch_losses, batch_tool_counts = [], [], []

            for idx in indices[start:start + args.batch_size]:
                s = ds[int(idx)]
                msgs = s["messages"]
                metadata = s.get("metadata", {})
                img_paths = extract_image_paths(msgs)
                if not img_paths: continue

                # Augment user prompt with tool instructions
                user_msgs_orig = [m for m in msgs if m["role"] == "user"]
                user_msgs = json.loads(json.dumps(user_msgs_orig))
                if isinstance(user_msgs[-1]["content"], list):
                    user_msgs[-1]["content"].append({"type": "text", "text": "\n\n" + TOOLS_DEF})
                else:
                    user_msgs[-1]["content"] += "\n\n" + TOOLS_DEF

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
                reward = agent_reward(completion, metadata)
                batch_rewards.append(reward)
                batch_tool_counts.append(completion.count("Action:"))

                # ── Full forward for log-prob ──
                full_msgs = user_msgs + [{"role": "assistant", "content": completion}]
                full_text = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
                full_enc = tok(text=[full_text], images=match_images(full_text, img_paths),
                              return_tensors="pt")
                full_enc = {k: v.to(device) for k, v in full_enc.items()}

                out = model(**full_enc)
                logits = out.logits

                # Response boundary
                prompt_full_text = tok.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
                prompt_enc = tok(text=[prompt_full_text],
                                images=match_images(prompt_full_text, img_paths), return_tensors="pt")
                prompt_len = prompt_enc.input_ids.shape[1]

                shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
                shift_labels = full_enc["input_ids"][:, prompt_len:].contiguous()
                token_lp = F.log_softmax(shift_logits, dim=-1)
                token_lp = token_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                gen_log_prob = token_lp.sum()

                pg_loss = -reward * gen_log_prob + args.kl_beta * (gen_log_prob ** 2)
                batch_losses.append(pg_loss)

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
                avg_t = sum(batch_tool_counts) / len(batch_tool_counts)
                epoch_r += avg_r; epoch_loss += loss.item() * args.grad_accum
                epoch_tools += avg_t; epoch_n += 1
                pbar.set_postfix(reward=f"{avg_r:.3f}", tools=f"{avg_t:.1f}",
                                 loss=f"{loss.item() * args.grad_accum:.4f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        print(f"[EPOCH {epoch+1}] reward={epoch_r/max(epoch_n,1):.4f}  tools/sample={epoch_tools/max(epoch_n,1):.1f}  loss={epoch_loss/max(epoch_n,1):.4f}")

    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)
    json.dump({"stage": "agent_grpo_v3"}, open(os.path.join(args.output, "training_config.json"), "w"), indent=2)
    print(f"\n[DONE] → {adapter_out}")


if __name__ == "__main__":
    main()
