#!/usr/bin/env python3
"""
Stage 4a: Agent SFT (Tool-Use Imitation Learning) — FIXED v3

Uses same image-conditioned manual loop as stage1_sft.py.
Supports real agent trajectories or synthetic ones built from nodule features.
"""

import os, sys, json, argparse, random
import torch
import torch.nn.functional as F
from datasets import Dataset
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm


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

工具调用格式:
Thought: <分析当前需要什么信息>
Action: <工具名>
Action Input: <JSON参数>
Observation: <工具返回结果>

最后以 "Final Answer:" 开头给出完整诊断报告。"""


def build_synthetic_trajectory(sft_sample: dict, num_steps: int = 2) -> dict:
    """Build a synthetic agent trajectory from SFT data."""
    msgs = sft_sample["messages"]
    metadata = sft_sample.get("metadata", {})

    user_msgs = [m for m in msgs if m["role"] == "user"]
    # Add tool instruction to last user message
    user_msgs_aug = json.loads(json.dumps(user_msgs))  # deep copy
    if isinstance(user_msgs_aug[-1]["content"], list):
        user_msgs_aug[-1]["content"].append({"type": "text", "text": "\n\n" + TOOLS_DEF})
    else:
        user_msgs_aug[-1]["content"] += "\n\n" + TOOLS_DEF

    assistant_text = ""
    for m in msgs:
        if m["role"] == "assistant":
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        assistant_text += item["text"]
            elif isinstance(content, str):
                assistant_text = content
            break

    # Build tool call steps from metadata
    steps = []
    tools_pool = [
        ("analyze_size", {"diameter_mm": metadata.get("diameter_mm", 10), "location": "lung"}),
        ("assess_margin", {"margin_score": metadata.get("margin", 3), "spiculation_score": metadata.get("spiculation", 1)}),
        ("classify_density", {"texture_score": metadata.get("texture", 5)}),
        ("calc_malignancy_risk", {"scores": {"malignancy": metadata.get("malignancy", 3), "texture": metadata.get("texture", 5)}}),
    ]
    chosen_tools = random.sample(tools_pool, min(num_steps, len(tools_pool)))

    for tool_name, params in chosen_tools:
        thought_map = {
            "analyze_size": "需要分析结节大小和位置的临床意义",
            "assess_margin": "需要评估边界特征和毛刺情况",
            "classify_density": "需要根据纹理评分确定密度类型",
            "calc_malignancy_risk": "需要综合多特征计算恶性风险",
        }
        obs_map = {
            "analyze_size": f"结节直径{params['diameter_mm']}mm，位于肺实质内，超过8mm阈值需关注",
            "assess_margin": f"边缘评分{params['margin_score']}/5，毛刺评分{params['spiculation_score']}/5",
            "classify_density": f"纹理评分{params['texture_score']}/5，符合{['非实性','部分实性','实性'][min(params['texture_score']//2, 2)]}结节特征",
            "calc_malignancy_risk": f"综合恶性风险评分：{metadata.get('malignancy', 3)}/5",
        }
        steps.append({
            "thought": thought_map.get(tool_name, "需要进一步分析"),
            "tool": tool_name,
            "params": params,
            "observation": obs_map.get(tool_name, "分析完成"),
        })

    # Build messages with tool calls
    new_msgs = list(user_msgs_aug)
    for step in steps:
        tool_text = f"Thought: {step['thought']}\nAction: {step['tool']}\nAction Input: {json.dumps(step['params'], ensure_ascii=False)}"
        new_msgs.append({"role": "assistant", "content": tool_text})
        new_msgs.append({"role": "user", "content": f"Observation: {step['observation']}"})

    # Final answer
    final_text = f"Final Answer:\n{assistant_text}"
    new_msgs.append({"role": "assistant", "content": final_text})

    return {"messages": new_msgs, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage3_grpo_v3/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/sft")
    parser.add_argument("--trajectories", default="")
    parser.add_argument("--build_synthetic", action="store_true", default=True,
                        help="Build synthetic trajectories from SFT data (for testing)")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage4a_agent_sft_v3")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print("=" * 60)
    print("  Stage 4a: Agent SFT (v3 — Image-Conditioned)")
    print("=" * 60)

    # ── 1. Build/load data ──
    if args.trajectories and os.path.exists(args.trajectories):
        print(f"[DATA] Loading trajectories: {args.trajectories}")
        data = []
        with open(args.trajectories) as f:
            for line in f:
                data.append(json.loads(line))
        print(f"  Loaded {len(data)} trajectories")
    elif args.build_synthetic:
        print("[DATA] Building synthetic trajectories from SFT data...")
        train_file = os.path.join(args.data_dir, "sft_train.jsonl")
        sft_data = []
        with open(train_file) as f:
            for line in f:
                sft_data.append(json.loads(line))
        if args.max_samples > 0:
            sft_data = sft_data[:args.max_samples]
        data = [build_synthetic_trajectory(s) for s in sft_data]
        print(f"  Built {len(data)} synthetic trajectories")
    else:
        print("[ERROR] No trajectories provided. Use --trajectories or --build_synthetic")
        return

    # ── 2. Load model ──
    print("[MODEL] Loading FastVisionModel …")
    model, tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    if os.path.exists(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,}")

    # ── 3. Optimizer ──
    steps_per_epoch = max(len(data) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    print(f"[TRAIN] ~{steps_per_epoch} steps/epoch\n")

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(data)
        pbar = tqdm(range(0, len(data), args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss_sum, epoch_n = 0.0, 0
        accum_loss = 0.0

        for start in pbar:
            batch_losses = []
            for i in range(start, min(start + args.batch_size, len(data))):
                s = data[i]
                msgs = s["messages"]
                img_paths = extract_image_paths(msgs)
                if not img_paths: continue

                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                enc = tok(text=[txt], images=match_images(txt, img_paths), return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}

                out = model(**enc)
                loss = out.loss
                if loss is None:
                    logits = out.logits
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = enc["input_ids"][:, 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                            shift_labels.view(-1))
                batch_losses.append(loss)

            if not batch_losses: continue
            loss = torch.stack(batch_losses).mean() / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (global_step + 1) % args.grad_accum == 0 or start + args.batch_size >= len(data):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                global_step += 1
                epoch_loss_sum += accum_loss * args.grad_accum
                epoch_n += 1
                pbar.set_postfix(loss=f"{accum_loss * args.grad_accum:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
                accum_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        print(f"[EPOCH {epoch+1}] loss={epoch_loss_sum/max(epoch_n,1):.4f}")

    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)
    print(f"\n[DONE] → {adapter_out}")


if __name__ == "__main__":
    main()
