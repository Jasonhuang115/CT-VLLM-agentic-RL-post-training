#!/usr/bin/env python3
"""
Stage 3: 纯文本 GRPO (无图像)

输入 = SFT 视觉描述，模型生成诊断意见+随访建议。
奖励函数优化:
  1. 诊断语言质量 (自然、有推理链、CT-RATE风格)
  2. 全面性 (涵盖鉴别诊断、分层随访)
  3. 长度奖励 (不过短)
  4. 特征利用 (实际使用了prompt中提供的特征)
  5. 非模板化 (不用序号标题、不说"建议结合临床")

使用:
  python training/stage3_text_grpo.py \
    --adapter /root/autodl-tmp/outputs/stage2_text_dpo/lora_adapter \
    --data_dir /root/autodl-tmp/data/dpo_text \
    --output /root/autodl-tmp/outputs/stage3_text_grpo \
    --n_generations 4 --lr 1e-6
"""

import os, sys, json, argparse, re, random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm


def reward_fn(completion: str, prompt_features: dict) -> float:
    """
    诊断语言质量奖励 (纯文本, 不需要图像)

    prompt_features: {diameter_mm, malignancy, texture, spiculation, margin, ...}
    """
    score = 0.0

    # 1. 长度奖励 (0.20) — 100-600字最佳区
    length = len(completion)
    if length > 500: score += 0.20
    elif length > 300: score += 0.18
    elif length > 100: score += 0.12
    else: score += 0.05

    # 2. 特征利用 (0.25) — 实际用了 prompt 中提供的特征词
    feature_kw = []
    diam = prompt_features.get("diameter_mm", 10)
    if "mm" in completion or str(int(diam)) in completion:
        feature_kw.append("size")
    if any(w in completion for w in ["实性", "磨玻璃", "部分实性", "密度"]):
        feature_kw.append("density")
    if any(w in completion for w in ["毛刺", "分叶", "边界", "边缘"]):
        feature_kw.append("margin")
    if any(w in completion for w in ["钙化"]):
        feature_kw.append("calc")
    score += 0.25 * (len(feature_kw) / 4)

    # 3. 推理链 (0.25) — 有特征→风险推断
    reasoning_patterns = [
        r'(因为|由于|鉴于|基于).*(建议|推荐|需|应)',
        r'(高危|风险|恶性|可疑).*(随访|复查|活检|手术|切除)',
        r'(如|若|如果).*(增大|进展|变化).*(则|应|需)',
        r'(直径|大小|mm).*(短期|立即|建议)',
    ]
    n_reason = sum(1 for pat in reasoning_patterns if re.search(pat, completion))
    score += 0.25 * min(n_reason / max(len(reasoning_patterns)-1, 1), 1.0)

    # 4. 非模板化 (0.15) — 不用序号标题、不说废话
    template_signals = [
        r'^\s*(一|二|三|四|1\.|2\.|3\.)',  # 序号章节标题
        r'建议结合临床',                     # 模板废话
        r'综合.*特征.*结论',                 # 模板句式
        r'请注意.*仅供参考',                  # 免责声明当诊断
    ]
    n_template = sum(1 for pat in template_signals if re.search(pat, completion))
    score += 0.15 * (1.0 - min(n_template / 3, 1.0))

    # 5. 严重度匹配 (0.15) — 恶性评价与 GT 一致性
    mal = prompt_features.get("malignancy", 3)
    if mal >= 4:  # GT 高度可疑
        if any(w in completion for w in ["可疑", "恶性", "高风险", "活检", "立即"]):
            score += 0.15
    elif mal <= 2:  # GT 良性
        if any(w in completion for w in ["良性", "低风险", "年度随访"]):
            score += 0.15
    else:  # GT 不确定
        score += 0.10  # base score

    return min(score, 1.0)


def response_log_prob(logits, input_ids, prompt_len):
    shift_logits = logits[:, prompt_len-1:-1, :].contiguous()
    shift_labels = input_ids[:, prompt_len:].contiguous()
    token_lp = F.log_softmax(shift_logits, dim=-1)
    token_lp = token_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(-1)


def generate_g_responses(model, tok, prompt_text, device, n_generations, max_new_tokens, temperature):
    enc = tok(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        gen_ids = model.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature,
            num_return_sequences=n_generations,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    completions = []
    for gi in range(n_generations):
        new_ids = gen_ids[gi, prompt_len:]
        completions.append(tok.decode(new_ids, skip_special_tokens=True))
    return completions, prompt_len, enc


def get_metadata_dict(ds_item, ds_full):
    """从数据中读取结节特征"""
    meta = ds_item.get("metadata", {})
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage2_text_dpo/lora_adapter")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--n_generations", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    print(f"[GRPO Text] G={args.n_generations}  LR={args.lr}")

    # 1. Data
    train_file = os.path.join(args.data_dir, "dpo_train.jsonl")
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    total = len(ds)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, total)))
    print(f"[DATA] {len(ds)} prompts")

    # 2. Model (4-bit)
    model, _ = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = PeftModel.from_pretrained(model, args.adapter)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        indices = torch.randperm(len(ds)).tolist()
        pbar = tqdm(range(0, len(indices), 1), desc=f"GRPO E{epoch+1}")
        epoch_r, epoch_loss, n = 0.0, 0.0, 0

        for start in pbar:
            idx = indices[start]
            s = ds[int(idx)]
            prompt_base = s["text_prompt"]
            prompt_text = prompt_base + "\n\n请基于上述影像学发现，给出诊断意见和随访建议。"
            meta = s.get("metadata", {})

            # Generate G responses
            completions, prompt_len_unused, enc_unused = generate_g_responses(
                model, tok, prompt_text, device,
                args.n_generations, args.max_new_tokens, args.temperature)

            # Score rewards
            rewards = [reward_fn(c, meta) for c in completions]
            r_tensor = torch.tensor(rewards, device=device)
            if r_tensor.std() < 1e-6:
                continue

            advantages = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-8)

            group_losses = []
            for gi, completion in enumerate(completions):
                adv = advantages[gi].detach()

                # Log-prob
                full_text = prompt_text + completion
                enc = tok(full_text, return_tensors="pt", truncation=True, max_length=2048)
                enc = {k: v.to(device) for k, v in enc.items()}
                prompt_enc = tok(prompt_text, return_tensors="pt")
                prompt_len = prompt_enc.input_ids.shape[1]

                out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                log_p = response_log_prob(out.logits, enc["input_ids"], prompt_len)

                pg_loss = -adv * log_p
                l2_reg = 0.001 * (log_p ** 2).mean()
                group_losses.append(pg_loss + l2_reg)

            if not group_losses:
                continue

            loss = torch.stack(group_losses).mean() / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()

            global_step += 1
            epoch_r += r_tensor.mean().item()
            epoch_loss += loss.item() * args.grad_accum
            n += 1

            if global_step % 10 == 0:
                pbar.set_postfix(reward=f"{r_tensor.mean().item():.3f}",
                                 loss=f"{loss.item()*args.grad_accum:.4f}")

        print(f"[EPOCH {epoch+1}] reward={epoch_r/n:.3f}  loss={epoch_loss/n:.4f}")

    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)
    print(f"[DONE] → {adapter_out}")


if __name__ == "__main__":
    main()
