#!/usr/bin/env python3
"""
Stage 2: 纯文本 DPO (无图像)

优化从"影像学发现"到"诊断建议+随访"的语言生成质量。
不需要 CT 图像，纯 LLM 前向，VRAM ~10GB。

使用:
  python training/stage2_text_dpo.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter /root/autodl-tmp/outputs/stage1_mv_cn/lora_adapter \
    --data_dir /root/autodl-tmp/data/dpo_text \
    --output /root/autodl-tmp/outputs/stage2_text_dpo \
    --beta 0.5 --lr 5e-5 --epochs 1
"""

import os, sys, json, argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wandb_utils import get_logger


def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict):
                if "text" in x: parts.append(str(x["text"]))
                if "content" in x: parts.append(extract_text(x["content"]))
            else: parts.append(str(x))
        return " ".join(parts)
    return str(content)


def response_log_prob(logits, input_ids, response_start):
    """计算 response 部分的 log-prob (mean)"""
    shift_logits = logits[:, response_start-1:-1, :].contiguous()
    shift_labels = input_ids[:, response_start:].contiguous()
    token_lp = F.log_softmax(shift_logits, dim=-1)
    token_lp = token_lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_lp.mean(-1)


def compute_text_log_prob(model, tok, prompt_text, completion_text, device):
    """纯文本 log-prob (不需要图像处理)"""
    full_text = prompt_text + completion_text
    enc = tok(full_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_enc = tok(prompt_text, return_tensors="pt")
    prompt_len = prompt_enc.input_ids.shape[1]

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return response_log_prob(out.logits, enc["input_ids"], prompt_len)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default="/root/autodl-tmp/outputs/stage1_mv_cn/lora_adapter")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--logging_steps", type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    wb_logger = get_logger("text_dpo", args.output, config=args)

    print("=" * 60)
    print("  Stage 2: 纯文本 DPO (无图像)")
    print("=" * 60)
    print(f"  Beta: {args.beta}  LR: {args.lr}")

    # 1. Load data
    train_file = os.path.join(args.data_dir, "dpo_train.jsonl")
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    print(f"[DATA] {len(ds)} 偏好对")

    # 2. Load models (4-bit for VRAM efficiency)
    print("[MODEL] Loading (4-bit) …")
    model, _ = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = PeftModel.from_pretrained(model, args.adapter)
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable: {sum(p.numel() for p in trainable):,} | VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Load reference model (frozen)
    print("[MODEL] Reference (frozen) …")
    ref_model, _ = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=True)
    ref_model = PeftModel.from_pretrained(ref_model, args.adapter)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Use bare tokenizer (no vision processing)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 3. Training
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    steps = len(ds) // args.grad_accum

    global_step, correct, total = 0, 0, 0
    for epoch in range(args.epochs):
        indices = torch.randperm(len(ds)).tolist()
        pbar = tqdm(range(0, len(indices), args.batch_size), desc=f"DPO E{epoch+1}")
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0

        for start in pbar:
            micro_losses = []
            batch_acc = []

            for idx in indices[start:start + args.batch_size]:
                s = ds[int(idx)]
                prompt = s["text_prompt"]
                chosen = s["chosen"]
                rejected = s["rejected"]

                # Log-prob under training model
                lp_c = compute_text_log_prob(model, tok, prompt, chosen, device)
                lp_r = compute_text_log_prob(model, tok, prompt, rejected, device)

                # Log-prob under reference model
                with torch.no_grad():
                    lp_c_ref = compute_text_log_prob(ref_model, tok, prompt, chosen, device)
                    lp_r_ref = compute_text_log_prob(ref_model, tok, prompt, rejected, device)

                # DPO loss
                log_ratio_c = lp_c - lp_c_ref
                log_ratio_r = lp_r - lp_r_ref
                loss = -F.logsigmoid(args.beta * (log_ratio_c - log_ratio_r))
                micro_losses.append(loss)
                batch_acc.append(float(lp_c > lp_r))
                correct += float(lp_c > lp_r)
                total += 1

            if not micro_losses:
                continue

            loss = torch.stack(micro_losses).mean() / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()

            global_step += 1
            epoch_loss += loss.item() * args.grad_accum
            epoch_acc += sum(batch_acc) / len(batch_acc)
            n += 1

            if global_step % args.logging_steps == 0:
                pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}",
                                 acc=f"{sum(batch_acc)/len(batch_acc):.3f}")
                wb_logger.log({"dpo/loss": loss.item() * args.grad_accum,
                               "dpo/acc": sum(batch_acc)/len(batch_acc)}, step=global_step)

        print(f"[EPOCH {epoch+1}] loss={epoch_loss/n:.4f}  acc={epoch_acc/n:.3f}")

    # Save
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    tok.save_pretrained(adapter_out)
    print(f"[DONE] → {adapter_out}")
    wb_logger.finish()


if __name__ == "__main__":
    main()
