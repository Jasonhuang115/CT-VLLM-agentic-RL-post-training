#!/usr/bin/env python3
"""
Stage 2: Preference Optimization (DPO or SimPO)

DPO (NeurIPS 2023): 需要参考模型约束，防止偏离太远
  Loss = -log σ(β × [log_ratio(chosen) - log_ratio(rejected)])
  log_ratio = log [π_θ(text) / π_ref(text)]

SimPO (ICML 2024): 无需参考模型，用长度归一化 + γ margin
  Loss = -log σ(β × [log_p(chosen)/|chosen| - log_p(rejected)/|rejected|] - γ)

两者都使用图像条件化前向 (v3): VLM tokenizer + pixel_values/image_grid_thw
"""

import os, sys, json, argparse, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from unsloth import FastVisionModel
from peft import PeftModel
from tqdm import tqdm

from wandb_utils import get_logger


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [extract_text(x) for x in content]
        return " ".join(t for t in texts if isinstance(t, str))
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return extract_text(content["content"])
    return ""


def extract_image_paths(messages: list) -> list:
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


def match_images(text: str, img_paths: list) -> list:
    n = text.count('<|vision_start|>')
    imgs = list(img_paths)
    while len(imgs) < n:
        imgs = imgs + img_paths
    return imgs[:n]


def response_log_prob(logits, input_ids, prompt_len, return_sum=True):
    """只计算 response token 的 log-prob"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Mask: only response tokens
    mask = torch.zeros_like(shift_labels)
    mask[:, prompt_len-1:] = 1

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * mask.float()

    if return_sum:
        return token_lp.sum(-1)
    else:
        return token_lp.sum(-1) / mask.sum(-1).clamp(min=1)


# ═══════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════

def forward_log_prob(model, vlm_tok, prompt_msgs, completion_text, img_paths, device):
    """图像条件化前向 → response log-prob (sum)"""
    assistant = {"role": "assistant",
                 "content": [{"type": "text", "text": completion_text}]}
    full_msgs = prompt_msgs + [assistant]

    text = vlm_tok.apply_chat_template(full_msgs, tokenize=False,
                                        add_generation_prompt=False)
    matched = match_images(text, img_paths)
    enc = vlm_tok(text=[text], images=matched, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    prompt_text = vlm_tok.apply_chat_template(prompt_msgs, tokenize=False,
                                               add_generation_prompt=True)
    prompt_enc = vlm_tok(text=[prompt_text],
                          images=match_images(prompt_text, img_paths),
                          return_tensors="pt")
    prompt_len = prompt_enc.input_ids.shape[1]

    out = model(**enc)
    return response_log_prob(out.logits, enc["input_ids"], prompt_len)


def dpo_loss_for_sample(model, ref_model, vlm_tok, prompt_msgs, chosen_text,
                         rejected_text, img_paths, beta, device):
    """DPO loss on one sample"""
    lp_c = forward_log_prob(model, vlm_tok, prompt_msgs, chosen_text,
                             img_paths, device)
    lp_r = forward_log_prob(model, vlm_tok, prompt_msgs, rejected_text,
                             img_paths, device)

    with torch.no_grad():
        lp_c_ref = forward_log_prob(ref_model, vlm_tok, prompt_msgs,
                                     chosen_text, img_paths, device)
        lp_r_ref = forward_log_prob(ref_model, vlm_tok, prompt_msgs,
                                     rejected_text, img_paths, device)

    log_ratio_c = lp_c - lp_c_ref
    log_ratio_r = lp_r - lp_r_ref
    diff = beta * (log_ratio_c - log_ratio_r)

    loss = -F.logsigmoid(diff)
    acc = (lp_c > lp_r).float()
    return loss, acc


def simpo_loss_for_sample(model, vlm_tok, prompt_msgs, chosen_text,
                           rejected_text, img_paths, beta, gamma, device):
    """SimPO loss on one sample (length-normalized, no reference model)"""
    # Tokenize full sequences to get response lengths
    assistant_c = {"role": "assistant",
                   "content": [{"type": "text", "text": chosen_text}]}
    assistant_r = {"role": "assistant",
                   "content": [{"type": "text", "text": rejected_text}]}

    text_c = vlm_tok.apply_chat_template(prompt_msgs + [assistant_c],
                                          tokenize=False,
                                          add_generation_prompt=False)
    text_r = vlm_tok.apply_chat_template(prompt_msgs + [assistant_r],
                                          tokenize=False,
                                          add_generation_prompt=False)

    # Get response lengths
    prompt_text = vlm_tok.apply_chat_template(prompt_msgs, tokenize=False,
                                               add_generation_prompt=True)
    prompt_enc = vlm_tok(text=[prompt_text],
                          images=match_images(prompt_text, img_paths),
                          return_tensors="pt")
    prompt_len = prompt_enc.input_ids.shape[1]

    enc_c = vlm_tok(text=[text_c], images=match_images(text_c, img_paths),
                     return_tensors="pt")
    len_c = enc_c.input_ids.shape[1] - prompt_len

    enc_r = vlm_tok(text=[text_r], images=match_images(text_r, img_paths),
                     return_tensors="pt")
    len_r = enc_r.input_ids.shape[1] - prompt_len

    # Forward → log-prob (sum)
    enc_c = {k: v.to(device) for k, v in enc_c.items()}
    enc_r = {k: v.to(device) for k, v in enc_r.items()}

    out_c = model(**enc_c)
    lp_c = response_log_prob(out_c.logits, enc_c["input_ids"], prompt_len)

    out_r = model(**enc_r)
    lp_r = response_log_prob(out_r.logits, enc_r["input_ids"], prompt_len)

    # Length-normalized SimPO
    ratio = beta * (lp_c / len_c - lp_r / len_r) - gamma
    loss = -F.logsigmoid(ratio)
    acc = (lp_c / len_c > lp_r / len_r).float()
    return loss, acc


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dpo", "simpo"], default="dpo",
                        help="偏好优化方法 (default: dpo)")
    parser.add_argument("--model_path",
                        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter",
                        default="/root/autodl-tmp/outputs/rest_round2/lora_adapter")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/dpo")
    parser.add_argument("--output", default="/root/autodl-tmp/outputs/stage2_pref")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="DPO beta / SimPO beta")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="SimPO margin (ignored in DPO)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--no_4bit", action="store_true",
                        help="禁用 4-bit 量化 (BF16)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")
    use_4bit = not args.no_4bit

    print("=" * 60)
    print("  Stage 2: {} (image-conditioned v3)".format(
          "DPO" if args.method == "dpo" else "SimPO"))
    print("=" * 60)
    print(f"  Model:   {args.model_path}")
    print(f"  Adapter: {args.adapter}")
    print(f"  Data:    {args.data_dir}")
    print(f"  Beta: {args.beta}  {'Gamma: ' + str(args.gamma) if args.method == 'simpo' else ''}")
    print(f"  4-bit: {use_4bit}")

    wb_logger = get_logger(args.method, args.output, config=args)

    # ── 1. Load data ──
    train_file = os.path.join(args.data_dir, "dpo_train.jsonl")
    val_file = os.path.join(args.data_dir, "dpo_val.jsonl")
    if not os.path.exists(val_file):
        val_file = train_file

    ds = load_dataset("json", data_files={"train": train_file,
                                           "validation": val_file})
    if args.max_samples > 0:
        ds["train"] = ds["train"].select(
            range(min(args.max_samples, len(ds["train"]))))
    print(f"[DATA] train={len(ds['train'])}, val={len(ds['validation'])}")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size,
                              shuffle=True, num_workers=0,
                              collate_fn=lambda batch: batch)

    # ── 2. Load training model ──
    print(f"[MODEL] Loading training model (4bit={use_4bit}) …")
    model, vlm_tok = FastVisionModel.from_pretrained(
        args.model_path, load_in_4bit=use_4bit,
        use_gradient_checkpointing="unsloth")
    if os.path.exists(args.adapter):
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"  Loaded adapter: {args.adapter}")
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad = "lora" in n.lower()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[MODEL] Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── 3. Load reference model (DPO only) ──
    ref_model = None
    if args.method == "dpo":
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
    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # ── 5. Training ──
    print(f"\n[TRAIN] {total_steps} steps ({steps_per_epoch}/epoch)\n")
    global_step = 0
    accum_loss = 0.0

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"{args.method.upper()} Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_n = 0

        for bidx, samples in enumerate(pbar):
            batch_losses = []
            batch_accs = []

            for s in samples:
                prompt_msgs = s["prompt"]
                chosen_text = extract_text(s["chosen"])
                rejected_text = extract_text(s["rejected"])
                img_paths = extract_image_paths(prompt_msgs)

                if not img_paths or not chosen_text or not rejected_text:
                    continue

                if args.method == "dpo":
                    loss, acc = dpo_loss_for_sample(
                        model, ref_model, vlm_tok, prompt_msgs,
                        chosen_text, rejected_text, img_paths,
                        args.beta, device)
                else:  # simpo
                    loss, acc = simpo_loss_for_sample(
                        model, vlm_tok, prompt_msgs,
                        chosen_text, rejected_text, img_paths,
                        args.beta, args.gamma, device)

                if loss.item() == 0.0:
                    continue

                batch_losses.append(loss)
                batch_accs.append(acc)

            if not batch_losses:
                continue

            loss = torch.stack(batch_losses).mean() / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (bidx + 1) % args.grad_accum == 0 or (bidx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()

                global_step += 1
                avg_l = accum_loss * args.grad_accum
                avg_a = torch.stack(batch_accs).mean().item()
                epoch_loss += avg_l
                epoch_acc += avg_a
                epoch_n += 1

                pbar.set_postfix(loss=f"{avg_l:.4f}", acc=f"{avg_a:.3f}",
                                 lr=f"{sched.get_last_lr()[0]:.2e}")
                wb_logger.log({
                    f"{args.method}/loss": avg_l,
                    f"{args.method}/acc": avg_a,
                    f"{args.method}/lr": sched.get_last_lr()[0],
                }, step=global_step)
                accum_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    model.save_pretrained(ckpt)
                    print(f"\n[SAVE] {ckpt}")

        avg_l = epoch_loss / max(epoch_n, 1)
        avg_a = epoch_acc / max(epoch_n, 1)
        print(f"[EPOCH {epoch+1}] loss={avg_l:.4f}  acc={avg_a:.4f}")

    # ── 6. Save ──
    adapter_out = os.path.join(args.output, "lora_adapter")
    model.save_pretrained(adapter_out)
    vlm_tok.save_pretrained(adapter_out)

    cfg = {"stage": args.method, "beta": args.beta, "gamma": args.gamma,
           "epochs": args.epochs, "lr": args.lr, "4bit": use_4bit}
    with open(os.path.join(args.output, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[DONE] → {adapter_out}")
    if args.method == "dpo":
        print(f"Next: python training/stage3_grpo.py --adapter {adapter_out}")

    wb_logger.finish()


if __name__ == "__main__":
    main()
