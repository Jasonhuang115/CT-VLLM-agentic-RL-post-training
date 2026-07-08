# 后训练修复计划

> 基于专家诊断：数据管线方向性错误 + Vision 冻结 + Loss 验证
> 核心原则：**不换模型，先修根本缺陷**

---

## 总览

```
当前状态:  SFT loss=6.4, 模型输出泛泛教科书式描述
目标:      SFT loss<2.5, 模型能基于CT描述结节具体特征
时间线:    2周, 分5个Phase
```

| Phase | 内容 | 预期耗时 | 准入标准 |
|---|---|---|---|
| **P0** | Loss 正确性验证 + Token 统计 | 1天 | 确认 response mask 正确 |
| **P1** | 数据结构化 prompt 改造 | 1-2天 | prompt 含坐标+直径 |
| **P2** | Vision connector + ViT 最后2层解冻 | 1天 | 分组LR, 显存可控 |
| **P3** | 重跑 SFT | 2-3天 | loss<3.0 |
| **P4** | ReST 自蒸馏 1轮 | 2-3天 | report quality↑ |

---

## P0: Loss 正确性验证

### 目标
确认当前 loss=6.4 是真实值，排查中文 token 稀疏问题。

### 任务清单

- [ ] **打印 label tensor**：抽1条样本，确认 `labels != -100` 的位置数 = assistant token 数（不含 prompt/image token）
- [ ] **打印特殊 token**：确认 `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>` 系列都在 mask 范围内
- [ ] **中英对比**：同一条数据跑中英两个版本，对比 loss 差异。如果英文 loss 显著更低 → 中文 token 稀疏问题
- [ ] **手算验证**：`loss_sum / n_tokens` 的值与 `F.cross_entropy(reduction='mean')` 一致

### 脚本
```bash
# AutoDL 上验证
python -c "
import torch, json
from unsloth import FastVisionModel

# 加载一条样本，打印 token 统计
# 输出: prompt_len, response_len, labels nonzero count
"
```

### 通过标准
- Response mask: `n_nonmasked ≈ assistant_text_token_count`（±5%）
- 英文 loss 不应显著低于中文（差 < 1.0）
- `loss_sum / n_tokens` 与手动 CE mean 一致

---

## P1: SFT 数据结构化 Prompt 改造

### 目标
将 LIDC 结构化特征注入 user prompt，让模型有明确锚点。

### 改造内容

**改造前**（当前）:
```
User: "请分析这张肺部CT图像中的结节..."
      [CT PNG 图像]

Assistant: "**肺部CT结节分析报告**
            - 位置：右上叶后段
            - 大小：10.0mm
            - 密度类型：实性结节
            ..."
```

**改造后**:
```
User: "请分析这张肺部CT图像中的结节。

       临床提示：
       - 结节位于 右上肺 区域
       - 估计直径约 10mm
       
       请重点分析以下特征：
       1. 密度类型（磨玻璃/部分实性/实性）
       2. 边界特征（清晰/模糊/毛刺/分叶）
       3. 钙化状态（有无钙化及类型）
       4. 给出 Lung-RADS 分级和随访建议"

       [CT PNG 图像]

Assistant: [同上，不变]
```

### 任务清单

- [ ] 在 `sft_dataset_builder.py` 中新增 `build_structured_prompt(feats)` 函数
  - 从 metadata 提取坐标→粗略位置（左上叶/左下叶/右上叶/右中叶/右下叶）
  - 从 metadata 提取 diameter_mm → 估计直径
  - 用坐标反推肺叶（改进现有 `lobe_from_coord`）
- [ ] Instruction templates 加入结构化前缀（保留原有多样性）
- [ ] **不需要重新生成报告**——只改 user prompt 拼接逻辑，assistant 内容不变
- [ ] 重新 build SFT 数据集: `python data/sft_dataset_builder.py --features ... --output /root/autodl-tmp/data/sft_structured`
- [ ] 验证: 打印一条新样本确认 prompt 包含结构化信息

### 通过标准
- 新 prompt 包含：位置提示 + 直径估计 + 分析要点清单
- Assistant 报告内容不变
- Train/val split 不变（按 seriesuid 分）

---

## P2: Vision Connector 解冻 + 分组 LR

### 目标
让 vision 信号能真正流入 LLM，同时避免过拟合。

### 策略

| 组件 | 操作 | LR |
|---|---|---|
| Vision Projector/Connector | **解冻** | = LLM LoRA LR (2e-4) |
| ViT 最后 2 层 | **解冻** | = LLM LoRA LR × 0.1 (2e-5) |
| ViT 其余层 | 冻结 | — |
| LLM LoRA | 保持当前 | 2e-4 |

### 任务清单

- [ ] 修改 `stage1_sft.py`:
  - 添加 `--unfreeze_vision` flag（默认 True）
  - 添加 `--unfreeze_vit_layers` 参数（默认 2）
  - 添加 `--vit_lr_ratio` 参数（默认 0.1）
- [ ] 解冻逻辑：
  ```python
  # 1. 解冻 vision-language projector
  for n, p in model.named_parameters():
      if "visual" in n and "merger" in n:  # Qwen VL projector 名字
          p.requires_grad = True
  
  # 2. 解冻 ViT 最后 N 层
  vit_layers = [n for n, p in model.named_parameters() 
                if "visual" in n and "blocks" in n]
  last_layers = sorted(set(l.split('.blocks.')[1].split('.')[0] 
                           for l in vit_layers))[-args.unfreeze_vit_layers:]
  for n, p in model.named_parameters():
      if any(f"blocks.{l}." in n for l in last_layers):
          p.requires_grad = True
  ```
- [ ] 分组 optimizer:
  ```python
  vit_params = [p for n, p in model.named_parameters() 
                if p.requires_grad and "visual" in n]
  lora_params = [p for n, p in model.named_parameters() 
                 if p.requires_grad and "visual" not in n]
  opt = AdamW([
      {"params": lora_params, "lr": args.lr},
      {"params": vit_params, "lr": args.lr * args.vit_lr_ratio},
  ], weight_decay=0.01)
  ```
- [ ] 显存预估：+2-3GB，总 < 15GB → 5090 24GB 安全

### 通过标准
- 训练不 OOM
- `visual.merger` 参数 requires_grad=True
- ViT 最后 2 层参数 requires_grad=True
- ViT 前 N-2 层参数 requires_grad=False

---

## P3: 重跑 SFT

### 目标
loss 收敛到 < 3.0（最好是 < 2.5）。

### 配置

```bash
python training/stage1_sft.py \
    --data_dir /root/autodl-tmp/data/sft_structured \
    --output /root/autodl-tmp/outputs/stage1_structured_v1 \
    --epochs 3 \
    --lr 2e-4 \
    --vit_lr_ratio 0.1 \
    --unfreeze_vit_layers 2 \
    --batch_size 1 \
    --grad_accum 8
```

### 监控指标

| 指标 | 目标 | 警戒值 |
|---|---|---|
| Train loss (epoch 1) | < 5.0 | > 8.0 |
| Train loss (epoch 3) | < 3.0 | > 4.0 |
| Val loss | 不高于 train loss + 0.5 | 高于 train loss + 1.0 (过拟合) |
| 生成报告含具体数值 | ≥50% 样本 | < 30% |
| 生成报告含密度描述 | ≥70% 样本 | < 50% |

### 验收标准
- 生成报告不再出现"约 2cm"这种幻觉数值
- 生成报告至少包含 3/5 种特征（密度/边界/钙化/Lung-RADS/位置）
- 不满足 → 回 P1/P2 排查，不进入 P4

---

## P4: ReST 自蒸馏 1 轮

### 目标
用模型自己生成的高分样本扩数据，进一步拉高质量。

### 为什么先做 ReST 不做 GRPO
- SFT 没收敛 → 做 GRPO = 奖励欺骗（往输出塞关键词骗分）
- ReST 是 SFT 的平滑延伸：生成 → 打分 → 过滤 → 加训
- 特别适合数据稀缺场景

### 流程

```
Step 1: 用 P3 最优 checkpoint 对全部 527 个 CT 各生成 4 条报告
Step 2: composite_reward 对每条报告打分
Step 3: 取 top-50% 高分报告，加入训练集
Step 4: 用扩充后的数据再做 1 epoch SFT (lr=1e-4, 更保守)
```

### 任务清单

- [ ] 写 `scripts/rest_self_distill.py`:
  ```python
  # 核心逻辑
  for sample in sft_data:
      for _ in range(4):  # 每个CT生成4条
          completion = model.generate(ct_image, prompt, temperature=0.8)
          reward = composite_reward(completion, gt)
          candidates.append((sample, completion, reward))
  
  # 按 reward 排序，取 top 50%
  candidates.sort(key=lambda x: x[2], reverse=True)
  good = candidates[:len(candidates)//2]
  
  # 构建新的 SFT 数据
  new_data = [{"image": c[0]["image"], 
               "report": c[1],
               "source": "self-distilled"} for c in good]
  
  # 1 epoch 保守微调
  sft_train(model, original_data + new_data, lr=1e-4, epochs=1)
  ```

### 通过标准
- 生成样本 >= 500 条
- Top-50% reward 中位数 > 0.3
- ReST 后 loss 不反弹（≤ ReST 前 + 0.3）

---

## 里程碑

| 日期 | Phase | 验收 |
|---|---|---|
| Day 1 | P0 | Loss mask 验证通过 |
| Day 2-3 | P1 | 结构化 prompt 数据集构建完毕 |
| Day 3-4 | P2 | Vision 解冻训练不 OOM |
| Day 5-7 | P3 | SFT loss < 3.0, 生成报告有具体特征 |
| Day 8-10 | P4 | ReST 1 轮完毕, 报告质量提升 |
| Day 11-14 | — | 评估 + 决定是否扩数据到 2000+ |

---

## 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| 解冻 vision 后过拟合 | 中 | 加 weight_decay (0.01→0.05), 减 ViT LR |
| 结构化 prompt 帮助不大 | 低 | 升级到方案 B: PNG 叠加红色结节圈 |
| 中文 token 稀疏导致的 ppl 偏高 | 中 | 接受 loss=3-4 的 ceiling，英文评估做双验证 |
| ReST 后模型退化 | 中 | 只用 reward>0.5 的样本, 保留原始数据混合训练 |
| 显存不够 | 低 | 减 grad_accum→4, 减 unfreeze_vit_layers→1 |

---

## 不做的事情

- ❌ 不换 7B 模型
- ❌ 不跑 Stage 2 SimPO / Stage 3 GRPO（等 SFT loss<2.5 再说）
- ❌ 不做翻转数据增强（破坏左右肺解剖信息）
- ❌ 不加英文数据（专注中文，等中文达标再考虑）
