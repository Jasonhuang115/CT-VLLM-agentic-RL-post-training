# CT-VLLM 后训练：当前困境与专家咨询

> **项目目标**: 用 Qwen2.5-VL-3B + QLoRA 训练一个面向肺结节 CT 影像的多模态诊断报告生成模型，采用 SFT → SimPO → GRPO 三阶段后训练。训练完成后 Stage 1 SFT loss=6.4（预期 2~3），模型生成报告质量不及预期。

---

## 1 项目总览

```
最终目标:  CT 图像 → VLM → 结构化放射科诊断报告
                 （位置/尺寸/密度/边界/毛刺/钙化/Lung-RADS/随访建议）

训练管线:
  Stage 1  SFT           监督微调         CT+DeepSeek报告 → 逐token交叉熵
  Stage 2  SimPO         偏好优化         Chosen(好报告) vs Rejected(扰动报告) 对比学习
  Stage 3  REINFORCE     强化学习         composite_reward 打分 → 策略梯度
  Stage 4a Agent SFT     (未来)          工具调用模仿学习
  Stage 4b Agent GRPO    (未来)          工具调用策略优化
```

---

## 2 当前状态

### 2.1 硬件配置

| 项目 | 规格 |
|---|---|
| 平台 | AutoDL 云实例 |
| GPU | RTX 5090 24GB |
| CPU/RAM | 16 vCPU / 60GB |
| 磁盘 | 280GB（已用约 150GB） |
| 框架 | PyTorch + Unsloth + Transformers |

### 2.2 基础模型 & 微调方法

| 项目 | 配置 |
|---|---|
| 基座 | Qwen2.5-VL-3B-Instruct |
| 量化 | 4-bit QLoRA (bitsandbytes) |
| LoRA | r=16, alpha=32, dropout=0.05 |
| 训练层 | language_layers + attention + MLP |
| Vision | **冻结**（finetune_vision_layers=False）|
| 梯度检查点 | unsloth gradient checkpointing |
| 优化器 | AdamW, lr=2e-4, cosine schedule |
| 梯度累积 | 8 step（有效 batch=8） |

### 2.3 训练数据

| 数据集 | 数量 | 来源 |
|---|---|---|
| CT 扫描 | 527 个系列 (subset0-7) | LUNA16 公开数据集 |
| 结节标注 | 501 个结节匹配到 LIDC | pylidc + lidc_match.py |
| 报告生成 | 870 train / 132 val | DeepSeek-V3 API 基于 LIDC 特征生成 |
| 图像格式 | 512×512 PNG 灰度图 (窗宽窗位) | mhd_to_png.py |
| 语言 | 中英文各一条/结节 → 501×2=1002 样本 |
| 训练数据格式 | OpenAI Vision Format (messages+image) | jsonl |

**数据示例**（一条训练样本）：
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/1.3.6.1.4.1...png"},
        {"type": "text", "text": "请分析这张肺部CT图像中的结节..."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "**肺部CT结节分析报告**\n\n**影像学发现**\n- 位置：右上叶后段\n..."}]
    }
  ],
  "metadata": {
    "seriesuid": "1.3.6.1.4.1...",
    "diameter_mm": 12.3,
    "malignancy": 4,
    "texture": 2,
    ...
  }
}
```

### 2.4 当前训练结果

| 阶段 | 状态 | 结果 |
|---|---|---|
| Stage 1 SFT | ✅ 已训完 | loss=6.39（3 epoch, 870 样本） |
| Stage 2 SimPO | ⏳ 待跑 | 需先验证图像条件修复是否有效 |
| Stage 3 GRPO | ⏳ 待跑 | 小批量验证 reward=0.4075 |

**模型实际输出**（训完 Stage 1 后生成）：
```
根据提供的CT图像，我们可以观察到以下几个特征：
1. 影像学发现：
   - 图像左侧的肺部区域存在一个大小约2cm的圆形结节。
   - 结节位于左下肺叶的近中心位置。
2. 恶性风险评估：
   - 对于直径小于3厘米的结节，通常被认为是良性的可能性更大。
   - 然而，不能仅凭影像学检查做出明确的恶性或良性诊断...
3. 随访建议：
   - 由于结节较小且有较高的良性风险，建议定期进行复查...

⚠ 问题: "2cm" 是幻觉（GT实际直径=10mm），未提及任何LIDC特征
  （密度类型/边界评分/毛刺/钙化/分叶），报告是教科书式的泛泛描述
```

Ground Truth (DeepSeek 生成的参考报告):
```
**肺部CT结节分析报告**
**影像学发现**
- 位置：右上叶后段
- 大小：10.0mm
- 密度类型：实性结节
- 边界特征：边界欠清，边缘轻度分叶
- 毛刺征：未见毛刺征
- 钙化：无钙化

**恶性风险评估**
该结节综合评估为可疑恶性（恶性度评分 4/5）。
根据影像学特征，Lung-RADS 分级为 4B。

**随访建议**
建议胸外科会诊，考虑活检或短期随访
```

---

## 3 核心困境

### 3.1 问题描述

**Stage 1 SFT loss=6.4 (perplexity≈600)，模型学到了"这是肺"但没学到"看看这个结节的细节"。**

```
我们期望模型做的:                   模型实际做的:
  CT → 读密度 → "实性结节"              CT → "这是肺" → 背教科书
  CT → 读尺寸 → "10.0mm"               CT → 没有精确读尺寸 → "约2cm" (幻觉)
  CT → 读边界 → "边界欠清，轻度分叶"     CT → 没有特征提取 → 不提具体特征
  CT → 读钙化 → "无钙化"               CT → 跳过 → 给通用建议
```

### 3.2 我们的困惑

**困惑 A: 是模型容量不够，还是数据/训练方式不对？**

- Qwen2.5-VL-3B 的 vision encoder 在 COCO/LAION 等自然图像上预训练，从未见过 CT 灰度图。要求一个 3B 的 VLM 同时完成 (a) 学会看 CT (b) 提取结节微特征 (c) 生成专业中文报告，是否超出了它的能力上限？
- 如果换 7B/8B 模型，预期改善幅度有多大？
- 还是说 3B 够用，但我们的训练策略（冻结 vision encoder、数据增强不足）限制了它？

**困惑 B: 数据量和训练策略是否合理？**

- 870 条训练数据，3 epoch ≈ 2600 optimizer steps。对于 domain gap 这么大的医学场景，是否严重不足？
- 我们每个结节生成了一中一英两份报告（501×2=1002）。中英混合数据对中文场景是帮助还是干扰？
- DeepSeek 基于 LIDC 特征描述（不含 CT 图像）生成的报告，作为训练目标是否合理？模型要从图像学到这些文本描述，本身就是一个交叉模态映射任务，难度是否被低估？

**困惑 C: Vision encoder 应该解冻吗？**

- 当前 `finetune_vision_layers=False`（省显存 + 防过拟合）
- 但 CT 图像的视觉特征（HU 值、纹理模式、结节轮廓）与自然图像完全不同
- 如果解冻 vision encoder，870 条 CT 数据会不会直接过拟合？
- 有没有中间方案——比如只解冻最后几层 vision layer？

**困惑 D: 三阶段后训练架构是否最优？**

当前设计: SFT → SimPO → GRPO

- 有没有更合适的路线？比如：
  - 多轮 ReST 自进化（generate → reward filter → SFT，循环 3 轮）
  - SFT 阶段先做医学文本继续预训练（在放射科报告语料上做 text-only CPT）
  - 在 SFT 之前加一个 medical VQA 预训练阶段
- 当前 reward 函数（composite_reward，基于正则表达式 + 规则，见附录 A）能否真正驱动 GRPO/RL 优化？

**困惑 E: 多模态训练的 Loss 计算**

- Qwen2.5-VL 内部使用 sum reduction（我们实测确认），导致 loss 数值不可读（原始 16877 → 修正后 6.4）
- VLM SFT 应该只计算 response token 的 loss（mask prompt+图像token），我们已修复
- 但 3B 模型的大词表 (152K) + 动态分辨率的长序列 (3000-8000 tokens) = logits 物化即 OOM
- 这些 VLM 训练的工程细节是否有最佳实践参考？

---

## 4 决策点 — 需要专家建议

### 决策 1: 换模型 vs 继续优化当前模型

| 选项 | 优点 | 缺点 |
|---|---|---|
| A. 维持 3B，优化训练 | 省显存，5090够用；已有基建 | loss 天花板可能不够低 |
| B. 换 Qwen2.5-VL-7B | 更大容量，vision encoder 更强 | 24GB 显存吃紧，可能需要多卡 |
| C. 换医学专用 VLM (如 MedVersa) | 医学预训练，domain gap 小 | 可能需要不同工具链 |

**您的建议是？**

### 决策 2: 解冻 vision encoder 的时机和幅度

| 选项 | 风险 |
|---|---|
| A. 全程冻结 | 模型可能永远学不会 CT 特征 |
| B. Stage 1 后半段逐步解冻 | 870 条数据太小，可能过拟合 |
| C. 全部解冻 + 强正则化 | 显存翻倍 |
| D. 只解冻 vision tower 最后 2-4 层 | 折中方案，不知道效果如何 |

**对于 3B 模型 + 870 条医学图像数据，vision encoder 应该怎么处理？**

### 决策 3: 是否需要 pretraining 阶段

当前是 Qwen2.5-VL Base → SFT。是否应该加入：

- **Medical CPT**（在放射科报告语料上做 text-only 继续预训练，让 LLM 先学会医学语言）
- **Medical VQA pretraining**（在公开医学 VQA 数据集上先做图文对齐）
- **CT-specific contrastive pretraining**（用对比学习让 vision encoder 适应 CT 域）

**哪个投入产出比最高？**

### 决策 4: 后训练架构

| 当前路线 | 替代方案 |
|---|---|
| SFT → SimPO → GRPO | SFT → 多轮 ReST 自进化 |
| | SFT → DPO → GRPO |
| | SFT → 医学 VQA 预训练 → SFT → RL |

**对于"生成准确的专业诊断报告"这个目标，有没有更高效的训练架构？**

### 决策 5: 数据策略

- 当前 870 条，目标 ~5000+。在数据量有限时，数据增强（旋转/翻转/对比度变化）有多大价值？
- 每个结节中英双份报告 vs 纯中文报告，哪种更有效？
- DeepSeek 基于 LIDC **文本特征**生成的报告（未看 CT 图像）作为训练目标，这个 pipeline 本身是否存在根本性缺陷？

---

## 5 附录

### 附录 A: 技术栈概览

```
模型层:
  Qwen2.5-VL-3B-Instruct  (HuggingFace)
  ↓ QLoRA 4-bit (bitsandbytes)
  ↓ LoRA: r=16, alpha=32, target_modules=all-linear
  ↓ Vision: ViT-G/14, 冻结

训练层:
  Unsloth FastVisionModel  (优化版 VLM 训练封装)
  ↓ 手动训练循环 (非 TRL Trainer，兼容性问题)
  ↓ AdamW + CosineAnnealingLR
  ↓ Gradient Checkpointing (unsloth 内置)

数据层:
  LUNA16 (888 CT 扫描, .mhd/.raw 格式)
  ↓ lidc_match.py (pylidc → SimpleITK 坐标系转换)
  ↓ mhd_to_png.py (HU窗宽窗位 → 512×512 灰度PNG)
  ↓ sft_dataset_builder.py + DeepSeek-V3 API
  ↓ OpenAI Vision Format .jsonl

评估层:
  composite_reward (5信号+防坍缩，见附件 reward_functions.py)
  Stage 2 SimPO acc (chosen vs rejected 二分类准确率)
```

### 附录 B: 已解决的工程问题 (14个)

完整记录见 `TRAINING_ISSUES.md`，关键问题：

1. **VLM Loss 计算**: Qwen2.5-VL 内部用 sum reduction，需手动 `/n_tokens` 转 mean；必须 response-only mask
2. **动态分辨率图像占位符**: 一张图 → N 个 `<vision_start>` 块，tokenizer 需重复传 N 次图像路径
3. **Unsloth 版本差异**: Mac/AutoDL 版本不同，`UnslothVisionTrainer` 不可用
4. **坐标系匹配**: pylidc voxel 坐标 vs LUNA16 world mm，需 4 种轴序组合
5. **CUDA OOM**: 手动物化 logits [L, 152064] 立即爆显存，必须传 labels 走 fused kernel
6. **zip 文件损坏**: Zenodo CDN 重定向 corrupts central directory

### 附录 C: GPU 显存占用详情

| 组件 | 显存 |
|---|---|
| Qwen2.5-VL-3B (4-bit) | ~4.5 GB |
| LoRA 适配器 + 梯度 | ~0.5 GB |
| 优化器状态 (AdamW) | ~1.0 GB |
| KV Cache (generate 时) | ~2-4 GB |
| Vision Encoder 前向 | ~1-2 GB |
| **总计** | **~10-12 GB** |

单卡 RTX 5090 24GB 当前有余量，但如果解冻 vision encoder 或换 7B 可能不够。

---

## 6 期望的专家建议

以下是我们最想听到的建议：

1. **对于模型生成"约2cm圆形结节"这种泛泛输出，病因诊断是什么？** 是模型容量天花板，还是训练策略可修复的问题？

2. **如果维持 3B，有哪些低成本的改进手段？** 比如数据增强、训练技巧、prompt 工程等？

3. **如果要升级，推荐的最小代价路径是什么？** 7B 够吗？还是需要更大的？有没有医学专用的预训练 VLM 推荐？

4. **后训练架构建议？** SFT→SimPO→GRPO 这个路线是否有根本性问题？ReST 自进化是否更适合这个场景？

5. **数据量对效果影响的经验估计？** 从 870 到 5000 条，预期改善幅度有多大？什么时候会到天花板？

---

*文档生成: 2026-07-07 | 项目路径: `CT-VLLM-agentic-RL-post-training-main/` | 共 ~3600 行训练+数据处理代码*
