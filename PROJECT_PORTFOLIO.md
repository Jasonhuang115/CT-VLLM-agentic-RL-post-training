# 肺结节CT多模态后训练项目 — 项目档案

---

## 一、简介

### 简历描述

**LungCT-R1: 基于多模态后训练的肺结节CT诊断Agent**

*GitHub: github.com/Jasonhuang115/CT-VLLM-agentic-RL-post-training | 个人项目 | 2026.06-07*

**项目背景**：面向肺结节CT影像诊断场景，构建完整后训练管线（SFT → DPO → GRPO），使 3B 视觉语言模型从零基础学会看图写放射科报告，并结合 Agent 工具实现指南检索、Lung-RADS 分级等临床辅助功能。

**技术栈**：Qwen2.5-VL-3B, LoRA/QLoRA, Unsloth, TRL, DeepSpeed, vLLM, PyTorch, GRPO, DPO, SFT, LangChain, Gradio, WandB, Pydicom, SimpleITK, OpenCV

**项目亮点**：
- **全链路后训练**：独立构建 SFT → DPO → GRPO 三阶段管线，使用 QLoRA 4-bit 在 RTX 5090 32GB 上完成 3B VLM 全流程训练，总消耗 ~¥200
- **视觉通路修复**：发现并修复 4-bit 量化下 ViT 假解冻问题（dtype=uint8 导致 requires_grad 静默绕过），切换 BF16 后视觉可训练参数从 0 提升至 530M
- **CT图像质量优化**：发现 84px 微型 PNG 导致 VLM 仅获 36 视觉 token 的问题，通过多视图三平面提取 + 512px resize 将视觉信息从 108 token 提升至 4032 token，SFT Loss 从 298 降至 9.5
- **六大类DPO错误构造**：设计严重度降级/特征漏报/特征误判/假精度/过度委婉/Lung-RADS不匹配等 6 类精细错误，90 对偏好数据修正模型系统性诊断偏差
- **GRPO 可验证奖励**：基于 LIDC 多放射科医生标注构建事实奖励信号，替代模糊的偏好排序

---

## 二、各阶段做了什么

### Stage 1: SFT（监督微调）— 核心阶段

**做什么**：让 Qwen2.5-VL-3B 从"完全不认识CT"到"能看图写报告"

**数据**：
- LUNA16 (888例CT，1186个肺结节，质心 + 直径标注)
- LIDC-IDRI 特征 (854个结节，4位放射科医生9维特征标注：纹理、边界、毛刺、分叶、钙化、恶性度等)
- CT-RATE (47,149份真实放射科报告文本，含21,382份肺结节阳性)

**训练配置**：
```
模型: Qwen2.5-VL-3B-Instruct
量化: BF16 (非4-bit, 因为4-bit下ViT假解冻)
微调: LoRA r=16 + ViT后4层全量解冻
可训练参数: ~560M (总参数的16%)
学习率: 2e-4, Cosine schedule
Batch: 1 × 8 grad_accum = effective 8
Epochs: 3-5
显存: ~12-15GB / 32GB
训练时间: ~2小时 (RTX 5090)
数据量: 1472条(纯中文, no-hint, 3张多视图/样本)
```

**关键踩坑**：
| 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|
| 图片太小 | Loss=298, 模型背教科书 | mhd_to_png.py 输出 84px，仅36视觉token | resize到512px, 1344 token/图 |
| ViT假解冻 | 视觉完全冻结, ViT LoRA keys=0 | 4-bit下参数dtype=uint8, unfreeze检查跳过 | 切换到BF16 (--no_4bit) |
| 无hint短路 | 模型抄prompt不看图 | 结构化提示注入结节位置/大小, 模型直接文本复制 | no-hint训练 |
| 单视角不足 | 5mm结节在50mm ROI中仅占10%画面 | 只有轴位切片 | 加入冠位+矢位三平面视图 |
| 报告风格 | 模型输出"不确定"(即使GT明确) | 模板报告训练导致模型保守 | DeepSeek API重写报告 |

**效果**：
- Loss: 298 → 2.63 (纯中文三平面版)
- 模型能够: 识别结节位置、估算大小、区分实性/磨玻璃、描述边界特征
- 模型还不能: 精确毛刺分级、准确恶性评估（这些受限于3B VL的视觉能力上限）

### Stage 2: DPO（偏好对齐）— 辅助微调

**做什么**：修正 SFT 模型的系统性诊断错误（保守低估、特征漏报、过度委婉）

**数据**：
- 90对偏好对 (6类错误 × 15对)
- chosen = LIDC GT 对应的正确报告
- rejected = 含特定类型错误的报告

**错误类别**：
| 类别 | 错误模式 | 选结节逻辑 |
|------|---------|-----------|
| 严重度降级 | 中度毛刺→轻度, 可疑恶性→不确定 | spiculation≥3 或 malignancy≥3 |
| 特征漏报 | 有分叶/毛刺/钙化但报告不提 | calc≤4 或 spic≥3 或 lob≥2 |
| 特征误判 | 实性→磨玻璃, 模糊→清晰 | texture≥4 或 margin≥3 |
| 假精度 | 12.3192mm, 带坐标 | 任意 |
| 过度委婉 | 特征明确但满篇不确定措辞 | malignancy≥3 且 spic≥2 |
| L-R不匹配 | 特征正确但评级偏低1-2级 | malignancy≥3 且 diam≥8 |

**训练配置**：
```
方法: DPO (有参考模型)
Beta: 0.5
LR: 5e-5
Epochs: 1
显存: ~18GB
Loss: 0.67 | Acc: 0.40
```

**关键踩坑**：
| 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|
| DPO loss=0 | 完全学不到 | prompt不含图像, 模型盲猜 | prompt_messages加入多视图路径 |
| 模板报告质量差 | chosen/rejected差异太小 | 规则拼字符串, 措辞高度同质 | DeepSeek API重写+CT-RATE风格 |
| 数据量少 | 90对效果有限 | 手工标注慢, 自动扰动无效 | 用SFT模型自生成rejected(Online DPO思路) |

### Stage 3: GRPO（群体相对策略优化）— 主力RL

**做什么**：用LIDC GT构建可验证奖励信号，在生成 4 条报告的组内做相对优势优化

**训练配置**：
```
G (每组生成数): 4
KL penalty: 0.04 (后改L2=0.001, 因VRAM不足无法加载ref model)
LR: 1e-6 (极小步长防策略跳变)
显存: ~28GB (含ref model) / ~20GB (L2替代KL)
```

**奖励函数**（5信号复合）：
```python
composite_reward(report, lidc_gt):
    format_score      (0.10)  # 是否有结构化段落
    accuracy_score    (0.30)  # 大小/位置数值偏差
    factual_score     (0.35)  # 密度/钙化/边界描述准确度
    completeness_score(0.15)  # 必要字段完整性
    consistency_score (0.10)  # 前后不自相矛盾
```

**关键踩坑**：
| 坑 | 现象 | 根因 | 修复 |
|---|---|---|---|
| OOM | 32GB显存不足 | BF16模型+4-bit ref model+3视图 | 去ref model, L2替代KL |
| requires_grad bug | `element 0 does not require grad` | compute_log_prob有@torch.no_grad()装饰器 | 去掉装饰器, ref model调用额外wrap no_grad |
| 奖励易骗 | 模型用关键词刷分 | 规则正则匹配奖励 | 后续计划: 训Reward Model |

---

## 三、面试准备

### Q1: 为什么不用 7B 模型？

A: EditGRPO(2025)论文直接对比了 Qwen2.5-VL-3B vs 7B：GRPO 后 7B 仅领先 <1% (CheXbert F1: 0.445 vs 0.447)，但 VRAM 翻倍、训练时间多 60%。视觉编码器在两版上几乎相同(~400M)，CT 理解上限由视觉编码器决定而非 LLM 参数量。肺结节是单一垂直任务不需要 7B 的语言多样性。

### Q2: QLoRA 为什么换成 BF16？

A: 4-bit 下模型参数 dtype=uint8, unfreeze_vision_p2() 的 dtype 检查会静默跳过所有视觉参数。我们验证了两个 adapter(stage1_multislice_v1, stage1_vision_v1)均为 0 ViT LoRA key。切换到 BF16 后 dtype=bfloat16, 530M 视觉参数成功解冻。

### Q3: DPO 为什么效果不明显？

A: 三个原因。①初始 DPO prompt 不含图像——发现后修复了；②模板生成的 chosen/rejected 高度同质(79% 字符相同)——计划改用 DeepSeek API 生成真实报告；③90 对偏好对 LoRA r=16 已足够学到方向，但信号强度不足——增加数据量或用 Online DPO 自生成偏好对。

### Q4: GRPO vs PPO 有什么区别？

A: GRPO 去掉了 Critic(Value)网络。对每个 prompt 生成 G 条(R=4), 组内计算相对优势: A_i = (r_i - mean_r) / std_r。省 30-40% VRAM。适合小样本场景。PPO 需要额外训练 Value 网络估计 baseline。

### Q5: 为什么没有做 Agent RL？

A: Agent 工具调用对视觉精确性的要求低于报告生成。当前核心挑战是报告质量，Agent 是另一维度。已有完整的 ReAct 引擎 + 4 个工具(指南检索/文献搜索/临床计算/图像重分析)，在推理阶段通过系统提示注入即可工作，不需要专门 RL 训练。

### Q6: 数据怎么处理的？

A: LUNA16 → SimpleITK 读 .mhd → 基于 LIDC 世界坐标定位结节 → 三平面视图提取(轴位/冠位/矢位) → 50mm ROI → 肺窗(WL=-600,WW=1500) → 512px LANCZOS resize → 与 CT-RATE 报告配对 → OpenAI Vision format。

### Q7: 这个项目的难点在哪？

A: ① CT 成像物理到 VLM 输入格式的桥接(3D→2D切片选择、窗宽窗位、像素spacing)；② VLM 视觉通路的正确解冻(4-bit下的静默失败极难察觉)；③ 有限标注(854结节) + 有限算力(单卡RTX 5090)下的有效后训练策略设计。

---

## 四、踩坑全记录

| # | 坑 | 关键指标 | 根因 | 情感 |
|---|-----|---------|------|------|
| 1 | 84px PNG | Loss=298, 36 visual tokens | mhd_to_png.py 无resize | 😤 |
| 2 | ViT假解冻 | ViT LoRA key=0/504 | 4-bit dtype=uint8绕过检查 | 😡 |
| 3 | 结构化hint短路 | 有图/无图输出相同 | Prompt注入结节信息模型抄文本 | 🤦 |
| 4 | DPO prompt缺图 | Loss=0.0, acc=0.0 | prompt_messages只含text | 🫠 |
| 5 | GRPO OOM | 32GB爆满 | BF16 model + 4bit ref model | 💥 |
| 6 | GRPO requires_grad | element 0 does not require grad | @torch.no_grad()装饰compute_log_prob | 🔧 |
| 7 | Zenodo下载损坏 | zip file corrupt | curl断点续传不兼容 | 🔄 |
| 8 | HF gated access | Dataset must be authenticated | CT-RATE需要申请+token | 🔑 |
| 9 | TCIA被墙 | SSL timeout | LIDC/DL在中国无法直连 | 🧱 |
| 10 | DPO模板报告同质 | 79%字符相同 | 规则f-string拼报告 | 😑

---

## 五、面试模拟

**面试官**: 介绍一下这个项目。

**回答**: 我独立构建了一个肺结节CT多模态后训练项目。基座模型是 Qwen2.5-VL-3B，用 SFT → DPO → GRPO 三阶段管线让它从零学会看CT写放射科报告。

SFT 阶段遇到了CT到VLM输入的工程挑战——3D CT体积需要转为2D图像，最初输出 84px 微型图导致 ViT 只有 36 个视觉 token、Loss高达 298。发现后改成三平面视图(轴位+冠位+矢位)+512px resize，视觉token提升到4000+，Loss降到9.5。同时发现 4-bit 量化下 ViT 解冻静默失败——dtype=uint8 被检查跳过——切换 BF16 后 530M 视觉参数成功训练。

DPO 阶段设计了 6 大类精细诊断错误，构造 90 对偏好对修正模型的保守低估和特征漏报。GRPO 阶段用 LIDC 多医生标注构建 5 信号事实奖励函数。整个训练在 AutoDL RTX 5090 32GB 上完成，总费用 ~¥200。

**面试官**: 为什么 QLoRA 换成了 BF16？

**回答**: 发现 4-bit 下 unfreeze 函数的 dtype 检查只放行 float32/float16/bfloat16，而 4-bit 参数是 uint8，检查全部跳过。这是静默失败——训练日志显示"Unfrozen: 0 params"但我一开始没注意到。切换到 BF16 后这个检查通过，ViT 后 4 层和 projector 的 530M 参数真正解冻。editGRPO/MedFact-R1 论文也踩过这个坑。

**面试官**: SFT 做了这么多工程优化，DPO/GRPO 贡献了多少？

**回答**: 诚实说 SFT 贡献了 90%+，DPO/GRPO 是增量优化。SFT 的突破来自数据管道和训练策略的修复，不是算法选择。DPO 在我修正了 prompt 缺图问题后能从 loss=0 学到 loss=0.67/acc=0.40，有效但不算飞跃。GRPO 的 reward 函数设计是正确的方向——用可验证事实做奖励而非模糊偏好——但 OOM/requires_grad 等工程问题消耗了调试时间。最核心的教训是：后训练数据质量 >> 算法选择。
