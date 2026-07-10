# CT-VLLM 训练日志 — 2026-07-06

## 环境

- **GPU**: NVIDIA RTX 5090 32GB (AutoDL)
- **CUDA**: 13.0 / Torch 2.11.0+cu130
- **Python**: 3.12
- **基座模型**: Qwen2.5-VL-3B-Instruct (4-bit QLoRA via Unsloth 2026.6.9)
- **Transformers**: 5.12.1
- **训练框架**: Unsloth + 自定义 PyTorch 训练循环

---

## 一、今天踩的坑总览

### 坑 1：Stage 2 SimPO 纯文本过拟合

**现象**: Train acc 0.91, Val acc 0.65, gap 0.26

**根因**: DPO 偏好对不含 CT 图像。模型在 Stage 1 学了 CT → 报告，Stage 2 蒙着眼睛学"这段文字比那段好"，LoRA 权重把视觉映射覆盖了。

**解决**: DPO prompt 必须带 `{"type": "image", "image": "/path/to/.mhd"}`，保持视觉锚定。

### 坑 2：pylidc 坐标匹配

**现象**: 第一次匹配 0/47 成功

**根因**:
1. pylidc centroid 是**体素坐标** (x,y,z voxel index)，不是世界坐标 mm
2. pylidc 和 LUNA16 的 X/Y 轴可能互换（图像坐标 vs 世界坐标约定不同）
3. `cluster_annotations()` 某些 scan 失败（"Failed to reduce all groups"）

**解决**:
1. 用 `SimpleITK.TransformPhysicalPointToIndex()` 把 LUNA16 世界坐标转体素
2. 匹配时尝试两种轴序：`(x,y,z)` 和 `(y,x,z)` 取 min distance  
3. cluster 失败时回退到原始 `scan.annotations`，每个 annotation 单独作为一个结节
4. 体素空间匹配阈值：`max(15, 3 × diameter / spacing)`

### 坑 3：Numpy 版本兼容

**现象**: `AttributeError: module 'numpy' has no attribute 'int'`

**解决**: `sed -i 's/\.astype(np\.int)/.astype(np.int64)/g' /root/miniconda3/lib/python3.12/site-packages/pylidc/Contour.py`

### 坑 4：Unsloth tokenizer 拦截

**现象**: `RuntimeError: Unsupported image file. Only jpeg, png, webp and gif are currently supported.`

**根因**: Unsloth 2026.6.9 修补了 Qwen2.5-VL 的 processor.`__call__`，所有 tokenizer 调用都被路由到 image_processor，即使输入是纯文本。

**解决**: 在 Stage 2 中加载 `AutoTokenizer.from_pretrained()` (bare tokenizer)，绕过 Unsloth 补丁。**注意**：bare tokenizer 不处理图像，所以 SimPO 的 log-prob 是无条件 P(text)，不是 P(text|image)。需要后续改进为图像条件化的 SimPO。

### 坑 5：DataLoader collate 错误

**现象**: `TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'NoneType'>`

**根因**: DPO 数据格式是 `{prompt: [{role, content: [{image}, {text}]}], chosen: [...], rejected: [...]}`，嵌套深度 3-4 层。PyTorch 默认 collate 无法处理。

**解决**: 自定义 `collate_fn=lambda batch: batch` 返回原始列表，然后在训练循环里手动解包。

### 坑 6：Stage 1 pickle 保存错误

**现象**: `_pickle.PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>`

**解决**: 不用 `trainer.save_model()`，改为 `model.save_pretrained() + tokenizer.save_pretrained()`

### 坑 7：LUNA16 .mhd 文件损坏

178/267 个 .mhd/.raw 文件实际大小与 header 声明的 DimSize 不匹配（truncated raw data）。只有 89 个 subset0 数据完好。损坏文件移至 `corrupted_backup/`。

---

## 二、数据集获取进度

### 下载状态

| 数据集 | 进度 | 大小 | 位置 |
|--------|------|------|------|
| LUNA16 subset0 | ✅ 已下载，89 个完好 | ~8GB | `/root/autodl-tmp/data/LUNA16/images/` |
| LUNA16 subset1 | ❌ 解压失败，需重下 | | 本地 Mac 下载中 |
| LUNA16 subset2-6 | 🔄 本地 Mac 下载中 | ~10GB each | `~/Downloads/LUNA16/subset{2-6}.zip` |
| LUNA16 subset7 | ✅ 已下载 | | `/root/autodl-tmp/data/LUNA16/images/` |
| LUNA16 subset8 | 🔄 本地 Mac 下载中 | | `~/Downloads/LUNA16/subset8.zip` |
| LUNA16 subset9 | ❌ 未下载 | | |
| CT-RATE reports | ✅ 已下载 | 47,149 条 | `/root/autodl-tmp/data/CT-RATE/reports.jsonl` |
| LIDC-IDRI 标注 | ✅ pylidc 自动获取 | 1018 scans | `~/.pylidc/` (pylidc 缓存) |

### 本地下载命令

```bash
# Mac 本地 — 逐个下载，支持断点续传
for i in 2 3 4 5 6 8; do
  curl -C - -L -o ~/Downloads/LUNA16/subset${i}.zip \
    "https://zenodo.org/records/3723295/files/subset${i}.zip"
done
```

### 上传到 AutoDL

```bash
# 单个文件 scp
scp -P <端口> ~/Downloads/LUNA16/subset*.zip root@<IP>:/root/autodl-tmp/data/LUNA16/

# AutoDL 上解压
cd /root/autodl-tmp/data/LUNA16
for f in subset*.zip; do
  unzip -t "$f" && unzip -o "$f" -d images/ && echo "$f done"
done
```

---

## 三、训练数据构造逻辑 (v2)

### 整体架构

```
LUNA16 .mhd CT 扫描 (图像输入)
    │
    ├── annotations.csv (coordX/Y/Z + diameter)
    │
    └──→ ① lidc_match.py: pylidc 提取 9D 特征 + 坐标匹配
            │
            ├── 输出: nodule_features.json
            │   每个结节有: subtlety, internalStructure, calcification, 
            │   sphericity, margin, lobulation, spiculation, texture, malignancy
            │
            └──→ ② sft_dataset_builder.py: 特征 → 报告
                   │
                   │  两种模式:
                   │  • 增强模板 (无需 API): 用特征值生成结构化报告
                   │  • DeepSeek API (可选): 用 CT-RATE 风格示例做 few-shot 生成专业报告
                   │
                   ├── 输出: sft_train.jsonl + sft_val.jsonl
                   │   格式: {messages: [{role: user, content: [{image}, {text}]},
                   │                    {role: assistant, content: [{text}]}],
                   │           metadata: {seriesuid, diameter, malignancy, ...}}
                   │
                   └──→ ③ dpo_dataset_builder.py: SFT → DPO 偏好对
                          │
                          │  prompt = SFT 的 user message (含 CT 图像!) ← 关键改动
                          │  chosen = SFT 的高质量报告
                          │  rejected = 扰动版本 (尺寸偏差/恶性翻转/密度翻转/段落删除)
                          │
                          输出: dpo_train.jsonl + dpo_val.jsonl
```

### 关键设计决策

1. **prompt 必须含图像** — DPO 训练时模型眼睛是睁开的，不会丢视觉映射
2. **LIDC 特征是 GT** — GRPO reward function 的 accuracy + factual 分量现在能用
3. **报告生成分层** — 有 DeepSeek API 用 API，没有用增强模板（比旧版 "Lung nodule, 12mm at (x,y,z)" 好 10 倍）

---

## 四、训练流程

### 准备阶段（一次性）

```bash
# 1. LIDC 特征提取（首次运行自动下载 pylidc 缓存）
python data/lidc_match.py
# 输出: /root/autodl-tmp/data/nodule_features.json

# 2. SFT 数据构建
python data/sft_dataset_builder.py \
  --features /root/autodl-tmp/data/nodule_features.json \
  --output /root/autodl-tmp/data/sft_v2
# 如果设置了 DEEPSEEK_API_KEY 环境变量，自动用 API 生成报告
# 输出: /root/autodl-tmp/data/sft_v2/{sft_train.jsonl, sft_val.jsonl}

# 3. DPO 数据构建
python data/dpo_dataset_builder.py \
  --sft_data /root/autodl-tmp/data/sft_v2/sft_train.jsonl \
  --output /root/autodl-tmp/data/dpo_v2 \
  --variants_per_sample 3
# 输出: /root/autodl-tmp/data/dpo_v2/{dpo_train.jsonl, dpo_val.jsonl}
```

### Stage 1: SFT

```bash
python training/stage1_sft.py \
  --data_dir /root/autodl-tmp/data/sft_v2 \
  --output /root/autodl-tmp/outputs/stage1_sft_v2 \
  --epochs 3 \
  --batch_size 4 --grad_accum 4
```

**已知问题**: 训练完成时 pickle 保存失败，需手动保存 adapter:
```bash
python3 -c "
from unsloth import FastVisionModel
from peft import PeftModel
model, tok = FastVisionModel.from_pretrained('/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct', load_in_4bit=True, use_gradient_checkpointing='unsloth')
model = PeftModel.from_pretrained(model, '/root/autodl-tmp/outputs/stage1_sft_v2/checkpoint-xxx')
model.save_pretrained('/root/autodl-tmp/outputs/stage1_sft_v2/lora_adapter')
tok.save_pretrained('/root/autodl-tmp/outputs/stage1_sft_v2/lora_adapter')
"
```

### Stage 2: SimPO

```bash
python training/stage2_simpo.py \
  --adapter /root/autodl-tmp/outputs/stage1_sft_v2/lora_adapter \
  --data_dir /root/autodl-tmp/data/dpo_v2 \
  --output /root/autodl-tmp/outputs/stage2_simpo_v2 \
  --epochs 1 \
  --beta 0.5 --gamma 0.3 \
  --batch_size 2 --grad_accum 8
```

**保守参数说明**: 
- `epochs=1`: 避免过拟合
- `beta=0.5, gamma=0.3`: 比默认(0.1, 0.1)更保守，对 stage1 权重的改动更小

### Stage 3: GRPO

```bash
python training/stage3_grpo.py \
  --adapter /root/autodl-tmp/outputs/stage2_simpo_v2/lora_adapter \
  --output /root/autodl-tmp/outputs/stage3_grpo_v2
```

### Val 评估（每个 Stage 结束后跑）

```bash
python3 2>/dev/null << 'PYEOF'
import torch, torch.nn as nn, statistics, json, warnings, os
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from unsloth import FastVisionModel
from peft import PeftModel
from datasets import load_dataset

MODEL = "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
ADAPTER = "/root/autodl-tmp/outputs/stage2_simpo_v2/lora_adapter"

def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        parts = [extract_text(x) for x in content]
        return ' '.join(p for p in parts if isinstance(p, str))
    if isinstance(content, dict):
        if 'text' in content: return str(content['text'])
        if 'content' in content: return extract_text(content['content'])
    return ''

def avg_log_prob(logits, input_ids, mask):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    lp = nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1) * shift_mask
    return token_lp.sum(-1) / shift_mask.sum(-1).clamp(min=1)

model, _ = FastVisionModel.from_pretrained(MODEL, load_in_4bit=True, use_gradient_checkpointing='unsloth')
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.pad_token = tok.eos_token

ds = load_dataset('json', data_files={'val': '/root/autodl-tmp/data/dpo_v2/dpo_val.jsonl'})['val']
print(f"Val samples: {len(ds)}")

correct, total = 0, 0
with torch.no_grad():
    for item in ds:
        ct = extract_text(item['chosen'])
        rt = extract_text(item['rejected'])
        c = tok(ct, return_tensors='pt', padding=True, truncation=True, max_length=2048).to('cuda')
        r = tok(rt, return_tensors='pt', padding=True, truncation=True, max_length=2048).to('cuda')
        c_lp = avg_log_prob(model(input_ids=c.input_ids, attention_mask=c.attention_mask).logits, c.input_ids, c.attention_mask)
        r_lp = avg_log_prob(model(input_ids=r.input_ids, attention_mask=r.attention_mask).logits, r.input_ids, r.attention_mask)
        correct += (c_lp > r_lp).float().item()
        total += 1

print(f"Val Acc: {correct/total:.4f} ({int(correct)}/{total})")
PYEOF
```

**判断标准**:
- Val acc > 0.80 → ✅ 良好
- Val acc 0.70-0.80 → ⚠️ 轻微 gap，可接受  
- Val acc < 0.70 → ❌ 过拟合，需要调参

---

## 五、当前卡点

### 卡点 1：Stage 2 SimPO DataLoader collate 修复尚未跑通

`stage2_simpo.py` 的 sed 修改可能不完整。**最可靠的方式**：在 AutoDL 上直接 `cat > file << 'EOF'` 全量重写该脚本。

需要修改的两处：
1. DataLoader 加 `collate_fn=lambda batch: batch`
2. 训练循环的 `for bidx, batch` → `for bidx, samples`，然后 `batch = {"chosen": [...], "rejected": [...]}`

### 卡点 2：SimPO 是无条件 log-prob，未使用图像条件化

当前 `bare_tok` 绕过 VLM processor，log-prob 是 P(report_text) 不是 P(report_text | CT_image)。

**影响**：SimPO update 方向仍是纯文本偏好，没有视觉锚定。虽然数据格式里 prompt 有图像，但前向传播时没用到。

**修复方向**（下一步改进）：
- 用 Unsloth tokenizer 对完整 `[prompt_messages + chosen/rejected]` 做前向
- 只对 response 部分的 token 计算 log-prob
- .mhd 需要先处理成 Qwen2.5-VL image processor 支持的格式
- 或者先用 SimpleITK 把 .mhd 在结节坐标处提取 PNG 切片

### 卡点 3：数据量仍有限

当前 179 个完好的 CT 扫描，约 246 个结节。全量 LIDC 匹配后预期 ~800+ 结节。需下载完 subset 1-9 并验证每个 .raw 的完整性。

### 卡点 4：报告质量取决于 DeepSeek API

增强模板虽然比旧版模板好，但仍是规则生成的，缺少真实放射科报告的自然语言多样性。

**建议**：全量训练时一定要用 `--deepseek_api_key` 或设 `DEEPSEEK_API_KEY` 环境变量。成本很低（~$0.03/1000 结节）。

---

## 六、快速恢复训练

换机子后，从零开始 4 步：

```bash
# 0. 确认环境
python3 -c "import pylidc, SimpleITK; from unsloth import FastVisionModel; print('OK')"
ls /root/autodl-tmp/data/LUNA16/images/*.mhd | wc -l
ls /root/autodl-tmp/data/CT-RATE/reports.jsonl

# 1. LIDC 特征
python data/lidc_match.py
# → /root/autodl-tmp/data/nodule_features.json

# 2. SFT 数据
python data/sft_dataset_builder.py \
  --features /root/autodl-tmp/data/nodule_features.json \
  --output /root/autodl-tmp/data/sft_v2
# → /root/autodl-tmp/data/sft_v2/{sft_train,sft_val}.jsonl

# 3. DPO 数据  
python data/dpo_dataset_builder.py \
  --sft_data /root/autodl-tmp/data/sft_v2/sft_train.jsonl \
  --output /root/autodl-tmp/data/dpo_v2
# → /root/autodl-tmp/data/dpo_v2/{dpo_train,dpo_val}.jsonl

# 4. 训练
python training/stage1_sft.py --data_dir ... --output ...
# (手动保存 adapter)
python training/stage2_simpo.py --adapter ... --data_dir ... --output ...
```

---

## 七、文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `data/lidc_match.py` | ✅ v2 | pylidc 特征提取，体素坐标匹配（已修） |
| `data/sft_dataset_builder.py` | ✅ v2 | DeepSeek/增强模板报告生成 |
| `data/dpo_dataset_builder.py` | ✅ v2 | 图像锚定 DPO 对 |
| `training/stage1_sft.py` | ⚠️ 可用 | 有 pickle 保存问题，需手动保存 |
| `training/stage2_simpo.py` | ⚠️ 待修复 | DataLoader collate + 无条件 log-prob 两个问题 |
| `training/stage3_grpo.py` | 未测试 | |

---

## 八、关键命令速查

```bash
# pylidc numpy 兼容
sed -i 's/\.astype(np\.int)/.astype(np.int64)/g' \
  /root/miniconda3/lib/python3.12/site-packages/pylidc/Contour.py

# 验证 .mhd 完整性
python3 -c "
import os, struct
d = '/root/autodl-tmp/data/LUNA16/images'
for f in sorted(os.listdir(d)):
    if f.endswith('.mhd'):
        p = os.path.join(d, f)
        # 读 header 获取 DimSize
        with open(p) as fh:
            for l in fh:
                if 'DimSize' in l:
                    expected = int(l.split('=')[1].strip().split()[2])
        raw = p.replace('.mhd','.raw')
        actual = os.path.getsize(raw)
        if actual != expected:
            print(f'BAD: {f} ({actual} != {expected})')
"

# wandb offline
export WANDB_MODE=offline
```

---

---

## 九、最终训练策略（832 样本）— 2026-07-10

### 数据确认

| 指标 | 实际值 |
|------|--------|
| CT 扫描 (.mhd) | 704 |
| 匹配结节 | 854 |
| 训练样本 | 832 |
| 损坏文件 | 1 |

### 为什么砍掉原计划的部分 Stage

| Stage | 决策 | 原因 |
|-------|------|------|
| **SimPO** | ❌ 跳过 | DPO 偏好对差异太小（模板扰动），梯度信号弱；当前实现有图像条件化 bug（用 bare tok 算无条件 P(text)，模型看不到 CT）。832 样本的偏好对质量不够 |
| **Agent SFT/GRPO (4a/4b)** | ❌ 跳过 | 需要 ReAct 工具调用轨迹数据，当前没有。且基础报告生成任务未收敛前不应引入工具使用 |
| **GRPO (Stage 3)** | ⏸️ 延后到 ReST 收敛后 | GRPO 需要足够多的 prompt → 多样探索，ReST 2-3 轮后模型和数据质量都提升了再做，lr 1e-5 极小步长 |

### 最终 Pipeline

```
Stage 1 SFT (已完成, stage1_full_v2)
    ↓
ReST Round 1 (当前正在跑, stage1_rest_v1)
    ↓
ReST Round 2 (adapter=rest_v1, output=rest_v2)
    ↓
[ReST Round 3 — 如果 reward_median 还在涨]
    ↓
轻量 GRPO (lr=1e-5, max_steps=200, beta=0.01)
    ↓
评估
```

### ReST 收敛判断

每轮完成后看 `rest_stats.json` 的 `reward_median`：
- Round N+1 > Round N × 1.05 → 继续下一轮
- Round N+1 ≈ Round N → 收敛，进入 GRPO

### 各阶段命令

```bash
# ReST Round 2
python scripts/rest_self_distill.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter /root/autodl-tmp/outputs/stage1_rest_v1/lora_adapter \
    --data_dir /root/autodl-tmp/data/sft_full_v2 \
    --output /root/autodl-tmp/outputs/stage1_rest_v2 \
    --no_4bit --batch_generate \
    --n_generations 4 --max_new_tokens 512

# ReST Round 3（可选）
# adapter → rest_v2, output → rest_v3

# 轻量 GRPO（ReST 收敛后）
python training/stage3_grpo.py \
    --adapter /root/autodl-tmp/outputs/stage1_rest_v2/lora_adapter \
    --data_dir /root/autodl-tmp/data/sft_full_v2 \
    --output /root/autodl-tmp/outputs/stage3_grpo_light \
    --num_generations 8 --lr 1e-5 --beta 0.01 --max_steps 200
```

### 为什么 ReST 是小数据最优解

- 不需要偏好对（不像 SimPO）
- 不需要大量探索（不像 GRPO）
- generate → filter → SFT 是封闭循环，数据质量逐轮提升
- 每轮用更好的模型生成更好的样本 → 正向飞轮

---

## 十、P4 ReST 自蒸馏性能优化 — 2026-07-10

### 问题

`rest_self_distill.py` 在 RTX 5090 上跑 832 样本 × 4 次生成 ≈ **9+ 小时**，每个样本 ~41 秒。

### 根因分析

| 瓶颈 | 严重度 | 说明 |
|------|--------|------|
| 图像重复编码 | 🔴🔴🔴 | `tok(text=..., images=...)` 在 4 次生成的循环内，每次重新 load/resize/normalize CT 图 |
| Prompt KV-cache 不复用 | 🔴🔴🔴 | 同一 prompt 的 4 次 `model.generate()` 各自独立计算完整前向 |
| 4-bit 量化开销 | 🟡🟡 | RTX 5090 32GB 跑 3B 模型绰绰有余，dequantize 反而拖慢推理 |
| 无 Flash Attention 2 | 🟡 | 环境 `FA2 = False`，长序列 attention 慢 |
| 无批处理 | 🟡 | 串行逐个样本，GPU 利用率低 |

### 改动内容 (`scripts/rest_self_distill.py`)

1. **新增 `--no_4bit` 参数** — 禁用 4-bit 量化，用 BF16 推理（3B 模型仅需 ~6GB，RTX 5090 完全够）
2. **新增 `--batch_generate` 参数** — 用 `num_return_sequences=N` 一次 `generate()` 调用生成 N 条，共享 prompt KV-cache
3. **图像预处理提到循环外** — `tok()` 只在样本级调用一次，不在 N 次生成循环内重复调用
4. **训练阶段也尊重 `use_4bit`** — 两处模型加载（推理 + 训练）统一使用 `use_4bit` 变量

### 优化后启动命令

```bash
# 推荐：全量跑
python scripts/rest_self_distill.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter /root/autodl-tmp/outputs/stage1_full_v2/lora_adapter \
    --data_dir /root/autodl-tmp/data/sft_full_v2 \
    --output /root/autodl-tmp/outputs/stage1_rest_v1 \
    --no_4bit \
    --batch_generate \
    --n_generations 4 \
    --max_new_tokens 512

# 极速验证模式（200 样本，~30-40 分钟）
python scripts/rest_self_distill.py \
    ... \
    --no_4bit \
    --batch_generate \
    --n_generations 2 \
    --max_new_tokens 256 \
    --max_samples 200
```

### 预估提速

| 优化 | 单样本耗时 | 832 样本总时间 |
|------|-----------|---------------|
| 原始 (4-bit, 无缓存复用) | ~41s | ~9h |
| + `--no_4bit` | ~25s | ~5.5h |
| + `--batch_generate` (KV-cache 共享) | ~12s | ~2.5h |
| + `n_generations=2, max_tokens=256` | ~4s | ~1h |
| + 安装 flash-attn | ~2-3s | ~30-40min |

### 可选：AutoDL 上装 flash-attn

```bash
pip install flash-attn --no-build-isolation
```

---

## 十一、多切片 Pipeline 执行记录 — 2026-07-10

### 关键发现：原始 PNG 切片不含结节

`mhd_to_png.py` 取的是 CT 扫描**中心轴向切片** (`z_center = shape[0] // 2`)，而不是结节实际 Z 坐标。`sft_dataset_builder.py` 用 `{seriesuid}.png` 完全不使用 `coordZ`。VLM 可能根本没看到结节。

### 改动

| 文件 | 改动 | 状态 |
|------|------|------|
| `data/mhd_to_png.py` | 新增 `--mode nodule_slices`：读取 `nodule_features.json`，在每结节 `coordZ` 处取 N 层 PNG | ✅ |
| `data/sft_dataset_builder.py` | 新增 `--slices_per_nodule`：glob 匹配多切片 PNG → 每样本含多张图 | ✅ |

### 执行记录

```bash
# Step 1: 多切片 PNG 提取
python data/mhd_to_png.py --mode nodule_slices \
    --images_dir /root/autodl-tmp/data/LUNA16/images \
    --features /root/autodl-tmp/data/nodule_features.json \
    --output_dir /root/autodl-tmp/data/LUNA16/images_png \
    --n_slices 3 --slice_spacing_mm 2.0
# 结果: 854 结节 → 2562 张 PNG (0 失败), 2.5 分钟

# Step 2: 重建 SFT 数据
python data/sft_dataset_builder.py \
    --features /root/autodl-tmp/data/nodule_features.json \
    --output /root/autodl-tmp/data/sft_multislice \
    --slices_per_nodule 3
# 结果: 1438 train / 270 val (CN+EN), 3 张图/样本

# Step 3: 多切片 SFT (进行中)
python training/stage1_sft.py \
    --data_dir /root/autodl-tmp/data/sft_multislice \
    --output /root/autodl-tmp/outputs/stage1_multislice_v1 \
    --epochs 3 --lr 2e-4 \
    --batch_size 1 --grad_accum 8 \
    --unfreeze_vision 1 --unfreeze_vit_layers 2 --vit_lr_ratio 0.1
```

### 数据对比

| 版本 | 样本数 | 图/样本 | 有效视觉数据 | 切片位置 |
|------|--------|---------|-------------|---------|
| 旧 (中心切片) | 832 CN+EN | 1 | 832 张 | CT 中心（可能无结节） |
| 新 (多切片) | 1438 CN+EN | 3 | 4314 张 | 结节 Z 坐标 ±2mm |

### 完整管线 (待执行)

```
Stage 1 SFT (多切片) → ReST×2 → SimPO → 轻量 REINFORCE → 评估
```

详见 [plan-fuzzy-mountain.md](.claude/plans/plan-fuzzy-mountain.md)。

---

## 十二、重大坑：P1 Structured Hints 导致 Text-Copy 短路 — 2026-07-10

### 现象

SFT 训练 loss 正常下降（302→255→203），但模型生成质量极差——输出"无法确定"、"请提供图像"等泛泛回复，完全不用视觉信息。

### 诊断方法

对比**有图像**和**无图像**生成结果：
```python
# 有图 vs 无图 → 输出完全相同 → 模型在抄 prompt，没看图
```

### 根因

P1 的 "structured clinical hints" 把结节位置+直径注入 prompt：
```
请评估这个肺结节的恶性风险。
[临床提示：结节位于右上叶，直径约 12.3mm，建议重点关注其边界特征和密度类型...]
```
模型学会直接从 hint 文本复制答案，**完全绕过了视觉通路**。LoRA 权重学到的是 text→text mapping，不是 image→text。

### 教训

> **任何注入 prompt 的提示信息都会成为模型偷懒的捷径。**
> VLM 训练中，"让模型看图"的唯一方式是不给它任何文字线索。

### 修复

1. `sft_dataset_builder.py` 新增 `--no_structured_hint` 参数
2. 无条件时 prompt 变成：
   ```
   请评估这个肺结节的恶性风险。包括大小、边界、密度类型、钙化状态等关键指标，并给出Lung-RADS分级。
   ```
3. 重建数据：1472 train / 236 val（no-hint + 多切片）

### 验证标准

训练完成后必须跑有图 vs 无图对比测试，输出**明显不同**才算视觉学习成功。

---

## 十三、训练策略澄清：视觉感知 vs 报告质量 — 2026-07-10

### 关键认知

| 阶段 | 目标 | 数据依赖 |
|------|------|---------|
| **Luna16 SFT** | VLM 学会"看到结节"（视觉感知） | CT 图像 + 结节特征 GT |
| **SimPO / GRPO** | 优化报告质量（文本生成） | 偏好对 / reward 打分 |

这两个是**不同的能力维度**。Luna16 数据少不影响 SimPO/GRPO——后者优化的是"报告写得好不好"，前者确保"模型看图说话"。不能用"视觉数据少"为理由砍掉 SimPO/GRPO。

### 最终管线

```
Stage 1 SFT (多切片, no-hint) → ReST × 2 → Stage 2 SimPO (image-conditioned v3) → Stage 3 轻量 REINFORCE → 评估
```

---

## 十四、WandB 日志集成 — 2026-07-10

### 新增文件

| 文件 | 说明 |
|------|------|
| `training/wandb_utils.py` | 共享 wandb logger，自动 fallback（未安装或 DISABLED 时不报错） |

### 各阶段日志 key

| Stage | 指标 |
|-------|------|
| SFT | `train/loss`, `train/lr`, `train/epoch` |
| ReST | `generate/r_mean`, `generate/r_max`, `filter/reward_*`, `rest_sft/loss` |
| SimPO | `simpo/loss`, `simpo/acc`, `simpo/lr` |
| GRPO | `grpo/reward`, `grpo/loss`, `grpo/lr` |

### 使用

```python
from wandb_utils import get_logger
wb_logger = get_logger("sft", args.output, config=args)
wb_logger.log({"train/loss": 3.5}, step=100)
wb_logger.finish()
```

---

## 十五、ReST 多切片运行记录 — 2026-07-10

### 数据确认

- sft_nohint: 1472 train / 236 val, 3 slices/sample
- 验证: `stage1_multislice_v1` adapter **视觉在用** ✓（有图 vs 无图输出不同）

### ReST Round 1 运行参数

```bash
python scripts/rest_self_distill.py \
    --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --adapter /root/autodl-tmp/outputs/stage1_multislice_v1/lora_adapter \
    --data_dir /root/autodl-tmp/data/sft_nohint \
    --output /root/autodl-tmp/outputs/rest_round1 \
    --no_4bit --batch_generate \
    --n_generations 4 --max_new_tokens 384
```

### 速度基准

| 配置 | 速度 | 总时间(1472样本) |
|------|------|-------------------|
| 单切片 + 4bit + 串行 | 41s/it | ~9h |
| 单切片 + BF16 + batch | 9.5s/it | ~2h |
| 3切片 + BF16 + batch, max_tokens=512 | 8.4s/it | ~3.4h |
| 3切片 + BF16 + batch, max_tokens=384 | ~6s/it | ~2.5h |

### max_new_tokens 选择

中文肺结节报告典型长度 200-400 字 → tokenizer 约 250-500 tokens。
- 512: 冗余，不截断
- 384: 覆盖大部分报告，少量长报告截断不影响 reward
- 256: 太紧，容易截断关键内容

---

## 十六、Stage 2 SimPO v3 修复 (image-conditioned) — 2026-07-10

### v2 bug

`stage2_simpo.py` v2 用 bare tokenizer（`AutoTokenizer.from_pretrained`）算无条件 P(text)，模型看不到 CT 图像。acc≈0.50 随机。

### v3 修复

改用 VLM tokenizer + 完整前向（`pixel_values` + `image_grid_thw`），只在 assistant response 部分计算 log-prob。模型"看着"CT 图像判断报告优劣。

### 关键代码模式

所有训练脚本统一使用：
```python
def match_images(text, img_paths):
    n = text.count('<|vision_start|>')
    imgs = list(img_paths)
    while len(imgs) < n: imgs += imgs
    return imgs[:n]

enc = tok(text=[text], images=match_images(text, img_paths), return_tensors="pt")
```

---

## 十七、Stage 3 不是真正的 GRPO — 2026-07-10

当前 `stage3_grpo.py` 实际是**单样本 REINFORCE + L2 log-prob 惩罚**：

| 维度 | 真正 GRPO | 当前实现 |
|------|----------|---------|
| 优势估计 | 同 prompt 生成 G 条，组内相对优势 | 单条 reward |
| KL 正则 | KL(π ‖ π_ref) | (log_prob)² L2 |
| 参考模型 | 保存 frozen checkpoint | 无 |

832 样本场景下单样本 REINFORCE 更合适（组内归一化不稳定），命名保持 `grpo` 但心里有数。

---

## 十八、4-bit 下 ViT 解冻静默失败 — 2026-07-10

### 发现

`unfreeze_vision_p2` 函数有 dtype 检查：
```python
if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
    p.requires_grad = True
```
4-bit 量化下参数 dtype 是 `torch.uint8`，检查永远 False → ViT 完全冻结。

### 验证

`stage1_multislice_v1` 虽然 `--unfreeze_vision 1`，但实际 ViT 参数更新为 0。

### 修复

1. 加 `--no_4bit` flag 加载 BF16 权重
2. dtype 检查改为 warn 而非静默跳过：不匹配时打印 WARNING
3. 训练完成后保存 **merged model**（包含 ViT 更新），下游阶段以此为 base model
4. ReST SFT 阶段自动检测无 adapter 时添加新 LoRA

### 教训

> 4-bit + unfreeze 是假组合。要真正训练 ViT，必须 BF16。

---

## 十九、True GRPO 重写 — 2026-07-10

### 旧版问题

`stage3_grpo.py` 是单样本 REINFORCE + L2 log-prob 惩罚，不是真正的 GRPO。

### 新版设计 (`stage3_grpo.py`)

| 特性 | 旧 | 新 |
|------|-----|-----|
| 每 prompt 生成数 | 1 | G=4 (共享 KV-cache) |
| 优势估计 | reward 直接作为优势 | (r_i - mean) / std 组内归一化 |
| KL 正则 | (log_p)² L2 | exp(log_ratio) - log_ratio - 1 (无偏估计) |
| 参考模型 | 无 | Frozen copy of DPO output |

### 关键参数

- `--n_generations 4`: 每组 4 条，组内相对比较
- `--kl_beta 0.04`: DeepSeek 默认 KL 权重
- `--lr 1e-6`: 极小步长，防止策略跳变
- 参考模型与训练模型同构，冻结

---

## 二十、Stage 2: DPO + SimPO — 2026-07-10

### 新增文件

`training/stage2_preference.py` — 统一偏好优化脚本

### 两种方法

```bash
# DPO (推荐): 有参考模型约束，稳定
python training/stage2_preference.py --method dpo \
    --model_path /path/to/stage1_vision_v2/merged_model \
    --adapter /path/to/rest_round2/lora_adapter \
    --data_dir /path/to/dpo_data \
    --beta 0.5 --lr 5e-5 --epochs 1

# SimPO: 无参考模型，省显存
python training/stage2_preference.py --method simpo \
    --model_path ... --adapter ... --data_dir ... \
    --beta 0.5 --gamma 0.3
```

### DPO vs SimPO 对比

| | DPO | SimPO |
|------|-----|------|
| 参考模型 | 需要 (+6GB) | 不需要 |
| 长度偏差 | 有 | 无 (avg log-prob) |
| 稳定性 | 高 | 中 |
| 论文 | NeurIPS 2023 | ICML 2024 |

### DPO 参考模型

DPO 参考模型 = 训练前的 frozen 副本。防止模型为了增大 chosen 概率扭曲语言分布。公式：
```
Loss = -log σ(β × [log(π_θ(c)/π_ref(c)) - log(π_θ(r)/π_ref(r))])
```

---

## 二十一、最终管线 (v4) — 2026-07-10

### 完整流程

```
Stage 1 SFT (BF16, ViT 4层解冻, no-hint)
  Input:  Qwen2.5-VL-3B + sft_nohint (1472, 3 slices/sample)
  Output: stage1_vision_v2/merged_model/ + lora_adapter/
    ↓
DPO 数据构建
  Input:  sft_nohint/sft_train.jsonl
  Output: dpo_nohint/dpo_train.jsonl + dpo_val.jsonl
    ↓
ReST Round 1
  Input:  merged_model + sft_nohint
  Output: rest_round1/lora_adapter/
    ↓
ReST Round 2 (if reward_median improving)
  Input:  merged_model + rest_round1 adapter + augmented data
  Output: rest_round2/lora_adapter/
    ↓
Stage 2 DPO
  Input:  merged_model + rest_round2 adapter + dpo_nohint
  Output: stage2_dpo/lora_adapter/
    ↓
Stage 3 True GRPO (G=4)
  Input:  merged_model + stage2_dpo adapter + sft_nohint
  Output: stage3_grpo/lora_adapter/
    ↓
评估 (有图/无图对比 + clinical_accuracy)
```

### 各阶段职责

| 阶段 | 学什么 | 优化目标 |
|------|--------|---------|
| SFT | 看图 → 写报告 | Cross-entropy (per-token) |
| ReST | 生成更好报告 → SFT | composite_reward (top-50%) |
| DPO | chosen > rejected | -log σ(β × Δlog_ratio) |
| GRPO | 探索高 reward 生成 | Group-relative advantage + KL |

### Base model 传递

Stage 1 产出 **merged_model**（BF16, ViT 更新已合并）。
所有下游阶段都以 merged_model 为 base，在此基础上叠加新的 LoRA adapter。
ReST 首次运行无 adapter 时自动添加新 LoRA。

### 预估时间 (RTX 5090)

| 阶段 | 时间 |
|------|------|
| Stage 1 SFT (5 epoch, BF16) | ~4-5h |
| DPO 数据构建 | ~5min |
| ReST × 2 | ~5-6h |
| Stage 2 DPO | ~1-2h |
| Stage 3 GRPO | ~2-3h |
| **总计** | **~14-17h** |

---

**最后更新**: 2026-07-10  
**新文件**:
- `training/stage2_preference.py` — DPO + SimPO 偏好优化
- `training/stage3_grpo.py` — True GRPO (重写)

**待完成**: 
1. 推送到 AutoDL
2. Stage 1 BF16 SFT: `--no_4bit --unfreeze_vit_layers 4 --vit_lr_ratio 0.5`
3. 构建 DPO 数据 (`dpo_dataset_builder.py --sft_data sft_nohint`)
4. ReST Round 1 → Round 2 → DPO → GRPO
5. CT-RATE reward model 训练（未来工作）
