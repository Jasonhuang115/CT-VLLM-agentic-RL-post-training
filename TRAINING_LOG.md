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

**最后更新**: 2026-07-06  
**待完成**: 
1. 本地下载 LUNA16 subset 1-6, 8, 9
2. AutoDL 上修复 stage2_simpo.py DataLoader 并跑通小批量验证
3. 对比 v1 vs v2 的 Val acc
4. 如有 DeepSeek API key，批量生成高质量报告
