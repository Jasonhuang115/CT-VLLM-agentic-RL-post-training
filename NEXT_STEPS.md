# 肺结节CT Agentic RL 后训练 — 操作指南

## 当前状态

| 项目 | 状态 |
|------|:---:|
| 环境安装 | ✅ 已验证 |
| 模型 (Qwen2.5-VL-3B) | ✅ 已下载 |
| CT-RATE 报告文本 (47,149条) | ✅ 已下载 |
| LUNA16 标注 (1,186结节) | ✅ 已下载 |
| LUNA16 subset0 CT图像 (89个) | ✅ 已下载 |
| LUNA16 subset1-9 CT图像 | ❌ 待下载 |
| Stage 1 SFT (小规模) | ✅ 已验证 (194样本, loss 10.2→2.4) |

## 数据缺口

LUNA16 共 10 个子集 (subset0-9)，我们只有 subset0。
每个子集约 8GB，还需下载 **subset1-9，约 72GB**。

下载地址: https://zenodo.org/records/3723295

## 在新电脑上操作步骤

### 1. 开 AutoDL 实例

- GPU: RTX 5090 (¥2.78/h) 或 4090 (¥1.88/h)
- 数据盘: 250GB
- 镜像: CUDA 13.0 + PyTorch 2.12 + Python 3.12 + Ubuntu 22.04
- 计费: 按量付费

### 2. 克隆代码

```bash
# 把 TOKEN 换成 GitHub Personal Access Token
cd /root
git clone https://TOKEN@github.com/Jasonhuang115/CT-VLLM-agentic-RL-post-training.git
cd CT-VLLM-agentic-RL-post-training
```

### 3. 装环境

```bash
bash scripts/setup_env.sh

# 如果 Unsloth 安装失败(国内GitHub慢)，用清华镜像:
pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple

# 装 qwen-vl-utils
pip install qwen-vl-utils -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 下载模型权重

```bash
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct')
"

# 挪出子目录 (ModelScope 会在 cache_dir 下再建子目录)
find /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct -name "*.safetensors" -type f | head -1
ls /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/Qwen/Qwen2.5-VL-3B-Instruct/ 2>/dev/null && {
    mv /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/Qwen/Qwen2.5-VL-3B-Instruct/* /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/
    rm -rf /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/Qwen/
}
```

### 5. 设置 HuggingFace Token

```bash
export HF_TOKEN="你的HF token"
export HF_ENDPOINT=https://hf-mirror.com
```

### 6. 下载数据

**6a. LUNA16 标注 + subset0 (已经有的跳过)**
```bash
# 标注文件 (几KB)
python data/download/download_luna16.py --annotations_only --output /root/autodl-tmp/data/LUNA16

# subset0 (8GB) - 如果已经下过就跳过
python data/download/download_luna16.py --subset --n_cases 1 --output /root/autodl-tmp/data/LUNA16
```

**6b. CT-RATE 报告**
```bash
python data/download/download_ctrate.py --reports --lung_only --output /root/autodl-tmp/data/CT-RATE
```

**6c. LUNA16 所有子集 (最耗时)**

总计约 80GB，下载约 2-4 小时。

```bash
cd /root/autodl-tmp/data/LUNA16

# 下载并解压每个子集
for i in 1 2 3 4 5 6 7 8 9; do
    echo "=== subset$i ==="
    wget -c "https://zenodo.org/records/3723295/files/subset$i.zip" -O "subset$i.zip"
    unzip -o "subset$i.zip" -d images/
    rm "subset$i.zip"  # 解压后删除zip省空间
done

# 把所有 .mhd 和 .raw 收集到一个目录
mkdir -p /root/autodl-tmp/data/LUNA16/all
find /root/autodl-tmp/data/LUNA16 -name "*.mhd" -o -name "*.raw" | xargs -I{} cp {} /root/autodl-tmp/data/LUNA16/all/
cp /root/autodl-tmp/data/LUNA16/annotations.csv /root/autodl-tmp/data/LUNA16/all/
```

### 7. 构建训练数据

```bash
cd /root/CT-VLLM-agentic-RL-post-training
python data/sft_dataset_builder.py \
    --luna16_dir /root/autodl-tmp/data/LUNA16/all \
    --ctrate_reports /root/autodl-tmp/data/CT-RATE/reports.jsonl \
    --output /root/autodl-tmp/data/sft
```

### 8. 开始训练

全量数据下，各阶段预估时间：
- Stage 1 SFT: 4-5小时
- Stage 2 DPO: 1.5-2小时  
- Stage 3 GRPO: 10-14小时 (过夜跑)
- Stage 4 Agent RL: 8-10小时

```bash
# Stage 1: SFT
python training/stage1_sft.py --data_dir /root/autodl-tmp/data/sft --output /root/autodl-tmp/outputs/stage1_sft --epochs 3

# Stage 2: DPO (先构建偏好对)
python data/dpo_dataset_builder.py --sft_data /root/autodl-tmp/data/sft/sft_train.jsonl --output /root/autodl-tmp/data/dpo --n_pairs 2000
python training/stage2_dpo.py --adapter /root/autodl-tmp/outputs/stage1_sft/lora_adapter --data_dir /root/autodl-tmp/data/dpo --output /root/autodl-tmp/outputs/stage2_dpo

# Stage 3: GRPO (过夜)
python training/stage3_grpo.py --adapter /root/autodl-tmp/outputs/stage2_dpo/lora_adapter --data_dir /root/autodl-tmp/data/sft --output /root/autodl-tmp/outputs/stage3_grpo

# Stage 4: Agent RL
python training/stage4a_agent_sft.py --adapter /root/autodl-tmp/outputs/stage3_grpo/lora_adapter --trajectories /root/autodl-tmp/data/agent_trajectories/agent_trajectories.jsonl --output /root/autodl-tmp/outputs/stage4a_agent_sft
python training/stage4b_agent_grpo.py --adapter /root/autodl-tmp/outputs/stage4a_agent_sft/lora_adapter --output /root/autodl-tmp/outputs/stage4b_agent_grpo
```

### 9. 验证结果

```bash
# 测试推理
python inference/gradio_app.py --adapter /root/autodl-tmp/outputs/stage4b_agent_grpo/lora_adapter
```

## 费用预估 (全量)

| 项目 | 费用 |
|------|------|
| GPU (5090 × 40h) | ~¥111 |
| 数据盘 (250GB × 5天) | ~¥8 |
| **总计** | **~¥120** |

## 不训练时记得关机

关机后只收数据盘费，GPU 不计费。**不要点释放/删除**。

## 常见问题

1. **Unsloth 装不上**: 用清华镜像 `pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple`
2. **flash-attn 装不上**: 脚本里已改为 `sdpa`，不需要 flash-attn
3. **HF 数据集报 gated**: 确认 `export HF_TOKEN=你的token` 已设置
4. **内存不足**: 5090 32GB 足够全程；4090 24GB 需要降低 batch_size=1, grad_accum=4
