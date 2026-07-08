# AutoDL 部署指南

## 1. 租用实例

### 推荐配置
| 项 | 选择 |
|----|------|
| GPU | RTX 5090 32GB (¥2.78/h) |
| CPU | 25核 Xeon Platinum 8470Q |
| 内存 | 92GB |
| 系统盘 | 30GB (默认) |
| 数据盘 | **250GB** (扩容到) |
| 镜像 | CUDA 12.4 + Python 3.11 预置镜像 |

### 操作步骤
1. 登录 [AutoDL](https://www.autodl.com)
2. 控制台 → 实例容器 → 租用新实例
3. 筛选 GPU: RTX 5090
4. **重要**: 数据盘扩容到 250GB（实例详情页 → 数据盘 → 扩容）
5. 开机后获取 JupyterLab / SSH 连接信息

## 2. 连接实例

### 方式A: JupyterLab (推荐)
浏览器打开 AutoDL 提供的 JupyterLab 地址 → Terminal

### 方式B: SSH
```bash
# AutoDL 控制台获取 SSH 命令
ssh -p <端口> root@<IP>
```

## 3. 部署项目

```bash
# 克隆项目（或 scp 上传）
cd /root
# git clone <your-repo>   # 如果用 git
# 或从本地上传:
# scp -P <端口> -r PostTraining/ root@<IP>:/root/

cd /root/PostTraining

# 安装环境
bash scripts/setup_env.sh

# 确认 GPU 可用
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 应输出: NVIDIA GeForce RTX 5090
```

## 4. 下载模型权重（使用国内镜像）

```bash
# 设置国内镜像（setup_env.sh 已自动设置）
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Qwen2.5-VL-3B-Instruct (~7GB)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
    --resume-download

# 验证
ls /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct/
```

## 5. 下载数据

```bash
# 所有数据统一放到数据盘
mkdir -p /root/autodl-tmp/data

# Stage 0 最小验证（先下 2-3 个病例测试管线）
python data/download/download_lidc.py --subset --n_patients 3

# 确认管线 OK 后，下载全量数据
python data/download/download_lidc.py --full
python data/download/download_radgenome.py
```

## 6. 运行训练流程

```bash
# Stage 0: 验证管线
python training/stage0_check.py

# 数据预处理
python data/sft_dataset_builder.py

# Stage 1-4: 按顺序训练
python training/stage1_sft.py      # ~4-5h
python training/stage2_dpo.py      # ~1.5-2h
python training/stage3_grpo.py     # ~10-14h (建议 overnight)
python training/stage4a_agent_sft.py  # ~1.5-2h
python training/stage4b_agent_grpo.py # ~6-10h (建议 overnight)
```

## 7. 注意事项

- **不训练时关机**: 计费继续，关机后只收数据盘费用
- **数据盘持久化**: 关机后数据不丢，下次开机直接继续
- **监控 VRAM**: 如果 OOM，检查 `nvidia-smi` 有无残留进程
- **下载加速**: HuggingFace 已设 hf-mirror.com 镜像
- **模型路径**: 所有脚本默认从 `/root/autodl-tmp/models/` 读取
