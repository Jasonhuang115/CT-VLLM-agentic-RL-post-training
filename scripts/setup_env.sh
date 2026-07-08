#!/bin/bash
# ============================================================
# PostTraining 环境安装脚本
# 适用于 AutoDL RTX 5090 / 4090 实例 (CUDA 12.4+, Python 3.11+)
# ============================================================
set -e

echo "============================================"
echo " PostTraining 环境安装"
echo "============================================"

# --- 1. 确认基础环境 ---
echo "[1/5] 检查基础环境..."
python --version
nvcc --version 2>/dev/null || echo "  (nvcc not found, using pre-installed CUDA)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not found"

# --- 2. 配置 HuggingFace 镜像（国内加速） ---
echo "[2/5] 配置 HuggingFace 镜像..."
export HF_ENDPOINT=https://hf-mirror.com
if ! grep -q "HF_ENDPOINT" ~/.bashrc 2>/dev/null; then
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
    echo "  Added HF_ENDPOINT to ~/.bashrc"
fi

# --- 3. 安装 PyTorch (如果预装版本不匹配) ---
echo "[3/5] 检查 PyTorch..."
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    echo "  Installing PyTorch 2.5.1 with CUDA 12.4..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
fi

# --- 4. 安装项目依赖 ---
echo "[4/5] 安装项目依赖..."
pip install --upgrade pip

# Core ML (先装，版本锁定)
pip install transformers==4.51.0 accelerate==1.3.0 peft==0.14.0
pip install bitsandbytes==0.45.0

# Training
pip install trl>=0.18.0 datasets>=3.2.0
pip install deepspeed>=0.16.0

# Unsloth (从官方安装)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# vLLM (AutoDL 可能已预装)
pip install vllm>=0.7.0 2>/dev/null || echo "  vLLM install skipped (check CUDA compat)"

# Medical imaging
pip install pydicom>=3.0.0 nibabel>=5.3.0
pip install opencv-python-headless>=4.10.0
pip install scikit-image>=0.24.0
pip install SimpleITK>=2.4.0

# Utilities
pip install Pillow matplotlib seaborn pandas numpy tqdm
pip install pyarrow huggingface_hub
pip install pyyaml omegaconf wandb rich fire

# Agent
pip install langchain langgraph duckduckgo-search

# Eval & Demo
pip install scikit-learn nltk rouge-score gradio

# --- 5. 验证关键组件 ---
echo "[5/5] 验证安装..."
echo ""
echo "  PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:         $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPU:          $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")')"
echo "  Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "  TRL:          $(python -c 'import trl; print(trl.__version__)')"
echo "  Unsloth:      $(python -c 'import unsloth; print(unsloth.__version__)' 2>/dev/null || echo 'checking...')"
echo "  Pydicom:      $(python -c 'import pydicom; print(pydicom.__version__)')"
echo "  Nibabel:      $(python -c 'import nibabel; print(nibabel.__version__)')"
echo ""
echo "============================================"
echo " 环境安装完成!"
echo " 下一步: 下载模型权重"
echo "   huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir models/Qwen2.5-VL-3B-Instruct"
echo "============================================"
