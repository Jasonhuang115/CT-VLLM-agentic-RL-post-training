#!/bin/bash
# ============================================================
# 一键执行全量数据管线
# 数据集: LUNA16 + CT-RATE + TotalSegmentator
# ============================================================
set -e

echo "============================================"
echo " 数据管线: LUNA16 + CT-RATE + TotalSegmentator"
echo "============================================"

DATA_ROOT="/root/autodl-tmp/data"

# Step 1: 下载 LUNA16 标注
echo "[1/4] 下载 LUNA16 标注..."
python data/download/download_luna16.py --annotations_only --output "$DATA_ROOT/LUNA16"

# 手动提示: 下载 LUNA16 图像子集
echo ""
echo "⚠️  LUNA16 图像文件 (~80GB) 需要在 MacBook+VPN 下载后上传, 或直接从 Zenodo 在 AutoDL 下载"
echo "   下载命令: python data/download/download_luna16.py --full --output $DATA_ROOT/LUNA16"
echo "   或只下1个子集测试: python data/download/download_luna16.py --subset --n_cases 1 --output $DATA_ROOT/LUNA16"
echo ""

# Step 2: 下载 CT-RATE 报告
echo "[2/4] 下载 CT-RATE 报告文本..."
python data/download/download_ctrate.py --reports --lung_only --output "$DATA_ROOT/CT-RATE"

# Step 3: 下载 TotalSegmentator
echo "[3/4] 下载 TotalSegmentator..."
python data/download/download_totalsegmentator.py --subset --output "$DATA_ROOT/TotalSegmentator"

# Step 4: 构建 SFT 数据
echo "[4/4] 构建 SFT 数据..."
python data/sft_dataset_builder.py \
    --luna16_dir "$DATA_ROOT/LUNA16" \
    --ctrate_reports "$DATA_ROOT/CT-RATE/reports.jsonl" \
    --output "$DATA_ROOT/sft" \
    --lang both

echo ""
echo "============================================"
echo " 数据管线完成!"
echo " 下一步: bash scripts/run_full_training.sh"
echo "============================================"
