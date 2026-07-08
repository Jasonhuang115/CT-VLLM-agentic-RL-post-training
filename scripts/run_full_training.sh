#!/bin/bash
# ============================================================
# 一键执行全流程训练: SFT → DPO → GRPO → Agent SFT → Agent GRPO
# ============================================================
set -e

OUTPUT_ROOT="/root/autodl-tmp/outputs"
DATA_ROOT="/root/autodl-tmp/data"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"

echo "============================================"
echo " 全流程训练"
echo "============================================"

# Stage 1: SFT
echo ""
echo "[1/5] Stage 1: SFT (~4-5h)..."
python training/stage1_sft.py \
    --data_dir "$DATA_ROOT/sft" \
    --output "$OUTPUT_ROOT/stage1_sft"
echo "Stage 1 完成: $OUTPUT_ROOT/stage1_sft"

# Stage 2: DPO
echo ""
echo "[2/5] Stage 2: SimPO (~1.5-2h)..."
python training/stage2_dpo.py \
    --adapter "$OUTPUT_ROOT/stage1_sft/lora_adapter" \
    --data_dir "$DATA_ROOT/dpo" \
    --output "$OUTPUT_ROOT/stage2_dpo" \
    --method simpo
echo "Stage 2 完成: $OUTPUT_ROOT/stage2_dpo"

# Stage 3: GRPO (过夜)
echo ""
echo "[3/5] Stage 3: GRPO (~10-14h, 建议 overnight)..."
python training/stage3_grpo.py \
    --adapter "$OUTPUT_ROOT/stage2_dpo/lora_adapter" \
    --data_dir "$DATA_ROOT/sft" \
    --output "$OUTPUT_ROOT/stage3_grpo"
echo "Stage 3 完成: $OUTPUT_ROOT/stage3_grpo"

# Stage 4a: Agent SFT
echo ""
echo "[4/5] Stage 4a: Agent SFT (~1.5-2h)..."
python training/stage4a_agent_sft.py \
    --adapter "$OUTPUT_ROOT/stage3_grpo/lora_adapter" \
    --trajectories "$DATA_ROOT/agent_trajectories/agent_trajectories.jsonl" \
    --output "$OUTPUT_ROOT/stage4a_agent_sft"
echo "Stage 4a 完成: $OUTPUT_ROOT/stage4a_agent_sft"

# Stage 4b: Agent GRPO (过夜)
echo ""
echo "[5/5] Stage 4b: Agent GRPO (~6-10h, 建议 overnight)..."
python training/stage4b_agent_grpo.py \
    --adapter "$OUTPUT_ROOT/stage4a_agent_sft/lora_adapter" \
    --output "$OUTPUT_ROOT/stage4b_agent_grpo"
echo "Stage 4b 完成: $OUTPUT_ROOT/stage4b_agent_grpo"

echo ""
echo "============================================"
echo " 🎉 全流程训练完成!"
echo "============================================"
echo ""
echo " 最终适配器: $OUTPUT_ROOT/stage4b_agent_grpo/lora_adapter"
echo ""
echo " 演示:"
echo "   python inference/gradio_app.py --adapter $OUTPUT_ROOT/stage4b_agent_grpo/lora_adapter"
