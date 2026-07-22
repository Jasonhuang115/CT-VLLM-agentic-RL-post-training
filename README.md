# 肺结节CT多模态后训练项目

> 完整后训练管线: SFT → DPO/SimPO → GRPO 
> 基座模型: Qwen2.5-VL-3B-Instruct | 训练框架: Unsloth + TRL
> GPU: RTX 5090 32GB | 费用: ~¥150-200 | 时间: 3-5天

## 快速开始

### 1. 环境

```bash
# AutoDL 实例上:
git clone <this-repo> && cd PostTraining
bash scripts/setup_env.sh
```

### 2. 下载模型权重

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

### 3. 数据管线

```bash
# Stage 0: 验证
python training/stage0_check.py

# 完整数据管线 (下载 → 预处理 → SFT/DPO/Agent数据)
bash scripts/run_data_pipeline.sh
```

### 4. 全流程训练

```bash
bash scripts/run_full_training.sh
```

### 5. 演示

```bash
python inference/gradio_app.py --adapter /root/autodl-tmp/outputs/stage4b_agent_grpo/lora_adapter
```

## 项目结构

```
PostTraining/
├── config/                      # 配置文件
├── data/
│   ├── download/                # 数据下载
│   ├── preprocessing/           # DICOM→NIfTI, 结节提取
│   ├── sft_dataset_builder.py   # SFT数据构建
│   ├── dpo_dataset_builder.py   # DPO偏好对
│   └── agent_trajectory_builder.py  # Agent轨迹
├── training/
│   ├── stage0_check.py          # 管线验证
│   ├── stage1_sft.py            # SFT (~4-5h)
│   ├── stage2_dpo.py            # SimPO (~1.5-2h)
│   ├── stage3_grpo.py           # GRPO (~10-14h)
│   └── reward_functions.py      # 复合奖励函数
├── agent/
│   ├── react_engine.py          # ReAct推理循环
│   ├── tool_registry.py         # 工具注册
│   └── tools/                   # 4个工具实现
├── inference/
│   ├── gradio_app.py            # 演示界面
│   └── report_generator.py      # 报告生成
├── evaluation/
│   └── clinical_accuracy.py     # 临床准确性评估
└── scripts/
    ├── setup_env.sh             # 环境安装
    ├── DEPLOY.md                # AutoDL部署指南
    ├── run_data_pipeline.sh     # 一键数据管线
    └── run_full_training.sh     # 一键全流程训练
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 基座模型 | Qwen2.5-VL-3B-Instruct (4-bit QLoRA) |
| SFT/DPO | Unsloth + TRL SFTTrainer / DPOTrainer |
| GRPO | TRL GRPOTrainer + vLLM colocate |
| Agent | LangGraph + ReAct + 自定义工具 |
| 评估 | 关键字段规则评分 + 人工抽检 |

## 关键参考

- [Med-R1](https://github.com/Yuxiang-Lai117/Med-R1) - GRPO 医学VLM
- [EditGRPO](https://github.com/taokz/EditGRPO) - GRPO 后编辑防坍缩
- [MedFact-R1](https://github.com/Garfieldgengliang/MEDFACT-R1) - 多信号事实性奖励
- [MedRAX](https://github.com/bowang-lab/MedRAX) - Agentic 胸片解释
- [Meissa](https://github.com/Schuture/Meissa) - Agent轨迹蒸馏

## License

MIT
