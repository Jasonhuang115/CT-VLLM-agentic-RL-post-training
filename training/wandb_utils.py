#!/usr/bin/env python3
"""
共享 WandB 日志工具

所有训练脚本统一调用，支持:
  - 自动 init (用 stage 名做 run name)
  - step-wise logging
  - 训练结束后自动 finish
  - 环境变量 WANDB_MODE=offline 时离线模式

使用方式:
  from training.wandb_utils import WandBLogger
  logger = WandBLogger(stage="sft", output_dir="/root/autodl-tmp/outputs/stage1", config=args)
  logger.log({"train/loss": 3.2, "train/lr": 2e-4}, step=10)
  logger.finish()
"""

import os
import json
from datetime import datetime


class WandBLogger:
    def __init__(self, stage: str, output_dir: str, config: dict = None,
                 project: str = "ct-vllm", enabled: bool = True):
        self.stage = stage
        self.output_dir = output_dir
        self.enabled = enabled
        self._wandb = None
        self._step = 0

        if not enabled:
            return

        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            print("[WandB] wandb 未安装，跳过日志记录。pip install wandb")
            self.enabled = False
            return

        # 检查是否被禁用
        if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
            print("[WandB] WANDB_DISABLED=1，跳过")
            self.enabled = False
            return

        run_name = f"{stage}_{datetime.now().strftime('%m%d_%H%M')}"
        cfg_dict = vars(config) if hasattr(config, '__dict__') else (config or {})

        try:
            self._wandb.init(
                project=project,
                name=run_name,
                dir=output_dir,
                config=cfg_dict,
                reinit=True,
            )
            print(f"[WandB] run: {run_name}")
        except Exception as e:
            print(f"[WandB] init 失败: {e}")
            self.enabled = False

    def log(self, metrics: dict, step: int = None):
        """记录指标。step=None 则自动递增。"""
        if not self.enabled:
            return
        if step is None:
            step = self._step
            self._step += 1
        try:
            self._wandb.log(metrics, step=step)
        except Exception:
            pass

    def log_scalar(self, key: str, value: float, step: int = None):
        self.log({key: value}, step=step)

    def finish(self):
        if self.enabled and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


class DummyLogger:
    """无 wandb 时的空 logger，接口兼容"""
    def __init__(self, *args, **kwargs):
        pass
    def log(self, *args, **kwargs):
        pass
    def log_scalar(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def get_logger(stage: str, output_dir: str, config=None, enabled: bool = True) -> WandBLogger:
    """工厂函数：自动降级到 DummyLogger"""
    logger = WandBLogger(stage=stage, output_dir=output_dir, config=config, enabled=enabled)
    if not logger.enabled:
        return DummyLogger()
    return logger
