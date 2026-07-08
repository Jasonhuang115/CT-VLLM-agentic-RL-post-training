#!/usr/bin/env python3
"""Agent 轨迹记录器 (用于 RL 训练)"""

import json
import os
from typing import List, Dict
from datetime import datetime


class TrajectoryLogger:
    def __init__(self, output_dir: str = "/root/autodl-tmp/outputs/trajectories"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.trajectories = []

    def log_trajectory(self, steps: List, final_answer: str):
        """记录一条完整的 Agent 执行轨迹"""
        trajectory = {
            "timestamp": datetime.now().isoformat(),
            "steps": [
                {"step": s.step, "thought": s.thought,
                 "action": s.action, "tool_name": s.tool_name,
                 "action_input": s.action_input, "observation": s.observation}
                for s in steps
            ],
            "final_answer": final_answer,
            "n_steps": len(steps),
        }
        self.trajectories.append(trajectory)

    def save(self, filename: str = None):
        """保存所有轨迹到文件"""
        if filename is None:
            filename = f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            for t in self.trajectories:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        return path
