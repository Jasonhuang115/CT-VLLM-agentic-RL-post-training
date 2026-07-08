#!/usr/bin/env python3
"""VRAM 监控工具"""

import time
import torch
from threading import Thread


class VRAMMonitor(Thread):
    def __init__(self, interval: float = 5.0, device: int = 0):
        super().__init__(daemon=True)
        self.interval = interval
        self.device = device
        self.running = True
        self.peak = 0.0
        self.history = []

    def run(self):
        while self.running:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            self.peak = max(self.peak, allocated)
            self.history.append((time.time(), allocated, reserved))
            print(f"  [VRAM] {allocated:.1f}G / {total:.1f}G (峰值: {self.peak:.1f}G, 缓存: {reserved:.1f}G)")
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        print(f"  [VRAM] 最终峰值: {self.peak:.1f} GB")


if __name__ == "__main__":
    monitor = VRAMMonitor(interval=2.0)
    monitor.start()
    time.sleep(30)
    monitor.stop()
