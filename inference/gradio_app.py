#!/usr/bin/env python3
"""
Gradio 演示界面

提供交互式 CT 肺结节诊断演示:
  - 上传 CT 多视图图像
  - 查看模型诊断过程 (含工具调用)
  - 下载结构化诊断报告

使用方式:
  python inference/gradio_app.py --adapter /root/autodl-tmp/outputs/stage4b_agent_grpo/lora_adapter
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoProcessor
from unsloth import FastVisionModel
from peft import PeftModel

from agent.tool_registry import ToolRegistry
from agent.react_engine import ReActEngine

os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# 免责声明
DISCLAIMER = """---
⚠️ 医疗免责声明

本报告由AI辅助诊断系统生成, 仅供临床参考, 不能替代专业医师的诊断意见。
所有诊断结论必须在执业医师审核确认后方可用于临床决策。
AI系统可能遗漏重要病变或产生错误判断, 请务必结合患者完整病史、
体格检查及其他辅助检查结果综合评估。

本系统不承担因使用此报告而产生的任何医疗责任。
如有疑问, 请咨询具有执业资格的放射科或呼吸科医师。"""


class AppState:
    def __init__(self, adapter_path: str, model_path: str):
        self.model = None
        self.tokenizer = None
        self.engine = None
        self.adapter_path = adapter_path
        self.model_path = model_path

    def load_model(self):
        """延迟加载模型"""
        if self.model is not None:
            return

        print("[INFO] 加载模型...")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.model_path,
            load_in_4bit=True,
        )

        if self.adapter_path and os.path.exists(self.adapter_path):
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        tools = ToolRegistry()
        self.engine = ReActEngine(
            self.model, self.tokenizer, tools,
            max_steps=5, verbose=False,
        )
        print("[INFO] 模型加载完成")

    def diagnose(self, *images):
        """运行诊断"""
        if self.engine is None:
            self.load_model()

        # 过滤掉空图像
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return "请至少上传一张CT图像", "", "", ""

        # 保存上传的图像到临时目录
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="gradio_ct_")
        image_paths = []
        for i, img in enumerate(valid_images):
            path = os.path.join(tmpdir, f"view_{i}.png")
            from PIL import Image
            if isinstance(img, str):
                img = Image.open(img)
            if hasattr(img, "save"):
                img.save(path)
            image_paths.append(path)

        # 运行 ReAct
        task = "请对CT图像中的肺结节进行完整诊断分析, 包括影像发现、恶性评估、指南对照和临床建议。"

        try:
            result = self.engine.run(image_paths, task)
        except Exception as e:
            return f"诊断出错: {str(e)}", "", "", ""

        # 格式化输出
        final_report = result.final_answer + DISCLAIMER

        # 工具调用过程
        tool_trace = ""
        for step in result.steps:
            tool_trace += f"### Step {step.step}\n"
            if step.thought:
                tool_trace += f"**思考**: {step.thought[:300]}\n\n"
            if step.action:
                tool_trace += f"**工具**: `{step.tool_name}`\n"
                tool_trace += f"**参数**: `{json.dumps(step.action_input, ensure_ascii=False)}`\n"
            if step.observation:
                tool_trace += f"**结果**: {str(step.observation)[:300]}\n\n"
            tool_trace += "---\n"

        # 摘要
        summary = (
            f"## 诊断摘要\n\n"
            f"- 工具调用次数: {result.total_tool_calls}\n"
            f"- 推理时间: {result.total_time:.1f}秒\n"
            f"- 状态: {'✅ 成功' if result.success else '⚠️ 异常'}\n"
        )

        return final_report, tool_trace, summary, json.dumps(
            {"steps": [{"step": s.step, "thought": s.thought, "action": s.action} for s in result.steps],
             "total_calls": result.total_tool_calls, "time": result.total_time},
            ensure_ascii=False, indent=2,
        )


def create_app(adapter_path: str, model_path: str):
    state = AppState(adapter_path, model_path)

    theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

    with gr.Blocks(theme=theme, title="肺结节CT AI诊断系统") as app:
        gr.Markdown("""
        # 🫁 肺结节CT AI辅助诊断系统

        上传CT结节多视图图像, AI将自动分析并通过工具调用(指南检索、文献搜索等)生成完整诊断报告。

        **推荐上传**: 轴位 + 冠状面 + 矢状面 + 九宫格视图
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 上传CT图像")
                img1 = gr.Image(label="轴位视图 (Axial)", type="filepath")
                img2 = gr.Image(label="冠状面 (Coronal)", type="filepath")
                img3 = gr.Image(label="矢状面 (Sagittal)", type="filepath")
                img4 = gr.Image(label="九宫格 (Montage)", type="filepath")
                img5 = gr.Image(label="MIP投影", type="filepath")

                run_btn = gr.Button("🔍 开始诊断", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📊 诊断结果")
                summary = gr.Markdown("等待诊断...")

                with gr.Tabs():
                    with gr.TabItem("诊断报告"):
                        report = gr.Markdown("请上传图像并点击'开始诊断'", label="诊断报告")
                    with gr.TabItem("推理过程"):
                        trace = gr.Markdown("", label="推理过程")
                    with gr.TabItem("原始JSON"):
                        json_out = gr.Code(language="json", label="结构化输出")

        run_btn.click(
            fn=state.diagnose,
            inputs=[img1, img2, img3, img4, img5],
            outputs=[report, trace, summary, json_out],
        )

        # 示例
        gr.Markdown("""
        ---
        ### 使用说明

        1. **上传图像**: 最少1张轴位CT图像, 推荐上传全部5张多视图
        2. **点击诊断**: AI将自动分析结节特征, 查询相关指南, 生成完整报告
        3. **查看过程**: 切换标签页查看推理过程和工具调用详情

        **当前模型**: Qwen2.5-VL-3B + GRPO + Agentic RL
        """)

    return app


def main():
    parser = argparse.ArgumentParser(description="Gradio 演示")
    parser.add_argument("--adapter", type=str, default="",
                        help="最终 LoRA adapter 路径")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="创建公网分享链接")
    args = parser.parse_args()

    app = create_app(args.adapter, args.model_path)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
