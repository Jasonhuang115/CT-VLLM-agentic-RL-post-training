#!/usr/bin/env python3
"""
ReAct 推理引擎

实现 Thought → Action → Observation 循环,
支持多轮工具调用和轨迹记录。

使用方式:
  from agent.react_engine import ReActEngine

  engine = ReActEngine(model, tokenizer, tools)
  result = engine.run(image_paths, "请诊断这个肺结节")
"""

import re
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from agent.tool_registry import ToolRegistry
from agent.trajectory_logger import TrajectoryLogger


@dataclass
class ReActStep:
    """单步 ReAct 记录"""
    step: int
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class ReActResult:
    """ReAct 运行结果"""
    final_answer: str
    steps: List[ReActStep] = field(default_factory=list)
    total_tool_calls: int = 0
    total_time: float = 0.0
    success: bool = True
    error: Optional[str] = None


class ReActEngine:
    """ReAct 推理引擎"""

    # ReAct 解析正则
    THOUGHT_PATTERN = re.compile(r"(?:Thought|思考|分析)[：:]\s*(.+?)(?=\n(?:Action|行动|操作|Observation|观察)|$)", re.DOTALL | re.IGNORECASE)
    ACTION_PATTERN = re.compile(r"(?:Action|行动|操作)[：:]\s*(\w+)\s*[\n\r]+.*?(?:Action Input|输入|参数)[：:]\s*(\{.*?\}|.+?)(?=\n(?:Observation|观察|Thought|思考)|$)", re.DOTALL | re.IGNORECASE)
    FINAL_ANSWER_PATTERN = re.compile(r"(?:Final Answer|最终回答|诊断报告|报告)[：:]?\s*(.+)", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        model,
        tokenizer,
        tools: ToolRegistry,
        max_steps: int = 10,
        verbose: bool = True,
        logger: Optional[TrajectoryLogger] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.logger = logger or TrajectoryLogger()

    def _build_prompt(self, image_paths: List[str], task: str) -> List[Dict]:
        """构建初始 prompt (包含图像 + ReAct 指令)"""
        tool_descriptions = self.tools.get_descriptions()

        system_prompt = f"""你是一名胸部放射科AI诊断助手。你可以使用以下工具来辅助诊断:

{tool_descriptions}

请使用以下格式进行推理:

Thought: <你的分析思路, 需要什么信息, 打算用什么工具>
Action: <工具名称>
Action Input: <JSON格式的工具参数>

Observation: <工具返回的结果>
(可以多轮Thought→Action→Observation)

当你有了足够的信息后, 给出最终回答:

Final Answer: <完整的诊断报告>

重要提示:
- 先仔细分析CT图像中的结节特征
- 不确定时, 查询指南 (guideline_retrieval) 或搜索最新文献 (web_search)
- 使用 measurement_calculator 计算 Lung-RADS 分级
- 如需更详细的图像分析, 使用 image_reanalyzer
- 最终报告必须包含: 影像发现, 恶性评估, 指南对照, 临床建议, 免责声明"""

        content = []
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
        content.append({"type": "text", "text": f"{system_prompt}\n\n任务: {task}"})

        return [{"role": "user", "content": content}]

    def _parse_react_output(self, text: str) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str]]:
        """
        解析模型输出，提取 Thought, Action, Action Input, Observation

        Returns: (thought, tool_name, tool_input, final_answer)
        """
        thought = None
        tool_name = None
        tool_input = None
        final_answer = None

        # 检查是否为最终回答
        final_match = self.FINAL_ANSWER_PATTERN.search(text)
        if final_match:
            final_answer = final_match.group(1).strip()

        # 提取思考
        thought_match = self.THOUGHT_PATTERN.search(text)
        if thought_match:
            thought = thought_match.group(1).strip()

        # 提取行动
        action_match = self.ACTION_PATTERN.search(text)
        if action_match:
            tool_name = action_match.group(1).strip()
            raw_input = action_match.group(2).strip()
            try:
                tool_input = json.loads(raw_input)
            except json.JSONDecodeError:
                # 非 JSON 输入 → 尝试解析为简单参数
                tool_input = {"query": raw_input}

        return thought, tool_name, tool_input, final_answer

    def _run_model(self, messages: List[Dict]) -> str:
        """运行模型推理"""
        from qwen_vl_utils import process_vision_info

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.tokenizer(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.model.device)

        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output

    def run(self, image_paths: List[str], task: str) -> ReActResult:
        """
        执行完整的 ReAct 推理循环

        Args:
            image_paths: CT 多视图图像路径列表
            task: 诊断任务描述

        Returns:
            ReActResult
        """
        steps = []
        total_calls = 0
        start_time = time.time()
        messages = self._build_prompt(image_paths, task)

        for step_idx in range(self.max_steps):
            if self.verbose:
                print(f"\n{'─'*40}")
                print(f"  Step {step_idx + 1}/{self.max_steps}")
                print(f"{'─'*40}")

            # 运行模型
            output = self._run_model(messages)

            # 解析输出
            thought, tool_name, tool_input, final_answer = self._parse_react_output(output)
            react_step = ReActStep(step=step_idx + 1, thought=thought or "")

            if self.verbose:
                if thought:
                    print(f"  💭 Thought: {thought[:200]}...")
                if tool_name:
                    print(f"  🔧 Action: {tool_name}({json.dumps(tool_input, ensure_ascii=False) if tool_input else 'None'})")

            # 如果是最终回答, 结束循环
            if final_answer:
                react_step.thought = thought or "给出最终诊断"
                steps.append(react_step)
                if self.verbose:
                    print(f"\n  ✅ Final Answer (长度: {len(final_answer)} 字符)")
                break

            # 执行工具调用
            if tool_name and tool_input:
                total_calls += 1
                react_step.action = tool_name
                react_step.action_input = tool_input
                react_step.tool_name = tool_name

                observation = self.tools.execute(tool_name, **tool_input)
                react_step.observation = observation

                if self.verbose:
                    print(f"  📊 Observation: {str(observation)[:200]}...")

                # 将观察追加到对话
                observation_text = self.tools.format_observation(tool_name, observation)
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Observation: {observation_text}\n\n请继续分析或给出最终诊断。"})
            else:
                # 没有解析到行动 → 可能是最终回答或被截断
                if self.verbose:
                    print(f"  ⚠️ 未检测到有效 Action, 结束循环")
                react_step.thought = thought or "(输出被截断或格式异常)"
                steps.append(react_step)
                final_answer = output  # 把当前输出作为最终回答
                break

            steps.append(react_step)

        # 如果没有提取到 final_answer, 用最后一步输出
        if not final_answer:
            final_answer = output if steps else "无法生成诊断报告"

        # 记录轨迹
        self.logger.log_trajectory(steps, final_answer)

        elapsed = time.time() - start_time
        result = ReActResult(
            final_answer=final_answer,
            steps=steps,
            total_tool_calls=total_calls,
            total_time=elapsed,
            success=(total_calls > 0 or len(final_answer) > 100),
        )

        if self.verbose:
            print(f"\n{'='*40}")
            print(f"  ReAct 完成: {total_calls} 次工具调用, {elapsed:.1f}s")
            print(f"{'='*40}")

        return result


if __name__ == "__main__":
    print("ReActEngine 模块已加载。")
    print("在实际使用时, 需要传入已加载的 Qwen2.5-VL 模型和工具注册表。")
    print()
    print("使用示例:")
    print("  from agent.react_engine import ReActEngine")
    print("  from agent.tool_registry import ToolRegistry")
    print("  engine = ReActEngine(model, tokenizer, registry)")
    print("  result = engine.run(['ct_slice.png'], '分析这个肺结节')")
    print("  print(result.final_answer)")
