#!/usr/bin/env python3
"""
工具注册表: 管理和调度所有 Agent 工具
"""

import json
from typing import Dict, List, Optional, Any

from agent.tools.guideline_retrieval import GuidelineRetrieval
from agent.tools.web_search import WebSearch
from agent.tools.measurement_calc import MeasurementCalculator
from agent.tools.image_reanalyzer import analyze_roi, format_for_llm as format_image_result
from agent.tools import __all__ as tool_modules


class ToolRegistry:
    """工具注册和调度"""

    def __init__(self):
        self._tools = {}
        self._instances = {}

        # 注册内置工具
        self.register("guideline_retrieval", GuidelineRetrieval())
        self.register("web_search", WebSearch())
        self.register("measurement_calculator", MeasurementCalculator())
        self.register("image_reanalyzer", None)  # 无状态工具, 直接调用函数

    def register(self, name: str, instance: Any):
        """注册一个工具"""
        self._tools[name] = {
            "name": name,
            "instance": instance,
        }
        self._instances[name] = instance

    def get(self, name: str) -> Any:
        """获取工具实例"""
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._instances.get(name)

    def list_tools(self) -> List[str]:
        """列出所有注册的工具"""
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs) -> Dict:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        if tool_name == "guideline_retrieval":
            instance = self.get("guideline_retrieval")
            return instance.search(
                query=kwargs.get("query", ""),
                guideline_type=kwargs.get("guideline_type"),
            )

        elif tool_name == "web_search":
            instance = self.get("web_search")
            return instance.search(
                query=kwargs.get("query", ""),
                max_results=kwargs.get("max_results", 5),
            )

        elif tool_name == "measurement_calculator":
            instance = self.get("measurement_calculator")
            func = kwargs.pop("function", "lung_rads_classify")
            if func == "lung_rads_classify":
                return instance.lung_rads_classify(**kwargs)
            elif func == "volume_doubling_time":
                return instance.volume_doubling_time(**kwargs)
            elif func == "malignancy_probability":
                return instance.malignancy_probability(**kwargs)
            else:
                return {"error": f"Unknown function: {func}"}

        elif tool_name == "image_reanalyzer":
            # 需要 ROI 图像, 这里简化: 接受 HU 值列表
            roi_hu = kwargs.get("roi_hu_data")
            if roi_hu is not None:
                import numpy as np
                roi_array = np.array(roi_hu)
                return analyze_roi(roi_array, analysis_type=kwargs.get("analysis_type", "all"))
            else:
                return {"error": "image_reanalyzer 需要 roi_hu_data (numpy array)"}

        else:
            return {"error": f"未知工具: {tool_name}"}

    def format_observation(self, tool_name: str, result: Dict) -> str:
        """将工具返回结果格式化为 LLM 可读文本"""
        if tool_name == "guideline_retrieval":
            instance = self.get("guideline_retrieval")
            return instance.format_for_llm(result)
        elif tool_name == "web_search":
            instance = self.get("web_search")
            return instance.format_for_llm(result)
        elif tool_name == "measurement_calculator":
            instance = self.get("measurement_calculator")
            return instance.format_for_llm(result)
        elif tool_name == "image_reanalyzer":
            return format_image_result(result)
        else:
            return json.dumps(result, ensure_ascii=False, indent=2)

    def get_descriptions(self) -> str:
        """获取所有工具的描述 (用于 prompt)"""
        descs = []
        descs.append("- **guideline_retrieval**: 检索 Fleischner, Lung-RADS, NCCN, 中国共识等指南内容。参数: query, guideline_type (可选)")
        descs.append("- **web_search**: 搜索最新医学文献和临床研究。参数: query, max_results=5")
        descs.append("- **measurement_calculator**: 计算 Lung-RADS 分级、VDT、恶性概率。参数: function (lung_rads_classify/volume_doubling_time/malignancy_probability), ...")
        descs.append("- **image_reanalyzer**: 对CT结节ROI进行定量图像分析 (HU统计/纹理/边界)。参数: roi_hu_data, analysis_type (density/texture/boundary/all)")
        return "\n".join(descs)


if __name__ == "__main__":
    registry = ToolRegistry()
    print("已注册工具:", registry.list_tools())

    # 测试
    result = registry.execute("guideline_retrieval", query="部分实性结节 随访")
    print("\n指南检索结果:")
    print(registry.format_observation("guideline_retrieval", result)[:500])
