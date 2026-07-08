#!/usr/bin/env python3
"""
指南检索工具 (基于本地 RAG)

从本地临床指南知识库检索相关内容。
支持: Fleischner, Lung-RADS, NCCN, 中国共识, ACCP

使用方式:
  from agent.tools.guideline_retrieval import GuidelineRetrieval

  retriever = GuidelineRetrieval(knowledge_base_dir="agent/tools/guidelines")
  result = retriever.search("部分实性结节 随访建议", guideline_type="fleischner")
"""

import os
import re
import json
from typing import List, Dict, Optional
from pathlib import Path

# ============================================================
# 指南知识库 (内置)
# ============================================================

# 如果本地文件存在就读取, 否则使用内置数据
GUIDELINE_DATA = {
    "fleischner_2017": {
        "name": "Fleischner Society Guidelines 2017",
        "description": "肺结节管理指南 (国际金标准)",
        "sections": [
            {
                "title": "实性结节 (Solid Nodules)",
                "content": """
实性结节管理:
- <6mm: 低风险者无需随访; 高风险者可考虑12个月随访
- 6-8mm: 低风险者6-12个月CT随访, 18-24个月再次; 高风险者6-12个月
- >8mm: 建议3个月CT随访, PET/CT或组织活检

高风险因素: 年龄>50, 重度吸烟史(>30包年), 一级亲属肺癌史, 石棉暴露, COPD/肺纤维化
"""
            },
            {
                "title": "部分实性结节 (Part-Solid Nodules)",
                "content": """
部分实性结节管理:
- <6mm: 无需随访
- ≥6mm且实性成分<6mm: 建议年度CT随访
- ≥6mm且实性成分≥6mm: 建议3-6个月CT随访

如果持续存在且实性成分≥6mm, 高度可疑恶性。
实性成分的大小比总大小更重要。
"""
            },
            {
                "title": "磨玻璃结节 (Ground-Glass Nodules)",
                "content": """
纯磨玻璃结节管理:
- <6mm (约<100mm³): 无需常规随访, 但>30岁建议年度筛查
- ≥6mm (约>100mm³): 建议6-12个月CT随访, 然后每2年一次至5年

纯GGO生长缓慢, 惰性生物学行为, 但约10-20%会进展为部分实性或实性。
"""
            },
        ],
    },
    "lung_rads_v2022": {
        "name": "Lung-RADS v2022 (ACR)",
        "description": "肺结节报告与数据系统",
        "sections": [
            {
                "title": "Lung-RADS 分级",
                "content": """
Lung-RADS v2022 分级系统:

Category 0: 不完整 (需补充检查)
  - 既往CT不可用

Category 1 (阴性):
  - 无结节, 或
  - 钙化结节 (良性), 或
  - 裂隙旁结节 <10mm, 实性

Category 2 (良性表现):
  - 实性结节 <6mm (基线) / <4mm (年度), 或
  - 部分实性结节 <6mm (基线), 或
  - GGN <30mm, 或
  - 气道结节, 或
  - 叶间裂结节 实性≥10mm (良性特征)

Category 3 (可能良性):
  - 实性结节 ≥6 to <8mm (基线) / ≥4 to <8mm (年度), 或
  - 部分实性结节 ≥6mm, 实性成分<6mm (基线) / 新发或<6mm (年度), 或
  - 新发实性结节 4 to <6mm

Category 4A (可疑恶性):
  - 实性结节 ≥8 to <15mm (基线) / 新发6 to <8mm / ≥8 to <15mm 增长, 或
  - 部分实性结节 ≥6mm, 实性成分≥6 to <8mm, 或
  - 新发部分实性结节 <6mm实性成分, 或
  - 支气管内结节

Category 4B (高度可疑恶性):
  - 实性结节 ≥15mm (基线) / 新发≥8mm, 或
  - 部分实性结节 实性成分≥8mm, 或
  - 新发部分实性结节 ≥6mm实性成分且≥8mm总径

Category 4X (极高度可疑恶性伴其他征象):
  - Category 3/4 且: 淋巴结肿大, 胸膜侵犯, 或 其他恶性征象

Category S (临床显著, 非肺癌):
  - 临床显著的异常发现, 与肺癌筛查相关但非结节
"""
            },
            {
                "title": "Lung-RADS 管理建议",
                "content": """
1-2: 继续年度低剂量CT筛查
3: 6个月LDCT
4A: 3个月LDCT; 如持续 → PET/CT ± 组织活检
4B: 立即胸部CT增强; 考虑PET/CT; 如高度可疑→穿刺活检
4X: 立即进一步检查; 多学科会诊
"""
            },
        ],
    },
    "nccn_lcs_2024": {
        "name": "NCCN Lung Cancer Screening v2024",
        "sections": [
            {
                "title": "筛查人群",
                "content": """
符合以下条件者进入年度LDCT筛查:
- 年龄50-80岁
- 吸烟史≥20包年
- 目前吸烟或戒烟<15年
"""
            },
            {
                "title": "结节管理",
                "content": """
实性结节:
- <5mm: 年度LDCT
- 5-7mm: 6个月LDCT
- 8-14mm: 3个月LDCT 或 PET/CT
- ≥15mm: 立即胸部CT增强; 考虑PET/CT ± 活检

部分实性结节:
- <6mm: 年度LDCT
- ≥6mm且实性成分<6mm: 3-6个月LDCT
- 实性成分≥6mm: 3个月LDCT; 持续→PET/CT ± 活检
"""
            },
        ],
    },
    "china_consensus_2024": {
        "name": "中国肺结节诊疗专家共识 2024",
        "description": "中国肺癌筛查与早诊早治指南",
        "sections": [
            {
                "title": "中国肺结节管理原则",
                "content": """
中国肺癌高危人群定义 (中位年龄>50岁):
- 吸烟≥20包年 (含既往吸烟但戒烟<15年)
- 有环境或高危职业暴露史 (石棉、铍、铀、氡等)
- 合并COPD、弥漫性肺纤维化或既往有肺结核病史
- 既往罹患恶性肿瘤或有肺癌家族史者

结节管理 (基于中国数据):
- 纯GGO ≤5mm: 无需常规随访; 鼓励年度体检
- 纯GGO >5mm: 3个月首次随访; 稳定后年度随访
- 部分实性结节 (任意大小): 3个月随访; 如持续→MDT讨论
- 实性结节 6-8mm: 6-12个月随访; ≥8mm: 3个月随访或PET/CT

特别提示:
- 中国人肺结节中磨玻璃比例较高 (约30-40%)
- 亚裔非吸烟者肺癌发病率高于其他人群
- 推荐使用低剂量CT (≤1mSv)
"""
            },
        ],
    },
}


class GuidelineRetrieval:
    """指南检索工具"""

    def __init__(self, knowledge_base_dir: Optional[str] = None):
        """
        Args:
            knowledge_base_dir: 本地指南文件目录 (可选)
               如果目录中有文件, 优先使用本地文件
               否则使用内置数据
        """
        self.guidelines = {}

        # 尝试加载本地文件
        if knowledge_base_dir and os.path.isdir(knowledge_base_dir):
            for fname in os.listdir(knowledge_base_dir):
                if fname.endswith(".txt") or fname.endswith(".md"):
                    fpath = os.path.join(knowledge_base_dir, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        key = fname.replace(".txt", "").replace(".md", "")
                        self.guidelines[key] = {
                            "name": key,
                            "content": f.read(),
                        }

        # 如果本地无数据, 使用内置
        if not self.guidelines:
            self.guidelines = GUIDELINE_DATA
            print(f"[GuidelineRetrieval] 使用内置指南数据 ({len(self.guidelines)} 份指南)")

    def search(
        self,
        query: str,
        guideline_type: Optional[str] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        检索指南内容

        Args:
            query: 搜索查询 (如 "部分实性结节 >8mm 管理")
            guideline_type: 限定指南类型 (fleischner / lung_rads / nccn / china)
            top_k: 返回的最大段落数

        Returns:
            {
                "results": [{"source": str, "title": str, "content": str, "score": float}, ...],
                "summary": str,
            }
        """
        results = []

        # 确定要搜索的指南
        if guideline_type:
            targets = {k: v for k, v in self.guidelines.items()
                       if guideline_type.lower() in k.lower()}
            if not targets:
                # fallback: 搜索所有
                targets = self.guidelines
        else:
            targets = self.guidelines

        # 简单的 BM25 搜索
        query_terms = set(query.lower().split())

        for guide_key, guide_data in targets.items():
            sections = guide_data.get("sections", [])

            for section in sections:
                title = section.get("title", "")
                content = section.get("content", "")

                # 计算匹配分
                text_lower = (title + " " + content).lower()
                score = sum(1 for term in query_terms if term in text_lower)
                # 归一化: 匹配项数 / 查询词数
                score = score / len(query_terms) if query_terms else 0

                if score > 0:
                    results.append({
                        "source": guide_data.get("name", guide_key),
                        "title": title,
                        "content": content.strip(),
                        "score": round(score, 3),
                    })

        # 排序, 取 top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]

        # 生成摘要
        if results:
            sources = list(set(r["source"] for r in results))
            summary = f"从 {len(sources)} 份指南中检索到 {len(results)} 条相关内容: {', '.join(sources)}"
        else:
            summary = f"未找到与 '{query}' 相关的指南内容。请尝试更通用的查询。"

        return {
            "results": results,
            "summary": summary,
            "query": query,
        }

    def format_for_llm(self, search_result: Dict) -> str:
        """格式化为 LLM 可用的文本"""
        lines = [search_result["summary"], ""]

        for i, r in enumerate(search_result["results"], 1):
            lines.append(f"[{i}] {r['source']} — {r['title']}")
            # 截断过长内容
            content = r["content"].strip()[:500]
            if len(r["content"]) > 500:
                content += "..."
            lines.append(content)
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    retriever = GuidelineRetrieval()
    result = retriever.search("部分实性结节 8mm 管理", guideline_type="fleischner")
    print(retriever.format_for_llm(result))
