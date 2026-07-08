#!/usr/bin/env python3
"""
网络搜索工具 (基于 Tavily Search API)

Tavily 专为 AI Agent 设计, 搜索结果质量优于通用搜索引擎,
特别适合医学文献和临床研究检索。

使用前设置 API Key:
  export TAVILY_API_KEY="tvly-xxxxxxxxxxxxx"

使用方式:
  from agent.tools.web_search import WebSearch

  searcher = WebSearch()
  result = searcher.search("part-solid nodule malignancy risk 2025")
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime


class WebSearch:
    """医学文献网络搜索 (基于 Tavily API)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: int = 15,
        include_raw_content: bool = False,
    ):
        """
        Args:
            api_key: Tavily API key (默认从环境变量 TAVILY_API_KEY 读取)
            max_results: 最大结果数
            timeout: 请求超时 (秒)
            include_raw_content: 是否包含原始页面内容 (token 消耗更大)
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self.max_results = max_results
        self.timeout = timeout
        self.include_raw_content = include_raw_content

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict:
        """
        搜索网络

        Args:
            query: 搜索查询
            max_results: 最大结果数
            search_depth: "basic" (快速) 或 "advanced" (深入, 推荐医学查询)
            include_domains: 限定搜索域名 (如 ["pubmed.ncbi.nlm.nih.gov", "radiopaedia.org"])
            exclude_domains: 排除域名

        Returns:
            {
                "results": [{"title", "url", "content", "score", "raw_content"}],
                "summary": str,
                "query": str,
                "answer": str (Tavily AI 生成的摘要, advanced 模式),
            }
        """
        n_results = max_results or self.max_results

        if not self.api_key:
            return self._fallback_search(query, n_results)

        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)

            # 医学查询优化: 自动追加医学域名
            if include_domains is None:
                include_domains = [
                    "pubmed.ncbi.nlm.nih.gov",
                    "radiopaedia.org",
                    "nih.gov",
                    "mayoclinic.org",
                    "uptodate.com",
                    "thelancet.com",
                    "nejm.org",
                    "cnki.net",          # 中国知网
                    "yiigle.com",        # 中华医学期刊
                ]

            response = client.search(
                query=query,
                max_results=n_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains or [],
                include_answer="advanced",
                include_raw_content=self.include_raw_content,
                include_images=False,
            )

            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],
                    "score": round(r.get("score", 0), 3),
                    "raw_content": r.get("raw_content", "")[:1000] if self.include_raw_content else "",
                    "source": "Tavily",
                })

            # Tavily AI 摘要
            answer = response.get("answer", "")

            if results:
                summary = f"搜索 '{query}' 获得 {len(results)} 条结果"
                if answer:
                    summary += f"\n\nAI 摘要: {answer[:300]}"
            else:
                summary = f"未找到 '{query}' 的相关结果"

            return {
                "results": results,
                "summary": summary,
                "query": query,
                "answer": answer,
                "response_time": response.get("response_time", 0),
                "timestamp": datetime.now().isoformat(),
            }

        except ImportError:
            print("[WARN] tavily-python 未安装, 使用离线 fallback。pip install tavily-python")
            return self._fallback_search(query, n_results)
        except Exception as e:
            return {
                "results": [],
                "summary": f"搜索失败: {str(e)}",
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _fallback_search(self, query: str, n_results: int) -> Dict:
        """离线 Fallback: 返回预置的医学参考信息"""
        pn_patterns = {
            "part-solid": (
                "根据 Fleischner 2017 指南, ≥8mm 的部分实性结节建议3个月CT随访。"
                "部分实性结节的实性成分大小是恶性风险评估的关键指标。"
            ),
            "solid nodule": (
                "实性结节 <6mm 通常无需随访。≥8mm 实性结节需短期随访或进一步检查"
                "(PET/CT, 活检)。"
            ),
            "ground glass": (
                "纯磨玻璃结节生长缓慢, 惰性行为。≥6mm 建议6-12个月随访, "
                "后每2年一次至5年。约10-20%会进展为部分实性或实性结节。"
            ),
            "lung-rads": (
                "Lung-RADS v2022 将结节分为 0-4X 级。4B 级(高度可疑)建议立即增强CT "
                "和/或 PET/CT。"
            ),
            "malignancy": (
                "肺结节恶性风险因素: 直径>8mm, 边缘毛刺, 分叶状, 部分实性, "
                "上叶位置, 患者年龄>50岁, 吸烟史。"
            ),
            "follow-up": (
                "肺结节随访策略应根据结节大小、类型和患者风险因素个体化制定。"
                "参照 Fleischner 2017 或 Lung-RADS v2022 指南。"
            ),
        }

        matched = ""
        for key, text in pn_patterns.items():
            if key in query.lower():
                matched += text + "\n\n"

        if not matched:
            matched = (
                f"未找到与 '{query}' 精确匹配的离线参考信息。"
                f"建议: 1) 设置 TAVILY_API_KEY 环境变量启用在线搜索; "
                f"2) 查询 PubMed (pubmed.ncbi.nlm.nih.gov) 获取最新文献。"
            )

        return {
            "results": [{
                "title": "医学参考 (离线模式)",
                "content": matched,
                "url": "",
                "score": 1.0,
                "source": "built-in",
            }],
            "summary": f"搜索 '{query}' (离线模式)",
            "query": query,
            "answer": "",
            "timestamp": datetime.now().isoformat(),
            "note": "离线模式: pip install tavily-python 并设置 TAVILY_API_KEY 以获得完整搜索功能",
        }

    def format_for_llm(self, search_result: Dict) -> str:
        """格式化为 LLM 可读文本"""
        lines = [search_result["summary"], ""]

        if "note" in search_result:
            lines.append(f"[注意] {search_result['note']}")
            lines.append("")

        # Tavily AI 答案
        answer = search_result.get("answer", "")
        if answer:
            lines.append(f"📝 AI 摘要: {answer}")
            lines.append("")

        # 搜索结果
        for i, r in enumerate(search_result.get("results", []), 1):
            lines.append(f"[{i}] {r.get('title', '无标题')}")
            lines.append(f"    {r.get('content', '')[:400]}")
            url = r.get("url", "")
            if url:
                lines.append(f"    URL: {url}")
            score = r.get("score", 0)
            if score:
                lines.append(f"    相关性: {score}")
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    searcher = WebSearch()
    result = searcher.search(
        "part-solid pulmonary nodule management guidelines 2025",
        search_depth="advanced",
    )
    print(searcher.format_for_llm(result))
