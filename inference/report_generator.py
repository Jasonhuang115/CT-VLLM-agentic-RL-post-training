#!/usr/bin/env python3
"""结构化报告生成器 (后处理格式化为标准报告)"""

import re
from datetime import datetime

DISCLAIMER = """---
⚠️ 医疗免责声明

本报告由AI辅助诊断系统生成, 仅供临床参考, 不能替代专业医师的诊断意见。
所有诊断结论必须在执业医师审核确认后方可用于临床决策。
AI系统可能遗漏重要病变或产生错误判断, 请务必结合患者完整病史、
体格检查及其他辅助检查结果综合评估。"""


def format_structured_report(raw_output: str) -> str:
    """确保报告有完整结构"""
    has_findings = bool(re.search(r"(影像学发现|Findings|发现)", raw_output, re.IGNORECASE))
    has_assessment = bool(re.search(r"(恶性风险|Assessment|Lung-RADS|评估)", raw_output, re.IGNORECASE))
    has_recommendation = bool(re.search(r"(随访|建议|Recommendation)", raw_output, re.IGNORECASE))

    if has_findings and has_assessment and has_recommendation:
        report = raw_output
    else:
        report = raw_output + "\n\n" if has_findings else "**影像学发现**: " + raw_output
        if not has_assessment:
            report += "\n\n**恶性风险评估**: 请参考上述特征综合判断"
        if not has_recommendation:
            report += "\n\n**建议**: 建议结合临床及实验室检查综合评估"

    report += f"\n{DISCLAIMER}\n\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    return report


if __name__ == "__main__":
    test = "右上叶后段见12.3mm结节, 边缘毛刺, 部分实性。Lung-RADS 4B, 建议3个月随访。"
    print(format_structured_report(test))
