#!/usr/bin/env python3
"""
测量计算工具

计算结节相关临床指标:
  - Lung-RADS 分级
  - 体积倍增时间 (VDT)
  - 恶性概率 (基于梅奥/VA模型简化版)

使用方式:
  from agent.tools.measurement_calc import MeasurementCalculator

  calc = MeasurementCalculator()
  result = calc.lung_rads_classify(nodule_type="part-solid", size_mm=12.3, solid_mm=5.0)
"""

import math
from typing import Dict, Optional


class MeasurementCalculator:
    """结节临床指标计算器"""

    # ============================================================
    # 1. Lung-RADS 分级
    # ============================================================

    @staticmethod
    def lung_rads_classify(
        nodule_type: str,
        size_mm: float,
        solid_component_mm: Optional[float] = None,
        is_new: bool = False,
        is_growing: bool = False,
        is_baseline: bool = True,
    ) -> Dict:
        """
        Lung-RADS v2022 分级

        Args:
            nodule_type: "solid" | "part-solid" | "ggn"
            size_mm: 结节总长径 (mm)
            solid_component_mm: 实性成分长径 (仅 part-solid 需要)
            is_new: 是否新发结节
            is_growing: 是否正在增大
            is_baseline: 是否基线筛查

        Returns:
            {category, description, recommendation}
        """
        if nodule_type == "solid":
            result = MeasurementCalculator._classify_solid(
                size_mm, is_new, is_growing, is_baseline
            )
        elif nodule_type == "part-solid":
            if solid_component_mm is None:
                solid_component_mm = size_mm * 0.3  # 默认估计
            result = MeasurementCalculator._classify_part_solid(
                size_mm, solid_component_mm, is_new, is_baseline
            )
        elif nodule_type in ("ggn", "ground-glass", "nonsolid"):
            result = MeasurementCalculator._classify_ggn(size_mm)
        else:
            return {"error": f"未知结节类型: {nodule_type}"}

        return result

    @staticmethod
    def _classify_solid(size, is_new, is_growing, is_baseline):
        if size >= 15:
            cat = "4B"
            desc = "高度可疑恶性"
            rec = "立即胸部增强CT; 考虑PET/CT; 如持续→穿刺活检"
        elif size >= 8:
            cat = "4A"
            desc = "可疑恶性"
            rec = "3个月LDCT; 如持续→PET/CT或活检"
        elif size >= 6:
            cat = "3"
            desc = "可能良性"
            rec = "6个月LDCT"
        elif is_new and size >= 4:
            cat = "3"
            desc = "可能良性 (新发)"
            rec = "6个月LDCT"
        elif size >= 4 and not is_baseline:
            cat = "2"
            desc = "良性表现"
            rec = "继续年度LDCT筛查"
        else:
            cat = "1"
            desc = "阴性"
            rec = "继续年度LDCT筛查"

        return {
            "category": f"Lung-RADS {cat}",
            "description": desc,
            "recommendation": rec,
            "nodule_type": "实性 (solid)",
            "size_mm": size,
        }

    @staticmethod
    def _classify_part_solid(total_size, solid_mm, is_new, is_baseline):
        if solid_mm >= 8:
            cat = "4B"
            desc = "高度可疑恶性"
            rec = "立即增强CT; PET/CT; 考虑穿刺活检"
        elif solid_mm >= 6:
            cat = "4A"
            desc = "可疑恶性"
            rec = "3个月LDCT; 如持续→PET/CT"
        elif total_size >= 6:
            if is_baseline:
                cat = "3"
                desc = "可能良性 (基线)"
                rec = "6个月LDCT; 稳定后年度随访"
            else:
                cat = "4A"
                desc = "可疑恶性 (新发/增长)"
                rec = "3个月LDCT"
        else:
            cat = "2"
            desc = "良性表现"
            rec = "年度LDCT"

        return {
            "category": f"Lung-RADS {cat}",
            "description": desc,
            "recommendation": rec,
            "nodule_type": f"部分实性 (part-solid, 实性成分 {solid_mm}mm)",
            "total_size_mm": total_size,
            "solid_component_mm": solid_mm,
        }

    @staticmethod
    def _classify_ggn(size):
        if size >= 30:
            cat = "2"
            desc = "良性表现 (大GGO但纯磨玻璃)"
            rec = "年度LDCT"
        elif size >= 6:
            cat = "2"
            desc = "良性表现"
            rec = "12个月LDCT; 如稳定则每2年一次至5年"
        else:
            cat = "2"
            desc = "良性表现 (小GGO)"
            rec = "年度LDCT (如>30岁)"

        return {
            "category": f"Lung-RADS {cat}",
            "description": desc,
            "recommendation": rec,
            "nodule_type": "纯磨玻璃 (pure GGN)",
            "size_mm": size,
        }

    # ============================================================
    # 2. 体积倍增时间 (VDT)
    # ============================================================

    @staticmethod
    def volume_doubling_time(
        diameter1_mm: float,
        diameter2_mm: float,
        interval_days: int,
    ) -> Dict:
        """
        计算结节体积倍增时间

        公式: VDT = t × ln(2) / ln(V2/V1)
        体积 ≈ π/6 × d³ (假设球形)

        Args:
            diameter1_mm: 首次测量长径
            diameter2_mm: 第二次测量长径
            interval_days: 两次检查间隔天数

        Returns:
            {vdt_days, annual_growth_rate_pct, risk_assessment}
        """
        if diameter2_mm <= diameter1_mm:
            return {
                "vdt_days": "无法计算 (结节未增大)",
                "annual_growth_rate_pct": 0.0,
                "risk_assessment": "结节稳定或缩小, 通常为良性表现",
            }

        # 体积比
        v1 = diameter1_mm ** 3
        v2 = diameter2_mm ** 3
        volume_ratio = v2 / v1

        # VDT
        vdt = interval_days * math.log(2) / math.log(volume_ratio)

        # 年增长率 (百分比)
        days_per_year = 365
        annual_growth = (volume_ratio ** (days_per_year / interval_days) - 1) * 100

        # 风险评估
        if vdt < 30:
            risk = "极快增长 (<30天), 提示感染或炎症 (非肿瘤典型)"
        elif vdt < 100:
            risk = "快速增长 (<100天), 高度可疑恶性 (小细胞肺癌可能)"
        elif vdt < 400:
            risk = "中等增长 (100-400天), 符合恶性肿瘤倍增时间"
        elif vdt < 800:
            risk = "缓慢增长 (400-800天), 可为恶性也可为良性"
        else:
            risk = "极慢增长 (>800天), 更倾向良性或惰性肿瘤"

        return {
            "vdt_days": round(vdt, 0),
            "annual_growth_rate_pct": round(annual_growth, 1),
            "risk_assessment": risk,
        }

    # ============================================================
    # 3. 恶性概率 (简化梅奥模型)
    # ============================================================

    @staticmethod
    def malignancy_probability(
        diameter_mm: float,
        location_upper_lobe: bool = False,
        spiculation_present: bool = False,
        age_years: int = 60,
        smoking_current_or_former: bool = True,
        family_history: bool = False,
    ) -> Dict:
        """
        简化梅奥模型恶性概率估计

        梅奥模型 (Mayo Clinic Model):
        P(malignant) = 1 / (1 + e^(-X))
        X = -6.8272 + 0.0391*age + 0.7917*smoke + 1.3388*cancer_hist
            + 0.1274*diameter + 1.0407*spiculation + 0.7838*upper_lobe
        """
        # 简化计算
        try:
            x = -6.8272
            x += 0.0391 * age_years
            x += 0.7917 * (1 if smoking_current_or_former else 0)
            x += 0.1274 * diameter_mm
            x += 1.0407 * (1 if spiculation_present else 0)
            x += 0.7838 * (1 if location_upper_lobe else 0)
            # family_history 用 cancer_history 替代 (近似)
            if family_history:
                x += 1.3388

            prob = 1.0 / (1.0 + math.exp(-x))
            prob_pct = prob * 100

            if prob_pct < 5:
                risk_level = "低风险 (<5%)"
            elif prob_pct < 25:
                risk_level = "中低风险 (5-25%)"
            elif prob_pct < 65:
                risk_level = "中等风险 (25-65%)"
            elif prob_pct < 90:
                risk_level = "高风险 (65-90%)"
            else:
                risk_level = "极高风险 (>90%)"

            return {
                "malignancy_probability_pct": round(prob_pct, 1),
                "risk_level": risk_level,
                "model": "Mayo Clinic Model (简化)",
                "note": "本模型仅供辅助评估, 不能替代临床综合判断",
            }

        except Exception as e:
            return {"error": str(e)}

    # ============================================================
    # 格式化
    # ============================================================

    def format_for_llm(self, result: Dict) -> str:
        """格式化为 LLM 可读文本"""
        lines = []

        if "error" in result:
            return f"[错误] {result['error']}"

        if "category" in result:
            # Lung-RADS 分级结果
            lines.append(f"Lung-RADS 分级: {result['category']}")
            lines.append(f"描述: {result['description']}")
            lines.append(f"结节类型: {result.get('nodule_type', 'N/A')}")
            lines.append(f"建议: {result['recommendation']}")

        if "vdt_days" in result:
            lines.append(f"\n体积倍增时间 (VDT): {result['vdt_days']} 天")
            lines.append(f"年增长率: {result['annual_growth_rate_pct']}%")
            lines.append(f"风险评估: {result['risk_assessment']}")

        if "malignancy_probability_pct" in result:
            lines.append(f"\n恶性概率: {result['malignancy_probability_pct']}%")
            lines.append(f"风险等级: {result['risk_level']}")
            lines.append(f"模型: {result['model']}")
            lines.append(f"注意: {result.get('note', '')}")

        return "\n".join(lines)


if __name__ == "__main__":
    calc = MeasurementCalculator()

    print("=== Lung-RADS 分级 ===")
    r = calc.lung_rads_classify("part-solid", 12.3, solid_component_mm=5.0)
    print(calc.format_for_llm(r))

    print("\n=== VDT 计算 ===")
    r = calc.volume_doubling_time(10.0, 13.0, 180)
    print(calc.format_for_llm(r))

    print("\n=== 恶性概率 ===")
    r = calc.malignancy_probability(12.3, upper_lobe=True, spiculation=True)
    print(calc.format_for_llm(r))
