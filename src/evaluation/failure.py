"""失效分析（方案 E18）。

分类 F1–F7（F6a/F6b 细分）。每类报双列：占进入 LLM 样本比 + 占总样本比（消除幸存者偏差）。
双人标注 + κ≥0.7，不足引第三人。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FailureLabels:
    F1: list[str] = field(default_factory=list)   # SAF 误放（异常仍传图但实际应拦截等）
    F2: list[str] = field(default_factory=list)   # SAF 误拦
    F3: list[str] = field(default_factory=list)   # CSC错→LLM纠错（LLM 纠正成功）
    F4: list[str] = field(default_factory=list)   # CSC错→LLM被带偏（LLM 跟着错）
    F5: list[str] = field(default_factory=list)   # LLM 推理错误（细分类型/程度/建议/矛盾）
    F6a: list[str] = field(default_factory=list)  # schema 非法（解析失败）
    F6b: list[str] = field(default_factory=list)  # 字段退化（合法但空字段/越界/截断）
    F7: list[str] = field(default_factory=list)   # 幻觉

    def all_ids(self) -> list[str]:
        keys = ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b", "F7"]
        return [k for k in keys if getattr(self, k)]


def annotate_failure(sample: dict[str, Any]) -> FailureLabels:
    """单样本自动标注（辅助自动管线；E18 要求双人标注，此处为半自动初标）。

    sample 需含字段：saf_decision, csc_correct, llm_correct, llm_was_corrected,
    schema_valid, field_degraded, hallucination, llm_parse_failed。
    """
    lab = FailureLabels()
    entered_llm = sample.get("entered_llm", True)
    if not entered_llm:
        # 未进入 LLM：F1/F2 由 SAF 判定
        if sample.get("saf_decision") == "rejected" and not sample.get("should_reject"):
            lab.F1.append(sample.get("id", ""))
        if sample.get("saf_decision") == "passed" and sample.get("should_reject", False):
            lab.F2.append(sample.get("id", ""))
        return lab
    # 进入 LLM
    if sample.get("llm_parse_failed"):
        lab.F6a.append(sample.get("id", ""))
    elif sample.get("field_degraded"):
        lab.F6b.append(sample.get("id", ""))
    if sample.get("hallucination"):
        lab.F7.append(sample.get("id", ""))
    # CSC 状态
    if not sample.get("csc_correct") and sample.get("llm_correct"):
        lab.F3.append(sample.get("id", ""))
    elif not sample.get("csc_correct") and not sample.get("llm_correct") and not sample.get("schema_valid") is False:
        # 无法归因到 F6 时记为被带偏
        lab.F4.append(sample.get("id", ""))
    elif sample.get("csc_correct") and not sample.get("llm_correct"):
        lab.F5.append(sample.get("id", ""))
    return lab


def double_column_frequencies(
    labels: FailureLabels,
    n_entered_llm: int,
    n_total: int,
) -> dict[str, dict[str, float]]:
    """每类双列：占进入 LLM 样本比 + 占总样本比。"""
    out = {}
    for key in ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b", "F7"]:
        ids = getattr(labels, key)
        out[key] = {
            "count": len(ids),
            "frac_of_llm": len(ids) / max(1, n_entered_llm),
            "frac_of_total": len(ids) / max(1, n_total),
        }
    return out


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """双人标注 Cohen's κ（F1–F7 类别标签）。"""
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(labels_a, labels_b))