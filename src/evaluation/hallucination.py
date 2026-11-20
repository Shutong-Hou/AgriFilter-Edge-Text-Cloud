"""幻觉率管线（方案附录 B.1）。

实体抽取（别名词典归一化）→ 与"输入 ∪ KG ∪ 预注册白名单"比对 → 未匹配人工复核 →
200 例金标准校准自动管线 precision/recall → 校正幻觉率 + CI。
规则 S3 冻结。
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

TOKEN_SPLIT = re.compile(r"[\s,;：:，。、（）()/·\-—_]+")


def extract_entities(text: str, alias_dict: dict[str, str]) -> set[str]:
    """从文本中抽取实体：先按别名词典做最长匹配归一化，再取未被词典覆盖的生物学名词。"""
    lowered = text.lower()
    found: set[str] = set()
    # 别名词典归一化（别名 → 规范实体）
    for alias, canon in alias_dict.items():
        if alias.lower() in lowered:
            found.add(canon)
    # 兜底：按 token 抽取长度>=2 且非停用的词（可配置）
    for tok in TOKEN_SPLIT.split(text):
        t = tok.strip().lower()
        if len(t) >= 2 and t not in _STOPWORDS:
            found.add(t)
    return found


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are",
    "this", "that", "请", "给出", "输出", "诊断", "建议", "然后", "以及", "属于", "该", "作物",
}


def is_hallucinated(
    entities: set[str],
    allowed: set[str],
    whitelist: set[str],
) -> tuple[bool, set[str]]:
    """实体不外于 (输入∪KG∪白名单) 视为正常；否则（不在 allowed/whitelist）为疑似幻觉。"""
    unknown = entities - allowed - whitelist
    # 去除纯停用残留
    unknown = {u for u in unknown if u not in _STOPWORDS}
    return bool(unknown), unknown


def calibrate_auto_pipeline(
    auto_flags: list[bool],
    manual_flags: list[bool],
) -> dict:
    """用 200 例金标准（人工复核）校准自动管线，返回 precision/recall/校正幻觉率。"""
    from sklearn.metrics import precision_score, recall_score

    p = precision_score(manual_flags, auto_flags, zero_division=0)
    r = recall_score(manual_flags, auto_flags, zero_division=0)
    return {"precision": float(p), "recall": float(r)}


def corrected_hallucination_rate(
    auto_rate: float,
    precision: float,
    recall: float,
) -> float:
    """校正幻觉率：auto_rate 按自动管线 precision/recall 校正。"""
    if recall == 0:
        return float("nan")
    return auto_rate * precision / recall


def load_whitelist(path: Path | None = None) -> set[str]:
    """预注册白名单/别名词典（可放 configs/kg/ 下）。"""
    from ..utils.io import load_yaml

    if path is None:
        path = Path(project_path("configs", "kg", "whitelist.yaml"))
    if Path(path).exists():
        data = load_yaml(path)
        return set(data.get("whitelist", []))
    return set()