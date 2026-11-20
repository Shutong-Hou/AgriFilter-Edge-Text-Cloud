"""输出三级完整性阶梯（方案附录 B.8 / E5/E8/E9/E15/E18）。

L1 结构有效率  : JSON 可解析且通过冻结 Schema 校验（四 required 字段、类型、confidence∈[0,1]、minLength≥1）。
L2 字段非退化率: L1 基础上四字段非空、reasoning 长度≥阈值、action 非占位符。
L3 截断率     : finish_reason≠"stop" 的比例。
红线15：guided 臂 L1=100% 时为构造性保证，须报 CP 上界并注明；L1/L2 与语义准确率成对呈现。
"""
from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft7Validator

from ..utils.io import load_yaml, project_path, load_json

CFG = load_yaml(project_path("configs", "eval_config.yaml"))["integrity_ladder"]
_PLACEHOLDERS = set(CFG["placeholders"])


def load_validator() -> Draft7Validator:
    schema = load_json(project_path("configs", "schema.json"))
    return Draft7Validator(schema)


def validate_record(
    text: str,
    validator: Draft7Validator | None = None,
    finish_reason: str | None = "stop",
) -> dict[str, Any]:
    """对单条 LLM 输出做三级完整性判定。返回判定明细。"""
    validator = validator or load_validator()
    result = {
        "l1_valid": False,
        "l1_errors": [],
        "l2_non_degraded": False,
        "l2_degrade_reasons": [],
        "l3_truncated": False,
        "parsed": None,
    }
    # L3 截断
    result["l3_truncated"] = bool(finish_reason is not None and finish_reason != "stop")

    # L1 解析 + schema 校验
    try:
        obj = json.loads(text)
        result["parsed"] = obj
    except json.JSONDecodeError as e:
        result["l1_errors"] = [f"json_parse_error: {e}"]
        return result
    errors = sorted(validator.iter_errors(obj), key=lambda e: str(e.path))
    if errors:
        result["l1_errors"] = [e.message for e in errors]
        return result
    result["l1_valid"] = True

    # L2 字段非退化
    degrade = []
    for f in CFG["required_fields"]:
        v = obj.get(f)
        if v is None or (isinstance(v, str) and len(v.strip()) == 0):
            degrade.append(f"empty_field:{f}")
        elif isinstance(v, str) and v.strip().lower() in _PLACEHOLDERS:
            degrade.append(f"placeholder_field:{f}")
    if isinstance(obj.get("reasoning"), str) and len(obj["reasoning"]) < CFG["reasoning_min_len"]:
        degrade.append("reasoning_too_short")
    if isinstance(obj.get("action"), str) and len(obj["action"]) < CFG["action_min_len"]:
        degrade.append("action_too_short")
    if not isinstance(obj.get("confidence"), (int, float)):
        degrade.append("confidence_type")
    elif not (CFG["confidence_min"] <= obj["confidence"] <= CFG["confidence_max"]):
        degrade.append("confidence_out_of_range")
    result["l2_degrade_reasons"] = degrade
    result["l2_non_degraded"] = len(degrade) == 0
    return result


def ladder_summary(records: list[dict[str, Any]], n_total: int) -> dict:
    """汇总一批判定结果，报三级比例 + 0 事件时 CP 上界。

    records: validate_record 返回的列表。
    """
    from .statistics import wilson_ci, cp_zero_bound

    n = len(records)
    l1 = sum(1 for r in records if r["l1_valid"])
    l2 = sum(1 for r in records if r["l2_non_degraded"])
    l3 = sum(1 for r in records if r["l3_truncated"])
    out = {
        "n": n,
        "l1_valid": l1 / n if n else 0.0,
        "l1_ci": wilson_ci(l1, n) if n else (0, 0),
        "l1_cp_upper_when_zero": cp_zero_bound(n) if n and l1 == 0 else None,
        "l2_non_degraded": l2 / n if n else 0.0,
        "l2_ci": wilson_ci(l2, n) if n else (0, 0),
        "l3_truncation_rate": l3 / n if n else 0.0,
    }
    return out