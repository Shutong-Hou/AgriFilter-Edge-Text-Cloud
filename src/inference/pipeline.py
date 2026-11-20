"""端到端管线（SAF → CSC → 云端文本 LLM）与载荷构建（B.7）。

- 载荷：L0 仅标签 / L2 KG 硬描述 / L2e 无关上下文控制 / L3 软描述(Top-3+u) / L4 完全体。
- L3 载荷 = {"top3":[[类,概率]×3], "u": SAF-EDL}；L4 追加各类 KG 描述。
- 路由（B.6）：u>τ* → 拒识；u≤τ* → 映射流水线。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.io import load_yaml, project_path

PROMPTS = load_yaml(project_path("configs", "prompts.yaml"))
SYSTEM = PROMPTS["system_instruct"]
U_CONTEXT = PROMPTS["u_context"]


def build_top3_payload(top3: list[tuple[str, float]]) -> dict:
    return {"top3": [[cls, float(p)] for cls, p in top3], "u": None}


def ternary(x: Any, default: str = "N/A") -> str:
    return x if x else default


def build_prompt(
    arm: str,
    classes: list[str],
    top3: list[tuple[str, float]] | None = None,
    u: float | None = None,
    kg_descriptions: dict[str, str] | None = None,
    filler: str = "",
    examples: str = "",
) -> str:
    cls_str = ", ".join(classes)
    top3_str = ""
    if top3:
        top3_str = "; ".join(f"{c}({p:.3f})" for c, p in top3)
    if arm == "L0":
        return PROMPTS["L0"].format(system=SYSTEM, classes=cls_str)
    if arm == "L2":
        # L2 = 纯 KG 硬描述（方案 E5 定义，不含 top3 概率；top3 是 L3 才引入的增量）
        desc = "\n".join(
            f"- {c}: {kg_descriptions.get(c, '')}" for c in classes
        ) if kg_descriptions else cls_str
        return PROMPTS["L2"].format(
            system=SYSTEM, class_descriptions=desc
        )
    if arm == "L2e":
        return PROMPTS["L2e"].format(
            system=SYSTEM, classes=cls_str, filler=filler,
            fill_instruction=PROMPTS["guided_attach"],
        )
    if arm == "L3":
        top3u = json.dumps({"top3": [[c, p] for c, p in top3], "u": u}, ensure_ascii=False)
        return PROMPTS["L3"].format(
            system=SYSTEM, classes=cls_str, top3_with_uncertainty=top3u, u_context=U_CONTEXT
        )
    if arm == "L4":
        desc = "\n".join(
            f"- {c}: {kg_descriptions.get(c, '')}" for c in classes
        ) if kg_descriptions else cls_str
        top3u = json.dumps({"top3": [[c, p] for c, p in top3], "u": u}, ensure_ascii=False)
        return PROMPTS["L4"].format(
            system=SYSTEM, class_descriptions=desc,
            top3_with_uncertainty=top3u, u_context=U_CONTEXT,
        )
    if arm == "zero_shot":
        return PROMPTS["zero_shot"].format(system=SYSTEM, classes=cls_str)
    if arm == "cot":
        return PROMPTS["chain_of_thought"].format(system=SYSTEM, classes=cls_str)
    if arm == "few_shot_3":
        return PROMPTS["few_shot_3"].format(
            system=SYSTEM, classes=cls_str, examples=examples
        )
    raise ValueError(f"未知臂: {arm}")


def route(u: float, tau_star: float) -> str:
    """u>τ* → 'reject'；否则 'pass'。"""
    return "reject" if u > tau_star else "pass"


def build_pipeline_sample(
    arm: str,
    classes: list[str],
    top3: list[tuple[str, float]],
    u: float,
    kg_descriptions: dict[str, str] | None,
    filler: str = "",
    examples: str = "",
) -> str:
    return build_prompt(arm, classes, top3, u, kg_descriptions, filler, examples)