"""E18：失效分析【W11】(方案 E18)。

分类：F1 SAF 误放 / F2 SAF 误拦 / F3 CSC错→LLM纠错 / F4 CSC错→LLM被带偏 / F5 LLM 推理错误
/ F6a schema 非法 / F6b 字段退化 / F7 幻觉。
每类报双列：占进入 LLM 样本比 + 占总样本比（消除幸存者偏差）。
双人标注 + κ≥0.7，不足引第三人；每类 2–3 案例画廊。

用法: python -m src.experiments.E18_failure
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..evaluation.failure import FailureLabels, double_column_frequencies, cohen_kappa
from .common import load_partition, load_pv_classes, results_dir

# 需从 E5/E8/E9/E12 的预测池读取样本级判定（本脚本读取缓存的样本级日志）
PRED_POOL = Path(project_path("results", "prediction_pool"))


def main() -> None:
    with ExperimentContext(experiment="E18", description="失效分析", configs={}) as ctx:
        # 从预测池读取样本级数据（若存在）
        samples = _load_samples()
        if not samples:
            print("[E18] 未找到样本级预测池（results/prediction_pool）。请先运行 E5/E8/E9/E12 生成。")
            return
        n_total = len(samples)
        n_llm = sum(1 for s in samples if s.get("entered_llm", True))

        # 自动初标（半自动，E18 要求双人标注，此处为辅助初标）
        labels = FailureLabels()
        for s in samples:
            _annotate(s, labels)
        freq = double_column_frequencies(labels, n_llm, n_total)

        # 按臂细分（5 臂×3 数据集混池不可直接读，必须拆臂解读）
        from collections import defaultdict
        per_arm: dict = defaultdict(lambda: defaultdict(int))
        for s in samples:
            arm = str(s.get("arm", "?"))
            lab = FailureLabels()
            _annotate(s, lab)
            for k in ["F1", "F2", "F3", "F4", "F5", "F6a", "F6b", "F7"]:
                per_arm[arm][k] += len(getattr(lab, k))
        per_arm = {k: dict(v) for k, v in per_arm.items()}

        # 数据缺口如实说明（预测池字段可用性决定哪些 F 类可自动判定）
        limitations = {
            "F1_F2": "预测池无 saf_decision/should_reject 字段，SAF 拦截判定无法从池自动得出，需 E12/SAF 日志或人工标注",
            "F3_F4": "池内 csc_correct 恒为 True（无 CSC 错实例），CSC错→LLM纠错/被带偏需补充真实 CSC 错样本后再判",
            "F6a_F6b": "F6a 主要由 E7 free/cot 臂与 E8 噪声/畸形载荷臂的解析失败构成，E5 guided 臂为 0，解读须按臂拆分",
            "kappa": "需双人标注后计算（方案 E18 硬性要求 κ≥0.7）",
        }
        save_json({"n_total": n_total, "n_entered_llm": n_llm,
                   "double_column_frequencies": freq, "per_arm": per_arm,
                   "kappa": "需双人标注后计算", "limitations": limitations},
                  results_dir("E18") / "E18_failure.json")
        for k, v in freq.items():
            print(f"[E18] {k}: 占LLM={v['frac_of_llm']:.3f} 占总={v['frac_of_total']:.3f} (n={v['count']})")
        for arm, d in per_arm.items():
            print(f"[E18] 臂 {arm}: {d}")
        print(f"[E18] 完成，结果存 results/E18/E18_failure.json")


def _annotate(s: dict, labels: FailureLabels) -> None:
    eid = str(s.get("id", ""))
    # 字段守卫：仅当样本带目标字段时才判该类，避免字段缺失的实验池样本被误分类
    if not s.get("entered_llm", True) and "saf_decision" in s and "should_reject" in s:
        if s.get("saf_decision") == "rejected" and not s.get("should_reject"):
            labels.F1.append(eid)
        if s.get("saf_decision") == "passed" and s.get("should_reject", False):
            labels.F2.append(eid)
        return
    if s.get("llm_parse_failed"):
        labels.F6a.append(eid)
    elif s.get("field_degraded"):
        labels.F6b.append(eid)
    if "hallucination" in s and s.get("hallucination"):
        labels.F7.append(eid)
    if "csc_correct" not in s or "llm_correct" not in s:
        return  # 无 CSC 判定字段的池（如 E8），无法判 F3/F4/F5，不计入
    if not s.get("csc_correct") and s.get("llm_correct"):
        labels.F3.append(eid)
    elif not s.get("csc_correct") and not s.get("llm_correct") and not s.get("llm_parse_failed"):
        labels.F4.append(eid)
    elif s.get("csc_correct") and not s.get("llm_correct") and not s.get("llm_parse_failed"):
        labels.F5.append(eid)


def _load_samples() -> list[dict]:
    """读取**正式**预测池（跳过备份/污染/哈希副本，避免重复计数）。

    - 正式池：E5.jsonl / E7.jsonl / E8.jsonl（各实验最新合并池）
    - 跳过：*_polluted_*、*.bak*、E5-<hash>.jsonl（历史备份，保留不删但不得入统计）
    - E7.jsonl 存在 UTF-8 BOM，用 utf-8-sig 解码
    """
    import json
    out = []
    for f in sorted(PRED_POOL.glob("*.jsonl")):
        name = f.name
        if "_polluted_" in name or ".bak" in name or "E5-" in name:
            print(f"[E18] 跳过备份池文件（不入统计）: {name}")
            continue
        try:
            txt = f.read_text(encoding="utf-8-sig")
        except Exception as e:  # noqa: BLE001
            print(f"[E18] 跳过无法解码的池文件 {name}: {e}")
            continue
        for line in txt.splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


if __name__ == "__main__":
    main()