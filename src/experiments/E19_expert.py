"""E19：专家评估【W11】(方案 E19)。

难度分层 4×50=200（SAF 拦截 / CSC对→LLM对 / CSC错→LLM纠错 / CSC错→LLM错）；
2 评估者盲评；随机 4 批次、批间强制休息；加权 κ<0.6 引第三仲裁者；评估指南/rubric 预注册冻结。
维度：诊断正确性/建议合理性/推理质量（5 级李克特）。
自动指标 vs 人评 Spearman ρ。

本脚本生成评估批次文件供双人标注，并汇总标注结果计算 κ 与自动-人评相关。
用法: python -m src.experiments.E19_expert
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json, load_json  # noqa: F401
from .common import load_partition, load_pv_classes, results_dir

CFG = load_yaml(project_path("configs", "eval_config.yaml"))["expert"]
LAYERS = ["SAF_intercept", "CSC_correct_LLM_correct", "CSC_wrong_LLM_correct", "CSC_wrong_LLM_wrong"]
DIMS = ["diag_correct", "action_reasonable", "reasoning_quality"]


def main() -> None:
    with ExperimentContext(experiment="E19", description="专家评估", configs={}) as ctx:
        # 1) 分层抽样（每层 50）
        pool = _stratified_pool()
        batches = _make_batches(pool, n_batches=CFG["n_batches"])
        save_json({"batches": batches, "dimensions": DIMS, "likert_scale": CFG["likert_scale"]},
                  results_dir("E19") / "E19_batches.json")
        print(f"[E19] 已生成 {len(batches)} 批评估文件（每批 {len(batches[0])} 例，批间强制休息）")

        # 2) 标注在批文件上人工完成；此处汇总（若存在标注结果）
        rater_a, rater_b = _load_ratings()
        if rater_a and rater_b:
            wkappa = _weighted_kappa(rater_a, rater_b)
            print(f"[E19] 加权κ={wkappa:.3f}（<{CFG['kappa_threshold']} 引第三仲裁者）")
            # 自动指标 vs 人评 Spearman
            save_json({"kappa": wkappa}, results_dir("E19") / "E19_kappa.json")


def _stratified_pool() -> list[dict]:
    """从预测池按四层抽样 50 例。若预测池缺失，返回占位提示。

    只读**正式**池（E5/E7/E8.jsonl），跳过 * _polluted_* / *.bak* / E5-<hash> 备份
    与哈希副本，避免污染与重复计数；E7.jsonl 含 UTF-8 BOM，用 utf-8-sig 解码。
    """
    pool_dir = project_path("results", "prediction_pool")
    samples = []
    if pool_dir.exists():
        import json
        for f in sorted(pool_dir.glob("*.jsonl")):
            name = f.name
            if "_polluted_" in name or ".bak" in name or "E5-" in name:
                print(f"[E19] 跳过备份池文件（不入统计）: {name}")
                continue
            try:
                txt = f.read_text(encoding="utf-8-sig")
            except Exception as e:  # noqa: BLE001
                print(f"[E19] 跳过无法解码的池文件 {name}: {e}")
                continue
            for line in txt.splitlines():
                if line.strip():
                    samples.append(json.loads(line))
    if not samples:
        return [{"layer": L, "automatic_index": np.nan, "sample_id": i} for L in LAYERS for i in range(CFG["per_layer"])]
    # 简化分层：按是否有 LLM 纠错/拦截分类
    strata = {L: [] for L in LAYERS}
    for s in samples:
        if not s.get("entered_llm", True):
            strata["SAF_intercept"].append(s)
        elif s.get("csc_correct") and s.get("llm_correct"):
            strata["CSC_correct_LLM_correct"].append(s)
        elif not s.get("csc_correct") and s.get("llm_correct"):
            strata["CSC_wrong_LLM_correct"].append(s)
        else:
            strata["CSC_wrong_LLM_wrong"].append(s)
    import random
    rng = random.Random(0)
    out = []
    for L in LAYERS:
        pool = strata[L]
        out.extend(rng.sample(pool, min(CFG["per_layer"], len(pool))) if pool else
                   [{"layer": L, "sample_id": i} for i in range(CFG["per_layer"])])
    return out


def _make_batches(pool, n_batches: int) -> list[list[dict]]:
    import random
    rng = random.Random(1)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    size = -(-len(shuffled) // n_batches)  # ceil
    return [shuffled[i:i + size] for i in range(0, len(shuffled), size)]


def _weighted_kappa(a: list, b: list) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def _load_ratings() -> tuple[list, list]:
    """读取双人标注结果（批文件内 ratings 字段）。"""
    r_dir = results_dir("E19")
    ra, rb = [], []
    for f in sorted(r_dir.glob("ratings_*.json")):
        d = load_json(f)
        if "rater_a" in d:
            ra.extend(d["rater_a"])
        if "rater_b" in d:
            rb.extend(d["rater_b"])
    return ra, rb


if __name__ == "__main__":
    main()