"""跨域标签映射与路由逻辑（方案 2.3 / B.6）。

- load_label_mapping：读 configs/label_mapping.csv（# 开头为注释）。
- 路由：u > τ* → 拒识；u ≤ τ* → 映射进 PV 类走正常流水线。
- 三指标分拆：已知类映射准确率 / 未知类正确拒识率 / 未知类误收率。
- 先验匹配：按 PV 训练集健康/病害比例对 D2/D3 分层重采样，报 n_eff。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.io import project_path


def load_label_mapping(path: Path | str | None = None) -> pd.DataFrame:
    path = path or project_path("configs", "label_mapping.csv")
    df = pd.read_csv(path, comment="#", dtype=str)
    df = df.dropna(subset=["source_class", "pv_class"])
    return df


def mapping_dict(df: pd.DataFrame) -> dict[str, str]:
    """{domain__source_class: pv_class}。"""
    return {
        f"{r['domain'].strip()}__{r['source_class'].strip()}": r["pv_class"].strip()
        for _, r in df.iterrows()
    }


def route_decision(u: float, tau_star: float) -> bool:
    """返回 True=拒识转人工（u>τ*）；False=进映射流水线（u≤τ*）。"""
    return u > tau_star


def prior_matching_resample(
    domain_samples: list[dict],
    pv_healthy_ratio: float,
    seed: int = 42,
) -> tuple[list[dict], float]:
    """按 PV 训练集健康/病害比例对跨域样本分层重采样，返回 (重采样集, n_eff)。

    domain_samples: 每项含 {"path", "is_healthy": bool, ...}。
    n_eff = 重采样后健康数/病害数 对齐 pv 比例（简单实现：按比例取整）。
    """
    import random

    rng = random.Random(seed)
    healthy = [s for s in domain_samples if s["is_healthy"]]
    diseased = [s for s in domain_samples if not s["is_healthy"]]
    # pv 比例：健康/病害
    if healthy and diseased and pv_healthy_ratio > 0:
        # 目标：n_healthy / n_diseased ≈ pv_healthy/(1-pv_healthy)
        n_d = len(diseased)
        n_h_target = int(round(n_d * pv_healthy_ratio / (1 - pv_healthy_ratio)))
        n_h = min(len(healthy), max(1, n_h_target))
    else:
        n_h, n_d = len(healthy), len(diseased)
    h_s = rng.sample(healthy, n_h) if n_h <= len(healthy) else healthy
    out = h_s + rng.sample(diseased, n_d) if n_d <= len(diseased) else h_s + diseased
    n_eff = len(out)
    return out, n_eff