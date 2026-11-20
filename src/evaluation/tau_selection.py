"""τ* 冻结协议（方案第三部分）。dev 集扫描 τ∈[0,1]（步长 0.01），
按"漏检率≤1% 选最大 τ"冻结（最小化过滤率，保证不漏检至少 99%）。

【偏离说明·经用户批准】方案原文写"选最小 τ"，但 EDL 的 u=K/S∈(0,1] 恒>0，
"选最小"恒得 τ*=0→过滤率=100%（退化）。经用户确认改为"满足漏检率≤1% 的最大 τ"，
得到有区分度的工作点。红线12：τ* 只能由 dev 集产生并全局冻结；验证/跨域集只应用不调优；
报 τ*±0.05 领域敏感性。
"""
from __future__ import annotations

import numpy as np

from ..utils.io import load_yaml, project_path

CFG = load_yaml(project_path("configs", "eval_config.yaml"))["tau"]


def select_tau(
    dev_u: np.ndarray,
    dev_y: np.ndarray,   # 二分类标签：1=病害(需过滤)，0=健康
    miss_rate_cap: float | None = None,
    step: float | None = None,
) -> dict:
    """按漏检率≤cap 选最大 τ 冻结。

    漏检率 = 预测为健康(u≤τ)但实际为病害 的比例（在病害样本上计算）。
    返回 {"tau_star", "miss_rate_at_tau", "working_curve"}。

    选"最大 τ"：在保证漏检率≤cap 的前提下尽可能少过滤（τ 越大被判病害越少）。
    """
    cap = miss_rate_cap if miss_rate_cap is not None else CFG["miss_rate_cap"]
    step = step if step is not None else CFG["search_step"]
    u = np.asarray(dev_u)
    y = np.asarray(dev_y)
    curve = []
    tau_star = None
    for tau in np.arange(0.0, 1.0 + 1e-9, step):
        pred_diseased = u > tau          # u>τ 判为病害(需过滤)
        # 漏检：实际病害但被判为健康
        miss_idx = (y == 1) & (~pred_diseased)
        miss_rate = miss_idx.sum() / max(1, (y == 1).sum())
        curve.append({"tau": float(tau), "miss_rate": float(miss_rate)})
        if miss_rate <= cap:
            tau_star = float(tau)  # 保留最后一个（最大）满足约束的 τ
    if tau_star is None:
        tau_star = 1.0  # 无法满足，取上限（如实报）
    return {
        "tau_star": tau_star,
        "miss_rate_cap": cap,
        "miss_rate_at_tau_star": next(c["miss_rate"] for c in curve if abs(c["tau"] - tau_star) < step / 2),
        "working_curve": curve,
    }


def apply_tau(u: np.ndarray, tau_star: float) -> np.ndarray:
    """应用冻结 τ*：返回是否需要过滤（u>τ* → True）。"""
    return np.asarray(u) > tau_star


def sensitivity_neighborhood(u: np.ndarray, y: np.ndarray, tau_star: float, window: float | None = None) -> dict:
    """τ*±window 邻域敏感性。"""
    w = window if window is not None else CFG["sensitivity_window"]
    out = {}
    for tau in [tau_star - w, tau_star, tau_star + w]:
        pred = apply_tau(u, tau)
        miss = ((y == 1) & (~pred)).sum() / max(1, (y == 1).sum())
        out[round(float(tau), 3)] = float(miss)
    return out