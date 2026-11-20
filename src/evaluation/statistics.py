"""统计方法学工具箱（方案第四部分）。

- McNemar 精确检验
- 配对 Bootstrap Δ95%CI（默认 10,000 次）
- Top-3 配对 Bootstrap / 等级分配对 Wilcoxon（scipy）
- 比例 Wilson CI
- 零事件强制 Clopper-Pearson 单侧 95% 界（≈3/n）
- Holm 族内校正
- ECE（10 bins）/ AUROC（confidence vs 正确性，B.9）
"""
from __future__ import annotations

import numpy as np

from ..utils.io import load_yaml, project_path

CFG = load_yaml(project_path("configs", "eval_config.yaml"))["statistics"]


# ---------- 比例 CI ----------
def wilson_ci(k: int, n: int, alpha: float | None = None) -> tuple[float, float]:
    """Wilson score interval for proportion k/n. 返回 (lo, hi)。"""
    alpha = alpha or CFG["wilson_alpha"]
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n if n else 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def clopper_pearson_upper(k: int, n: int, alpha: float | None = None, one_sided: bool = True) -> float:
    """零事件时 Clopper-Pearson 单侧 95% 上界（红线10）。k=0 时 ≈3/n。"""
    from scipy.stats import beta

    alpha = alpha or CFG["cp_alpha"]
    if one_sided:
        # 单侧上界：P(X<=k) = alpha → 上界 = beta.ppf(1-alpha, k+1, n-k)
        return float(beta.ppf(1 - alpha, k + 1, n - k))
    return float(beta.ppf(1 - alpha / 2, k + 1, n - k))


def cp_zero_bound(n: int) -> float:
    """k=0 时的 Clopper-Pearson 单侧 95% 上界（≈3/n）。"""
    return clopper_pearson_upper(0, n)


# ---------- 配对检验 ----------
def mcnemar(a: int, b: int):
    """McNemar 精确检验（二项）。a=臂1对臂2错，b=臂1错臂2对。返回 p 值。"""
    from scipy.stats import binom

    d = a + b
    if d == 0:
        return 1.0
    smaller = min(a, b)
    p = 2.0 * binom.cdf(smaller, d, 0.5)  # 双侧
    return float(min(1.0, p))


def paired_bootstrap_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_iter: int | None = None,
    seed: int = 0,
) -> dict:
    """配对 Bootstrap 的 Δ95%CI（A-B）。scores 为逐样本 0/1 正确性（或得分）。"""
    n_iter = n_iter or CFG["bootstrap_iterations"]
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    assert n == len(scores_b)
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        da = scores_a[idx].mean()
        db = scores_b[idx].mean()
        deltas[i] = da - db
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": float(scores_a.mean() - scores_b.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_iter": n_iter,
    }


def wilcoxon_paired(a: np.ndarray, b: np.ndarray):
    """配对 Wilcoxon（等级分）。返回 (statistic, p)。a-b 差分。"""
    from scipy.stats import wilcoxon

    try:
        res = wilcoxon(a, b)
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return None, None


def holm_correct(pvals: list[float], alpha: float | None = None) -> list[float]:
    """Holm 族内校正。返回校正后 p 值。

    标准实现：排序 → 乘因子(m+1-rank) → 在排序后序列上从前到后保持非递减
    → 映射回原序。修正历史版本在原始顺序上做 max 导致的数值错误。
    """
    import numpy as np

    alpha = alpha or CFG["holm_alpha"]
    p = np.array(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj = sorted_p * (m + 1 - (np.arange(m) + 1))  # rank=i+1, 因子=m-i
    adj = np.maximum.accumulate(adj)              # 排序后从前到后保持非递减
    corrected = np.empty_like(p)
    corrected[order] = adj
    return np.clip(corrected, 0.0, 1.0).tolist()


# ---------- 校准（B.9） ----------
def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int | None = None) -> float:
    """Expected Calibration Error（10 bins）。"""
    n_bins = n_bins or CFG["ece_bins"]
    conf_bins = np.clip(confidences, 0, 1)
    bin_ids = np.clip(np.floor(conf_bins * n_bins).astype(int), 0, n_bins - 1)
    ece_v = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = correct[mask].mean()
        conf = conf_bins[mask].mean()
        ece_v += (mask.sum() / len(conf_bins)) * abs(acc - conf)
    return float(ece_v)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC（scores 越高越接近正类）。"""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman 秩相关。返回 (rho, p)。"""
    from scipy.stats import spearmanr

    rho, p = spearmanr(a, b)
    return float(rho), float(p)