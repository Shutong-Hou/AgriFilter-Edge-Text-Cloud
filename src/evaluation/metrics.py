"""分类与系统指标（E1/E2/E5/E9 等）。

- 二分类：过滤率/漏检率（+Wilson CI 或 CP 上界）、AUROC、macro-F1、每类分解。
- 多分类：Top-1/3/5 Acc、macro-F1、38×38 混淆矩阵。
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

from .statistics import wilson_ci, cp_zero_bound


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> dict:
    """二分类（SAF 过滤）。过滤率=预测为正(需过滤/病害)比例；漏检率=真正例被漏检。"""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    tp = int(((yt == pos_label) & (yp == pos_label)).sum())
    fn = int(((yt == pos_label) & (yp != pos_label)).sum())
    fp = int(((yt != pos_label) & (yp == pos_label)).sum())
    tn = int(((yt != pos_label) & (yp != pos_label)).sum())
    filter_rate = (tp + fp) / n if n else 0.0
    # 漏检率：正类中被判为负的比例（= 1 - 召回率）
    miss_rate = fn / max(1, tp + fn)
    # Wilson CI / CP 上界
    fr_ci = wilson_ci(tp + fp, n)
    mr_ci = wilson_ci(fn, tp + fn) if (tp + fn) else (0.0, cp_zero_bound(1))
    auroc = None
    return {
        "n": n,
        "filter_rate": filter_rate,
        "filter_rate_ci": fr_ci,
        "miss_rate": miss_rate,
        "miss_rate_ci": mr_ci,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": (tp / max(1, tp + fp)),
        "recall": (tp / max(1, tp + fn)),
        "f1": f1_score(yt, yp, pos_label=pos_label, zero_division=0),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred_topk: np.ndarray, n_classes: int) -> dict:
    """多分类 Top-1/3/5 + macro-F1 + 混淆矩阵。

    y_pred_topk: (N, K) 每行按概率降序的类别索引。
    """
    yt = np.asarray(y_true)
    top1 = y_pred_topk[:, 0]
    acc1 = accuracy_score(yt, top1)
    acc3 = float(np.mean([yt[i] in y_pred_topk[i, :3] for i in range(len(yt))]))
    k5 = min(5, y_pred_topk.shape[1])
    acc5 = float(np.mean([yt[i] in y_pred_topk[i, :k5] for i in range(len(yt))]))
    f1 = f1_score(yt, top1, average="macro", zero_division=0)
    cm = confusion_matrix(yt, top1, labels=list(range(n_classes)))
    return {
        "n": len(yt),
        "top1": float(acc1),
        "top3": acc3,
        "top5": acc5,
        "macro_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": [
            float(f1_score(yt, top1, average="macro", labels=[c], zero_division=0))
            for c in range(n_classes)
        ],
    }


def auroc_from_scores(scores: np.ndarray, y_true: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y_true), np.asarray(scores)))


def fpr_at_tpr(scores: np.ndarray, y_true: np.ndarray, tpr_target: float = 0.95) -> dict:
    """FPR@95%TPR（OOD/ID 判别，E3+）。"""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(scores))
    idx = np.argmin(np.abs(tpr - tpr_target))
    return {"fpr_at_tpr": float(fpr[idx]), "tpr_actual": float(tpr[idx])}


def risk_coverage_aurc(confidences: np.ndarray, correct: np.ndarray) -> dict:
    """risk-coverage 曲线 + AURC（E3+）。置信度降序覆盖，越靠前越可信。

    coverage = 已覆盖样本占比；risk = 1 - 已覆盖样本准确率。
    AURC = 曲线下面积（越低越好，风险随覆盖增长越慢）。
    """
    conf = np.asarray(confidences)
    corr = np.asarray(correct).astype(float)
    order = np.argsort(-conf)  # 置信度降序
    corr_s = corr[order]
    n = len(corr_s)
    cum = np.cumsum(corr_s)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cum / np.arange(1, n + 1)
    aurc = float(np.trapz(risk, coverage))
    return {"aurc": aurc, "coverage": coverage.tolist(), "risk": risk.tolist()}