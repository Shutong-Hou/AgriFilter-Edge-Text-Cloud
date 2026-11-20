"""E7 few-shot 示例选取：K-Means medoid 自动选取 + LOO 敏感性 + 5 随机种子对照。

方案 E7：few-shot-3，K-Means medoid 自动选取 + LOO 敏感性 + 5 随机种子对照。
"""
from __future__ import annotations

import random

import numpy as np


def select_medoids(
    features: np.ndarray,
    labels: list[str],
    k: int = 3,
    seed: int = 42,
) -> tuple[list[int], dict]:
    """每类内 K-Means→medoid 选取 k 个示例。返回 (indices, 元数据)。

    features: (N, D) 嵌入；labels: 对应类标签。
    """
    from sklearn.cluster import KMeans

    rng = np.random.RandomState(seed)
    idx_by_class: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        idx_by_class.setdefault(lab, []).append(i)

    selected, meta = [], {}
    for lab, idxs in idx_by_class.items():
        feats = features[idxs]
        k_eff = min(k, len(idxs))
        km = KMeans(n_clusters=k_eff, n_init=10, random_state=seed).fit(feats)
        for c in range(k_eff):
            members = np.where(km.labels_ == c)[0]
            center = km.cluster_centers_[c]
            # medoid：簇内到中心距离最小的样本
            dists = np.linalg.norm(feats[members] - center, axis=1)
            medoid = members[int(dists.argmin())]
            selected.append(idxs[medoid])
        meta[lab] = {"k": k_eff, "n_examples": len(idxs)}
    return sorted(selected), meta


def loo_sensitivity(
    features: np.ndarray,
    labels: list[str],
    selected: list[int],
) -> dict:
    """留一法敏感性：逐一移除示例，重算 medoid 选择，返回替换率。"""
    replaced = 0
    for rem in selected:
        rem_idxs = [i for i in selected if i != rem]
        # 用剩余 + 补齐同类的可替换候选：简化——仅统计移除后是否仍能选到同类
        class_of = labels[rem]
        candidates = [i for i in range(len(labels)) if labels[i] == class_of and i not in rem_idxs]
        if not candidates:
            continue
        # 若候选集与 medoid 选择不同则认为敏感（此处用特征中心变更近似）
        new_feats = features[rem_idxs + candidates[:1]]
        old_center = features[rem_idxs].mean(axis=0)
        new_center = new_feats.mean(axis=0)
        if np.linalg.norm(old_center - new_center) > 1e-6:
            replaced += 1
    return {"n_selected": len(selected), "n_sensitive": replaced, "sensitivity": replaced / max(1, len(selected))}


def random_seed_contrast(
    features: np.ndarray,
    labels: list[str],
    k: int = 3,
    seeds: list[int] = (1234, 2345, 3456, 4567, 5678),
) -> dict:
    """5 随机种子的对照选择，返回每种子选择的稳定度。"""
    selections = [
        select_medoids(features, labels, k=k, seed=s)[0] for s in seeds
    ]
    # 稳定度：不同种子所选索引的交集比例
    if not selections:
        return {"seeds": seeds, "overlap_ratio": 0.0}
    base = set(selections[0])
    inter = base.intersection(*[set(s) for s in selections[1:]])
    return {
        "seeds": seeds,
        "n_total": len(base),
        "common_over_seeds": len(inter),
        "overlap_ratio": len(inter) / max(1, len(base)),
    }