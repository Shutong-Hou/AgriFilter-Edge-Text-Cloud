"""组感知划分（方案 E0b/E0d）与防泄露单测。

- E0b：拍摄组级 80/20（seed=42），整组不可拆。
- E0d：训练集内划 10% dev（组感知同步，seed=7）。
- 单测：跨集路径交集为空 + 跨集无 Hamming≤5 近重复对（红线5）。
- 旧随机划分 vs 新划分的泄露量化（E0c）由调用方在 E0 脚本完成。
"""
from __future__ import annotations

import random
from pathlib import Path

from ..utils.seeds import set_seed
from ..utils.io import save_json


def _group_assign_group_id(groups: list[list[str]]) -> dict[str, int]:
    """给每个样本分配组 id。groups 为连通分量（拍摄组）。"""
    gid: dict[str, int] = {}
    for i, g in enumerate(groups):
        for p in g:
            gid[p] = i
    return gid


def _gid_of(gid: dict[str, int], p: str, next_id: list) -> int:
    """取样本组 id；若样本不在任何组（外部传入的 groups 漏了单图组），
    视为独立组并分配新 id（单图即一组）。【辅助补充，保证划分不漏样本】"""
    if p in gid:
        return gid[p]
    gid[p] = next_id[0]
    next_id[0] += 1
    return gid[p]


def group_aware_split(
    sample_paths: list[str],
    groups: list[list[str]],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> dict[str, list[str]]:
    """拍摄组级 80/20 划分。整组落入 train 或 eval，组内不拆分。

    返回 {"train": [...], "eval": [...]}。
    """
    set_seed(seed)
    gid = _group_assign_group_id(groups)
    next_id = [len(set(gid.values()))]
    group_ids = sorted({_gid_of(gid, p, next_id) for p in sample_paths})
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_train_groups = int(round(len(group_ids) * train_fraction))
    train_gids = set(group_ids[:n_train_groups])
    train = [p for p in sample_paths if _gid_of(gid, p, next_id) in train_gids]
    eval_ = [p for p in sample_paths if _gid_of(gid, p, next_id) not in train_gids]
    return {"train": sorted(train), "eval": sorted(eval_)}


def split_dev_from_train(
    train_paths: list[str],
    groups: list[list[str]],
    dev_fraction: float = 0.1,
    seed: int = 7,
) -> dict[str, list[str]]:
    """E0d：训练集内组感知同步划 10% 为 dev（供早停/调参/τ*选择/few-shot取样）。

    注意：返回的 train 是"从原 train 切出 dev 后的剩余"，调用方必须用它作为训练集，
    否则 dev 会与原 train 重叠（违反防泄露）。
    """
    gid = _group_assign_group_id(groups)
    # 传入的 groups 必须是完整组表；若某样本缺失组 id，视为独立组（单图一组）
    next_id = [len(set(gid.values()))]
    present = [p for p in train_paths if p in gid] or train_paths
    group_ids = sorted({_gid_of(gid, p, next_id) for p in present})
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_dev = max(1, int(round(len(group_ids) * dev_fraction)))
    dev_gids = set(group_ids[:n_dev])
    dev = [p for p in present if _gid_of(gid, p, next_id) in dev_gids]
    train = [p for p in present if _gid_of(gid, p, next_id) not in dev_gids]
    return {"train": sorted(train), "dev": sorted(dev)}


def check_no_path_overlap(splits: dict[str, list[str]]) -> bool:
    """跨集路径交集为空。"""
    paths = list(splits.keys())
    for i in range(len(paths)):
        a = set(splits[paths[i]])
        for j in range(i + 1, len(paths)):
            if a & set(splits[paths[j]]):
                return False
    return True


def check_no_cross_near_dup(
    splits: dict[str, list[str]],
    hashes: dict[str, int],
    hamming_max: int = 5,
) -> bool:
    """跨集无 Hamming≤hamming_max 近重复对。

    numpy 分块向量化：对每对跨集集合，按块做 XOR+popcount+阈值掩码，
    避免 O(n²) 纯 Python 循环（否则 train×eval 约 6 亿次会跑数分钟）。
    """
    import numpy as np

    from ..data.phash import _popcount_u64

    keys = list(splits.keys())
    arrs = {
        k: np.fromiter(
            (int(hashes[p]) for p in splits[k] if p in hashes),
            dtype=np.uint64,
        )
        for k in keys
    }
    CHUNK = 2000
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_h = arrs[keys[i]]
            b_h = arrs[keys[j]]
            for a0 in range(0, len(a_h), CHUNK):
                a1 = min(len(a_h), a0 + CHUNK)
                sub = a_h[a0:a1]
                xor = np.bitwise_xor(sub[:, None], b_h[None, :])  # (chunk, nb)
                dist = _popcount_u64(np.ascontiguousarray(xor.ravel())).reshape(xor.shape)
                if np.any(dist <= hamming_max):
                    return False
    return True