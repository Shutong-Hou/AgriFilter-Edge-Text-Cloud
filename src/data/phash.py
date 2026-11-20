"""pHash（64-bit）近重复审计（方案 E0a）。

实现：灰度图 → 32x32 → DCT → 取左上 8x8 低频系数(舍去DC) → 二值化 → 64bit 哈希。
连通分量（threshold 阈值下）＝ 拍摄组。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

HASH_SIZE = 8  # 8x8 = 64 bit


def phash(image: Image.Image) -> np.ndarray:
    """返回 64-bit pHash 作为 uint64。"""
    img = image.convert("L").resize((32, 32), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    dct = np.zeros((32, 32), dtype=np.float32)
    # 2D DCT（ll-8 近似）：用 scipy 或 numpy 手写。优先 scipy。
    try:
        from scipy.fftpack import dct, idct  # noqa: F401

        dct = _dct2(arr)
    except ImportError:  # 无 scipy 时退化为简化 DCT（慢，仅用于结构示意）
        dct = _dct2_numpy(arr)
    low = dct[:HASH_SIZE, :HASH_SIZE]
    # 去 DC 后按中位数二值化
    low_nodc = low.ravel()[1:]
    med = np.median(low_nodc)
    bits = (low_nodc > med).astype(np.uint8)
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return np.uint64(val)


def _dct2(a: np.ndarray) -> np.ndarray:
    from scipy.fftpack import dct

    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def _dct2_numpy(a: np.ndarray) -> np.ndarray:
    """纯 numpy DCT（低效，仅 scipy 缺失时兜底）。"""
    n = a.shape[0]
    out = np.zeros_like(a)
    for u in range(n):
        for v in range(n):
            cu = 1.0 if u == 0 else np.sqrt(2.0)
            cv = 1.0 if v == 0 else np.sqrt(2.0)
            s = 0.0
            for x in range(n):
                for y in range(n):
                    s += a[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * n)) * np.cos(
                        (2 * y + 1) * v * np.pi / (2 * n)
                    )
            out[u, v] = (2 / n) * cu * cv * s
    return out


def hamming(a: np.uint64, b: np.uint64) -> int:
    x = int(a) ^ int(b)
    return bin(x).count("1")


def connected_components(
    paths: list[str],
    hashes: list[np.uint64],
    threshold: int,
) -> list[list[str]]:
    """在 hamming<=threshold 图上做连通分量，返回拍摄组列表。

    完备性（鸽巢原理）：把 64 位哈希切到 S 段。若 S > threshold，则任两个
    hamming<=threshold 的哈希至少有一段完全相同（否则 S 段各至少 1 位不同，
    共需 >=S 位 > threshold，矛盾）。故取 S = threshold+1 段即可保证完备捕获
    所有 hamming<=threshold 的近重复对，再对同桶候选做精确 Hamming 校验。

    分桶方式：轮转取位（第 s 段取位 {s, s+S, s+2S, ...}）。pHash 低位（近 DC 的
    低频系数）高度相关，若按连续位切段，最大段（含最多相关位）会形成超大桶，
    候选对爆炸。轮转把相关位摊到各段，桶更均匀，不改完备性。

    提速：桶内成对 Hamming 用 numpy 向量化（XOR + popcount + 阈值掩码），避免
    Python 双层循环；对 t=8/9/10 仍能在可接受时间内完成（纯 CPU，GPU 不参与）。
    """
    import numpy as np
    from collections import defaultdict

    n = len(paths)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 段数 S = threshold+1；每段轮转取位
    S = threshold + 1
    seg_bits: list[list[int]] = [[] for _ in range(S)]
    for bitno in range(64):
        seg_bits[bitno % S].append(bitno)

    h = np.fromiter((int(x) for x in hashes), dtype=np.uint64, count=n)

    # 构建 (段号, 段值) -> 索引列表（轮转取位，向量化）
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for s_id, bpos in enumerate(seg_bits):
        val = np.zeros(n, dtype=np.int64)
        for shift, bitno in enumerate(bpos):
            val |= ((h >> bitno) & np.uint64(1)).astype(np.int64) << shift
        for idx in range(n):
            buckets[(s_id, int(val[idx]))].append(idx)

    # 桶内成对 Hamming 精确校验（numpy 向量化）
    for (_, _s_id), members in buckets.items():
        m = len(members)
        if m < 2:
            continue
        idx = np.array(members, dtype=np.int64)
        col = h[idx]  # (m,) uint64
        # 分块处理，控制峰值内存
        CHUNK = 4000
        for a0 in range(0, m, CHUNK):
            a1 = min(m, a0 + CHUNK)
            sub = col[a0:a1]  # (chunk,) uint64
            xor = sub[:, None] ^ col[None, :]  # (chunk, m) uint64
            dist = _popcount_u64(np.ascontiguousarray(xor.ravel())).reshape(a1 - a0, m)
            for ra in range(a1 - a0):
                a = a0 + ra
                row = dist[ra]
                cand = np.nonzero(row[a + 1 : m] <= threshold)[0]
                ai = idx[a]
                for c in cand:
                    union(ai, idx[a + 1 + c])

    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(paths[i])
    return [v for v in groups.values()]


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """uint64 数组逐元素 popcount（兼容 numpy 1.x / 2.x）。"""
    import numpy as np

    if hasattr(np, "bitwise_count"):  # numpy>=2.0
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(-1, 8)  # 逐字节 unpack，popcount 与端序无关
    return np.unpackbits(b, axis=1).sum(axis=1)


def audit_dataset(
    paths: list[str],
    thresholds: list[int] = (3, 5, 10),
    hash_cache: dict[str, np.uint64] | None = None,
) -> dict:
    """对一批图像路径做 pHash 审计，返回各阈值下的连通分量（拍摄组）统计。"""
    import tqdm

    hashes = {}
    for p in tqdm.tqdm(paths, desc="phash"):
        if hash_cache is not None and p in hash_cache:
            hashes[p] = hash_cache[p]
            continue
        try:
            with Image.open(p) as im:
                hashes[p] = phash(im)
            if hash_cache is not None:
                hash_cache[p] = hashes[p]
        except Exception:
            hashes[p] = np.uint64(0)
    result = {}
    for t in thresholds:
        groups = connected_components(paths, [hashes[p] for p in paths], threshold=t)
        result[int(t)] = {
            "n_groups": len(groups),
            "n_singleton": sum(1 for g in groups if len(g) == 1),
            "max_group_size": max(len(g) for g in groups) if groups else 0,
            "mean_group_size": float(np.mean([len(g) for g in groups])) if groups else 0.0,
            "n_duplicate_pairs": sum(
                len(g) * (len(g) - 1) // 2 for g in groups if len(g) > 1
            ),
            "groups": [g for g in groups if len(g) > 1],
            # 完整组表（含单图组）：供组感知划分使用，确保每个样本都有确定组 id
            # （报告中只列多图组以控制体积，划分必须用完整表避免惰性分配不一致）
            "_all_groups": groups,
        }
    result["_hash_cache"] = hash_cache
    return result