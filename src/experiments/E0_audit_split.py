"""E0：泄露审计与组感知划分【W1】(方案 2.2/2.3)。

产出:
- dataset cards（D1/D2/D3）
- pHash 近重复审计（阈值 {3,5,10}，连通分量=拍摄组）
- 拍摄组级 80/20 划分（seed=42）
- 训练集内 10% dev（组感知同步，seed=7）
- 防泄露单测（跨集交集为空 + 跨集无 Hamming≤5 近重复对）
- 发布 partition.json / pv_classes.json / phash_cache.json

用法: python -m src.experiments.E0
"""
from __future__ import annotations

import sys
from pathlib import Path

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..data.datasets import (
    build_dataset_card,
    save_dataset_card,
    index_images,
)
from ..data.phash import audit_dataset
from ..data.split import (
    group_aware_split,
    split_dev_from_train,
    check_no_path_overlap,
    check_no_cross_near_dup,
)

CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    with ExperimentContext(
        experiment="E0",
        description="泄露审计与组感知划分",
        configs=CFG,
    ) as ctx:
        data_root = Path(CFG["data_root"])
        out = results_dir = project_path("results", "E0")
        out.mkdir(parents=True, exist_ok=True)

        # 1) dataset cards
        cards = {}
        for name, meta in CFG["datasets"].items():
            card = build_dataset_card(
                name,
                Path(meta["dir"]),
                license=meta.get("license", ""),
                download_date="TBD",
            )
            save_dataset_card(card, results_dir)
            cards[name] = card.to_dict()
            print(f"[E0] {name}: {card.n_images} 图 / {card.n_classes} 类 / 健康类={card.has_healthy}")

        # 2) pHash 审计（D1 全量）
        pv_dir = Path(CFG["datasets"]["plantvillage"]["dir"])
        idx = index_images(pv_dir)
        pv_paths = [str(p) for v in idx.values() for p in v]
        print(f"[E0] PlantVillage 全量 {len(pv_paths)} 张，开始 pHash 审计...")
        phash_cache: dict = {}
        audit = audit_dataset(
            pv_paths,
            thresholds=CFG["partition"]["phash_hamming_thresholds"],
            hash_cache=phash_cache,  # 【辅助补充】收集哈希供防泄露单测，避免空跑
        )

        # 3) 组感知划分（seed=42）
        from ..utils.seeds import get_seeds

        groups = audit[5]["_all_groups"]  # 完整组表（含单图组），划分必须用完整表
        split = group_aware_split(
            pv_paths, groups,
            train_fraction=CFG["partition"]["train_fraction"],
            seed=get_seeds()["group_split_seed"],
        )
        dev = split_dev_from_train(
            split["train"], groups,
            dev_fraction=CFG["partition"]["dev_fraction"],
            seed=get_seeds()["dev_split_seed"],
        )

        # 4) 防泄露单测（红线5）
        splits = {"train": dev["train"], "eval": split["eval"], "dev": dev["dev"]}
        overlap_ok = check_no_path_overlap(splits)
        hashes = {p: int(phash_cache[p]) for p in pv_paths if p in phash_cache}
        near_dup_ok = check_no_cross_near_dup(splits, hashes, hamming_max=5)
        tests = {
            "no_path_overlap": overlap_ok,
            "no_cross_near_dup_le5": near_dup_ok,
            "all_green": overlap_ok and near_dup_ok,
        }
        print(f"[E0] 防泄露单测: 路径交集空={overlap_ok}, 跨集近重复≤5={near_dup_ok}")

        # 5) dataset card 落盘 + 划分结果保存
        pv_classes = sorted(idx.keys())
        save_json(pv_classes, project_path("results", "E0", "pv_classes.json"))
        save_json(
            {
                "train": dev["train"], "eval": split["eval"], "dev": dev["dev"],
                "group_assignments": {str(i): g for i, g in enumerate(groups)},
                "n_train": len(dev["train"]), "n_eval": len(split["eval"]),
                "n_dev": len(dev["dev"]),
                # 报告仅存各阈值统计量（去掉完整组表避免 JSON 冗余膨胀；
                # 完整组表只对 t=5 划分需要，已单独存为 group_assignments）
                "phash_audit": {
                    k: {kk: vv for kk, vv in v.items() if kk != "_all_groups"}
                    for k, v in audit.items()
                    if k != "_hash_cache"
                },
                "anti_leak_tests": tests,
                "dataset_cards": cards,
            },
            project_path("results", "E0", "partition.json"),
        )
        save_json({str(k): int(v) for k, v in phash_cache.items()},
                  project_path("results", "E0", "phash_cache.json"))
        print(f"[E0] 完成。划分已保存至 results/E0/partition.json")


if __name__ == "__main__":
    main()