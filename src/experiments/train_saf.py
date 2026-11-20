"""训练 SAF（MobileNetV3-Small + EDL 头，K=2）。3 种子 (42/43/44)。

用法: python -m src.experiments.train_saf
产出: results/train_saf/saf_seed{42,43,44}.pt + train_log
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path
from ..utils.seeds import experiment_seeds
from ..data.datasets import (
    PlantDataset, default_train_transform, default_eval_transform,
)
from ..models.saf import SAF
from ..training.trainer import train
from .common import load_partition, results_dir

CFG = load_yaml(project_path("configs", "model_config.yaml"))["saf"]
DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    with ExperimentContext(experiment="train_saf", description="SAF 训练 3 种子", configs=CFG) as ctx:
        part = load_partition()
        # 直接复用 E0 组感知划分的 train/dev（不再重复划分）
        train_paths = part["train"]
        dev = part["dev"]
        # SAF 二分类标签：类含 healthy 为 0，否则 1
        class_to_idx = {"healthy": 0, "diseased": 1}
        train_ds = PlantDataset(
            Path(DATA_CFG["datasets"]["plantvillage"]["dir"]),
            transform=default_train_transform(),
            sample_paths=[Path(p) for p in train_paths],
            class_to_idx=class_to_idx,
        )
        # 需按病理重建二分类标签：覆盖 targets
        train_ds.targets = [_binary_label(p) for p in train_ds.samples]
        dev_ds = PlantDataset(
            DATA_CFG["datasets"]["plantvillage"]["dir"],
            transform=default_eval_transform(),
            sample_paths=[Path(p) for p in dev],
            class_to_idx=class_to_idx,
        )
        dev_ds.targets = [_binary_label(p) for p in dev_ds.samples]
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=4)
        dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=4)

        out = results_dir("train_saf")
        for seed in experiment_seeds():
            ckpt = out / f"saf_seed{seed}.pt"
            if ckpt.exists():
                print(f"[train_saf] seed={seed} 已存在，跳过")
                continue
            model = SAF(n_classes=2)
            train(
                model, train_loader, dev_loader, head_type="edl",
                epochs=CFG["epochs"],
                head_lr=CFG["optimizer"]["head_lr"],
                backbone_lr=CFG["optimizer"]["backbone_lr"],
                weight_decay=CFG["optimizer"]["weight_decay"],
                seed=seed,
                kl_anneal_step=CFG["kl_anneal"]["step"],
                out_dir=out,
            )
            # 统一检查点命名：best_seed→saf_seed（与 common.load_saf_for_exp / E1 读取口径一致）
            best = out / f"best_seed{seed}.pt"
            if best.exists():
                shutil.copy(best, out / f"saf_seed{seed}.pt")
            print(f"[train_saf] seed={seed} 完成")


def _binary_label(p) -> int:
    """p 为 Path 或 str；健康→0，病害→1。"""
    return 0 if "healthy" in str(p).lower() else 1


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(project_path("src")))
    main()