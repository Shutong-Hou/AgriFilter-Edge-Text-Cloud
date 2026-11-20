"""训练 CSC（MobileNetV3-Small + Linear→38）。3 种子。标签平滑 0.1。

用法: python -m src.experiments.train_csc
产出: results/train_csc/csc_seed{42,43,44}.pt
"""
from __future__ import annotations

import shutil
from pathlib import Path
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path
from ..utils.seeds import experiment_seeds
from ..data.datasets import PlantDataset, default_train_transform, default_eval_transform
from ..models.csc import CSC
from ..training.trainer import train
from .common import load_partition, load_pv_classes, results_dir

CFG = load_yaml(project_path("configs", "model_config.yaml"))["csc"]
DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    with ExperimentContext(experiment="train_csc", description="CSC 训练 3 种子", configs=CFG) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        train_ds = PlantDataset(root, transform=default_train_transform(),
                                sample_paths=[Path(p) for p in part["train"]],
                                class_to_idx=class_to_idx)
        dev_ds = PlantDataset(root, transform=default_eval_transform(),
                              sample_paths=[Path(p) for p in part["dev"]],
                              class_to_idx=class_to_idx)
        train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=4)
        dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=4)
        out = results_dir("train_csc")
        for seed in experiment_seeds():
            ckpt = out / f"csc_seed{seed}.pt"
            if ckpt.exists():
                print(f"[train_csc] seed={seed} 已存在，跳过")
                continue
            model = CSC(n_classes=len(classes), label_smoothing=CFG["label_smoothing"])
            train(model, train_loader, dev_loader, head_type="ce",
                  epochs=CFG["epochs"],
                  head_lr=CFG["optimizer"]["head_lr"],
                  backbone_lr=CFG["optimizer"]["backbone_lr"],
                  weight_decay=CFG["optimizer"]["weight_decay"],
                  seed=seed, label_smoothing=CFG["label_smoothing"],
                  out_dir=out)
            # 统一检查点命名：best_seed→csc_seed（与 common.load_csc_for_exp / E2 读取口径一致）
            best = out / f"best_seed{seed}.pt"
            if best.exists():
                shutil.copy(best, out / f"csc_seed{seed}.pt")
            print(f"[train_csc] seed={seed} 完成")


if __name__ == "__main__":
    main()