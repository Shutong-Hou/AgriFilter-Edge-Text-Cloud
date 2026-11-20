"""E3+：分类头消融训练（固定骨干，H1 CE / H2 CE+标签平滑 / H3 EDL 头）。3 种子。

方案 E3+：固定骨干，比较三种头。评估在 E3plus_eval 中完成（二分类 + OOD + 校准 + risk-coverage）。

用法: python -m src.experiments.train_head_ablation
产出: results/train_head_ablation/head_{H1|H2|H3}_seed{42,43,44}.pt
"""
from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path
from ..utils.seeds import experiment_seeds
from ..data.datasets import PlantDataset, default_train_transform, default_eval_transform
from ..models.backbone import build_backbone, backbone_dim
from ..models.heads import build_head
from ..training.trainer import train
from .common import load_partition, load_pv_classes, results_dir

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
MODEL_CFG = load_yaml(project_path("configs", "model_config.yaml"))
HEADS = {"H1": "ce", "H2": "ce_ls", "H3": "edl"}


def main() -> None:
    with ExperimentContext(experiment="train_head_ablation",
                           description="E3+ 分类头消融训练", configs=HEADS) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        n_cls = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        train_ds = PlantDataset(root, transform=default_train_transform(),
                                sample_paths=[Path(p) for p in part["train"]], class_to_idx=class_to_idx)
        dev_ds = PlantDataset(root, transform=default_eval_transform(),
                              sample_paths=[Path(p) for p in part["dev"]], class_to_idx=class_to_idx)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
        dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=4)
        out = results_dir("train_head_ablation")

        for hname, htype in HEADS.items():
            for seed in experiment_seeds():
                backbone = build_backbone("mobilenet_v3_small", pretrained=True)
                ls = 0.0 if hname == "H1" else MODEL_CFG["csc"]["label_smoothing"]
                head = build_head(htype, backbone_dim("mobilenet_v3_small"), n_cls, label_smoothing=ls)
                model = _attach(backbone, head)
                train(model, train_loader, dev_loader, head_type=htype,
                      epochs=MODEL_CFG["csc"]["epochs"],
                      head_lr=0.001, backbone_lr=0.0001, weight_decay=0.0005,
                      seed=seed, label_smoothing=ls,
                      out_dir=out,
                      )
                # 重命名保存为 head_{hname}
                import shutil
                p = out / f"best_seed{seed}.pt"
                if p.exists():
                    shutil.copy2(p, out / f"head_{hname}_seed{seed}.pt")
                print(f"[head_ablation] {hname} seed={seed} 完成")


def _attach(backbone, head):
    import torch.nn as nn
    from ..models.heads import EDLHead

    is_edl = isinstance(head, EDLHead)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def _feat(self, x):
            f = self.backbone(x)
            if f.dim() == 4:
                f = f.flatten(1)
            return f

        def forward(self, x):
            f = self._feat(x)
            if is_edl:
                return self.head.predict(f)   # dict with evidence/probs/u
            return self.head(f)               # logits

        def predict_dict(self, x):
            f = self._feat(x)
            return self.head.predict(f)

    return M()


if __name__ == "__main__":
    main()