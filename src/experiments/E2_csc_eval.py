"""E2：CSC 分类性能【W2–W3】(方案 E2)。

全量验证集 × 3 种子；Top-1/3/5 Acc、macro-F1、38×38 混淆矩阵；易混类对供 E8/E18。

用法: python -m src.experiments.E2_csc_eval
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..utils.seeds import experiment_seeds
from ..data.datasets import PlantDataset, default_eval_transform
from ..evaluation.metrics import multiclass_metrics
from ..inference.saf_csc_infer import load_csc_ckpt, csc_infer
from .common import load_partition, load_pv_classes, results_dir

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    with ExperimentContext(experiment="E2", description="CSC 分类性能", configs={"seeds": experiment_seeds()}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_ds = PlantDataset(root, transform=default_eval_transform(),
                               sample_paths=[Path(p) for p in part["eval"]],
                               class_to_idx=class_to_idx)
        eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=4)

        results = {}
        for seed in experiment_seeds():
            model = load_csc_ckpt(project_path("results", "train_csc", f"csc_seed{seed}.pt"),
                                  n_classes=len(classes))
            pred = csc_infer(model, eval_loader, topk=5)
            y = np.array(pred["targets"])
            m = multiclass_metrics(y, pred["topk_idx"].numpy(), len(classes))
            results[seed] = m
            print(f"[E2] seed={seed} Top1={m['top1']:.4f} Top3={m['top3']:.4f} "
                  f"Top5={m['top5']:.4f} macroF1={m['macro_f1']:.4f}")

        # 易混类对：从混淆矩阵提取（全部种子平均）
        top1s = [results[s]["top1"] for s in results]
        summary = {
            "n_seeds": len(results),
            "top1_mean_std": [float(np.mean(top1s)), float(np.std(top1s))],
            "per_seed": results,
        }
        save_json(summary, results_dir("E2") / "E2_summary.json")
        # 混淆矩阵（取第一个种子）另存，文件名用真实种子号
        first_seed = experiment_seeds()[0]
        save_json(results[first_seed]["confusion_matrix"],
                  results_dir("E2") / f"confusion_matrix_seed{first_seed}.json")
        print(f"[E2] 汇总已保存 results/E2/E2_summary.json")


if __name__ == "__main__":
    main()