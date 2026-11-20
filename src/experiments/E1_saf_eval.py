"""E1：SAF 性能评估【W2–W3】(方案第五部分 E1)。

组感知验证集全量 × 3 种子；指标：过滤率/漏检率（Wilson CI）、AUROC、macro-F1、每类分解。

过滤口径（经用户批准）：SAF 的"过滤"=病害二分类，决策分用病害概率 probs[:,1]，
τ* 在 dev 上按"漏检率≤1% 选最大 τ"冻结并套用到 eval（红线12）。
u 的"拒识/降级"路由（u>τ*）属 E12，不在 E1 当病害分类用。

用法: python -m src.experiments.E1_saf_eval
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..utils.seeds import experiment_seeds
from ..data.datasets import PlantDataset, default_eval_transform
from ..evaluation.statistics import wilson_ci, cp_zero_bound, auroc
from ..evaluation.metrics import binary_metrics
from ..evaluation.tau_selection import select_tau, apply_tau, sensitivity_neighborhood
from ..inference.saf_csc_infer import load_saf_ckpt, saf_infer
from .common import load_partition, load_pv_classes, results_dir

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def _binary_label(p) -> int:
    """p 为 Path 或 str；健康→0，病害→1。"""
    return 0 if "healthy" in str(p).lower() else 1


def main() -> None:
    with ExperimentContext(experiment="E1", description="SAF 性能评估", configs={"seeds": experiment_seeds()}) as ctx:
        part = load_partition()
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        class_to_idx = {"healthy": 0, "diseased": 1}
        eval_ds = PlantDataset(root, transform=default_eval_transform(),
                               sample_paths=[Path(p) for p in part["eval"]],
                               class_to_idx=class_to_idx)
        eval_ds.targets = [_binary_label(p) for p in eval_ds.samples]
        eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=4)

        dev_ds = PlantDataset(root, transform=default_eval_transform(),
                              sample_paths=[Path(p) for p in part["dev"]],
                              class_to_idx=class_to_idx)
        dev_ds.targets = [_binary_label(p) for p in dev_ds.samples]
        dev_loader = DataLoader(dev_ds, batch_size=64, shuffle=False, num_workers=4)

        results = {}
        all_tau = []
        for seed in experiment_seeds():
            model = load_saf_ckpt(project_path("results", "train_saf", f"saf_seed{seed}.pt"), n_classes=2)
            dev_pred = saf_infer(model, dev_loader)
            # 病害概率 probs[:,1]；τ* 只在 dev 上选并冻结（红线12）
            dev_p = dev_pred["probs"].numpy()[:, 1]
            tau = select_tau(dev_p, np.array(dev_pred["targets"]))
            all_tau.append(tau["tau_star"])

            pred = saf_infer(model, eval_loader)
            y = np.array(pred["targets"])
            p_disease = pred["probs"].numpy()[:, 1]
            pred_diseased = apply_tau(p_disease, tau["tau_star"])
            # 过滤率/漏检率：过滤=判病害
            fr = binary_metrics(y, pred_diseased.astype(int), pos_label=1)
            auroc_v = auroc(p_disease, y)
            macro_f1 = float(f1_score(y, pred_diseased.astype(int), average="macro", zero_division=0))
            # 每类分解（健康/病害）：健康过滤率=健康中被判病害的比例（除以健康数）
            n_healthy = int((y == 0).sum())
            per_class = {
                "healthy": {"filter_rate": float(((y == 0) & pred_diseased).sum() / n_healthy) if n_healthy else None},
                "diseased": {"miss_rate": fr["miss_rate"]},
            }
            results[seed] = {
                "tau_star": tau["tau_star"],
                "tau_working_curve": tau["working_curve"],
                "filter_rate": fr["filter_rate"],
                "filter_rate_wilson": fr["filter_rate_ci"],
                "miss_rate": fr["miss_rate"],
                "miss_rate_wilson": fr["miss_rate_ci"],
                "auroc": auroc_v,
                "macro_f1": macro_f1,
                **per_class,
            }
            print(f"[E1] seed={seed} τ*={tau['tau_star']:.2f} 过滤率={fr['filter_rate']:.4f} "
                  f"(CI {fr['filter_rate_ci'][0]:.4f}-{fr['filter_rate_ci'][1]:.4f}) "
                  f"漏检率={fr['miss_rate']:.4f} AUROC={auroc_v:.4f} F1={macro_f1:.4f}")

        # 汇总 3 种子
        frs = [results[s]["filter_rate"] for s in results]
        mrs = [results[s]["miss_rate"] for s in results]
        summary = {
            "n_seeds": len(results),
            "filter_rate_mean_std": [float(np.mean(frs)), float(np.std(frs))],
            "miss_rate_mean_std": [float(np.mean(mrs)), float(np.std(mrs))],
            "tau_star_mean": float(np.mean(all_tau)),
            "per_seed": results,
        }
        save_json(summary, results_dir("E1") / "E1_summary.json")
        print(f"[E1] 汇总已保存 results/E1/E1_summary.json")


if __name__ == "__main__":
    main()