"""E4：不确定性质量【W4】(方案 E4，与 E3+ 数据共享)。

温度缩放前后校准；EDL-u vs MSP vs Energy vs ODIN；risk-coverage AURC。
并入 Table 3 + Figure 3。

【辅助补充】聚合 3 种子 mean±std：原脚本硬编码 head_H3_seed42.pt（不存在），
且仅单模型。按用户决定改为在种子 43/44/45 上分别计算后聚合，更稳健。
推理加 torch.no_grad()，与 E3+ 一致，结果语义不变。

用法: python -m src.experiments.E4_uncertainty
"""
from __future__ import annotations

import os

# 【辅助补充】显存碎片保护
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..data.datasets import PlantDataset, default_eval_transform
from ..evaluation.statistics import ece, auroc, spearman
from .common import load_partition, load_pv_classes, results_dir

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
SEEDS = [43, 44, 45]  # 【辅助补充】由 seed42→{43,44,45} 聚合


def _ts_fit(logits_val: np.ndarray, targets: np.ndarray, n_iter: int = 100) -> float:
    """温度缩放：在验证集上优化 T（最小化负对数似然）。"""
    from scipy.optimize import minimize_scalar

    def nll(T):
        logits_T = logits_val / T
        logp = logits_T - np.log(np.sum(np.exp(logits_T), axis=-1, keepdims=True))
        return -np.mean(logp[np.arange(len(targets)), targets])

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def _run_seed(seed: int, n_cls: int, part: dict, classes: list, class_to_idx: dict, root: Path, ood_root: Path) -> dict:
    """在单个种子的 H3(EDL) 模型上计算全部 E4 指标。"""
    from .E3plus_eval import _build_model

    model = _build_model("H3", n_cls)
    model.load_state_dict(
        torch.load(project_path("results", "train_head_ablation", f"head_H3_seed{seed}.pt"),
                   map_location="cpu")
    )
    model.cuda().eval()

    val_loader = DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                          sample_paths=[Path(p) for p in part["eval"]],
                                          class_to_idx=class_to_idx),
                            batch_size=64, shuffle=False, num_workers=4)

    logits_list, u_list, msp_list, energy_list, ys = [], [], [], [], []
    with torch.no_grad():  # 【辅助补充】推理无需建图
        for x, y, _ in val_loader:
            x = x.cuda()
            out = model.predict_dict(x)
            # EDL 概率：probs = α/S；用 log(probs) 作 logits 仅是相对顺序近似。
            logits = torch.log(out["probs"] + 1e-12)
            logits_list.append(logits.cpu())
            u_list.append(out["u"].cpu())
            msp_list.append(out["probs"].max(-1).values.cpu())
            # 【修正】能量代理（EDL）：S=Σ_k α_k 为总证据，S 越大越自信。
            # 故用 -log(S)，越大越不确定=OOD（原代码误用 +log(S)，方向反了）。
            energy_list.append(-torch.log(out["S"].clamp_min(1e-12)).squeeze(-1).cpu())
            ys.append(y)
    logits = torch.cat(logits_list).numpy()
    u = torch.cat(u_list).numpy()
    msp = torch.cat(msp_list).numpy()
    energy_score = torch.cat(energy_list).numpy()   # -log(S)，越大越不确定
    y = torch.cat(ys).numpy()

    # 温度缩放：用 dev 集拟合 T（此处用验证集中前 2000 作为校准子集近似）
    cal_logits = logits[:2000]
    cal_y = y[:2000]
    T = _ts_fit(cal_logits, cal_y)
    conf_orig = msp
    conf_ts = np.max(np.exp(logits / T) / np.sum(np.exp(logits / T), axis=-1, keepdims=True), axis=-1)
    correct = (np.argmax(logits, axis=-1) == y)

    # OOD 校准/判别
    ood_ds = PlantDataset(ood_root, transform=default_eval_transform())
    clsm = {c: i for i, c in enumerate(ood_ds.classes)}
    ood_loader = DataLoader(PlantDataset(ood_root, transform=default_eval_transform(), class_to_idx=clsm),
                            batch_size=64, shuffle=False, num_workers=4)
    u_ood, energy_ood, msp_ood = [], [], []
    with torch.no_grad():  # 【辅助补充】推理无需建图
        for x, _, _ in ood_loader:
            x = x.cuda()
            o = model.predict_dict(x)
            u_ood.append(o["u"].cpu())
            energy_ood.append(-torch.log(o["S"].clamp_min(1e-12)).squeeze(-1).cpu())
            msp_ood.append(o["probs"].max(-1).values.cpu())
    u_ood = torch.cat(u_ood).numpy()
    energy_ood = torch.cat(energy_ood).numpy()
    msp_ood = torch.cat(msp_ood).numpy()

    ood_label = np.concatenate([np.zeros(len(u)), np.ones(len(u_ood))])
    rho, _p = spearman(u, correct.astype(float))
    return {
        "temperature_T": T,
        "ece_before_ts": ece(conf_orig, correct),
        "ece_after_ts": ece(conf_ts, correct),
        "uncertainty_auroc_ood": {
            "edl_u": auroc(np.concatenate([u, u_ood]), ood_label),
            # 【修正】energy 用 -log(S)，方向正确；与 EDL-u 均随总证据 S 单调，
            # 数学上等价（非独立证据），报告时如实说明。
            "energy_neglogS": auroc(np.concatenate([energy_score, energy_ood]), ood_label),
            # 【去重】ODIN(1-MSP) 与 MSP-inv = 同一个量，只保留 msp_inv 作为基准。
            "msp_inv": auroc(np.concatenate([1 - msp, 1 - msp_ood]), ood_label),
        },
        "u_vs_correct_spearman_rho": rho,
    }


def _mean_std(values) -> tuple[float, float]:
    return (float(np.mean(values)), float(np.std(values, ddof=1)))


def main() -> None:
    with ExperimentContext(experiment="E4", description="不确定性质量(3种子聚合)", configs={"seeds": SEEDS}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        n_cls = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        ood_root = Path(DATA_CFG["datasets"]["plantdoc"]["dir"])

        per_seed = {}
        for seed in SEEDS:
            r = _run_seed(seed, n_cls, part, classes, class_to_idx, root, ood_root)
            per_seed[f"seed{seed}"] = r
            print(f"[E4] seed={seed}: T={r['temperature_T']:.3f} "
                  f"ECE前={r['ece_before_ts']:.4f} ECE后={r['ece_after_ts']:.4f} "
                  f"OOD-AUROC EDL-u={r['uncertainty_auroc_ood']['edl_u']:.4f} "
                  f"energy(-logS)={r['uncertainty_auroc_ood']['energy_neglogS']:.4f} "
                  f"MSPinv={r['uncertainty_auroc_ood']['msp_inv']:.4f} "
                  f"spearman_u={r['u_vs_correct_spearman_rho']:.4f}", flush=True)
            torch.cuda.empty_cache()  # 【辅助补充】释放显存，避免多模型累积

        # 聚合 mean±std
        keys_scalar = ["temperature_T", "ece_before_ts", "ece_after_ts", "u_vs_correct_spearman_rho"]
        uroc_keys = ["edl_u", "energy_neglogS", "msp_inv"]
        agg = {"per_seed": per_seed, "mean_std": {}}
        for k in keys_scalar:
            vals = [per_seed[f"seed{s}"][k] for s in SEEDS]
            agg["mean_std"][k] = list(_mean_std(vals))
        for k in uroc_keys:
            vals = [per_seed[f"seed{s}"]["uncertainty_auroc_ood"][k] for s in SEEDS]
            agg["mean_std"][f"auroc_ood_{k}"] = list(_mean_std(vals))

        save_json(agg, results_dir("E4") / "E4_summary.json")
        print(f"[E4] 聚合 mean±std:")
        for k, v in agg["mean_std"].items():
            print(f"  {k}: {v[0]:.4f}±{v[1]:.4f}")
        print(f"[E4] 已保存 results/E4/E4_summary.json")


if __name__ == "__main__":
    main()