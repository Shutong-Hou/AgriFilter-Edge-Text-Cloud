"""E3+：分类头消融评估（方案 E3+）。与 E4 共享数据。

二分类指标 + OOD（PV=ID vs D2/D3=OOD：AUROC、FPR@95%TPR）+ 校准（ECE/Brier/reliability）+ risk-coverage/AURC。
EDL 未显著优则如实报告（红线7）。

用法: python -m src.experiments.E3plus_eval
"""
from __future__ import annotations

import os

# 【辅助补充】显存碎片保护：不改变任何评估逻辑，仅避免显存碎片化导致 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..utils.seeds import experiment_seeds
from ..data.datasets import PlantDataset, default_eval_transform
from ..evaluation.statistics import ece, auroc
from ..evaluation.metrics import fpr_at_tpr, multiclass_metrics, risk_coverage_aurc
from ..models.backbone import build_backbone, backbone_dim
from ..models.heads import build_head, EDLHead
from .common import load_partition, load_pv_classes, results_dir

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
HEADS = ["H1", "H2", "H3"]
EVAL_BATCH = 32  # 【辅助补充】评估 batch 64→32：降低峰值显存/内存，eval 模式不改变结果


def _uncertainty(out, head_type: str) -> torch.Tensor:
    if head_type == "edl":
        return out["u"]
    probs = out["probs"]
    return 1.0 - probs.max(-1).values  # 1-MSP（越大越不确定）


def _conf(out, head_type: str) -> torch.Tensor:
    if head_type == "edl":
        return out["expected_p"]
    return out["probs"].max(-1).values


def _extract(model, loader, head_type: str):
    u_list, conf_list, probs_list, ys = [], [], [], []
    # 【辅助补充】推理无需构建 autograd 图，禁用可减少内存/图开销，结果不变
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.cuda()
            out = model.predict_dict(x)
            u_list.append(_uncertainty(out, head_type).cpu())
            conf_list.append(_conf(out, head_type).cpu())
            probs_list.append(out["probs"].cpu())
            ys.append(y)
    return (torch.cat(u_list), torch.cat(conf_list), torch.cat(probs_list), torch.cat(ys))


def _build_model(hname: str, n_cls: int):
    import torch.nn as nn

    edl = hname == "H3"
    backbone = build_backbone("mobilenet_v3_small", pretrained=False)
    ls = 0.0 if hname == "H1" else 0.1  # H1 标准CE；H2 CE+标签平滑
    head = build_head("edl" if edl else ("ce_ls" if hname == "H2" else "ce"),
                      backbone_dim("mobilenet_v3_small"), n_cls, label_smoothing=ls)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def _feat(self, x):
            f = self.backbone(x)
            return f.flatten(1) if f.dim() == 4 else f

        def predict_dict(self, x):
            return self.head.predict(self._feat(x))

    return M()


def main() -> None:
    with ExperimentContext(experiment="E3plus", description="分类头消融评估+OOD+校准", configs={"heads": HEADS}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        n_cls = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        id_loader = DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                             sample_paths=[Path(p) for p in part["eval"]], class_to_idx=class_to_idx),
                               batch_size=EVAL_BATCH, shuffle=False, num_workers=0)
        # OOD：D2 PlantDoc（真实田间）
        ood_root = Path(DATA_CFG["datasets"]["plantdoc"]["dir"])
        ood_ds = PlantDataset(ood_root, transform=default_eval_transform())
        cls_map = {c: i for i, c in enumerate(ood_ds.classes)}
        ood_ds.class_to_idx = cls_map
        ood_ds.targets = [cls_map.get(Path(p).parent.name, 0) for p in ood_ds.samples]
        ood_loader = DataLoader(ood_ds, batch_size=EVAL_BATCH, shuffle=False, num_workers=0)

        summary = {}
        for hname in HEADS:
            head_type = "edl" if hname == "H3" else "ce"
            for seed in experiment_seeds():
                model = _build_model(hname, n_cls)
                ckpt = project_path("results", "train_head_ablation", f"head_{hname}_seed{seed}.pt")
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.cuda().eval()

                u_id, conf_id, probs_id, y_id = _extract(model, id_loader, head_type)
                u_ood, _, _, _ = _extract(model, ood_loader, head_type)

                # ID 分类（Top-1/3/5）
                pred_id = probs_id.argmax(-1)
                topk = torch.topk(probs_id, k=min(5, n_cls), dim=-1).indices
                m = multiclass_metrics(y_id.numpy(), topk.numpy(), n_cls)

                # OOD 检测：ID 低不确定 vs OOD 高不确定
                scores = torch.cat([u_id, u_ood]).numpy()
                ood_label = np.concatenate([np.zeros(len(u_id)), np.ones(len(u_ood))])
                auroc_ood = auroc(scores, ood_label)
                fpr = fpr_at_tpr(scores, ood_label)

                # 校准 + risk-coverage/AURC（ID）
                correct = (pred_id == y_id).numpy()
                ece_v = ece(conf_id.numpy(), correct)
                brier = float(((conf_id.numpy() - correct) ** 2).mean())
                rc = risk_coverage_aurc(conf_id.numpy(), correct)

                summary[f"{hname}_seed{seed}"] = {
                    "top1": m["top1"], "macro_f1": m["macro_f1"],
                    "ood_auroc": auroc_ood, "fpr_at_95tpr": fpr["fpr_at_tpr"],
                    "ece": ece_v, "brier": brier, "aurc": rc["aurc"],
                }
                print(f"[E3+] {hname} seed={seed}: Top1={m['top1']:.4f} OOD-AUROC={auroc_ood:.4f} "
                      f"FPR@95TPR={fpr['fpr_at_tpr']:.4f} ECE={ece_v:.4f} Brier={brier:.4f} AURC={rc['aurc']:.4f}")
                # 【辅助补充】释放本轮模型并清空 CUDA 缓存，避免多模型连续加载导致显存累积/碎片化
                del model
                torch.cuda.empty_cache()
        save_json(summary, results_dir("E3plus") / "E3plus_summary.json")
        print(f"[E3+] 已保存 results/E3plus/E3plus_summary.json")


if __name__ == "__main__":
    main()