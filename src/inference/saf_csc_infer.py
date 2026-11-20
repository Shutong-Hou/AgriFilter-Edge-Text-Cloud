"""SAF/CSC 图像模型推理（批量预测 + u 输出）。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.saf import SAF
from ..models.csc import CSC
from ..data.datasets import PlantDataset, default_eval_transform


@torch.no_grad()
def saf_infer(model: SAF, loader: DataLoader) -> dict:
    """返回 {probs, u, pred_class(病害/健康), pred_binary, paths, targets}。"""
    model.cuda().eval()
    probs, u, pred_bin = [], [], []
    paths, targets = [], []
    for x, y, p in loader:
        x = x.cuda()
        out = model(x)
        probs.append(out["probs"].cpu())
        u.append(out["u"].cpu())
        pred_bin.append(out["probs"].argmax(-1).cpu())
        paths.extend(p)
        targets.extend(y.tolist())
    return {
        "probs": torch.cat(probs),
        "u": torch.cat(u),
        "pred_binary": torch.cat(pred_bin),
        "paths": paths,
        "targets": targets,
    }


@torch.no_grad()
def csc_infer(model: CSC, loader: DataLoader, topk: int = 5) -> dict:
    """返回 {probs, topk_idx, top1, paths, targets}。"""
    model.cuda().eval()
    probs_list, paths, targets = [], [], []
    for x, y, p in loader:
        x = x.cuda()
        out = model.predict_dict(x)
        probs_list.append(out["probs"].cpu())
        paths.extend(p)
        targets.extend(y.tolist())
    probs = torch.cat(probs_list)
    topk_idx = torch.topk(probs, k=min(topk, probs.shape[1]), dim=-1).indices
    return {
        "probs": probs,
        "topk_idx": topk_idx,
        "top1": topk_idx[:, 0],
        "paths": paths,
        "targets": targets,
    }


def load_saf_ckpt(ckpt: Path, **kwargs) -> SAF:
    m = SAF(**kwargs)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m


def load_csc_ckpt(ckpt: Path, n_classes: int = 38, **kwargs) -> CSC:
    m = CSC(n_classes=n_classes, **kwargs)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m


def make_loader(sample_paths: list[str], class_to_idx: dict, batch_size: int = 64):
    ds = PlantDataset(Path(sample_paths[0]).parent.parent if sample_paths else ".",
                      transform=default_eval_transform(),
                      sample_paths=[Path(p) for p in sample_paths],
                      class_to_idx=class_to_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)