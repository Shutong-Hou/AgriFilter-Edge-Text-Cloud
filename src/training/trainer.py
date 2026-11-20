"""训练器（SAF/CSC/E3+ 头消融共用）。

- SAF（EDL 头）：Sensoy Type-II MLE + KL 退火；AdamW（头1e-3/骨干1e-4）。
- CSC（CE 头，标签平滑 0.1）：CE 损失。
- E3+ H1/H2/H3：三头统一入口。
按 epoch 保存检查点，返回最佳/最后指标。
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.seeds import set_seed
from ..utils.io import ensure_dir, save_json
from .losses import sensoy_edl_loss, cross_entropy_loss


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, head_type: str) -> dict:
    model.eval()
    correct = 0
    total = 0
    u_list = []
    for x, y, _ in loader:
        x, y = x.cuda(), y.cuda()
        if head_type == "edl":
            out = model(x)
            pred = out["probs"].argmax(-1)
            u_list.append(out["u"].cpu())
        else:
            out = model.predict_dict(x)
            pred = out["probs"].argmax(-1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / max(1, total)
    res = {"accuracy": acc, "n": total}
    if u_list:
        res["mean_u"] = float(torch.cat(u_list).mean().item())
    return res


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    head_type: str,
    epochs: int,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    seed: int,
    label_smoothing: float = 0.0,
    kl_anneal_step: int = 10,
    log_every: int = 50,
    out_dir: Path | str | None = None,
) -> dict:
    set_seed(seed)
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        ensure_dir(out_dir)

    # 分组参数：头用 head_lr，骨干用 backbone_lr
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params = [p for n, p in model.named_parameters() if "head" in n]
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    model.cuda()
    best_acc = -1.0
    history = {"train_acc": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        t0 = time.time()
        for i, (x, y, _) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            if head_type == "edl":
                out = model(x)
                loss = sensoy_edl_loss(
                    out["evidence"], y, kl_scale=1.0, anneal_epoch=ep, anneal_step=kl_anneal_step
                )
                pred = out["probs"].argmax(-1)
            else:
                logits = model(x)
                loss = cross_entropy_loss(logits, y, label_smoothing=label_smoothing)
                pred = logits.argmax(-1)
            loss.backward()
            opt.step()
            correct += (pred == y).sum().item()
            total += y.size(0)
            running += loss.item()
            if (i + 1) % log_every == 0:
                print(f"  epoch {ep} step {i+1}: loss={running/(i+1):.4f}", flush=True)
        train_acc = correct / max(1, total)
        history["train_acc"].append(train_acc)
        print(f"[epoch {ep}] train_acc={train_acc:.4f} loss={running/len(train_loader):.4f} ({time.time()-t0:.1f}s)", flush=True)
        if val_loader is not None:
            val = evaluate_model(model, val_loader, head_type)
            history["val_acc"].append(val["accuracy"])
            print(f"          val_acc={val['accuracy']:.4f}", flush=True)
            if val["accuracy"] > best_acc:
                best_acc = val["accuracy"]
                if out_dir:
                    torch.save(model.state_dict(), out_dir / f"best_seed{seed}.pt")
    if out_dir:
        torch.save(model.state_dict(), out_dir / f"last_seed{seed}.pt")
        save_json({"history": history, "seed": seed, "best_val_acc": best_acc}, out_dir / f"train_log_seed{seed}.json")
    return {"best_val_acc": best_acc, "history": history, "seed": seed}