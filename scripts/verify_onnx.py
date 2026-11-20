# -*- coding: utf-8 -*-
"""校验导出的 ONNX 与 PyTorch 模型数值一致性（同一随机输入）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\agri_review_run")
sys.path.insert(0, str(ROOT))

from src.experiments.common import load_pv_classes  # noqa: E402
from src.inference.saf_csc_infer import load_csc_ckpt, load_saf_ckpt  # noqa: E402
from scripts.export_onnx import MEAN, STD  # noqa: E402


def main() -> None:
    import onnxruntime as ort

    n_cls = len(load_pv_classes())
    x = torch.rand(1, 3, 224, 224, dtype=torch.float32) * 255.0

    cases = {
        "saf": load_saf_ckpt(ROOT / "results" / "train_saf" / "saf_seed43.pt",
                             n_classes=2, pretrained=False),
        "csc": load_csc_ckpt(ROOT / "results" / "train_csc" / "csc_seed43.pt",
                             n_classes=n_cls, pretrained=False),
    }
    for name, model in cases.items():
        model.eval()
        with torch.no_grad():
            xn = (x / 255.0 - MEAN) / STD
            out = model.predict_dict(xn)
            keys = ["probs", "u"] if name == "saf" else ["probs"]
            ref = [out[k].numpy().reshape(-1) for k in keys]

        sess = ort.InferenceSession(str(ROOT / "export" / f"{name}.onnx"),
                                    providers=["CPUExecutionProvider"])
        outs = sess.run(None, {"image": x.numpy()})
        print(f"[{name}] torch outs={[r.shape for r in ref]} onnx outs={[o.shape for o in outs]}")
        for k, r, o in zip(keys, ref, outs):
            d = np.abs(r - o.reshape(-1)).max()
            print(f"  {k:5s} max|Δ| = {d:.3e}   torch[0:3]={np.round(r[:3],4)}  onnx[0:3]={np.round(o.reshape(-1)[:3],4)}")


if __name__ == "__main__":
    main()
