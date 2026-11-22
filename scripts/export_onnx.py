# -*- coding: utf-8 -*-
"""导出 SAF / CSC 为 ONNX（含归一化预处理），供手机端 ONNX Runtime Web 真机测量。

输入：float32 (1,3,224,224)，取值 0-255 —— 端侧只做 decode + resize + 转 float
内部：/255 -> ImageNet 归一化 -> backbone -> head
输出：SAF: probs(1,2), u(1)；CSC: probs(1,38)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.inference.saf_csc_infer import load_csc_ckpt, load_saf_ckpt  # noqa: E402
from src.experiments.common import load_pv_classes  # noqa: E402

OUT = ROOT / "export"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Wrap(nn.Module):
    def __init__(self, model: nn.Module, kind: str):
        super().__init__()
        self.model = model
        self.kind = kind
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x: torch.Tensor):
        x = x.float().div(255.0)
        x = (x - self.mean) / self.std
        out = self.model.predict_dict(x)
        if self.kind == "saf":
            return out["probs"], out["u"]
        return out["probs"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_cls = len(load_pv_classes())
    dummy = torch.rand(1, 3, 224, 224, dtype=torch.float32) * 255.0

    jobs = [
        ("saf", load_saf_ckpt(ROOT / "models" / "saf_seed43.pt",
                              n_classes=2, pretrained=False),
         ["probs", "u"]),
        ("csc", load_csc_ckpt(ROOT / "models" / "csc_seed43.pt",
                              n_classes=n_cls, pretrained=False),
         ["probs"]),
    ]

    for name, model, out_names in jobs:
        model.eval()
        wrap = Wrap(model, name).eval()
        with torch.no_grad():
            ref = wrap(dummy)
        path = OUT / f"{name}.onnx"
        torch.onnx.export(
            wrap, dummy, str(path),
            input_names=["image"], output_names=out_names,
            opset_version=13, do_constant_folding=True, dynamo=False,
        )
        shapes = [tuple(t.shape) for t in (ref if isinstance(ref, tuple) else (ref,))]
        print(f"[export] {name}.onnx  {path.stat().st_size/1024:.0f} KB  outputs={shapes}", flush=True)


if __name__ == "__main__":
    main()
