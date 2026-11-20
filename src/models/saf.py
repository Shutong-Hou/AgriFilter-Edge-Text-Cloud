"""SAF 模型：MobileNetV3-Small 骨干 + EDL 头（K=2，健康/病害过滤）。

方案第三部分：SAF（EDL 过滤）+ Softplus evidence、α=e+1、u=K/S、K=2。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone, backbone_dim
from .heads import EDLHead


class SAF(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        n_classes: int = 2,
    ):
        super().__init__()
        self.backbone_type = backbone
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        self.head = EDLHead(backbone_dim(backbone), n_classes=n_classes)
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if feat.dim() == 4:  # (B, C, 1, 1) -> (B, C)
            feat = feat.flatten(1)
        return self.head.predict(feat)

    def uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["u"]

    def predict_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward(x)


def build_saf(**kwargs) -> SAF:
    return SAF(**kwargs)