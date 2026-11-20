"""CSC 模型：MobileNetV3-Small 骨干 + Linear→38（38 类病害分类）。

方案第三部分：CSC（MobileNetV3-Small + Linear→38，标签平滑0.1，epochs=5）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import build_backbone, backbone_dim
from .heads import CEHead


class CSC(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        pretrained: bool = True,
        n_classes: int = 38,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.backbone_type = backbone
        self.backbone = build_backbone(backbone, pretrained=pretrained)
        self.head = CEHead(backbone_dim(backbone), n_classes, label_smoothing=label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = feat.flatten(1)
        return self.head(feat)

    def predict_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if feat.dim() == 4:
            feat = feat.flatten(1)
        return self.head.predict(feat)


def build_csc(**kwargs) -> CSC:
    return CSC(**kwargs)