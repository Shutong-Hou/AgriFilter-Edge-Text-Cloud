"""骨干网络工厂（MobileNetV3-Small；E3 扩展可加更多骨干，见 model_config）。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def _backbone_dim(name: str) -> int:
    return {"mobilenet_v3_small": 576}.get(name, 576)


def build_backbone(name: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    """返回去掉分类头的骨干特征提取器。"""
    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        # 去掉最后的 classifier（分类头），保留 features + avgpool
        m.classifier = nn.Identity()
        return m
    raise ValueError(f"骨干 {name} 未实现（Tier2 扩展）")


def backbone_dim(name: str = "mobilenet_v3_small") -> int:
    return _backbone_dim(name)