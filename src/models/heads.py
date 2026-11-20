"""分类头（方案 E3+ 头消融 H1/H2/H3）。

- H1 标准 CE 头：softmax 概率，MSP/熵代理不确定性。
- H2 CE + 标签平滑（0.1）。
- H3 EDL 头：Softplus evidence, α=e+1, u=K/S, K=类别数（二分类=2）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEHead(nn.Module):
    """H1 标准 CE 头（MSP/熵代理标注）。"""

    def __init__(self, backbone_dim: int, n_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(backbone_dim, n_classes)
        self.label_smoothing = label_smoothing

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)  # logits

    def predict(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward(feat)
        probs = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "msp": probs.max(-1).values,
            "entropy": -(probs * (probs + 1e-12).log()).sum(-1),
        }


class H2Head(CEHead):
    """H2 = H1 + 标签平滑（复用 CEHead，label_smoothing>0）。"""

    pass


class EDLHead(nn.Module):
    """H3 EDL 头（Sensoy Type-II MLE）。Softplus evidence → α=e+1 → u=K/S。"""

    def __init__(self, backbone_dim: int, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(backbone_dim, n_classes)
        self.n_classes = n_classes

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # 证据 e = softplus(logits) ≥ 0
        return F.softplus(self.fc(feat))  # evidence (>=0)

    def predict(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.forward(feat)
        alpha = evidence + 1.0  # α = e + 1
        S = alpha.sum(-1, keepdim=True)  # S = ΣK α_k
        probs = alpha / S  # 期望概率
        u = self.n_classes / S  # u = K/S
        return {
            "evidence": evidence,
            "alpha": alpha,
            "S": S,
            "probs": probs,
            "u": u.squeeze(-1),
            "expected_p": probs.max(-1).values,
        }

    def uncertainty(self, feat: torch.Tensor) -> torch.Tensor:
        return self.predict(feat)["u"]


def build_head(
    head_type: str,
    backbone_dim: int,
    n_classes: int,
    label_smoothing: float = 0.1,
) -> nn.Module:
    if head_type in ("ce", "H1"):
        return CEHead(backbone_dim, n_classes, label_smoothing=0.0)
    if head_type in ("ce_ls", "H2"):
        return CEHead(backbone_dim, n_classes, label_smoothing=label_smoothing)
    if head_type in ("edl", "H3"):
        return EDLHead(backbone_dim, n_classes)
    raise ValueError(f"未知头类型: {head_type}")