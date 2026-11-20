"""损失函数（方案第三部分 SAF 训练规范）。

Sensoy Type-II MLE 损失 + KL 退火 λ_t=min(1, t/10)：
  L = Σ_k y_k (log S - log α_k) + λ_t * KL(Dir(α) || Dir(α̃))
其中 α=e+1（Softplus evidence），S=Σα_k，α̃ 移除正确类证据后（正确类 α=1）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def sensoy_edl_loss(
    evidence: torch.Tensor,
    target: torch.Tensor,
    kl_scale: float = 1.0,
    anneal_epoch: int | None = None,
    anneal_step: int = 10,
) -> torch.Tensor:
    """Sensoy Type-II MLE 总损失。

    evidence: (B, K) Softplus evidence ≥0。
    target:   (B,) int 类别标签。
    """
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)  # (B,1)
    K = alpha.shape[-1]

    # 正则化：移除正确类证据 (α̃)：正确类 α=1，其余不变
    alpha_tilde = alpha.clone()
    alpha_tilde.scatter_(-1, target.unsqueeze(-1), 1.0)

    # Type-II MLE 项：Σ_k y_k (log S - log α_k)
    log_alpha = (alpha + 1e-12).log()
    log_S = (S + 1e-12).log()
    one_hot = F.one_hot(target, num_classes=K).float()
    mle = (one_hot * (log_S - log_alpha)).sum(-1)  # (B,)

    # KL 退火项
    kl = _kl_dirichlet(alpha, alpha_tilde)  # (B,)
    if anneal_epoch is not None:
        lam = min(1.0, anneal_epoch / anneal_step)
    else:
        lam = kl_scale
    loss = (mle + lam * kl).mean()
    return loss


def _kl_dirichlet(alpha: torch.Tensor, alpha_tilde: torch.Tensor) -> torch.Tensor:
    """KL(Dirichlet(alpha) || Dirichlet(alpha_tilde)) per sample (B,)。"""
    lgamma = torch.lgamma
    digamma = torch.digamma
    S_alpha = alpha.sum(-1)
    S_tilde = alpha_tilde.sum(-1)
    term = (
        lgamma(S_alpha)
        - lgamma(S_tilde)
        - (lgamma(alpha) - lgamma(alpha_tilde)).sum(-1)
        + ((alpha - alpha_tilde) * (digamma(alpha) - digamma(S_alpha.unsqueeze(-1)).expand_as(alpha))).sum(-1)
    )
    return term


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor, label_smoothing: float = 0.0):
    return F.cross_entropy(logits, target, label_smoothing=label_smoothing)