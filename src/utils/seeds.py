"""种子管理（红线4：全环节种子登记）。唯一种子来源 configs/seeds.yaml。"""
from __future__ import annotations

import random

import numpy as np
import torch

from .io import project_path, load_yaml

_SEEDS = load_yaml(project_path("configs", "seeds.yaml"))


def get_seeds() -> dict:
    return _SEEDS


def experiment_seeds() -> list[int]:
    return list(_SEEDS["experiment_seeds"])


def llm_inference_seeds() -> list[int]:
    return list(_SEEDS["llm_inference_seeds"])


def set_seed(seed: int, cudnn_deterministic: bool = True) -> None:
    """确定性种子设置（红线4）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 注意：cudnn.benchmark=False 保证确定性但会略慢；实验默认关闭 benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = not cudnn_deterministic