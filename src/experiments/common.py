"""实验公共辅助：加载 E0 划分产物、模型检查点、PV 类表。"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import load_json, project_path


def results_dir(exp: str) -> Path:
    d = project_path("results", exp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_partition() -> dict:
    """读取 E0 生成的划分（含 train/dev/eval 路径、拍摄组、phash）。"""
    p = project_path("results", "E0", "partition.json")
    return load_json(p)


def load_pv_classes() -> list[str]:
    """PlantVillage 38 类（按 DatasetCard 生成产物读取）。"""
    p = project_path("results", "E0", "pv_classes.json")
    return load_json(p)


def load_phash_cache() -> dict[str, int]:
    p = project_path("results", "E0", "phash_cache.json")
    return load_json(p) if p.exists() else {}


def ckpt_path(exp: str, kind: str, seed: int) -> Path:
    """返回模型检查点路径。kind ∈ {saf, csc, head_<type>}。"""
    return project_path("results", exp, f"{kind}_seed{seed}.pt")


def load_saf_for_exp(exp: str, seed: int):
    from ..models.saf import SAF
    from ..inference.saf_csc_infer import load_saf_ckpt

    return load_saf_ckpt(ckpt_path(exp, "saf", seed))


def load_csc_for_exp(exp: str, seed: int, n_classes: int = 38):
    from ..inference.saf_csc_infer import load_csc_ckpt

    return load_csc_ckpt(ckpt_path(exp, "csc", seed), n_classes=n_classes)