"""E15：压缩率与生成式对照【核心，W5–W6 并行】(方案 E15)。

实测字节/token 分布与压缩比（对数坐标图）；生成式基线 = Qwen2-VL caption → 文本 LLM。
准确率 + 幻觉率 + 三级完整性对比。

用法: python -m src.experiments.E15_compression --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..data.datasets import PlantDataset, default_eval_transform
from ..inference.saf_csc_infer import load_csc_ckpt, load_saf_ckpt, csc_infer, saf_infer
from ..inference.pipeline import build_prompt
from ..kg.kg import KG
from .common import load_partition, load_pv_classes, results_dir
from .llm_common import sample_subset, eval_llm_arm

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    with ExperimentContext(experiment="E15", description="压缩率与生成式对照", configs={}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]
        subset = sample_subset(eval_disease, n=1000, seed=42)

        # 加载 CSC/SAF（【修正】对齐 E5：原硬编码 seed42 的 csc_seed42 不存在、saf_seed42 的 u 塌缩
        # → 改用 seed43，二者均存在且健康）
        csc = load_csc_ckpt(project_path("results", "train_csc", "csc_seed43.pt"), n_classes=len(classes))
        saf = load_saf_ckpt(project_path("results", "train_saf", "saf_seed43.pt"), n_classes=2)
        loader = DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                         sample_paths=[Path(p) for p in eval_disease], class_to_idx=class_to_idx),
                            batch_size=64, shuffle=False, num_workers=4)
        csc_pred = csc_infer(csc, loader, topk=5)
        saf_pred = saf_infer(saf, loader)

        def _can(x):
            """路径规范化：统一为正斜杠绝对路径，规避 Windows 下 str(Path) 反斜杠与 partition 正斜杠不匹配。"""
            return str(Path(x).resolve()).replace("\\", "/")

        u_dict = {_can(x): v for x, v in zip(saf_pred["paths"], saf_pred["u"].tolist())}
        probs_map = {_can(x): v for x, v in zip(csc_pred["paths"], np.array(csc_pred["probs"].tolist()))}
        topk_map = {_can(x): v for x, v in zip(csc_pred["paths"], csc_pred["topk_idx"].tolist())}
        canon_paths = [_can(x) for x in csc_pred["paths"]]
        kg = KG.load(project_path("configs", "kg", "kg.json"))
        kg_desc = kg.descriptions_map()

        # 字节/token 分布与压缩比
        img_bytes = [Path(p).stat().st_size for p in subset]
        text_prompts = []
        for p in subset:
            idx = canon_paths.index(_can(p))
            top3 = [(classes[topk_map[_can(p)][k]], probs_map[_can(p)][topk_map[_can(p)][k]]) for k in range(3)]
            text_prompts.append(build_prompt("L4", classes, top3, u_dict[_can(p)], kg_desc))
        text_bytes = [len(t.encode("utf-8")) for t in text_prompts]
        compression = [i / max(1, t) for i, t in zip(img_bytes, text_bytes)]
        result = {
            "img_bytes_med": float(np.median(img_bytes)),
            "text_bytes_med": float(np.median(text_bytes)),
            "compression_ratio_med": float(np.median(compression)),
            "img_bytes_dist": img_bytes,
            "text_bytes_dist": text_bytes,
            "compression_ratio_dist": compression,
            "generative_baseline": "见 E15_generative（Qwen2-VL caption → 文本 LLM）",
        }
        save_json(result, results_dir("E15") / "E15_compression.json")
        print(f"[E15] 原图中位 {np.median(img_bytes):.0f} B，文本中位 {np.median(text_bytes):.0f} B，"
              f"压缩比中位 {np.median(compression):.1f}x")
        print(f"[E15] 完成，结果存 results/E15/E15_compression.json")


if __name__ == "__main__":
    main()