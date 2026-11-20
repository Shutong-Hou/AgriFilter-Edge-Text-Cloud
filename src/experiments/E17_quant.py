"""E17：量化权衡【W10】(方案 E17)。

Qwen2.5-7B：FP16 vs AWQ-Int4，seed=42 子集 n=1000，报准确率差（服务 E11 并报）、吞吐、显存。
T2：GPTQ 档 + 边缘 INT8。

用法: python -m src.experiments.E17_quant --fp16 Qwen/Qwen2.5-7B-Instruct --awq Qwen/Qwen2.5-7B-Instruct-AWQ
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..inference.saf_csc_infer import load_csc_ckpt, load_saf_ckpt, csc_infer, saf_infer
from ..inference.pipeline import build_prompt
from ..inference.llm_client import get_llm_client
from ..kg.kg import KG
from .common import load_partition, load_pv_classes, results_dir
from .llm_common import sample_subset, eval_llm_arm

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--awq", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    args = ap.parse_args()

    with ExperimentContext(experiment="E17", description="量化权衡 FP16 vs AWQ", configs={}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]
        subset = sample_subset(eval_disease, n=1000, seed=42)

        csc = load_csc_ckpt(project_path("results", "train_csc", "csc_seed42.pt"), n_classes=len(classes))
        saf = load_saf_ckpt(project_path("results", "train_saf", "saf_seed42.pt"), n_classes=2)
        loader = DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                         sample_paths=[Path(p) for p in eval_disease], class_to_idx=class_to_idx),
                            batch_size=64, shuffle=False, num_workers=4)
        csc_pred = csc_infer(csc, loader, topk=5)
        saf_pred = saf_infer(saf, loader)
        u_dict = dict(zip(saf_pred["paths"], saf_pred["u"].tolist()))
        probs_map = dict(zip(csc_pred["paths"], np.array(csc_pred["probs"].tolist())))
        topk_map = dict(zip(csc_pred["paths"], csc_pred["topk_idx"].tolist()))
        true_map = dict(zip(csc_pred["paths"], csc_pred["targets"]))
        kg = KG.load(project_path("configs", "kg", "kg.json"))
        kg_desc = kg.descriptions_map()
        prompts, true_labels = [], []
        for p in subset:
            idx = csc_pred["paths"].index(p)
            top3 = [(classes[topk_map[p][k]], probs_map[p][topk_map[p][k]]) for k in range(3)]
            prompts.append(build_prompt("L4", classes, top3, u_dict[p], kg_desc))
            true_labels.append(classes[true_map[p]])

        out = {}
        for tag, repo in [("fp16", args.fp16), ("awq", args.awq)]:
            client = get_llm_client(repo).load()
            items = client.generate(prompts, temperature=0.0, guided=True)
            res = eval_llm_arm(items, true_labels, classes, validator=None)
            out[tag] = {
                "acc": res["semantic"]["diag_acc"],
                "vram_mb": _vram_mb(),
                "ladder": res["structure_ladder"],
            }
            print(f"[E17] {tag}: acc={res['semantic']['diag_acc']:.4f} 显存={out[tag]['vram_mb']:.0f}MB")
        out["acc_delta_awq_minus_fp16"] = out["awq"]["acc"] - out["fp16"]["acc"]
        save_json(out, results_dir("E17") / "E17_quant.json")
        print(f"[E17] 完成，结果存 results/E17/E17_quant.json（服务 E11 的 C-B 声明）")


def _vram_mb() -> float:
    import subprocess
    try:
        r = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                           capture_output=True, text=True, shell=True, timeout=15)
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()