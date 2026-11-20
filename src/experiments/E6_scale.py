"""E6：云端 LLM 规模消融【W7】(方案 E6)。

3B / 7B / 14B-GPTQ × L4 输入 × 3 子集。指标：准确率、三级完整性、幻觉率、
confidence 校准、实测延迟 p50/p95、吞吐、显存峰值、每千次成本。

用法: python -m src.experiments.E6_scale --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import torch

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..utils.seeds import get_seeds
from ..data.datasets import PlantDataset, default_eval_transform
from ..inference.saf_csc_infer import load_csc_ckpt, load_saf_ckpt, csc_infer, saf_infer
from ..inference.pipeline import build_prompt
from ..inference.llm_client import get_llm_client
from ..evaluation.integrity import load_validator
from ..kg.kg import KG
from torch.utils.data import DataLoader
from .common import load_partition, load_pv_classes, results_dir
from .llm_common import sample_subset, eval_llm_arm

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    with ExperimentContext(experiment="E6", description=f"云端规模消融 {args.model}", configs={"model": args.model}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]

        # 【修正】与 E5 一致使用 seed43：csc_seed42.pt 不存在，saf_seed42 的 u 塌缩。
        csc = load_csc_ckpt(project_path("results", "train_csc", "csc_seed43.pt"), n_classes=len(classes))
        saf = load_saf_ckpt(project_path("results", "train_saf", "saf_seed43.pt"), n_classes=2)
        loader = DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                         sample_paths=[Path(p) for p in eval_disease], class_to_idx=class_to_idx),
                            batch_size=64, shuffle=False, num_workers=4)
        csc_pred = csc_infer(csc, loader, topk=5)
        saf_pred = saf_infer(saf, loader)
        u_dict = dict(zip(saf_pred["paths"], saf_pred["u"].tolist()))
        probs_map = dict(zip(csc_pred["paths"], csc_pred["probs"].tolist()))
        topk_map = dict(zip(csc_pred["paths"], csc_pred["topk_idx"].tolist()))
        true_map = dict(zip(csc_pred["paths"], csc_pred["targets"]))
        kg = KG.load(project_path("configs", "kg", "kg.json"))
        kg_desc = kg.descriptions_map()

        client = get_llm_client(args.model).load()
        validator = load_validator()
        # 显存峰值
        vram_peak = _vram_peak_mb()

        latencies = []
        results = {}
        for s_seed in args.seeds:
            subset = sample_subset(eval_disease, n=1000, seed=s_seed)
            prompts, true_labels, us = [], [], []
            for p in subset:
                idx = csc_pred["paths"].index(p)
                top3 = [(classes[topk_map[p][k]], probs_map[p][topk_map[p][k]]) for k in range(3)]
                u = u_dict[p]
                prompts.append(build_prompt("L4", classes, top3, u, kg_desc))
                true_labels.append(classes[true_map[p]])
                us.append(u)
            t0 = time.time()
            items = client.generate(prompts, temperature=0.0, guided=True)
            latencies.append((time.time() - t0) / len(prompts))
            res = eval_llm_arm(items, true_labels, classes, u_values=np.array(us), validator=validator)
            res["avg_per_sample_s"] = latencies[-1]
            results[s_seed] = res
            print(f"[E6] {args.model} seed={s_seed} acc={res['semantic']['diag_acc']:.4f} "
                  f"L1={res['structure_ladder']['l1_valid']:.4f}")

        lat = np.array(latencies)
        summary = {
            "model": args.model,
            "vram_peak_mb": vram_peak,
            "latency_p50_s": float(np.median(lat)),
            "latency_p95_s": float(np.percentile(lat, 95)),
            "throughput_req_s": float(1.0 / np.median(lat)),
            "per_1k_cost_est": _cost_estimate(lat),
            "per_seed": results,
        }
        save_json(summary, results_dir("E6") / f"E6_{Path(args.model).name}.json")
        print(f"[E6] 完成，结果存 results/E6/E6_{Path(args.model).name}.json")


def _vram_peak_mb() -> float:
    import subprocess, re
    try:
        r = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                           capture_output=True, text=True, shell=True, timeout=15)
        return float(r.stdout.strip().split("\n")[0])
    except Exception:
        return float("nan")


def _cost_estimate(lat: np.ndarray) -> float:
    """B.3 每千张成本（模型估算，标注来源）。"""
    P_gpu = 350.0  # W（3090 满载典型值，来源：硬件规格）
    H_life = 8760.0  # 小时/年
    W_avg = 350.0  # W
    r_elec = 0.6  # 元/kWh（来源：中国工商业电价约 0.6）
    t_1k = np.median(lat) * 1000 / 3600  # 小时
    C = (P_gpu / H_life + W_avg * r_elec / 1000) * t_1k
    return float(C)


if __name__ == "__main__":
    main()