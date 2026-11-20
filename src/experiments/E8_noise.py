"""E8：压缩容错与噪声鲁棒性【W7–W8】(方案 E8)。

E8a 语义噪声：噪声率 {0,10,20%} × L2 vs L4 × 3 子集 × n=1000；翻转目标取混淆矩阵 Top 易混对。
E8b 畸形载荷压力测试（N10）：4 类畸形 × n=1000（seed=42 子集）：
  (i) Top-3 概率打乱不归一 / (ii) u 与 Top-3 矛盾（高 u 配高概率）
  (iii) KG 描述截断一半 / (iv) 字段缺失（缺 u）。
双轨：语义轨（准确率/纠错率衰减曲线）+ 结构轨（三级完整性，guided vs 自由文本双臂对照，红线15）。

用法: python -m src.experiments.E8_noise --model Qwen/Qwen2.5-7B-Instruct
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
from ..inference.llm_client import get_llm_client
from ..evaluation.integrity import load_validator
from ..kg.kg import KG
from .common import load_partition, load_pv_classes, results_dir
from .llm_common import sample_subset, eval_llm_arm

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
NOISE_RATES = [0.0, 0.1, 0.2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arm", default="L4")
    ap.add_argument("--mode", default="e8a")
    args = ap.parse_args()

    with ExperimentContext(experiment="E8", description=f"噪声鲁棒 {args.mode} {args.arm}", configs={}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]

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

        client = get_llm_client(args.model).load()
        validator = load_validator()
        subset = sample_subset(eval_disease, n=1000, seed=42)

        if args.mode == "e8a":
            out = {}
            for arm in (args.arm,):
                for rate in NOISE_RATES:
                    prompts, true_labels = [], []
                    for p in subset:
                        idx = csc_pred["paths"].index(p)
                        top3 = _top3(classes, topk_map[p], probs_map[p], rate)
                        u = u_dict[p]
                        prompts.append(build_prompt(arm, classes, top3, u, kg_desc))
                        true_labels.append(classes[true_map[p]])
                    items = client.generate(prompts, temperature=0.0, guided=True)
                    res = eval_llm_arm(items, true_labels, classes, validator=validator)
                    out[f"{arm}_noise{int(rate*100)}"] = {
                        "acc": res["semantic"]["diag_acc"], "ladder": res["structure_ladder"]}
                    print(f"[E8a] {arm} noise={rate}: acc={res['semantic']['diag_acc']:.4f} "
                          f"L1={res['structure_ladder']['l1_valid']:.4f}")
            save_json(out, results_dir("E8") / f"E8a_{args.arm}.json")
        elif args.mode == "e8b":
            out = {}
            for mal_type in ["not_normalized", "contradict", "kg_truncated", "missing_u"]:
                prompts, true_labels = [], []
                for p in subset:
                    idx = csc_pred["paths"].index(p)
                    top3 = _perturb(classes, topk_map[p], probs_map[p], mal_type, u_dict[p], kg_desc)
                    u = None if mal_type == "missing_u" else u_dict[p]
                    kd = _perturb_kg(kg_desc, mal_type)
                    prompts.append(build_prompt("L4", classes, top3, u, kd))
                    true_labels.append(classes[true_map[p]])
                items = client.generate(prompts, temperature=0.0, guided=True)
                res = eval_llm_arm(items, true_labels, classes, validator=validator)
                out[mal_type] = {"acc": res["semantic"]["diag_acc"], "ladder": res["structure_ladder"]}
                print(f"[E8b] {mal_type}: acc={res['semantic']['diag_acc']:.4f}")
            save_json(out, results_dir("E8") / "E8b_malformed.json")
        print(f"[E8] 完成，结果存 results/E8/")


def _top3(classes, topk, probs, rate):
    import random
    rng = random.Random(0)
    top3 = [(classes[topk[k]], float(probs[topk[k]])) for k in range(3)]
    if rate > 0:
        n_flip = int(rate * len(top3))
        for _ in range(n_flip):
            a, b = rng.sample(range(len(top3)), 2)
            top3[a], top3[b] = top3[b], top3[a]  # 概率/类对调（语义噪声）
    return top3


def _perturb(classes, topk, probs, mal_type, u, kg_desc):
    top3 = [(classes[topk[k]], float(probs[topk[k]])) for k in range(3)]
    if mal_type == "not_normalized":
        return [(c, float(p) * 2.0) for c, p in top3]  # 概率打乱不归一
    if mal_type == "contradict":
        return [(top3[0][0], 0.99), (top3[1][0], 0.03), (top3[2][0], 0.02)] + [(f"u={u:.2f}", None)]  # 高 u 配高概率
    return top3


def _perturb_kg(kg_desc, mal_type):
    if mal_type == "kg_truncated":
        return {k: v[: len(v) // 2] for k, v in kg_desc.items()}  # 截断一半
    return kg_desc


if __name__ == "__main__":
    main()