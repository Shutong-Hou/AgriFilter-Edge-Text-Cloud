"""E7：Prompt 消融【精简，W7】(方案 E7)。

变体：zero-shot 直出 / CoT / few-shot-3（K-Means medoid 自动选取 + LOO 敏感性 + 5 随机种子对照）
/ guided vs 自由文本+正则（比准确率 + 三级完整性，不止比解析率）。
协议：seed=42 子集筛选 → 最优/最差变体 3 子集复测。

用法: python -m src.experiments.E7_prompt --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..utils.seeds import get_seeds
from ..data.datasets import PlantDataset, default_eval_transform
from ..data.kmeans_fewshot import select_medoids, loo_sensitivity, random_seed_contrast
from ..models.csc import CSC
from ..evaluation.integrity import load_validator
from ..inference.llm_client import get_llm_client
from ..inference.pipeline import build_prompt
from .common import load_partition, load_pv_classes, results_dir
from .llm_common import sample_subset, eval_llm_arm, _match_diag

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
VARIANTS = ["zero_shot", "cot", "few_shot_3"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    with ExperimentContext(experiment="E7", description="Prompt 消融", configs={"variants": VARIANTS}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]
        subset = sample_subset(eval_disease, n=1000, seed=42)

        # few-shot 示例：用 CSC 特征做 K-Means medoid（特征近似用 CSC head 输出 probs）
        csc = CSC(n_classes=len(classes))
        csc.load_state_dict(torch.load(project_path("results", "train_csc", "csc_seed42.pt"), map_location="cpu"))
        csc.cuda().eval()
        csc_loader = DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                             sample_paths=[Path(p) for p in subset], class_to_idx=class_to_idx),
                                batch_size=64, shuffle=False, num_workers=4)
        feats, labs = [], []
        with torch.no_grad():
            for x, y, _ in csc_loader:
                out = csc.predict_dict(x.cuda())
                feats.append(out["probs"].cpu().numpy())
                labs.extend([classes[i] for i in y.tolist()])
        feats = np.concatenate(feats)
        med_selected, med_meta = select_medoids(feats, labs, k=1, seed=get_seeds()["fewshot_seed"])
        loo = loo_sensitivity(feats, labs, med_selected)
        rsc = random_seed_contrast(feats, labs, k=1)
        ex_str = "\n".join(f"类别:{labs[i]}, 特征:{', '.join(f'{v:.2f}' for v in feats[i][:3])}..." for i in med_selected[:3])

        client = get_llm_client(args.model).load()
        validator = load_validator()
        results = {}
        for v in VARIANTS:
            prompts, true_labels = [], []
            for p in subset:
                prompts.append(build_prompt(v, classes, None, None, None, examples=ex_str if v == "few_shot_3" else ""))
                true_labels.append(classes[class_to_idx[Path(p).parent.name]])
            items = client.generate(prompts, temperature=0.0, guided=True)
            res = eval_llm_arm(items, true_labels, classes, validator=validator)
            results[v] = {"acc": res["semantic"]["diag_acc"], "ladder": res["structure_ladder"]}
            print(f"[E7] {v}: acc={res['semantic']['diag_acc']:.4f} L1={res['structure_ladder']['l1_valid']:.4f}")

        # guided vs 自由文本+正则（同批）
        prompts = [build_prompt("zero_shot", classes, None, None, None) for p in subset]
        items_free = client.generate(prompts, temperature=0.0, guided=False)
        res_free = eval_llm_arm(items_free, true_labels, classes, validator=validator)
        results["free_text_regex"] = {"acc": res_free["semantic"]["diag_acc"], "ladder": res_free["structure_ladder"]}
        print(f"[E7] free_text_regex: acc={res_free['semantic']['diag_acc']:.4f} L1={res_free['structure_ladder']['l1_valid']:.4f}")

        save_json({"variants": results, "fewshot": {"medoid_meta": med_meta, "loo": loo, "random_seed_contrast": rsc}},
                  results_dir("E7") / "E7_summary.json")
        print(f"[E7] 完成，结果存 results/E7/E7_summary.json")


if __name__ == "__main__":
    main()