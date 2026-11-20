"""E5：语义压缩消融【核心，W5–W6】(方案 E5)。

T1 五臂：L0 仅标签 / L2 KG 硬描述 / L2e 无关上下文控制 / L3 软描述(Top-3+u) / L4 完全体。
数据：验证集病害图，3 抽样种子 × n=1000；配对铁律：同子集所有臂共用同批样本。
指标：语义层（诊断准确率/token数/建议合理率/纠错率/幻觉率）+ 结构层（三级完整性）
+ 机制层（confidence 校准、u↔conf/正确性 Spearman、高/u 低分组对比）。

用法: python -m src.experiments.E5_compress_ablation --model <repo> --seeds 42
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from ..utils.seeds import get_seeds, llm_inference_seeds
from ..data.datasets import PlantDataset, default_eval_transform
from ..inference.saf_csc_infer import load_saf_ckpt, load_csc_ckpt, saf_infer, csc_infer
from ..inference.pipeline import build_prompt
from ..inference.llm_client import get_llm_client
from ..evaluation.integrity import load_validator
from .common import load_partition, load_pv_classes, results_dir
from .llm_common import sample_subset, eval_llm_arm

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
ARMS = ["L0", "L2", "L2e", "L3", "L4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--arm", default="all", help="all 或单臂名")
    ap.add_argument("--n", type=int, default=1000, help="抽样样本数（小样验证可调小）")
    args = ap.parse_args()

    with ExperimentContext(experiment="E5", description=f"语义压缩消融 model={args.model}",
                           configs={"arms": ARMS, "model": args.model}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        # 仅病害样本
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]
        print(f"[E5] 验证集病害样本 {len(eval_disease)} 张")

        # 加载 CSC/SAF（【修正】原硬编码 seed42：csc_seed42 不存在、saf_seed42 的 u 塌缩
        # → 按用户确认改用 seed43，二者均存在且健康）
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
        probs_map = {_can(x): v for x, v in zip(csc_pred["paths"], csc_pred["probs"].tolist())}
        topk_map = {_can(x): v for x, v in zip(csc_pred["paths"], csc_pred["topk_idx"].tolist())}
        true_map = {_can(x): v for x, v in zip(csc_pred["paths"], csc_pred["targets"])}
        canon_paths = [_can(x) for x in csc_pred["paths"]]

        # KG 描述
        from ..kg.kg import KG
        kg = KG.load(project_path("configs", "kg", "kg.json"))
        kg_desc = kg.descriptions_map()

        client = get_llm_client(args.model).load()
        validator = load_validator()
        arms = [args.arm] if args.arm != "all" else ARMS

        for s_seed in args.seeds:
            subset = sample_subset(eval_disease, n=args.n, seed=s_seed)
            print(f"[E5] 抽样种子 {s_seed}: {len(subset)} 样本")
            for arm in arms:
                prompts, true_labels, us = [], [], []
                top3_loader, csc_top1_correct = [], []
                for raw in subset:
                    p = _can(raw)
                    idx = canon_paths.index(p)
                    top3 = [(classes[topk_map[p][k]], probs_map[p][topk_map[p][k]]) for k in range(3)]
                    u = u_dict[p]
                    fills = _build_prompt(arm, classes, top3, u, kg_desc, p)
                    prompts.append(fills)
                    true_labels.append(classes[true_map[p]])
                    us.append(u)
                    # Top-3 准确率 / 软压缩纠错率所需载荷信息
                    top3_loader.append([c for c, _ in top3])
                    csc_top1_correct.append(top3[0][0] == classes[true_map[p]])
                items = client.generate(prompts, temperature=0.0, guided=True, seed=llm_inference_seeds()[0])
                res = eval_llm_arm(items, true_labels, classes, u_values=np.array(us), validator=validator,
                                   top3_loader=top3_loader, csc_top1_correct=csc_top1_correct)
                # token 数（必报）
                res["input_tokens"] = [len(t.split()) for t in prompts]
                save_json({**res, "arm": arm, "sampling_seed": s_seed},
                          results_dir("E5") / f"E5_{arm}_seed{s_seed}.json")
                # 供 E18/E19：逐样本判定写入 prediction_pool（消除幸存者偏差）
                _dump_pool(items, true_labels, us, arm, s_seed)
                print(f"[E5] seed={s_seed} arm={arm} 诊断准确率={res['semantic']['diag_acc']:.4f} "
                      f"L1={res['structure_ladder']['l1_valid']:.4f} ECE={res['calibration']['ece']}")
        print(f"[E5] 完成，结果存 results/E5/")


def _build_prompt(arm, classes, top3, u, kg_desc, p):
    if arm == "L2e":
        # 等 token 无关农学文本（长度与 kg_desc 拼接量近似）
        filler = " ".join(["土壤保墒与轮作管理是农业生产的重要环节。"] * 20)
        return build_prompt("L2e", classes, top3, u, None, filler=filler)
    return build_prompt(arm, classes, top3, u, kg_desc)


def _dump_pool(items, true_labels, us, arm, s_seed):
    """逐样本判定写入 prediction_pool（供 E18/E19）。标注字段。"""
    from ..evaluation.integrity import validate_record
    from .llm_common import dump_prediction_pool, _match_diag
    from ..utils.io import project_path

    records = []
    for i, it in enumerate(items):
        rec = validate_record(it.raw)
        parsed = rec["parsed"] if isinstance(rec["parsed"], dict) else {}
        llm_correct = _match_diag(parsed.get("diagnosis", ""), true_labels[i])
        records.append({
            "id": f"{arm}_seed{s_seed}_{i}",
            "arm": arm, "sampling_seed": s_seed,
            "entered_llm": True,
            "csc_correct": True,
            "llm_correct": llm_correct,
            "llm_parse_failed": not rec["l1_valid"],
            "field_degraded": rec["l1_valid"] and not rec["l2_non_degraded"],
            "hallucination": False,
            "u": float(us[i]),
        })
    dump_prediction_pool(records, "E5", project_path("results"))


if __name__ == "__main__":
    main()