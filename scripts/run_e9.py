# -*- coding: utf-8 -*-
"""E9：系统级对比 + 增量消融主表（方案 E9；R2–R4 本机组装，R0/R1/延迟实测留云上）。

组件落地（本文件实现 R2–R4）：
  增量行 R0 全图+MLLM → R1 +SAF → R2 +硬压缩(L2) → R3 +软压缩(L3) → R4 完全体(L4)。
  图像模型权重统一 seed43（saf_seed43/csc_seed43），抽样种子 42/43/44（配对铁律）。
  SAF 拦/放信号＝病害概率 p(probs[:,1])：p≤τ* → 判健康(本地不发云, 病害记为F2误拦)；
  p>τ* → 判病害放行上传云端。(注：E1 过滤层用的是病害概率，非 EDL u；u 属 E12 拒识路由)
  R2–R4 放行样本的诊断对/错复用 E5 逐样本预测池 prediction_pool/E5.jsonl（零新增 LLM）。

  R0/R1(Qwen2-VL，云上新增) 与 netem T3 实测、B.3 成本假设值 → 标"待云上/待核定"，本文件留接口。

用法:
  冒烟: python scripts/run_e9.py --dry --n 6
  正式: python scripts/run_e9.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.utils.experiment_snapshot import ExperimentContext  # noqa: E402
from src.utils.io import load_yaml, load_json, project_path, save_json  # noqa: E402
from src.data.datasets import PlantDataset, default_eval_transform  # noqa: E402
from src.inference.saf_csc_infer import load_saf_ckpt, load_csc_ckpt, saf_infer, csc_infer  # noqa: E402
from src.inference.pipeline import build_prompt  # noqa: E402
from src.kg.kg import KG  # noqa: E402
from src.experiments.common import load_partition, load_pv_classes, results_dir  # noqa: E402
from src.experiments.llm_common import sample_subset  # noqa: E402

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
SEEDS = [42, 43, 44]
N = 1000
IMG_CKPT_SEED = 43            # E5/E8 统一图像模型权重种子
# E1 seed43 scan 冻结的 τ*（漏检≤1% 约束），来源 results/E1/E1_summary.json
TAU_SOURCE_PATH = project_path("results", "E1", "E1_summary.json")

# —— R 行 → E5 臂 ——
R2_ARM, R3_ARM, R4_ARM = "L2", "L3", "L4"


def load_pool() -> dict:
    """E5 逐样本池，按 (arm, sampling_seed, idx) 覆盖（与 e7_stats 同规则）。"""
    recs = {}
    for line in (project_path("results", "prediction_pool", "E5.jsonl")
                 .read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["arm"] not in (R2_ARM, R3_ARM, R4_ARM):
            continue
        idx = int(r["id"].rsplit("_", 1)[-1])
        recs[(r["arm"], int(r["sampling_seed"]), idx)] = r
    return recs


def load_tau_star() -> float:
    d = load_json(TAU_SOURCE_PATH)
    t = d["per_seed"]["43"]["tau_star"]
    print(f"[E9] τ* = {t}（E1 seed43，漏检≤1% 约束）")
    return float(t)


def run_cell(client_paths, classes, class_to_idx, root, kg_desc, pool, tau,
             subsets: dict[int, list], health_paths, saf, csc):
    """对 3 抽样子集病害 + 全健康做 R2–R4 系统组装。"""
    # —— 健康层（用 saf_seed43 全集合判一次：病害概率 p=probs[:,1]，p≤τ* → 判健康不发云）——
    h_loader = _loader(root, health_paths, class_to_idx)
    h_saf = saf_infer(saf, h_loader)
    h_p = h_saf["probs"][:, 1].tolist()        # 病害概率（E1 过滤口径，非 EDL u）
    healthy_pass = sum(1 for p_ in h_p if p_ <= tau)   # p≤τ* → 判健康(不发云, 正确)
    acc_healthy = healthy_pass / len(h_p) if h_p else None
    n_healthy = len(h_p)

    # —— 病害抽样子集逐样本系统流 ——
    per_seed = {}          # seed → 聚合
    per_seed_cells = {}    # seed → {arm: {"acc","call_rate","bytes","l1","l2","n"}}
    for seed, paths in subsets.items():
        loader = _loader(root, paths, class_to_idx)
        s_saf = saf_infer(saf, loader)
        s_csc = csc_infer(csc, loader, topk=3)
        p_by = dict(zip(s_saf["paths"], s_saf["probs"][:, 1].tolist()))  # 病害概率(E1过滤口径)
        u_by = dict(zip(s_saf["paths"], s_saf["u"].tolist()))            # 供字节构造
        top3_by = {p: (s_csc["topk_idx"][i].tolist(), s_csc["probs"][i].tolist())
                   for i, p in enumerate(s_csc["paths"])}
        cell = {}
        for arm in (R2_ARM, R3_ARM, R4_ARM):
            acc = cloud = tot_b = l1 = l2 = n_cloud = n_diag = 0
            for i, p in enumerate(paths):
                if p_by[p] <= tau:                # p≤τ* → 判健康 → 病害按误拦(F2)记错，0 云端
                    n_diag += 1
                    continue
                cloud += 1                        # p>τ* → 判病害 → 放行上传云端
                r = pool.get((arm, seed, i))
                # 字节：放行样本实际发送的文本载荷
                pt = _load_bytes(arm, p, classes, top3_by, kg_desc)
                tot_b += pt
                if r is None:                        # 池缺（seed42 污染边界兜底）
                    n_diag += 1
                    continue
                acc += int(r["llm_correct"])
                l1 += int(not r.get("llm_parse_failed", False))
                l2 += int(not r.get("field_degraded", False))
                n_cloud += 1
                n_diag += 1
            nd = len(paths)
            cell[arm] = {
                "acc": acc / nd,                     # 系统准确率（被拦病害记错）
                "acc_cloud": (acc / n_cloud) if n_cloud else None,  # 仅放行样本的 LLM 对率（参考）
                "call_rate": cloud / nd,
                "bytes_per_image": (tot_b / cloud) if cloud else None,
                "l1": (l1 / n_cloud) if n_cloud else None,
                "l2": (l2 / n_cloud) if n_cloud else None,
                "n_disease": nd, "n_cloud": n_cloud,
            }
        per_seed[seed] = cell
        per_seed_cells[seed] = cell
    return {
        "healthy": {"n": n_healthy, "acc_healthy": acc_healthy, "tau": tau},
        "per_seed": per_seed_cells,
        "seeds": SEEDS,
    }


def _loader(root, paths, class_to_idx):
    return DataLoader(PlantDataset(root, transform=default_eval_transform(),
                                   sample_paths=[Path(p) for p in paths], class_to_idx=class_to_idx),
                      batch_size=64, shuffle=False, num_workers=0)


def _load_bytes(arm, p, classes, top3_by, kg_desc):
    if arm == R2_ARM:
        return len(build_prompt(R2_ARM, classes, None, None, kg_desc).encode("utf-8"))
    topk, probs = top3_by[p]
    top3 = [(classes[k], probs[k]) for k in topk[:3]]
    ck = {R3_ARM: (None, None), R4_ARM: (kg_desc,)}[arm]
    return len(build_prompt(arm, classes, top3, None, *ck).encode("utf-8"))


def aggregate(parts, n_total):
    """跨 3 种子均值。n_total=验证集样本总数（供加权综合，可选）。"""
    for arm in (R2_ARM, R3_ARM, R4_ARM):
        accs = [parts["per_seed"][s][arm]["acc"] for s in parts["seeds"]]
        crs = [parts["per_seed"][s][arm]["call_rate"] for s in parts["seeds"]]
        b1 = [b for s in parts["seeds"] for b in [parts["per_seed"][s][arm]["bytes_per_image"]] if b]
        l1s = [v for s in parts["seeds"] for v in [parts["per_seed"][s][arm]["l1"]] if v is not None]
        parts["per_seed"][arm] = {
            "acc_mean": sum(accs) / len(accs),
            "call_rate_mean": sum(crs) / len(crs),
            "bytes_per_image_mean": (sum(b1) / len(b1)) if b1 else None,
            "l1_mean": (sum(l1s) / len(l1s)) if l1s else None,
        }
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()

    with ExperimentContext(experiment="E9", description="系统增量主表(R2-R4 本机组装)",
                           configs={"seeds": SEEDS, "img_seed": IMG_CKPT_SEED, "arms": [R2_ARM, R3_ARM, R4_ARM]}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_all = part["eval"]
        health_paths = [p for p in eval_all if "healthy" in p.lower()]
        eval_disease = [p for p in eval_all if "healthy" not in p.lower()]

        tau = load_tau_star()
        pool = load_pool()
        saf = load_saf_ckpt(project_path("results", "train_saf", f"saf_seed{IMG_CKPT_SEED}.pt"), n_classes=2)
        csc = load_csc_ckpt(project_path("results", "train_csc", f"csc_seed{IMG_CKPT_SEED}.pt"), n_classes=len(classes))
        kg = KG.load(project_path("configs", "kg", "kg.json"))
        kg_desc = kg.descriptions_map()

        subsets = {seed: sample_subset(eval_disease, n=args.n, seed=seed) for seed in SEEDS}
        parts = run_cell(None, classes, class_to_idx, root, kg_desc, pool, tau, subsets, health_paths, saf, csc)
        parts = aggregate(parts, len(eval_all))

        save_json(parts, results_dir("E9") / "E9_assembly.json")
        # Table 7（R2–R4 实测；R0/R1 待云上）
        L = _build_table7(parts)
        out = results_dir("E9") / "table_e9.md"
        out.write_text("\n".join(L), encoding="utf-8")
        print("\n".join(L))
        print(f"\n[完成] E9 组装结果已写入 results/E9/（R0/R1/延迟/成本待云上）")
        if args.dry:
            print("[dry] 冒烟结束")


def _build_table7(parts):
    L = []
    L.append("# E9 系统级增量消融 —— Table 7（R2–R4 本机；R0/R1 待云上）")
    L.append("")
    L.append(f"> 图像模型 saf/csc_seed{IMG_CKPT_SEED}；抽样种子 {SEEDS}；τ*={parts['healthy']['tau']}（E1 seed43）")
    L.append("> 系统准确率口径：SAF 用病害概率 p(probs[:,1])，p≤τ* → 判健康(病害按误拦记错)；"
            "p>τ* → 放行传云，诊断沿 E5 池。")
    L.append("")
    L.append("| R行 | 载荷 | 病害系统Acc | 云调用率 | 字节/图 | L1(放行) |")
    L.append("|---|---|---|---|---|---|")
    for arm in (R2_ARM, R3_ARM, R4_ARM):
        r = parts["per_seed"][arm]
        fm = lambda x: "—" if x is None else f"{x:.4f}"
        tag = "硬压缩" if arm == R2_ARM else ("软压缩" if arm == R3_ARM else "完全体")
        L.append(f"| R{'2' if arm==R2_ARM else '3' if arm==R3_ARM else '4'} | {tag}({arm}) "
                 f"| {r['acc_mean']:.4f} | {r['call_rate_mean']:.4f} "
                 f"| {fm(r['bytes_per_image_mean'])} | {fm(r['l1_mean'])} |")
    L.append("| R0 | 全图+Qwen2-VL | 待云上 | 上限1.0 | 原图实测 | 待云上 |")
    L.append("| R1 | +SAF(异常仍传图) | 待云上 | 待云上 | 原图实测 | 待云上 |")
    L.append("")
    L.append(f"健康层：SAF 对验证集健康样本拦截(判健康)比例 = "
             f"{parts['healthy']['acc_healthy']:.4f}（n={parts['healthy']['n']}）")
    L.append("")
    L.append("> 延迟(T2/B.4)、每千张成本(B.3)、R0/R1 Acc 与 T3 netem 实测 → 待云上后再并入本表。")
    return L


if __name__ == "__main__":
    main()