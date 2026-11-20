"""reproduce_minimal 端到端 Demo（S2，CPU 可跑）。

加载若干 PV 图 → SAF 过滤 → CSC 分类 → 构建 L4 载荷 →（可选）云端 LLM 诊断。
CPU 5 分钟跑通。LLM 部分可选（无 GPU/vLLM 时跳过，仅输出压缩载荷）。

用法: python -m src.experiments.reproduce_minimal_demo --images data/minimal --csc ... --saf ... --kg ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..utils.io import load_yaml, project_path
from ..data.datasets import PlantDataset, default_eval_transform
from ..inference.saf_csc_infer import load_saf_ckpt, load_csc_ckpt, saf_infer, csc_infer
from ..inference.pipeline import build_prompt
from ..kg.kg import KG
from .common import load_pv_classes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="reproduce_minimal/data")
    ap.add_argument("--csc", default="reproduce_minimal/weights/csc_seed43.pt")
    ap.add_argument("--saf", default="reproduce_minimal/weights/saf_seed43.pt")
    ap.add_argument("--kg", default="reproduce_minimal/kg.json")
    ap.add_argument("--llm", default=None, help="可选：LLM 模型目录")
    args = ap.parse_args()

    classes = load_pv_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    img_dir = Path(args.images)
    n_imgs = sum(1 for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if n_imgs == 0:
        print(f"[reproduce] data 目录为空（{img_dir}）：请按 data/README.txt 放入 10 张 PV 图（需确认再分发许可）后再运行。")
        return
    print(f"[reproduce] 读取 {n_imgs} 张图")
    ds = PlantDataset(img_dir, transform=default_eval_transform(), class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    saf = load_saf_ckpt(Path(args.saf), n_classes=2)
    csc = load_csc_ckpt(Path(args.csc), n_classes=len(classes))
    kg_path = Path(args.kg)
    if not kg_path.is_absolute() and not kg_path.exists():
        kg_path = project_path(args.kg)
    kg = KG.load(kg_path)
    kg_desc = kg.descriptions_map()

    sp = saf_infer(saf, loader)
    cp = csc_infer(csc, loader, topk=5)
    print("=" * 60)
    for i in range(len(sp["paths"])):
        u = sp["u"][i].item()
        top3 = [(classes[cp["topk_idx"][i][k].item()], cp["probs"][i][cp["topk_idx"][i][k]].item()) for k in range(3)]
        routed = "拒识转人工" if u > 0.5 else "进正常流水线"
        print(f"[{Path(sp['paths'][i]).name}] u={u:.3f} → {routed}")
        print(f"    Top-3: {[(c, round(p,3)) for c,p in top3]}")
        print(f"    L4 载荷token数: {len(build_prompt('L4', classes, top3, u, kg_desc).split())}")
    print("=" * 60)
    if args.llm:
        from ..inference.llm_client import get_llm_client
        client = get_llm_client(args.llm).load()
        prompts = []
        for i in range(len(sp["paths"])):
            u = sp["u"][i].item()
            top3 = [(classes[cp["topk_idx"][i][k].item()], cp["probs"][i][cp["topk_idx"][i][k]].item()) for k in range(3)]
            prompts.append(build_prompt("L4", classes, top3, u, kg_desc))
        items = client.generate(prompts, temperature=0.0, guided=True)
        for it in items:
            print("LLM 诊断:", (it.parsed or {}).get("diagnosis"))


if __name__ == "__main__":
    main()