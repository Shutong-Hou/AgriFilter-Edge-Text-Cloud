# -*- coding: utf-8 -*-
"""审稿意见 R1-4：KG 38 类全覆盖后的 L2 / L4 重跑。

背景：现有 KG 仅覆盖 10/38 类，L2/L4 载荷里其余 28 类为空串。
      本实验把 KG 补到 38/38（configs/kg/kg_full_38.json），重跑 L2/L4，
      看「KG 覆盖不足」是否真的是 L2/L4 落后的原因。

严格同口径：csc_seed43 / saf_seed43、同 seed 子集、同 Qwen2.5-7B-Instruct、
同 guided JSON 解码、temperature=0、同一 build_prompt 与 _match_diag。
唯一变量：KG 描述来自 38 类全覆盖版。

用法：
  python scripts/review_r1_kg_full.py --arm L4 --seeds 42 43
  python scripts/review_r1_kg_full.py --arm L2 --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from src.data.datasets import default_eval_transform  # noqa: E402
from src.inference.saf_csc_infer import csc_infer, load_csc_ckpt, load_saf_ckpt, saf_infer  # noqa: E402
from src.inference.pipeline import build_prompt  # noqa: E402

DATA_ROOT = Path(os.environ.get("AGRI_DATA_ROOT", ROOT / "data"))
OUT = ROOT / "results" / "review_r1"


def resolve(p: str) -> Path:
    parts = Path(p).parts
    for marker in ("plantvillage", "plantdoc", "cassava"):
        if marker in parts:
            return DATA_ROOT.joinpath(*parts[parts.index(marker):])
    return Path(p)


class PathListDataset(Dataset):
    def __init__(self, paths: list[Path], class_to_idx: dict[str, int]):
        self.paths = paths
        self.targets = [class_to_idx[p.parent.name] for p in paths]
        self.tf = default_eval_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        img = Image.open(p)
        try:
            img.draft("RGB", (256, 256))
            img.load()
        except Exception:
            pass
        if max(img.size) > 2560:
            img.thumbnail((2560, 2560), Image.LANCZOS)
        return self.tf(img.convert("RGB")), self.targets[i], str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arm", nargs="+", default=["L4"], choices=["L2", "L4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--kg", default=str(ROOT / "configs" / "kg" / "kg_full_38.json"))
    ap.add_argument("--tag", default="fullkg", help="输出文件名中的实验标签（如完整 KG=fullkg，原 KG=orig）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用，输出加 _smokeN 后缀）")
    args = ap.parse_args()

    from src.experiments.common import load_partition, load_pv_classes  # noqa: E402
    from src.experiments.llm_common import _match_diag, sample_subset  # noqa: E402
    from src.kg.kg import KG  # noqa: E402
    from src.inference.llm_client import get_llm_client  # noqa: E402
    from src.utils.seeds import llm_inference_seeds  # noqa: E402

    part = load_partition()
    classes = load_pv_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]

    kg_desc = KG.load(Path(args.kg)).descriptions_map()
    filled = sum(1 for c in classes if kg_desc.get(c, "").strip())
    print(f"[KG-full] {Path(args.kg).name}：{filled}/{len(classes)} 类有描述", flush=True)

    subsets = {s: sample_subset(eval_disease, n=args.n, seed=s) for s in args.seeds}
    need = sorted({p for v in subsets.values() for p in v})
    cache = OUT / f"evidence_cache_n{args.n}.npz"
    if not cache.exists():
        raise SystemExit(f"缺少证据缓存 {cache}，请先跑 review_r1_l3_nou.py --evidence-only")
    z = np.load(cache, allow_pickle=True)
    probs, topk, u_arr, targets = z["probs"], z["topk"], z["u"], z["targets"]
    cache_paths = [str(x) for x in z["paths"]]
    evidence = {p: i for i, p in enumerate(cache_paths)}
    miss = [p for p in need if p not in evidence]
    assert not miss, f"证据缓存缺少 {len(miss)} 张（缓存为 seeds 42/43 并集，改 seeds 需重算）"

    client = get_llm_client(args.model).load()
    gen_seed = llm_inference_seeds()[0]

    for arm in args.arm:
        for s_seed in args.seeds:
            sfx = f"_smoke{args.limit}" if args.limit else ""
            out_json = OUT / f"{arm}_{args.tag}_seed{s_seed}{sfx}.json"
            if out_json.exists():
                print(f"[KG-full] {arm} seed{s_seed} 已完成，跳过", flush=True)
                continue
            seq = subsets[s_seed][: args.limit] if args.limit else subsets[s_seed]
            prompts, true_labels = [], []
            for raw in seq:
                i = evidence[raw]
                top3 = [(classes[topk[i][k]], float(probs[i][topk[i][k]])) for k in range(3)]
                u = float(u_arr[i])
                if arm == "L2":
                    prompts.append(build_prompt("L2", classes, kg_descriptions=kg_desc))
                else:
                    prompts.append(build_prompt("L4", classes, top3, u, kg_desc))
                true_labels.append(classes[targets[i]])

            tok = float(np.mean([len(t.split()) for t in prompts]))
            print(f"[KG-full] {arm} seed{s_seed} n={len(prompts)} 平均input_tokens={tok:.1f}", flush=True)
            t0 = time.time()
            items = client.generate(prompts, temperature=0.0, guided=True, seed=gen_seed)
            dt = time.time() - t0

            correct = []
            raw_path = OUT / f"{arm}_{args.tag}_raw_seed{s_seed}{sfx}.jsonl"
            with raw_path.open("w", encoding="utf-8") as f:
                for k, it in enumerate(items):
                    diag = ""
                    try:
                        diag = json.loads(it.raw).get("diagnosis", "")
                    except Exception:
                        pass
                    ok = _match_diag(diag, true_labels[k])
                    correct.append(ok)
                    f.write(json.dumps({"i": k, "true_label": true_labels[k], "diag": diag,
                                        "correct": bool(ok), "raw": it.raw}, ensure_ascii=False) + "\n")

            n_ok = int(sum(correct))
            res = {"arm": f"{arm}-fullKG", "sampling_seed": s_seed, "n": len(correct),
                   "correct": n_ok, "diag_acc": n_ok / len(correct),
                   "mean_input_tokens": tok, "seconds": dt, "model": args.model,
                   "kg_file": Path(args.kg).name, "kg_covered": filled}
            out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[KG-full] {arm} seed{s_seed} diag_acc={res['diag_acc']:.4f} ({n_ok}/{len(correct)}) "
                  f"{dt:.0f}s", flush=True)

    print("[KG-full] 全部完成", flush=True)


if __name__ == "__main__":
    main()
