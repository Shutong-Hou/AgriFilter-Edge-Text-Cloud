# -*- coding: utf-8 -*-
"""R1-2 加固：给 Qwen2-VL 全图基线"最好的机会"——3-shot 与 CoT 两种强提示。

与 E9 R0 完全同批样本（seed42 的 1000 张病害图）、同一模型、同一 JSON schema、同一判定口径，
只改变提示强度，用来回答"7.37% 是不是提示没给够"。

用法（agrifilter_vl 环境）:
  python scripts/review_r1_vl_hardened.py --cond fewshot3 --limit 20      # 冒烟
  python scripts/review_r1_vl_hardened.py --cond fewshot3                # 正式
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.environ.get("AGRI_DATA_ROOT", ROOT / "data"))

JSON_INSTR = (
    "请仅输出一个 JSON 对象，字段：diagnosis(病害类别名，从给定候选选一个)、"
    "action(建议措施)、reasoning(简短依据)、confidence(0到1的置信度)。不要输出其他文字。"
)
COT_INSTR = (
    "请先分步观察并描述叶片的病斑特征（颜色、形状、边缘、分布），"
    "然后在最后输出且仅输出一个 JSON 对象，字段："
    "diagnosis(病害类别名，必须从给定候选类中选一个)、action(建议措施)、"
    "reasoning(简短依据)、confidence(0到1的置信度)。JSON 必须位于回答末尾。"
)


def norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


try:  # 与项目完全同一判定函数
    from src.experiments.llm_common import _match_diag  # type: ignore

    print("[R1-2VL] 使用项目 _match_diag 口径")
except Exception as e:  # pragma: no cover
    print(f"[R1-2VL] 项目判定函数导入失败({e})，使用等价内联实现")

    def _match_diag(diag, true_class):
        d, tc = str(diag).strip().lower(), str(true_class).strip().lower()
        if not d:
            return False
        return tc in d or d in tc


def local(p: str) -> Path:
    parts = Path(p).parts
    for marker in ("plantvillage", "plantdoc", "cassava"):
        if marker in parts:
            return DATA_ROOT.joinpath(*parts[parts.index(marker):])
    return Path(p)


def parse_diag(raw: str, last: bool = False) -> tuple[str, bool]:
    """从原始输出提取 diagnosis；last=True 时取最后一个 JSON 对象（CoT 用）。"""
    blocks = []
    i = 0
    while True:
        a = raw.find("{", i)
        if a < 0:
            break
        b = raw.rfind("}")
        if b <= a:
            break
        blocks.append(raw[a:b + 1])
        i = a + 1
        if not last:
            break
    cands = blocks if last else blocks[:1]
    for blk in (reversed(cands) if last else cands):
        try:
            d = json.loads(blk)
            if isinstance(d, dict) and d.get("diagnosis"):
                return str(d["diagnosis"]), True
        except Exception:
            pass
    m = re.findall(r'"diagnosis"\s*:\s*"([^"]*)"', raw)
    if m:
        return m[-1], True
    return "", False


def build_msgs(cond, classes, exemplars, query_img):
    cls_line = "候选类：" + ", ".join(classes)
    if cond == "fewshot3":
        content = []
        for img, label in exemplars:
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"该图的真实类别是：{label}"})
        content.append({"type": "image", "image": query_img})
        content.append({"type": "text", "text": f"{cls_line}\n请判断最后这张图。{JSON_INSTR}"})
        return [{"role": "user", "content": content}]
    # cot
    return [{"role": "user", "content": [
        {"type": "image", "image": query_img},
        {"type": "text", "text": f"{cls_line}\n{COT_INSTR}"},
    ]}]


def pick_exemplars(part, classes):
    """从 train 划分取 3 个不同作物的示例（不触碰 eval，避免泄漏）。"""
    want = ["Tomato___Late_blight", "Corn_(maize)___Common_rust_", "Apple___Apple_scab"]
    out = []
    from PIL import Image
    for cls in want:
        hit = next((p for p in part["train"] if Path(p).parent.name == cls), None)
        if hit is None:
            continue
        q = local(hit)
        out.append((Image.open(q).convert("RGB"), cls))
    print(f"[R1-2VL] exemplars: {[c for _, c in out]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", choices=["fewshot3", "cot"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    part = json.loads((ROOT / "results" / "E0" / "partition.json").read_text(encoding="utf-8"))
    classes = json.loads((ROOT / "results" / "E0" / "pv_classes.json").read_text(encoding="utf-8"))
    dis = [p for p in part["eval"] if "healthy" not in p.lower()]
    subset = random.Random(args.seed).sample(dis, 1000)
    if args.limit:
        subset = subset[: args.limit]
    print(f"[R1-2VL] cond={args.cond} seed={args.seed} n={len(subset)}")

    model_dir = args.model
    print(f"[R1-2VL] model={model_dir}")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForVision2Seq.from_pretrained(model_dir, torch_dtype="auto",
                                                   device_map="auto")
    model.eval()

    exemplars = pick_exemplars(part, classes) if args.cond == "fewshot3" else []
    max_new = args.max_new or (768 if args.cond == "cot" else 512)

    outdir = ROOT / "results" / "review_r1" / f"vl_{args.cond}"
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / f"raw_seed{args.seed}.jsonl"

    done = set()
    if raw_path.exists() and not args.limit:
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
        print(f"[R1-2VL] 断点续跑：已完成 {len(done)}")

    ok = n = 0
    t0 = time.time()
    todo = [(i, p) for i, p in enumerate(subset) if i not in done and local(p).exists()]
    bs = args.batch
    with open(raw_path, "a", encoding="utf-8") as f:
        for s0 in range(0, len(todo), bs):
            chunk = todo[s0:s0 + bs]
            texts, imgs, metas = [], [], []
            for i, p in chunk:
                img = Image.open(local(p)).convert("RGB")
                msgs = build_msgs(args.cond, classes, exemplars, img)
                texts.append(processor.apply_chat_template(msgs, add_generation_prompt=True))
                if args.cond == "fewshot3":
                    imgs.extend([im for im, _ in exemplars] + [img])
                else:
                    imgs.append(img)
                metas.append((i, p, Path(p).parent.name))
            inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=processor.tokenizer.pad_token_id)
            gens = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for (i, p, true), gen in zip(metas, gens):
                gen = gen.strip()
                diag, parsed = parse_diag(gen, last=(args.cond == "cot"))
                correct = bool(_match_diag(diag, true)) if parsed else False
                ok += int(correct)
                n += 1
                f.write(json.dumps({"i": i, "path": p, "true_label": true, "diag": diag,
                                    "parsed": parsed, "correct": correct, "raw": gen},
                                   ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"  {n}  acc={ok/max(n,1):.4f}  {el:.0f}s ({el/max(n,1):.2f}s/img)", flush=True)

    print(f"[R1-2VL] 完成 {args.cond}: {ok}/{n} = {ok/max(n,1):.4f}  用时 {time.time()-t0:.0f}s")
    (outdir / f"summary_seed{args.seed}.json").write_text(json.dumps(
        {"cond": args.cond, "seed": args.seed, "n": n, "correct": ok,
         "acc": ok / max(n, 1), "max_new_tokens": max_new,
         "prompt": (JSON_INSTR if args.cond == "fewshot3" else COT_INSTR)},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
