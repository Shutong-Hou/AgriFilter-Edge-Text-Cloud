# -*- coding: utf-8 -*-
"""E17 vLLM 版（云端 Linux，同引擎 FP16 vs AWQ）。

【为什么换 vLLM】本地/云端 transformers+outlines 路径：本地与 neurovein 抢卡、
云端 gptqmodel+outlines 的 AWQ 卡 CPU 病态地慢，且与 FP16 的 transformers 引擎不同，
d_pp 混入引擎差异。改走 **同一 vLLM 引擎**：xgrammar guided JSON、温度=0，
FP16 与 AWQ 同引擎、同 L4 载荷、同 seed42 子集 → 干净可复现的 d_pp；吞吐也可信，
且 vLLM 即 E11 所用引擎，一趟打通。

口径：与 run_e17.py 完全一致（build_l4_prompts / _match_diag / extract_diag_primary），
仅把生成后端从 transformers+outlines 换成 vLLM(CloudLLMClient guided)。子集/载荷/评估零改动。

依赖（云端）：pip install vllm transformers torch qwen-vl-utils pillow "numpy<2"
用法：
  python scripts/run_e17_vllm.py --model-fp16 models/Qwen2.5-7B-Instruct \
      --model-awq models/Qwen2.5-7B-Instruct-AWQ --only fp16 --n 1000
  python scripts/run_e17_vllm.py --model-awq models/Qwen2.5-7B-Instruct-AWQ --only awq --n 1000
输出: results/E17/E17_vllm_{fp16|awq}_seed42.json + summary（d_pp 用两侧拼接）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

# 复用 L4 载荷构建与评估（不触发 run_e17 的 main）
from run_e17 import (  # noqa: E402
    build_l4_prompts, _load_candidates, extract_diag_primary,
    FP16_DEFAULT, AWQ_DEFAULT, N, SEED, P_ARM,
)
from src.evaluation.integrity import load_validator, validate_record, ladder_summary  # noqa: E402
from src.utils.io import load_yaml, load_json, project_path, save_json, ensure_dir  # noqa: E402
from src.utils.experiment_snapshot import ExperimentContext  # noqa: E402
from src.experiments.common import load_partition, load_pv_classes, results_dir  # noqa: E402
from src.experiments.llm_common import sample_subset, _match_diag, _action_ok  # noqa: E402

DATA_CFG = load_yaml(project_path("configs", "dataset_config.yaml"))
SCHEMA = json.dumps(load_json(project_path("configs", "schema.json")))

try:
    from vllm import LLM, SamplingParams
    # vLLM >= 0.12：guided_json 已移除，改用 structured_outputs=StructuredOutputsParams(json=...)
    from vllm.sampling_params import StructuredOutputsParams as SOP
    VLLM_OK = True
except Exception as e:  # pragma: no cover
    VLLM_OK = False
    _VLLM_ERR = e


def vllm_client(model_repo):
    if not VLLM_OK:
        raise RuntimeError(f"vLLM 不可用（云端需 pip install vllm）：{_VLLM_ERR}")
    llm = LLM(model=model_repo, max_model_len=8192, gpu_memory_utilization=0.90,
              trust_remote_code=True)  # AWQ 由 config.json quantization 自动识别
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512, seed=SEED,
                        structured_outputs=SOP(json=SCHEMA))
    return llm, sp


def gen_and_eval(model_repo, prompts, true_labels, classes, candidates, label):
    llm, sp = vllm_client(model_repo)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=True)
    wall = time.perf_counter() - t0

    items = [o.outputs[0].text for o in outs]
    n = len(items)
    validator = load_validator()
    CFG = load_yaml(project_path("configs", "eval_config.yaml"))["integrity_ladder"]
    placeholders, action_min = set(CFG["placeholders"]), CFG["action_min_len"]
    records = []
    c_primary, cj, conf, advise = [], [], [], 0
    for i, raw in enumerate(items):
        rec = validate_record(raw, validator)
        records.append(rec)
        tl = true_labels[i]
        diag_p = extract_diag_primary(raw, rec["parsed"], candidates)
        c_primary.append(int(_match_diag(diag_p, tl)))
        cj.append(int(isinstance(rec["parsed"], dict) and _match_diag(rec["parsed"].get("diagnosis", ""), tl)))
        rc = rec["parsed"].get("confidence") if isinstance(rec["parsed"], dict) else None
        conf.append(rc if (rc is not None and 0 <= float(rc) <= 1) else np.nan)
        advise += int(isinstance(rec["parsed"], dict) and _action_ok(rec["parsed"].get("action"), placeholders, action_min))
    ladder = ladder_summary(records, n)
    valid_idx = [i for i, r in enumerate(records) if r["l1_valid"]]
    cp = np.array(c_primary, float)
    res = {
        "quant": label, "engine": "vllm", "arm": P_ARM, "n": n, "seed": SEED,
        "semantic": {
            "diag_acc_primary": float(cp.mean()),
            "diag_acc_strict_json": float(np.array(cj, float).mean()),
            "diag_acc_primary_valid": float(cp[valid_idx].mean()) if valid_idx else None,
            "json_parse_rate": sum(isinstance(r["parsed"], dict) for r in records) / n,
            "advise_rate": advise / n,
        },
        "structure_ladder": ladder,
        "perf": {"wall_s": wall, "throughput_img_s": n / wall, "per_img_s": wall / n},
    }
    # 逐样本 raw 落盘（干净覆写）
    out_dir = ensure_dir(results_dir("E17") / "raw")
    with open(out_dir / f"E17_vllm_{label}_seed{SEED}.jsonl", "w", encoding="utf-8") as f:
        for i, raw in enumerate(items):
            f.write(json.dumps({"id": f"vllm_{label}_{i}", "quant": label, "engine": "vllm",
                                "seed": SEED, "raw_output": raw, "true_label": true_labels[i]},
                               ensure_ascii=False) + "\n")
    del llm
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-fp16", default=FP16_DEFAULT)
    ap.add_argument("--model-awq", default=AWQ_DEFAULT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--only", choices=["fp16", "awq"], default=None)
    args = ap.parse_args()

    with ExperimentContext("E17", description="E17 vLLM 同引擎 FP16 vs AWQ (L4)",
                           configs={"model_fp16": args.model_fp16, "model_awq": args.model_awq,
                                    "seed": args.seed, "n": args.n, "engine": "vllm"}) as ctx:
        part = load_partition()
        classes = load_pv_classes()
        c2i = {c: i for i, c in enumerate(classes)}
        candidates = _load_candidates(classes)
        root = Path(DATA_CFG["datasets"]["plantvillage"]["dir"])
        eval_disease = [p for p in part["eval"] if "healthy" not in p.lower()]
        subset = sample_subset(eval_disease, n=args.n, seed=args.seed)
        true_labels = [classes[c2i[Path(p).parent.name]] for p in subset]
        from src.kg.kg import KG
        kg_desc = KG.load(project_path("configs", "kg", "kg.json")).descriptions_map()

        print(f"[E17_vllm] 构建 R4 同款 L4 prompt ({len(subset)}) ...", flush=True)
        prompts = build_l4_prompts(subset, classes, c2i, root, kg_desc)
        results = {}
        if args.only in (None, "fp16"):
            results["fp16"] = gen_and_eval(args.model_fp16, prompts, true_labels, classes, candidates, "fp16")
            save_json(results["fp16"], results_dir("E17") / f"E17_vllm_fp16_seed{args.seed}.json")
            print(f"[E17_vllm] FP16 acc={results['fp16']['semantic']['diag_acc_primary']:.4f} "
                  f"thr={results['fp16']['perf']['throughput_img_s']:.3f}/s")
        if args.only in (None, "awq"):
            results["awq"] = gen_and_eval(args.model_awq, prompts, true_labels, classes, candidates, "awq")
            save_json(results["awq"], results_dir("E17") / f"E17_vllm_awq_seed{args.seed}.json")
            print(f"[E17_vllm] AWQ acc={results['awq']['semantic']['diag_acc_primary']:.4f} "
                  f"thr={results['awq']['perf']['throughput_img_s']:.3f}/s")
        if set(results) >= {"fp16", "awq"}:
            summary = {
                "engine": "vllm", "seed": args.seed, "n": args.n,
                "acc": {k: v["semantic"]["diag_acc_primary"] for k, v in results.items()},
                "acc_diff_awq_minus_fp16_pp": (results["awq"]["semantic"]["diag_acc_primary"]
                                               - results["fp16"]["semantic"]["diag_acc_primary"]) * 100.0,
                "thr": {k: v["perf"]["throughput_img_s"] for k, v in results.items()},
            }
            save_json(summary, results_dir("E17") / "E17_vllm_summary.json")
            print(f"[E17_vllm] d_pp(AWQ-FP16)={summary['acc_diff_awq_minus_fp16_pp']:+.3f} pp")


if __name__ == "__main__":
    main()