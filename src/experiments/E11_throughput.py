"""E11：并发吞吐【W9–W10】(方案 E11)。

vLLM benchmark_serving，λ∈{1,2,4,8,16,32}（T2:64），每档 ≥3 次重复报 mean±std（N12），单次 ≥60s。
C-A 同精度档：1×VL-7B FP16 vs 1×文本7B FP16；C-B 显存等价档：1×VL FP16 vs 2×文本7B AWQ-Int4；
C-C 成本换算（纯算术，B.3）。强制并报 C-B 附 E17 实测 AWQ-FP16 准确率差。

用法: python -m src.experiments.E11_throughput --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from .common import results_dir

LAMBDAS = [1, 2, 4, 8, 16, 32]
REPEATS = 3
DURATION_S = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--backend", default="vllm")
    ap.add_argument("--port", default=8000)
    args = ap.parse_args()

    with ExperimentContext(experiment="E11", description=f"并发吞吐 {args.model}", configs={"lambdas": LAMBDAS}) as ctx:
        results = {}
        for lam in LAMBDAS:
            runs = []
            for rep in range(REPEATS):
                out = _bench_run(args.model, args.backend, args.port, lam)
                runs.append(out)
                print(f"[E11] λ={lam} rep={rep}: throughput={out.get('throughput', float('nan'))} req/s")
            # mean±std
            th = np.array([r.get("throughput", float("nan")) for r in runs])
            lt = np.array([r.get("latency_p50", float("nan")) for r in runs])
            results[lam] = {
                "throughput_mean": float(np.nanmean(th)), "throughput_std": float(np.nanstd(th)),
                "latency_p50_mean": float(np.nanmean(lt)), "latency_p50_std": float(np.nanstd(lt)),
                "n_repeats": REPEATS, "runs": runs,
            }
        # C-A / C-B 对比（需分别对 text-7B 与 VL-7B 运行后填入）
        comparison = {
            "C_A_same_precision": "1xVL-7B FP16 vs 1xtext-7B FP16（分别运行后填入）",
            "C_B_mem_equal": "1xVL FP16 vs 2xtext-7B AWQ-Int4（需 E17 附准确率差）",
            "C_C_cost": _cost_conversion(),
        }
        save_json({"per_lambda": results, "comparison": comparison},
                  results_dir("E11") / f"E11_{Path(args.model).name}.json")
        print(f"[E11] 完成，结果存 results/E11/E11_{Path(args.model).name}.json")


def _bench_run(model: str, backend: str, port: int, lam: int) -> dict:
    """调用 vllm benchmark_serving.py（子进程）。需先以 server 模式起 vllm。"""
    import glob
    cands = glob.glob(str(Path(sys.prefix).joinpath("**", "vllm", "benchmarks", "benchmark_serving.py")), recursive=True)
    if not cands:
        print("[E11] 未找到 benchmark_serving.py，跳过实测（请先安装 vllm 并起服务）")
        return {}
    script = cands[0]
    cmd = [
        sys.executable, script,
        "--model", model, "--backend", backend, "--port", str(port),
        "--request-rate", str(lam), "--duration", str(DURATION_S),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=DURATION_S * 3)
    # 解析 vllm 输出（best_of/throughput 行）
    tp = lat = float("nan")
    for line in r.stdout.splitlines():
        if "Throughput" in line and "req/s" in line:
            try:
                tp = float(line.split(":")[-1].split("req")[0].strip())
            except ValueError:
                pass
    return {"throughput": tp, "latency_p50": lat}


def _cost_conversion() -> dict:
    """B.3 每千张成本换算（纯算术，模型估算）。"""
    P_gpu, H_life, r_elec = 350.0, 8760.0, 0.6
    return {"note": "C-C 由 C-A/C-B 实测吞吐按 B.3 公式换算，见 E6 的 _cost_estimate"}


if __name__ == "__main__":
    main()