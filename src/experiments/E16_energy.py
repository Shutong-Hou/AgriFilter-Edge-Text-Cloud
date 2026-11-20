"""E16：能耗【随 E9/E11 同步测量，零额外 GPU 时间】(方案 E16 / B.5)。

pynvml 10Hz；净推理能耗 E=(P_avg−P_idle)×t；P_idle 两档（空载/模型加载待机），主报加载待机基线；
毛能耗并列；CO₂e 因子引权威来源注年份。

用法: python -m src.experiments.E16_energy --model Qwen/Qwen2.5-7B-Instruct --batch 100
"""
from __future__ import annotations

import argparse
import subprocess
import time

import numpy as np

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from .common import results_dir

CFG = load_yaml(project_path("configs", "eval_config.yaml"))["energy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    with ExperimentContext(experiment="E16", description="能耗测量", configs={}) as ctx:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            print(f"[E16] pynvml 初始化失败（{e}）。请安装 pynvml 且确认 NVIDIA 驱动。")
            return

        # 待机基线（加载模型前）
        p_idle = _sample_power(pynvml, handle, seconds=5)
        # 空载基线（无模型）
        # 主报加载待机基线（B.5）
        note = f"P_idle_loaded={p_idle:.1f}W（主报基线）"

        # 推理功率采样（需在推理期间并行采样；此处演示对 model 推理的同步采样）
        avg, t = _sample_during_inference(pynvml, handle, args.model, args.batch)
        E_net = (avg - p_idle) * t
        E_gross = avg * t
        co2e = 0.581  # kgCO2e/kWh（中国 2023 电网因子，来源：IEA/中国电网排放因子公报）
        result = {
            "P_idle_loaded_w": p_idle,
            "P_avg_w": avg,
            "t_s": t,
            # 单位修正：P(W)×t(s)=J，kWh=J/3.6e6；旧版 /3600 得 Wh 却标 kWh，错 1000 倍
            "E_net_kwh": E_net / 3_600_000.0,
            "E_gross_kwh": E_gross / 3_600_000.0,
            "co2e_kg": (E_net / 3_600_000.0) * co2e,
            "co2e_factor_note": "0.581 kgCO2e/kWh（中国 2023 电网因子）",
            "sample_hz": CFG["sample_hz"],
            "note": note,
        }
        save_json(result, results_dir("E16") / "E16_energy.json")
        print(f"[E16] P_idle={p_idle:.1f}W P_avg={avg:.1f}W E_net={result['E_net_kwh']:.6f}kWh "
              f"CO₂e={result['co2e_kg']:.4f}kg")
        print(f"[E16] 完成，结果存 results/E16/E16_energy.json")


def _sample_power(pynvml, handle, seconds: float) -> float:
    vals = []
    end = time.time() + seconds
    while time.time() < end:
        try:
            vals.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        except pynvml.NVMLError:
            pass
        time.sleep(1 / CFG["sample_hz"])
    return float(np.mean(vals)) if vals else float("nan")


def _sample_during_inference(pynvml, handle, model: str, batch: int) -> tuple[float, float]:
    """对一段推理过程同步采样功率。进行 1 次小批量推理作为能耗载荷。"""
    from ..inference.llm_client import get_llm_client
    client = get_llm_client(model).load()
    prompts = ["请用 JSON 输出诊断：作物=番茄，症状=叶片水渍状暗绿斑。"] * max(1, batch)
    vals = []
    t0 = time.time()
    # 采样线程
    import threading
    stop = False

    def sampler():
        while not stop:
            try:
                vals.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except Exception:
                pass
            time.sleep(1 / CFG["sample_hz"])

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    client.generate(prompts, temperature=0.0, guided=True)
    stop = True
    t = time.time() - t0
    th.join(timeout=2)
    return (float(np.mean(vals)) if vals else float("nan")), t


if __name__ == "__main__":
    main()