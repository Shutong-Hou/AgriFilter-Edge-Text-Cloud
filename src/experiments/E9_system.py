"""E9：系统级对比 + 增量消融主表【核心，W8–W9】(方案 E9)。

增量行：R0 全图+MLLM → R1 +SAF（异常仍传图）→ R2 +硬压缩 → R3 +软压缩 → R4 完全体。
延迟双列：全流平均 / 病害流端到端。边缘延迟三轨（T2 解析估算区间 † / T3 真实设备实测点）。
指标：准确率、双列延迟分解、字节/图、云端调用率、每千张成本、能耗（随 E16）、三级完整性。

本脚本为分析/汇总器：读取 E5/E6/E15/E16 产物拼装增量表。延迟/能耗实测由对应实验填充。
用法: python -m src.experiments.E9_system
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from .common import results_dir

EDGE_CFG = load_yaml(project_path("configs", "eval_config.yaml"))["edge"]


def main() -> None:
    with ExperimentContext(experiment="E9", description="系统增量主表", configs={}) as ctx:
        # 读取 E5 各臂准确率（L4 为 R4）
        rows = _collect_rows()
        table = []
        for r in ["R0", "R1", "R2", "R3", "R4"]:
            row = rows.get(r, {})
            table.append({
                "row": r,
                "accuracy": row.get("acc"),
                "cloud_call_rate": row.get("cloud_call_rate"),
                "bytes_per_image": row.get("bytes_per_image"),
                "latency_full_avg_s": row.get("latency_full_avg_s"),
                "latency_disease_e2e_s": row.get("latency_disease_e2e_s"),
                "edge_latency_est_range_s": _edge_est_range(),
                "per_1k_cost": row.get("per_1k_cost"),
                "integrity_l1": row.get("l1"),
            })
        # 边缘延迟三轨（红线9：主表只认 T2 估算区间或 T3 真实设备）
        save_json({"table": table, "edge_three_tracks": _edge_tracks()},
                  results_dir("E9") / "E9_table.json")
        for r in table:
            print(f"[E9] {r['row']}: acc={r['accuracy']} 字节/图={r['bytes_per_image']} "
                  f"延迟={r['latency_full_avg_s']}")


def _collect_rows() -> dict:
    data = {}
    # R4 = E5 L4 结果
    e5 = project_path("results", "E5")
    if e5.exists():
        for f in e5.glob("E5_L4_seed42.json"):
            import json
            d = json.loads(f.read_text(encoding="utf-8"))
            data["R4"] = {"acc": d.get("semantic", {}).get("diag_acc")}
    # R0 由 E15 MLLM 基线填充（占位，需真实运行）
    data.setdefault("R0", {})
    data.setdefault("R1", {})
    data.setdefault("R2", {})
    data.setdefault("R3", {})
    return data


def _edge_est_range() -> list:
    """B.2 边缘延迟解析估算区间（†）。t_edge ≈ FLOPs/(η·P) + Bytes/BW。"""
    flops_mnet = 0.6e9  # MobileNetV3-Small ~0.6 GFLOPs
    P_edge = 3.0  # W（树莓派4B 典型，来源：硬件）
    bw = 1e6  # B/s
    lo = flops_mnet / (EDGE_CFG["eta_max"] * P_edge) / 1e9 + 1 / bw
    hi = flops_mnet / (EDGE_CFG["eta_min"] * P_edge) / 1e9 + 1 / bw
    return [round(lo, 4), round(hi, 4)]


def _edge_tracks() -> dict:
    return {
        "T1_x86_singlecore": "工程参考线（体系结构差异，不得作边缘基线，红线9）",
        "T2_analytic_interval": _edge_est_range(),
        "T3_real_device": "待树莓派/Jetson 实测（强烈建议购置）",
    }


if __name__ == "__main__":
    main()