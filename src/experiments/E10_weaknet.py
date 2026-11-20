"""E10：弱网生存【核心，W9】(方案 E10)。

两区制传输模型（B.4）：小载荷（≤W0≈14.6KB）≈1–2×RTT；大载荷=慢启动+Mathis 稳态上界。
先验证后断言：netem 下 curl 实测 KB 级载荷各 RTT 传输时间，验证小载荷区模型后才准写洞察句。
解析矩阵 72 格（p×RTT×B，零成本）；netem 实测代表性子集 ~12 格。

注意：netem 实测需 Linux（tc netem）。Windows 下本脚本仅输出解析矩阵并提示跳过实测。
用法: python -m src.experiments.E10_weaknet
"""
from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import load_yaml, project_path, save_json
from .common import results_dir

NET_CFG = load_yaml(project_path("configs", "eval_config.yaml"))["network"]
RTT_MS = [20, 50, 100, 200, 400]
LOSS = [0.0, 0.01, 0.05]
B_TXT_BYTES = [512, 2048, 14600, 30000, 100000]  # 文本载荷（B.7 载荷规模）
B_IMG_KB = [100, 500, 1024, 2048]                # 全图载荷（R0 基线）


def analytic_time_bytes(B: int, rtt_s: float) -> float:
    """两区制传输时间估算（B.4）。小载荷区：RTT握手 + ceil(L/W0)·RTT + L/B。"""
    W0 = NET_CFG["initial_window_bytes"]
    bw = 1e6  # B/s 假设带宽
    if B <= W0:
        t = 1.5 * rtt_s + rtt_s + B / bw
    else:
        mss = NET_CFG["mss_bytes"]
        p = NET_CFG["p_loss_assumed"]
        t_steady = mss / (rtt_s * np.sqrt(p)) * (B / mss) / (bw / 1)  # Mathis 上界（秒）
        t = 1.5 * rtt_s + rtt_s + t_steady
    return t


def main() -> None:
    with ExperimentContext(experiment="E10", description="弱网生存", configs={}) as ctx:
        # 解析矩阵 72 格（p×RTT×B）：文本载荷成功率
        analytic = {}
        for p, rtt, b in itertools.product(LOSS, RTT_MS, B_TXT_BYTES):
            t = analytic_time_bytes(b, rtt / 1000.0)
            # 成功率：与 RTT+2s 超时预算比较（简化：>2s 视为失败）
            success = 1.0 if t <= 2.0 else 0.0
            analytic[f"p{p}_rtt{rtt}_B{b}"] = {"t_s": round(t, 4), "success": success}

        # netem 实测代表性子集（~12 格），仅 Linux 可执行
        measured = {}
        is_linux = sys.platform.startswith("linux")
        if is_linux:
            for p, rtt in [(0.0, "20ms"), (0.0, "100ms"), (0.01, "200ms"), (0.05, "400ms")]:
                for b in [512, 2048, 14600]:
                    t = _netem_measure(rtt, p, b)
                    measured[f"p{p}_rtt{rtt}_B{b}"] = t
        else:
            print("[E10] 非 Linux 环境跳过 netem 实测（需 tc netem）。仅输出解析矩阵。")

        # 模型 vs 实测偏差表
        dev = {}
        for k, vm in measured.items():
            if k in analytic:
                dev[k] = {"model": analytic[k]["t_s"], "measured": vm,
                          "bias": round(vm - analytic[k]["t_s"], 4)}
        save_json({"analytic_matrix": analytic, "measured": measured, "model_vs_measured": dev},
                  results_dir("E10") / "E10_weaknet.json")
        print(f"[E10] 解析矩阵 {len(analytic)} 格，实测 {len(measured)} 格，已保存 results/E10/E10_weaknet.json")


def _netem_measure(rtt: str, loss: float, b: int) -> float:
    """netem + 本地 HTTP 实测载荷传输时间（秒）。需 Linux + sudo + requests。

    在 127.0.0.1 起临时 HTTP 服务返回指定字节数载荷，netem 对 lo 施加
    delay/loss 后测量下载耗时。小载荷区模型验证即由此类实测完成（方案 N2）。
    """
    import http.server
    import socketserver
    import threading
    import time

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(b))
            self.end_headers()
            self.wfile.write(b"x" * b)

        def log_message(self, *a):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as srv:
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        subprocess.run(["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem",
                        "delay", rtt, "loss", f"{loss*100}%"], check=True)
        import requests
        t0 = time.time()
        requests.get(f"http://127.0.0.1:{port}/", timeout=30)
        dt = time.time() - t0
        srv.shutdown()
        return round(dt, 4)


if __name__ == "__main__":
    main()