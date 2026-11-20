"""实验快照机制（用户硬性要求：每次实验后保存终端真实输出记录）。

SnapshotLogger 以 tee 方式把终端 stdout/stderr **真实输出**同时写到控制台和
snapshots/<exp>/<timestamp>/terminal.log，并额外落盘：
  - command.txt   : 完整命令行与调用参数
  - meta.yaml     : 时间、种子、configs 哈希、git tag、nvidia-smi、pip freeze
  - manifest.sha256 : 结果 JSON 的 SHA-256 清单（红线 S2 冻结存证）
用法见各实验脚本入口 `with ExperimentContext(...)`。
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .io import ensure_dir, project_path, load_yaml


class _Tee(io.TextIOBase):
    """把写入同时派发给两个流。"""

    def __init__(self, primary: TextIO, mirror: TextIO):
        self.primary = primary
        self.mirror = mirror
        self._errors = getattr(primary, "errors", None)

    def write(self, s: str) -> int:
        self.primary.write(s)
        self.mirror.write(s)
        self.mirror.flush()
        return len(s)

    def flush(self) -> None:
        self.primary.flush()
        self.mirror.flush()

    @property
    def errors(self):
        return self._errors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExperimentContext:
    """实验快照上下文管理器，负责 tee 终端输出 + 落盘元数据。"""

    experiment: str            # 如 "E5"
    description: str = ""
    configs: dict[str, Any] = field(default_factory=dict)
    keep_running: bool = True  # 退出时是否恢复原始流

    snap_dir: Path = field(default=None, repr=False)  # type: ignore
    _log_file: TextIO = field(default=None, repr=False)  # type: ignore
    _old_out: TextIO = field(default=None, repr=False)  # type: ignore
    _old_err: TextIO = field(default=None, repr=False)  # type: ignore

    def __post_init__(self):
        if self.snap_dir is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.snap_dir = ensure_dir(
                project_path("snapshots", self.experiment, ts)
            )

    def __enter__(self) -> "ExperimentContext":
        self._old_out = sys.stdout
        self._old_err = sys.stderr
        self._log_file = open(self.snap_dir / "terminal.log", "a", encoding="utf-8")
        sys.stdout = _Tee(self._old_out, self._log_file)  # type: ignore
        sys.stderr = _Tee(self._old_err, self._log_file)  # type: ignore
        self._write_meta()
        print(f"[snapshot] 实验 {self.experiment} 快照目录: {self.snap_dir}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.keep_running:
            sys.stdout = self._old_out
            sys.stderr = self._old_err
        if self._log_file is not None:
            self._log_file.close()
        self._write_manifest()
        print(f"[snapshot] 终端输出已保存: {self.snap_dir / 'terminal.log'}", flush=True)
        return False

    def _write_meta(self) -> None:
        import yaml

        env_lines = []
        nvidia_smi_text = ""
        try:
            smi = subprocess.run(
                "nvidia-smi", capture_output=True, text=True, timeout=30
            )
            env_lines.append("=== nvidia-smi ===")
            env_lines.append(smi.stdout)
            nvidia_smi_text = smi.stdout
        except Exception as e:  # noqa: BLE001
            env_lines.append(f"(nvidia-smi 不可用: {e})")
        try:
            freeze = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            env_lines.append("=== pip freeze ===")
            env_lines.append(freeze.stdout)
        except Exception as e:  # noqa: BLE001
            env_lines.append(f"(pip freeze 失败: {e})")

        git_tag = "n/a"
        try:
            r = subprocess.run(
                ["git", "-C", str(project_path()), "describe", "--tags"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            git_tag = r.stdout.strip() or "no-tag"
        except Exception:  # noqa: BLE001
            pass

        meta = {
            "experiment": self.experiment,
            "description": self.description,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "git_tag": git_tag,
            "python": sys.version,
            "configs": self.configs,
            "nvidia_smi": nvidia_smi_text,
        }
        (self.snap_dir / "meta.yaml").write_text(
            yaml.safe_dump(meta, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (self.snap_dir / "env_snapshot.txt").write_text(
            "\n".join(env_lines), encoding="utf-8"
        )
        (self.snap_dir / "command.txt").write_text(
            " ".join(sys.argv), encoding="utf-8"
        )

    def _write_manifest(self) -> None:
        """结果 JSON 的 SHA-256 清单（红线 S2 冻结存证）。"""
        lines = []
        for f in sorted(self.snap_dir.rglob("*.json")):
            lines.append(f"{_sha256(f)}  {f.relative_to(self.snap_dir)}")
        if lines:
            (self.snap_dir / "manifest.sha256").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    @contextlib.contextmanager
    def artifact(self, *parts: str) -> Path:
        """在快照目录下创建子目录并返回路径。"""
        d = ensure_dir(self.snap_dir.joinpath(*parts))
        yield d


def load_experiment_config(exp_name: str) -> dict[str, Any]:
    """读取 configs/rg_<exp>/ 下的实验配置片（若存在）。"""
    cfg_dir = project_path("configs", "experiments", exp_name)
    merged: dict[str, Any] = {}
    if cfg_dir.exists():
        for p in sorted(cfg_dir.glob("*.yaml")):
            merged.update(load_yaml(p))
    return merged