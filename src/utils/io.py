"""路径与通用 I/O 工具。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repository root


def env_path(key: str, default: str) -> Path:
    """读取环境变量路径，支持 ${VAR:-default} 形式已在 yaml 中展开不了，这里手动兜底。"""
    val = os.environ.get(key, default)
    return Path(val)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    # 两段式：
    #  1) 仅展开"真实环境变量"(${VAR} 或 ${VAR:-default})，未定义且无默认值的原样保留；
    #  2) YAML 解析后再解析内部引用 ${key}（引用同一 YAML 内的其它键，如 ${data_root}）。
    raw = path.read_text(encoding="utf-8")
    expanded = _expand_env(raw)
    data = yaml.safe_load(expanded)
    return _resolve_refs(data)


def _expand_env(text: str) -> str:
    import re

    def repl(m: re.Match) -> str:
        key, default = m.group(1), m.group(2)
        if key in os.environ:            # 真实环境变量优先
            return os.environ[key]
        if default is not None:          # 有默认值则用默认
            return default
        return m.group(0)                # 否则保留原文（内部引用，留待 _resolve_refs）
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}", repl, text)


def _resolve_refs(data: Any, lookup: dict | None = None) -> Any:
    """递归解析 YAML 内部 ${key} 引用（key 为本 YAML 顶层键）。"""
    import re

    if lookup is None:
        lookup = data if isinstance(data, dict) else {}
    if isinstance(data, dict):
        return {k: _resolve_refs(v, lookup) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_refs(x, lookup) for x in data]

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in lookup and isinstance(lookup[key], str):
            return lookup[key]
        return m.group(0)
    if isinstance(data, str):
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, data)
    return data


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: Path | str) -> Any:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path | str, indent: int = 2) -> Path:
    import json

    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    return p


def copy_artifact(src: Path | str, dst: Path | str) -> Path:
    """复制产物到目标位置（用于冻结存证）。"""
    s, d = Path(src), Path(dst)
    ensure_dir(d.parent)
    shutil.copy2(s, d)
    return d