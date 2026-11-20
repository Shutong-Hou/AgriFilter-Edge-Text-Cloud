"""KG 构建与审计（方案 S1，W5 前，E5 依赖）。

五元组 schema：(subject, relation, object, source_url, note)。
- 来源 URL 逐条登记
- 100 条双人核对报一致率
- 38 类全覆盖单测
提供 kg_descriptions(class) 供 L2/L4 载荷使用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KG:
    entries: list[dict[str, Any]]
    alias_dict: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "KG":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(entries=data.get("entries", []), alias_dict=data.get("alias_dict", {}))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"entries": self.entries, "alias_dict": self.alias_dict}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def classes_covered(self) -> set[str]:
        return {e["subject"] for e in self.entries}

    def all_source_urls(self) -> list[str]:
        return [e.get("source_url", "") for e in self.entries if e.get("source_url")]

    def description_for(self, cls: str) -> str:
        """把某类所有三元组拼成 KG 描述文本（供 L2/L4）。"""
        parts = []
        for e in self.entries:
            if e["subject"] == cls:
                parts.append(f"{e['relation']} {e['object']}")
        return "; ".join(parts)

    def descriptions_map(self) -> dict[str, str]:
        return {c: self.description_for(c) for c in sorted(self.classes_covered())}

    def full_coverage_check(self, all_classes: list[str]) -> dict:
        covered = self.classes_covered()
        missing = sorted(set(all_classes) - covered)
        return {
            "n_classes": len(all_classes),
            "covered": len(covered),
            "missing": missing,
            "full_coverage": not missing,
        }

    def double_check_statistics(self, checker_b: list[dict]) -> dict:
        """100 条双人核对一致率（checker_b 为第二位核对结果，字段结构同 entries）。"""
        agree = 0
        for a, b in zip(self.entries, checker_b):
            if a.get("subject") == b.get("subject") and a.get("relation") == b.get("relation") and a.get("object") == b.get("object"):
                agree += 1
        n = max(1, len(self.entries))
        return {"n_checked": len(self.entries), "agree": agree, "agreement": agree / n}


def validate_five_tuple(e: dict[str, Any]) -> bool:
    required = {"subject", "relation", "object", "source_url"}
    return required.issubset(e.keys()) and bool(e.get("subject")) and bool(e.get("relation")) and bool(e.get("object"))