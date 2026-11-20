"""Action-content audit against the knowledge graph's prevention entries.

The audited L4 runs consumed the symptom-only 38-class graph, so the prevention
entries in configs/kg/kg_full.json were never shown to the model. For each
sampled record we compare the management categories named in the generated
action with those named in the guideline for the model's own diagnosis, and
with a mismatched guideline as a null.

Usage:
    python scripts/audit_action.py --sheet audit/content_audit_sheet.csv \
        --kg configs/kg/kg_full.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CATEGORIES = {
    "chemical": ["药剂", "杀菌剂", "喷施", "喷雾", "铜", "波尔多液", "代森锰锌", "三唑",
                 "抗生素", "霜脲氰", "百菌清", "多菌灵", "甲基硫菌灵", "氢氧化铜"],
    "resistant": ["抗病品种", "抗性品种", "抗病", "品种"],
    "rotation": ["轮作", "间作", "套种", "换茬"],
    "removal": ["清除", "销毁", "移除", "拔除", "剪除", "摘除", "病叶", "病枝", "病果", "病株"],
    "sanitation": ["清园", "杂草", "田园卫生", "卫生", "消毒", "深耕", "翻耕"],
    "moisture": ["通风", "透光", "排水", "灌溉", "控水", "湿度", "密植", "种植密度"],
    "nutrition": ["施肥", "有机肥", "钾肥", "氮肥", "水肥", "树势"],
    "biological": ["生物防治", "天敌", "拮抗", "生防"],
    "quarantine": ["隔离", "检疫", "封锁"],
}


def cats(text: str) -> set[str]:
    return {c for c, words in CATEGORIES.items() if any(w in text for w in words)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="audit/content_audit_sheet.csv")
    ap.add_argument("--kg", default="configs/kg/kg_full.json")
    args = ap.parse_args()

    kg = json.loads(Path(args.kg).read_text(encoding="utf-8"))
    prevention = {e["subject"]: e["object"] for e in kg["entries"] if e["relation"] == "prevention"}

    rows = list(csv.DictReader(Path(args.sheet).open(encoding="utf-8-sig")))
    used, skipped = [], 0
    for r in rows:
        cls = r["ModelDiagnosis"]
        if cls in prevention:
            used.append((r, cls))
        else:
            skipped += 1

    own_any = own_primary = 0
    null_any = null_primary = 0.0
    inter = union = 0
    classes = list(prevention)
    for r, cls in used:
        a = cats(r["Action"])
        g = cats(prevention[cls])
        own_any += bool(a & g)
        first = next((c for c in CATEGORIES if any(w in prevention[cls] for w in CATEGORIES[c])), None)
        own_primary += bool(first and first in a)
        inter += len(a & g)
        union += len(a | g)
        others = [c for c in classes if c != cls]
        for o in others:
            go = cats(prevention[o])
            null_any += bool(a & go) / len(others)
            first_o = next((c for c in CATEGORIES if any(w in prevention[o] for w in CATEGORIES[c])), None)
            null_primary += bool(first_o and first_o in a) / len(others)

    n = len(used)
    print(json.dumps({
        "records": len(rows),
        "mapped": n,
        "skipped": skipped,
        "own_any_category": round(own_any / max(1, n), 3),
        "null_any_category": round(null_any / max(1, n), 3),
        "own_primary_category": round(own_primary / max(1, n), 3),
        "null_primary_category": round(null_primary / max(1, n), 3),
        "jaccard_own": round(inter / max(1, union), 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
