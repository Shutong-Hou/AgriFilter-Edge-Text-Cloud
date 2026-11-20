"""Reproduce the 100-record content-audit sheet from the L4 per-sample records.

Usage:
    python audit/sample_l4.py \
        --jsonl results/review_r1/L4_fullkg_raw_seed42.jsonl \
        --kg configs/kg/kg_full.json \
        --out audit/content_audit_sheet.csv

The sampling seed is fixed (20261120). Rating columns are left blank for two
independent readers; see content_audit_rubric.md for the 1--3 scale.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

FIELDS = [
    "ID", "TrueLabel", "ModelDiagnosis", "DiagnosisCorrect", "Confidence",
    "Action", "Reasoning", "Rater1_Action", "Rater1_Reasoning",
    "Rater2_Action", "Rater2_Reasoning", "Notes",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/review_r1/L4_fullkg_raw_seed42.jsonl")
    ap.add_argument("--kg", default="configs/kg/kg_full.json")
    ap.add_argument("--out", default="audit/content_audit_sheet.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261120)
    args = ap.parse_args()

    kg = json.loads(Path(args.kg).read_text(encoding="utf-8"))
    to_class: dict[str, str] = {}
    for cls, vals in kg.get("alias_dict", {}).items():
        if cls.startswith("_"):
            continue
        for v in vals if isinstance(vals, list) else [vals]:
            to_class[v] = cls

    records = [json.loads(line) for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = random.Random(args.seed).sample(range(len(records)), args.n)

    with Path(args.out).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i in sorted(picked):
            r = records[i]
            out = json.loads(r["raw"])
            diag = out.get("diagnosis", "")
            w.writerow({
                "ID": i,
                "TrueLabel": r["true_label"],
                "ModelDiagnosis": to_class.get(diag, diag),
                "DiagnosisCorrect": r["correct"],
                "Confidence": out.get("confidence", ""),
                "Action": out.get("action", ""),
                "Reasoning": out.get("reasoning", ""),
            })


if __name__ == "__main__":
    main()
