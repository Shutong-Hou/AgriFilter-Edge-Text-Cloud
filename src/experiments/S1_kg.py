"""S1：KG 构建与审计（W5 前，E5 依赖）(方案 S1)。

- 五元组 schema：(subject, relation, object, source_url, note)
- 来源 URL 逐条登记
- 100 条双人核对报一致率
- 38 类全覆盖单测
对 configs/kg/kg.json 的种子做全覆盖检查与五元组校验，输出审计报告。

用法: python -m src.experiments.S1_kg
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..utils.experiment_snapshot import ExperimentContext
from ..utils.io import project_path, save_json
from ..kg.kg import KG, validate_five_tuple
from .common import load_pv_classes, results_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(project_path("configs", "kg", "kg.json")),
                    help="审计对象：默认配置加载的冻结种子；也可传 kg_full.json 审计 38 类草案")
    ap.add_argument("--tag", default="", help="输出文件后缀标记，如 full")
    args = ap.parse_args()

    with ExperimentContext(experiment="S1", description=f"KG 构建与审计（{Path(args.kg).name}）",
                           configs={"kg": args.kg}) as ctx:
        kg = KG.load(args.kg)
        classes = load_pv_classes()

        # 五元组校验
        invalid = [e for e in kg.entries if not validate_five_tuple(e)]
        # 38 类全覆盖单测
        coverage = kg.full_coverage_check(classes)
        # 来源 URL 登记（空 URL 需按 S1 补全）
        missing_src = [e["subject"] for e in kg.entries if not e.get("source_url")]

        report = {
            "kg_file": str(Path(args.kg).name),
            "n_entries": len(kg.entries),
            "invalid_five_tuples": len(invalid),
            "coverage": coverage,
            "missing_source_url_subjects": missing_src,
            "double_check_statistics": "需双人标注后计算（人工环节）",
            "note": ("freeze=configs/kg/kg.json（E5/E6 运行加载，冻结为保证可复现）；"
                     "full=configs/kg/kg_full.json（S1 38 类全覆盖草案，来源URL待人工登记）"),
        }
        tag = f"_{args.tag}" if args.tag else ""
        save_json(report, results_dir("S1") / f"S1_kg_audit{tag}.json")
        cov = coverage
        print(f"[S1] 审计 {Path(args.kg).name}：条目 {len(kg.entries)}，非法五元组 {len(invalid)}，"
              f"覆盖 {cov['covered']}/{cov['n_classes']}，缺失来源 {len(missing_src)}")
        if cov["missing"]:
            print(f"[S1] 未覆盖类: {cov['missing']}")
        print(f"[S1] 完成，审计存 results/S1/S1_kg_audit{tag}.json")


if __name__ == "__main__":
    main()