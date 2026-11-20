"""LLM 实验公共（E5/E6/E7/E8 复用）。

- sample_subset：seed 固定抽样 n=1000（reproduce 配对铁律：同子集所有臂共用）。
- eval_llm_arm：对一批 LLMItem 做语义层 + 结构层 + 机制层评估。
- 机制层（N8/N9，零新增调用）：confidence 校准 ECE/AUROC、u↔confidence/正确性 Spearman、高置信错误率。
"""
from __future__ import annotations

import numpy as np
import json
from pathlib import Path

from ..evaluation.statistics import ece, auroc, spearman
from ..evaluation.integrity import validate_record, ladder_summary
from ..utils.io import load_json, load_yaml, project_path
from ..utils.seeds import get_seeds

# 建议合理率自动规则复用 eval_config 的 placeholders/action_min_len
CFG = load_yaml(project_path("configs", "eval_config.yaml"))["integrity_ladder"]

# 【修正】诊断命中匹配口径：模型在含中文 KG 描述时可能输出中文诊断名，
# 仅用英文类名字串匹配会把正确中文诊断误判为错（E5 L4 曾因此被系统性低估）。
# 加载 38 类中英别名表，匹配时同时比对英文类名与中文别名。
_ALIAS_FILE = project_path("configs", "class_aliases.json")
_CN_TO_EN: dict[str, str] = {}
try:
    _alias = load_json(_ALIAS_FILE)
    for _en, _cns in (_alias or {}).items():
        if isinstance(_cns, list):
            for _cn in _cns:
                _CN_TO_EN[str(_cn).strip().lower()] = _en
except Exception:
    _CN_TO_EN = {}


def sample_subset(paths: list[str], n: int = 1000, seed: int | None = None) -> list[str]:
    """固定抽样种子（默认 42），返回 n 个样本路径。"""
    import random

    seed = seed if seed is not None else get_seeds()["sampling_seed"]
    rng = random.Random(seed)
    if len(paths) <= n:
        return list(paths)
    return rng.sample(paths, n)


def eval_llm_arm(
    items: list,  # LLMItem
    true_labels: list,  # 每个样本真实 PV 类索引或类名
    classes: list[str],
    u_values: np.ndarray | None = None,
    validator=None,
    top3_loader: list | None = None,        # 每样本载荷 top3 类名列表（Top-3 准确率）
    csc_top1_correct: list | None = None,   # 每样本 CSC top1 是否判对（软压缩纠错率）
) -> dict:
    """对一条臂的全部输出做三类评估。返回 dict。"""
    n = len(items)
    # ---- 结构层（三级完整性）----
    records = [validate_record(it.raw, validator, finish_reason=it.finish_reason) for it in items]
    ladder = ladder_summary(records, n)

    # ---- 语义层 ----
    diag_correct = 0
    conf_list = []
    top3_hit = 0
    for i, (it, rec) in enumerate(zip(items, records)):
        if rec["parsed"] is None or not isinstance(rec["parsed"], dict):
            continue
        diag = rec["parsed"].get("diagnosis", "")
        conf = rec["parsed"].get("confidence")
        if conf is not None:
            conf_list.append(float(conf) if isinstance(conf, (int, float)) else np.nan)
        if _match_diag(diag, true_labels[i]):
            diag_correct += 1
        # Top-3 准确率：diagnosis 命中载荷 top3 类名集合
        if top3_loader is not None and _match_top3(diag, top3_loader[i]):
            top3_hit += 1
    diag_acc = diag_correct / n if n else 0.0

    # 结构层：进入语义分析的样本（L1 通过）
    valid_idx = [i for i, r in enumerate(records) if r["l1_valid"]]
    n_valid = len(valid_idx)
    diag_acc_valid = sum(
        1 for i in valid_idx if _match_diag(records[i]["parsed"].get("diagnosis", ""), true_labels[i])
    ) / n_valid if n_valid else 0.0

    # Top-3 准确率（valid 样本内）
    top3_hit_valid = sum(
        1 for i in valid_idx
        if top3_loader is not None and _match_top3(records[i]["parsed"].get("diagnosis", ""), top3_loader[i])
    ) / n_valid if n_valid and top3_loader is not None else None

    # 软压缩纠错率：CSC top1 错 且 LLM diagnosis 对（方案 H1 主指标）
    correct_rate = None
    if csc_top1_correct is not None and top3_loader is not None:
        csc_wrong = [i for i in valid_idx if not csc_top1_correct[i]]
        n_csc_wrong = len(csc_wrong)
        if n_csc_wrong:
            llm_correct_over_cscwrong = sum(
                1 for i in csc_wrong
                if _match_diag(records[i]["parsed"].get("diagnosis", ""), true_labels[i])
            )
            correct_rate = llm_correct_over_cscwrong / n_csc_wrong
    # 建议合理率：action 非空、非占位符、长度≥阈值（复用 L2 的 action 判定，自动 rule）
    advise_rate = None
    if validator is not None:
        placeholders = set(CFG["placeholders"])
        action_min = CFG["action_min_len"]
        n_advise = sum(
            1 for rec in records if rec["parsed"] is not None and isinstance(rec["parsed"], dict)
            and _action_ok(rec["parsed"].get("action"), placeholders, action_min)
        )
        advise_rate = n_advise / n if n else 0.0

    semantic = {
        "diag_acc": diag_acc,
        "diag_acc_valid": diag_acc_valid,
        "top3_acc": top3_hit / n if n and top3_loader is not None else None,
        "top3_acc_valid": top3_hit_valid,
        "correct_rate": correct_rate,          # 软压缩纠错率（CSC错→LLM对）
        "advise_rate": advise_rate,            # 建议合理率（自动字段规则）
    }

    # ---- 机制层（N8/N9）----
    conf_arr = np.array(conf_list, dtype=float)
    correctness = np.zeros(n_valid)
    for j, i in enumerate(valid_idx):
        correctness[j] = 1.0 if _match_diag(records[i]["parsed"].get("diagnosis", ""), true_labels[i]) else 0.0
    valid_conf = np.array([conf_arr[j] for j in range(n_valid) if not np.isnan(conf_arr[j])])
    valid_corr = np.array([correctness[j] for j in range(n_valid) if not np.isnan(conf_arr[j])])
    cal = {
        "ece": ece(valid_conf, valid_corr) if len(valid_conf) else None,
        "conf_auroc": auroc(valid_conf, valid_corr) if len(valid_conf) and valid_conf.std() > 0 else None,
        "high_conf_err_rate": float(((valid_conf >= 0.9) & (valid_corr == 0)).mean()) if len(valid_conf) else None,
        "n_with_conf": len(valid_conf),
    }
    # u ↔ confidence / 正确性 Spearman
    ucorr = {}
    if u_values is not None and len(valid_conf) and len(u_values) >= n:
        u_val = np.array([u_values[i] for i in valid_idx])[: len(valid_conf)]
        ucorr["u_conf_spearman"] = spearman(u_val[: len(valid_conf)], valid_conf)
        ucorr["u_correct_spearman"] = spearman(u_val, correctness)
    # 高/低 u 中位数分组对比（N9）
    u_group = {}
    if u_values is not None and len(u_values) >= n and len(valid_conf):
        u_aligned = np.array([u_values[i] for i in valid_idx])[: len(valid_conf)]
        med = np.median(u_aligned)
        hi = u_aligned > med
        lo = ~hi
        u_group = {
            "u_median": float(med),
            "high_u_acc": float(correctness[hi].mean()) if hi.any() else None,
            "low_u_acc": float(correctness[lo].mean()) if lo.any() else None,
            "high_u_mean_conf": float(valid_conf[hi].mean()) if hi.any() else None,
            "low_u_mean_conf": float(valid_conf[lo].mean()) if lo.any() else None,
            "n_high_u": int(hi.sum()),
            "n_low_u": int(lo.sum()),
        }
    return {
        "n": n,
        "semantic": semantic,
        "structure_ladder": ladder,
        "calibration": cal,
        "u_correlation": ucorr,
        "u_group": u_group,
        "hallucination_note": "E5 不报幻觉率：自动管线不可靠，需 E19 人工复核（B.1）",
    }


def _match_diag(diag: str, true_class: str) -> bool:
    """诊断是否命中真实类。

    【修正】双向匹配口径：
    1) 英文类名直接子串匹配（模型输出英文类名时，如 L3 复用 top3）；
    2) 中文别名命中：诊断文本若包含真实类对应的任一中文学名，也判命中
       （模型在含中文 KG 描述时倾向输出中文诊断，如 L4）。
    """
    d = str(diag).strip().lower()
    if not d:
        return False
    tc = str(true_class).strip().lower()
    if tc in d or d in tc:
        return True
    return any(cn in d for cn in _CN_TO_EN if _CN_TO_EN[cn] == true_class)


def _match_top3(diag: str, top3_classes: list[str]) -> bool:
    """诊断是否命中载荷 top3 类名集合（Top-3 准确率）。

    对 top3 中任一类做英文/中文双语匹配（复用 _match_diag 口径）。
    """
    d = str(diag).strip().lower()
    if not d:
        return False
    for c in top3_classes:
        tc = str(c).strip().lower()
        if tc in d or d in tc:
            return True
        if any(cn in d for cn in _CN_TO_EN if _CN_TO_EN[cn] == c):
            return True
    return False


def _action_ok(action, placeholders: set[str], action_min: int) -> bool:
    """建议合理率自动规则：action 非空、非占位符、长度≥阈值。"""
    if action is None:
        return False
    if not isinstance(action, str):
        return False
    a = action.strip()
    if len(a) == 0:
        return False
    if a.lower() in placeholders:
        return False
    return len(a) >= action_min


# ---- 幻觉率（B.1 自动管线）----
# 【决策】E5 不报自动幻觉率：AI 无法可靠判定幻觉（实体比对会误判合法防治术语），
# 方案 B.1 要求 200 例人工复核校准，属 E19 范畴。E5 仅靠 prediction_pool 保留逐样本字段供 E19 复用。
_HALLU_WHITELIST = set()
try:
    _HALLU_WHITELIST = set(load_yaml(project_path("configs", "kg", "whitelist.yaml")).get("whitelist", []))
except Exception:
    _HALLU_WHITELIST = set()


def dump_prediction_pool(
    records: list[dict],
    exp: str,
    out_root,
) -> None:
    """把逐样本判定写入 prediction_pool（供 E18/E19 失效/专家评估，消除幸存者偏差）。

    【修复】改为按 (arm, sampling_seed, id) **幂等 upsert**，不再纯追加：
    同一 (arm,seed,id) 重复写入会自动替换旧记录，避免"同一实验跑多次/干燥+正式"
    造成 prediction_pool 出现重复 id（此前 E5 seed42 被两次写入，L0 重复 1000、
    其余臂各重复 100，导致 E9 R4 逐样本取值不稳）。E18/E19 统计也避免双计。
    """
    import json

    pool_dir = out_root / "prediction_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    path = pool_dir / f"{exp}.jsonl"
    # 读现有 → 按主键去重
    existing: dict[tuple, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                existing[(r.get("arm"), r.get("sampling_seed"), r.get("id"))] = r
    for r in records:
        existing[(r.get("arm"), r.get("sampling_seed"), r.get("id"))] = r
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(existing):
            f.write(json.dumps(existing[key], ensure_ascii=False) + "\n")