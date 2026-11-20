# R1-2 / R1-4 补充实验总表

> 全部与 E5 同口径（csc/saf seed43、同抽样种子、Qwen2.5-7B-Instruct、guided JSON、temp=0），
> 配对键为同一子集内的样本序号，McNemar 精确检验。

| 对照 | n | 原方案 | 新方案 | Δ | 原错→新对 | 原对→新错 | McNemar p |
|---|---|---|---|---|---|---|---|
| L3 vs L3-noU（seed42） | 1000 | 94.10% | 77.30% | -16.80 pp | 43 | 211 | 0.000 |
| L3 vs L3-noU（seed43） | 1000 | 93.60% | 78.10% | -15.50 pp | 39 | 194 | 0.000 |
| L4 vs L4-fullKG（seed42） | 1000 | 86.80% | 77.90% | -8.90 pp | 104 | 193 | 0.000 |
| L3 vs L4-fullKG（seed42） | 1000 | 94.10% | 77.90% | -16.20 pp | 41 | 203 | 0.000 |

## 预测多样性（判断载荷是否携带样本信息）

| 对照 | 不同预测类数 | 最高频预测类 | 占比 |
|---|---|---|---|
| L3 vs L3-noU（seed42） | 33 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 15.0% |
| L3 vs L3-noU（seed43） | 36 | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 13.0% |
| L4 vs L4-fullKG（seed42） | 56 | 番茄黄叶卷曲病毒 | 7.8% |
| L3 vs L4-fullKG（seed42） | 56 | 番茄黄叶卷曲病毒 | 7.8% |

## 尚未完成

- L4 vs L4-fullKG（seed43）
- L2 vs L2-fullKG（seed42）
- L2 vs L2-fullKG（seed43）
- L3 vs L4-fullKG（seed43）