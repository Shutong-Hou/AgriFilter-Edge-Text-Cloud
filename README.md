# AgriFilter — edge–text–cloud plant disease diagnosis

Artifact repository for the manuscript **“Low-Bandwidth and Efficient Plant Disease Diagnosis Based on Perceptual Evidence”** (Frontiers in Plant Science, manuscript ID 1999473).

The pipeline converts each leaf photograph into a short structured evidence payload with two lightweight edge classifiers (SAF, CSC) and sends only that text to a small cloud language model, which decodes it under a frozen JSON schema.

## What is in this repository

| Path | Contents |
|---|---|
| `models/saf.onnx`, `models/csc.onnx` | ONNX exports of the two deployed edge models (inputs normalised inside the graph) |
| `models/saf_seed43.pt`, `models/csc_seed43.pt` | PyTorch checkpoints of the deployed seed-43 models |
| `mobile_bench/AgriFilter_mobile_bench.html` | Self-contained on-device measurement page (models embedded); open in Chrome |
| `mobile_bench/make_mobile_bench.py`, `mobile_bench/mobile_bench_template.html` | Scripts that build the measurement page |
| `scripts/export_onnx.py`, `scripts/verify_onnx.py` | ONNX export and export-verification scripts |
| `results/review_r1/*.jsonl` | Per-sample prediction records (whole-image VLM baseline, chain-of-thought and three-shot controls, L3/L4 arms) |
| `results/review_r1/*.json` | Summary statistics corresponding to those records |
| `prereg/S3_preregistration.md` | Pre-registration plan, frozen before the language-model runs |
| `configs/kg/` | Knowledge-graph files: the frozen 10-entry seed graph, the 38-class draft with prevention entries, and the hallucination whitelist |
| `audit/` | 100-record content-audit sheet with a 1--3 rubric for action appropriateness and reasoning support, plus the sampling script |
| `src/`, `configs/`, `scripts/` | Source code, schemas and configuration for the reported experiments |

## Reproducing the on-device measurement

1. Open `mobile_bench/AgriFilter_mobile_bench.html` in Chrome (desktop or Android).
2. The page runs SAF and CSC through ONNX Runtime Web on the WebAssembly/SIMD backend, with 10 warm-up and 200 timed iterations per model.
3. Latency statistics are shown in the page; the reported medians come from three independent repetitions.

No installation is required for the measurement page. To rebuild it from the ONNX files:

```bash
python mobile_bench/make_mobile_bench.py
```

## Reproducing the per-sample records

The JSONL files in `results/review_r1/` contain one record per evaluated image:

```json
{"i": 0, "true_label": "...", "diag": "...", "correct": true, "raw": "{...four-field JSON...}"}
```

They can be re-scored without a GPU. The cloud language model itself requires a GPU; see `requirements.txt` for the pinned environment.

## Environment

Python 3.10 with the packages in `requirements.txt`. The reported runs used PyTorch 2.1.2+cu121, Transformers 4.49.0 and Outlines 0.2.3 locally, and vLLM 0.27/0.28 on Linux for the serving and quantisation experiments.

## Data

PlantVillage, PlantDoc and iCassava 2019 are public datasets and are **not** redistributed here. See `data/README.txt` for download instructions.

## Pre-registration

`prereg/S3_preregistration.md` is the plan frozen before the first language-model run. The commit that first added this file to this repository is the timestamped record of that plan.

## License

Code is released under the MIT License. Prediction records and summary statistics are released under CC BY 4.0. The datasets retain their original licenses.

## Contact

Xiongwei He — pinfanxiaohe@sxau.edu.cn
Corresponding author: Xiaoying Zhang — xiaoyingzhang@sxau.edu.cn
