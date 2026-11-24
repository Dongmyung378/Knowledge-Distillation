# One-Shot Verification & L4 Reproduction Guide

[English README](../README.md) · [한국어](./reproduction.ko.md) · [Methodology & Decisions](./methodology-and-decisions.md) · [Results](./results.md)

Instead of toy synthetic demos, this repository provides a single self-contained notebook to **verify the exact SHA-256 frozen ONNX FP32/INT8 deployment artifacts (`v4.0-models`) and reports in one shot**, or **retrain the complete `v3-max` + `v4-final` study on an NVIDIA L4**.

---

## 1. One-Shot Verification Notebook (Recommended · CPU or GPU)

Upload [`notebooks/colab_enterprise_l4_one_shot.ipynb`](../notebooks/colab_enterprise_l4_one_shot.ipynb) to Google Colab, Vertex AI Colab Enterprise, or local Jupyter and click **Runtime → Run All**. All 23 source, test, and frozen report files are embedded inside a SHA-256-verified bundle.

### Mode A: `RUN_MODE = "verify_saved"` *(Default · 1–3 min One-Shot Verification)*
1. **Integrity & Unit Tests**: Unpacks and SHA-256-verifies the embedded `seg_kd` bundle, runs `seg_kd doctor`, and passes all `12/12` `unittest` checks.
2. **Frozen ONNX SHA-256 Check**: Downloads all 4 release ONNX artifacts from GitHub Release [`v4.0-models`](https://github.com/Dongmyung378/Knowledge-Distillation/releases/tag/v4.0-models) and verifies byte-for-byte SHA-256 equality against `evaluation-freeze*.json`.
3. **Live CPU ONNXRuntime Benchmark**: Measures live `384×768` single-frame 4-thread CPU latency (p50 ms, FPS), FP32→INT8 speedup, size reduction (`-72.5%`), and re-verifies the validation-only INT8 gate logic.
4. ***(Optional) Live Dataset Re-Evaluation (`EVALUATE_LIVE_DATASETS = True`)***: Automatically downloads `Chris1/cityscapes` (`locked_eval` 500 imgs) and KITTI Semantics (`200` imgs) without login to re-measure `67.95% / 66.96%` mIoU, KITTI Zero-Shot, and temperature-scaled ECE (`T*=1.70`) live against `reports/final-summary.json`.

### Mode B: `RUN_MODE = "reproduce_full"` *(NVIDIA L4 24GB · End-to-End Retraining)*
Retrains the complete `v3-max` (`SegFormer-B5` + 3-seed paired + 3-tier Pareto + QAT + ONNX INT8 gates) and `v4-final` (Medium Baseline + `384×768` 4-method ablation + Asymmetric Robust KD + BF16 QAT-KD + validation temperature scaling) study from scratch with automatic resumability.

---

## 2. Local CLI One-Shot Verification

```bash
pip install -e ".[export]"
python -m unittest discover -s tests -v
python scripts/verify_release_models.py --threads 4 --runs 20
```

---

## 3. Published Frozen ONNX Artifacts ([GitHub Release `v4.0-models`](https://github.com/Dongmyung378/Knowledge-Distillation/releases/tag/v4.0-models))

| Artifact | Size | Freeze SHA-256 (prefix) | Role |
|---|---:|---|---|
| `v3-max-mobilenetv4-medium-int8.onnx` | 7.90 MiB | `8cda8dff653f4561` | Clean-peak INT8 (`67.95%`, `40.49 FPS`, Gate PASS) |
| `v3-max-mobilenetv4-medium-fp32.onnx` | 28.71 MiB | `434c8e4243a55f35` | `v3-max` FP32 reference (`68.51%`, `26.24 FPS`) |
| `v4-final-robust-medium-int8.onnx` | 7.90 MiB | `fac9853786e9bb35` | Adverse-weather INT8 (`59.59%` noise > teacher, Gate PASS) |
| `v4-final-robust-medium-fp32.onnx` | 28.71 MiB | `1cb99d88e685d1f6` | `v4-final` FP32 reference (`60.61%` noise) |
