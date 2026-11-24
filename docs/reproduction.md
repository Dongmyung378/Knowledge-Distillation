# Reproduction & Execution Guide

[English README](../README.md) · [한국어](./reproduction.ko.md) · [Architecture](./architecture.md) · [Results](./results.md)

## 1. Fast Local CPU Verification (~2 Minutes)

Verify mathematical invariance, quantization-friendliness audits, ONNX FP32/INT8 export, SHA-256 evaluation freezing, and post-hoc temperature scaling locally without a GPU or external datasets:

```bash
# 1. Create virtual environment & install package
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -e ".[export]"

# 2. Diagnostics & unit/integration test suite (12 tests)
python -m seg_kd doctor
python -m unittest discover -s tests -v

# 3. End-to-end synthetic driving smoke pipeline
python -m seg_kd prepare-data --dataset synthetic --data ./tmp/syn-data --output ./tmp/syn-manifest.json --image-size 128x256
python scripts/run_l4_study.py \
  --plan configs/l4-plan.json \
  --manifest ./tmp/syn-manifest.json \
  --data ./tmp/syn-data \
  --output ./tmp/syn-out \
  --smoke
```

---

## 2. One-Shot Vertex AI Colab Enterprise (NVIDIA L4) Reproduction

To reproduce the full `384×768` Cityscapes + KITTI Zero-Shot `SegFormer-B5 → MobileNetV4` study:

1. Upload [`notebooks/colab_enterprise_l4_one_shot.ipynb`](../notebooks/colab_enterprise_l4_one_shot.ipynb) to Colab Enterprise (`g2-standard-4` + NVIDIA L4 24GB, `100GB+` disk).
2. In Cell 1, select `DATA_SOURCE="hf_cityscapes"`, `ENABLE_KITTI_ZEROSHOT=True`, and `RUN_MODE="full"` (or `"pilot"` for a fast 1.5h sweep).
3. Click **Runtime → Run All**. The notebook verifies its embedded SHA-256 source bundle, installs missing dependencies, runs the 12 unit tests, downloads Cityscapes & KITTI, and executes every stage end-to-end with automatic resumability.

---

## 3. Manual Staged CLI Execution

```bash
# 1. Build city-disjoint Cityscapes manifest
python -m seg_kd prepare-data \
  --dataset cityscapes \
  --data /data/cityscapes \
  --output ./manifests/cityscapes-manifest.json \
  --seed 378

# 2. Build KITTI Semantics zero-shot manifest
python -m seg_kd prepare-data \
  --dataset kitti \
  --data /data/kitti \
  --output ./manifests/kitti-manifest.json

# 3. Run all resumable stages
python scripts/run_l4_study.py \
  --plan configs/l4-plan.json \
  --manifest ./manifests/cityscapes-manifest.json \
  --data /data/cityscapes \
  --kitti-manifest ./manifests/kitti-manifest.json \
  --kitti-data /data/kitti \
  --output ./runs/driving-seg-kd-l4 \
  --stage all \
  --device auto
```

---

## 4. Regenerating Reports & Publication Figures

With `outputs/v3-max/` and `outputs/v4-final/` present locally, regenerate `reports/final-summary.json` and all PNG/SVG figures deterministically via:

```bash
python scripts/generate_reports_and_figures.py
```

---

## 5. Verified Runtime Specification

| Item | Verified Configuration |
|---|---|
| GPU Training Runtime | NVIDIA L4 (`22.03 GiB` VRAM), CUDA `12.8`, PyTorch `2.11.0+cu128`, Python `3.13.15`, Linux |
| CPU Benchmark Runtime | 4-thread ONNXRuntime Static QDQ INT8 / FP32 (`384×768`, `batch=1`) |
| Local Verification | Windows 11, Python `3.12`, `unittest` 12/12 PASS |
| Random Seeds | Paired study `378, 379, 380`; deployment reference seed `378` |
