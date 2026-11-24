# 원샷 검증 및 L4 전체 재현 가이드

[한국어 README](../README.ko.md) · [English](./reproduction.md) · [설계 판단 및 과정](./methodology-and-decisions.ko.md) · [실험 결과](./results.ko.md)

본 프로젝트는 의미 없는 합성 데이터 데모 대신, **실제 학습·동결된 ONNX FP32/INT8 배포 모델(`v4.0-models`)과 `v3-max` / `v4-final` 동결 리포트를 원샷으로 검증**하거나 **NVIDIA L4에서 전체 실험을 처음부터 재현**할 수 있도록 단일 자립형 노트북으로 정리했습니다.

---

## 1. 원샷 노트북으로 즉시 검증하기 (권장 · CPU/GPU 모두 지원)

[`notebooks/colab_enterprise_l4_one_shot.ipynb`](../notebooks/colab_enterprise_l4_one_shot.ipynb) 파일 하나만 Google Colab, Vertex AI Colab Enterprise, 또는 로컬 Jupyter에 올리고 **런타임 → 모두 실행**(Run All)을 누르면 됩니다. (외부 Git 클론·Google Drive 마운트 불필요 — 코드·테스트·동결 매니페스트 23개 파일이 SHA-256 검증 번들로 내장됨)

### 모드 A: `RUN_MODE = "verify_saved"` *(기본값 · 1~3분 원샷 검증)*
1. **내장 코드 무결성 & 단위 테스트**: SHA-256 번들 추출 후 `seg_kd doctor` 및 `unittest` **12/12 PASS** 확인
2. **배포 모델 SHA-256 동결 검증**: GitHub Release [`v4.0-models`](https://github.com/Dongmyung378/Knowledge-Distillation/releases/tag/v4.0-models)에서 실제 배포 ONNX 모델 4종(`v3 Clean` FP32/INT8, `v4 Robust` FP32/INT8)을 자동 다운로드해 `evaluation-freeze*.json`의 SHA-256과 **바이트 단위 100% 일치** 검증
3. **실제 CPU ONNXRuntime 속도 실측**: 현재 머신(`384×768`, `batch=1`, 4스레드)에서 FP32 vs Static QDQ INT8 지연 시간(p50 ms, FPS)과 가속비·용량 축소비(`-72.5%`)를 실시간 측정하고 Validation-Only INT8 게이트(4/4) 로직 검증
4. ***(선택) 실제 데이터셋 라이브 재평가 (`EVALUATE_LIVE_DATASETS = True`)***: 로그인 없이 `Chris1/cityscapes` (`locked_eval` 500장)와 공식 KITTI Semantics (`200장`)를 자동 다운로드해, 실제 ONNX INT8 모델의 `67.95% / 66.96%` mIoU · KITTI Zero-Shot · 온도 보정(`T*=1.70`) ECE를 눈앞에서 직접 재측정해 동결 리포트와 대조

### 모드 B: `RUN_MODE = "reproduce_full"` *(NVIDIA L4 24GB · 전체 재학습·재현)*
- `Chris1/cityscapes` (`384×768`, `train 2,529 / val 446 / locked_eval 500`) + KITTI(`200장`) 자동 준비
- `SegFormer-B5` 교사 → `v3-max` (3시드 쌍체 비교 + 3-Tier Pareto + QAT-KD + ONNX INT8 게이트 + 동결 평가) → `v4-final` (Medium Baseline + `384×768` 4종 소거 + 비대칭 강건성 KD 45ep + BF16 교사 QAT-KD 6ep + ONNX INT8 게이트 + 사후 온도 보정) 전체 원샷 재현

---

## 2. 로컬 CLI 원샷 검증 (노트북 없이 터미널 실행)

```bash
# 1. 패키지 설치 (ONNXRuntime 포함)
pip install -e ".[export]"

# 2. 단위·무결성 테스트 12종 실행
python -m unittest discover -s tests -v

# 3. GitHub Release 배포 ONNX 모델 4종 SHA-256 동결 대조 + CPU 4스레드 속도 실측 + 마스터 요약 감사
python scripts/verify_release_models.py --threads 4 --runs 20
```

---

## 3. 공개 배포 모델 아티팩트 ([GitHub Release `v4.0-models`](https://github.com/Dongmyung378/Knowledge-Distillation/releases/tag/v4.0-models))

| 파일명 | 크기 | 동결 SHA-256 (앞 16자리) | 용도 |
|---|---:|---|---|
| `v3-max-mobilenetv4-medium-int8.onnx` | 7.90 MiB | `8cda8dff653f4561` | 주간 최고 성능 INT8 (`67.95%`, `40.49 FPS`, Gate PASS) |
| `v3-max-mobilenetv4-medium-fp32.onnx` | 28.71 MiB | `434c8e4243a55f35` | `v3-max` FP32 기준 대조 모델 (`68.51%`, `26.24 FPS`) |
| `v4-final-robust-medium-int8.onnx` | 7.90 MiB | `fac9853786e9bb35` | 악천후·노이즈 강건 INT8 (`59.59%` 노이즈 교사 추월, Gate PASS) |
| `v4-final-robust-medium-fp32.onnx` | 28.71 MiB | `1cb99d88e685d1f6` | `v4-final` FP32 기준 대조 모델 (`60.61%` 노이즈) |

---

## 4. 검증 완료 환경 명세

| 항목 | 검증 환경 |
|---|---|
| GPU 학습 런타임 | NVIDIA L4 (`22.03 GiB` VRAM), CUDA `12.8`, PyTorch `2.11.0+cu128`, Python `3.13.15`, Linux |
| CPU 벤치마크 런타임 | 4-thread ONNXRuntime Static QDQ INT8 / FP32 (`384×768`, `batch=1`) |
| CI / 로컬 단위 검증 | GitHub Actions Linux & Windows 11, `unittest` 12/12 PASS + Release SHA-256 4/4 일치 |
| 난수 시드 | 3시드 쌍체 검증 `378, 379, 380` · 배포 기준 시드 `378` |
