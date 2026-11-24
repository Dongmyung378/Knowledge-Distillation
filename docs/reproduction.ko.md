# 재현 및 실행 가이드

[한국어 README](../README.ko.md) · [English](./reproduction.md) · [아키텍처](./architecture.ko.md) · [실험 결과](./results.ko.md)

## 1. 로컬 CPU 즉시 검증 (약 2분)

GPU나 대용량 외부 데이터셋 없이 전체 파이프라인의 수학적 무결성, 양자화 감사, ONNX 변환, SHA-256 평가 동결, 사후 온도 보정을 로컬에서 검증할 수 있습니다.

```bash
# 1. 가상환경 구성 및 패키지 설치
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -e ".[export]"

# 2. 런타임 진단 및 단위·통합 테스트 (12 tests)
python -m seg_kd doctor
python -m unittest discover -s tests -v

# 3. 합성 도로 데이터 End-to-End 스모크 파이프라인 실행
python -m seg_kd prepare-data --dataset synthetic \
  --data ./tmp/syn-data --output ./tmp/syn-manifest.json --image-size 128x256
python scripts/run_l4_study.py \
  --plan configs/l4-plan.json \
  --manifest ./tmp/syn-manifest.json \
  --data ./tmp/syn-data \
  --output ./tmp/syn-out \
  --smoke
```

---

## 2. Vertex AI Colab Enterprise (NVIDIA L4) 원샷 재현

전체 `384×768` Cityscapes + KITTI Zero-Shot + `SegFormer-B5` → `MobileNetV4` 연구를 재현하려면 자립형 노트북을 사용합니다.

1. [`notebooks/colab_enterprise_l4_one_shot.ipynb`](../notebooks/colab_enterprise_l4_one_shot.ipynb) 파일을 Colab Enterprise (`g2-standard-4` / NVIDIA L4 24GB, 디스크 `100GB+`)에 업로드합니다.
2. 셀 1의 폼 파라미터에서 `DATA_SOURCE="hf_cityscapes"`, `ENABLE_KITTI_ZEROSHOT=True`, `RUN_MODE="full"`(또는 빠른 검증 시 `"pilot"`)을 선택합니다.
3. **런타임 → 모두 실행**(Run All)을 클릭하면 내장된 SHA-256 소스 번들을 자동 검증·압축 해제한 뒤 데이터 다운로드, 단위 테스트, 단계별 학습·QAT·ONNX 내보내기·동결 평가·시각화·ZIP 번들 생성까지 자동 완주합니다.

---

## 3. CLI 단계별 수동 실행

직접 데이터 디렉터리를 보유한 Linux GPU 워크스테이션에서는 `seg_kd` CLI와 오케스트레이터를 단계별로 실행할 수 있습니다.

```bash
# 1. Cityscapes 매니페스트 생성 (도시 단위 분리 + SHA-256 해싱)
python -m seg_kd prepare-data \
  --dataset cityscapes \
  --data /data/cityscapes \
  --output ./manifests/cityscapes-manifest.json \
  --seed 378

# 2. KITTI Semantics Zero-Shot 매니페스트 생성
python -m seg_kd prepare-data \
  --dataset kitti \
  --data /data/kitti \
  --output ./manifests/kitti-manifest.json

# 3. 단계별 재개 가능 실행 (teacher -> ablation -> pareto -> export -> freeze -> eval -> plots)
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

## 4. 리포트 및 포트폴리오 차트 재생성

`outputs/v3-max/` 및 `outputs/v4-final/` 번들이 준비된 상태에서 아래 명령을 실행하면 `reports/final-summary.json`과 `assets/figures/`의 모든 차트·한/영 SVG를 결정론적으로 재생성합니다.

```bash
python scripts/generate_reports_and_figures.py
```

---

## 5. 환경 및 하드웨어 명세 (검증 완료 기준)

| 항목 | 검증 환경 |
|---|---|
| GPU 학습 런타임 | NVIDIA L4 (`22.03 GiB` VRAM), CUDA `12.8`, PyTorch `2.11.0+cu128`, Python `3.13.15`, Linux |
| CPU 벤치마크 런타임 | 4-thread ONNXRuntime Static QDQ INT8 / FP32 (`384×768`, `batch=1`) |
| 로컬 단위 테스트 검증 | Windows 11, Python `3.12`, `unittest` 12/12 PASS |
| 난수 시드 | 주력 쌍체 검증 `378, 379, 380` · 배포 기준 시드 `378` |
