> **앞선 4개 후보 연구의 보존 문서입니다.** 아래 결과, 기본 다운로드 명세와 학습 명령은 이전 연구에 해당합니다. 최신 9개 조합 연구에는 [확장 결과와 명시적 재현 명령](./full-grid.ko.md) 및 [교사와 학생 분석](./teacher-student-analysis.ko.md)을 사용하십시오. 기존 실행기의 기본 명세를 최신 전체 조합 모델로 간주하지 않습니다. 원본 수치와 설명은 아래에 유지합니다.

<div align="right"><a href="./training.md">English</a> | <strong>한국어</strong></div>

# 실험 예산을 제한한 Kaggle 학습

전체 학습은 GPU가 활성화된 Kaggle에서 실행합니다. 실험 진입점은 노트북 컴퓨터에서 실제 학습을 실행하는 것을 거부합니다. 로컬 CPU 테스트는 작은 합성 자료로 구현을 확인하며, GTSRB 성능을 입증하는 학습 결과로 사용하지 않습니다.

실험 계획은 [`configs/experiment-plan.json`](../configs/experiment-plan.json)에 고정합니다. Kaggle 실행에서는 실제 Python, TensorFlow, Keras, GPU와 데이터 목록을 기록합니다. Kaggle 시스템의 Python과 TensorFlow는 프로젝트와 독립적으로 변경될 수 있어, `scripts/setup_kaggle_runtime.py`가 Python 3.12.15를 준비하고 고정된 TensorFlow 2.19.0 및 호환 CUDA 의존성을 별도 환경에 설치합니다. 시스템 설치와 다른 프로젝트의 환경을 변경하지 않습니다. 모델과 데이터 해시, 저장된 체크포인트 및 호환성 검증으로 로컬에서 재현 가능한 범위를 확인합니다.

## 단계

1. **탐색:** MobileNetV2 입력 범위를 수정하고 분류 헤드를 최대 5에포크 학습합니다. 이후 배치 정규화 계층을 고정한 상태로 백본의 마지막 최대 60개 계층을 최대 15에포크 미세 조정합니다. 시드 378에서 독립 학생과 증류 후보 4개를 같은 조건으로 학습합니다.
2. **반복 확인:** 검증 집합으로만 선택합니다. 선택한 고정 설정과 기준 모델을 시드 379 및 380에서 반복하며, 시드 378은 탐색에서 완료한 짝을 재사용합니다. 각 시드의 기준 모델과 증류 모델은 동일한 분할과 초기 가중치를 사용합니다.
3. **동결:** 최종 동결 평가 및 공식 테스트 감사 평가 전에 완료한 각 체크포인트, 선택한 설정, 데이터 명세와 소스 해시를 고정합니다. 이 단계는 해시 계산이므로 Kaggle 출력을 내려받은 뒤 로컬에서도 실행할 수 있습니다.

학생 모델은 34,406개 파라미터를 유지합니다. 기준 학생과 증류 학생은 동일한 배치 크기, 학습률 정책, 최대 30에포크, 조기 종료 patience 5, 증강 및 클래스 가중치를 사용합니다. 검증에서 선택한 최적 체크포인트를 다시 불러와 확인합니다. 증류 과정에서는 교사 가중치를 변경하지 않습니다.

## 탐색 범위 및 중단 기준

| 후보 | 온도 | 정답 손실 비중 alpha |
| --- | ---: | ---: |
| `T3-a03` | 3 | 0.3 |
| `T5-a03` | 5 | 0.3 |
| `T10-a03` | 10 | 0.3 |
| `T5-a05` | 5 | 0.5 |

검증에서는 클래스 가중치를 적용하지 않은 분류 정확도를 사용합니다. 기준 모델 대비 **최소 0.2%p** 향상한 후보만 선택 대상으로 인정합니다. 정확도가 같으면 검증 교차 엔트로피가 낮은 후보, 이후 계획에 명시한 순서를 적용합니다. 반복 확인에서는 시드별 평균 정확도 향상이 최소 0.2%p이고 3개 중 최소 2개 시드에서 개선해야 합니다. 확인 기준을 만족하지 못하면 기준 모델을 배포 권장 모델로 유지하고 실패한 증류 결과도 보관합니다.

배포 시드 **378**은 최종 평가 전에 고정합니다. 최종 결과를 근거로 다른 시드, 온도, 구조 또는 체크포인트를 선택하지 않습니다. 정해진 patience 동안 검증 지표가 개선되지 않으면 조기 종료하며, 최종 지표를 본 뒤 탐색 후보를 추가하지 않습니다.

## Kaggle 실행

GPU를 활성화하고 소스 패키지와 고정된 GTSRB 버전을 연결합니다. 시스템 Python은 준비 도구를 실행하고, 준비 도구는 격리된 Python 3.12.15, 고정된 TensorFlow 및 CUDA 라이브러리로 요청한 단계를 실행합니다. 프로젝트를 `/tmp/kd-source`에 추출한 경우 다음 명령을 사용합니다.

```bash
python /tmp/kd-source/scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- -m knowledge_distillation train --data DATASET_ROOT --output /kaggle/working/kd-run --plan /tmp/kd-source/configs/experiment-plan.json --stage pilot
python /tmp/kd-source/scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- -m knowledge_distillation train --data DATASET_ROOT --output /kaggle/working/kd-run --plan /tmp/kd-source/configs/experiment-plan.json --stage confirmation
python /tmp/kd-source/scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- -m knowledge_distillation train --data DATASET_ROOT --output /kaggle/working/kd-run --plan /tmp/kd-source/configs/experiment-plan.json --stage freeze
```

`DATASET_ROOT`를 검증한 데이터의 실제 연결 또는 준비 경로로 바꿉니다. 모든 단계는 동일한 출력 폴더를 사용합니다. CPU 전용 로컬 실행기는 학습을 거부하므로 위 명령에 사용하지 않습니다.

자동 생성한 비공개 Kaggle 실행기는 세 단계를 프로그램에서 호출하고, 시작 전에 데이터 해시를 확인합니다. 학습 공간에는 최초 학습 CSV와 그 CSV가 지정한 이미지만 복사하며 공식 테스트 CSV는 읽지 않습니다. 로컬에서 작업을 시작하지 않고 소스 패키지만 준비하는 명령은 다음과 같습니다.

```bash
python scripts/package_kaggle.py --output .local/kaggle-package --kernel-id YOUR_ACCOUNT/YOUR_PRIVATE_SLUG --manifest reports/split-manifest.json
```

출력 폴더는 아직 존재하지 않아야 합니다. 계정과 슬러그는 자신의 Kaggle 계정 및 비공개 스크립트 이름으로 바꿉니다. 생성한 설정은 T4 GPU, 인터넷 연결과 데이터셋을 지정하고 커널을 비공개로 유지합니다. 패키지 준비는 파일만 생성하며 업로드 및 실행 제출은 별도 동작입니다. 내장 소스 압축에는 명시된 코드, 계획 및 영어와 한국어 Markdown 설명이 포함됩니다. 이는 학습용 전달 파일이며 공개 저장소 전체의 압축 파일은 아닙니다. 제출 성공만으로 학습 성공을 주장하지 않으며, 커널 완료와 출력 파일의 다운로드 및 검증이 필요합니다.

자동 생성한 실행 스크립트는 시스템 인터프리터로 다음 준비 도구를 호출합니다.

```bash
python scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- /tmp/kd-source/kaggle-worker.py
```

`--` 뒤 인자는 별도 환경의 Python에 전달됩니다. 준비 도구는 공식 uv 압축 파일과 실행 파일 해시를 검증하고, Kaggle에 할당된 GPU 설정을 유지하며, 별도로 설치한 CUDA 라이브러리를 우선 사용합니다. 실행 스크립트를 시작하기 전에 실제 GPU 행렬 곱을 확인합니다. 기본적으로 패키지는 `/tmp`에 두어 공개 모델 출력에 포함되지 않도록 합니다. Windows 및 일반 Linux 환경에서는 GPU 준비 전에 거부합니다. 결과 JSON에는 버전과 GPU 정보를 기록하며 인증 정보와 무관한 환경 변수를 포함하지 않습니다.

중단된 단계는 설정, 분할 및 모델 해시가 일치하는 완료 실행을 재사용할 수 있습니다. 시작한 출력 폴더에서 계획을 변경하면 거부합니다. 완료한 탐색이나 반복 보고서를 덮어쓰면서 모델 선택을 다시 수행할 수 없습니다. 동결 기록이 있으면 해당 실행의 추가 학습 및 재선택을 금지합니다.

## 출력 및 공개

실행 폴더에는 `plan-snapshot.json`, `splits.json`, `pilot-report.json`, `selected-configuration.json`, `confirmation-report.json`, `evaluation-freeze.json`, 표준 `.keras` 체크포인트 및 학습 기록을 저장합니다. 공개 결과 보고서는 완료한 실행, 중단 판단과 실제 최종 지표를 명시합니다. 성공한 후보만 제시하지 말고 실패하거나 중단한 실험도 구조화된 기록으로 남깁니다.

`splits.json`은 Kaggle 실행 폴더 내부의 파일명입니다. 공개 저장소는 동일한 학습 분할을 `reports/split-manifest.json`으로 유지합니다. 평가 반복 명령은 별도 명세를 복제하지 않고 공개된 이 경로를 사용합니다.

공개 모델, 평가 명세와 해시를 통해 재학습 없이 저장 모델을 검증할 수 있습니다. 학습 기록과 계획은 방법의 근거를 제공합니다. 최초 노트북과 최초 성능은 과거 개발 자료로 보관하며, 개선 실험은 별도 평가 방법을 사용합니다. 개선했다고 최초 성능이 독립적인 미사용 평가 결과로 바뀌지는 않습니다.

## 학습 후 변환과 최종 평가

완료된 Kaggle 실행 결과를 내려받으면 `scripts/evaluate_release.py`로 소스, 저장 모델과 최초 동결 기록을 검증할 수 있습니다. 이 도구는 FP32 및 INT8 변환, 검증 집합의 양자화 평가와 CPU 측정을 수행한 뒤, 최종 추론 전에 배포 모델 전체를 다시 동결합니다. 모든 대응 체크포인트를 최종 평가 집합에 적용하며, 과거에 모델 선택에 사용했던 공식 테스트는 별도 참고 평가로 기록합니다. 결과 폴더는 새 경로를 지정하고, 공식 테스트 이미지도 포함한 검증된 전체 데이터셋을 사용합니다.

```powershell
.\launch-kd.bat --setup-only
.\.venv\Scripts\python.exe scripts/evaluate_release.py --source . --run .local/kaggle-output/kd-results --data .local/data/gtsrb-v1 --output .local/release-evaluation
```

Linux에서는 `bash launch-kd.sh --setup-only`로 CPU 환경을 준비한 뒤 `.venv/bin/python`으로 같은 도구를 실행합니다. `--run`은 `kernel-completion.json`, `pilot-report.json`, `confirmation-report.json`, `splits.json`과 최초 동결 파일이 들어 있는 실제 다운로드 폴더를 지정합니다. 모델, 설정과 소스의 해시가 기존 기록과 일치해야 합니다. 공개 성능만 직접 재현하려면 공개된 배포 동결 파일과 [저장 모델 평가 명령](./runtime.ko.md)을 사용합니다.

실측 보고서로 결과 그림을 다시 생성할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py --summary reports/release-summary.json --output .local/reproduced-figures
```

참고: [Keras 전이 학습](https://keras.io/guides/transfer_learning/), [Keras 지식증류 예제](https://keras.io/examples/vision/knowledge_distillation/), [Kaggle API](https://github.com/Kaggle/kaggle-api).
