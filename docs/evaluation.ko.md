<div align="right"><a href="./evaluation.md">English</a> | <strong>한국어</strong></div>

# 평가 방법

최초 노트북은 공식 GTSRB 테스트 자료를 체크포인트 및 하이퍼파라미터 선택에 사용했습니다. 기록된 증류 정확도 80.97%, 기준 모델 54.61%, 교사 모델 88.37%는 과거 실험에서 선택한 에포크의 결과이며, 독립적인 최종 테스트 성능이 아닙니다. 기존 속도 측정에는 `model.predict()` 호출 비용이 포함되어 배포 속도 개선을 입증하지 못합니다.

## 모델 선택과 최종 측정 분리

개선 실험에서는 공식 학습 자료를 학습, 검증, 동결 평가 집합으로 나눕니다. 클래스별 층화를 적용하면서 트랙 ID를 분리 단위로 사용합니다. 명세에는 이미지 목록과 해시를 기록합니다. 체크포인트, 학습 일정, 증류 설정은 검증 집합으로 선택합니다. 동결 평가 이미지는 선택 과정에서 제외하며 설정을 고정한 뒤 평가합니다.

동결 평가 집합은 이번 개선 실험에서 새로 분리했지만 최초 노트북의 전체 학습 자료에 포함됐던 이미지입니다. 모든 과거 모델에 완전히 미사용한 자료라고 표현하면 안 됩니다. ImageNet 사전 학습 자료와의 중복 여부도 이번 분할만으로 입증하거나 배제하지 않습니다. 공식 테스트 결과는 과거 노출 이력이 있는 감사 평가로 표시합니다. 최초 실험과 개선 실험은 모델 선택 조건이 달라 동일 조건의 단일 전후 비교로 제시하지 않습니다.

## 추론 전 동결

공개 모델의 평가를 반복할 때는 공개된 동결 기록을 그대로 사용합니다. 아래 명령은 새로운 실험에서 최종 측정 전에 수행하는 개발 절차의 예시이며, 공개 체크포인트를 실행하기 위한 사전 작업이 아닙니다.

```powershell
.\launch-kd.bat freeze --model teacher=artifacts/runtime/models/teacher.keras --model baseline=artifacts/runtime/models/baseline.keras --model student=artifacts/runtime/models/student.keras --manifest reports/split-manifest.json --config reports/selected-configuration.json --output reports/evaluation-freeze.json --source-root .
```

동결 기록은 체크포인트 해시, 분할 명세, 선택한 설정 및 관련 소스 파일을 연결합니다. `locked_holdout`과 `official_test`는 동결 기록이 없거나 일치하지 않으면 평가를 거부합니다. 보고서는 정확한 체크포인트와 평가 분모를 명시해야 합니다. 최종 결과를 본 뒤 변경된 모델을 감추려고 동결 기록을 새로 만들면 안 됩니다.

## 분류 성능

```powershell
.\launch-kd.bat evaluate --model artifacts/runtime/models/student.keras --manifest reports/split-manifest.json --data DATASET_ROOT --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/student-final
```

정답 수와 전체 평가 이미지 수에 따른 정확도, macro F1, 클래스별 재현율, 혼동행렬 및 평가 ID를 보고합니다. 같은 조건의 여러 시드를 반복하여 선택한 학습 방법의 변동을 확인합니다. 각 시드에서 기준 모델과 증류 모델의 구조, 분할, 최적화 예산 및 시드 정책을 일치시킵니다. 가장 좋은 시드만 제시하지 말고 완료한 모든 시드와 중단한 실행을 공개합니다.

모든 후보에 동일한 검증 선택 기준을 적용하며, 미리 정한 개선 기준을 만족하지 못한 후보는 중단합니다. 최종 평가 결과는 해당 방법의 일반화 성능을 기록하는 용도이며 탐색을 재개하는 데 사용하지 않습니다. 더 작은 구조나 다른 환경으로 바뀐 실험은 별도 비교로 표시합니다.

## 공식 테스트 감사 평가

```powershell
.\launch-kd.bat official-test-manifest --data DATASET_ROOT --training-manifest reports/split-manifest.json --freeze reports/evaluation-freeze.json --output .local/official-test.json
.\launch-kd.bat evaluate --model artifacts/runtime/models/student.keras --manifest .local/official-test.json --data DATASET_ROOT --split official_test --freeze reports/evaluation-freeze.json --output .local/student-official-audit
```

모델 선택을 동결한 뒤에만 감사 평가를 시작합니다. 공식 테스트 이미지 경로와 라벨은 데이터셋의 테스트 CSV에서 읽습니다. 과거 노출 이력이 있는 추가 점검이며, 독립적인 미사용 벤치마크나 다른 논문보다 우수하다는 근거로 표현하지 않습니다.

## CPU 비용 및 양자화

교사, 기준 모델, 증류 모델은 동일 CPU, 이미지 목록, 스레드 설정 및 예열한 호출 경로에서 측정합니다. 중앙값과 꼬리 지연, 처리량, 파일 크기, 파라미터 수, 측정 범위와 환경을 보고합니다. 프레임워크 시작 시간과 전처리 범위를 구분하며, 속도 결과는 사용한 하드웨어에 한정됩니다.

INT8 보정에는 학습 이미지만 사용합니다. FP32 TFLite와 INT8 TFLite는 원본 모델과 같은 고정 입력으로 평가합니다. 채택 여부는 정확도 손실과 실제 CPU 지연을 함께 판단하며 파일 크기 감소만으로 속도 개선을 주장하지 않습니다. 허용 정확도 손실은 최종 평가 전에 고정합니다. INT8이 기준을 만족하지 못하면 중단 후보로 기록하고 검증된 FP32 경로를 권장합니다.

## 재현 범위

모델, 소스, 데이터 해시를 통해 다른 사용자가 재학습 없이 평가를 반복할 수 있습니다. 재학습의 정확한 진행 경로나 CPU 시간은 드라이버, TensorFlow와 하드웨어에 따라 달라질 수 있습니다. 동일 컴퓨터에서 수행한 Windows 및 Linux 점검은 별도 환경 검증이며 외부 독립 평가자가 확인한 결과는 아닙니다. 다운로드와 간단한 추론은 배포 및 실행 경로를 검증하며, 보고서에 전체 평가를 명시하지 않았다면 전체 데이터 성능 검증을 뜻하지 않습니다.

최종 측정값은 결과 보고서에 기록합니다. 평가 방법을 구현한 사실만으로 학습이나 최종 성능 측정을 완료했다고 주장하지 않습니다.
