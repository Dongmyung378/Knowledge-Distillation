<div align="right">
  <a href="./README.md">English</a> | <strong>한국어</strong>
</div>

# 지식 증류 기반 교통 표지판 인식 모델 경량화

교통 표지판 영역을 잘라낸 이미지를 입력하면 어떤 표지판인지 분류합니다. 큰 모델의 예측을 작은 모델이 학습하도록 해 인식 품질을 유지하면서 실행에 필요한 메모리와 CPU 시간을 줄일 수 있는지 살펴보는 프로젝트입니다. 표지판이 있는 이미지를 분류하며, 전체 도로 장면에서 표지판의 위치를 찾는 기능과는 구분합니다.

GTSRB의 43개 교통 표지판 클래스를 분류하는 TensorFlow 모델 경량화 프로젝트입니다. MobileNetV2 교사의 예측 분포를 **34,406개 파라미터**의 경량 CNN에 전달합니다. 동일 조건의 독립 학습 기준 모델, 전체 9개 설정과 순수 증류 대조를 비교하며 실험 결과와 실제 배포 결정을 구분합니다.

**최초 개발 기간은 2026년 2월부터 4월입니다.** 이후의 개선은 이 기간을 유지하고 통제된 실험, CPU 배포 비용 측정과 저장 모델 재현 절차를 추가합니다.

[대표 배포와 프로젝트 전체 결과](./docs/deployment.ko.md) | [전체 9개 조합 연구](./docs/full-grid.ko.md) | [학생이 교사를 넘는 이유 분석](./docs/teacher-student-analysis.ko.md) | [구조 설명](./docs/architecture.ko.md) | [최초 실험 노트북](./notebooks/knowledge_distillation_gtsrb.ipynb)

## 실험을 이해하기 위한 안내

| 결과에 등장하는 용어 | 이 프로젝트에서의 의미 |
| --- | --- |
| 교사, 학생과 지식 증류 | 큰 교사 모델이 표지판별 확률을 제공하고 작은 학생 모델이 그 분포를 학습합니다. 교사가 고른 정답 후보 하나만 그대로 복사하는 방식은 아닙니다. |
| 동일 조건의 독립 학습 기준 모델 | 학생과 같은 작은 구조를 사용하되 교사의 도움 없이 정답 레이블로 학습한 모델입니다. 시작 조건을 맞춰 학습 방식의 차이를 비교합니다. |
| 온도와 alpha | 온도는 증류에 사용하는 교사와 학생의 확률 분포를 부드럽게 조정합니다. alpha는 정답 손실과 교사 분포를 따라가는 손실을 섞는 계수이며, 실제 기울기에서 차지하는 비율은 아닙니다. |
| 시드와 시드별 대응 비교 | 시드는 초기 가중치나 데이터 순서 같은 의사 난수 선택의 시작값입니다. 세 시드는 시작 조건을 바꿔 학습을 반복한 것이며, 각 시드 안에서는 기준 모델과 증류 모델의 시작 조건을 맞춥니다. 데이터셋 세 개를 뜻하지 않습니다. |
| 검증 집합과 참고 평가 집합 | 검증 집합은 학습 설정과 저장 모델을 고르는 데 사용합니다. 참고 평가 집합은 선택 이후의 결과를 보고하며, 과거에 결과를 확인한 자료라는 한계도 아래에 유지합니다. |
| FP32 TFLite와 INT8 | TFLite는 경량 추론 실행기에서 사용하는 저장 형식입니다. FP32는 부동소수점 값을 유지하고, INT8 양자화는 수치 정밀도를 낮춰 파일 크기와 실행 비용을 줄이려는 방식이므로 품질 손실을 함께 확인해야 합니다. |

표지판 트랙은 같은 실제 표지판을 촬영한 이미지들의 묶음입니다. 트랙을 한 집합 안에 두어 서로 연관된 사진이 학습과 평가에 나뉘어 들어가는 것을 막습니다. 에포크는 학습 데이터를 한 차례 학습하는 단위이고, 체크포인트는 학습 중 특정 시점의 가중치를 저장한 모델입니다.

설정 이름 T5-a05는 온도 5와 alpha 0.5를 뜻하고, T5-a01은 alpha 0.1을 사용합니다. 실험에서 선택한 설정과 실제 제공하는 배포 모델의 결정 근거를 각각 설명합니다.

**표를 읽는 방법:** 정확도는 맞힌 이미지의 비율입니다. 정밀도는 특정 클래스로 예측한 것 중 맞힌 비율이고, 재현율은 실제 그 클래스인 사례 중 찾아낸 비율입니다. Macro F1은 이 둘을 함께 평가한 뒤 클래스를 같은 비중으로 평균하고, 균형 정확도는 클래스별 재현율의 평균입니다. %p는 백분율을 직접 뺀 차이입니다. ± 뒤의 표본 표준편차는 학습 시드에 따른 변동이며 신뢰구간이 아닙니다. CPU p50은 처리 시간의 중앙값이고 p95는 측정한 요청 중 상대적으로 느린 쪽의 시간을 보여줍니다. 파라미터 수, 파일 크기와 실행 시간은 각각 다른 비용입니다.

## 대표 배포: 기존 87.06% 모델 유지

대표 모델은 **T5-a05, 시드 378, FP32 TFLite**를 유지합니다. 이전에 공개한 체크포인트이며 재학습하거나 가중치를 변경하지 않았습니다. 전체 9개 조합 연구는 그대로 남겼으며 검증으로 선택된 설정은 **T5-a01**입니다. 실험에서 선택한 설정과 대표 배포 모델은 역할이 다릅니다.

| 시드 378 모델, 동일 참고 평가 5,850장 | 정답 / 평가 | 정확도 | macro F1 | Balanced accuracy |
| --- | --- | --- | --- | --- |
| 교사 | 4,887 / 5,850 | 83.54% | 0.7465 | 0.7548 |
| 동일 조건의 독립 학습 기준 | 4,303 / 5,850 | 73.56% | 0.6253 | 0.6329 |
| **유지한 대표 배포, T5-a05 FP32** | **5,093 / 5,850** | **87.06%** | **0.7903** | **0.7817** |
| 9개 조합의 검증 선택, T5-a01 | 5,013 / 5,850 | 85.69% | 0.7719 | 0.7620 |

| 선택 또는 평가 범위 | 유지한 T5-a05 설정 | 연구에서 선택한 T5-a01 설정 |
| --- | --- | --- |
| 세 시드의 검증 정확도 평균 | 86.65% | 87.08% |
| 세 시드의 참고 정확도 평균 ± 표본 SD | 87.51% ± 2.18%p | 87.81% ± 2.08%p |
| 시드 378의 참고 정확도 | 87.06% | 85.69% |

연구는 세 시드의 검증 정확도 평균으로 설정을 선택하고 시드 378은 사전에 고정했습니다. 새 설정의 참고 정확도 평균이 0.30%p 높다는 사실만으로 확실한 배포 개선을 입증하지는 않으며, 고정된 시드 378의 체크포인트는 기존보다 1.37%p 낮습니다. 기존 모델 유지는 배포의 연속성과 관찰된 참고 성능을 고려한 결정입니다. 이미 확인한 결과를 고려한 후속 배포 결정이라는 점을 공개하며, 연구의 선택 결과를 바꾸거나 미사용 평가로 표현하지 않습니다. 정책은 [`reports/deployment/deployment-policy.json`](./reports/deployment/deployment-policy.json)에 기록합니다.

![유지한 T5-a05 대표 모델의 성능과 해당 모델의 CPU 배포 비용](./assets/figures/ko/knowledge-distillation-release.png)

## 프로젝트 전체 결과와 교사 및 학생 분석 결론

대표 학생은 같은 시드의 독립 학습 기준보다 참고 정확도가 **13.50%p**, 교사보다 **3.52%p** 높습니다. T5-a05의 세 시드에서 기준 모델 대비 정확도 차이는 평균 **+9.43%p ± 3.93%p**입니다. 파라미터는 교사의 2,313,067개에서 **34,406개로 약 98.5% 감소**했습니다. 앞선 동일 CPU 벤치마크의 Keras p50은 교사 **3.823ms**, 해당 학생 **1.040ms**입니다. 대표 FP32 TFLite는 **p50 0.429ms**, **p95 0.456ms**, **140.0KiB**를 기록했습니다. 한 컴퓨터의 실측값이며 다른 기기의 속도를 보장하지 않습니다.

대표 학생은 **교사의 오답 473개를 수정하고, 교사가 맞힌 267개를 틀려 정답이 206개 증가**했습니다. 순수 증류 학생도 세 시드 모두에서 교사 정확도를 넘었으므로 학생의 정답 CE 추가만으로 이 결과를 설명할 수 없습니다. 증류는 교사의 최댓값 클래스를 그대로 복제하는 대신 확률 분포를 학습합니다. 서로 다른 구조, 학생 전체 가중치의 적응, BN 정책과 최적화 경로는 서로 다른 판단을 만들 수 있는 설명 후보입니다. 각 요인의 인과 기여는 분리해서 입증하지 않았으며 교사와 혼합 증류 학생 모두 정답 레이블을 사용합니다. [저장 예측과 대조 실험의 상세 분석](./docs/teacher-student-analysis.ko.md)을 제공합니다.

이 프로젝트는 동일 조건의 실험 설계, 모델 경량화, 오류 분석, CPU 벤치마크와 파일 검증 역량을 보여줍니다. **INT8은 두 연구의 변환 모두에서 큰 검증 정확도 감소와 속도 기준 미달을 확인해 배포에서 제외**했습니다. 최초 9개 조합 노트북, 두 통제 실험, 실패한 변환과 동결된 근거도 보존합니다.

![개발 시드 3개에서 비교한 전체 9개 설정, 검증 선택은 T5-a01 유지](./assets/figures/ko/full-grid-comparison.png)

전체 조합의 세 시드는 모두 개발 선택에 참여했습니다. 참고 평가 결과는 이미 공개됐고 공식 테스트는 과거에 모델 선택에 사용됐습니다. 이 결과는 제한된 교통 표지판 분류 비교를 뒷받침하며 실제 도로 환경의 신뢰성을 입증하지 않습니다.

## 대표 저장 모델 실행

Windows PowerShell:

```powershell
.\launch-kd.bat doctor
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
.\launch-kd.bat fetch-data
.\launch-kd.bat evaluate --model artifacts/runtime/models/student-fp32.tflite --manifest reports/split-manifest.json --data .local/data/gtsrb-v1 --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-representative-student
```

Linux:

```bash
bash launch-kd.sh doctor
bash launch-kd.sh download --manifest reports/artifact-download.json --output artifacts/runtime
bash launch-kd.sh download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
bash launch-kd.sh fetch-data
bash launch-kd.sh evaluate --model artifacts/runtime/models/student-fp32.tflite --manifest reports/split-manifest.json --data .local/data/gtsrb-v1 --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-representative-student
```

[대표 모델 번들](https://park-dongmyung.com/downloads/knowledge-distillation-models-v2.zip)은 **10.83MiB, 11,353,209바이트**이며 `artifacts/runtime/models/`에 설치됩니다. 기존에 동결한 명세와 모델 체크섬을 유지합니다. 저장 모델의 CPU 평가에는 Docker, GPU와 API 키가 필요하지 않습니다. 첫 설치와 다운로드에는 인터넷이 필요하며 쓰기 가능한 Windows 또는 Linux x86-64 폴더를 사용합니다.

별도의 [9개 조합 모델 번들](https://park-dongmyung.com/downloads/knowledge-distillation-full-grid-models.zip)과 [초기 실험 근거 번들](https://park-dongmyung.com/downloads/knowledge-distillation-full-grid-base-evidence.zip)은 전체 연구 체크포인트와 T5-a01 변환 파일을 보존합니다. 해당 실험을 설치하거나 재현할 때는 [전체 조합 안내](./docs/full-grid.ko.md)의 명시적 다운로드 명세와 명령을 사용합니다. 다운로드 ZIP에서 대표 모델과 연구 선택 모델을 구분합니다.

<details>
<summary>앞선 4개 후보 연구와 최초 노트북 결과의 보존 기록</summary>

아래 수치, 실행 명령과 환경 검증은 앞선 연구의 기록입니다. 위 확장 실험의 새 모델에 대한 재검증 완료 주장으로 사용하지 않습니다. 최초 노트북의 9개 조합 결과도 동일한 과거 기록으로 유지합니다.

### 결과와 근거

모델과 평가 절차를 고정한 뒤, 이번 실험의 모델 선택에서 제외한 **최종 평가(holdout) 집합 5,850장**에서 시드 378 결과를 측정했습니다. 전체 대응 결과와 검증 근거는 [측정 결과 보고서](./docs/results.ko.md)에 정리했습니다.

| 측정 항목 | 교사 모델 | 동일 조건의 기준 모델 | 고정된 증류 후보 |
| --- | ---: | ---: | ---: |
| 최종 평가 정확도, 정답 수 / 평가 수 | 83.54%, 4,887 / 5,850 | 73.56%, 4,303 / 5,850 | 87.06%, 5,093 / 5,850 |
| 최종 평가 macro F1 | 0.7465 | 0.6253 | 0.7903 |
| 3개 시드 정확도 평균 ± 표본 표준편차 | N/A | 78.08% ± 3.94%p | 87.51% ± 2.18%p |
| 3개 시드의 대응 차이 평균 ± 표본 표준편차 | N/A | 기준 | +9.43 ± 3.93%p |
| CPU 단일 이미지 p50 / p95 | 3.823 / 4.641 ms | 1.008 / 1.128 ms | 1.040 / 1.130 ms |
| 표준 Keras 추론 파일 크기 | 9.39 MiB (9,843,117 bytes) | 182.8 KiB (187,164 bytes) | 182.8 KiB (187,164 bytes) |
| 파라미터 수 | 2,313,067 | 34,406 | 34,406 |

평균과 표본 표준편차는 시드 378, 379, 380의 기술 통계이며 신뢰구간이 아닙니다. CPU 수치는 동일한 검증 이미지로 측정한 Keras 호출 비용이며 TFLite 배포 비용과 구분합니다.

CPU 측정 환경: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (Microsoft Windows 11 Pro), 스레드 1개, 동일 이미지 128장, 워밍업 10회, 반복 3회. 다른 하드웨어의 지연 시간을 보장하지 않습니다.

**검증 단계의 선택:** `T5-a05`, 대응 정확도 차이 평균 +9.16%p, 증가한 짝 3/3, 사전 기준 통과. 최종 평가의 대응 차이는 위 표에 별도로 기록하며 모델을 다시 선택하는 데 사용하지 않습니다.

**공개 모델 선택:** 시드 378 증류 모델, **FP32 TFLite**. INT8은 검증 정확도 감소 +61.863%p, CPU p50 속도 비율 0.771배로 채택하지 않았습니다.
통과하지 못한 기준: 검증 정확도 감소, CPU p50 속도.
선택한 배포 파일의 최종 평가 정확도는 **87.06% (5,093/5,850)**, macro F1은 **0.7903**, CPU p50 / p95는 **0.429 / 0.456 ms**, 크기는 <strong>140.0 KiB (143,316 bytes)</strong>입니다.
최종 평가 결과로 시드, 체크포인트 또는 양자화 채택 여부를 다시 정하지 않습니다.

![최종 평가와 CPU 배포 비용의 실측 비교](./assets/figures/ko/four-candidate-release.png)

#### 최초 개발 결과

| 최초 실험의 측정 항목 | 기록된 값 |
| --- | ---: |
| 교사 모델의 선택된 최고 에포크 정확도 | 88.37% |
| 경량 기준 모델의 선택된 최고 에포크 정확도 | 54.61% |
| 최고 증류 모델의 선택된 에포크 정확도 | 80.97% |
| 학생 / 교사 모델 파라미터 수 | 34,406 / 2,313,067 |
| 최초 실험의 최고 설정 | `temperature=10`, `alpha=0.3` |

이 수치는 하나의 시드로 9개 조건을 비교한 최초 노트북의 결과입니다. 공식 GTSRB 테스트 데이터를 체크포인트와 학습 설정 선택에 사용했으므로, 모델 선택에 관여하지 않은 평가 데이터의 성능으로 해석할 수 없습니다. 기준 모델의 마지막 메모리 가중치는 54.12%였지만 표에는 학습 이력의 최고 에포크인 54.61%를 표시했습니다. 최초 `model.predict()` 지연 시간에는 프레임워크 호출 비용이 포함됐으며 교사 모델 대비 배포 속도는 측정하지 않았습니다. 약 67.2배의 파라미터 감소를 실제 속도 향상으로 해석하지 않습니다.

개선 버전은 데이터 분할과 모델 선택 절차가 다릅니다. 최초 결과와 새로운 결과의 백분율을 동일한 최종 평가 집합에서 측정한 통제된 전후 비교처럼 제시하지 않습니다.

### 개선 구현이 보여주는 역량

1. **실제 표지판 단위의 평가 분리:** 클래스별 구성을 고려한 학습, 검증, 최종 평가 집합에서 같은 표지판 트랙의 이미지가 서로 다른 집합에 섞이지 않게 합니다. 상대 경로, 레이블, 트랙 ID, 파일 크기와 SHA-256으로 이미지 구성을 고정합니다. 공식 테스트는 모델 동결 이후 별도로 확인합니다.
2. **기준 모델과 증류 모델의 통제된 비교:** 두 학생 모델의 34,406개 파라미터 구조, 시드별 초기 가중치, 데이터 순서, 증강, 클래스 가중치와 최대 학습 예산을 맞춥니다. 시드 378, 379, 380의 대응 실험으로 변동을 확인하며, 배포 시드는 최종 평가 전에 378로 정합니다.
3. **교사 모델의 올바른 적응:** `[0,1]` RGB 입력을 증강한 뒤 교사 모델 내부에서 MobileNetV2의 `[-1,1]` 입력 범위로 변환합니다. 분류기 학습 후 작은 학습률로 미세 조정하고 배치 정규화를 고정합니다. 증류 과정에서 교사 가중치와 저장 파일이 변하지 않았는지도 확인합니다.
4. **예산과 검증 기준이 명확한 증류:** 사전에 정한 네 가지 후보에서 클래스 가중치를 적용한 정답 CE와 가중치를 적용하지 않은 교사에서 학생 방향의 KL을 결합하고 온도의 제곱으로 보정합니다. 실제 검증 CE를 기록하고 최적 `.keras` 모델을 저장한 뒤 다시 읽어 평가합니다. 검증 정확도 0.2%p 개선 기준과 3개 시드 확인으로 효과 없는 후보를 중단합니다.
5. **실제 배포 비용의 측정:** 학습용 사용자 정의 래퍼 없이 표준 `.keras` 모델로 추론합니다. FP32 TFLite 내보내기와 완전 INT8 양자화를 구분하며, INT8 보정에는 학습 이미지만 사용합니다. 동일 이미지, 스레드 정책과 워밍업 조건에서 CPU 지연 시간, 처리량과 파일 크기를 비교하고, 검증 집합으로 양자화 채택 여부를 정합니다.
6. **재현 가능한 실행과 근거 관리:** 격리된 CPU 실행 환경, 고정 의존성, 소스 스냅샷, 정확한 파일 목록과 SHA-256으로 재학습 없이 저장 모델을 검증할 수 있게 합니다. 테스트는 손실 기울기, 교사 불변성, 실제 최적화 갱신, 분할 격리, 체크포인트 재로드와 변조 거부를 확인합니다.

위 항목은 구현된 기능입니다. 실제 학습, 모델 변환과 환경 검증의 완료 여부는 공개 결과에 별도로 표시하며, 코드가 존재한다는 사실만으로 실행 성공을 주장하지 않습니다.

### 재학습 없이 실행하기

포트폴리오 프로젝트 페이지의 소스 ZIP을 사용하는 경우 쓰기 가능한 폴더에 추출한 뒤 동일한 명령을 실행합니다.

로컬 기본 실행 환경은 CPU입니다. 공개 저장 모델의 추론에는 Docker, GPU와 API 키가 필요하지 않습니다. 최초 설치와 다운로드에는 인터넷 연결이 필요합니다. 지원 범위의 Windows 또는 Linux x86-64에서 쓰기 가능한 소스 폴더를 사용하며, Python 3.12가 없으면 실행기가 고정된 Python 환경을 준비할 수 있습니다.

Windows PowerShell:

```powershell
.\launch-kd.bat doctor
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
.\launch-kd.bat fetch-data
.\launch-kd.bat fetch-data --verify-only
.\launch-kd.bat predict --model artifacts/runtime/models/student-fp32.tflite --image .local/data/gtsrb-v1/Train/20/00020_00000_00000.png
```

Linux:

```bash
bash launch-kd.sh doctor
bash launch-kd.sh download --manifest reports/artifact-download.json --output artifacts/runtime
bash launch-kd.sh download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
bash launch-kd.sh fetch-data
bash launch-kd.sh fetch-data --verify-only
bash launch-kd.sh predict --model artifacts/runtime/models/student-fp32.tflite --image .local/data/gtsrb-v1/Train/20/00020_00000_00000.png
```

예제 이미지는 추론 경로를 확인하기 위한 학습 데이터의 입력이며 정확도 평가 사례가 아닙니다. 모델 다운로드 명세는 체크포인트를 `artifacts/runtime/models/`에 설치하며 공개 파일 목록에서 실제 파일명을 확인할 수 있습니다. 공개 모델 번들을 [검증한 주소](https://park-dongmyung.com/downloads/knowledge-distillation-models-v2.zip)에서 로그인 없이 전체 다운로드하고 SHA-256 및 파일 크기를 대조했습니다 (10.83 MiB (11,353,209 bytes)). `reports/artifact-download.json`의 정확한 파일 목록으로 모델을 설치합니다.

#### 동결된 평가 반복하기

공개된 `reports/split-manifest.json`과 최종 평가 전 동결 기록을 그대로 사용합니다. 공개 모델의 평가를 반복할 때는 원래 동결 기록을 유지합니다.

```powershell
.\launch-kd.bat evaluate --model artifacts/runtime/models/student-fp32.tflite --manifest reports/split-manifest.json --data .local/data/gtsrb-v1 --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-student
```

Linux에서는 `.\launch-kd.bat`를 `bash launch-kd.sh`로 바꿉니다. 실행마다 새로운 출력 디렉터리를 사용합니다. 공개 데이터 목록과 동결 파일에는 정확한 모델, 평가 이미지 수, 설정과 소스가 기록됩니다. 평가 결과는 집계 지표, 전체 클래스 혼동행렬과 이미지 ID별 예측으로 저장됩니다.

#### 구현 검증 테스트

`doctor`로 `.venv`를 준비한 뒤 실행합니다.

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m unittest discover -s tests -v
```

**실행 환경 및 저장 모델 재현:** Windows, Python 3.12.15: 테스트 63개 중 62개 통과, 1개 제외, 실패 0개. 저장 모델의 최종 평가 5,850장 전체를 다시 실행해 정확도와 macro F1이 일치했습니다 (`baseline` 4,303/5,850, `distilled` 5,093/5,850, `fp32` 5,093/5,850). Linux (WSL2), Python 3.12.15: 테스트 63개 중 63개 통과, 0개 제외, 실패 0개. 저장 모델의 최종 평가 5,850장 전체를 다시 실행해 정확도와 macro F1이 일치했습니다 (`baseline` 4,303/5,850, `distilled` 5,093/5,850, `fp32` 5,093/5,850). 두 운영체제 검증은 동일 컴퓨터에서 수행했으며 외부 독립 검토를 뜻하지 않습니다. 합성 데이터 테스트와 실제 저장 모델의 전체 평가 수행 범위는 별도로 보고합니다.

### 학습과 재현의 범위

전체 프로젝트 학습은 GPU를 사용하는 Kaggle 환경에서만 수행합니다. [사전 계획](./configs/experiment-plan.json)은 교사 모델과 증류 후보 수를 제한하고, 최초 후보 선택과 시드별 확인을 구분하며, 최종 추론 전에 모델, 소스와 설정의 해시를 고정합니다. 명령과 저장 결과는 [학습 안내](./docs/training.ko.md)에 정리했습니다. 최초 노트북은 현재 학습 진입점이 아닌 기존 개발 근거입니다.

최종 평가 집합은 새 실험의 학습과 모델 선택에서 제외하지만, 그 이미지들은 최초 노트북이 사용한 전체 학습 데이터에 포함돼 있었습니다. 새 모델은 ImageNet 교사 가중치와 새로운 학생 초기화에서 출발합니다. 이것만으로 모든 과거 모델과 외부 사전학습 모델이 해당 공개 데이터를 본 적 없다고 입증할 수는 없습니다. 공식 테스트 역시 이전에 모델 선택에 사용했으므로 별도 확인 결과로 표시합니다.

소스, 모델과 데이터 해시가 같으면 저장 모델의 평가를 반복할 수 있습니다. 하드웨어, 드라이버와 부동소수점 구현에 따라 실행 시간과 일부 계산값은 달라질 수 있습니다. 동일 컴퓨터의 다른 운영체제 환경에서 수행한 검증은 환경 검증이며 외부 독립 검토를 의미하지 않습니다. GTSRB 분류는 제한된 연구 과제이며 실제 도로 장면 인식 시스템의 완성을 입증하지 않습니다.

### 저장소 안내

| 위치 | 용도 |
| --- | --- |
| `knowledge_distillation/` | 트랙 분할, 모델, 학습, 평가, 변환과 CPU 추론 |
| `configs/experiment-plan.json` | 사전 탐색 예산, 대응 시드와 중단 정책 |
| `scripts/`와 `launch-kd.*` | 격리 환경 설치와 검증 가능한 다운로드 |
| `tests/` | 합성 데이터 기반 구현, 무결성과 환경 검증 |
| `reports/` | 공개 결과, 데이터 목록, 동결과 검증 근거 |
| `docs/` | 영어 기본 설명과 대응하는 한국어 문서 |
| `notebooks/knowledge_distillation_gtsrb.ipynb` | 최초 실험 기록 |

방법론 출처: [GTSRB 데이터셋](https://benchmark.ini.rub.de/gtsrb_dataset.html), [지식 증류 논문](https://arxiv.org/abs/1503.02531), [MobileNetV2 전처리](https://keras.io/api/applications/mobilenet/mobilenet_models/), [Keras 전이학습](https://keras.io/guides/transfer_learning/).

</details>
