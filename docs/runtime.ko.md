> **앞선 4개 후보 연구의 보존 문서입니다.** 아래 결과, 기본 다운로드 명세와 학습 명령은 이전 연구에 해당합니다. 최신 9개 조합 연구에는 [확장 결과와 명시적 재현 명령](./full-grid.ko.md) 및 [교사와 학생 분석](./teacher-student-analysis.ko.md)을 사용하십시오. 기존 실행기의 기본 명세를 최신 전체 조합 모델로 간주하지 않습니다. 원본 수치와 설명은 아래에 유지합니다.

<div align="right"><a href="./runtime.md">English</a> | <strong>한국어</strong></div>

# CPU 실행 및 검증

저장된 Keras 및 TFLite 모델은 재학습 없이 평가하거나 이미지 추론에 사용할 수 있습니다. 전체 학습은 GPU를 활성화한 Kaggle 커널에서 실행합니다. 로컬 실행기는 항상 CPU를 선택하며 CUDA 환경을 설치하지 않습니다.

## 최초 설치

쓰기 가능한 폴더에 소스를 압축 해제합니다. Windows x86-64:

```powershell
.\launch-kd.bat doctor
```

Linux x86-64:

```bash
bash launch-kd.sh doctor
```

실행기는 사용 가능한 64비트 Python 3.12를 선택하거나, 해시를 검증한 uv 0.13.0으로 Python 3.12.15를 준비합니다. 시스템 패키지를 사용하지 않는 `.venv`에 고정 버전 의존성을 설치합니다. 기존 환경이 호환되지 않으면 `.local/environment-backups`에 보관합니다. 전역 Python 패키지를 변경하지 않습니다. `KD_PYTHON`으로 Python 3.12 실행 파일을 지정하거나 `KD_USE_MANAGED_PYTHON=1`로 자동 Python 설치 경로를 사용할 수 있습니다.

최초 설치에는 Python, 패키지, 공개 모델과 데이터셋을 받을 인터넷 연결이 필요합니다. 소스 ZIP에 모델 가중치는 포함하지 않습니다. 저장 모델의 CPU 추론에는 Docker, GPU, API 키가 필요하지 않습니다. Windows에는 TensorFlow가 요구하는 Microsoft Visual C++ 런타임이 필요합니다. 자동 Python 준비는 Windows 및 Linux x86-64용이며, 다른 플랫폼은 기록된 검증 범위에 포함하지 않습니다.

`doctor`는 패키지 버전, 가상 환경 분리 상태, 실제 TensorFlow CPU 텐서 연산을 확인합니다. `--model` 또는 `--artifact-manifest`를 지정하지 않았다면 모델이나 데이터 설치까지 확인한 결과로 해석하면 안 됩니다.

## 모델 다운로드 및 검증

```powershell
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
```

Linux에서는 `.\launch-kd.bat` 대신 `bash launch-kd.sh`를 사용합니다. 다운로드 명세는 HTTPS 주소, 파일 크기, SHA-256 및 정확한 압축 해제 파일 목록을 고정합니다. 다운로드는 임시 폴더에서 검증한 뒤 설치합니다. 크기나 해시 불일치, 안전하지 않은 ZIP 경로, 추가 파일 또는 누락 파일이 있으면 설치를 중단합니다. 검증되지 않은 기존 파일을 자동으로 덮어쓰지 않습니다. URL이 null이면 공개 다운로드 확인이 완료되지 않은 상태이며, 임의의 대체 주소를 사용하지 않고 명확한 오류를 반환합니다.

공개 다운로드 명세는 모델을 `artifacts/runtime/models/`에 설치합니다. `student.keras`는 선택한 시드 378 학생 모델의 표준 Keras 파일입니다. FP32와 INT8 변환 파일은 `student-fp32.tflite`, `student-int8.tflite`이며, 결과 보고서에서 검증 집합의 채택 기준을 통과한 형식을 확인할 수 있습니다. 명세에 실제 검증한 공개 주소를 넣기 전까지 모델 전달은 대기 상태입니다.

## 데이터 및 저장 모델 추론

```powershell
.\launch-kd.bat fetch-data
.\launch-kd.bat predict --model artifacts/runtime/models/student.keras --image path/to/sign.png
```

`fetch-data`는 공개 데이터 ZIP의 고정 해시를 검증한 뒤 `.local/data/gtsrb-v1`에 설치합니다. 해당 경로를 `--data`에 입력합니다. 원본 ZIP에는 `Train`과 `train`처럼 대소문자만 다른 중복 폴더가 있어, 루트 `Train.csv`, `Test.csv`, `Meta.csv`에 명시된 정확한 경로만 추출합니다. Windows와 Linux에서 같은 이름을 보존하며 압축 해제한 목록과 파일 해시도 기록합니다. `fetch-data --archive path/to/gtsrb.zip`은 같은 크기 및 SHA-256 검증 후 기존 ZIP을 재사용합니다. `fetch-data --verify-only`는 네트워크 없이 기존 설치를 검사합니다. 측정을 반복할 때는 공개 모델 목록과 결과 보고서에 명시된 파일 경로를 유지합니다.

이미지 추론은 로컬 이미지와 `.keras` 또는 `.tflite` 모델을 받습니다. 평가와 동일한 RGB 변환, 96 x 96 크기 조정, `[0,1]` 정규화를 적용합니다. 교사 모델에는 MobileNetV2용 `[-1,1]` 변환이 포함되어 있습니다. 기본 출력은 GTSRB 숫자 클래스 ID이며, `--class-map`으로 호환되는 클래스명 JSON을 지정할 수 있습니다. `--sha256`을 지정하면 로딩 전에 모델 해시를 확인합니다. 이 분류 결과는 도로 안전용 실서비스 검증을 의미하지 않습니다.

## 평가 재현

```powershell
.\launch-kd.bat evaluate --model artifacts/runtime/models/student.keras --manifest reports/split-manifest.json --data DATASET_ROOT --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-student
```

공개 결과에는 사용한 분할 명세, 동결 파일 및 체크포인트 경로가 명시됩니다. 공개 학습 분할 명세는 `reports/split-manifest.json`이며 Kaggle 실행 내부의 `splits.json`을 별도 공개 파일로 복제하지 않습니다. `locked_holdout`과 `official_test`에는 평가 전에 생성한 적절한 동결 파일이 필요합니다. 공개 모델에는 기존 공개 동결 기록을 그대로 사용합니다. 공식 테스트 평가는 과거에 모델 선택에 사용한 자료의 감사 평가입니다. 평가 결과로 집계 지표, 혼동행렬, ID 기반 예측 기록을 저장합니다. 실행마다 새 출력 폴더를 사용합니다.

```powershell
.\launch-kd.bat benchmark --model student=artifacts/runtime/models/student.keras --model int8=artifacts/runtime/models/student-int8.tflite --manifest reports/split-manifest.json --data DATASET_ROOT --output .local/cpu-benchmark --threads 1 --samples 128 --warmup 10 --repetitions 3
```

벤치마크는 하드웨어와 스레드 설정을 기록합니다. 컴퓨터에 따라 실행 시간과 일부 부동소수점 계산값이 달라질 수 있습니다. 소스, 모델, 데이터 해시가 같다고 모든 TensorFlow 환경에서 비트 단위로 동일한 출력을 보장하지 않습니다. 공개 검증 보고서는 실제 수행 범위를 명시하며, 명령이 준비되어 있다는 사실만으로 실행을 완료했다고 주장하지 않습니다.

## 오프라인 실행 및 문제 해결

설치와 모델 다운로드 후에는 가중치를 다시 받지 않고 로컬 이미지 추론을 실행할 수 있습니다. 데이터 평가에는 로컬 데이터셋도 필요합니다. `--verify-only`는 네트워크 없이 설치된 파일을 검증합니다. 네트워크 문제 때문에 정상 `.venv`를 삭제할 필요는 없습니다. 환경 복구가 필요하면 실행기가 호환되지 않는 기존 환경을 보관합니다.

설치 실패 시 오류를 보관하고 연결을 복구한 뒤 `doctor`를 다시 실행합니다. 해시가 다르면 검증을 끄지 말고 공개된 정확한 파일을 받습니다. 최종 평가에서 동결 검증이 실패하면 해당 모델, 분할 명세, 소스 및 설정이 동결 내용과 일치하는지 확인합니다. 전체 학습은 Kaggle GPU에서 실행하므로 로컬 `launch-kd`로 학습하는 명령은 거부됩니다.

참고: [TensorFlow 설치](https://www.tensorflow.org/install/pip), [Python 가상 환경](https://docs.python.org/3.12/library/venv.html), [Kaggle API](https://github.com/Kaggle/kaggle-api).
