# 아키텍처 및 핵심 알고리즘 설계

[한국어 README](../README.ko.md) · [English](./architecture.md) · [실험 결과](./results.ko.md) · [재현 가이드](./reproduction.ko.md)

## 1. 문제 정의 및 설계 목표

자율주행 도로 장면 시맨틱 세그멘테이션(19개 Cityscapes 클래스)에서 고정밀 Transformer 교사(`SegFormer-B5`, `84.6M`)는 단일 프레임 CPU 추론에 `1,631 ms`(`0.61 FPS`)가 소요되어 엣지 실시간 주행에 투입할 수 없습니다. 본 시스템은 다음 4가지 엔지니어링 목표를 동시에 달성하도록 설계되었습니다.

1. **이종 아키텍처 지식 증류**: 전역 어텐션 기반 Vision Transformer(`MiT-B5`)의 공간·의미 지식을 순수 경량 CNN(`MobileNetV4 + FPN`)으로 전이합니다.
2. **안전 핵심(Safety-6) 소수 클래스 보존**: 보행자(`person`), 탑승자(`rider`), 자전거(`bicycle`), 오토바이(`motorcycle`), 신호등(`traffic_light`), 표지판(`traffic_sign`)처럼 픽셀 점유율이 극도로 작은 클래스가 증류 과정에서 희석되지 않도록 합니다.
3. **악천후·센서 열화 강건성**: 깨끗한 입력만을 보는 교사의 내부 표상과 로짓을 열화된 입력을 받는 학생에게 전이하여 야간·안개·블러·센서 노이즈 환경에서 붕괴를 방지합니다.
4. **무누설 양자화 배포 및 확률 보정**: `validation` 분할만으로 모델 선택·정적 QDQ INT8 게이트·사후 온도 보정을 끝내고 `locked_eval`과 KITTI Zero-Shot은 SHA-256 동결 후 단 1회 평가합니다.

---

## 2. 교사·학생 모델 구조 (`seg_kd/models.py`)

### 교사 (`SegformerTeacher`)
- **백본**: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` (`MiT-B5`, `84,607,955` 파라미터, 체크포인트 `323.14 MiB`)
- **출력**: 19클래스 로짓 `(B, 19, H, W)` 및 4단계 계층 특징 맵(`s4`, `s8`, `s16`, `s32`: 채널 `[64, 128, 320, 512]`)
- **역할**: 학생 학습 시 `eval()` + `torch.no_grad()` + CUDA BF16으로 동결(frozen)되어 깨끗한 입력 프레임의 정밀 공간 분포와 다중 스케일 특징을 제공합니다.

### 학생 (`EdgeSegStudent` + `LiteFPNDecoder`)
- **공통 디코더**: 양자화 융합에 친화적인 `Conv2d(bias=False) → BatchNorm2d → ReLU6` 경량 FPN 디코더(모든 업샘플은 ONNX Opset 17 호환 `bilinear`)
- **3단계 Pareto 백본 티어**:
  - `MobileNetV4-Small` (`1.50M` params, `fpn_dim=96`, 57 Conv / 55 BN): 초경량 UIB 블록 (`FP32 5.72 MiB`, `INT8 1.76 MiB`)
  - `MobileNetV3-Large` (`3.30M` params, `fpn_dim=128`, 74 Conv / 56 BN): 18개 Squeeze-and-Excitation(SE) + HardSwish 포함(정적 INT8 민감성 대조군)
  - `MobileNetV4-Medium` (`7.54M` params, `fpn_dim=128`, 88 Conv / 86 BN): 순수 `Conv-BN-ReLU` UIB 구조 (`FP32 28.71 MiB`, `INT8 7.90 MiB`)
- **양자화 친화 자동 감사 (`audit_quantization_friendliness`)**: 모듈 시퀀스에서 `Conv → ReLU → BN` 비융합 안티패턴을 사전 검출하며 모든 학생은 감사 통과 후 학습됩니다.
- **학습 전용 프로젝터 (`FeatureProjectorBank`)**: 학생과 교사의 4단계 채널 차원을 맞추는 `1×1 Conv-BN` 프로젝터 은행으로, 배포(ONNX export) 시에는 완전히 제거되어 **추론 오버헤드가 0**입니다.

---

## 3. 손실 함수 설계 (`seg_kd/losses.py`)

### 3.1 Channel-Wise Spatial Distillation (`CWD`)
도로 세그멘테이션에서는 도로·건물·하늘이 전체 픽셀의 대다수를 차지하고 신호등·표지판·보행자는 극소수입니다. 픽셀별 Softmax KL(`pixel_kl`)은 배경 다수 클래스에 그래디언트가 지배되어 소수 클래스를 무너뜨립니다. `CWD`는 각 채널 $c$ 내부의 공간 좌표 $(h, w) \in H \times W$에 대해 정규화된 공간 확률 분포를 정렬합니다.

$$p_c^{T}(i) = \frac{\exp(z_{c,i}^{T} / \tau)}{\sum_{j \in \Omega} \exp(z_{c,j}^{T} / \tau)}, \qquad \mathcal{L}_{\mathrm{CWD}} = \frac{\tau^2}{C} \sum_{c=1}^{C} \sum_{i \in \Omega} p_c^{T}(i) \log \frac{p_c^{T}(i)}{p_c^{S}(i)}$$

유효 마스크 $\Omega$에서 `ignore_index=255` 픽셀은 마스킹 제외하여 손실 왜곡을 차단합니다(`T = 4.0`).

### 3.2 4단계 하이브리드 특징 정렬 (`cwd_feat`)
이종 아키텍처(Transformer ↔ CNN)의 4단계(`s4, s8, s16, s32`) 특징 맵을 단일 정규화에만 의존하지 않고 공간 집중도(`CWD`)와 채널 방향성(`Cosine`)을 절반씩 결합합니다.

$$\mathcal{L}_{\mathrm{feat}} = \frac{1}{|S|} \sum_{s \in \lbrace s4, s8, s16, s32 \rbrace} \Big( 0.5 \cdot \mathcal{L}_{\mathrm{CWD}}(\hat{F}_s^{S}, F_s^{T}) + 0.5 \cdot (1 - \cos(\hat{F}_s^{S}, F_s^{T})) \Big)$$

전체 학생 목적 함수는 무가중 CE, 다중 클래스 Dice, 공간 CWD, 4단계 하이브리드 특징 정렬의 합(`configs/l4-plan.json` 기준 `λ_dice = 0.25`, `α_kd = 0.8`, `β_feat = 0.4`, `T = 4.0`)입니다.

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{dice}} \mathcal{L}_{\mathrm{Dice}} + \alpha_{\mathrm{kd}} \mathcal{L}_{\mathrm{CWD}} + \beta_{\mathrm{feat}} \mathcal{L}_{\mathrm{feat}}$$

---

## 4. 비대칭 강건성 증류 (`v4-final`, `seg_kd/training.py`)

표준 Clean 증류(`v3-max`) 학생은 깨끗한 주간 장면에서는 교사의 `90.8%`까지 근접하지만, 고주파 가우시안 센서 노이즈(`σ=18`)가 주입되면 CNN 초기 필터가 교란되어 `13.75%`로 붕괴합니다. 이를 해결하기 위해 **비대칭 입력 경로**(Asymmetric Clean-Teacher / Corrupted-Student)를 도입했습니다.

1. **초기 5에포크 램프 스케줄**: 에포크 1~5 동안 열화 주입 확률 $p$를 `0.07 → 0.35`로 선형 증가시켜 초기 수렴 안정성을 확보합니다.
2. **GPU 배치 결정론적 열화 주입 (`apply_gpu_adverse_corruption`)**: 학생에게 전달되는 텐서에만 4가지 열화(야간 저조도 감마, 모션 블러, 안개 저대비 혼합, 가우시안 센서 노이즈)를 무작위 조합으로 적용합니다.
3. **깨끗한 교사 감독**: 동결된 `SegFormer-B5`는 항상 **열화 전 깨끗한 이미지**에서 특징과 로짓을 추출합니다. 결과적으로 학생은 열화 입력으로부터 깨끗한 도로 구조와 안전 객체 경계를 복원하도록 학습되며, 노이즈 환경에서는 오히려 교사(열화 입력 시 `59.19%`)를 추월(`59.59%` INT8 / `60.61%` FP32)합니다.

---

## 5. QAT-KD · 정적 QDQ ONNX INT8 게이트 · 사후 온도 보정

1. **QAT-KD (`v3-max` 8ep / `v4-final` BF16 교사 6ep)**: 학습 완료 학생에 양자화 인식 미세조정을 6에포크 수행해 정적 INT8 전환 시의 정확도 낙폭을 `1.0%p` 이내로 억제합니다.
2. **ONNX FP32/INT8 내보내기 (`seg_kd/export.py`)**: Opset 17로 FP32 ONNX를 추출해 PyTorch 대비 `100.0%` 픽셀 argmax 일치(`max_abs_logit_diff <= 3.6e-5`)를 검증한 뒤, 훈련 분할 128장 결정론적 캘리브레이션 세트로 per-channel 정적 QDQ INT8(`student-int8.onnx`)을 생성합니다.
3. **Validation-Only 배포 게이트 (`assess_int8_acceptance`)**: `validation` 분할에서만 다음 4개 조건을 모두 만족해야 `INT8`을 승인합니다.
   - `val_mIoU_drop <= 1.0%p`
   - `val_safety_mIoU_drop <= 1.5%p`
   - `INT8 file_size < FP32 file_size`
   - `CPU_p50_speedup >= 1.20×`
4. **Validation-Only 사후 온도 보정 (`seg_kd/evaluation.py`)**: CWD 증류 학생의 로짓 샤프닝(ECE 상승)을 해소하기 위해 `validation` NLL 최소화 스칼라 온도 $`T^*`$를 1차원 그리드 탐색으로 추정합니다. 로짓을 $`z / T^*`$로 나누면 `argmax`가 수학적으로 불변(`mIoU 변화 0.00%p`)이면서 `locked_eval` ECE가 `69.5~78.5%` 감소합니다.
