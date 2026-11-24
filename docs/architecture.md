# Architecture & Core Algorithmic Design

[English README](../README.md) · [한국어](./architecture.ko.md) · [Results](./results.md) · [Reproduction](./reproduction.md)

## 1. Problem Formulation & Design Goals

In autonomous driving semantic segmentation (19 Cityscapes classes), a high-capacity Vision Transformer teacher (`SegFormer-B5`, `84.6M` parameters) takes `1,631 ms` per frame (`0.61 FPS`) on CPU—far too slow for real-time edge deployment. `seg_kd` is engineered to satisfy four simultaneous constraints:

1. **Cross-Architecture Knowledge Transfer**: Transfer global self-attention spatial context from `MiT-B5` to lightweight quantization-ready Edge CNNs (`MobileNetV4 + FPN`).
2. **Safety-Critical Class Preservation**: Prevent minority vulnerable road-user classes (`traffic_light`, `traffic_sign`, `person`, `rider`, `motorcycle`, `bicycle`) from being drowned out by majority background classes (`road`, `building`, `sky`).
3. **Adverse-Weather & Sensor-Noise Resilience**: Transfer clean-scene teacher representations to corrupted-input students so deployment models stay reliable under night, fog, motion blur, and camera sensor noise.
4. **Leakage-Free Deployment & Calibration**: Fit model selection, static QDQ INT8 acceptance gating, and post-hoc temperature scaling exclusively on `validation`, freezing all artifacts via SHA-256 before touching `locked_eval` or KITTI Zero-Shot.

---

## 2. Teacher & Student Architectures (`seg_kd/models.py`)

### Teacher (`SegformerTeacher`)
- **Backbone**: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` (`MiT-B5`, `84,607,955` params, `323.14 MiB`)
- **Outputs**: 19-class logits `(B, 19, H, W)` and 4-stage hierarchical feature pyramid (`s4, s8, s16, s32` with channels `[64, 128, 320, 512]`).
- **Role**: Frozen in `eval()` + CUDA BF16 during student distillation, always consuming clean inputs.

### Student (`EdgeSegStudent` + `LiteFPNDecoder`)
- **Decoder**: Quantization-friendly `Conv2d(bias=False) → BatchNorm2d → ReLU6` lightweight FPN with ONNX Opset 17 compatible bilinear upsampling.
- **3-Tier Pareto Backbones**:
  - `MobileNetV4-Small` (`1.50M` params, `fpn_dim=96`, 57 Conv / 55 BN): ultra-light UIB blocks (`FP32 5.72 MiB`, `INT8 1.76 MiB`).
  - `MobileNetV3-Large` (`3.30M` params, `fpn_dim=128`, 74 Conv / 56 BN): includes 18 Squeeze-and-Excitation (SE) + HardSwish paths (quantization-sensitivity control).
  - `MobileNetV4-Medium` (`7.54M` params, `fpn_dim=128`, 88 Conv / 86 BN): pure `Conv-BN-ReLU` UIB blocks (`FP32 28.71 MiB`, `INT8 7.90 MiB`).
- **Quantization Friendliness Audit (`audit_quantization_friendliness`)**: Programmatically rejects `Conv → ReLU → BN` non-fusible antipatterns prior to training.
- **Training-Only Projector Bank (`FeatureProjectorBank`)**: `1×1 Conv-BN` projections align student and teacher pyramid channels during training and are completely discarded at export (**zero inference overhead**).

---

## 3. Distillation Objectives (`seg_kd/losses.py`)

### 3.1 Channel-Wise Spatial Distillation (`CWD`)
Pixel-wise KL (`softmax` across `C` channels) is dominated by frequent background classes. `CWD` instead normalizes each channel $c$ spatially over valid pixels $`\Omega = \lbrace (h, w) : y_{h,w} \neq 255 \rbrace`$:

$$p_c^{T}(i) = \frac{\exp(z_{c,i}^{T} / \tau)}{\sum_{j \in \Omega} \exp(z_{c,j}^{T} / \tau)}, \qquad \mathcal{L}_{\mathrm{CWD}} = \frac{\tau^2}{C} \sum_{c=1}^{C} \sum_{i \in \Omega} p_c^{T}(i) \log \frac{p_c^{T}(i)}{p_c^{S}(i)}$$

Every class channel receives equal spatial gradient weight regardless of pixel area (`T = 4.0`).

### 3.2 4-Stage Hybrid Feature Alignment (`cwd_feat`)
Across stages `s4, s8, s16, s32`, projected student features match frozen teacher features via a 50/50 hybrid of spatial CWD and channel-wise cosine distance:

$$\mathcal{L}_{\mathrm{feat}} = \frac{1}{|S|} \sum_{s \in \lbrace s4, s8, s16, s32 \rbrace} \Big( 0.5 \cdot \mathcal{L}_{\mathrm{CWD}}(\hat{F}_s^{S}, F_s^{T}) + 0.5 \cdot (1 - \cos(\hat{F}_s^{S}, F_s^{T})) \Big)$$

The total student objective combines unweighted CE, multi-class Dice, spatial CWD, and 4-stage hybrid feature alignment (`λ_dice = 0.25`, `α_kd = 0.8`, `β_feat = 0.4`, `T = 4.0` in `configs/l4-plan.json`):

$$\mathcal{L} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{dice}} \mathcal{L}_{\mathrm{Dice}} + \alpha_{\mathrm{kd}} \mathcal{L}_{\mathrm{CWD}} + \beta_{\mathrm{feat}} \mathcal{L}_{\mathrm{feat}}$$

---

## 4. Asymmetric Robustness KD (`v4-final`, `seg_kd/training.py`)

Clean-distilled students (`v3-max`) match `90.8%` of teacher mIoU on clean frames but collapse to `13.75%` under high-frequency sensor noise (`σ=18`). Asymmetric Robustness KD resolves this cleanly:

1. **5-Epoch Ramp Schedule**: Linearly increases corruption probability $p$ from `0.07` to `0.35` over epochs 1–5 to stabilize early convergence.
2. **GPU-Batch Corrupted Student Stream (`apply_gpu_adverse_corruption`)**: Injects night gamma darkening, motion blur, fog low-contrast haze, and Gaussian sensor noise exclusively into student inputs.
3. **Clean Teacher Supervision**: The frozen `SegFormer-B5` always processes the uncorrupted frame, forcing the student to reconstruct clean-scene spatial logits and pyramid features directly from degraded sensor inputs.

---

## 5. QAT-KD, Validation-Only INT8 Gate & Temperature Scaling

1. **QAT-KD (`v3-max` 8 ep / `v4-final` BF16-Teacher 6 ep)**: Quantization-aware fine-tuning keeps static INT8 accuracy loss strictly below `1.0%p`.
2. **ONNX Export & Static QDQ INT8 (`seg_kd/export.py`)**: Verifies `100.0%` pixel argmax parity (`max_abs_logit_diff <= 3.6e-5`) between PyTorch and Opset 17 FP32 ONNX, then applies per-channel static QDQ INT8 calibration over 128 deterministic training frames.
3. **Validation-Only INT8 Gate (`assess_int8_acceptance`)**: Accepts INT8 strictly when all 4 validation checks pass (`val_mIoU_drop <= 1.0%p`, `val_safety_drop <= 1.5%p`, `size_ratio < 1.0`, `CPU_p50_speedup >= 1.20×`).
4. **Post-Hoc Validation Temperature Scaling (`seg_kd/evaluation.py`)**: Fits a single scalar temperature $`T^*`$ on `validation` NLL (`T* ≈ 1.70` across all CWD students), cutting `locked_eval` ECE by `69.5–78.5%` with mathematically zero mIoU change (`0.00%p`) and `0 ms` inference overhead.
