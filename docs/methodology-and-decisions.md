# Experimental Progression, Design Rationale & Troubleshooting

[English README](../README.md) · [한국어](./methodology-and-decisions.ko.md) · [Architecture](./architecture.md) · [Results](./results.md) · [Verification Guide](./reproduction.md)

This document records the chronological experimental progression—**why** each dataset resolution, architecture, loss objective, quantization gate, robustness scheme, and post-hoc calibration choice was adopted, **which alternatives were rejected and on what quantitative evidence**, and how key engineering issues were resolved.

---

## 1. Protocol & Evaluation Design Decisions

| Decision | Rationale | Rejected Alternative |
| :--- | :--- | :--- |
| **`384×768` Resolution (`Chris1/cityscapes` full-res source)** | Preserves Cityscapes' `1:2` aspect ratio so small pedestrians/traffic lights remain resolvable while fitting a real-time 40 FPS CPU edge budget. | `128×256` (destroys small safety objects) or `1024×2048` (non-real-time on CPU). |
| **Strict 3-Way Split + SHA-256 Evaluation Freeze** | Splits the 2,975 training frames into `train 2,529 / validation 446` and locks the official 500-frame val set as `locked_eval`. Selection, INT8 gating, and temperature scaling `T*` are fit on `validation` only; `locked_eval` is evaluated once after SHA-256 freezing. | Tuning thresholds or INT8 acceptance on `locked_eval`. |
| **Shared-Init 3-Seed Paired Study (`378, 379, 380`)** | Eliminates random initialization luck by starting `baseline` and `cwd_feat` from identical SHA-256-verified initial checkpoints. | Unpaired single-seed comparisons. |
| **KITTI Semantics Zero-Shot (`200` imgs)** | Tests cross-sensor, cross-city generalization with zero fine-tuning. | In-domain Cityscapes evaluation only. |

---

## 2. Chronological Experimental Progression

| Stage | Core Question | Key Measured Result (`Locked Eval` / `Val Gate`) | Decision |
| :--- | :--- | :--- | :--- |
| **Stage 1. Teacher** | What upper bound does a ViT teacher set at `384×768`? | `SegFormer-B5` (`84.6M`, `323.14 MiB`): Val `82.96%`, Locked **`74.83%`** (Safety `65.57%`), KITTI `67.94%`, CPU `1631 ms` (`0.61 FPS`) | Freeze as BF16 teacher. |
| **Stage 2. 4-Method Loss Ablation** (`1.50M`, `s378`) | Pixel KL vs Spatial CWD vs 4-Stage Feature KD under `56×` ViT→CNN compression? | `baseline` `61.65%` → `pixel_kl` **`59.25%` (`-2.41%p`, Safety `-5.71%p`)** → **`cwd` `63.07%` (`+1.42%p`, KITTI Safety `+7.12%p`)** → `cwd_feat` `62.00%` | **Reject `pixel_kl`**; adopt spatial `CWD`. |
| **Stage 3. 3-Seed Paired Study** (`1.50M`, `378/379/380`) | Are gains reproducible across seeds and across all 6 safety classes? | `61.42% ± 0.21%` → **`62.03% ± 0.08%`** (`+0.61%p`, **3/3 wins**, `2.7×` smaller SD); Safety `49.52%` → **`51.40%`** (`+1.88%p`, **6/6 positive**) | Reproducibility & variance stabilization confirmed. |
| **Stage 4. 3-Tier Pareto & INT8 Gate** (`v3-max`) | Which backbone survives static QDQ INT8 with real CPU speedup? | `V4-Small` INT8: `1.04× < 1.20×` (rejected); `V3-Large` INT8: **`-6.46%p` / `0.72×` collapse (blocked)**; **`V4-Medium` INT8: 4/4 PASS (`67.95%`, `7.90 MiB`, `40.49 FPS`, `66.0×` speedup)** | **Adopt `MobileNetV4-Medium INT8`** (beats FP32 baseline `67.15%` by `+0.80%p` at `1/3.6` size). |
| **Stage 5. Adverse-Weather Diagnosis & `v4-final`** | Does clean INT8 survive night, blur, fog, and camera noise? | `v3 Clean INT8` collapsed to **`13.75%`** under Gaussian noise (`σ=18`) → **Asymmetric Robust KD (Clean Teacher → Corrupted Student) + BF16 QAT-KD** → **`59.59%` (`+45.84%p`, surpasses 84.6M teacher `59.19%`)**, Fog `66.33%` | Establish **two-profile deployment** (`v3 Clean` vs `v4 Robust`). |
| **Stage 6. Post-Hoc Temperature Scaling** | Can CWD overconfidence (`ECE 0.025–0.031`) be fixed at zero latency/mIoU cost? | Validation-only NLL fit converges consistently to `T* = 1.70–1.72` across all CWD models → **ECE `-69.5–78.5%`**, `0.00%p` mIoU change, `0 ms` overhead | Adopt scalar `T*` post-processing. |

---

## 3. Why Each Key Choice Was Made

1. **Why `CWD` over `Pixel KL`**: Pixel-wise KL weights gradients by pixel area, allowing road/sky/building to drown out tiny vulnerable road users (`-5.71%p` safety collapse). Channel-wise spatial softmax (`H×W` per class channel) gives every safety class equal spatial supervision (`+1.42%p` overall, `+7.12%p` KITTI safety). Moreover, pure `cwd` beats `cwd_feat` at `56×` compression (`1.50M`), whereas 4-stage `0.5·CWD + 0.5·Cosine` feature alignment shines at `11.2×` compression (`7.54M`).
2. **Why `MobileNetV4-Medium` over `MobileNetV3-Large` / `V4-Small`**: `MobileNetV3-Large`'s 18 unfused SE + HardSwish branches (`74 Conv / 56 BN`) broke static QDQ INT8 (`-6.46%p`, `0.72×` slowdown), and `V4-Small` was too small to amortize QDQ overhead (`1.04×`). Pure `Conv-BN-ReLU` UIB blocks in `MobileNetV4-Medium` (`88 Conv / 86 BN`) passed all four gate checks in both `v3-max` and `v4-final` (`1.55×` CPU speedup, `-72.5%` file size).
3. **Why Asymmetric Robustness KD (`Clean Teacher → Corrupted Student`)**: Feeding corrupted inputs to the teacher degrades its own supervision (`59.19%` under `σ=18`). Keeping the frozen `SegFormer-B5` strictly on clean frames while corrupting student inputs (5-epoch ramp `p: 0.07 → 0.35`) trains the student as a cross-architecture denoising segmenter, recovering `+45.84%p` under noise and surpassing the 84.6M teacher (`59.59%` INT8 / `60.61%` FP32 vs `59.19%`).
4. **Why Validation-Only Temperature Scaling (`T* ≈ 1.70`)**: Spatial `T=4.0` CWD sharpens student logit norms, roughly doubling ECE (`0.025–0.031`). Dividing logits by a single validation-fit scalar `T*` preserves argmax identically (`0.00%p` mIoU delta, `0 ms` latency) while cutting ECE by up to `78.5%`.

---

## 4. Engineering Troubleshooting Log

1. **Unweighted CE vs Class-Weighted CE**: Heavy inverse-frequency CE weighting caused false-positive halo artifacts around small classes; keeping CE unweighted while delegating minority-class focus to spatial `CWD` raised both overall mIoU and 6/6 safety classes.
2. **BF16-Teacher QAT-KD Stability**: Isolated student FakeQuantize execution from `channels_last` conflicts via explicit `cfg.get("channels_last", False)` guards and ran only the frozen teacher forward pass under CUDA BF16 autocast (~148 s/epoch on L4).
3. **100.0% PyTorch ↔ ONNX FP32 Parity**: Wrapped export with `_PlainLogitsExportWrapper` to strip training-only projector banks cleanly, enforcing automated `100.0%` pixel argmax parity (`max_abs_logit_diff <= 3.6e-5`) before INT8 calibration.
