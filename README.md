<div align="right">
  <strong>English</strong> | <a href="./README.ko.md">한국어</a>
</div>

# Knowledge Distillation for Traffic Sign Recognition

A TensorFlow model-compression project for GTSRB's 43 traffic-sign classes. A MobileNetV2 teacher guides a compact CNN with **34,406 parameters**, while a matched independently trained CNN measures the benefit of distillation without changing the student's inference architecture.

**Original development period: February-April 2026.** The later engineering improvements retain that period and add a separate, controlled evaluation protocol.

[Architecture](./docs/architecture.md) | [Evaluation](./docs/evaluation.md) | [CPU runtime](./docs/runtime.md) | [Kaggle training](./docs/training.md) | [Historical notebook](./notebooks/knowledge_distillation_gtsrb.ipynb)

## Results and evidence

The seed-378 checkpoints were frozen before evaluation on the same **5,850-image** locked holdout. All paired outcomes and verification evidence are in the [measured results report](./docs/results.md).

| Measured protocol result | Teacher | Matched baseline | Fixed distilled candidate |
| --- | ---: | ---: | ---: |
| Locked-holdout accuracy, correct / evaluated | 83.54%, 4,887 / 5,850 | 73.56%, 4,303 / 5,850 | 87.06%, 5,093 / 5,850 |
| Locked-holdout macro F1 | 0.7465 | 0.6253 | 0.7903 |
| Three-seed accuracy mean ± sample SD | N/A | 78.08% ± 3.94 pp | 87.51% ± 2.18 pp |
| Three-seed paired gain mean ± sample SD | N/A | Reference | +9.43 ± 3.93 pp |
| CPU batch-one p50 / p95 | 3.823 / 4.641 ms | 1.008 / 1.128 ms | 1.040 / 1.130 ms |
| Standard Keras inference artifact size | 9.39 MiB (9,843,117 bytes) | 182.8 KiB (187,164 bytes) | 182.8 KiB (187,164 bytes) |
| Parameters | 2,313,067 | 34,406 | 34,406 |

Means and sample standard deviations describe seeds 378, 379 and 380; they are not confidence intervals. CPU entries above measure matched Keras calls on validation images and are separate from TFLite deployment costs.

CPU measurement: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz (Microsoft Windows 11 Pro), 1 thread(s), 128 matched images, 10 warmup calls and 3 repetitions. These timings do not guarantee latency on another device.

**Validation selection:** `T5-a05`, mean paired accuracy gain +9.16 pp, 3/3 positive pairs, declared gate passed. The holdout gain is reported separately above and does not reopen selection.

**Release decision:** seed-378 distilled, **FP32 TFLite**. INT8 was rejected with validation accuracy loss +61.863 pp and measured CPU p50 speed ratio 0.771x.
Failed criteria: validation accuracy loss, CPU p50 speedup.
The selected deployment artifact records **87.06% (5,093/5,850)** locked-holdout accuracy, **0.7903** macro F1, **0.429 / 0.456 ms** CPU p50 / p95 and **140.0 KiB (143,316 bytes)** size.
Final outcomes do not reselect the seed, checkpoint or quantization format.

![Measured holdout quality and CPU deployment costs](./assets/figures/knowledge-distillation-release.png)

### Historical development results

| Historical measurement | Recorded value |
| --- | ---: |
| Teacher best selected-epoch accuracy | 88.37% |
| Compact baseline best selected-epoch accuracy | 54.61% |
| Best distilled selected-epoch accuracy | 80.97% |
| Student parameters / teacher parameters | 34,406 / 2,313,067 |
| Best historical setting | `temperature=10`, `alpha=0.3` |

These values come from the original single-seed, nine-condition notebook. Its official GTSRB test split was used for checkpoint and hyperparameter selection. They are historical development results, not untouched-test scores. The baseline's final in-memory model scored 54.12%, whereas the table reports its best history epoch, 54.61%. Historical `model.predict()` timing includes framework overhead and did not measure a teacher-to-student deployment speedup. Parameter reduction of approximately 67.2 times is not a measured speedup.

The new split and selection procedure differ from the historical experiment. Old and new percentages must not be treated as a controlled before/after comparison on the same untouched evaluation population.

## What the improved implementation demonstrates

1. **Evaluation by physical sign.** Class-stratified train, validation and locked-holdout sets keep every traffic-sign track in one split. Relative paths, class labels, track IDs, image bytes and SHA-256 hashes bind the exact inventory. Official-test inference is a separate audit after freezing.
2. **Matched Baseline/KD controls.** Both students share the 34,406-parameter architecture, initial weights within each seed, deterministic data order, augmentation, class weights and maximum training budget. Seeds 378, 379 and 380 measure paired variation. The deployment seed is declared as 378 before final evaluation.
3. **Correct teacher adaptation.** `[0,1]` RGB inputs are augmented and rescaled inside the teacher to MobileNetV2's `[-1,1]` input range. Head training precedes low-learning-rate fine-tuning with frozen batch normalization. The teacher's weights and saved artifact are checked for immutability during distillation.
4. **Bounded, testable distillation.** Four declared KD recipes use class-weighted hard-label CE plus unweighted Teacher-to-Student KL with temperature-squared compensation. Validation records real, unweighted CE. Best plain `.keras` checkpoints are saved and reloaded. A 0.2 percentage-point validation improvement gate and three-seed confirmation stop ineffective candidates.
5. **Measured deployment tradeoffs.** Standard `.keras` models support inference without the training wrapper. FP32 TFLite conversion and full INT8 quantization are evaluated separately. INT8 calibration uses training records only. CPU benchmarks compare the same images, thread policy and warmed invocation path, reporting latency, throughput and file size; validation controls quantization acceptance.
6. **Portable, auditable delivery.** Isolated CPU launchers, pinned dependencies, source snapshots, exact artifact inventories and SHA-256 checks support saved-model verification without retraining. Correctness tests exercise loss gradients, immutable teachers, real optimizer updates, split isolation, checkpoint reloads and tampering rejection.

These are implemented capabilities. Completed training, export and environment checks are separately identified in the release evidence rather than inferred from the presence of code.

## Run without retraining

If you use a source ZIP from the portfolio project page, extract it to a writable folder and run the same commands.

The default local runtime uses CPU. Docker, a GPU and an API key are unnecessary for public saved-model inference. First setup and downloads need internet access. Use a writable source folder on supported Windows or Linux x86-64 with Python 3.12; the launcher can obtain the pinned Python runtime when it is missing.

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

The example image is a training-collection input for checking the inference path, not a quality benchmark. The model descriptor extracts checkpoints under `artifacts/runtime/models/`; the published inventory identifies the exact filenames. The model bundle was downloaded in full without authentication from the [verified endpoint](https://park-dongmyung.com/downloads/knowledge-distillation-models-v2.zip) and matched its SHA-256 and byte count (10.83 MiB (11,353,209 bytes)). `reports/artifact-download.json` installs its exact declared inventory.

### Repeat the frozen evaluation

Use the published `reports/split-manifest.json` and pre-evaluation freeze. Keep the original freeze when repeating a released model's evaluation.

```powershell
.\launch-kd.bat evaluate --model artifacts/runtime/models/student-fp32.tflite --manifest reports/split-manifest.json --data .local/data/gtsrb-v1 --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-student
```

On Linux, replace `.\launch-kd.bat` with `bash launch-kd.sh`. Use a new output directory for each run. Published manifests and freeze files identify the exact model, denominator, configuration and source. Evaluation produces aggregate metrics, an all-class confusion matrix and image-ID predictions.

### Run correctness tests

After `doctor` has prepared `.venv`:

```powershell
$env:CUDA_VISIBLE_DEVICES = "-1"
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -m unittest discover -s tests -v
```

**Runtime and saved-model verification:** Windows, Python 3.12.15: 62/63 correctness tests passed, 1 skipped, 0 failures. Full 5,850-image saved-model holdout reruns matched reference accuracy and macro F1 (`baseline` 4,303/5,850, `distilled` 5,093/5,850, `fp32` 5,093/5,850). Linux (WSL2), Python 3.12.15: 63/63 correctness tests passed, 0 skipped, 0 failures. Full 5,850-image saved-model holdout reruns matched reference accuracy and macro F1 (`baseline` 4,303/5,850, `distilled` 5,093/5,850, `fp32` 5,093/5,850). The operating-system checks use the same physical computer and are not an independent external review. Synthetic correctness checks and full saved-model evaluation are reported as separate scopes.

## Training and reproduction limits

Full project training runs only in a GPU-enabled Kaggle kernel. The [declared plan](./configs/experiment-plan.json) bounds teacher variants and KD candidates, separates pilot selection from paired confirmation, and freezes model, source and configuration hashes before final inference. See the [training guide](./docs/training.md) for commands and saved outputs. The original notebook is historical evidence rather than the current training entry point.

The locked holdout is excluded from training and selection in the new experiment, but those images belonged to the historical notebook's full training collection. New models start from fresh ImageNet teacher weights and fresh student initialization. This does not establish that the public data were unseen by every historical or upstream pretrained model. The official test was historically exposed and remains labelled as an audit.

Same source, model and data hashes support repeated saved-model evaluation. Hardware, drivers and floating-point implementations can affect timing and small numerical differences. Checks in another operating-system environment on the same computer are environment verification, not an independent external review. GTSRB classification is a bounded research task and does not demonstrate production road-scene perception.

## Repository guide

| Location | Purpose |
| --- | --- |
| `knowledge_distillation/` | Track splits, models, training, evaluation, export and CPU inference |
| `configs/experiment-plan.json` | Declared search budget, paired seeds and stopping policy |
| `scripts/` and `launch-kd.*` | Isolated runtime setup and verified downloads |
| `tests/` | Synthetic correctness, integrity and portability checks |
| `reports/` | Published results, manifests, freezes and verification evidence |
| `docs/` | English documentation with corresponding Korean versions |
| `notebooks/knowledge_distillation_gtsrb.ipynb` | Labelled historical experiment |

Method references: [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html), [knowledge distillation](https://arxiv.org/abs/1503.02531), [MobileNetV2 preprocessing](https://keras.io/api/applications/mobilenet/mobilenet_models/), [Keras transfer learning](https://keras.io/guides/transfer_learning/).
