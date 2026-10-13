<div align="right"><strong>English</strong> | <a href="./deployment.ko.md">한국어</a></div>

# Representative deployment and project outcome

The representative deployment retains the released **T5-a05, seed 378, FP32 TFLite** checkpoint. The complete nine-recipe experiment keeps **T5-a01** as its validation-selected recipe. Existing checkpoints, selection reports, evaluation freezes, prediction records and model bundles remain unchanged. The [deployment policy](../reports/deployment/deployment-policy.json) adds a transparent record of this later continuity decision.

## Why retain the prior deployment?

| Comparison scope | Retained T5-a05 | Study-selected T5-a01 |
| --- | --- | --- |
| Temperature / hard-label mixing coefficient | 5 / 0.5 | 5 / 0.1 |
| Three-seed validation accuracy mean | 86.65% | 87.08% |
| Three-seed reference accuracy mean ± sample SD | 87.51% ± 2.18 pp | 87.81% ± 2.08 pp |
| Seed-378 reference correct / count | 5,093 / 5,850 | 5,013 / 5,850 |
| Seed-378 reference accuracy | 87.06% | 85.69% |
| Seed-378 macro F1 | 0.7903 | 0.7719 |
| Seed-378 balanced accuracy | 0.7817 | 0.7620 |

The nine-cell study selects the **recipe** with the highest eligible mean validation accuracy. Its deployment/evaluation seed was fixed at 378 before final inference; it did not select the highest-scoring individual checkpoint after reading reference outcomes. That procedure legitimately selects T5-a01, but does not require replacing an already released checkpoint.

T5-a01's reference mean is 0.30 pp higher across three development seeds. This small descriptive difference does not establish a reliable deployment benefit. Its fixed seed-378 checkpoint produces 80 fewer correct predictions, a 1.37 pp decrease on the same reference collection. Retaining the prior release prioritizes continuity and its known reference performance. This is an observed-results-informed deployment decision, not a new validation-only selection or an independent confirmation result. The study winner, all nine recipes and unsuccessful exports remain reported.

## Representative model results and cost

All classification rows below use the same 5,850 images. CPU costs are from the **original T5-a05 release benchmark**, not substituted from the later T5-a01 benchmark.

| Metric | Teacher | Matched baseline, seed 378 | Retained KD, seed 378 |
| --- | --- | --- | --- |
| Correct / count | 4,887 / 5,850 | 4,303 / 5,850 | 5,093 / 5,850 |
| Accuracy | 83.54% | 73.56% | 87.06% |
| macro F1 | 0.7465 | 0.6253 | 0.7903 |
| Parameters | 2,313,067 | 34,406 | 34,406 |
| Standard Keras file | 9,843,117 bytes | 187,164 bytes | 187,164 bytes |
| Keras CPU p50 / p95 | 3.823 / 4.641 ms | 1.008 / 1.128 ms | 1.040 / 1.130 ms |

The student has approximately **98.5% fewer parameters** and **98.1% smaller standard Keras file size** than the teacher. The matched Keras p50 ratio is approximately **3.68x** on this host. The retained student's separate FP32 TFLite artifact has **143,316 bytes (140.0 KiB)**, **0.429 ms p50** and **0.456 ms p95**. Architecture compression, file-size reduction and measured speedup are different quantities.

The original benchmark uses an Intel Core i7-11800H CPU under Windows, one CPU thread, 128 matched validation images, batch one, ten warmup calls and three repetitions. Timings exclude framework import, model loading and image decoding. They do not guarantee latency on other devices. Full measurements are in [`reports/release-summary.json`](../reports/release-summary.json) and the [earlier results guide](./results.md).

T5-a05's reference accuracy across three seeds is **87.51% ± 2.18 pp**, versus **78.08% ± 3.94 pp** for paired supervised baselines. The mean paired gain is **+9.43 ± 3.93 pp**. These are sample means and standard deviations, not confidence intervals or teacher-seed uncertainty.

![Retained deployment and its original CPU measurements](../assets/figures/knowledge-distillation-release.png)

## Why did a student outperform the teacher?

The retained student repairs **473** teacher errors and introduces **267** errors on teacher-correct images, for **206** additional correct predictions and **+3.52 pp** accuracy. The study-selected T5-a01 separately repairs 465 and introduces 339, for a net 126. Both results are retained and identified by their actual checkpoint.

Distillation optimizes the teacher's softened class distribution rather than enforcing identical top-class predictions. The student can therefore learn different decision boundaries. Pure-soft T5-a00 students also exceed the teacher in **all three seeds**, so adding hard-label CE to the student objective cannot be the sole explanation. The teacher and mixed students both receive training-label supervision; labels also drive validation checkpoint selection for the pure-soft control.

Verified pipeline differences include a partially fine-tuned ImageNet teacher with frozen BN versus a fully trainable compact student with updating BN, different dropout, maximum budgets, optimization histories and selected epochs. These are plausible contributors to different predictions, not individually proven causes. The experiments establish observed student-over-teacher performance on this reference collection; they do not establish that compression itself improves accuracy or that students always outperform teachers. The [detailed analysis](./teacher-student-analysis.md) retains error counts, physical-track resampling, class-level results and matched controls.

## FP32 and rejected INT8 exports

FP32 uses 32-bit floating-point values. INT8 TFLite is a quantized export of the same trained student using 8-bit integer representations for weights and activations; it is not a different architecture or another trained model. Quantization may reduce storage and accelerate compatible hardware, but both accuracy and speed must be measured. See [Google's quantization specification](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/quantization_spec).

| Recorded release | FP32 bytes | INT8 bytes | Validation accuracy loss, FP32 minus INT8 | FP32-to-INT8 CPU p50 speedup | Decision |
| --- | --- | --- | --- | --- | --- |
| Retained T5-a05 | 143,316 | 55,472 | 61.863 pp | 0.771x | Reject INT8; retain FP32 |
| Study-selected T5-a01 | 143,316 | 55,472 | 46.410 pp | 0.786x | Reject INT8; retain study FP32 export |

Both files shrink by about 61%, but fail the at-most-1.0-pp accuracy-loss and at-least-1.05x CPU-speedup gates. Calibration uses training images only. The exact internal cause of the conversion accuracy loss has not been established. The earlier [INT8 diagnostic](../reports/int8-validation-diagnostic.json) excludes the tested runtime quantization path as the cause and records suspected range handling, without claiming a proven converter defect. INT8 files remain available as rejected reproducible experiments, not deployment recommendations.

Regenerate the representative figure from the unchanged measured summary and the explicit retention policy:

```bash
python scripts/plot_results.py --summary reports/release-summary.json --deployment-policy reports/deployment/deployment-policy.json --output .local/representative-figures
```

## Reproduce the correct role

The [representative model ZIP](https://park-dongmyung.com/downloads/knowledge-distillation-models-v2.zip) and original [`reports/artifact-download.json`](../reports/artifact-download.json) retain the 87.06% model and original SHA-256 bindings. Use the original evaluation freeze when reproducing that checkpoint:

```powershell
.\launch-kd.bat doctor
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
.\launch-kd.bat fetch-data
.\launch-kd.bat evaluate --model artifacts/runtime/models/student-fp32.tflite --manifest reports/split-manifest.json --data .local/data/gtsrb-v1 --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-representative-student
```

On Linux, use `bash launch-kd.sh` in place of `.\launch-kd.bat`. Use a fresh output directory. No Docker, GPU or API key is needed for saved-model CPU replay.

The [nine-recipe model ZIP](https://park-dongmyung.com/downloads/knowledge-distillation-full-grid-models.zip), [starting-evidence ZIP](https://park-dongmyung.com/downloads/knowledge-distillation-full-grid-base-evidence.zip), [`reports/full-grid/artifact-download.json`](../reports/full-grid/artifact-download.json) and [`reports/full-grid/evaluation-freeze.json`](../reports/full-grid/evaluation-freeze.json) remain unchanged. They reproduce the study under `artifacts/full-grid/`, including T5-a01, every other recipe and pure-soft controls. The [full-grid guide](./full-grid.md) provides the separate all-checkpoint verifier and Kaggle retraining commands.

## What the project demonstrates

- **Controlled experiments:** a complete 9-recipe by 3-seed KD grid, matched supervised baselines, pure-soft controls and preserved historical evidence.
- **Compression with measured tradeoffs:** the same compact inference structure across student objectives, parameter and file-size accounting, CPU latency measurement and rejected INT8 exports.
- **Result interpretation:** macro F1, balanced accuracy, class recall, paired prediction errors and transparent limits on causal explanations.
- **Reproducible delivery:** pinned CPU environments, track-disjoint data manifests, checkpoint and source hashes, frozen model bindings, public model downloads and saved-model replay commands.
- **Deployment judgment:** keeping a released checkpoint when the expanded experiment does not establish a replacement benefit, without rewriting the experiment outcome.

The project's value is an auditable model-compression workflow, not a claim of record-setting GTSRB performance. Original development remains **February-April 2026**. Reference outcomes were already observed and the official test was historically used for selection. Windows and Linux checks on the same computer verify those environments, not an independent external review. This 43-class cropped-sign study does not demonstrate production road-scene perception.
