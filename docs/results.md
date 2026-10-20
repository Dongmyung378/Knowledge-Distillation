> **Two separate roles.** The representative deployment retains the released **T5-a05, seed 378, FP32 model (87.06%)**; see the [deployment policy and project outcome](./deployment.md). The complete nine-recipe experiment still selects **T5-a01** on validation; use the [full-grid guide](./full-grid.md) for its separate frozen artifacts and replay commands. The earlier four-candidate documentation below is preserved, including the original descriptor and freeze now used for representative-model replay.

<div align="right"><strong>English</strong> | <a href="./results.ko.md">한국어</a></div>

# Measured results and deployment decision

Measured results with supplied saved-model reproduction and anonymous-delivery receipts.

The improved experiment evaluates a fixed 34,406-parameter student against a matched independently trained student and an ImageNet-initialized MobileNetV2 teacher. Current model selection uses a class-stratified, physical-sign-track validation split. Model, configuration, source and data hashes are frozen before locked-holdout inference. The declared deployment seed is 378.

## Reading the controls and metrics

The teacher is the larger source of class probabilities. The baseline and distilled candidate share the smaller architecture; only the latter learns from the teacher as part of its objective. A seed identifies repeatable pseudorandom starting choices. Matching baseline and distilled runs within each seed lets their difference be compared before averaging across repetitions. Sample SD after ± describes the spread of those repetitions, not a confidence interval.

Precision asks how often a predicted sign class is correct, and recall asks how many actual images of that class were found. Macro F1 balances these measures within each class and gives every class equal weight. Percentage-point gains subtract accuracies directly. CPU p50 is median latency and p95 summarizes slower calls; parameter reduction alone is not a speedup measurement. FP32 TFLite is a saved floating-point inference artifact, whereas INT8 is a lower-precision conversion evaluated for its own accuracy and runtime costs. The broader [experiment guide](full-grid.md) explains the recipe labels and the separate deployment decision.

## Locked-holdout quality

All rows below use the same **5,850 images** and the same record inventory. Accuracy is correct predictions divided by evaluated images; macro F1 averages all 43 classes and uses the 0-1 scale.

| Seed-378 model | Correct / evaluated | Accuracy | Macro F1 | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 4,887 / 5,850 | 83.54% | 0.7465 | 2,313,067 |
| Matched baseline | 4,303 / 5,850 | 73.56% | 0.6253 | 34,406 |
| Fixed distilled candidate | 5,093 / 5,850 | 87.06% | 0.7903 | 34,406 |

For declared seed 378, KD minus baseline accuracy is +13.50 percentage points (+790 correct predictions). Class recall increased in 33 classes, decreased in 6, and was unchanged in 4. These are observed errors and gains; no causal explanation of soft-label transfer is inferred from these counts.

Teacher describes its role in the distillation loss, not guaranteed superior accuracy. The seed-378 distilled student achieved 87.06% versus the teacher's 83.54% on this evaluation.

The paired table records every completed seed under the fixed recipe; final outcomes do not select a preferred seed.

| Training seed | Baseline accuracy | Fixed KD accuracy | KD minus baseline |
| --- | ---: | ---: | ---: |
| 378 | 73.56% | 87.06% | +13.50 pp |
| 379 | 80.75% | 89.88% | +9.13 pp |
| 380 | 79.93% | 85.59% | +5.66 pp |

Mean paired gain: **+9.43 percentage points**. Sample standard deviation: **3.93 percentage points**, a descriptive measure over three training seeds. It is not a confidence interval or a population-level significance claim. Teacher training is fixed across student seeds.

## Bounded improvement experiments

The preregistered plan contains four KD candidates, one teacher policy and at most 30 student epochs with validation patience five. Baseline and KD share initialization within each seed, input order, augmentation, class weights and the maximum budget. Their realized stopping epochs may differ. KL is unweighted, hard-label CE uses class weights and validation CE is unweighted.

Selected pilot recipe: **T5-a05**. Three-seed validation confirmation: **passed; mean validation gain +9.16 pp, 3/3 positive pairs**. Stopped or rejected candidates: **No candidate failed the pilot improvement gate; unselected candidates remain recorded**. A candidate must exceed the matched baseline by at least 0.2 percentage points in the pilot, then meet the declared three-seed confirmation rule. Final outcomes did not select a new configuration or deployment seed.

## CPU cost and model formats

Host: **11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz**. Threads: **1**. The benchmark uses batch size one, **128 identical validation images**, **10 warmups** and **3 repetitions**. Each artifact runs in a fresh process. Pure inference includes synchronized output but excludes framework import, model load and image decoding; the machine-readable report also records end-to-end timing and process memory.

| Inference artifact | p50 | p95 | File size |
| --- | ---: | ---: | ---: |
| Teacher Keras | 3.823 ms | 4.641 ms | 9.39 MiB (9,843,117 bytes) |
| Baseline Keras | 1.008 ms | 1.128 ms | 182.8 KiB (187,164 bytes) |
| Fixed KD Keras | 1.040 ms | 1.130 ms | 182.8 KiB (187,164 bytes) |
| Selected student FP32 TFLite | 0.429 ms | 0.456 ms | 140.0 KiB (143,316 bytes) |
| Rejected INT8 TFLite export | 0.556 ms | 0.602 ms | 54.2 KiB (55,472 bytes) |

INT8 validation accuracy loss (FP32 minus INT8): **+61.863 percentage points**. Same-CPU p50 speedup relative to FP32 TFLite: **0.771 times**. Acceptance: **rejected; failed checks: validation accuracy loss, CPU p50 speedup**. Calibration uses training images only; the validation gate requires at most one percentage-point accuracy loss, a smaller file and at least 1.05 times CPU speedup.

The [INT8 diagnostic](../reports/int8-validation-diagnostic.json) found no runtime quantization error: independent float32 quantization and direct interpreter calls exactly matched the runner on 43 validation images. On the same 256 calibration images, FP32 pre-BN convolution maxima were 1.103, 4.265 and 11.847, while the corresponding INT8 ranges reached 13.198, 28.672 and 246.213, matching the following BN-MUL maxima. Converter or calibrator range handling is suspected, but the internal cause is not proven. This INT8 file is retained as an experimental artifact; FP32 is the recommended runtime. No further training, re-export or deployment reselection was performed after freezing.

Recommended deployment: **distilled / FP32, seed 378**, determined before final holdout evaluation. File-size reduction and parameter reduction are not themselves measured speedups. Timings describe this host and thread policy.

## Historically exposed official-test audit

The additional **12,630-image** audit started after freezing. Teacher, baseline and fixed KD accuracies were **82.72%**, **75.30%** and **86.63%**, respectively. These are separate audit results because the original notebook used the official test for selection; they are not independent untouched-test scores.

The original notebook's 54.61% baseline and 80.97% KD scores remain historical selected-epoch results. Its single-seed, nine-condition selection procedure differs from the improved experiment, so those percentages are not placed on a before/after axis with the new locked-holdout results.

## Verification and practical limits

Saved-model reproduction: **windows: complete saved-model holdout rerun with matching accuracy/F1 (baseline 4,303/5,850, distilled 5,093/5,850, fp32 5,093/5,850); linux: complete saved-model holdout rerun with matching accuracy/F1 (baseline 4,303/5,850, distilled 5,093/5,850, fp32 5,093/5,850); Windows clean CPU setup and synthetic correctness tests; Linux (WSL2, same physical host) clean CPU setup and synthetic correctness tests**. Tests: **Windows: 62/63 passed, 1 skipped, 0 failures; Linux (WSL2, same physical host): 63/63 passed, 0 skipped, 0 failures**. Public artifact delivery: **Anonymous full download and SHA-256 match, 10.83 MiB (11,353,209 bytes); SHA-256 d1b7de4bc7fa67fb0a7ccabfaf32d7b5cd3b84b4686c0d714427fc73aae50e5b; [verified endpoint](https://park-dongmyung.com/downloads/knowledge-distillation-models-v2.zip)**. Exact source, model and input hashes are recorded in **release-summary.json, evaluation-freeze.json, reports/split-manifest.json; release freeze f833aa3b35baf9be1bae39f112ffd3a5157aae2a9d111f9753bf668332b853ba; split-manifest content adf978947af03876338282d798bf9c179267752e8c36f71b21865031cbe7a546**. A smoke test demonstrates the tested execution path, not full-dataset accuracy unless that complete evaluation is explicitly recorded.

The holdout is excluded from the new model's training and selection, but its images belonged to the historical project's original training collection. Fresh ImageNet teacher initialization and new student initialization do not prove that every historical or upstream model was unfamiliar with these public images. Small floating-point differences and CPU timings can vary across environments. Checks on another operating system on the same host are environment verification rather than an external independent review.

The conclusion supported by this release is **the fixed KD recipe has a descriptive mean held-out accuracy gain of +9.43 pp over three matched seeds, with deployment selected on validation**. The project demonstrates a bounded model-compression comparison and verifiable deployment workflow; it does not establish production road-scene reliability or superiority over unrelated benchmark publications.
