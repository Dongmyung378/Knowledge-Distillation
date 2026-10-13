> **Two separate roles.** The representative deployment retains the released **T5-a05, seed 378, FP32 model (87.06%)**; see the [deployment policy and project outcome](./deployment.md). The complete nine-recipe experiment still selects **T5-a01** on validation; use the [full-grid guide](./full-grid.md) for its separate frozen artifacts and replay commands. The earlier four-candidate documentation below is preserved, including the original descriptor and freeze now used for representative-model replay.

<div align="right"><strong>English</strong> | <a href="./evaluation.ko.md">한국어</a></div>

# Evaluation protocol

The original notebook used the official GTSRB test collection during checkpoint and hyperparameter selection. Its recorded 80.97% distilled accuracy, 54.61% baseline accuracy and 88.37% teacher accuracy are historical selected-epoch results. They are not measurements on an untouched final test set. Original latency includes `model.predict()` overhead and cannot establish a deployment speedup.

## Separate selection from final measurement

The improved protocol builds train, validation and locked-holdout splits from the official training collection. Track IDs are the grouping unit, with per-class stratification. The manifest stores the exact image inventory and hashes. Validation selects checkpoints, schedules and distillation parameters. Locked-holdout images are excluded from those decisions and evaluated only after freezing.

The locked holdout is new to this improvement experiment, but its images belonged to the original notebook's full training collection. It must not be described as entirely unseen by every historical model. Pretrained ImageNet exposure is not established or excluded by this project's split. Official test results are explicitly labelled a historically exposed audit. Old and improved percentages have different selection conditions and should not be presented as a single controlled before/after experiment.

## Freeze before inference

Released-model reproduction uses the published freeze directly. The command below illustrates the developer step for a new experiment before any final measurements; it is not a required setup step for a released checkpoint.

```powershell
.\launch-kd.bat freeze --model teacher=artifacts/runtime/models/teacher.keras --model baseline=artifacts/runtime/models/baseline.keras --model student=artifacts/runtime/models/student.keras --manifest reports/split-manifest.json --config reports/selected-configuration.json --output reports/evaluation-freeze.json --source-root .
```

The freeze binds checkpoint hashes, the split manifest, selected configuration and relevant source files. `locked_holdout` and `official_test` evaluation reject a missing or incompatible freeze. Reports must state the exact checkpoint and denominator. Do not recompute a freeze after reading final outcomes to conceal a changed model.

## Classification quality

```powershell
.\launch-kd.bat evaluate --model artifacts/runtime/models/student.keras --manifest reports/split-manifest.json --data DATASET_ROOT --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/student-final
```

Report accuracy as correct predictions divided by evaluated images, macro F1, per-class recall, a confusion matrix and evaluated record IDs. Repeated matched seeds measure variation in the selected training recipe. Each seed pairs the baseline and distilled student under the same architecture, input split, optimization budget and seed policy. Report every completed seed and any stopped run, not only the best seed.

Model selection uses the same fixed validation rule across candidates. An experimental candidate that fails the declared improvement criterion is stopped. Final held-out results document generalization under the protocol and are not used to reopen that search. A smaller model or a changed environment is a different comparison and must be labelled accordingly.

## Official test audit

```powershell
.\launch-kd.bat official-test-manifest --data DATASET_ROOT --training-manifest reports/split-manifest.json --freeze reports/evaluation-freeze.json --output .local/official-test.json
.\launch-kd.bat evaluate --model artifacts/runtime/models/student.keras --manifest .local/official-test.json --data DATASET_ROOT --split official_test --freeze reports/evaluation-freeze.json --output .local/student-official-audit
```

This audit starts only after selection is frozen. Official test image paths and labels come from the dataset's test CSV. It is an additional historically exposed dataset check, not an independent untouched benchmark or proof of superiority over other publications.

## CPU cost and quantization

Measure teacher, baseline and distilled models on the same CPU, sample set, thread policy and warmed invocation path. Report median and tail latency, throughput, file size, parameter count, measurement scope and environment. Framework startup and preprocessing should be documented separately. Benchmark results are hardware-specific.

INT8 calibration uses training images only. FP32 TFLite and INT8 TFLite are evaluated using the same fixed inputs as their source model. Acceptance combines accuracy loss and actual CPU latency; size reduction alone does not prove a speedup. The maximum allowed accuracy drop is declared before final evaluation. If INT8 does not meet the preset criteria, retain it as a documented rejected candidate and recommend the verified FP32 path.

## Reproduction limits

Saved model, source and data hashes let another user repeat the evaluation without retraining. Exact retraining trajectories and CPU timing can differ across drivers, TensorFlow builds and hardware. Windows/Linux checks performed on the same host are separate environment tests, not an external independent reviewer. A download or smoke test demonstrates delivery and inference, not full-dataset quality unless the report explicitly records that evaluation.

Final measured results belong in the result reports. No training or final score is implied merely by implementing this protocol.
