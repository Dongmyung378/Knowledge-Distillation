> **Preserved documentation for the earlier four-candidate study.** Results, default download descriptors and training commands below belong to that study. For the current nine-cell extension use the [full-grid results and explicit reproduction commands](./full-grid.md) and [teacher/student analysis](./teacher-student-analysis.md). The launcher's original default descriptor must not be read as the new full-grid release. Original measurements and prose remain below.

<div align="right"><strong>English</strong> | <a href="./training.ko.md">한국어</a></div>

# Bounded Kaggle training

Full training runs on Kaggle with a visible GPU. The experiment entry point rejects real training on the local laptop. Local CPU tests use tiny synthetic fixtures to verify implementation; they are not training evidence for GTSRB performance.

The declared plan is [`configs/experiment-plan.json`](../configs/experiment-plan.json). The Kaggle run records the actual Python, TensorFlow, Keras, GPU and dataset inventory used. Kaggle's system Python and TensorFlow can change independently of this project. `scripts/setup_kaggle_runtime.py` therefore obtains a managed Python 3.12.15 and installs the pinned TensorFlow 2.19.0 environment plus its matching CUDA dependencies in a separate Kaggle runtime. It leaves the system installation and other projects' environments alone. Model and data hashes, saved checkpoints and compatibility tests determine which results can be repeated locally.

## Stages

1. **Pilot:** correct the MobileNetV2 input range; train its classification head for at most 5 epochs, then fine-tune up to the last 60 backbone layers for at most 15 epochs while freezing batch normalization. Train a matched independent student and four declared distillation candidates with seed 378.
2. **Confirmation:** select by validation only. Repeat the chosen fixed recipe and the matched baseline with seeds 379 and 380, reusing the pilot pair for seed 378. Preserve the exact same split and initialization within each paired seed.
3. **Freeze:** bind every completed paired checkpoint, selected configuration, data manifest and source hash before final holdout or official-test audit inference. This stage performs hashing and can run locally after the Kaggle outputs are downloaded.

The student stays at 34,406 parameters. Both baseline and distilled students use the same batch size, learning-rate policy, maximum 30 epochs, early-stopping patience of 5, augmentation and class weights. Best validation checkpoints are reloaded and verified. The teacher is frozen during distillation.

## Declared search and stopping rule

| Candidate | Temperature | Hard-label weight alpha |
| --- | ---: | ---: |
| `T3-a03` | 3 | 0.3 |
| `T5-a03` | 5 | 0.3 |
| `T10-a03` | 10 | 0.3 |
| `T5-a05` | 5 | 0.5 |

Validation uses unweighted classification accuracy. A candidate must improve the matched baseline by at least **0.2 percentage points** to qualify. Ties use lower validation cross-entropy, then declared candidate order. Confirmation requires mean paired validation gain of at least 0.2 percentage points and positive gains in at least two of the three seeds. If confirmation fails, the baseline remains the recommended deployment model and the unsuccessful distillation results are retained.

Deployment seed **378** is declared before final evaluation. Final outcomes cannot select another seed, temperature, architecture or checkpoint. Early stopping ends training when the validation rule has not improved within the declared patience. There are no additional search candidates after inspecting final metrics.

## Kaggle execution

Enable a GPU and attach the source package plus the pinned GTSRB dataset version. The system interpreter runs the runtime helper; the helper then executes the requested stage with isolated Python 3.12.15, pinned TensorFlow and CUDA libraries. From an extracted project at `/tmp/kd-source`:

```bash
python /tmp/kd-source/scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- -m knowledge_distillation train --data DATASET_ROOT --output /kaggle/working/kd-run --plan /tmp/kd-source/configs/experiment-plan.json --stage pilot
python /tmp/kd-source/scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- -m knowledge_distillation train --data DATASET_ROOT --output /kaggle/working/kd-run --plan /tmp/kd-source/configs/experiment-plan.json --stage confirmation
python /tmp/kd-source/scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- -m knowledge_distillation train --data DATASET_ROOT --output /kaggle/working/kd-run --plan /tmp/kd-source/configs/experiment-plan.json --stage freeze
```

Replace `DATASET_ROOT` with the verified mounted or staged dataset root. All stages reuse the same output folder. The CPU-only local launcher rejects training and must not be used for these commands.

The generated private Kaggle runner invokes all three stages programmatically and checks dataset hashes before starting. It copies only the original training CSV and the image paths it names into its training workspace; it does not open the official-test CSV. Create its source package locally without starting a job:

```bash
python scripts/package_kaggle.py --output .local/kaggle-package --kernel-id YOUR_ACCOUNT/YOUR_PRIVATE_SLUG --manifest reports/split-manifest.json
```

The output folder must not exist. Replace the account and slug with your own Kaggle account and a private script name. The generated metadata enables a T4 GPU and internet access, attaches the dataset and keeps the kernel private. Packaging creates files only; uploading or submitting them is a separate action. The embedded source archive includes the declared code, plan and bilingual Markdown documentation. It is a training payload, not the complete public repository archive. A successful submission alone is not a successful training run; kernel completion and downloaded, verified output artifacts are required.

For a generated worker script, the system interpreter invokes the bootstrap as follows:

```bash
python scripts/setup_kaggle_runtime.py --project-root /tmp/kd-source --runtime /tmp/knowledge-distillation-runtime --metadata-file /kaggle/working/kd-results/kaggle-runtime.json -- /tmp/kd-source/kaggle-worker.py
```

Arguments after `--` are Python arguments for the isolated interpreter. The helper verifies the official uv archive and binary hashes, preserves Kaggle's assigned GPU visibility, prioritizes the isolated CUDA libraries and verifies an actual GPU matrix multiplication before running the worker. Runtime packages remain under `/tmp` by default so they are not included in the published model outputs. A Windows or ordinary Linux machine is rejected before any GPU setup. The resulting JSON records versions and GPU metadata without credentials or unrelated environment variables.

An interrupted stage can reuse complete runs whose configuration, split and saved-model hashes match. Changing the plan in a started output folder is rejected. Completed pilot or confirmation reports cannot be overwritten to silently rerun selection. Once an evaluation freeze exists, further training or reselection in that run is prohibited.

## Outputs and publication

The run directory contains `plan-snapshot.json`, `splits.json`, `pilot-report.json`, `selected-configuration.json`, `confirmation-report.json` and `evaluation-freeze.json`, plus each standard `.keras` checkpoint and its training history. The public result report identifies completed runs, rejection decisions and actual final metrics. Retain failed or stopped experiments as structured results instead of presenting only successful candidates.

`splits.json` is the internal Kaggle run filename. The public repository retains the same declared training split as `reports/split-manifest.json`; reproduction commands use that published path rather than creating a duplicate manifest.

Public runtime models, evaluation manifests and hashes allow saved-model validation without retraining. Training histories and the source plan provide the method's provenance. Original notebooks and original scores remain historical development evidence; the improved experiment uses a separate evaluation protocol and cannot turn those old scores into an untouched benchmark.

## Export and final evaluation after training

After downloading a completed Kaggle run, `scripts/evaluate_release.py` verifies its source, saved checkpoints and training freeze. It exports FP32 and INT8 models, evaluates quantization on validation data, measures CPU latency, and creates the joint deployment freeze before final inference. All paired checkpoints are evaluated on the locked holdout; the historically exposed official test remains a separate audit. Use a new output directory and the full verified dataset, including the official test images.

```powershell
.\launch-kd.bat --setup-only
.\.venv\Scripts\python.exe scripts/evaluate_release.py --source . --run .local/kaggle-output/kd-results --data .local/data/gtsrb-v1 --output .local/release-evaluation
```

On Linux, prepare the CPU environment with `bash launch-kd.sh --setup-only` and invoke the same script with `.venv/bin/python`. The `--run` argument must identify the actual downloaded folder containing `kernel-completion.json`, `pilot-report.json`, `confirmation-report.json`, `splits.json` and the original freeze. Existing model, configuration and source bindings must match. To reproduce published scores directly, use the [saved-model evaluation commands](./runtime.md) with the published joint freeze.

Recreate the measured figure from an actual release summary:

```powershell
.\.venv\Scripts\python.exe scripts/plot_results.py --summary reports/release-summary.json --output .local/reproduced-figures
```

Sources: [Keras transfer learning](https://keras.io/guides/transfer_learning/), [Keras distillation example](https://keras.io/examples/vision/knowledge_distillation/), [Kaggle API](https://github.com/Kaggle/kaggle-api).
