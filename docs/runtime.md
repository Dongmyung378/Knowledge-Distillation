> **Preserved documentation for the earlier four-candidate study.** Results, default download descriptors and training commands below belong to that study. For the current nine-cell extension use the [full-grid results and explicit reproduction commands](./full-grid.md) and [teacher/student analysis](./teacher-student-analysis.md). The launcher's original default descriptor must not be read as the new full-grid release. Original measurements and prose remain below.

<div align="right"><strong>English</strong> | <a href="./runtime.ko.md">한국어</a></div>

# CPU runtime and verification

The saved Keras and TFLite models can be evaluated and used for image inference without retraining. Full training uses a GPU-enabled Kaggle kernel. The local launcher always selects CPU and does not install a CUDA stack.

## Install once

Use a writable extracted source folder. On Windows x86-64:

```powershell
.\launch-kd.bat doctor
```

On Linux x86-64:

```bash
bash launch-kd.sh doctor
```

The launcher selects an available 64-bit Python 3.12 or obtains Python 3.12.15 with a hash-verified uv 0.13.0 bootstrap. It creates `.venv` without system site packages and installs the pinned runtime there. An incompatible existing environment is preserved under `.local/environment-backups`. No global Python packages are modified. `KD_PYTHON` can select a Python 3.12 executable; `KD_USE_MANAGED_PYTHON=1` exercises the managed-Python path.

First setup needs internet access for Python, packages, published models and the public dataset. Model files are not bundled with the source ZIP. The launcher does not require Docker, a GPU or an API key for saved-model CPU inference. Windows needs the Microsoft Visual C++ runtime required by TensorFlow. Automatic Python bootstrap is provided for Windows and Linux x86-64; other platforms are outside the recorded verification scope.

`doctor` checks package versions, virtual-environment isolation and an actual TensorFlow CPU tensor operation. It does not claim that models or a dataset have been installed unless explicitly supplied with `--model` or `--artifact-manifest`.

## Obtain and verify model files

```powershell
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime
.\launch-kd.bat download --manifest reports/artifact-download.json --output artifacts/runtime --verify-only
```

Use `bash launch-kd.sh` in place of `.\launch-kd.bat` on Linux. The descriptor binds HTTPS URLs, byte counts, SHA-256 digests and the exact extracted inventory. Downloads are staged before installation. A size/hash failure, unsafe ZIP path, extra file or incomplete inventory stops installation. Existing unverified artifacts are not silently overwritten. A null download URL means publication has not been verified and is an explicit failure rather than a fabricated fallback.

The published descriptor installs model files under `artifacts/runtime/models/`. `student.keras` is the selected seed-378 student's standard Keras artifact. The FP32 and INT8 exports use `student-fp32.tflite` and `student-int8.tflite`; the release results identify which format passed the validation acceptance rule. Until the descriptor contains verified public URLs, model delivery is pending.

## Data and saved-model inference

```powershell
.\launch-kd.bat fetch-data
.\launch-kd.bat predict --model artifacts/runtime/models/student.keras --image path/to/sign.png
```

`fetch-data` verifies the pinned public dataset archive and installs it under `.local/data/gtsrb-v1`. Use that root in `--data`. The original archive includes case-variant duplicate folders such as `Train` and `train`; the installer extracts only the exact paths referenced by the root `Train.csv`, `Test.csv` and `Meta.csv`, preserving canonical names on Windows and Linux. It records the extracted inventory and verifies source hashes. `fetch-data --archive path/to/gtsrb.zip` reuses a locally available archive after the same size/SHA-256 check, and `fetch-data --verify-only` checks an existing installation without network access. Keep the paths named by the published model inventory and results report when repeating measurements.

Inference accepts a local image and a `.keras` or `.tflite` model. It applies the same RGB conversion, 96 by 96 resize and `[0,1]` input normalization used for evaluation. The teacher contains its own MobileNetV2 `[-1,1]` rescaling. Default labels are numeric GTSRB class IDs; a compatible class-name JSON can be supplied through `--class-map`. An optional `--sha256` verifies the model before loading. Predictions are class estimates, not a production road-safety system.

## Repeat an evaluation

```powershell
.\launch-kd.bat evaluate --model artifacts/runtime/models/student.keras --manifest reports/split-manifest.json --data DATASET_ROOT --split locked_holdout --freeze reports/evaluation-freeze.json --output .local/reproduced-student
```

Published results identify the exact split manifest, freeze file and checkpoint paths. `reports/split-manifest.json` is the public training-split manifest; the Kaggle run's internal `splits.json` is not a second public file. `locked_holdout` and `official_test` evaluation require an appropriate pre-evaluation freeze. Reuse the published freeze for released models. Official test evaluation is a historically exposed audit. Evaluation writes aggregate metrics, a confusion matrix and ID-based predictions. Use a new output directory for each run.

```powershell
.\launch-kd.bat benchmark --model student=artifacts/runtime/models/student.keras --model int8=artifacts/runtime/models/student-int8.tflite --manifest reports/split-manifest.json --data DATASET_ROOT --output .local/cpu-benchmark --threads 1 --samples 128 --warmup 10 --repetitions 3
```

The benchmark records the hardware and thread policy. Timing and small floating-point differences can vary across hosts. Same source/model/data hashes do not guarantee bit-identical output on every TensorFlow build. Published verification reports state what was actually run; a configured command is not proof that it has been executed.

## Offline use and troubleshooting

After installation and model download, local image inference can run without downloading weights again. Dataset evaluation additionally requires the local dataset. `--verify-only` checks local artifacts without network access. Do not delete a working `.venv` to repair an unrelated network problem; the launcher already preserves incompatible environments when repair is needed.

If installation fails, retain the error and rerun `doctor` after connectivity is restored. If hashes differ, obtain the exact published artifact instead of disabling verification. If a final evaluation rejects a freeze, confirm that its bound model, manifest, source and configuration match. Full training through `launch-kd` is rejected because training is reserved for Kaggle GPU execution.

Sources: [TensorFlow installation](https://www.tensorflow.org/install/pip), [Python virtual environments](https://docs.python.org/3.12/library/venv.html), [Kaggle API](https://github.com/Kaggle/kaggle-api).
