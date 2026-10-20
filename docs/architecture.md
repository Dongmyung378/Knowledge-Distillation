<div align="right"><strong>English</strong> | <a href="./architecture.ko.md">한국어</a></div>

# Architecture

The runtime takes a cropped sign image and predicts a sign class. Training additionally uses a larger teacher to guide a smaller student with its class-probability distribution. The supervised baseline uses the same small architecture without teacher guidance. A seed fixes pseudorandom starting choices so the two small-model training paths can be matched. Model export then packages the saved student for CPU inference; exporting, evaluating accuracy and measuring latency are separate steps.

The project compares a MobileNetV2 teacher, an independently trained compact student and a distilled student on GTSRB's 43 classes. Baseline and distilled students share the same 34,406-parameter architecture. Their inference graphs are the same; distillation changes the training objective, not the deployment model.

## Input and model boundary

Every model accepts 96 by 96 RGB images normalized to `[0,1]` and returns 43 unnormalized class logits. The teacher contains its own `Rescaling(2.0, offset=-1.0)` before ImageNet-pretrained MobileNetV2, so its backbone receives the required `[-1,1]` range. This avoids a separate teacher-only normalization in data loaders or the public inference CLI.

The compact student uses three depthwise-separable convolution blocks, global average pooling and a small dense classifier. Model files use standard Keras layers and can be loaded without a custom distillation wrapper. The inference path applies softmax once to produce class probabilities. TFLite inference reads tensor quantization metadata and applies the correct input quantization and output dequantization.

## Training objective

```text
hard_loss = cross_entropy(labels, student_logits)
soft_loss = KL(softmax(teacher_logits / T), softmax(student_logits / T))
loss = alpha * hard_loss + (1 - alpha) * T**2 * soft_loss
```

The teacher is called with `training=False`, and its weights remain unchanged during student training. Class weights, augmentation, training budget and initialization policy are matched between baseline and distillation comparisons. Validation reports a real classification loss and accuracy rather than a placeholder zero. The best validation checkpoint is saved and reloaded for final evaluation.

## Data boundary

The official training collection is divided by traffic-sign track, stratified within class, into training, validation and locked holdout. Images from the same track cannot cross those boundaries. Relative paths, labels, track IDs, sizes and SHA-256 digests are recorded in the manifest. Data loading checks records against that inventory.

Training and model selection can use training and validation only. Final holdout inference is permitted after model, data, source and configuration hashes have been frozen. The official test collection is kept as a separate audit because the original notebook already used it for model selection. Neither audit scores nor locked-holdout outcomes are used to retune the published configuration.

## Modules

| Module | Responsibility |
| --- | --- |
| `data.py` | Dataset inventory, track-disjoint splits and deterministic evaluation inputs |
| `models.py` | Standard teacher and compact-student models |
| `training.py` | Supervised and distilled optimization, checkpoints and validation |
| `experiments.py` | Bounded Kaggle experiment stages and stop decisions |
| `evaluation.py` | Frozen-model evaluation, confusion matrices and prediction records |
| `benchmark.py` | Warmed CPU measurements using the same input images |
| `export.py` | FP32 and INT8 TFLite conversion with training-only calibration |
| `runtime.py` | Safe saved-model loading and local image inference |
| `__main__.py` | Explicit command-line entry points |

Actual file names and public results are listed in the repository. A diagram or module interface alone does not prove that a training run or deployment verification succeeded; the results and verification reports provide that evidence.

Sources: [MobileNetV2 preprocessing](https://keras.io/api/applications/mobilenet/), [knowledge distillation paper](https://arxiv.org/abs/1503.02531), [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html).
