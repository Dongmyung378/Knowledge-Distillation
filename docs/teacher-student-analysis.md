<div align="right"><strong>English</strong> | <a href="./teacher-student-analysis.ko.md">한국어</a></div>

# Teacher/student error differences and fixed-T5 controls

## Current deployment and supported conclusion

The representative deployment retains **T5-a05, seed 378**: **87.06% (5,093/5,850)** accuracy, **0.7903** macro F1 and **0.7817** balanced accuracy. The complete nine-recipe experiment still selects **T5-a01** by mean validation accuracy. Neither the study's frozen reports nor its selection has been rewritten. The detailed T5-a01 error analysis below remains labelled as the study-selected model.

| Seed-378 comparison with the same teacher | Teacher wrong, student correct | Teacher correct, student wrong | Net correct | Accuracy gain |
| --- | --- | --- | --- | --- |
| **Retained deployment, T5-a05** | **473** | **267** | **206** | **+3.52 pp** |
| Study-selected T5-a01 | 465 | 339 | 126 | +2.15 pp |
| Pure-soft control, T5-a00 | 455 | 345 | 110 | +1.88 pp |

The T5-a05 track-resampling interval is **[+1.15, +5.79] pp** on the already observed reference collection. This is a descriptive interval over observed physical-sign tracks, not evidence of independent confirmation. Counts come from the saved prediction comparison in [`summary.json`](../reports/full-grid/teacher-student-analysis/summary.json), including `grid_T5-a05_seed_378-vs-teacher`; the compact [deployment comparison](../reports/deployment/teacher-student-comparison.json) records the representative model's role.

**Conclusion:** the distilled students have higher observed reference accuracy in these runs. Their objective fits softened class probabilities rather than enforcing the teacher's top-class decision, so different decisions and teacher-surpassing accuracy are possible. All three pure-soft T5-a00 controls exceeded the teacher, so the extra hard-label CE term cannot be the sole explanation. The teacher and mixed student both have ground-truth training supervision; this is not a case in which only the student receives labels.

Teacher partial fine-tuning with frozen BN differs from the student's full compact-model adaptation and updating BN. Architecture, pretraining, dropout, training budgets, learning-rate histories and validation checkpoint choice also differ. These verified differences are plausible explanations for different predictions, but their causal contributions have not been isolated. The supported finding is higher observed reference accuracy in these runs, not proof that compression itself raises accuracy or that a student is always better than its teacher. The [deployment and project outcome guide](./deployment.md) combines these findings with cost, reproducibility and scope limits.

## Preserved study-selected T5-a01 analysis

The validation-selected recipe is **T5-a01**. This diagnostic reads saved predictions and training histories from the new joint freeze. It performs no model calls, retraining or reselection. Results describe 5,850 images and 195 observed physical-sign tracks on a previously observed reference set.

| Model | Correct / count | Accuracy | macro F1 | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Teacher | 4,887 / 5,850 | 83.54% | 0.7465 | 0.7548 |
| Selected student, seed 378 | 5,013 / 5,850 | 85.69% | 0.7719 | 0.7620 |

![Selected-student error differences and fixed-T5 controls](../assets/figures/teacher-student-analysis.png)

Selected student minus teacher macro F1 is **+0.0254**, and balanced-accuracy difference is **+0.72 pp**. The conditional teacher-advantage discussion is triggered by accuracy/correct count on this same reference set. Higher accuracy is not broadened into superiority on every metric or another collection.

## Differences established by saved predictions

| Seed | Teacher wrong, student correct | Teacher correct, student wrong | Net correct | Accuracy gain (pp) | Track resampling interval (pp) |
| --- | --- | --- | --- | --- | --- |
| 378 | 465 | 339 | 126 | +2.15 | [-0.14, +4.55] |
| 379 | 552 | 183 | 369 | +6.31 | [+3.90, +8.44] |
| 380 | 517 | 263 | 254 | +4.34 | [+1.79, +6.87] |

Of 465 teacher errors repaired by seed 378, **282 were already correct in the supervised baseline**. The student repaired **183 cases where both teacher and baseline were wrong**. All teacher-error repairs are not attributed to KD alone. Student recall is higher in 22 classes, teacher recall in 14, with 7 ties.

Cluster intervals are 95% percentile resampling intervals over whole observed sign tracks. They do not establish untouched confirmation, teacher-seed uncertainty, causal distillation effects or road-scene reliability. Intervals that contain zero remain reported unchanged.

## Matched controls for the selected T5 recipe

The selected **T5-a01** is paired directly with the same-seed supervised baseline and **T5-a00** pure-soft control. These are measured accuracy and macro F1 on the same reference collection. The later fixed-alpha-0.5 analysis is supplementary and does not substitute for the selected alpha 0.1 recipe.

| Seed | Hard CE accuracy | Hard CE macro F1 | Selected accuracy | Selected macro F1 | Pure KL accuracy | Pure KL macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 378 | 73.56% | 0.6253 | 85.69% | 0.7719 | 85.42% | 0.7684 |
| 379 | 80.75% | 0.7216 | 89.85% | 0.8111 | 90.03% | 0.8329 |
| 380 | 79.93% | 0.7118 | 87.88% | 0.8164 | 87.25% | 0.7876 |

| Paired contrast | Mean accuracy gain ± sample SD (pp) | Mean macro F1 gain |
| --- | --- | --- |
| Selected minus hard CE | +9.73 ± 2.16 | +0.1135 |
| Selected minus pure KL | +0.24 ± 0.41 | +0.0035 |

| Seed | Same realized epochs | Same realized LR history |
| --- | --- | --- |
| 378 | True | False |
| 379 | False | False |
| 380 | True | False |

Temperature five, shared initialization and maximum-budget policies are verified. Changing alpha from zero to 0.1 both adds hard-label CE and changes the KL coefficient from one to 0.9. Realized learning rates and validation checkpoint choice also respond to the objective, so this comparison does not isolate hard-label CE as the sole cause. Means and sample SDs describe the three development-selection repetitions.

## Supplementary fixed-T5 hard, mixed and pure-soft controls

Controls compare fixed temperature **5**, mixed **alpha 0.5**, pure-soft **alpha 0**, and supervised **alpha 1**. They can differ from the recipe selected from the nine-cell grid; their contrast is not automatically the selected recipe's effect. Alpha-zero runs are excluded from recipe selection.

Alpha is the hard-label CE mixing coefficient, not a guaranteed fraction of loss or gradients. Comparing alpha 0.5 with alpha zero both adds hard-label CE and changes the KL mixing coefficient from one to 0.5. That difference does not isolate the causal effect of hard-label CE alone.

| Seed | Hard CE, alpha 1 | Mixed, alpha 0.5 | Pure KL, alpha 0 |
| --- | --- | --- | --- |
| 378 | 73.56% | 87.06% | 85.42% |
| 379 | 80.75% | 89.88% | 90.03% |
| 380 | 79.93% | 85.59% | 87.25% |

Pure-soft students also exceeded teacher reference accuracy in **3 of three** seeds. Their teacher-surpassing outcomes cannot be explained solely by a hard-label CE term in the student objective. The teacher and validation checkpoint selection still use labels; this is not label-free learning.

| Paired contrast | Mean accuracy gain ± sample SD (pp) | Mean macro F1 gain |
| --- | --- | --- |
| Mixed minus hard | +9.43 ± 3.93 | +0.1096 |
| Mixed minus pure-soft | -0.06 ± 1.65 | -0.0004 |
| Pure-soft minus hard | +9.49 ± 2.28 | +0.1100 |

| Seed | Same realized epochs | Same realized LR history |
| --- | --- | --- |
| 378 | True | False |
| 379 | False | False |
| 380 | True | False |

Initialization, data, augmentation, class weights and maximum-budget rules are matched. Realized learning-rate schedules, early stopping and validation-selected epochs can depend on the objective. These are pipeline contrasts, not a causal decomposition of the KL term alone.

## Interpretation limits

Mixed KD optimizes hard-label CE and KL to softened teacher probabilities; it need not reproduce the teacher's argmax. Ground-truth learning, student architecture, augmentation and validation checkpoint choice are plausible sources of different predictions. Error differences and controls do not establish every proposed cause.

Training accuracy uses augmentation and train-mode dropout/normalization; validation uses clean inference-mode inputs. Their gap is not a same-condition overfitting measure. Saved confidence is the predicted-class maximum probability; true-label probability, per-example KL, entropy and training gradients cannot be reconstructed. Three student seeds sharing one teacher do not measure teacher-training-seed variation.

## Recipe facts established by the frozen code

Teacher and student both use the same training labels. Class weights are `N / (43 * n_class)` and apply to both teacher CE and student hard-label CE. Mixed students additionally optimize unweighted Teacher-to-Student KL with `T^2` compensation. The student is not the only model with ground-truth supervision. Sources: [data.py:242](../knowledge_distillation/data.py#L242), [teacher weighted CE](../knowledge_distillation/training.py#L159), [student weighted CE](../knowledge_distillation/training.py#L250), and [loss mixture](../knowledge_distillation/losses.py#L17).

| Verified property | Teacher | Student | Code evidence |
| --- | --- | --- | --- |
| Architecture and initialization | ImageNet MobileNetV2 plus a fresh 43-class head | Freshly initialized three-block separable MicroCNN | [models.py:20](../knowledge_distillation/models.py#L20), [training.py:147](../knowledge_distillation/training.py#L147) |
| Adaptation scope | Train head, then fine-tune up to the last 60 backbone layers | Optimize all trainable weights of the compact model | [models.py:69](../knowledge_distillation/models.py#L69), [losses.py:104](../knowledge_distillation/losses.py#L104) |
| Batch normalization | Freeze MobileNetV2 BN; call backbone in inference mode | Three BN layers update in student training mode | [models.py:62](../knowledge_distillation/models.py#L62), [models.py:80](../knowledge_distillation/models.py#L80), [losses.py:97](../knowledge_distillation/losses.py#L97) |
| Classifier dropout | 0.3 | 0.2 | [models.py:34](../knowledge_distillation/models.py#L34), [models.py:64](../knowledge_distillation/models.py#L64) |
| Maximum budget and initial learning rates | Adam: head 5 epochs at 0.001, fine-tuning 15 at 0.00001 | Adam: at most 30 epochs at 0.001 | [training.py:152](../knowledge_distillation/training.py#L152), [training.py:255](../knowledge_distillation/training.py#L255) |
| Augmentation rules | Brightness 0.2, contrast 0.2, zoom 0.05 | Same operations and configured strengths | [models.py:8](../knowledge_distillation/models.py#L8) |
| Input contract | RGB96 bilinear [0,1] external input, scaled to [-1,1] inside graph | RGB96 bilinear [0,1] input, clipped after augmentation | [data.py:292](../knowledge_distillation/data.py#L292), [models.py:55](../knowledge_distillation/models.py#L55), [losses.py:88](../knowledge_distillation/losses.py#L88) |

During distillation the teacher retains its saved weights and is called with `training=False`; teacher dropout and internal augmentation are not reactivated to add noise. The student is called in training mode. Shared augmentation rules do not imply identical realized teacher/student images across every seed, stage and epoch. Sources: [losses.py:86](../knowledge_distillation/losses.py#L86) and [teacher immutability](../knowledge_distillation/training.py#L246).

Both pipelines select the best validation-accuracy checkpoint and use patience five, while ReduceLROnPlateau responds to each model's validation CE history. Actual learning-rate paths, stopping epochs and selected epochs can differ. The control table reports realized agreement; matched policies are not described as identical realized paths. Source: [shared callbacks](../knowledge_distillation/training.py#L91).

Verified differences include pretraining and partial versus full adaptation, BN/dropout policies, optimization budgets and objectives. Their contribution to generalization or repairs remains a hypothesis to interpret alongside actual error differences and fixed-T5 controls. This comparison does not establish compression itself as the cause of accuracy gain, or hard-label CE as the sole cause of teacher-surpassing accuracy.

Evidence: [`reports/full-grid/teacher-student-analysis/summary.json`](../reports/full-grid/teacher-student-analysis/summary.json). See the [full-grid report](./full-grid.md) for the complete grid and public evaluation commands.
