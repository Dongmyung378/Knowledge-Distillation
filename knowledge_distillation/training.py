"""Kaggle-only real training, best-model artifacts and paired student controls."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

from .data import canonical_sha256, class_weights_for_records, file_sha256, make_dataset, records_for_split
from .experiments import source_inventory
from .losses import Distiller
from .models import build_augmentation, build_student, build_teacher, configure_teacher_stage, weights_sha256


def write_json(value: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def require_kaggle_gpu() -> None:
    """Real project training is explicitly restricted to a Kaggle GPU kernel."""
    in_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").is_dir()
    if not in_kaggle:
        raise RuntimeError("Real training is allowed only in Kaggle. Local use is CPU inference/evaluation or tiny correctness tests.")
    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("Enable a Kaggle GPU accelerator before starting a training experiment")
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            # A caller may have initialized TensorFlow already; training still
            # remains within the explicitly checked Kaggle environment.
            pass


def _check_execution(execution_mode: str, record_count: int, max_epochs: int) -> str:
    if execution_mode == "kaggle":
        require_kaggle_gpu()
        return "/GPU:0"
    if execution_mode == "cpu-smoke" and record_count <= 32 and max_epochs <= 2:
        return "/CPU:0"
    raise RuntimeError("Only Kaggle GPU training or <=32-image, <=2-epoch CPU correctness tests are allowed")


def _prepare_run(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty experiment directory: {path.name}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _configure_seed(seed: int, deterministic: bool = True) -> None:
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(seed)
    tf.keras.mixed_precision.set_global_policy("float32")
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def _runtime() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": getattr(tf.keras, "__version__", "unknown"),
        "platform": platform.system(),
        "training_location": "kaggle" if (os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir()) else "cpu-smoke",
        "gpu_names": [tf.config.experimental.get_device_details(device).get("device_name", device.name) for device in tf.config.list_physical_devices("GPU")],
    }


def _compile_classifier(model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        jit_compile=False,
    )


def _callbacks(checkpoint: Path, *, patience: int, student: bool = False):
    checkpoint_callback = BestStudentCheckpoint(checkpoint) if student else tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint), monitor="val_accuracy", mode="max", save_best_only=True, verbose=0
    )
    return [
        checkpoint_callback,
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


class BestStudentCheckpoint(tf.keras.callbacks.Callback):
    """Save a plain student, never the custom training wrapper or its teacher."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)
        self.best = float("-inf")
        self.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        score = (logs or {}).get("val_accuracy")
        if score is not None and float(score) > self.best:
            self.best = float(score)
            self.best_epoch = epoch + 1
            temporary = self.path.with_name(self.path.stem + ".tmp.keras")
            self.model.student.save(temporary)
            temporary.replace(self.path)


def train_teacher(
    dataset_root: str | Path,
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 378,
    config: dict[str, Any] | None = None,
    execution_mode: str = "kaggle",
) -> dict[str, Any]:
    config = dict(config or {})
    train_records = records_for_split(manifest, "train")
    validation_records = records_for_split(manifest, "validation")
    head_epochs = int(config.get("head_epochs", 5))
    fine_epochs = int(config.get("fine_tune_epochs", 15))
    if min(head_epochs, fine_epochs) < 1:
        raise ValueError("Both teacher stages require at least one epoch")
    device = _check_execution(execution_mode, len(train_records) + len(validation_records), head_epochs + fine_epochs)
    output = _prepare_run(output_dir)
    source_snapshot = source_inventory()
    _configure_seed(seed, bool(config.get("deterministic", True)))
    variant = config.get("variant", "frozen_bn")
    batch_size = int(config.get("batch_size", 64))
    model = None
    started = time.perf_counter()
    with tf.device(device):
        model = build_teacher(
            manifest["image_size"], manifest["num_classes"], variant=variant, seed=seed,
            weights="imagenet" if execution_mode == "kaggle" else None,
        )
        stage_results = []
        for stage, epochs, learning_rate in [
            ("head", head_epochs, float(config.get("head_learning_rate", 1e-3))),
            ("fine_tune", fine_epochs, float(config.get("fine_tune_learning_rate", 1e-5))),
        ]:
            configure_teacher_stage(model, stage, variant=variant, unfreeze_last=int(config.get("unfreeze_last", 60)))
            _compile_classifier(model, learning_rate)
            checkpoint = output / f"{stage}-best.keras"
            history = model.fit(
                make_dataset(train_records, dataset_root, training=True, seed=seed, batch_size=batch_size, image_size=manifest["image_size"]),
                validation_data=make_dataset(validation_records, dataset_root, batch_size=batch_size, image_size=manifest["image_size"]),
                class_weight=class_weights_for_records(train_records, manifest["num_classes"]),
                epochs=epochs,
                callbacks=_callbacks(checkpoint, patience=int(config.get("patience", 5))),
                verbose=int(config.get("verbose", 2)),
            )
            write_json(history.history, output / f"{stage}-history.json")
            if not checkpoint.is_file():
                raise RuntimeError("Teacher produced no finite best checkpoint")
            model = tf.keras.models.load_model(checkpoint, compile=False, safe_mode=True)
            _compile_classifier(model, learning_rate)
            score = model.evaluate(
                make_dataset(validation_records, dataset_root, batch_size=batch_size, image_size=manifest["image_size"]),
                verbose=0, return_dict=True,
            )
            stage_results.append({
                "stage": stage, "epochs_run": len(history.history["val_accuracy"]),
                "best_epoch": 1 + max(range(len(history.history["val_accuracy"])), key=lambda i: history.history["val_accuracy"][i]),
                "validation_accuracy": float(score["accuracy"]), "validation_ce": float(score["loss"]),
                "checkpoint": checkpoint.name, "checkpoint_sha256": file_sha256(checkpoint),
            })
        chosen = max(stage_results, key=lambda item: (item["validation_accuracy"], -item["validation_ce"]))
        final_path = output / "teacher.keras"
        model = tf.keras.models.load_model(output / chosen["checkpoint"], compile=False, safe_mode=True)
        # Deployment comparisons use inference artifacts without Adam state.
        # Compiled stage checkpoints remain available as training provenance.
        model.save(final_path)
        if source_inventory() != source_snapshot:
            raise RuntimeError("Training source changed during the teacher run")
        report = {
            "status": "completed", "kind": "teacher", "seed": seed,
            "configuration": config, "configuration_sha256": canonical_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"], "inventory_sha256": manifest["inventory_sha256"],
            "source_sha256": source_snapshot,
            "stages": stage_results, "selected_stage": chosen["stage"],
            "validation_accuracy": chosen["validation_accuracy"], "validation_ce": chosen["validation_ce"],
            "model_filename": final_path.name, "model_sha256": file_sha256(final_path),
            "weights_sha256": weights_sha256(model), "parameter_count": model.count_params(),
            "seconds": time.perf_counter() - started, "runtime": _runtime(),
            "selection_split": "validation", "locked_holdout_evaluated": False, "official_test_evaluated": False,
            "initialization": "Fresh ImageNet MobileNetV2 and fresh class head; historical GTSRB project checkpoints are not reused",
        }
    write_json(report, output / "training-report.json")
    del model
    gc.collect()
    return {"model_path": str(final_path.resolve()), "report_path": str((output / "training-report.json").resolve()), "report": report}


def train_student(
    dataset_root: str | Path,
    manifest: dict[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 378,
    config: dict[str, Any] | None = None,
    teacher_path: str | Path | None = None,
    initial_weights_path: str | Path | None = None,
    execution_mode: str = "kaggle",
) -> dict[str, Any]:
    config = dict(config or {})
    train_records = records_for_split(manifest, "train")
    validation_records = records_for_split(manifest, "validation")
    epochs = int(config.get("max_epochs", 30))
    if epochs < 1 or epochs > 30:
        raise ValueError("Student budget is one to thirty epochs")
    device = _check_execution(execution_mode, len(train_records) + len(validation_records), epochs)
    output = _prepare_run(output_dir)
    source_snapshot = source_inventory()
    _configure_seed(seed, bool(config.get("deterministic", True)))
    alpha = float(config.get("alpha", 1.0))
    temperature = float(config.get("temperature", 5.0))
    batch_size = int(config.get("batch_size", 64))
    started = time.perf_counter()
    with tf.device(device):
        student = build_student(manifest["image_size"], manifest["num_classes"])
        if initial_weights_path is not None:
            student.load_weights(initial_weights_path)
        initial_file_sha = file_sha256(initial_weights_path) if initial_weights_path is not None else None
        initial_sha = weights_sha256(student)
        teacher = None
        teacher_sha_before = None
        teacher_file_sha_before = None
        if alpha < 1.0:
            if teacher_path is None:
                raise ValueError("A saved teacher is required for distillation")
            teacher_file_sha_before = file_sha256(teacher_path)
            teacher = tf.keras.models.load_model(teacher_path, compile=False, safe_mode=True)
            teacher.trainable = False
            teacher_sha_before = weights_sha256(teacher)
        trainer = Distiller(
            student, teacher, alpha=alpha, temperature=temperature,
            class_weights=class_weights_for_records(train_records, manifest["num_classes"]),
            augmentation=build_augmentation(seed) if config.get("augmentation", True) else None,
        )
        trainer.compile(optimizer=tf.keras.optimizers.Adam(float(config.get("learning_rate", 1e-3))), jit_compile=False)
        train_dataset = make_dataset(train_records, dataset_root, training=True, seed=seed, batch_size=batch_size, image_size=manifest["image_size"])
        validation_dataset = make_dataset(validation_records, dataset_root, batch_size=batch_size, image_size=manifest["image_size"])
        checkpoint = output / "student.keras"
        history = trainer.fit(
            train_dataset, validation_data=validation_dataset, epochs=epochs,
            callbacks=_callbacks(checkpoint, patience=int(config.get("patience", 5)), student=True),
            verbose=int(config.get("verbose", 2)),
        )
        if not checkpoint.is_file():
            raise RuntimeError("Student produced no finite best checkpoint")
        teacher_sha_after = weights_sha256(teacher) if teacher is not None else None
        if teacher_sha_before != teacher_sha_after:
            raise AssertionError("Teacher weights changed during distillation")
        if teacher_file_sha_before is not None and file_sha256(teacher_path) != teacher_file_sha_before:
            raise AssertionError("Teacher artifact changed during distillation")
        if initial_file_sha is not None and file_sha256(initial_weights_path) != initial_file_sha:
            raise AssertionError("Shared initialization artifact changed during student training")
        restored = tf.keras.models.load_model(checkpoint, compile=False, safe_mode=True)
        _compile_classifier(restored, float(config.get("learning_rate", 1e-3)))
        scores = restored.evaluate(validation_dataset, return_dict=True, verbose=0)
        best_epoch = 1 + max(range(len(history.history["val_accuracy"])), key=lambda i: history.history["val_accuracy"][i])
        best_log_score = float(history.history["val_accuracy"][best_epoch - 1])
        if abs(float(scores["accuracy"]) - best_log_score) > 1e-6:
            raise AssertionError("Reloaded checkpoint differs from the selected validation epoch")
        if source_inventory() != source_snapshot:
            raise RuntimeError("Training source changed during the student run")
        report = {
            "status": "completed", "kind": "baseline" if alpha == 1 else "distilled", "seed": seed,
            "configuration": config, "configuration_sha256": canonical_sha256(config),
            "manifest_sha256": manifest["manifest_sha256"], "inventory_sha256": manifest["inventory_sha256"],
            "source_sha256": source_snapshot,
            "initial_weights_sha256": initial_sha,
            "initial_weights_file_sha256": initial_file_sha,
            "teacher_file_sha256": teacher_file_sha_before,
            "teacher_weights_before_sha256": teacher_sha_before, "teacher_weights_after_sha256": teacher_sha_after,
            "teacher_immutable": teacher_sha_before == teacher_sha_after,
            "epochs_run": len(history.history["val_accuracy"]), "best_epoch": best_epoch,
            "validation_accuracy": float(scores["accuracy"]), "validation_ce": float(scores["loss"]),
            "validation_metrics_reloaded_from_checkpoint": True,
            "model_filename": checkpoint.name, "model_sha256": file_sha256(checkpoint),
            "weights_sha256": weights_sha256(restored), "parameter_count": restored.count_params(),
            "loss_policy": "Class-weighted hard CE (batch-size denominator); unweighted Teacher-to-Student KL; T squared compensation; validation CE and accuracy unweighted",
            "validation_loss_policy": "Unweighted hard-label cross entropy, independent of alpha and temperature",
            "pairing_policy": "Fresh same-seed pipeline each run, explicit shared initial weights, same augmentation and maximum epoch/stopping budget; realized stopping epochs may differ",
            "seconds": time.perf_counter() - started, "runtime": _runtime(),
            "selection_split": "validation", "locked_holdout_evaluated": False, "official_test_evaluated": False,
        }
        write_json(history.history, output / "history.json")
    write_json(report, output / "training-report.json")
    del trainer, student, teacher, restored
    gc.collect()
    return {"model_path": str(checkpoint.resolve()), "report_path": str((output / "training-report.json").resolve()), "report": report}
