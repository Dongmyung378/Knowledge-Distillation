"""Frozen, streamed evaluation with explicit dataset and artifact provenance."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import NUM_CLASSES, file_sha256, load_class_map, load_model, probabilities, tensorflow

LOCKED_SPLITS = {"locked_holdout", "official_test"}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_manifest(value: str | Path | dict) -> dict:
    result = value if isinstance(value, dict) else json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(result.get("records"), list):
        raise ValueError("Manifest must contain a records list.")
    return result


def safe_record_path(dataset_root: str | Path, record: dict) -> Path:
    root = Path(dataset_root).resolve(strict=True)
    relative = Path(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset records must use safe relative paths.")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Dataset path is outside its root or is not a file.")
    return path


def records_for_evaluation(manifest: dict, split: str) -> list[dict]:
    records = [r for r in manifest["records"] if r["split"] == split]
    if not records:
        raise ValueError(f"Manifest contains no records for {split!r}.")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Evaluation record IDs must be unique.")
    if any(not isinstance(r["label"], int) or not 0 <= r["label"] < NUM_CLASSES for r in records):
        raise ValueError("Evaluation labels must be integer class IDs 0 through 42.")
    return records


def verify_record_files(records: list[dict], dataset_root: str | Path) -> None:
    for record in records:
        path = safe_record_path(dataset_root, record)
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Dataset file hash mismatch for record {record['id']}.")


def _source_hashes(root: Path) -> dict[str, str]:
    paths = sorted((root / "knowledge_distillation").glob("*.py"))
    paths += [root / name for name in ("requirements.txt", "requirements-runtime.txt", "pyproject.toml")
              if (root / name).is_file()]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}


def freeze_evaluation(models: dict[str, str | Path], manifest: str | Path | dict,
                      config: dict, output_path: str | Path,
                      source_root: str | Path | None = None) -> dict:
    """Freeze selection using validation only, before reading locked labels.

    The freeze does not certify absence of earlier human exposure. It prevents
    accidental artifact, configuration, source, or dataset changes after this
    selected experiment is frozen.
    """
    manifest = read_manifest(manifest)
    if not models or not isinstance(config, dict):
        raise ValueError("At least one selected model and its configuration are required.")
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    model_bindings = {}
    for name, value in sorted(models.items()):
        path = Path(value).resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in {".keras", ".tflite"}:
            raise ValueError("Freeze models must be saved .keras or .tflite artifacts.")
        model_bindings[name] = {"filename": path.name, "sha256": file_sha256(path),
                                "bytes": path.stat().st_size}
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "selection_split": "validation", "models": model_bindings,
               "manifest_sha256": canonical_sha256(manifest),
               "inventory_sha256": manifest.get("inventory_sha256"),
               "class_names": load_class_map(manifest.get("class_names")),
               "split_counts": dict(sorted(Counter(r["split"] for r in manifest["records"]).items())),
               "config": config, "config_sha256": canonical_sha256(config),
               "source_sha256": _source_hashes(root),
               "evaluation_policy": {"outputs": "43 logits", "decision": "argmax",
                                     "external_input_range": [0, 1], "image_size": [96, 96],
                                     "official_test_previously_used_for_selection": True}}
    payload["freeze_sha256"] = canonical_sha256(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("A freeze is immutable; choose a new output path.")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def validate_freeze(freeze_path: str | Path, manifest: str | Path | dict,
                    model_path: str | Path | None = None,
                    source_root: str | Path | None = None) -> dict:
    manifest = read_manifest(manifest)
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    claimed = freeze.get("freeze_sha256")
    if claimed != canonical_sha256({k: v for k, v in freeze.items() if k != "freeze_sha256"}):
        raise ValueError("Freeze content checksum mismatch.")
    if freeze.get("config_sha256") != canonical_sha256(freeze.get("config")):
        raise ValueError("Frozen configuration checksum mismatch.")
    if manifest.get("external_audit"):
        if manifest.get("parent_manifest_sha256") != freeze["manifest_sha256"]:
            raise ValueError("External audit does not reference the frozen training manifest.")
        if manifest.get("freeze_sha256") != claimed:
            raise ValueError("External audit references a different freeze.")
        content = {k: v for k, v in manifest.items() if k != "external_manifest_sha256"}
        if manifest.get("external_manifest_sha256") != canonical_sha256(content):
            raise ValueError("External manifest checksum mismatch.")
    elif canonical_sha256(manifest) != freeze["manifest_sha256"]:
        raise ValueError("Manifest differs from the frozen dataset inventory/splits.")
    if load_class_map(manifest.get("class_names")) != freeze["class_names"]:
        raise ValueError("Class IDs differ from the frozen class map.")
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    if _source_hashes(root) != freeze["source_sha256"]:
        raise ValueError("Source or dependency files changed since model selection was frozen.")
    if model_path is not None:
        path = Path(model_path)
        sha = file_sha256(path)
        if not any(binding["sha256"] == sha and binding["bytes"] == path.stat().st_size
                   for binding in freeze["models"].values()):
            raise ValueError("Model artifact is not part of the frozen evaluation.")
    return freeze


def build_official_test_manifest(dataset_root: str | Path, training_manifest: str | Path | dict,
                                 freeze_path: str | Path, csv_path: str | Path | None = None) -> dict:
    """Read official test labels only after a valid selected-model freeze exists."""
    original = read_manifest(training_manifest)
    freeze = validate_freeze(freeze_path, original)
    root = Path(dataset_root).resolve(strict=True)
    csv_file = Path(csv_path) if csv_path else root / "Test.csv"
    records = []
    with csv_file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Path", "ClassId"}.issubset(reader.fieldnames or []):
            raise ValueError("Official Test.csv must provide Path and ClassId columns.")
        for row in reader:
            relative = row["Path"].replace("\\", "/")
            record = {"id": "official_test:" + relative, "relative_path": relative,
                      "label": int(row["ClassId"]), "track_id": "official_test:" + relative,
                      "split": "official_test"}
            path = safe_record_path(root, record)
            record.update(sha256=file_sha256(path), bytes=path.stat().st_size)
            records.append(record)
    manifest = {"schema_version": 1, "external_audit": True, "records": records,
                "num_classes": NUM_CLASSES, "image_size": 96,
                "class_names": freeze["class_names"],
                "parent_manifest_sha256": freeze["manifest_sha256"],
                "freeze_sha256": freeze["freeze_sha256"],
                "official_test_previously_used_for_selection": True,
                "csv_sha256": file_sha256(csv_file), "inventory_sha256": canonical_sha256(records)}
    records_for_evaluation(manifest, "official_test")
    manifest["external_manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def classification_metrics(labels, predictions, num_classes: int = NUM_CLASSES) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("Labels/predictions must be nonempty aligned one-dimensional arrays.")
    if np.any(labels < 0) or np.any(labels >= num_classes) or np.any(predictions < 0) or np.any(predictions >= num_classes):
        raise ValueError("Class IDs are outside the confusion matrix.")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    tp = np.diag(confusion).astype(float)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {"count": int(len(labels)), "correct": int(tp.sum()),
            "accuracy": float(tp.sum() / len(labels)), "accuracy_percent": float(100 * tp.sum() / len(labels)),
            "macro_f1": float(f1.mean()), "macro_f1_policy": f"All {num_classes} classes; absent class F1 = 0.",
            "balanced_accuracy": float(recall[present].mean()),
            "balanced_accuracy_policy": "Mean recall over classes with evaluation support.",
            "per_class": [{"class_id": i, "support": int(support[i]), "predicted_count": int(predicted[i]),
                           "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
                          for i in range(num_classes)], "confusion_matrix": confusion.tolist()}


def _evaluate(model_path, manifest, dataset_root, split, output_dir,
              freeze_path=None, batch_size=64, threads=1) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = read_manifest(manifest)
    freeze = None
    if split in LOCKED_SPLITS and freeze_path is None:
        raise ValueError("Locked evaluation requires an immutable selection freeze.")
    if freeze_path is not None:
        freeze = validate_freeze(freeze_path, manifest, model_path)
    records = records_for_evaluation(manifest, split)
    verify_record_files(records, dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    from .data import make_dataset
    begin = time.perf_counter()
    runner = load_model(model_path, threads=threads)
    load_seconds = time.perf_counter() - begin
    labels, predictions = [], []
    offset = 0
    start = time.perf_counter()
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for images, batch_labels in make_dataset(records, dataset_root, training=False,
                                                 seed=378, batch_size=batch_size):
            values = images.numpy() if hasattr(images, "numpy") else images
            target = np.asarray(batch_labels, dtype=np.int64)
            scores = probabilities(runner.predict(values))
            pred = scores.argmax(axis=1)
            if not np.array_equal(target, [r["label"] for r in records[offset:offset + len(target)]]):
                raise ValueError("Dataset iteration order differs from frozen record order.")
            for j, label in enumerate(target):
                item = {"id": records[offset + j]["id"], "label": int(label),
                        "prediction": int(pred[j]), "confidence": float(scores[j, pred[j]])}
                stream.write(json.dumps(item, sort_keys=True) + "\n")
            labels.extend(target.tolist())
            predictions.extend(pred.tolist())
            offset += len(target)
    if offset != len(records):
        raise ValueError("Evaluation did not consume every frozen record.")
    metric = classification_metrics(labels, predictions)
    confusion = metric.pop("confusion_matrix")
    tf = tensorflow()
    metric.update(schema_version=1, split=split, model_filename=Path(model_path).name,
                  model_sha256=file_sha256(model_path), manifest_sha256=canonical_sha256(manifest),
                  training_manifest_sha256=manifest.get("parent_manifest_sha256", canonical_sha256(manifest)),
                  inventory_sha256=manifest.get("inventory_sha256"),
                  evaluation_records_sha256=canonical_sha256(records),
                  freeze_sha256=freeze["freeze_sha256"] if freeze else None,
                  class_names=load_class_map(manifest.get("class_names")),
                  preprocessing={"image_size": [96, 96], "channels": "RGB", "range": [0, 1],
                                 "resize": "bilinear", "teacher_scaling": "Inside saved graph"},
                  model_parameters=runner.parameters, model_load_seconds=load_seconds,
                  evaluation_seconds=time.perf_counter() - start,
                  tensorflow_version=tf.__version__, python_version=platform.python_version(),
                  visible_devices=[d.device_type for d in tf.config.get_visible_devices()],
                  official_test_previously_used_for_selection=split == "official_test",
                  selection_permitted=split not in LOCKED_SPLITS)
    (output / "metrics.json").write_text(json.dumps(metric, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "confusion.json").write_text(json.dumps({"rows": "true_class_id", "columns": "predicted_class_id",
        "class_names": metric["class_names"], "matrix": confusion}, indent=2) + "\n", encoding="utf-8")
    return metric


def evaluate_keras(model_path, manifest, dataset_root, split, output_dir,
                   freeze_path=None, batch_size=64, threads=1):
    if Path(model_path).suffix.lower() != ".keras":
        raise ValueError("evaluate_keras requires a .keras artifact.")
    return _evaluate(model_path, manifest, dataset_root, split, output_dir, freeze_path, batch_size, threads)


def evaluate_tflite(model_path, manifest, dataset_root, split, output_dir,
                    freeze_path=None, batch_size=64, threads=1):
    if Path(model_path).suffix.lower() != ".tflite":
        raise ValueError("evaluate_tflite requires a .tflite artifact.")
    return _evaluate(model_path, manifest, dataset_root, split, output_dir, freeze_path, batch_size, threads)
