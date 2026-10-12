"""Deterministic, track-disjoint GTSRB inventories and bounded image streaming.

Manifest creation deliberately does not inspect Test.csv or official test images.
The public dataset was used historically; ``locked_holdout`` only describes its
role in this new protocol, not an assertion that these images were never seen.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

SPLIT_NAMES = ("train", "validation", "locked_holdout")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_dataset_path(dataset_root: str | Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if relative.is_absolute() or not relative.parts or any(p in {"..", "."} for p in relative.parts):
        raise ValueError(f"Unsafe dataset-relative path: {relative_path!r}")
    if ":" in normalized:
        raise ValueError(f"Drive-qualified dataset path is forbidden: {relative_path!r}")
    root = Path(dataset_root).resolve()
    result = root.joinpath(*relative.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Dataset path or symlink leaves the dataset root")
    return result


def track_id_from_path(relative_path: str, label: int) -> str:
    """Handle Kaggle class_track_frame PNG names and official track_frame names.

    Unknown naming conventions are rejected instead of silently treating every
    image as a different physical sign, which would invalidate grouped splits.
    """
    stem = PurePosixPath(relative_path.replace("\\", "/")).stem
    parts = stem.split("_")
    if len(parts) == 3 and all(part.isdecimal() for part in parts):
        if int(parts[0]) != label:
            raise ValueError(f"Filename class disagrees with annotation: {relative_path}")
        track = int(parts[1])
    elif len(parts) == 2 and all(part.isdecimal() for part in parts):
        track = int(parts[0])
    else:
        raise ValueError(f"Cannot recover a physical-sign track from {relative_path!r}")
    return f"{label:02d}:{track:05d}"


def _training_entries(root: Path, num_classes: int) -> list[tuple[str, int]]:
    csv_path = root / "Train.csv"
    entries: list[tuple[str, int]] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"Path", "ClassId"}.issubset(reader.fieldnames or []):
                raise ValueError("Train.csv must contain Path and ClassId columns")
            for row in reader:
                relative = row["Path"].replace("\\", "/")
                if not PurePosixPath(relative).parts or PurePosixPath(relative).parts[0].lower() != "train":
                    raise ValueError("Train.csv points outside its Train image directory")
                entries.append((relative, int(row["ClassId"])))
    else:
        train_root = root / "Train"
        if not train_root.is_dir():
            raise FileNotFoundError("Expected Train.csv or Train/<class>/ image directories")
        for image in sorted(train_root.rglob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                entries.append((image.relative_to(root).as_posix(), int(image.parent.name)))
    if not entries:
        raise ValueError("Training inventory is empty")
    if any(label < 0 or label >= num_classes for _, label in entries):
        raise ValueError("Training label is outside the configured class range")
    if len({path for path, _ in entries}) != len(entries):
        raise ValueError("Duplicate training image paths")
    return sorted(entries, key=lambda row: (row[1], row[0]))


def build_split_manifest(
    dataset_root: str | Path,
    seed: int = 378,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    *,
    num_classes: int = 43,
    image_size: int = 96,
) -> dict[str, Any]:
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Three positive split ratios summing to one are required")
    root = Path(dataset_root).resolve()
    groups: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for relative, label in _training_entries(root, num_classes):
        image = safe_dataset_path(root, relative)
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Missing or unsupported training image: {relative}")
        track = track_id_from_path(relative, label)
        record = {
            "id": relative,
            "relative_path": relative,
            "label": label,
            "track_id": track,
            "sha256": file_sha256(image),
            "bytes": image.stat().st_size,
        }
        groups[label][track].append(record)
    if set(groups) != set(range(num_classes)):
        raise ValueError("Every configured class must be represented in the training inventory")
    records: list[dict[str, Any]] = []
    for label in sorted(groups):
        tracks = sorted(groups[label])
        if len(tracks) < 3:
            raise ValueError(f"Class {label} needs at least three independent sign tracks")
        random.Random(seed * 1_000_003 + label).shuffle(tracks)
        n_val = max(1, round(len(tracks) * ratios[1]))
        n_holdout = max(1, round(len(tracks) * ratios[2]))
        while n_val + n_holdout >= len(tracks):
            if n_val >= n_holdout and n_val > 1:
                n_val -= 1
            elif n_holdout > 1:
                n_holdout -= 1
            else:
                raise ValueError("Split allocation leaves no training track")
        boundaries = (len(tracks) - n_val - n_holdout, len(tracks) - n_holdout)
        for index, track in enumerate(tracks):
            split = SPLIT_NAMES[0 if index < boundaries[0] else 1 if index < boundaries[1] else 2]
            for record in groups[label][track]:
                records.append({**record, "split": split})
    records.sort(key=lambda record: record["id"])
    inventory = [{k: v for k, v in record.items() if k != "split"} for record in records]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "GTSRB Kaggle PNG inventory or equivalent official tracks",
        "seed": seed,
        "requested_track_ratios": dict(zip(SPLIT_NAMES, ratios)),
        "split_policy": "Class-stratified physical-sign tracks; per-class rounded counts; minimum one track per split",
        "historical_exposure": "Original project used all training images and official test labels; new models start fresh and current selection uses validation only.",
        "num_classes": num_classes,
        "image_size": image_size,
        "class_names": [str(label) for label in range(num_classes)],
        "records": records,
        "inventory_sha256": canonical_sha256(inventory),
        "split_ids": {name: [r["id"] for r in records if r["split"] == name] for name in SPLIT_NAMES},
        "counts": {name: sum(r["split"] == name for r in records) for name in SPLIT_NAMES},
        "track_counts": {name: len({r["track_id"] for r in records if r["split"] == name}) for name in SPLIT_NAMES},
        "training_csv_sha256": file_sha256(root / "Train.csv") if (root / "Train.csv").is_file() else None,
        "official_test_used_in_split_construction": False,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], dataset_root: str | Path | None = None, *, verify_files: bool = False
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    if manifest.get("manifest_sha256") != canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest checksum mismatch")
    if verify_files and dataset_root is None:
        raise ValueError("Dataset root is required for file verification")
    if verify_files and manifest.get("training_csv_sha256") is not None:
        csv_path = Path(dataset_root) / "Train.csv"
        if not csv_path.is_file() or file_sha256(csv_path) != manifest["training_csv_sha256"]:
            raise ValueError("Training CSV checksum mismatch")
    records = list(manifest.get("records", []))
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Empty inventory or duplicate image IDs")
    if len({r["relative_path"] for r in records}) != len(records):
        raise ValueError("Duplicate image paths")
    inventory = [{k: v for k, v in record.items() if k != "split"} for record in records]
    if manifest.get("inventory_sha256") != canonical_sha256(inventory):
        raise ValueError("Inventory checksum mismatch")
    track_splits: dict[str, str] = {}
    for record in records:
        if record["split"] not in SPLIT_NAMES:
            raise ValueError("Unknown split name")
        if not 0 <= record["label"] < manifest["num_classes"]:
            raise ValueError("Invalid record class")
        relative = record["relative_path"]
        if track_id_from_path(relative, record["label"]) != record["track_id"]:
            raise ValueError("Recorded sign track disagrees with filename")
        if record["id"] != relative:
            raise ValueError("Record ID must be its unique relative path")
        path = safe_dataset_path(dataset_root or Path.cwd(), relative)
        if record["track_id"] in track_splits and track_splits[record["track_id"]] != record["split"]:
            raise ValueError("Physical-sign track crosses split boundaries")
        track_splits[record["track_id"]] = record["split"]
        if verify_files:
            if not path.is_file() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                raise ValueError(f"Image integrity check failed: {relative}")
    for name in SPLIT_NAMES:
        selected = [r for r in records if r["split"] == name]
        if not selected or {r["label"] for r in selected} != set(range(manifest["num_classes"])):
            raise ValueError("Every split must contain every class")
        if [r["id"] for r in selected] != manifest["split_ids"][name]:
            raise ValueError("Split IDs disagree with inventory order")
        if len(selected) != manifest["counts"][name]:
            raise ValueError("Split counts disagree with records")
        if len({r["track_id"] for r in selected}) != manifest["track_counts"][name]:
            raise ValueError("Split track count disagrees with records")


def save_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    validate_manifest(manifest)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def records_for_split(manifest: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {name}")
    return [dict(record) for record in manifest["records"] if record["split"] == name]


def class_weights_for_records(records: Iterable[Mapping[str, Any]], num_classes: int = 43) -> dict[int, float]:
    counts = [0] * num_classes
    for record in records:
        counts[int(record["label"])] += 1
    if not all(counts):
        raise ValueError("Class weights require at least one training example in every class")
    total = sum(counts)
    return {label: total / (num_classes * count) for label, count in enumerate(counts)}


def make_dataset(
    records: Iterable[Mapping[str, Any]],
    dataset_root: str | Path,
    training: bool = False,
    seed: int = 378,
    batch_size: int = 64,
    *,
    image_size: int = 96,
    shuffle_buffer: int | None = None,
):
    """Create a fresh deterministic pipeline per run, without a full-RAM cache.

    Evaluation follows the supplied record order exactly. Recreating this
    pipeline with the same seed gives the same epoch shuffle sequence for each
    paired baseline/KD run. Only two decode workers and two prefetched batches
    are used, independent of dataset size. The default full shuffle stores only
    file paths and integer labels before decoding, not all float32 images. A
    small buffer on a class-sorted inventory would bias class order strongly.
    """
    import tensorflow as tf

    chosen = list(records)
    if not chosen or batch_size < 1 or image_size < 1:
        raise ValueError("Nonempty records and positive batch/image sizes are required")
    paths = [str(safe_dataset_path(dataset_root, r["relative_path"])) for r in chosen]
    labels = [int(record["label"]) for record in chosen]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        if shuffle_buffer is not None and shuffle_buffer < 1:
            raise ValueError("An explicit shuffle buffer must be positive")
        buffer_size = len(chosen) if shuffle_buffer is None else min(len(chosen), shuffle_buffer)
        dataset = dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)

    def read_ppm(path_bytes):
        import numpy as np
        from PIL import Image

        with Image.open(path_bytes.decode("utf-8")) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def decode(path, label):
        def native():
            return tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)

        def ppm():
            return tf.numpy_function(read_ppm, [path], tf.uint8)

        image = tf.cond(tf.strings.regex_full_match(tf.strings.lower(path), ".*\\.ppm"), ppm, native)
        image.set_shape((None, None, 3))
        image = tf.image.resize(tf.cast(image, tf.float32), [image_size, image_size], method="bilinear") / 255.0
        return image, tf.cast(label, tf.int32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return dataset.map(decode, num_parallel_calls=2, deterministic=True).batch(batch_size).with_options(options).prefetch(2)
