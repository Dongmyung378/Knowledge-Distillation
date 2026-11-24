"""Deterministic driving segmentation manifests, integrity checks, and PyTorch datasets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from .taxonomy import (
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    map_mask_to_train_ids,
)

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "locked_eval", "zero_shot_eval")
IMAGE_EXTENSIONS: set[str] = {".png", ".jpg", ".jpeg"}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_dataset_path(dataset_root: str | Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    rel = PurePosixPath(normalized)
    if rel.is_absolute() or not rel.parts or any(part in {"..", "."} for part in rel.parts):
        raise ValueError(f"Unsafe relative path: {relative_path!r}")
    if ":" in normalized:
        raise ValueError(f"Drive-qualified relative path is forbidden: {relative_path!r}")
    root = Path(dataset_root).resolve()
    resolved = root.joinpath(*rel.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative_path!r}")
    return resolved


def _load_bdd100k_attributes(root: Path) -> dict[str, dict[str, str]]:
    """Optionally read BDD100K image attribute JSONs for weather/timeofday slicing."""
    attrs: dict[str, dict[str, str]] = {}
    candidate_jsons = [
        root / "labels" / "bdd100k_labels_images_train.json",
        root / "labels" / "bdd100k_labels_images_val.json",
        root / "labels" / "det_20" / "det_train.json",
        root / "labels" / "det_20" / "det_val.json",
        root / "labels" / "sem_seg" / "polygons" / "sem_seg_train.json",
        root / "labels" / "sem_seg" / "polygons" / "sem_seg_val.json",
    ]
    for path in candidate_jsons:
        if not path.is_file():
            continue
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            name = item.get("name")
            raw_attr = item.get("attributes")
            if isinstance(name, str) and isinstance(raw_attr, dict):
                stem = Path(name).stem
                attrs[stem] = {
                    "weather": str(raw_attr.get("weather", "unknown")).lower(),
                    "timeofday": str(raw_attr.get("timeofday", "unknown")).lower(),
                    "scene": str(raw_attr.get("scene", "unknown")).lower(),
                }
    return attrs


def _is_grayscale_mask_file(path: Path) -> bool:
    name = path.name.lower()
    if "color" in name or "colormaps" in str(path.parent).lower() or "color_labels" in str(path.parent).lower():
        return False
    try:
        with Image.open(path) as im:
            return im.mode in {"L", "P", "I", "I;16"}
    except Exception:
        return False


def _canonical_bdd_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_train_id", "_trainid", "_mask", "_label"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_bdd100k_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_candidates = [
        root / "images" / "10k" / split,
        root / "bdd100k" / "images" / "10k" / split,
        root / "data" / "images_10k" / "10k" / split,
        root / "data" / "images_10k" / split,
        root / "images_10k" / "10k" / split,
        root / "images_10k" / split,
        root / "seg" / "images" / split,
        root / "images" / split,
    ]
    mask_candidates = [
        root / "labels" / "sem_seg" / "masks" / split,
        root / "bdd100k" / "labels" / "sem_seg" / "masks" / split,
        root / "data" / "labels_10k" / "10k" / split,
        root / "data" / "labels_10k" / split,
        root / "data" / "sem_seg" / "masks" / split,
        root / "labels" / "10k" / split,
        root / "labels_10k" / "10k" / split,
        root / "labels_10k" / split,
        root / "masks" / split,
        root / "seg" / "labels" / split,
        root / "labels" / split,
    ]
    img_dir = next((p for p in image_candidates if p.is_dir()), None)
    mask_dir = next((p for p in mask_candidates if p.is_dir()), None)

    if img_dir is None or mask_dir is None:
        # Fallback: scan subdirectories named <split> and pair RGB images with grayscale masks.
        split_dirs = [p for p in sorted(root.rglob(split)) if p.is_dir()]
        jpg_dirs = [d for d in split_dirs if any(f.suffix.lower() in {".jpg", ".jpeg"} for f in d.iterdir() if f.is_file())]
        png_mask_dirs = []
        for d in split_dirs:
            if "color" in str(d).lower():
                continue
            pngs = [f for f in sorted(d.glob("*.png")) if "color" not in f.name.lower()]
            if pngs and _is_grayscale_mask_file(pngs[0]):
                png_mask_dirs.append(d)
        if img_dir is None and jpg_dirs:
            img_dir = jpg_dirs[0]
        if img_dir is None:
            # Support PNG RGB image directories if JPGs are not present.
            for d in split_dirs:
                if d not in png_mask_dirs and any(f.suffix.lower() == ".png" for f in d.iterdir() if f.is_file()):
                    img_dir = d
                    break
        if mask_dir is None and png_mask_dirs:
            mask_dir = png_mask_dirs[0]

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"Could not find BDD100K {split} images/masks under {root}. "
            "Expected images/10k/<split> and labels/sem_seg/masks/<split> (or HuggingFace mirror layout)."
        )
    return img_dir, mask_dir


def build_bdd100k_manifest(
    dataset_root: str | Path,
    *,
    seed: int = 378,
    val_ratio: float = 0.15,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Build a deterministic manifest for BDD100K semantic segmentation (19 classes).

    Official ``train`` (7,000 images) is stratified by timeofday/weather into
    ``train`` and ``validation`` for hyperparameter/model selection.
    Official ``val`` (1,000 images) is preserved untouched as ``locked_eval``.
    """
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val_ratio must lie in (0, 0.5)")
    root = Path(dataset_root).resolve()
    attrs_by_stem = _load_bdd100k_attributes(root)

    def collect_pairs(official_split: str) -> list[dict[str, Any]]:
        img_dir, mask_dir = _find_bdd100k_dirs(root, official_split)
        mask_lookup: dict[str, Path] = {}
        for m in sorted(mask_dir.glob("*.png")):
            if "color" in m.name.lower():
                continue
            stem = _canonical_bdd_stem(m)
            mask_lookup[stem] = m
        items: list[dict[str, Any]] = []
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask = mask_lookup.get(img.stem)
            if mask is None:
                continue
            attr = attrs_by_stem.get(
                img.stem, {"weather": "unknown", "timeofday": "unknown", "scene": "unknown"}
            )
            rel_img = img.relative_to(root).as_posix()
            rel_mask = mask.relative_to(root).as_posix()
            rec: dict[str, Any] = {
                "id": f"bdd100k:{official_split}:{img.stem}",
                "image_path": rel_img,
                "mask_path": rel_mask,
                "label_format": "train_id",
                "group_id": f"bdd100k:{img.stem}",
                "official_split": official_split,
                "attributes": attr,
                "image_bytes": img.stat().st_size,
                "mask_bytes": mask.stat().st_size,
            }
            if hash_files:
                rec["image_sha256"] = file_sha256(img)
                rec["mask_sha256"] = file_sha256(mask)
            items.append(rec)
        if not items:
            raise ValueError(f"No matched BDD100K image/mask pairs found for split {official_split!r}")
        return items

    official_train = collect_pairs("train")
    official_val = collect_pairs("val")

    # Stratified split of official train into train + validation by (timeofday, weather).
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in official_train:
        key = (rec["attributes"]["timeofday"], rec["attributes"]["weather"])
        buckets[key].append(rec)

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda r: r["id"])
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * val_ratio)) if len(bucket) >= 2 else 0
        for idx, rec in enumerate(bucket):
            split = "validation" if idx < n_val else "train"
            records.append({**rec, "split": split})

    for rec in official_val:
        records.append({**rec, "split": "locked_eval"})

    return _finalize_manifest(
        dataset_name="bdd100k_sem_seg",
        split_policy="Official train stratified into train/validation by (timeofday, weather); official val locked as locked_eval.",
        seed=seed,
        records=records,
        hash_files=hash_files,
    )


def build_cityscapes_manifest(
    dataset_root: str | Path,
    *,
    seed: int = 378,
    val_ratio: float = 0.17,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Build a city-disjoint manifest for Cityscapes (19 classes).

    Official ``train`` cities (18 cities) are partitioned by whole city into
    ``train`` and ``validation`` so no two frames from the same city cross
    the train/validation boundary. Official ``val`` (3 cities, 500 images) is
    preserved untouched as ``locked_eval``.
    """
    root = Path(dataset_root).resolve()
    img_root = root / "leftImg8bit"
    gt_root = root / "gtFine"
    if not img_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError(
            f"Expected leftImg8bit/ and gtFine/ under {root}."
        )

    def collect_city_split(official_split: str) -> dict[str, list[dict[str, Any]]]:
        split_img = img_root / official_split
        split_gt = gt_root / official_split
        if not split_img.is_dir() or not split_gt.is_dir():
            raise FileNotFoundError(f"Missing Cityscapes {official_split} directories under {root}")
        by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for img in sorted(split_img.rglob("*_leftImg8bit.png")):
            city = img.parent.name
            prefix = img.name[: -len("_leftImg8bit.png")]
            train_id_mask = split_gt / city / f"{prefix}_gtFine_labelTrainIds.png"
            raw_id_mask = split_gt / city / f"{prefix}_gtFine_labelIds.png"
            if train_id_mask.is_file():
                mask, label_format = train_id_mask, "train_id"
            elif raw_id_mask.is_file():
                mask, label_format = raw_id_mask, "label_id"
            else:
                continue
            rec: dict[str, Any] = {
                "id": f"cityscapes:{official_split}:{prefix}",
                "image_path": img.relative_to(root).as_posix(),
                "mask_path": mask.relative_to(root).as_posix(),
                "label_format": label_format,
                "group_id": f"city:{city}",
                "official_split": official_split,
                "attributes": {"city": city, "weather": "clear", "timeofday": "daytime", "scene": "city_street"},
                "image_bytes": img.stat().st_size,
                "mask_bytes": mask.stat().st_size,
            }
            if hash_files:
                rec["image_sha256"] = file_sha256(img)
                rec["mask_sha256"] = file_sha256(mask)
            by_city[city].append(rec)
        if not by_city:
            raise ValueError(f"No matched Cityscapes image/mask pairs found in {official_split}")
        return by_city

    train_cities_map = collect_city_split("train")
    val_cities_map = collect_city_split("val")
    cities = sorted(train_cities_map)
    if len(cities) < 2:
        raise ValueError("City-disjoint split requires at least two training cities")
    rng = random.Random(seed)
    rng.shuffle(cities)
    n_val_cities = max(1, min(len(cities) - 1, round(len(cities) * val_ratio)))
    val_cities = set(cities[:n_val_cities])

    records: list[dict[str, Any]] = []
    for city, items in sorted(train_cities_map.items()):
        split = "validation" if city in val_cities else "train"
        for rec in items:
            records.append({**rec, "split": split})
    for city, items in sorted(val_cities_map.items()):
        for rec in items:
            records.append({**rec, "split": "locked_eval"})

    return _finalize_manifest(
        dataset_name="cityscapes",
        split_policy="City-disjoint partition of official train into train/validation; official val locked as locked_eval.",
        seed=seed,
        records=records,
        hash_files=hash_files,
        enforce_group_disjoint=True,
    )


def build_kitti_manifest(
    dataset_root: str | Path,
    *,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Build a zero-shot cross-dataset evaluation manifest for KITTI Semantics (200 images)."""
    root = Path(dataset_root).resolve()
    img_dir = root / "training" / "image_2"
    mask_dir = root / "training" / "semantic"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected training/image_2 and training/semantic under {root}."
        )
    records: list[dict[str, Any]] = []
    for img in sorted(img_dir.glob("*.png")):
        mask = mask_dir / img.name
        if not mask.is_file():
            continue
        rec: dict[str, Any] = {
            "id": f"kitti:{img.stem}",
            "image_path": img.relative_to(root).as_posix(),
            "mask_path": mask.relative_to(root).as_posix(),
            "label_format": "label_id",
            "group_id": f"kitti:{img.stem}",
            "official_split": "training",
            "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "road"},
            "image_bytes": img.stat().st_size,
            "mask_bytes": mask.stat().st_size,
            "split": "zero_shot_eval",
        }
        if hash_files:
            rec["image_sha256"] = file_sha256(img)
            rec["mask_sha256"] = file_sha256(mask)
        records.append(rec)
    if not records:
        raise ValueError(f"No matched KITTI Semantics pairs found under {root}")
    return _finalize_manifest(
        dataset_name="kitti_semantics",
        split_policy="All 200 KITTI Semantics labeled frames reserved exclusively for zero_shot_eval.",
        seed=0,
        records=records,
        hash_files=hash_files,
    )


def create_synthetic_dataset(
    output_dir: str | Path,
    *,
    num_train: int = 12,
    num_val: int = 4,
    num_eval: int = 4,
    image_size: tuple[int, int] = (128, 256),
    seed: int = 378,
) -> dict[str, Any]:
    """Generate a deterministic synthetic driving-scene dataset for CPU smoke tests."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    height, width = image_size
    records: list[dict[str, Any]] = []

    splits = [("train", num_train), ("validation", num_val), ("locked_eval", num_eval)]
    time_choices = ("daytime", "night")
    weather_choices = ("clear", "rainy")

    for split, count in splits:
        img_dir = root / "images" / split
        mask_dir = root / "masks" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            mask = np.full((height, width), 10, dtype=np.uint8)  # sky

            horizon = height // 2 + int(rng.integers(-6, 7))
            # Sky & lighting variation
            timeofday = time_choices[i % len(time_choices)]
            weather = weather_choices[(i // 2) % len(weather_choices)]
            sky_base = np.array([135, 180, 225] if timeofday == "daytime" else [25, 32, 55], dtype=np.int32)
            img[:horizon, :] = np.clip(sky_base + rng.integers(-12, 13, size=(horizon, width, 3)), 0, 255)

            # Buildings on upper-left / right
            b_h = max(8, horizon // 2)
            mask[horizon - b_h : horizon, : width // 3] = 2  # building
            img[horizon - b_h : horizon, : width // 3] = [80, 82, 90]
            mask[horizon - b_h : horizon, 2 * width // 3 :] = 8  # vegetation
            img[horizon - b_h : horizon, 2 * width // 3 :] = [55, 110, 48]

            # Road & sidewalks
            mask[horizon:, :] = 0  # road
            road_color = np.array([65, 65, 72] if timeofday == "daytime" else [28, 28, 34], dtype=np.int32)
            img[horizon:, :] = np.clip(road_color + rng.integers(-8, 9, size=(height - horizon, width, 3)), 0, 255)
            mask[horizon:, : width // 8] = 1  # sidewalk
            img[horizon:, : width // 8] = [160, 150, 165]

            # Safety-critical objects: car(13), person(11), traffic_sign(7), traffic_light(6)
            car_x = int(rng.integers(width // 4, width // 2))
            car_y = min(height - 20, horizon + 6)
            mask[car_y : car_y + 18, car_x : car_x + 34] = 13
            img[car_y : car_y + 18, car_x : car_x + 34] = [30, 50, 175]

            px = int(rng.integers(width // 8 + 4, width // 4))
            py = min(height - 22, horizon + 2)
            mask[py : py + 20, px : px + 10] = 11
            img[py : py + 20, px : px + 10] = [210, 45, 55]

            sx = min(width - 18, 2 * width // 3 + int(rng.integers(0, 14)))
            sy = max(4, horizon - 20)
            mask[sy : sy + 12, sx : sx + 12] = 7
            img[sy : sy + 12, sx : sx + 12] = [230, 210, 30]

            lx = max(4, width // 3 + int(rng.integers(0, 14)))
            ly = max(4, horizon - 22)
            mask[ly : ly + 12, lx : lx + 8] = 6
            img[ly : ly + 12, lx : lx + 8] = [245, 140, 25]

            # Ignore border (tests ignore_index=255 handling!)
            mask[:2, :] = IGNORE_INDEX
            mask[-2:, :] = IGNORE_INDEX

            stem = f"{split}_{i:04d}"
            img_path = img_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"
            Image.fromarray(img, mode="RGB").save(img_path)
            Image.fromarray(mask, mode="L").save(mask_path)

            records.append(
                {
                    "id": f"synthetic:{split}:{stem}",
                    "image_path": img_path.relative_to(root).as_posix(),
                    "mask_path": mask_path.relative_to(root).as_posix(),
                    "label_format": "train_id",
                    "group_id": f"synthetic:{split}:{i:04d}",
                    "official_split": split,
                    "attributes": {"weather": weather, "timeofday": timeofday, "scene": "synthetic_road"},
                    "image_bytes": img_path.stat().st_size,
                    "mask_bytes": mask_path.stat().st_size,
                    "image_sha256": file_sha256(img_path),
                    "mask_sha256": file_sha256(mask_path),
                    "split": split,
                }
            )

    manifest = _finalize_manifest(
        dataset_name="synthetic_driving_smoke",
        split_policy="Deterministic synthetic 19-class road scenes for CPU verification only.",
        seed=seed,
        records=records,
        hash_files=True,
        enforce_group_disjoint=True,
    )
    save_manifest(manifest, root / "manifest.json")
    return manifest


def _finalize_manifest(
    *,
    dataset_name: str,
    split_policy: str,
    seed: int,
    records: list[dict[str, Any]],
    hash_files: bool,
    enforce_group_disjoint: bool = False,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda r: r["id"])
    inventory = [{k: v for k, v in rec.items() if k != "split"} for rec in ordered]
    present_splits = [name for name in SPLIT_NAMES if any(r["split"] == name for r in ordered)]
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "task": "driving_semantic_segmentation",
        "dataset": dataset_name,
        "seed": seed,
        "split_policy": split_policy,
        "enforce_group_disjoint": enforce_group_disjoint,
        "files_hashed": hash_files,
        "num_classes": NUM_CLASSES,
        "ignore_index": IGNORE_INDEX,
        "class_names": list(CLASS_NAMES),
        "records": ordered,
        "inventory_sha256": canonical_sha256(inventory),
        "counts": {name: sum(r["split"] == name for r in ordered) for name in present_splits},
        "group_counts": {
            name: len({r["group_id"] for r in ordered if r["split"] == name})
            for name in present_splits
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    dataset_root: str | Path | None = None,
    *,
    verify_files: bool = False,
) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("Unsupported manifest schema_version (expected 2)")
    expected_sha = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected_sha:
        raise ValueError("Manifest checksum mismatch")
    records = list(manifest.get("records", []))
    if not records:
        raise ValueError("Manifest contains no records")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate record IDs in manifest")
    if len({r["image_path"] for r in records}) != len(records):
        raise ValueError("Duplicate image_path entries in manifest")
    inventory = [{k: v for k, v in rec.items() if k != "split"} for rec in records]
    if manifest.get("inventory_sha256") != canonical_sha256(inventory):
        raise ValueError("Inventory checksum mismatch")
    if verify_files and dataset_root is None:
        raise ValueError("dataset_root is required when verify_files=True")

    group_splits: dict[str, str] = {}
    enforce_groups = bool(manifest.get("enforce_group_disjoint", False))
    for rec in records:
        split = rec.get("split")
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split name: {split!r}")
        if rec.get("label_format") not in {"train_id", "label_id"}:
            raise ValueError(f"Unsupported label_format: {rec.get('label_format')!r}")
        group_id = rec["group_id"]
        if enforce_groups and group_id in group_splits and group_splits[group_id] != split:
            raise ValueError(f"Group {group_id!r} crosses split boundary ({group_splits[group_id]} vs {split})")
        group_splits[group_id] = split
        if verify_files and dataset_root is not None:
            img_path = safe_dataset_path(dataset_root, rec["image_path"])
            mask_path = safe_dataset_path(dataset_root, rec["mask_path"])
            if not img_path.is_file() or not mask_path.is_file():
                raise ValueError(f"Missing image or mask for record {rec['id']}")
            if img_path.stat().st_size != rec["image_bytes"] or mask_path.stat().st_size != rec["mask_bytes"]:
                raise ValueError(f"File size mismatch for record {rec['id']}")
            if "image_sha256" in rec and file_sha256(img_path) != rec["image_sha256"]:
                raise ValueError(f"Image SHA-256 mismatch for record {rec['id']}")
            if "mask_sha256" in rec and file_sha256(mask_path) != rec["mask_sha256"]:
                raise ValueError(f"Mask SHA-256 mismatch for record {rec['id']}")


def save_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    validate_manifest(manifest)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def load_manifest(path: str | Path, dataset_root: str | Path | None = None, *, verify_files: bool = False) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest, dataset_root=dataset_root, verify_files=verify_files)
    return manifest


def records_for_split(manifest: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unknown split: {split!r}")
    return [dict(r) for r in manifest["records"] if r["split"] == split]


class DrivingSegDataset(Dataset):
    """Paired RGB image and 19-class semantic mask dataset with driving augmentations."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        dataset_root: str | Path,
        *,
        image_size: tuple[int, int] = (512, 1024),
        training: bool = False,
        scale_range: tuple[float, float] = (0.5, 1.5),
        hflip_prob: float = 0.5,
        color_jitter: bool = True,
        adverse_aug_prob: float = 0.15,
    ) -> None:
        self.records = [dict(r) for r in records]
        if not self.records:
            raise ValueError("DrivingSegDataset requires at least one record")
        self.dataset_root = Path(dataset_root).resolve()
        self.image_size = (int(image_size[0]), int(image_size[1]))  # (H, W)
        self.training = bool(training)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.hflip_prob = float(hflip_prob)
        self.color_jitter = bool(color_jitter)
        self.adverse_aug_prob = float(adverse_aug_prob)

    def __len__(self) -> int:
        return len(self.records)

    def _random_scale_crop(self, img: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        target_h, target_w = self.image_size
        orig_w, orig_h = img.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        # Base scale aligns the image proportionally toward target_size before jitter.
        base_scale = max(target_h / max(orig_h, 1), target_w / max(orig_w, 1))
        eff_scale = base_scale * scale
        new_h = max(16, int(round(orig_h * eff_scale)))
        new_w = max(16, int(round(orig_w * eff_scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Pad if scaled image is smaller than crop size.
        pad_h = max(0, target_h - new_h)
        pad_w = max(0, target_w - new_w)
        if pad_h > 0 or pad_w > 0:
            padded_img = Image.new("RGB", (new_w + pad_w, new_h + pad_h), (0, 0, 0))
            padded_img.paste(img, (0, 0))
            padded_mask = Image.new("L", (new_w + pad_w, new_h + pad_h), IGNORE_INDEX)
            padded_mask.paste(mask, (0, 0))
            img, mask = padded_img, padded_mask
            new_w, new_h = img.size

        top = random.randint(0, new_h - target_h)
        left = random.randint(0, new_w - target_w)
        img = img.crop((left, top, left + target_w, top + target_h))
        mask = mask.crop((left, top, left + target_w, top + target_h))
        return img, mask

    def _photometric_augment(self, img: Image.Image) -> Image.Image:
        if self.color_jitter:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.75, 1.25))
        if self.adverse_aug_prob > 0 and random.random() < self.adverse_aug_prob:
            # Mild camera defocus / night dimness simulation for driving robustness.
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            else:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.45, 0.70))
        return img

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        img_path = safe_dataset_path(self.dataset_root, rec["image_path"])
        mask_path = safe_dataset_path(self.dataset_root, rec["mask_path"])
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
        with Image.open(mask_path) as raw_mask:
            mask_np = map_mask_to_train_ids(np.asarray(raw_mask), rec.get("label_format", "train_id"))
        mask = Image.fromarray(mask_np, mode="L")

        if self.training:
            img, mask = self._random_scale_crop(img, mask)
            if random.random() < self.hflip_prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            img = self._photometric_augment(img)
        else:
            target_h, target_w = self.image_size
            img = img.resize((target_w, target_h), Image.BILINEAR)
            mask = mask.resize((target_w, target_h), Image.NEAREST)

        image_tensor = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64))
        attrs = rec.get("attributes", {})
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": rec["id"],
            "weather": str(attrs.get("weather", "unknown")),
            "timeofday": str(attrs.get("timeofday", "unknown")),
            "scene": str(attrs.get("scene", "unknown")),
        }


def make_dataloader(
    records: Iterable[Mapping[str, Any]],
    dataset_root: str | Path,
    *,
    image_size: tuple[int, int] = (512, 1024),
    batch_size: int = 8,
    training: bool = False,
    seed: int = 378,
    num_workers: int = 4,
    pin_memory: bool = False,
    drop_last: bool = False,
    adverse_aug_prob: float = 0.15,
) -> DataLoader:
    dataset = DrivingSegDataset(
        records,
        dataset_root,
        image_size=image_size,
        training=training,
        adverse_aug_prob=adverse_aug_prob,
    )

    def _seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=drop_last and training and len(dataset) >= batch_size,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
    )
