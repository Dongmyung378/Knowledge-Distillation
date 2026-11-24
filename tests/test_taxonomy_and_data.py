"""Tests for Cityscapes-19 taxonomy, label mapping, manifests, and driving datasets."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from seg_kd.data import (
    DrivingSegDataset,
    build_cityscapes_manifest,
    build_kitti_manifest,
    create_synthetic_dataset,
    load_manifest,
    records_for_split,
    save_manifest,
    validate_manifest,
)
from seg_kd.taxonomy import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    SAFETY_CRITICAL_INDICES,
    colorize_mask,
    map_mask_to_train_ids,
)


class TaxonomyAndDataTests(unittest.TestCase):
    def test_taxonomy_and_label_id_mapping(self) -> None:
        self.assertEqual(len(CLASS_NAMES), NUM_CLASSES)
        self.assertEqual(len(SAFETY_CRITICAL_INDICES), 6)
        raw = np.array([[0, 7, 19, 20, 24, 26, 33, 255]], dtype=np.uint8)
        mapped = map_mask_to_train_ids(raw, label_format="label_id")
        expected = np.array([[255, 0, 6, 7, 11, 13, 18, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(mapped, expected)

        rgb = colorize_mask(mapped)
        self.assertEqual(rgb.shape, (1, 8, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb[0, 0], np.array([0, 0, 0], dtype=np.uint8))

    def test_synthetic_manifest_integrity_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "synth"
            manifest = create_synthetic_dataset(root, num_train=4, num_val=2, num_eval=2, image_size=(64, 128))
            manifest_path = root / "manifest.json"
            reloaded = load_manifest(manifest_path, dataset_root=root, verify_files=True)
            self.assertEqual(reloaded["manifest_sha256"], manifest["manifest_sha256"])
            self.assertEqual(reloaded["counts"], {"train": 4, "validation": 2, "locked_eval": 2})

            # Tampering with a record split without updating checksum must fail.
            tampered = dict(reloaded)
            tampered["records"] = [dict(r) for r in reloaded["records"]]
            tampered["records"][0]["split"] = (
                "validation" if tampered["records"][0]["split"] == "locked_eval" else "locked_eval"
            )
            with self.assertRaises(ValueError):
                validate_manifest(tampered)

            # Dataset loader yields normalized tensors and valid mask values.
            ds = DrivingSegDataset(records_for_split(reloaded, "train"), root, image_size=(64, 128), training=True)
            sample = ds[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 64, 128))
            self.assertEqual(tuple(sample["mask"].shape), (64, 128))
            uniq = set(sample["mask"].unique().tolist())
            self.assertTrue(uniq.issubset(set(range(NUM_CLASSES)) | {IGNORE_INDEX}))

    def test_cityscapes_city_disjoint_and_kitti_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cs_root = Path(tmp) / "cityscapes"
            for split, cities in (("train", ("aachen", "bochum", "bremen", "cologne")), ("val", ("frankfurt",))):
                for city in cities:
                    img_dir = cs_root / "leftImg8bit" / split / city
                    gt_dir = cs_root / "gtFine" / split / city
                    img_dir.mkdir(parents=True, exist_ok=True)
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    for idx in range(2):
                        prefix = f"{city}_{idx:06d}_000019"
                        Image.new("RGB", (64, 32), (80, 90, 100)).save(img_dir / f"{prefix}_leftImg8bit.png")
                        Image.new("L", (64, 32), 7).save(gt_dir / f"{prefix}_gtFine_labelIds.png")

            cs_manifest = build_cityscapes_manifest(cs_root, seed=378, val_ratio=0.25)
            save_manifest(cs_manifest, cs_root / "manifest.json")
            validate_manifest(cs_manifest, cs_root, verify_files=True)
            train_cities = {r["group_id"] for r in records_for_split(cs_manifest, "train")}
            val_cities = {r["group_id"] for r in records_for_split(cs_manifest, "validation")}
            self.assertEqual(len(train_cities & val_cities), 0)
            self.assertEqual(len(records_for_split(cs_manifest, "locked_eval")), 2)

            kitti_root = Path(tmp) / "kitti"
            (kitti_root / "training" / "image_2").mkdir(parents=True)
            (kitti_root / "training" / "semantic").mkdir(parents=True)
            Image.new("RGB", (64, 32), (10, 20, 30)).save(kitti_root / "training" / "image_2" / "000000_10.png")
            Image.new("L", (64, 32), 24).save(kitti_root / "training" / "semantic" / "000000_10.png")
            km = build_kitti_manifest(kitti_root)
            self.assertEqual(km["counts"], {"zero_shot_eval": 1})

    def test_bdd100k_hf_layout_discovery_ignores_color_masks(self) -> None:
        from seg_kd.data import build_bdd100k_manifest

        with tempfile.TemporaryDirectory() as tmp:
            bdd_root = Path(tmp) / "bdd_hf"
            for split, n in (("train", 4), ("val", 2)):
                img_dir = bdd_root / "data" / "images_10k" / "10k" / split
                mask_dir = bdd_root / "data" / "labels_10k" / "10k" / split
                color_dir = bdd_root / "data" / "color_labels" / split
                img_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                color_dir.mkdir(parents=True)
                for i in range(n):
                    stem = f"frame_{split}_{i:04d}"
                    Image.new("RGB", (64, 32), (90, 90, 90)).save(img_dir / f"{stem}.jpg")
                    Image.new("L", (64, 32), 0).save(mask_dir / f"{stem}_train_id.png")
                    Image.new("RGB", (64, 32), (128, 64, 128)).save(mask_dir / f"{stem}_color.png")
                    Image.new("RGB", (64, 32), (128, 64, 128)).save(color_dir / f"{stem}.png")
            bm = build_bdd100k_manifest(bdd_root, seed=378, val_ratio=0.25)
            validate_manifest(bm, bdd_root, verify_files=True)
            self.assertEqual(bm["counts"]["locked_eval"], 2)
            self.assertEqual(bm["counts"]["train"] + bm["counts"]["validation"], 4)


if __name__ == "__main__":
    unittest.main()
