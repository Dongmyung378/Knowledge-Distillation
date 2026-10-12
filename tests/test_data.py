import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from knowledge_distillation.data import (
    build_split_manifest, canonical_sha256, class_weights_for_records,
    load_manifest, records_for_split, safe_dataset_path, save_manifest,
    track_id_from_path, validate_manifest,
)


def fake_dataset(root, classes=2, tracks=7, frames=2):
    rows = []
    for label in range(classes):
        directory = root / "Train" / str(label)
        directory.mkdir(parents=True)
        for track in range(tracks):
            for frame in range(frames):
                name = f"{label:05d}_{track:05d}_{frame:05d}.png"
                path = directory / name
                path.write_bytes(f"synthetic {label} {track} {frame}".encode())
                rows.append({"Path": path.relative_to(root).as_posix(), "ClassId": label})
    with (root / "Train.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Path", "ClassId"])
        writer.writeheader()
        writer.writerows(rows)


class DataProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fake_dataset(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self, seed=378):
        return build_split_manifest(self.root, seed=seed, num_classes=2)

    def test_physical_sign_groups_never_cross_splits_and_all_classes_survive(self):
        manifest = self.manifest()
        groups = {}
        paths = set()
        for name in ("train", "validation", "locked_holdout"):
            selected = records_for_split(manifest, name)
            self.assertEqual({r["label"] for r in selected}, {0, 1})
            for record in selected:
                self.assertNotIn(record["id"], paths)
                paths.add(record["id"])
                groups.setdefault(record["track_id"], set()).add(name)
        self.assertTrue(all(len(names) == 1 for names in groups.values()))
        self.assertEqual(len(paths), 28)

    def test_repeatability_and_seed_affects_assignment_only(self):
        one = self.manifest()
        two = self.manifest()
        other = self.manifest(379)
        self.assertEqual(one, two)
        self.assertEqual(one["inventory_sha256"], other["inventory_sha256"])
        self.assertNotEqual(one["split_ids"], other["split_ids"])

    def test_official_test_metadata_not_consulted(self):
        (self.root / "Test.csv").write_text("not a readable valid test CSV", encoding="utf-8")
        self.assertFalse(self.manifest()["official_test_used_in_split_construction"])

    def test_round_trip_and_checksum_tamper(self):
        manifest = self.manifest()
        saved = save_manifest(manifest, self.root / "splits.json")
        self.assertEqual(load_manifest(saved), manifest)
        changed = json.loads(saved.read_text(encoding="utf-8"))
        changed["seed"] = 5
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_manifest(changed)

    def test_resealed_cross_split_track_is_still_rejected(self):
        changed = copy.deepcopy(self.manifest())
        track = changed["records"][0]["track_id"]
        records = [r for r in changed["records"] if r["track_id"] == track]
        records[0]["split"] = "validation" if records[0]["split"] != "validation" else "train"
        changed["manifest_sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "manifest_sha256"})
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_manifest(changed)

    def test_file_corruption_and_annotation_changes_are_rejected(self):
        manifest = self.manifest()
        validate_manifest(manifest, self.root, verify_files=True)
        path = safe_dataset_path(self.root, manifest["records"][0]["relative_path"])
        path.write_bytes(b"changed pixels")
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_manifest(manifest, self.root, verify_files=True)
        (self.root / "Train.csv").write_text("changed annotations", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CSV checksum"):
            validate_manifest(manifest, self.root, verify_files=True)

    def test_safe_relative_paths_and_explicit_track_parsing(self):
        for unsafe in ("../secret.png", "/tmp/image.png", "C:/secret.png", "Train/../../secret.png"):
            with self.assertRaises(ValueError):
                safe_dataset_path(self.root, unsafe)
        self.assertEqual(track_id_from_path("Train/4/00004_00013_00007.png", 4), "04:00013")
        self.assertEqual(track_id_from_path("Train/4/00013_00007.ppm", 4), "04:00013")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            track_id_from_path("Train/4/00005_00013_00007.png", 4)
        with self.assertRaisesRegex(ValueError, "Cannot recover"):
            track_id_from_path("Train/4/arbitrary.png", 4)

    def test_inverse_frequency_weights_computed_only_from_given_records(self):
        weights = class_weights_for_records([{"label": 0}] * 3 + [{"label": 1}], num_classes=2)
        self.assertAlmostEqual(weights[0], 4 / 6)
        self.assertEqual(weights[1], 2)
        with self.assertRaises(ValueError):
            class_weights_for_records([{"label": 0}], num_classes=2)


if __name__ == "__main__":
    unittest.main()
