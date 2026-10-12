import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from knowledge_distillation.evaluation import (canonical_sha256, classification_metrics,
                                              build_official_test_manifest,
                                              evaluate_keras, freeze_evaluation, safe_record_path,
                                              validate_freeze)
from knowledge_distillation.runtime import file_sha256

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class EvaluationIntegrityTests(unittest.TestCase):
    def test_metrics_class_support_and_macro_policy(self):
        result = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1])
        self.assertEqual(result["correct"], 2)
        self.assertAlmostEqual(result["accuracy"], .5)
        self.assertAlmostEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["per_class"][2]["recall"], 0)
        self.assertEqual(result["per_class"][42]["support"], 0)
        self.assertEqual(np.asarray(result["confusion_matrix"]).shape, (43, 43))
        self.assertAlmostEqual(result["macro_f1"], ((2 / 3) + .5) / 43)

    def test_freeze_detects_model_manifest_and_source_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "knowledge_distillation"
            package.mkdir()
            source = package / "models.py"
            source.write_text("MODEL_VERSION = 1\n")
            model = root / "tiny.keras"
            model.write_bytes(b"artifact-v1")
            manifest = {"records": [{"id": "a", "split": "train"}], "class_names": [str(i) for i in range(43)],
                        "inventory_sha256": "inventory"}
            path = root / "freeze.json"
            freeze_evaluation({"tiny": model}, manifest, {"seed": 378}, path, root)
            validate_freeze(path, manifest, model, root)
            with self.assertRaises(FileExistsError):
                freeze_evaluation({"tiny": model}, manifest, {}, path, root)
            with self.assertRaises(ValueError):
                validate_freeze(path, dict(manifest, inventory_sha256="changed"), model, root)
            model.write_bytes(b"artifact-v2")
            with self.assertRaisesRegex(ValueError, "artifact"):
                validate_freeze(path, manifest, model, root)
            model.write_bytes(b"artifact-v1")
            source.write_text("MODEL_VERSION = 2\n")
            with self.assertRaisesRegex(ValueError, "Source"):
                validate_freeze(path, manifest, model, root)

    def test_dataset_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                safe_record_path(temporary, {"relative_path": "../outside.png"})

    def test_locked_evaluation_requires_freeze_before_dataset_access(self):
        manifest = {"records": []}
        with self.assertRaisesRegex(ValueError, "freeze"):
            evaluate_keras("missing.keras", manifest, "missing", "locked_holdout", "unused")

    def test_external_audit_is_bound_to_selected_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "tiny.keras"
            model.write_bytes(b"artifact")
            training = {"records": [{"id": "train:1", "split": "train"}],
                        "class_names": [str(i) for i in range(43)], "inventory_sha256": "inventory"}
            freeze_path = root / "freeze.json"
            freeze_evaluation({"tiny": model}, training, {"seed": 378}, freeze_path)
            (root / "Test").mkdir()
            (root / "Test" / "00001.png").write_bytes(b"hashed image payload")
            (root / "Test.csv").write_text("Path,ClassId\nTest/00001.png,2\n", encoding="utf-8")
            external = build_official_test_manifest(root, training, freeze_path)
            self.assertTrue(external["official_test_previously_used_for_selection"])
            self.assertEqual(external["records"][0]["label"], 2)
            validate_freeze(freeze_path, external, model)
            external["records"][0]["label"] = 3
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_freeze(freeze_path, external, model)


@unittest.skipUnless(HAS_TF, "TensorFlow runtime is not installed")
class SavedEvaluationTests(unittest.TestCase):
    def test_streamed_saved_model_evaluation_ids_and_exact_counts(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for i, label in enumerate([3, 4, 3]):
                image = root / f"sample-{i}.png"
                image.write_bytes(tf.io.encode_png(np.full((12, 12, 3), 50 + i * 30, np.uint8)).numpy())
                records.append({"id": f"validation:{i}", "relative_path": image.name, "label": label,
                                "track_id": f"track:{i}", "split": "validation", "sha256": file_sha256(image)})
            manifest = {"records": records, "class_names": [str(i) for i in range(43)],
                        "inventory_sha256": canonical_sha256(records), "image_size": 96}
            inputs = tf.keras.Input((96, 96, 3))
            outputs = tf.keras.layers.Dense(43, kernel_initializer="zeros")(tf.keras.layers.GlobalAveragePooling2D()(inputs))
            model = tf.keras.Model(inputs, outputs)
            bias = np.zeros(43, np.float32)
            bias[3] = 5
            model.layers[-1].set_weights([np.zeros((3, 43), np.float32), bias])
            model_path = root / "tiny.keras"
            model.save(model_path)
            result = evaluate_keras(model_path, manifest, root, "validation", root / "evaluation", batch_size=2)
            self.assertEqual(result["count"], 3)
            self.assertEqual(result["correct"], 2)
            items = [json.loads(line) for line in (root / "evaluation" / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([item["id"] for item in items], [r["id"] for r in records])
            self.assertTrue(all(set(item) == {"id", "label", "prediction", "confidence"} for item in items))
            from knowledge_distillation.benchmark import benchmark_models
            benchmark = benchmark_models({"tiny": model_path}, manifest, root, root / "benchmark",
                                         sample_count=2, warmup=1, repetitions=2, threads=1)
            actual = benchmark["models"]["tiny"]
            self.assertEqual(actual["pure_model"]["count"], 4)
            self.assertEqual(actual["end_to_end"]["count"], 4)
            self.assertEqual(actual["model_sha256"], file_sha256(model_path))
            self.assertEqual(actual["threads"], 1)
            self.assertEqual(actual["device"], "CPU")
            self.assertGreater(actual["pure_model"]["p50_ms"], 0)
            records[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                evaluate_keras(model_path, manifest, root, "validation", root / "bad-evaluation")
