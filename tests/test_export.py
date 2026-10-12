import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from knowledge_distillation.export import assess_int8_acceptance, calibration_records, export_student
from knowledge_distillation.runtime import file_sha256, load_model

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class CalibrationTests(unittest.TestCase):
    def test_calibration_never_uses_validation_or_holdout(self):
        records = [{"id": f"train:{i}", "label": i % 4, "split": "train"} for i in range(20)]
        records += [{"id": "sentinel-final", "label": 0, "split": "locked_holdout"}]
        manifest = {"records": records}
        chosen = calibration_records(manifest, 8, seed=378)
        self.assertEqual(chosen, calibration_records(manifest, 8, seed=378))
        self.assertEqual(len(chosen), 8)
        self.assertEqual({r["label"] for r in chosen}, {0, 1, 2, 3})
        self.assertTrue(all(r["split"] == "train" for r in chosen))
        with self.assertRaisesRegex(ValueError, "training"):
            calibration_records(manifest, input_split="validation")

    def test_smaller_but_slower_int8_is_not_accepted(self):
        metric = {"accuracy": .95, "split": "validation", "evaluation_records_sha256": "same", "freeze_sha256": "same",
                  "model_sha256": "same-model", "training_manifest_sha256": "same-data"}
        common = {"sample_ids_sha256": "same", "threads": 1, "repetitions": 3, "warmup": 10, "device": "CPU",
                  "model_sha256": "same-model", "manifest_sha256": "same-data"}
        fp32 = dict(common, model_bytes=1000, pure_model={"p50_ms": 2.0})
        int8 = dict(common, model_bytes=300, pure_model={"p50_ms": 3.0})
        result = assess_int8_acceptance(metric, metric, fp32, int8)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["selected_format"], "FP32")
        int8["pure_model"]["p50_ms"] = 1.99
        self.assertFalse(assess_int8_acceptance(metric, metric, fp32, int8)["accepted"])
        int8["pure_model"]["p50_ms"] = 2 / 1.05
        self.assertTrue(assess_int8_acceptance(metric, metric, fp32, int8)["accepted"])
        int8["pure_model"]["p50_ms"] = 1
        self.assertTrue(assess_int8_acceptance(metric, dict(metric, accuracy=.945), fp32, int8)["accepted"])
        self.assertFalse(assess_int8_acceptance(metric, dict(metric, accuracy=.93), fp32, int8)["accepted"])
        with self.assertRaises(ValueError):
            assess_int8_acceptance(metric, metric, fp32, int8, max_accuracy_drop_pp=2)
        with self.assertRaisesRegex(ValueError, "artifact"):
            assess_int8_acceptance(metric, metric, fp32, dict(int8, model_sha256="another-model"))
        with self.assertRaisesRegex(ValueError, "validation"):
            assess_int8_acceptance(dict(metric, split="locked_holdout"), dict(metric, split="locked_holdout"), fp32, int8)
        with self.assertRaisesRegex(ValueError, "1.05"):
            assess_int8_acceptance(metric, metric, fp32, int8, minimum_speedup=1.01)


@unittest.skipUnless(HAS_TF, "TensorFlow runtime is not installed")
class ExportRoundtripTests(unittest.TestCase):
    def test_real_fp32_and_integer_conversion_reloads(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for i in range(4):
                path = root / f"train-{i}.png"
                image = np.random.default_rng(i).integers(0, 256, (16, 16, 3), dtype=np.uint8)
                path.write_bytes(tf.io.encode_png(image).numpy())
                records.append({"id": f"train:{i}", "relative_path": path.name, "label": i,
                                "track_id": str(i), "split": "train", "sha256": file_sha256(path)})
            manifest = {"records": records, "class_names": [str(i) for i in range(43)]}
            inputs = tf.keras.Input((96, 96, 3))
            value = tf.keras.layers.Conv2D(4, 3, activation="relu")(inputs)
            outputs = tf.keras.layers.Dense(43)(tf.keras.layers.GlobalAveragePooling2D()(value))
            model = tf.keras.Model(inputs, outputs)
            model_path = root / "student.keras"
            model.save(model_path)
            report = export_student(model_path, manifest, root, root / "export", calibration_count=4)
            self.assertEqual(report["calibration_split"], "train")
            self.assertEqual(report["exports"]["int8"]["floating_tensor_count"], 0)
            self.assertLess(report["fp32_max_absolute_logit_difference"], .001)
            runner = load_model(root / "export" / "student-int8.tflite")
            predictions = runner.predict(np.ones((2, 96, 96, 3), np.float32) * .5)
            self.assertEqual(predictions.shape, (2, 43))
            self.assertTrue(np.isfinite(predictions).all())
