import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from knowledge_distillation.experiments import bind_source_snapshot, load_plan, validate_plan, verify_completed_result
from knowledge_distillation.data import file_sha256

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
if HAS_TENSORFLOW:
    import numpy as np
    import tensorflow as tf

    from knowledge_distillation.data import build_split_manifest, make_dataset, records_for_split
    from knowledge_distillation.models import build_student, weights_sha256
    from knowledge_distillation.training import train_student, train_teacher, require_kaggle_gpu


class ProtocolBudgetTests(unittest.TestCase):
    def test_plan_fixes_search_budget_and_predeclares_deployment_seed(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "experiment-plan.json"
        plan = load_plan(path)
        self.assertLessEqual(len(plan["kd_candidates"]), 4)
        self.assertEqual(plan["confirmation_seeds"], [378, 379, 380])
        self.assertEqual(plan["deployment_seed"], 378)
        changed = json.loads(json.dumps(plan))
        changed["student_training"]["max_epochs"] = 31
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_source_change_rejected_before_next_training_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("knowledge_distillation.experiments.source_inventory", return_value={"data.py": "first"}):
                self.assertEqual(bind_source_snapshot(output), {"data.py": "first"})
                bind_source_snapshot(output)
            with patch("knowledge_distillation.experiments.source_inventory", return_value={"data.py": "changed"}):
                with self.assertRaisesRegex(ValueError, "changed"):
                    bind_source_snapshot(output)

    def test_altered_completed_model_or_report_cannot_be_frozen_as_same_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model = output / "student.keras"
            model.write_bytes(b"first saved model")
            report = {"model_sha256": file_sha256(model), "manifest_sha256": "data-hash", "source_sha256": {"core.py": "source-hash"}}
            report_path = output / "training-report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = {"model_path": model.name, "report_path": report_path.name, "report": report}
            with patch("knowledge_distillation.experiments.source_inventory", return_value=report["source_sha256"]):
                verify_completed_result(result, output, {"manifest_sha256": "data-hash"})
                report_path.write_text(json.dumps({**report, "validation_accuracy": 0.999}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_completed_result(result, output, {"manifest_sha256": "data-hash"})
                report_path.write_text(json.dumps(report), encoding="utf-8")
                model.write_bytes(b"a different saved model")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_completed_result(result, output, {"manifest_sha256": "data-hash"})


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is required for tiny CPU checkpoint tests")
class TinyCPUTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "dataset"
        with tf.device("/CPU:0"):
            for label in range(2):
                directory = self.dataset_root / "Train" / str(label)
                directory.mkdir(parents=True)
                for track in range(3):
                    pixels = np.full((16, 16, 3), 30 + label * 140 + track * 5, np.uint8)
                    encoded = tf.io.encode_png(pixels).numpy()
                    (directory / f"{label:05d}_{track:05d}_00000.png").write_bytes(encoded)
        self.manifest = build_split_manifest(self.dataset_root, num_classes=2, image_size=16)

    def tearDown(self):
        tf.keras.backend.clear_session()
        self.temporary.cleanup()

    def test_best_plain_model_saved_reloaded_and_validation_loss_recorded(self):
        config = {"max_epochs": 1, "patience": 5, "batch_size": 2, "augmentation": False, "alpha": 1.0, "verbose": 0}
        result = train_student(self.dataset_root, self.manifest, self.root / "baseline", config=config, execution_mode="cpu-smoke")
        self.assertTrue(Path(result["model_path"]).is_file())
        self.assertTrue(result["report"]["validation_metrics_reloaded_from_checkpoint"])
        self.assertGreater(result["report"]["validation_ce"], 0.0)
        restored = tf.keras.models.load_model(result["model_path"], compile=False, safe_mode=True)
        self.assertEqual(weights_sha256(restored), result["report"]["weights_sha256"])
        history = json.loads((self.root / "baseline" / "history.json").read_text(encoding="utf-8"))
        self.assertGreater(history["val_loss"][0], 0.0)
        with self.assertRaises(FileExistsError):
            train_student(self.dataset_root, self.manifest, self.root / "baseline", config=config, execution_mode="cpu-smoke")

    def test_two_stage_teacher_checkpoint_is_plain_reloadable_keras(self):
        manifest = build_split_manifest(self.dataset_root, num_classes=2, image_size=32)
        result = train_teacher(
            self.dataset_root, manifest, self.root / "teacher-training",
            config={"variant": "frozen_bn", "head_epochs": 1, "fine_tune_epochs": 1, "batch_size": 2, "verbose": 0},
            execution_mode="cpu-smoke",
        )
        self.assertEqual(len(result["report"]["stages"]), 2)
        self.assertEqual(result["report"]["selection_split"], "validation")
        restored = tf.keras.models.load_model(result["model_path"], compile=False, safe_mode=True)
        self.assertEqual(weights_sha256(restored), result["report"]["weights_sha256"])
        self.assertGreater(result["report"]["validation_ce"], 0.0)
        with zipfile.ZipFile(result["model_path"]) as archive:
            config = json.loads(archive.read("config.json"))
        self.assertFalse(config.get("compile_config"))

    def test_paired_initial_weights_and_kd_teacher_do_not_change(self):
        with tf.device("/CPU:0"):
            tf.keras.utils.set_random_seed(378)
            student = build_student(16, 2)
            initial = self.root / "initial.weights.h5"
            student.save_weights(initial)
            initial_sha = weights_sha256(student)
            teacher = tf.keras.Sequential([tf.keras.layers.Input((16, 16, 3)), tf.keras.layers.GlobalAveragePooling2D(), tf.keras.layers.Dense(2)])
            teacher_path = self.root / "teacher.keras"
            teacher.save(teacher_path)
        common = {"max_epochs": 1, "batch_size": 2, "augmentation": False, "verbose": 0}
        baseline = train_student(self.dataset_root, self.manifest, self.root / "paired-baseline", config={**common, "alpha": 1.0}, initial_weights_path=initial, execution_mode="cpu-smoke")
        distilled = train_student(self.dataset_root, self.manifest, self.root / "paired-kd", config={**common, "alpha": 0.3, "temperature": 3.0}, initial_weights_path=initial, teacher_path=teacher_path, execution_mode="cpu-smoke")
        self.assertEqual(baseline["report"]["initial_weights_sha256"], initial_sha)
        self.assertEqual(distilled["report"]["initial_weights_sha256"], initial_sha)
        self.assertEqual(distilled["report"]["teacher_weights_before_sha256"], distilled["report"]["teacher_weights_after_sha256"])

    def test_fresh_pipelines_repeat_shuffle_and_evaluation_keeps_manifest_order(self):
        records = records_for_split(self.manifest, "train") + records_for_split(self.manifest, "validation")
        first = make_dataset(records, self.dataset_root, training=True, seed=378, batch_size=2, image_size=16)
        second = make_dataset(records, self.dataset_root, training=True, seed=378, batch_size=2, image_size=16)
        for _ in range(2):
            a = np.concatenate([x.numpy() for x, _ in first])
            b = np.concatenate([x.numpy() for x, _ in second])
            np.testing.assert_array_equal(a, b)
        ordered = make_dataset(records, self.dataset_root, batch_size=2, image_size=16)
        labels = np.concatenate([y.numpy() for _, y in ordered]).tolist()
        self.assertEqual(labels, [record["label"] for record in records])

    def test_default_shuffle_covers_all_metadata_before_image_decoding(self):
        records = [self.manifest["records"][0]] * 5000
        original_shuffle = tf.data.Dataset.shuffle
        seen = []

        def record_buffer(dataset, buffer_size, *args, **kwargs):
            seen.append(buffer_size)
            return original_shuffle(dataset, buffer_size, *args, **kwargs)

        with patch.object(tf.data.Dataset, "shuffle", new=record_buffer):
            dataset = make_dataset(records, self.dataset_root, training=True, image_size=16)
        self.assertEqual(seen, [5000])
        self.assertEqual(int(dataset.cardinality().numpy()), 79)

    def test_full_training_does_not_run_on_local_gpu(self):
        with patch("knowledge_distillation.training.Path.is_dir", return_value=False), patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "only in Kaggle"):
                require_kaggle_gpu()

    def test_cpu_smoke_cannot_be_used_for_large_training_budget(self):
        with self.assertRaises(RuntimeError):
            train_student(self.dataset_root, self.manifest, self.root / "forbidden", config={"max_epochs": 3}, execution_mode="cpu-smoke")


if __name__ == "__main__":
    unittest.main()
