"""Protocol/integrity tests with synthetic files, never GPU or image inference."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_distillation.data import canonical_sha256, file_sha256, track_id_from_path
from knowledge_distillation.evaluation import freeze_evaluation
from knowledge_distillation.experiments import source_inventory
from scripts import full_grid_study as study

ROOT = Path(__file__).resolve().parents[1]


def write_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def metric(accuracy=.75, ce=1.):
    return {"status": "completed", "selection_split": "validation", "locked_holdout_evaluated": False,
            "official_test_evaluated": False, "validation_accuracy": accuracy, "validation_ce": ce}


def selection_fixture(plan, accuracy=.75, ce=1.):
    baselines = {seed: metric(.70) for seed in study.SEEDS}
    reports = {(seed, recipe["id"]): metric(accuracy, ce) for seed in study.SEEDS for recipe in plan["kd_candidates"]}
    return baselines, reports


class SyntheticBase:
    """Build metadata-consistent fake artifacts; no tensors or real performance."""

    def __init__(self, root):
        self.base = root / "base"
        self.data = root / "data"
        self.output = root / "extended-study"
        self.plan_path = root / "full-grid-plan.json"
        self.plan = study.load_plan(ROOT / "configs/full-grid-plan.json")
        self.core = source_inventory()
        self.original_plan = study.read_json(ROOT / "configs/experiment-plan.json")
        self.base.mkdir()
        records = []
        for label in range(43):
            for track, split in enumerate(("train", "validation", "locked_holdout")):
                relative = f"Train/{label}/{label:05d}_{track:05d}_00000.png"
                path = self.data / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic metadata fixture {label} {track}".encode())
                records.append({"id": relative, "relative_path": relative, "label": label,
                                "track_id": track_id_from_path(relative, label), "split": split,
                                "sha256": file_sha256(path), "bytes": path.stat().st_size})
        self.manifest = {"schema_version": 1, "image_size": 96, "num_classes": 43,
                         "class_names": [str(i) for i in range(43)], "records": records,
                         "inventory_sha256": canonical_sha256([{k: v for k, v in r.items() if k != "split"} for r in records]),
                         "split_ids": {s: [r["id"] for r in records if r["split"] == s]
                                       for s in ("train", "validation", "locked_holdout")},
                         "counts": {s: 43 for s in ("train", "validation", "locked_holdout")},
                         "track_counts": {s: 43 for s in ("train", "validation", "locked_holdout")}}
        self.manifest["manifest_sha256"] = canonical_sha256(self.manifest)
        write_json(self.manifest, self.base / "splits.json")
        write_json(self.core, self.base / "source-snapshot.json")
        write_json(self.original_plan, self.base / "plan-snapshot.json")
        self.teacher = {**metric(.8), "kind": "teacher", "seed": 378,
                        "configuration": self.original_plan["teacher_variants"][0],
                        "configuration_sha256": canonical_sha256(self.original_plan["teacher_variants"][0]),
                        "source_sha256": self.core, "manifest_sha256": self.manifest["manifest_sha256"],
                        "inventory_sha256": self.manifest["inventory_sha256"], "parameter_count": 2_313_067,
                        "weights_sha256": hashlib.sha256(b"teacher fake tensors").hexdigest(),
                        "model_filename": "teacher.keras"}
        teacher_dir = self.base / "teachers/variant-0"
        teacher_dir.mkdir(parents=True)
        (teacher_dir / "teacher.keras").write_bytes(b"Synthetic fake teacher artifact, no tensors")
        self.teacher["model_sha256"] = file_sha256(teacher_dir / "teacher.keras")
        write_json(self.teacher, teacher_dir / "training-report.json")
        for stage in ("head", "fine_tune"):
            write_json({"val_accuracy": [.8], "val_loss": [1.]}, teacher_dir / f"{stage}-history.json")
        self.initializations = {}
        for seed in study.SEEDS:
            initial = self.base / "initial-weights" / f"seed-{seed}.weights.h5"
            initial.parent.mkdir(parents=True, exist_ok=True)
            initial.write_bytes(f"synthetic initial weights {seed}".encode())
            item = {"seed": seed, "parameter_count": 34_406, "file_sha256": file_sha256(initial),
                    "weights_sha256": hashlib.sha256(f"fake tensor content {seed}".encode()).hexdigest()}
            self.initializations[seed] = item
            write_json(item, initial.with_name(f"seed-{seed}.json"))
        self.baselines = {}
        self.kd = {}
        for seed in study.SEEDS:
            self.baselines[seed] = self.write_student(self.base, seed, "baseline",
                                                      study._student_config(self.plan, 1., 5.), .7)
        for task in study.declared_tasks(self.plan):
            if task["imported"]:
                self.kd[(task["seed"], task["id"])] = self.write_student(
                    self.base, task["seed"], task["id"],
                    study._student_config(self.plan, task["alpha"], task["temperature"]), .75)
        teacher_result = {"model_path": "teachers/variant-0/teacher.keras",
                          "report_path": "teachers/variant-0/training-report.json", "report": self.teacher}
        common = {"status": "completed", "plan_sha256": canonical_sha256(self.original_plan),
                  "manifest_sha256": self.manifest["manifest_sha256"], "source_sha256": self.core,
                  "locked_holdout_evaluated": False, "official_test_evaluated": False,
                  "selected_candidate_id": "T5-a05"}
        pilot = {**common, "selected_teacher": teacher_result, "baseline": self.baselines[378],
                 "kd_candidates": [{"id": identifier, "result": self.kd[(seed, identifier)]}
                                   for seed, identifier in self.kd if seed == 378]}
        confirmation = {**common, "pairs": [{"seed": seed, "baseline": self.baselines[seed],
                                             "distilled": self.kd[(seed, "T5-a05")]} for seed in study.SEEDS]}
        write_json(pilot, self.base / "pilot-report.json")
        write_json(confirmation, self.base / "confirmation-report.json")
        write_json({"candidate_id": "T5-a05"}, self.base / "selected-configuration.json")
        write_json({"status": "completed", "stages": ["pilot", "confirmation", "freeze"],
                    "locked_holdout_evaluated": False, "official_test_evaluated": False}, self.base / "kernel-completion.json")
        write_json({"python": "3.12.15", "versions": self.plan["environment_pins"]["packages"],
                    "local_gpu_used": False}, self.base / "kaggle-runtime.json")
        write_json({"status": "passed"}, self.base / "gpu-runtime.json")
        frozen = {"teacher": self.base / teacher_result["model_path"]}
        for seed in study.SEEDS:
            frozen[f"baseline_{seed}"] = self.base / self.baselines[seed]["model_path"]
            frozen[f"distilled_{seed}"] = self.base / self.kd[(seed, "T5-a05")]["model_path"]
        freeze = freeze_evaluation(frozen, self.manifest, {"plan_sha256": canonical_sha256(self.original_plan)},
                                   self.base / "evaluation-freeze.json")
        self.plan["base_bindings"] = {"plan_sha256": canonical_sha256(self.original_plan),
                                      "training_freeze_sha256": freeze["freeze_sha256"],
                                      "core_source_sha256": canonical_sha256(self.core),
                                      "manifest_sha256": self.manifest["manifest_sha256"],
                                      "artifact_inventory_sha256": canonical_sha256(self.inventory())}
        write_json(self.plan, self.plan_path)

    def inventory(self):
        return {p.relative_to(self.base).as_posix(): study.file_binding(p)
                for p in sorted(self.base.rglob("*")) if p.is_file()}

    def write_student(self, output, seed, identifier, config, accuracy):
        directory = output / f"students/seed-{seed}/{identifier}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "student.keras").write_bytes(f"Synthetic fake artifact {seed} {identifier}".encode())
        kd = config["alpha"] < 1
        report = {**metric(accuracy), "kind": "distilled" if kd else "baseline", "seed": seed,
                  "configuration": config, "configuration_sha256": canonical_sha256(config),
                  "manifest_sha256": self.manifest["manifest_sha256"], "inventory_sha256": self.manifest["inventory_sha256"],
                  "source_sha256": self.core, "parameter_count": 34_406,
                  "initial_weights_file_sha256": self.initializations[seed]["file_sha256"],
                  "initial_weights_sha256": self.initializations[seed]["weights_sha256"],
                  "teacher_file_sha256": self.teacher["model_sha256"] if kd else None,
                  "teacher_weights_before_sha256": self.teacher["weights_sha256"] if kd else None,
                  "teacher_weights_after_sha256": self.teacher["weights_sha256"] if kd else None,
                  "teacher_immutable": True, "validation_metrics_reloaded_from_checkpoint": True,
                  "epochs_run": 2, "best_epoch": 2, "model_filename": "student.keras",
                  "model_sha256": file_sha256(directory / "student.keras"),
                  "runtime": {"python": "3.12.15", "tensorflow": "2.19.0", "keras": "3.9.2",
                              "platform": "Linux", "training_location": "kaggle", "gpu_names": ["Tesla T4"]}}
        write_json(report, directory / "training-report.json")
        write_json({"val_accuracy": [accuracy - .01, accuracy], "val_loss": [1.1, 1.]}, directory / "history.json")
        return {"model_path": (directory / "student.keras").relative_to(output).as_posix(),
                "report_path": (directory / "training-report.json").relative_to(output).as_posix(), "report": report}

    def fake_train(self, data, output, manifest, seed, config, relative_dir, teacher_path=None):
        identifier = relative_dir.rsplit("/", 1)[1]
        if (output / relative_dir / "training-report.json").exists():
            return study._result(output, relative_dir)
        accuracy = .99 if identifier == "T5-a00" else (.82 if identifier == "T3-a01" else .76)
        return self.write_student(output, seed, identifier, config, accuracy)


class FullGridPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = study.load_plan(ROOT / "configs/full-grid-plan.json")

    def test_exact_nine_by_three_grid_and_excluded_pure_soft_control(self):
        tasks = study.declared_tasks(self.plan)
        self.assertEqual(len([t for t in tasks if t["group"] == "grid"]), 27)
        self.assertEqual(len([t for t in tasks if t["group"] == "control"]), 3)
        self.assertEqual(sum(t["imported"] for t in tasks), 6)
        self.assertEqual({t["alpha"] for t in tasks if t["group"] == "grid"}, {.1, .3, .5})
        for kind in ("missing", "duplicate", "mislabeled"):
            changed = copy.deepcopy(self.plan)
            if kind == "missing":
                changed["kd_candidates"].pop()
            elif kind == "duplicate":
                changed["kd_candidates"][-1] = changed["kd_candidates"][0]
            else:
                changed["kd_candidates"][0]["id"] = "different"
            with self.assertRaises(ValueError):
                study.validate_plan(changed)

    def test_two_workers_are_disjoint_and_cover_exactly_the_24_new_tasks(self):
        one = study.worker_tasks(self.plan, 0, 2)
        two = study.worker_tasks(self.plan, 1, 2)
        keys = lambda ts: {(t["seed"], t["id"]) for t in ts}
        self.assertEqual((len(one), len(two)), (12, 12))
        self.assertFalse(keys(one) & keys(two))
        self.assertEqual(keys(one) | keys(two), keys(study.worker_tasks(self.plan)))
        self.assertFalse((keys(one) | keys(two)) & study.REUSED_KD)
        self.assertEqual(one, study.worker_tasks(self.plan, 0, 2))
        with self.assertRaises(ValueError):
            study.worker_tasks(self.plan, 2, 2)

    def test_validation_means_ce_tie_and_declared_order_determine_selection(self):
        baseline, reports = selection_fixture(self.plan)
        winner = study.select_configuration(self.plan, baseline, reports)
        self.assertEqual(winner["selected_candidate_id"], "T3-a01")
        for seed in study.SEEDS:
            reports[(seed, "T5-a05")]["validation_ce"] = .5
        winner = study.select_configuration(self.plan, baseline, reports)
        self.assertEqual(winner["selected_candidate_id"], "T5-a05")
        self.assertEqual(winner["deployment"]["seed"], 378)
        self.assertFalse(winner["development_seeds_are_independent_confirmation"])

    def test_mean_gain_does_not_replace_two_positive_pairs_gate(self):
        baseline, reports = selection_fixture(self.plan, .699)
        reports[(378, "T3-a01")]["validation_accuracy"] = .9
        result = study.select_configuration(self.plan, baseline, reports)
        self.assertIsNone(result["selected_candidate_id"])
        self.assertEqual(result["deployment"]["kind"], "baseline")
        self.assertFalse(result["candidates"][0]["eligible"])

    def test_missing_reports_controls_or_holdout_metrics_cannot_enter_selection(self):
        baseline, reports = selection_fixture(self.plan)
        partial = dict(reports)
        partial.pop(next(iter(partial)))
        with self.assertRaises(ValueError):
            study.select_configuration(self.plan, baseline, partial)
        with self.assertRaises(ValueError):
            study.select_configuration(self.plan, baseline, {**reports, (378, "T5-a00"): metric(.999)})
        reports[(378, "T3-a01")]["locked_holdout_evaluated"] = True
        with self.assertRaisesRegex(ValueError, "validation-only"):
            study.select_configuration(self.plan, baseline, reports)


class FullGridArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticBase(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def run_stage(self, stage, **kwargs):
        f = self.fixture
        return study.run_full_grid_study(f.data, f.base, f.output, f.plan_path, stage, **kwargs)

    def test_prepare_copies_exact_whitelist_and_never_changes_base_evidence(self):
        f = self.fixture
        before = f.inventory()
        report = self.run_stage("prepare")
        self.assertEqual(report["copied_files"], 47)
        self.assertFalse(report["base_run_written"])
        self.assertEqual(before, f.inventory())
        self.assertEqual(report, self.run_stage("prepare"))
        snapshot = study.read_json(f.output / "study-snapshot.json")
        self.assertEqual(snapshot["core_source_sha256"], source_inventory())
        self.assertEqual(snapshot["planned_grid_runs"], 27)

    def test_tampered_model_or_unreferenced_raw_base_metadata_are_rejected(self):
        f = self.fixture
        model = f.base / "students/seed-378/T3-a03/student.keras"
        original = model.read_bytes()
        model.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            self.run_stage("prepare")
        model.write_bytes(original)
        write_json({"candidate_id": "tampered"}, f.base / "selected-configuration.json")
        with self.assertRaisesRegex(ValueError, "base inventory"):
            self.run_stage("prepare")

    def test_imported_initialization_or_valid_plan_change_cannot_resume_silently(self):
        f = self.fixture
        self.run_stage("prepare")
        copied = f.output / "initial-weights/seed-378.weights.h5"
        original = copied.read_bytes()
        copied.write_bytes(b"changed initialization")
        with self.assertRaisesRegex(ValueError, "modified imported"):
            self.run_stage("prepare")
        copied.write_bytes(original)
        changed = copy.deepcopy(f.plan)
        changed["kd_candidates"] = list(reversed(changed["kd_candidates"]))
        write_json(changed, f.plan_path)
        with self.assertRaisesRegex(ValueError, "completed record"):
            self.run_stage("prepare")

    def test_core_mutation_is_rejected_without_training(self):
        self.run_stage("prepare")
        changed = dict(source_inventory())
        changed["knowledge_distillation/training.py"] = "0" * 64
        with patch.object(study, "source_inventory", return_value=changed):
            with self.assertRaisesRegex(ValueError, "immutable base"):
                self.run_stage("finalize")

    def test_partial_study_cannot_finalize_and_full_workers_freeze_all_34_models(self):
        f = self.fixture
        before = f.inventory()
        self.run_stage("prepare")
        with patch.object(study, "_runtime_receipt", return_value={"synthetic": True}), \
             patch("knowledge_distillation.experiments._student_run", side_effect=f.fake_train) as training:
            self.run_stage("train-worker", worker_index=0, worker_count=2)
            self.assertEqual(training.call_count, 12)
            with self.assertRaises(FileNotFoundError):
                self.run_stage("finalize")
            self.assertEqual(training.call_count, 12)
            self.run_stage("train-worker", worker_index=1, worker_count=2)
            self.assertEqual(training.call_count, 24)
            result = self.run_stage("finalize")
            self.assertEqual(result["grid_run_count"], 27)
            self.assertEqual(result["diagnostic_control_count"], 3)
            self.assertNotEqual(result["selection"]["selected_candidate_id"], "T5-a00")
            frozen = self.run_stage("freeze")
            self.assertEqual(len(frozen["freeze"]["models"]), 34)
            self.assertEqual(frozen["selected_candidate_id"], "T3-a01")
            with self.assertRaisesRegex(RuntimeError, "frozen"):
                self.run_stage("train-worker", worker_index=0, worker_count=2)
            self.assertEqual(training.call_count, 24)
        self.assertEqual(before, f.inventory())

    def test_completed_worker_history_change_cannot_be_reused(self):
        f = self.fixture
        self.run_stage("prepare")
        with patch.object(study, "_runtime_receipt", return_value={"synthetic": True}), \
             patch("knowledge_distillation.experiments._student_run", side_effect=f.fake_train):
            self.run_stage("train-worker", worker_index=0, worker_count=2)
            first = study.worker_tasks(f.plan, 0, 2)[0]
            history = f.output / first["relative_dir"] / "history.json"
            value = study.read_json(history)
            value["val_loss"][0] += .2  # Valid metrics, changed provenance bytes.
            write_json(value, history)
            with self.assertRaisesRegex(ValueError, "checksums changed"):
                self.run_stage("train-worker", worker_index=0, worker_count=2)


if __name__ == "__main__":
    unittest.main()
