"""A separate, checksum-bound 9x3 extension using the unchanged training core.

This runner never performs holdout/test inference, never edits its base run,
and never changes the original four-candidate protocol or evaluation freezes.
TensorFlow is imported only by a requested training stage, after preparation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_distillation.data import canonical_sha256, file_sha256, load_manifest, validate_manifest
from knowledge_distillation.experiments import source_inventory, verify_completed_result

PROTOCOL = "gtsrb-full-grid-extension-v1"
SEEDS = (378, 379, 380)
GRID = {(float(t), float(a)) for t in (3, 5, 10) for a in (.1, .3, .5)}
REUSED_KD = {(378, "T3-a03"), (378, "T5-a03"), (378, "T10-a03"),
             (378, "T5-a05"), (379, "T5-a05"), (380, "T5-a05")}
BASE_METADATA = ("plan-snapshot.json", "source-snapshot.json", "pilot-report.json",
                 "confirmation-report.json", "evaluation-freeze.json", "kernel-completion.json",
                 "kaggle-runtime.json", "gpu-runtime.json", "selected-configuration.json")


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError("A report or plan must be a JSON object")
    return value


def _write_immutable(value: dict, path: Path) -> None:
    if path.exists():
        if canonical_sha256(read_json(path)) != canonical_sha256(value):
            raise ValueError(f"Refusing to change a completed record: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Distinct worker output paths avoid concurrent writes to shared records.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe(root: Path, relative: str) -> Path:
    item = Path(relative)
    if item.is_absolute() or ".." in item.parts or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Artifact paths must be safe relative POSIX paths")
    path = (root / item).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path leaves its declared root")
    return path


def file_binding(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path.name}")
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def validate_plan(plan: dict) -> None:
    if plan.get("schema_version") != 1 or plan.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported full-grid extension protocol")
    if plan.get("training_location") != "kaggle_gpu_only":
        raise ValueError("Full-grid training remains Kaggle GPU only")
    if plan.get("seeds") != list(SEEDS) or plan.get("deployment_seed") != 378:
        raise ValueError("The three development seeds and deployment seed are fixed")
    if plan.get("split_seed") != 378 or plan.get("split_track_ratios") != [.7, .15, .15]:
        raise ValueError("The original track split must be preserved")
    candidates = plan.get("kd_candidates", [])
    pairs = [(float(c["temperature"]), float(c["alpha"])) for c in candidates]
    ids = [c["id"] for c in candidates]
    if len(candidates) != 9 or len(set(ids)) != 9 or len(set(pairs)) != 9 or set(pairs) != GRID:
        raise ValueError("Exactly nine unique Cartesian grid candidates are required")
    for item in candidates:
        expected_id = f"T{int(item['temperature'])}-a{round(float(item['alpha']) * 10):02d}"
        if item["id"] != expected_id:
            raise ValueError("Grid candidate IDs must identify their declared temperature and alpha")
    control = plan.get("diagnostic_controls")
    if control != [{"id": "T5-a00", "temperature": 5.0, "alpha": 0.0, "eligible_for_selection": False}]:
        raise ValueError("Only the excluded T5/alpha-zero pure-soft control is declared")
    selection = plan.get("selection", {})
    if (selection.get("split") != "validation"
        or selection.get("metric") != "mean_unweighted_validation_accuracy_over_three_seeds"
        or selection.get("minimum_mean_accuracy_gain") != .002
        or selection.get("minimum_positive_pairs") != 2
        or selection.get("all_grid_runs_required") is not True
        or selection.get("controls_eligible_for_selection") is not False
        or selection.get("final_outcomes_used_for_selection") is not False):
        raise ValueError("Validation-only selection and the declared paired improvement gate are fixed")
    training = plan.get("student_training", {})
    if training != {"max_epochs": 30, "patience": 5, "batch_size": 64, "learning_rate": .001,
                    "augmentation": True, "deterministic": True, "verbose": 2}:
        raise ValueError("The exact original student training configuration must be preserved")
    reuse = plan.get("reuse_policy", {})
    entries = reuse.get("kd_runs", [])
    imported = [(r["seed"], r["id"]) for r in entries]
    if len(imported) != 6 or set(imported) != REUSED_KD or reuse.get("baseline_seeds") != list(SEEDS):
        raise ValueError("Exactly the six existing KD runs and three baselines are reusable")
    exposure = plan.get("exposure", {})
    if any(exposure.get(key) is not True for key in ("old_locked_holdout_results_already_public",
                                                   "historical_full_train_and_official_test_exposed",
                                                   "new_study_is_registered_after_previous_results")):
        raise ValueError("Historical and already-published evaluation exposure must be declared")
    if plan.get("worker_policy", {}).get("maximum_workers") != 2:
        raise ValueError("At most two isolated workers are declared")
    pins = plan.get("environment_pins", {})
    if pins.get("python") != "3.12.15" or pins.get("gpu_name") != "Tesla T4":
        raise ValueError("Original Python and T4 runtime pins are required")
    if not isinstance(pins.get("packages"), dict) or not pins["packages"]:
        raise ValueError("Exact package versions must be declared")
    for key in ("plan_sha256", "training_freeze_sha256", "core_source_sha256", "manifest_sha256", "artifact_inventory_sha256"):
        value = plan.get("base_bindings", {}).get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Base-study canonical checksums must be explicitly declared")


def load_plan(path: str | Path) -> dict:
    plan = read_json(path)
    validate_plan(plan)
    return plan


def declared_tasks(plan: dict) -> list[dict]:
    validate_plan(plan)
    tasks = []
    for group, recipes in (("grid", plan["kd_candidates"]), ("control", plan["diagnostic_controls"])):
        for seed in SEEDS:
            for recipe in recipes:
                tasks.append({"seed": seed, "id": recipe["id"], "group": group,
                              "temperature": float(recipe["temperature"]), "alpha": float(recipe["alpha"]),
                              "relative_dir": f"students/seed-{seed}/{recipe['id']}",
                              "imported": (seed, recipe["id"]) in REUSED_KD})
    return tasks


def worker_tasks(plan: dict, worker_index: int = 0, worker_count: int = 1) -> list[dict]:
    if worker_count not in (1, 2) or not isinstance(worker_index, int) or not 0 <= worker_index < worker_count:
        raise ValueError("Worker index must belong to a declared one- or two-worker assignment")
    pending = [t for t in declared_tasks(plan) if not t["imported"]]
    return pending[worker_index::worker_count]


def _student_config(plan: dict, alpha: float, temperature: float) -> dict:
    return {**plan["student_training"], "alpha": alpha, "temperature": temperature}


def _metric_report(report: dict) -> None:
    if (report.get("status") != "completed" or report.get("selection_split") != "validation"
        or report.get("locked_holdout_evaluated") is not False or report.get("official_test_evaluated") is not False):
        raise ValueError("Only completed validation-only training reports may enter selection")
    for key in ("validation_accuracy", "validation_ce"):
        if not isinstance(report.get(key), (int, float)) or not math.isfinite(report[key]):
            raise ValueError("Selection metrics must be finite numbers")
    if not 0 <= report["validation_accuracy"] <= 1 or report["validation_ce"] < 0:
        raise ValueError("Invalid validation accuracy or CE")


def _result(root: Path, relative_dir: str) -> dict:
    report = read_json(_safe(root, relative_dir + "/training-report.json"))
    if report.get("model_filename") not in {"student.keras", "teacher.keras"}:
        raise ValueError("Unexpected model filename in a training report")
    return {"model_path": relative_dir + "/" + report["model_filename"],
            "report_path": relative_dir + "/training-report.json", "report": report}


def _verify_student(root: Path, relative_dir: str, seed: int, config: dict,
                    manifest: dict, core: dict, teacher: dict, initialization: dict) -> dict:
    result = _result(root, relative_dir)
    report = result["report"]
    _metric_report(report)
    verify_completed_result(result, root, manifest)
    expected_kind = "baseline" if config["alpha"] == 1 else "distilled"
    if (report.get("kind") != expected_kind or report.get("seed") != seed
        or report.get("parameter_count") != 34_406 or report.get("source_sha256") != core
        or report.get("configuration") != config
        or report.get("configuration_sha256") != canonical_sha256(config)
        or report.get("inventory_sha256") != manifest["inventory_sha256"]
        or report.get("initial_weights_sha256") != initialization["weights_sha256"]
        or report.get("initial_weights_file_sha256") != initialization["file_sha256"]
        or report.get("validation_metrics_reloaded_from_checkpoint") is not True):
        raise ValueError("Completed student configuration, seed, initialization, or data binding mismatch")
    if config["alpha"] < 1:
        if (report.get("teacher_file_sha256") != teacher["model_sha256"]
            or report.get("teacher_weights_before_sha256") != teacher["weights_sha256"]
            or report.get("teacher_weights_after_sha256") != teacher["weights_sha256"]
            or report.get("teacher_immutable") is not True):
            raise ValueError("Completed KD run did not use the exact immutable teacher")
    elif any(report.get(k) is not None for k in ("teacher_file_sha256", "teacher_weights_before_sha256",
                                                "teacher_weights_after_sha256")):
        raise ValueError("A supervised baseline must not use teacher logits")
    runtime = report.get("runtime", {})
    if (runtime.get("python") != "3.12.15" or runtime.get("tensorflow") != "2.19.0"
        or runtime.get("keras") != "3.9.2" or runtime.get("platform") != "Linux"
        or runtime.get("training_location") != "kaggle" or not runtime.get("gpu_names")
        or any(name != "Tesla T4" for name in runtime["gpu_names"])):
        raise ValueError("Completed student used a different training runtime")
    history = read_json(_safe(root, relative_dir + "/history.json"))
    accuracy = history.get("val_accuracy", [])
    ce = history.get("val_loss", [])
    if (not accuracy or len(accuracy) != report["epochs_run"] or len(ce) != len(accuracy)
        or not 1 <= len(accuracy) <= config["max_epochs"]
        or any(not math.isfinite(float(x)) for x in accuracy + ce)
        or report["best_epoch"] != 1 + max(range(len(accuracy)), key=lambda i: accuracy[i])
        or abs(report["validation_accuracy"] - accuracy[report["best_epoch"] - 1]) > 1e-6):
        raise ValueError("Completed checkpoint metrics disagree with the saved training history")
    return result


def inspect_base_run(base_run: str | Path, plan: dict) -> dict:
    """Verify imported evidence without loading a model or any evaluation images."""
    from knowledge_distillation.evaluation import validate_freeze

    validate_plan(plan)
    base = Path(base_run).resolve(strict=True)
    manifest = load_manifest(base / "splits.json")
    core = read_json(base / "source-snapshot.json")
    bindings = plan["base_bindings"]
    if (len(core) != 14 or core != source_inventory()
        or canonical_sha256(core) != bindings["core_source_sha256"]
        or manifest["manifest_sha256"] != bindings["manifest_sha256"]
        or manifest.get("image_size") != 96 or manifest.get("num_classes") != 43):
        raise ValueError("Original core or split differs from the declared immutable base")
    original_plan = read_json(base / "plan-snapshot.json")
    if (canonical_sha256(original_plan) != bindings["plan_sha256"]
        or original_plan["student_training"] != plan["student_training"]):
        raise ValueError("Original plan does not match the declared training configuration")
    freeze = validate_freeze(base / "evaluation-freeze.json", manifest)
    if freeze["freeze_sha256"] != bindings["training_freeze_sha256"]:
        raise ValueError("The original training freeze does not match the declared parent study")
    if freeze["config"]["plan_sha256"] != bindings["plan_sha256"]:
        raise ValueError("The original freeze references a different base plan")
    completion = read_json(base / "kernel-completion.json")
    if (completion.get("status") != "completed" or completion.get("stages") != ["pilot", "confirmation", "freeze"]
        or completion.get("locked_holdout_evaluated") is not False or completion.get("official_test_evaluated") is not False):
        raise ValueError("The base Kaggle training run must be complete without final inference")
    runtime = read_json(base / "kaggle-runtime.json")
    if (runtime.get("python") != plan["environment_pins"]["python"]
        or runtime.get("versions") != plan["environment_pins"]["packages"]
        or runtime.get("local_gpu_used") is not False):
        raise ValueError("Base runtime pins do not match the extension plan")
    pilot, confirmation = (read_json(base / name) for name in ("pilot-report.json", "confirmation-report.json"))
    for report in (pilot, confirmation):
        if (report.get("status") != "completed" or report.get("plan_sha256") != bindings["plan_sha256"]
            or report.get("source_sha256") != core or report.get("manifest_sha256") != manifest["manifest_sha256"]
            or report.get("locked_holdout_evaluated") is not False or report.get("official_test_evaluated") is not False):
            raise ValueError("Base selection record failed its declared provenance checks")
    if pilot.get("selected_candidate_id") != "T5-a05" or confirmation.get("selected_candidate_id") != "T5-a05":
        raise ValueError("Unexpected selected recipe in the immutable base study")
    teacher_result = pilot["selected_teacher"]
    verify_completed_result(teacher_result, base, manifest)
    teacher = teacher_result["report"]
    _metric_report(teacher)
    if (teacher_result["model_path"] != "teachers/variant-0/teacher.keras"
        or teacher["kind"] != "teacher" or teacher["seed"] != 378 or teacher["parameter_count"] != 2_313_067
        or teacher["configuration"] != original_plan["teacher_variants"][0]
        or teacher["configuration_sha256"] != canonical_sha256(teacher["configuration"])
        or teacher.get("source_sha256") != core):
        raise ValueError("Unexpected teacher architecture or training provenance")
    validate_freeze(base / "evaluation-freeze.json", manifest, base / teacher_result["model_path"])
    initializations = {}
    for seed in SEEDS:
        item = read_json(base / "initial-weights" / f"seed-{seed}.json")
        if (item.get("seed") != seed or item.get("parameter_count") != 34_406
            or file_sha256(base / "initial-weights" / f"seed-{seed}.weights.h5") != item["file_sha256"]):
            raise ValueError("Original shared student initialization checksum mismatch")
        initializations[seed] = item
    embedded = {}
    for item in pilot["kd_candidates"]:
        embedded[(378, item["id"])] = item["result"]
    baseline_embedded = {}
    for pair in confirmation["pairs"]:
        seed = pair["seed"]
        if seed in baseline_embedded or seed not in SEEDS:
            raise ValueError("Duplicate or unsupported seed in base confirmation")
        baseline_embedded[seed] = pair["baseline"]
        if pair["distilled"]:
            embedded[(seed, "T5-a05")] = pair["distilled"]
    if set(baseline_embedded) != set(SEEDS) or set(embedded) != REUSED_KD:
        raise ValueError("The base reports do not contain exactly the declared reusable runs")
    runs = []
    for seed in SEEDS:
        relative = f"students/seed-{seed}/baseline"
        result = _verify_student(base, relative, seed, _student_config(plan, 1., 5.), manifest, core,
                                 teacher, initializations[seed])
        if canonical_sha256(result) != canonical_sha256(baseline_embedded[seed]):
            raise ValueError("Base inline baseline and saved training report disagree")
        validate_freeze(base / "evaluation-freeze.json", manifest, base / result["model_path"])
        runs.append(relative)
    for task in declared_tasks(plan):
        if task["imported"]:
            seed = task["seed"]
            result = _verify_student(base, task["relative_dir"], seed, _student_config(plan, task["alpha"], task["temperature"]),
                                     manifest, core, teacher, initializations[seed])
            if canonical_sha256(result) != canonical_sha256(embedded[(seed, task["id"])]):
                raise ValueError("Base inline KD record and saved training report disagree")
            runs.append(task["relative_dir"])
    file_map = {"splits.json": "splits.json"}
    file_map.update({name: "base-evidence/" + name for name in BASE_METADATA})
    file_map["source-snapshot.json"] = "source-snapshot.json"
    for name in ("teacher.keras", "training-report.json", "head-history.json", "fine_tune-history.json"):
        relative = "teachers/variant-0/" + name
        file_map[relative] = relative
    for seed in SEEDS:
        for suffix in ("json", "weights.h5"):
            relative = f"initial-weights/seed-{seed}.{suffix}"
            file_map[relative] = relative
    for relative_dir in runs:
        for name in ("student.keras", "training-report.json", "history.json"):
            relative = relative_dir + "/" + name
            file_map[relative] = relative
    inventory = {name: file_binding(_safe(base, name)) for name in sorted(file_map)}
    if canonical_sha256(inventory) != bindings["artifact_inventory_sha256"]:
        raise ValueError("Original model, history, or evidence bytes differ from the declared base inventory")
    return {"manifest": manifest, "core": core, "teacher": teacher, "initializations": initializations,
            "file_map": file_map, "inventory": inventory, "base_freeze": freeze}


def _snapshot(plan: dict, plan_path: Path, info: dict) -> dict:
    if source_inventory() != info["core"]:
        raise ValueError("Immutable core changed after base evidence was checked")
    if load_plan(plan_path) != plan:
        raise ValueError("Plan contents changed during this invocation")
    return {"schema_version": 1, "protocol": PROTOCOL, "plan_sha256": canonical_sha256(plan),
            "plan_file": file_binding(plan_path), "runner_file": file_binding(Path(__file__)),
            "core_source_sha256": info["core"], "environment_pins": plan["environment_pins"],
            "manifest_sha256": info["manifest"]["manifest_sha256"],
            "parent_training_freeze_sha256": info["base_freeze"]["freeze_sha256"],
            "base_files": info["inventory"], "copied_paths": info["file_map"],
            "planned_grid_runs": 27, "imported_grid_runs": 6, "additional_grid_runs": 21,
            "planned_diagnostic_controls": 3, "imported_baselines": 3,
            "initialization_files_reused": 3, "holdout_results_already_public": True,
            "new_holdout_inference_performed_by_runner": False}


def _verify_copies(output: Path, info: dict) -> None:
    for source, target in info["file_map"].items():
        if file_binding(_safe(output, target)) != info["inventory"][source]:
            raise ValueError("An imported artifact or original evidence file was modified")


def _bind_or_check(output: Path, snapshot: dict, plan: dict, *, prepare: bool = False) -> None:
    path = output / "study-snapshot.json"
    if not path.exists() and not prepare:
        raise ValueError("Run prepare once before training workers or finalization")
    if not path.exists() and output.exists() and any(output.iterdir()):
        raise FileExistsError("A new study must start in an absent or empty output directory")
    _write_immutable(snapshot, path)
    _write_immutable(plan, output / "plan-snapshot.json")


def prepare_study(base_run: str | Path, output_dir: str | Path, plan_path: str | Path) -> dict:
    """Copy the explicitly verified base artifacts; this stage does not import TF."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan, info = load_plan(plan_path), None
    base, output = Path(base_run).resolve(strict=True), Path(output_dir).resolve()
    if output == base or output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("The new study and immutable base run must have disjoint directories")
    info = inspect_base_run(base, plan)
    snapshot = _snapshot(plan, plan_path, info)
    _bind_or_check(output, snapshot, plan, prepare=True)
    for source, target in info["file_map"].items():
        destination = _safe(output, target)
        if destination.exists():
            if file_binding(destination) != info["inventory"][source]:
                raise ValueError("Refusing to replace a modified imported artifact")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_safe(base, source), destination)
    _verify_copies(output, info)
    if inspect_base_run(base, plan)["inventory"] != info["inventory"] or _snapshot(plan, plan_path, info) != snapshot:
        raise ValueError("Base evidence, runner, or core changed during preparation")
    report = {"status": "prepared", "study_snapshot_sha256": canonical_sha256(snapshot),
              "copied_files": len(info["file_map"]), "copied_bytes": sum(v["bytes"] for v in info["inventory"].values()),
              "base_run_written": False, "gpu_training_performed": False,
              "grid_tasks": 27, "reused_kd_runs": 6, "new_kd_tasks": 21, "new_control_tasks": 3}
    _write_immutable(report, output / "preparation-report.json")
    return report


def _runtime_receipt(plan: dict, worker_count: int) -> dict:
    """Validate pins before any fit; the parent has assigned GPU visibility."""
    pins = plan["environment_pins"]
    versions = {name: importlib.metadata.version(name) for name in pins["packages"]}
    if platform.python_version() != pins["python"] or versions != pins["packages"]:
        raise RuntimeError("The actual training Python/packages differ from the declared original runtime")
    from knowledge_distillation.training import require_kaggle_gpu
    import tensorflow as tf

    require_kaggle_gpu()
    devices = tf.config.list_physical_devices("GPU")
    names = [tf.config.experimental.get_device_details(device).get("device_name") for device in devices]
    if any(name != pins["gpu_name"] for name in names) or (worker_count == 2 and len(names) != 1):
        raise RuntimeError("Each parallel worker must see exactly one assigned Tesla T4")
    return {"python": platform.python_version(), "packages": versions, "gpu_names": names,
            "tensorflow_training_device": "/GPU:0", "dtype": "float32"}


def select_configuration(plan: dict, baseline_reports: dict[int, dict],
                         candidate_reports: dict[tuple[int, str], dict]) -> dict:
    """Choose only from nine complete, three-seed validation comparisons."""
    validate_plan(plan)
    if set(baseline_reports) != set(SEEDS):
        raise ValueError("All three paired baselines are required")
    required = {(seed, recipe["id"]) for seed in SEEDS for recipe in plan["kd_candidates"]}
    if set(candidate_reports) != required:
        raise ValueError("Exactly 27 grid reports are required; diagnostics cannot enter selection")
    for report in list(baseline_reports.values()) + list(candidate_reports.values()):
        _metric_report(report)
    rows = []
    for recipe in plan["kd_candidates"]:
        reports = [candidate_reports[(seed, recipe["id"])] for seed in SEEDS]
        accuracies = [r["validation_accuracy"] for r in reports]
        gains = [r["validation_accuracy"] - baseline_reports[seed]["validation_accuracy"] for seed, r in zip(SEEDS, reports)]
        mean_gain = statistics.mean(gains)
        positive = sum(g > 0 for g in gains)
        rows.append({"id": recipe["id"], "temperature": recipe["temperature"], "alpha": recipe["alpha"],
                     "seeds": list(SEEDS), "validation_accuracy_by_seed": accuracies,
                     "mean_validation_accuracy": statistics.mean(accuracies),
                     "sample_std_validation_accuracy": statistics.stdev(accuracies),
                     "mean_validation_ce": statistics.mean(r["validation_ce"] for r in reports),
                     "paired_validation_gains": gains, "mean_validation_accuracy_gain": mean_gain,
                     "sample_std_paired_gain": statistics.stdev(gains), "positive_pairs": positive,
                     "eligible": mean_gain >= plan["selection"]["minimum_mean_accuracy_gain"]
                                 and positive >= plan["selection"]["minimum_positive_pairs"]})
    eligible = [row for row in rows if row["eligible"]]
    # max is stable: equal accuracy and CE retain the earlier declared recipe.
    winner = max(eligible, key=lambda row: (row["mean_validation_accuracy"], -row["mean_validation_ce"])) if eligible else None
    selected = next((recipe for recipe in plan["kd_candidates"] if winner and recipe["id"] == winner["id"]), None)
    return {"status": "completed", "selection_split": "validation", "candidates": rows,
            "selected_candidate_id": winner["id"] if winner else None,
            "selected_configuration": _student_config(plan, selected["alpha"], selected["temperature"]) if selected else _student_config(plan, 1., 5.),
            "deployment_kind": "distilled" if winner else "baseline", "deployment_seed": 378,
            "deployment": {"candidate_id": winner["id"] if winner else None, "seed": 378,
                           "kind": "distilled" if winner else "baseline",
                           "model_path": f"students/seed-378/{winner['id'] if winner else 'baseline'}/student.keras",
                           "selected_on": "validation"},
            "mean_selected_validation_gain": winner["mean_validation_accuracy_gain"] if winner else None,
            "controls_eligible_for_selection": False, "final_outcomes_used_for_selection": False,
            "development_seeds_are_independent_confirmation": False,
            "old_holdout_results_already_public": True,
            "decision": "Choose the eligible fixed recipe by three-seed validation mean; retain baseline if none qualifies."}


def _all_completed(output: Path, plan: dict, info: dict) -> tuple[dict, dict, dict]:
    baselines, candidates, controls = {}, {}, {}
    for seed in SEEDS:
        baselines[seed] = _verify_student(output, f"students/seed-{seed}/baseline", seed, _student_config(plan, 1., 5.),
                                           info["manifest"], info["core"], info["teacher"], info["initializations"][seed])
    for task in declared_tasks(plan):
        result = _verify_student(output, task["relative_dir"], task["seed"], _student_config(plan, task["alpha"], task["temperature"]),
                                 info["manifest"], info["core"], info["teacher"], info["initializations"][task["seed"]])
        target = candidates if task["group"] == "grid" else controls
        target[(task["seed"], task["id"])] = result
    return baselines, candidates, controls


def _finalize(output: Path, plan: dict, info: dict, snapshot: dict) -> dict:
    _verify_copies(output, info)
    baselines, candidates, controls = _all_completed(output, plan, info)
    selection = select_configuration(plan, {s: r["report"] for s, r in baselines.items()},
                                     {key: r["report"] for key, r in candidates.items()})
    _write_immutable(selection, output / "selection-report.json")
    def brief(result):
        report = result["report"]
        return {"model_path": result["model_path"], "report_path": result["report_path"],
                "model_sha256": report["model_sha256"], "report_file": file_binding(output / result["report_path"]),
                "history_file": file_binding(output / Path(result["report_path"]).parent / "history.json"),
                "seed": report["seed"], "validation_accuracy": report["validation_accuracy"],
                "validation_ce": report["validation_ce"], "best_epoch": report["best_epoch"], "epochs_run": report["epochs_run"]}
    report = {"status": "completed", "stage": "finalize", "protocol": PROTOCOL,
              "study_snapshot_sha256": canonical_sha256(snapshot), "plan_sha256": canonical_sha256(plan),
              "source_sha256": info["core"], "manifest_sha256": info["manifest"]["manifest_sha256"],
              "base_training_freeze_sha256": info["base_freeze"]["freeze_sha256"],
              "baselines": {str(seed): brief(result) for seed, result in baselines.items()},
              "grid_runs": [{"seed": seed, "id": identifier, **brief(result)} for (seed, identifier), result in candidates.items()],
              "diagnostic_controls": [{"seed": seed, "id": identifier, "eligible_for_selection": False, **brief(result)}
                                      for (seed, identifier), result in controls.items()],
              "selection": selection, "grid_run_count": 27, "imported_grid_run_count": 6,
              "additional_grid_run_count": 21, "diagnostic_control_count": 3,
              "holdout_evaluated_by_runner": False, "official_test_evaluated_by_runner": False,
              "final_outcomes_used_for_selection": False, "exposure": plan["exposure"]}
    _write_immutable(report, output / "study-report.json")
    return report


def _freeze(output: Path, plan: dict, info: dict, snapshot: dict) -> dict:
    from knowledge_distillation.evaluation import freeze_evaluation

    report = _finalize(output, plan, info, snapshot)
    models = {"teacher": output / "teachers/variant-0/teacher.keras"}
    for seed in SEEDS:
        models[f"baseline_seed_{seed}"] = output / f"students/seed-{seed}/baseline/student.keras"
    for task in declared_tasks(plan):
        models[f"{task['group']}_{task['id']}_seed_{task['seed']}"] = output / task["relative_dir"] / "student.keras"
    selected = report["selection"]["selected_candidate_id"]
    config = {"protocol": PROTOCOL, "plan": plan, "plan_sha256": canonical_sha256(plan),
              "study_snapshot_sha256": canonical_sha256(snapshot), "runner_file": snapshot["runner_file"],
              "selection_report_file": file_binding(output / "selection-report.json"),
              "study_report_file": file_binding(output / "study-report.json"),
              "base_training_freeze_sha256": info["base_freeze"]["freeze_sha256"],
              "base_inventory_sha256": canonical_sha256(info["inventory"]),
              "selected_candidate_id": selected, "selected_configuration": report["selection"]["selected_configuration"],
              "deployment_kind": report["selection"]["deployment_kind"], "deployment_seed": 378,
              "deployment_model_name": f"grid_{selected}_seed_378" if selected else "baseline_seed_378",
              "controls_eligible_for_selection": False, "final_reselection_allowed": False,
              "final_outcomes_used_for_selection": False, "exposure": plan["exposure"]}
    path = output / "evaluation-freeze.json"
    if path.exists():
        raise FileExistsError("This separate study is already frozen; its freeze must not be overwritten")
    freeze = freeze_evaluation(models, info["manifest"], config, path)
    return {"status": "frozen", "freeze_path": str(path.resolve()), "freeze": freeze,
            "models": {name: str(value.resolve()) for name, value in models.items()},
            "selected_candidate_id": selected, "base_freeze_modified": False}


def run_full_grid_study(data: str | Path, base_run: str | Path, output: str | Path,
                        plan: str | Path, stage: str = "train", *,
                        worker_index: int = 0, worker_count: int = 1) -> dict:
    """Prepare, train assigned jobs, finalize validation selection, or freeze.

    Parallel mode: parent prepares once, invokes train-worker with GPU visibility
    assigned before Python starts, then invokes finalize and freeze after joins.
    Sequential train evaluates all pending training jobs and finalizes only.
    """
    if stage not in {"prepare", "train", "train-worker", "finalize", "freeze"}:
        raise ValueError("Unknown full-grid stage")
    plan_path = Path(plan).resolve(strict=True)
    plan = load_plan(plan_path)
    base, output = Path(base_run).resolve(strict=True), Path(output).resolve()
    if output == base or output.is_relative_to(base) or base.is_relative_to(output):
        raise ValueError("The new study must be separate from the immutable base run")
    if stage in {"train", "train-worker"} and (output / "evaluation-freeze.json").exists():
        raise RuntimeError("This study is frozen; additional training and reselection are prohibited")
    if stage == "prepare":
        return prepare_study(base, output, plan_path)
    if stage == "train" and not (output / "study-snapshot.json").exists():
        prepare_study(base, output, plan_path)
    info = inspect_base_run(base, plan)
    snapshot = _snapshot(plan, plan_path, info)
    _bind_or_check(output, snapshot, plan)
    _verify_copies(output, info)
    # Dataset verification reads hashes only. Neither training nor selection
    # opens held-out labels for model inference; the existing split stays fixed.
    validate_manifest(info["manifest"], data, verify_files=True)
    if stage == "finalize":
        return _finalize(output, plan, info, snapshot)
    if stage == "freeze":
        return _freeze(output, plan, info, snapshot)
    if (output / "study-report.json").exists():
        raise RuntimeError("Selection is already complete; freeze rather than repeat training")
    if stage == "train" and (worker_index != 0 or worker_count != 1):
        raise ValueError("Sequential train uses worker 0/1; parallel execution uses train-worker")
    assigned = worker_tasks(plan, worker_index, worker_count)
    runtime = _runtime_receipt(plan, worker_count)
    from knowledge_distillation.experiments import _student_run

    worker_path = output / f"worker-{worker_index}-report.json"
    if worker_path.exists():
        previous = read_json(worker_path)
        if (previous.get("worker_count") != worker_count or previous.get("assigned_tasks") != assigned
            or previous.get("study_snapshot_sha256") != canonical_sha256(snapshot)
            or previous.get("runtime") != runtime or previous.get("status") != "completed"):
            raise ValueError("A worker completion report does not match this declared assignment")
    start = time.perf_counter()
    results = []
    for task in assigned:
        config = _student_config(plan, task["alpha"], task["temperature"])
        report_path = _safe(output, task["relative_dir"] + "/training-report.json")
        already_completed = report_path.exists()
        print(json.dumps({"event": "full-grid-task-start", "worker_index": worker_index,
                          "seed": task["seed"], "id": task["id"], "group": task["group"],
                          "reused_completed": already_completed}), flush=True)
        if already_completed:
            _verify_student(output, task["relative_dir"], task["seed"], config, info["manifest"], info["core"],
                            info["teacher"], info["initializations"][task["seed"]])
        elif worker_path.exists():
            raise ValueError("A completed worker has a missing task artifact")
        if _snapshot(plan, plan_path, info) != snapshot or file_binding(output / "teachers/variant-0/teacher.keras") != info["inventory"]["teachers/variant-0/teacher.keras"]:
            raise ValueError("The runner, source, plan, or teacher changed before a task")
        _student_run(data, output, info["manifest"], task["seed"], config, task["relative_dir"],
                     teacher_path=output / "teachers/variant-0/teacher.keras")
        result = _verify_student(output, task["relative_dir"], task["seed"], config, info["manifest"], info["core"],
                                 info["teacher"], info["initializations"][task["seed"]])
        results.append({"seed": task["seed"], "id": task["id"], "group": task["group"],
                        "model_path": result["model_path"], "report_path": result["report_path"],
                        "model_sha256": result["report"]["model_sha256"],
                        "report_file": file_binding(output / result["report_path"]),
                        "history_file": file_binding(output / task["relative_dir"] / "history.json")})
        print(json.dumps({"event": "full-grid-task-complete", "worker_index": worker_index,
                          "seed": task["seed"], "id": task["id"], "group": task["group"],
                          "reported_seconds": result["report"].get("seconds"),
                          "epochs_run": result["report"]["epochs_run"], "best_epoch": result["report"]["best_epoch"],
                          "validation_accuracy": result["report"]["validation_accuracy"],
                          "validation_ce": result["report"]["validation_ce"]}), flush=True)
    if inspect_base_run(base, plan)["inventory"] != info["inventory"] or _snapshot(plan, plan_path, info) != snapshot:
        raise ValueError("Immutable base files, runner, or core changed during worker execution")
    _verify_copies(output, info)
    if worker_path.exists():
        if read_json(worker_path).get("results") != results:
            raise ValueError("A completed worker's output artifact checksums changed")
        worker = read_json(worker_path)
    else:
        worker = {"status": "completed", "worker_index": worker_index, "worker_count": worker_count,
                  "study_snapshot_sha256": canonical_sha256(snapshot), "assigned_tasks": assigned,
                  "runtime": runtime, "results": results, "seconds_this_invocation": time.perf_counter() - start,
                  "final_inference_performed": False, "base_run_written": False}
        _write_immutable(worker, worker_path)
    return _finalize(output, plan, info, snapshot) if stage == "train" else worker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", default=str(Path(__file__).resolve().parents[1] / "configs/full-grid-plan.json"))
    parser.add_argument("--stage", choices=("prepare", "train", "train-worker", "finalize", "freeze"), default="train")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args(argv)
    result = run_full_grid_study(args.data, args.base_run, args.output, args.plan, args.stage,
                                 worker_index=args.worker_index, worker_count=args.worker_count)
    # The persisted records contain complete evidence. Console output remains small.
    print(json.dumps({"status": result["status"], "stage": args.stage,
                      "output": str(Path(args.output).resolve()),
                      "selected_candidate_id": result.get("selected_candidate_id", result.get("selection", {}).get("selected_candidate_id"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
