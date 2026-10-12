"""A bounded, auditable pilot -> paired confirmation -> freeze protocol.

This module never evaluates the locked holdout or official test. Selection is
finished first, and final evaluation lives behind the separate freeze guard.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .data import build_split_manifest, canonical_sha256, file_sha256, load_manifest, save_manifest, validate_manifest


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema_version") != 1 or plan.get("training_location") != "kaggle_gpu_only":
        raise ValueError("Only the declared Kaggle GPU protocol is supported")
    candidates = plan.get("kd_candidates", [])
    teachers = plan.get("teacher_variants", [])
    if not 1 <= len(candidates) <= 4 or not 1 <= len(teachers) <= 2:
        raise ValueError("The preregistered budget is at most four KD recipes and two teacher variants")
    if len({candidate["id"] for candidate in candidates}) != len(candidates):
        raise ValueError("KD recipe IDs must be unique")
    for candidate in candidates:
        if not 0 < float(candidate["alpha"]) < 1 or float(candidate["temperature"]) <= 0:
            raise ValueError("KD recipes require positive temperatures and alpha strictly between zero and one")
        if not candidate["id"].replace("-", "").replace("_", "").isalnum():
            raise ValueError("KD recipe ID is not safe as an output directory name")
    student = plan["student_training"]
    if not 1 <= int(student["max_epochs"]) <= 30 or int(student["patience"]) != 5:
        raise ValueError("Student protocol requires <=30 epochs and validation stopping patience five")
    seeds = plan.get("confirmation_seeds")
    if seeds != [378, 379, 380] or plan.get("pilot_seed") != 378 or plan.get("deployment_seed") != 378:
        raise ValueError("Three paired seeds and the deployment seed are declared in advance")
    if plan.get("split_seed") != 378 or plan.get("split_track_ratios") != [0.7, 0.15, 0.15]:
        raise ValueError("The track split seed and ratios are fixed by this protocol")
    if plan["selection"].get("metric") != "unweighted_validation_accuracy":
        raise ValueError("Validation accuracy is the declared selection metric")
    if float(plan["selection"]["minimum_absolute_accuracy_gain"]) < 0:
        raise ValueError("The minimum validation improvement cannot be negative")


def load_plan(path: str | Path) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def _write(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_inventory(source_root: str | Path | None = None) -> dict[str, str]:
    root = Path(source_root or Path(__file__).resolve().parents[1]).resolve()
    paths = sorted((root / "knowledge_distillation").glob("*.py"))
    paths += [root / name for name in ("requirements.txt", "requirements-runtime.txt", "pyproject.toml") if (root / name).is_file()]
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def bind_source_snapshot(output: Path) -> dict[str, str]:
    current = source_inventory()
    path = output / "source-snapshot.json"
    if path.exists():
        if _load_json(path) != current:
            raise ValueError("Source or dependencies changed after this protocol run started")
    else:
        _write(current, path)
    return current


def _relative_result(result: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "model_path": Path(result["model_path"]).resolve().relative_to(output.resolve()).as_posix(),
        "report_path": Path(result["report_path"]).resolve().relative_to(output.resolve()).as_posix(),
        "report": result["report"],
    }


def verify_completed_result(result: dict[str, Any], output: Path, manifest: dict[str, Any]) -> None:
    model_path = (output / result["model_path"]).resolve()
    report_path = (output / result["report_path"]).resolve()
    if not model_path.is_relative_to(output.resolve()) or not report_path.is_relative_to(output.resolve()):
        raise ValueError("Completed artifact paths leave the experiment output directory")
    recorded = result["report"]
    if (
        not model_path.is_file() or not report_path.is_file()
        or file_sha256(model_path) != recorded["model_sha256"]
        or canonical_sha256(_load_json(report_path)) != canonical_sha256(recorded)
        or recorded["manifest_sha256"] != manifest["manifest_sha256"]
        or recorded.get("source_sha256") != source_inventory()
    ):
        raise ValueError("Completed artifact, report, source, or dataset checksum mismatch")


def _read_completed_run(output: Path, relative_dir: str, config: dict[str, Any], manifest: dict[str, Any], *, teacher_path=None):
    run = output / relative_dir
    report_path = run / "training-report.json"
    if not report_path.is_file():
        return None
    report = _load_json(report_path)
    model = run / report["model_filename"]
    if (
        report.get("status") != "completed"
        or report["configuration_sha256"] != canonical_sha256(config)
        or report["manifest_sha256"] != manifest["manifest_sha256"]
        or report.get("source_sha256") != source_inventory()
        or (teacher_path is not None and float(config.get("alpha", 1.0)) < 1.0 and report.get("teacher_file_sha256") != file_sha256(teacher_path))
        or not model.is_file()
        or report["model_sha256"] != file_sha256(model)
    ):
        raise ValueError("Existing run failed its completion/configuration/model integrity checks")
    return {"model_path": str(model.resolve()), "report_path": str(report_path.resolve()), "report": report}


def _initial_weights(output: Path, seed: int, manifest: dict[str, Any]) -> Path:
    import tensorflow as tf

    from .models import build_student, weights_sha256

    directory = output / "initial-weights"
    directory.mkdir(parents=True, exist_ok=True)
    weights_path = directory / f"seed-{seed}.weights.h5"
    metadata_path = directory / f"seed-{seed}.json"
    if weights_path.is_file():
        metadata = _load_json(metadata_path)
        if metadata["file_sha256"] != file_sha256(weights_path) or metadata["seed"] != seed:
            raise ValueError("Shared initial student weights failed their checksum")
        return weights_path
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    student = build_student(manifest["image_size"], manifest["num_classes"])
    student.save_weights(weights_path)
    _write({
        "seed": seed, "file_sha256": file_sha256(weights_path), "weights_sha256": weights_sha256(student),
        "parameter_count": student.count_params(), "initialization": "Fresh same-seed random initialization, shared by baseline and KD",
    }, metadata_path)
    del student
    return weights_path


def _teacher_pilot(dataset_root, output, plan, manifest):
    from .training import train_teacher

    results = []
    for index, teacher_config in enumerate(plan["teacher_variants"]):
        relative_dir = f"teachers/variant-{index}"
        result = _read_completed_run(output, relative_dir, teacher_config, manifest)
        if result is None:
            result = train_teacher(dataset_root, manifest, output / relative_dir, seed=plan["pilot_seed"], config=teacher_config)
        results.append(_relative_result(result, output))
    # Stable sorting retains declared candidate order when both scores tie.
    best_index = max(range(len(results)), key=lambda i: (
        results[i]["report"]["validation_accuracy"], -results[i]["report"]["validation_ce"]
    ))
    return results, results[best_index]


def _student_run(dataset_root, output, manifest, seed, config, relative_dir, *, teacher_path=None):
    from .training import train_student

    result = _read_completed_run(output, relative_dir, config, manifest, teacher_path=teacher_path)
    if result is None:
        result = train_student(
            dataset_root, manifest, output / relative_dir, seed=seed, config=config,
            teacher_path=teacher_path, initial_weights_path=_initial_weights(output, seed, manifest),
        )
    return _relative_result(result, output)


def _pilot(dataset_root, output, plan, manifest):
    teachers, teacher = _teacher_pilot(dataset_root, output, plan, manifest)
    baseline_config = {**plan["student_training"], "alpha": 1.0, "temperature": 5.0}
    seed = plan["pilot_seed"]
    baseline = _student_run(dataset_root, output, manifest, seed, baseline_config, f"students/seed-{seed}/baseline")
    candidates = []
    minimum = float(plan["selection"]["minimum_absolute_accuracy_gain"])
    for candidate in plan["kd_candidates"]:
        verify_completed_result(teacher, output, manifest)
        config = {**plan["student_training"], "alpha": candidate["alpha"], "temperature": candidate["temperature"]}
        result = _student_run(
            dataset_root, output, manifest, seed, config,
            f"students/seed-{seed}/{candidate['id']}", teacher_path=output / teacher["model_path"],
        )
        if result["report"]["initial_weights_sha256"] != baseline["report"]["initial_weights_sha256"]:
            raise AssertionError("Baseline and KD do not share initial student weights")
        gain = result["report"]["validation_accuracy"] - baseline["report"]["validation_accuracy"]
        candidates.append({
            "id": candidate["id"], "configuration": config, "result": result,
            "validation_accuracy_gain": gain, "accepted": gain >= minimum and gain > 0,
        })
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = max(accepted, key=lambda item: (
        item["result"]["report"]["validation_accuracy"], -item["result"]["report"]["validation_ce"]
    )) if accepted else None
    report = {
        "status": "completed", "stage": "pilot", "plan_sha256": canonical_sha256(plan),
        "source_sha256": source_inventory(),
        "manifest_sha256": manifest["manifest_sha256"], "selection_split": "validation",
        "teacher_candidates": teachers, "selected_teacher": teacher,
        "baseline": baseline, "kd_candidates": candidates,
        "selected_candidate_id": selected["id"] if selected else None,
        "selected_configuration": selected["configuration"] if selected else baseline_config,
        "decision": "fixed KD recipe proceeds to paired confirmation" if selected else "no KD recipe met the declared improvement gate; retain baseline and stop KD search",
        "locked_holdout_evaluated": False, "official_test_evaluated": False,
    }
    _write(report, output / "pilot-report.json")
    _write({
        "candidate_id": report["selected_candidate_id"], "configuration": report["selected_configuration"],
        "plan_sha256": report["plan_sha256"], "manifest_sha256": report["manifest_sha256"],
        "selection_split": "validation", "deployment_seed": plan["deployment_seed"],
    }, output / "selected-configuration.json")
    return report


def _confirmation(dataset_root, output, plan, manifest):
    pilot = _load_json(output / "pilot-report.json")
    if pilot["plan_sha256"] != canonical_sha256(plan) or pilot["manifest_sha256"] != manifest["manifest_sha256"] or pilot.get("source_sha256") != source_inventory():
        raise ValueError("Pilot protocol or split changed before confirmation")
    verify_completed_result(pilot["selected_teacher"], output, manifest)
    candidate_id = pilot["selected_candidate_id"]
    selected_pilot = next((item for item in pilot["kd_candidates"] if item["id"] == candidate_id), None)
    pairs = []
    for seed in plan["confirmation_seeds"]:
        verify_completed_result(pilot["selected_teacher"], output, manifest)
        baseline_config = {**plan["student_training"], "alpha": 1.0, "temperature": 5.0}
        if seed == plan["pilot_seed"]:
            baseline = pilot["baseline"]
            distilled = selected_pilot["result"] if selected_pilot else None
        else:
            baseline = _student_run(dataset_root, output, manifest, seed, baseline_config, f"students/seed-{seed}/baseline")
            distilled = _student_run(
                dataset_root, output, manifest, seed, pilot["selected_configuration"],
                f"students/seed-{seed}/{candidate_id}", teacher_path=output / pilot["selected_teacher"]["model_path"],
            ) if candidate_id is not None else None
        # Reused completed runs are rechecked before their measurements enter
        # confirmation; a modified model or report is not silently accepted.
        for result in (baseline, distilled):
            if result is not None:
                verify_completed_result(result, output, manifest)
        if distilled and distilled["report"]["teacher_file_sha256"] != pilot["selected_teacher"]["report"]["model_sha256"]:
            raise ValueError("A confirmation student used a different teacher artifact")
        if distilled and baseline["report"]["initial_weights_sha256"] != distilled["report"]["initial_weights_sha256"]:
            raise AssertionError("Confirmation pair initial weights differ")
        pairs.append({
            "seed": seed, "baseline": baseline, "distilled": distilled,
            "validation_accuracy_gain": distilled["report"]["validation_accuracy"] - baseline["report"]["validation_accuracy"] if distilled else None,
        })
    baseline_accuracies = [pair["baseline"]["report"]["validation_accuracy"] for pair in pairs]
    gains = [pair["validation_accuracy_gain"] for pair in pairs if pair["distilled"] is not None]
    confirmed = bool(gains) and statistics.mean(gains) >= float(plan["selection"]["minimum_absolute_accuracy_gain"]) and sum(gain > 0 for gain in gains) >= int(plan["selection"]["confirmation_required_positive_pairs"])
    report = {
        "status": "completed", "stage": "confirmation", "plan_sha256": canonical_sha256(plan),
        "source_sha256": source_inventory(),
        "manifest_sha256": manifest["manifest_sha256"], "selection_split": "validation",
        "selected_candidate_id": candidate_id, "configuration_not_retuned": True,
        "pairs": pairs, "baseline_validation_accuracy_mean": statistics.mean(baseline_accuracies),
        "baseline_validation_accuracy_std": statistics.stdev(baseline_accuracies),
        "mean_validation_accuracy_gain": statistics.mean(gains) if gains else None,
        "std_validation_accuracy_gain": statistics.stdev(gains) if len(gains) > 1 else None,
        "positive_pairs": sum(gain > 0 for gain in gains), "distillation_confirmed": confirmed,
        "deployment_kind": "distilled" if confirmed else "baseline", "deployment_seed": plan["deployment_seed"],
        "teacher_policy": "One validation-selected fresh teacher fixed for every paired student seed",
        "decision": "freeze the declared seed-378 distilled deployment model" if confirmed else "KD did not meet the paired development gate; keep the stronger supervised baseline without further KD search",
        "locked_holdout_evaluated": False, "official_test_evaluated": False,
    }
    _write(report, output / "confirmation-report.json")
    return report


def _freeze(output, plan, manifest):
    from .evaluation import freeze_evaluation

    pilot = _load_json(output / "pilot-report.json")
    confirmation = _load_json(output / "confirmation-report.json")
    for report in (pilot, confirmation):
        if report["plan_sha256"] != canonical_sha256(plan) or report["manifest_sha256"] != manifest["manifest_sha256"] or report.get("source_sha256") != source_inventory():
            raise ValueError("Completed protocol changed before final freeze")
    verify_completed_result(pilot["selected_teacher"], output, manifest)
    for pair in confirmation["pairs"]:
        for result in (pair["baseline"], pair["distilled"]):
            if result is not None:
                verify_completed_result(result, output, manifest)
        if pair["distilled"] and pair["distilled"]["report"]["teacher_file_sha256"] != pilot["selected_teacher"]["report"]["model_sha256"]:
            raise ValueError("Frozen students do not share the selected teacher artifact")
    models = {"teacher": output / pilot["selected_teacher"]["model_path"]}
    deployment_pair = next(pair for pair in confirmation["pairs"] if pair["seed"] == plan["deployment_seed"])
    models["baseline"] = output / deployment_pair["baseline"]["model_path"]
    if deployment_pair["distilled"]:
        # The fixed candidate is preserved even when the validation gate rejects
        # it; deployment_kind records which model is actually recommended.
        models["distilled"] = output / deployment_pair["distilled"]["model_path"]
    for pair in confirmation["pairs"]:
        if pair["seed"] != plan["deployment_seed"]:
            models[f"baseline_seed_{pair['seed']}"] = output / pair["baseline"]["model_path"]
            if pair["distilled"]:
                models[f"distilled_seed_{pair['seed']}"] = output / pair["distilled"]["model_path"]
    config = {
        "protocol": plan, "plan_sha256": canonical_sha256(plan),
        "selected_configuration": pilot["selected_configuration"],
        "selected_candidate_id": pilot["selected_candidate_id"],
        "deployment_kind": confirmation["deployment_kind"], "deployment_seed": plan["deployment_seed"],
        "distillation_confirmed_on_validation": confirmation["distillation_confirmed"],
        "historical_exposure": plan["historical_exposure"],
        "final_reselection_allowed": False,
    }
    freeze_path = output / "evaluation-freeze.json"
    if freeze_path.exists():
        raise FileExistsError("Final freeze already exists; do not overwrite a protocol after final access")
    freeze = freeze_evaluation(
        models=models, manifest=manifest, config=config, output_path=freeze_path,
        source_root=Path(__file__).resolve().parents[1],
    )
    return {
        "status": "frozen", "stage": "freeze", "freeze_path": str(freeze_path.resolve()),
        "models": {name: str(path.resolve()) for name, path in models.items()},
        "deployment_kind": confirmation["deployment_kind"], "freeze": freeze,
        "locked_holdout_evaluated": False, "official_test_evaluated": False,
    }


def run_experiment(
    dataset_root: str | Path,
    output_dir: str | Path,
    plan_path: str | Path,
    stage: str = "pilot",
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one declared stage; real training never runs on the user's laptop."""
    if stage not in {"pilot", "confirmation", "freeze"}:
        raise ValueError("Stage must be pilot, confirmation, or freeze")
    plan = load_plan(plan_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation-freeze.json").exists():
        raise RuntimeError("This run is already frozen; further training or reselection is prohibited")
    bind_source_snapshot(output)
    snapshot_path = output / "plan-snapshot.json"
    if snapshot_path.exists() and canonical_sha256(_load_json(snapshot_path)) != canonical_sha256(plan):
        raise ValueError("Experiment plan changed after the run started")
    if not snapshot_path.exists():
        _write(plan, snapshot_path)
    split_path = output / "splits.json"
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        if split_path.exists() and load_manifest(split_path)["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Supplied split manifest differs from the existing run")
        if not split_path.exists():
            save_manifest(manifest, split_path)
    elif split_path.exists():
        manifest = load_manifest(split_path)
    else:
        manifest = build_split_manifest(dataset_root, seed=plan["split_seed"], ratios=tuple(plan["split_track_ratios"]))
        save_manifest(manifest, split_path)
    # Source files are hash checked before training. A manifest is not merely a
    # user-provided list of labels that can change between paired runs.
    validate_manifest(manifest, dataset_root, verify_files=True)
    if stage in {"pilot", "confirmation"}:
        from .training import require_kaggle_gpu

        require_kaggle_gpu()
    if stage == "pilot":
        if (output / "pilot-report.json").exists():
            raise FileExistsError("Pilot report is already complete; proceed to confirmation instead of rerunning selection")
        return _pilot(dataset_root, output, plan, manifest)
    if stage == "confirmation":
        if (output / "confirmation-report.json").exists():
            raise FileExistsError("Paired confirmation is already complete; freeze instead of repeating selection")
        return _confirmation(dataset_root, output, plan, manifest)
    return _freeze(output, plan, manifest)
