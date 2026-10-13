"""Replay every published full-grid model on CPU without changing selection.

The existing downloader installs the exact descriptor under models/ and
original-base-run/. This verifier never trains, calibrates, exports, benchmarks,
creates a freeze, or ranks models using evaluation outcomes. Reference prediction
files are optional; without them it reports metrics-only agreement explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from knowledge_distillation.data import canonical_sha256, file_sha256, load_manifest, validate_manifest
from knowledge_distillation.evaluation import classification_metrics, records_for_evaluation, validate_freeze
from scripts.download_artifacts import checked_path, load_descriptor, verify_artifacts
from scripts.full_grid_study import inspect_base_run, validate_plan

SEEDS = (378, 379, 380)
GRID_IDS = tuple(f"T{t}-a{a:02d}" for t in (3, 5, 10) for a in (1, 3, 5))
METRIC_TOLERANCE = 1e-12
METRIC_KEYS = ("count", "correct", "accuracy", "accuracy_percent", "macro_f1", "balanced_accuracy", "per_class")
BINDING_KEYS = ("split", "model_sha256", "manifest_sha256", "training_manifest_sha256",
                "inventory_sha256", "evaluation_records_sha256", "freeze_sha256",
                "class_names", "preprocessing", "selection_permitted", "model_parameters")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=invalid)


def write(value, path):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def expected_roles():
    study = {"teacher"} | {f"baseline_seed_{s}" for s in SEEDS}
    study |= {f"grid_{identifier}_seed_{s}" for identifier in GRID_IDS for s in SEEDS}
    study |= {f"control_T5-a00_seed_{s}" for s in SEEDS}
    return study | {"deployment_student", "deployment_fp32", "deployment_int8"}


def checked_selection(freeze, summary):
    """Check the already-recorded decision; deliberately do not recompute it."""
    config = freeze["config"]
    training = config["training_config"]
    plan = training["plan"]
    validate_plan(plan)
    require(training["plan_sha256"] == canonical_sha256(plan), "Frozen full-grid plan hash differs")
    selection = config["selection"]
    require(summary.get("selection") == selection, "Release summary changed frozen selection")
    require(selection.get("status") == "completed" and selection.get("selection_split") == "validation"
            and selection.get("final_outcomes_used_for_selection") is False
            and selection.get("controls_eligible_for_selection") is False,
            "Only the recorded validation-only selection is permitted")
    candidate = selection.get("selected_candidate_id")
    require(candidate is None or candidate in GRID_IDS, "Deployment cannot be a diagnostic control")
    require(selection.get("deployment_seed") == 378 and training.get("deployment_seed") == 378
            and config.get("deployment_seed") == 378, "Deployment seed must remain 378")
    require(candidate == training.get("selected_candidate_id") == config.get("deployment_candidate_id")
            and selection["selected_configuration"] == training["selected_configuration"],
            "Frozen selection and deployment configuration differ")
    require(selection["deployment"]["selected_on"] == "validation"
            and selection["deployment"]["candidate_id"] == candidate
            and selection["deployment"]["seed"] == 378
            and selection["deployment_kind"] == ("distilled" if candidate else "baseline"),
            "Deployment differs from the validation decision")
    require(config.get("deployment_selection_uses_reference_outcomes") is False
            and training.get("final_outcomes_used_for_selection") is False
            and training.get("final_reselection_allowed") is False
            and training.get("controls_eligible_for_selection") is False
            and summary.get("evaluation_does_not_select_models") is True,
            "Reference evaluation must not select or modify models")
    candidates = selection.get("candidates", [])
    require(len(candidates) == 9 and {r["id"] for r in candidates} == set(GRID_IDS)
            and all(r.get("seeds") == list(SEEDS) for r in candidates),
            "The recorded selection must cover all nine three-seed candidates")
    selected_role = f"grid_{candidate}_seed_378" if candidate else "baseline_seed_378"
    require(training["deployment_model_name"] == selected_role, "Frozen selected model role differs")
    require(freeze["models"]["deployment_student"] == freeze["models"][selected_role]
            or all(freeze["models"]["deployment_student"][k] == freeze["models"][selected_role][k]
                   for k in ("bytes", "sha256")), "Deployment checkpoint is not the selected artifact")
    require(config["deployment_format"] in ("FP32", "INT8")
            and config["int8_acceptance"]["selected_format"] == config["deployment_format"]
            and summary["deployment"]["format"] == config["deployment_format"]
            and summary["deployment"]["candidate_id"] == candidate
            and summary["deployment"]["seed"] == 378
            and summary["deployment"]["int8_acceptance"] == config["int8_acceptance"],
            "Deployment format must remain the frozen validation acceptance decision")
    require(summary.get("training_freeze_sha256") == config["training_freeze_sha256"],
            "Training freeze binding differs")
    return plan, selected_role


def checked_metric(metric, role, freeze, manifest, records):
    require(metric.get("split") == "locked_holdout" and metric.get("selection_permitted") is False,
            "A reference must be frozen locked-holdout evaluation")
    require(metric.get("count") == len(records), "Evaluation denominator differs")
    expected = {"model_sha256": freeze["models"][role]["sha256"],
                "manifest_sha256": canonical_sha256(manifest),
                "training_manifest_sha256": canonical_sha256(manifest),
                "inventory_sha256": manifest["inventory_sha256"],
                "evaluation_records_sha256": canonical_sha256(records),
                "freeze_sha256": freeze["freeze_sha256"], "class_names": manifest["class_names"]}
    for key, value in expected.items():
        require(metric.get(key) == value, "Evaluation binding differs: " + role + "/" + key)
    preprocessing = metric.get("preprocessing", {})
    require(preprocessing == {"image_size": [96, 96], "channels": "RGB", "range": [0, 1],
                             "resize": "bilinear", "teacher_scaling": "Inside saved graph"},
            "Evaluation preprocessing differs")
    require(metric.get("model_parameters") == (2313067 if role == "teacher" else
            None if role in {"deployment_fp32", "deployment_int8"} else 34406),
            "Frozen architecture parameter count differs")
    for key in ("accuracy", "accuracy_percent", "macro_f1", "balanced_accuracy"):
        require(type(metric.get(key)) in (int, float) and math.isfinite(metric[key]), "Metric must be finite")
    require(0 <= metric["accuracy"] <= 1 and 0 <= metric["macro_f1"] <= 1
            and 0 <= metric["balanced_accuracy"] <= 1, "Metric range differs")
    require(len(metric.get("per_class", [])) == 43
            and [r.get("class_id") for r in metric["per_class"]] == list(range(43)),
            "All 43 per-class results must be present in class order")


def checked_predictions(directory, metric, records):
    rows = [read_line(line) for line in checked_path(directory, "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(rows) == len(records), "Saved predictions have a different denominator")
    for row, record in zip(rows, records, strict=True):
        require(row.get("id") == record["id"] and type(row.get("label")) is int
                and row["label"] == record["label"], "Saved prediction ID/label/order differs")
        require(type(row.get("prediction")) is int and 0 <= row["prediction"] < 43,
                "Saved predicted class is invalid")
        require(type(row.get("confidence")) in (int, float) and math.isfinite(row["confidence"])
                and 0 <= row["confidence"] <= 1, "Saved confidence must be finite and bounded")
    observed = classification_metrics([r["label"] for r in rows], [r["prediction"] for r in rows])
    require(all(equal_values(observed[key], metric[key]) for key in METRIC_KEYS),
            "Metrics disagree with saved predictions")
    confusion = read(checked_path(directory, "confusion.json"))
    require(confusion.get("matrix") == observed["confusion_matrix"]
            and confusion.get("class_names") == [str(i) for i in range(43)]
            and confusion.get("rows") == "true_class_id" and confusion.get("columns") == "predicted_class_id",
            "Confusion matrix disagrees with saved predictions")
    return rows


def read_line(line):
    return json.loads(line, object_pairs_hook=_unique, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def equal_values(left, right):
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= METRIC_TOLERANCE
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal_values(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal_values(left[k], right[k]) for k in left)
    return type(left) is type(right) and left == right


def compare_outputs(actual, reference, actual_rows, reference_rows=None):
    metric_differences = [key for key in METRIC_KEYS if not equal_values(actual[key], reference[key])]
    binding_differences = [key for key in BINDING_KEYS if actual.get(key) != reference.get(key)]
    result = {"metrics_match": not metric_differences, "metric_differences": metric_differences,
              "metric_absolute_tolerance": METRIC_TOLERANCE,
              "bindings_match": not binding_differences, "binding_differences": binding_differences,
              "prediction_comparison_available": reference_rows is not None,
              "confidence_comparison_policy": "Report-only; top-class confidence is not asserted bit-identical across CPU implementations."}
    if reference_rows is not None:
        require(len(actual_rows) == len(reference_rows), "Prediction comparison denominator differs")
        require(all(a["id"] == b["id"] and a["label"] == b["label"] for a, b in zip(actual_rows, reference_rows, strict=True)),
                "Prediction comparison IDs/labels/order differ")
        mismatches = [a["id"] for a, b in zip(actual_rows, reference_rows, strict=True) if a["prediction"] != b["prediction"]]
        differences = [abs(a["confidence"] - b["confidence"]) for a, b in zip(actual_rows, reference_rows, strict=True)]
        result.update(prediction_mismatches=len(mismatches), prediction_mismatch_ids=mismatches,
                      predictions_match=not mismatches, maximum_top_class_confidence_difference=max(differences),
                      mean_top_class_confidence_difference=sum(differences) / len(differences))
    passed = result["metrics_match"] and result["bindings_match"] and result.get("predictions_match", True)
    result["status"] = ("passed_saved_model_outcomes" if reference_rows is not None else "passed_metrics_only") if passed else "failed_comparison"
    return result


def preflight(descriptor_path, artifact_root, manifest_path, freeze_path, summary_path,
              dataset_root, reference_root=None, source_root=ROOT, training_freeze_path=None,
              selection_path=None):
    descriptor = load_descriptor(descriptor_path)
    installed = verify_artifacts(descriptor, artifact_root)
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, dataset_root, verify_files=True)
    records = records_for_evaluation(manifest, "locked_holdout")
    require(len(records) == 5850 and len({r["track_id"] for r in records}) == 195
            and manifest["num_classes"] == 43 and manifest["image_size"] == 96
            and manifest["class_names"] == [str(i) for i in range(43)], "Declared full-grid corpus differs")
    freeze = validate_freeze(freeze_path, manifest, source_root=source_root)
    summary = read(summary_path)
    require(summary.get("status") == "completed" and summary.get("freeze_sha256") == freeze["freeze_sha256"]
            and summary.get("manifest_sha256") == manifest["manifest_sha256"]
            and summary.get("counts") == manifest["counts"], "Release summary provenance differs")
    require(descriptor.get("release_freeze_sha256") == freeze["freeze_sha256"], "Download descriptor binds another release")
    roles = descriptor.get("model_roles", {})
    require(set(roles) == set(freeze["models"]) == set(summary["models"]) == expected_roles()
            == set(summary["evaluations"]["locked_holdout"]), "Every one of the 37 frozen roles is required")
    require(len(set(roles.values())) == len(roles), "Two roles cannot share a delivery pathname")
    artifacts = {a["path"]: a for a in descriptor["artifacts"]}
    require(set(artifacts) == {"models", "original-base-run"}
            and all(a.get("kind") == "zip" for a in artifacts.values()), "Expected the two full-grid archive layouts")
    model_inventory = {r["path"]: {k: r[k] for k in ("bytes", "sha256")} for r in artifacts["models"]["inventory"]}
    require(set(model_inventory) == set(roles.values()), "Model delivery inventory differs from role mapping")
    plan, selected_role = checked_selection(freeze, summary)
    training_path = training_freeze_path or freeze_path.parent / "training-evaluation-freeze.json"
    selected_path = selection_path or freeze_path.parent / "selection-report.json"
    training_freeze = validate_freeze(training_path, manifest, source_root=source_root)
    require(training_freeze["freeze_sha256"] == freeze["config"]["training_freeze_sha256"]
            and training_freeze["config"] == freeze["config"]["training_config"],
            "Published training freeze differs from joint release binding")
    study_roles = expected_roles() - {"deployment_student", "deployment_fp32", "deployment_int8"}
    require(set(training_freeze["models"]) == study_roles
            and all(training_freeze["models"][r] == freeze["models"][r] for r in study_roles),
            "Every released study model must match its pre-evaluation training freeze")
    selection_binding = training_freeze["config"]["selection_report_file"]
    require(selected_path.stat().st_size == selection_binding["bytes"]
            and file_sha256(selected_path) == selection_binding["sha256"]
            and read(selected_path) == freeze["config"]["selection"],
            "The original validation selection report bytes changed")
    require(descriptor.get("base_inventory_sha256") == plan["base_bindings"]["artifact_inventory_sha256"],
            "Starting-evidence descriptor binds another study")
    base_inventory = {r["path"]: {k: r[k] for k in ("bytes", "sha256")} for r in artifacts["original-base-run"]["inventory"]}
    require(canonical_sha256(base_inventory) == descriptor["base_inventory_sha256"], "Starting-evidence inventory differs")
    base = inspect_base_run(checked_path(artifact_root, "original-base-run"), plan)
    require(base["manifest"] == manifest and base["inventory"] == base_inventory,
            "Imported original study or split changed")
    training = freeze["config"]["training_config"]
    for path, binding in ((source_root / "scripts/full_grid_study.py", training["runner_file"]),):
        require(path.stat().st_size == binding["bytes"] and file_sha256(path) == binding["sha256"],
                "Frozen study runner source changed")
    require(file_sha256(source_root / "scripts/evaluate_full_grid.py") == freeze["config"]["evaluator_sha256"],
            "Frozen reference evaluator source changed")
    require(canonical_sha256(read(source_root / "configs/full-grid-plan.json")) == training["plan_sha256"],
            "Published full-grid plan differs from frozen plan")
    models, reference_rows = {}, {}
    for role in sorted(roles):
        binding = freeze["models"][role]
        require(model_inventory[roles[role]] == {k: binding[k] for k in ("bytes", "sha256")}
                and all(summary["models"][role][k] == binding[k] for k in ("bytes", "sha256")),
                "Frozen measured and delivered bytes differ: " + role)
        model = checked_path(artifact_root / "models", roles[role])
        require(model.suffix == (".tflite" if role in {"deployment_fp32", "deployment_int8"} else ".keras"),
                "Model format differs from frozen role")
        metric = summary["evaluations"]["locked_holdout"][role]
        checked_metric(metric, role, freeze, manifest, records)
        if reference_root is not None:
            directory = checked_path(reference_root, role)
            require(read(checked_path(directory, "metrics.json")) == metric, "Reference directory differs from release summary: " + role)
            reference_rows[role] = checked_predictions(directory, metric, records)
        models[role] = model
    return {"descriptor": descriptor, "installed": installed, "manifest": manifest, "records": records,
            "freeze": freeze, "summary": summary, "models": models, "reference_rows": reference_rows,
            "selected_role": selected_role}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("descriptor", "artifact-root", "manifest", "freeze", "reference-summary", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference-evaluations", type=Path, help="Directory containing the 37 frozen role subdirectories")
    parser.add_argument("--training-freeze", type=Path, help="Defaults to training-evaluation-freeze.json beside the release freeze")
    parser.add_argument("--selection-report", type=Path, help="Defaults to selection-report.json beside the release freeze")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true", help="Verify all inventories/bindings without TensorFlow/model execution")
    args = parser.parse_args()
    require(args.threads > 0 and args.batch_size > 0, "Threads and batch size must be positive")
    require(not args.output.exists(), "Use a new verification output directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("TF_NUM_INTRAOP_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    state = preflight(args.descriptor, args.artifact_root, args.manifest, args.freeze,
                      args.reference_summary, args.data, args.reference_evaluations,
                      training_freeze_path=args.training_freeze, selection_path=args.selection_report)
    inputs = {"descriptor": args.descriptor, "manifest": args.manifest, "release_freeze": args.freeze,
              "reference_summary": args.reference_summary,
              "training_freeze": args.training_freeze or args.freeze.parent / "training-evaluation-freeze.json",
              "selection_report": args.selection_report or args.freeze.parent / "selection-report.json",
              "study_runner": ROOT / "scripts/full_grid_study.py", "reference_evaluator": ROOT / "scripts/evaluate_full_grid.py",
              "study_plan": ROOT / "configs/full-grid-plan.json", "verifier": Path(__file__)}
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    require(not any(args.output.resolve().is_relative_to(root.resolve()) for root in
                    (args.artifact_root, args.data, args.reference_evaluations) if root is not None),
            "Verification output must be separate from immutable artifacts/data/reference outputs")
    args.output.mkdir(parents=True)
    receipt = {"schema_version": 1, "status": "passed_inventory_preflight",
               "platform": platform.system(), "python_version": platform.python_version(),
               "created_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": state["freeze"]["freeze_sha256"],
               "manifest_sha256": state["manifest"]["manifest_sha256"],
               "evaluation_records_sha256": canonical_sha256(state["records"]),
               "source_sha256": state["freeze"]["source_sha256"], "verifier_sha256": file_sha256(Path(__file__)),
               "input_file_sha256": input_hashes,
               "descriptor_file_sha256": file_sha256(args.descriptor), "reference_summary_sha256": file_sha256(args.reference_summary),
               "study_keras_roles": 34, "deployment_roles": 3, "total_role_evaluations": 37,
               "planned_count_per_role": 5850, "evaluation_cpu_threads": args.threads, "batch_size": args.batch_size,
               "selected_model_role_unchanged": state["selected_role"], "deployment_format_unchanged": state["freeze"]["config"]["deployment_format"],
               "installed_artifacts": state["installed"], "model_inference_performed": False, "completed_role_evaluations": 0,
               "training_performed": False, "calibration_refitted": False, "models_reselected": False,
               "official_test_replayed": False, "reference_prediction_comparison_available": args.reference_evaluations is not None,
               "historical_exposure": state["freeze"]["config"]["reference_evaluation_exposure"], "models": {}}
    write(receipt, args.output / "verification.json")
    if args.preflight_only:
        print(json.dumps({"status": receipt["status"], "models_evaluated": 0}), flush=True)
        return
    from knowledge_distillation.evaluation import evaluate_keras, evaluate_tflite
    from knowledge_distillation.runtime import configure_cpu, tensorflow
    receipt["runtime"] = configure_cpu(args.threads)
    try:
        for index, (role, model) in enumerate(state["models"].items(), 1):
            print(f"FULL GRID CPU REPLAY {index}/37 {role}", flush=True)
            destination = args.output / "evaluations" / "locked_holdout" / role
            evaluator = evaluate_tflite if model.suffix == ".tflite" else evaluate_keras
            receipt.update(status="running", model_inference_performed=True, active_model_role=role)
            write(receipt, args.output / "verification.json")
            metric = evaluator(model, state["manifest"], args.data, "locked_holdout", destination,
                               freeze_path=args.freeze, threads=args.threads, batch_size=args.batch_size)
            checked_metric(metric, role, state["freeze"], state["manifest"], state["records"])
            rows = checked_predictions(destination, metric, state["records"])
            comparison = compare_outputs(metric, state["summary"]["evaluations"]["locked_holdout"][role], rows,
                                         state["reference_rows"].get(role))
            receipt["models"][role] = {"metrics": metric, "comparison": comparison,
                                       "predictions_sha256": file_sha256(destination / "predictions.jsonl")}
            receipt.update(status="running", model_inference_performed=True, completed_role_evaluations=index)
            write(receipt, args.output / "verification.json")
            tensorflow().keras.backend.clear_session()
            gc.collect()
        verify_artifacts(state["descriptor"], args.artifact_root)
        validate_manifest(state["manifest"], args.data, verify_files=True)
        validate_freeze(args.freeze, state["manifest"], source_root=ROOT)
        require({name: file_sha256(path) for name, path in inputs.items()} == input_hashes,
                "Verifier inputs or study/verification scripts changed during replay")
        passed = all(r["comparison"]["status"].startswith("passed_") for r in receipt["models"].values())
        receipt["status"] = ("passed_saved_model_outcomes" if args.reference_evaluations else "passed_metrics_only") if passed else "failed_comparison"
        receipt["source_data_artifacts_exact_before_after"] = True
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write(receipt, args.output / "verification.json")
        print(json.dumps({"status": receipt["status"], "models_evaluated": len(receipt["models"])}), flush=True)
        if not passed:
            raise SystemExit(1)
    except Exception as exc:
        receipt.update(status="failed_execution", error_type=type(exc).__name__, error=str(exc),
                       completed_utc=datetime.now(timezone.utc).isoformat())
        write(receipt, args.output / "verification.json")
        raise


if __name__ == "__main__":
    main()
