"""Create deployment evidence before final inference, without training or retuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-host", type=Path)
    args = p.parse_args()
    for key in ("TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(args.source.resolve()))
    from knowledge_distillation.runtime import configure_cpu, file_sha256
    configure_cpu(1)
    from knowledge_distillation.data import load_manifest
    from knowledge_distillation.experiments import verify_completed_result
    from knowledge_distillation.evaluation import (evaluate_keras, evaluate_tflite, freeze_evaluation,
                                                 validate_freeze, build_official_test_manifest)
    from knowledge_distillation.export import export_student, assess_int8_acceptance
    from knowledge_distillation.benchmark import benchmark_models
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("Release output must be new; preserve previous evidence and never reselect after final access")
    out.mkdir(parents=True)
    read = lambda name: json.loads((run / name).read_text(encoding="utf-8"))
    completion = read("kernel-completion.json")
    if completion.get("status") != "completed" or completion.get("locked_holdout_evaluated") is not False:
        raise ValueError("Expected a completed, not yet finally evaluated training run")
    manifest = load_manifest(run / "splits.json")
    pilot, confirmation = read("pilot-report.json"), read("confirmation-report.json")
    training_freeze = run / "evaluation-freeze.json"
    validate_freeze(training_freeze, manifest, source_root=args.source)
    selected = {"teacher": pilot["selected_teacher"]}
    for pair in confirmation["pairs"]:
        suffix = "" if pair["seed"] == 378 else f"_seed_{pair['seed']}"
        selected["baseline" + suffix] = pair["baseline"]
        if pair["distilled"] is not None:
            selected["distilled" + suffix] = pair["distilled"]
    models, runtime = {}, out / "runtime-models"
    runtime.mkdir()
    for name, result in selected.items():
        verify_completed_result(result, run, manifest)
        destination = runtime / (name.replace("_", "-") + ".keras")
        shutil.copyfile(run / result["model_path"], destination)
        models[name] = destination
        if file_sha256(destination) != result["report"]["model_sha256"]:
            raise ValueError("Copied checkpoint checksum mismatch")
    kind = confirmation["deployment_kind"]
    student = runtime / "student.keras"
    shutil.copyfile(models[kind], student)
    print("RELEASE EXPORT: training-only calibration", flush=True)
    export = export_student(student, manifest, args.data, out / "export", calibration_count=256,
                            seed=378, freeze_path=training_freeze)
    for name, filename in (("fp32", "student-fp32.tflite"), ("int8", "student-int8.tflite")):
        shutil.copyfile(out / "export" / filename, runtime / filename)
        models[name] = runtime / filename
    print("RELEASE VALIDATION: quantization selection only", flush=True)
    validation = {name: evaluate_tflite(models[name], manifest, args.data, "validation", out / "validation" / name)
                  for name in ("fp32", "int8")}
    benchmark_models_map = {k: models[k] for k in ("teacher", "baseline", "distilled", "fp32", "int8") if k in models}
    benchmark = benchmark_models(benchmark_models_map, manifest, args.data, out / "cpu-benchmark",
                                 sample_count=128, warmup=10, repetitions=3, threads=1)
    acceptance = assess_int8_acceptance(validation["fp32"], validation["int8"],
                                        benchmark["models"]["fp32"], benchmark["models"]["int8"])
    save(acceptance, out / "int8-acceptance.json")
    config = {**read("evaluation-freeze.json")["config"],
              "training_freeze_sha256": read("evaluation-freeze.json")["freeze_sha256"],
              "deployment_format": acceptance["selected_format"], "int8_acceptance": acceptance,
              "deployment_selection_uses_final_outcomes": False,
              "export_report_sha256": file_sha256(out / "export/export-report.json"),
              "cpu_benchmark_sha256": file_sha256(out / "cpu-benchmark/benchmark.json"),
              "quantization_validation_model_sha256": {k: v["model_sha256"] for k, v in validation.items()}}
    freeze_path = out / "evaluation-freeze.json"
    freeze = freeze_evaluation({**models, "student": student}, manifest, config, freeze_path, source_root=args.source)
    print("RELEASE FROZEN", freeze["freeze_sha256"], flush=True)
    evaluation = {"locked_holdout": {}, "official_test": {}}
    for name, model in models.items():
        print("RELEASE LOCKED EVALUATION", name, flush=True)
        function = evaluate_tflite if model.suffix == ".tflite" else evaluate_keras
        evaluation["locked_holdout"][name] = function(model, manifest, args.data, "locked_holdout",
                                                     out / "evaluations/locked_holdout" / name, freeze_path=freeze_path)
    official = build_official_test_manifest(args.data, manifest, freeze_path)
    save(official, out / "official-test-manifest.json")
    for name in ("teacher", "baseline", "distilled", "fp32", "int8"):
        if name not in models:
            continue
        print("RELEASE HISTORICAL OFFICIAL AUDIT", name, flush=True)
        model = models[name]
        function = evaluate_tflite if model.suffix == ".tflite" else evaluate_keras
        evaluation["official_test"][name] = function(model, official, args.data, "official_test",
                                                   out / "evaluations/official_test" / name, freeze_path=freeze_path)
    summary = {"schema_version": 1, "protocol": "gtsrb-track-disjoint-kaggle-v1",
               "training_location": "Kaggle GPU only", "final_evaluation_device": "CPU",
               "deployment": {"kind": kind, "seed": 378, "format": acceptance["selected_format"],
                              "int8_acceptance": acceptance}, "confirmation": confirmation,
               "evaluations": evaluation, "benchmark": benchmark, "export": export,
               "freeze_sha256": freeze["freeze_sha256"],
               "manifest_sha256": manifest["manifest_sha256"], "counts": manifest["counts"],
               "historical_exposure": config["historical_exposure"],
               "models": {k: {"filename": v.name, "sha256": file_sha256(v), "bytes": v.stat().st_size}
                          for k, v in {**models, "student": student}.items()}}
    if args.cpu_host:
        summary["cpu_host"] = json.loads(args.cpu_host.read_text(encoding="utf-8-sig"))
    save(summary, out / "release-summary.json")
    print("RELEASE COMPLETED", json.dumps(summary["deployment"]), flush=True)


if __name__ == "__main__":
    main()
