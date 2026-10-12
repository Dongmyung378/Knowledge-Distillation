"""Command-line entry points for portable evaluation and Kaggle training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def _named_models(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError("Use distinct --model NAME=PATH entries")
        result[name] = Path(path)
    return result


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def doctor(model: Path | None = None, descriptor: Path | None = None, artifact_root: Path | None = None) -> dict:
    from scripts.setup_runtime import PROJECT_ROOT
    problems = []
    versions = {}
    for requirement in (PROJECT_ROOT / "requirements-runtime.txt").read_text(encoding="utf-8").splitlines():
        if not requirement.strip() or requirement.startswith("#"):
            continue
        name, wanted = requirement.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
            versions[name] = actual
            if actual != wanted:
                problems.append(f"{name}: expected {wanted}, found {actual}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name}: not installed")
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32 or sys.prefix == sys.base_prefix:
        problems.append("Use the isolated 64-bit Python 3.12 launcher runtime")
    configuration = Path(sys.prefix) / "pyvenv.cfg"
    if not configuration.is_file() or "include-system-site-packages = true" in configuration.read_text(encoding="utf-8").lower():
        problems.append("System site packages must be disabled")
    tensorflow = {}
    try:
        from .runtime import configure_cpu
        configure_cpu(threads=1)
        import tensorflow as tf
        value = float(tf.reduce_sum(tf.constant([1.0, 2.0])).numpy())
        tensorflow = {"cpu_tensor_result": value, "visible_gpus": len(tf.config.list_physical_devices("GPU"))}
        if value != 3.0 or tensorflow["visible_gpus"]:
            problems.append("The default runtime must perform the CPU check with GPUs hidden")
    except (ImportError, OSError, RuntimeError) as error:
        problems.append(f"TensorFlow CPU import failed: {error}")
    if descriptor is not None:
        from scripts.download_artifacts import verify_artifacts
        try:
            verify_artifacts(descriptor, artifact_root or PROJECT_ROOT / "artifacts" / "runtime")
        except (OSError, ValueError) as error:
            problems.append(str(error))
    if model is not None:
        from .runtime import load_model
        try:
            load_model(model)
        except (OSError, ValueError, RuntimeError) as error:
            problems.append(f"Model check failed: {error}")
    return {"healthy": not problems, "python": sys.version.split()[0], "isolated": sys.prefix != sys.base_prefix,
            "device": "cpu", "versions": versions, "tensorflow": tensorflow, "problems": problems}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Verify isolated dependencies and CPU operation")
    check.add_argument("--model", type=Path)
    check.add_argument("--artifact-manifest", type=Path)
    check.add_argument("--artifact-root", type=Path)

    download = commands.add_parser("download", help="Download and verify published models")
    download.add_argument("--manifest", type=Path, default=Path("reports/artifact-download.json"))
    download.add_argument("--output", type=Path, default=Path("artifacts/runtime"))
    download.add_argument("--verify-only", action="store_true")

    fetch = commands.add_parser("fetch-data", help="Download and safely extract the hash-pinned GTSRB dataset")
    fetch.add_argument("--source-manifest", type=Path, default=Path("reports/dataset-download.json"))
    fetch.add_argument("--output", type=Path, default=Path(".local/data/gtsrb-v1"))
    fetch.add_argument("--archive", type=Path, help="Reuse an already downloaded archive after size/SHA verification")
    fetch.add_argument("--verify-only", action="store_true")

    prepare = commands.add_parser("prepare-data", help="Create track-disjoint train/validation/locked_holdout manifest")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=378)

    predict = commands.add_parser("predict", help="Predict one RGB image with a saved Keras or TFLite model")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--class-map", type=Path)
    predict.add_argument("--sha256")

    evaluate = commands.add_parser("evaluate", help="Evaluate a saved model on a fixed split")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "locked_holdout", "official_test"), default="locked_holdout")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--freeze", type=Path)
    evaluate.add_argument("--batch-size", type=int, default=64)

    freeze = commands.add_parser("freeze", help="Bind model hashes, data inventory and configuration before final evaluation")
    freeze.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--source-root", type=Path)

    audit = commands.add_parser("official-test-manifest", help="Prepare the historically exposed official test audit after freezing")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--training-manifest", type=Path, required=True)
    audit.add_argument("--freeze", type=Path, required=True)
    audit.add_argument("--csv", type=Path)
    audit.add_argument("--output", type=Path, required=True)

    benchmark = commands.add_parser("benchmark", help="Measure warmed CPU latency and throughput")
    benchmark.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    benchmark.add_argument("--manifest", type=Path, required=True)
    benchmark.add_argument("--data", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--split", choices=("train", "validation", "locked_holdout", "official_test"), default="validation")
    benchmark.add_argument("--samples", type=int, default=128)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--repetitions", type=int, default=3)
    benchmark.add_argument("--threads", type=int, default=1)
    benchmark.add_argument("--freeze", type=Path)

    export = commands.add_parser("export", help="Export FP32 and INT8 TFLite with training-only calibration")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--data", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--calibration-count", type=int, default=128)
    export.add_argument("--seed", type=int, default=378)
    export.add_argument("--freeze", type=Path)

    train = commands.add_parser("train", help="Run a bounded stage on Kaggle with a visible GPU")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--plan", type=Path, required=True)
    train.add_argument("--stage", choices=("pilot", "confirmation", "freeze"), required=True)
    train.add_argument("--manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        if args.command == "doctor":
            result = doctor(args.model, args.artifact_manifest, args.artifact_root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return int(not result["healthy"])
        if args.command == "download":
            from scripts.download_artifacts import install_artifacts, verify_artifacts
            result = verify_artifacts(args.manifest, args.output) if args.verify_only else install_artifacts(args.manifest, args.output)
        elif args.command == "fetch-data":
            from scripts.download_data import install_dataset, verify_dataset
            result = verify_dataset(args.source_manifest, args.output) if args.verify_only else install_dataset(args.source_manifest, args.output, archive=args.archive)
        elif args.command == "prepare-data":
            from .data import build_split_manifest, save_manifest
            result = build_split_manifest(args.data, seed=args.seed)
            save_manifest(result, args.output)
            result = {"manifest": str(args.output), "counts": result.get("counts", result.get("split_counts")), "manifest_sha256": result.get("manifest_sha256")}
        elif args.command == "predict":
            from .runtime import infer_image
            result = infer_image(args.model, args.image, class_map=args.class_map, expected_sha256=args.sha256)
        elif args.command == "evaluate":
            from .evaluation import evaluate_keras, evaluate_tflite
            function = evaluate_tflite if args.model.suffix.lower() == ".tflite" else evaluate_keras
            result = function(args.model, args.manifest, args.data, args.split, args.output, freeze_path=args.freeze, batch_size=args.batch_size)
        elif args.command == "freeze":
            from .evaluation import freeze_evaluation
            result = freeze_evaluation(_named_models(args.model), args.manifest, _json(args.config), args.output, source_root=args.source_root)
        elif args.command == "official-test-manifest":
            from .evaluation import build_official_test_manifest
            result = build_official_test_manifest(args.data, args.training_manifest, args.freeze, csv_path=args.csv)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result = {"manifest": str(args.output), "evaluation_scope": "historically_exposed_official_test_audit"}
        elif args.command == "benchmark":
            from .benchmark import benchmark_models
            result = benchmark_models(_named_models(args.model), args.manifest, args.data, args.output,
                                      split=args.split, sample_count=args.samples, warmup=args.warmup,
                                      repetitions=args.repetitions, threads=args.threads, freeze_path=args.freeze)
        elif args.command == "export":
            from .export import export_student
            result = export_student(args.model, args.manifest, args.data, args.output,
                                    calibration_count=args.calibration_count, seed=args.seed, input_split="train", freeze_path=args.freeze)
        elif args.command == "train":
            from .experiments import run_experiment
            result = run_experiment(args.data, args.output, args.plan, args.stage, manifest_path=args.manifest)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(json.dumps({"success": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
