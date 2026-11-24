"""CLI entrypoint for L4 Autonomous Driving Semantic Segmentation Knowledge Distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark import benchmark_model
from .data import (
    build_bdd100k_manifest,
    build_cityscapes_manifest,
    build_kitti_manifest,
    create_synthetic_dataset,
    load_manifest,
    records_for_split,
    save_manifest,
)
from .evaluation import create_evaluation_freeze, evaluate_artifact
from .export import assess_int8_acceptance, export_student_onnx
from .losses import VALID_KD_METHODS
from .models import STUDENT_ARCHS, TEACHER_VARIANTS
from .training import (
    create_shared_initial_student_weights,
    detect_runtime_metadata,
    resolve_device,
    train_student,
    train_teacher,
    write_json,
)
from .visualize import render_prediction_panel


def _parse_hw(spec: str) -> tuple[int, int]:
    parts = spec.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected HxW or H,W image size, got {spec!r}")
    return int(parts[0]), int(parts[1])


def cmd_doctor(_args: argparse.Namespace) -> None:
    device = resolve_device("auto")
    info = detect_runtime_metadata(device, precision="bf16" if device.type == "cuda" else "fp32")
    optional = {}
    for pkg in ("timm", "transformers", "onnx", "onnxruntime"):
        try:
            mod = __import__(pkg)
            optional[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            optional[pkg] = "missing"
    info["packages"] = optional
    print(json.dumps(info, indent=2))


def cmd_prepare_data(args: argparse.Namespace) -> None:
    dataset = args.dataset.lower()
    if dataset == "bdd100k":
        manifest = build_bdd100k_manifest(
            args.data, seed=args.seed, val_ratio=args.val_ratio, hash_files=not args.no_hash
        )
    elif dataset == "cityscapes":
        manifest = build_cityscapes_manifest(
            args.data, seed=args.seed, val_ratio=args.val_ratio, hash_files=not args.no_hash
        )
    elif dataset == "kitti":
        manifest = build_kitti_manifest(args.data, hash_files=not args.no_hash)
    elif dataset == "synthetic":
        h, w = _parse_hw(args.image_size)
        manifest = create_synthetic_dataset(
            args.data,
            num_train=args.synthetic_train,
            num_val=args.synthetic_val,
            num_eval=args.synthetic_eval,
            image_size=(h, w),
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset!r}")

    if args.output:
        save_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "counts": manifest["counts"],
                "manifest_sha256": manifest["manifest_sha256"],
                "output": str(Path(args.output).resolve()) if args.output else None,
            },
            indent=2,
        )
    )


def cmd_init_student(args: argparse.Namespace) -> None:
    info = create_shared_initial_student_weights(
        arch=args.arch,
        output_path=args.output,
        seed=args.seed,
        pretrained=args.pretrained,
        fpn_dim=args.fpn_dim,
    )
    print(json.dumps(info, indent=2))


def cmd_train_teacher(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest, dataset_root=args.data, verify_files=args.verify_files)
    h, w = _parse_hw(args.image_size)
    cfg: dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [h, w],
        "learning_rate": args.lr,
        "precision": args.precision,
        "num_workers": args.workers,
        "pretrained": not args.no_pretrained,
    }
    res = train_teacher(
        dataset_root=args.data,
        manifest=manifest,
        output_dir=args.output,
        variant=args.variant,
        seed=args.seed,
        config=cfg,
        device_spec=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(res["report"], indent=2))


def cmd_train_student(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest, dataset_root=args.data, verify_files=args.verify_files)
    h, w = _parse_hw(args.image_size)
    cfg: dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [h, w],
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "lambda_logit_kd": args.lambda_logit_kd,
        "lambda_feat_kd": args.lambda_feat_kd,
        "safety_boost": args.safety_boost,
        "precision": args.precision,
        "num_workers": args.workers,
        "pretrained_backbone": args.pretrained_backbone,
        "compile_model": args.compile,
        "qat_epochs": args.qat_epochs,
        "fpn_dim": args.fpn_dim,
    }
    res = train_student(
        dataset_root=args.data,
        manifest=manifest,
        output_dir=args.output,
        arch=args.arch,
        method=args.method,
        seed=args.seed,
        teacher_path=args.teacher,
        initial_weights_path=args.init_weights,
        config=cfg,
        device_spec=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(res["report"], indent=2))


def cmd_freeze(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    models: dict[str, str] = {}
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"Expected --model label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        models[label.strip()] = path.strip()
    freeze = create_evaluation_freeze(manifest, models, args.output, plan_path=args.plan)
    print(json.dumps(freeze, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest, dataset_root=args.data, verify_files=args.verify_files)
    h, w = _parse_hw(args.image_size)
    metrics = evaluate_artifact(
        model_path=args.model,
        manifest=manifest,
        dataset_root=args.data,
        split=args.split,
        image_size=(h, w),
        batch_size=args.batch_size,
        num_workers=args.workers,
        device_spec=args.device,
        threads=args.threads,
        freeze_path=args.freeze,
    )
    if args.output:
        write_json(metrics, args.output)
    print(json.dumps(metrics, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest) if args.manifest else None
    h, w = _parse_hw(args.image_size)
    report = export_student_onnx(
        checkpoint_path=args.model,
        output_dir=args.output,
        manifest=manifest,
        dataset_root=args.data,
        image_size=(h, w),
        calibration_count=args.calibration_count,
        seed=args.seed,
        export_int8=not args.fp32_only,
    )
    print(json.dumps(report, indent=2))


def cmd_benchmark(args: argparse.Namespace) -> None:
    h, w = _parse_hw(args.image_size)
    result = benchmark_model(
        model_path=args.model,
        image_size=(h, w),
        batch_size=args.batch_size,
        warmup=args.warmup,
        runs=args.runs,
        device_spec=args.device,
        precision=args.precision,
        threads=args.threads,
    )
    if args.output:
        write_json(result, args.output)
    print(json.dumps(result, indent=2))


def cmd_int8_gate(args: argparse.Namespace) -> None:
    fp32_m = json.loads(Path(args.fp32_metrics).read_text(encoding="utf-8"))
    int8_m = json.loads(Path(args.int8_metrics).read_text(encoding="utf-8"))
    fp32_b = json.loads(Path(args.fp32_benchmark).read_text(encoding="utf-8"))
    int8_b = json.loads(Path(args.int8_benchmark).read_text(encoding="utf-8"))
    verdict = assess_int8_acceptance(
        fp32_m,
        int8_m,
        fp32_b,
        int8_b,
        max_miou_drop_pp=args.max_miou_drop_pp,
        max_safety_miou_drop_pp=args.max_safety_drop_pp,
        minimum_speedup=args.min_speedup,
    )
    if args.output:
        write_json(verdict, args.output)
    print(json.dumps(verdict, indent=2))


def cmd_visualize(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    h, w = _parse_hw(args.image_size)
    records = records_for_split(manifest, args.split)
    models: dict[str, str] = {}
    for spec in args.model:
        label, path = spec.split("=", 1)
        models[label.strip()] = path.strip()
    out = render_prediction_panel(
        records,
        args.data,
        models,
        args.output,
        image_size=(h, w),
        device_spec=args.device,
        max_samples=args.samples,
    )
    print(json.dumps({"figure": str(out.resolve())}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seg-kd", description="L4 Autonomous Driving Semantic Segmentation KD CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Inspect GPU/CPU runtime, BF16 support, and package versions")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("prepare-data", help="Build validated dataset manifest (bdd100k, cityscapes, kitti, synthetic)")
    p.add_argument("--dataset", required=True, choices=["bdd100k", "cityscapes", "kitti", "synthetic"])
    p.add_argument("--data", required=True, help="Dataset root directory")
    p.add_argument("--output", help="Output manifest JSON path")
    p.add_argument("--seed", type=int, default=378)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--no-hash", action="store_true", help="Skip SHA-256 hashing of image/mask files")
    p.add_argument("--image-size", default="128x256", help="Synthetic image HxW")
    p.add_argument("--synthetic-train", type=int, default=16)
    p.add_argument("--synthetic-val", type=int, default=6)
    p.add_argument("--synthetic-eval", type=int, default=6)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("init-student", help="Create shared initial student weights for paired seed comparisons")
    p.add_argument("--arch", required=True, choices=sorted(STUDENT_ARCHS))
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=378)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--fpn-dim", type=int, default=None)
    p.set_defaults(func=cmd_init_student)

    p = sub.add_parser("train-teacher", help="Fine-tune SegFormer teacher on L4 (BF16 AMP)")
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", default="segformer_b4", choices=sorted(TEACHER_VARIANTS))
    p.add_argument("--seed", type=int, default=378)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", default="512x1024")
    p.add_argument("--lr", type=float, default=6e-5)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--verify-files", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_train_teacher)

    p = sub.add_parser("train-student", help="Train Edge CNN student (baseline, pixel_kl, cwd, cwd_feat, cwd_safety)")
    p.add_argument("--data", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arch", default="mobilenetv4_small", choices=sorted(STUDENT_ARCHS))
    p.add_argument("--method", default="cwd", choices=list(VALID_KD_METHODS))
    p.add_argument("--teacher", default=None)
    p.add_argument("--init-weights", default=None)
    p.add_argument("--seed", type=int, default=378)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", default="512x1024")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--lambda-logit-kd", type=float, default=3.0)
    p.add_argument("--lambda-feat-kd", type=float, default=1.5)
    p.add_argument("--safety-boost", type=float, default=2.0)
    p.add_argument("--qat-epochs", type=int, default=0)
    p.add_argument("--fpn-dim", type=int, default=None)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--pretrained-backbone", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--verify-files", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_train_student)

    p = sub.add_parser("freeze", help="Create cryptographic freeze JSON before locked evaluation")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", action="append", required=True, help="label=path (repeatable)")
    p.add_argument("--plan", default=None)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_freeze)

    p = sub.add_parser("evaluate", help="Evaluate PyTorch (.pt) or ONNX (.onnx) artifact on a manifest split")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="locked_eval")
    p.add_argument("--image-size", default="512x1024")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--freeze", default=None)
    p.add_argument("--verify-files", action="store_true")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("export", help="Export plain student to ONNX FP32 and Static INT8 (QDQ)")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--image-size", default="512x1024")
    p.add_argument("--calibration-count", type=int, default=64)
    p.add_argument("--seed", type=int, default=378)
    p.add_argument("--fp32-only", action="store_true")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("benchmark", help="Benchmark CPU/CUDA latency (p50/p95 ms) and FPS")
    p.add_argument("--model", required=True)
    p.add_argument("--image-size", default="512x1024")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("int8-gate", help="Apply validation-only INT8 acceptance gate")
    p.add_argument("--fp32-metrics", required=True)
    p.add_argument("--int8-metrics", required=True)
    p.add_argument("--fp32-benchmark", required=True)
    p.add_argument("--int8-benchmark", required=True)
    p.add_argument("--max-miou-drop-pp", type=float, default=1.0)
    p.add_argument("--max-safety-drop-pp", type=float, default=1.5)
    p.add_argument("--min-speedup", type=float, default=1.20)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_int8_gate)

    p = sub.add_parser("visualize", help="Render road-scene overlay comparison grid")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="locked_eval")
    p.add_argument("--model", action="append", required=True, help="label=path (repeatable)")
    p.add_argument("--image-size", default="512x1024")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_visualize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
