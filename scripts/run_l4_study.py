"""Phased, resumable L4 study orchestrator for Driving Semantic Segmentation KD.

Stages:
  1. teacher  - Fine-tune SegFormer teacher on L4 (BF16)
  2. ablation - Paired multi-seed comparison (baseline, pixel_kl, cwd, cwd_feat, cwd_safety)
  3. pareto   - Multi-size Edge CNN Pareto frontier (deployment seed)
  4. export   - Select best KD method on validation, export FP32 & INT8 ONNX, run INT8 gate
  5. freeze   - Cryptographically freeze manifest + models before locked_eval
  6. eval     - Evaluate frozen models on locked_eval (and optional KITTI zero-shot) + benchmarks
  7. plots    - Generate portfolio figures (overlays, Pareto, per-class safety IoU gains)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from seg_kd.benchmark import benchmark_model
from seg_kd.data import load_manifest, records_for_split
from seg_kd.evaluation import create_evaluation_freeze, evaluate_artifact
from seg_kd.export import assess_int8_acceptance, export_student_onnx
from seg_kd.training import (
    create_shared_initial_student_weights,
    train_student,
    train_teacher,
    write_json,
)
from seg_kd.visualize import plot_ablation_and_pareto, plot_per_class_gain, render_prediction_panel


def _load_existing_report(run_dir: Path, expected_manifest_sha: str) -> dict[str, Any] | None:
    report_path = run_dir / "training-report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") == "completed" and report.get("manifest_sha256") == expected_manifest_sha:
        return report
    return None


def _select_best_kd_method(ablation_reports: dict[str, dict[int, dict[str, Any]]], seeds: list[int]) -> dict[str, Any]:
    baseline_mious = [ablation_reports["baseline"][s]["validation_miou"] for s in seeds]
    baseline_mean = float(np.mean(baseline_mious))

    candidates = []
    for method, seed_map in ablation_reports.items():
        if method == "baseline":
            continue
        mious = [seed_map[s]["validation_miou"] for s in seeds]
        safeties = [seed_map[s]["validation_safety_miou"] for s in seeds]
        eces = [seed_map[s]["validation_ece"] for s in seeds]
        gains = [m - b for m, b in zip(mious, baseline_mious)]
        candidates.append(
            {
                "method": method,
                "mean_val_miou": float(np.mean(mious)),
                "std_val_miou": float(np.std(mious, ddof=1)) if len(mious) > 1 else 0.0,
                "mean_val_safety_miou": float(np.mean(safeties)),
                "mean_val_ece": float(np.mean(eces)),
                "mean_gain_over_baseline_pp": float(np.mean(gains) * 100.0),
                "positive_seed_pairs": int(sum(g > 0 for g in gains)),
                "seed_mious": {str(s): round(seed_map[s]["validation_miou"], 6) for s in seeds},
            }
        )

    candidates.sort(
        key=lambda c: (c["mean_val_miou"], c["mean_val_safety_miou"], -c["mean_val_ece"]),
        reverse=True,
    )
    selected = candidates[0] if candidates else None
    return {
        "selection_split": "validation",
        "baseline_mean_val_miou": baseline_mean,
        "selected_method": selected["method"] if selected else "baseline",
        "candidates": candidates,
    }


def run_study(
    *,
    plan_path: str | Path,
    manifest_path: str | Path,
    dataset_root: str | Path,
    output_root: str | Path,
    stages: list[str],
    device: str = "auto",
    kitti_manifest_path: str | Path | None = None,
    kitti_root: str | Path | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    manifest = load_manifest(manifest_path, dataset_root=dataset_root, verify_files=False)
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    if smoke:
        plan["seeds"] = [378]
        plan["deployment_seed"] = 378
        plan["image_size"] = [128, 256]
        plan["teacher"].update(
            {"variant": "segformer_tiny_smoke", "pretrained": False, "epochs": 1, "batch_size": 2, "num_workers": 0, "precision": "fp32"}
        )
        plan["main_student"].update(
            {"arch": "tiny_cnn", "fpn_dim": 48, "pretrained_backbone": False, "epochs": 1, "batch_size": 2, "num_workers": 0, "precision": "fp32", "qat_epochs": 1}
        )
        plan["ablation_methods"] = ["baseline", "pixel_kl", "cwd", "cwd_feat", "cwd_safety"]
        plan["pareto_students"] = [{"arch": "tiny_cnn", "fpn_dim": 48}]
        plan["benchmark"].update({"warmup": 1, "runs": 3, "cpu_threads": 1})
        plan["deployment_gate"]["calibration_count"] = 4

    seeds: list[int] = [int(s) for s in plan["seeds"]]
    deploy_seed: int = int(plan["deployment_seed"])
    image_size = (int(plan["image_size"][0]), int(plan["image_size"][1]))
    run_all = "all" in stages

    # 1. Teacher stage
    teacher_dir = out / "teacher" / plan["teacher"]["variant"]
    teacher_ckpt = teacher_dir / "teacher.pt"
    if run_all or "teacher" in stages:
        existing = _load_existing_report(teacher_dir, manifest["manifest_sha256"])
        if existing is None:
            t_cfg = {**plan["teacher"], "image_size": list(image_size)}
            train_teacher(
                dataset_root=dataset_root,
                manifest=manifest,
                output_dir=teacher_dir,
                variant=plan["teacher"]["variant"],
                seed=deploy_seed,
                config=t_cfg,
                device_spec=device,
                overwrite=True,
            )

    # 2. Paired multi-seed KD ablation
    ablation_reports: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in plan["ablation_methods"]}
    main_arch = plan["main_student"]["arch"]
    if run_all or "ablation" in stages:
        for seed in seeds:
            init_ckpt = out / "initial-weights" / f"{main_arch}-seed-{seed}.pt"
            if not init_ckpt.is_file():
                create_shared_initial_student_weights(
                    arch=main_arch,
                    output_path=init_ckpt,
                    seed=seed,
                    pretrained=bool(plan["main_student"].get("pretrained_backbone", False)),
                    fpn_dim=plan["main_student"].get("fpn_dim"),
                )
            for method in plan["ablation_methods"]:
                run_dir = out / "ablation" / f"seed-{seed}" / method
                existing = _load_existing_report(run_dir, manifest["manifest_sha256"])
                if existing is not None:
                    ablation_reports[method][seed] = existing
                    continue
                s_cfg = {**plan["main_student"], "image_size": list(image_size), "qat_epochs": 0}
                res = train_student(
                    dataset_root=dataset_root,
                    manifest=manifest,
                    output_dir=run_dir,
                    arch=main_arch,
                    method=method,
                    seed=seed,
                    teacher_path=teacher_ckpt if method != "baseline" else None,
                    initial_weights_path=init_ckpt,
                    config=s_cfg,
                    device_spec=device,
                    overwrite=True,
                )
                ablation_reports[method][seed] = res["report"]

        selection = _select_best_kd_method(ablation_reports, seeds)
        write_json(selection, out / "selection-report.json")
    else:
        sel_path = out / "selection-report.json"
        selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.is_file() else {"selected_method": "cwd"}

    selected_method = selection["selected_method"]

    # 3. Multi-size Pareto sweep (deployment seed)
    if run_all or "pareto" in stages:
        for spec in plan.get("pareto_students", []):
            arch = spec["arch"]
            fpn_dim = spec.get("fpn_dim")
            if arch == main_arch and fpn_dim == plan["main_student"].get("fpn_dim"):
                continue
            run_dir = out / "pareto" / f"{arch}-fpn{fpn_dim}" / selected_method
            if _load_existing_report(run_dir, manifest["manifest_sha256"]) is None:
                s_cfg = {**plan["main_student"], "image_size": list(image_size), "fpn_dim": fpn_dim, "qat_epochs": 0}
                train_student(
                    dataset_root=dataset_root,
                    manifest=manifest,
                    output_dir=run_dir,
                    arch=arch,
                    method=selected_method,
                    seed=deploy_seed,
                    teacher_path=teacher_ckpt,
                    config=s_cfg,
                    device_spec=device,
                    overwrite=True,
                )

    # 4. QAT + ONNX FP32/INT8 export + validation INT8 gate
    deploy_student_ckpt = out / "ablation" / f"seed-{deploy_seed}" / selected_method / "student.pt"
    export_dir = out / "export" / f"{main_arch}-{selected_method}-seed-{deploy_seed}"
    if run_all or "export" in stages:
        qat_epochs = int(plan["main_student"].get("qat_epochs", 0))
        export_source_ckpt = deploy_student_ckpt
        if qat_epochs > 0:
            qat_dir = out / "qat" / f"seed-{deploy_seed}" / selected_method
            if _load_existing_report(qat_dir, manifest["manifest_sha256"]) is None:
                s_cfg = {
                    **plan["main_student"],
                    "image_size": list(image_size),
                    "epochs": 1,
                    "warmup_epochs": 0,
                    "learning_rate": float(plan["main_student"]["learning_rate"]) * 0.2,
                    "qat_epochs": qat_epochs,
                }
                train_student(
                    dataset_root=dataset_root,
                    manifest=manifest,
                    output_dir=qat_dir,
                    arch=main_arch,
                    method=selected_method,
                    seed=deploy_seed,
                    teacher_path=teacher_ckpt,
                    initial_weights_path=deploy_student_ckpt,
                    config=s_cfg,
                    device_spec=device,
                    overwrite=True,
                )
            qat_ckpt = qat_dir / "student-qat.pt"
            if qat_ckpt.is_file():
                export_source_ckpt = qat_ckpt

        try:
            export_report = export_student_onnx(
                checkpoint_path=export_source_ckpt,
                output_dir=export_dir,
                manifest=manifest,
                dataset_root=dataset_root,
                image_size=image_size,
                calibration_count=int(plan["deployment_gate"]["calibration_count"]),
                seed=deploy_seed,
                export_int8=True,
            )
        except ImportError as exc:
            if not smoke:
                raise
            export_report = {"skipped": True, "reason": str(exc), "int8": {"exported": False}}
            write_json(export_report, export_dir / "export-report.json")
        if export_report["int8"].get("exported"):
            fp32_onnx = export_dir / "student-fp32.onnx"
            int8_onnx = export_dir / "student-int8.onnx"
            fp32_val = evaluate_artifact(
                fp32_onnx, manifest, dataset_root, split="validation", image_size=image_size, batch_size=1, threads=1
            )
            int8_val = evaluate_artifact(
                int8_onnx, manifest, dataset_root, split="validation", image_size=image_size, batch_size=1, threads=1
            )
            fp32_bench = benchmark_model(
                fp32_onnx,
                image_size=image_size,
                batch_size=1,
                warmup=int(plan["benchmark"]["warmup"]),
                runs=int(plan["benchmark"]["runs"]),
                device_spec="cpu",
                threads=int(plan["benchmark"]["cpu_threads"]),
            )
            int8_bench = benchmark_model(
                int8_onnx,
                image_size=image_size,
                batch_size=1,
                warmup=int(plan["benchmark"]["warmup"]),
                runs=int(plan["benchmark"]["runs"]),
                device_spec="cpu",
                threads=int(plan["benchmark"]["cpu_threads"]),
            )
            gate = assess_int8_acceptance(
                fp32_val,
                int8_val,
                fp32_bench,
                int8_bench,
                max_miou_drop_pp=float(plan["deployment_gate"]["max_miou_drop_pp"]),
                max_safety_miou_drop_pp=float(plan["deployment_gate"]["max_safety_miou_drop_pp"]),
                minimum_speedup=float(plan["deployment_gate"]["minimum_cpu_p50_speedup"]),
            )
            write_json(gate, export_dir / "int8-acceptance.json")

    # 5. Freeze before locked_eval
    freeze_models: dict[str, Path] = {}
    if teacher_ckpt.is_file():
        freeze_models["teacher"] = teacher_ckpt
    for method in plan["ablation_methods"]:
        p = out / "ablation" / f"seed-{deploy_seed}" / method / "student.pt"
        if p.is_file():
            freeze_models[f"student_{method}"] = p
    for onnx_name in ("student-fp32.onnx", "student-int8.onnx"):
        p = export_dir / onnx_name
        if p.is_file():
            freeze_models[onnx_name.replace(".onnx", "")] = p

    freeze_path = out / "evaluation-freeze.json"
    if run_all or "freeze" in stages:
        create_evaluation_freeze(manifest, freeze_models, freeze_path, plan_path=plan_path)

    # 6. Locked Evaluation + Benchmarks (+ optional KITTI Zero-Shot)
    eval_summary: dict[str, Any] = {"locked_eval": {}, "benchmarks_cpu": {}, "zero_shot_kitti": {}}
    if run_all or "eval" in stages:
        for label, path in sorted(freeze_models.items()):
            metrics = evaluate_artifact(
                model_path=path,
                manifest=manifest,
                dataset_root=dataset_root,
                split="locked_eval",
                image_size=image_size,
                batch_size=1 if path.suffix == ".onnx" else (2 if smoke else 4),
                device_spec="cpu" if path.suffix == ".onnx" else device,
                threads=1,
                freeze_path=freeze_path if freeze_path.is_file() else None,
            )
            write_json(metrics, out / "evaluations" / "locked_eval" / f"{label}.json")
            eval_summary["locked_eval"][label] = metrics

            bench = benchmark_model(
                model_path=path,
                image_size=image_size,
                batch_size=1,
                warmup=int(plan["benchmark"]["warmup"]),
                runs=int(plan["benchmark"]["runs"]),
                device_spec="cpu",
                threads=int(plan["benchmark"]["cpu_threads"]),
            )
            write_json(bench, out / "benchmarks" / "cpu" / f"{label}.json")
            eval_summary["benchmarks_cpu"][label] = bench

        if kitti_manifest_path and kitti_root:
            kitti_manifest = load_manifest(kitti_manifest_path, dataset_root=kitti_root, verify_files=False)
            for label in ("teacher", "student_baseline", f"student_{selected_method}"):
                if label in freeze_models:
                    km = evaluate_artifact(
                        model_path=freeze_models[label],
                        manifest=kitti_manifest,
                        dataset_root=kitti_root,
                        split="zero_shot_eval",
                        image_size=image_size,
                        batch_size=2,
                        device_spec=device,
                    )
                    write_json(km, out / "evaluations" / "kitti_zero_shot" / f"{label}.json")
                    eval_summary["zero_shot_kitti"][label] = km

        write_json(eval_summary, out / "release-summary.json")

    # 7. Figures
    if run_all or "plots" in stages:
        fig_dir = out / "figures"
        summary_path = out / "release-summary.json"
        if summary_path.is_file():
            loaded_sum = json.loads(summary_path.read_text(encoding="utf-8"))
            entries = []
            for label, m in loaded_sum["locked_eval"].items():
                b = loaded_sum["benchmarks_cpu"].get(label, {})
                entries.append(
                    {
                        "label": label,
                        "miou": m["miou"],
                        "safety_critical_miou": m["safety_critical_miou"],
                        "latency_p50_ms": b.get("latency_ms", {}).get("p50"),
                        "model_mib": b.get("model_mib", 1.0),
                    }
                )
            if entries:
                plot_ablation_and_pareto(entries, fig_dir / "ablation-and-pareto.png")
            base_m = loaded_sum["locked_eval"].get("student_baseline")
            dist_m = loaded_sum["locked_eval"].get(f"student_{selected_method}")
            if base_m and dist_m:
                plot_per_class_gain(base_m, dist_m, fig_dir / "per-class-safety-gain.png")

        panel_models = {}
        if teacher_ckpt.is_file():
            panel_models["Teacher"] = teacher_ckpt
        base_ckpt = out / "ablation" / f"seed-{deploy_seed}" / "baseline" / "student.pt"
        if base_ckpt.is_file():
            panel_models["Baseline"] = base_ckpt
        if deploy_student_ckpt.is_file():
            panel_models[f"Distilled ({selected_method})"] = deploy_student_ckpt
        if panel_models:
            render_prediction_panel(
                records_for_split(manifest, "locked_eval"),
                dataset_root,
                panel_models,
                fig_dir / "road-scene-overlays.png",
                image_size=image_size,
                device_spec=device,
                max_samples=4,
            )

    return {
        "output_root": str(out.resolve()),
        "selected_method": selected_method,
        "manifest_sha256": manifest["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phased L4 Driving Semantic Segmentation KD study")
    parser.add_argument("--plan", default="configs/l4-plan.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", "teacher", "ablation", "pareto", "export", "freeze", "eval", "plots"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kitti-manifest", default=None)
    parser.add_argument("--kitti-data", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run tiny 1-epoch CPU/GPU verification smoke pipeline")
    args = parser.parse_args()
    res = run_study(
        plan_path=args.plan,
        manifest_path=args.manifest,
        dataset_root=args.data,
        output_root=args.output,
        stages=args.stage,
        device=args.device,
        kitti_manifest_path=args.kitti_manifest,
        kitti_root=args.kitti_data,
        smoke=args.smoke,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
