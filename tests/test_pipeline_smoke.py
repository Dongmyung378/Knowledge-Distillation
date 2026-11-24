"""End-to-end CPU smoke test covering manifest, teacher, paired KD, QAT, freeze, eval, benchmark, and plots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seg_kd.benchmark import benchmark_model
from seg_kd.data import create_synthetic_dataset, load_manifest, make_dataloader, records_for_split
from seg_kd.evaluation import (
    create_evaluation_freeze,
    evaluate_artifact,
    evaluate_temperature_scaled_model,
    fit_temperature_scaling,
    validate_evaluation_freeze,
)
from seg_kd.export import assess_int8_acceptance
from seg_kd.models import load_model_checkpoint
from seg_kd.training import (
    apply_gpu_adverse_corruption,
    create_shared_initial_student_weights,
    train_student,
    train_teacher,
)
from seg_kd.visualize import plot_ablation_and_pareto, plot_per_class_gain, render_prediction_panel


class EndToEndPipelineSmokeTests(unittest.TestCase):
    def test_full_paired_kd_pipeline_on_synthetic_driving_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "synth"
            manifest = create_synthetic_dataset(
                data_dir, num_train=4, num_val=2, num_eval=2, image_size=(64, 128), seed=378
            )
            manifest = load_manifest(data_dir / "manifest.json", dataset_root=data_dir, verify_files=True)

            # 1. Train tiny smoke SegFormer teacher
            t_res = train_teacher(
                dataset_root=data_dir,
                manifest=manifest,
                output_dir=root / "teacher",
                variant="segformer_tiny_smoke",
                seed=378,
                config={
                    "epochs": 1,
                    "batch_size": 2,
                    "image_size": [64, 128],
                    "precision": "fp32",
                    "num_workers": 0,
                    "pretrained": False,
                },
                device_spec="cpu",
            )
            teacher_ckpt = Path(t_res["model_path"])
            self.assertTrue(teacher_ckpt.is_file())

            # 2. Create shared initial student weights for paired comparison
            init_info = create_shared_initial_student_weights(
                arch="tiny_cnn", output_path=root / "init-seed-378.pt", seed=378, pretrained=False
            )
            self.assertTrue(init_info["quantization_audit"]["passed"])

            common_cfg = {
                "epochs": 1,
                "batch_size": 2,
                "image_size": [64, 128],
                "precision": "fp32",
                "num_workers": 0,
                "pretrained_backbone": False,
                "channels_last": False,
            }
            base_res = train_student(
                dataset_root=data_dir,
                manifest=manifest,
                output_dir=root / "student-baseline",
                arch="tiny_cnn",
                method="baseline",
                seed=378,
                initial_weights_path=init_info["path"],
                config=common_cfg,
                device_spec="cpu",
            )
            dist_res = train_student(
                dataset_root=data_dir,
                manifest=manifest,
                output_dir=root / "student-cwd-safety",
                arch="tiny_cnn",
                method="cwd_safety",
                seed=378,
                teacher_path=teacher_ckpt,
                initial_weights_path=init_info["path"],
                config={**common_cfg, "qat_epochs": 1, "asymmetric_robust_prob": 0.5},
                device_spec="cpu",
            )

            self.assertEqual(
                base_res["report"]["initial_weights_sha256"],
                dist_res["report"]["initial_weights_sha256"],
            )
            self.assertTrue(dist_res["report"]["teacher_immutable"])
            self.assertIsNotNone(dist_res["report"]["qat_finetune"])
            self.assertTrue((root / "student-cwd-safety" / "student-qat.pt").is_file())

            # 3. Freeze before locked evaluation and verify tamper protection
            freeze_path = root / "evaluation-freeze.json"
            create_evaluation_freeze(
                manifest,
                {
                    "teacher": teacher_ckpt,
                    "baseline": base_res["model_path"],
                    "distilled": dist_res["model_path"],
                },
                freeze_path,
            )
            validate_evaluation_freeze(freeze_path, manifest=manifest, model_path=dist_res["model_path"])

            # 4. Evaluate on locked_eval (checks mIoU, safety mIoU, ECE, weather/time slices)
            base_eval = evaluate_artifact(
                base_res["model_path"],
                manifest,
                data_dir,
                split="locked_eval",
                image_size=(64, 128),
                batch_size=2,
                device_spec="cpu",
                freeze_path=freeze_path,
            )
            dist_eval = evaluate_artifact(
                dist_res["model_path"],
                manifest,
                data_dir,
                split="locked_eval",
                image_size=(64, 128),
                batch_size=2,
                device_spec="cpu",
                freeze_path=freeze_path,
            )
            self.assertIn("timeofday:daytime", dist_eval["slices"])
            self.assertIn("timeofday:night", dist_eval["slices"])
            self.assertGreaterEqual(dist_eval["ece"], 0.0)

            # 5. Benchmark CPU latency & verify INT8 gate logic
            base_bench = benchmark_model(
                base_res["model_path"], image_size=(64, 128), warmup=1, runs=3, device_spec="cpu", threads=1
            )
            dist_bench = benchmark_model(
                dist_res["model_path"], image_size=(64, 128), warmup=1, runs=3, device_spec="cpu", threads=1
            )
            self.assertGreater(dist_bench["latency_ms"]["p50"], 0.0)

            gate_pass = assess_int8_acceptance(
                {"split": "validation", "miou": 0.650, "safety_critical_miou": 0.550},
                {"split": "validation", "miou": 0.644, "safety_critical_miou": 0.542},
                {"model_bytes": 10_000_000, "latency_ms": {"p50": 40.0}},
                {"model_bytes": 2_800_000, "latency_ms": {"p50": 18.0}},
            )
            self.assertTrue(gate_pass["accepted"])
            self.assertEqual(gate_pass["selected_format"], "INT8")

            gate_reject = assess_int8_acceptance(
                {"split": "validation", "miou": 0.650, "safety_critical_miou": 0.550},
                {"split": "validation", "miou": 0.620, "safety_critical_miou": 0.510},
                {"model_bytes": 10_000_000, "latency_ms": {"p50": 40.0}},
                {"model_bytes": 2_800_000, "latency_ms": {"p50": 39.0}},
            )
            self.assertFalse(gate_reject["accepted"])
            self.assertEqual(gate_reject["selected_format"], "FP32")

            # 6. Render visual figures
            fig_panel = render_prediction_panel(
                records_for_split(manifest, "locked_eval"),
                data_dir,
                {"Teacher": teacher_ckpt, "Baseline": base_res["model_path"], "Distilled": dist_res["model_path"]},
                root / "figs" / "panel.png",
                image_size=(64, 128),
                device_spec="cpu",
                max_samples=2,
            )
            fig_pareto = plot_ablation_and_pareto(
                [
                    {"label": "Baseline", "miou": base_eval["miou"], "safety_critical_miou": base_eval["safety_critical_miou"], "latency_p50_ms": base_bench["latency_ms"]["p50"], "model_mib": base_bench["model_mib"]},
                    {"label": "Distilled", "miou": dist_eval["miou"], "safety_critical_miou": dist_eval["safety_critical_miou"], "latency_p50_ms": dist_bench["latency_ms"]["p50"], "model_mib": dist_bench["model_mib"]},
                ],
                root / "figs" / "pareto.png",
            )
            fig_gain = plot_per_class_gain(base_eval, dist_eval, root / "figs" / "gain.png")
            for fig in (fig_panel, fig_pareto, fig_gain):
                self.assertTrue(fig.is_file() and fig.stat().st_size > 1000)

            # 7. Verify GPU/CPU adverse corruption helper & Post-hoc Temperature Scaling mIoU parity
            import torch
            sample_batch = torch.randn(4, 3, 64, 128, dtype=torch.float32)
            corrupted = apply_gpu_adverse_corruption(sample_batch, prob=1.0)
            self.assertEqual(corrupted.shape, sample_batch.shape)
            self.assertTrue(bool(torch.isfinite(corrupted).all().item()))
            self.assertGreater(float((corrupted - sample_batch).abs().mean().item()), 0.0)

            dist_model, _ = load_model_checkpoint(dist_res["model_path"], map_location="cpu")
            val_loader = make_dataloader(
                records_for_split(manifest, "validation"), data_dir, image_size=(64, 128), batch_size=2, training=False, num_workers=0
            )
            eval_loader = make_dataloader(
                records_for_split(manifest, "locked_eval"), data_dir, image_size=(64, 128), batch_size=2, training=False, num_workers=0
            )
            t_star = fit_temperature_scaling(dist_model, val_loader, device=torch.device("cpu"), coarse_steps=9)
            self.assertGreater(t_star, 0.0)
            cal_res = evaluate_temperature_scaled_model(
                dist_model, eval_loader, temperature=t_star, device=torch.device("cpu"), split="locked_eval"
            )
            self.assertAlmostEqual(cal_res["miou_before"], cal_res["miou_after"], places=6)


if __name__ == "__main__":
    unittest.main()
