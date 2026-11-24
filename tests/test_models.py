"""Tests for quantization-friendly Edge CNN students, SegFormer teacher, and feature alignment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from seg_kd.models import (
    FeatureProjectorBank,
    audit_quantization_friendliness,
    build_student,
    build_teacher,
    load_model_checkpoint,
    save_model_checkpoint,
    weights_sha256,
)
from seg_kd.taxonomy import NUM_CLASSES


class ModelArchitectureTests(unittest.TestCase):
    def test_students_pass_quantization_audit_and_emit_aligned_pyramids(self) -> None:
        x = torch.randn(2, 3, 128, 256)
        for arch in ("tiny_cnn", "mobilenetv4_small", "mobilenetv3_large"):
            student = build_student(arch=arch, pretrained=False)
            audit = audit_quantization_friendliness(student)
            self.assertTrue(audit["passed"], f"{arch} failed quantization audit: {audit['violations']}")

            logits = student(x)
            self.assertEqual(tuple(logits.shape), (2, NUM_CLASSES, 128, 256))

            rich = student(x, return_features=True)
            self.assertEqual(tuple(rich["logits"].shape), (2, NUM_CLASSES, 128, 256))
            self.assertEqual(tuple(rich["features"]["s4"].shape[-2:]), (32, 64))
            self.assertEqual(tuple(rich["features"]["s8"].shape[-2:]), (16, 32))
            self.assertEqual(tuple(rich["features"]["s16"].shape[-2:]), (8, 16))
            self.assertEqual(tuple(rich["features"]["s32"].shape[-2:]), (4, 8))

    def test_quantization_audit_catches_conv_relu_bn_antipattern(self) -> None:
        bad_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        audit = audit_quantization_friendliness(bad_model)
        self.assertFalse(audit["passed"])
        self.assertEqual(len(audit["violations"]), 1)

    def test_teacher_and_projector_alignment_and_checkpoint_roundtrip(self) -> None:
        x = torch.randn(1, 3, 128, 256)
        teacher = build_teacher(variant="segformer_tiny_smoke", pretrained=False).eval()
        student = build_student(arch="tiny_cnn", pretrained=False).eval()

        t_out = teacher(x, return_features=True)
        s_out = student(x, return_features=True)
        self.assertEqual(tuple(t_out["logits"].shape), (1, NUM_CLASSES, 128, 256))

        bank = FeatureProjectorBank(student.feature_channels, teacher.feature_channels, stages=("s8", "s16", "s32"))
        projected = bank(s_out["features"])
        for stage in ("s8", "s16", "s32"):
            self.assertEqual(projected[stage].shape, t_out["features"][stage].shape)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = save_model_checkpoint(student, Path(tmp) / "student.pt", role="student", arch="tiny_cnn")
            reloaded, payload = load_model_checkpoint(ckpt)
            reloaded.eval()
            self.assertEqual(payload["weights_sha256"], weights_sha256(reloaded))
            with torch.no_grad():
                torch.testing.assert_close(student(x), reloaded(x))
            # Verify TorchScript tracing via the plain export wrapper succeeds cleanly.
            from seg_kd.export import _PlainLogitsExportWrapper
            traced = torch.jit.trace(_PlainLogitsExportWrapper(reloaded).eval(), x)
            with torch.no_grad():
                torch.testing.assert_close(reloaded(x), traced(x))

    def test_segformer_b4_offline_config_matches_real_b4_and_roundtrips(self) -> None:
        teacher = build_teacher(variant="segformer_b4", pretrained=False).eval()
        params = sum(p.numel() for p in teacher.parameters())
        # Real MiT-B4 + 768-dim MLP decode head is ~64M parameters (not ~3.7M B0).
        self.assertGreater(params, 60_000_000)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = save_model_checkpoint(teacher, Path(tmp) / "teacher-b4.pt", role="teacher", arch="segformer_b4")
            reloaded, payload = load_model_checkpoint(ckpt)
            self.assertEqual(payload["weights_sha256"], weights_sha256(reloaded))


if __name__ == "__main__":
    unittest.main()
