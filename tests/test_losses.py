"""Tests for dense segmentation KD losses: exactness, ignore_index masking, BF16 safety, and gradients."""

from __future__ import annotations

import unittest

import torch

from seg_kd.losses import (
    SegmentationDistillationLoss,
    VALID_KD_METHODS,
    channel_wise_kd_loss,
    feature_cosine_loss,
    pixel_kl_loss,
)
from seg_kd.taxonomy import IGNORE_INDEX, NUM_CLASSES, safety_channel_weights


class DistillationLossTests(unittest.TestCase):
    def test_zero_kl_when_student_equals_teacher(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, NUM_CLASSES, 16, 32)
        valid = torch.ones(2, 16, 32, dtype=torch.bool)
        valid[:, :2, :] = False

        pkl = pixel_kl_loss(logits, logits, temperature=4.0, valid_mask=valid)
        cwd = channel_wise_kd_loss(logits, logits, temperature=4.0, valid_mask=valid)
        cwd_w = channel_wise_kd_loss(
            logits, logits, temperature=4.0, valid_mask=valid, channel_weights=safety_channel_weights(2.0)
        )
        self.assertAlmostEqual(float(pkl.item()), 0.0, places=5)
        self.assertAlmostEqual(float(cwd.item()), 0.0, places=5)
        self.assertAlmostEqual(float(cwd_w.item()), 0.0, places=5)
        cos = feature_cosine_loss(logits, logits, valid_mask=valid)
        self.assertAlmostEqual(float(cos.item()), 0.0, places=5)

    def test_ignore_index_pixels_do_not_affect_losses(self) -> None:
        torch.manual_seed(1)
        s = torch.randn(2, NUM_CLASSES, 8, 16, requires_grad=True)
        t = torch.randn(2, NUM_CLASSES, 8, 16)
        t_corrupted = t.clone()
        t_corrupted[:, :, :2, :] = 100.0  # Extreme values inside ignored border

        valid = torch.ones(2, 8, 16, dtype=torch.bool)
        valid[:, :2, :] = False

        self.assertAlmostEqual(
            float(pixel_kl_loss(s, t, valid_mask=valid).item()),
            float(pixel_kl_loss(s, t_corrupted, valid_mask=valid).item()),
            places=5,
        )
        self.assertAlmostEqual(
            float(channel_wise_kd_loss(s, t, valid_mask=valid).item()),
            float(channel_wise_kd_loss(s, t_corrupted, valid_mask=valid).item()),
            places=5,
        )

    def test_all_methods_produce_finite_losses_and_gradients_under_bf16_inputs(self) -> None:
        torch.manual_seed(2)
        targets = torch.randint(0, NUM_CLASSES, (2, 16, 32))
        targets[:, :2, :] = IGNORE_INDEX
        t_logits = torch.randn(2, NUM_CLASSES, 16, 32, dtype=torch.bfloat16)
        t_feats = {
            "s8": torch.randn(2, 32, 4, 8, dtype=torch.bfloat16),
            "s16": torch.randn(2, 64, 2, 4, dtype=torch.bfloat16),
            "s32": torch.randn(2, 128, 1, 2, dtype=torch.bfloat16),
        }

        for method in VALID_KD_METHODS:
            s_logits = torch.randn(2, NUM_CLASSES, 16, 32, dtype=torch.bfloat16, requires_grad=True)
            s_feats = {
                k: torch.randn_like(v, requires_grad=True) for k, v in t_feats.items()
            }
            loss_fn = SegmentationDistillationLoss(method=method, temperature=4.0)
            out = loss_fn(
                s_logits,
                targets,
                teacher_logits=None if method == "baseline" else t_logits,
                projected_student_features=s_feats if method in {"cwd_feat", "cwd_safety"} else None,
                teacher_features=t_feats if method in {"cwd_feat", "cwd_safety"} else None,
            )
            self.assertEqual(out["loss"].dtype, torch.float32)
            self.assertTrue(bool(torch.isfinite(out["loss"]).item()))
            out["loss"].backward()
            self.assertIsNotNone(s_logits.grad)
            self.assertGreater(float(s_logits.grad.abs().sum().item()), 0.0)
            if method in {"cwd_feat", "cwd_safety"}:
                for stage, tensor in s_feats.items():
                    self.assertIsNotNone(tensor.grad, f"Missing feature grad for {method}:{stage}")
                    self.assertGreater(float(tensor.grad.abs().sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
