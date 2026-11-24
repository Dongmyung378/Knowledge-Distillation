"""Dense semantic segmentation KD losses in strict FP32 (BF16 AMP-safe).

Supported distillation objectives:
- ``baseline``: Supervised CE (+ optional Dice) only.
- ``pixel_kl``: Masked pixel-wise logit KL divergence * T^2.
- ``cwd``: Channel-wise Knowledge Distillation (Shu et al., ICCV 2021) over spatial maps.
- ``cwd_feat``: Logit CWD + Multi-scale projected feature CWD alignment.
- ``cwd_safety``: Safety-critical class-weighted Logit CWD + Feature CWD + safety-boosted CE.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .taxonomy import IGNORE_INDEX, NUM_CLASSES, safety_channel_weights

VALID_KD_METHODS: tuple[str, ...] = (
    "baseline",
    "pixel_kl",
    "cwd",
    "cwd_feat",
    "cwd_safety",
)


def supervised_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    logits_f32 = logits.float()
    weight = class_weights.float().to(logits_f32.device) if class_weights is not None else None
    valid = targets != ignore_index
    if not bool(valid.any()):
        return logits_f32.sum() * 0.0
    return F.cross_entropy(logits_f32, targets.long(), weight=weight, ignore_index=ignore_index)


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = IGNORE_INDEX,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft multiclass Dice loss computed over valid pixels present in the batch."""
    logits_f32 = logits.float()
    num_classes = logits_f32.shape[1]
    valid = targets != ignore_index
    if not bool(valid.any()):
        return logits_f32.sum() * 0.0

    probs = F.softmax(logits_f32, dim=1)
    safe_targets = targets.long().clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    mask = valid.unsqueeze(1).float()
    probs = probs * mask
    one_hot = one_hot * mask

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    present = one_hot.sum(dim=dims) > 0
    if not bool(present.any()):
        return logits_f32.sum() * 0.0
    dice = (2.0 * intersection[present] + smooth) / (denom[present] + smooth)
    return 1.0 - dice.mean()


def pixel_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 4.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked pixel-wise Teacher -> Student KL divergence scaled by T^2."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = student_logits.float()
    t = teacher_logits.detach().float()
    if s.shape[-2:] != t.shape[-2:]:
        s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)

    t_log_p = F.log_softmax(t / temperature, dim=1)
    s_log_p = F.log_softmax(s / temperature, dim=1)
    kl_map = (t_log_p.exp() * (t_log_p - s_log_p)).sum(dim=1)  # [B, H, W]

    if valid_mask is not None:
        mask = valid_mask.float()
        if mask.shape[-2:] != kl_map.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1), size=kl_map.shape[-2:], mode="nearest").squeeze(1)
        denom = mask.sum()
        if float(denom.item()) <= 0.0:
            return s.sum() * 0.0
        kl = (kl_map * mask).sum() / denom
    else:
        kl = kl_map.mean()
    return kl * (temperature**2)


def channel_wise_kd_loss(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    *,
    temperature: float = 4.0,
    valid_mask: torch.Tensor | None = None,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Channel-wise Knowledge Distillation (Shu et al., ICCV 2021) over spatial locations.

    Computes KL divergence between spatial probability distributions of each channel:
    ``KL(softmax_HW(T_c / tau) || softmax_HW(S_c / tau)) * tau^2``.
    Optional ``valid_mask`` excludes ``ignore_index`` spatial locations cleanly.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = student_map.float()
    t = teacher_map.detach().float()
    if s.shape[-2:] != t.shape[-2:]:
        s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)

    b, c, h, w = t.shape
    s_flat = s.reshape(b, c, h * w) / temperature
    t_flat = t.reshape(b, c, h * w) / temperature

    if valid_mask is not None:
        mask = valid_mask.bool()
        if mask.shape[-2:] != (h, w):
            # Mark a coarse feature cell valid if any pixel in its receptive field is valid.
            mask = F.adaptive_max_pool2d(mask.float().unsqueeze(1), output_size=(h, w)).squeeze(1) > 0.5
        mask_flat = mask.reshape(b, 1, h * w)
        sample_valid = mask_flat.any(dim=-1)  # [B, 1]
        if not bool(sample_valid.any()):
            return s.sum() * 0.0
        neg_inf = torch.finfo(torch.float32).min / 2.0
        # Replace masked-out spatial locations with large negative values for safe log_softmax.
        # Ensure completely masked samples have at least index 0 unmasked before filtering them out.
        safe_mask = mask_flat.clone()
        empty_batch = ~sample_valid.squeeze(-1)
        if bool(empty_batch.any()):
            safe_mask[empty_batch, :, 0] = True
        s_masked = s_flat.masked_fill(~safe_mask, neg_inf)
        t_masked = t_flat.masked_fill(~safe_mask, neg_inf)
        t_log_p = F.log_softmax(t_masked, dim=-1)
        s_log_p = F.log_softmax(s_masked, dim=-1)
        t_p = t_log_p.exp().masked_fill(~safe_mask, 0.0)
        diff = (t_log_p - s_log_p).masked_fill(~safe_mask, 0.0)
        kl_bc = (t_p * diff).sum(dim=-1)  # [B, C]
        kl_bc = kl_bc[sample_valid.squeeze(-1)]
    else:
        t_log_p = F.log_softmax(t_flat, dim=-1)
        s_log_p = F.log_softmax(s_flat, dim=-1)
        kl_bc = (t_log_p.exp() * (t_log_p - s_log_p)).sum(dim=-1)  # [B, C]

    if channel_weights is not None:
        w_c = channel_weights.float().to(kl_bc.device)
        if w_c.numel() != c:
            raise ValueError(f"channel_weights length {w_c.numel()} != channels {c}")
        w_norm = w_c / w_c.sum()
        kl = (kl_bc * w_norm.unsqueeze(0)).sum(dim=-1).mean()
    else:
        kl = kl_bc.mean()
    return kl * (temperature**2)


def feature_cosine_loss(
    student_map: torch.Tensor,
    teacher_map: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked spatial cosine distance (1 - cos) between projected student and teacher features."""
    s = student_map.float()
    t = teacher_map.detach().float()
    if s.shape[-2:] != t.shape[-2:]:
        s = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
    cos = F.cosine_similarity(s, t, dim=1, eps=1e-6)
    loss_map = 1.0 - cos
    if valid_mask is not None:
        mask = valid_mask.bool()
        if mask.shape[-2:] != loss_map.shape[-2:]:
            mask = F.adaptive_max_pool2d(mask.float().unsqueeze(1), output_size=loss_map.shape[-2:]).squeeze(1) > 0.5
        denom = mask.float().sum()
        if float(denom.item()) <= 0.0:
            return s.sum() * 0.0
        return (loss_map * mask.float()).sum() / denom
    return loss_map.mean()


class SegmentationDistillationLoss(nn.Module):
    """Unified, strictly FP32 loss module supporting all ablation methods."""

    def __init__(
        self,
        method: str = "cwd",
        *,
        temperature: float = 4.0,
        lambda_dice: float = 0.25,
        lambda_logit_kd: float = 3.0,
        lambda_feat_kd: float = 1.5,
        feature_stages: Sequence[str] = ("s8", "s16", "s32"),
        safety_boost: float = 2.0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if method not in VALID_KD_METHODS:
            raise ValueError(f"Unknown KD method {method!r}. Expected one of {VALID_KD_METHODS}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.method = method
        self.temperature = float(temperature)
        self.lambda_dice = float(lambda_dice)
        self.lambda_logit_kd = float(lambda_logit_kd)
        self.lambda_feat_kd = float(lambda_feat_kd)
        self.feature_stages = tuple(feature_stages)
        self.safety_boost = float(safety_boost)
        self.ignore_index = int(ignore_index)

    def forward(
        self,
        student_logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        teacher_logits: torch.Tensor | None = None,
        projected_student_features: Mapping[str, torch.Tensor] | None = None,
        teacher_features: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        device = student_logits.device
        zero = student_logits.float().sum() * 0.0
        valid_mask = targets != self.ignore_index

        ch_weights = (
            safety_channel_weights(self.safety_boost, device=device)
            if self.method == "cwd_safety"
            else None
        )
        # Keep supervised CE unweighted so rare large-vehicle classes (e.g. truck/bus) do not collapse;
        # safety-critical emphasis is applied exclusively to CWD logit channels when method == "cwd_safety".
        hard_loss = supervised_ce_loss(
            student_logits,
            targets,
            ignore_index=self.ignore_index,
            class_weights=None,
        )
        dice_loss = (
            multiclass_dice_loss(student_logits, targets, ignore_index=self.ignore_index)
            if self.lambda_dice > 0
            else zero
        )

        logit_kd = zero
        feat_kd = zero

        if self.method != "baseline":
            if teacher_logits is None:
                raise ValueError(f"Method {self.method!r} requires teacher_logits")
            if self.method == "pixel_kl":
                logit_kd = pixel_kl_loss(
                    student_logits,
                    teacher_logits,
                    temperature=self.temperature,
                    valid_mask=valid_mask,
                )
            elif self.method in {"cwd", "cwd_feat", "cwd_safety"}:
                logit_kd = channel_wise_kd_loss(
                    student_logits,
                    teacher_logits,
                    temperature=self.temperature,
                    valid_mask=valid_mask,
                    channel_weights=ch_weights,
                )

        if self.method in {"cwd_feat", "cwd_safety"}:
            if projected_student_features is None or teacher_features is None:
                raise ValueError(f"Method {self.method!r} requires projected_student_features and teacher_features")
            stage_losses = []
            for stage in self.feature_stages:
                if stage not in projected_student_features or stage not in teacher_features:
                    raise ValueError(f"Missing feature stage {stage!r} for feature distillation")
                cwd_f = channel_wise_kd_loss(
                    projected_student_features[stage],
                    teacher_features[stage],
                    temperature=self.temperature,
                    valid_mask=valid_mask,
                )
                cos_f = feature_cosine_loss(
                    projected_student_features[stage],
                    teacher_features[stage],
                    valid_mask=valid_mask,
                )
                stage_losses.append(0.5 * cwd_f + 0.5 * cos_f)
            feat_kd = torch.stack(stage_losses).mean()

        total = (
            hard_loss
            + self.lambda_dice * dice_loss
            + (self.lambda_logit_kd * logit_kd if self.method != "baseline" else zero)
            + (self.lambda_feat_kd * feat_kd if self.method in {"cwd_feat", "cwd_safety"} else zero)
        )
        return {
            "loss": total,
            "hard_loss": hard_loss.detach(),
            "dice_loss": dice_loss.detach(),
            "logit_kd_loss": logit_kd.detach(),
            "feature_kd_loss": feat_kd.detach(),
        }
