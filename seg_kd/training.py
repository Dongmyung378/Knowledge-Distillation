"""L4-optimized BF16 AMP training, paired KD ablation, teacher immutability, and QAT fine-tuning."""

from __future__ import annotations

import copy
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import canonical_sha256, file_sha256, make_dataloader, records_for_split
from .evaluation import evaluate_pytorch_model
from .losses import SegmentationDistillationLoss, VALID_KD_METHODS, multiclass_dice_loss, supervised_ce_loss
from .models import (
    FeatureProjectorBank,
    audit_quantization_friendliness,
    build_student,
    build_teacher,
    count_parameters,
    load_model_checkpoint,
    save_model_checkpoint,
    weights_sha256,
)
from .taxonomy import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def apply_gpu_adverse_corruption(images: torch.Tensor, prob: float = 0.35) -> torch.Tensor:
    """Apply randomized sensor/weather corruptions exclusively to student inputs on GPU.

    Used for Asymmetric Robustness KD (Clean ViT Teacher -> Corrupted Edge CNN Student)
    with continuous randomized severities across Gaussian sensor noise, low-light shot noise,
    atmospheric fog veil, and directional/defocus blur.
    """
    if prob <= 0.0:
        return images
    batch_size = int(images.shape[0])
    device = images.device
    orig_dtype = images.dtype
    is_cl = images.is_contiguous(memory_format=torch.channels_last)
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    rgb = (images.float() * std + mean).clamp(0.0, 1.0)
    out = rgb.clone()

    for i in range(batch_size):
        if random.random() >= prob:
            continue
        mode = random.choices(
            ("sensor_noise", "night_shot", "fog_veil", "motion_defocus"),
            weights=(0.35, 0.22, 0.23, 0.20),
            k=1,
        )[0]
        x = out[i : i + 1]
        if mode == "sensor_noise":
            sigma = random.uniform(4.0, 24.0) / 255.0
            x = (x + torch.randn_like(x) * sigma).clamp(0.0, 1.0)
        elif mode == "night_shot":
            gain = random.uniform(0.25, 0.65)
            gamma = random.uniform(1.0, 1.35)
            shot = random.uniform(2.0, 9.0) / 255.0
            x = (x.pow(gamma) * gain + torch.randn_like(x) * shot).clamp(0.0, 1.0)
        elif mode == "fog_veil":
            alpha = random.uniform(0.18, 0.48)
            veil = random.uniform(0.68, 0.92)
            x = (x * (1.0 - alpha) + veil * alpha).clamp(0.0, 1.0)
        else:
            k = random.choice((3, 5, 7))
            if random.random() < 0.5:
                kern = torch.ones((3, 1, 1, k), device=device, dtype=torch.float32) / float(k)
                x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), kern, groups=3)
            else:
                kern = torch.ones((3, 1, k, k), device=device, dtype=torch.float32) / float(k * k)
                x = F.conv2d(F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode="reflect"), kern, groups=3)
            x = x.clamp(0.0, 1.0)
        out[i : i + 1] = x

    normed = ((out - mean) / std).to(dtype=orig_dtype)
    if is_cl:
        normed = normed.to(memory_format=torch.channels_last)
    return normed


def write_json(value: Any, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return dest


def resolve_device(device_spec: str = "auto") -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no NVIDIA GPU is available")
        return torch.device(device_spec)
    if device_spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device specification: {device_spec!r}")


def detect_runtime_metadata(device: torch.device, *, precision: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.system(),
        "device": str(device),
        "precision": precision,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_vram_gib": round(props.total_memory / (1024**3), 2),
                "cuda_version": torch.version.cuda,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return info


def configure_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def _resolve_amp(device: torch.device, precision: str) -> tuple[bool, torch.dtype, bool]:
    """Return (use_amp, amp_dtype, use_grad_scaler)."""
    if device.type != "cuda" or precision == "fp32":
        return False, torch.float32, False
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return True, torch.float16, True
    if precision == "fp16":
        return True, torch.float16, True
    raise ValueError(f"Unsupported precision {precision!r}. Use 'bf16', 'fp16', or 'fp32'.")


def _prepare_output_dir(output_dir: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


class ModelEMA:
    """Exponential Moving Average with step warmup and direct BN buffer synchronization."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        eff_decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for ema_p, src_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(eff_decay).add_(src_p.detach(), alpha=1.0 - eff_decay)
        for ema_b, src_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(src_b.detach())


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = max(1, int(total_steps))
    warmup = max(0, min(int(warmup_steps), total - 1))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = float(step - warmup) / float(max(1, total - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def create_shared_initial_student_weights(
    arch: str,
    output_path: str | Path,
    *,
    seed: int = 378,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    fpn_dim: int | None = None,
) -> dict[str, Any]:
    """Create and persist shared initial weights so Baseline and KD runs start identically per seed."""
    configure_seed(seed, deterministic=True)
    student = build_student(arch=arch, num_classes=num_classes, pretrained=pretrained, fpn_dim=fpn_dim)
    audit = audit_quantization_friendliness(student)
    ckpt = save_model_checkpoint(
        student,
        output_path,
        role="student",
        arch=arch,
        extra={"seed": seed, "pretrained_backbone": pretrained, "quantization_audit": audit},
    )
    return {
        "path": str(ckpt.resolve()),
        "file_sha256": file_sha256(ckpt),
        "weights_sha256": weights_sha256(student),
        "parameter_count": count_parameters(student),
        "quantization_audit": audit,
    }


def train_teacher(
    dataset_root: str | Path,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    variant: str = "segformer_b4",
    seed: int = 378,
    config: Mapping[str, Any] | None = None,
    device_spec: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fine-tune a SegFormer teacher on L4 (BF16 AMP) and save best validation mIoU checkpoint."""
    cfg = {
        "epochs": 25,
        "batch_size": 8,
        "image_size": [512, 1024],
        "learning_rate": 6e-5,
        "head_lr_multiplier": 5.0,
        "weight_decay": 0.01,
        "warmup_epochs": 2,
        "lambda_dice": 0.25,
        "grad_clip": 1.0,
        "precision": "bf16",
        "num_workers": 4,
        "pretrained": True,
        "hf_model_id": None,
        "deterministic": False,
        "compile_model": False,
        "verbose": True,
    }
    if config:
        cfg.update(dict(config))

    device = resolve_device(device_spec)
    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    configure_seed(seed, deterministic=bool(cfg["deterministic"]))
    use_amp, amp_dtype, use_scaler = _resolve_amp(device, str(cfg["precision"]))

    image_size = (int(cfg["image_size"][0]), int(cfg["image_size"][1]))
    train_records = records_for_split(manifest, "train")
    val_records = records_for_split(manifest, "validation")
    train_loader = make_dataloader(
        train_records,
        dataset_root,
        image_size=image_size,
        batch_size=int(cfg["batch_size"]),
        training=True,
        seed=seed,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = make_dataloader(
        val_records,
        dataset_root,
        image_size=image_size,
        batch_size=max(1, int(cfg["batch_size"])),
        training=False,
        seed=seed,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    teacher = build_teacher(
        variant=variant,
        num_classes=int(manifest.get("num_classes", NUM_CLASSES)),
        pretrained=bool(cfg["pretrained"]),
        hf_model_id=cfg.get("hf_model_id"),
    ).to(device)

    backbone_params, head_params = [], []
    for name, param in teacher.named_parameters():
        if not param.requires_grad:
            continue
        if "decode_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    param_groups = [
        {"params": backbone_params, "lr": float(cfg["learning_rate"])},
        {"params": head_params, "lr": float(cfg["learning_rate"]) * float(cfg["head_lr_multiplier"])},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg["weight_decay"]))
    total_steps = max(1, int(cfg["epochs"]) * max(1, len(train_loader)))
    warmup_steps = int(cfg["warmup_epochs"]) * max(1, len(train_loader))
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_miou = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    ckpt_path = output / "teacher.pt"
    started = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        ep_t0 = time.perf_counter()
        teacher.train()
        running_loss = 0.0
        steps = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = teacher(images, return_features=False)
            ce = supervised_ce_loss(logits, masks, ignore_index=IGNORE_INDEX)
            dice = multiclass_dice_loss(logits, masks, ignore_index=IGNORE_INDEX) if float(cfg["lambda_dice"]) > 0 else ce * 0.0
            loss = ce + float(cfg["lambda_dice"]) * dice

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite teacher loss at epoch {epoch}")
            scaler.scale(loss).backward()
            if float(cfg["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(teacher.parameters(), float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += float(loss.item())
            steps += 1

        val_metrics = evaluate_pytorch_model(
            teacher,
            val_loader,
            device=device,
            split="validation",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, steps),
            "val_miou": val_metrics["miou"],
            "val_safety_miou": val_metrics["safety_critical_miou"],
            "val_pixel_acc": val_metrics["pixel_accuracy"],
            "val_ece": val_metrics["ece"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": round(time.perf_counter() - ep_t0, 2),
        }
        history.append(epoch_row)
        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = float(val_metrics["miou"])
            best_epoch = epoch
            best_val_metrics = val_metrics
            save_model_checkpoint(
                teacher,
                ckpt_path,
                role="teacher",
                arch=variant,
                extra={"seed": seed, "best_epoch": epoch, "val_miou": best_miou},
            )
        if bool(cfg.get("verbose", True)):
            star = " *" if is_best else ""
            print(
                f"[Teacher {variant} E{epoch:02d}/{int(cfg['epochs']):02d}] "
                f"loss={epoch_row['train_loss']:.4f} val_mIoU={val_metrics['miou']*100:.2f}% "
                f"safety={val_metrics['safety_critical_miou']*100:.2f}% "
                f"lr={epoch_row['lr']:.2e} ({epoch_row['epoch_seconds']:.1f}s){star}",
                flush=True,
            )

    reloaded, _ = load_model_checkpoint(ckpt_path, map_location=device)
    reloaded.to(device).eval()
    verified_val = evaluate_pytorch_model(
        reloaded, val_loader, device=device, split="validation", use_amp=use_amp, amp_dtype=amp_dtype
    )

    write_json(history, output / "history.json")
    write_json(verified_val, output / "validation-metrics.json")
    report = {
        "schema_version": 2,
        "status": "completed",
        "role": "teacher",
        "variant": variant,
        "seed": seed,
        "configuration": cfg,
        "configuration_sha256": canonical_sha256(cfg),
        "manifest_sha256": manifest["manifest_sha256"],
        "best_epoch": best_epoch,
        "validation_miou": verified_val["miou"],
        "validation_safety_miou": verified_val["safety_critical_miou"],
        "validation_ece": verified_val["ece"],
        "parameter_count": count_parameters(reloaded),
        "model_filename": ckpt_path.name,
        "model_sha256": file_sha256(ckpt_path),
        "weights_sha256": weights_sha256(reloaded),
        "seconds": round(time.perf_counter() - started, 3),
        "runtime": detect_runtime_metadata(device, precision=str(cfg["precision"])),
        "best_logged_val_miou": best_val_metrics.get("miou"),
    }
    write_json(report, output / "training-report.json")
    return {"model_path": str(ckpt_path.resolve()), "report_path": str((output / "training-report.json").resolve()), "report": report}


def train_student(
    dataset_root: str | Path,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    arch: str = "mobilenetv4_small",
    method: str = "cwd",
    seed: int = 378,
    teacher_path: str | Path | None = None,
    initial_weights_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    device_spec: str = "auto",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train a quantization-friendly Edge CNN student with paired controls & teacher immutability."""
    if method not in VALID_KD_METHODS:
        raise ValueError(f"Unknown KD method {method!r}. Supported: {VALID_KD_METHODS}")
    cfg = {
        "epochs": 60,
        "batch_size": 16,
        "image_size": [512, 1024],
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "temperature": 4.0,
        "lambda_dice": 0.25,
        "lambda_logit_kd": 3.0,
        "lambda_feat_kd": 1.5,
        "feature_stages": ["s8", "s16", "s32"],
        "safety_boost": 2.0,
        "grad_clip": 2.0,
        "ema_decay": 0.995,
        "fpn_dim": None,
        "dropout": 0.1,
        "pretrained_backbone": False,
        "precision": "bf16",
        "num_workers": 4,
        "deterministic": False,
        "compile_model": False,
        "channels_last": True,
        "asymmetric_robust_prob": 0.0,
        "qat_epochs": 0,
        "qat_lr_scale": 0.1,
        "verbose": True,
    }
    if config:
        cfg.update(dict(config))

    device = resolve_device(device_spec)
    output = _prepare_output_dir(output_dir, overwrite=overwrite)
    configure_seed(seed, deterministic=bool(cfg["deterministic"]))
    use_amp, amp_dtype, use_scaler = _resolve_amp(device, str(cfg["precision"]))

    image_size = (int(cfg["image_size"][0]), int(cfg["image_size"][1]))
    asym_prob = float(cfg.get("asymmetric_robust_prob", 0.0))
    train_records = records_for_split(manifest, "train")
    val_records = records_for_split(manifest, "validation")
    train_loader = make_dataloader(
        train_records,
        dataset_root,
        image_size=image_size,
        batch_size=int(cfg["batch_size"]),
        training=True,
        seed=seed,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
        adverse_aug_prob=0.0 if asym_prob > 0.0 else 0.15,
    )
    val_loader = make_dataloader(
        val_records,
        dataset_root,
        image_size=image_size,
        batch_size=max(1, int(cfg["batch_size"])),
        training=False,
        seed=seed,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    student = build_student(
        arch=arch,
        num_classes=int(manifest.get("num_classes", NUM_CLASSES)),
        pretrained=bool(cfg["pretrained_backbone"]),
        fpn_dim=cfg.get("fpn_dim"),
        dropout=float(cfg["dropout"]),
    )
    initial_file_sha = None
    if initial_weights_path is not None:
        init_path = Path(initial_weights_path)
        initial_file_sha = file_sha256(init_path)
        init_payload = torch.load(init_path, map_location="cpu", weights_only=True)
        student.load_state_dict(init_payload["state_dict"], strict=True)
    initial_weights_sha = weights_sha256(student)
    quant_audit = audit_quantization_friendliness(student)
    if not quant_audit["passed"]:
        raise AssertionError(f"Student failed quantization-friendliness audit: {quant_audit['violations']}")

    student = student.to(device)
    if bool(cfg.get("channels_last", False)) and device.type == "cuda":
        student = student.to(memory_format=torch.channels_last)

    teacher = None
    teacher_file_sha_before = None
    teacher_weights_sha_before = None
    projector_bank = None
    need_features = method in {"cwd_feat", "cwd_safety"}

    if method != "baseline":
        if teacher_path is None:
            raise ValueError(f"Method {method!r} requires teacher_path")
        t_path = Path(teacher_path)
        teacher_file_sha_before = file_sha256(t_path)
        teacher, _ = load_model_checkpoint(t_path, map_location=device)
        teacher.to(device).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher_weights_sha_before = weights_sha256(teacher)
        if need_features:
            projector_bank = FeatureProjectorBank(
                student.feature_channels,
                teacher.feature_channels,
                stages=tuple(cfg["feature_stages"]),
            ).to(device)

    criterion = SegmentationDistillationLoss(
        method=method,
        temperature=float(cfg["temperature"]),
        lambda_dice=float(cfg["lambda_dice"]),
        lambda_logit_kd=float(cfg["lambda_logit_kd"]),
        lambda_feat_kd=float(cfg["lambda_feat_kd"]),
        feature_stages=tuple(cfg["feature_stages"]),
        safety_boost=float(cfg["safety_boost"]),
        ignore_index=IGNORE_INDEX,
    )

    trainable_params = list(student.parameters()) + (list(projector_bank.parameters()) if projector_bank is not None else [])
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    total_steps = max(1, int(cfg["epochs"]) * max(1, len(train_loader)))
    warmup_steps = int(cfg["warmup_epochs"]) * max(1, len(train_loader))
    scheduler = _build_warmup_cosine_scheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    use_ema = float(cfg["ema_decay"]) > 0.0
    ema = ModelEMA(student, decay=float(cfg["ema_decay"])) if use_ema else None

    train_student_module: nn.Module = student
    if bool(cfg["compile_model"]) and device.type == "cuda" and hasattr(torch, "compile"):
        train_student_module = torch.compile(student)

    best_miou = -1.0
    best_epoch = 0
    best_source = "student"
    history: list[dict[str, Any]] = []
    ckpt_path = output / "student.pt"
    started = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        ep_t0 = time.perf_counter()
        student.train()
        if projector_bank is not None:
            projector_bank.train()
        sums = {"loss": 0.0, "hard_loss": 0.0, "dice_loss": 0.0, "logit_kd_loss": 0.0, "feature_kd_loss": 0.0}
        steps = 0

        eff_asym_prob = asym_prob * min(1.0, epoch / 5.0) if asym_prob > 0.0 else 0.0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            if bool(cfg.get("channels_last", False)) and device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            student_images = apply_gpu_adverse_corruption(images, prob=eff_asym_prob) if eff_asym_prob > 0.0 else images
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            teacher_logits = None
            teacher_feats = None
            if teacher is not None:
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    t_out = teacher(images, return_features=need_features)
                    if need_features:
                        teacher_logits = t_out["logits"]
                        teacher_feats = t_out["features"]
                    else:
                        teacher_logits = t_out

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                s_out = train_student_module(student_images, return_features=need_features)
                if need_features:
                    student_logits = s_out["logits"]
                    proj_feats = projector_bank(s_out["features"])
                else:
                    student_logits = s_out
                    proj_feats = None

            # Loss runs strictly in FP32 outside autocast for numerical stability.
            terms = criterion(
                student_logits,
                masks,
                teacher_logits=teacher_logits,
                projected_student_features=proj_feats,
                teacher_features=teacher_feats,
            )
            loss = terms["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite student loss at epoch {epoch}")

            scaler.scale(loss).backward()
            if float(cfg["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_params, float(cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(student)

            for k in sums:
                sums[k] += float(terms[k].item())
            steps += 1

        eval_model = ema.ema if ema is not None else student
        val_metrics = evaluate_pytorch_model(
            eval_model, val_loader, device=device, split="validation", use_amp=use_amp, amp_dtype=amp_dtype
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / max(1, steps) for k, v in sums.items()},
            "val_miou": val_metrics["miou"],
            "val_safety_miou": val_metrics["safety_critical_miou"],
            "val_pixel_acc": val_metrics["pixel_accuracy"],
            "val_ece": val_metrics["ece"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": round(time.perf_counter() - ep_t0, 2),
        }
        history.append(row)

        is_best = val_metrics["miou"] > best_miou
        if is_best:
            best_miou = float(val_metrics["miou"])
            best_epoch = epoch
            best_source = "ema" if ema is not None else "student"
            # Always save plain EdgeSegStudent weights only (never projectors or teacher).
            save_model_checkpoint(
                eval_model,
                ckpt_path,
                role="student",
                arch=arch,
                extra={
                    "method": method,
                    "seed": seed,
                    "best_epoch": epoch,
                    "source": best_source,
                    "val_miou": best_miou,
                },
            )
        if bool(cfg.get("verbose", True)):
            star = " *" if is_best else ""
            print(
                f"[Student {arch}/{method} s{seed} E{epoch:02d}/{int(cfg['epochs']):02d}] "
                f"loss={row['train_loss']:.4f} (ce={row['train_hard_loss']:.3f} kd={row['train_logit_kd_loss']:.3f} feat={row['train_feature_kd_loss']:.3f}) "
                f"val_mIoU={val_metrics['miou']*100:.2f}% safety={val_metrics['safety_critical_miou']*100:.2f}% "
                f"({row['epoch_seconds']:.1f}s){star}",
                flush=True,
            )

    # Optional QAT + KD fine-tuning phase (straight-through quantization noise awareness).
    qat_summary = None
    if int(cfg.get("qat_epochs", 0)) > 0:
        qat_summary = _run_qat_kd_finetune(
            ckpt_path=ckpt_path,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=int(cfg["qat_epochs"]),
            lr=float(cfg["learning_rate"]) * float(cfg.get("qat_lr_scale", 0.1)),
            weight_decay=float(cfg["weight_decay"]),
            grad_clip=float(cfg["grad_clip"]),
            output_path=output / "student-qat.pt",
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            asymmetric_robust_prob=min(0.25, asym_prob) if asym_prob > 0.0 else 0.0,
        )

    # Verify strict immutability of teacher and shared initial weights.
    teacher_weights_sha_after = None
    if teacher is not None:
        teacher_weights_sha_after = weights_sha256(teacher)
        if teacher_weights_sha_after != teacher_weights_sha_before:
            raise AssertionError("Teacher weights mutated during distillation!")
        if teacher_path is not None and file_sha256(teacher_path) != teacher_file_sha_before:
            raise AssertionError("Teacher checkpoint file mutated during distillation!")
    if initial_weights_path is not None and file_sha256(initial_weights_path) != initial_file_sha:
        raise AssertionError("Shared initial weights file mutated during student training!")

    # Reload plain student checkpoint from disk and verify validation metric parity.
    reloaded, _ = load_model_checkpoint(ckpt_path, map_location=device)
    reloaded = reloaded.to(device)
    if bool(cfg.get("channels_last", False)) and device.type == "cuda":
        reloaded = reloaded.to(memory_format=torch.channels_last)
    reloaded.eval()
    verified_val = evaluate_pytorch_model(
        reloaded, val_loader, device=device, split="validation", use_amp=use_amp, amp_dtype=amp_dtype
    )
    reload_tol = 2e-3 if use_amp else 1e-4
    if abs(verified_val["miou"] - best_miou) > reload_tol:
        raise AssertionError(
            f"Reloaded student validation mIoU ({verified_val['miou']:.6f}) differs from best logged ({best_miou:.6f})"
        )

    write_json(history, output / "history.json")
    write_json(verified_val, output / "validation-metrics.json")
    report = {
        "schema_version": 2,
        "status": "completed",
        "role": "student",
        "arch": arch,
        "method": method,
        "seed": seed,
        "configuration": cfg,
        "configuration_sha256": canonical_sha256(cfg),
        "manifest_sha256": manifest["manifest_sha256"],
        "initial_weights_sha256": initial_weights_sha,
        "initial_weights_file_sha256": initial_file_sha,
        "teacher_file_sha256": teacher_file_sha_before,
        "teacher_weights_before_sha256": teacher_weights_sha_before,
        "teacher_weights_after_sha256": teacher_weights_sha_after,
        "teacher_immutable": teacher_weights_sha_before == teacher_weights_sha_after,
        "quantization_audit": quant_audit,
        "best_epoch": best_epoch,
        "best_source": best_source,
        "validation_miou": verified_val["miou"],
        "validation_safety_miou": verified_val["safety_critical_miou"],
        "validation_pixel_accuracy": verified_val["pixel_accuracy"],
        "validation_ece": verified_val["ece"],
        "parameter_count": count_parameters(reloaded),
        "model_filename": ckpt_path.name,
        "model_sha256": file_sha256(ckpt_path),
        "weights_sha256": weights_sha256(reloaded),
        "qat_finetune": qat_summary,
        "seconds": round(time.perf_counter() - started, 3),
        "runtime": detect_runtime_metadata(device, precision=str(cfg["precision"])),
    }
    write_json(report, output / "training-report.json")
    return {"model_path": str(ckpt_path.resolve()), "report_path": str((output / "training-report.json").resolve()), "report": report}


def _fake_quantize_per_channel(weight: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Differentiable symmetric per-output-channel STE fake quantization for QAT-KD."""
    qmax = float(2 ** (bits - 1) - 1)
    reduce_dims = tuple(range(1, weight.ndim))
    scale = weight.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-6) / qmax
    q = torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale
    return weight + (q - weight).detach()


def _run_qat_kd_finetune(
    *,
    ckpt_path: Path,
    teacher: nn.Module | None,
    train_loader,
    val_loader,
    criterion: SegmentationDistillationLoss,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    output_path: Path,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    asymmetric_robust_prob: float = 0.0,
) -> dict[str, Any]:
    """Lightweight weight-STE QAT + logit-KD fine-tuning that preserves clean FP32 exportability."""
    student, payload = load_model_checkpoint(ckpt_path, map_location=device)
    student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)

    # Register forward pre-hooks simulating INT8 symmetric weight quantization during training.
    handles = []
    conv_originals: dict[nn.Module, torch.Tensor] = {}

    def _make_pre_hook():
        def pre_hook(mod: nn.Conv2d, _inp):
            if mod.training:
                conv_originals[mod] = mod.weight
                mod._parameters["weight"] = _fake_quantize_per_channel(mod.weight, bits=8)
        return pre_hook

    def _make_post_hook():
        def post_hook(mod: nn.Conv2d, _inp, _out):
            if mod in conv_originals:
                mod._parameters["weight"] = conv_originals.pop(mod)
        return post_hook

    for mod in student.modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_pre_hook(_make_pre_hook()))
            handles.append(mod.register_forward_hook(_make_post_hook()))

    best_miou = -1.0
    best_epoch = 0
    # Use logit-only KD criterion during QAT fine-tuning (no feature projectors needed).
    qat_method = "baseline" if teacher is None else ("cwd" if criterion.method != "pixel_kl" else "pixel_kl")
    qat_criterion = SegmentationDistillationLoss(
        method=qat_method,
        temperature=criterion.temperature,
        lambda_dice=criterion.lambda_dice,
        lambda_logit_kd=criterion.lambda_logit_kd,
        ignore_index=criterion.ignore_index,
    )

    try:
        for epoch in range(1, epochs + 1):
            student.train()
            for batch in train_loader:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                stu_images = (
                    apply_gpu_adverse_corruption(images, prob=asymmetric_robust_prob)
                    if asymmetric_robust_prob > 0.0
                    else images
                )
                optimizer.zero_grad(set_to_none=True)
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        teacher_logits = teacher(images, return_features=False)
                student_logits = student(stu_images, return_features=False)
                loss = qat_criterion(student_logits, masks, teacher_logits=teacher_logits)["loss"]
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()

            val_metrics = evaluate_pytorch_model(
                student, val_loader, device=device, split="validation", use_amp=use_amp, amp_dtype=amp_dtype
            )
            if val_metrics["miou"] > best_miou:
                best_miou = float(val_metrics["miou"])
                best_epoch = epoch
                save_model_checkpoint(
                    student,
                    output_path,
                    role="student",
                    arch=payload["arch"],
                    extra={"qat_kd": True, "best_epoch": epoch, "val_miou": best_miou},
                )
    finally:
        for h in handles:
            h.remove()
        for mod, orig in list(conv_originals.items()):
            mod._parameters["weight"] = orig
        conv_originals.clear()

    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "validation_miou": best_miou,
        "model_filename": output_path.name,
        "model_sha256": file_sha256(output_path),
    }
