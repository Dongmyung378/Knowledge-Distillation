"""Portfolio-grade visualizers for road-scene overlays, Pareto curves, and safety-class gains."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from .data import safe_dataset_path
from .models import load_model_checkpoint
from .taxonomy import (
    CLASS_NAMES,
    IGNORE_INDEX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SAFETY_CRITICAL_CLASSES,
    colorize_mask,
    map_mask_to_train_ids,
)


def _blend_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    color = colorize_mask(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * color
    return np.clip(blended, 0, 255).astype(np.uint8)


def _error_map(rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Green = correct, Red = misclassified, Dark = ignore_index."""
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    valid = gt != IGNORE_INDEX
    correct = valid & (pred == gt)
    wrong = valid & (pred != gt)
    out[correct] = np.clip(out[correct].astype(np.int32) + np.array([0, 120, 40]), 0, 255).astype(np.uint8)
    out[wrong] = np.clip(out[wrong].astype(np.int32) + np.array([210, 25, 25]), 0, 255).astype(np.uint8)
    return out


@torch.no_grad()
def render_prediction_panel(
    records: Sequence[Mapping[str, Any]],
    dataset_root: str | Path,
    model_checkpoints: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    image_size: tuple[int, int] = (512, 1024),
    device_spec: str = "cpu",
    max_samples: int = 4,
) -> Path:
    """Render RGB | Ground Truth | Model Overlays | Distilled Error Map grid."""
    device = torch.device(device_spec if device_spec != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded = {}
    for label, ckpt in model_checkpoints.items():
        model, _ = load_model_checkpoint(ckpt, map_location=device)
        model.to(device).eval()
        loaded[label] = model

    chosen = list(records)[:max_samples]
    model_names = list(loaded.keys())
    cols = 2 + len(model_names) + (1 if model_names else 0)
    rows = len(chosen)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.2 * rows), squeeze=False)
    target_h, target_w = int(image_size[0]), int(image_size[1])

    for r_idx, rec in enumerate(chosen):
        img_path = safe_dataset_path(dataset_root, rec["image_path"])
        mask_path = safe_dataset_path(dataset_root, rec["mask_path"])
        with Image.open(img_path) as im:
            rgb_img = im.convert("RGB").resize((target_w, target_h), Image.BILINEAR)
        with Image.open(mask_path) as mk:
            gt_raw = map_mask_to_train_ids(np.asarray(mk), rec.get("label_format", "train_id"))
            gt_resized = np.asarray(Image.fromarray(gt_raw, mode="L").resize((target_w, target_h), Image.NEAREST))

        rgb_np = np.asarray(rgb_img, dtype=np.uint8)
        inp = TF.normalize(TF.to_tensor(rgb_img), mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)

        axes[r_idx, 0].imshow(rgb_np)
        axes[r_idx, 0].set_title("Input RGB" if r_idx == 0 else "")
        axes[r_idx, 1].imshow(_blend_overlay(rgb_np, gt_resized))
        axes[r_idx, 1].set_title("Ground Truth" if r_idx == 0 else "")

        last_pred = None
        for c_offset, name in enumerate(model_names):
            logits = loaded[name](inp)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            last_pred = pred
            axes[r_idx, 2 + c_offset].imshow(_blend_overlay(rgb_np, pred))
            axes[r_idx, 2 + c_offset].set_title(name if r_idx == 0 else "")

        if last_pred is not None:
            axes[r_idx, cols - 1].imshow(_error_map(rgb_np, last_pred, gt_resized))
            axes[r_idx, cols - 1].set_title(f"{model_names[-1]} Error" if r_idx == 0 else "")

        for c in range(cols):
            axes[r_idx, c].axis("off")

    plt.tight_layout()
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_ablation_and_pareto(
    entries: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot mIoU vs Safety-Critical mIoU and (when latency is present) mIoU vs Latency Pareto."""
    labels = [str(e["label"]) for e in entries]
    mious = [float(e["miou"]) * 100.0 for e in entries]
    safety = [float(e["safety_critical_miou"]) * 100.0 for e in entries]
    has_latency = all(e.get("latency_p50_ms") is not None for e in entries)

    fig, axes = plt.subplots(1, 2 if has_latency else 1, figsize=(12 if has_latency else 7, 4.5))
    ax0 = axes[0] if has_latency else axes

    x = np.arange(len(labels))
    width = 0.36
    ax0.bar(x - width / 2, mious, width, label="Overall mIoU (%)", color="#2563eb")
    ax0.bar(x + width / 2, safety, width, label="Safety-Critical mIoU (%)", color="#dc2626")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=20, ha="right")
    ax0.set_ylabel("mIoU (%)")
    ax0.set_title("KD Method Comparison (Overall vs Safety-Critical)")
    ax0.grid(axis="y", alpha=0.3)
    ax0.legend()

    if has_latency:
        ax1 = axes[1]
        lats = [float(e["latency_p50_ms"]) for e in entries]
        sizes = [max(40.0, float(e.get("model_mib", 5.0)) * 18.0) for e in entries]
        ax1.scatter(lats, mious, s=sizes, c="#0f766e", alpha=0.85, edgecolors="black")
        for lbl, lx, my in zip(labels, lats, mious):
            ax1.annotate(lbl, (lx, my), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax1.set_xlabel("p50 Latency (ms, lower is faster)")
        ax1.set_ylabel("mIoU (%)")
        ax1.set_title("Accuracy-Latency Pareto Frontier (bubble = model MiB)")
        ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest


def plot_per_class_gain(
    baseline_metrics: Mapping[str, Any],
    distilled_metrics: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Plot per-class IoU delta (Distilled - Baseline) highlighting safety-critical classes."""
    base_pc = baseline_metrics["per_class_iou"]
    dist_pc = distilled_metrics["per_class_iou"]
    names, deltas, colors = [], [], []
    for cls_name in CLASS_NAMES:
        b = base_pc.get(cls_name)
        d = dist_pc.get(cls_name)
        if b is None or d is None:
            continue
        names.append(cls_name)
        deltas.append((float(d) - float(b)) * 100.0)
        colors.append("#dc2626" if cls_name in SAFETY_CRITICAL_CLASSES else "#2563eb")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(names))
    ax.bar(x, deltas, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("IoU Gain (%p, Distilled - Baseline)")
    ax.set_title("Per-Class IoU Gain (Red = Safety-Critical Classes)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return dest
