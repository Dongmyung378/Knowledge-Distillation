"""Frozen evaluation, 19-class & safety-critical mIoU, ECE calibration, and weather/time slicing."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .data import canonical_sha256, file_sha256, make_dataloader, records_for_split
from .models import count_parameters, load_model_checkpoint, weights_sha256
from .taxonomy import (
    CLASS_NAMES,
    IGNORE_INDEX,
    NUM_CLASSES,
    SAFETY_CRITICAL_CLASSES,
    SAFETY_CRITICAL_INDICES,
)


class SegmentationMeter:
    """Exact confusion-matrix + ECE + attribute-slice accumulator for semantic segmentation."""

    def __init__(self, num_classes: int = NUM_CLASSES, *, ignore_index: int = IGNORE_INDEX, ece_bins: int = 15) -> None:
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ece_bins = int(ece_bins)
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.slice_confusions: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        )
        self.bin_counts = np.zeros(self.ece_bins, dtype=np.int64)
        self.bin_correct = np.zeros(self.ece_bins, dtype=np.float64)
        self.bin_conf_sum = np.zeros(self.ece_bins, dtype=np.float64)
        self.image_count = 0

    def _accumulate_cm(self, cm: np.ndarray, preds: np.ndarray, targets: np.ndarray) -> None:
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        if not np.any(valid):
            return
        t = targets[valid].astype(np.int64)
        p = preds[valid].astype(np.int64)
        idx = t * self.num_classes + p
        cm += np.bincount(idx, minlength=self.num_classes * self.num_classes).reshape(self.num_classes, self.num_classes)

    def update(
        self,
        logits: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        *,
        weathers: list[str] | None = None,
        timeofdays: list[str] | None = None,
    ) -> None:
        if isinstance(logits, np.ndarray):
            logits_t = torch.from_numpy(logits).float()
        else:
            logits_t = logits.detach().float().cpu()
        if isinstance(targets, np.ndarray):
            targets_t = torch.from_numpy(targets).long()
        else:
            targets_t = targets.detach().long().cpu()

        if logits_t.shape[-2:] != targets_t.shape[-2:]:
            logits_t = F.interpolate(logits_t, size=targets_t.shape[-2:], mode="bilinear", align_corners=False)

        probs = F.softmax(logits_t, dim=1)
        conf, preds = probs.max(dim=1)

        preds_np = preds.numpy()
        targets_np = targets_t.numpy()
        conf_np = conf.numpy()

        batch_size = int(targets_np.shape[0])
        self.image_count += batch_size
        self._accumulate_cm(self.confusion, preds_np, targets_np)

        # Slice confusion matrices by weather and timeofday (e.g., daytime / night / rainy).
        for i in range(batch_size):
            if timeofdays is not None:
                key = f"timeofday:{timeofdays[i]}"
                self._accumulate_cm(self.slice_confusions[key], preds_np[i], targets_np[i])
            if weathers is not None:
                key = f"weather:{weathers[i]}"
                self._accumulate_cm(self.slice_confusions[key], preds_np[i], targets_np[i])

        # Pixel-level Expected Calibration Error (ECE) over valid pixels.
        valid = (targets_np != self.ignore_index) & (targets_np >= 0) & (targets_np < self.num_classes)
        if np.any(valid):
            v_conf = np.clip(conf_np[valid], 0.0, 1.0)
            v_corr = (preds_np[valid] == targets_np[valid]).astype(np.float64)
            bin_idx = np.minimum((v_conf * self.ece_bins).astype(np.int64), self.ece_bins - 1)
            self.bin_counts += np.bincount(bin_idx, minlength=self.ece_bins)
            self.bin_correct += np.bincount(bin_idx, weights=v_corr, minlength=self.ece_bins)
            self.bin_conf_sum += np.bincount(bin_idx, weights=v_conf, minlength=self.ece_bins)

    @staticmethod
    def _metrics_from_cm(cm: np.ndarray) -> dict[str, Any]:
        tp = np.diag(cm).astype(np.float64)
        gt = cm.sum(axis=1).astype(np.float64)
        pred = cm.sum(axis=0).astype(np.float64)
        union = gt + pred - tp
        valid_classes = union > 0
        ious = np.full(cm.shape[0], np.nan, dtype=np.float64)
        ious[valid_classes] = tp[valid_classes] / union[valid_classes]

        recalls = np.full(cm.shape[0], np.nan, dtype=np.float64)
        gt_present = gt > 0
        recalls[gt_present] = tp[gt_present] / gt[gt_present]

        total_pixels = float(cm.sum())
        pixel_acc = float(tp.sum() / total_pixels) if total_pixels > 0 else 0.0
        miou = float(np.nanmean(ious)) if np.any(valid_classes) else 0.0
        mean_acc = float(np.nanmean(recalls)) if np.any(gt_present) else 0.0

        safety_vals = [ious[idx] for idx in SAFETY_CRITICAL_INDICES if idx < len(ious) and not np.isnan(ious[idx])]
        safety_miou = float(np.mean(safety_vals)) if safety_vals else 0.0

        return {
            "miou": miou,
            "safety_critical_miou": safety_miou,
            "pixel_accuracy": pixel_acc,
            "mean_class_accuracy": mean_acc,
            "valid_pixels": int(total_pixels),
            "present_classes": int(valid_classes.sum()),
            "per_class_iou": {
                CLASS_NAMES[i]: (None if np.isnan(ious[i]) else round(float(ious[i]), 6))
                for i in range(min(cm.shape[0], len(CLASS_NAMES)))
            },
        }

    def compute(self, *, split: str = "validation") -> dict[str, Any]:
        summary = self._metrics_from_cm(self.confusion)
        total_ece_pixels = int(self.bin_counts.sum())
        if total_ece_pixels > 0:
            non_empty = self.bin_counts > 0
            bin_acc = np.zeros_like(self.bin_correct)
            bin_conf = np.zeros_like(self.bin_conf_sum)
            bin_acc[non_empty] = self.bin_correct[non_empty] / self.bin_counts[non_empty]
            bin_conf[non_empty] = self.bin_conf_sum[non_empty] / self.bin_counts[non_empty]
            weights = self.bin_counts / float(total_ece_pixels)
            ece = float(np.sum(weights * np.abs(bin_acc - bin_conf)))
        else:
            ece = 0.0

        slices = {}
        for slice_key, cm in sorted(self.slice_confusions.items()):
            if int(cm.sum()) == 0:
                continue
            s_res = self._metrics_from_cm(cm)
            slices[slice_key] = {
                "miou": round(s_res["miou"], 6),
                "safety_critical_miou": round(s_res["safety_critical_miou"], 6),
                "pixel_accuracy": round(s_res["pixel_accuracy"], 6),
                "valid_pixels": s_res["valid_pixels"],
            }

        return {
            "schema_version": 2,
            "split": split,
            "image_count": self.image_count,
            "miou": round(summary["miou"], 6),
            "miou_percent": round(summary["miou"] * 100.0, 3),
            "safety_critical_classes": list(SAFETY_CRITICAL_CLASSES),
            "safety_critical_miou": round(summary["safety_critical_miou"], 6),
            "safety_critical_miou_percent": round(summary["safety_critical_miou"] * 100.0, 3),
            "pixel_accuracy": round(summary["pixel_accuracy"], 6),
            "mean_class_accuracy": round(summary["mean_class_accuracy"], 6),
            "ece": round(ece, 6),
            "valid_pixels": summary["valid_pixels"],
            "present_classes": summary["present_classes"],
            "per_class_iou": summary["per_class_iou"],
            "slices": slices,
        }


@torch.no_grad()
def evaluate_pytorch_model(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    split: str = "validation",
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    model.eval()
    meter = SegmentationMeter()
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(images, return_features=False) if _supports_return_features(model) else model(images)
        logits = out["logits"] if isinstance(out, dict) else out
        meter.update(
            logits,
            masks,
            weathers=list(batch["weather"]),
            timeofdays=list(batch["timeofday"]),
        )
    return meter.compute(split=split)


def _supports_return_features(model: torch.nn.Module) -> bool:
    return hasattr(model, "feature_channels")


def evaluate_onnx_model(
    onnx_path: str | Path,
    dataloader,
    *,
    split: str = "validation",
    threads: int = 1,
    provider: str = "CPUExecutionProvider",
) -> dict[str, Any]:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = int(threads)
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=[provider])
    input_name = session.get_inputs()[0].name

    meter = SegmentationMeter()
    for batch in dataloader:
        images = batch["image"].numpy().astype(np.float32)
        masks = batch["mask"]
        logits_np = session.run(None, {input_name: images})[0]
        meter.update(
            logits_np,
            masks,
            weathers=list(batch["weather"]),
            timeofdays=list(batch["timeofday"]),
        )
    result = meter.compute(split=split)
    result["runtime_backend"] = f"onnxruntime:{provider}"
    result["model_sha256"] = file_sha256(onnx_path)
    result["model_bytes"] = Path(onnx_path).stat().st_size
    return result


def evaluate_artifact(
    model_path: str | Path,
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    split: str = "locked_eval",
    image_size: tuple[int, int] = (512, 1024),
    batch_size: int = 4,
    num_workers: int = 0,
    device_spec: str = "cpu",
    threads: int = 1,
    freeze_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(model_path)
    if freeze_path is not None:
        validate_evaluation_freeze(freeze_path, manifest=manifest, model_path=path)

    records = records_for_split(manifest, split)
    loader = make_dataloader(
        records,
        dataset_root,
        image_size=image_size,
        batch_size=batch_size,
        training=False,
        num_workers=num_workers,
        pin_memory=device_spec.startswith("cuda"),
    )
    if path.suffix.lower() == ".onnx":
        result = evaluate_onnx_model(path, loader, split=split, threads=threads)
    elif path.suffix.lower() in {".pt", ".pth"}:
        device = torch.device(device_spec if (device_spec != "auto") else ("cuda" if torch.cuda.is_available() else "cpu"))
        model, payload = load_model_checkpoint(path, map_location=device)
        model.to(device).eval()
        result = evaluate_pytorch_model(model, loader, device=device, split=split)
        result["runtime_backend"] = f"pytorch:{device}"
        result["model_sha256"] = file_sha256(path)
        result["weights_sha256"] = weights_sha256(model)
        result["parameter_count"] = count_parameters(model)
        result["arch"] = payload.get("arch")
        result["role"] = payload.get("role")
        result["model_bytes"] = path.stat().st_size
    else:
        raise ValueError(f"Unsupported model extension: {path.suffix}")

    result["manifest_sha256"] = manifest["manifest_sha256"]
    result["image_size"] = [int(image_size[0]), int(image_size[1])]
    if freeze_path is not None:
        result["freeze_sha256"] = file_sha256(freeze_path)
    return result


def create_evaluation_freeze(
    manifest: Mapping[str, Any],
    model_paths: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    plan_path: str | Path | None = None,
    note: str = "Frozen before locked_eval or zero_shot_eval inference.",
) -> dict[str, Any]:
    """Cryptographically freeze manifest and candidate models prior to locked evaluation."""
    models_block = {}
    for label, p in sorted(model_paths.items()):
        path = Path(p)
        models_block[label] = {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    freeze = {
        "schema_version": 2,
        "dataset": manifest["dataset"],
        "manifest_sha256": manifest["manifest_sha256"],
        "inventory_sha256": manifest["inventory_sha256"],
        "plan_sha256": file_sha256(plan_path) if plan_path is not None else None,
        "models": models_block,
        "note": note,
    }
    freeze["freeze_sha256"] = canonical_sha256(freeze)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return freeze


def validate_evaluation_freeze(
    freeze_path: str | Path,
    *,
    manifest: Mapping[str, Any],
    model_path: str | Path,
) -> dict[str, Any]:
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    expected = canonical_sha256({k: v for k, v in freeze.items() if k != "freeze_sha256"})
    if freeze.get("freeze_sha256") != expected:
        raise ValueError("Evaluation freeze checksum mismatch")
    if freeze.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise ValueError("Manifest checksum does not match evaluation freeze")
    model_sha = file_sha256(model_path)
    recorded_shas = {entry["sha256"] for entry in freeze.get("models", {}).values()}
    if model_sha not in recorded_shas:
        raise ValueError(f"Model {Path(model_path).name} SHA-256 is not present in evaluation freeze")
    return freeze


@torch.no_grad()
def fit_temperature_scaling(
    model: torch.nn.Module,
    val_loader,
    *,
    device: torch.device,
    t_min: float = 0.40,
    t_max: float = 2.60,
    coarse_steps: int = 45,
) -> float:
    """Fit a single scalar temperature T* minimizing pixel NLL strictly on validation."""
    model.eval()
    temps = torch.linspace(float(t_min), float(t_max), int(coarse_steps), device=device)
    nll_sums = torch.zeros_like(temps)
    total_valid = 0

    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        out = model(images, return_features=False) if _supports_return_features(model) else model(images)
        logits = (out["logits"] if isinstance(out, dict) else out).float()
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        valid_px = int((masks != IGNORE_INDEX).sum().item())
        if valid_px == 0:
            continue
        total_valid += valid_px
        for ti, t_val in enumerate(temps):
            nll_sums[ti] += F.cross_entropy(logits / t_val, masks, ignore_index=IGNORE_INDEX, reduction="sum")

    if total_valid == 0:
        return 1.0
    t_coarse = float(temps[int(torch.argmin(nll_sums / float(total_valid)).item())].item())

    fine_temps = torch.linspace(max(0.15, t_coarse - 0.05), t_coarse + 0.05, 11, device=device)
    fine_sums = torch.zeros_like(fine_temps)
    for batch in val_loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        out = model(images, return_features=False) if _supports_return_features(model) else model(images)
        logits = (out["logits"] if isinstance(out, dict) else out).float()
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        for ti, t_val in enumerate(fine_temps):
            fine_sums[ti] += F.cross_entropy(logits / t_val, masks, ignore_index=IGNORE_INDEX, reduction="sum")
    return round(float(fine_temps[int(torch.argmin(fine_sums).item())].item()), 4)


@torch.no_grad()
def evaluate_temperature_scaled_model(
    model: torch.nn.Module,
    dataloader,
    *,
    temperature: float,
    device: torch.device,
    split: str = "locked_eval",
) -> dict[str, Any]:
    """Compare raw (T=1.0) vs temperature-scaled (T=temperature) ECE and mIoU parity."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    meter_raw = SegmentationMeter()
    meter_cal = SegmentationMeter()
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"]
        out = model(images, return_features=False) if _supports_return_features(model) else model(images)
        logits = (out["logits"] if isinstance(out, dict) else out).float()
        weathers = list(batch["weather"])
        timeofdays = list(batch["timeofday"])
        meter_raw.update(logits, masks, weathers=weathers, timeofdays=timeofdays)
        meter_cal.update(logits / float(temperature), masks, weathers=weathers, timeofdays=timeofdays)

    raw_res = meter_raw.compute(split=split)
    cal_res = meter_cal.compute(split=split)
    return {
        "split": split,
        "temperature": round(float(temperature), 4),
        "miou_before": raw_res["miou"],
        "miou_after": cal_res["miou"],
        "ece_before": raw_res["ece"],
        "ece_after": cal_res["ece"],
        "ece_reduction_pct": round((1.0 - cal_res["ece"] / max(raw_res["ece"], 1e-9)) * 100.0, 2),
    }
