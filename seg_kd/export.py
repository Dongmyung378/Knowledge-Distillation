"""Plain ONNX FP32 export, train-calibrated Static INT8 (QDQ) quantization, and acceptance gating."""

from __future__ import annotations

import inspect
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data import DrivingSegDataset, canonical_sha256, file_sha256, records_for_split
from .models import audit_quantization_friendliness, count_parameters, load_model_checkpoint


def select_calibration_records(
    manifest: Mapping[str, Any],
    *,
    count: int = 128,
    seed: int = 378,
) -> list[dict[str, Any]]:
    """Select train-only records for INT8 calibration (strictly prohibiting val/eval leakage)."""
    train_records = records_for_split(manifest, "train")
    if count < 1:
        raise ValueError("calibration count must be positive")
    rng = random.Random(seed)
    ordered = sorted(train_records, key=lambda r: r["id"])
    rng.shuffle(ordered)
    selected = ordered[: min(count, len(ordered))]
    if any(r["split"] != "train" for r in selected):
        raise ValueError("Calibration leakage detected: non-training record selected")
    return selected


class _PlainLogitsExportWrapper(torch.nn.Module):
    """Single-argument forward wrapper guaranteeing clean ONNX tracing across PyTorch versions."""

    def __init__(self, student: torch.nn.Module) -> None:
        super().__init__()
        self.student = student

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.student(image)


def export_student_onnx(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    dataset_root: str | Path | None = None,
    image_size: tuple[int, int] = (512, 1024),
    calibration_count: int = 64,
    seed: int = 378,
    opset_version: int = 17,
    export_int8: bool = True,
) -> dict[str, Any]:
    """Export plain EdgeSegStudent to ONNX FP32 and (optionally) Static QDQ INT8.

    Verifies PyTorch vs ONNX FP32 numerical parity on sample inputs before writing
    ``export-report.json``.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'onnx' package is required for ONNX export. Install dependencies via: pip install -r requirements-l4.txt"
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, payload = load_model_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    if payload.get("role") != "student":
        raise ValueError("export_student_onnx requires a plain student checkpoint")

    audit = audit_quantization_friendliness(model)
    height, width = int(image_size[0]), int(image_size[1])
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)

    fp32_path = out_dir / "student-fp32.onnx"
    export_module = _PlainLogitsExportWrapper(model).eval()
    # Export plain single-input / single-output graph (no projectors, no teacher).
    export_kwargs: dict[str, Any] = {
        "input_names": ["image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "do_constant_folding": True,
        "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(export_module, (dummy,), str(fp32_path), **export_kwargs)

    parity: dict[str, Any] = {"checked": False}
    int8_info: dict[str, Any] = {"exported": False, "reason": "disabled"}

    try:
        import onnxruntime as ort
    except ImportError:
        parity = {"checked": False, "reason": "onnxruntime not installed"}
        int8_info = {"exported": False, "reason": "onnxruntime not installed"}
    else:
        sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pt_logits = model(dummy).numpy()
        ort_logits = sess.run(None, {"image": dummy.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(pt_logits - ort_logits)))
        argmax_agree = float(np.mean(pt_logits.argmax(axis=1) == ort_logits.argmax(axis=1)))
        if not np.isfinite(ort_logits).all():
            raise ValueError("Exported FP32 ONNX produced non-finite logits")
        parity = {
            "checked": True,
            "max_abs_logit_diff": round(max_abs_diff, 6),
            "pixel_argmax_agreement": round(argmax_agree, 6),
        }

        if export_int8:
            if manifest is None or dataset_root is None:
                int8_info = {"exported": False, "reason": "manifest and dataset_root required for train-only calibration"}
            else:
                int8_info = _quantize_onnx_static_int8(
                    fp32_path=fp32_path,
                    out_dir=out_dir,
                    manifest=manifest,
                    dataset_root=dataset_root,
                    image_size=(height, width),
                    calibration_count=calibration_count,
                    seed=seed,
                )

    report = {
        "schema_version": 2,
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "arch": payload["arch"],
        "parameter_count": count_parameters(model),
        "image_size": [height, width],
        "opset_version": opset_version,
        "quantization_audit": audit,
        "fp32": {
            "filename": fp32_path.name,
            "bytes": fp32_path.stat().st_size,
            "sha256": file_sha256(fp32_path),
            "parity": parity,
        },
        "int8": int8_info,
        "acceptance_policy": {
            "max_miou_drop_pp": 1.0,
            "max_safety_miou_drop_pp": 1.5,
            "minimum_cpu_p50_speedup": 1.20,
            "requires_smaller_file_than_fp32": True,
            "selection_split": "validation",
        },
    }
    (out_dir / "export-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _quantize_onnx_static_int8(
    *,
    fp32_path: Path,
    out_dir: Path,
    manifest: Mapping[str, Any],
    dataset_root: str | Path,
    image_size: tuple[int, int],
    calibration_count: int,
    seed: int,
) -> dict[str, Any]:
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    chosen = select_calibration_records(manifest, count=calibration_count, seed=seed)
    dataset = DrivingSegDataset(chosen, dataset_root, image_size=image_size, training=False)

    class _SegCalibrationReader(CalibrationDataReader):
        def __init__(self) -> None:
            self._idx = 0

        def get_next(self) -> dict[str, np.ndarray] | None:
            if self._idx >= len(dataset):
                return None
            sample = dataset[self._idx]
            self._idx += 1
            img = sample["image"].unsqueeze(0).numpy().astype(np.float32)
            return {"image": img}

        def rewind(self) -> None:
            self._idx = 0

    prep_path = out_dir / "student-fp32-prep.onnx"
    int8_path = out_dir / "student-int8.onnx"
    quant_input_path = fp32_path
    try:
        quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=False)
        quant_input_path = prep_path
    except Exception:
        try:
            quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)
            quant_input_path = prep_path
        except Exception:
            quant_input_path = fp32_path
    quantize_static(
        model_input=str(quant_input_path),
        model_output=str(int8_path),
        calibration_data_reader=_SegCalibrationReader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    if prep_path.is_file():
        prep_path.unlink()

    calib_manifest = {
        "schema_version": 2,
        "split": "train",
        "seed": seed,
        "count": len(chosen),
        "manifest_sha256": manifest["manifest_sha256"],
        "record_ids": [r["id"] for r in chosen],
        "records_sha256": canonical_sha256([r["id"] for r in chosen]),
    }
    calib_path = out_dir / "calibration-manifest.json"
    calib_path.write_text(json.dumps(calib_manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "exported": True,
        "filename": int8_path.name,
        "bytes": int8_path.stat().st_size,
        "sha256": file_sha256(int8_path),
        "quant_format": "QDQ",
        "per_channel": True,
        "calibration_count": len(chosen),
        "calibration_manifest_sha256": file_sha256(calib_path),
    }


def assess_int8_acceptance(
    fp32_val_metrics: Mapping[str, Any],
    int8_val_metrics: Mapping[str, Any],
    fp32_benchmark: Mapping[str, Any],
    int8_benchmark: Mapping[str, Any],
    *,
    max_miou_drop_pp: float = 1.0,
    max_safety_miou_drop_pp: float = 1.5,
    minimum_speedup: float = 1.20,
) -> dict[str, Any]:
    """Validation-only deployment gate for accepting INT8 over FP32."""
    if fp32_val_metrics.get("split") != "validation" or int8_val_metrics.get("split") != "validation":
        raise ValueError("INT8 acceptance gate must use validation metrics only (never locked_eval)")

    miou_drop_pp = 100.0 * (float(fp32_val_metrics["miou"]) - float(int8_val_metrics["miou"]))
    safety_drop_pp = 100.0 * (
        float(fp32_val_metrics["safety_critical_miou"]) - float(int8_val_metrics["safety_critical_miou"])
    )
    fp32_p50 = float(fp32_benchmark["latency_ms"]["p50"])
    int8_p50 = float(int8_benchmark["latency_ms"]["p50"])
    speedup = fp32_p50 / max(int8_p50, 1e-9)
    size_ratio = float(int8_benchmark["model_bytes"]) / max(1.0, float(fp32_benchmark["model_bytes"]))

    checks = {
        "miou_drop_within_limit": miou_drop_pp <= max_miou_drop_pp,
        "safety_miou_drop_within_limit": safety_drop_pp <= max_safety_miou_drop_pp,
        "file_size_smaller": int(int8_benchmark["model_bytes"]) < int(fp32_benchmark["model_bytes"]),
        "cpu_p50_speedup_met": speedup >= minimum_speedup,
    }
    accepted = all(checks.values())
    return {
        "accepted": accepted,
        "selected_format": "INT8" if accepted else "FP32",
        "checks": checks,
        "miou_drop_pp": round(miou_drop_pp, 4),
        "safety_miou_drop_pp": round(safety_drop_pp, 4),
        "cpu_p50_speedup": round(speedup, 4),
        "file_size_ratio_int8_over_fp32": round(size_ratio, 4),
        "thresholds": {
            "max_miou_drop_pp": max_miou_drop_pp,
            "max_safety_miou_drop_pp": max_safety_miou_drop_pp,
            "minimum_speedup": minimum_speedup,
        },
    }
