"""Warmup + timed latency/FPS benchmarking on CPU (ONNX Runtime / PyTorch) and CUDA (L4)."""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import file_sha256
from .models import count_parameters, load_model_checkpoint


def _summarize_timings(timings_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(timings_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mean = float(arr.mean())
    return {
        "mean": round(mean, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "fps_p50": round(1000.0 / max(p50, 1e-9), 2),
        "fps_mean": round(1000.0 / max(mean, 1e-9), 2),
    }


def benchmark_model(
    model_path: str | Path,
    *,
    image_size: tuple[int, int] = (512, 1024),
    batch_size: int = 1,
    warmup: int = 10,
    runs: int = 50,
    device_spec: str = "cpu",
    precision: str = "fp32",
    threads: int = 4,
) -> dict[str, Any]:
    """Measure pure forward-pass latency (p50/p95 ms) and FPS under controlled conditions."""
    path = Path(model_path)
    height, width = int(image_size[0]), int(image_size[1])
    ext = path.suffix.lower()

    if ext == ".onnx":
        return _benchmark_onnx(
            path,
            image_size=(height, width),
            batch_size=batch_size,
            warmup=warmup,
            runs=runs,
            device_spec=device_spec,
            threads=threads,
        )
    if ext in {".pt", ".pth"}:
        return _benchmark_pytorch(
            path,
            image_size=(height, width),
            batch_size=batch_size,
            warmup=warmup,
            runs=runs,
            device_spec=device_spec,
            precision=precision,
            threads=threads,
        )
    raise ValueError(f"Unsupported model file for benchmarking: {path}")


@torch.no_grad()
def _benchmark_pytorch(
    path: Path,
    *,
    image_size: tuple[int, int],
    batch_size: int,
    warmup: int,
    runs: int,
    device_spec: str,
    precision: str,
    threads: int,
) -> dict[str, Any]:
    if device_spec == "cpu":
        torch.set_num_threads(int(threads))
    device = torch.device(device_spec if device_spec != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_model_checkpoint(path, map_location=device)
    model.to(device).eval()

    x = torch.randn(batch_size, 3, image_size[0], image_size[1], device=device, dtype=torch.float32)
    use_amp = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    def step() -> None:
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _ = model(x)

    for _ in range(max(1, warmup)):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "schema_version": 2,
        "backend": "pytorch",
        "device": str(device),
        "precision": precision,
        "threads": int(threads) if device.type == "cpu" else None,
        "host": platform.processor() or platform.machine(),
        "platform": platform.system(),
        "model_filename": path.name,
        "model_sha256": file_sha256(path),
        "model_bytes": path.stat().st_size,
        "model_mib": round(path.stat().st_size / (1024**2), 3),
        "arch": payload.get("arch"),
        "role": payload.get("role"),
        "parameter_count": count_parameters(model),
        "image_size": [image_size[0], image_size[1]],
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "runs": int(runs),
        "latency_ms": _summarize_timings(timings_ms),
    }


def _benchmark_onnx(
    path: Path,
    *,
    image_size: tuple[int, int],
    batch_size: int,
    warmup: int,
    runs: int,
    device_spec: str,
    threads: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    provider = "CUDAExecutionProvider" if device_spec.startswith("cuda") else "CPUExecutionProvider"
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = int(threads)
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=sess_opts, providers=[provider])
    input_name = session.get_inputs()[0].name
    x = np.random.default_rng(378).standard_normal((batch_size, 3, image_size[0], image_size[1])).astype(np.float32)

    for _ in range(max(1, warmup)):
        session.run(None, {input_name: x})

    timings_ms: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "schema_version": 2,
        "backend": f"onnxruntime:{provider}",
        "device": device_spec,
        "precision": "int8" if "int8" in path.name.lower() else "fp32",
        "threads": int(threads) if provider == "CPUExecutionProvider" else None,
        "host": platform.processor() or platform.machine(),
        "platform": platform.system(),
        "model_filename": path.name,
        "model_sha256": file_sha256(path),
        "model_bytes": path.stat().st_size,
        "model_mib": round(path.stat().st_size / (1024**2), 3),
        "image_size": [image_size[0], image_size[1]],
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "runs": int(runs),
        "latency_ms": _summarize_timings(timings_ms),
    }
