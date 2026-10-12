"""Same-image, same-thread CPU benchmarks in one fresh process per artifact."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from .evaluation import (canonical_sha256, read_manifest, records_for_evaluation,
                         safe_record_path, validate_freeze, verify_record_files)
from .runtime import configure_cpu, file_sha256, load_model, preprocess_image


def _rss_bytes() -> int | None:
    """Current resident process memory, not TensorFlow allocator or model size."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
            return int(counter.WorkingSetSize)
        return None
    path = Path("/proc/self/statm")
    if path.exists():
        return int(path.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


def timing_summary(seconds: list[float]) -> dict:
    values = np.asarray(seconds, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Timings must contain finite positive durations.")
    return {"count": int(len(values)), "mean_ms": float(values.mean() * 1000),
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "throughput_images_per_second": float(len(values) / values.sum()),
            "serial_total_seconds": float(values.sum())}


def _worker(request: dict) -> dict:
    framework_start = time.perf_counter()
    config = configure_cpu(request["threads"])
    framework_import_seconds = time.perf_counter() - framework_start
    memory_samples = []
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.01):
            value = _rss_bytes()
            if value is not None:
                memory_samples.append(value)

    memory_before = _rss_bytes()
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    try:
        load_start = time.perf_counter()
        runner = load_model(request["model_path"], threads=request["threads"])
        load_seconds = time.perf_counter() - load_start
        memory_after_load = _rss_bytes()
        paths = [safe_record_path(request["dataset_root"], r) for r in request["records"]]
        preload_start = time.perf_counter()
        images = [preprocess_image(path) for path in paths]
        preload_seconds = time.perf_counter() - preload_start
        for index in range(request["warmup"]):
            runner.predict(images[index % len(images)][None])
        pure, total, preprocess = [], [], []
        prediction_digest = hashlib.sha256()
        pure_predictions = []
        for _ in range(request["repetitions"]):
            for image in images:
                start = time.perf_counter()
                output = runner.predict(image[None])  # Includes synchronized host output.
                pure.append(time.perf_counter() - start)
                if not np.isfinite(output).all():
                    raise ValueError("Benchmark model produced nonfinite output.")
                pure_predictions.append(int(output.argmax()))
                prediction_digest.update(np.asarray(output.argmax(axis=1), dtype="<i4").tobytes())
        for repetition in range(request["repetitions"]):
            for index, path in enumerate(paths):
                start = time.perf_counter()
                image = preprocess_image(path)
                ready = time.perf_counter()
                output = runner.predict(image[None])
                end = time.perf_counter()
                preprocess.append(ready - start)
                total.append(end - start)
                if int(output.argmax()) != pure_predictions[repetition * len(paths) + index]:
                    raise ValueError("Pure and end-to-end preprocessing predictions differ.")
        memory_end = _rss_bytes()
        if memory_end is not None:
            memory_samples.append(memory_end)
        return {"schema_version": 1, "model_filename": Path(request["model_path"]).name,
                "model_sha256": file_sha256(request["model_path"]),
                "model_bytes": Path(request["model_path"]).stat().st_size,
                "model_parameters": runner.parameters, "device": "CPU",
                "threads": request["threads"], "inter_op_threads": 1,
                "warmup": request["warmup"], "repetitions": request["repetitions"],
                "sample_count": len(paths), "sample_ids_sha256": canonical_sha256([r["id"] for r in request["records"]]),
                "manifest_sha256": request["manifest_sha256"],
                "framework": config, "framework_import_seconds": framework_import_seconds,
                "model_load_seconds": load_seconds, "dataset_preload_seconds": preload_seconds,
                "pure_model": timing_summary(pure), "read_decode_resize": timing_summary(preprocess),
                "end_to_end": timing_summary(total),
                "memory": {"unit": "bytes", "process_rss_before_model_load": memory_before,
                           "process_rss_after_model_load": memory_after_load,
                           "sampled_peak_process_rss": max(memory_samples) if memory_samples else None,
                           "sampling_interval_ms": 10,
                           "scope": "Fresh process RSS including framework, model, image batch and temporary allocations; not model-only memory."},
                "prediction_ids_sha256": prediction_digest.hexdigest(),
                "synchronization": "Keras .numpy() or TFLite get_tensor() before stopping timer",
                "filesystem_cache_scope": "Warm OS cache; cache was not evicted between image reads.",
                "host": {"system": platform.system(), "machine": platform.machine(),
                         "processor": platform.processor(), "python": platform.python_version()},
                "raw_timings_seconds": {"pure_model": pure, "read_decode_resize": preprocess,
                                       "end_to_end": total}}
    finally:
        stop.set()
        monitor.join(timeout=2)
        gc.collect()


def benchmark_models(model_paths: dict, manifest, dataset_root, output_dir,
                      split: str = "validation", sample_count: int = 128,
                      warmup: int = 10, repetitions: int = 3, threads: int = 1,
                      freeze_path=None) -> dict:
    if not model_paths or min(sample_count, warmup, repetitions, threads) < 1:
        raise ValueError("Models and positive sample, warmup, repetition and thread counts are required.")
    if split != "validation":
        raise ValueError("Deployment latency uses validation images, never final holdout labels.")
    manifest = read_manifest(manifest)
    records = sorted(records_for_evaluation(manifest, split), key=lambda r: r["id"])
    indices = np.random.default_rng(378).permutation(len(records))[:min(sample_count, len(records))]
    chosen = [records[int(index)] for index in indices]
    verify_record_files(chosen, dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    results, raw = {}, {}
    root = Path(__file__).resolve().parents[1]
    for name, model in sorted(model_paths.items()):
        model = Path(model).resolve(strict=True)
        if freeze_path is not None:
            validate_freeze(freeze_path, manifest, model)
        request = {"model_path": str(model), "dataset_root": str(Path(dataset_root).resolve()),
                   "records": chosen, "warmup": warmup, "repetitions": repetitions, "threads": threads,
                   "manifest_sha256": canonical_sha256(manifest)}
        with tempfile.TemporaryDirectory(prefix="kd-cpu-benchmark-") as temporary:
            request_path = Path(temporary) / "request.json"
            result_path = Path(temporary) / "result.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS=str(threads),
                       TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS="1")
            completed = subprocess.run([sys.executable, "-m", "knowledge_distillation.benchmark", "--worker",
                                        str(request_path), str(result_path)], cwd=root, env=env,
                                       capture_output=True, text=True, encoding="utf-8", errors="replace")
            if completed.returncode != 0:
                raise RuntimeError(f"CPU benchmark failed for {name}: {completed.stderr[-6000:]}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            raw[name] = result.pop("raw_timings_seconds")
            results[name] = result
    report = {"schema_version": 1, "split": split, "device": "CPU", "batch_size": 1,
              "manifest_sha256": canonical_sha256(manifest), "sample_seed": 378,
              "sample_ids": [r["id"] for r in chosen], "models": results,
              "comparison_scope": "One fresh child process per model; same images and thread counts. No model.predict() overhead."}
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "raw-timings.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return report


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("REQUEST_JSON", "RESULT_JSON"), required=True)
    args = parser.parse_args()
    request_path, result_path = map(Path, args.worker)
    result = _worker(json.loads(request_path.read_text(encoding="utf-8")))
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    _main()
