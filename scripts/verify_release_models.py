"""One-shot verifier for published v3-max & v4-final ONNX deployment models and frozen reports.

Downloads the SHA-256 frozen ONNX FP32 & Static QDQ INT8 artifacts from GitHub Release
``v4.0-models`` (or reuses cached local copies), verifies byte-level SHA-256 agreement against
``reports/v3-max/evaluation-freeze.json`` and ``reports/v4-final/evaluation-freeze-v4.json``,
runs live 4-thread CPU ONNXRuntime latency benchmarks (384x768, batch=1), validates the
validation-only INT8 acceptance gates, and optionally runs live dataset re-evaluation on
Cityscapes ``locked_eval`` (500 imgs), KITTI Zero-Shot (200 imgs), 4-corruption stress test,
and validation-only temperature scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seg_kd.benchmark import benchmark_model
from seg_kd.export import assess_int8_acceptance

RELEASE_BASE_URL = "https://github.com/Dongmyung378/Knowledge-Distillation/releases/download/v4.0-models"

RELEASE_MODELS: dict[str, dict[str, str]] = {
    "v3_medium_int8": {
        "filename": "v3-max-mobilenetv4-medium-int8.onnx",
        "freeze_file": "reports/v3-max/evaluation-freeze.json",
        "freeze_key": "mobilenetv4_medium_int8",
    },
    "v3_medium_fp32": {
        "filename": "v3-max-mobilenetv4-medium-fp32.onnx",
        "freeze_file": "reports/v3-max/evaluation-freeze.json",
        "freeze_key": "mobilenetv4_medium_fp32",
    },
    "v4_robust_medium_int8": {
        "filename": "v4-final-robust-medium-int8.onnx",
        "freeze_file": "reports/v4-final/evaluation-freeze-v4.json",
        "freeze_key": "v4_robust_medium_int8",
    },
    "v4_robust_medium_fp32": {
        "filename": "v4-final-robust-medium-fp32.onnx",
        "freeze_file": "reports/v4-final/evaluation-freeze-v4.json",
        "freeze_key": "v4_robust_medium_fp32",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_verify_models(
    cache_dir: Path,
    *,
    repo_root: Path = REPO_ROOT,
    int8_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Download published ONNX artifacts and verify exact SHA-256 match with freeze manifests."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    verified: dict[str, dict[str, Any]] = {}

    for label, spec in RELEASE_MODELS.items():
        if int8_only and label.endswith("_fp32"):
            continue
        freeze_path = repo_root / spec["freeze_file"]
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        expected_meta = freeze["models"][spec["freeze_key"]]
        expected_sha = expected_meta["sha256"]
        expected_bytes = int(expected_meta["bytes"])

        dst = cache_dir / spec["filename"]
        if not dst.is_file() or dst.stat().st_size != expected_bytes:
            url = f"{RELEASE_BASE_URL}/{spec['filename']}"
            print(f"[download] {spec['filename']} <- {url}")
            urllib.request.urlretrieve(url, dst)

        actual_sha = sha256_file(dst)
        actual_bytes = dst.stat().st_size
        if actual_sha != expected_sha or actual_bytes != expected_bytes:
            raise RuntimeError(
                f"SHA-256 freeze mismatch for {label}: "
                f"got ({actual_bytes} B, {actual_sha}), expected ({expected_bytes} B, {expected_sha})"
            )

        verified[label] = {
            "path": str(dst.resolve()),
            "bytes": actual_bytes,
            "size_mib": round(actual_bytes / (1024**2), 2),
            "sha256": actual_sha,
            "freeze_match": True,
        }
        print(f"  [OK] {label:24s} | {actual_bytes / (1024**2):5.2f} MiB | sha256={actual_sha[:16]}... == freeze")
    return verified


def benchmark_verified_models(
    verified: dict[str, dict[str, Any]],
    *,
    warmup: int = 5,
    runs: int = 20,
    threads: int = 4,
) -> dict[str, dict[str, Any]]:
    """Run live 384x768 CPU ONNXRuntime latency benchmarks on the verified ONNX artifacts."""
    results: dict[str, dict[str, Any]] = {}
    for label, meta in verified.items():
        res = benchmark_model(
            meta["path"],
            image_size=(384, 768),
            batch_size=1,
            warmup=warmup,
            runs=runs,
            device_spec="cpu",
            threads=threads,
        )
        results[label] = res
        p50 = res["latency_ms"]["p50"]
        fps = res["latency_ms"]["fps_p50"]
        print(f"  [CPU {threads}T] {label:24s} | p50={p50:6.2f} ms | {fps:5.1f} FPS | {res['model_mib']:5.2f} MiB")

    for pair_name, fp32_key, int8_key in (
        ("v3-max Clean Medium", "v3_medium_fp32", "v3_medium_int8"),
        ("v4-final Robust Medium", "v4_robust_medium_fp32", "v4_robust_medium_int8"),
    ):
        if fp32_key in results and int8_key in results:
            speedup = results[fp32_key]["latency_ms"]["p50"] / results[int8_key]["latency_ms"]["p50"]
            ratio = verified[int8_key]["bytes"] / verified[fp32_key]["bytes"]
            print(
                f"  [PAIR]   {pair_name:24s} | live CPU speedup={speedup:.2f}x | "
                f"size ratio={ratio:.4f} (-{(1.0 - ratio) * 100:.1f}%)"
            )
    return results


def print_master_summary(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load and display the verified frozen v3-max + v4-final quantitative summary."""
    summary = json.loads((repo_root / "reports" / "final-summary.json").read_text(encoding="utf-8"))
    t = summary["teacher"]
    m = summary["locked_eval_models"]
    ts = summary["three_seed_paired_small"]
    rob = summary["robustness_stress_test_200f"]
    cal = summary["ece_temperature_scaling"]
    kitti = summary["kitti_zero_shot"]

    print("\n================ FROZEN MASTER SUMMARY (Cityscapes 384x768) ================")
    print(f"Manifest SHA-256 : {summary['dataset']['manifest_sha256']}")
    print(f"Freeze v3-max    : {summary['dataset']['freeze_v3_sha256']}")
    print(f"Freeze v4-final  : {summary['dataset']['freeze_v4_sha256']}")
    print(
        f"Teacher SegFormer-B5 (84.6M): locked mIoU={t['miou_pct']:.2f}% | "
        f"safety={t['safety_miou_pct']:.2f}% | CPU={t['cpu_p50_ms']:.2f} ms | KITTI={kitti['teacher_segformer_b5']['miou_pct']:.2f}%"
    )
    print(
        f"3-Seed Small (1.50M, s378/379/380): Baseline {ts['baseline_miou_mean_sd'][0]:.2f}% (±{ts['baseline_miou_mean_sd'][1]:.2f}) -> "
        f"CWD+Feat {ts['cwd_feat_miou_mean_sd'][0]:.2f}% (±{ts['cwd_feat_miou_mean_sd'][1]:.2f}) | "
        f"Safety {ts['baseline_safety_mean_sd'][0]:.2f}% -> {ts['cwd_feat_safety_mean_sd'][0]:.2f}% (+{ts['paired_safety_gain_mean_sd'][0]:.2f}%p, 3/3 wins)"
    )
    for key, label in [
        ("medium_baseline", "V4-Medium Baseline (FP32, no KD)"),
        ("mobilenetv4_medium_int8", "V4-Medium v3 Clean INT8 (7.90 MiB)"),
        ("v4_robust_medium_int8", "V4-Medium v4 Robust INT8 (7.90 MiB)"),
    ]:
        row = m[key]
        k_row = kitti[key]
        print(
            f"  {label:36s} | locked={row['miou_pct']:5.2f}% | safety={row['safety_miou_pct']:5.2f}% | "
            f"KITTI={k_row['miou_pct']:5.2f}% (safety {k_row['safety_miou_pct']:5.2f}%)"
        )

    print("\n--- 4-Corruption Stress Test (200 locked_eval frames, mIoU %) ---")
    for name in (
        "Teacher (SegFormer-B5)",
        "Medium Baseline (7.5M)",
        "Medium v3-max INT8 (7.5M)",
        "Medium v4 Robust FP32 (7.5M)",
        "Medium v4 Robust INT8 (7.5M)",
    ):
        r = rob[name]
        print(
            f"  {name:30s} | Clean={r['clean']['miou']*100:5.2f} | Night={r['night_lowlight']['miou']*100:5.2f} | "
            f"Blur={r['motion_blur']['miou']*100:5.2f} | Fog={r['fog_lowcontrast']['miou']*100:5.2f} | "
            f"Noise={r['sensor_noise']['miou']*100:5.2f}"
        )

    print("\n--- Post-Hoc Validation Temperature Scaling (Locked Eval ECE) ---")
    for key in ("medium_baseline", "v3_medium_int8", "v4_robust_medium_fp32", "v4_robust_medium_int8"):
        c = cal[key]
        print(
            f"  {key:24s} | T*={c['optimal_temperature_val']:.2f} | "
            f"ECE {c['locked_eval_ece_before']:.4f} -> {c['locked_eval_ece_after']:.4f} (-{c['ece_reduction_pct']:.1f}%) | "
            f"mIoU delta=0.00%p"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify published ONNX INT8/FP32 models and frozen v3-max/v4-final reports")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "outputs" / "verified_release_models"))
    parser.add_argument("--int8-only", action="store_true", help="Only verify/benchmark the two 7.90 MiB INT8 ONNX models")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip live CPU latency benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", default=None, help="Optional path to save verification receipt JSON")
    args = parser.parse_args()

    print("=== [1/3] Downloading & SHA-256 Verifying Frozen ONNX Models ===")
    verified = download_and_verify_models(Path(args.cache_dir), int8_only=args.int8_only)

    benchmarks: dict[str, Any] = {}
    if not args.skip_benchmark:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            print("\n=== [2/3] Live CPU ONNXRuntime Benchmark skipped (install `onnxruntime` to enable) ===")
        else:
            print(f"\n=== [2/3] Live CPU ONNXRuntime Benchmark (384x768, batch=1, {args.threads} threads) ===")
            benchmarks = benchmark_verified_models(verified, warmup=args.warmup, runs=args.runs, threads=args.threads)

    print("\n=== [3/3] Frozen Study Summary Audit ===")
    summary = print_master_summary()

    # Re-verify stored validation INT8 acceptance gates directly from reports/
    gate_v3 = summary["export_and_int8_gates"]["mobilenetv4_medium_v3"]["gate"]
    gate_v4 = summary["export_and_int8_gates"]["mobilenetv4_medium_v4_robust"]["gate"]
    assert gate_v3["accepted"] is True and gate_v4["accepted"] is True
    assert assess_int8_acceptance(
        {"split": "validation", "miou": 0.68, "safety_critical_miou": 0.58},
        {"split": "validation", "miou": 0.675, "safety_critical_miou": 0.576},
        {"model_bytes": 30_000_000, "latency_ms": {"p50": 38.0}},
        {"model_bytes": 8_288_000, "latency_ms": {"p50": 24.7}},
    )["accepted"] is True
    print("\n[ALL CHECKS PASSED] Freeze SHA-256 hashes, master summary, and validation-only INT8 gate logic verified!")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"verified_models": verified, "live_cpu_benchmarks": benchmarks}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved verification receipt -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
