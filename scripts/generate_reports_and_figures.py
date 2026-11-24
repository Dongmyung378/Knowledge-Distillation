"""Curate verified v3-max & v4-final JSON reports and render publication-quality figures.

Reads strictly from ``outputs/v3-max/`` and ``outputs/v4-final/`` (produced on Colab Enterprise
NVIDIA L4) and writes:
  - ``reports/final-summary.json`` + curated lightweight JSON artifacts under ``reports/``
  - ``assets/figures/kd-thumbnail.svg`` (Korean) & ``assets/figures/en/kd-thumbnail.svg`` (English)
  - ``assets/figures/kd-pareto-frontier.png`` (zero-overlap Pareto frontier)
  - ``assets/figures/kd-robustness-breakthrough.png`` (4-corruption stress test)
  - ``assets/figures/kd-method-ablation-384.png`` (384x768 4-method ablation)
  - ``assets/figures/kd-3seed-per-class.png`` (3-seed paired stats + 3-seed mean 19-class IoU delta)
  - ``assets/figures/kd-ece-calibration.png`` (validation temperature scaling ECE before/after)
  - ``assets/figures/kd-road-overlays-clean.png`` & ``kd-road-overlays-noise.png`` (web-optimized)
"""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_DIR = REPO_ROOT / "outputs" / "v3-max"
V4_DIR = REPO_ROOT / "outputs" / "v4-final"
REPORTS_DIR = REPO_ROOT / "reports"
FIG_DIR = REPO_ROOT / "assets" / "figures"

SAFETY_CLASSES = {"traffic_light", "traffic_sign", "person", "rider", "motorcycle", "bicycle"}
ALL_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def curate_reports() -> dict[str, Any]:
    """Copy verified JSON evidence from outputs/ into reports/ and build final-summary.json."""
    if REPORTS_DIR.exists():
        shutil.rmtree(REPORTS_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy all lightweight JSON reports from v3-max and v4-final (excluding training histories)
    for tag, src_root in (("v3-max", V3_DIR), ("v4-final", V4_DIR)):
        for p in sorted(src_root.rglob("*.json")):
            rel = p.relative_to(src_root)
            if rel.name in {"history.json", "calibration-manifest.json"}:
                continue
            dst = REPORTS_DIR / tag / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

    teacher_report = _load(V3_DIR / "teacher" / "segformer_b5" / "training-report.json")
    v4_master = _load(V4_DIR / "v4-final-master-summary.json")
    three_seed = _load(V3_DIR / "three-seed-stats.json")
    freeze_v3 = _load(V3_DIR / "evaluation-freeze.json")
    freeze_v4 = _load(V4_DIR / "evaluation-freeze-v4.json")
    kitti_v3 = _load(V3_DIR / "evaluations" / "kitti-zero-shot-summary.json")
    kitti_v4 = _load(V4_DIR / "evaluations" / "kitti-zero-shot-v4.json")
    ece_cal = _load(V4_DIR / "evaluations" / "ece-temperature-scaling.json")
    robust_v4 = _load(V4_DIR / "evaluations" / "robustness-stress-test-v4.json")

    # Compute exact 3-seed mean per-class IoU on locked_eval
    per_class_3seed: dict[str, dict[str, float]] = {}
    base_evals = [_load(V3_DIR / "evaluations" / "locked_eval" / f"s{s}_baseline.json") for s in (378, 379, 380)]
    kd_evals = [_load(V3_DIR / "evaluations" / "locked_eval" / f"s{s}_cwd_feat.json") for s in (378, 379, 380)]
    for cls in ALL_CLASSES:
        b_vals = [e["per_class_iou"][cls] * 100.0 for e in base_evals]
        k_vals = [e["per_class_iou"][cls] * 100.0 for e in kd_evals]
        deltas = [k - b for k, b in zip(k_vals, b_vals)]
        per_class_3seed[cls] = {
            "baseline_mean_pct": round(float(np.mean(b_vals)), 3),
            "cwd_feat_mean_pct": round(float(np.mean(k_vals)), 3),
            "delta_mean_pp": round(float(np.mean(deltas)), 3),
            "delta_sd_pp": round(float(np.std(deltas, ddof=1)), 3),
            "is_safety_critical": cls in SAFETY_CLASSES,
        }

    # Gather locked_eval + CPU/GPU benchmarks + export gates
    locked_models = {}
    for name, src_dir in [
        ("teacher_segformer_b5", V3_DIR),
        ("s378_baseline", V3_DIR),
        ("s378_pixel_kl", V4_DIR),
        ("s378_cwd", V4_DIR),
        ("s378_cwd_feat", V3_DIR),
        ("mobilenetv4_small_fp32", V3_DIR),
        ("mobilenetv4_small_int8", V3_DIR),
        ("mobilenetv3_large_fp32", V3_DIR),
        ("mobilenetv3_large_int8", V3_DIR),
        ("medium_baseline", V4_DIR),
        ("mobilenetv4_medium_fp32", V3_DIR),
        ("mobilenetv4_medium_int8", V3_DIR),
        ("v4_robust_medium_fp32", V4_DIR),
        ("v4_robust_medium_int8", V4_DIR),
    ]:
        ev = _load(src_dir / "evaluations" / "locked_eval" / f"{name}.json")
        cpu_path = src_dir / "benchmarks" / "cpu" / f"{name}.json"
        cpu = _load(cpu_path) if cpu_path.is_file() else None
        locked_models[name] = {
            "miou_pct": round(ev["miou"] * 100.0, 2),
            "safety_miou_pct": round(ev["safety_critical_miou"] * 100.0, 2),
            "pixel_accuracy_pct": round(ev["pixel_accuracy"] * 100.0, 2),
            "ece": round(ev["ece"], 4),
            "cpu_p50_ms": round(cpu["latency_ms"]["p50"], 2) if cpu else None,
            "cpu_fps_p50": round(cpu["latency_ms"]["fps_p50"], 2) if cpu else None,
            "model_mib": round(cpu["model_mib"], 2) if cpu else None,
            "parameter_count": cpu.get("parameter_count") if cpu else None,
        }

    # Exact GPU L4 BF16 benchmarks
    gpu_bench = {}
    for key, path in [
        ("teacher_segformer_b5", V3_DIR / "benchmarks" / "l4_cuda_bf16" / "teacher_segformer_b5.json"),
        ("s378_baseline", V3_DIR / "benchmarks" / "l4_cuda_bf16" / "s378_baseline.json"),
        ("s378_cwd_feat", V3_DIR / "benchmarks" / "l4_cuda_bf16" / "s378_cwd_feat.json"),
        ("v4_robust_medium", V4_DIR / "benchmarks" / "l4_cuda_bf16" / "v4_robust_medium.json"),
    ]:
        g = _load(path)
        gpu_bench[key] = {
            "l4_bf16_p50_ms": round(g["latency_ms"]["p50"], 2),
            "l4_bf16_fps_p50": round(g["latency_ms"]["fps_p50"], 2),
            "parameter_count": g.get("parameter_count"),
        }

    # Export gates
    export_gates = {}
    for key, exp_dir in [
        ("mobilenetv4_small", V3_DIR / "export" / "mobilenetv4_small"),
        ("mobilenetv3_large", V3_DIR / "export" / "mobilenetv3_large"),
        ("mobilenetv4_medium_v3", V3_DIR / "export" / "mobilenetv4_medium"),
        ("mobilenetv4_medium_v4_robust", V4_DIR / "export" / "v4_robust_medium"),
    ]:
        er = _load(exp_dir / "export-report.json")
        gt = _load(exp_dir / "int8-acceptance.json")
        export_gates[key] = {
            "parameter_count": er["parameter_count"],
            "conv_count": er["quantization_audit"]["conv_count"],
            "bn_count": er["quantization_audit"]["bn_count"],
            "fp32_mib": round(er["fp32"]["bytes"] / (1024**2), 2),
            "int8_mib": round(er["int8"]["bytes"] / (1024**2), 2),
            "size_reduction_pct": round((1.0 - er["int8"]["bytes"] / er["fp32"]["bytes"]) * 100.0, 1),
            "fp32_parity_argmax": er["fp32"]["parity"]["pixel_argmax_agreement"],
            "fp32_max_abs_logit_diff": er["fp32"]["parity"]["max_abs_logit_diff"],
            "gate": gt,
        }

    # Combine KITTI zero-shot
    kitti_combined = {}
    for k, v in kitti_v3.items():
        kitti_combined[k] = {
            "miou_pct": round(v["miou"] * 100.0, 2),
            "safety_miou_pct": round(v["safety_critical_miou"] * 100.0, 2),
            "ece": round(v["ece"], 4),
        }
    for k, v in kitti_v4.items():
        kitti_combined[k] = {
            "miou_pct": round(v["miou"] * 100.0, 2),
            "safety_miou_pct": round(v["safety_critical_miou"] * 100.0, 2),
            "ece": round(v["ece"], 4),
        }

    summary = {
        "schema_version": 4,
        "study": "driving-seg-kd-l4-v3max-v4final",
        "dataset": {
            "primary": "Cityscapes 19-class (Chris1/cityscapes, 384x768)",
            "splits": {"train": 2529, "validation": 446, "locked_eval": 500},
            "zero_shot": "KITTI Semantics (200 official images, zero fine-tuning)",
            "manifest_sha256": freeze_v3["manifest_sha256"],
            "freeze_v3_sha256": freeze_v3["freeze_sha256"],
            "freeze_v4_sha256": freeze_v4["freeze_sha256"],
        },
        "teacher": {
            "variant": "segformer_b5",
            "parameter_count": teacher_report["parameter_count"],
            "val_miou_pct": round(teacher_report["validation_miou"] * 100.0, 2),
            "val_safety_miou_pct": round(teacher_report["validation_safety_miou"] * 100.0, 2),
            "runtime": teacher_report["runtime"],
            **locked_models["teacher_segformer_b5"],
            **gpu_bench["teacher_segformer_b5"],
        },
        "three_seed_paired_small": {
            **three_seed,
            "seeds_detail": {
                str(s): {
                    "baseline_miou_pct": round(base_evals[i]["miou"] * 100.0, 2),
                    "baseline_safety_pct": round(base_evals[i]["safety_critical_miou"] * 100.0, 2),
                    "cwd_feat_miou_pct": round(kd_evals[i]["miou"] * 100.0, 2),
                    "cwd_feat_safety_pct": round(kd_evals[i]["safety_critical_miou"] * 100.0, 2),
                }
                for i, s in enumerate((378, 379, 380))
            },
            "per_class_3seed_mean": per_class_3seed,
        },
        "ablation_384_seed378": v4_master["ablation_384_seed378_locked_eval"],
        "locked_eval_models": locked_models,
        "l4_cuda_bf16_benchmarks": gpu_bench,
        "export_and_int8_gates": export_gates,
        "kitti_zero_shot": kitti_combined,
        "robustness_stress_test_200f": robust_v4,
        "ece_temperature_scaling": ece_cal,
    }
    _write_json(summary, REPORTS_DIR / "final-summary.json")
    return summary


def _set_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FCFDFB",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def plot_pareto_frontier(summary: dict[str, Any]) -> None:
    """Zero-overlap latency vs mIoU Pareto frontier on locked_eval (500 Cityscapes 384x768 imgs)."""
    _set_pub_style()
    m = summary["locked_eval_models"]

    # (key, label, color, marker, size, tx_data, ty_data, ha, rad)
    points = [
        ("mobilenetv4_small_int8", "V4-Small INT8\n61.43% · 15.9 ms · 1.76 MiB", "#0F766E", "D", 105, 9.2, 57.9, "left", 0.12),
        ("mobilenetv4_small_fp32", "V4-Small FP32\n61.95% · 16.8 ms · 5.72 MiB", "#0D9488", "o", 120, 38.0, 61.8, "left", -0.08),
        ("mobilenetv3_large_fp32", "V3-Large FP32\n64.84% · 24.2 ms · 12.6 MiB", "#D97706", "o", 145, 9.2, 66.0, "left", -0.10),
        ("mobilenetv3_large_int8", "V3-Large INT8 (REJECTED)\n59.32% · 33.2 ms (-5.52%p)", "#DC2626", "X", 150, 64.0, 58.1, "left", 0.08),
        ("mobilenetv4_medium_int8", f"V4-Med v3 Clean INT8 ★\n67.95% · 24.7 ms · 7.91 MiB\n({m['teacher_segformer_b5']['cpu_p50_ms'] / m['mobilenetv4_medium_int8']['cpu_p50_ms']:.1f}× faster · Gate PASS)", "#15803D", "*", 360, 9.2, 71.7, "left", -0.12),
        ("mobilenetv4_medium_fp32", "V4-Med v3 FP32\n68.51% · 38.1 ms · 28.7 MiB", "#2563EB", "o", 190, 74.0, 73.1, "left", 0.10),
        ("v4_robust_medium_fp32", "V4-Med v4 Robust FP32\n67.35% · 40.3 ms · 28.7 MiB", "#3B82F6", "s", 160, 145.0, 69.0, "left", -0.08),
        ("v4_robust_medium_int8", "V4-Med v4 Robust INT8 ★\n66.96% · 26.0 ms · 7.91 MiB\n(Noise 59.59% > Teacher!)", "#047857", "P", 220, 145.0, 63.7, "left", 0.12),
        ("teacher_segformer_b5", "Teacher SegFormer-B5\n74.83% · 1631 ms · 323 MiB", "#475569", "^", 240, 1420.0, 72.1, "right", -0.10),
    ]

    fig, ax = plt.subplots(figsize=(11.8, 6.4), dpi=240)

    # Pareto frontier dashed line
    frontier_keys = [
        "mobilenetv4_small_int8",
        "mobilenetv4_small_fp32",
        "mobilenetv4_medium_int8",
        "mobilenetv4_medium_fp32",
        "teacher_segformer_b5",
    ]
    fx = [m[k]["cpu_p50_ms"] for k in frontier_keys]
    fy = [m[k]["miou_pct"] for k in frontier_keys]
    ax.plot(fx, fy, linestyle="--", color="#16A34A", linewidth=1.8, alpha=0.55, zorder=2, label="Clean Pareto Frontier")

    # Medium Baseline reference line (no teacher)
    mb_miou = m["medium_baseline"]["miou_pct"]
    ax.axhline(mb_miou, color="#94A3B8", linestyle=":", linewidth=1.3, zorder=1)
    ax.text(
        1980, mb_miou - 0.45,
        f"MobileNetV4-Medium Baseline (no KD, FP32): {mb_miou:.2f}%",
        ha="right", va="top", fontsize=8.8, color="#475569", style="italic",
    )

    for key, label, color, marker, size, tx, ty, ha, rad in points:
        x = m[key]["cpu_p50_ms"]
        y = m[key]["miou_pct"]
        edge = "#0F172A" if marker in {"*", "P", "X"} else "#FFFFFF"
        lw = 1.1 if marker in {"*", "P", "X"} else 1.0
        ax.scatter([x], [y], s=size, c=color, marker=marker, edgecolors=edge, linewidths=lw, zorder=5)

        box_bg = "#F0FDF4" if "★" in label else ("#FEF2F2" if "REJECTED" in label else "#FFFFFF")
        box_ec = color if ("★" in label or "REJECTED" in label) else "#CBD5E1"
        fw = "bold" if ("★" in label or "REJECTED" in label) else "normal"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=8.3,
            fontweight=fw,
            color="#0F172A" if "REJECTED" not in label else "#991B1B",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=box_bg, edgecolor=box_ec, linewidth=0.95, alpha=0.97),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.05, alpha=0.82, shrinkA=2, shrinkB=6, connectionstyle=f"arc3,rad={rad}"),
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_xlim(8.4, 2250)
    ax.set_ylim(56.0, 77.2)
    ax.set_xticks([10, 15, 25, 40, 100, 300, 1000, 1631])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Single-Frame CPU Inference Latency p50 (ms, log scale, 4-thread ONNXRuntime, 384x768)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Cityscapes Locked Eval mIoU (%)", fontsize=10.5, labelpad=8)
    ax.set_title("Accuracy-Latency Pareto Frontier & Validation-Only INT8 Gate (Cityscapes 384x768, 500 Locked Eval)", fontsize=12.0, pad=12)
    ax.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "kd-pareto-frontier.png", dpi=240)
    plt.close(fig)


def plot_robustness_breakthrough(summary: dict[str, Any]) -> None:
    """4-corruption stress test (200 locked_eval frames) highlighting v4 Robust INT8 beating 84.6M teacher."""
    _set_pub_style()
    rob = summary["robustness_stress_test_200f"]

    conditions = ["clean", "night_lowlight", "motion_blur", "fog_lowcontrast", "sensor_noise"]
    cond_labels = ["Clean\n(Daylight)", "Night\n(Low-Light)", "Motion Blur\n(k=9)", "Fog\n(Low-Contrast)", "Sensor Noise\n(Gaussian σ=18)"]
    models = [
        ("Teacher (SegFormer-B5)", "Teacher SegFormer-B5 (84.6M, 323 MiB)", "#475569"),
        ("Medium Baseline (7.5M)", "V4-Med Baseline (7.5M, FP32)", "#94A3B8"),
        ("Medium v3-max INT8 (7.5M)", "V4-Med v3 Clean INT8 (7.91 MiB)", "#D97706"),
        ("Medium v4 Robust FP32 (7.5M)", "V4-Med v4 Robust FP32 (28.7 MiB)", "#2563EB"),
        ("Medium v4 Robust INT8 (7.5M)", "V4-Med v4 Robust INT8 (7.91 MiB ★)", "#15803D"),
    ]

    x = np.arange(len(conditions))
    width = 0.155
    fig, ax = plt.subplots(figsize=(12.2, 6.2), dpi=240)

    for i, (mkey, mlabel, color) in enumerate(models):
        vals = [rob[mkey][c]["miou"] * 100.0 for c in conditions]
        offset = (i - 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=mlabel, color=color, edgecolor="#FFFFFF", linewidth=0.6, zorder=3)
        for b, v in zip(bars, vals):
            if i in (0, 2, 4):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    v + 0.9,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    fontweight="bold" if i == 4 else "normal",
                    color="#0F172A" if i != 2 else "#92400E",
                )

    # Highlight the sensor noise breakthrough in completely clean upper-right space
    ax.annotate(
        "7.91 MiB INT8 (59.59%) surpasses 84.6M Teacher (59.19%)!\n+45.84%p recovery vs v3 Clean INT8 (13.75%)",
        xy=(4 + 2 * width, 64.4),
        xytext=(3.52, 81.5),
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#14532D",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#DCFCE7", edgecolor="#15803D", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#15803D", lw=1.6, connectionstyle="arc3,rad=-0.12"),
        zorder=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=9.8, fontweight="bold")
    ax.set_ylim(0, 89)
    ax.set_ylabel("Cityscapes Stress-Test mIoU (%)", fontsize=10.5)
    ax.set_title("Asymmetric Robustness KD (Clean Teacher -> Corrupted Student) Stress Test (200 Locked Eval Frames)", fontsize=11.8, pad=42)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "kd-robustness-breakthrough.png", dpi=240)
    plt.close(fig)


def plot_method_ablation_384(summary: dict[str, Any]) -> None:
    """Complete 384x768 4-method KD ablation on MobileNetV4-Small (1.50M, seed 378)."""
    _set_pub_style()
    ab = summary["ablation_384_seed378"]
    kitti = summary["kitti_zero_shot"]

    methods = ["baseline", "pixel_kl", "cwd", "cwd_feat"]
    labels = ["Baseline\n(CE Only)", "Pixel KL\n(Naive Softmax)", "CWD\n(Channel Spatial)", "CWD + 4-Stage\nHybrid Feat"]
    overall = [ab[k]["miou"] * 100.0 for k in methods]
    safety = [ab[k]["safety"] * 100.0 for k in methods]
    kitti_vals = [
        kitti["s378_baseline"]["miou_pct"],
        kitti["s378_pixel_kl"]["miou_pct"],
        kitti["s378_cwd"]["miou_pct"],
        kitti["s378_cwd_feat"]["miou_pct"],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.1), dpi=240, gridspec_kw={"width_ratios": [1.35, 1.0]})

    x = np.arange(len(methods))
    w = 0.36
    colors_o = ["#64748B", "#DC2626", "#15803D", "#2563EB"]
    colors_s = ["#94A3B8", "#F87171", "#4ADE80", "#60A5FA"]

    b1 = ax1.bar(x - w / 2, overall, w, color=colors_o, edgecolor="#FFFFFF", label="Overall 19-Class mIoU", zorder=3)
    b2 = ax1.bar(x + w / 2, safety, w, color=colors_s, edgecolor="#FFFFFF", label="Safety-6 Critical mIoU", zorder=3)

    base_o, base_s = overall[0], safety[0]
    for i, (bar_o, bar_s) in enumerate(zip(b1, b2)):
        vo, vs = overall[i], safety[i]
        d_o = "" if i == 0 else f"\n({vo - base_o:+.2f}%p)"
        d_s = "" if i == 0 else f"\n({vs - base_s:+.2f}%p)"
        ax1.text(bar_o.get_x() + bar_o.get_width() / 2, vo + 0.5, f"{vo:.2f}%{d_o}", ha="center", va="bottom", fontsize=7.8, fontweight="bold" if i == 2 else "normal", color="#0F172A")
        ax1.text(bar_s.get_x() + bar_s.get_width() / 2 + 0.03, vs + 0.5, f"{vs:.2f}%{d_s}", ha="center", va="bottom", fontsize=7.3, color="#334155")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.2)
    ax1.set_ylim(38, 74.5)
    ax1.set_ylabel("Cityscapes Locked Eval mIoU (%)", fontsize=10.0)
    ax1.set_title("In-Domain Cityscapes (384x768, 1.50M Student)", fontsize=11.0, pad=10)
    ax1.legend(loc="upper center", ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: KITTI Zero-Shot
    b3 = ax2.bar(x, kitti_vals, 0.52, color=colors_o, edgecolor="#FFFFFF", zorder=3)
    base_k = kitti_vals[0]
    for i, bar_k in enumerate(b3):
        vk = kitti_vals[i]
        dk = "" if i == 0 else f"\n({vk - base_k:+.2f}%p)"
        ax2.text(bar_k.get_x() + bar_k.get_width() / 2, vk + 0.45, f"{vk:.2f}%{dk}", ha="center", va="bottom", fontsize=8.4, fontweight="bold" if i == 2 else "normal", color="#0F172A")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Baseline", "Pixel KL", "CWD", "CWD+Feat"], fontsize=9.2)
    ax2.set_ylim(38, 53)
    ax2.set_ylabel("KITTI Zero-Shot mIoU (%, 200 imgs)", fontsize=10.0)
    ax2.set_title("Cross-Dataset KITTI Zero-Shot Transfer", fontsize=11.0, pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("384x768 KD Objective Ablation (56x ViT-to-CNN Compression, Seed 378)", fontsize=12.4, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(FIG_DIR / "kd-method-ablation-384.png", dpi=240)
    plt.close(fig)


def plot_3seed_and_per_class(summary: dict[str, Any]) -> None:
    """Left: 3-seed paired comparison; Right: exact 3-seed mean 19-class IoU delta (6/6 safety positive)."""
    _set_pub_style()
    ts = summary["three_seed_paired_small"]
    detail = ts["seeds_detail"]
    per_cls = ts["per_class_3seed_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 6.2), dpi=240, gridspec_kw={"width_ratios": [1.0, 1.38]})

    # Left panel: 3-seed paired overall & safety mIoU
    seeds = ["378", "379", "380"]
    b_o = [detail[s]["baseline_miou_pct"] for s in seeds]
    k_o = [detail[s]["cwd_feat_miou_pct"] for s in seeds]
    b_s = [detail[s]["baseline_safety_pct"] for s in seeds]
    k_s = [detail[s]["cwd_feat_safety_pct"] for s in seeds]

    groups = ["Overall mIoU\n(19 Classes)", "Safety-6 mIoU\n(Vulnerable Road Users)"]
    b_means = [ts["baseline_miou_mean_sd"][0], ts["baseline_safety_mean_sd"][0]]
    b_sds = [ts["baseline_miou_mean_sd"][1], ts["baseline_safety_mean_sd"][1]]
    k_means = [ts["cwd_feat_miou_mean_sd"][0], ts["cwd_feat_safety_mean_sd"][0]]
    k_sds = [ts["cwd_feat_miou_mean_sd"][1], ts["cwd_feat_safety_mean_sd"][1]]

    gx = np.arange(2)
    w = 0.32
    ax1.bar(gx - w / 2, b_means, w, yerr=b_sds, capsize=5, color="#64748B", edgecolor="#FFFFFF", label="Baseline (3-seed mean ± SD)", zorder=3)
    ax1.bar(gx + w / 2, k_means, w, yerr=k_sds, capsize=5, color="#15803D", edgecolor="#FFFFFF", label="CWD+Feat KD (3-seed mean ± SD)", zorder=3)

    # Overlay individual seed dots
    for i, (bv, kv) in enumerate(zip([b_o, b_s], [k_o, k_s])):
        jitter = np.array([-0.05, 0.0, 0.05])
        ax1.scatter(np.full(3, gx[i] - w / 2) + jitter, bv, color="#0F172A", s=24, zorder=5)
        ax1.scatter(np.full(3, gx[i] + w / 2) + jitter, kv, color="#052E16", s=24, zorder=5)

    ax1.text(0 - w / 2, b_means[0] + b_sds[0] + 0.6, f"{b_means[0]:.2f}%\n(±{b_sds[0]:.2f})", ha="center", va="bottom", fontsize=8.5, color="#334155")
    ax1.text(0 + w / 2, k_means[0] + k_sds[0] + 0.6, f"{k_means[0]:.2f}%\n(±{k_sds[0]:.2f})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#14532D")
    ax1.text(1 - w / 2, b_means[1] + b_sds[1] + 0.6, f"{b_means[1]:.2f}%\n(±{b_sds[1]:.2f})", ha="center", va="bottom", fontsize=8.5, color="#334155")
    ax1.text(1 + w / 2, k_means[1] + k_sds[1] + 0.6, f"{k_means[1]:.2f}%\n(±{k_sds[1]:.2f})", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#14532D")

    ax1.annotate("+0.61%p (3/3 wins)\nSD 2.7× smaller", xy=(0 + w / 2, 62.6), xytext=(0.05, 67.3), ha="center", fontsize=8.5, fontweight="bold", color="#15803D", bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor="#15803D"))
    ax1.annotate("+1.88%p (3/3 wins)\nSD 3.1× smaller", xy=(1 + w / 2, 52.0), xytext=(1.0, 57.5), ha="center", fontsize=8.5, fontweight="bold", color="#15803D", bbox=dict(boxstyle="round,pad=0.3", facecolor="#DCFCE7", edgecolor="#15803D"))

    ax1.set_xticks(gx)
    ax1.set_xticklabels(groups, fontsize=9.6, fontweight="bold")
    ax1.set_ylim(43, 71.5)
    ax1.set_ylabel("Locked Eval mIoU (%)", fontsize=10.2)
    ax1.set_title("Paired 3-Seed Reproducibility (Seeds 378, 379, 380)", fontsize=10.8, pad=10)
    ax1.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: 3-seed mean 19-class IoU delta sorted descending
    sorted_cls = sorted(ALL_CLASSES, key=lambda c: per_cls[c]["delta_mean_pp"])
    deltas = [per_cls[c]["delta_mean_pp"] for c in sorted_cls]
    bar_colors = []
    for c, d in zip(sorted_cls, deltas):
        if c in SAFETY_CLASSES:
            bar_colors.append("#15803D" if d >= 0 else "#B91C1C")
        else:
            bar_colors.append("#60A5FA" if d >= 0 else "#DC2626")

    y_pos = np.arange(len(sorted_cls))
    ax2.barh(y_pos, deltas, color=bar_colors, height=0.72, edgecolor="#FFFFFF", zorder=3)
    ax2.axvline(0.0, color="#334155", linewidth=1.0, zorder=4)

    ylabels = [f"★ {c}" if c in SAFETY_CLASSES else c for c in sorted_cls]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(ylabels, fontsize=8.6)
    for tick, c in zip(ax2.get_yticklabels(), sorted_cls):
        if c in SAFETY_CLASSES:
            tick.set_fontweight("bold")
            tick.set_color("#14532D")

    for i, d in enumerate(deltas):
        ha = "left" if d >= 0 else "right"
        pad = 0.18 if d >= 0 else -0.18
        ax2.text(d + pad, i, f"{d:+.2f}%p", va="center", ha=ha, fontsize=7.8, fontweight="bold" if sorted_cls[i] in SAFETY_CLASSES else "normal", color="#0F172A")

    ax2.set_xlim(-9.5, 4.6)
    ax2.set_xlabel("3-Seed Mean Per-Class IoU Gain: CWD+Feat minus Baseline (%p)", fontsize=9.8)
    ax2.set_title("Per-Class IoU Delta (★ = Safety-6: 6/6 Positive on 3-Seed Mean)", fontsize=10.8, pad=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "kd-3seed-per-class.png", dpi=240)
    plt.close(fig)


def plot_ece_calibration(summary: dict[str, Any]) -> None:
    """Validation-only post-hoc temperature scaling ECE reduction with 0.00%p mIoU change."""
    _set_pub_style()
    cal = summary["ece_temperature_scaling"]

    order = [
        ("s378_baseline", "V4-Small\nBaseline"),
        ("s378_cwd_feat", "V4-Small\nCWD+Feat"),
        ("medium_baseline", "V4-Medium\nBaseline"),
        ("v3_medium_int8", "V4-Med v3\nClean INT8"),
        ("v4_robust_medium_fp32", "V4-Med v4\nRobust FP32"),
        ("v4_robust_medium_int8", "V4-Med v4\nRobust INT8"),
    ]
    labels = [x[1] for x in order]
    before = [cal[x[0]]["locked_eval_ece_before"] for x in order]
    after = [cal[x[0]]["locked_eval_ece_after"] for x in order]
    temps = [cal[x[0]]["optimal_temperature_val"] for x in order]
    reds = [cal[x[0]]["ece_reduction_pct"] for x in order]

    x = np.arange(len(order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11.2, 5.0), dpi=240)

    b1 = ax.bar(x - w / 2, before, w, color="#F59E0B", edgecolor="#FFFFFF", label="Before Calibration (T = 1.00)", zorder=3)
    b2 = ax.bar(x + w / 2, after, w, color="#15803D", edgecolor="#FFFFFF", label="After Validation Temp Scaling (0.00%p mIoU change)", zorder=3)

    for i, (bar1, bar2) in enumerate(zip(b1, b2)):
        ax.text(bar1.get_x() + bar1.get_width() / 2, before[i] + 0.0006, f"{before[i]:.4f}", ha="center", va="bottom", fontsize=8.0, color="#92400E")
        ax.text(bar2.get_x() + bar2.get_width() / 2, after[i] + 0.0006, f"{after[i]:.4f}\n(-{reds[i]:.1f}%)", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#14532D")
        ax.text(
            x[i],
            max(before[i], after[i]) + 0.0038,
            f"T* = {temps[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            color="#1E3A8A" if temps[i] > 1.5 else "#475569",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#EFF6FF" if temps[i] > 1.5 else "#F1F5F9", edgecolor="#93C5FD" if temps[i] > 1.5 else "#CBD5E1", lw=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2, fontweight="bold")
    ax.set_ylim(0, 0.039)
    ax.set_ylabel("Locked Eval Expected Calibration Error (ECE, lower is better)", fontsize=10.0)
    ax.set_title("Post-Hoc Validation Temperature Scaling: Resolving CWD Confidence Over-Sharpening (T* ≈ 1.70)", fontsize=11.6, pad=12)
    ax.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "kd-ece-calibration.png", dpi=240)
    plt.close(fig)


def optimize_road_overlays() -> None:
    """Resize and optimize raw Colab road overlay PNGs for crisp fast-loading web/GitHub viewing."""
    pairs = [
        (V3_DIR / "figures" / "v3-max-road-overlays.png", FIG_DIR / "kd-road-overlays-clean.png"),
        (V4_DIR / "figures" / "v4-robust-overlays-noise.png", FIG_DIR / "kd-road-overlays-noise.png"),
    ]
    for src, dst in pairs:
        with Image.open(src) as im:
            im_rgb = im.convert("RGB")
            max_w = 1800
            if im_rgb.width > max_w:
                new_h = int(round(im_rgb.height * max_w / im_rgb.width))
                im_rgb = im_rgb.resize((max_w, new_h), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im_rgb.save(dst, format="PNG", optimize=True)


def render_thumbnail_svg(summary: dict[str, Any], *, lang: str) -> Path:
    """Render 1200x760 Swiss editorial SVG matching nlu/dm100/hex portfolio cards."""
    t = summary["teacher"]
    m = summary["locked_eval_models"]
    rob = summary["robustness_stress_test_200f"]
    v3_int8 = m["mobilenetv4_medium_int8"]
    v4_int8 = m["v4_robust_medium_int8"]
    speedup_vs_teacher = t["cpu_p50_ms"] / v3_int8["cpu_p50_ms"]
    t_noise = rob["Teacher (SegFormer-B5)"]["sensor_noise"]["miou"] * 100.0
    v4_noise = rob["Medium v4 Robust INT8 (7.5M)"]["sensor_noise"]["miou"] * 100.0
    v3_noise = rob["Medium v3-max INT8 (7.5M)"]["sensor_noise"]["miou"] * 100.0
    noise_gain = v4_noise - v3_noise

    if lang == "ko":
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">자율주행 도로 세그멘테이션 지식 증류: SegFormer-B5 → MobileNetV4 INT8</title>
<desc id="desc">84.6M SegFormer-B5 ViT 교사를 7.5M MobileNetV4-Medium ONNX INT8(7.91 MiB, CPU 40.5 FPS)로 이종 아키텍처 지식 증류하고, 비대칭 강건성 증류로 센서 노이즈 환경에서 교사(59.19%)를 추월(59.59%)한 결과 요약입니다.</desc>
<style>text{{font-family:'Noto Sans KR','Malgun Gothic','Apple SD Gothic Neo','Segoe UI',sans-serif;font-variant-numeric:tabular-nums}}</style>
<rect width="1200" height="760" fill="#F4F7F2"/>
<text x="48" y="76" fill="#182218" font-size="42" font-weight="750">자율주행 세그멘테이션 지식 증류</text>
<text x="48" y="122" fill="#4D5C4C" font-size="22">SegFormer-B5 (ViT) → MobileNetV4 (CNN) · 비대칭 강건성 KD · Validation-Only ONNX INT8 게이트</text>
<line x1="48" y1="148" x2="1152" y2="148" stroke="#D5E0D0" stroke-width="1.4"/>

<!-- Left Panel: Cross-Architecture Asymmetric KD -->
<rect x="48" y="172" width="434" height="410" rx="14" fill="#FFFFFF" stroke="#D5E0D0" stroke-width="1.3"/>
<text x="74" y="214" fill="#182218" font-size="24" font-weight="750">이종 아키텍처 비대칭 증류</text>
<text x="74" y="241" fill="#4D5C4C" font-size="15">Cityscapes 384×768 (19클래스) · KITTI Zero-Shot</text>

<rect x="74" y="264" width="382" height="94" rx="10" fill="#F4F7F2" stroke="#D5E0D0" stroke-width="1.2"/>
<text x="92" y="294" fill="#2B4C27" font-size="13.2" font-weight="700">TEACHER · CLEAN INPUT ONLY</text>
<text x="92" y="323" fill="#182218" font-size="19.2" font-weight="750">SegFormer-B5 (84.6M · 323 MiB)</text>
<text x="92" y="346" fill="#4D5C4C" font-size="13.5">mIoU {t['miou_pct']:.2f}% · Safety {t['safety_miou_pct']:.2f}% · CPU {t['cpu_p50_ms']:.0f} ms</text>

<path d="M265 358V404" stroke="#2B4C27" stroke-width="3" stroke-dasharray="5 4"/>
<path d="M257 395L265 405L273 395" fill="none" stroke="#2B4C27" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="108" y="369" width="314" height="24" rx="12" fill="#E3EFE0"/>
<text x="265" y="385" text-anchor="middle" fill="#1F421B" font-size="12.5" font-weight="700">CWD 공간 증류 + 4-Stage Cosine + QAT-KD</text>

<rect x="74" y="408" width="382" height="96" rx="10" fill="#F0F8EE" stroke="#527D49" stroke-width="1.6"/>
<text x="92" y="437" fill="#2B4C27" font-size="12.6" font-weight="700">STUDENT · CORRUPTED INPUT (야간·안개·블러·노이즈)</text>
<text x="92" y="466" fill="#182218" font-size="18.2" font-weight="750">MobileNetV4-Medium INT8 (7.91 MiB)</text>
<text x="92" y="490" fill="#2B4C27" font-size="13" font-weight="600">11.2× 파라미터 압축 · 40.9× 용량 축소 · CPU 38~40 FPS</text>

<rect x="74" y="520" width="382" height="42" rx="8" fill="#E3EFE0"/>
<text x="265" y="546" text-anchor="middle" fill="#1F421B" font-size="14.5" font-weight="700">SHA-256 동결 Locked Eval · 검증 전용 INT8 게이트</text>

<!-- Connectors -->
<path d="M482 377H534V233H586" fill="none" stroke="#7A8A78" stroke-width="3" stroke-linejoin="round"/>
<path d="M482 377H586" fill="none" stroke="#4D6B48" stroke-width="3"/>
<path d="M482 377H534V520H586" fill="none" stroke="#2B4C27" stroke-width="3.5" stroke-linejoin="round"/>
<circle cx="534" cy="377" r="5" fill="#182218"/>
<path d="M576 226L586 233L576 240" fill="none" stroke="#7A8A78" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M576 370L586 377L576 384" fill="none" stroke="#4D6B48" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M576 513L586 520L576 527" fill="none" stroke="#2B4C27" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>

<!-- Card 01 -->
<rect x="594" y="172" width="558" height="122" rx="14" fill="#FFFFFF" stroke="#D5E0D0" stroke-width="1.3"/>
<rect x="594" y="172" width="8" height="122" rx="4" fill="#7A8A78"/>
<text x="624" y="209" fill="#4D5C4C" font-size="14" font-weight="700">TRACK 01 · 3-SEED ABLATION &amp; GATE AUDIT (1.5M / 3.3M)</text>
<text x="624" y="243" fill="#182218" font-size="23" font-weight="750">3시드 3/3 승 · 안전 클래스 6/6 전원 상승</text>
<text x="624" y="274" fill="#4D5C4C" font-size="14.2">CWD +1.42%p (Pixel KL -2.41%p) · V3-Large INT8 붕괴(-6.46%p) 자동 차단</text>

<!-- Card 02 -->
<rect x="594" y="314" width="558" height="122" rx="14" fill="#FFFFFF" stroke="#D5E0D0" stroke-width="1.3"/>
<rect x="594" y="314" width="8" height="122" rx="4" fill="#4D6B48"/>
<text x="624" y="351" fill="#3B5936" font-size="14" font-weight="700">TRACK 02 · CLEAN PEAK INT8 · 7.91 MiB · 40.5 FPS</text>
<text x="624" y="385" fill="#182218" font-size="23" font-weight="750">MobileNetV4-Medium Clean INT8 (Gate PASS)</text>
<text x="624" y="416" fill="#4D5C4C" font-size="14.2">mIoU {v3_int8['miou_pct']:.2f}% (교사 90.8%) · Safety {v3_int8['safety_miou_pct']:.2f}% · KITTI 51.32% (+2.23%p)</text>

<!-- Card 03 -->
<rect x="594" y="456" width="558" height="126" rx="14" fill="#FFFFFF" stroke="#2B4C27" stroke-width="2.2"/>
<rect x="594" y="456" width="10" height="126" rx="4" fill="#2B4C27"/>
<rect x="1024" y="473" width="108" height="26" rx="13" fill="#E3EFE0"/>
<text x="1078" y="491" text-anchor="middle" fill="#1F421B" font-size="12.8" font-weight="700">FINAL MODEL</text>
<text x="624" y="492" fill="#2B4C27" font-size="13.8" font-weight="700">TRACK 03 · v4 ROBUST INT8 · 7.91 MiB · 38.4 FPS</text>
<text x="624" y="529" fill="#182218" font-size="23" font-weight="750">MobileNetV4-Medium Robust INT8 + 온도 보정</text>
<text x="624" y="559" fill="#1F421B" font-size="14.2" font-weight="600">노이즈 {v4_noise:.2f}% (교사 {t_noise:.2f}% 추월 · +{noise_gain:.2f}%p) · 안개 66.33% · ECE -74.2%</text>

<!-- Bottom Forest Scorecard -->
<rect x="48" y="606" width="1104" height="114" rx="14" fill="#2B4C27"/>
<text x="80" y="652" fill="#FFFFFF" font-size="26" font-weight="750">Final Locked Eval (Cityscapes 384×768)</text>
<text x="80" y="692" fill="#DDEBD9" font-size="16.2">84.6M SegFormer-B5 → 7.5M INT8 · 7.91 MiB · 4/4 게이트 통과</text>
<line x1="636" y1="624" x2="636" y2="702" stroke="#85A97E" stroke-width="1.2" opacity="0.55"/>

<text x="794" y="654" text-anchor="end" fill="#FFFFFF" font-size="35" font-weight="750">{v3_int8['miou_pct']:.2f}%</text>
<text x="794" y="690" text-anchor="end" fill="#DDEBD9" font-size="14.2">Clean INT8 mIoU</text>

<text x="962" y="654" text-anchor="end" fill="#D5F591" font-size="35" font-weight="750">{v4_noise:.2f}%</text>
<text x="962" y="690" text-anchor="end" fill="#DDEBD9" font-size="14.2">Noise σ=18 (교사 추월)</text>

<text x="1124" y="654" text-anchor="end" fill="#FFFFFF" font-size="35" font-weight="750">{speedup_vs_teacher:.1f}×</text>
<text x="1124" y="690" text-anchor="end" fill="#DDEBD9" font-size="14.2">CPU 가속 (40.5 FPS)</text>
</svg>
"""
        out_path = FIG_DIR / "kd-thumbnail.svg"
    else:
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title desc">
<title id="title">Driving Semantic Segmentation Knowledge Distillation: SegFormer-B5 to MobileNetV4 INT8</title>
<desc id="desc">Cross-architecture knowledge distillation from 84.6M SegFormer-B5 ViT teacher to 7.5M MobileNetV4-Medium ONNX INT8 (7.91 MiB, 40.5 FPS CPU), achieving 67.95% clean mIoU and surpassing the 84.6M teacher under Gaussian sensor noise (59.59% vs 59.19%).</desc>
<style>text{{font-family:'Inter','Segoe UI','DejaVu Sans',sans-serif;font-variant-numeric:tabular-nums}}</style>
<rect width="1200" height="760" fill="#F4F7F2"/>
<text x="48" y="76" fill="#182218" font-size="40" font-weight="750">Driving Semantic Segmentation KD</text>
<text x="48" y="122" fill="#4D5C4C" font-size="21.5">SegFormer-B5 (ViT) → MobileNetV4 (CNN) · Asymmetric Robust KD · Validation-Only ONNX INT8 Gate</text>
<line x1="48" y1="148" x2="1152" y2="148" stroke="#D5E0D0" stroke-width="1.4"/>

<!-- Left Panel -->
<rect x="48" y="172" width="434" height="410" rx="14" fill="#FFFFFF" stroke="#D5E0D0" stroke-width="1.3"/>
<text x="74" y="214" fill="#182218" font-size="23" font-weight="750">Cross-Architecture Asymmetric KD</text>
<text x="74" y="241" fill="#4D5C4C" font-size="15">Cityscapes 384×768 (19 Classes) · KITTI Zero-Shot</text>

<rect x="74" y="264" width="382" height="94" rx="10" fill="#F4F7F2" stroke="#D5E0D0" stroke-width="1.2"/>
<text x="92" y="294" fill="#2B4C27" font-size="13" font-weight="700">TEACHER · CLEAN INPUT ONLY</text>
<text x="92" y="323" fill="#182218" font-size="19" font-weight="750">SegFormer-B5 (84.6M · 323 MiB)</text>
<text x="92" y="346" fill="#4D5C4C" font-size="13.5">mIoU {t['miou_pct']:.2f}% · Safety {t['safety_miou_pct']:.2f}% · CPU {t['cpu_p50_ms']:.0f} ms</text>

<path d="M265 358V404" stroke="#2B4C27" stroke-width="3" stroke-dasharray="5 4"/>
<path d="M257 395L265 405L273 395" fill="none" stroke="#2B4C27" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="108" y="369" width="314" height="24" rx="12" fill="#E3EFE0"/>
<text x="265" y="385" text-anchor="middle" fill="#1F421B" font-size="12.2" font-weight="700">CWD Spatial KD + 4-Stage Cosine + QAT-KD</text>

<rect x="74" y="408" width="382" height="96" rx="10" fill="#F0F8EE" stroke="#527D49" stroke-width="1.6"/>
<text x="92" y="437" fill="#2B4C27" font-size="12.2" font-weight="700">STUDENT · CORRUPTED INPUT (Night/Fog/Blur/Noise)</text>
<text x="92" y="466" fill="#182218" font-size="18" font-weight="750">MobileNetV4-Medium INT8 (7.91 MiB)</text>
<text x="92" y="490" fill="#2B4C27" font-size="12.8" font-weight="600">11.2× Fewer Params · 40.9× Smaller File · 38–40 FPS</text>

<rect x="74" y="520" width="382" height="42" rx="8" fill="#E3EFE0"/>
<text x="265" y="546" text-anchor="middle" fill="#1F421B" font-size="14" font-weight="700">SHA-256 Frozen Locked Eval · Validation-Only Gate</text>

<!-- Connectors -->
<path d="M482 377H534V233H586" fill="none" stroke="#7A8A78" stroke-width="3" stroke-linejoin="round"/>
<path d="M482 377H586" fill="none" stroke="#4D6B48" stroke-width="3"/>
<path d="M482 377H534V520H586" fill="none" stroke="#2B4C27" stroke-width="3.5" stroke-linejoin="round"/>
<circle cx="534" cy="377" r="5" fill="#182218"/>
<path d="M576 226L586 233L576 240" fill="none" stroke="#7A8A78" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M576 370L586 377L576 384" fill="none" stroke="#4D6B48" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M576 513L586 520L576 527" fill="none" stroke="#2B4C27" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>

<!-- Card 01 -->
<rect x="594" y="172" width="558" height="122" rx="14" fill="#FFFFFF" stroke="#D5E0D0" stroke-width="1.3"/>
<rect x="594" y="172" width="8" height="122" rx="4" fill="#7A8A78"/>
<text x="624" y="209" fill="#4D5C4C" font-size="14" font-weight="700">TRACK 01 · 3-SEED ABLATION &amp; GATE AUDIT (1.5M / 3.3M)</text>
<text x="624" y="243" fill="#182218" font-size="22.5" font-weight="750">3/3 Seed Wins · 6/6 Safety Classes Positive</text>
<text x="624" y="274" fill="#4D5C4C" font-size="14.2">CWD +1.42%p (Pixel KL -2.41%p) · V3-Large INT8 collapse (-6.46%p) blocked</text>

<!-- Card 02 -->
<rect x="594" y="314" width="558" height="122" rx="14" fill="#FFFFFF" stroke="#D5E0D0" stroke-width="1.3"/>
<rect x="594" y="314" width="8" height="122" rx="4" fill="#4D6B48"/>
<text x="624" y="351" fill="#3B5936" font-size="14" font-weight="700">TRACK 02 · CLEAN PEAK INT8 · 7.91 MiB · 40.5 FPS</text>
<text x="624" y="385" fill="#182218" font-size="22.5" font-weight="750">MobileNetV4-Medium Clean INT8 (Gate PASS)</text>
<text x="624" y="416" fill="#4D5C4C" font-size="14.2">mIoU {v3_int8['miou_pct']:.2f}% (90.8% of teacher) · Safety {v3_int8['safety_miou_pct']:.2f}% · KITTI 51.32%</text>

<!-- Card 03 -->
<rect x="594" y="456" width="558" height="126" rx="14" fill="#FFFFFF" stroke="#2B4C27" stroke-width="2.2"/>
<rect x="594" y="456" width="10" height="126" rx="4" fill="#2B4C27"/>
<rect x="1024" y="473" width="108" height="26" rx="13" fill="#E3EFE0"/>
<text x="1078" y="491" text-anchor="middle" fill="#1F421B" font-size="12.8" font-weight="700">FINAL MODEL</text>
<text x="624" y="492" fill="#2B4C27" font-size="13.8" font-weight="700">TRACK 03 · v4 ROBUST INT8 · 7.91 MiB · 38.4 FPS</text>
<text x="624" y="529" fill="#182218" font-size="19.8" font-weight="750">MobileNetV4-Medium Robust INT8 + Temp Scaling</text>
<text x="624" y="559" fill="#1F421B" font-size="14.2" font-weight="600">Noise {v4_noise:.2f}% (beats {t_noise:.2f}% teacher · +{noise_gain:.2f}%p) · Fog 66.33% · ECE -74.2%</text>

<!-- Bottom Forest Scorecard -->
<rect x="48" y="606" width="1104" height="114" rx="14" fill="#2B4C27"/>
<text x="80" y="652" fill="#FFFFFF" font-size="26" font-weight="750">Final Locked Eval (Cityscapes 384×768)</text>
<text x="80" y="692" fill="#DDEBD9" font-size="16.2">84.6M SegFormer-B5 → 7.5M INT8 · 7.91 MiB · 4/4 Gate Checks Passed</text>
<line x1="636" y1="624" x2="636" y2="702" stroke="#85A97E" stroke-width="1.2" opacity="0.55"/>

<text x="794" y="654" text-anchor="end" fill="#FFFFFF" font-size="35" font-weight="750">{v3_int8['miou_pct']:.2f}%</text>
<text x="794" y="690" text-anchor="end" fill="#DDEBD9" font-size="14.2">Clean INT8 mIoU</text>

<text x="962" y="654" text-anchor="end" fill="#D5F591" font-size="35" font-weight="750">{v4_noise:.2f}%</text>
<text x="962" y="690" text-anchor="end" fill="#DDEBD9" font-size="14.2">Noise σ=18 (&gt;Teacher)</text>

<text x="1124" y="654" text-anchor="end" fill="#FFFFFF" font-size="35" font-weight="750">{speedup_vs_teacher:.1f}×</text>
<text x="1124" y="690" text-anchor="end" fill="#DDEBD9" font-size="14.2">CPU Speedup (40.5 FPS)</text>
</svg>
"""
        out_path = FIG_DIR / "en" / "kd-thumbnail.svg"

    # Strict XML parse check before writing
    ET.fromstring(svg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    return out_path


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = curate_reports()
    plot_pareto_frontier(summary)
    plot_robustness_breakthrough(summary)
    plot_method_ablation_384(summary)
    plot_3seed_and_per_class(summary)
    plot_ece_calibration(summary)
    optimize_road_overlays()
    ko_svg = render_thumbnail_svg(summary, lang="ko")
    en_svg = render_thumbnail_svg(summary, lang="en")
    print("Generated reports/final-summary.json and all verified figures:")
    for p in sorted(FIG_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(REPO_ROOT).as_posix():45s} {p.stat().st_size / 1024:7.1f} KiB")
    print("SVGs XML-validated:", ko_svg.name, en_svg.name)


if __name__ == "__main__":
    main()
