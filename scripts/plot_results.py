"""Plot measured release evidence without importing the frozen project package.

Example:
  python plot_release.py --summary release-summary.json --output release-assets

The summary contains actual evaluate_keras metrics under
evaluations.locked_holdout and actual benchmark_models output under benchmark.
An explicitly marked synthetic fixture requires --allow-synthetic-fixture and
is watermarked. Historical notebook scores are never used by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "teacher": "#66758a", "baseline": "#3275a5", "distilled": "#148b79",
    "fp32": "#8a62b0", "int8": "#d29526",
}
SEED_MARKERS = {378: "o", 379: "D", 380: "^"}


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def finite(value: Any, label: str, *, lower: float | None = None, upper: float | None = None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: numeric value required")
    result = float(value)
    require(math.isfinite(result), f"{label}: finite value required")
    require(lower is None or result >= lower, f"{label}: value below {lower}")
    require(upper is None or result <= upper, f"{label}: value above {upper}")
    return result


def check_metrics(metric: dict, name: str) -> dict:
    require(isinstance(metric, dict), f"evaluations.locked_holdout.{name}: metrics object required")
    for key in ("count", "correct", "accuracy", "macro_f1", "per_class", "model_parameters"):
        require(key in metric, f"evaluations.locked_holdout.{name}.{key}: missing measured field")
    count = finite(metric["count"], f"{name}.count", lower=1)
    correct = finite(metric["correct"], f"{name}.correct", lower=0, upper=count)
    require(count.is_integer() and correct.is_integer(), f"{name}: count and correct must be integers")
    accuracy = finite(metric["accuracy"], f"{name}.accuracy", lower=0, upper=1)
    finite(metric["macro_f1"], f"{name}.macro_f1", lower=0, upper=1)
    parameters = finite(metric["model_parameters"], f"{name}.model_parameters", lower=1)
    require(parameters.is_integer(), f"{name}: parameter count must be integer")
    require(abs(accuracy - correct / count) < 1e-9, f"{name}: accuracy disagrees with correct/count")
    require(metric.get("split") == "locked_holdout", f"{name}: only locked_holdout quality belongs on this figure")
    classes = metric["per_class"]
    require(isinstance(classes, list) and len(classes) == 43, f"{name}: all 43 per-class records required")
    require(all(isinstance(item, dict) for item in classes), f"{name}: per-class records must be objects")
    require({item.get("class_id") for item in classes} == set(range(43)), f"{name}: class IDs must be exactly 0..42")
    support_sum = 0
    recalled = 0.0
    for item in classes:
        support = finite(item.get("support"), f"{name}.class_{item['class_id']}.support", lower=0)
        recall = finite(item.get("recall"), f"{name}.class_{item['class_id']}.recall", lower=0, upper=1)
        require(support.is_integer(), f"{name}: class support must be integer")
        support_sum += int(support)
        recalled += support * recall
    require(support_sum == int(count), f"{name}: class supports do not sum to the measured denominator")
    require(abs(recalled - correct) < 1e-5, f"{name}: class recalls disagree with correct predictions")
    return metric


def artifact_category(name: str, result: dict) -> str:
    key = name.lower().replace("-", "_")
    filename = str(result.get("model_filename", "")).lower()
    if "int8" in key or "int8" in filename:
        return "int8"
    if "fp32" in key or "fp32" in filename:
        return "fp32"
    if "teacher" in key:
        return "teacher"
    if "baseline" in key:
        return "baseline"
    return "distilled"


def artifact_label(name: str, result: dict) -> str:
    filename = str(result.get("model_filename", ""))
    category = artifact_category(name, result)
    if category in {"fp32", "int8"}:
        return f"Student / {category.upper()} TFLite"
    kind = "Teacher" if category == "teacher" else "Baseline" if category == "baseline" else "Distilled" if name.lower() == "distilled" else name.replace("_", " ").title()
    suffix = "Keras" if filename.lower().endswith(".keras") else "TFLite" if filename.lower().endswith(".tflite") else "artifact"
    return f"{kind} / {suffix}"


def validate_summary(summary: dict) -> tuple[dict, list[tuple[str, dict]], dict, list[int]]:
    require(isinstance(summary, dict), "Release summary must be a JSON object")
    evaluations = summary.get("evaluations")
    require(isinstance(evaluations, dict), "Missing evaluations object")
    held = evaluations.get("locked_holdout")
    require(isinstance(held, dict), "Missing evaluations.locked_holdout measured metrics")
    for name in ("teacher", "baseline"):
        require(name in held, f"Missing locked_holdout.{name}; evaluate that frozen model first")
        check_metrics(held[name], name)
    require(held["teacher"]["count"] == held["baseline"]["count"], "Teacher and baseline use different denominators")
    deployment = summary.get("deployment")
    require(isinstance(deployment, dict), "Missing deployment decision object")
    require(deployment.get("kind") in {"baseline", "distilled"}, "deployment.kind must be baseline or distilled")
    require(deployment.get("format") in {"FP32", "INT8"}, "deployment.format must be FP32 or INT8")
    require(deployment["kind"] in held, f"Missing locked_holdout.{deployment['kind']} selected float-model metrics")
    if "distilled" in held:
        check_metrics(held["distilled"], "distilled")
    plotted = ["teacher", "baseline"] + (["distilled"] if "distilled" in held else [])
    seeds = [378]
    for seed in (379, 380):
        names = [f"baseline_seed_{seed}"] + ([f"distilled_seed_{seed}"] if "distilled" in held else [])
        present = [name in held for name in names]
        require(not any(present) or all(present), f"Incomplete seed {seed} pair; supply {', '.join(names)}")
        if all(present):
            seeds.append(seed)
            for name in names:
                check_metrics(held[name], name)
                plotted.append(name)
    require(seeds in ([378], [378, 379, 380]), "Partial confirmation: provide both seeds 379 and 380 or only the declared seed 378")
    reference = held["baseline"]
    for name in plotted:
        metric = held[name]
        require(metric["count"] == reference["count"], f"{name}: denominator differs from baseline")
        if name != "teacher":
            require(metric["model_parameters"] == reference["model_parameters"], f"{name}: compact architecture parameter count differs from baseline")
        for key in ("evaluation_records_sha256", "training_manifest_sha256"):
            require(metric.get(key) and metric.get(key) == reference.get(key), f"{name}: missing or different {key}; comparisons require identical inputs")
        support_by_id = {item["class_id"]: item["support"] for item in metric["per_class"]}
        require(support_by_id == {item["class_id"]: item["support"] for item in reference["per_class"]}, f"{name}: class supports differ")
    benchmark = summary.get("benchmark")
    require(isinstance(benchmark, dict) and isinstance(benchmark.get("models"), dict), "Missing actual benchmark.models result")
    require(benchmark.get("device") == "CPU" and benchmark.get("batch_size") == 1, "CPU batch-one benchmark required")
    require(benchmark.get("split") == "validation", "Deployment benchmark must use validation samples")
    require(all(isinstance(result, dict) for result in benchmark["models"].values()), "benchmark.models must contain measured result objects")
    rows = sorted(benchmark["models"].items(), key=lambda item: ({"teacher": 0, "baseline": 1, "distilled": 2, "fp32": 3, "int8": 4}[artifact_category(*item)], item[0]))
    require(rows, "No measured benchmark models")
    require(len(rows) <= 8, "At most eight artifact rows fit this summary; pass the deployment benchmark rather than all seed benchmarks")
    first = rows[0][1]
    for name, result in rows:
        require(result.get("device") == "CPU", f"benchmark.{name}: CPU measurements required")
        for key in ("threads", "sample_count", "sample_ids_sha256", "repetitions", "warmup", "manifest_sha256"):
            require(result.get(key) is not None and result.get(key) == first.get(key), f"benchmark.{name}: missing or unmatched {key}")
        model_bytes = finite(result.get("model_bytes"), f"benchmark.{name}.model_bytes", lower=1)
        require(model_bytes.is_integer(), f"benchmark.{name}: model bytes must be integer")
        pure = result.get("pure_model", {})
        require(isinstance(pure, dict), f"benchmark.{name}.pure_model: timing object required")
        p50 = finite(pure.get("p50_ms"), f"benchmark.{name}.pure_model.p50_ms", lower=1e-9)
        p95 = finite(pure.get("p95_ms"), f"benchmark.{name}.pure_model.p95_ms", lower=p50)
        require(p95 >= p50, f"benchmark.{name}: p95 below p50")
        for core in ("teacher", "baseline", "distilled"):
            if name == core and core in held:
                require(result.get("model_sha256") == held[core].get("model_sha256"), f"benchmark.{name}: quality and latency refer to different artifacts")
    acceptance = deployment.get("int8_acceptance")
    if any(artifact_category(*row) == "int8" for row in rows):
        require(isinstance(acceptance, dict) and isinstance(acceptance.get("accepted"), bool), "INT8 timing is present but deployment.int8_acceptance is missing its actual accepted decision")
    if deployment["format"] == "INT8":
        require(isinstance(acceptance, dict) and acceptance.get("accepted") is True, "INT8 deployment requires an accepted measured validation/CPU gate")
    return held, rows, deployment, seeds


def byte_label(size: float) -> str:
    return f"{size / 2**20:.2f} MiB" if size >= 2**20 else f"{size / 2**10:.1f} KiB"


def clean_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#aeb8c2")
    axis.tick_params(colors="#3b4653", labelsize=9)
    axis.set_axisbelow(True)


def make_plot(summary: dict, output: Path, prefix: str, *, allow_synthetic_fixture: bool = False) -> dict:
    held, rows, deployment, seeds = validate_summary(summary)
    fixture = summary.get("_synthetic_fixture") is True
    require(not fixture or allow_synthetic_fixture, "Synthetic fixture requires --allow-synthetic-fixture and must remain in private output")
    require(prefix and all(c.isalnum() or c in "-_" for c in prefix), "Output prefix must contain only letters, numbers, hyphens or underscores")
    if fixture:
        prefix = "SYNTHETIC-PRIVATE-" + prefix
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelsize": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })
    figure = plt.figure(figsize=(17.2, 10.2))
    grid = figure.add_gridspec(2, 3, left=0.065, right=0.98, top=0.84, bottom=0.155, hspace=0.43, wspace=0.65, height_ratios=[1.0, 0.85])
    accuracy_axis = figure.add_subplot(grid[0, 0])
    latency_axis = figure.add_subplot(grid[0, 1])
    size_axis = figure.add_subplot(grid[0, 2])
    recall_axis = figure.add_subplot(grid[1, :])
    names = ["teacher", "baseline"] + (["distilled"] if "distilled" in held else [])
    for index, name in enumerate(names):
        metric = held[name]
        accuracy_axis.bar(index, 100 * metric["accuracy"], width=0.61, color=PALETTE[name], alpha=0.88, zorder=2)
        accuracy_axis.text(index, 102, f"{100 * metric['accuracy']:.2f}%\nF1 {100 * metric['macro_f1']:.2f}%", ha="center", va="bottom", fontsize=9.5)
        if name != "teacher":
            for offset, seed in zip(np.linspace(-0.14, 0.14, len(seeds)), seeds):
                key = name if seed == 378 else f"{name}_seed_{seed}"
                accuracy_axis.scatter(index + offset, 100 * held[key]["accuracy"], marker=SEED_MARKERS[seed], s=39, facecolor="white", edgecolor="#263443", linewidth=1.0, zorder=4)
    if "distilled" in held:
        baseline_index, distilled_index = names.index("baseline"), names.index("distilled")
        for offset, seed in zip(np.linspace(-0.14, 0.14, len(seeds)), seeds):
            baseline_key = "baseline" if seed == 378 else f"baseline_seed_{seed}"
            distilled_key = "distilled" if seed == 378 else f"distilled_seed_{seed}"
            accuracy_axis.plot([baseline_index + offset, distilled_index + offset], [100 * held[baseline_key]["accuracy"], 100 * held[distilled_key]["accuracy"]], color="#374151", alpha=0.30, linewidth=0.8, zorder=3)
    accuracy_axis.set_xticks(range(len(names)), [f"{name.title()}\n{held[name]['correct']:,}/{held[name]['count']:,}" for name in names])
    accuracy_axis.set_ylim(0, 116)
    accuracy_axis.set_yticks([0, 20, 40, 60, 80, 100])
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_title("Locked-holdout classification", loc="left", pad=20)
    accuracy_axis.grid(axis="y", color="#e6ebef", linewidth=0.8)
    handles = [plt.Line2D([0], [0], marker=SEED_MARKERS[seed], linestyle="none", markerfacecolor="white", markeredgecolor="#263443", label=str(seed)) for seed in seeds]
    accuracy_axis.legend(handles=handles, title="Training seed", loc="lower left", fontsize=8, title_fontsize=8, frameon=True, facecolor="white", edgecolor="#e6ebef", ncols=len(seeds))
    labels = [artifact_label(name, result) for name, result in rows]
    y = np.arange(len(rows))
    p50s = [result["pure_model"]["p50_ms"] for _, result in rows]
    p95s = [result["pure_model"]["p95_ms"] for _, result in rows]
    sizes = [result["model_bytes"] for _, result in rows]
    for index, (name, result) in enumerate(rows):
        color = PALETTE[artifact_category(name, result)]
        latency_axis.hlines(index, p50s[index], p95s[index], color=color, linewidth=3)
        latency_axis.scatter(p50s[index], index, color=color, marker="o", s=40, zorder=3)
        latency_axis.scatter(p95s[index], index, facecolor="white", edgecolor=color, marker="s", s=30, zorder=3)
        latency_axis.annotate(f"{p50s[index]:.2f} / {p95s[index]:.2f}", (p95s[index], index), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)
        size_axis.scatter(sizes[index] / 2**20, index, color=color, s=48, zorder=3)
        size_axis.annotate(byte_label(sizes[index]), (sizes[index] / 2**20, index), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
    latency_axis.set_xscale("log")
    latency_axis.set_xlim(min(p50s) / 1.4, max(p95s) * 3.8)
    latency_axis.set_yticks(y, labels)
    latency_axis.invert_yaxis()
    latency_axis.set_ylim(len(rows) - 0.5, -0.5)
    latency_axis.set_xlabel("p50 / p95 milliseconds (log scale)")
    latency_axis.set_title("Warmed CPU inference", loc="left", pad=20)
    latency_axis.grid(axis="x", which="major", color="#e6ebef", linewidth=0.8)
    size_axis.set_xscale("log")
    size_axis.set_xlim(min(sizes) / 2**20 / 1.6, max(sizes) / 2**20 * 4)
    size_axis.set_yticks(y, labels)
    size_axis.set_ylim(len(rows) - 0.5, -0.5)
    size_axis.set_xlabel("Inference artifact MiB (log scale)")
    size_axis.set_title("Actual model file sizes", loc="left", pad=20)
    size_axis.grid(axis="x", which="major", color="#e6ebef", linewidth=0.8)
    class_ids = np.arange(43)
    baseline_recall = [100 * item["recall"] for item in sorted(held["baseline"]["per_class"], key=lambda item: item["class_id"])]
    recall_axis.plot(class_ids, baseline_recall, color=PALETTE["baseline"], marker="o", markersize=4, linewidth=1.4, label="Baseline, seed 378")
    selected_kind = deployment["kind"]
    if selected_kind == "distilled":
        selected_recall = [100 * item["recall"] for item in sorted(held["distilled"]["per_class"], key=lambda item: item["class_id"])]
        recall_axis.plot(class_ids, selected_recall, color=PALETTE["distilled"], marker="s", markersize=3.5, linewidth=1.4, label="Selected distilled float checkpoint, seed 378")
    recall_axis.set_title("Per-class recall on the locked holdout (float checkpoints)", loc="left", pad=13)
    recall_axis.set_xticks(class_ids)
    recall_axis.set_xlim(-0.7, 42.7)
    recall_axis.set_ylim(0, 106)
    recall_axis.set_yticks([0, 20, 40, 60, 80, 100])
    recall_axis.set_ylabel("Recall (%)")
    recall_axis.set_xlabel("GTSRB class ID (all 43 classes)")
    recall_axis.grid(axis="y", color="#e6ebef", linewidth=0.8)
    recall_axis.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", edgecolor="#e6ebef")
    for axis in (accuracy_axis, latency_axis, size_axis, recall_axis):
        clean_axis(axis)
    benchmark_reference = rows[0][1]
    host = benchmark_reference.get("host", {})
    host_name = " ".join(str(host.get(key, "")) for key in ("system", "machine")).strip() or "Recorded host"
    title = "Traffic sign recognition: measured release evidence"
    if fixture:
        title = "PRIVATE SYNTHETIC FIXTURE - NOT PROJECT RESULTS"
    figure.suptitle(title, x=0.065, y=0.971, ha="left", fontsize=20, fontweight="bold", color="#203046" if not fixture else "#a32929")
    subtitle = f"{held['baseline']['count']:,} locked-holdout images  |  {held['baseline']['model_parameters']:,}-parameter baseline  |  {host_name}, {benchmark_reference['threads']} CPU thread(s)"
    figure.text(0.065, 0.926, subtitle, ha="left", va="top", fontsize=11, color="#566474")
    pairing = "; lines pair baseline and KD" if "distilled" in held else "; no distilled model in this release"
    note1 = f"Quality bars: seed 378; F1 labels: macro F1. Points: {', '.join(str(seed) for seed in seeds)}{pairing}. No confidence intervals."
    note2 = f"CPU: batch 1, {benchmark_reference['sample_count']} identical validation images, {benchmark_reference['warmup']} warmups, {benchmark_reference['repetitions']} repetitions; synchronized inference, excluding load and image decoding."
    acceptance = deployment.get("int8_acceptance")
    gate_status = "accepted" if acceptance and acceptance.get("accepted") is True else "rejected" if acceptance and acceptance.get("accepted") is False else "not measured"
    note3 = f"Declared deployment: {selected_kind} / {deployment['format']}. INT8 validation/CPU gate: {gate_status}. Holdout excluded from current selection; the public data were historically exposed."
    for ypos, note in zip((0.101, 0.075, 0.049), (note1, note2, note3)):
        figure.text(0.065, ypos, note, fontsize=9.0, ha="left", color="#566474")
    png = output / f"{prefix}.png"
    svg = output / f"{prefix}.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight", pad_inches=0.2)
    figure.savefig(svg, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)
    return {"png": str(png.resolve()), "svg": str(svg.resolve()), "locked_holdout_count": held["baseline"]["count"], "plotted_training_seeds": seeds, "synthetic_fixture": fixture, "historical_results_on_axes": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="knowledge-distillation-release")
    parser.add_argument("--allow-synthetic-fixture", action="store_true")
    args = parser.parse_args()
    try:
        content = args.summary.read_bytes()
        summary = json.loads(content)
        result = make_plot(summary, args.output, args.prefix, allow_synthetic_fixture=args.allow_synthetic_fixture)
        result["summary_sha256"] = hashlib.sha256(content).hexdigest()
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"success": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
