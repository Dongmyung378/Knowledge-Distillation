"""Recolor the two historical figures using only saved notebook text outputs.

No notebook cell executes. Historical rounded scores, axis labels, ticks,
legends, temperature/alpha order and best-cell marker are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np


def saved_values(path):
    content = path.read_bytes()
    notebook = json.loads(content)
    text = "\n".join("".join(output.get("text", [])) for cell in notebook["cells"] for output in cell.get("outputs", []))
    rows = re.findall(r"^T=(3|5|10)\s+(\d+\.\d+)%\s+(\d+\.\d+)%\s*\*?\s+(\d+\.\d+)%", text, re.MULTILINE)
    if len(rows) != 3 or [row[0] for row in rows] != ["3", "5", "10"]:
        raise ValueError("Expected the original recorded three-by-three score table")
    grid = np.array([[float(value) for value in row[1:]] for row in rows])
    teacher = re.search(r"^Teacher \(MobileNetV2\)\s+2,313,067\s+(\d+\.\d+)%", text, re.MULTILINE)
    baseline = re.search(r"^Student Baseline\s+34,406\s+(\d+\.\d+)%", text, re.MULTILINE)
    if not teacher or not baseline:
        raise ValueError("The original teacher/baseline summary is absent")
    return grid, float(teacher.group(1)) / 100, float(baseline.group(1)) / 100, hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grid, teacher, baseline, notebook_sha = saved_values(args.notebook)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.labelcolor": "#403846", "text.color": "#403846",
                         "axes.titlecolor": "#694483", "figure.facecolor": "#FBFAFC",
                         "axes.facecolor": "white", "savefig.facecolor": "#FBFAFC"})
    colors_t = ["#B1BB86", "#88994A", "#657A35"]
    colors_a = ["#657A35", "#88994A", "#B1BB86"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Knowledge Distillation — Accuracy Comparison", fontsize=15, fontweight="bold", color="#694483")
    for axis, values, colors, labels, title, xlabel in (
            (axes[0], grid[:, 0] / 100, colors_t, ["T=3", "T=5", "T=10"], "Experiment A: Temperature Sweep\n(alpha=0.1 fixed)", "Temperature (T)"),
            (axes[1], grid[2] / 100, colors_a, ["α=0.1", "α=0.3", "α=0.5"], "Experiment B: Alpha Sweep\n(T=10 fixed)", "Alpha (α)")):
        bars = axis.bar(labels, values, color=colors, width=.5, zorder=3)
        axis.axhline(baseline, color="#939099", ls="--", lw=2, label=f"Student Baseline ({baseline:.2%})")
        axis.axhline(teacher, color="#694483", ls=":", lw=2, label=f"Teacher ({teacher:.2%})")
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + .005, f"{value:.2%}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        axis.set_title(title, fontsize=13)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Best Validation Accuracy")
        axis.set_ylim(baseline - .05, min(teacher + .08, 1.0))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.grid(axis="y", color="#EAE5EE", zorder=0)
        axis.legend(fontsize=10, edgecolor="#DDD4E4")
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#B6ADBF")
    fig.tight_layout()
    fig.savefig(args.output / "accuracy-comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5))
    palette = LinearSegmentedColormap.from_list("KD_olive", ["#F3F5E9", "#B1BB86", "#657A35"])
    image = axis.imshow(grid, cmap=palette, aspect="auto", vmin=grid.min() - 1, vmax=grid.max() + 1)
    fig.colorbar(image, ax=axis, label="Val Accuracy (%)")
    axis.set_xticks(range(3), ["α=0.1", "α=0.3", "α=0.5"])
    axis.set_yticks(range(3), ["T=3", "T=5", "T=10"])
    axis.set_xlabel("Alpha (α)")
    axis.set_ylabel("Temperature (T)")
    axis.set_title("KD Grid Search — Validation Accuracy (%)", fontweight="bold")
    for row in range(3):
        for column in range(3):
            suffix = " ★" if grid[row, column] == grid.max() else ""
            position = (grid[row, column] - (grid.min() - 1)) / (grid.max() - grid.min() + 2)
            axis.text(column, row, f"{grid[row, column]:.2f}%{suffix}", ha="center", va="center", fontsize=11, fontweight="bold", color="white" if position > .63 else "#403846")
    fig.tight_layout()
    fig.savefig(args.output / "hyperparameter-grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"display_only": True, "source_notebook_sha256": notebook_sha,
                      "historical_recorded_grid_percent": grid.tolist(), "teacher_percent": 100 * teacher,
                      "baseline_percent": 100 * baseline, "notebook_cells_executed": False}))


if __name__ == "__main__":
    main()
