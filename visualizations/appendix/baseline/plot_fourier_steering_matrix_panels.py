"""Fourier steering matrix comparison plot from stored steering JSONLs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_das_appendix import BACKGROUND, GRID, REPO_ROOT, SPINE, TEXT, style_matplotlib


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "visualizations" / "appendix" / "baseline" / "fourier_steering_matrix"
)

DEFAULT_PANELS = [
    (
        "Llama L18, pos 8",
        REPO_ROOT
        / "results"
        / "baseline"
        / "llama3_1_8b"
        / "digits"
        / "fourier_probes"
        / "addition"
        / "fourier_steering_matrix"
        / "result_layer18_pos8_ridge"
        / "periods_2-5-10-20-50-100_alpha10.jsonl",
    ),
    (
        "Gemma L36, pos 11",
        REPO_ROOT
        / "results"
        / "baseline"
        / "gemma4_12b_it"
        / "digits"
        / "fourier_probes"
        / "addition"
        / "fourier_steering_matrix"
        / "result_layer36_pos11_ridge"
        / "periods_2-5-10-20-50-100_alpha10.jsonl",
    ),
    (
        "Gemma L43, pos 11",
        REPO_ROOT
        / "results"
        / "baseline"
        / "gemma4_12b_it"
        / "digits"
        / "fourier_probes"
        / "addition"
        / "fourier_steering_matrix"
        / "result_layer43_pos11_ridge"
        / "periods_2-5-10-20-50-100_alpha10.jsonl",
    ),
    (
        "Gemma L43, pos 17",
        REPO_ROOT
        / "results"
        / "baseline"
        / "gemma4_12b_it"
        / "digits"
        / "fourier_probes"
        / "addition"
        / "fourier_steering_matrix"
        / "result_layer43_pos17_ridge"
        / "periods_2-5-10-20-50-100_alpha10.jsonl",
    ),
]


def long_path(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_jsonl(path: Path) -> list[dict]:
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_matrix(path: Path):
    rows = load_jsonl(path)
    matrix_rows = [row for row in rows if row.get("row_type") == "matrix"]
    if not matrix_rows:
        raise ValueError(f"No matrix rows found in {path}")
    summary_rows = [row for row in rows if row.get("row_type") == "summary"]
    summary = summary_rows[0] if summary_rows else {}

    targets = sorted(int(row["steering_target"]) for row in matrix_rows)
    output_values = summary.get("output_values")
    if output_values is None:
        output_values = sorted(
            {
                int(candidate)
                for row in matrix_rows
                for candidate in row.get("avg_steered_probs", {})
            }
        )
    output_values = [int(value) for value in output_values]

    by_target = {int(row["steering_target"]): row for row in matrix_rows}
    matrix = np.zeros((len(targets), len(output_values)), dtype=float)
    for row_index, target in enumerate(targets):
        probs = by_target[target].get("avg_steered_probs", {})
        for col_index, value in enumerate(output_values):
            matrix[row_index, col_index] = float(probs.get(str(value), probs.get(value, 0.0)))
    return matrix, targets, output_values, summary


def setup_heatmap_axis(ax):
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.75)
    ax.tick_params(width=0.75, length=2.8)
    ax.grid(False)


def draw_matrix(ax, matrix, targets, output_values, title, summary, cmap, vmax):
    setup_heatmap_axis(ax)
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(title, pad=7, fontweight="normal")
    tick_step = 5
    xticks = [index for index, value in enumerate(output_values) if value % tick_step == 0]
    yticks = [index for index, value in enumerate(targets) if value % tick_step == 0]
    ax.set_xticks(xticks, labels=[str(output_values[index]) for index in xticks])
    ax.set_yticks(yticks, labels=[str(targets[index]) for index in yticks])
    accuracy = summary.get("matrix_top1_diagonal_accuracy")
    if accuracy is not None:
        ax.text(
            0.04,
            0.96,
            f"diag. top-1: {float(accuracy):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color=TEXT,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": BACKGROUND,
                "edgecolor": SPINE,
                "linewidth": 0.55,
                "alpha": 0.88,
            },
        )
    return image


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="addition_fourier_steering_matrix_three_panel")
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.45,
        help="Shared colorbar maximum. Lower values make weak off-diagonal structure visible.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    style_matplotlib()
    plt.rcParams.update(
        {
            "axes.titlesize": 10.8,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
        }
    )

    loaded = []
    for title, path in DEFAULT_PANELS:
        loaded.append((title, *load_matrix(path)))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper_fourier_steering_matrix",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )
    fig, axes = plt.subplots(1, len(loaded), figsize=(12.1, 3.35), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)

    image = None
    for ax, (title, matrix, targets, output_values, summary) in zip(axes, loaded):
        image = draw_matrix(
            ax,
            matrix,
            targets,
            output_values,
            title,
            summary,
            cmap,
            args.vmax,
        )
    axes[0].set_ylabel("Steering target")
    fig.supxlabel("Output candidate", y=0.055, color=TEXT)
    fig.suptitle("Fourier steering for addition", y=0.965, color=TEXT)
    fig.subplots_adjust(left=0.065, right=0.895, top=0.80, bottom=0.22, wspace=0.14)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.02)
        cbar.set_label("Candidate probability", color=TEXT)
        cbar.outline.set_edgecolor(SPINE)
        cbar.ax.tick_params(colors=TEXT, width=0.7, length=2.6)

    for output in save_figure(fig, args.output_dir, args.stem):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
