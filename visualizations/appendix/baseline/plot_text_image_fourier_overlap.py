"""Compact text/image Fourier-plane overlap plot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "src"
    / "banks"
    / "baseline_bank"
    / "gemma4_12b_it"
    / "digits"
    / "subspace_overlap"
    / "fourier_text_image"
    / "addition_result_layers_42_43_44_periods_2_5_10_20_50_100"
    / "text_image_addition_result_layers_42_43_44_periods_2_5_10_20_50_100_pos17_ridge.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "visualizations"
    / "appendix"
    / "baseline"
    / "fourier_text_image"
)

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"
LAYERS = [42, 43, 44]
PERIODS = [2, 5, 10, 20, 50, 100]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", default="min_rank_containment")
    parser.add_argument("--vmax", type=float, default=0.10)
    return parser.parse_args()


def long_path(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_jsonl(path):
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib():
    plt.rcParams.update(
        {
            "font.family": [
                "Palatino Linotype",
                "Georgia",
                "Cambria",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.2)


def overlap_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "text_image_fourier_overlap",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f"],
    )


def matching_text_image_matrix(rows, metric):
    matrix = np.full((len(LAYERS), len(PERIODS)), np.nan, dtype=float)
    text_r2 = np.full_like(matrix, np.nan)
    image_r2 = np.full_like(matrix, np.nan)
    for row in rows:
        if row.get("first_root_label") != "text" or row.get("second_root_label") != "image":
            continue
        if int(row["first_layer"]) != int(row["second_layer"]):
            continue
        if int(row["first_period"]) != int(row["second_period"]):
            continue
        layer = int(row["first_layer"])
        period = int(row["first_period"])
        if layer not in LAYERS or period not in PERIODS:
            continue
        row_idx = LAYERS.index(layer)
        col_idx = PERIODS.index(period)
        matrix[row_idx, col_idx] = float(row[metric])
        text_r2[row_idx, col_idx] = float(row["first_probe_r2_mean"])
        image_r2[row_idx, col_idx] = float(row["second_probe_r2_mean"])
    return matrix, text_r2, image_r2


def plot_heatmap(matrix, text_r2, image_r2, args):
    style_matplotlib()
    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    image = ax.imshow(
        matrix,
        cmap=overlap_cmap(),
        vmin=0.0,
        vmax=args.vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Text-image Fourier overlap: addition result", pad=10)
    ax.set_xlabel("Fourier period $T$")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(PERIODS)), [str(period) for period in PERIODS])
    ax.set_yticks(range(len(LAYERS)), [str(layer) for layer in LAYERS])

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                continue
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color="white" if value > args.vmax * 0.55 else TEXT,
            )

    for boundary in np.arange(0.5, len(PERIODS), 1):
        ax.axvline(boundary, color=BACKGROUND, linewidth=0.8, alpha=0.7)
    for boundary in np.arange(0.5, len(LAYERS), 1):
        ax.axhline(boundary, color=BACKGROUND, linewidth=0.8, alpha=0.7)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.025)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Subspace overlap", fontsize=9.4)

    mean_overlap = np.nanmean(matrix)
    mean_text_r2 = np.nanmean(text_r2)
    mean_image_r2 = np.nanmean(image_r2)
    ax.text(
        0.0,
        -0.34,
        (
            f"Mean overlap: {mean_overlap:.3f}  "
            f"(mean probe $R^2$: text {mean_text_r2:.2f}, image {mean_image_r2:.2f})"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color="#4f5658",
    )

    fig.subplots_adjust(left=0.12, right=0.92, top=0.86, bottom=0.27)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "text_image_addition_result_fourier_overlap_compact"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    rows = load_jsonl(args.input)
    matrix, text_r2, image_r2 = matching_text_image_matrix(rows, args.metric)
    for output in plot_heatmap(matrix, text_r2, image_r2, args):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
