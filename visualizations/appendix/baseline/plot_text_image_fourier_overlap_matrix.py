"""Paper-style text/image Fourier overlap matrix plot."""

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
            "axes.labelsize": 10.3,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
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
    ax.tick_params(width=0.75, length=2.8)


def overlap_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "text_image_fourier_matrix",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )


def label(layer, period):
    return f"L{layer} T{period}"


def ordered_pairs():
    return [(layer, period) for layer in LAYERS for period in PERIODS]


def text_image_matrix(rows, metric):
    pairs = ordered_pairs()
    matrix = np.full((len(pairs), len(pairs)), np.nan, dtype=float)
    lookup = {}
    for row in rows:
        if row.get("first_root_label") != "text" or row.get("second_root_label") != "image":
            continue
        key = (
            int(row["first_layer"]),
            int(row["first_period"]),
            int(row["second_layer"]),
            int(row["second_period"]),
        )
        lookup[key] = float(row[metric])

    for row_idx, (text_layer, text_period) in enumerate(pairs):
        for col_idx, (image_layer, image_period) in enumerate(pairs):
            matrix[row_idx, col_idx] = lookup[
                (text_layer, text_period, image_layer, image_period)
            ]
    return matrix, pairs


def draw_group_guides(ax, n_periods):
    for boundary in np.arange(n_periods - 0.5, len(LAYERS) * n_periods, n_periods):
        ax.axhline(boundary, color=BACKGROUND, linewidth=1.8, alpha=0.95)
        ax.axvline(boundary, color=BACKGROUND, linewidth=1.8, alpha=0.95)
    for boundary in np.arange(0.5, len(LAYERS) * n_periods, 1):
        ax.axhline(boundary, color=BACKGROUND, linewidth=0.45, alpha=0.35)
        ax.axvline(boundary, color=BACKGROUND, linewidth=0.45, alpha=0.35)


def plot_matrix(matrix, pairs, args):
    style_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    image = ax.imshow(
        matrix,
        cmap=overlap_cmap(),
        vmin=0.0,
        vmax=args.vmax,
        interpolation="nearest",
    )
    tick_labels = [label(layer, period) for layer, period in pairs]
    ax.set_xticks(range(len(pairs)), tick_labels, rotation=48, ha="right")
    ax.set_yticks(range(len(pairs)), tick_labels)
    ax.set_xlabel("Image Fourier plane")
    ax.set_ylabel("Text Fourier plane")
    ax.set_title("Text-image Fourier overlap: addition result", pad=10)
    draw_group_guides(ax, len(PERIODS))

    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Symmetric Subspace Overlap", fontsize=9.2)
    colorbar.ax.tick_params(colors=TEXT, width=0.7, length=2.6)

    mean = float(np.nanmean(matrix))
    maximum = float(np.nanmax(matrix))
    ax.text(
        0.0,
        -0.24,
        f"Mean: {mean:.3f}; max: {maximum:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#4f5658",
    )

    fig.subplots_adjust(left=0.20, right=0.92, top=0.90, bottom=0.25)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "text_image_addition_result_fourier_overlap_matrix"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    rows = load_jsonl(args.input)
    matrix, pairs = text_image_matrix(rows, args.metric)
    for output in plot_matrix(matrix, pairs, args):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
