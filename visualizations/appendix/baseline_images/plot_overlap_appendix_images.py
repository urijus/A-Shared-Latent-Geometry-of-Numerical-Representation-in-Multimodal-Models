"""Appendix-style overlap plots for image-baseline subspaces."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OVERLAP_ROOT = (
    REPO_ROOT
    / "results"
    / "baseline_images"
    / "gemma4_12b_it"
    / "digits"
    / "subspace_overlap"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "visualizations"
    / "appendix"
    / "baseline_images"
    / "overlap"
)

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"

DAS_LABELS = {
    "add_result": "Add\nresult",
    "sub_result": "Sub\nresult",
    "mul_result": "Mul\nresult",
    "mul_c0_hat": r"Mul $\hat{c}_0$",
    "mul_c1_hat": r"Mul $\hat{c}_1$",
}

MODALITY_LABELS = {
    "add": "Add",
    "sub": "Sub",
    "mul": "Mul",
}

PERIODS = [2, 5, 10, 20, 50, 100]
MODALITIES = ["add", "sub", "mul"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap_root", type=Path, default=DEFAULT_OVERLAP_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", default="min_rank_containment")
    parser.add_argument(
        "--das_metric",
        default="symmetric_overlap",
        help="Metric to use for DAS-DAS plots.",
    )
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
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "figure.titlesize": 14,
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
    ax.tick_params(width=0.8, length=0)


def overlap_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "image_overlap_appendix",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )


def labels_from_rows(rows):
    labels = []
    seen = set()
    for row in rows:
        for key in ("first_label", "second_label"):
            label = row[key]
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def matrix_from_rows(rows, labels, metric):
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for row_index, first in enumerate(labels):
        for col_index, second in enumerate(labels):
            matrix[row_index, col_index] = lookup[(first, second)][metric]
    return matrix


def save_matrix(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def clean_stem(path):
    return Path(path).stem.replace("image_", "").replace("_pos-1", "_posminus1")


def plot_das_overlap(path, output_dir, metric):
    rows = load_jsonl(path)
    labels = labels_from_rows(rows)
    matrix = matrix_from_rows(rows, labels, metric)

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(5.45, 4.65))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    cmap = overlap_cmap().copy()
    cmap.set_bad("#d8dee0")
    vmax = max(0.18, min(0.65, float(np.nanpercentile(plot_matrix, 98))))
    image = ax.imshow(plot_matrix, cmap=cmap, vmin=0, vmax=vmax)

    tick_labels = [DAS_LABELS.get(label, label.replace("_", "\n")) for label in labels]
    ax.set_xticks(range(len(labels)), tick_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), tick_labels)
    ax.set_title("Image DAS overlap, layer 44, k=32", pad=12)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            value = matrix[row_index, col_index]
            text = "1" if row_index == col_index else f"{value:.2f}"
            color = "white" if value > vmax * 0.52 and row_index != col_index else "#171717"
            if row_index == col_index:
                color = "#5e6668"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=8.6, color=color)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.035)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(metric.replace("_", " "), fontsize=9.2)

    fig.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.24)
    return save_matrix(fig, output_dir, f"das_overlap_{clean_stem(path)}_{metric}")


def fourier_label_order(labels):
    available = set(labels)
    ordered = [
        f"{modality}_T{period}"
        for modality in MODALITIES
        for period in PERIODS
        if f"{modality}_T{period}" in available
    ]
    extras = [label for label in labels if label not in ordered]
    return ordered + extras


def parse_fourier_label(label):
    match = re.match(r"(?P<modality>[^_]+)_T(?P<period>\d+)", label)
    if not match:
        return label, ""
    modality = MODALITY_LABELS.get(match.group("modality"), match.group("modality"))
    return modality, f"T{match.group('period')}"


def plot_fourier_overlap(path, output_dir, metric):
    rows = load_jsonl(path)
    labels = fourier_label_order(labels_from_rows(rows))
    matrix = matrix_from_rows(rows, labels, metric)

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(7.65, 6.55))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    cmap = overlap_cmap().copy()
    cmap.set_bad("#d8dee0")
    off_diag = plot_matrix[~np.eye(plot_matrix.shape[0], dtype=bool)]
    vmax = max(0.15, min(1.0, float(np.nanpercentile(off_diag, 99))))
    image = ax.imshow(plot_matrix, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    tick_labels = [parse_fourier_label(label)[1] or label for label in labels]
    ax.set_xticks(range(len(labels)), tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), tick_labels)
    ax.set_title("Image Fourier-plane overlap, result, layer 43", pad=12)

    group_size = len(PERIODS)
    for boundary in range(group_size, len(labels), group_size):
        ax.axhline(boundary - 0.5, color=BACKGROUND, linewidth=1.2)
        ax.axvline(boundary - 0.5, color=BACKGROUND, linewidth=1.2)

    for group_index, modality in enumerate(MODALITIES):
        start = group_index * group_size
        end = start + group_size - 1
        if end >= len(labels):
            continue
        center = (start + end) / 2
        ax.text(center, len(labels) + 0.75, MODALITY_LABELS[modality], ha="center", va="top", fontsize=9.4, clip_on=False)
        ax.text(-1.55, center, MODALITY_LABELS[modality], ha="center", va="center", rotation=90, fontsize=9.4, clip_on=False)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            value = matrix[row_index, col_index]
            if row_index == col_index:
                ax.text(col_index, row_index, "1", ha="center", va="center", fontsize=6.6, color="#5e6668")
            elif value >= max(0.30, vmax * 0.48):
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color="white" if value > vmax * 0.58 else TEXT,
                )

    ax.set_xlabel("Second Fourier plane", labelpad=16)
    ax.set_ylabel("First Fourier plane", labelpad=16)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.03)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(metric.replace("_", " "), fontsize=9.2)

    fig.subplots_adjust(left=0.145, right=0.92, top=0.89, bottom=0.17)
    return save_matrix(fig, output_dir, f"fourier_overlap_{clean_stem(path)}_{metric}")


def main():
    args = parse_args()
    outputs = []

    das_paths = sorted((args.overlap_root / "das_to_das").glob("**/*.jsonl"))
    for path in das_paths:
        outputs.extend(plot_das_overlap(path, args.output_dir, args.das_metric))

    fourier_paths = sorted((args.overlap_root / "fourier_to_fourier").glob("**/*.jsonl"))
    for path in fourier_paths:
        outputs.extend(plot_fourier_overlap(path, args.output_dir, args.metric))

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
