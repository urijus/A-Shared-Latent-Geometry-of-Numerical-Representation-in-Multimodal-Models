"""Plot Procrustes causal-transport matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "final_exps" / "procrustes" / "basic"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "procrustes" / "basic"

BACKGROUND = "#fdfdfd"
AX_FACE = "#f7f8f8"
SPINE = "#b9c0c2"
TEXT = "#171717"

TASK_LABELS = {
    "text_add": "Text\nadd",
    "text_sub": "Text\nsub",
    "image_add": "Image\nadd",
    "image_sub": "Image\nsub",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="procrustes_matrices")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_value(row: dict, key: str) -> float:
    value = row.get(key)
    return np.nan if value is None else float(value)


def matrices(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    for row in rows:
        if row["source_label"] not in labels:
            labels.append(row["source_label"])
    lookup = {(row["source_label"], row["destination_label"]): row for row in rows}
    mean = np.array(
        [[row_value(lookup[(src, dst)], "mean") for dst in labels] for src in labels],
        dtype=float,
    )
    std = np.array(
        [[row_value(lookup[(src, dst)], "std") for dst in labels] for src in labels],
        dtype=float,
    )
    counts = np.array([[lookup[(src, dst)]["n"] for dst in labels] for src in labels], dtype=int)
    return labels, mean, std, counts


def display_labels(labels: list[str]) -> list[str]:
    return [TASK_LABELS.get(label, label.replace("_", "\n")) for label in labels]


def style_axis(ax, labels: list[str], title: str) -> None:
    ax.set_facecolor(AX_FACE)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    shown_labels = display_labels(labels)
    ax.set_title(title, pad=9)
    ax.set_xticks(range(len(labels)), labels=shown_labels)
    ax.set_yticks(range(len(labels)), labels=shown_labels)
    ax.tick_params(axis="both", length=0)


def draw_matrix(
    ax,
    labels: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    title: str,
    colorbar_label: str,
    cmap,
    norm,
) -> None:
    masked = np.ma.masked_invalid(mean)
    cmap = cmap.copy()
    cmap.set_bad("#eceff0")
    style_axis(ax, labels, title)
    image = ax.imshow(masked, cmap=cmap, norm=norm)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            value = mean[row_index, col_index]
            if np.isnan(value):
                ax.text(
                    col_index,
                    row_index,
                    "n/a",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="#6b6f72",
                )
                continue
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.48 else "#181818"
            ax.text(
                col_index,
                row_index - 0.09,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=color,
            )
            if not np.isnan(std[row_index, col_index]):
                ax.text(
                    col_index,
                    row_index + 0.16,
                    fr"$\pm${std[row_index, col_index]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color=color,
                )

    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(colorbar_label, fontsize=9)


def finite_abs_max(values: np.ndarray, minimum: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return minimum
    return max(minimum, float(np.nanmax(np.abs(finite))))


def main() -> None:
    args = parse_args()
    style_matplotlib()

    transfer_rows = load_rows(args.input_dir / "destination_normalized_transfer_matrix.jsonl")
    gain_rows = load_rows(args.input_dir / "destination_normalized_transfer_gain_over_causal_matrix.jsonl")
    labels, transfer_mean, transfer_std, transfer_counts = matrices(transfer_rows)
    gain_labels, gain_mean, gain_std, gain_counts = matrices(gain_rows)
    if labels != gain_labels:
        raise ValueError(f"Matrix labels differ: {labels} vs {gain_labels}")
    if not np.array_equal(transfer_counts, gain_counts):
        raise ValueError("Transfer and gain matrices use different cell counts.")

    transfer_cmap = mcolors.LinearSegmentedColormap.from_list(
        "procrustes_transfer",
        ["#d8dee0", "#b7d1ce", "#7db9aa", "#419287", "#226a74"],
    )
    gain_cmap = mcolors.LinearSegmentedColormap.from_list(
        "procrustes_gain",
        ["#8f343f", "#d9b6ad", "#f3f3f0", "#a9cfc2", "#236f78"],
    )
    transfer_norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, finite_abs_max(transfer_mean, 1.0)))
    gain_limit = finite_abs_max(gain_mean, 0.25)
    gain_norm = mcolors.TwoSlopeNorm(vmin=-gain_limit, vcenter=0.0, vmax=gain_limit)

    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.85))
    fig.patch.set_facecolor(BACKGROUND)
    draw_matrix(
        axes[0],
        labels,
        transfer_mean,
        transfer_std,
        "Procrustes transfer",
        "Normalized transfer",
        transfer_cmap,
        transfer_norm,
    )
    draw_matrix(
        axes[1],
        labels,
        gain_mean,
        gain_std,
        "Gain over causal transfer",
        "Delta normalized transfer",
        gain_cmap,
        gain_norm,
    )

    fig.supxlabel("Destination task", x=0.5, y=0.035, fontsize=10.5)
    fig.supylabel("Source alignment", x=0.025, fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.965, top=0.87, bottom=0.18, wspace=0.34)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
