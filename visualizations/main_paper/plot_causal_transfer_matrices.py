"""Plot causal-transfer matrices with mean and seed-pair standard deviation."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "final_exps" / "causal_tranfer"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "causal_transfer"

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
    parser.add_argument("--stem", default="causal_transfer_matrices")
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


def matrices(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels = []
    for row in rows:
        if row["source_label"] not in labels:
            labels.append(row["source_label"])
    lookup = {(row["source_label"], row["destination_label"]): row for row in rows}
    mean = np.array([[lookup[(src, dst)]["mean"] for dst in labels] for src in labels], dtype=float)
    std = np.array([[lookup[(src, dst)]["std"] for dst in labels] for src in labels], dtype=float)
    counts = np.array([[lookup[(src, dst)]["n"] for dst in labels] for src in labels], dtype=int)
    return labels, mean, std, counts


def display_labels(labels: list[str]) -> list[str]:
    return [TASK_LABELS.get(label, label.replace("_", "\n")) for label in labels]


def draw_matrix(ax, labels: list[str], mean: np.ndarray, std: np.ndarray, title: str, colorbar_label: str):
    display_mean = np.clip(mean, 0.0, 1.0)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "causal_transfer",
        ["#d8dee0", "#b7d1ce", "#7db9aa", "#419287", "#226a74"],
    )
    ax.set_facecolor(AX_FACE)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)

    image = ax.imshow(display_mean, vmin=0, vmax=1.0, cmap=cmap)
    shown_labels = display_labels(labels)
    ax.set_title(title, pad=9)
    ax.set_xticks(range(len(labels)), labels=shown_labels)
    ax.set_yticks(range(len(labels)), labels=shown_labels)
    ax.tick_params(axis="both", length=0)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            value = mean[row_index, col_index]
            display_value = display_mean[row_index, col_index]
            color = "white" if display_value > 0.52 else "#181818"
            ax.text(
                col_index,
                row_index - 0.09,
                f"{display_value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=color,
            )
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


def main() -> None:
    args = parse_args()
    style_matplotlib()

    raw_rows = load_rows(args.input_dir / "raw_autoregressive_iia_matrix.jsonl")
    normalized_rows = load_rows(args.input_dir / "destination_normalized_transfer_matrix.jsonl")
    labels, raw_mean, raw_std, raw_counts = matrices(raw_rows)
    normalized_labels, normalized_mean, normalized_std, normalized_counts = matrices(normalized_rows)
    if labels != normalized_labels:
        raise ValueError(f"Raw and normalized matrices use different labels: {labels} vs {normalized_labels}")
    if not np.array_equal(raw_counts, normalized_counts):
        raise ValueError("Raw and normalized matrices use different cell counts.")

    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.85))
    fig.patch.set_facecolor(BACKGROUND)
    draw_matrix(axes[0], labels, raw_mean, raw_std, "Autoregressive IIA", "AR IIA")
    draw_matrix(
        axes[1],
        labels,
        normalized_mean,
        normalized_std,
        "Destination-normalized transfer",
        "Normalized transfer",
    )

    fig.supxlabel("Destination condition", x=0.5, y=0.035, fontsize=10.5)
    fig.supylabel("Source DAS subspace", x=0.025, fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.965, top=0.87, bottom=0.18, wspace=0.34)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    left_fig, left_ax = plt.subplots(1, 1, figsize=(4.65, 3.85))
    left_fig.patch.set_facecolor(BACKGROUND)
    draw_matrix(left_ax, labels, raw_mean, raw_std, "Autoregressive IIA", "AR IIA")
    left_fig.supxlabel("Destination condition", x=0.52, y=0.035, fontsize=10.5)
    left_fig.supylabel("Source DAS subspace", x=0.035, fontsize=10.5)
    left_fig.subplots_adjust(left=0.16, right=0.92, top=0.87, bottom=0.18)

    left_png_path = args.output_dir / f"{args.stem}_left.png"
    left_pdf_path = args.output_dir / f"{args.stem}_left.pdf"
    left_fig.savefig(left_png_path, dpi=350, facecolor=left_fig.get_facecolor())
    left_fig.savefig(left_pdf_path, facecolor=left_fig.get_facecolor())
    plt.close(left_fig)
    print(f"Saved {left_png_path}")
    print(f"Saved {left_pdf_path}")


if __name__ == "__main__":
    main()
