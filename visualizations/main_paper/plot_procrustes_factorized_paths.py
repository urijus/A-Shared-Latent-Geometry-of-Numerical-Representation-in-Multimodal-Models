"""Plot direct and composed Procrustes factorized-path results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "final_exps" / "procrustes" / "factorized_paths"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "procrustes" / "factorized_paths"

BACKGROUND = "#fdfdfd"
TEXT = "#171717"
GRID = "#d8ddde"
COLORS = {
    "direct": "#226a74",
    "operation_first": "#a95642",
    "modality_first": "#5d6f9f",
}
TRANSPORT_LABELS = {
    "direct": "Direct",
    "operation_first": "Operation first",
    "modality_first": "Modality first",
}
TASK_LABELS = {
    "text_add": "T+",
    "text_sub": "T-",
    "image_add": "I+",
    "image_sub": "I-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="procrustes_factorized_paths")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
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


def short_task(label: str) -> str:
    return TASK_LABELS.get(label, label)


def direction_label(row: dict) -> str:
    return f"{short_task(row['source_label'])} -> {short_task(row['destination_label'])}"


def as_float(value) -> float:
    return np.nan if value is None else float(value)


def ordered(rows: list[dict]) -> tuple[list[str], list[str], dict[tuple[str, str], dict]]:
    directions = []
    transports = ["direct", "operation_first", "modality_first"]
    lookup = {}
    for row in rows:
        direction = direction_label(row)
        if direction not in directions:
            directions.append(direction)
        lookup[(direction, row["transport"])] = row
    return directions, transports, lookup


def draw_grouped_bars(ax, directions: list[str], transports: list[str], lookup: dict, metric: str, title: str) -> None:
    x = np.arange(len(directions))
    width = 0.24
    offsets = np.linspace(-width, width, len(transports))
    ax.set_facecolor("#f7f8f8")
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    ax.axhline(0.0, color="#9aa1a3", linewidth=0.8)

    for offset, transport in zip(offsets, transports):
        values = [as_float(lookup.get((direction, transport), {}).get(metric)) for direction in directions]
        label = TRANSPORT_LABELS[transport]
        ax.bar(x + offset, values, width=width, color=COLORS[transport], label=label)
    ax.set_title(title)
    ax.set_xticks(x, directions)
    ax.tick_params(axis="x", length=0)


def main() -> None:
    args = parse_args()
    style_matplotlib()
    rows = load_rows(args.input_dir / "factorized_paths_summary.jsonl")
    if not rows:
        raise ValueError(f"No factorized-path rows found under {args.input_dir}.")
    directions, transports, lookup = ordered(rows)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.2))
    fig.patch.set_facecolor(BACKGROUND)
    draw_grouped_bars(
        axes[0, 0],
        directions,
        transports,
        lookup,
        "autoregressive_iia_mean",
        "Raw autoregressive IIA",
    )
    draw_grouped_bars(
        axes[0, 1],
        directions,
        transports,
        lookup,
        "destination_normalized_transfer_mean",
        "Destination-normalized transfer",
    )
    draw_grouped_bars(
        axes[1, 0],
        directions,
        transports,
        lookup,
        "prediction_cosine_to_direct_mean",
        "Predicted displacement cosine to direct",
    )
    draw_grouped_bars(
        axes[1, 1],
        directions,
        transports,
        lookup,
        "operator_relative_distance_to_direct_mean",
        "Relative operator distance to direct",
    )
    axes[1, 0].set_ylim(-1.0, 1.05)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.09, hspace=0.35, wspace=0.22)

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
