"""Plot baseline result linear-probe accuracy by layer for text and image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_ROOT = REPO_ROOT / "results" / "baseline" / "gemma4_12b_it" / "digits"
IMAGE_ROOT = REPO_ROOT / "results" / "baseline_images" / "gemma4_12b_it" / "digits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "baseline_linear_probes"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"

TASK_SPECS = [
    ("T+", "text", "addition", TEXT_ROOT, "17", "#226a74", "o"),
    ("T-", "text", "subtraction", TEXT_ROOT, "17", "#a95642", "^"),
    ("I+", "image", "addition", IMAGE_ROOT, "-1", "#7a5b98", "s"),
    ("I-", "image", "subtraction", IMAGE_ROOT, "-1", "#b8872d", "D"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-root", type=Path, default=TEXT_ROOT)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="baseline_result_probe_layer_accuracy")
    parser.add_argument("--min-layer", type=int, default=32)
    parser.add_argument("--max-layer", type=int, default=48)
    return parser.parse_args()


def long_path(path: Path) -> str:
    path = Path(path)
    value = str(path)
    if os.name == "nt" and path.is_absolute() and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def load_jsonl(path: Path) -> list[dict]:
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
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


def setup_axis(ax, title: str) -> None:
    ax.set_title(title, pad=6)
    ax.set_facecolor(BACKGROUND)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)


def metric_value(row: dict) -> float:
    key = "top_1_accuracy" if "top_1_accuracy" in row else "accuracy"
    return float(row[key])


def collect_series(
    rows: list[dict],
    *,
    operation: str,
    position: str,
    min_layer: int,
    max_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row.get("modality") == operation
        and row.get("target") == "result"
        and str(row.get("position_name")) == position
        and min_layer <= int(row["layer"]) <= max_layer
    ]
    by_layer = {}
    for row in selected:
        by_layer[int(row["layer"])] = metric_value(row)
    layers = np.asarray(sorted(by_layer), dtype=int)
    values = np.asarray([by_layer[int(layer)] for layer in layers], dtype=float)
    return layers, values


def display_layer(layer: int) -> float:
    if layer <= 40:
        return float(layer)
    return 40.8 + 0.09 * float(layer - 40)


def main() -> None:
    args = parse_args()
    style_matplotlib()

    task_specs = [
        (label, modality, operation, args.text_root if modality == "text" else args.image_root, position, color, marker)
        for label, modality, operation, _root, position, color, marker in TASK_SPECS
    ]
    rows_by_modality = {}
    for _label, modality, _operation, root, _position, _color, _marker in task_specs:
        rows_by_modality.setdefault(
            modality,
            load_jsonl(root / "linear_probes" / "add_sub_probe_results.jsonl"),
        )

    fig, ax = plt.subplots(1, 1, figsize=(4.9, 2.55), facecolor=BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    summary: dict[str, dict] = {}

    setup_axis(ax, "Result linear probe")
    for label, modality, operation, _root, position, color, marker in task_specs:
        layers, values = collect_series(
            rows_by_modality[modality],
            operation=operation,
            position=position,
            min_layer=args.min_layer,
            max_layer=args.max_layer,
        )
        if layers.size == 0:
            raise ValueError(f"No rows for {label} at position {position}")
        ax.plot(
            [display_layer(int(layer)) for layer in layers],
            values,
            color=color,
            linewidth=1.15,
            linestyle="-",
            marker=marker,
            markersize=2.9,
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.4,
            label=label,
            zorder=3,
        )
        summary[label] = {
            "modality": modality,
            "operation": operation,
            "position": position,
            "metric": "top_1_accuracy" if "top_1_accuracy" in rows_by_modality[modality][0] else "accuracy",
            "layers": layers.astype(int).tolist(),
            "accuracy": values.tolist(),
            "max_accuracy": float(np.max(values)),
            "best_layer": int(layers[int(np.argmax(values))]),
        }
    ax.set_xlim(display_layer(args.min_layer), display_layer(args.max_layer) + 0.25)
    ax.set_xticks(
        [display_layer(layer) for layer in [32, 36, 40, 48]],
        ["32", "36", "40", "48"],
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy")
    ax.text(
        display_layer(40) + 0.36,
        -0.105,
        "...",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9.2,
        color="#555f62",
        clip_on=False,
    )
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            linestyle="",
            marker=marker,
            markersize=4.2,
            markerfacecolor=color,
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.4,
            label=label,
        )
        for label, _modality, _operation, _root, _position, color, marker in task_specs
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=2,
        frameon=False,
        labelspacing=0.24,
        columnspacing=0.65,
        handlelength=0.8,
        handletextpad=0.25,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.18)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    summary_path = args.output_dir / f"{args.stem}_summary.json"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
