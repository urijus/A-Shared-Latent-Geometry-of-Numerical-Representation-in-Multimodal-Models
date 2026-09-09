"""Render the main shared numerical geometry figure for causal L."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "experiments" / "closing" / "causal_L_shared_geometry"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "causal_L_shared_geometry"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"

TASK_ORDER = ["text:addition", "text:subtraction", "image:addition", "image:subtraction"]
TASK_LABELS = {
    "text:addition": "T+",
    "text:subtraction": "T-",
    "image:addition": "I+",
    "image:subtraction": "I-",
}
SPACE_SPECS = [
    ("causal_L", "causal L", "#226a74"),
    ("random_readout_orthogonal_13d", "random 13D RF", "#7a5b98"),
    ("shuffled_causal_L", "shuffled", "#a95642"),
]


def compact_value_label(value: float) -> str:
    if value < 0 and abs(value) < 0.01:
        return f"{value:.3f}"
    return f"{value:.2f}"


def control_label_y(value: float, error: float) -> float:
    if abs(value) < 0.01:
        return 0.055
    return value + error + 0.025
MATRIX_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_l_matrix",
    ["#d8dee0", "#b7d1ce", "#7db9aa", "#419287", "#226a74"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="causal_L_shared_numerical_geometry")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.6,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.1,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.0,
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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values) / math.sqrt(len(values)))


def relation_key(source: str, destination: str) -> str:
    return f"{TASK_LABELS[source]}->{TASK_LABELS[destination]}"


def row_relation(row: dict) -> str:
    source = row.get("source_task")
    destination = row.get("destination_task")
    if source in TASK_LABELS and destination in TASK_LABELS:
        return relation_key(source, destination)
    return row.get("task_relation", "")


def relation_level_values(rows: list[dict], metric: str, space: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("space_type") != space:
            continue
        value = to_float(row.get(metric))
        if value is None:
            continue
        grouped[row_relation(row)].append(value)
    return [mean(values) for _relation, values in sorted(grouped.items()) if values]


def matrix_from_rows(rows: list[dict], metric: str) -> tuple[np.ndarray, dict]:
    matrix = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    summary = {}
    for i, source in enumerate(TASK_ORDER):
        for j, destination in enumerate(TASK_ORDER):
            if source == destination:
                continue
            values = [
                to_float(row.get(metric))
                for row in rows
                if row.get("space_type") == "causal_L"
                and row.get("source_task") == source
                and row.get("destination_task") == destination
            ]
            values = [value for value in values if value is not None]
            if values:
                matrix[i, j] = mean(values)
                summary[relation_key(source, destination)] = {
                    "mean": mean(values),
                    "sem": sem(values),
                    "n": len(values),
                }
    return matrix, summary


def rsa_matrix_from_rows(rows: list[dict]) -> tuple[np.ndarray, dict]:
    matrix = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    summary = {}
    for i, task_a in enumerate(TASK_ORDER):
        for j, task_b in enumerate(TASK_ORDER):
            if task_a == task_b:
                continue
            pair_rows = [
                row
                for row in rows
                if row.get("space_type") == "causal_L"
                and (
                    (row.get("task_a") == task_a and row.get("task_b") == task_b)
                    or (row.get("task_a") == task_b and row.get("task_b") == task_a)
                )
            ]
            values = [to_float(row.get("spearman_rsa")) for row in pair_rows]
            values = [value for value in values if value is not None]
            pvalues = [to_float(row.get("permutation_pvalue")) for row in pair_rows]
            pvalues = [value for value in pvalues if value is not None]
            if values:
                matrix[i, j] = mean(values)
                summary[relation_key(task_a, task_b)] = {
                    "mean": mean(values),
                    "sem": sem(values),
                    "n": len(values),
                    "mean_pvalue": mean(pvalues),
                }
    return matrix, summary


def style_panel(
    ax,
    heading: str,
    panel: str,
    panel_x: float = 0.02,
    panel_y: float = 0.98,
    panel_fontsize: float = 10.0,
) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.text(panel_x, panel_y, panel, transform=ax.transAxes, ha="left", va="top", fontsize=panel_fontsize, clip_on=False)
    ax.text(0.5, 1.025, heading, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.0)


def draw_controls(ax, transition_rows: list[dict], retrieval_rows: list[dict], control_rows: list[dict]) -> dict:
    style_panel(ax, "Controls", "(a)", panel_x=0.02, panel_y=0.985, panel_fontsize=9.4)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)

    metrics = [
        ("transition", "Transition\ncosine", "heldout_transition_cosine_mean", transition_rows, control_rows),
        ("retrieval", "Top-1\naccuracy", "top1_value_retrieval", retrieval_rows, control_rows),
    ]
    x = np.arange(len(metrics), dtype=float)
    width = 0.22
    offsets = [-width, 0.0, width]
    summary = {}

    for offset, (space, label, color) in zip(offsets, SPACE_SPECS):
        means = []
        errors = []
        for metric_key, _metric_label, metric, primary, controls in metrics:
            source = controls if space == "shuffled_causal_L" else primary
            values = relation_level_values(source, metric, space)
            means.append(mean(values))
            errors.append(sem(values))
            summary.setdefault(metric_key, {})[space] = {
                "mean": mean(values),
                "sem": sem(values),
                "n_relations": len(values),
            }
        ax.bar(
            x + offset,
            means,
            yerr=errors,
            width=width,
            color=color,
            edgecolor=SPINE,
            linewidth=0.78,
            capsize=2.1,
            error_kw={"elinewidth": 0.8, "ecolor": TEXT, "capthick": 0.8},
            label=label,
            zorder=3,
        )
        for xpos, value, error in zip(x + offset, means, errors):
            ax.text(
                xpos,
                control_label_y(value, error),
                compact_value_label(value),
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

    ax.set_xticks(x, [label for _key, label, _metric, _primary, _control in metrics])
    ax.set_ylabel("Mean held-out score")
    ax.set_xlim(-0.5, 1.82)
    ax.set_ylim(-0.08, 1.14)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.99),
        ncol=3,
        columnspacing=0.38,
        handlelength=1.0,
        handletextpad=0.26,
        fontsize=7.4,
    )
    return summary


def draw_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    heading: str,
    panel: str,
    xlabel: str,
    ylabel: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    note: str | None = None,
):
    style_panel(ax, heading, panel)
    cmap = MATRIX_CMAP.copy()
    cmap.set_bad("#d7dcde")
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_yticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.tick_params(length=0, labelsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "-", ha="center", va="center", fontsize=9.0, color="#697174")
                continue
            normed = (value - vmin) / max(vmax - vmin, 1e-12)
            color = "white" if normed > 0.52 else TEXT
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.8, color=color)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.026)
    colorbar.set_label(cbar_label, fontsize=9.3)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.75)
    colorbar.ax.tick_params(labelsize=8.0, width=0.7, length=2.5)
    if note:
        ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.2, color="#555f62")
    return colorbar


def save_outputs(fig, output_dir: Path, stem: str, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    summary_path = output_dir / f"{stem}_summary.json"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {summary_path}")


def align_axis_to_reference_group(ax, left_ax, right_ax) -> None:
    """Match a panel's horizontal footprint to a reference axis plus colorbar."""
    fig = ax.figure
    fig.canvas.draw()
    ax_pos = ax.get_position()
    left_pos = left_ax.get_position()
    right_pos = right_ax.get_position()
    ax.set_position([left_pos.x0, ax_pos.y0, right_pos.x1 - left_pos.x0, ax_pos.height])



def shift_axis_group(ax, colorbar, dx: float) -> None:
    ax.figure.canvas.draw()
    for obj in (ax, colorbar.ax):
        box = obj.get_position()
        obj.set_position([box.x0 + dx, box.y0, box.width, box.height])


def main() -> None:
    args = parse_args()
    style_matplotlib()
    transition_rows = read_csv(args.input_dir / "L_transition_procrustes.csv")
    retrieval_rows = read_csv(args.input_dir / "L_value_retrieval.csv")
    rsa_rows = read_csv(args.input_dir / "L_rsa.csv")
    control_rows = read_csv(args.input_dir / "L_controls.csv")
    comparison_path = args.input_dir / "L_final_comparison.csv"
    comparison_rows = read_csv(comparison_path) if comparison_path.exists() else []
    comparison = {row["space_type"]: row for row in comparison_rows}

    transition_matrix, transition_summary = matrix_from_rows(transition_rows, "heldout_transition_cosine_mean")
    retrieval_matrix, retrieval_summary = matrix_from_rows(retrieval_rows, "top1_value_retrieval")
    rsa_matrix, rsa_summary = rsa_matrix_from_rows(rsa_rows)
    rsa_mean = float(np.nanmean(rsa_matrix))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
        facecolor=BACKGROUND,
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    fig.patch.set_facecolor(BACKGROUND)

    # convert 2x2 array into a flat list:
    # axes[0] top-left, axes[1] top-right, axes[2] bottom-left, axes[3] bottom-right
    axes = axes.ravel()

    panel_a = draw_controls(axes[0], transition_rows, retrieval_rows, control_rows)

    cbar_b = draw_heatmap(
        axes[1],
        transition_matrix,
        heading="Transition geometry",
        panel="(b)",
        xlabel="Destination",
        ylabel="Source",
        vmin=0.70,
        vmax=0.90,
        cbar_label="Cosine",
    )

    cbar_c = draw_heatmap(
        axes[2],
        retrieval_matrix,
        heading="Top-1 accuracy",
        panel="(c)",
        xlabel="Destination",
        ylabel="Source",
        vmin=0.60,
        vmax=0.90,
        cbar_label="Accuracy",
        note="chance = 0.05",
    )

    draw_heatmap(
        axes[3],
        rsa_matrix,
        heading="No-fit RSA",
        panel="(d)",
        xlabel="Destination",
        ylabel="Source",
        vmin=0.50,
        vmax=0.90,
        cbar_label="Spearman RSA",
        note=f"cross-condition mean = {rsa_mean:.3f}",
    )

    summary = {
        "panel_a_controls": panel_a,
        "panel_b_transition": transition_summary,
        "panel_c_retrieval": retrieval_summary,
        "panel_d_rsa": {"cells": rsa_summary, "cross_task_mean": rsa_mean},
        "global_means": comparison,
        "source": {
            "input_dir": str(args.input_dir),
            "transition_rows": len(transition_rows),
            "retrieval_rows": len(retrieval_rows),
            "rsa_rows": len(rsa_rows),
            "control_rows": len(control_rows),
        },
    }

    fig.subplots_adjust(
    left=0.08,
    right=0.97,
    top=0.90,
    bottom=0.10,
    wspace=0.03,
    hspace=0.32)

    # Make panel A have exactly the same horizontal footprint
    # as panel C + its colorbar
    align_axis_to_reference_group(
        axes[0],
        axes[2],
        cbar_c.ax,
    )

    save_outputs(fig, args.output_dir, args.stem, summary)


if __name__ == "__main__":
    main()
