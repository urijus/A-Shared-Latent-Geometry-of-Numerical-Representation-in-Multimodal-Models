"""Appendix Procrustes diagnostics: seed stability, rank sweep, factorized paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "image"
DAS_AUDIT_ROOT = REPO_ROOT / "results" / "final_exps" / "DAS_audit_k_22"
PROCRUSTES_ROOT = REPO_ROOT / "results" / "final_exps" / "procrustes"

BACKGROUND = "#fdfdfd"
TEXT = "#171717"
SPINE = "#b9c0c2"
GRID = "#d8dddd"
BLUE = "#446b8f"
TEAL = "#2f7f7b"
RED = "#b97068"
GOLD = "#c49a4a"
VIOLET = "#76608a"

TASK_LABELS = {
    "text:addition": r"$T_+$",
    "text:subtraction": r"$T_-$",
    "image:addition": r"$I_+$",
    "image:subtraction": r"$I_-$",
}

DIRECTION_ORDER = [
    ("text:addition", "image:addition"),
    ("image:addition", "text:addition"),
    ("text:subtraction", "image:subtraction"),
    ("image:subtraction", "text:subtraction"),
]

DIRECTION_COLORS = {
    ("text:addition", "image:addition"): BLUE,
    ("image:addition", "text:addition"): TEAL,
    ("text:subtraction", "image:subtraction"): RED,
    ("image:subtraction", "text:subtraction"): VIOLET,
}

FACTOR_DIRECTIONS = [
    ("image:addition", "text:subtraction"),
    ("image:subtraction", "text:addition"),
    ("text:addition", "image:subtraction"),
    ("text:subtraction", "image:addition"),
]

TRANSPORTS = ["direct", "modality_first", "operation_first"]
TRANSPORT_LABELS = {
    "direct": "Direct",
    "modality_first": "Modality first",
    "operation_first": "Operation first",
}
TRANSPORT_COLORS = {"direct": BLUE, "modality_first": TEAL, "operation_first": GOLD}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--das-audit-root", type=Path, default=DAS_AUDIT_ROOT)
    parser.add_argument("--procrustes-root", type=Path, default=PROCRUSTES_ROOT)
    return parser.parse_args()


def long_path(path: Path) -> str:
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_jsonl(path: Path) -> list[dict]:
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib() -> None:
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
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.titlesize": 14,
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


def setup_axis(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.0)


def setup_grid_axis(ax) -> None:
    setup_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.68)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.28)


def heat_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "appendix_overlap_heat",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )


def save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(paths[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def direction_label(source: str, destination: str) -> str:
    return f"{TASK_LABELS[source]} $\\rightarrow$ {TASK_LABELS[destination]}"


def task_label(modality: str, operation: str) -> str:
    return TASK_LABELS[f"{modality}:{operation}"]


def annotate_matrix(ax, matrix: np.ndarray) -> None:
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            color = BACKGROUND if value > 0.78 else TEXT
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8.3, color=color)


def plot_seed_stability(das_audit_root: Path, output_dir: Path) -> list[Path]:
    rows = load_jsonl(das_audit_root / "subspace_reliability_matrices.jsonl")
    lookup = {
        (row["modality"], row["operation"], row["target"], row["condition"]): row
        for row in rows
    }
    panels = [
        ("text", "addition"),
        ("text", "subtraction"),
        ("image", "addition"),
        ("image", "subtraction"),
    ]

    style_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    last_image = None
    for ax, (modality, operation) in zip(axes.ravel(), panels):
        row = lookup[(modality, operation, "result", "das_pca_initialized")]
        matrix = np.array(row["matrix"], dtype=float)
        seeds = [str(seed) for seed in row["seeds"]]
        setup_axis(ax)
        ax.tick_params(length=0)
        last_image = ax.imshow(matrix, cmap=heat_cmap(), vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(
            f"{task_label(modality, operation)}  mean off-diag = {row['mean_off_diagonal']:.2f}",
            pad=8,
            fontweight="normal",
        )
        ax.set_xticks(range(len(seeds)), seeds)
        ax.set_yticks(range(len(seeds)), seeds)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Seed")
        ax.set_xticks(np.arange(-0.5, len(seeds), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(seeds), 1), minor=True)
        ax.grid(which="minor", color=BACKGROUND, linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        annotate_matrix(ax, matrix)

    fig.subplots_adjust(left=0.075, right=0.83, top=0.89, bottom=0.09, wspace=0.32, hspace=0.36)
    cax = fig.add_axes([0.885, 0.20, 0.018, 0.58])
    cbar = fig.colorbar(last_image, cax=cax)
    cbar.outline.set_edgecolor(SPINE)
    cbar.outline.set_linewidth(0.7)
    cbar.set_label("Symmetric subspace overlap", fontsize=9.5)
    fig.suptitle("Cross-Seed DAS Subspace Stability", y=0.985, color=TEXT)
    return save(fig, output_dir, "four_task_seed_stability")


def rank_groups(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["source_task"], row["destination_task"])
        groups.setdefault(key, []).append(row)
    for part in groups.values():
        part.sort(key=lambda row: int(row["rank"]))
    return groups


def threshold_rank(rows: list[dict], threshold: float) -> tuple[int, float] | None:
    final = float(rows[-1]["destination_normalized_transfer_mean"])
    if final <= 0:
        return None
    target = threshold * final
    for row in rows:
        value = float(row["destination_normalized_transfer_mean"])
        if value >= target:
            return int(row["rank"]), value
    return int(rows[-1]["rank"]), float(rows[-1]["destination_normalized_transfer_mean"])


def plot_rank_sweep(procrustes_root: Path, output_dir: Path) -> list[Path]:
    rows = load_jsonl(procrustes_root / "rank_sweep" / "rank_sweep_summary.jsonl")
    groups = rank_groups(rows)
    metrics = [
        ("cumulative_singular_value_fraction_mean", "(a) Cumulative singular-value mass", "Cumulative mass", (0.0, 1.03)),
        ("alignment_test_cosine_mean", "(b) Held-out displacement cosine", "Cosine", (0.0, 1.03)),
        (
            "destination_normalized_transfer_mean",
            "(c) Destination-normalized causal transfer",
            "Normalized transfer",
            (0.0, 1.15),
        ),
    ]

    style_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.2), sharex=True, constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    curve_handles = []
    curve_labels = []
    threshold_handles = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor=TEXT, markerfacecolor=BACKGROUND, markersize=5.0, label="80%"),
        Line2D([0], [0], marker="s", color="none", markeredgecolor=TEXT, markerfacecolor=BACKGROUND, markersize=5.0, label="90%"),
        Line2D([0], [0], marker="D", color="none", markeredgecolor=TEXT, markerfacecolor=BACKGROUND, markersize=4.7, label="95%"),
    ]
    threshold_markers = [(0.80, "o"), (0.90, "s"), (0.95, "D")]

    for ax, (metric, title, ylabel, ylim) in zip(axes, metrics):
        setup_grid_axis(ax)
        for direction in DIRECTION_ORDER:
            part = groups[direction]
            color = DIRECTION_COLORS[direction]
            x = np.array([int(row["rank"]) for row in part], dtype=float)
            y = np.array([float(row[metric]) for row in part], dtype=float)
            line = ax.plot(x, y, color=color, linewidth=1.75, label=direction_label(*direction))[0]
            std_key = metric.replace("_mean", "_std")
            if std_key in part[0]:
                std = np.array([float(row.get(std_key) or 0.0) for row in part], dtype=float)
                ax.fill_between(x, np.clip(y - std, ylim[0], ylim[1]), np.clip(y + std, ylim[0], ylim[1]), color=color, alpha=0.12, linewidth=0)
            if ax is axes[0]:
                curve_handles.append(line)
                curve_labels.append(direction_label(*direction))
            if metric == "destination_normalized_transfer_mean":
                for threshold, marker in threshold_markers:
                    point = threshold_rank(part, threshold)
                    if point is None:
                        continue
                    rank, value = point
                    ax.scatter(
                        [rank],
                        [value],
                        marker=marker,
                        s=34,
                        facecolor=BACKGROUND,
                        edgecolor=color,
                        linewidth=1.2,
                        zorder=4,
                    )
        ax.set_title(title, pad=7, fontweight="normal")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    axes[-1].set_xlabel(r"Retained rank $m$")
    axes[-1].set_xticks(range(1, 23, 2))
    axes[0].legend(curve_handles, curve_labels, loc="lower right", ncol=2, frameon=False, handlelength=1.8)
    axes[2].legend(handles=threshold_handles, title="Full-rank effect", frameon=False, loc="lower right", ncol=3, handlelength=1.0, columnspacing=0.9)
    fig.suptitle(r"Rank-$m$ Procrustes Diagnostics", y=0.985, color=TEXT)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.92, bottom=0.075, hspace=0.33)
    return save(fig, output_dir, "rank_sweep_full_diagnostics")


def plot_factorized_paths(procrustes_root: Path, output_dir: Path) -> list[Path]:
    rows = [
        row
        for row in load_jsonl(procrustes_root / "factorized_paths" / "factorized_paths_summary.jsonl")
        if row.get("test_family") == "cross_operation_factorized"
    ]
    lookup = {(row["source_task"], row["destination_task"], row["transport"]): row for row in rows}
    metrics = [
        ("destination_normalized_transfer_mean", "(a) Destination-normalized transfer", (0.0, 1.08), True),
        ("alignment_test_cosine_mean", "(b) Cosine to true destination displacement", (0.82, 0.92), False),
        ("prediction_cosine_to_direct_mean", "(c) Prediction cosine to direct map", (0.94, 1.005), False),
        ("operator_relative_distance_to_direct_mean", "(d) Relative Frobenius distance to direct", (0.0, 0.55), False),
    ]

    style_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 5.9), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    x = np.arange(len(FACTOR_DIRECTIONS), dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    tick_labels = [direction_label(source, destination) for source, destination in FACTOR_DIRECTIONS]

    for ax, (metric, title, ylim, show_error) in zip(axes.ravel(), metrics):
        setup_grid_axis(ax)
        for offset, transport in zip(offsets, TRANSPORTS):
            vals = [float(lookup[(source, destination, transport)][metric]) for source, destination in FACTOR_DIRECTIONS]
            std_key = metric.replace("_mean", "_std")
            errs = [
                float(lookup[(source, destination, transport)].get(std_key) or 0.0)
                for source, destination in FACTOR_DIRECTIONS
            ]
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=TRANSPORT_COLORS[transport],
                label=TRANSPORT_LABELS[transport],
                yerr=errs if show_error else None,
                error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8, "ecolor": "#5d6264"},
            )
        ax.set_title(title, pad=7, fontweight="normal")
        ax.set_ylim(*ylim)
        ax.set_xticks(x, tick_labels, rotation=24, ha="right", rotation_mode="anchor")
        if metric == "operator_relative_distance_to_direct_mean":
            ax.set_ylabel("Relative distance")
        elif "cosine" in metric:
            ax.set_ylabel("Cosine")
        else:
            ax.set_ylabel("Normalized transfer")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Direct and Composed Cross-Operation Transports", y=0.985, color=TEXT)
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.15, wspace=0.25, hspace=0.47)
    return save(fig, output_dir, "factorized_paths_full_summary")


def main() -> None:
    args = parse_args()
    outputs: list[Path] = []
    outputs.extend(plot_seed_stability(args.das_audit_root, args.output_dir))
    outputs.extend(plot_rank_sweep(args.procrustes_root, args.output_dir))
    outputs.extend(plot_factorized_paths(args.procrustes_root, args.output_dir))
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
