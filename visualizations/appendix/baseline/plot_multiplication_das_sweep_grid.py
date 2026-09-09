"""Four-by-three multiplication DAS sweep grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_das_appendix import (
    BACKGROUND,
    K_COLORS,
    REPO_ROOT,
    TEXT,
    deduplicate_rows,
    load_jsonl,
    setup_axis,
    style_matplotlib,
)


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "das_v2"
    / "runs"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "baseline" / "das"

VARIABLES = [
    (
        "Result",
        "multiplication_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
    (
        r"$\hat{c}_1$",
        "multiplication_c1_hat_full_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
    (
        r"controlled $\hat{c}_1$",
        "multiplication_c1_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
    (
        r"$\hat{c}_0$",
        "multiplication_c0_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
]

METRICS = [
    ("autoregressive_iia", "Autoregressive IIA"),
    ("full_answer_teacher_forced_iia", "Teacher-forced IIA"),
    ("variable_teacher_forced_iia", "Variable IIA"),
]

KS = [16, 32]


def rows_for_file(path: Path) -> list[dict]:
    rows = [
        row
        for row in load_jsonl(path)
        if "layer" in row and "k" in row and int(row["k"]) in KS
    ]
    return deduplicate_rows(rows)


def plot_cell(ax, rows: list[dict], metric: str, show_xticks: bool):
    setup_axis(ax)
    lookup = {(int(row["layer"]), int(row["k"])): row for row in rows}
    layers = sorted({int(row["layer"]) for row in rows})
    handles = []
    labels = []
    for k in KS:
        k_layers = [layer for layer in layers if (layer, k) in lookup]
        if not k_layers:
            continue
        y = [float(lookup[(layer, k)][metric]) for layer in k_layers]
        handle = ax.plot(
            k_layers,
            y,
            color=K_COLORS.get(k, "#5d6264"),
            marker="o",
            markersize=3.0,
            linewidth=1.2,
            label=f"k={k}",
        )[0]
        handles.append(handle)
        labels.append(f"k={k}")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(min(layers) - 0.4, max(layers) + 0.4)
    ax.set_xticks(layers)
    if not show_xticks:
        ax.set_xticklabels([])
    else:
        ax.tick_params(axis="x", labelrotation=0)
    return handles, labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    style_matplotlib()
    plt.rcParams.update(
        {
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.2,
        }
    )

    fig, axes = plt.subplots(
        len(VARIABLES),
        len(METRICS),
        figsize=(10.6, 8.2),
        sharey=True,
        constrained_layout=False,
    )
    fig.patch.set_facecolor(BACKGROUND)

    legend_handles = []
    legend_labels = []
    for row_idx, (variable_label, filename) in enumerate(VARIABLES):
        rows = rows_for_file(args.run_dir / filename)
        for col_idx, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            handles, labels = plot_cell(
                ax,
                rows,
                metric,
                show_xticks=row_idx == len(VARIABLES) - 1,
            )
            if row_idx == 0:
                ax.set_title(metric_label, pad=7, fontweight="normal")
            if col_idx == 0:
                ax.text(
                    -0.24,
                    0.5,
                    variable_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=10.0,
                    color=TEXT,
                )
            if row_idx == 0 and col_idx == 0:
                legend_handles = handles
                legend_labels = labels

    fig.suptitle("Multiplication variables DAS sweep", y=0.975, color=TEXT)
    fig.supxlabel("Layer", y=0.055, color=TEXT)
    fig.supylabel("IIA", x=0.006, color=TEXT)
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=len(legend_labels),
        handlelength=1.8,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.12, right=0.992, top=0.91, bottom=0.13, wspace=0.12, hspace=0.20)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "multiplication_variables_das_sweep_grid.png"
    pdf_path = args.output_dir / "multiplication_variables_das_sweep_grid.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
