"""Paper-style DAS comparison panels from stored DAS JSONL runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_das_appendix import (
    BACKGROUND,
    GRID,
    K_COLORS,
    REPO_ROOT,
    SPINE,
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

METRIC = "autoregressive_iia"
METRIC_LABEL = "Autoregressive IIA"


RESULT_PANELS = [
    (
        "Addition",
        "addition_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
    (
        "Subtraction",
        "subtraction_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
    (
        "Multiplication",
        "multiplication_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
]

DIGIT_PANELS = [
    (
        r"$\hat{c}_1$",
        "multiplication_c1_hat_full_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
    (
        r"$\hat{c}_0$",
        "multiplication_c0_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
    ),
]


def rows_for_file(path: Path, metric: str) -> list[dict]:
    rows = [row for row in load_jsonl(path) if metric in row and "layer" in row and "k" in row]
    return deduplicate_rows(rows)


def ks_with_full_layer_span(rows: list[dict], requested_ks: list[int] | None = None) -> list[int]:
    layers = sorted({int(row["layer"]) for row in rows})
    layer_set = set(layers)
    available_ks = sorted({int(row["k"]) for row in rows})
    if requested_ks is not None:
        available_ks = [k for k in requested_ks if k in available_ks]
    full_ks = []
    for k in available_ks:
        k_layers = {int(row["layer"]) for row in rows if int(row["k"]) == k}
        if k_layers == layer_set:
            full_ks.append(k)
    return full_ks


def plot_panel(ax, title: str, rows: list[dict], ks: list[int], metric: str):
    setup_axis(ax)
    layers = sorted({int(row["layer"]) for row in rows})
    lookup = {(int(row["layer"]), int(row["k"])): row for row in rows}
    handles = []
    labels = []
    for k in ks:
        y = [float(lookup[(layer, k)][metric]) for layer in layers]
        line = ax.plot(
            layers,
            y,
            color=K_COLORS.get(k, "#5d6264"),
            marker="o",
            markersize=3.3,
            linewidth=1.25,
            label=f"k={k}",
        )[0]
        handles.append(line)
        labels.append(f"k={k}")
    ax.set_title(title, pad=7, fontweight="normal")
    ax.set_xlabel("Layer")
    ax.set_xticks(layers)
    ax.set_ylim(-0.02, 1.02)
    return handles, labels


def unique_legend_entries(panel_entries):
    seen = set()
    handles = []
    labels = []
    for panel_handles, panel_labels in panel_entries:
        for handle, label in zip(panel_handles, panel_labels):
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def save_figure(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def plot_comparison(
    panels: list[tuple[str, str]],
    run_dir: Path,
    output_dir: Path,
    stem: str,
    title: str,
    figsize: tuple[float, float],
    requested_ks: list[int] | None,
):
    style_matplotlib()
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=True)
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BACKGROUND)

    legend_entries = []
    for ax, (panel_title, filename) in zip(axes, panels):
        rows = rows_for_file(run_dir / filename, METRIC)
        ks = ks_with_full_layer_span(rows, requested_ks)
        if not ks:
            raise ValueError(f"No k values span the full layer range in {filename}")
        legend_entries.append(plot_panel(ax, panel_title, rows, ks, METRIC))

    axes[0].set_ylabel(METRIC_LABEL)
    handles, labels = unique_legend_entries(legend_entries)
    fig.suptitle(title, y=0.965, color=TEXT)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 5),
        handlelength=1.8,
        columnspacing=1.35,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.78, bottom=0.27, wspace=0.18)
    return save_figure(fig, output_dir, stem)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = []
    outputs.extend(
        plot_comparison(
            RESULT_PANELS,
            args.run_dir,
            args.output_dir,
            stem="das_result_operations_autoregressive_iia",
            title="DAS result subspaces",
            figsize=(10.8, 3.35),
            requested_ks=args.ks,
        )
    )
    outputs.extend(
        plot_comparison(
            DIGIT_PANELS,
            args.run_dir,
            args.output_dir,
            stem="das_multiplication_c0hat_c1hat_autoregressive_iia",
            title=r"DAS multiplication digit subspaces",
            figsize=(7.4, 3.35),
            requested_ks=args.ks,
        )
    )
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
