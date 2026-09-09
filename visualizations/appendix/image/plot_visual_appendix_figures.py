"""Generate appendix figures for visual arithmetic and transport results."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "image"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
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
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 8.0,
            "figure.titlesize": 12.0,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def setup_axis(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.75)
    ax.tick_params(width=0.7, length=2.6)


def heat_cmap():
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "appendix_heat",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )
    cmap.set_bad(BACKGROUND)
    return cmap


def line_palette() -> dict[str, str]:
    return {
        "text:addition->image:addition": BLUE,
        "text:subtraction->image:subtraction": RED,
        "image:addition->text:addition": TEAL,
        "image:subtraction->text:subtraction": VIOLET,
        "image:addition->text:subtraction": TEAL,
        "image:subtraction->text:addition": VIOLET,
        "text:addition->image:subtraction": BLUE,
        "text:subtraction->image:addition": RED,
    }


def save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(paths[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return paths


def matrix_from_rows(rows: list[dict], value_key: str, *, target: str | None = None) -> tuple[np.ndarray, list[int], list[str]]:
    if target is not None:
        rows = [row for row in rows if row.get("target") == target]
    layers = sorted({int(row["layer"]) for row in rows})
    positions = sorted({str(row.get("position_name", row.get("position"))) for row in rows}, key=lambda x: int(x))
    index_l = {layer: i for i, layer in enumerate(layers)}
    index_p = {pos: i for i, pos in enumerate(positions)}
    matrix = np.full((len(layers), len(positions)), np.nan)
    for row in rows:
        layer = int(row["layer"])
        pos = str(row.get("position_name", row.get("position")))
        if row.get(value_key) is not None:
            matrix[index_l[layer], index_p[pos]] = float(row[value_key])
    return matrix, layers, positions


def draw_heatmap(
    ax,
    matrix: np.ndarray,
    layers: list[int],
    positions: list[str],
    title: str,
    *,
    vmin=0.0,
    vmax=1.0,
    missing_label: str | None = None,
):
    setup_axis(ax)
    image = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto", cmap=heat_cmap(), vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=7)
    ax.set_xlabel("Position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(len(positions)), positions)
    yticks = np.linspace(0, len(layers) - 1, min(6, len(layers)), dtype=int)
    ax.set_yticks(yticks, [str(layers[i]) for i in yticks])
    if missing_label and np.isnan(matrix).all():
        ax.text(
            0.5,
            0.5,
            missing_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#6f6f6f",
        )
    return image


def aggregate_patch(op: str, hook: str = "resid_post", metric: str = "patched_clean_iia"):
    path = (
        REPO_ROOT
        / "results"
        / "baseline_images"
        / "gemma4_12b_it"
        / "digits"
        / "activation_patching"
        / op
        / f"{op}_activation_patching_layers33-34-35-36-37-38-39-40-41-42-43-44-45-46-47-48.jsonl"
    )
    rows = load_jsonl(path)
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["hook"] != hook:
            continue
        grouped[(int(row["layer"]), int(row["position"]))].append(float(row[metric]))
    layers = sorted({key[0] for key in grouped})
    positions = sorted({key[1] for key in grouped})
    matrix = np.full((len(layers), len(positions)), np.nan)
    for (layer, position), values in grouped.items():
        matrix[layers.index(layer), positions.index(position)] = float(np.mean(values))
    return matrix, layers, [str(pos) for pos in positions]


def das_reference_scores(base: str = "DAS_audit_k_22") -> dict[str, float]:
    out = {}
    for op in ["addition", "subtraction"]:
        path = REPO_ROOT / "results" / "final_exps" / base / "image" / op / "summary.jsonl"
        rows = load_jsonl(path)
        vals = [float(row["autoregressive_iia"]) for row in rows if row.get("condition") == "das_pca_initialized"]
        if vals:
            out[op] = float(np.mean(vals))
    return out


def plot_visual_result_localization(output_dir: Path) -> list[Path]:
    probe_rows = load_jsonl(
        REPO_ROOT
        / "results"
        / "baseline_images"
        / "gemma4_12b_it"
        / "digits"
        / "linear_probes"
        / "add_sub_probe_results.jsonl"
    )
    add_probe = [row for row in probe_rows if row["modality"] == "addition"]
    sub_probe = [row for row in probe_rows if row["modality"] == "subtraction"]
    add_matrix, add_layers, add_positions = matrix_from_rows(add_probe, "top_1_accuracy", target="result")
    sub_matrix, sub_layers, sub_positions = matrix_from_rows(sub_probe, "top_1_accuracy", target="result")
    patch_add, patch_layers, patch_positions = aggregate_patch("addition")
    patch_sub, _, _ = aggregate_patch("subtraction")
    patch_mean = np.nanmean(np.stack([patch_add, patch_sub]), axis=0)
    das_scores = das_reference_scores("DAS_audit_k_22")

    style_matplotlib()
    fig = plt.figure(figsize=(9.4, 5.9))
    fig.patch.set_facecolor(BACKGROUND)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.78], height_ratios=[1, 1], wspace=0.34, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0:2])
    ax_d = fig.add_subplot(grid[:, 2])

    image = draw_heatmap(ax_a, add_matrix, add_layers, add_positions, "A. Image addition probe", vmax=1.0)
    draw_heatmap(ax_b, sub_matrix, sub_layers, sub_positions, "B. Image subtraction probe", vmax=1.0)
    patch_image = draw_heatmap(ax_c, patch_mean, patch_layers, patch_positions, "C. Visual resid-post patching", vmax=max(0.40, float(np.nanmax(patch_mean))))
    setup_axis(ax_d)
    ops = ["addition", "subtraction"]
    vals = [das_scores.get(op, np.nan) for op in ops]
    ax_d.bar([0, 1], vals, color=[BLUE, RED], width=0.58)
    ax_d.set_xticks([0, 1], ["Addition", "Subtraction"], rotation=25, ha="right")
    ax_d.set_ylim(0, 1)
    ax_d.set_ylabel("Autoregressive IIA")
    ax_d.set_title("D. Reference DAS", pad=7)
    ax_d.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)
    for idx, val in enumerate(vals):
        if not math.isnan(val):
            ax_d.text(idx, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8.2)

    cbar = fig.colorbar(image, ax=[ax_a, ax_b], shrink=0.82, pad=0.025)
    cbar.set_label("Probe top-1 accuracy")
    cbar.outline.set_edgecolor(SPINE)
    cbar2 = fig.colorbar(patch_image, ax=ax_c, shrink=0.88, pad=0.015)
    cbar2.set_label("Mean patched-clean IIA")
    cbar2.outline.set_edgecolor(SPINE)
    fig.suptitle("Visual Result Localization", y=0.99)
    return save(fig, output_dir, "visual_result_localization_summary")


def load_why_image_summaries() -> list[dict]:
    root = REPO_ROOT / "results" / "final_exps" / "why_image_worse"
    rows = []
    for path in root.rglob("summary.jsonl"):
        for row in load_jsonl(path):
            row["_path"] = str(path)
            rows.append(row)
    return rows


def plot_image_addition_diagnostics(output_dir: Path) -> list[Path]:
    rows = [row for row in load_why_image_summaries() if row.get("modality") == "image" and row.get("operation") == "addition"]
    das = [row for row in rows if row.get("condition") == "das_pca_initialized"]
    k_rows = sorted(
        [row for row in das if str(row.get("position")) == "-1" and row.get("k") is not None],
        key=lambda row: int(row["k"]),
    )
    pos_rows = sorted(
        [row for row in das if int(row.get("k", -1)) == 22],
        key=lambda row: int(row["position"]),
    )
    controls = [row for row in rows if str(row.get("position")) == "-1" and int(row.get("k") or -1) == 16]

    style_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1.15, 0.9]})
    fig.patch.set_facecolor(BACKGROUND)

    ax = axes[0]
    setup_axis(ax)
    xs = [int(row["position"]) for row in pos_rows]
    ys = [float(row["autoregressive_iia"]) for row in pos_rows]
    ax.plot(xs, ys, marker="o", color=BLUE, linewidth=1.8, markersize=4.8)
    ax.set_title("A. Position sweep", pad=7)
    ax.set_xlabel("Image/token position")
    ax.set_ylabel("Autoregressive IIA")
    ax.set_ylim(-0.03, 0.80)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)

    ax = axes[1]
    setup_axis(ax)
    xs = [int(row["k"]) for row in k_rows]
    ys = [float(row["autoregressive_iia"]) for row in k_rows]
    ax.plot(xs, ys, marker="o", color=TEAL, linewidth=1.8, markersize=4.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_title("B. Dimension sweep", pad=7)
    ax.set_xlabel(r"DAS dimension $k$")
    ax.set_ylim(-0.03, 0.80)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)

    ax = axes[2]
    setup_axis(ax)
    labels = []
    vals = []
    colors = []
    for row in sorted(controls, key=lambda r: str(r.get("condition"))):
        labels.append(str(row["condition"]).replace("_", "\n"))
        vals.append(float(row["autoregressive_iia"]))
        colors.append(GOLD if row["condition"] == "das_pca_initialized" else SPINE)
    ax.bar(range(len(vals)), vals, color=colors, width=0.62)
    ax.set_xticks(range(len(vals)), labels, rotation=35, ha="right")
    ax.set_title("C. Controls at k=16", pad=7)
    ax.set_ylim(0, 0.80)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)
    for idx, val in enumerate(vals):
        ax.text(idx, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7.7)

    fig.suptitle("Image Addition Causal-Control Diagnostics", y=1.04)
    return save(fig, output_dir, "image_addition_diagnostics")


def direction_label(source: str, destination: str) -> str:
    return f"{TASK_LABELS[source]} $\\rightarrow$ {TASK_LABELS[destination]}"


def plot_rank_sweep(output_dir: Path) -> list[Path]:
    rows = load_jsonl(REPO_ROOT / "results" / "final_exps" / "procrustes" / "rank_sweep" / "rank_sweep_summary.jsonl")
    directions = list(dict.fromkeys((row["source_task"], row["destination_task"]) for row in rows))
    metrics = [
        ("cumulative_singular_value_fraction_mean", "Cumulative singular mass", (0, 1.02)),
        ("alignment_test_cosine_mean", "Held-out cosine", (0, 1.02)),
        ("autoregressive_iia_mean", "Raw autoregressive IIA", (0, 1.05)),
        ("destination_normalized_transfer_mean", "Destination-normalized transfer", (0, 1.18)),
    ]
    palette = line_palette()

    style_matplotlib()
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 8.3), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)
    for ax, (metric, ylabel, ylim) in zip(axes, metrics):
        setup_axis(ax)
        for source, destination in directions:
            key = f"{source}->{destination}"
            part = sorted([row for row in rows if row["source_task"] == source and row["destination_task"] == destination], key=lambda r: r["rank"])
            x = np.array([row["rank"] for row in part], dtype=float)
            y = np.array([row[metric] for row in part], dtype=float)
            color = palette.get(key, BLUE)
            ax.plot(x, y, color=color, linewidth=1.7, label=direction_label(source, destination))
            std_key = metric.replace("_mean", "_std")
            if std_key in part[0]:
                std = np.array([row.get(std_key) or 0.0 for row in part], dtype=float)
                ax.fill_between(x, y - std, y + std, color=color, alpha=0.14, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)
    axes[-1].set_xlabel(r"Retained rank $m$")
    axes[-1].set_xticks(range(1, 23, 2))
    axes[0].legend(loc="lower right", ncol=2, frameon=False)
    fig.suptitle(r"Cumulative Rank-$m$ Transport Diagnostics", y=0.995)
    return save(fig, output_dir, "rank_sweep_full_diagnostics")


def plot_factorized_paths(output_dir: Path) -> list[Path]:
    rows = load_jsonl(
        REPO_ROOT
        / "results"
        / "final_exps"
        / "procrustes"
        / "factorized_paths"
        / "factorized_paths_summary.jsonl"
    )
    rows = [row for row in rows if row["test_family"] == "cross_operation_factorized"]
    directions = [
        ("image:addition", "text:subtraction"),
        ("image:subtraction", "text:addition"),
        ("text:addition", "image:subtraction"),
        ("text:subtraction", "image:addition"),
    ]
    transports = ["direct", "modality_first", "operation_first"]
    labels = {"direct": "Direct", "modality_first": "Modality first", "operation_first": "Operation first"}
    colors = {"direct": BLUE, "modality_first": TEAL, "operation_first": GOLD}
    lookup = {(row["source_task"], row["destination_task"], row["transport"]): row for row in rows}
    metrics = [
        ("destination_normalized_transfer_mean", "Normalized transfer", (0, 1.12)),
        ("prediction_cosine_to_direct_mean", "Prediction cosine to direct", (0.94, 1.01)),
        ("operator_relative_distance_to_direct_mean", "Relative Frobenius distance", (0, 0.56)),
        ("alignment_test_cosine_mean", "Destination cosine", (0.82, 0.92)),
    ]

    style_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.8))
    fig.patch.set_facecolor(BACKGROUND)
    x = np.arange(len(directions))
    width = 0.23
    tick_labels = [direction_label(s, d) for s, d in directions]
    for ax, (metric, ylabel, ylim) in zip(axes.flat, metrics):
        setup_axis(ax)
        for idx, transport in enumerate(transports):
            vals = [lookup[(s, d, transport)].get(metric, 0.0) for s, d in directions]
            std_key = metric.replace("_mean", "_std")
            errs = [lookup[(s, d, transport)].get(std_key, 0.0) or 0.0 for s, d in directions]
            ax.bar(x + (idx - 1) * width, vals, width=width, yerr=errs if "transfer" in metric else None, color=colors[transport], label=labels[transport])
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x, tick_labels, rotation=28, ha="right")
        ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.1, 1.28))
    fig.suptitle("Direct and Composed Cross-Operation Transport", y=1.01)
    return save(fig, output_dir, "factorized_paths_full_summary")


def plot_visual_fourier(output_dir: Path) -> list[Path]:
    base = REPO_ROOT / "results" / "baseline_images" / "gemma4_12b_it" / "digits" / "fourier_probes"
    rows = {
        "addition": load_jsonl(base / "addition" / "add_sub_fourier_results_ridge.jsonl"),
        "subtraction": load_jsonl(base / "subtraction" / "add_sub_fourier_results_ridge.jsonl"),
    }
    periods = [2, 5, 10, 20, 50, 100]
    ops = ["addition", "subtraction"]
    default_layers = list(range(1, 49))
    default_positions = ["-4", "-3", "-2", "-1"]

    style_matplotlib()
    fig, axes = plt.subplots(2, len(periods), figsize=(12.2, 4.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)
    last_image = None
    for row_idx, op in enumerate(ops):
        op_rows = [row for row in rows[op] if row["target"] == "result"]
        for col_idx, period in enumerate(periods):
            part = [row for row in op_rows if int(row["period"]) == period]
            matrix, layers, positions = matrix_from_rows(part, "r2_mean")
            if matrix.size == 0:
                layers = default_layers
                positions = default_positions
                matrix = np.full((len(layers), len(positions)), np.nan)
            ax = axes[row_idx, col_idx]
            title = f"T={period}" if row_idx == 0 else ""
            last_image = draw_heatmap(ax, matrix, layers, positions, title, vmin=0.0, vmax=1.0, missing_label="not run")
            if col_idx != 0:
                ax.set_ylabel("")
            else:
                ax.set_ylabel(f"{op.title()}\nLayer")
            if row_idx == 0:
                ax.set_xlabel("")
    cbar = fig.colorbar(last_image, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015)
    cbar.set_label(r"Mean sine-cosine $R^2$")
    cbar.outline.set_edgecolor(SPINE)
    fig.suptitle("Visual Fourier-Probe Sweeps", y=1.02)
    return save(fig, output_dir, "visual_fourier_probe_summary")


def main() -> None:
    args = parse_args()
    outputs: list[Path] = []
    outputs.extend(plot_visual_result_localization(args.output_dir))
    outputs.extend(plot_image_addition_diagnostics(args.output_dir))
    outputs.extend(plot_rank_sweep(args.output_dir))
    outputs.extend(plot_factorized_paths(args.output_dir))
    outputs.extend(plot_visual_fourier(args.output_dir))
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
