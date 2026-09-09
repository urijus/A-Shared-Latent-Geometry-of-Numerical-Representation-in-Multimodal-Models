"""Four-panel image appendix figure: linear probes and DAS sweeps."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "results" / "baseline_images" / "gemma4_12b_it" / "digits"
OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "image"

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"

K_COLORS = {
    8: "#b85f5d",
    16: "#d08b57",
    32: "#5f9b92",
    64: "#6f7fae",
    128: "#8a6aa3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--hook", default="resid_post")
    parser.add_argument("--das-position", default="-1")
    parser.add_argument("--das-seed", default="8")
    parser.add_argument("--das-ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
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
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.5,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def appendix_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "appendix_probe_warm",
        ["#f5f1ef", "#ead8ce", "#ddb09d", "#c97f6d", "#8f4f59"],
    )


def patch_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "appendix_patch_teal",
        ["#f3f6f5", "#d9e7e2", "#adcfc6", "#6fa99b", "#3f7f75"],
    )


def setup_axis(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def rows_to_matrix(rows: list[dict], metric: str) -> tuple[np.ndarray, list[int], list[str]]:
    if not rows:
        raise ValueError("Cannot build a matrix from zero rows.")
    layers = sorted({int(row["layer"]) for row in rows})
    positions = sorted({row["position_name"] for row in rows}, key=lambda value: int(value))
    matrix = np.full((len(positions), len(layers)), np.nan, dtype=float)
    layer_index = {layer: index for index, layer in enumerate(layers)}
    position_index = {position: index for index, position in enumerate(positions)}
    for row in rows:
        matrix[position_index[row["position_name"]], layer_index[int(row["layer"])]] = float(row[metric])
    return matrix, layers, positions


def linear_probe_rows(result_root: Path, operation: str) -> list[dict]:
    rows = load_jsonl(result_root / "linear_probes" / "add_sub_probe_results.jsonl")
    return [
        row
        for row in rows
        if row.get("modality") == operation
        and row.get("target") == "result"
        and "normalized_gain" in row
    ]


def activation_patch_rows(result_root: Path, operation: str) -> list[dict]:
    path = (
        result_root
        / "activation_patching"
        / operation
        / f"{operation}_activation_patching_layers33-34-35-36-37-38-39-40-41-42-43-44-45-46-47-48.jsonl"
    )
    return load_jsonl(path)


def das_rows(result_root: Path, operation: str, *, position: str, hook: str, seed: str) -> list[dict]:
    path = (
        result_root
        / "das_v2"
        / "runs"
        / f"{operation}_result_pos{position}_{hook}_initrandom_pca_seed{seed}.jsonl"
    )
    return load_jsonl(path)


def draw_probe_heatmap(ax, rows: list[dict], title: str):
    matrix, layers, positions = rows_to_matrix(rows, "normalized_gain")
    matrix = np.clip(matrix, 0.0, 1.0)

    setup_axis(ax)
    ax.tick_params(length=0)
    image = ax.imshow(matrix, aspect="auto", cmap=appendix_cmap(), vmin=0.0, vmax=1.0)
    ax.set_title(title, pad=8, fontweight="normal")
    ax.set_xlabel("Layer", labelpad=6)
    ax.set_ylabel("Token position", labelpad=6)
    xtick_positions = [index for index, layer in enumerate(layers) if layer == 1 or layer % 4 == 0]
    ax.set_xticks(xtick_positions, [str(layers[index]) for index in xtick_positions])
    ax.set_yticks(range(len(positions)), positions)
    for boundary in np.arange(0.5, len(positions), 1):
        ax.axhline(boundary, color=BACKGROUND, linewidth=0.55, alpha=0.7)
    return image


def activation_patch_matrix(
    rows: list[dict],
    *,
    hook: str,
    metric: str,
) -> tuple[np.ndarray, list[int], list[str]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("hook") != hook or metric not in row:
            continue
        grouped[(int(row["layer"]), int(row["position"]))].append(float(row[metric]))
    if not grouped:
        raise ValueError(f"No activation-patching rows found for hook={hook}, metric={metric}.")
    layers = sorted({layer for layer, _ in grouped})
    positions = sorted({position for _, position in grouped})
    matrix = np.full((len(positions), len(layers)), np.nan, dtype=float)
    layer_index = {layer: index for index, layer in enumerate(layers)}
    position_index = {position: index for index, position in enumerate(positions)}
    for (layer, position), values in grouped.items():
        matrix[position_index[position], layer_index[layer]] = float(np.mean(values))
    return matrix, layers, [str(position) for position in positions]


def draw_patch_heatmap(ax, rows: list[dict], title: str, *, hook: str, metric: str):
    matrix, layers, positions = activation_patch_matrix(rows, hook=hook, metric=metric)
    matrix = np.clip(matrix, 0.0, 1.0)

    setup_axis(ax)
    ax.tick_params(length=0)
    image = ax.imshow(matrix, aspect="auto", cmap=patch_cmap(), vmin=0.0, vmax=1.0)
    ax.set_title(title, pad=8, fontweight="normal")
    ax.set_xlabel("Layer", labelpad=6)
    ax.set_ylabel("Token position", labelpad=6)
    xtick_positions = [index for index, layer in enumerate(layers) if layer == 33 or layer % 4 == 0]
    ax.set_xticks(xtick_positions, [str(layers[index]) for index in xtick_positions])
    ax.set_yticks(range(len(positions)), positions)
    for boundary in np.arange(0.5, len(positions), 1):
        ax.axhline(boundary, color=BACKGROUND, linewidth=0.55, alpha=0.7)
    return image


def deduplicate_das_rows(rows: list[dict]) -> list[dict]:
    best: dict[tuple[int, int], dict] = {}
    for row in rows:
        if "layer" not in row or "k" not in row:
            continue
        key = (int(row["layer"]), int(row["k"]))
        current = best.get(key)
        if current is None or float(row.get("autoregressive_iia", -1.0)) >= float(
            current.get("autoregressive_iia", -1.0)
        ):
            best[key] = row
    return list(best.values())


def draw_das_autoregressive_sweep(
    ax,
    rows: list[dict],
    title: str,
    *,
    hook: str,
    position: str,
    ks: list[int],
):
    rows = [
        row
        for row in deduplicate_das_rows(rows)
        if row.get("target") == "result"
        and str(row.get("hook")) == str(hook)
        and str(row.get("position")) == str(position)
        and "autoregressive_iia" in row
    ]
    if not rows:
        raise ValueError(f"No DAS rows found for hook={hook}, position={position}.")

    layers = sorted({int(row["layer"]) for row in rows})
    available_ks = sorted({int(row["k"]) for row in rows})
    ordered_ks = [k for k in ks if k in available_ks]

    setup_axis(ax)
    ax.tick_params(width=0.8, length=3.2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.68)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.28)

    handles = []
    labels = []
    for k in ordered_ks:
        selected = sorted(
            (row for row in rows if int(row["k"]) == k),
            key=lambda row: int(row["layer"]),
        )
        if not selected:
            continue
        line = ax.plot(
            [int(row["layer"]) for row in selected],
            [float(row["autoregressive_iia"]) for row in selected],
            color=K_COLORS.get(k, "#5d6264"),
            marker="o",
            markersize=3.6,
            linewidth=1.5,
            label=f"k={k}",
        )[0]
        handles.append(line)
        labels.append(f"k={k}")

    ax.set_title(title, pad=8, fontweight="normal")
    ax.set_xlabel("Layer", labelpad=6)
    ax.set_ylabel("Autoregressive IIA", labelpad=6)
    ax.set_xticks(layers)
    ax.set_ylim(-0.02, 1.02)
    return handles, labels


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def plot_four_panel(args: argparse.Namespace) -> list[Path]:
    style_matplotlib()
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.9), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)

    heat_image = draw_probe_heatmap(
        axes[0, 0],
        linear_probe_rows(args.result_root, "addition"),
        "(a) Image addition linear probe: exact result",
    )
    draw_probe_heatmap(
        axes[0, 1],
        linear_probe_rows(args.result_root, "subtraction"),
        "(b) Image subtraction linear probe: exact result",
    )
    handles, labels = draw_das_autoregressive_sweep(
        axes[1, 0],
        das_rows(
            args.result_root,
            "addition",
            position=args.das_position,
            hook=args.hook,
            seed=args.das_seed,
        ),
        "(c) Image addition DAS: autoregressive IIA",
        hook=args.hook,
        position=args.das_position,
        ks=args.das_ks,
    )
    draw_das_autoregressive_sweep(
        axes[1, 1],
        das_rows(
            args.result_root,
            "subtraction",
            position=args.das_position,
            hook=args.hook,
            seed=args.das_seed,
        ),
        "(d) Image subtraction DAS: autoregressive IIA",
        hook=args.hook,
        position=args.das_position,
        ks=args.das_ks,
    )
    patch_image = draw_patch_heatmap(
        axes[2, 0],
        activation_patch_rows(args.result_root, "addition"),
        "(e) Image addition resid-post patching",
        hook=args.hook,
        metric="patched_clean_iia",
    )
    draw_patch_heatmap(
        axes[2, 1],
        activation_patch_rows(args.result_root, "subtraction"),
        "(f) Image subtraction resid-post patching",
        hook=args.hook,
        metric="patched_clean_iia",
    )

    fig.subplots_adjust(left=0.07, right=0.88, top=0.925, bottom=0.115, wspace=0.24, hspace=0.58)
    cax = fig.add_axes([0.915, 0.708, 0.012, 0.16])
    colorbar = fig.colorbar(heat_image, cax=cax)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Normalized gain", fontsize=9.5)
    patch_cax = fig.add_axes([0.915, 0.18, 0.012, 0.16])
    patch_colorbar = fig.colorbar(patch_image, cax=patch_cax)
    patch_colorbar.outline.set_edgecolor(SPINE)
    patch_colorbar.outline.set_linewidth(0.7)
    patch_colorbar.set_label("Patched-clean IIA", fontsize=9.5)

    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.475, 0.025),
        ncol=min(len(labels), 5),
        handlelength=1.7,
        columnspacing=1.3,
    )

    fig.suptitle("Image Exact Result Probes, DAS and Activation Patching", y=0.985, color=TEXT)
    return save_figure(fig, args.output_dir, "image_exact_result_probe_das_four_panel")


def main() -> None:
    args = parse_args()
    for output in plot_four_panel(args):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
