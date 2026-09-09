import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pathlib
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BANK_DIGITS_ROOT = (
    REPO_ROOT
    / "src"
    / "banks"
    / "baseline_bank"
    / "gemma4_12b_it"
    / "digits"
)
BANK_ROOT = (
    BANK_DIGITS_ROOT
    / "subspace_overlap"
)
FOURIER_PROBE_ROOT = (
    BANK_DIGITS_ROOT
    / "fourier_probes"
    / "addition"
    / "fourier_projections"
    / "result_pos17_ridge"
    / "probes"
    / "result_addition"
)
DAS_SUBSPACE_ROOT = BANK_DIGITS_ROOT / "das" / "subspaces"
OUTPUT_DIR = (
    REPO_ROOT
    / "visualizations"
    / "appendix"
    / "baseline"
    / "appendix_subspace_overlap"
)

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
SOFT_GRID = "#d5dcde"
PERIODS = [5, 10, 20, 50, 100]
ADD_SUB_PERIODS = [2, 5, 10, 20, 50, 100]
LAYERS = list(range(40, 47))

DAS_FOURIER_SPECS = [
    (
        "addition_result_fourier_result",
        "addition_result_fourier_result_layer44_pos17_resid_post_seed8.jsonl",
        "Add result",
        "#b85f5d",
    ),
    (
        "subtraction_result_fourier_result",
        "subtraction_result_fourier_result_layer44_pos17_resid_post_seed8.jsonl",
        "Sub result",
        "#d08b57",
    ),
    (
        "multiplication_result_fourier_result",
        "multiplication_result_fourier_result_layer44_pos17_resid_post_seed8.jsonl",
        "Mul result",
        "#5f9b92",
    ),
    (
        "multiplication_c0_hat_fourier_c0_hat",
        "multiplication_c0_hat_fourier_c0_hat_layer44_pos17_resid_post_seed8.jsonl",
        r"Mul $\hat{c}_0$",
        "#6f7fae",
    ),
    (
        "multiplication_c1_hat_full_fourier_c1_hat",
        "multiplication_c1_hat_full_fourier_c1_hat_layer44_pos17_resid_post_seed8.jsonl",
        r"Mul $\hat{c}_1$",
        "#8a6aa3",
    ),
]

FOURIER_FOURIER_SPECS = [
    (
        "addition_result_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100",
        "addition_result_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100_pos17_ridge.jsonl",
        "addition_result",
        "Add result",
    ),
    (
        "subtraction_result_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100",
        "subtraction_result_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100_pos17_ridge.jsonl",
        "subtraction_result",
        "Sub result",
    ),
    (
        "multiplication_result_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100",
        "multiplication_result_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100_pos17_ridge.jsonl",
        "multiplication_result",
        "Mul result",
    ),
    (
        "multiplication_c0_hat_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100",
        "multiplication_c0_hat_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100_pos17_ridge.jsonl",
        "multiplication_c0_hat",
        r"Mul $\hat{c}_0$",
    ),
    (
        "multiplication_c1_hat_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100",
        "multiplication_c1_hat_layers_40_41_42_43_44_45_46_periods_5_10_20_50_100_pos17_ridge.jsonl",
        "multiplication_c1_hat",
        r"Mul $\hat{c}_1$",
    ),
]


def long_path(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_jsonl(path):
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def torch_load_portable(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except NotImplementedError as error:
        if "PosixPath" not in str(error) or not hasattr(pathlib, "WindowsPath"):
            raise
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            return torch.load(path, map_location="cpu", weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path


def orthonormal_columns(matrix):
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    if matrix.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {tuple(matrix.shape)}")
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    return left[:, :rank]


def fourier_in_das(das_basis, fourier_weight):
    das_basis = orthonormal_columns(das_basis)
    fourier_basis = orthonormal_columns(fourier_weight)
    squared_projection = torch.linalg.matrix_norm(das_basis.T @ fourier_basis).square()
    containment = float(squared_projection / fourier_basis.shape[1])
    random_expectation = das_basis.shape[1] / das_basis.shape[0]
    return containment, containment / random_expectation


def symmetric_overlap_from_weights(first_weight, second_weight):
    first_basis = orthonormal_columns(first_weight)
    second_basis = orthonormal_columns(second_weight)
    squared_projection = torch.linalg.matrix_norm(first_basis.T @ second_basis).square()
    return float(squared_projection / min(first_basis.shape[1], second_basis.shape[1]))


def result_fourier_probe_path(modality, layer, period):
    return (
        BANK_DIGITS_ROOT
        / "fourier_probes"
        / modality
        / "fourier_projections"
        / "result_pos17_ridge"
        / "probes"
        / f"result_{modality}"
        / f"result_T{period}_layer{layer}_pos17_ridge_probe.pt"
    )


def style_matplotlib():
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
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.7,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.2)


def plot_das_fourier_lines():
    style_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)

    for ax in axes:
        setup_axis(ax)
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100], labels=["2", "5", "10", "20", "50", "100"])
        ax.grid(axis="y", color=SOFT_GRID, linewidth=0.7, alpha=0.62)
        ax.grid(axis="x", color=SOFT_GRID, linewidth=0.45, alpha=0.26)
        ax.set_xlabel("Fourier period $T$")

    for folder, filename, label, color in DAS_FOURIER_SPECS:
        rows = load_jsonl(BANK_ROOT / "das_to_fourier" / folder / "layer44" / filename)
        rows = sorted(rows, key=lambda row: int(row["period"]))
        periods = [int(row["period"]) for row in rows]
        containment = [float(row["fourier_in_das"]) for row in rows]
        enrichment = [float(row["fourier_in_das_over_random"]) for row in rows]
        axes[0].plot(
            periods,
            containment,
            marker="o",
            markersize=4.2,
            linewidth=1.85,
            color=color,
            label=label,
        )
        axes[1].plot(
            periods,
            enrichment,
            marker="o",
            markersize=4.2,
            linewidth=1.85,
            color=color,
            label=label,
        )

    axes[0].set_title(r"Fourier containment in DAS, L44", pad=8)
    axes[0].set_ylabel(r"$C(\mathcal{F}_T \subset \mathcal{D})$")
    axes[0].set_ylim(bottom=0)

    axes[1].set_title("Enrichment over random subspaces, L44", pad=8)
    axes[1].set_ylabel(r"$C / \mathbb{E}[C_{\mathrm{rand}}]$")
    axes[1].set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        handlelength=1.55,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.28, wspace=0.27)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_DIR / "das_fourier_layer44_period_lines.png",
        OUTPUT_DIR / "das_fourier_layer44_period_lines.pdf",
    ]
    fig.savefig(outputs[0], dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(outputs[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return outputs


def find_addition_das_subspace(layer):
    matches = sorted(
        DAS_SUBSPACE_ROOT.glob(
            f"addition_result_layer{layer}_pos17_resid_post_initrandom_pca_k32_seed8*.pt"
        )
    )
    if not matches:
        raise FileNotFoundError(f"No addition result DAS subspace found for layer {layer}")
    return matches[-1]


def plot_addition_same_layer_das_fourier():
    style_matplotlib()
    layer_paths = []
    for layer in [40, 41, 42, 43, 44, 45, 46]:
        matches = sorted(
            DAS_SUBSPACE_ROOT.glob(
                f"addition_result_layer{layer}_pos17_resid_post_initrandom_pca_k32_seed8*.pt"
            )
        )
        if matches:
            layer_paths.append((layer, matches[-1]))

    periods = [2, 5, 10, 20, 50, 100]
    period_colors = {
        2: "#b85f5d",
        5: "#d08b57",
        10: "#c3a35c",
        20: "#5f9b92",
        50: "#6f7fae",
        100: "#8a6aa3",
    }

    fig, ax = plt.subplots(figsize=(5.75, 4.05), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)
    ax.grid(axis="y", color=SOFT_GRID, linewidth=0.7, alpha=0.62)
    ax.grid(axis="x", color=SOFT_GRID, linewidth=0.45, alpha=0.26)

    best = None
    for period in periods:
        x_values = []
        y_values = []
        for layer, das_path in layer_paths:
            probe_path = FOURIER_PROBE_ROOT / f"result_T{period}_layer{layer}_pos17_ridge_probe.pt"
            if not probe_path.exists():
                continue
            das_artifact = torch_load_portable(das_path)
            probe_artifact = torch_load_portable(probe_path)
            containment, enrichment = fourier_in_das(
                das_artifact["bases"][layer],
                probe_artifact["weight"],
            )
            x_values.append(layer)
            y_values.append(containment)
            if best is None or containment > best[2]:
                best = (layer, period, containment, enrichment)
        if x_values:
            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=4.3,
                linewidth=1.85,
                color=period_colors[period],
                label=f"T={period}",
            )

    if best is not None:
        layer, period, containment, _ = best
        ax.scatter(
            [layer],
            [containment],
            s=54,
            facecolor=BACKGROUND,
            edgecolor="#5d6264",
            linewidth=1.1,
            zorder=5,
        )
        ax.annotate(
            f"max: L{layer}, T={period}",
            xy=(layer, containment),
            xytext=(-98, 10),
            textcoords="offset points",
            fontsize=8.6,
            color="#4f5658",
            arrowprops={
                "arrowstyle": "-",
                "color": "#697174",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax.set_title("Same-layer DAS/Fourier alignment, addition result", pad=10, fontsize=13)
    ax.set_xlabel("Layer", labelpad=3)
    ax.set_ylabel(r"$C(\mathcal{F}_T \subset \mathcal{D}_{\ell})$")
    ax.set_xticks([layer for layer, _ in layer_paths])
    y_top = ax.get_ylim()[1]
    ax.set_ylim(bottom=0, top=y_top * 1.10)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.47),
        ncol=3,
        handlelength=1.45,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.86, bottom=0.39)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_DIR / "addition_result_same_layer_das_fourier_overlap.png",
        OUTPUT_DIR / "addition_result_same_layer_das_fourier_overlap.pdf",
    ]
    fig.savefig(outputs[0], dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(outputs[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return outputs


def label_for(layer, period):
    return f"L{layer}, T{period}"


def matrix_from_fourier_rows(rows, variable_prefix):
    labels = [f"{variable_prefix}_L{layer}_T{period}" for layer in LAYERS for period in PERIODS]
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for row_index, first_label in enumerate(labels):
        for col_index, second_label in enumerate(labels):
            if first_label == second_label:
                matrix[row_index, col_index] = 1.0
            else:
                matrix[row_index, col_index] = lookup[(first_label, second_label)][
                    "symmetric_overlap"
                ]
    return matrix


def matrix_from_labels(rows, labels, metric="symmetric_overlap"):
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for row_index, first_label in enumerate(labels):
        for col_index, second_label in enumerate(labels):
            matrix[row_index, col_index] = lookup[(first_label, second_label)][metric]
    return matrix


def plot_add_sub_fourier_period_matrix():
    style_matplotlib()
    labels = [
        f"{prefix}_result_T{period}"
        for prefix in ["add", "sub"]
        for period in ADD_SUB_PERIODS
    ]
    label_specs = [
        (modality, period)
        for modality in ["addition", "subtraction"]
        for period in ADD_SUB_PERIODS
    ]
    tick_labels = [f"T{period}" for _ in ["add", "sub"] for period in ADD_SUB_PERIODS]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper_add_sub_fourier_overlap",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )

    matrices = []
    max_value = 0.0
    for layer in [42, 43, 44]:
        weights = {}
        for modality, period in label_specs:
            probe_path = result_fourier_probe_path(modality, layer, period)
            weights[(modality, period)] = torch_load_portable(probe_path)["weight"]
        matrix = np.full((len(label_specs), len(label_specs)), np.nan, dtype=float)
        for row_index, first_spec in enumerate(label_specs):
            for col_index, second_spec in enumerate(label_specs):
                matrix[row_index, col_index] = symmetric_overlap_from_weights(
                    weights[first_spec],
                    weights[second_spec],
                )
        matrices.append((layer, matrix))
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        max_value = max(max_value, float(np.nanpercentile(off_diag, 99)))
    vmax = max(0.18, min(0.75, max_value))

    fig, axes = plt.subplots(1, 3, figsize=(10.85, 3.55), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    image = None
    for ax, (layer, matrix) in zip(axes, matrices):
        setup_axis(ax)
        plot_matrix = matrix.copy()
        np.fill_diagonal(plot_matrix, np.nan)
        plot_cmap = cmap.copy()
        plot_cmap.set_bad("#d9dee0")
        image = ax.imshow(
            plot_matrix,
            cmap=plot_cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"Layer {layer}", pad=8)
        ax.set_xticks(range(len(labels)), tick_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), tick_labels)
        ax.tick_params(axis="both", length=0)
        ax.axhline(5.5, color=BACKGROUND, linewidth=1.1)
        ax.axvline(5.5, color=BACKGROUND, linewidth=1.1)
        ax.text(
            2.5,
            len(labels) + 1.45,
            "Addition",
            ha="center",
            va="top",
            fontsize=9,
            clip_on=False,
        )
        ax.text(
            8.5,
            len(labels) + 1.45,
            "Subtraction",
            ha="center",
            va="top",
            fontsize=9,
            clip_on=False,
        )
        ax.text(
            -2.45,
            2.5,
            "Addition",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            clip_on=False,
        )
        ax.text(
            -2.45,
            8.5,
            "Subtraction",
            ha="center",
            va="center",
            rotation=90,
            fontsize=9,
            clip_on=False,
        )
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if row_index == col_index:
                    ax.text(
                        col_index,
                        row_index,
                        "1",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="#5d6669",
                    )
                elif value >= 0.50:
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.3,
                        color="white" if value > 0.48 else TEXT,
                    )

    fig.subplots_adjust(left=0.065, right=0.905, top=0.88, bottom=0.27, wspace=0.22)
    if image is not None:
        cax = fig.add_axes([0.93, 0.26, 0.014, 0.58])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.outline.set_edgecolor(SPINE)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label("Symmetric overlap", fontsize=9.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_DIR / "add_sub_fourier_layers42_43_44_period_overlap_matrix.png",
        OUTPUT_DIR / "add_sub_fourier_layers42_43_44_period_overlap_matrix.pdf",
    ]
    fig.savefig(outputs[0], dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(outputs[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return outputs


def draw_layer_period_matrix(ax, matrix, title, cmap, vmax):
    setup_axis(ax)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, pad=7)
    centers = [layer_index * len(PERIODS) + (len(PERIODS) - 1) / 2 for layer_index in range(len(LAYERS))]
    ax.set_xticks(centers, labels=[str(layer) for layer in LAYERS])
    ax.set_yticks(centers, labels=[str(layer) for layer in LAYERS])
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")

    for boundary in np.arange(len(PERIODS) - 0.5, matrix.shape[0], len(PERIODS)):
        ax.axhline(boundary, color=BACKGROUND, linewidth=0.9, alpha=0.95)
        ax.axvline(boundary, color=BACKGROUND, linewidth=0.9, alpha=0.95)

    for layer_index, center in enumerate(centers):
        if layer_index % 2 == 0:
            start = layer_index * len(PERIODS)
            ax.axhspan(start - 0.5, start + len(PERIODS) - 0.5, color="#ffffff", alpha=0.045)
            ax.axvspan(start - 0.5, start + len(PERIODS) - 0.5, color="#ffffff", alpha=0.045)

    return image


def draw_guide_panel(ax):
    ax.set_facecolor(BACKGROUND)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.93, "Reading guide", ha="left", va="top", fontsize=13, color=TEXT)
    ax.text(
        0.03,
        0.76,
        "Each block compares two layers.\n"
        "Rows and columns inside a block\n"
        "are periods: 5, 10, 20, 50, 100.",
        ha="left",
        va="top",
        fontsize=10.0,
        linespacing=1.45,
        color=TEXT,
    )
    ax.text(
        0.03,
        0.43,
        "Color is symmetric overlap:\n"
        r"$\mathrm{tr}(P_1P_2) / \min(r_1,r_2)$.",
        ha="left",
        va="top",
        fontsize=10.0,
        linespacing=1.45,
        color=TEXT,
    )
    ax.text(
        0.03,
        0.19,
        "Layer labels sit at block centers\n"
        "so the structure remains readable.",
        ha="left",
        va="top",
        fontsize=10.0,
        linespacing=1.45,
        color=TEXT,
    )


def plot_fourier_fourier_grid():
    style_matplotlib()
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "paper_fourier_overlap",
        ["#f6f2ef", "#ead3c8", "#d8a68f", "#b97068", "#7e546f", "#435b79"],
    )
    matrices = []
    max_value = 0.0
    for folder, filename, variable_prefix, title in FOURIER_FOURIER_SPECS:
        rows = load_jsonl(BANK_ROOT / "fourier_to_fourier" / folder / filename)
        matrix = matrix_from_fourier_rows(rows, variable_prefix)
        matrices.append((matrix, title))
        off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        max_value = max(max_value, float(np.nanpercentile(off_diag, 99)))
    vmax = max(0.18, min(0.65, max_value))

    fig = plt.figure(figsize=(11.2, 6.35), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    grid = fig.add_gridspec(2, 6)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]

    image = None
    for ax, (matrix, title) in zip(axes, matrices):
        image = draw_layer_period_matrix(ax, matrix, title, cmap, vmax)

    fig.subplots_adjust(left=0.055, right=0.88, top=0.93, bottom=0.08, wspace=0.36, hspace=0.34)
    if image is not None:
        cax = fig.add_axes([0.91, 0.20, 0.016, 0.61])
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.outline.set_edgecolor(SPINE)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label("Symmetric overlap", fontsize=9.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        OUTPUT_DIR / "fourier_fourier_layers40_46_symmetric_overlap_grid.png",
        OUTPUT_DIR / "fourier_fourier_layers40_46_symmetric_overlap_grid.pdf",
    ]
    fig.savefig(outputs[0], dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(outputs[1], facecolor=fig.get_facecolor())
    plt.close(fig)
    return outputs


def main():
    outputs = []
    outputs.extend(plot_das_fourier_lines())
    outputs.extend(plot_addition_same_layer_das_fourier())
    outputs.extend(plot_add_sub_fourier_period_matrix())
    outputs.extend(plot_fourier_fourier_grid())
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
