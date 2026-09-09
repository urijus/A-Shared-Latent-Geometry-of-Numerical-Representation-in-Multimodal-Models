import json
import os
import pathlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "visualizations" / "01_arithmetic_reference" / "reference_outputs"
BANK_ROOT = REPO_ROOT / "src" / "banks" / "baseline_bank" / "gemma4_12b_it" / "digits"
REFERENCE_MAP_PATH = REPO_ROOT / "visualizations" / "main_paper" / "DAS_audit" / "das_text_reference_map.jsonl"

FOURIER_PROBE_ROOT = BANK_ROOT / "fourier_probes"
FOURIER_CROSS_PATH = (
    BANK_ROOT
    / "subspace_overlap"
    / "fourier_to_fourier"
    / "result_c1_c0_modalities_layer43_periods_2_5_10_20_50_100"
    / "result_c1_c0_add_sub_mul_layer43_periods_2_5_10_20_50_100_pos17_ridge.jsonl"
)
FOURIER_LAYER_PATHS = {
    "add_result": (
        BANK_ROOT
        / "subspace_overlap_layers_32_48"
        / "fourier_to_fourier"
        / "addition_result_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100"
        / "addition_result_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100_pos17_ridge.jsonl"
    ),
    "sub_result": (
        BANK_ROOT
        / "subspace_overlap_layers_32_48"
        / "fourier_to_fourier"
        / "subtraction_result_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100"
        / "subtraction_result_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100_pos17_ridge.jsonl"
    ),
    "mul_result": (
        BANK_ROOT
        / "subspace_overlap_layers_32_48"
        / "fourier_to_fourier"
        / "multiplication_result_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100"
        / "multiplication_result_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100_pos17_ridge.jsonl"
    ),
    "mul_c1_hat": (
        BANK_ROOT
        / "subspace_overlap_layers_32_48"
        / "fourier_to_fourier"
        / "multiplication_c1_hat_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100"
        / "multiplication_c1_hat_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100_pos17_ridge.jsonl"
    ),
    "mul_c0_hat": (
        BANK_ROOT
        / "subspace_overlap_layers_32_48"
        / "fourier_to_fourier"
        / "multiplication_c0_hat_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100"
        / "multiplication_c0_hat_layers_32_33_34_35_36_37_38_39_40_41_42_43_44_45_46_47_48_periods_5_10_20_50_100_pos17_ridge.jsonl"
    ),
}

BACKGROUND = "#fdfdfd"
AX_FACE = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"


def load_jsonl(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    with open(open_path, "r", encoding="utf-8") as handle:
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
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 16,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def muted_tab10(amount=0.88):
    base = plt.get_cmap("tab10").colors
    gray = mcolors.to_rgb(BACKGROUND)
    return [
        tuple(
            amount * channel + (1.0 - amount) * gray_channel
            for channel, gray_channel in zip(mcolors.to_rgb(color), gray)
        )
        for color in base
    ]


def setup_axis(ax):
    ax.set_facecolor(AX_FACE)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(length=3.5, width=0.8)


def probe_path(modality, target, period, layer, position=17, method="ridge"):
    return (
        FOURIER_PROBE_ROOT
        / modality
        / "fourier_projections"
        / f"{target}_pos{position}_{method}"
        / "probes"
        / f"{target}_{modality}"
        / f"{target}_T{period}_layer{layer}_pos{position}_{method}_probe.pt"
    )


def orthonormal_columns(matrix, d_model):
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    if matrix.shape[0] == d_model:
        columns = matrix
    elif matrix.shape[1] == d_model:
        columns = matrix.T
    else:
        raise ValueError(f"Cannot infer d_model={d_model} axis from {tuple(matrix.shape)}")
    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    tolerance = max(columns.shape) * torch.finfo(columns.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    return left[:, :rank]


def subspace_overlap(first_basis, second_basis):
    first_basis = torch.as_tensor(first_basis).detach().float().squeeze()
    second_basis = torch.as_tensor(second_basis).detach().float().squeeze()
    d_model = max(first_basis.shape)
    first = orthonormal_columns(first_basis, d_model)
    second = orthonormal_columns(second_basis, d_model)
    singular_values = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    return float(singular_values.square().sum() / min(first.shape[1], second.shape[1]))


def load_fourier_weight(modality, target, period, layer):
    path = probe_path(modality, target, period, layer)
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = torch_load_portable(path)
    return torch.as_tensor(artifact["weight"]).detach().float(), path


def plot_fourier_layer_lines(ax):
    setup_axis(ax)
    layers = list(range(32, 49))
    periods = [5, 10, 20, 50, 100]
    colors = muted_tab10()
    series = [
        ("add_result", "addition_result", "Add result", colors[0]),
        ("sub_result", "subtraction_result", "Sub result", colors[1]),
        ("mul_result", "multiplication_result", "Mul result", colors[2]),
        ("mul_c1_hat", "multiplication_c1_hat", r"Mul $\hat{c}_1$", colors[3]),
        ("mul_c0_hat", "multiplication_c0_hat", r"Mul $\hat{c}_0$", colors[4]),
    ]

    ax.axvspan(42, 44, color="#d7dde0", alpha=0.95, zorder=0)
    ax.text(
        43,
        0.292,
        "42-44",
        ha="center",
        va="top",
        fontsize=8,
        color="#5b6264",
    )

    for key, row_label_prefix, label, color in series:
        rows = load_jsonl(FOURIER_LAYER_PATHS[key])
        lookup = {(row["first_label"], row["second_label"]): row for row in rows}
        values = []
        for layer in layers:
            overlaps = []
            for period in periods:
                first = f"{row_label_prefix}_L{layer}_T{period}"
                for other_layer in layers:
                    if other_layer == layer:
                        continue
                    second = f"{row_label_prefix}_L{other_layer}_T{period}"
                    overlaps.append(lookup[(first, second)]["symmetric_overlap"])
            values.append(float(np.mean(overlaps)))
        ax.plot(
            layers,
            values,
            marker="o",
            markersize=4.2,
            linewidth=1.9,
            color=color,
            label=label,
        )

    ax.set_title("Fourier layer centrality", pad=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean same-period overlap")
    ax.set_xticks(list(range(32, 49, 2)))
    ax.set_ylim(0.0, 0.30)
    ax.grid(axis="y", color="#cbd2d4", linewidth=0.7, alpha=0.55)
    ax.legend(frameon=False, ncol=1, loc="upper left", handlelength=1.4)


def matrix_from_rows(rows, labels, metric):
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    return np.array(
        [[lookup[(first, second)][metric] for second in labels] for first in labels],
        dtype=float,
    )


def draw_heatmap(
    ax,
    matrix,
    labels,
    title,
    cmap,
    vmax,
    colorbar_label,
    annotate_threshold=0.035,
    annotation_fontsize=7.5,
    xtick_rotation=35,
    mask_diagonal=False,
):
    setup_axis(ax)
    plot_matrix = matrix.copy()
    if mask_diagonal:
        np.fill_diagonal(plot_matrix, np.nan)

    cmap_obj = cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d8dee0")
    image = ax.imshow(plot_matrix, vmin=0, vmax=vmax, cmap=cmap_obj)
    ax.set_title(title, pad=9)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=xtick_rotation, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="both", length=0)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            value = matrix[row_index, col_index]
            if mask_diagonal and row_index == col_index:
                text = "1"
                color = "#5e6668"
            elif value >= annotate_threshold:
                text = "1" if value >= 0.995 else f"{value:.2f}"
                color = "white" if value > vmax * 0.52 else "#181818"
            else:
                continue
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=annotation_fontsize,
                color=color,
            )

    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.76, pad=0.025)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(colorbar_label, fontsize=9)
    return image


def plot_das_matrix(ax):
    rows = load_jsonl(REFERENCE_MAP_PATH)
    labels = [
        "text/addition/result",
        "text/subtraction/result",
        "text/multiplication/result",
        "text/multiplication/c1_hat_full",
        "text/multiplication/c0_hat",
    ]
    display = [
        "Add\nresult",
        "Sub\nresult",
        "Mul\nresult",
        r"Mul $\hat{c}_1$",
        r"Mul $\hat{c}_0$",
    ]
    lookup = {(row["first"], row["second"]): row for row in rows}
    matrix = np.array(
        [[lookup[(first, second)]["symmetric_overlap_mean"] for second in labels] for first in labels],
        dtype=float,
    )
    std_matrix = np.array(
        [[lookup[(first, second)]["symmetric_overlap_std"] for second in labels] for first in labels],
        dtype=float,
    )
    das_cmap = mcolors.LinearSegmentedColormap.from_list(
        "muted_das_overlap",
        ["#d8dee0", "#c7d7d4", "#96c8af", "#59aa84", "#2f7f75"],
    )
    setup_axis(ax)
    image = ax.imshow(matrix, vmin=0, vmax=0.8, cmap=das_cmap)
    ax.set_title("DAS reference map, L43, k=22", pad=9)
    ax.set_xticks(range(len(display)), labels=display, rotation=35, ha="right")
    ax.set_yticks(range(len(display)), labels=display)
    ax.tick_params(axis="both", length=0)
    for row_index in range(len(display)):
        for col_index in range(len(display)):
            value = matrix[row_index, col_index]
            color = "white" if value > 0.8 * 0.52 else "#181818"
            ax.text(
                col_index,
                row_index - 0.09,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
            )
            ax.text(
                col_index,
                row_index + 0.15,
                fr"$\pm${std_matrix[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=color,
            )
    colorbar = ax.figure.colorbar(image, ax=ax, shrink=0.76, pad=0.025)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Symmetric overlap", fontsize=9)
    return image


def plot_fourier_cross_matrix(ax):
    rows = load_jsonl(FOURIER_CROSS_PATH)
    periods = [2, 5, 10, 20, 50, 100]
    groups = [
        ("mul_c1_hat", r"$\hat{c}_1$"),
        ("mul_c0_hat", r"$\hat{c}_0$"),
        ("mul_result", "Mul"),
    ]
    labels = [f"{prefix}_T{period}" for prefix, _ in groups for period in periods]
    display = [f"T{period}" for _ in groups for period in periods]
    matrix = matrix_from_rows(rows, labels, "symmetric_overlap")
    draw_heatmap(
        ax,
        matrix,
        display,
        "Multiplication Fourier overlap, L43",
        cmap="magma",
        vmax=1.0,
        colorbar_label="Symmetric overlap",
        annotate_threshold=0.50,
        annotation_fontsize=5.6,
        xtick_rotation=45,
        mask_diagonal=False,
    )

    group_centers = [2.5, 8.5, 14.5]
    group_names = [short for _, short in groups]
    for center, name in zip(group_centers, group_names):
        ax.text(
            center,
            len(labels) + 1.55,
            name,
            ha="center",
            va="top",
            fontsize=8.5,
            color=TEXT,
            clip_on=False,
        )
        ax.text(
            -2.65,
            center,
            name,
            ha="center",
            va="center",
            rotation=90,
            fontsize=8.5,
            color=TEXT,
            clip_on=False,
        )

    for separator in [5.5, 11.5]:
        ax.axhline(separator, color="#f3f0e8", linewidth=0.9, alpha=0.72)
        ax.axvline(separator, color="#f3f0e8", linewidth=0.9, alpha=0.72)

    highlights = [
        ("mul_c1_hat_T2", "mul_result_T20"),
        ("mul_c1_hat_T5", "mul_result_T50"),
        ("mul_c1_hat_T10", "mul_result_T100"),
        ("mul_result_T20", "mul_c1_hat_T2"),
        ("mul_result_T50", "mul_c1_hat_T5"),
        ("mul_result_T100", "mul_c1_hat_T10"),
        ("mul_result_T2", "mul_c0_hat_T2"),
        ("mul_result_T5", "mul_c0_hat_T5"),
        ("mul_result_T10", "mul_c0_hat_T10"),
        ("mul_c0_hat_T2", "mul_result_T2"),
        ("mul_c0_hat_T5", "mul_result_T5"),
        ("mul_c0_hat_T10", "mul_result_T10"),
    ]
    highlight_color = "#b45a55"
    n_labels = len(labels)
    for first, second in highlights:
        row = labels.index(first)
        col = labels.index(second)
        value = matrix[row, col]
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor=highlight_color,
                linewidth=1.35,
                zorder=5,
            )
        )
        ax.plot(
            [col, col],
            [row + 0.5, n_labels - 0.5],
            color=highlight_color,
            linewidth=0.9,
            alpha=0.48,
            zorder=4,
        )
        ax.plot(
            [-0.5, col - 0.5],
            [row, row],
            color=highlight_color,
            linewidth=0.9,
            alpha=0.48,
            zorder=4,
        )
        ax.text(
            col,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=5.8,
            color="white",
            zorder=6,
        )


def make_figure():
    style_matplotlib()
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12.8, 4.25),
        gridspec_kw={"width_ratios": [1.04, 0.035, 1.04, 1.16]},
    )
    fig.patch.set_facecolor(BACKGROUND)

    plot_fourier_layer_lines(axes[0])
    axes[1].axis("off")
    plot_das_matrix(axes[2])
    plot_fourier_cross_matrix(axes[3])

    fig.subplots_adjust(left=0.075, right=0.975, top=0.90, bottom=0.255, wspace=0.10)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "overlap_summary_three_panel.png"
    pdf_path = OUTPUT_DIR / "overlap_summary_three_panel.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main():
    for output_path in make_figure():
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
