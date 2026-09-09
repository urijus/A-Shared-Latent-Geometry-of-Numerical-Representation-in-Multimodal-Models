import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes"
    / "cross_linear_probe_results_balanced_multiplication.jsonl"
)
EXPANDED_RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes_expanded"
    / "result_100_900_n50.jsonl"
)
EXPANDED_MOD10_PATH = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes_expanded"
    / "result_mod_10_n1000.jsonl"
)
EXPANDED_MOD100_PATH = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes_expanded"
    / "result_mod_100_n400.jsonl"
)
IMAGE_EXPANDED_RESULT_PATH = (
    REPO_ROOT
    / "results"
    / "baseline_images"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes_expanded"
    / "result_100_900_n5.jsonl"
)
IMAGE_EXPANDED_MOD10_PATH = (
    REPO_ROOT
    / "results"
    / "baseline_images"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes_expanded"
    / "result_mod_10_n100.jsonl"
)
IMAGE_EXPANDED_MOD100_PATH = (
    REPO_ROOT
    / "results"
    / "baseline_images"
    / "gemma4_12b_it"
    / "digits"
    / "cross_linear_probes_expanded"
    / "result_mod_100_n25.jsonl"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "visualizations"
    / "appendix"
    / "baseline"
    / "appendix_cross_format_probes"
)
IMAGE_OUTPUT_DIR = (
    REPO_ROOT
    / "visualizations"
    / "appendix"
    / "baseline_images"
    / "appendix_cross_format_probes"
)

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"

TARGET_LABELS = {
    "result": "Result",
    "result_mod_10": "Result mod 10",
    "result_mod_100": "Result mod 100",
}

MODALITY_LABELS = {
    "addition": "Add",
    "subtraction": "Sub",
    "multiplication": "Mul",
}


def load_jsonl(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    with open(open_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.titlesize": 13,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


TARGET_CMAPS = {
    "result": ["#f6f2f0", "#ead8ce", "#ddb09d", "#c97f6d", "#8f4f59"],
    "result_mod_10": ["#f3f6f5", "#d9e7e2", "#adcfc6", "#6fa99b", "#3f7f75"],
    "result_mod_100": ["#f5f3f7", "#e1dae8", "#beb0cf", "#927aa8", "#654f7a"],
}


def appendix_cmap(target):
    return mcolors.LinearSegmentedColormap.from_list(
        f"cross_format_{target}",
        TARGET_CMAPS.get(target, TARGET_CMAPS["result"]),
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)


def target_label(target):
    return TARGET_LABELS.get(target, target.replace("_", " "))


def modality_order(rows):
    preferred = ["addition", "subtraction", "multiplication"]
    present = {row["train_modality"] for row in rows} | {row["test_modality"] for row in rows}
    return [modality for modality in preferred if modality in present]


def matrix_for_layer(rows, layer, train_modalities, test_modalities):
    matrix = np.full((len(test_modalities), len(train_modalities)), np.nan, dtype=float)
    train_index = {modality: index for index, modality in enumerate(train_modalities)}
    test_index = {modality: index for index, modality in enumerate(test_modalities)}
    for row in rows:
        if int(row["layer"]) != int(layer):
            continue
        matrix[test_index[row["test_modality"]], train_index[row["train_modality"]]] = row[
            "normalized_gain"
        ]
    return matrix


def metric_matrix_for_layer(rows, layer, train_modalities, test_modalities, metric):
    matrix = np.full((len(test_modalities), len(train_modalities)), np.nan, dtype=float)
    train_index = {modality: index for index, modality in enumerate(train_modalities)}
    test_index = {modality: index for index, modality in enumerate(test_modalities)}
    for row in rows:
        if int(row["layer"]) != int(layer):
            continue
        if row["train_modality"] not in train_index or row["test_modality"] not in test_index:
            continue
        matrix[test_index[row["test_modality"]], train_index[row["train_modality"]]] = row[
            metric
        ]
    return matrix


def annotate_matrix(ax, matrix):
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                continue
            color = "#fdfdfd" if value > 0.72 else TEXT
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=color,
            )


def draw_panel(ax, rows, layer, train_modalities, test_modalities, metric, title, cmap):
    setup_axis(ax)
    matrix = metric_matrix_for_layer(rows, layer, train_modalities, test_modalities, metric)
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title, pad=8, fontweight="normal")
    ax.set_xticks(
        range(len(train_modalities)),
        [MODALITY_LABELS.get(modality, modality.title()) for modality in train_modalities],
        rotation=28,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(
        range(len(test_modalities)),
        [MODALITY_LABELS.get(modality, modality.title()) for modality in test_modalities],
    )
    ax.set_xticks(np.arange(-0.5, len(train_modalities), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(test_modalities), 1), minor=True)
    ax.grid(which="minor", color=BACKGROUND, linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    annotate_matrix(ax, matrix)
    return image


def rows_for(rows, target, modalities):
    return [
        row
        for row in rows
        if row["target"] == target
        and row["train_modality"] in modalities
        and row["test_modality"] in modalities
    ]


def plot_add_sub_exact_summary(
    output_dir=OUTPUT_DIR,
    old_result_path=RESULT_PATH,
    expanded_result_path=EXPANDED_RESULT_PATH,
    output_prefix="cross_format",
    figure_title="Cross-operation probes: exact result",
):
    old_rows = load_jsonl(old_result_path)
    expanded_rows = load_jsonl(expanded_result_path)
    layer = 48
    modalities = ["addition", "subtraction"]

    panels = [
        (
            rows_for(old_rows, "result", modalities),
            "accuracy",
            "Controlled result\n0-99, top-1",
        ),
        (
            rows_for(expanded_rows, "result", modalities),
            "accuracy",
            "Expanded result\n100-900, top-1",
        ),
        (
            rows_for(expanded_rows, "result", modalities),
            "top_5_accuracy",
            "Expanded result\n100-900, top-5",
        ),
    ]

    style_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.55), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)

    image = None
    for ax, (panel_rows, metric, title) in zip(axes, panels):
        image = draw_panel(
            ax,
            panel_rows,
            layer,
            modalities,
            modalities,
            metric,
            title,
            appendix_cmap("result"),
        )

    axes[0].set_ylabel("Test operation", labelpad=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    panel_center_x = (0.095 + 0.86) / 2
    fig.text(
        panel_center_x,
        0.975,
        figure_title,
        ha="center",
        va="top",
        fontsize=15,
    )
    fig.text(panel_center_x, 0.075, "Train operation", ha="center", va="center", fontsize=11.5)
    fig.subplots_adjust(left=0.095, right=0.86, top=0.70, bottom=0.27, wspace=0.35)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.025)
        colorbar.outline.set_edgecolor(SPINE)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label("Accuracy", fontsize=9.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_prefix}_exact_result_add_sub_summary.png"
    pdf_path = output_dir / f"{output_prefix}_exact_result_add_sub_summary.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def plot_modular_summary(
    output_dir=OUTPUT_DIR,
    mod10_path=EXPANDED_MOD10_PATH,
    mod100_path=EXPANDED_MOD100_PATH,
    output_prefix="cross_format",
    figure_title="Cross-operation probes: modular targets",
):
    mod10_rows = load_jsonl(mod10_path)
    mod100_rows = load_jsonl(mod100_path)
    layer = 48
    modalities = ["addition", "subtraction", "multiplication"]

    panels = [
        (
            rows_for(mod10_rows, "result_mod_10", modalities),
            "accuracy",
            "Expanded mod 10\ntop-1",
            "result_mod_10",
        ),
        (
            rows_for(mod100_rows, "result_mod_100", modalities),
            "accuracy",
            "Expanded mod 100\ntop-1",
            "result_mod_100",
        ),
        (
            rows_for(mod100_rows, "result_mod_100", modalities),
            "top_5_accuracy",
            "Expanded mod 100\ntop-5",
            "result_mod_100",
        ),
    ]

    style_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.25), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)

    image = None
    for ax, (panel_rows, metric, title, cmap_target) in zip(axes, panels):
        image = draw_panel(
            ax,
            panel_rows,
            layer,
            modalities,
            modalities,
            metric,
            title,
            appendix_cmap(cmap_target),
        )

    axes[0].set_ylabel("Test operation", labelpad=8)
    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.text(0.5, 0.982, figure_title, ha="center", va="top", fontsize=15)
    fig.text(0.49, 0.055, "Train operation", ha="center", va="center", fontsize=11.5)
    fig.subplots_adjust(left=0.09, right=0.89, top=0.78, bottom=0.24, wspace=0.32)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.025)
        colorbar.outline.set_edgecolor(SPINE)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label("Accuracy", fontsize=9.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_prefix}_modular_expanded_summary.png"
    pdf_path = output_dir / f"{output_prefix}_modular_expanded_summary.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def plot_target(rows, target):
    target_rows = [row for row in rows if row["target"] == target]
    layers = sorted({int(row["layer"]) for row in target_rows})
    train_modalities = modality_order(target_rows)
    test_modalities = modality_order(target_rows)

    style_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(8.7, 6.0), constrained_layout=False)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor(BACKGROUND)

    image = None
    for ax, layer in zip(axes, layers):
        setup_axis(ax)
        matrix = matrix_for_layer(target_rows, layer, train_modalities, test_modalities)
        image = ax.imshow(matrix, cmap=appendix_cmap(target), vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"Layer {layer}", pad=7, fontweight="normal")
        ax.set_xticks(
            range(len(train_modalities)),
            [MODALITY_LABELS.get(modality, modality.title()) for modality in train_modalities],
            rotation=28,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(
            range(len(test_modalities)),
            [MODALITY_LABELS.get(modality, modality.title()) for modality in test_modalities],
        )
        ax.set_xticks(np.arange(-0.5, len(train_modalities), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(test_modalities), 1), minor=True)
        ax.grid(which="minor", color=BACKGROUND, linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        annotate_matrix(ax, matrix)

    for ax in axes[len(layers) :]:
        ax.axis("off")

    panel_center_x = (0.11 + 0.875) / 2
    fig.text(
        panel_center_x,
        0.982,
        f"Cross-operation linear probe: {target_label(target)}",
        ha="center",
        va="top",
        fontsize=15,
    )
    fig.text(panel_center_x, 0.05, "Train operation", ha="center", va="center", fontsize=11.5)
    fig.text(0.036, 0.52, "Test operation", ha="center", va="center", rotation=90, fontsize=11.5)
    fig.subplots_adjust(left=0.11, right=0.875, top=0.89, bottom=0.15, wspace=0.32, hspace=0.55)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes[: len(layers)], shrink=0.78, pad=0.025)
        colorbar.outline.set_edgecolor(SPINE)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label("Normalized gain", fontsize=9.5)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_stem = f"cross_format_{target}_normalized_gain"
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def main():
    rows = load_jsonl(RESULT_PATH)
    outputs = []
    for target in ["result", "result_mod_10", "result_mod_100"]:
        outputs += plot_target(rows, target)
    outputs += plot_add_sub_exact_summary()
    outputs += plot_modular_summary()
    outputs += plot_add_sub_exact_summary(
        output_dir=IMAGE_OUTPUT_DIR,
        expanded_result_path=IMAGE_EXPANDED_RESULT_PATH,
        output_prefix="cross_format_images",
        figure_title="Image cross-operation probes: exact result",
    )
    outputs += plot_modular_summary(
        output_dir=IMAGE_OUTPUT_DIR,
        mod10_path=IMAGE_EXPANDED_MOD10_PATH,
        mod100_path=IMAGE_EXPANDED_MOD100_PATH,
        output_prefix="cross_format_images",
        figure_title="Image cross-operation probes: modular targets",
    )
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
