import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "activation_patching"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "visualizations"
    / "appendix"
    / "baseline"
    / "activation_patching"
)

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"

OPERATIONS = ["addition", "subtraction", "multiplication"]
HOOKS = ["resid_post", "attn_out", "mlp_input"]

OPERATION_LABELS = {
    "addition": "Addition",
    "subtraction": "Subtraction",
    "multiplication": "Multiplication",
}

HOOK_LABELS = {
    "resid_post": "post-block residual",
    "attn_out": "attention output",
    "mlp_input": "MLP input",
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
            "axes.labelsize": 10,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.5,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def recovery_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "activation_recovery",
        ["#f6f2f0", "#ead8ce", "#ddb09d", "#c97f6d", "#8f4f59"],
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.75)


def operation_rows(operation):
    path = RESULT_ROOT / operation / f"{operation}_activation_patching.jsonl"
    return load_jsonl(path)


def recovery_matrix(rows, hook):
    hook_rows = [
        row
        for row in rows
        if row["hook"] == hook
        and row["layer_mode"] == "individual"
        and isinstance(row["layer"], int)
    ]
    layers = sorted({int(row["layer"]) for row in hook_rows})
    positions = sorted({int(row["position"]) for row in hook_rows})
    values = {
        (layer, position): []
        for layer in layers
        for position in positions
    }
    for row in hook_rows:
        values[(int(row["layer"]), int(row["position"]))].append(float(row["recovery"]))

    matrix = np.full((len(positions), len(layers)), np.nan, dtype=float)
    for row_index, position in enumerate(positions):
        for col_index, layer in enumerate(layers):
            cell_values = values[(layer, position)]
            if cell_values:
                matrix[row_index, col_index] = np.mean(cell_values)
    return matrix, layers, positions


def plot_activation_patching():
    rows_by_operation = {operation: operation_rows(operation) for operation in OPERATIONS}
    matrices = {}
    all_values = []
    for operation in OPERATIONS:
        for hook in HOOKS:
            matrix, layers, positions = recovery_matrix(rows_by_operation[operation], hook)
            matrices[(operation, hook)] = (matrix, layers, positions)
            all_values.extend(matrix[np.isfinite(matrix)].tolist())

    vmax = max(0.01, float(np.nanpercentile(all_values, 99)))
    vmin = min(0.0, float(np.nanpercentile(all_values, 1)))

    style_matplotlib()
    fig, axes = plt.subplots(
        len(OPERATIONS),
        len(HOOKS),
        figsize=(11.2, 7.4),
        constrained_layout=False,
    )
    fig.patch.set_facecolor(BACKGROUND)

    image = None
    for row_index, operation in enumerate(OPERATIONS):
        for col_index, hook in enumerate(HOOKS):
            ax = axes[row_index, col_index]
            setup_axis(ax)
            matrix, layers, positions = matrices[(operation, hook)]
            image = ax.imshow(
                matrix,
                cmap=recovery_cmap(),
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                interpolation="nearest",
            )

            xtick_positions = [
                index
                for index, layer in enumerate(layers)
                if layer == layers[0] or layer == layers[-1] or layer % 4 == 0
            ]
            ax.set_xticks(xtick_positions, [str(layers[index]) for index in xtick_positions])
            ax.set_yticks(range(len(positions)), [str(position) for position in positions])
            ax.set_xticks(np.arange(-0.5, len(layers), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(positions), 1), minor=True)
            ax.grid(which="minor", color=BACKGROUND, linewidth=0.65)
            ax.tick_params(which="minor", length=0)

            if col_index == 0:
                ax.set_ylabel("Position", labelpad=8)
                ax.text(
                    -0.34,
                    0.5,
                    OPERATION_LABELS[operation],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=12,
                )
            else:
                ax.set_yticklabels([])

            if row_index != len(OPERATIONS) - 1:
                ax.set_xticklabels([])

    panel_center_x = 0.48
    fig.text(panel_center_x, 0.982, "Activation patching recovery", ha="center", va="top", fontsize=15)
    fig.subplots_adjust(left=0.13, right=0.90, top=0.92, bottom=0.105, wspace=0.12, hspace=0.16)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.025)
        colorbar.outline.set_edgecolor(SPINE)
        colorbar.outline.set_linewidth(0.7)
        colorbar.set_label("Recovery", fontsize=9.5)

    for col_index, hook in enumerate(HOOKS):
        box = axes[-1, col_index].get_position()
        fig.text(
            (box.x0 + box.x1) / 2,
            0.047,
            HOOK_LABELS[hook],
            ha="center",
            va="center",
            fontsize=11,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "activation_patching_recovery_resid_post_attn_out_mlp_input.png"
    pdf_path = OUTPUT_DIR / "activation_patching_recovery_resid_post_attn_out_mlp_input.pdf"
    legacy_png_path = OUTPUT_DIR / "activation_patching_recovery_resid_post_mlp_input.png"
    legacy_pdf_path = OUTPUT_DIR / "activation_patching_recovery_resid_post_mlp_input.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    fig.savefig(legacy_png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(legacy_pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path, legacy_png_path, legacy_pdf_path]


def main():
    for path in plot_activation_patching():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
