import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "results" / "baseline" / "gemma4_12b_it" / "digits"
OUTPUT_ROOT = REPO_ROOT / "visualizations" / "appendix" / "baseline"

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"
PERIOD_COLORS = {
    2: "#8f4f59",
    5: "#a65f55",
    10: "#c97f6d",
    20: "#d6a85f",
    50: "#a77968",
    100: "#795f72",
    891: "#8f4f59",
    997: "#8f4f59",
}


TARGET_LABELS = {
    "result": "Result",
    "result_mod_10": "Result mod 10",
    "result_mod_50": "Result mod 50",
    "result_mod_100": "Result mod 100",
    "c1_hat": r"$\hat{c}_1$",
    "c0_hat": r"$\hat{c}_0$",
    "r0": r"$r_0$",
}

MODALITY_LABELS = {
    "addition": "Addition",
    "subtraction": "Subtraction",
    "multiplication": "Multiplication",
}


def load_jsonl(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    with open(open_path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


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
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8.5,
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


def blend_with_background(color, amount):
    rgb = np.array(mcolors.to_rgb(color))
    bg = np.array(mcolors.to_rgb(BACKGROUND))
    return tuple(amount * rgb + (1.0 - amount) * bg)


def period_cmap(period):
    color = PERIOD_COLORS.get(int(period), "#8f4f59")
    return mcolors.LinearSegmentedColormap.from_list(
        f"appendix_fourier_T{period}",
        [
            blend_with_background(color, 0.05),
            blend_with_background(color, 0.22),
            blend_with_background(color, 0.52),
            blend_with_background(color, 0.78),
            color,
        ],
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(length=0)


def label_target(target):
    return TARGET_LABELS.get(target, target.replace("_", " "))


def label_modality(modality):
    return MODALITY_LABELS.get(modality, modality.replace("_", " ").title())


def filter_rows(rows, modality=None, target=None, period=None):
    out = rows
    if modality is not None:
        out = [row for row in out if row.get("modality") == modality]
    if target is not None:
        out = [row for row in out if row.get("target") == target]
    if period is not None:
        out = [row for row in out if int(row.get("period", -1)) == int(period)]
    return out


def rows_to_matrix(rows, metric):
    if not rows:
        raise ValueError("Cannot build a matrix from zero rows.")
    layers = sorted({int(row["layer"]) for row in rows})
    positions = sorted({row["position_name"] for row in rows}, key=lambda value: int(value))
    matrix = np.full((len(positions), len(layers)), np.nan, dtype=float)
    layer_index = {layer: index for index, layer in enumerate(layers)}
    position_index = {position: index for index, position in enumerate(positions)}
    for row in rows:
        matrix[position_index[row["position_name"]], layer_index[int(row["layer"])]] = row[metric]
    return matrix, layers, positions


def plot_heatmap(
    rows,
    metric,
    title,
    output_stem,
    output_dir,
    colorbar_label,
    vmin=0.0,
    vmax=1.0,
    cmap=None,
):
    if not rows:
        return []
    matrix, layers, positions = rows_to_matrix(rows, metric)
    matrix = np.clip(matrix, vmin, vmax)

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap if cmap is not None else appendix_cmap(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, pad=8, fontweight="normal")
    ax.set_xlabel("Layer", labelpad=6)
    ax.set_ylabel("Token position", labelpad=6)

    xtick_positions = [index for index, layer in enumerate(layers) if layer == 1 or layer % 4 == 0]
    ax.set_xticks(xtick_positions, [str(layers[index]) for index in xtick_positions])
    ax.set_yticks(range(len(positions)), positions)

    for boundary in np.arange(0.5, len(positions), 1):
        ax.axhline(boundary, color=BACKGROUND, linewidth=0.55, alpha=0.7)

    colorbar = fig.colorbar(image, ax=ax, shrink=0.80, pad=0.018)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label(colorbar_label, fontsize=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def plot_multiplication_result_summary(seed_rows_list, output_dir):
    by_seed = []
    for seed_rows in seed_rows_list:
        rows = filter_rows(seed_rows, modality="multiplication", target="result")
        rows = [row for row in rows if row["position_name"] == "17"]
        by_seed.append({int(row["layer"]): row for row in rows})
    layers = sorted(set.intersection(*(set(seed.keys()) for seed in by_seed)))

    metrics = [
        ("top_1_accuracy", "Top 1", "#8f4f59"),
        ("top_2_accuracy", "Top 2", "#5f9b8f"),
        ("top_5_accuracy", "Top 5", "#c97f6d"),
        ("top_10_accuracy", "Top 10", "#d6a85f"),
        ("top_50_accuracy", "Top 50", "#60798f"),
    ]

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.65)
    ax.grid(axis="x", visible=False)

    x = np.array(layers, dtype=float)
    for metric, label, color in metrics:
        values = np.array(
            [[seed[layer][metric] for layer in layers] for seed in by_seed],
            dtype=float,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        ax.fill_between(
            x,
            np.clip(mean - std, 0.0, 1.0),
            np.clip(mean + std, 0.0, 1.0),
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(x, mean, linewidth=2.0, color=color, label=label)

    ax.set_title("Multiplication result top-k accuracy", pad=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(list(range(4, 49, 4)))
    ax.legend(frameon=False, ncol=3, loc="lower right", handlelength=1.8, columnspacing=0.9)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "linear_multiplication_result_topk_summary.png"
    pdf_path = output_dir / "linear_multiplication_result_topk_summary.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def save_linear_appendix():
    output_dir = OUTPUT_ROOT / "appendix_linear_probes"
    outputs = []

    add_sub = load_jsonl(RESULT_ROOT / "linear_probes" / "add_sub_probe_results.jsonl")
    add_sub_shuffled = load_jsonl(RESULT_ROOT / "linear_probes" / "add_sub_probe_results_shuffled.jsonl")
    for modality in ["addition", "subtraction"]:
        for target in ["result", "result_mod_10", "result_mod_100"]:
            rows = filter_rows(add_sub, modality=modality, target=target)
            outputs += plot_heatmap(
                rows,
                metric="normalized_gain",
                title=f"{label_modality(modality)} linear probe: {label_target(target)}",
                output_stem=f"linear_{modality}_{target}",
                output_dir=output_dir,
                colorbar_label="Normalized gain",
            )
        rows = filter_rows(add_sub_shuffled, modality=modality, target="result")
        outputs += plot_heatmap(
            rows,
            metric="normalized_gain",
            title=f"{label_modality(modality)} shuffled linear probe: Result",
            output_stem=f"linear_{modality}_result_shuffled",
            output_dir=output_dir,
            colorbar_label="Normalized gain",
        )

    mul = load_jsonl(RESULT_ROOT / "linear_probes" / "mul_probe_results.jsonl")
    mul_shuffled = load_jsonl(RESULT_ROOT / "linear_probes" / "mul_probe_results_shuffled.jsonl")
    for target in ["c1_hat", "result_mod_100", "c0_hat"]:
        rows = filter_rows(mul, modality="multiplication", target=target)
        outputs += plot_heatmap(
            rows,
            metric="normalized_gain",
            title=f"Multiplication linear probe: {label_target(target)}",
            output_stem=f"linear_multiplication_{target}",
            output_dir=output_dir,
            colorbar_label="Normalized gain",
        )
    rows = filter_rows(mul_shuffled, modality="multiplication", target="c1_hat")
    outputs += plot_heatmap(
        rows,
        metric="normalized_gain",
        title=r"Multiplication shuffled linear probe: $\hat{c}_1$",
        output_stem="linear_multiplication_c1_hat_shuffled",
        output_dir=output_dir,
        colorbar_label="Normalized gain",
    )

    seed_dir = RESULT_ROOT / "linear_probes" / "multiplication_result_test"
    seed_paths = sorted(seed_dir.glob("result_seed*.jsonl"))
    if seed_paths:
        seed0 = load_jsonl(seed_dir / "result_seed0.jsonl")
        outputs += plot_heatmap(
            filter_rows(seed0, modality="multiplication", target="result"),
            metric="normalized_gain",
            title="Multiplication linear probe: Result",
            output_stem="linear_multiplication_result_seed0",
            output_dir=output_dir,
            colorbar_label="Normalized gain",
        )
        seed_rows_list = [load_jsonl(path) for path in seed_paths]
        outputs += plot_multiplication_result_summary(seed_rows_list, output_dir)
    return outputs


def save_fourier_appendix():
    output_dir = OUTPUT_ROOT / "appendix_fourier_probes"
    outputs = []
    periods = [2, 5, 10, 20, 50, 100]
    shuffled_period = 10

    for modality in ["addition", "subtraction"]:
        rows = load_jsonl(
            RESULT_ROOT / "fourier_probes" / modality / "add_sub_fourier_results_ridge.jsonl"
        )
        shuffled = load_jsonl(
            RESULT_ROOT / "fourier_probes" / modality / "add_sub_fourier_results_shuffled_ridge.jsonl"
        )
        for period in periods:
            outputs += plot_heatmap(
                filter_rows(rows, modality=modality, target="result", period=period),
                metric="r2_mean",
                title=f"{label_modality(modality)} Fourier probe: Result, T={period}",
                output_stem=f"fourier_{modality}_result_T{period}",
                output_dir=output_dir,
                colorbar_label=r"Mean $R^2$",
                cmap=period_cmap(period),
            )
        outputs += plot_heatmap(
            filter_rows(shuffled, modality=modality, target="result", period=shuffled_period),
            metric="r2_mean",
            title=f"{label_modality(modality)} shuffled Fourier probe: Result, T={shuffled_period}",
            output_stem=f"fourier_{modality}_result_T{shuffled_period}_shuffled",
            output_dir=output_dir,
            colorbar_label=r"Mean $R^2$",
            cmap=period_cmap(shuffled_period),
        )

    mul = load_jsonl(RESULT_ROOT / "fourier_probes" / "multiplication" / "mul_fourier_results_ridge.jsonl")
    mul_shuffled = load_jsonl(
        RESULT_ROOT / "fourier_probes" / "multiplication" / "mul_fourier_results_shuffled_ridge.jsonl"
    )
    for target in ["result", "c1_hat", "c0_hat"]:
        for period in periods:
            outputs += plot_heatmap(
                filter_rows(mul, modality="multiplication", target=target, period=period),
                metric="r2_mean",
                title=f"Multiplication Fourier probe: {label_target(target)}, T={period}",
                output_stem=f"fourier_multiplication_{target}_T{period}",
                output_dir=output_dir,
                colorbar_label=r"Mean $R^2$",
                cmap=period_cmap(period),
            )
    outputs += plot_heatmap(
        filter_rows(mul_shuffled, modality="multiplication", target="c1_hat", period=shuffled_period),
        metric="r2_mean",
        title=r"Multiplication shuffled Fourier probe: $\hat{c}_1$, T=10",
        output_stem="fourier_multiplication_c1_hat_T10_shuffled",
        output_dir=output_dir,
        colorbar_label=r"Mean $R^2$",
        cmap=period_cmap(shuffled_period),
    )

    held_out_path = (
        RESULT_ROOT
        / "fourier_probes"
        / "multiplication"
        / "held_out_result_mul_fourier_results_ridge.jsonl"
    )
    if held_out_path.exists():
        held_out = load_jsonl(held_out_path)
        outputs += plot_heatmap(
            filter_rows(held_out, modality="multiplication", target="result", period=891),
            metric="r2_mean",
            title="Multiplication validation Fourier probe: Result, T=891",
            output_stem="fourier_multiplication_result_T891_held_out",
            output_dir=output_dir,
            colorbar_label=r"Mean $R^2$",
            cmap=period_cmap(891),
        )
    return outputs


def main():
    args = parse_args()
    global RESULT_ROOT, OUTPUT_ROOT
    RESULT_ROOT = args.result_root
    OUTPUT_ROOT = args.output_root

    outputs = save_linear_appendix()
    outputs += save_fourier_appendix()
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
