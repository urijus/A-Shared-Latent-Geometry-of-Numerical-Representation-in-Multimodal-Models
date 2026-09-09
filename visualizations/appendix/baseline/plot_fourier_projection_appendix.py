import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results" / "baseline" / "gemma4_12b_it" / "digits"
OUTPUT_DIR = REPO_ROOT / "visualizations" / "baseline" / "appendix_fourier_projections"
POSITION = 17

BACKGROUND = "#fdfdfd"
SPINE = "#c4c9ca"
ZERO_LINE = "#aab3b6"
TEXT = "#171717"
PHASE_COLORS = [
    "#8f4f59",
    "#c97f6d",
    "#d6a85f",
    "#9aa66f",
    "#5f9b8f",
    "#60798f",
    "#795f72",
]

TARGET_LABELS = {
    "result": "Result",
    "c1_hat": r"$\hat{c}_1$",
    "c0_hat": r"$\hat{c}_0$",
}

MODALITY_LABELS = {
    "addition": "Addition",
    "subtraction": "Subtraction",
    "multiplication": "Multiplication",
}


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
            "axes.titlesize": 10.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def label_target(target):
    return TARGET_LABELS.get(target, target.replace("_", " "))


def label_modality(modality):
    return MODALITY_LABELS.get(modality, modality.replace("_", " ").title())


def safe_name(value):
    return str(value).replace("-", "minus").replace("*", "times").replace(":", "_")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--position", default=str(POSITION))
    return parser.parse_args()


def projection_path(modality, target, period, layer, position=None, method="ridge"):
    if position is None:
        position = POSITION
    return (
        RESULT_ROOT
        / "fourier_probes"
        / modality
        / "fourier_projections"
        / f"{target}_pos{position}_{method}"
        / "projections"
        / f"{target}_{modality}"
        / safe_name(f"{target}_T{period}_layer{layer}_pos{position}_{method}_projections.pt")
    )


def load_projection(path):
    data = torch.load(path, map_location="cpu")
    metadata = data["metadata"]
    coords = data["orthonormal_plane_projection"].detach().cpu().float()
    target = metadata["target"]
    period = int(metadata["period"])
    residues = torch.tensor([int(row[target]) % period for row in data["labels"]])
    return {"path": path, "metadata": metadata, "coords": coords, "residues": residues}


def blend(color, amount):
    rgb = np.array(mcolors.to_rgb(color))
    bg = np.array(mcolors.to_rgb(BACKGROUND))
    return tuple(amount * rgb + (1.0 - amount) * bg)


def phase_cmap():
    return mcolors.LinearSegmentedColormap.from_list(
        "muted_fourier_phase",
        PHASE_COLORS + [PHASE_COLORS[0]],
    )


def residue_colors(period):
    cmap = phase_cmap()
    return [blend(cmap(index / period), 0.78) for index in range(period)]


def centered_coords(coords):
    return coords - coords.mean(dim=0, keepdim=True)


def set_square_limits(ax, coords):
    max_abs = float(coords.abs().max())
    half_width = max_abs * 1.12 if max_abs > 0 else 1.0
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_width, half_width)


def setup_panel(ax):
    ax.set_facecolor(BACKGROUND)
    ax.axhline(0.0, color=ZERO_LINE, linewidth=0.78, zorder=0)
    ax.axvline(0.0, color=ZERO_LINE, linewidth=0.78, zorder=0)
    ax.grid(False)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.75)


def plot_projection_panel(ax, item, title, show_ylabel, show_xlabel=True):
    coords = centered_coords(item["coords"])
    metadata = item["metadata"]
    period = int(metadata["period"])
    residues = item["residues"]
    colors = residue_colors(period)

    setup_panel(ax)
    for residue in range(period):
        mask = residues == residue
        if not bool(mask.any()):
            continue
        xy = coords[mask]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=9.0 if period <= 20 else 5.2,
            color=colors[residue],
            alpha=0.50 if period <= 20 else 0.34,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(title, pad=7, fontweight="normal")
    if show_xlabel:
        ax.set_xlabel(r"$\hat{w}_{\cos}$", labelpad=4)
    if show_ylabel:
        ax.set_ylabel(r"$\hat{w}_{\sin}$", labelpad=5)
    ax.set_aspect("equal", adjustable="box")
    set_square_limits(ax, coords)


def legend_handles(period):
    colors = residue_colors(period)
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[residue],
            markeredgewidth=0,
            markersize=7,
            label=str(residue),
        )
        for residue in range(period)
    ]


def save_projection_grid(
    items,
    titles,
    output_stem,
    caption=None,
    legend_period=None,
    legend_title=None,
    ncols=None,
):
    style_matplotlib()
    n_panels = len(items)
    ncols = ncols or n_panels
    nrows = int(np.ceil(n_panels / ncols))
    fig_width = 3.0 * ncols + (0.25 if ncols <= 3 else 0.7)
    fig_height = 3.05 * nrows + (0.35 if nrows > 1 else 0.30)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=False)
    axes = np.array(axes, dtype=object).reshape(-1)
    fig.patch.set_facecolor(BACKGROUND)

    for index, (ax, item, title) in enumerate(zip(axes, items, titles)):
        row = index // ncols
        col = index % ncols
        plot_projection_panel(
            ax,
            item,
            title=title,
            show_ylabel=col == 0,
            show_xlabel=row == nrows - 1,
        )
    for ax in axes[n_panels:]:
        ax.axis("off")

    if caption is not None:
        fig.text(0.5, 0.992, caption, ha="center", va="top", fontsize=13, color=TEXT)
    bottom = 0.29 if legend_period is not None else 0.18
    top = 0.90 if caption is not None else 0.94
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        top=top,
        bottom=bottom,
        wspace=0.10,
        hspace=0.28,
    )

    if legend_period is not None:
        fig.legend(
            handles=legend_handles(legend_period),
            title=legend_title or f"Mod {legend_period}",
            loc="lower center",
            ncol=legend_period,
            frameon=False,
            handlelength=1.0,
            handletextpad=0.32,
            columnspacing=0.8,
            title_fontsize=9.5,
            bbox_to_anchor=(0.5, 0.015),
            alignment="center",
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def collect_items(modality, target, pairs):
    items = []
    missing = []
    for period, layer in pairs:
        path = projection_path(modality, target, period, layer)
        if path.exists():
            items.append(load_projection(path))
        else:
            missing.append(path)
    if missing:
        print("Skipped missing projection files:")
        for path in missing:
            print(f"  {path}")
    return items


def layer_legend_title(modality, target, period):
    if target == "result":
        return f"{label_modality(modality)} mod {period}"
    return rf"{label_target(target)} mod {period}"


def save_layer_comparisons():
    outputs = []
    period = 10
    jobs = [
        ("subtraction", "result", [34, 35, 38], "layers34_35_38"),
        ("multiplication", "result", [34, 35, 38], "layers34_35_38"),
        ("multiplication", "c0_hat", [34, 35, 38], "layers34_35_38"),
        ("multiplication", "c1_hat", [34, 35, 38], "layers34_35_38"),
    ]
    for modality, target, layers, layer_suffix in jobs:
        items = collect_items(modality, target, [(period, layer) for layer in layers])
        if len(items) != len(layers):
            continue
        caption = f"{label_modality(modality)} {label_target(target)}, T={period}"
        stem = f"projection_{modality}_{target}_T{period}_{layer_suffix}"
        outputs += save_projection_grid(
            items=items,
            titles=[f"Layer {layer}" for layer in layers],
            output_stem=stem,
            caption=None,
            legend_period=period,
            legend_title=layer_legend_title(modality, target, period),
        )
    return outputs


def save_layer44_period_comparisons():
    outputs = []
    layer = 44
    periods = [2, 5, 10, 20, 50, 100]
    jobs = [
        ("addition", "result"),
        ("subtraction", "result"),
        ("multiplication", "result"),
        ("multiplication", "c1_hat"),
        ("multiplication", "c0_hat"),
    ]
    for modality, target in jobs:
        items = collect_items(modality, target, [(period, layer) for period in periods])
        if len(items) != len(periods):
            continue
        caption = f"{label_modality(modality)} {label_target(target)}, layer {layer}"
        stem = f"projection_{modality}_{target}_layer{layer}_periods2_5_10_20_50_100"
        outputs += save_projection_grid(
            items=items,
            titles=[f"T={period}" for period in periods],
            output_stem=stem,
            caption=caption,
            legend_period=None,
            ncols=3,
        )
    return outputs


def main():
    args = parse_args()
    global RESULT_ROOT, OUTPUT_DIR, POSITION
    RESULT_ROOT = args.result_root
    OUTPUT_DIR = args.output_dir
    POSITION = args.position

    outputs = save_layer_comparisons()
    outputs += save_layer44_period_comparisons()
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
