import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROJECTION_DIR = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "fourier_probes"
    / "addition"
    / "fourier_projections"
    / "result_pos17_ridge"
    / "projections"
    / "result_addition"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "01_arithmetic_reference" / "reference_outputs"
BACKGROUND = "#fdfdfd"


def default_projection_dir(operation):
    return (
        REPO_ROOT
        / "results"
        / "baseline"
        / "gemma4_12b_it"
        / "digits"
        / "fourier_probes"
        / operation
        / "fourier_projections"
        / "result_pos17_ridge"
        / "projections"
        / f"result_{operation}"
    )


def load_projection(path):
    data = torch.load(path, map_location="cpu")
    metadata = data["metadata"]
    coords = data["orthonormal_plane_projection"].detach().cpu().float()
    labels = data["labels"]

    target = metadata["target"]
    period = int(metadata["period"])
    residues = torch.tensor([int(row[target]) % period for row in labels])

    return {
        "path": path,
        "metadata": metadata,
        "coords": coords,
        "residues": residues,
    }


def style_matplotlib():
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 13,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
        }
    )


def centered_coords(coords, center):
    if not center:
        return coords
    return coords - coords.mean(dim=0, keepdim=True)


def set_square_limits(ax, coords, pad_fraction=0.12):
    max_abs = float(coords.abs().max())
    half = max_abs * (1.0 + pad_fraction)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)


def plot_panel(ax, item, colors, label, center):
    coords = centered_coords(item["coords"], center=center)
    residues = item["residues"]
    metadata = item["metadata"]
    period = int(metadata["period"])

    ax.set_facecolor(BACKGROUND)
    ax.axhline(0.0, color="#aab3b6", linewidth=0.85, zorder=0)
    ax.axvline(0.0, color="#aab3b6", linewidth=0.85, zorder=0)
    ax.grid(False)

    for residue in range(period):
        mask = residues == residue
        xy = coords[mask]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=10.5,
            color=colors[residue],
            alpha=0.48,
            linewidths=0,
            rasterized=True,
        )

    ax.set_title(f"Layer {metadata['layer']}", pad=8, fontweight="normal")
    ax.set_xlabel(r"$\hat{w}_{\cos}$")
    ax.set_ylabel(r"$\hat{w}_{\sin}$")
    ax.set_aspect("equal", adjustable="box")
    set_square_limits(ax, coords)

    for spine in ax.spines.values():
        spine.set_color("#c2c2c2")
        spine.set_linewidth(0.8)


def make_legend(colors, period):
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[i],
            markeredgewidth=0,
            markersize=8,
            label=str(i),
        )
        for i in range(period)
    ]


def muted_tab10(amount=0.72):
    base = plt.get_cmap("tab10").colors
    gray = mcolors.to_rgb(BACKGROUND)
    muted = []
    for color in base:
        rgb = mcolors.to_rgb(color)
        muted.append(
            tuple(
                amount * channel + (1.0 - amount) * gray_channel
                for channel, gray_channel in zip(rgb, gray)
            )
        )
    return muted


def plot_figure(items, output_dir, output_stem, title, center):
    style_matplotlib()
    period = int(items[0]["metadata"]["period"])
    target = items[0]["metadata"]["target"]
    colors = muted_tab10(amount=0.86)

    fig, axes = plt.subplots(
        1,
        len(items),
        figsize=(11.8, 4.3),
        constrained_layout=False,
    )
    if len(items) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BACKGROUND)

    for ax, item in zip(axes, items):
        plot_panel(
            ax=ax,
            item=item,
            colors=colors,
            label=None,
            center=center,
        )

    visual_center_x = 0.525
    fig.legend(
        handles=make_legend(colors, period),
        title=f"addition {target} mod {period}",
        loc="lower center",
        ncol=10,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        title_fontsize=10,
        bbox_to_anchor=(visual_center_x, 0.01),
        alignment="center",
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.84, bottom=0.27, wspace=0.05)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return png_path, pdf_path


def load_items(projection_dir, layers, period, position, target, method):
    paths = [
        projection_dir
        / f"{target}_T{period}_layer{layer}_pos{position}_{method}_projections.pt"
        for layer in layers
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing projection files: " + ", ".join(map(str, missing)))
    return [load_projection(path) for path in paths]


def plot_two_operation_figure(
    rows,
    output_dir,
    output_stem,
    center,
):
    style_matplotlib()
    first_items = next(iter(rows.values()))
    period = int(first_items[0]["metadata"]["period"])
    target = first_items[0]["metadata"]["target"]
    colors = muted_tab10(amount=0.86)
    row_labels = list(rows.keys())
    n_cols = len(first_items)

    fig, axes = plt.subplots(
        len(row_labels),
        n_cols,
        figsize=(11.8, 7.4),
        constrained_layout=False,
        sharex=False,
        sharey=False,
    )
    fig.patch.set_facecolor(BACKGROUND)

    for row_index, row_label in enumerate(row_labels):
        for col_index, item in enumerate(rows[row_label]):
            ax = axes[row_index, col_index]
            plot_panel(
                ax=ax,
                item=item,
                colors=colors,
                label=None,
                center=center,
            )
            if row_index > 0:
                ax.set_title("")

    row_y = [0.695, 0.365]
    for y, row_label in zip(row_y, row_labels):
        fig.text(
            0.018,
            y,
            row_label.title(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=13,
            color="#171717",
        )

    visual_center_x = 0.53
    fig.legend(
        handles=make_legend(colors, period),
        title=f"{target} mod {period}",
        loc="lower center",
        ncol=10,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        title_fontsize=10,
        bbox_to_anchor=(visual_center_x, 0.01),
        alignment="center",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.91, bottom=0.16, wspace=0.05, hspace=0.38)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)

    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final Result mod 10 orthonormal-plane plot."
    )
    parser.add_argument("--projection-dir", type=Path, default=DEFAULT_PROJECTION_DIR)
    parser.add_argument("--layers", type=int, nargs="+", default=[34, 35, 38])
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--position", type=int, default=17)
    parser.add_argument("--target", type=str, default="result")
    parser.add_argument("--method", type=str, default="ridge")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-stem",
        type=str,
        default="result_mod10_layers34_35_38_orthonormal_planes",
    )
    parser.add_argument(
        "--add-subtraction-row",
        action="store_true",
        help="Also write a two-row addition/subtraction version of the plot.",
    )
    parser.add_argument(
        "--two-row-output-stem",
        type=str,
        default="result_mod10_layers34_35_38_add_sub_orthonormal_planes",
    )
    parser.add_argument("--title", type=str, default="Addition: Result Mod 10")
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Plot absolute projection coordinates instead of centering each panel.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_items(
        projection_dir=args.projection_dir,
        layers=args.layers,
        period=args.period,
        position=args.position,
        target=args.target,
        method=args.method,
    )
    output_paths = plot_figure(
        items=items,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        title=args.title,
        center=not args.no_center,
    )
    for path in output_paths:
        print(f"Saved {path}")

    if args.add_subtraction_row:
        rows = {
            "addition": items,
            "subtraction": load_items(
                projection_dir=default_projection_dir("subtraction"),
                layers=args.layers,
                period=args.period,
                position=args.position,
                target=args.target,
                method=args.method,
            ),
        }
        output_paths = plot_two_operation_figure(
            rows=rows,
            output_dir=args.output_dir,
            output_stem=args.two_row_output_stem,
            center=not args.no_center,
        )
        for path in output_paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
