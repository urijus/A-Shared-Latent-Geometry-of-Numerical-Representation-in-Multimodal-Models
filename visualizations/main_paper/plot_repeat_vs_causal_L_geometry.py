from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_GEOMETRY_DIR = Path("results/experiments/closing/repeat_vs_causal_L_geometry")
DEFAULT_NOISE_DIR = Path("results/experiments/closing/repeat_centroid_noise_control")
DEFAULT_OUTPUT_DIR = Path("visualizations/main_paper/repeat_vs_causal_L_geometry")
DEFAULT_STEM = "repeat_vs_causal_L_geometry"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
MUTED = "#5e696d"
SECONDARY_TEXT = "#4f5a5e"
SPINE = "#202426"
GRID = "#d8ddde"
TEAL = "#226a74"
RUST = "#a95642"
PURPLE = "#7a5b98"
GOLD = "#c99a3d"
GREEN = "#5d8f7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--noise-dir", type=Path, default=DEFAULT_NOISE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser.parse_args()


def style_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "figure.dpi": 120,
            "savefig.dpi": 350,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.9,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def style_panel(ax: plt.Axes, heading: str, label: str) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.set_title(heading, pad=8)
    ax.text(0.018, 0.982, label, transform=ax.transAxes, ha="left", va="top", fontsize=10.0)


def rounded_box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    edge: str,
    face: str = "#ffffff",
    fontsize: float = 7.0,
    weight: str = "normal",
    zorder: int = 4,
) -> FancyBboxPatch:
    x, y = center
    w, h = size
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.95,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, linespacing=1.14, zorder=zorder + 1)
    return patch


def arrow_between(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = SPINE,
    mutation_scale: float = 8.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=0.95,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def draw_projection_icon(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.plot([x - w / 2, x + w / 2], [y - h / 2, y - h / 2], color=GRID, linewidth=0.65, zorder=5)
    ax.plot([x - w / 2, x - w / 2], [y - h / 2, y + h / 2], color=GRID, linewidth=0.65, zorder=5)
    ax.plot([x - w * 0.36, x + w * 0.34], [y - h * 0.28, y + h * 0.24], color=TEAL, linewidth=1.2, zorder=6)
    ax.plot([x - w * 0.34, x + w * 0.33], [y - h * 0.20, y + h * 0.14], color=RUST, linewidth=1.2, zorder=6)
    ax.scatter([x + w * 0.35, x + w * 0.33], [y + h * 0.24, y + h * 0.14], s=10, color=[TEAL, RUST], zorder=7)


def draw_vector_comparison(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    x0 = x - w * 0.34
    y0 = y - h * 0.20
    ax.plot([x0 - w * 0.05, x0 + w * 0.72], [y0, y0], color=GRID, linewidth=0.65, zorder=6)
    ax.plot([x0, x0], [y0 - h * 0.10, y0 + h * 0.58], color=GRID, linewidth=0.65, zorder=6)
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x0 + w * 0.60, y0 + h * 0.42),
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=1.05,
            color=TEAL,
            zorder=8,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x0 + w * 0.62, y0 + h * 0.35),
            arrowstyle="-|>",
            mutation_scale=6.5,
            linewidth=1.05,
            color=RUST,
            zorder=8,
        )
    )
    ax.text(x0 + w * 0.34, y0 + h * 0.53, "cos", ha="center", va="center", fontsize=5.9, color=SECONDARY_TEXT, zorder=7)


def draw_schematic(ax: plt.Axes) -> None:
    style_panel(ax, "Experimental concept", "(a)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    stage_y = 0.895
    for x, label in [(0.18, "Task"), (0.50, r"$L$-geometry"), (0.82, "Comparison")]:
        ax.text(x, stage_y, label, ha="center", va="center", fontsize=6.7, color=SECONDARY_TEXT)

    top_y = 0.675
    bot_y = 0.385
    left_x = 0.18
    mid_x = 0.50
    right_x = 0.82
    box = (0.245, 0.142)
    mid_box = (0.235, 0.142)

    rounded_box(ax, (left_x, top_y), box, "38 + 5 =", edge=TEAL, face="#edf3f1", fontsize=7.6)
    rounded_box(ax, (left_x, bot_y), box, '"Repeat 43"', edge=RUST, face="#f1e6e2", fontsize=7.6)

    rounded_box(ax, (mid_x, top_y), mid_box, r"extract $h_{43}$" + "\nat final token", edge=TEAL, fontsize=6.95)
    rounded_box(ax, (mid_x, bot_y), mid_box, r"project $h_{43}$" + "\ninto arithmetic", edge=RUST, fontsize=6.95)

    compare_box = FancyBboxPatch(
        (right_x - 0.235 / 2, 0.53 - 0.315 / 2),
        0.235,
        0.315,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.95,
        edgecolor=PURPLE,
        facecolor="#f3eff7",
        zorder=4,
    )
    ax.add_patch(compare_box)
    ax.text(
        right_x,
        0.604,
        "same numerical\nidentity?",
        ha="center",
        va="center",
        fontsize=6.95,
        fontweight="bold",
        linespacing=1.12,
        zorder=7,
    )
    draw_vector_comparison(ax, right_x, 0.425, 0.160, 0.145)

    gap = 0.016
    arrow_between(ax, (left_x + box[0] / 2 + gap, top_y), (mid_x - mid_box[0] / 2 - gap, top_y), color=TEAL)
    arrow_between(ax, (left_x + box[0] / 2 + gap, bot_y), (mid_x - mid_box[0] / 2 - gap, bot_y), color=RUST)
    arrow_between(ax, (mid_x + mid_box[0] / 2 + gap, top_y), (compare_box.get_x() - gap, top_y - 0.052), color=TEAL)
    arrow_between(ax, (mid_x + mid_box[0] / 2 + gap, bot_y), (compare_box.get_x() - gap, bot_y + 0.052), color=RUST)

    ax.text(
        0.50,
        0.140,
        r"Repeat $h_{43}$ is projected into the arithmetic $L$ basis."
        + "\n"
        + "No repeat DAS or repeat intervention is trained.",
        ha="center",
        va="center",
        fontsize=6.15,
        color=SECONDARY_TEXT,
        linespacing=1.12,
    )


def load_distributions(noise_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    geo = pd.read_csv(noise_dir / "one_example_geometry.csv")
    geo = geo[
        (geo["alignment"] == "procrustes")
        & (geo["value_filter"] == "all_values")
        & geo["heldout_transition_cosine"].notna()
        & geo["comparison_family"].isin(["arithmetic_arithmetic", "repeat_arithmetic"])
    ]
    rsa = pd.read_csv(noise_dir / "one_example_rsa.csv")
    rsa = rsa[
        (rsa["value_filter"] == "all_values")
        & rsa["comparison_family"].isin(["arithmetic_arithmetic", "repeat_arithmetic"])
    ]

    distributions = {
        "Transition cosine": {
            "AA": geo[geo["comparison_family"] == "arithmetic_arithmetic"].groupby("replicate")["heldout_transition_cosine"].mean().to_numpy(),
            "RA": geo[geo["comparison_family"] == "repeat_arithmetic"].groupby("replicate")["heldout_transition_cosine"].mean().to_numpy(),
        },
        "RSA": {
            "AA": rsa[rsa["comparison_family"] == "arithmetic_arithmetic"].groupby("replicate")["spearman_rsa"].mean().to_numpy(),
            "RA": rsa[rsa["comparison_family"] == "repeat_arithmetic"].groupby("replicate")["spearman_rsa"].mean().to_numpy(),
        },
    }
    summary = {}
    for metric, groups in distributions.items():
        summary[metric] = {
            key: {
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
            }
            for key, values in groups.items()
        }
    return distributions, summary


def draw_distribution_axis(ax: plt.Axes, metric: str, groups: dict[str, np.ndarray], *, ylim: tuple[float, float]) -> None:
    colors = {"AA": TEAL, "RA": RUST}
    labels = {"AA": r"arith $\to$ arith", "RA": r"repeat $\to$ arith"}
    positions = [0.0, 1.0]
    values = [groups["AA"], groups["RA"]]

    violins = ax.violinplot(values, positions=positions, widths=0.62, showmeans=False, showextrema=False, showmedians=False)
    for body, key in zip(violins["bodies"], ["AA", "RA"]):
        body.set_facecolor(colors[key])
        body.set_edgecolor(colors[key])
        body.set_alpha(0.23)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(17)
    label_offsets = {"AA": 0.34, "RA": 0.27}
    for pos, key, vals in zip(positions, ["AA", "RA"], values):
        jitter = rng.normal(0, 0.040, size=vals.size)
        ax.scatter(np.full(vals.size, pos) + jitter, vals, s=7, color=colors[key], alpha=0.20, linewidth=0, zorder=2)
        mean = float(np.mean(vals))
        ax.scatter([pos], [mean], s=26, facecolor="#ffffff", edgecolor=colors[key], linewidth=1.05, zorder=5)
        ax.text(
            pos + label_offsets[key],
            mean + (ylim[1] - ylim[0]) * 0.045,
            f"{mean:.3f}",
            ha="left",
            va="bottom",
            fontsize=7.0,
            color=colors[key],
            zorder=6,
        )

    ax.set_title(metric, pad=5, fontsize=8.4)
    ax.set_xticks(positions, [labels["AA"], labels["RA"]])
    ax.set_xlim(-0.48, 1.68)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.75)


def draw_estimator_panel(ax: plt.Axes, distributions: dict[str, dict[str, np.ndarray]]) -> None:
    style_panel(ax, "Estimator-matched evidence", "(b)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    left = ax.inset_axes([0.115, 0.205, 0.380, 0.635])
    right = ax.inset_axes([0.580, 0.205, 0.350, 0.635])
    left.set_facecolor(PANEL_BG)
    right.set_facecolor(PANEL_BG)
    draw_distribution_axis(left, "Transition cosine", distributions["Transition cosine"], ylim=(0.605, 0.735))
    draw_distribution_axis(right, "RSA", distributions["RSA"], ylim=(0.335, 0.765))
    left.set_ylabel("score")
    right.set_ylabel("")


def save_outputs(fig: plt.Figure, output_dir: Path, stem: str, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    summary_path = output_dir / f"{stem}_summary.json"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {summary_path}")


def main() -> None:
    args = parse_args()
    style_matplotlib()
    distributions, dist_summary = load_distributions(args.noise_dir)

    fig = plt.figure(figsize=(8.9, 3.35), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.28])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    draw_schematic(ax_a)
    draw_estimator_panel(ax_b, distributions)

    fig.subplots_adjust(left=0.030, right=0.985, top=0.825, bottom=0.150, wspace=0.155)
    save_outputs(
        fig,
        args.output_dir,
        args.stem,
        {
            "geometry_dir": str(args.geometry_dir),
            "noise_dir": str(args.noise_dir),
            "panel_b_distributions": dist_summary,
            "panel_c": "not rendered; exported files contain aggregate retrieval metrics, not per-value prediction assignments",
        },
    )


if __name__ == "__main__":
    main()
