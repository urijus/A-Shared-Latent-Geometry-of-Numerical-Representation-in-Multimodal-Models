"""Render main-paper shared-frame figures from stored result summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "final_exps"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"
TEAL = "#226a74"
BLUE = "#5d6f9f"
GOLD = "#b8872d"
PURPLE = "#7a5b98"
GRAY = "#6d7477"
LIGHT_GRAY = "#d8ddde"

TASK_LABELS = {
    "text_add": "T+",
    "image_add": "I+",
    "text_sub": "T-",
    "image_sub": "I-",
    "text:addition": "T+",
    "image:addition": "I+",
    "text:subtraction": "T-",
    "image:subtraction": "I-",
}
TASK_ORDER = ["text:addition", "image:addition", "text:subtraction", "image:subtraction"]
HELDOUT_DIRECTIONS = [
    ("text:addition", "image:addition"),
    ("image:addition", "text:addition"),
    ("text:subtraction", "image:subtraction"),
    ("image:subtraction", "text:subtraction"),
]
HELDOUT_DIR_TO_FILE = {
    ("text:addition", "image:addition"): "text_addition-_image_addition",
    ("image:addition", "text:addition"): "text_addition-_image_addition",
    ("text:subtraction", "image:subtraction"): "text_subtraction-_image_subtraction",
    ("image:subtraction", "text:subtraction"): "text_subtraction-_image_subtraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fig7_stem", default="shared_frame_synchronization")
    parser.add_argument("--fig8_stem", default="shared_frame_registration_equivariance")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value) -> float:
    return np.nan if value is None else float(value)


def sample_std(values: list[float]) -> float:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def setup_axis(ax, title: str | None = None, *, grid: bool = True) -> None:
    if title:
        ax.set_title(title, pad=6)
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(grid, color=GRID, linewidth=0.6, alpha=0.72)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.85)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.075,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def direction_label(source: str, destination: str) -> str:
    return f"{TASK_LABELS[source]} $\\rightarrow$ {TASK_LABELS[destination]}"


def read_fig7_bar_data(results_dir: Path) -> tuple[list[dict], dict]:
    rows_by_file = {}
    for folder in sorted(set(HELDOUT_DIR_TO_FILE.values())):
        path = results_dir / "synchronization_held_out" / folder / "synchronization_summary.jsonl"
        rows_by_file[folder] = load_rows(path)

    data = []
    for source, destination in HELDOUT_DIRECTIONS:
        folder = HELDOUT_DIR_TO_FILE[(source, destination)]
        rows = rows_by_file[folder]
        selected = [
            row
            for row in rows
            if row["source_task"] == source and row["destination_task"] == destination
        ]
        by_transport = {row["transport"]: row for row in selected}
        data.append(
            {
                "source": source,
                "destination": destination,
                "label": direction_label(source, destination),
                "heldout_edges_key": by_transport["pairwise_direct"]["heldout_edges_key"],
                "direct_mean": finite(by_transport["pairwise_direct"]["autoregressive_iia_mean"]),
                "direct_std": finite(by_transport["pairwise_direct"]["autoregressive_iia_std"]),
                "direct_n": by_transport["pairwise_direct"]["n"],
                "heldout_mean": finite(by_transport["synchronized_hub"]["autoregressive_iia_mean"]),
                "heldout_std": finite(by_transport["synchronized_hub"]["autoregressive_iia_std"]),
                "heldout_n": by_transport["synchronized_hub"]["n"],
            }
        )
    return data, {"source_files": sorted(set(HELDOUT_DIR_TO_FILE.values()))}


def draw_fig7_schematic(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box", anchor="E")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.85)

    positions = {
        "T+": (0.20, 0.82),
        "I+": (0.80, 0.82),
        "T-": (0.20, 0.18),
        "I-": (0.80, 0.18),
    }
    colors = {"T+": TEAL, "I+": BLUE, "T-": "#a95642", "I-": GOLD}
    hub = (0.50, 0.50)

    hub_circle = patches.Circle(hub, 0.12, facecolor="white", edgecolor=SPINE, linewidth=1.15, zorder=4)
    ax.add_patch(hub_circle)
    ax.text(*hub, r"$H$", ha="center", va="center", fontsize=20, zorder=5)

    for label, (x, y) in positions.items():
        node = patches.Circle((x, y), 0.105, facecolor="white", edgecolor=colors[label], linewidth=2.0, zorder=5)
        ax.add_patch(node)
        ax.text(x, y, rf"${label}$", ha="center", va="center", fontsize=15.0, zorder=6)
        ax.annotate(
            "",
            xy=(hub[0] + 0.093 * np.sign(x - hub[0]), hub[1] + 0.093 * np.sign(y - hub[1])),
            xytext=(x - 0.087 * np.sign(x - hub[0]), y - 0.087 * np.sign(y - hub[1])),
            arrowprops=dict(
                arrowstyle="<->",
                color=colors[label],
                lw=1.55,
                alpha=0.9,
                shrinkA=2,
                shrinkB=2,
                mutation_scale=11,
            ),
            zorder=3,
        )

    ax.text(
        0.50,
        0.70,
        r"$\Delta z_i \leftrightarrow \Delta u$",
        ha="center",
        va="center",
        fontsize=8.8,
        color=GRAY,
        zorder=6,
    )


def draw_fig7_bars(ax, data: list[dict]) -> None:
    setup_axis(ax, grid=True)
    x = np.arange(len(data), dtype=float)
    width = 0.34
    direct = np.asarray([item["direct_mean"] for item in data])
    hub = np.asarray([item["heldout_mean"] for item in data])
    direct_std = np.asarray([item["direct_std"] for item in data])
    hub_std = np.asarray([item["heldout_std"] for item in data])

    ax.bar(
        x - width / 2,
        direct,
        width,
        yerr=direct_std,
        capsize=2.2,
        label="Direct pairwise map",
        color=GRAY,
        edgecolor=SPINE,
        linewidth=0.75,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        hub,
        width,
        yerr=hub_std,
        capsize=2.2,
        label="Held-out synchronized map",
        color=TEAL,
        edgecolor=SPINE,
        linewidth=0.75,
        zorder=3,
    )
    ax.set_xticks(x, [item["label"] for item in data])
    ax.set_ylabel("Autoregressive IIA", labelpad=1)
    ax.yaxis.set_label_coords(-0.055, 0.5)
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper left", frameon=True, facecolor=BACKGROUND, edgecolor=GRID, framealpha=0.95)
    ax.text(
        0.5,
        1.065,
        "Relation excluded in both directions during synchronization",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        clip_on=False,
    )


def read_fig8_registration_data(results_dir: Path) -> tuple[list[dict], dict]:
    rows = load_rows(results_dir / "equivariance_registered_linear" / "shared_space_sanity.jsonl")
    observations = []
    for row in rows:
        before_pairs = {
            (item["source_task"], item["destination_task"]): item
            for item in row["heldout_before_registration"]["pair_rows"]
        }
        after_pairs = {
            (item["source_task"], item["destination_task"]): item
            for item in row["heldout_after_registration"]["pair_rows"]
        }
        for key, before in before_pairs.items():
            after = after_pairs[key]
            observations.append(
                {
                    "seed": row["seed"],
                    "source_task": key[0],
                    "destination_task": key[1],
                    "before_cosine": finite(before["mean_cosine"]),
                    "after_cosine": finite(after["mean_cosine"]),
                    "before_rmse": finite(before["rmse"]),
                    "after_rmse": finite(after["rmse"]),
                    "n_values": before["n_values"],
                }
            )
    metadata = {
        "heldout_seed_mean_before_cosine": [row["heldout_before_registration"]["mean_cosine"] for row in rows],
        "heldout_seed_mean_after_cosine": [row["heldout_after_registration"]["mean_cosine"] for row in rows],
    }
    return observations, metadata


def read_fig8_successor_data(results_dir: Path) -> tuple[list[dict], dict]:
    summary_rows = load_rows(results_dir / "equivariance_registered_linear" / "geometry_summary.jsonl")
    result_rows = load_rows(results_dir / "equivariance_registered_linear" / "geometry_results.jsonl")
    models = ["orthogonal", "identity", "random_orthogonal"]
    data = []
    for task in TASK_ORDER:
        for model in models:
            summary = next(
                row
                for row in summary_rows
                if row["task"] == task and row["model"] == model and row["step"] == 1 and row["subset"] == "all"
            )
            seed_values = [
                finite(row["nearest_result_centroid_accuracy"])
                for row in result_rows
                if row["task"] == task and row["model"] == model and row["step"] == 1 and row["subset"] == "all"
            ]
            data.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "model": model,
                    "mean": finite(summary["nearest_result_centroid_accuracy_mean"]),
                    "std": sample_std(seed_values),
                    "n": summary["n"],
                    "seed_values": seed_values,
                }
            )
    return data, {"models": models}


def draw_fig8_registration(ax, observations: list[dict]) -> None:
    setup_axis(ax, grid=True)
    rng = np.random.default_rng(1)
    for obs in observations:
        jitter = float(rng.uniform(-0.035, 0.035))
        ax.plot(
            [0 + jitter, 1 + jitter],
            [obs["before_cosine"], obs["after_cosine"]],
            color="#8c9497",
            alpha=0.36,
            linewidth=0.75,
            zorder=2,
        )
        ax.scatter([0 + jitter, 1 + jitter], [obs["before_cosine"], obs["after_cosine"]], s=12, color="#8c9497", alpha=0.5, zorder=3)
    before = np.asarray([obs["before_cosine"] for obs in observations])
    after = np.asarray([obs["after_cosine"] for obs in observations])
    means = [float(np.nanmean(before)), float(np.nanmean(after))]
    sems = [sample_std(before.tolist()) / np.sqrt(len(before)), sample_std(after.tolist()) / np.sqrt(len(after))]
    ax.plot([0, 1], means, color=TEAL, linewidth=2.0, marker="o", markersize=4.8, zorder=5)
    ax.errorbar([0, 1], means, yerr=sems, fmt="none", color=TEAL, linewidth=1.0, capsize=2.5, zorder=5)
    ax.set_xticks([0, 1], ["Before\nalignment", "After offset\ncorrection"])
    ax.set_ylabel("Same value cross-task\ncosine similarity")
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(0.0, 1.05)
    rmse_before = np.nanmean([obs["before_rmse"] for obs in observations])
    rmse_after = np.nanmean([obs["after_rmse"] for obs in observations])
    ax.text(
        0.5,
        0.09,
        rf"$\mathrm{{RMSE}}:\ {rmse_before:.1f}\rightarrow {rmse_after:.1f}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        bbox=dict(facecolor=BACKGROUND, edgecolor=GRID, linewidth=0.6, pad=2.2),
    )


def draw_fig8_successor(ax, data: list[dict]) -> None:
    setup_axis(ax, grid=True)
    models = [
        ("orthogonal", r"$G_{+1}$", TEAL),
        ("identity", "Identity", GRAY),
        ("random_orthogonal", r"Random $Q$", PURPLE),
    ]
    x = np.arange(len(TASK_ORDER), dtype=float)
    width = 0.24
    for offset, (model, label, color) in zip([-width, 0.0, width], models):
        rows = [item for item in data if item["model"] == model]
        values = np.asarray([item["mean"] for item in rows])
        stds = np.asarray([item["std"] for item in rows])
        ax.bar(
            x + offset,
            values,
            width,
            yerr=stds,
            capsize=2.0,
            label=label,
            color=color,
            edgecolor=SPINE,
            linewidth=0.7,
            zorder=3,
        )
    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylabel(r"Nearest $y+1$ centroid accuracy")
    ax.set_ylim(0.0, 0.82)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        framealpha=0.95,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.4,
        fontsize=6.5,
        columnspacing=0.8,
    )
    ax.text(
        0.5,
        1.035,
        r"$G_{+1}$ fitted on $T+$ only",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4,
        clip_on=False,
    )


def save_figure(fig, output_dir: Path, stem: str, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}_summary.json"
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    fig.savefig(png, dpi=350, facecolor=fig.get_facecolor())
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {pdf}")
    print(f"Saved {png}")
    print(f"Saved {json_path}")


def draw_figure7(results_dir: Path, output_dir: Path, stem: str) -> None:
    bar_data, metadata = read_fig7_bar_data(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(7.75, 3.18), gridspec_kw={"width_ratios": [1.05, 1.28]})
    fig.patch.set_facecolor(BACKGROUND)
    draw_fig7_schematic(axes[0])
    draw_fig7_bars(axes[1], bar_data)
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    fig.subplots_adjust(left=0.052, right=0.99, top=0.925, bottom=0.18, wspace=0.14)
    save_figure(fig, output_dir, stem, {"panel_b": bar_data, **metadata})


def draw_figure8(results_dir: Path, output_dir: Path, stem: str) -> None:
    registration_data, registration_metadata = read_fig8_registration_data(results_dir)
    successor_data, successor_metadata = read_fig8_successor_data(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(4.85, 2.65), gridspec_kw={"width_ratios": [1.0, 1.28]})
    fig.patch.set_facecolor(BACKGROUND)
    draw_fig8_registration(axes[0], registration_data)
    draw_fig8_successor(axes[1], successor_data)
    add_panel_label(axes[0], "(a)")
    add_panel_label(axes[1], "(b)")
    fig.subplots_adjust(left=0.105, right=0.995, top=0.78, bottom=0.25, wspace=0.34)
    save_figure(
        fig,
        output_dir,
        stem,
        {
            "panel_a": registration_data,
            "panel_a_metadata": registration_metadata,
            "panel_b": successor_data,
            "panel_b_metadata": successor_metadata,
        },
    )


def main() -> None:
    args = parse_args()
    style_matplotlib()
    draw_figure7(args.results_dir, args.output_dir, args.fig7_stem)
    draw_figure8(args.results_dir, args.output_dir, args.fig8_stem)


if __name__ == "__main__":
    main()
