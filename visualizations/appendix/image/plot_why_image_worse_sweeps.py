"""Two-panel appendix plot for the why-image-worse DAS sweeps."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = REPO_ROOT / "results" / "final_exps" / "why_image_worse"
OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "image"

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"
LINE = "#5f9b92"
LIGHT_LINE = "#9ec9c0"
ACCENT = "#8f4f59"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--position-k", type=int, default=22)
    parser.add_argument("--image-position", default="-1")
    parser.add_argument("--positions", type=int, nargs="+", default=[-6, -5, -4, -3, -2, -1])
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 22, 32, 64, 128])
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
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.6,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def setup_axis(ax) -> None:
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.68)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.28)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return np.nan, np.nan
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def position_sweep_summary(result_root: Path, positions: list[int], k: int):
    rows = load_jsonl(result_root / "aggregate_results.jsonl")
    by_position: dict[int, dict[str, list[float]]] = {
        position: {"autoregressive_iia": [], "variable_teacher_forced_iia": []}
        for position in positions
    }
    for row in rows:
        if (
            row.get("experiment") != "position_sweep"
            or row.get("condition") != "das_pca_initialized"
            or row.get("modality") != "image"
            or row.get("operation") != "addition"
            or row.get("target") != "result"
            or int(row.get("requested_k", -1)) != int(k)
        ):
            continue
        position = int(row["requested_position"])
        if position not in by_position:
            continue
        by_position[position]["autoregressive_iia"].append(float(row["autoregressive_iia"]))
        by_position[position]["variable_teacher_forced_iia"].append(float(row["variable_teacher_forced_iia"]))

    summary = {}
    for position in positions:
        ar_mean, ar_std = mean_std(by_position[position]["autoregressive_iia"])
        tf_mean, tf_std = mean_std(by_position[position]["variable_teacher_forced_iia"])
        summary[position] = {
            "autoregressive_mean": ar_mean,
            "autoregressive_std": ar_std,
            "teacher_forced_mean": tf_mean,
            "teacher_forced_std": tf_std,
            "n": len(by_position[position]["autoregressive_iia"]),
        }
    return summary


def iter_k_sweep_rows(result_root: Path, layer: int, image_position: str, k: int):
    root = result_root / "k_sweep" / f"layer{layer}" / f"image_pos{image_position}" / f"k{k}"
    seen: set[tuple[int, str]] = set()
    for path in sorted(root.rglob("results.jsonl")):
        for row in load_jsonl(path):
            if (
                row.get("condition") != "das_pca_initialized"
                or row.get("modality") != "image"
                or row.get("operation") != "addition"
                or row.get("target") != "result"
                or int(row.get("k", -1)) != int(k)
            ):
                continue
            key = (int(row.get("seed", 0)), str(path))
            if key in seen:
                continue
            seen.add(key)
            yield row


def k_sweep_summary(result_root: Path, layer: int, image_position: str, ks: list[int]):
    summary = {}
    for k in ks:
        ar_values = []
        for row in iter_k_sweep_rows(result_root, layer, image_position, k):
            ar_values.append(float(row["autoregressive_iia"]))
        ar_mean, ar_std = mean_std(ar_values)
        summary[k] = {"autoregressive_mean": ar_mean, "autoregressive_std": ar_std, "n": len(ar_values)}
    return summary


def draw_position_panel(ax, summary: dict[int, dict], positions: list[int]) -> None:
    setup_axis(ax)
    x = np.array(positions, dtype=float)
    ar_mean = np.array([summary[position]["autoregressive_mean"] for position in positions])
    ar_std = np.array([summary[position]["autoregressive_std"] for position in positions])
    tf_mean = np.array([summary[position]["teacher_forced_mean"] for position in positions])
    tf_std = np.array([summary[position]["teacher_forced_std"] for position in positions])

    ax.errorbar(
        x,
        ar_mean,
        yerr=ar_std,
        color=LINE,
        marker="o",
        markersize=4.2,
        linewidth=1.65,
        capsize=2.8,
        capthick=0.8,
        label="Autoregressive",
    )
    ax.errorbar(
        x,
        tf_mean,
        yerr=tf_std,
        color=LIGHT_LINE,
        marker="o",
        markersize=3.5,
        linewidth=1.35,
        linestyle="--",
        capsize=2.4,
        capthick=0.7,
        alpha=0.78,
        label="Teacher-forced",
    )
    ax.set_title("(a) Position sweep", pad=8, fontweight="normal")
    ax.set_xlabel("Prompt-relative position")
    ax.set_ylabel("IIA")
    ax.set_xticks(positions, [str(position) for position in positions])
    ax.set_ylim(-0.03, 0.82)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)


def draw_k_panel(ax, summary: dict[int, dict], ks: list[int]) -> None:
    setup_axis(ax)
    x = np.array(ks, dtype=float)
    means = np.array([summary[k]["autoregressive_mean"] for k in ks])
    stds = np.array([summary[k]["autoregressive_std"] for k in ks])

    ax.errorbar(
        x,
        means,
        yerr=stds,
        color=ACCENT,
        marker="o",
        markersize=4.4,
        linewidth=1.65,
        capsize=2.8,
        capthick=0.8,
    )
    ax.set_xscale("log", base=2)
    ax.set_title("(b) Dimension sweep", pad=8, fontweight="normal")
    ax.set_xlabel(r"DAS dimension $k$")
    ax.set_ylabel("Autoregressive IIA")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_ylim(-0.03, 0.82)


def save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def plot_sweeps(args: argparse.Namespace) -> list[Path]:
    position_summary = position_sweep_summary(args.result_root, args.positions, args.position_k)
    dimension_summary = k_sweep_summary(args.result_root, args.layer, args.image_position, args.ks)

    style_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.45), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    draw_position_panel(axes[0], position_summary, args.positions)
    draw_k_panel(axes[1], dimension_summary, args.ks)
    fig.suptitle("Image DAS Sweep Diagnostics", y=0.985, color=TEXT)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.19, wspace=0.28)
    return save_figure(fig, args.output_dir, "why_image_worse_position_dimension_sweeps")


def main() -> None:
    args = parse_args()
    for output in plot_sweeps(args):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
