"""Plot DAS autoregressive IIA across token positions for one layer."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "das_v2"
    / "runs"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "baseline" / "das"

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"
LINE = "#5f9b92"
ACCENT = "#8f4f59"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--operation", default="addition")
    parser.add_argument("--target", default="result")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--max_position", type=int, default=17)
    parser.add_argument("--hook", default="resid_post")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--metric", default="autoregressive_iia")
    return parser.parse_args()


def long_path(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_jsonl(path):
    with open(long_path(path), "r", encoding="utf-8") as handle:
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
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
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
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.68)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.28)


def position_from_name(path):
    match = re.search(r"_pos(-?\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1))


def matching_row(rows, args, position):
    matches = [
        row
        for row in rows
        if row.get("target") == args.target
        and int(row.get("layer", -1)) == args.layer
        and int(row.get("k", -1)) == args.k
        and str(row.get("position")) == str(position)
        and row.get("hook") == args.hook
        and int(row.get("seed", -1)) == args.seed
        and args.metric in row
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: float(row.get(args.metric, -1.0)))


def collect_points(args):
    pattern = (
        f"{args.operation}_{args.target}_pos*_{args.hook}_"
        f"initrandom_pca_seed{args.seed}.jsonl"
    )
    points = []
    for path in sorted(args.run_dir.glob(pattern)):
        position = position_from_name(path)
        if position is None or position > args.max_position:
            continue
        row = matching_row(load_jsonl(path), args, position)
        if row is None:
            continue
        points.append((position, float(row[args.metric])))
    points.sort()
    return points


def label_operation(operation, target):
    return f"{operation.replace('_', ' ').title()} {target.replace('_', ' ')}"


def plot(points, args):
    if not points:
        raise ValueError("No matching DAS rows found for the requested sweep.")

    positions = [position for position, _ in points]
    values = [value for _, value in points]

    style_matplotlib()
    fig, ax = plt.subplots(figsize=(5.55, 3.45))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    ax.plot(
        positions,
        values,
        color=LINE,
        marker="o",
        markersize=4.2,
        linewidth=1.55,
        label=f"k={args.k}",
    )
    ax.scatter(
        [positions[-1]],
        [values[-1]],
        s=52,
        facecolor=BACKGROUND,
        edgecolor=ACCENT,
        linewidth=1.2,
        zorder=4,
    )

    ax.set_title(
        f"DAS position sweep: {label_operation(args.operation, args.target)}, L{args.layer}",
        pad=10,
    )
    ax.set_xlabel("Token position")
    ax.set_ylabel("Autoregressive IIA")
    ax.set_xticks(positions)
    ax.set_ylim(0.0, min(1.04, max(1.0, max(values) * 1.10)))
    ax.legend(frameon=False, loc="lower right", handlelength=1.7)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.85, bottom=0.18)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"das_{args.operation}_{args.target}_layer{args.layer}_"
        f"k{args.k}_position_sweep_{args.metric}"
    )
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    points = collect_points(args)
    for output_path in plot(points, args):
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
