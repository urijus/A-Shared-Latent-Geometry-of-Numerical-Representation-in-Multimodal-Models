"""Two-panel readout diagnostic figure for DAS subspaces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_ROOT = REPO_ROOT / "results" / "experiments" / "check_unembeeding"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "readout_diagnostic"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"
ORIGINAL = "#226a74"
REMOVED = "#a95642"
RANDOM = "#9aa1a3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat_original",
        type=Path,
        default=CHECK_ROOT / "repeat_transfer" / "repeat_transfer_1digit.jsonl",
    )
    parser.add_argument(
        "--repeat_ablated",
        type=Path,
        default=CHECK_ROOT / "repeat_transfer" / "repeat_transfer_1digit_das_readout_ablated_seed1.jsonl",
    )
    parser.add_argument(
        "--self_ablated",
        type=Path,
        default=CHECK_ROOT / "self_ablated_eval" / "text_add_seed1_ablated_vs_original.jsonl",
    )
    parser.add_argument(
        "--procrustes_before",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "final_exps"
        / "procrustes"
        / "scaled_displacement"
        / "procrustes_results.jsonl",
    )
    parser.add_argument(
        "--procrustes_after",
        type=Path,
        default=CHECK_ROOT / "procrustes_readout_ablated" / "scaled_displacement" / "procrustes_results.jsonl",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="das_readout_diagnostic")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.2,
            "axes.titlesize": 9.8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.6,
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


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def style_axis(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)


def repeat_tf(path: Path) -> float:
    for row in load_jsonl(path):
        if row.get("condition") == "arithmetic_das_subspace":
            return float(row["variable_teacher_forced_iia"])
    raise ValueError(f"No arithmetic_das_subspace row in {path}")


def self_ar_pair(path: Path) -> tuple[float, float]:
    rows = load_jsonl(path)
    values = {row["label"]: float(row["autoregressive_iia"]) for row in rows}
    return values["original"], values["ablated"]


def procrustes_by_operation(path: Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    aligned = defaultdict(list)
    random = defaultdict(list)
    for row in load_jsonl(path):
        same_operation = row.get("source_operation") == row.get("destination_operation")
        cross_modality = row.get("source_modality") != row.get("destination_modality")
        if not (same_operation and cross_modality):
            continue
        operation = row["source_operation"]
        aligned[operation].append(float(row["alignment_fit_test"]["mean_cosine"]))
        random[operation].append(float(row["random_control_test"]["mean_cosine"]))
    return aligned, random


def add_value_labels(ax, bars, *, dy: float = 0.014) -> None:
    for bar in bars:
        value = bar.get_height()
        shown = 0.0 if abs(value) < 0.005 else value
        y = value + dy if value >= 0 else dy
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{shown:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )


def draw_behavior_panel(ax, self_before, self_after, repeat_before, repeat_after) -> dict:
    labels = ["Arithmetic\nAR IIA", "Repeat-number\nTF IIA"]
    before = [self_before, repeat_before]
    after = [self_after, repeat_after]
    x = np.arange(len(labels), dtype=float)
    width = 0.32

    bars_before = ax.bar(
        x - width / 2,
        before,
        width=width,
        color=ORIGINAL,
        edgecolor=SPINE,
        linewidth=0.8,
        label="Original DAS",
        zorder=3,
    )
    bars_after = ax.bar(
        x + width / 2,
        after,
        width=width,
        color=REMOVED,
        edgecolor=SPINE,
        linewidth=0.8,
        label="Readout removed",
        zorder=3,
    )
    add_value_labels(ax, bars_before)
    add_value_labels(ax, bars_after)
    ax.set_title("A. Behavioral effect")
    ax.set_ylabel("IIA")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8)
    style_axis(ax)
    return {
        "arithmetic_ar_iia": {"original": self_before, "readout_removed": self_after},
        "repeat_tf_iia": {"original": repeat_before, "readout_removed": repeat_after},
    }


def draw_geometry_panel(ax, before, after, rand_before, rand_after) -> dict:
    operations = ["addition", "subtraction"]
    labels = ["Addition", "Subtraction"]
    original = [mean(before[op]) for op in operations]
    removed = [mean(after[op]) for op in operations]
    random_ref = [mean(rand_before[op] + rand_after[op]) for op in operations]
    x = np.arange(len(labels), dtype=float)
    width = 0.24

    bars_original = ax.bar(
        x - width,
        original,
        width=width,
        color=ORIGINAL,
        edgecolor=SPINE,
        linewidth=0.8,
        label="Before removal",
        zorder=3,
    )
    bars_removed = ax.bar(
        x,
        removed,
        width=width,
        color=REMOVED,
        edgecolor=SPINE,
        linewidth=0.8,
        label="After removal",
        zorder=3,
    )
    bars_random = ax.bar(
        x + width,
        random_ref,
        width=width,
        color=RANDOM,
        edgecolor=SPINE,
        linewidth=0.8,
        label="Random rotation",
        zorder=3,
    )
    add_value_labels(ax, bars_original)
    add_value_labels(ax, bars_removed)
    add_value_labels(ax, bars_random, dy=0.018)
    ax.axhline(0.0, color="#9aa1a3", linewidth=0.85)
    ax.text(
        0.5,
        0.035,
        "random rotation",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#5a6265",
    )
    ax.set_title("B. Held-out geometry")
    ax.set_ylabel("Procrustes cosine")
    ax.set_xticks(x, labels)
    ax.set_ylim(-0.08, 1.10)
    style_axis(ax)
    return {
        op: {
            "before": original[index],
            "after": removed[index],
            "random_rotation": random_ref[index],
        }
        for index, op in enumerate(operations)
    }


def save_outputs(fig, output_dir: Path, stem: str, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    summary_path = output_dir / f"{stem}_summary.json"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {summary_path}")


def main() -> None:
    args = parse_args()
    style_matplotlib()

    self_before, self_after = self_ar_pair(args.self_ablated)
    repeat_before = repeat_tf(args.repeat_original)
    repeat_after = repeat_tf(args.repeat_ablated)
    proc_before, rand_before = procrustes_by_operation(args.procrustes_before)
    proc_after, rand_after = procrustes_by_operation(args.procrustes_after)

    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.2))
    fig.patch.set_facecolor(BACKGROUND)
    summary = {
        "panel_a_behavior": draw_behavior_panel(
            axes[0],
            self_before,
            self_after,
            repeat_before,
            repeat_after,
        ),
        "panel_b_geometry": draw_geometry_panel(
            axes[1],
            proc_before,
            proc_after,
            rand_before,
            rand_after,
        ),
    }
    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.20, wspace=0.26)
    save_outputs(fig, args.output_dir, args.stem, summary)


if __name__ == "__main__":
    main()
