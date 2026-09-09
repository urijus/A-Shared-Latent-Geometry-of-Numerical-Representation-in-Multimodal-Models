"""Render the main readout decomposition figure from saved audit summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "experiments" / "closing" / "readout_causal_audit"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "readout_decomposition"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"

TASK_ORDER = ["text_addition", "text_subtraction", "image_addition", "image_subtraction"]
TASK_LABELS = {
    "text_addition": "T+",
    "text_subtraction": "T-",
    "image_addition": "I+",
    "image_subtraction": "I-",
}
TASK_COLORS = {
    "text_addition": "#226a74",
    "text_subtraction": "#a95642",
    "image_addition": "#5d6f9f",
    "image_subtraction": "#b8872d",
}
ORDERING_STYLES = {
    "aligned_first": ("-", "readout-aligned first"),
    "orthogonal_first": ("--", "readout-orthogonal first"),
}
BAR_SPECS = [
    ("original", "original DAS", "#226a74"),
    ("digit_ablated", "digit-readout ablated", "#a95642"),
    ("random_ablated", "random rank-9 ablation", "#7a5b98"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="readout_decomposition_main")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.8,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.6,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 7.7,
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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def sd(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def sem(values: list[float]) -> float:
    return sd(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def style_axis(ax, *, xgrid: bool = False) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, linewidth=0.45, alpha=0.32)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)


def grouped_curve_rows(rows: list[dict]) -> dict[tuple[str, str, int], list[dict]]:
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        task = row.get("task")
        ordering = row.get("ordering")
        seed = row.get("das_seed")
        if task not in TASK_ORDER or ordering not in ORDERING_STYLES or seed is None:
            continue
        groups[(task, ordering, int(seed))].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: int(row["m"]))
    return groups


def curve_summary(rows: list[dict]) -> dict[tuple[str, str, int], dict[str, float]]:
    by_m: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        task = row.get("task")
        ordering = row.get("ordering")
        recovery = to_float(row.get("relative_ar_recovery"))
        if task not in TASK_ORDER or ordering not in ORDERING_STYLES or recovery is None:
            continue
        by_m[(task, ordering, int(row["m"]))].append(recovery)
    return {
        key: {"mean": mean(values), "sd": sd(values), "sem": sem(values), "n": len(values)}
        for key, values in by_m.items()
    }


def draw_cumulative_panel(ax, rows: list[dict], summary_json: dict) -> dict:
    style_axis(ax, xgrid=True)
    seed_groups = grouped_curve_rows(rows)
    summary = curve_summary(rows)

    for task in TASK_ORDER:
        color = TASK_COLORS[task]
        for ordering, (linestyle, _label) in ORDERING_STYLES.items():
            for (group_task, group_ordering, _seed), group_rows in seed_groups.items():
                if group_task != task or group_ordering != ordering:
                    continue
                xs = [int(row["m"]) for row in group_rows]
                ys = [to_float(row.get("relative_ar_recovery")) for row in group_rows]
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.65,
                    alpha=0.16,
                    zorder=1,
                )

            xs = np.arange(1, 23)
            means = np.asarray([summary[(task, ordering, int(m))]["mean"] for m in xs], dtype=float)
            errors = np.asarray([summary[(task, ordering, int(m))]["sem"] for m in xs], dtype=float)
            ax.plot(
                xs,
                means,
                color=color,
                linestyle=linestyle,
                linewidth=2.05 if ordering == "aligned_first" else 1.75,
                marker="o" if ordering == "aligned_first" else None,
                markersize=2.6,
                markevery=[8, 12, 21] if ordering == "aligned_first" else None,
                zorder=3,
            )
            ax.fill_between(xs, means - errors, means + errors, color=color, alpha=0.075, linewidth=0, zorder=2)

    ax.axvline(9, color=SPINE, linewidth=0.9, linestyle=(0, (4, 3)), alpha=0.72)
    ax.axvline(13, color="#9aa1a3", linewidth=0.85, linestyle=":", alpha=0.82)
    ax.text(9.22, 1.045, r"rank($U_D$) = 9", ha="left", va="top", fontsize=8.1)
    ax.text(13.2, 0.965, "L component\nm = 13", ha="left", va="top", fontsize=7.4, color="#555f62")

    m9_lines = []
    for task in TASK_ORDER:
        key = f"{task}.aligned_recovery_m_9_mean"
        task_data = summary_json.get("tasks", {}).get(task, {})
        value = task_data.get("aligned_recovery_m_9_mean")
        if value is None:
            value = summary[(task, "aligned_first", 9)]["mean"]
        m9_lines.append(f"{TASK_LABELS[task]} {float(value):.3f}")
    ax.text(
        9.35,
        0.745,
        "aligned m=9\n" + "\n".join(m9_lines),
        ha="left",
        va="top",
        fontsize=7.1,
        linespacing=1.18,
        bbox={"facecolor": BACKGROUND, "edgecolor": GRID, "linewidth": 0.65, "alpha": 0.92, "pad": 3.0},
        zorder=5,
    )

    ax.set_xlim(1, 22)
    ax.set_ylim(0, 1.08)
    ax.set_xticks([1, 5, 9, 13, 17, 22])
    ax.set_xlabel("Number of DAS dimensions m")
    ax.set_ylabel("Normalized autoregressive causal recovery")
    ax.text(0.015, 0.965, "A", transform=ax.transAxes, ha="left", va="top", fontsize=12.5)

    task_handles = [
        mlines.Line2D([], [], color=TASK_COLORS[task], lw=2.0, label=TASK_LABELS[task])
        for task in TASK_ORDER
    ]
    ordering_handles = [
        mlines.Line2D([], [], color=SPINE, lw=1.8, linestyle=style, label=label)
        for style, label in ORDERING_STYLES.values()
    ]
    ax.legend(
        handles=task_handles + ordering_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        columnspacing=0.95,
        handlelength=1.8,
        handletextpad=0.42,
        borderaxespad=0.0,
    )

    return {
        TASK_LABELS[task]: {
            "aligned_m9": summary[(task, "aligned_first", 9)]["mean"],
            "orthogonal_m13": summary[(task, "orthogonal_first", 13)]["mean"],
            "aligned_m22": summary[(task, "aligned_first", 22)]["mean"],
            "orthogonal_m22": summary[(task, "orthogonal_first", 22)]["mean"],
        }
        for task in TASK_ORDER
    }


def seed_level_ablation(rows: list[dict]) -> dict[tuple[str, str], dict[str, list[float]]]:
    per_seed: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        task = row.get("task")
        condition = row.get("condition")
        value = to_float(row.get("raw_ar_iia"))
        seed = row.get("das_seed")
        if task not in TASK_ORDER or condition not in {spec[0] for spec in BAR_SPECS} or value is None or seed is None:
            continue
        per_seed[(task, condition)][int(seed)].append(value)

    result: dict[tuple[str, str], dict[str, list[float]]] = {}
    for key, seed_values in per_seed.items():
        seed_means = [mean(values) for _seed, values in sorted(seed_values.items())]
        result[key] = {"seed_means": seed_means}
    return result


def draw_ablation_panel(ax, rows: list[dict]) -> dict:
    style_axis(ax)
    grouped = seed_level_ablation(rows)
    x = np.arange(len(TASK_ORDER), dtype=float)
    width = 0.22
    offsets = np.linspace(-width, width, len(BAR_SPECS))
    summary: dict[str, dict[str, float]] = {}

    for offset, (condition, label, color) in zip(offsets, BAR_SPECS):
        values = []
        errors = []
        for task in TASK_ORDER:
            seed_values = grouped[(task, condition)]["seed_means"]
            values.append(mean(seed_values))
            errors.append(sd(seed_values))
            summary.setdefault(TASK_LABELS[task], {})[condition] = mean(seed_values)
            summary[TASK_LABELS[task]][f"{condition}_sd"] = sd(seed_values)
            summary[TASK_LABELS[task]][f"{condition}_n"] = len(seed_values)

        bars = ax.bar(
            x + offset,
            values,
            yerr=errors,
            width=width,
            color=color,
            edgecolor=SPINE,
            linewidth=0.78,
            capsize=2.2,
            error_kw={"elinewidth": 0.8, "ecolor": TEXT, "capthick": 0.8},
            label=label,
            zorder=3,
        )
        for bar, value, error in zip(bars, values, errors):
            shown = 0.0 if abs(value) < 0.005 else value
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(value + error, 0.0) + 0.028,
                f"{shown:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.1,
                rotation=0,
            )

    ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Autoregressive IIA")
    ax.text(0.015, 0.965, "B", transform=ax.transAxes, ha="left", va="top", fontsize=12.5)

    handles = [
        mpatches.Patch(facecolor=color, edgecolor=SPINE, linewidth=0.7, label=label)
        for _condition, label, color in BAR_SPECS
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.55,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    return summary


def shared_legend(fig) -> None:
    task_handles = [
        mlines.Line2D([], [], color=TASK_COLORS[task], lw=2.0, label=TASK_LABELS[task])
        for task in TASK_ORDER
    ]
    ordering_handles = [
        mlines.Line2D([], [], color=SPINE, lw=1.8, linestyle=style, label=label)
        for style, label in ORDERING_STYLES.values()
    ]
    bar_handles = [
        mpatches.Patch(facecolor=color, edgecolor=SPINE, linewidth=0.7, label=label)
        for _condition, label, color in BAR_SPECS
    ]
    fig.legend(
        handles=task_handles + ordering_handles + bar_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=1.05,
        handlelength=1.85,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def save_outputs(fig, output_dir: Path, stem: str, summary: dict) -> None:
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

    cumulative_rows = read_csv(args.input_dir / "cumulative_readout_results.csv")
    ablation_rows = read_csv(args.input_dir / "ablation_control_results.csv")
    summary_json_path = args.input_dir / "summary.json"
    summary_json = json.loads(summary_json_path.read_text(encoding="utf-8")) if summary_json_path.exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.75), facecolor=BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    summary = {
        "panel_a_cumulative": draw_cumulative_panel(axes[0], cumulative_rows, summary_json),
        "panel_b_ablation": draw_ablation_panel(axes[1], ablation_rows),
        "source": {
            "input_dir": str(args.input_dir),
            "cumulative_rows": len(cumulative_rows),
            "ablation_rows": len(ablation_rows),
        },
    }
    fig.subplots_adjust(left=0.070, right=0.992, top=0.735, bottom=0.185, wspace=0.245)
    save_outputs(fig, args.output_dir, args.stem, summary)


if __name__ == "__main__":
    main()
