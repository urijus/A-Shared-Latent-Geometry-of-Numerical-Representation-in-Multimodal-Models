from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = Path("results/experiments/closing/latent_readout_causal_interaction")
DEFAULT_OUTPUT_DIR = Path("visualizations/main_paper/latent_identity_causal_swap")
DEFAULT_STEM = "latent_identity_causal_swap"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
SPINE = "#202426"
GRID = "#d8ddde"
TEAL = "#226a74"
RUST = "#a95642"
PURPLE = "#7a5b98"
GOLD = "#c99a3d"

FIRST_DIGIT_CATEGORY = "different_first_digit"
VALUE_SPECIFIC_CATEGORY = "same_first_diff_second"

FIRST_DIGIT_CONDITIONS = [
    ("C_only_d", "$\\Delta h_D^{b\\to y}$"),
    ("L_only_d", "$\\Delta h_L^{b\\to y}$"),
    ("mismatch_Cd_Lc", "$\\Delta h_D^{b\\to y}+\\Delta h_L^{b\\to\\tilde{y}}$"),
]

FIRST_DIGIT_TARGETS = [
    ("prob_d_first", "$P(y_1)$", TEAL),
    ("prob_c_first", "$P(\\tilde{y}_1)$", RUST),
]

VALUE_SPECIFIC_CONDITIONS = [
    ("matched_Cd_Ld", "matched\n$D_y + L_y$"),
    ("mismatch_Cd_Lc", "latent mismatch\n$D_y + L_{\\tilde{y}}$"),
    ("random_latent_mismatch", "random latent\n$D_y + L_r$"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
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
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.9,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: object) -> float:
    return 1.0 if str(value).strip().lower() == "true" else 0.0


def float_value(value: object) -> float:
    return float(value)


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def config_key(row: dict) -> tuple[str, str]:
    return row["task_key"], row["das_seed"]


def grouped_metric(
    rows: list[dict],
    condition: str,
    metric: str,
    *,
    mismatch_category: str,
    boolean: bool = True,
) -> dict:
    per_config: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["mismatch_category"] != mismatch_category or row["condition"] != condition:
            continue
        value = bool_value(row[metric]) if boolean else float_value(row[metric])
        per_config[config_key(row)].append(value)
    config_means = [mean(values) for values in per_config.values() if values]
    return {
        "mean": mean(config_means),
        "sem": sem(config_means),
        "n_configs": len(config_means),
        "n_rows": sum(len(values) for values in per_config.values()),
    }


def style_panel(ax, heading: str, panel: str, *, title_y: float | None = None) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, ha="left", va="top", fontsize=10.0)
    title_y = title_y if title_y is not None else (1.035 if "\n" not in heading else 1.022)
    ax.text(
        0.5,
        title_y,
        heading,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.2,
        linespacing=1.05,
    )


def label_bars(ax, bars, errors: list[float], dy: float = 0.018, x_nudge: float = 0.0) -> None:
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        if height < 0.015:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_nudge,
            height + error + dy,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )


def draw_first_digit_panel(ax, rows: list[dict], *, panel: str) -> dict:
    style_panel(ax, "Immediate digit follows D", panel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)

    x = np.arange(len(FIRST_DIGIT_CONDITIONS), dtype=float)
    width = 0.28
    offsets = [-width / 2, width / 2]
    summary: dict[str, dict] = {}

    for offset, (metric, label, color) in zip(offsets, FIRST_DIGIT_TARGETS):
        means = []
        errors = []
        for condition, _display in FIRST_DIGIT_CONDITIONS:
            stats = grouped_metric(
                rows,
                condition,
                metric,
                mismatch_category=FIRST_DIGIT_CATEGORY,
                boolean=False,
            )
            means.append(stats["mean"])
            errors.append(stats["sem"])
            summary.setdefault(condition, {})[metric] = stats
        bars = ax.bar(
            x + offset,
            means,
            yerr=errors,
            width=width,
            color=color,
            edgecolor=SPINE,
            linewidth=0.75,
            capsize=2.0,
            error_kw={"elinewidth": 0.75, "ecolor": TEXT, "capthick": 0.75},
            label=label,
            zorder=3,
        )
        label_bars(ax, bars, errors, x_nudge=-0.022 if metric == "prob_d_first" else 0.022)

    ax.set_xticks(x, [display for _condition, display in FIRST_DIGIT_CONDITIONS])
    ax.tick_params(axis="x", labelsize=7.8)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.72, len(FIRST_DIGIT_CONDITIONS) - 0.5)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=2, handlelength=1.25)
    return summary


def draw_value_specific_panel(ax, rows: list[dict], *, panel: str) -> dict:
    style_panel(ax, "Latent identity redirects continuation", panel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)

    x = np.arange(len(VALUE_SPECIFIC_CONDITIONS), dtype=float)
    width = 0.27
    offsets = [-width / 2, width / 2]
    targets = [("ar_iia_to_d", "toward $y$", TEAL), ("ar_iia_to_c", "toward $\\tilde{y}$", RUST)]
    summary: dict[str, dict] = {}

    for offset, (metric, label, color) in zip(offsets, targets):
        means = []
        errors = []
        for condition, _display in VALUE_SPECIFIC_CONDITIONS:
            stats = grouped_metric(
                rows,
                condition,
                metric,
                mismatch_category=VALUE_SPECIFIC_CATEGORY,
                boolean=True,
            )
            means.append(stats["mean"])
            errors.append(stats["sem"])
            summary.setdefault(condition, {})[metric] = stats
        bars = ax.bar(
            x + offset,
            means,
            yerr=errors,
            width=width,
            color=color,
            edgecolor=SPINE,
            linewidth=0.75,
            capsize=2.0,
            error_kw={"elinewidth": 0.75, "ecolor": TEXT, "capthick": 0.75},
            label=label,
            zorder=3,
        )
        label_bars(ax, bars, errors, x_nudge=-0.025 if metric == "ar_iia_to_d" else 0.025)

    ax.set_xticks(x, [display for _condition, display in VALUE_SPECIFIC_CONDITIONS])
    ax.set_ylabel("Autoregressive IIA")
    ax.set_ylim(0, 0.98)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=2, handlelength=1.25)
    return summary


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

    intervention_rows = read_csv(args.input_dir / "intervention_metrics.csv")
    digit_rows = read_csv(args.input_dir / "digit_level_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.15), facecolor=BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)

    summary = {
        "input_dir": str(args.input_dir),
        "panel_a_first_digit_category": FIRST_DIGIT_CATEGORY,
        "panel_a_l_only_note": "The available result files contain L_only_d = Delta h_L^{b->y}; no L_only_c / Delta h_L^{b->tilde y} condition was saved.",
        "panel_b_value_specific_category": VALUE_SPECIFIC_CATEGORY,
        "panel_a_first_digit": draw_first_digit_panel(axes[0], digit_rows, panel="(a)"),
        "panel_b_full_answer": draw_value_specific_panel(axes[1], intervention_rows, panel="(b)"),
    }

    fig.subplots_adjust(left=0.072, right=0.988, top=0.845, bottom=0.25, wspace=0.24)
    save_outputs(fig, args.output_dir, args.stem, summary)


if __name__ == "__main__":
    main()
