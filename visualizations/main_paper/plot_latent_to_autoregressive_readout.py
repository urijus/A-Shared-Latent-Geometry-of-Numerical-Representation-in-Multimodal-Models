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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_INPUT_DIR = Path("results/experiments/closing/latent_to_next_digit_readout")
DEFAULT_OUTPUT_DIR = Path("visualizations/main_paper/latent_to_autoregressive_readout")
DEFAULT_STEM = "latent_to_autoregressive_readout"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
MUTED = "#5e696d"
SPINE = "#202426"
GRID = "#d8ddde"
TEAL = "#226a74"
RUST = "#a95642"
PURPLE = "#7a5b98"
GOLD = "#c99a3d"
GREEN = "#5d8f7b"

TASK_COLORS = {
    "T+": TEAL,
    "T-": RUST,
    "I+": PURPLE,
    "I-": GOLD,
}

PRIMARY_CATEGORY = "same_first_diff_second"
LAYERS = [44, 45, 46, 47]
PROB_CONDITIONS = [
    ("matched_Cd_Ld", "matched", TEAL),
    ("mismatch_Cd_Lc", "mismatch", RUST),
    ("C_only_d", "C-only", PURPLE),
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
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.7,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.5,
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


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def style_panel(ax, heading: str, panel: str, panel_x: float = 0.015) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.text(panel_x, 0.985, panel, transform=ax.transAxes, ha="left", va="top", fontsize=10.0)
    ax.text(0.5, 1.035, heading, transform=ax.transAxes, ha="center", va="bottom", fontsize=10.0)


def box(
    ax,
    xy: tuple[float, float],
    text: str,
    *,
    color: str,
    width: float = 0.26,
    height: float = 0.14,
    alpha: float = 0.16,
    fontsize: float = 7.6,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=SPINE,
        facecolor=color,
        alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = SPINE, style: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            linestyle=style,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def draw_panel_a(ax) -> None:
    style_panel(ax, "Mechanism", "(a)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    y_header = 0.73
    y_matched = 0.53
    y_mismatch = 0.35
    y_note = 0.20

    x_intervention = 0.18
    x_tf = 0.50
    x_pred = 0.82
    col_width = 0.235
    row_height = 0.12
    arrow_clearance = 0.018

    def link(y: float, color: str) -> None:
        arrow(
            ax,
            (x_intervention + col_width / 2 + arrow_clearance, y),
            (x_tf - col_width / 2 - arrow_clearance, y),
            color=color,
        )
        arrow(
            ax,
            (x_tf + col_width / 2 + arrow_clearance, y),
            (x_pred - col_width / 2 - arrow_clearance, y),
            color=color,
        )

    box(ax, (x_intervention, y_header), r"intervene at $t_0$", color="#ffffff", width=col_width, height=0.10, alpha=0.92, fontsize=6.35)
    box(ax, (x_tf, y_header), 'shared digit "4"', color="#ffffff", width=col_width, height=0.10, alpha=0.92, fontsize=5.9)
    box(ax, (x_pred, y_header), "next digit", color="#ffffff", width=col_width, height=0.10, alpha=0.92, fontsize=6.35)
    link(y_header, MUTED)

    box(ax, (x_intervention, y_matched), "matched\n" + r"$C\leftarrow49,\ L\leftarrow49$", color=TEAL, width=col_width, height=row_height, alpha=0.14, fontsize=5.95)
    box(ax, (x_intervention, y_mismatch), "mismatch\n" + r"$C\leftarrow49,\ L\leftarrow46$", color=RUST, width=col_width, height=row_height, alpha=0.14, fontsize=5.8)
    box(ax, (x_tf, y_matched), 'TF "4"', color=GREEN, width=col_width, height=row_height, alpha=0.15, fontsize=6.85)
    box(ax, (x_tf, y_mismatch), 'TF "4"', color=GREEN, width=col_width, height=row_height, alpha=0.15, fontsize=6.85)
    box(ax, (x_pred, y_matched), '"9"', color=TEAL, width=col_width, height=row_height, alpha=0.14, fontsize=7.2)
    box(ax, (x_pred, y_mismatch), '"6"', color=RUST, width=col_width, height=row_height, alpha=0.14, fontsize=7.2)

    for y, color in [(y_matched, TEAL), (y_mismatch, RUST)]:
        link(y, color)

    ax.text(0.50, y_note, "intervened KV cache carried forward", ha="center", va="center", fontsize=7.3, color=MUTED)


def task_seed_layer_means(rows: list[dict]) -> dict[str, dict[int, dict]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["mismatch_category"] != PRIMARY_CATEGORY or row["conversion_cosine"] == "":
            continue
        layer = int(row["layer"])
        if layer not in LAYERS:
            continue
        grouped[(row["task_label"], row["das_seed"], layer)].append(float(row["conversion_cosine"]))

    summary: dict[str, dict[int, dict]] = defaultdict(dict)
    for task in TASK_COLORS:
        for layer in LAYERS:
            config_values = [
                mean(values)
                for (task_label, _seed, layer_id), values in grouped.items()
                if task_label == task and layer_id == layer
            ]
            summary[task][layer] = {
                "mean": mean(config_values),
                "sem": sem(config_values),
                "n_configs": len(config_values),
            }
    return summary


def draw_panel_b(ax, rows: list[dict]) -> dict:
    style_panel(ax, "Latent-to-readout conversion", "(b)", panel_x=0.04)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    summary = task_seed_layer_means(rows)

    for task, color in TASK_COLORS.items():
        means = [summary[task][layer]["mean"] for layer in LAYERS]
        errors = [summary[task][layer]["sem"] for layer in LAYERS]
        ax.plot(LAYERS, means, color=color, linewidth=1.6, marker="o", markersize=3.7, label=task, zorder=3)
        ax.fill_between(LAYERS, np.array(means) - np.array(errors), np.array(means) + np.array(errors), color=color, alpha=0.13, linewidth=0)

    ax.axvline(44, color="#9aa1a3", linestyle=(0, (3, 2)), linewidth=0.9)
    ax.text(44.05, -0.11, "first layer with access to\nintervention-affected cache", ha="left", va="bottom", fontsize=6.7, color=MUTED)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Conversion cosine")
    ax.set_ylim(-0.16, 1.02)
    ax.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.9, handlelength=1.5)
    return summary


def probability_summary(rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for condition, _label, _color in PROB_CONDITIONS:
        per_config: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["mismatch_category"] != PRIMARY_CATEGORY or row["condition"] != condition:
                continue
            key = (row["task_label"], row["das_seed"])
            per_config[key]["prob_units_d"].append(float(row["prob_units_d"]))
            per_config[key]["prob_units_c"].append(float(row["prob_units_c"]))
            per_config[key]["argmax_c"].append(1.0 if row["t1_argmax_is_units_c"].lower() == "true" else 0.0)
        summary[condition] = {}
        for metric in ["prob_units_d", "prob_units_c", "argmax_c"]:
            values = [mean(config_values[metric]) for config_values in per_config.values()]
            summary[condition][metric] = {
                "mean": mean(values),
                "sem": sem(values),
                "n_configs": len(values),
            }
    return summary


def summarize_sanity(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    if not rows:
        return None
    return {
        "max_U_T_L_norm": max(float(row["U_T_L_norm"]) for row in rows if "U_T_L_norm" in row),
        "all_patched_kv_cache_reused": all(bool(row.get("patched_kv_cache_reused")) for row in rows),
        "any_t1_patch_applied": any(bool(row.get("t1_patch_applied")) for row in rows),
        "n_configs": len(rows),
    }


def summarize_random_controls(rows: list[dict]) -> dict:
    per_config: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["mismatch_category"] != PRIMARY_CATEGORY or int(row["layer"]) != 47:
            continue
        per_config[(row["task_label"], row["das_seed"])].append(float(row["prob_units_c"]))
    values = [mean(config_values) for config_values in per_config.values()]
    if not values:
        return {"n_configs": 0}
    return {"prob_units_c_mean": mean(values), "prob_units_c_sem": sem(values), "n_configs": len(values)}


def label_bars(ax, bars, errors: list[float], x_nudge: float = 0.0) -> None:
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        if height < 0.015:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_nudge,
            height + error + 0.018,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.9,
        )


def draw_panel_c(ax, rows: list[dict]) -> dict:
    style_panel(ax, "Next token probabilities\nafter cached propagation", "(c)")
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    summary = probability_summary(rows)

    x = np.arange(2, dtype=float)
    width = 0.23
    offsets = [-width, 0.0, width]
    metrics = [("prob_units_d", r"$P(y_2)$"), ("prob_units_c", r"$P(\tilde{y}_2)$")]

    for offset, (condition, label, color) in zip(offsets, PROB_CONDITIONS):
        means = [summary[condition][metric]["mean"] for metric, _display in metrics]
        errors = [summary[condition][metric]["sem"] for metric, _display in metrics]
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
        label_bars(ax, bars, errors, x_nudge=offset * 0.05)

    ax.set_xticks(x, [display for _metric, display in metrics])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1.16)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=3, columnspacing=0.55, handlelength=1.0, handletextpad=0.28, fontsize=6.8)
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

    conversion_rows = read_csv(args.input_dir / "conversion_metrics.csv")
    probability_rows = read_csv(args.input_dir / "second_digit_logits.csv")
    sanity_path = args.input_dir / "sanity_checks.json"
    random_path = args.input_dir / "random_latent_controls.csv"
    random_rows = read_csv(random_path) if random_path.exists() else []

    fig = plt.figure(figsize=(9.45, 3.35), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.16, 1.08])
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    fig.patch.set_facecolor(BACKGROUND)

    draw_panel_a(axes[0])
    panel_b = draw_panel_b(axes[1], conversion_rows)
    panel_c = draw_panel_c(axes[2], probability_rows)

    summary = {
        "input_dir": str(args.input_dir),
        "primary_mismatch_category": PRIMARY_CATEGORY,
        "panel_b_conversion": panel_b,
        "panel_c_probabilities": panel_c,
        "sanity_checks": summarize_sanity(sanity_path),
        "random_same_tens_control": summarize_random_controls(random_rows),
    }

    fig.subplots_adjust(left=0.045, right=0.99, top=0.82, bottom=0.23, wspace=0.29)
    save_outputs(fig, args.output_dir, args.stem, summary)


if __name__ == "__main__":
    main()
