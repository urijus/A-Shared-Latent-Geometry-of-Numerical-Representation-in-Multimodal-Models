"""Render the main Procrustes causal-transport summary figure.

The figure is local-only: it reads the JSONL summaries already produced by the
causal-transfer, Procrustes ablation, scaled-displacement, and rank-sweep jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAUSAL_DIR = REPO_ROOT / "results" / "final_exps" / "causal_tranfer"
DEFAULT_PROCRUSTES_DIR = REPO_ROOT / "results" / "final_exps" / "procrustes"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "procrustes" / "paper"
DEFAULT_FACTORIZED_SUMMARY = DEFAULT_PROCRUSTES_DIR / "factorized_paths" / "factorized_paths_summary.jsonl"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"
TRANSFER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_transfer",
    ["#d8dee0", "#b7d1ce", "#7db9aa", "#419287", "#226a74"],
)

TASK_ORDER = ["text_add", "text_sub", "image_add", "image_sub"]
TASK_LABELS = {
    "text_add": "T+",
    "image_add": "I+",
    "text_sub": "T-",
    "image_sub": "I-",
}
SAME_OPERATION_DIRECTIONS = [
    ("text_add", "image_add"),
    ("image_add", "text_add"),
    ("text_sub", "image_sub"),
    ("image_sub", "text_sub"),
]
OFF_DIAGONAL_GROUPS = [
    (
        "Cross modality",
        [
            ("image_add", "text_add"),
            ("image_sub", "text_sub"),
            ("text_add", "image_add"),
            ("text_sub", "image_sub"),
        ],
    ),
    (
        "Cross operation",
        [
            ("text_add", "text_sub"),
            ("text_sub", "text_add"),
            ("image_add", "image_sub"),
            ("image_sub", "image_add"),
        ],
    ),
    (
        "Cross modality-operation",
        [
            ("image_sub", "text_add"),
            ("image_add", "text_sub"),
            ("text_add", "image_sub"),
            ("text_sub", "image_add"),
        ],
    ),
]
DIRECTION_LABELS = {
    ("text_add", "image_add"): "T+ -> I+",
    ("image_add", "text_add"): "I+ -> T+",
    ("text_sub", "image_sub"): "T- -> I-",
    ("image_sub", "text_sub"): "I- -> T-",
    ("text_add", "text_sub"): "T+ -> T-",
    ("text_sub", "text_add"): "T- -> T+",
    ("image_add", "image_sub"): "I+ -> I-",
    ("image_sub", "image_add"): "I- -> I+",
    ("text_add", "image_sub"): "T+ -> I-",
    ("image_sub", "text_add"): "I- -> T+",
    ("text_sub", "image_add"): "T- -> I+",
    ("image_add", "text_sub"): "I+ -> T-",
}
DIRECTION_COLORS = {
    ("text_add", "image_add"): "#226a74",
    ("image_add", "text_add"): "#5d6f9f",
    ("text_sub", "image_sub"): "#a95642",
    ("image_sub", "text_sub"): "#b8872d",
    ("text_add", "text_sub"): "#7a5b98",
    ("text_sub", "text_add"): "#9c6f33",
    ("image_add", "image_sub"): "#6d7477",
    ("image_sub", "image_add"): "#3f7f49",
    ("text_add", "image_sub"): "#226a74",
    ("image_sub", "text_add"): "#5d6f9f",
    ("text_sub", "image_add"): "#a95642",
    ("image_add", "text_sub"): "#b8872d",
}
GROUP_COLORS = {
    "Cross modality": "#226a74",
    "Cross operation": "#7a5b98",
    "Cross modality-operation": "#b8872d",
}
METHODS = [
    ("direct", "Direct", None, "#6d7477"),
    ("basic", "Q", "basic", "#5d6f9f"),
    ("alpha_identity", r"$\alpha I$", "alpha_identity", "#b8872d"),
    ("random_orthogonal", r"$Q_{\rm rand}$", "random_orthogonal", "#9aa1a3"),
    ("scaled_random_orthogonal", r"$\alpha Q_{\rm rand}$", "scaled_random_orthogonal", "#7a5b98"),
    ("scaled_displacement", r"$\alpha Q$", "scaled_displacement", "#226a74"),
    ("unrestricted_linear", r"$A_{\rm lin}$", "unrestricted_linear", "#3f7f49"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal_dir", type=Path, default=DEFAULT_CAUSAL_DIR)
    parser.add_argument("--procrustes_dir", type=Path, default=DEFAULT_PROCRUSTES_DIR)
    parser.add_argument("--factorized_summary", type=Path, default=DEFAULT_FACTORIZED_SUMMARY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="main_four_panel_procrustes_results")
    return parser.parse_args()


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.7,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
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


def read_matrix(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["source_label"], row["destination_label"]): row for row in load_rows(path)}


def read_factorized_scaled_matrix(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    families = {
        "same_operation_transition_mean",
        "operation_only_transition_mean",
        "cross_operation_factorized",
    }
    for row in load_rows(path):
        if row.get("transport") != "direct":
            continue
        if row.get("fit_mode") != "transition_mean":
            continue
        if row.get("test_family") not in families:
            continue
        rows[(row["source_label"], row["destination_label"])] = {
            "source_label": row["source_label"],
            "destination_label": row["destination_label"],
            "mean": row.get("destination_normalized_transfer_mean"),
            "std": row.get("destination_normalized_transfer_std"),
            "n": row.get("n"),
            "alpha_mean": row.get("alpha_mean"),
            "alpha_std": row.get("alpha_std"),
            "test_family": row.get("test_family"),
            "transport": row.get("transport"),
            "fit_mode": row.get("fit_mode"),
        }
    return rows


def task_parts(task: str) -> tuple[str, str]:
    modality, operation = task.split("_", 1)
    return modality, operation


def relation_group(source: str, destination: str) -> str:
    source_modality, source_operation = task_parts(source)
    destination_modality, destination_operation = task_parts(destination)
    if source == destination:
        return "Self"
    if source_operation == destination_operation:
        return "Cross modality"
    if source_modality == destination_modality:
        return "Cross operation"
    return "Cross modality-operation"


def all_off_diagonal_directions() -> list[tuple[str, str]]:
    return [direction for _group, directions in OFF_DIAGONAL_GROUPS for direction in directions]


def panel_a_pairwise_matrix(
    factorized_rows: dict[tuple[str, str], dict],
    fallback_same_operation_rows: dict[tuple[str, str], dict],
) -> dict[tuple[str, str], dict]:
    rows = {}
    for key in all_off_diagonal_directions():
        src, dst = key
        if relation_group(src, dst) == "Cross modality":
            row = fallback_same_operation_rows.get(key)
            if row is not None:
                rows[key] = {
                    **row,
                    "test_family": "same_operation_scaled_displacement",
                    "panel_a_source": "scaled_displacement_matrix",
                }
                continue
        row = factorized_rows.get(key)
        if row is not None:
            rows[key] = {**row, "panel_a_source": "factorized_transition_mean_direct"}
    return rows


def panel_d_alpha_rows(
    factorized_rows: dict[tuple[str, str], dict],
    scaled_alpha_rows: dict[tuple[str, str], dict],
) -> dict[tuple[str, str], dict]:
    rows = {}
    for key in all_off_diagonal_directions():
        src, dst = key
        if relation_group(src, dst) == "Cross modality":
            row = scaled_alpha_rows.get(key)
            if row is not None:
                rows[key] = {
                    "mean": row.get("mean"),
                    "std": row.get("std"),
                    "n": row.get("n"),
                    "alpha_source": "scaled_displacement_alpha_matrix",
                    "test_family": "same_operation_scaled_displacement",
                }
                continue
        row = factorized_rows.get(key)
        if row is not None:
            rows[key] = {
                "mean": row.get("alpha_mean"),
                "std": row.get("alpha_std"),
                "n": row.get("n"),
                "alpha_source": "factorized_transition_mean_direct",
                "test_family": row.get("test_family"),
            }
    return rows


def matrix_values(rows: dict[tuple[str, str], dict]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    std = np.full_like(mean, np.nan)
    for i, src in enumerate(TASK_ORDER):
        for j, dst in enumerate(TASK_ORDER):
            row = rows.get((src, dst))
            if row is None:
                continue
            mean[i, j] = finite(row.get("mean"))
            std[i, j] = finite(row.get("std"))
    return mean, std


def setup_axis(ax, title: str) -> None:
    ax.set_title(title, pad=7)
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)


def setup_matrix_axis(ax, title: str) -> None:
    ax.set_title(title, pad=7)
    ax.set_facecolor(PANEL_BG)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)


def group_title(fig, axes, title: str, *, y_pad: float = 0.035) -> None:
    bbox = Bbox.union([ax.get_position() for ax in axes])
    fig.text(
        (bbox.x0 + bbox.x1) / 2,
        bbox.y1 + y_pad,
        title,
        ha="center",
        va="bottom",
        fontsize=11.2,
    )


def draw_transfer_matrix(
    ax,
    mean: np.ndarray,
    title: str,
    norm: mcolors.Normalize,
    std: np.ndarray | None = None,
) -> None:
    setup_matrix_axis(ax, title)
    cmap = TRANSFER_CMAP.copy()
    cmap.set_bad("#eceff0")
    image = ax.imshow(np.ma.masked_invalid(mean), cmap=cmap, norm=norm)
    labels = [TASK_LABELS[task] for task in TASK_ORDER]
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="both", length=0, labelsize=9.8)
    ax.set_xlabel("Destination", fontsize=11.5, labelpad=6)
    ax.set_ylabel("Source", fontsize=11.5, labelpad=7)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            value = mean[i, j]
            if i == j:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#d7dcde", edgecolor="none")
                )
                continue
            if not np.isfinite(value):
                ax.text(j, i, "missing", ha="center", va="center", fontsize=6.9, color="#788083")
                continue
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.48 else "#181818"
            label = f"{value:.2f}"
            if std is not None and np.isfinite(std[i, j]):
                label = f"{value:.2f}\n+/-{std[i, j]:.2f}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8.3 if "\n" in label else 8.8,
                linespacing=1.18,
                color=color,
            )
    return image


def draw_destination_normalized_matrix(
    ax,
    rows: dict[tuple[str, str], dict],
    *,
    title: str = "A. Direct reuse",
) -> dict:
    mean, std = matrix_values(rows)
    setup_matrix_axis(ax, title)
    cmap = TRANSFER_CMAP.copy()
    display_mean = np.clip(mean, 0.0, 1.0)
    image = ax.imshow(display_mean, cmap=cmap, vmin=0.0, vmax=1.0)

    labels = [TASK_LABELS[task] for task in TASK_ORDER]
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="both", length=0, labelsize=9.8)
    ax.set_xlabel("Destination", fontsize=11.5, labelpad=6)
    ax.set_ylabel("Source", fontsize=11.5, labelpad=7)

    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            value = display_mean[i, j]
            rgba = cmap(value)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.48 else "#181818"
            ax.text(
                j,
                i - 0.09,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color=color,
            )
            ax.text(
                j,
                i + 0.16,
                fr"$\pm${std[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.028)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.8)

    entries = {}
    for src in TASK_ORDER:
        for dst in TASK_ORDER:
            row = rows.get((src, dst))
            if row is None:
                continue
            entries[f"{TASK_LABELS[src]} -> {TASK_LABELS[dst]}"] = {
                "mean": finite(row.get("mean")),
                "std": finite(row.get("std")),
                "n": row.get("n"),
            }
    return {"destination_normalized_transfer_matrix": entries}


def direction_values(matrix_rows: dict[tuple[str, str], dict]) -> list[float]:
    return [finite(matrix_rows[(src, dst)].get("mean")) for src, dst in SAME_OPERATION_DIRECTIONS]


def draw_panel_a(
    fig,
    subspec,
    scaled_rows,
    *,
    title: str = "A. Pairwise transport transfer",
    x_shift: float = -0.035,
) -> dict:
    ax = fig.add_subplot(subspec)
    scaled_mean, scaled_std = matrix_values(scaled_rows)
    finite_values = scaled_mean[np.isfinite(scaled_mean)]
    vmax = max(1.1, float(np.nanmax(finite_values)) if finite_values.size else 1.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    image = draw_transfer_matrix(ax, scaled_mean, title, norm, scaled_std)
    ax.set_anchor("W")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.028)
    for moved_ax in (ax, colorbar.ax):
        box = moved_ax.get_position()
        moved_ax.set_position([box.x0 + x_shift, box.y0, box.width, box.height])
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.8)
    entries = {}
    for src in TASK_ORDER:
        for dst in TASK_ORDER:
            row = scaled_rows.get((src, dst))
            if row is None:
                continue
            entries[f"{TASK_LABELS[src]} -> {TASK_LABELS[dst]}"] = {
                "mean": finite(row.get("mean")),
                "std": finite(row.get("std")),
                "n": row.get("n"),
                "alpha_mean": row.get("alpha_mean"),
                "alpha_std": row.get("alpha_std"),
                "test_family": row.get("test_family"),
                "panel_a_source": row.get("panel_a_source"),
            }
    return {"pairwise_transport_matrix": entries}


def draw_pairwise_scaled_transport(
    ax,
    scaled_rows,
    *,
    title: str = "C. Pairwise scaled transport",
) -> dict:
    scaled_mean, scaled_std = matrix_values(scaled_rows)
    finite_values = scaled_mean[np.isfinite(scaled_mean)]
    vmax = max(1.1, float(np.nanmax(finite_values)) if finite_values.size else 1.0)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    image = draw_transfer_matrix(ax, scaled_mean, title, norm, scaled_std)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.026)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.8)
    entries = {}
    for src in TASK_ORDER:
        for dst in TASK_ORDER:
            row = scaled_rows.get((src, dst))
            if row is None:
                continue
            entries[f"{TASK_LABELS[src]} -> {TASK_LABELS[dst]}"] = {
                "mean": finite(row.get("mean")),
                "std": finite(row.get("std")),
                "n": row.get("n"),
                "alpha_mean": row.get("alpha_mean"),
                "alpha_std": row.get("alpha_std"),
                "test_family": row.get("test_family"),
                "panel_a_source": row.get("panel_a_source"),
            }
    return {"pairwise_transport_matrix": entries}


def draw_panel_b(
    ax,
    matrix_by_method: dict[str, dict[tuple[str, str], dict]],
    *,
    title: str = "B. Transformation ablation",
) -> dict:
    setup_axis(ax, title)
    x = np.arange(len(METHODS), dtype=float)
    means = []
    stds = []
    per_direction = {}
    for key, _label, _folder, _color in METHODS:
        vals = np.asarray(direction_values(matrix_by_method[key]), dtype=float)
        per_direction[key] = vals.tolist()
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals)))
    colors = [color for _key, _label, _folder, color in METHODS]
    ax.bar(x, means, color=colors, edgecolor=SPINE, linewidth=0.8, width=0.68, zorder=3)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=TEXT, elinewidth=0.8, capsize=2.3, zorder=3)
    offsets = np.linspace(-0.20, 0.20, len(SAME_OPERATION_DIRECTIONS))
    for offset, direction in zip(offsets, SAME_OPERATION_DIRECTIONS):
        vals = [per_direction[key][SAME_OPERATION_DIRECTIONS.index(direction)] for key, *_ in METHODS]
        ax.scatter(
            x + offset,
            vals,
            s=18,
            marker="o",
            color=DIRECTION_COLORS[direction],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.92,
            zorder=4,
        )
    ax.set_xticks(x, [label for _key, label, _folder, _color in METHODS], rotation=0)
    ax.set_ylabel("Mean normalized transfer")
    ax.set_ylim(0.0, max(1.12, max(means) + max(stds) + 0.10))
    ax.axhline(1.0, color="#9aa1a3", linewidth=0.85, linestyle=":")
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5.5,
            markerfacecolor=DIRECTION_COLORS[direction],
            markeredgecolor="white",
            label=DIRECTION_LABELS[direction].replace("->", "\u2192"),
        )
        for direction in SAME_OPERATION_DIRECTIONS
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.25,
    )
    return {"method_means": dict(zip([key for key, *_ in METHODS], means)), "method_values": per_direction}


def rank_groups(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["source_label"], row["destination_label"])
        if key not in SAME_OPERATION_DIRECTIONS:
            continue
        groups.setdefault(key, []).append(row)
    for parts in groups.values():
        parts.sort(key=lambda row: int(row["rank"]))
    return groups


def first_threshold_rank(parts: list[dict], threshold: float) -> int | None:
    full = finite(parts[-1]["destination_normalized_transfer_mean"])
    target = threshold * full
    for row in parts:
        if finite(row["destination_normalized_transfer_mean"]) >= target:
            return int(row["rank"])
    return None


def draw_panel_c(ax, rank_rows: list[dict]) -> dict:
    setup_axis(ax, "C. Rank-m causal recovery")
    groups = rank_groups(rank_rows)
    summary = {}
    for direction in SAME_OPERATION_DIRECTIONS:
        parts = groups.get(direction, [])
        if not parts:
            continue
        ranks = np.asarray([int(row["rank"]) for row in parts], dtype=float)
        scores = np.asarray([finite(row["destination_normalized_transfer_mean"]) for row in parts], dtype=float)
        stds = np.asarray([finite(row["destination_normalized_transfer_std"]) for row in parts], dtype=float)
        color = DIRECTION_COLORS[direction]
        ax.plot(
            ranks,
            scores,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.2,
            label=DIRECTION_LABELS[direction].replace("->", "\u2192"),
            zorder=3,
        )
        ax.fill_between(ranks, scores - stds, scores + stds, color=color, alpha=0.12, linewidth=0)
        thresholds = {str(int(t * 100)): first_threshold_rank(parts, t) for t in (0.8, 0.9, 0.95)}
        marker_shapes = {"80": "o", "90": "^", "95": "s"}
        for name, rank in thresholds.items():
            if rank is None:
                continue
            row = next(row for row in parts if int(row["rank"]) == rank)
            score = finite(row["destination_normalized_transfer_mean"])
            ax.scatter(
                [rank],
                [score],
                marker=marker_shapes[name],
                s=42,
                color=color,
                edgecolor=SPINE,
                linewidth=0.6,
                zorder=5,
            )
        summary[DIRECTION_LABELS[direction]] = {
            "full": float(scores[-1]),
            "rank_at_fraction_of_full": thresholds,
        }
    ax.axhline(1.0, color="#9aa1a3", linewidth=0.85, linestyle=":")
    ax.set_xlim(1, 22)
    ax.set_xticks([1, 5, 10, 15, 20, 22])
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Retained rank m")
    ax.set_ylabel("Normalized transfer")
    direction_legend = ax.legend(
        loc="lower right",
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.25,
    )
    ax.add_artist(direction_legend)
    threshold_handles = [
        mlines.Line2D([], [], color=TEXT, marker="o", linestyle="", markersize=5, label="80%"),
        mlines.Line2D([], [], color=TEXT, marker="^", linestyle="", markersize=5, label="90%"),
        mlines.Line2D([], [], color=TEXT, marker="s", linestyle="", markersize=5, label="95%"),
    ]
    ax.legend(
        handles=threshold_handles,
        title="Fraction of full-rank effect",
        loc="upper left",
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.2,
        handletextpad=0.35,
    )
    return summary


def draw_panel_d(ax, alpha_rows: dict[tuple[str, str], dict]) -> dict:
    setup_axis(ax, "D. Fitted scales")
    labels = []
    values = []
    stds = []
    colors = []
    y_positions = []
    group_centers = {}
    group_label_y = {}
    group_summaries = {}
    y = 0.0
    for group_name, directions in OFF_DIAGONAL_GROUPS:
        start = y
        group_label_y[group_name] = start - 0.72
        group_entries = {}
        for direction in directions:
            row = alpha_rows.get(direction, {})
            labels.append(DIRECTION_LABELS[direction].replace("->", "\u2192"))
            values.append(finite(row.get("mean")))
            stds.append(finite(row.get("std")))
            colors.append(GROUP_COLORS[group_name])
            y_positions.append(y)
            group_entries[DIRECTION_LABELS[direction]] = {
                "mean": finite(row.get("mean")),
                "std": finite(row.get("std")),
                "n": row.get("n"),
                "test_family": row.get("test_family"),
                "alpha_source": row.get("alpha_source"),
            }
            y += 1.0
        group_centers[group_name] = (start + y - 1.0) / 2.0
        group_summaries[group_name] = group_entries
        y += 0.92

    values = np.asarray(values, dtype=float)
    stds = np.asarray(stds, dtype=float)
    y_positions = np.asarray(y_positions, dtype=float)
    ax.barh(
        y_positions,
        values,
        xerr=stds,
        capsize=2.0,
        color=colors,
        edgecolor=SPINE,
        linewidth=0.75,
        height=0.58,
        zorder=3,
    )
    ax.axvline(1.0, color="#9aa1a3", linewidth=0.9, linestyle=":")
    for yi, value in zip(y_positions, values):
        if np.isfinite(value):
            ax.text(value + 0.035, yi, f"{value:.2f}", ha="left", va="center", fontsize=8.0)
    ax.set_yticks(y_positions, labels)
    ax.tick_params(axis="y", labelsize=7.7, pad=2)
    ax.set_xlabel(r"Scale $\alpha$")
    finite_hi = values + np.nan_to_num(stds, nan=0.0)
    ax.set_xlim(0.0, max(1.9, float(np.nanmax(finite_hi)) + 0.26))
    ax.set_ylim(y_positions[-1] + 0.82, -1.08)
    ax.grid(True, axis="x", color=GRID, linewidth=0.65, alpha=0.75)
    ax.grid(False, axis="y")
    for group_name, label_y in group_label_y.items():
        ax.text(
            0.02,
            label_y,
            group_name,
            ha="left",
            va="center",
            fontsize=8.0,
            color=GROUP_COLORS[group_name],
        )
    return group_summaries


def save_outputs(fig, output_dir: Path, stem: str, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    json_path = output_dir / f"{stem}_summary.json"
    fig.savefig(png, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")
    print(f"Saved {json_path}")


def draw_standard_figure(panel_a_rows, matrix_by_method, alpha_rows, rank_rows, direct_rows):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.5, 7.2),
        facecolor=BACKGROUND,
        gridspec_kw={
            "width_ratios": [1.0, 1.15],
            "height_ratios": [1.0, 1.0],
        },
    )

    fig.patch.set_facecolor(BACKGROUND)
    axes = axes.ravel()

    summary = {
        # TOP LEFT -> Panel B
        "panel_b": draw_panel_b(
            axes[0],
            matrix_by_method,
            title="B. Transformation ablation",
        ),

        # TOP RIGHT -> Panel A
        "panel_a": draw_destination_normalized_matrix(
            axes[1],
            direct_rows,
            title="A. Direct reuse",
        ),

        # BOTTOM LEFT -> Panel C
        "panel_c": draw_pairwise_scaled_transport(
            axes[2],
            panel_a_rows,
            title="C. Pairwise scaled transport",
        ),

        # BOTTOM RIGHT -> Panel D
        "panel_d": draw_panel_d(
            axes[3],
            alpha_rows,
        ),

        "task_order": [TASK_LABELS[task] for task in TASK_ORDER],
        "directions": [DIRECTION_LABELS[d] for d in all_off_diagonal_directions()],
        "layout": "two_by_two",
    }

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.94,
        bottom=0.09,
        wspace=0.34,
        hspace=0.34,
    )

    return fig, summary

def draw_wide_matrix_figure(panel_a_rows, matrix_by_method, alpha_rows, rank_rows):
    fig = plt.figure(figsize=(11.4, 9.8))
    fig.patch.set_facecolor(BACKGROUND)
    grid = fig.add_gridspec(
        3,
        2,
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.065,
        wspace=0.22,
        hspace=0.36,
        height_ratios=[1.08, 0.95, 1.05],
    )
    summary = {
        "panel_b": draw_panel_a(fig, grid[0, :], panel_a_rows, title="B. Pairwise transport transfer"),
        "panel_a": draw_panel_b(
            fig.add_subplot(grid[1, 0]),
            matrix_by_method,
            title="A. Transformation ablation",
        ),
        "panel_d": draw_panel_d(fig.add_subplot(grid[1, 1]), alpha_rows),
        "panel_c": draw_panel_c(fig.add_subplot(grid[2, :]), rank_rows),
        "task_order": [TASK_LABELS[task] for task in TASK_ORDER],
        "directions": [DIRECTION_LABELS[d] for d in all_off_diagonal_directions()],
        "layout": "wide_matrix_headline",
    }
    return fig, summary


def main() -> None:
    args = parse_args()
    style_matplotlib()

    direct_rows = read_matrix(args.causal_dir / "destination_normalized_transfer_matrix.jsonl")
    scaled_rows = read_matrix(args.procrustes_dir / "scaled_displacement" / "destination_normalized_transfer_matrix.jsonl")
    factorized_rows = read_factorized_scaled_matrix(args.factorized_summary)
    panel_a_rows = panel_a_pairwise_matrix(factorized_rows, scaled_rows)
    matrix_by_method = {"direct": direct_rows}
    for key, _label, folder, _color in METHODS:
        if folder is None:
            continue
        matrix_by_method[key] = read_matrix(
            args.procrustes_dir / folder / "destination_normalized_transfer_matrix.jsonl"
        )
    scaled_alpha_rows = read_matrix(args.procrustes_dir / "scaled_displacement" / "alpha_matrix.jsonl")
    alpha_rows = panel_d_alpha_rows(factorized_rows, scaled_alpha_rows)
    rank_rows = load_rows(args.procrustes_dir / "rank_sweep" / "rank_sweep_summary.jsonl")

    fig, summary = draw_standard_figure(panel_a_rows, matrix_by_method, alpha_rows, rank_rows, direct_rows)
    save_outputs(fig, args.output_dir, args.stem, summary)

    wide_fig, wide_summary = draw_wide_matrix_figure(
        panel_a_rows, matrix_by_method, alpha_rows, rank_rows
    )
    save_outputs(wide_fig, args.output_dir, f"{args.stem}_wide_matrix", wide_summary)


if __name__ == "__main__":
    main()
