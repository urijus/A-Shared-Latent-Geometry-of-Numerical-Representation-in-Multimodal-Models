"""Render polished Procrustes PCA geometry figures from saved JSONL rows.

This script is local-only: it reads the rows produced by
src.experiments.cross_condition_transfer.procrustes.pca_plots.pca_plots and does not load the
model or re-extract activations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Ellipse
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "final_exps" / "procrustes" / "pca_plots"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "procrustes" / "pca_plots" / "paper"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
GRID = "#d8ddde"
SPINE = "#202426"
TASK_COLORS = {
    "text:addition": "#226a74",
    "text:subtraction": "#a95642",
    "image:addition": "#5d6f9f",
    "image:subtraction": "#b8872d",
}
TASK_LABELS = {
    "text:addition": "T+",
    "text:subtraction": "T-",
    "image:addition": "I+",
    "image:subtraction": "I-",
}
MARKERS = {"text": "o", "image": "^"}
TASK_MARKERS = {
    "text:addition": "o",
    "text:subtraction": "s",
    "image:addition": "^",
    "image:subtraction": "D",
}
TRANSPORT_COLORS = {
    "true": "#202426",
    "transported": "#226a74",
    "direct": "#226a74",
    "operation_first": "#a95642",
    "modality_first": "#5d6f9f",
}
TRANSPORT_LABELS = {
    "true": "True",
    "transported": "Transported",
    "direct": "Direct",
    "operation_first": "Operation first",
    "modality_first": "Modality first",
}
TRANSITION_COLORS = ["#226a74", "#a95642", "#5d6f9f", "#b8872d", "#7a5b98", "#3f7d45"]
VALUE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "paper_result_value",
    ["#d8dee0", "#b7d1ce", "#7db9aa", "#419287", "#226a74"],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem_prefix", default="")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
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


def task_label(task: str) -> str:
    return TASK_LABELS.get(task, task)


def pretty_transition(transition: str) -> str:
    return transition.replace("->", " " + "\u2192" + " ")


def task_title(task: str) -> str:
    return task_label(task).replace("+", "$_+$").replace("-", "$_-$")


def result_norm(rows: list[dict]) -> mcolors.Normalize:
    values = [float(row["result"]) for row in rows if "result" in row]
    if not values:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    return mcolors.Normalize(vmin=min(values), vmax=max(values))


def value_color(value: float, norm: mcolors.Normalize):
    return plt.get_cmap(VALUE_CMAP)(norm(value))


def add_value_colorbar(fig, ax, norm: mcolors.Normalize) -> None:
    mappable = cm.ScalarMappable(norm=norm, cmap=VALUE_CMAP)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Result value")
    cbar.outline.set_edgecolor(SPINE)
    cbar.outline.set_linewidth(0.75)


def setup_axis(ax, title: str, *, equal: bool = False) -> None:
    ax.set_title(title, pad=8)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
    ax.axhline(0.0, color="#9aa1a3", linewidth=0.75)
    ax.axvline(0.0, color="#9aa1a3", linewidth=0.75)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if equal:
        ax.set_aspect("equal", adjustable="box")


def save(fig, output_dir: Path, stem: str, prefix: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}{stem}" if prefix else stem
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def points(rows: list[dict]) -> np.ndarray:
    return np.asarray([[float(row["pc1"]), float(row["pc2"])] for row in rows], dtype=float)


def robust_limits(points_: np.ndarray, q: float = 0.985) -> tuple[tuple[float, float], tuple[float, float]]:
    if points_.size == 0:
        return (-1, 1), (-1, 1)
    lo = (1 - q) / 2
    hi = 1 - lo
    xmin, xmax = np.quantile(points_[:, 0], [lo, hi])
    ymin, ymax = np.quantile(points_[:, 1], [lo, hi])
    pad = max(xmax - xmin, ymax - ymin) * 0.06
    pad = max(float(pad), 0.5)
    return (float(xmin - pad), float(xmax + pad)), (float(ymin - pad), float(ymax + pad))


def covariance_ellipse(ax, pts: np.ndarray, color: str, *, scale: float = 2.0, alpha: float = 0.78) -> None:
    if pts.shape[0] < 4:
        return
    cov = np.cov(pts.T)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals = np.clip(vals, 1e-12, None)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ellipse = Ellipse(
        pts.mean(axis=0),
        width=scale * np.sqrt(vals[0]),
        height=scale * np.sqrt(vals[1]),
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.2,
        alpha=alpha,
    )
    ax.add_patch(ellipse)


def draw_reference_links(ax, rows: list[dict], *, reference_task: str = "text:addition") -> None:
    by_result: dict[int, dict[str, np.ndarray]] = {}
    for row in rows:
        by_result.setdefault(int(row["result"]), {})[row["task"]] = np.asarray([float(row["pc1"]), float(row["pc2"])])
    for task_points in by_result.values():
        if reference_task not in task_points:
            continue
        ref = task_points[reference_task]
        for task, point in task_points.items():
            if task == reference_task:
                continue
            ax.plot(
                [ref[0], point[0]],
                [ref[1], point[1]],
                color="#aeb5b8",
                lw=0.45,
                alpha=0.28,
                zorder=1,
            )


def draw_value_centroid_panel(ax, rows: list[dict], title: str, *, show_links: bool) -> mcolors.Normalize:
    setup_axis(ax, title)
    norm = result_norm(rows)
    if show_links:
        draw_reference_links(ax, rows)
    for task in TASK_COLORS:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        pts = points(task_rows)
        values = [float(row["result"]) for row in task_rows]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=32 if "text" in task else 38,
            marker=TASK_MARKERS[task],
            c=[value_color(value, norm) for value in values],
            edgecolor=SPINE,
            linewidth=0.58,
            alpha=0.92,
            zorder=3,
        )
    xlim, ylim = robust_limits(points(rows), q=0.99)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return norm


def add_centroid_legend(ax, *, position: str = "top") -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=TASK_MARKERS[task],
            color="none",
            markerfacecolor="#b8c0c3",
            markeredgecolor=SPINE,
            label=task_label(task),
            markersize=5.8,
        )
        for task in TASK_COLORS
    ]
    legend_kwargs = {"frameon": False, "ncol": 4, "handletextpad": 0.35, "columnspacing": 0.75, "fontsize": 7.6}
    if position == "top":
        legend_kwargs.update({"loc": "upper center", "bbox_to_anchor": (0.5, 1.13)})
    elif position == "bottom":
        legend_kwargs.update({"loc": "lower center", "bbox_to_anchor": (0.5, -0.23)})
    elif position == "inside_right":
        legend_kwargs.update({"loc": "upper right", "bbox_to_anchor": (0.985, 0.985)})
    else:
        legend_kwargs.update({"loc": "upper left"})
    ax.legend(
        handles=handles,
        **legend_kwargs,
    )


def plot_aligned_centroids(input_dir: Path, output_dir: Path, prefix: str) -> None:
    rows = load_rows(input_dir / "plot_a_aligned_result_centroids.jsonl")
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    fig.patch.set_facecolor(BACKGROUND)
    norm = draw_value_centroid_panel(ax, rows, "Aligned result-centroid geometry", show_links=True)
    add_centroid_legend(ax, position="top")
    add_value_colorbar(fig, ax, norm)
    save(fig, output_dir, "main_a_aligned_centroid_geometry", prefix)


def plot_raw_centroids(input_dir: Path, output_dir: Path, prefix: str) -> None:
    rows = load_rows(input_dir / "plot_a_raw_result_centroids.jsonl")
    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    fig.patch.set_facecolor(BACKGROUND)
    norm = draw_value_centroid_panel(ax, rows, "Raw result-centroid geometry", show_links=False)
    add_centroid_legend(ax, position="top")
    add_value_colorbar(fig, ax, norm)
    save(fig, output_dir, "appendix_a_raw_centroid_geometry", prefix)


def plot_pooled_residuals(input_dir: Path, output_dir: Path, prefix: str) -> None:
    rows = load_rows(input_dir / "plot_b_pooled_value_residuals.jsonl")
    fig, ax = plt.subplots(figsize=(6.0, 4.7))
    fig.patch.set_facecolor(BACKGROUND)
    draw_pooled_residual_panel(ax, rows, "Residual geometry after pooled value centering")
    task_color_legend(ax, position="top")
    save(fig, output_dir, "main_b_pooled_value_residual_geometry", prefix)


def plot_condition_specific_residuals(input_dir: Path, output_dir: Path, prefix: str) -> None:
    rows = load_rows(input_dir / "plot_b_condition_specific_residuals.jsonl")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)
    xlim, ylim = robust_limits(points(rows), q=0.995)
    for ax, task in zip(axes.ravel(), TASK_COLORS):
        setup_axis(ax, f"{task_label(task)} residuals")
        task_rows = [row for row in rows if row["task"] == task]
        if task_rows:
            pts = points(task_rows)
            marker = MARKERS.get(task_rows[0].get("modality"), "o")
            ax.scatter(pts[:, 0], pts[:, 1], s=8, marker=marker, color=TASK_COLORS[task], alpha=0.20, linewidths=0)
            covariance_ellipse(ax, pts, TASK_COLORS[task], scale=2.0, alpha=0.7)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    fig.subplots_adjust(hspace=0.28, wspace=0.20)
    save(fig, output_dir, "appendix_b_condition_specific_residuals", prefix)


def transition_map(rows: list[dict]) -> dict[str, str]:
    transitions = list(dict.fromkeys(row["transition"] for row in rows))
    return {transition: TRANSITION_COLORS[index % len(TRANSITION_COLORS)] for index, transition in enumerate(transitions)}


def draw_arrow(ax, x: float, y: float, *, color: str, linestyle: str, linewidth: float) -> None:
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(0.0, 0.0),
        annotation_clip=False,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "mutation_scale": 9,
            "alpha": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def draw_transport_panel(ax, rows: list[dict], title: str, *, equal: bool = True) -> None:
    setup_axis(ax, title, equal=equal)
    colors = transition_map(rows)
    for row in rows:
        color = colors[row["transition"]]
        linestyle = "-" if row["kind"] == "true" else "--"
        linewidth = 1.8 if row["kind"] == "true" else 1.55
        draw_arrow(ax, float(row["pc1"]), float(row["pc2"]), color=color, linestyle=linestyle, linewidth=linewidth)
    pts = np.asarray([[0.0, 0.0]] + [[float(row["pc1"]), float(row["pc2"])] for row in rows])
    span = max(float(np.abs(pts).max()), 1.0) * 1.12
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)


def plot_transport_arrows(input_dir: Path, output_dir: Path, prefix: str) -> None:
    rows = load_rows(input_dir / "plot_c_transported_displacement_arrows.jsonl")
    directions = list(dict.fromkeys((row["source_task"], row["destination_task"]) for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.0))
    fig.patch.set_facecolor(BACKGROUND)
    for ax, direction in zip(axes.ravel(), directions):
        panel = [row for row in rows if (row["source_task"], row["destination_task"]) == direction]
        draw_transport_panel(ax, panel, f"{task_title(direction[0])} $\\rightarrow$ {task_title(direction[1])}")
    for ax in axes.ravel()[len(directions):]:
        ax.axis("off")
    style_handles = [
        plt.Line2D([0], [0], color=SPINE, lw=1.8, linestyle="-", label="true destination"),
        plt.Line2D([0], [0], color=SPINE, lw=1.8, linestyle="--", label="transported source"),
    ]
    color_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=color, markeredgecolor=color, label=pretty_transition(transition), markersize=5.2)
        for transition, color in transition_map(rows).items()
    ]
    fig.legend(handles=style_handles + color_handles, loc="upper center", ncol=min(6, 2 + len(color_handles)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.88, hspace=0.32, wspace=0.24)
    save(fig, output_dir, "main_c_same_operation_transport_arrows", prefix)


def plot_factorized_path_agreement(input_dir: Path, output_dir: Path, prefix: str) -> None:
    two_step_path = input_dir / "plot_d_factorized_two_step_paths.jsonl"
    use_two_step = two_step_path.exists()
    rows = load_rows(two_step_path if use_two_step else input_dir / "plot_d_factorized_path_parallelogram.jsonl")
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.0))
    fig.patch.set_facecolor(BACKGROUND)
    if use_two_step:
        draw_two_step_path_grid(axes.ravel(), rows, compact=True, shared_limits=True, legend_axis_index=2)
        handles = two_step_path_handles()
    else:
        draw_factorized_arrow_grid(axes.ravel(), rows, title_prefix="", compact=True)
        handles = factorized_arrow_handles()
    if not use_two_step:
        fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4)
    fig.supxlabel("PC1", y=0.035, fontsize=9.5)
    fig.supylabel("PC2", x=0.03, fontsize=9.5)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.10, hspace=0.34, wspace=0.26)
    save(fig, output_dir, "main_d_factorized_path_agreement", prefix)


def factorized_distance_rows(rows: list[dict]) -> list[dict]:
    by_transition: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_transition.setdefault(row["transition"], {})[row["kind"]] = row
    distance_rows = []
    for transition, parts in by_transition.items():
        direct = parts.get("direct")
        if not direct:
            continue
        dx, dy = float(direct["pc1"]), float(direct["pc2"])
        for kind in ["operation_first", "modality_first", "true"]:
            row = parts.get(kind)
            if not row:
                continue
            ex = float(row["pc1"]) - dx
            ey = float(row["pc2"]) - dy
            distance_rows.append(
                {
                    "transition": transition,
                    "kind": kind,
                    "distance": float(np.sqrt(ex * ex + ey * ey)),
                }
            )
    return distance_rows


def draw_factorized_agreement_panel(ax, distance_rows: list[dict], title: str, *, show_legend: bool = True) -> None:
    ax.set_title(title, pad=8)
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    ax.axhline(0.0, color="#9aa1a3", linewidth=0.75)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    transitions = list(dict.fromkeys(row["transition"] for row in distance_rows))
    x = np.arange(len(transitions), dtype=float)
    offsets = {"operation_first": -0.17, "modality_first": 0.0, "true": 0.17}
    labels = {
        "operation_first": "operation first",
        "modality_first": "modality first",
        "true": "true destination",
    }
    for kind in ["operation_first", "modality_first", "true"]:
        ys = []
        for transition in transitions:
            match = [row for row in distance_rows if row["transition"] == transition and row["kind"] == kind]
            ys.append(match[0]["distance"] if match else np.nan)
        ax.scatter(
            x + offsets[kind],
            ys,
            s=42 if kind != "true" else 48,
            marker="o" if kind != "true" else "X",
            color=TRANSPORT_COLORS[kind],
            edgecolor=SPINE if kind == "true" else "none",
            linewidth=0.55,
            alpha=0.9,
            label=labels[kind],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_transition(transition) for transition in transitions], rotation=22, ha="right")
    ax.set_ylabel("Distance to direct endpoint")
    ax.set_xlabel("Transition")
    ax.set_ylim(bottom=0.0)
    if show_legend:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, fontsize=8.4)


def factorized_arrow_handles() -> list:
    return [
        plt.Line2D([0], [0], color=SPINE, lw=1.7, linestyle="-", label="true"),
        plt.Line2D([0], [0], color=TRANSPORT_COLORS["direct"], lw=1.7, linestyle="-", label="direct"),
        plt.Line2D([0], [0], color=TRANSPORT_COLORS["operation_first"], lw=1.7, linestyle="--", label="operation first"),
        plt.Line2D([0], [0], color=TRANSPORT_COLORS["modality_first"], lw=1.7, linestyle=":", label="modality first"),
    ]


def two_step_path_handles() -> list:
    return [
        plt.Line2D([0], [0], color=SPINE, marker="o", linestyle="none", markerfacecolor=PANEL_BG, markersize=5.2, label="source"),
        plt.Line2D([0], [0], color=TRANSPORT_COLORS["modality_first"], lw=1.7, marker="o", markersize=4.4, label="modality first"),
        plt.Line2D([0], [0], color=TRANSPORT_COLORS["operation_first"], lw=1.7, marker="o", markersize=4.4, label="operation first"),
        plt.Line2D([0], [0], color=TRANSPORT_COLORS["direct"], lw=1.7, marker="D", markersize=4.7, label="direct"),
        plt.Line2D([0], [0], color=SPINE, marker="X", linestyle="none", markersize=5.0, label="true"),
    ]


def draw_factorized_arrow(ax, row: dict) -> None:
    style = {
        "true": (SPINE, "-", 1.45, 0.82),
        "direct": (TRANSPORT_COLORS["direct"], "-", 1.55, 0.95),
        "operation_first": (TRANSPORT_COLORS["operation_first"], "--", 1.35, 0.9),
        "modality_first": (TRANSPORT_COLORS["modality_first"], ":", 1.6, 0.95),
    }
    color, linestyle, linewidth, alpha = style[row["kind"]]
    ax.annotate(
        "",
        xy=(float(row["pc1"]), float(row["pc2"])),
        xytext=(0.0, 0.0),
        annotation_clip=False,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "mutation_scale": 7.5,
            "alpha": alpha,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def draw_factorized_arrow_grid(axes, rows: list[dict], *, title_prefix: str, compact: bool = False) -> None:
    transitions = list(dict.fromkeys(row["transition"] for row in rows))[:4]
    for axis_index, (ax, transition) in enumerate(zip(axes, transitions)):
        panel = [row for row in rows if row["transition"] == transition]
        setup_axis(ax, f"{title_prefix}{pretty_transition(transition)}", equal=True)
        for kind in ["true", "direct", "operation_first", "modality_first"]:
            for row in [row for row in panel if row["kind"] == kind]:
                draw_factorized_arrow(ax, row)
        pts = np.asarray([[0.0, 0.0]] + [[float(row["pc1"]), float(row["pc2"])] for row in panel])
        span = max(float(np.abs(pts).max()), 1.0) * 1.18
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        if compact:
            ax.set_aspect("auto")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
    for ax in list(axes)[len(transitions):]:
        ax.axis("off")


def centered_limits_for_rows(rows: list[dict], *, pad_scale: float = 1.16) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.asarray([[0.0, 0.0]] + [[float(row["pc1"]), float(row["pc2"])] for row in rows], dtype=float)
    span = max(float(np.abs(pts).max()), 1.0) * pad_scale
    return (-span, span), (-span, span)


def draw_two_step_path_grid(
    axes,
    rows: list[dict],
    *,
    compact: bool = False,
    shared_limits: bool = False,
    legend_axis_index: int | None = None,
) -> None:
    transitions = list(dict.fromkeys(row["transition"] for row in rows))[:4]
    xlim, ylim = centered_limits_for_rows(rows) if shared_limits else (None, None)
    for axis_index, (ax, transition) in enumerate(zip(axes, transitions)):
        panel = [row for row in rows if row["transition"] == transition]
        setup_axis(ax, pretty_transition(transition), equal=True)
        ax.scatter(
            0.0,
            0.0,
            s=26,
            marker="o",
            facecolor=PANEL_BG,
            edgecolor=SPINE,
            linewidth=0.9,
            zorder=6,
        )
        direct = next((row for row in panel if row.get("kind") == "direct"), None)
        if direct:
            dx, dy = float(direct["pc1"]), float(direct["pc2"])
            ax.plot([0.0, dx], [0.0, dy], color=TRANSPORT_COLORS["direct"], lw=1.45, alpha=0.88, zorder=2)
        for path, color in [("modality_first", TRANSPORT_COLORS["modality_first"]), ("operation_first", TRANSPORT_COLORS["operation_first"])]:
            path_rows = [row for row in panel if row.get("path") == path]
            order = {"source": 0, "intermediate": 1, "final": 2}
            path_rows = sorted(path_rows, key=lambda row: order.get(row.get("stage"), 99))
            if len(path_rows) >= 3:
                xy = np.asarray([[float(row["pc1"]), float(row["pc2"])] for row in path_rows])
                ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.5, marker="o", markersize=3.2, alpha=0.92)
        for kind, marker, color in [("direct", "D", TRANSPORT_COLORS["direct"]), ("true", "X", SPINE)]:
            endpoints = [row for row in panel if row.get("kind") == kind]
            if endpoints:
                row = endpoints[0]
                ax.scatter(float(row["pc1"]), float(row["pc2"]), marker=marker, s=34, color=color, edgecolor=SPINE, linewidth=0.45, zorder=5)
        if shared_limits and xlim and ylim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        else:
            pxlim, pylim = centered_limits_for_rows(panel, pad_scale=1.18)
            ax.set_xlim(*pxlim)
            ax.set_ylim(*pylim)
        if legend_axis_index is not None and axis_index == legend_axis_index:
            ax.legend(
                handles=two_step_path_handles(),
                frameon=False,
                loc="upper left",
                ncol=3,
                fontsize=7.6,
                handlelength=1.45,
                handletextpad=0.38,
                columnspacing=0.75,
            )
        if compact:
            ax.set_aspect("auto")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
    for ax in list(axes)[len(transitions):]:
        ax.axis("off")


def draw_pooled_residual_panel(ax, rows: list[dict], title: str) -> None:
    setup_axis(ax, title)
    for task, color in TASK_COLORS.items():
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        pts = points(task_rows)
        marker = MARKERS.get(task_rows[0].get("modality"), "o")
        ax.scatter(pts[:, 0], pts[:, 1], s=8, marker=marker, color=color, alpha=0.18, linewidths=0, label=task_label(task))
        covariance_ellipse(ax, pts, color, scale=2.0, alpha=0.78)
        center = pts.mean(axis=0)
        ax.scatter(center[0], center[1], s=40, marker="X", color=color, edgecolor=SPINE, linewidth=0.55, zorder=5)
    xlim, ylim = robust_limits(points(rows), q=0.992)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def task_color_legend(ax, *, position: str = "top") -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=color, markeredgecolor=color, label=task_label(task), markersize=4.8)
        for task, color in TASK_COLORS.items()
    ]
    kwargs = {"frameon": False, "ncol": 4, "fontsize": 7.4, "handletextpad": 0.25, "columnspacing": 0.75}
    if position == "top":
        kwargs.update({"loc": "upper center", "bbox_to_anchor": (0.5, 1.13)})
    elif position == "inside_left":
        kwargs.update({"loc": "upper left", "bbox_to_anchor": (0.02, 0.985)})
    else:
        kwargs.update({"loc": "upper right"})
    ax.legend(handles=handles, **kwargs)


def plot_main_four_panel(input_dir: Path, output_dir: Path, prefix: str) -> None:
    aligned = load_rows(input_dir / "plot_a_aligned_result_centroids.jsonl")
    pooled = load_rows(input_dir / "plot_b_pooled_value_residuals.jsonl")
    transport = load_rows(input_dir / "plot_c_transported_displacement_arrows.jsonl")
    two_step_path = input_dir / "plot_d_factorized_two_step_paths.jsonl"
    use_two_step = two_step_path.exists()
    factorized = load_rows(two_step_path if use_two_step else input_dir / "plot_d_factorized_path_parallelogram.jsonl")

    fig = plt.figure(figsize=(11.2, 8.4), facecolor=BACKGROUND)
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.955, bottom=0.085, hspace=0.42, wspace=0.24)
    fig.patch.set_facecolor(BACKGROUND)

    ax_a = fig.add_subplot(gs[0, 0])
    norm = draw_value_centroid_panel(ax_a, aligned, "A. Aligned result centroids", show_links=True)
    add_centroid_legend(ax_a, position="inside_right")
    add_value_colorbar(fig, ax_a, norm)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_pooled_residual_panel(ax_b, pooled, "B. Pooled value residuals")
    task_color_legend(ax_b, position="inside_left")

    ax_c = fig.add_subplot(gs[1, 0])
    directions = list(dict.fromkeys((row["source_task"], row["destination_task"]) for row in transport))
    direction = directions[0]
    panel = [row for row in transport if (row["source_task"], row["destination_task"]) == direction]
    draw_transport_panel(ax_c, panel, f"C. Transported displacement: {task_title(direction[0])} $\\rightarrow$ {task_title(direction[1])}", equal=False)
    style_handles = [
        plt.Line2D([0], [0], color=SPINE, lw=1.8, linestyle="-", label="true"),
        plt.Line2D([0], [0], color=SPINE, lw=1.8, linestyle="--", label="transported"),
    ]
    transition_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="none", color="none", markerfacecolor=color, markeredgecolor=color, label=pretty_transition(transition), markersize=5.2)
        for transition, color in transition_map(panel).items()
    ]
    legend1 = ax_c.legend(handles=style_handles, frameon=False, loc="upper left", fontsize=8.8, handlelength=1.7)
    ax_c.add_artist(legend1)
    ax_c.legend(
        handles=transition_handles,
        frameon=False,
        loc="lower left",
        ncol=2,
        fontsize=8.0,
        handletextpad=0.32,
        columnspacing=0.85,
        labelspacing=0.25,
    )

    d_transitions = list(dict.fromkeys(row["transition"] for row in factorized))
    dgs = gs[1, 1].subgridspec(2, 2, hspace=0.30, wspace=0.18)
    if len(d_transitions) == 3:
        d_axes = [fig.add_subplot(dgs[0, 0]), fig.add_subplot(dgs[0, 1]), fig.add_subplot(dgs[1, :])]
    else:
        d_axes = [
            fig.add_subplot(dgs[0, 0]),
            fig.add_subplot(dgs[0, 1]),
            fig.add_subplot(dgs[1, 0]),
            fig.add_subplot(dgs[1, 1]),
        ]
    if use_two_step:
        draw_two_step_path_grid(d_axes, factorized, compact=True, shared_limits=True, legend_axis_index=2 if len(d_axes) >= 3 else None)
        d_handles = None
    else:
        draw_factorized_arrow_grid(d_axes, factorized, title_prefix="", compact=True)
        d_handles = factorized_arrow_handles()
    left = min(ax.get_position().x0 for ax in d_axes)
    right = max(ax.get_position().x1 for ax in d_axes)
    top = max(ax.get_position().y1 for ax in d_axes)
    bottom = min(ax.get_position().y0 for ax in d_axes)
    fig.text((left + right) / 2, top + 0.026, "D. Factorized path agreement", ha="center", va="bottom", fontsize=10.5)
    fig.text((left + right) / 2, bottom - 0.030, "PC1", ha="center", va="top", fontsize=8.3)
    fig.text(left - 0.026, (bottom + top) / 2, "PC2", ha="right", va="center", rotation=90, fontsize=8.3)
    if d_handles is not None:
        fig.legend(
            handles=d_handles,
            frameon=False,
            loc="center",
            bbox_to_anchor=((left + right) / 2, top + 0.035),
            ncol=4,
            fontsize=7.8,
            handlelength=1.7,
            columnspacing=0.9,
        )

    save(fig, output_dir, "main_four_panel_procrustes_geometry", prefix)


def main() -> None:
    args = parse_args()
    style_matplotlib()
    plot_aligned_centroids(args.input_dir, args.output_dir, args.stem_prefix)
    plot_raw_centroids(args.input_dir, args.output_dir, args.stem_prefix)
    plot_pooled_residuals(args.input_dir, args.output_dir, args.stem_prefix)
    plot_condition_specific_residuals(args.input_dir, args.output_dir, args.stem_prefix)
    plot_transport_arrows(args.input_dir, args.output_dir, args.stem_prefix)
    plot_factorized_path_agreement(args.input_dir, args.output_dir, args.stem_prefix)
    plot_main_four_panel(args.input_dir, args.output_dir, args.stem_prefix)


if __name__ == "__main__":
    main()
