"""Geometric PCA plots for Procrustes transport.

The plots are descriptive diagnostics:

Plot A: result centroids for the four task spaces in a shared 2D PCA plane.
Plot B: residuals after subtracting each example's result centroid.
Plot C: true vs transported same-operation transition arrows.
Plot D: direct vs composed cross-operation factorized arrows.

For map-based plots, coordinates are row vectors and maps act as alpha * x @ Q.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    jsonable,
    label,
    load_jsonl,
    parse_task,
    save_json,
    save_jsonl,
    selected_seeds,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.factorized_paths.factorized_paths import (
    TASKS,
    TaskSpace,
    compose_maps,
    fit_scaled_map,
    load_task_space,
    path_specs,
    same_result_width,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    activations_for_ids,
    cached_task_samples,
    load_model_bundle,
    sample_key,
    sample_split,
    stable_seed,
    target_value,
)


EXPERIMENT = "pca_plots"
DEFAULT_SAME_OPERATION_PAIRS = [
    "text:addition->image:addition",
    "image:addition->text:addition",
    "text:subtraction->image:subtraction",
    "image:subtraction->text:subtraction",
]
DEFAULT_FACTOR_PAIR = "text:addition->image:subtraction"
DEFAULT_TRANSITIONS = ["2->7", "4->9", "8->3", "1->6"]
DEFAULT_REFERENCE_TASK = "text:addition"
TASK_ORDER = ["text:addition", "text:subtraction", "image:addition", "image:subtraction"]
TASK_LABELS = {
    "text:addition": "T+",
    "text:subtraction": "T-",
    "image:addition": "I+",
    "image:subtraction": "I-",
}
MODALITY_MARKERS = {"text": "o", "image": "^"}
OPERATION_EDGES = {"addition": "#202426", "subtraction": "#a95642"}
COLORS = {
    "true": "#202426",
    "transported": "#226a74",
    "direct": "#226a74",
    "operation_first": "#a95642",
    "modality_first": "#5d6f9f",
}
TASK_COLORS = {
    "text:addition": "#226a74",
    "text:subtraction": "#a95642",
    "image:addition": "#5d6f9f",
    "image:subtraction": "#b8872d",
}
TRANSITION_PALETTE = ["#226a74", "#a95642", "#5d6f9f", "#b8872d", "#7a5b98", "#3f7d45"]
ARROW_STYLES = {
    "true": ("solid", 1.9),
    "transported": ("--", 1.9),
    "direct": ("solid", 2.0),
    "operation_first": ("--", 2.0),
    "modality_first": (":", 2.2),
}
BACKGROUND = "#fdfdfd"
GRID = "#d8ddde"
TEXT = "#171717"


@dataclass
class TaskData:
    space: TaskSpace
    samples: list[dict]
    coordinates: torch.Tensor
    centroids: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper/procrustes/pca_plots"))
    parser.add_argument("--plot_dir", type=Path, default=Path("visualizations/main_paper/procrustes/pca_plots"))
    parser.add_argument("--factorized_map_dir", type=Path, default=Path("results/paper/procrustes/factorized_paths"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--same_operation_pairs", nargs="+", default=DEFAULT_SAME_OPERATION_PAIRS)
    parser.add_argument("--factorized_pair", default=DEFAULT_FACTOR_PAIR)
    parser.add_argument("--reference_task", default=DEFAULT_REFERENCE_TASK)
    parser.add_argument("--transitions", nargs="+", default=DEFAULT_TRANSITIONS)
    parser.add_argument("--plot_split", choices=["all", "train", "validation", "test"], default="test")
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--seed_map", nargs="*", default=[], metavar="TASK=SEEDS")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--alignment_seed", type=int, default=0)
    parser.add_argument("--max_alignment_samples", type=int, default=2048)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_residual_points_per_task", type=int, default=500)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_pair(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Pair must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def parse_transition(spec: str) -> tuple[int, int]:
    if "->" not in spec:
        raise ValueError(f"Transition must be A->B, got {spec!r}.")
    start, end = spec.split("->", 1)
    return int(start.strip()), int(end.strip())


def task_short(task: str) -> str:
    return TASK_LABELS.get(task, label(task))


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
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


def choose_samples(args: argparse.Namespace, samples: list[dict]) -> list[dict]:
    if args.plot_split == "all":
        return samples
    return sample_split(args, samples, args.plot_split)


def coordinates_for_samples(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    space: TaskSpace,
    samples: list[dict],
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    ids = [sample_key(sample) for sample in samples]
    activations = activations_for_ids(
        args,
        activation_cache,
        sample_cache,
        task=space.task,
        row=space.row,
        ids=ids,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return activations @ space.basis


def compute_centroids(args: argparse.Namespace, samples: list[dict], coordinates: torch.Tensor) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for sample, coordinate in zip(samples, coordinates):
        grouped.setdefault(target_value(args, sample), []).append(coordinate)
    return {value: torch.stack(parts).mean(dim=0) for value, parts in grouped.items()}


def load_task_data(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    *,
    task: str,
    seed: int,
    hidden_size: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> TaskData:
    space = load_task_space(args, space_cache, task=task, seed=seed, hidden_size=hidden_size)
    samples = choose_samples(args, cached_task_samples(sample_cache, space.task, space.row))
    coordinates = coordinates_for_samples(
        args,
        activation_cache,
        sample_cache,
        space=space,
        samples=samples,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    centroids = compute_centroids(args, samples, coordinates)
    print(f"  {task}[{seed}] {args.plot_split}: samples={len(samples)}, result centroids={len(centroids)}")
    return TaskData(space=space, samples=samples, coordinates=coordinates, centroids=centroids)


def pca_project(vectors: torch.Tensor) -> tuple[np.ndarray, dict]:
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Need at least two vectors for PCA, got shape {tuple(vectors.shape)}.")
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    projected = centered @ components
    variance = singular_values.square()
    ratio = variance / variance.sum().clamp_min(1e-12)
    return projected.cpu().numpy(), {
        "explained_variance_ratio": [float(x) for x in ratio[:2]],
        "singular_values": [float(x) for x in singular_values[:2]],
    }


def pca_project_arrows(vectors: torch.Tensor) -> tuple[np.ndarray, dict]:
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Need at least two vectors for PCA, got shape {tuple(vectors.shape)}.")
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    projected = vectors.float() @ components
    variance = singular_values.square()
    ratio = variance / variance.sum().clamp_min(1e-12)
    return projected.cpu().numpy(), {
        "explained_variance_ratio": [float(x) for x in ratio[:2]],
        "singular_values": [float(x) for x in singular_values[:2]],
    }


def save_figure(fig, args: argparse.Namespace, stem: str) -> None:
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.plot_dir / f"{stem}.png"
    pdf_path = args.plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def setup_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_facecolor("#f7f8f8")
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
    ax.axhline(0.0, color="#a3aaac", linewidth=0.75)
    ax.axvline(0.0, color="#a3aaac", linewidth=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def plot_raw_result_centroids(args: argparse.Namespace, data: dict[tuple[str, int], TaskData]) -> dict:
    rows = []
    vectors = []
    for (task, seed), task_data in data.items():
        modality, operation = parse_task(task)
        for value, coordinate in sorted(task_data.centroids.items()):
            rows.append(
                {
                    "plot": "A_raw",
                    "geometry": "raw_task_das_coordinates",
                    "task": task,
                    "task_label": task_short(task),
                    "seed": seed,
                    "modality": modality,
                    "operation": operation,
                    "result": value,
                }
            )
            vectors.append(coordinate)
    projected, pca = pca_project(torch.stack(vectors))
    for row, point in zip(rows, projected):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])

    values = np.asarray([row["result"] for row in rows], dtype=float)
    norm = plt.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax, "Plot A raw - centroids in their own DAS axes")
    for row in rows:
        ax.scatter(
            row["pc1"],
            row["pc2"],
            s=32,
            marker=MODALITY_MARKERS[row["modality"]],
            c=[cmap(norm(row["result"]))],
            edgecolors=OPERATION_EDGES[row["operation"]],
            linewidths=0.85,
            alpha=0.86,
        )
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        if len(task_rows) < 2:
            continue
        center = np.asarray([[row["pc1"], row["pc2"]] for row in task_rows]).mean(axis=0)
        ax.text(center[0], center[1], task_short(task), fontsize=9, weight="bold", color=TASK_COLORS[task])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Result value")
    save_figure(fig, args, "plot_a_raw_result_centroids")
    save_jsonl(rows, args.output_dir / "plot_a_raw_result_centroids.jsonl")
    return {"plot": "A_raw", **pca, "n_points": len(rows)}


def align_vectors_to_reference(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    *,
    source: TaskSpace,
    reference: TaskSpace,
    vectors: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[torch.Tensor, dict]:
    if source.task == reference.task and source.seed == reference.seed:
        return vectors.float(), {"alpha": 1.0, "map": "identity"}
    fitted = map_for(
        args,
        activation_cache,
        sample_cache,
        map_cache,
        source=source,
        destination=reference,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return float(fitted["alpha"]) * (vectors.float() @ fitted["q"]), {
        "alpha": float(fitted["alpha"]),
        "map": str(fitted["path"]),
    }


def plot_aligned_result_centroids(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    data: dict[tuple[str, int], TaskData],
    *,
    reference: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    rows = []
    vectors = []
    for (task, seed), task_data in data.items():
        modality, operation = parse_task(task)
        values = sorted(task_data.centroids)
        centroid_matrix = torch.stack([task_data.centroids[value] for value in values])
        # The scaled maps are displacement maps, so remove each task's centroid
        # origin before transporting into the reference coordinate system.
        centered = centroid_matrix - centroid_matrix.mean(dim=0, keepdim=True)
        aligned, map_info = align_vectors_to_reference(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            source=task_data.space,
            reference=reference,
            vectors=centered,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        for value, coordinate in zip(values, aligned):
            rows.append(
                {
                    "plot": "A_aligned",
                    "geometry": "task_centered_centroids_aligned_to_reference",
                    "reference_task": reference.task,
                    "reference_seed": reference.seed,
                    "task": task,
                    "task_label": task_short(task),
                    "seed": seed,
                    "modality": modality,
                    "operation": operation,
                    "result": value,
                    "alignment_alpha_to_reference": map_info["alpha"],
                    "alignment_map_to_reference": map_info["map"],
                }
            )
            vectors.append(coordinate)
    projected, pca = pca_project(torch.stack(vectors))
    for row, point in zip(rows, projected):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])

    values = np.asarray([row["result"] for row in rows], dtype=float)
    norm = plt.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(6.8, 5.1))
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax, f"Plot A aligned - centroids in {task_short(reference.task)} coordinates")
    for row in rows:
        ax.scatter(
            row["pc1"],
            row["pc2"],
            s=34,
            marker=MODALITY_MARKERS[row["modality"]],
            c=[cmap(norm(row["result"]))],
            edgecolors=OPERATION_EDGES[row["operation"]],
            linewidths=0.85,
            alpha=0.86,
        )
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        task_rows.sort(key=lambda row: row["result"])
        if len(task_rows) >= 2:
            ax.plot(
                [row["pc1"] for row in task_rows],
                [row["pc2"] for row in task_rows],
                color="#687174",
                linewidth=0.65,
                alpha=0.35,
            )
            center = np.asarray([[row["pc1"], row["pc2"]] for row in task_rows]).mean(axis=0)
            ax.text(center[0], center[1], task_short(task), fontsize=9, weight="bold", color=TASK_COLORS[task])
    modality_handles = [
        plt.Line2D([0], [0], marker=marker, color="none", markerfacecolor="#8ab6b9", markeredgecolor="#222", label=name)
        for name, marker in MODALITY_MARKERS.items()
    ]
    operation_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#8ab6b9", markeredgecolor=color, label=name)
        for name, color in OPERATION_EDGES.items()
    ]
    ax.legend(handles=modality_handles + operation_handles, frameon=False, loc="upper right", fontsize=8)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Result value")
    save_figure(fig, args, "plot_a_aligned_result_centroids")
    save_jsonl(rows, args.output_dir / "plot_a_aligned_result_centroids.jsonl")
    return {"plot": "A_aligned", **pca, "n_points": len(rows)}


def robust_limits(points: np.ndarray, quantile: float = 0.985) -> tuple[tuple[float, float], tuple[float, float]]:
    if points.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    low = (1.0 - quantile) / 2.0
    high = 1.0 - low
    xmin, xmax = np.quantile(points[:, 0], [low, high])
    ymin, ymax = np.quantile(points[:, 1], [low, high])
    xpad = max((xmax - xmin) * 0.08, 0.5)
    ypad = max((ymax - ymin) * 0.08, 0.5)
    return (float(xmin - xpad), float(xmax + xpad)), (float(ymin - ypad), float(ymax + ypad))


def draw_cov_ellipse(ax, points: np.ndarray, color: str) -> None:
    if points.shape[0] < 4:
        return
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return
    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if values[1] <= 1e-12:
        return
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2.0 * np.sqrt(values)
    ellipse = Ellipse(
        points.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.15,
        alpha=0.8,
    )
    ax.add_patch(ellipse)


def aligned_sample_vectors(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    data: dict[tuple[str, int], TaskData],
    *,
    reference: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[list[dict], torch.Tensor]:
    rows = []
    vectors = []
    for (task, seed), task_data in data.items():
        modality, operation = parse_task(task)
        candidates = list(zip(task_data.samples, task_data.coordinates))
        rng = random.Random(stable_seed("plot_b_samples", task, seed, args.alignment_seed))
        rng.shuffle(candidates)
        if args.max_residual_points_per_task > 0:
            candidates = candidates[: args.max_residual_points_per_task]
        if not candidates:
            continue
        coordinate_matrix = torch.stack([coordinate for _sample, coordinate in candidates])
        aligned, map_info = align_vectors_to_reference(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            source=task_data.space,
            reference=reference,
            vectors=coordinate_matrix,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        for (sample, _coordinate), aligned_coordinate in zip(candidates, aligned):
            rows.append(
                {
                    "task": task,
                    "task_label": task_short(task),
                    "seed": seed,
                    "sample_id": sample_key(sample),
                    "modality": modality,
                    "operation": operation,
                    "result": target_value(args, sample),
                    "reference_task": reference.task,
                    "reference_seed": reference.seed,
                    "alignment_alpha_to_reference": map_info["alpha"],
                    "alignment_map_to_reference": map_info["map"],
                }
            )
            vectors.append(aligned_coordinate)
    return rows, torch.stack(vectors)


def draw_task_scatter(ax, rows: list[dict], *, title: str) -> None:
    setup_axis(ax, title)
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            continue
        points = np.asarray([[row["pc1"], row["pc2"]] for row in task_rows], dtype=float)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=11,
            alpha=0.24,
            marker=MODALITY_MARKERS[parse_task(task)[0]],
            color=TASK_COLORS[task],
            linewidths=0,
            label=task_short(task),
        )
        draw_cov_ellipse(ax, points, TASK_COLORS[task])
        center = points.mean(axis=0)
        ax.scatter(center[0], center[1], s=85, marker="X", color=TASK_COLORS[task], edgecolor="#111", linewidth=0.7)
        ax.text(center[0], center[1], task_short(task), fontsize=8, weight="bold", color=TASK_COLORS[task])


def plot_pooled_value_residuals(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    data: dict[tuple[str, int], TaskData],
    *,
    reference: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    rows, aligned = aligned_sample_vectors(
        args,
        activation_cache,
        sample_cache,
        space_cache,
        map_cache,
        data,
        reference=reference,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    grouped = {}
    for row, vector in zip(rows, aligned):
        grouped.setdefault(row["result"], []).append(vector)
    pooled_centroids = {value: torch.stack(parts).mean(dim=0) for value, parts in grouped.items()}
    residuals = torch.stack([vector - pooled_centroids[row["result"]] for row, vector in zip(rows, aligned)])
    for row in rows:
        row["plot"] = "B_pooled"
        row["geometry"] = "pooled_value_centering_in_reference_coordinates"
    projected, pca = pca_project(residuals)
    for row, point in zip(rows, projected):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    fig.patch.set_facecolor(BACKGROUND)
    draw_task_scatter(ax, rows, title="Plot B pooled - remove shared value centroid")
    ax.legend(frameon=False, loc="best", fontsize=8, ncol=2)
    save_figure(fig, args, "plot_b_pooled_value_residuals")
    save_jsonl(rows, args.output_dir / "plot_b_pooled_value_residuals.jsonl")
    return {"plot": "B_pooled", **pca, "n_points": len(rows)}


def plot_condition_specific_residuals(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    data: dict[tuple[str, int], TaskData],
    *,
    reference: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    rows, aligned = aligned_sample_vectors(
        args,
        activation_cache,
        sample_cache,
        space_cache,
        map_cache,
        data,
        reference=reference,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    grouped = {}
    for row, vector in zip(rows, aligned):
        grouped.setdefault((row["task"], row["result"]), []).append(vector)
    task_value_centroids = {key: torch.stack(parts).mean(dim=0) for key, parts in grouped.items()}
    residuals = torch.stack([vector - task_value_centroids[(row["task"], row["result"])] for row, vector in zip(rows, aligned)])
    for row in rows:
        row["plot"] = "B_condition_specific"
        row["geometry"] = "condition_specific_value_centering_in_reference_coordinates"
    projected, pca = pca_project(residuals)
    for row, point in zip(rows, projected):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)
    points = np.asarray([[row["pc1"], row["pc2"]] for row in rows], dtype=float)
    xlim, ylim = robust_limits(points)
    for ax, task in zip(axes.ravel(), TASK_ORDER):
        setup_axis(ax, f"Plot B condition-specific - {task_short(task)}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        task_rows = [row for row in rows if row["task"] == task]
        if not task_rows:
            ax.text(0.5, 0.5, "No points", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.scatter(
            [row["pc1"] for row in task_rows],
            [row["pc2"] for row in task_rows],
            s=9,
            alpha=0.28,
            marker=MODALITY_MARKERS[parse_task(task)[0]],
            color=TASK_COLORS[task],
            linewidths=0,
        )
        center = np.asarray([[row["pc1"], row["pc2"]] for row in task_rows]).mean(axis=0)
        ax.scatter(center[0], center[1], s=80, marker="X", color=TASK_COLORS[task], edgecolor="#111", linewidth=0.7)
        ax.text(0.02, 0.95, f"n={len(task_rows)}", transform=ax.transAxes, va="top", ha="left", fontsize=8)
    fig.subplots_adjust(hspace=0.24, wspace=0.12)
    save_figure(fig, args, "plot_b_condition_specific_residuals")
    save_jsonl(rows, args.output_dir / "plot_b_condition_specific_residuals.jsonl")
    return {"plot": "B_condition_specific", **pca, "n_points": len(rows)}


def map_for(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    map_cache: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    key = (source.task, source.seed, destination.task, destination.seed)
    if key not in map_cache:
        result_output_dir = args.output_dir
        args.output_dir = args.factorized_map_dir
        try:
            map_cache[key] = fit_scaled_map(
                args,
                activation_cache,
                sample_cache,
                source=source,
                destination=destination,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
            )
        finally:
            args.output_dir = result_output_dir
    return map_cache[key]


def valid_transition_rows(
    transitions: list[tuple[int, int]],
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
) -> list[tuple[int, int]]:
    rows = []
    for start, end in transitions:
        if start in source_centroids and end in source_centroids and start in destination_centroids and end in destination_centroids:
            rows.append((start, end))
    return rows


def automatic_transition_rows(
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
    *,
    max_count: int,
    require_same_width: bool,
) -> list[tuple[int, int]]:
    values = sorted(set(source_centroids) & set(destination_centroids))
    candidates = [
        (start, end)
        for start in values
        for end in values
        if start != end and (not require_same_width or same_result_width(start, end))
    ]
    positives = sorted([item for item in candidates if item[1] > item[0]], key=lambda item: (item[1] - item[0], item), reverse=True)
    negatives = sorted([item for item in candidates if item[1] < item[0]], key=lambda item: (item[0] - item[1], item), reverse=True)
    ordered = []
    while positives or negatives:
        if positives:
            ordered.append(positives.pop(0))
        if negatives:
            ordered.append(negatives.pop(0))
    return ordered[:max_count]


def nearest_requested_transition_rows(
    requested: list[tuple[int, int]],
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
    *,
    require_same_width: bool,
) -> list[tuple[int, int]]:
    values = sorted(set(source_centroids) & set(destination_centroids))
    if not values:
        return []

    def nearest(value: int, blocked: int | None = None) -> int | None:
        candidates = [item for item in values if item != blocked]
        if not candidates:
            return None
        return min(candidates, key=lambda item: (abs(item - value), item))

    rows = []
    seen = set()
    for start, end in requested:
        first = nearest(start)
        second = nearest(end, blocked=first)
        if first is None or second is None:
            continue
        if require_same_width and not same_result_width(first, second):
            same_width_values = [item for item in values if item != first and same_result_width(first, item)]
            if same_width_values:
                second = min(same_width_values, key=lambda item: (abs(item - end), item))
        if first == second or (require_same_width and not same_result_width(first, second)):
            continue
        transition = (first, second)
        if transition not in seen:
            rows.append(transition)
            seen.add(transition)
    return rows


def selected_transition_rows(
    requested: list[tuple[int, int]],
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
    *,
    require_same_width: bool = False,
) -> tuple[list[tuple[int, int]], str]:
    valid = valid_transition_rows(requested, source_centroids, destination_centroids)
    if require_same_width:
        valid = [(start, end) for start, end in valid if same_result_width(start, end)]
    if valid:
        return valid, "requested"
    nearest = nearest_requested_transition_rows(
        requested,
        source_centroids,
        destination_centroids,
        require_same_width=require_same_width,
    )
    if nearest:
        return nearest, "nearest_requested_fallback"
    fallback = automatic_transition_rows(
        source_centroids,
        destination_centroids,
        max_count=max(1, len(requested)),
        require_same_width=require_same_width,
    )
    return fallback, "auto_shared_transition_fallback"


def transition_color_map(panel_rows: list[dict]) -> dict[str, str]:
    transitions = list(dict.fromkeys(row["transition"] for row in panel_rows))
    return {transition: TRANSITION_PALETTE[index % len(TRANSITION_PALETTE)] for index, transition in enumerate(transitions)}


def plot_arrow_panel(
    ax,
    panel_rows: list[dict],
    title: str,
    *,
    color_by_transition: bool,
    xlim=None,
    ylim=None,
) -> None:
    setup_axis(ax, title)
    if xlim is None or ylim is None:
        endpoints = np.asarray([[0.0, 0.0]] + [[row["pc1"], row["pc2"]] for row in panel_rows], dtype=float)
        max_abs = max(float(np.abs(endpoints).max()), 1.0)
        pad = 1.18 * max_abs
        xlim = (-pad, pad) if xlim is None else xlim
        ylim = (-pad, pad) if ylim is None else ylim
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    transition_colors = transition_color_map(panel_rows)
    for row in panel_rows:
        linestyle, linewidth = ARROW_STYLES.get(row["kind"], ("solid", 1.6))
        color = transition_colors[row["transition"]] if color_by_transition else COLORS[row["kind"]]
        ax.annotate(
            "",
            xy=(row["pc1"], row["pc2"]),
            xytext=(0.0, 0.0),
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
    ax.set_aspect("equal", adjustable="box")


def arrow_legend_handles(panel_rows: list[dict], *, color_by_transition: bool) -> list:
    if color_by_transition:
        transition_handles = [
            plt.Line2D([0], [0], color=color, linewidth=2, label=transition)
            for transition, color in transition_color_map(panel_rows).items()
        ]
        style_handles = [
            plt.Line2D([0], [0], color="#202426", linewidth=2, linestyle=ARROW_STYLES["true"][0], label="True"),
            plt.Line2D(
                [0],
                [0],
                color="#202426",
                linewidth=2,
                linestyle=ARROW_STYLES["transported"][0],
                label="Transported",
            ),
        ]
        return style_handles + transition_handles
    return [
        plt.Line2D([0], [0], color=COLORS["true"], linewidth=2, linestyle=ARROW_STYLES["true"][0], label="True destination"),
        plt.Line2D([0], [0], color=COLORS["direct"], linewidth=2, linestyle=ARROW_STYLES["direct"][0], label="Direct"),
        plt.Line2D(
            [0],
            [0],
            color=COLORS["operation_first"],
            linewidth=2,
            linestyle=ARROW_STYLES["operation_first"][0],
            label="Operation first",
        ),
        plt.Line2D(
            [0],
            [0],
            color=COLORS["modality_first"],
            linewidth=2,
            linestyle=ARROW_STYLES["modality_first"][0],
            label="Modality first",
        ),
    ]


def projected_arrow_rows(rows: list[dict], vectors: list[torch.Tensor]) -> tuple[list[dict], dict]:
    if not vectors:
        raise ValueError("No displacement vectors were available to project. Check shared result transitions.")
    projected, pca = pca_project_arrows(torch.stack(vectors))
    for row, point in zip(rows, projected):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])
    return rows, pca


def plot_transported_arrows(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    data: dict[tuple[str, int], TaskData],
    *,
    seed_selection: dict[str, list[int]],
    hidden_size: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    transitions = [parse_transition(item) for item in args.transitions]
    rows = []
    pca_by_direction = {}
    for pair_spec in args.same_operation_pairs:
        source_task, destination_task = parse_pair(pair_spec)
        source_seed = seed_selection[source_task][0]
        destination_seed = seed_selection[destination_task][0]
        source = load_task_space(args, space_cache, task=source_task, seed=source_seed, hidden_size=hidden_size)
        destination = load_task_space(args, space_cache, task=destination_task, seed=destination_seed, hidden_size=hidden_size)
        source_data = data[(source_task, source_seed)]
        destination_data = data[(destination_task, destination_seed)]
        fitted = map_for(
            args,
            activation_cache,
            sample_cache,
            map_cache,
            source=source,
            destination=destination,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        valid, transition_source = selected_transition_rows(transitions, source_data.centroids, destination_data.centroids)
        print(
            f"  Plot C {source_task}->{destination_task}: using {len(valid)} transitions "
            f"({transition_source}; requested={len(transitions)})"
        )
        direction_rows = []
        direction_vectors = []
        for start, end in valid:
            source_delta = source_data.centroids[end] - source_data.centroids[start]
            true_delta = destination_data.centroids[end] - destination_data.centroids[start]
            transported = float(fitted["alpha"]) * (source_delta @ fitted["q"])
            for kind, vector in [("true", true_delta), ("transported", transported)]:
                direction_rows.append(
                    {
                        "plot": "C",
                        "source_task": source_task,
                        "destination_task": destination_task,
                        "source_seed": source_seed,
                        "destination_seed": destination_seed,
                        "transition": f"{start}->{end}",
                        "kind": kind,
                        "label": f"{start}->{end} {kind}",
                        "alpha": float(fitted["alpha"]),
                        "transition_source": transition_source,
                    }
                )
                direction_vectors.append(vector)
        if direction_vectors:
            direction_rows, pca = projected_arrow_rows(direction_rows, direction_vectors)
            direction_key = f"{source_task}->{destination_task}"
            pca_by_direction[direction_key] = pca
            rows.extend(direction_rows)
    directions = list(dict.fromkeys(f"{row['source_task']}->{row['destination_task']}" for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    fig.patch.set_facecolor(BACKGROUND)
    for ax, direction in zip(axes.ravel(), directions):
        panel_rows = [row for row in rows if f"{row['source_task']}->{row['destination_task']}" == direction]
        plot_arrow_panel(
            ax,
            panel_rows,
            f"Plot C - {task_short(direction.split('->')[0])} -> {task_short(direction.split('->')[1])}",
            color_by_transition=True,
        )
    for ax in axes.ravel()[len(directions):]:
        ax.axis("off")
    handles = arrow_legend_handles(rows, color_by_transition=True)
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.26)
    save_figure(fig, args, "plot_c_transported_displacement_arrows")
    save_jsonl(rows, args.output_dir / "plot_c_transported_displacement_arrows.jsonl")
    return {"plot": "C", "pca_by_direction": pca_by_direction, "n_arrows": len(rows)}


def edge_seed(source_task: str, destination_task: str, edge_task: str, source_seed: int, destination_seed: int) -> int:
    edge_modality, _edge_operation = parse_task(edge_task)
    source_modality, _source_operation = parse_task(source_task)
    return source_seed if edge_modality == source_modality else destination_seed


def factorized_maps_for_pair(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    *,
    source_task: str,
    destination_task: str,
    source_seed: int,
    destination_seed: int,
    hidden_size: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict[str, tuple[torch.Tensor, float]]:
    details = factorized_path_details_for_pair(
        args,
        activation_cache,
        sample_cache,
        space_cache,
        map_cache,
        source_task=source_task,
        destination_task=destination_task,
        source_seed=source_seed,
        destination_seed=destination_seed,
        hidden_size=hidden_size,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return {transport: (parts["q"], float(parts["alpha"])) for transport, parts in details.items()}


def factorized_path_details_for_pair(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    *,
    source_task: str,
    destination_task: str,
    source_seed: int,
    destination_seed: int,
    hidden_size: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict[str, dict]:
    specs = path_specs(source_task, destination_task)
    outputs = {}
    for transport, edges in specs.items():
        fitted_edges = []
        for edge_source_task, edge_destination_task in edges:
            edge_source_seed = edge_seed(source_task, destination_task, edge_source_task, source_seed, destination_seed)
            edge_destination_seed = edge_seed(source_task, destination_task, edge_destination_task, source_seed, destination_seed)
            edge_source = load_task_space(args, space_cache, task=edge_source_task, seed=edge_source_seed, hidden_size=hidden_size)
            edge_destination = load_task_space(
                args, space_cache, task=edge_destination_task, seed=edge_destination_seed, hidden_size=hidden_size
            )
            fitted_edges.append(
                map_for(
                    args,
                    activation_cache,
                    sample_cache,
                    map_cache,
                    source=edge_source,
                    destination=edge_destination,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    blocks=blocks,
                )
            )
        if len(fitted_edges) == 1:
            outputs[transport] = {
                "q": fitted_edges[0]["q"],
                "alpha": float(fitted_edges[0]["alpha"]),
                "edges": fitted_edges,
                "edge_tasks": edges,
            }
        else:
            q, alpha = compose_maps(fitted_edges[0], fitted_edges[1])
            outputs[transport] = {
                "q": q,
                "alpha": float(alpha),
                "edges": fitted_edges,
                "edge_tasks": edges,
            }
    return outputs


def plot_factorized_parallelogram(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    space_cache: dict,
    map_cache: dict,
    data: dict[tuple[str, int], TaskData],
    *,
    seed_selection: dict[str, list[int]],
    hidden_size: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    source_task, destination_task = parse_pair(args.factorized_pair)
    source_seed = seed_selection[source_task][0]
    destination_seed = seed_selection[destination_task][0]
    source_data = data[(source_task, source_seed)]
    destination_data = data[(destination_task, destination_seed)]
    path_details = factorized_path_details_for_pair(
        args,
        activation_cache,
        sample_cache,
        space_cache,
        map_cache,
        source_task=source_task,
        destination_task=destination_task,
        source_seed=source_seed,
        destination_seed=destination_seed,
        hidden_size=hidden_size,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    maps = {transport: (parts["q"], float(parts["alpha"])) for transport, parts in path_details.items()}
    requested_transitions = [parse_transition(item) for item in args.transitions]
    transitions, transition_source = selected_transition_rows(
        requested_transitions,
        source_data.centroids,
        destination_data.centroids,
        require_same_width=True,
    )
    print(
        f"  Plot D {source_task}->{destination_task}: using {len(transitions)} transitions "
        f"({transition_source}; requested={len(args.transitions)})"
    )
    rows = []
    vectors = []
    two_step_rows = []
    two_step_vectors = []
    for start, end in transitions:
        source_delta = source_data.centroids[end] - source_data.centroids[start]
        true_delta = destination_data.centroids[end] - destination_data.centroids[start]
        rows.append(
            {
                "plot": "D",
                "source_task": source_task,
                "destination_task": destination_task,
                "source_seed": source_seed,
                "destination_seed": destination_seed,
                "transition": f"{start}->{end}",
                "kind": "true",
                "label": f"{start}->{end} true",
                "transition_source": transition_source,
            }
        )
        vectors.append(true_delta)
        for transport, (q, alpha) in maps.items():
            predicted = float(alpha) * (source_delta @ q)
            rows.append(
                {
                    "plot": "D",
                    "source_task": source_task,
                    "destination_task": destination_task,
                    "source_seed": source_seed,
                    "destination_seed": destination_seed,
                    "transition": f"{start}->{end}",
                    "kind": transport,
                    "label": f"{start}->{end} {transport}",
                    "alpha": float(alpha),
                    "transition_source": transition_source,
                }
            )
            vectors.append(predicted)
        zero = torch.zeros_like(source_delta)
        for transport in ["operation_first", "modality_first"]:
            parts = path_details.get(transport)
            if not parts or len(parts["edges"]) != 2:
                continue
            first_edge = parts["edges"][0]
            intermediate_task = parts["edge_tasks"][0][1]
            intermediate = float(first_edge["alpha"]) * (source_delta @ first_edge["q"])
            final = float(parts["alpha"]) * (source_delta @ parts["q"])
            for stage, kind, vector in [
                ("source", "source", zero),
                ("intermediate", f"{transport}_intermediate", intermediate),
                ("final", transport, final),
            ]:
                two_step_rows.append(
                    {
                        "plot": "D_two_step",
                        "source_task": source_task,
                        "destination_task": destination_task,
                        "source_seed": source_seed,
                        "destination_seed": destination_seed,
                        "transition": f"{start}->{end}",
                        "path": transport,
                        "stage": stage,
                        "kind": kind,
                        "intermediate_task": intermediate_task if stage == "intermediate" else None,
                        "transition_source": transition_source,
                    }
                )
                two_step_vectors.append(vector)
        for path, kind, vector in [("direct", "direct", float(maps["direct"][1]) * (source_delta @ maps["direct"][0])), ("true", "true", true_delta)]:
            two_step_rows.append(
                {
                    "plot": "D_two_step",
                    "source_task": source_task,
                    "destination_task": destination_task,
                    "source_seed": source_seed,
                    "destination_seed": destination_seed,
                    "transition": f"{start}->{end}",
                    "path": path,
                    "stage": "endpoint",
                    "kind": kind,
                    "intermediate_task": None,
                    "transition_source": transition_source,
                }
            )
            two_step_vectors.append(vector)
    rows, pca = projected_arrow_rows(rows, vectors)
    if two_step_vectors:
        two_step_rows, two_step_pca = projected_arrow_rows(two_step_rows, two_step_vectors)
        save_jsonl(two_step_rows, args.output_dir / "plot_d_factorized_two_step_paths.jsonl")
    else:
        two_step_pca = {}
    transitions_present = list(dict.fromkeys(row["transition"] for row in rows))
    cols = min(2, max(1, len(transitions_present)))
    rows_n = int(np.ceil(len(transitions_present) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.7 * cols, 3.8 * rows_n), squeeze=False)
    fig.patch.set_facecolor(BACKGROUND)
    for ax, transition in zip(axes.ravel(), transitions_present):
        panel_rows = [row for row in rows if row["transition"] == transition]
        plot_arrow_panel(
            ax,
            panel_rows,
            f"Plot D - {task_short(source_task)} -> {task_short(destination_task)} | {transition}",
            color_by_transition=False,
        )
    for ax in axes.ravel()[len(transitions_present):]:
        ax.axis("off")
    handles = arrow_legend_handles(rows, color_by_transition=False)
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
    save_figure(fig, args, "plot_d_factorized_path_parallelogram")
    save_jsonl(rows, args.output_dir / "plot_d_factorized_path_parallelogram.jsonl")
    return {"plot": "D", **pca, "two_step_pca": two_step_pca, "n_arrows": len(rows), "n_two_step_points": len(two_step_rows)}


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    same_operation_pairs = [parse_pair(item) for item in args.same_operation_pairs]
    factorized_source, factorized_destination = parse_pair(args.factorized_pair)
    reference_task = task_key(*parse_task(args.reference_task))
    needed_tasks = list(
        dict.fromkeys(
            args.tasks
            + [task for pair in same_operation_pairs for task in pair]
            + [factorized_source, factorized_destination]
            + [reference_task]
            + [task for edge in path_specs(factorized_source, factorized_destination).values() for pair in edge for task in pair]
        )
    )
    args.tasks = needed_tasks
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    style_matplotlib()
    seed_selection = selected_seeds(args)
    print("Procrustes PCA plots")
    print(f"  split={args.plot_split}; tasks={', '.join(args.tasks)}")
    print(f"  reference task={reference_task}")
    print(f"  factorized map dir={args.factorized_map_dir}")
    print("  selected seeds:")
    for task in args.tasks:
        print(f"    {task}: {seed_selection[task]}")

    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    activation_cache = {}
    sample_cache = {}
    space_cache = {}
    map_cache = {}

    data = {}
    for task in TASK_ORDER:
        if task not in args.tasks:
            continue
        seed = seed_selection[task][0]
        data[(task, seed)] = load_task_data(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            task=task,
            seed=seed,
            hidden_size=hidden_size,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
    reference_seed = seed_selection[reference_task][0]
    reference = load_task_space(args, space_cache, task=reference_task, seed=reference_seed, hidden_size=hidden_size)

    summaries = [
        plot_raw_result_centroids(args, data),
        plot_aligned_result_centroids(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            data,
            reference=reference,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        ),
        plot_pooled_value_residuals(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            data,
            reference=reference,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        ),
        plot_condition_specific_residuals(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            data,
            reference=reference,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        ),
        plot_transported_arrows(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            data,
            seed_selection=seed_selection,
            hidden_size=hidden_size,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        ),
        plot_factorized_parallelogram(
            args,
            activation_cache,
            sample_cache,
            space_cache,
            map_cache,
            data,
            seed_selection=seed_selection,
            hidden_size=hidden_size,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        ),
    ]
    save_json(
        {
            "experiment": EXPERIMENT,
            "model": model_name,
            "config": jsonable(vars(args)),
            "summaries": summaries,
        },
        args.output_dir / "pca_plot_summary.json",
    )
    print(f"Saved {args.output_dir / 'pca_plot_summary.json'}")


if __name__ == "__main__":
    main()
