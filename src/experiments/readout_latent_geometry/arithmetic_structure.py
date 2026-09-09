"""Readout-free arithmetic shift structure in the shared numerical manifold.

This offline experiment asks whether readout-free value geometry supports
operators for +Delta that are learned from text-addition training values and
then transfer to held-out ranges and other tasks. It reuses the existing
activation loading, basis construction, Procrustes, centroid, and retrieval
utilities from the previous closing experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from src.experiments.readout_latent_geometry import readout_free_specificity as specificity
from src.experiments.readout_latent_geometry import value_heldout_geometry as geom
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    jsonable,
    parse_task,
    save_json,
    subspace_path,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed


EXPERIMENT = "arithmetic_structure_readout_free"
REFERENCE_TASK = "text:addition"
DEFAULT_REPRESENTATIONS = ["pca_leading_readout_free", "das_readout_free"]
OPERATOR_TYPES = ["translation", "scaled_orthogonal_affine", "ridge_affine"]
OFFSETS = [1, 2, 5, 10]
RIDGE_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


@dataclass
class AlignedSpace:
    task: str
    representation_type: str
    source_seed: int
    destination_seed: int
    centroids: dict[int, torch.Tensor]
    alignment: dict


@dataclass
class Operator:
    operator_type: str
    offset: int
    params: dict
    selected_lambda: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--readout_free_audit_root",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43"),
    )
    parser.add_argument(
        "--digit_readout_basis_path",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43/digit_readout_basis.pt"),
    )
    parser.add_argument(
        "--activation_dir_text",
        type=Path,
        default=Path("outputs/activations/baseline/gemma4_12b_it/digits"),
    )
    parser.add_argument(
        "--activation_dir_image",
        type=Path,
        default=Path("outputs/activations/baseline_images/gemma4_12b_it/digits"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/experiments/closing/arithmetic_structure_readout_free"),
    )
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=geom.DEFAULT_TASKS)
    parser.add_argument("--representations", nargs="+", default=DEFAULT_REPRESENTATIONS)
    parser.add_argument("--splits", nargs="+", default=["high", "low", "middle"], choices=["high", "low", "middle", "all"])
    parser.add_argument("--offsets", type=int, nargs="+", default=OFFSETS)
    parser.add_argument("--operator_types", nargs="+", default=OPERATOR_TYPES, choices=OPERATOR_TYPES + ["all"])
    parser.add_argument("--shuffled_controls", type=int, default=20)
    parser.add_argument("--ridge_lambdas", type=float, nargs="+", default=RIDGE_LAMBDAS)
    parser.add_argument("--values_min", type=int, default=10)
    parser.add_argument("--values_max", type=int, default=89, help="Inclusive maximum value.")
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="DAS seeds.")
    parser.add_argument("--rsa_permutations", type=int, default=0, help="Unused; accepted for wrapper symmetry.")
    parser.add_argument("--readout_overlap_tolerance", type=float, default=1e-6)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--skip_controls", action="store_true")
    parser.add_argument("--skip_composition", action="store_true")
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.tasks == geom.DEFAULT_TASKS:
        args.tasks = ["text:addition", "image:addition"]
    if args.representations == DEFAULT_REPRESENTATIONS:
        args.representations = ["pca_leading_readout_free"]
    if args.splits == ["high", "low", "middle"]:
        args.splits = ["high"]
    if args.offsets == OFFSETS:
        args.offsets = [1, 5]
    args.shuffled_controls = min(args.shuffled_controls, 2)


def normalize_args(args: argparse.Namespace) -> None:
    apply_smoke_defaults(args)
    args.tasks = specificity.normalize_tasks(args.tasks)
    if REFERENCE_TASK not in args.tasks:
        raise ValueError(f"{REFERENCE_TASK} must be included because all shared operators are fit in the T+ frame.")
    if "all" in args.representations:
        args.representations = list(DEFAULT_REPRESENTATIONS)
    args.representations = specificity.normalize_spaces(args.representations)
    args.space_types = list(args.representations)
    if "all" in args.operator_types:
        args.operator_types = list(OPERATOR_TYPES)
    if "all" in args.splits:
        args.splits = ["high", "low", "middle"]
    if args.values_min > args.values_max:
        raise ValueError("--values_min must be <= --values_max.")


def structured_value_splits(values_min: int, values_max: int, requested: list[str]) -> dict[str, dict[str, list[int]]]:
    if values_min != 10 or values_max != 89:
        values = list(range(values_min, values_max + 1))
        midpoint = len(values) // 2
        first = values[:midpoint]
        second = values[midpoint:]
        middle_start = len(values) // 4
        middle_end = len(values) - middle_start
        splits = {
            "high": {"train_values": first, "test_values": second},
            "low": {"train_values": second, "test_values": first},
            "middle": {"train_values": values[:middle_start] + values[middle_end:], "test_values": values[middle_start:middle_end]},
        }
    else:
        splits = {
            "high": {"train_values": list(range(10, 50)), "test_values": list(range(50, 90))},
            "low": {"train_values": list(range(50, 90)), "test_values": list(range(10, 50))},
            "middle": {"train_values": list(range(10, 30)) + list(range(70, 90)), "test_values": list(range(30, 70))},
        }
    return {name: splits[name] for name in requested}


def shift_pairs(values: list[int], offset: int) -> list[tuple[int, int]]:
    present = set(values)
    return [(value, value + offset) for value in values if value + offset in present]


def matrix_for_pairs(centroids: dict[int, torch.Tensor], pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([centroids[source] for source, _target in pairs])
    y = torch.stack([centroids[target] for _source, target in pairs])
    return x.float(), y.float()


def fit_alignment(
    source_centroids: dict[int, torch.Tensor],
    reference_centroids: dict[int, torch.Tensor],
    train_values: list[int],
) -> tuple[dict[int, torch.Tensor], dict]:
    transitions, x_train = geom.transition_matrix(source_centroids, train_values)
    _transitions_ref, y_train = geom.transition_matrix(reference_centroids, train_values)
    q, alpha = geom.fit_scaled_procrustes(x_train, y_train)
    offsets = [
        reference_centroids[value] - float(alpha) * (source_centroids[value] @ q)
        for value in train_values
    ]
    bias = torch.stack(offsets).mean(dim=0)
    aligned = {value: bias + float(alpha) * (centroid @ q) for value, centroid in source_centroids.items()}
    train_metrics = geom.transition_metrics(x_train, y_train, q, alpha)
    return aligned, {
        "alignment_alpha": alpha,
        "alignment_bias_norm": float(bias.norm()),
        "alignment_train_transition_cosine": train_metrics["heldout_transition_cosine_mean"],
        "alignment_train_relative_error": train_metrics["heldout_relative_error_mean"],
        "alignment_n_train_transitions": len(transitions),
    }


def build_aligned_spaces(
    args: argparse.Namespace,
    spaces: dict[tuple[str, str], list[specificity.SpaceItem]],
    split_name: str,
    train_values: list[int],
) -> tuple[dict[tuple[str, str, int, int], AlignedSpace], list[dict]]:
    aligned = {}
    diagnostics = []
    for representation in args.representations:
        ref_items = spaces[(REFERENCE_TASK, representation)]
        source_seeds = [item.space_seed for item in ref_items]
        if representation == "pca_leading_readout_free":
            source_seeds = [0]
        for source_seed in source_seeds:
            reference = next(item for item in ref_items if item.space_seed == source_seed)
            for task in args.tasks:
                task_items = spaces[(task, representation)]
                if representation == "pca_leading_readout_free":
                    task_items = [task_items[0]]
                for destination in task_items:
                    if task == REFERENCE_TASK and destination.space_seed == source_seed:
                        centroids = dict(reference.centroids)
                        alignment = {
                            "alignment_alpha": 1.0,
                            "alignment_bias_norm": 0.0,
                            "alignment_train_transition_cosine": 1.0,
                            "alignment_train_relative_error": 0.0,
                            "alignment_n_train_transitions": len(geom.transition_matrix(reference.centroids, train_values)[0]),
                        }
                    else:
                        centroids, alignment = fit_alignment(destination.centroids, reference.centroids, train_values)
                    key = (representation, task, source_seed, destination.space_seed)
                    aligned[key] = AlignedSpace(
                        task=task,
                        representation_type=representation,
                        source_seed=source_seed,
                        destination_seed=destination.space_seed,
                        centroids=centroids,
                        alignment=alignment,
                    )
                    diagnostics.append(
                        {
                            "split": split_name,
                            "representation_type": representation,
                            "reference_task": REFERENCE_TASK,
                            "evaluation_task": task,
                            "source_seed": source_seed,
                            "destination_seed": destination.space_seed,
                            **alignment,
                        }
                    )
    return aligned, diagnostics


def fit_translation(x: torch.Tensor, y: torch.Tensor) -> Operator:
    vector = (y - x).mean(dim=0)
    return Operator("translation", 0, {"v": vector.float()}, None)


def fit_scaled_orthogonal_affine(x: torch.Tensor, y: torch.Tensor) -> Operator:
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    q, alpha = geom.fit_scaled_procrustes(x_centered, y_centered)
    bias = y_mean - float(alpha) * (x_mean @ q)
    return Operator("scaled_orthogonal_affine", 0, {"q": q.float(), "alpha": float(alpha), "b": bias.float()}, None)


def fit_ridge_given_lambda(x: torch.Tensor, y: torch.Tensor, lambda_value: float) -> tuple[torch.Tensor, torch.Tensor]:
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    x_centered = x - x_mean
    y_centered = y - y_mean
    dim = x.shape[1]
    gram = x_centered.T @ x_centered
    rhs = x_centered.T @ y_centered
    if lambda_value == 0.0:
        transform = torch.linalg.pinv(gram) @ rhs
    else:
        regularized = gram + float(lambda_value) * torch.eye(dim, dtype=x.dtype)
        transform = torch.linalg.solve(regularized, rhs)
    bias = y_mean - x_mean @ transform
    return transform.float(), bias.float()


def cv_ridge_lambda(x: torch.Tensor, y: torch.Tensor, lambdas: list[float], seed: int) -> float:
    n = x.shape[0]
    if n < 5:
        return lambdas[0]
    k_folds = min(5, n)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    folds = [indices[i::k_folds] for i in range(k_folds)]
    scores = []
    for lambda_value in lambdas:
        fold_scores = []
        for fold in folds:
            train = [index for index in indices if index not in set(fold)]
            if not train or not fold:
                continue
            transform, bias = fit_ridge_given_lambda(x[train], y[train], lambda_value)
            pred = x[fold] @ transform + bias
            fold_scores.append(float((pred - y[fold]).square().mean()))
        scores.append((float(sum(fold_scores) / len(fold_scores)), lambda_value))
    scores.sort(key=lambda item: (item[0], item[1]))
    return float(scores[0][1])


def fit_ridge_affine(x: torch.Tensor, y: torch.Tensor, lambdas: list[float], seed: int) -> Operator:
    selected = cv_ridge_lambda(x, y, lambdas, seed)
    transform, bias = fit_ridge_given_lambda(x, y, selected)
    return Operator("ridge_affine", 0, {"a": transform, "b": bias}, selected)


def fit_operator(operator_type: str, x: torch.Tensor, y: torch.Tensor, lambdas: list[float], seed: int) -> Operator:
    if operator_type == "translation":
        return fit_translation(x, y)
    if operator_type == "scaled_orthogonal_affine":
        return fit_scaled_orthogonal_affine(x, y)
    if operator_type == "ridge_affine":
        return fit_ridge_affine(x, y, lambdas, seed)
    raise ValueError(f"Unknown operator_type={operator_type}")


def with_offset(operator: Operator, offset: int) -> Operator:
    return Operator(operator.operator_type, offset, operator.params, operator.selected_lambda)


def apply_operator(operator: Operator, x: torch.Tensor) -> torch.Tensor:
    if operator.operator_type == "identity":
        return x
    if operator.operator_type == "translation":
        return x + operator.params["v"]
    if operator.operator_type == "scaled_orthogonal_affine":
        return float(operator.params["alpha"]) * (x @ operator.params["q"]) + operator.params["b"]
    if operator.operator_type == "ridge_affine":
        return x @ operator.params["a"] + operator.params["b"]
    raise ValueError(f"Unknown operator_type={operator.operator_type}")


def operator_serializable_params(operator: Operator) -> dict:
    output = {"operator_type": operator.operator_type, "offset": operator.offset}
    for key, value in operator.params.items():
        output[key] = value.cpu() if isinstance(value, torch.Tensor) else value
    output["selected_lambda"] = operator.selected_lambda
    return output


def retrieve_value(predicted: torch.Tensor, centroids: dict[int, torch.Tensor], candidate_values: list[int], target_value: int) -> tuple[int, bool, bool, int]:
    matrix = torch.stack([centroids[value] for value in candidate_values])
    distances = (matrix - predicted).norm(dim=1)
    order = torch.argsort(distances).tolist()
    predicted_value = candidate_values[order[0]]
    top5_values = [candidate_values[index] for index in order[:5]]
    return predicted_value, predicted_value == target_value, target_value in top5_values, abs(predicted_value - target_value)


def evaluate_operator(
    operator: Operator,
    centroids: dict[int, torch.Tensor],
    test_pairs: list[tuple[int, int]],
    candidate_values: list[int],
) -> dict:
    cosines = []
    rel_errors = []
    exact = 0
    top5 = 0
    numeric_errors = []
    pair_rows = []
    for source_value, target_value in test_pairs:
        source = centroids[source_value]
        target = centroids[target_value]
        predicted = apply_operator(operator, source)
        predicted_displacement = predicted - source
        true_displacement = target - source
        cos = torch.nn.functional.cosine_similarity(predicted_displacement[None], true_displacement[None]).item()
        rel = float((predicted_displacement - true_displacement).norm() / true_displacement.norm().clamp_min(1e-12))
        predicted_value, exact_ok, top5_ok, abs_error = retrieve_value(predicted, centroids, candidate_values, target_value)
        cosines.append(float(cos))
        rel_errors.append(rel)
        exact += int(exact_ok)
        top5 += int(top5_ok)
        numeric_errors.append(abs_error)
        pair_rows.append(
            {
                "source_value": source_value,
                "target_value": target_value,
                "predicted_value": predicted_value,
                "displacement_cosine": float(cos),
                "relative_error": rel,
                "absolute_numeric_error": abs_error,
            }
        )
    n = len(test_pairs)
    if n == 0:
        raise ValueError(f"No test pairs for offset={operator.offset}")
    return {
        "n_test_pairs": n,
        "mean_displacement_cosine": float(sum(cosines) / n),
        "median_displacement_cosine": float(statistics.median(cosines)),
        "mean_relative_error": float(sum(rel_errors) / n),
        "median_relative_error": float(statistics.median(rel_errors)),
        "top1_exact_target": exact / n,
        "top5_target": top5 / n,
        "mean_absolute_numeric_error": float(sum(numeric_errors) / n),
        "median_absolute_numeric_error": float(statistics.median(numeric_errors)),
    }


def fit_shared_operator(
    operator_type: str,
    offset: int,
    reference_space: AlignedSpace,
    train_pairs: list[tuple[int, int]],
    args: argparse.Namespace,
    seed: int,
) -> Operator:
    x, y = matrix_for_pairs(reference_space.centroids, train_pairs)
    operator = fit_operator(operator_type, x, y, args.ridge_lambdas, seed)
    return with_offset(operator, offset)


def shuffled_train_targets(train_pairs: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    sources = [source for source, _target in train_pairs]
    targets = [target for _source, target in train_pairs]
    random.Random(seed).shuffle(targets)
    return list(zip(sources, targets))


def main_result_row(
    *,
    representation: str,
    split_name: str,
    offset: int,
    operator: Operator,
    fit_source: str,
    evaluation: AlignedSpace,
    n_train_pairs: int,
    metrics: dict,
    extra: dict | None = None,
) -> dict:
    return {
        "representation_type": representation,
        "split": split_name,
        "offset": offset,
        "operator_type": operator.operator_type,
        "fit_source": fit_source,
        "evaluation_task": evaluation.task,
        "evaluation_task_label": geom.TASK_LABELS[evaluation.task],
        "source_seed": evaluation.source_seed,
        "destination_seed": evaluation.destination_seed,
        "n_train_pairs": n_train_pairs,
        "selected_ridge_lambda": operator.selected_lambda,
        **metrics,
        **(extra or {}),
    }


def composition_metrics(
    first: Operator,
    direct: Operator,
    centroids: dict[int, torch.Tensor],
    test_values: list[int],
    step: int,
    candidate_values: list[int],
) -> dict:
    pairs = [(value, value + 2 * step) for value in test_values if value + 2 * step in set(test_values)]
    composed_cosines = []
    direct_cosines = []
    composed_top1 = 0
    direct_top1 = 0
    composed_errors = []
    direct_errors = []
    for source_value, target_value in pairs:
        source = centroids[source_value]
        target = centroids[target_value]
        composed = apply_operator(first, apply_operator(first, source))
        direct_pred = apply_operator(direct, source)
        true_displacement = target - source
        composed_displacement = composed - source
        direct_displacement = direct_pred - source
        composed_cosines.append(float(torch.nn.functional.cosine_similarity(composed_displacement[None], true_displacement[None])))
        direct_cosines.append(float(torch.nn.functional.cosine_similarity(direct_displacement[None], true_displacement[None])))
        _pred, exact, _top5, abs_err = retrieve_value(composed, centroids, candidate_values, target_value)
        composed_top1 += int(exact)
        composed_errors.append(abs_err)
        _pred, exact, _top5, abs_err = retrieve_value(direct_pred, centroids, candidate_values, target_value)
        direct_top1 += int(exact)
        direct_errors.append(abs_err)
    n = len(pairs)
    if n == 0:
        raise ValueError(f"No composition test pairs for step={step}")
    composed_top1_value = composed_top1 / n
    direct_top1_value = direct_top1 / n
    composed_cosine = float(sum(composed_cosines) / n)
    direct_cosine = float(sum(direct_cosines) / n)
    return {
        "n_test_pairs": n,
        "composed_target_cosine": composed_cosine,
        "direct_operator_target_cosine": direct_cosine,
        "composed_top1": composed_top1_value,
        "direct_operator_top1": direct_top1_value,
        "composed_mean_absolute_numeric_error": float(sum(composed_errors) / n),
        "direct_mean_absolute_numeric_error": float(sum(direct_errors) / n),
        "composition_gap_top1": composed_top1_value - direct_top1_value,
        "composition_gap_cosine": composed_cosine - direct_cosine,
    }


def summarize_results(results: list[dict], controls: list[dict], compositions: list[dict]) -> dict:
    groups = {}
    for row in results:
        if row["fit_source"] not in {"shared_Tplus", "task_specific"}:
            continue
        key = (
            row["representation_type"],
            row["operator_type"],
            row["offset"],
            row["evaluation_task"],
            row["fit_source"],
        )
        groups.setdefault(key, []).append(row)
    summary_rows = []
    for key, parts in sorted(groups.items()):
        representation, operator_type, offset, task, fit_source = key
        row = {
            "representation_type": representation,
            "operator_type": operator_type,
            "offset": offset,
            "evaluation_task": task,
            "fit_source": fit_source,
            "n": len(parts),
        }
        for metric in ("mean_displacement_cosine", "top1_exact_target", "mean_absolute_numeric_error"):
            values = [float(part[metric]) for part in parts]
            row[f"{metric}_mean"] = float(sum(values) / len(values))
            row[f"{metric}_std"] = None if len(values) < 2 else float(statistics.stdev(values))
        summary_rows.append(row)

    shuffled = {}
    for row in controls:
        key = (row["representation_type"], row["operator_type"], row["offset"], row["evaluation_task"])
        shuffled.setdefault(key, []).append(row)
    shuffled_summary = []
    for key, parts in sorted(shuffled.items()):
        representation, operator_type, offset, task = key
        values = [float(row["top1_exact_target"]) for row in parts]
        cosines = [float(row["mean_displacement_cosine"]) for row in parts]
        shuffled_summary.append(
            {
                "representation_type": representation,
                "operator_type": operator_type,
                "offset": offset,
                "evaluation_task": task,
                "n": len(parts),
                "shuffled_top1_mean": float(sum(values) / len(values)),
                "shuffled_top1_std": None if len(values) < 2 else float(statistics.stdev(values)),
                "shuffled_displacement_cosine_mean": float(sum(cosines) / len(cosines)),
                "shuffled_displacement_cosine_std": None if len(cosines) < 2 else float(statistics.stdev(cosines)),
            }
        )

    ratio_rows = []
    shared_lookup = {
        (row["representation_type"], row["operator_type"], row["offset"], row["evaluation_task"]): row
        for row in summary_rows
        if row["fit_source"] == "shared_Tplus"
    }
    for row in summary_rows:
        if row["fit_source"] != "task_specific":
            continue
        key = (row["representation_type"], row["operator_type"], row["offset"], row["evaluation_task"])
        shared = shared_lookup.get(key)
        if shared:
            upper = row["top1_exact_target_mean"]
            ratio_rows.append(
                {
                    **{field: row[field] for field in ("representation_type", "operator_type", "offset", "evaluation_task")},
                    "shared_top1": shared["top1_exact_target_mean"],
                    "task_specific_top1": upper,
                    "shared_task_specific_top1_ratio": None if abs(upper) < 1e-12 else shared["top1_exact_target_mean"] / upper,
                    "shared_displacement_cosine": shared["mean_displacement_cosine_mean"],
                    "task_specific_displacement_cosine": row["mean_displacement_cosine_mean"],
                }
            )

    composition_summary = geom.summarize(
        compositions,
        ["composed_top1", "direct_operator_top1", "composed_target_cosine", "direct_operator_target_cosine"],
        ["representation_type", "operator_type", "composition", "evaluation_task"],
    ) if compositions else []
    return {
        "operator_metric_summary": summary_rows,
        "shared_vs_task_specific": ratio_rows,
        "shuffled_control_summary": shuffled_summary,
        "composition_summary": composition_summary,
    }


def plot_figures(args: argparse.Namespace, results: list[dict], controls: list[dict], compositions: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    task_order = [task for task in geom.TASK_ORDER if task in args.tasks]
    op_labels = {
        "translation": "translation",
        "scaled_orthogonal_affine": "scaled orthogonal",
        "ridge_affine": "ridge affine",
    }

    primary = "pca_leading_readout_free"
    if primary in args.representations:
        fig, axes = plt.subplots(1, len(args.offsets), figsize=(4.1 * len(args.offsets), 3.6), sharey=True)
        if len(args.offsets) == 1:
            axes = [axes]
        for ax, offset in zip(axes, args.offsets):
            x = np.arange(len(task_order))
            width = 0.22
            for index, operator_type in enumerate(args.operator_types):
                values = []
                for task in task_order:
                    rows = [
                        row for row in results
                        if row["representation_type"] == primary
                        and row["fit_source"] == "shared_Tplus"
                        and row["offset"] == offset
                        and row["operator_type"] == operator_type
                        and row["evaluation_task"] == task
                    ]
                    values.append(np.nan if not rows else float(np.mean([float(row["top1_exact_target"]) for row in rows])))
                ax.bar(x + (index - 1) * width, values, width, label=op_labels[operator_type])
            control_rows = [
                row for row in controls
                if row["representation_type"] == primary and row["offset"] == offset
            ]
            if control_rows:
                ax.axhline(
                    float(np.mean([float(row["top1_exact_target"]) for row in control_rows])),
                    color="black",
                    linewidth=1.0,
                    alpha=0.45,
                    linestyle="--",
                )
            ax.set_title(f"+{offset}")
            ax.set_xticks(x, [geom.TASK_LABELS[task] for task in task_order])
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("exact target retrieval")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.suptitle("PCA readout-free arithmetic shift generalization")
        fig.tight_layout()
        fig.savefig(figure_dir / "figure1_arithmetic_shift_generalization_pca_rf.png", dpi=240)
        fig.savefig(figure_dir / "figure1_arithmetic_shift_generalization_pca_rf.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        labels = []
        shared = []
        task_specific = []
        for offset in args.offsets:
            for task in task_order:
                labels.append(f"+{offset} {geom.TASK_LABELS[task]}")
                for target_list, fit_source in ((shared, "shared_Tplus"), (task_specific, "task_specific")):
                    rows = [
                        row for row in results
                        if row["representation_type"] == primary
                        and row["operator_type"] == "scaled_orthogonal_affine"
                        and row["fit_source"] == fit_source
                        and row["offset"] == offset
                        and row["evaluation_task"] == task
                    ]
                    target_list.append(np.nan if not rows else float(np.mean([float(row["mean_displacement_cosine"]) for row in rows])))
        x = np.arange(len(labels))
        ax.bar(x - 0.18, shared, 0.36, label="T+ trained")
        ax.bar(x + 0.18, task_specific, 0.36, label="task-specific")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_ylabel("displacement cosine")
        ax.set_title("Shared vs task-specific scaled-orthogonal shifts")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "figure2_shared_vs_task_specific_pca_rf.png", dpi=240)
        fig.savefig(figure_dir / "figure2_shared_vs_task_specific_pca_rf.pdf")
        plt.close(fig)

        if compositions:
            fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
            for ax, composition in zip(axes, ["1+1=2", "5+5=10"]):
                rows = [
                    row for row in compositions
                    if row["representation_type"] == primary
                    and row["operator_type"] == "scaled_orthogonal_affine"
                    and row["composition"] == composition
                ]
                composed = []
                direct = []
                for task in task_order:
                    parts = [row for row in rows if row["evaluation_task"] == task]
                    composed.append(np.nan if not parts else float(np.mean([float(row["composed_top1"]) for row in parts])))
                    direct.append(np.nan if not parts else float(np.mean([float(row["direct_operator_top1"]) for row in parts])))
                x = np.arange(len(task_order))
                ax.bar(x - 0.18, direct, 0.36, label="direct")
                ax.bar(x + 0.18, composed, 0.36, label="composed")
                ax.set_title(composition)
                ax.set_xticks(x, [geom.TASK_LABELS[task] for task in task_order])
                ax.grid(axis="y", alpha=0.25)
            axes[0].set_ylabel("exact target retrieval")
            axes[-1].legend(frameon=False)
            fig.suptitle("PCA readout-free on-manifold composition")
            fig.tight_layout()
            fig.savefig(figure_dir / "figure3_operator_composition_pca_rf.png", dpi=240)
            fig.savefig(figure_dir / "figure3_operator_composition_pca_rf.pdf")
            plt.close(fig)

    if "das_readout_free" in args.representations:
        fig, ax = plt.subplots(figsize=(7.8, 4.0))
        labels = []
        values = []
        for offset in args.offsets:
            for task in task_order:
                labels.append(f"+{offset} {geom.TASK_LABELS[task]}")
                rows = [
                    row for row in results
                    if row["representation_type"] == "das_readout_free"
                    and row["fit_source"] == "shared_Tplus"
                    and row["operator_type"] == "scaled_orthogonal_affine"
                    and row["offset"] == offset
                    and row["evaluation_task"] == task
                ]
                values.append(np.nan if not rows else float(np.mean([float(row["top1_exact_target"]) for row in rows])))
        ax.bar(np.arange(len(labels)), values)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylabel("exact target retrieval")
        ax.set_title("Supplementary DAS readout-free shift generalization")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "supplement_das_rf_shift_generalization.png", dpi=240)
        fig.savefig(figure_dir / "supplement_das_rf_shift_generalization.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    normalize_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    values = list(range(args.values_min, args.values_max + 1))
    value_splits = structured_value_splits(args.values_min, args.values_max, args.splits)
    save_json(value_splits, args.output_dir / "arithmetic_value_splits.json")

    print("Arithmetic structure in readout-free spaces")
    print(f"  tasks={args.tasks}")
    print(f"  representations={args.representations}")
    print(f"  values={values[0]}..{values[-1]} n={len(values)}")
    print(f"  splits={list(value_splits)}")
    print(f"  offsets={args.offsets}")

    rows_by_task = {}
    data_by_task = {}
    for task in args.tasks:
        modality, operation = parse_task(task)
        row = geom.first_result_row(args, modality, operation, args.condition, args.seeds[0])
        geom.validate_result_metadata(args, row, task=task, path=subspace_path(args, modality, operation, args.seeds[0]))
        rows_by_task[task] = row
        data_by_task[task] = geom.load_task_data(args, task, row)

    value_count_rows = []
    for task, data in data_by_task.items():
        counts = [data.counts.get(value, 0) for value in values]
        missing = [value for value, count in zip(values, counts) if count == 0]
        if missing:
            raise ValueError(f"{task} is missing selected values: {missing[:20]}")
        print(f"  {task}: min/median/max examples per selected value = {min(counts)}/{statistics.median(counts)}/{max(counts)}")
        value_count_rows.append(
            {
                "task": task,
                "task_label": geom.TASK_LABELS[task],
                "min_examples_per_value": min(counts),
                "median_examples_per_value": statistics.median(counts),
                "max_examples_per_value": max(counts),
            }
        )
    geom.write_csv(value_count_rows, args.output_dir / "selected_value_counts.csv")

    spaces, space_diagnostics = specificity.build_spaces(args, data_by_task, rows_by_task, values)
    geom.write_csv(space_diagnostics, args.output_dir / "space_diagnostics.csv")

    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "representations": args.representations,
                "values": values,
                "value_splits": value_splits,
                "space_diagnostics_rows": len(space_diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("ARTIFACT_CHECK_ONLY complete")
        return

    result_rows = []
    control_rows = []
    composition_rows = []
    alignment_rows = []
    operator_parameters = {}
    candidate_values = values

    for split_name, split in value_splits.items():
        train_values = split["train_values"]
        test_values = split["test_values"]
        aligned_spaces, split_alignment_rows = build_aligned_spaces(args, spaces, split_name, train_values)
        alignment_rows.extend(split_alignment_rows)
        print(f"\nSPLIT {split_name}: train={train_values[0]}..{train_values[-1]} n={len(train_values)} test={test_values[0]}..{test_values[-1]} n={len(test_values)}")
        for offset in args.offsets:
            train_pairs = shift_pairs(train_values, offset)
            test_pairs = shift_pairs(test_values, offset)
            print(f"  offset=+{offset}: n_train_shift_pairs={len(train_pairs)} n_test_shift_pairs={len(test_pairs)}")
            if not train_pairs or not test_pairs:
                raise ValueError(f"Split {split_name} offset {offset} has train={len(train_pairs)} test={len(test_pairs)} pairs.")

            for representation in args.representations:
                ref_source_seeds = [0] if representation == "pca_leading_readout_free" else args.seeds
                for source_seed in ref_source_seeds:
                    reference = aligned_spaces[(representation, REFERENCE_TASK, source_seed, source_seed)]
                    shared_ops = {}
                    for operator_type in args.operator_types:
                        op_seed = stable_seed(EXPERIMENT, "operator", representation, split_name, offset, operator_type, source_seed)
                        operator = fit_shared_operator(operator_type, offset, reference, train_pairs, args, op_seed)
                        shared_ops[operator_type] = operator
                        operator_parameters[("shared_Tplus", representation, split_name, offset, operator_type, source_seed)] = operator_serializable_params(operator)

                        for task in args.tasks:
                            task_items = [key for key in aligned_spaces if key[0] == representation and key[1] == task and key[2] == source_seed]
                            for key in task_items:
                                evaluation = aligned_spaces[key]
                                metrics = evaluate_operator(operator, evaluation.centroids, test_pairs, candidate_values)
                                result_rows.append(
                                    main_result_row(
                                        representation=representation,
                                        split_name=split_name,
                                        offset=offset,
                                        operator=operator,
                                        fit_source="shared_Tplus",
                                        evaluation=evaluation,
                                        n_train_pairs=len(train_pairs),
                                        metrics=metrics,
                                    )
                                )

                        if not args.skip_controls:
                            for control_seed in range(args.shuffled_controls):
                                shuffled = shuffled_train_targets(
                                    train_pairs,
                                    stable_seed(EXPERIMENT, "shuffle", representation, split_name, offset, operator_type, source_seed, control_seed),
                                )
                                x_shuffle, y_shuffle = matrix_for_pairs(reference.centroids, shuffled)
                                control_op = with_offset(
                                    fit_operator(operator_type, x_shuffle, y_shuffle, args.ridge_lambdas, stable_seed(EXPERIMENT, "control_op", representation, split_name, offset, operator_type, source_seed, control_seed)),
                                    offset,
                                )
                                for task in args.tasks:
                                    task_items = [key for key in aligned_spaces if key[0] == representation and key[1] == task and key[2] == source_seed]
                                    for key in task_items:
                                        evaluation = aligned_spaces[key]
                                        metrics = evaluate_operator(control_op, evaluation.centroids, test_pairs, candidate_values)
                                        control_rows.append(
                                            {
                                                **main_result_row(
                                                    representation=representation,
                                                    split_name=split_name,
                                                    offset=offset,
                                                    operator=control_op,
                                                    fit_source="shuffled_Tplus",
                                                    evaluation=evaluation,
                                                    n_train_pairs=len(train_pairs),
                                                    metrics=metrics,
                                                ),
                                                "control_seed": control_seed,
                                            }
                                        )

                    identity = Operator("identity", offset, {}, None)
                    for task in args.tasks:
                        task_items = [key for key in aligned_spaces if key[0] == representation and key[1] == task and key[2] == source_seed]
                        for key in task_items:
                            evaluation = aligned_spaces[key]
                            identity_metrics = evaluate_operator(identity, evaluation.centroids, test_pairs, candidate_values)
                            result_rows.append(
                                main_result_row(
                                    representation=representation,
                                    split_name=split_name,
                                    offset=offset,
                                    operator=identity,
                                    fit_source="identity",
                                    evaluation=evaluation,
                                    n_train_pairs=0,
                                    metrics=identity_metrics,
                                )
                            )

        for representation in args.representations:
            ref_source_seeds = [0] if representation == "pca_leading_readout_free" else args.seeds
            for source_seed in ref_source_seeds:
                shared_by_offset = {}
                for offset in args.offsets:
                    train_pairs = shift_pairs(train_values, offset)
                    reference = aligned_spaces[(representation, REFERENCE_TASK, source_seed, source_seed)]
                    for operator_type in args.operator_types:
                        op_seed = stable_seed(EXPERIMENT, "operator", representation, split_name, offset, operator_type, source_seed)
                        shared_by_offset[(operator_type, offset)] = fit_shared_operator(operator_type, offset, reference, train_pairs, args, op_seed)
                if not args.skip_composition:
                    for operator_type, step, direct_offset, label in [
                        ("translation", 1, 2, "1+1=2"),
                        ("scaled_orthogonal_affine", 1, 2, "1+1=2"),
                        ("ridge_affine", 1, 2, "1+1=2"),
                        ("translation", 5, 10, "5+5=10"),
                        ("scaled_orthogonal_affine", 5, 10, "5+5=10"),
                        ("ridge_affine", 5, 10, "5+5=10"),
                    ]:
                        if operator_type not in args.operator_types or step not in args.offsets or direct_offset not in args.offsets:
                            continue
                        first = shared_by_offset[(operator_type, step)]
                        direct = shared_by_offset[(operator_type, direct_offset)]
                        for task in args.tasks:
                            task_items = [key for key in aligned_spaces if key[0] == representation and key[1] == task and key[2] == source_seed]
                            for key in task_items:
                                evaluation = aligned_spaces[key]
                                metrics = composition_metrics(first, direct, evaluation.centroids, test_values, step, candidate_values)
                                composition_rows.append(
                                    {
                                        "representation_type": representation,
                                        "split": split_name,
                                        "composition": label,
                                        "operator_type": operator_type,
                                        "evaluation_task": task,
                                        "evaluation_task_label": geom.TASK_LABELS[task],
                                        "source_seed": evaluation.source_seed,
                                        "destination_seed": evaluation.destination_seed,
                                        "step_offset": step,
                                        "direct_offset": direct_offset,
                                        **metrics,
                                    }
                                )

        for representation in args.representations:
            for task in args.tasks:
                source_seeds = [0] if representation == "pca_leading_readout_free" else args.seeds
                for source_seed in source_seeds:
                    task_items = [key for key in aligned_spaces if key[0] == representation and key[1] == task and key[2] == source_seed]
                    for key in task_items:
                        evaluation = aligned_spaces[key]
                        for offset in args.offsets:
                            train_pairs = shift_pairs(train_values, offset)
                            test_pairs = shift_pairs(test_values, offset)
                            x_task, y_task = matrix_for_pairs(evaluation.centroids, train_pairs)
                            for operator_type in args.operator_types:
                                task_operator = with_offset(
                                    fit_operator(
                                        operator_type,
                                        x_task,
                                        y_task,
                                        args.ridge_lambdas,
                                        stable_seed(EXPERIMENT, "task_specific", representation, split_name, offset, operator_type, source_seed, task, evaluation.destination_seed),
                                    ),
                                    offset,
                                )
                                operator_parameters[("task_specific", representation, split_name, offset, operator_type, source_seed, task, evaluation.destination_seed)] = operator_serializable_params(task_operator)
                                metrics = evaluate_operator(task_operator, evaluation.centroids, test_pairs, candidate_values)
                                result_rows.append(
                                    main_result_row(
                                        representation=representation,
                                        split_name=split_name,
                                        offset=offset,
                                        operator=task_operator,
                                        fit_source="task_specific",
                                        evaluation=evaluation,
                                        n_train_pairs=len(train_pairs),
                                        metrics=metrics,
                                    )
                                )

        geom.write_csv(result_rows, args.output_dir / "arithmetic_operator_results.csv")
        geom.write_csv(control_rows, args.output_dir / "arithmetic_operator_controls.csv")
        geom.write_csv(composition_rows, args.output_dir / "arithmetic_composition_results.csv")
        geom.write_csv(alignment_rows, args.output_dir / "task_alignment_diagnostics.csv")
        print(f"CHECKPOINT_ARITHMETIC_STRUCTURE split={split_name} output_dir={args.output_dir}")

    torch.save(operator_parameters, args.output_dir / "operator_parameters.pt")
    summary = summarize_results(result_rows, control_rows, composition_rows)
    summary.update(
        {
            "experiment": EXPERIMENT,
            "description": (
                "Offline readout-free arithmetic shift operators fit on T+ train values, "
                "evaluated on structured held-out ranges and transferred task manifolds."
            ),
            "tasks": args.tasks,
            "task_labels": geom.TASK_LABELS,
            "representations": args.representations,
            "values": values,
            "value_splits": value_splits,
            "offsets": args.offsets,
            "operator_types": args.operator_types,
            "n_result_rows": len(result_rows),
            "n_control_rows": len(control_rows),
            "n_composition_rows": len(composition_rows),
            "config": jsonable(vars(args)),
        }
    )
    save_json(summary, args.output_dir / "summary.json")
    if not args.skip_plots:
        plot_figures(args, result_rows, control_rows, composition_rows)

    print("\nArithmetic structure readout-free complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Result rows: {len(result_rows)}")
    print(f"Control rows: {len(control_rows)}")
    print(f"Composition rows: {len(composition_rows)}")


if __name__ == "__main__":
    main()
