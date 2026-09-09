"""Factorized Procrustes paths in the exact 13D readout-orthogonal L spaces.

This is the L-space analogue of the Phase 2 ``factorized_paths`` experiment.
It preserves the old path algebra:

    Q_path = Q_first @ Q_second
    alpha_path = alpha_first * alpha_second

for row-vector coordinates.  The added fit-3 analysis is deliberately separate:
one complete task condition is omitted from the factor/path fit, the other three
corners predict the omitted condition in the diagonal observed reference frame,
and the omitted task is used only afterward for evaluation.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path

import torch

from src.experiments.global_geometry import causal_subspace_geometry as lgeom
from src.experiments.readout_latent_geometry import value_heldout_geometry as geom
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    jsonable,
    parse_task,
    save_json,
    save_jsonl,
    selected_seeds,
)
from src.experiments.cross_condition_transfer.procrustes.factorized_paths import factorized_paths as old_factorization
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed
from src.experiments.global_geometry.synchronization import synchronization as sync
from src.experiments.global_geometry.synchronization import synchronization_L as lsync


EXPERIMENT = "readout_orthogonal_L_factorized_paths"
SPACE_TYPE = lsync.SPACE_TYPE
TASKS = list(lsync.TASKS)
FACTOR_TRANSPORTS = [
    "pairwise_direct",
    "synchronized_full",
    "factorized_full_direct",
    "factorized_full_operation_first",
    "factorized_full_modality_first",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--previous_geometry_dir",
        type=Path,
        default=Path("results/experiments/closing/value_heldout_readout_free_geometry"),
        help="Directory containing the exact value_splits.json to reuse.",
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
        default=Path("results/paper/procrustes/factorized_paths/readout_orthogonal_L"),
    )
    parser.add_argument(
        "--old_das_summary",
        type=Path,
        default=Path("results/paper/procrustes/factorized_paths/factorized_paths_summary.jsonl"),
    )
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--seed_map", nargs="*", default=[], metavar="TASK=SEEDS")
    parser.add_argument("--value_split_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--m_readout", type=int, default=9)
    parser.add_argument("--latent_dim", type=int, default=13)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--alignment_seed", type=int, default=0)
    parser.add_argument("--weight_metric", choices=["mean_cosine", "inverse_rmse", "uniform"], default="mean_cosine")
    parser.add_argument("--sanity_tolerance", type=float, default=1e-5)
    parser.add_argument("--shuffled_controls", type=int, default=4)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--synthetic_smoke_test", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_tasks(tasks: list[str]) -> list[str]:
    if any(task.lower() == "all" for task in tasks):
        return list(TASKS)
    return [old_factorization.task_key(*parse_task(task)) for task in tasks]


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.seeds == [0, 1, 2]:
        args.seeds = [0]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]
    args.shuffled_controls = min(args.shuffled_controls, 1)


def task_label(task: str) -> str:
    return lsync.task_label(task)


def directed_pairs(tasks: list[str]) -> list[tuple[str, str]]:
    return lsync.directed_pairs(tasks)


def same_modality_neighbor(task: str) -> str:
    modality, operation = parse_task(task)
    return old_factorization.task_key(modality, old_factorization.opposite_operation(operation))


def same_operation_neighbor(task: str) -> str:
    modality, operation = parse_task(task)
    return old_factorization.task_key(old_factorization.opposite_modality(modality), operation)


def diagonal_task(task: str) -> str:
    modality, operation = parse_task(task)
    return old_factorization.task_key(
        old_factorization.opposite_modality(modality),
        old_factorization.opposite_operation(operation),
    )


def compose_edges(first: sync.Edge, second: sync.Edge) -> tuple[torch.Tensor, float]:
    return old_factorization.compose_maps(
        {"q": first.q, "alpha": first.alpha},
        {"q": second.q, "alpha": second.alpha},
    )


def evaluate_edge_transport(
    args: argparse.Namespace,
    *,
    source: lsync.LSpace,
    destination: lsync.LSpace,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
    transport: str,
    q: torch.Tensor,
    alpha: float,
    direct_edge: sync.Edge,
    component_paths: list[Path],
    synchronization_path: Path | None = None,
    orientations: dict[str, torch.Tensor] | None = None,
    log_scales: dict[str, float] | None = None,
) -> dict:
    row = lsync.evaluate_transport(
        args,
        source=source,
        destination=destination,
        value_split_seed=value_split_seed,
        train_values=train_values,
        test_values=test_values,
        transport=transport,
        q=q,
        alpha=alpha,
        direct_edge=direct_edge,
        synchronization_path=synchronization_path,
        heldout_relation=None,
        orientations=orientations,
        log_scales=log_scales,
    )
    row.update(
        {
            "experiment": EXPERIMENT,
            "analysis": "full_four_condition_factorized_path",
            "component_map_paths": [str(path) for path in component_paths],
            "factorization_source": "old_phase2_factorized_paths.path_specs_and_compose_maps",
        }
    )
    return row


def path_transports(
    source_task: str,
    destination_task: str,
    edge_by_pair: dict[tuple[str, str], sync.Edge],
) -> dict[str, tuple[torch.Tensor, float, list[Path]]]:
    direct = edge_by_pair[(source_task, destination_task)]
    specs = old_factorization.path_specs(source_task, destination_task)
    transports = {
        "factorized_full_direct": (direct.q, direct.alpha, [direct.path]),
    }
    if "operation_first" in specs:
        first_src, first_dst = specs["operation_first"][0]
        second_src, second_dst = specs["operation_first"][1]
        first = edge_by_pair[(first_src, first_dst)]
        second = edge_by_pair[(second_src, second_dst)]
        q, alpha = compose_edges(first, second)
        transports["factorized_full_operation_first"] = (q, alpha, [first.path, second.path])
    if "modality_first" in specs:
        first_src, first_dst = specs["modality_first"][0]
        second_src, second_dst = specs["modality_first"][1]
        first = edge_by_pair[(first_src, first_dst)]
        second = edge_by_pair[(second_src, second_dst)]
        q, alpha = compose_edges(first, second)
        transports["factorized_full_modality_first"] = (q, alpha, [first.path, second.path])
    return transports


def path_consistency_row(
    args: argparse.Namespace,
    *,
    source: lsync.LSpace,
    destination: lsync.LSpace,
    edge_by_pair: dict[tuple[str, str], sync.Edge],
    seed: int,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
) -> dict | None:
    transports = path_transports(source.task, destination.task, edge_by_pair)
    if "factorized_full_operation_first" not in transports or "factorized_full_modality_first" not in transports:
        return None
    q_op, alpha_op, op_paths = transports["factorized_full_operation_first"]
    q_mod, alpha_mod, mod_paths = transports["factorized_full_modality_first"]
    direct = edge_by_pair[(source.task, destination.task)]
    _train_transitions, x_train = geom.transition_matrix(source.centroids, train_values)
    _test_transitions, x_test = geom.transition_matrix(source.centroids, test_values)
    _true_transitions, y_test = geom.transition_matrix(destination.centroids, test_values)
    pred_op = float(alpha_op) * (x_test @ q_op)
    pred_mod = float(alpha_mod) * (x_test @ q_mod)
    q_error = float((q_op - q_mod).norm() / q_mod.norm().clamp_min(1e-12))
    operator_error = sync.relative_operator_distance(q_op, alpha_op, q_mod, alpha_mod)
    prediction_error = (pred_op - pred_mod).norm(dim=1) / pred_mod.norm(dim=1).clamp_min(1e-12)
    prediction_cosine = torch.nn.functional.cosine_similarity(pred_op, pred_mod, dim=1)
    op_true_cosine = torch.nn.functional.cosine_similarity(pred_op, y_test, dim=1)
    mod_true_cosine = torch.nn.functional.cosine_similarity(pred_mod, y_test, dim=1)
    return {
        "experiment": EXPERIMENT,
        "space_type": SPACE_TYPE,
        "analysis": "commuting_path_consistency",
        "source_task": source.task,
        "source_label": task_label(source.task),
        "destination_task": destination.task,
        "destination_label": task_label(destination.task),
        "task_relation": lsync.directed_key(source.task, destination.task),
        "seed": seed,
        "value_split_seed": value_split_seed,
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "n_train_transitions": int(x_train.shape[0]),
        "n_test_transitions": int(x_test.shape[0]),
        "operation_first_paths": [str(path) for path in op_paths],
        "modality_first_paths": [str(path) for path in mod_paths],
        "direct_map_path": str(direct.path),
        "Q_path_relative_frobenius_error": q_error,
        "Q_path_trace_similarity": float(torch.trace(q_op.T @ q_mod) / q_op.shape[0]),
        "scale_operation_first": float(alpha_op),
        "scale_modality_first": float(alpha_mod),
        "scale_path_log_error": abs(math.log(max(alpha_op, 1e-12)) - math.log(max(alpha_mod, 1e-12))),
        "operator_path_relative_distance": operator_error,
        "path_prediction_cosine_mean": lsync.mean([float(v) for v in prediction_cosine.tolist()]),
        "path_prediction_cosine_median": float(statistics.median([float(v) for v in prediction_cosine.tolist()])),
        "path_prediction_relative_error_mean": lsync.mean([float(v) for v in prediction_error.tolist()]),
        "operation_first_true_cosine_mean": lsync.mean([float(v) for v in op_true_cosine.tolist()]),
        "modality_first_true_cosine_mean": lsync.mean([float(v) for v in mod_true_cosine.tolist()]),
        "fit_value_leakage_check": "passed_disjoint_train_test_values",
    }


def map_centroids(
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
    values: list[int],
    train_values: list[int],
    edge: sync.Edge,
) -> dict[int, torch.Tensor]:
    offsets = [
        destination_centroids[value] - float(edge.alpha) * (source_centroids[value] @ edge.q)
        for value in train_values
    ]
    offset = torch.stack(offsets).mean(dim=0)
    return {
        value: offset + float(edge.alpha) * (source_centroids[value] @ edge.q)
        for value in values
    }


def centered_same_value_metrics(
    predicted: dict[int, torch.Tensor],
    true: dict[int, torch.Tensor],
    values: list[int],
    prefix: str,
) -> dict:
    if not values:
        return {
            f"{prefix}_same_value_cosine_mean": None,
            f"{prefix}_different_value_cosine_mean": None,
        }
    pred_center = torch.stack([predicted[value] for value in values]).mean(dim=0)
    true_center = torch.stack([true[value] for value in values]).mean(dim=0)
    same = []
    different = []
    for source_value in values:
        pred = predicted[source_value] - pred_center
        same.append(float(torch.nn.functional.cosine_similarity(pred[None], (true[source_value] - true_center)[None])))
        for target_value in values:
            if source_value != target_value:
                different.append(
                    float(torch.nn.functional.cosine_similarity(pred[None], (true[target_value] - true_center)[None]))
                )
    return {
        f"{prefix}_same_value_cosine_mean": lsync.mean(same),
        f"{prefix}_same_value_cosine_median": float(statistics.median(same)),
        f"{prefix}_same_value_cosine_std": lsync.sample_std(same),
        f"{prefix}_same_value_cosine_sem": lsync.sem(same),
        f"{prefix}_different_value_cosine_mean": lsync.mean(different),
        f"{prefix}_different_value_cosine_median": float(statistics.median(different)) if different else None,
        f"{prefix}_different_value_cosine_std": lsync.sample_std(different),
        f"{prefix}_different_value_cosine_sem": lsync.sem(different),
    }


def centroid_retrieval_metrics(
    predicted: dict[int, torch.Tensor],
    true: dict[int, torch.Tensor],
    train_values: list[int],
    eval_values: list[int],
    prefix: str,
) -> dict:
    offsets = [true[value] - predicted[value] for value in train_values]
    offset = torch.stack(offsets).mean(dim=0)
    true_matrix = torch.stack([true[value] for value in eval_values])
    true_center = true_matrix.mean(dim=0)
    exact = 0
    top5 = 0
    abs_errors = []
    cosines = []
    for value in eval_values:
        mapped = offset + predicted[value]
        true_value = true[value]
        cosines.append(float(torch.nn.functional.cosine_similarity((mapped - true_center)[None], (true_value - true_center)[None])))
        distances = (true_matrix - mapped).norm(dim=1)
        order = torch.argsort(distances).tolist()
        predicted_value = eval_values[order[0]]
        exact += int(predicted_value == value)
        top5 += int(value in [eval_values[index] for index in order[:5]])
        abs_errors.append(abs(predicted_value - value))
    return {
        f"{prefix}_top1_value_retrieval": exact / len(eval_values),
        f"{prefix}_top5_value_retrieval": top5 / len(eval_values),
        f"{prefix}_mean_absolute_value_error": float(sum(abs_errors) / len(abs_errors)),
        f"{prefix}_median_absolute_value_error": float(statistics.median(abs_errors)),
        f"{prefix}_same_value_cosine_mean": float(sum(cosines) / len(cosines)),
        f"{prefix}_same_value_cosine_median": float(statistics.median(cosines)),
    }


def fit3_prediction(
    args: argparse.Namespace,
    *,
    spaces: dict[str, lsync.LSpace],
    edge_by_pair: dict[tuple[str, str], sync.Edge],
    heldout_task: str,
    seed: int,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
    control_seed: int | None = None,
) -> dict:
    reference_task = diagonal_task(heldout_task)
    mod_neighbor_task = same_modality_neighbor(heldout_task)
    op_neighbor_task = same_operation_neighbor(heldout_task)
    fit_tasks = sorted([task for task in spaces if task != heldout_task])
    assert heldout_task not in fit_tasks
    if {reference_task, mod_neighbor_task, op_neighbor_task} != set(fit_tasks):
        raise RuntimeError(f"Bad fit-3 corner construction for {heldout_task}: {fit_tasks}")

    all_values = sorted(set(train_values) | set(test_values))
    reference = spaces[reference_task]
    heldout = spaces[heldout_task]
    mod_neighbor = spaces[mod_neighbor_task]
    op_neighbor = spaces[op_neighbor_task]
    mod_in_ref = map_centroids(
        mod_neighbor.centroids,
        reference.centroids,
        all_values,
        train_values,
        edge_by_pair[(mod_neighbor_task, reference_task)],
    )
    op_in_ref = map_centroids(
        op_neighbor.centroids,
        reference.centroids,
        all_values,
        train_values,
        edge_by_pair[(op_neighbor_task, reference_task)],
    )
    reference_in_ref = {value: reference.centroids[value] for value in all_values}

    if control_seed is not None:
        shuffled_values = list(all_values)
        random.Random(control_seed).shuffle(shuffled_values)
        mod_in_ref = {value: mod_in_ref[shuffled_values[index]] for index, value in enumerate(all_values)}

    predicted_in_ref = {
        value: mod_in_ref[value] + op_in_ref[value] - reference_in_ref[value]
        for value in all_values
    }
    eval_edge = edge_by_pair[(heldout_task, reference_task)]
    true_in_ref = map_centroids(
        heldout.centroids,
        reference.centroids,
        all_values,
        train_values,
        eval_edge,
    )

    train_transitions, x_train = geom.transition_matrix(predicted_in_ref, train_values)
    train_transitions_b, y_train = geom.transition_matrix(true_in_ref, train_values)
    test_transitions, x_test = geom.transition_matrix(predicted_in_ref, test_values)
    test_transitions_b, y_test = geom.transition_matrix(true_in_ref, test_values)
    if train_transitions != train_transitions_b or test_transitions != test_transitions_b:
        raise RuntimeError("Fit-3 predicted/true transition order mismatch.")
    identity = torch.eye(args.latent_dim)
    q_posthoc, alpha_posthoc = geom.fit_scaled_procrustes(x_train, y_train)
    posthoc_metrics = lsync.metric_set(x_test, y_test, q_posthoc, alpha_posthoc, "heldout_posthoc_aligned_transition")
    return {
        "experiment": EXPERIMENT,
        "space_type": SPACE_TYPE,
        "analysis": "fit3_leave_one_condition_out",
        "control_type": "none" if control_seed is None else "shuffled_mod_neighbor_values",
        "control_seed": control_seed,
        "transport": "factorized_fit3_square_closure" if control_seed is None else "factorized_fit3_shuffled_control",
        "heldout_task": heldout_task,
        "heldout_label": task_label(heldout_task),
        "fit_tasks": fit_tasks,
        "reference_task": reference_task,
        "reference_label": task_label(reference_task),
        "same_modality_neighbor_task": mod_neighbor_task,
        "same_operation_neighbor_task": op_neighbor_task,
        "seed": seed,
        "value_split_seed": value_split_seed,
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "train_values": train_values,
        "test_values": test_values,
        "n_train_transitions": len(train_transitions),
        "n_test_transitions": len(test_transitions),
        "source_L_dim": args.latent_dim,
        "destination_L_dim": args.latent_dim,
        "fit3_task_leakage_check": "passed_heldout_task_excluded_from_factorial_prediction",
        "fit_value_leakage_check": "passed_disjoint_train_test_values",
        "offset_correction": "constant offset fit on train values in the diagonal observed reference frame",
        "evaluation_map_leakage_check": (
            "direct heldout->reference map is computed only after the three-corner prediction "
            "and is used only to express true heldout geometry for evaluation"
        ),
        "heldout_evaluation_map_path": str(eval_edge.path),
        "map_reconstruction_note": (
            "A full heldout coordinate-basis map is not identifiable from three observed "
            "coordinate spaces alone; identity/posthoc errors below quantify the residual "
            "map needed after square-closure prediction in the reference frame."
        ),
        **lsync.metric_set(x_train, y_train, identity, 1.0, "train_transition"),
        **lsync.metric_set(x_test, y_test, identity, 1.0, "heldout_transition"),
        **centroid_retrieval_metrics(predicted_in_ref, true_in_ref, train_values, train_values, "train"),
        **centroid_retrieval_metrics(predicted_in_ref, true_in_ref, train_values, test_values, "heldout"),
        **centered_same_value_metrics(predicted_in_ref, true_in_ref, test_values, "heldout_centered"),
        "fit3_posthoc_Q_relative_frobenius_error_to_identity": float((q_posthoc - identity).norm() / identity.norm()),
        "fit3_posthoc_Q_trace_similarity_to_identity": float(torch.trace(q_posthoc) / q_posthoc.shape[0]),
        "fit3_posthoc_scale": float(alpha_posthoc),
        "fit3_posthoc_scale_log_error_to_identity": abs(math.log(max(alpha_posthoc, 1e-12))),
        **posthoc_metrics,
    }


def comparison_parameters(k: int) -> dict:
    rotation_dim = k * (k - 1) // 2
    return {
        "dimension": k,
        "pairwise_direct_independent_directed_maps": 12,
        "pairwise_rotation_parameters": 12 * rotation_dim,
        "pairwise_scale_parameters": 12,
        "synchronization_task_frames": 4,
        "synchronization_rotation_parameters_after_gauge": 3 * rotation_dim,
        "synchronization_scale_parameters_after_gauge": 3,
        "factorized_path_primitives": 8,
        "factorized_path_note": (
            "The old Phase 2 code uses shared row/column two-step path compositions, "
            "not a learned low-parameter optimizer; diagonal relations are predicted "
            "from two one-factor edges."
        ),
    }


def summarize(rows: list[dict], group_fields: list[str], metrics: list[str]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(str(row.get(field)) for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, parts in sorted(groups.items()):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(parts)
        for metric in metrics:
            values = [part.get(metric) for part in parts]
            item[f"{metric}_mean"] = lsync.mean(values)
            item[f"{metric}_std"] = lsync.sample_std(values)
            item[f"{metric}_sem"] = lsync.sem(values)
            clean = [float(value) for value in values if value is not None]
            item[f"{metric}_median"] = float(statistics.median(clean)) if clean else None
        output.append(item)
    return output


def compact_fit3_table(fit3_summary: list[dict]) -> list[dict]:
    rows = []
    for row in fit3_summary:
        if row.get("control_type") != "none":
            continue
        rows.append(
            {
                "heldout_task": row["heldout_task"],
                "n": row["n"],
                "transition_cosine": row.get("heldout_transition_cosine_mean_mean"),
                "retrieval_top1": row.get("heldout_top1_value_retrieval_mean"),
                "same_value_cosine": row.get("heldout_same_value_cosine_mean_mean"),
                "identity_map_error": row.get("fit3_posthoc_Q_relative_frobenius_error_to_identity_mean"),
                "scale_log_error": row.get("fit3_posthoc_scale_log_error_to_identity_mean"),
            }
        )
    parts = [row for row in fit3_summary if row.get("control_type") == "none"]
    if parts:
        rows.append(
            {
                "heldout_task": "ALL",
                "n": sum(int(row["n"]) for row in parts),
                "transition_cosine": lsync.mean([row.get("heldout_transition_cosine_mean_mean") for row in parts]),
                "retrieval_top1": lsync.mean([row.get("heldout_top1_value_retrieval_mean") for row in parts]),
                "same_value_cosine": lsync.mean([row.get("heldout_same_value_cosine_mean_mean") for row in parts]),
                "identity_map_error": lsync.mean(
                    [row.get("fit3_posthoc_Q_relative_frobenius_error_to_identity_mean") for row in parts]
                ),
                "scale_log_error": lsync.mean([row.get("fit3_posthoc_scale_log_error_to_identity_mean") for row in parts]),
            }
        )
    return rows


def compact_model_comparison(full_summary: list[dict], fit3_all: list[dict]) -> list[dict]:
    rows = []
    labels = {
        "pairwise_direct": "Direct Procrustes",
        "synchronized_full": "Synchronization",
        "factorized_full_operation_first": "Full factorization: operation first",
        "factorized_full_modality_first": "Full factorization: modality first",
        "factorized_full_direct": "Full factorization: primitive direct",
    }
    for row in full_summary:
        transport = row["transport"]
        rows.append(
            {
                "model": labels.get(transport, transport),
                "transport": transport,
                "n": row["n"],
                "transition_cosine": row.get("heldout_transition_cosine_mean_mean"),
                "retrieval_top1": row.get("top1_value_retrieval_mean"),
                "same_value_cosine": row.get("same_value_cosine_mean_mean"),
                "map_error": row.get("Q_relative_frobenius_error_mean"),
                "scale_log_error": row.get("scale_log_error_mean"),
            }
        )
    for row in fit3_all:
        if row.get("control_type") != "none":
            continue
        rows.append(
            {
                "model": "Fit-3 factorization",
                "transport": "factorized_fit3_square_closure",
                "n": row["n"],
                "transition_cosine": row.get("heldout_transition_cosine_mean_mean"),
                "retrieval_top1": row.get("heldout_top1_value_retrieval_mean"),
                "same_value_cosine": row.get("heldout_same_value_cosine_mean_mean"),
                "map_error": row.get("fit3_posthoc_Q_relative_frobenius_error_to_identity_mean"),
                "scale_log_error": row.get("fit3_posthoc_scale_log_error_to_identity_mean"),
            }
        )
    return rows


def load_old_das_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            import json

            rows.append(json.loads(line))
    return rows


def save_outputs(
    args: argparse.Namespace,
    *,
    diagnostics: list[dict],
    map_rows: list[dict],
    result_rows: list[dict],
    path_rows: list[dict],
    fit3_rows: list[dict],
    sync_rows: list[dict],
    value_splits: dict,
    seed_selection: dict[str, list[int]],
    sync_seeds: list[int],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geom.write_csv(diagnostics, args.output_dir / "L_space_diagnostics.csv")
    geom.write_csv(map_rows, args.output_dir / "L_pairwise_map_summary.csv")
    geom.write_csv(result_rows, args.output_dir / "L_factorized_geometric_results.csv")
    geom.write_csv(path_rows, args.output_dir / "L_path_consistency.csv")
    geom.write_csv(fit3_rows, args.output_dir / "L_fit3_results.csv")
    geom.write_csv(sync_rows, args.output_dir / "L_synchronization_fit_summary.csv")
    save_jsonl(result_rows, args.output_dir / "L_factorized_geometric_results.jsonl")
    save_jsonl(path_rows, args.output_dir / "L_path_consistency.jsonl")
    save_jsonl(fit3_rows, args.output_dir / "L_fit3_results.jsonl")

    full_metrics = [
        "heldout_transition_cosine_mean",
        "heldout_transition_relative_error_mean",
        "top1_value_retrieval",
        "top5_value_retrieval",
        "same_value_cosine_mean",
        "Q_relative_frobenius_error",
        "Q_trace_similarity",
        "scale_log_error",
        "operator_relative_distance_to_direct",
    ]
    path_metrics = [
        "Q_path_relative_frobenius_error",
        "Q_path_trace_similarity",
        "scale_path_log_error",
        "operator_path_relative_distance",
        "path_prediction_cosine_mean",
        "path_prediction_relative_error_mean",
        "operation_first_true_cosine_mean",
        "modality_first_true_cosine_mean",
    ]
    fit3_metrics = [
        "heldout_transition_cosine_mean",
        "heldout_transition_relative_error_mean",
        "heldout_top1_value_retrieval",
        "heldout_top5_value_retrieval",
        "heldout_same_value_cosine_mean",
        "heldout_centered_same_value_cosine_mean",
        "heldout_centered_different_value_cosine_mean",
        "fit3_posthoc_Q_relative_frobenius_error_to_identity",
        "fit3_posthoc_Q_trace_similarity_to_identity",
        "fit3_posthoc_scale_log_error_to_identity",
    ]
    full_summary = summarize(result_rows, ["transport"], full_metrics)
    relation_summary = summarize(result_rows, ["transport", "task_relation"], full_metrics)
    path_summary = summarize(path_rows, ["task_relation"], path_metrics)
    fit3_summary = summarize(fit3_rows, ["control_type", "heldout_task"], fit3_metrics)
    fit3_all = summarize(fit3_rows, ["control_type"], fit3_metrics)
    table = compact_fit3_table(fit3_summary)
    geom.write_csv(full_summary, args.output_dir / "L_model_comparison_summary.csv")
    geom.write_csv(relation_summary, args.output_dir / "L_relation_summary.csv")
    geom.write_csv(path_summary, args.output_dir / "L_path_consistency_summary.csv")
    geom.write_csv(fit3_summary, args.output_dir / "L_fit3_by_task_summary.csv")
    geom.write_csv(fit3_all, args.output_dir / "L_fit3_aggregate_summary.csv")
    geom.write_csv(table, args.output_dir / "L_fit3_summary_table.csv")
    geom.write_csv(compact_model_comparison(full_summary, fit3_all), args.output_dir / "L_main_model_comparison_table.csv")
    save_json(
        {
            "experiment": EXPERIMENT,
            "space_type": SPACE_TYPE,
            "description": (
                "Phase 2 factorized Procrustes path experiment rerun on exact 13D readout-orthogonal L spaces, "
                "with direct pairwise, full synchronization, full path factorization, path consistency, and "
                "fit-3 leave-one-condition-out square-closure prediction."
            ),
            "original_das_factorization_path": "src/experiments/cross_condition_transfer/procrustes/factorized_paths/factorized_paths.py",
            "old_math": {
                "fit": "scaled orthogonal Procrustes on train-value transition mean displacements",
                "row_vector_composition": "Q_path = Q_first @ Q_second; alpha_path = alpha_first * alpha_second",
                "operation_first": "source -> same-modality destination-operation -> destination",
                "modality_first": "source -> destination-modality same-operation -> destination",
                "optimizer": "none in old script; factorization is evaluated by composing fitted edge maps",
            },
            "reused_functions": [
                "synchronization_L.build_l_spaces",
                "synchronization_L.fit_or_load_edge",
                "synchronization_L.evaluate_transport",
                "synchronization.synchronize_rotations/scales/hub_map",
                "factorized_paths.path_specs",
                "factorized_paths.compose_maps",
                "value_heldout_readout_free_geometry.transition_matrix/retrieval-style offset correction",
            ],
            "tasks": args.tasks,
            "task_labels": geom.TASK_LABELS,
            "seed_selection": seed_selection,
            "sync_seeds": sync_seeds,
            "value_splits": value_splits,
            "L_dimension": args.latent_dim,
            "parameter_count_comparison": comparison_parameters(args.latent_dim),
            "full_factorization_summary": full_summary,
            "path_consistency_summary": path_summary,
            "fit3_by_task_summary": fit3_summary,
            "fit3_aggregate_summary": fit3_all,
            "old_full_das_summary_available": args.old_das_summary.exists(),
            "old_full_das_summary_path": str(args.old_das_summary),
            "old_full_das_summary_preview": load_old_das_summary(args.old_das_summary)[:20],
            "controls_and_leakage_checks": {
                "L_dimension_assertion": f"all spaces exactly {args.latent_dim}D",
                "C_L_orthogonality": "checked by causal_L_shared_geometry.construct_exact_CL diagnostics",
                "heldout_value_leakage": "pairwise/factorized maps are fit on train_values; test_values are disjoint",
                "fit3_task_leakage": "heldout task is excluded from the three-corner prediction fit",
                "fit3_evaluation_anchor": "heldout->reference direct map is computed only for evaluation in the reference frame",
                "autoregressive_iia": "not run; all reported metrics are geometric",
            },
            "config": jsonable(vars(args)),
        },
        args.output_dir / "L_factorization_summary.json",
    )


def plot_figures(args: argparse.Namespace, result_rows: list[dict], fit3_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(
        result_rows,
        ["transport"],
        ["heldout_transition_cosine_mean", "top1_value_retrieval", "Q_relative_frobenius_error"],
    )
    labels = [row["transport"].replace("factorized_full_", "fact_").replace("synchronized_full", "sync") for row in summary]
    for metric, ylabel, name in [
        ("heldout_transition_cosine_mean_mean", "held-out transition cosine", "L_factorized_transition_cosine"),
        ("top1_value_retrieval_mean", "top-1 retrieval", "L_factorized_retrieval"),
        ("Q_relative_frobenius_error_mean", "map error", "L_factorized_map_error"),
    ]:
        fig, ax = plt.subplots(figsize=(9.0, 3.8))
        values = [np.nan if row.get(metric) is None else row.get(metric) for row in summary]
        ax.bar(np.arange(len(labels)), values)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{name}.png", dpi=240)
        plt.close(fig)

    fit3 = [row for row in fit3_rows if row["control_type"] == "none"]
    if fit3:
        fit3_summary = summarize(fit3, ["heldout_task"], ["heldout_transition_cosine_mean", "heldout_top1_value_retrieval"])
        labels = [task_label(row["heldout_task"]) for row in fit3_summary]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        x = np.arange(len(labels))
        ax.bar(x - 0.18, [row["heldout_transition_cosine_mean_mean"] for row in fit3_summary], 0.36, label="transition")
        ax.bar(x + 0.18, [row["heldout_top1_value_retrieval_mean"] for row in fit3_summary], 0.36, label="top-1")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "L_fit3_by_task.png", dpi=240)
        plt.close(fig)


def run_for_seed_split(
    args: argparse.Namespace,
    tasks: list[str],
    spaces_by_key: dict[tuple[str, int], lsync.LSpace],
    *,
    seed: int,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    lsync.assert_value_split(train_values, test_values, f"seed={seed} value_split={value_split_seed}")
    spaces = {task: spaces_by_key[(task, seed)] for task in tasks}
    edge_by_pair = {}
    map_rows = []
    for source_task, destination_task in directed_pairs(tasks):
        edge = lsync.fit_or_load_edge(
            args,
            spaces[source_task],
            spaces[destination_task],
            value_split_seed=value_split_seed,
            train_values=train_values,
            test_values=test_values,
        )
        edge_by_pair[(source_task, destination_task)] = edge
        map_rows.append(
            {
                "source_task": source_task,
                "destination_task": destination_task,
                "task_relation": lsync.directed_key(source_task, destination_task),
                "seed": seed,
                "value_split_seed": value_split_seed,
                "path": str(edge.path),
                "alpha": edge.alpha,
                "weight": edge.weight,
                **edge.fit_metrics,
            }
        )

    orientations, rotation_stats = sync.synchronize_rotations(tasks, list(edge_by_pair.values()), args.latent_dim)
    log_scales, scale_stats = sync.synchronize_scales(tasks, list(edge_by_pair.values()))
    sync_path = args.output_dir / "maps" / f"L_factorized_sync_seed{seed}_valuesplit{value_split_seed}_layer{args.layer}_k{args.latent_dim}.pt"
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "L_factorized_full_sync_baseline",
            "experiment": EXPERIMENT,
            "space_type": SPACE_TYPE,
            "tasks": tasks,
            "seed": seed,
            "value_split_seed": value_split_seed,
            "orientations_domain_to_hub": {task: value.cpu() for task, value in orientations.items()},
            "log_scales": log_scales,
            "rotation_stats": rotation_stats,
            "scale_stats": scale_stats,
        },
        sync_path,
    )
    sync_rows = [
        {
            "condition": "synchronized_full",
            "seed": seed,
            "value_split_seed": value_split_seed,
            "n_edges": len(edge_by_pair),
            "synchronization_path": str(sync_path),
            "rotation_residual_mean": rotation_stats["rotation_residual_mean"],
            "rotation_residual_std": rotation_stats["rotation_residual_std"],
            "scale_residual_mean_abs": scale_stats["scale_residual_mean_abs"],
            "scale_residual_std_abs": scale_stats["scale_residual_std_abs"],
        }
    ]

    result_rows = []
    path_rows = []
    for source_task, destination_task in directed_pairs(tasks):
        source = spaces[source_task]
        destination = spaces[destination_task]
        direct = edge_by_pair[(source_task, destination_task)]
        result_rows.append(
            evaluate_edge_transport(
                args,
                source=source,
                destination=destination,
                value_split_seed=value_split_seed,
                train_values=train_values,
                test_values=test_values,
                transport="pairwise_direct",
                q=direct.q,
                alpha=direct.alpha,
                direct_edge=direct,
                component_paths=[direct.path],
            )
        )
        q_sync, alpha_sync = sync.hub_map(source_task, destination_task, orientations, log_scales)
        result_rows.append(
            evaluate_edge_transport(
                args,
                source=source,
                destination=destination,
                value_split_seed=value_split_seed,
                train_values=train_values,
                test_values=test_values,
                transport="synchronized_full",
                q=q_sync,
                alpha=alpha_sync,
                direct_edge=direct,
                component_paths=[sync_path],
                synchronization_path=sync_path,
                orientations=orientations,
                log_scales=log_scales,
            )
        )
        for transport, (q, alpha, paths) in path_transports(source_task, destination_task, edge_by_pair).items():
            result_rows.append(
                evaluate_edge_transport(
                    args,
                    source=source,
                    destination=destination,
                    value_split_seed=value_split_seed,
                    train_values=train_values,
                    test_values=test_values,
                    transport=transport,
                    q=q,
                    alpha=alpha,
                    direct_edge=direct,
                    component_paths=paths,
                )
            )
        path_row = path_consistency_row(
            args,
            source=source,
            destination=destination,
            edge_by_pair=edge_by_pair,
            seed=seed,
            value_split_seed=value_split_seed,
            train_values=train_values,
            test_values=test_values,
        )
        if path_row is not None:
            path_rows.append(path_row)

    fit3_rows = []
    for heldout_task in tasks:
        fit3_rows.append(
            fit3_prediction(
                args,
                spaces=spaces,
                edge_by_pair=edge_by_pair,
                heldout_task=heldout_task,
                seed=seed,
                value_split_seed=value_split_seed,
                train_values=train_values,
                test_values=test_values,
            )
        )
        for control_index in range(args.shuffled_controls):
            fit3_rows.append(
                fit3_prediction(
                    args,
                    spaces=spaces,
                    edge_by_pair=edge_by_pair,
                    heldout_task=heldout_task,
                    seed=seed,
                    value_split_seed=value_split_seed,
                    train_values=train_values,
                    test_values=test_values,
                    control_seed=stable_seed(EXPERIMENT, "fit3_shuffle", heldout_task, seed, value_split_seed, control_index),
                )
            )
    return result_rows, map_rows, sync_rows, path_rows, fit3_rows


def synthetic_spaces(args: argparse.Namespace) -> tuple[dict[tuple[str, int], lsync.LSpace], list[dict]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    base = {value: torch.randn(args.latent_dim, generator=generator) for value in range(12)}
    transforms = {}
    scales = {}
    for task_index, task in enumerate(TASKS):
        q, _ = torch.linalg.qr(torch.randn(args.latent_dim, args.latent_dim, generator=generator))
        transforms[task] = q.float()
        scales[task] = 1.0 + 0.05 * task_index
    spaces = {}
    diagnostics = []
    for task in TASKS:
        modality, operation = parse_task(task)
        centroids = {value: scales[task] * (base[value] @ transforms[task]) for value in base}
        basis = torch.eye(args.latent_dim)
        c_basis = torch.empty(args.latent_dim, 0)
        diag = {
            "task": task,
            "task_label": task_label(task),
            "digit_overlap": 0.0,
            "C_L_overlap": 0.0,
            "orthogonality_error": 0.0,
            "split_projector_error": 0.0,
        }
        spaces[(task, 0)] = lsync.LSpace(task, modality, operation, 0, basis, c_basis, centroids, diag)
        diagnostics.append({**diag, "seed": 0, "L_dimension": args.latent_dim, "C_dimension": 0})
    return spaces, diagnostics


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = normalize_tasks(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.tasks) != 4:
        raise ValueError("This factorial experiment expects exactly the four T+/T-/I+/I- tasks.")

    print("Readout-orthogonal L factorized path experiment")
    print("Old DAS reference: src/experiments/cross_condition_transfer/procrustes/factorized_paths/factorized_paths.py")
    print("Old algebra: scaled row-vector path composition Q_path = Q_first @ Q_second; alpha_path = alpha_first * alpha_second")
    print("Old implementation has no learned optimizer and no fit-3 mode; fit-3 is added here as a separate square-closure test.")

    if args.synthetic_smoke_test:
        args.tasks = list(TASKS)
        seed_selection = {task: [0] for task in args.tasks}
        sync_seeds = [0]
        value_splits = {"0": {"train_values": list(range(8)), "test_values": list(range(8, 12))}}
        spaces, diagnostics = synthetic_spaces(args)
        common_values = list(range(12))
    else:
        seed_selection = selected_seeds(args)
        sync_seeds = sync.common_sync_seeds(seed_selection)
        value_splits = lsync.load_value_splits(args)
        requested_split_seeds = {str(seed) for seed in args.value_split_seeds}
        value_splits = {seed: split for seed, split in value_splits.items() if seed in requested_split_seeds}
        if not value_splits:
            raise ValueError(f"No requested value split seeds found: {args.value_split_seeds}")
        common_values = lgeom.all_split_values(value_splits)
        data_by_task = lsync.load_rows_and_data(args, args.tasks, seed_selection)
        for task, data in data_by_task.items():
            missing = [value for value in common_values if data.counts.get(value, 0) < 1]
            if missing:
                raise ValueError(f"{task} is missing value-split values: {missing[:10]}")
        spaces, diagnostics = lsync.build_l_spaces(args, args.tasks, seed_selection, data_by_task, common_values)

    save_json(value_splits, args.output_dir / "value_splits.json")
    geom.write_csv(diagnostics, args.output_dir / "L_space_diagnostics.csv")
    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "seed_selection": seed_selection,
                "sync_seeds": sync_seeds,
                "value_splits": value_splits,
                "common_values": common_values,
                "diagnostics_rows": len(diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("ARTIFACT_CHECK_ONLY complete")
        return

    result_rows: list[dict] = []
    map_rows: list[dict] = []
    sync_rows: list[dict] = []
    path_rows: list[dict] = []
    fit3_rows: list[dict] = []
    for split_seed_text, split in value_splits.items():
        value_split_seed = int(split_seed_text)
        train_values = [int(value) for value in split["train_values"]]
        test_values = [int(value) for value in split["test_values"]]
        print(f"\nVALUE_SPLIT seed={value_split_seed} train={len(train_values)} test={len(test_values)}")
        for seed in sync_seeds:
            print(f"  SEED {seed}")
            rows, maps, sync_fit, paths, fit3 = run_for_seed_split(
                args,
                args.tasks,
                spaces,
                seed=seed,
                value_split_seed=value_split_seed,
                train_values=train_values,
                test_values=test_values,
            )
            result_rows.extend(rows)
            map_rows.extend(maps)
            sync_rows.extend(sync_fit)
            path_rows.extend(paths)
            fit3_rows.extend(fit3)
            save_outputs(
                args,
                diagnostics=diagnostics,
                map_rows=map_rows,
                result_rows=result_rows,
                path_rows=path_rows,
                fit3_rows=fit3_rows,
                sync_rows=sync_rows,
                value_splits=value_splits,
                seed_selection=seed_selection,
                sync_seeds=sync_seeds,
            )

    save_outputs(
        args,
        diagnostics=diagnostics,
        map_rows=map_rows,
        result_rows=result_rows,
        path_rows=path_rows,
        fit3_rows=fit3_rows,
        sync_rows=sync_rows,
        value_splits=value_splits,
        seed_selection=seed_selection,
        sync_seeds=sync_seeds,
    )
    if not args.skip_plots:
        plot_figures(args, result_rows, fit3_rows)

    aggregate = summarize(
        [row for row in fit3_rows if row["control_type"] == "none"],
        ["control_type"],
        ["heldout_transition_cosine_mean", "heldout_top1_value_retrieval"],
    )
    print("\nReadout-orthogonal L factorization complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Full geometric rows: {len(result_rows)}")
    print(f"Path-consistency rows: {len(path_rows)}")
    print(f"Fit-3 rows: {len(fit3_rows)}")
    if aggregate:
        row = aggregate[0]
        print(f"Fit-3 ALL transition cosine: {row.get('heldout_transition_cosine_mean_mean')}")
        print(f"Fit-3 ALL top-1 retrieval: {row.get('heldout_top1_value_retrieval_mean')}")


if __name__ == "__main__":
    main()
