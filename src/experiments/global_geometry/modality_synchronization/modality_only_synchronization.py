"""Offline modality-only constrained synchronization in exact 13D L space.

This experiment reuses the exact-L synchronization artifacts and held-out value
splits.  It asks whether one orientation/scale per modality explains most of
the synchronized geometry that the full task-specific model captures.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
from src.experiments.global_geometry.synchronization import synchronization as sync
from src.experiments.global_geometry.synchronization import synchronization_L as lsync


EXPERIMENT = "reevaluating_sync_modality_only_synchronization"
SPACE_TYPE = lsync.SPACE_TYPE
TASKS = list(lsync.TASKS)
TRANSPORTS = [
    "direct_reuse_identity",
    "modality_only_sync",
    "full_task_specific_sync",
]
RELATION_TYPE_BY_RELATION = {
    "T+<->T-": "operation_only",
    "I+<->I-": "operation_only",
    "T+<->I+": "modality_only",
    "T-<->I-": "modality_only",
    "T+<->I-": "modality_plus_operation",
    "T-<->I+": "modality_plus_operation",
}
MODEL_LABELS = {
    "direct_reuse_identity": "direct reuse",
    "modality_only_sync": "modality-only sync",
    "full_task_specific_sync": "full task-specific sync",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--previous_geometry_dir",
        type=Path,
        default=Path("results/experiments/closing/value_heldout_readout_free_geometry"),
        help="Completed value-heldout geometry directory; value_splits.json is reused exactly.",
    )
    parser.add_argument(
        "--sync_L_dir",
        type=Path,
        default=Path("results/paper/synchronization/readout_orthogonal_L"),
        help="Existing exact-L synchronization output directory to reuse for pairwise/full-sync maps.",
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
        default=Path("results/experiments/closing/modality_only_synchronization"),
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
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.seeds == [0, 1, 2]:
        args.seeds = [0]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]


def sync_l_args(args: argparse.Namespace) -> argparse.Namespace:
    sync_args = copy.copy(args)
    sync_args.output_dir = args.sync_L_dir
    sync_args.force = False
    return sync_args


def modality(task: str) -> str:
    return parse_task(task)[0]


def directed_relation_type(source: str, destination: str) -> str:
    return RELATION_TYPE_BY_RELATION[lsync.undirected_key(source, destination)]


def summarize(rows: list[dict], group_fields: list[str], metrics: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, parts in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        item = {field: value for field, value in zip(group_fields, key)}
        item["n"] = len(parts)
        for metric in metrics:
            values = [part.get(metric) for part in parts]
            clean = [float(value) for value in values if value is not None]
            item[f"{metric}_mean"] = lsync.mean(values)
            item[f"{metric}_std"] = lsync.sample_std(values)
            item[f"{metric}_sem"] = lsync.sem(values)
            item[f"{metric}_median"] = float(statistics.median(clean)) if clean else None
        output.append(item)
    return output


def compact_model_rows(by_type_rows: list[dict]) -> list[dict]:
    fields = [
        "heldout_transition_cosine_mean_mean",
        "top1_value_retrieval_mean",
        "top5_value_retrieval_mean",
        "same_value_cosine_mean_mean",
        "heldout_transition_relative_error_mean_mean",
    ]
    rows = []
    for row in by_type_rows:
        rows.append(
            {
                "model": MODEL_LABELS.get(row["transport"], row["transport"]),
                "transport": row["transport"],
                "relation_type": row["relation_type"],
                "n": row["n"],
                "transition_cosine": row.get(fields[0]),
                "top1_retrieval": row.get(fields[1]),
                "top5_retrieval": row.get(fields[2]),
                "same_value_cosine": row.get(fields[3]),
                "relative_error": row.get(fields[4]),
            }
        )
    return rows


def comparison_rows(by_type_rows: list[dict]) -> list[dict]:
    lookup = {(row["transport"], row["relation_type"]): row for row in by_type_rows}
    relation_types = sorted({row["relation_type"] for row in by_type_rows if row["relation_type"] != "ALL"})
    specs = [
        ("full_minus_modality_only", "full_task_specific_sync", "modality_only_sync"),
        ("modality_only_minus_direct_reuse", "modality_only_sync", "direct_reuse_identity"),
    ]
    rows = []
    metrics = [
        ("transition_cosine", "heldout_transition_cosine_mean_mean"),
        ("top1_retrieval", "top1_value_retrieval_mean"),
        ("top5_retrieval", "top5_value_retrieval_mean"),
    ]
    for relation_type in relation_types + ["ALL"]:
        for comparison, left, right in specs:
            left_row = lookup.get((left, relation_type))
            right_row = lookup.get((right, relation_type))
            if left_row is None or right_row is None:
                continue
            item = {
                "comparison": comparison,
                "left_transport": left,
                "right_transport": right,
                "relation_type": relation_type,
                "n_left": left_row["n"],
                "n_right": right_row["n"],
            }
            for label, metric in metrics:
                l_value = left_row.get(metric)
                r_value = right_row.get(metric)
                item[f"{label}_delta"] = None if l_value is None or r_value is None else float(l_value) - float(r_value)
                item[f"{label}_left"] = l_value
                item[f"{label}_right"] = r_value
            rows.append(item)
    return rows


def fit_modality_only(
    tasks: list[str],
    edges: list[sync.Edge],
    k: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict, dict]:
    modality_tasks = ["text", "image"]
    cross_modal_edges = [
        sync.Edge(
            source=modality(edge.source),
            destination=modality(edge.destination),
            q=edge.q,
            alpha=edge.alpha,
            weight=edge.weight,
            path=edge.path,
            fit_metrics=edge.fit_metrics,
        )
        for edge in edges
        if modality(edge.source) != modality(edge.destination)
    ]
    modality_orientations, rotation_stats = sync.synchronize_rotations(modality_tasks, cross_modal_edges, k)
    modality_log_scales, scale_stats = sync.synchronize_scales(modality_tasks, cross_modal_edges)
    orientations = {task: modality_orientations[modality(task)] for task in tasks}
    log_scales = {task: modality_log_scales[modality(task)] for task in tasks}
    rotation_stats = dict(rotation_stats)
    scale_stats = dict(scale_stats)
    rotation_stats["all_edge_residuals_under_constraint"] = sync.rotation_residuals(tasks, edges, orientations)
    scale_stats["all_edge_residuals_under_constraint"] = scale_residuals(edges, log_scales)
    rotation_stats["all_edge_rotation_residual_mean"] = lsync.mean(
        [row["relative_frobenius"] for row in rotation_stats["all_edge_residuals_under_constraint"]]
    )
    scale_stats["all_edge_scale_residual_mean_abs"] = lsync.mean(
        [row["absolute_error"] for row in scale_stats["all_edge_residuals_under_constraint"]]
    )
    return orientations, log_scales, rotation_stats, scale_stats


def scale_residuals(edges: list[sync.Edge], log_scales: dict[str, float]) -> list[dict]:
    rows = []
    for edge in edges:
        predicted = log_scales[edge.destination] - log_scales[edge.source]
        observed = float(torch.log(torch.tensor(max(edge.alpha, 1e-12))).item())
        rows.append(
            {
                "source_task": edge.source,
                "destination_task": edge.destination,
                "observed_log_alpha": observed,
                "predicted_log_alpha": predicted,
                "absolute_error": abs(predicted - observed),
                "weight": edge.weight,
            }
        )
    return rows


def save_modality_payload(
    args: argparse.Namespace,
    tasks: list[str],
    edges: list[sync.Edge],
    orientations: dict[str, torch.Tensor],
    log_scales: dict[str, float],
    rotation_stats: dict,
    scale_stats: dict,
    *,
    seed: int,
    value_split_seed: int,
) -> Path:
    path = (
        args.output_dir
        / "maps"
        / f"L_modality_only_sync_seed{seed}_valuesplit{value_split_seed}_layer{args.layer}_k{args.latent_dim}.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "L_modality_only_constrained_synchronization",
            "experiment": EXPERIMENT,
            "space_type": SPACE_TYPE,
            "seed": seed,
            "value_split_seed": value_split_seed,
            "tasks": tasks,
            "modalities": ["text", "image"],
            "orientations_domain_to_hub": {task: value.cpu() for task, value in orientations.items()},
            "log_scales": log_scales,
            "scales": {task: float(torch.exp(torch.tensor(log_scale)).item()) for task, log_scale in log_scales.items()},
            "edges": [
                {
                    "source_task": edge.source,
                    "destination_task": edge.destination,
                    "alpha": edge.alpha,
                    "weight": edge.weight,
                    "path": str(edge.path),
                    "fit_metrics": edge.fit_metrics,
                }
                for edge in edges
            ],
            "rotation_stats": rotation_stats,
            "scale_stats": scale_stats,
            "config": jsonable(vars(args)),
        },
        path,
    )
    return path


def load_full_sync(args: argparse.Namespace, seed: int, value_split_seed: int) -> tuple[dict[str, torch.Tensor], dict[str, float], Path]:
    sync_args = sync_l_args(args)
    path = lsync.sync_payload_file(sync_args, seed, value_split_seed, "synchronized_full", None)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing existing exact-L full synchronization payload: {path}. "
            "Run src.experiments.global_geometry.synchronization.synchronization_L first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    orientations = {task: torch.as_tensor(value).float() for task, value in payload["orientations_domain_to_hub"].items()}
    log_scales = {task: float(value) for task, value in payload["log_scales"].items()}
    return orientations, log_scales, path


def full_sync_reproduction_check(args: argparse.Namespace, result_rows: list[dict]) -> dict:
    path = args.sync_L_dir / "L_synchronization_geometric_results.csv"
    metrics = ["heldout_transition_cosine_mean", "top1_value_retrieval", "top5_value_retrieval"]
    if not path.exists():
        return {"status": "skipped_missing_existing_results", "path": str(path)}
    existing = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("transport") != "synchronized_full":
                continue
            key = (
                int(row["sync_seed"]),
                int(row["value_split_seed"]),
                row["source_task"],
                row["destination_task"],
            )
            existing[key] = row
    max_abs_delta = 0.0
    checked = 0
    missing = 0
    for row in result_rows:
        if row["transport"] != "full_task_specific_sync":
            continue
        key = (int(row["sync_seed"]), int(row["value_split_seed"]), row["source_task"], row["destination_task"])
        previous = existing.get(key)
        if previous is None:
            missing += 1
            continue
        checked += 1
        for metric in metrics:
            delta = abs(float(row[metric]) - float(previous[metric]))
            max_abs_delta = max(max_abs_delta, delta)
    status = "passed" if missing == 0 and max_abs_delta <= args.sanity_tolerance else "warning"
    return {
        "status": status,
        "path": str(path),
        "checked_rows": checked,
        "missing_rows": missing,
        "max_abs_delta": max_abs_delta,
        "tolerance": args.sanity_tolerance,
    }


def saved_value_splits_check(args: argparse.Namespace, value_splits: dict[str, dict[str, list[int]]]) -> dict:
    path = args.sync_L_dir / "value_splits.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing existing exact-L synchronization value splits: {path}")
    with path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)
    normalized_existing = {
        str(seed): {
            "train_values": [int(value) for value in split["train_values"]],
            "test_values": [int(value) for value in split["test_values"]],
        }
        for seed, split in existing.items()
    }
    normalized_current = {
        str(seed): {
            "train_values": [int(value) for value in split["train_values"]],
            "test_values": [int(value) for value in split["test_values"]],
        }
        for seed, split in value_splits.items()
    }
    missing = sorted(set(normalized_current) - set(normalized_existing))
    mismatched = [
        seed
        for seed, split in normalized_current.items()
        if seed in normalized_existing and normalized_existing[seed] != split
    ]
    if missing or mismatched:
        raise ValueError(
            f"Current selected value splits do not match existing exact-L synchronization splits at {path}: "
            f"missing={missing}, mismatched={mismatched}."
        )
    return {
        "status": "passed",
        "path": str(path),
        "n_existing_splits": len(normalized_existing),
        "n_selected_splits": len(normalized_current),
        "split_seeds": sorted(normalized_current),
    }


def sanity_checks(
    args: argparse.Namespace,
    spaces: dict[str, lsync.LSpace],
    train_values: list[int],
    test_values: list[int],
    orientations: dict[str, torch.Tensor],
    log_scales: dict[str, float],
) -> dict:
    identity = torch.eye(args.latent_dim)
    operation_errors = []
    for source, destination in [("text:addition", "text:subtraction"), ("text:subtraction", "text:addition"), ("image:addition", "image:subtraction"), ("image:subtraction", "image:addition")]:
        q, alpha = sync.hub_map(source, destination, orientations, log_scales)
        operation_errors.append(
            {
                "source_task": source,
                "destination_task": destination,
                "Q_identity_max_abs_error": float((q - identity).abs().max()),
                "scale_unit_abs_error": abs(float(alpha) - 1.0),
            }
        )
    text_orientation_error = float((orientations["text:addition"] - orientations["text:subtraction"]).abs().max())
    image_orientation_error = float((orientations["image:addition"] - orientations["image:subtraction"]).abs().max())
    text_scale_error = abs(log_scales["text:addition"] - log_scales["text:subtraction"])
    image_scale_error = abs(log_scales["image:addition"] - log_scales["image:subtraction"])
    l_dims = {task: int(space.basis.shape[1]) for task, space in spaces.items()}
    lsync.assert_value_split(train_values, test_values, "modality-only sanity")
    failures = []
    if any(dim != args.latent_dim for dim in l_dims.values()):
        failures.append("L_dimension")
    if max(row["Q_identity_max_abs_error"] for row in operation_errors) > args.sanity_tolerance:
        failures.append("operation_identity_Q")
    if max(row["scale_unit_abs_error"] for row in operation_errors) > args.sanity_tolerance:
        failures.append("operation_unit_scale")
    if max(text_orientation_error, image_orientation_error, text_scale_error, image_scale_error) > 0.0:
        failures.append("tied_parameter_equality")
    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "L_dimensions": l_dims,
        "expected_L_dimension": args.latent_dim,
        "train_values": train_values,
        "test_values": test_values,
        "train_test_overlap": sorted(set(train_values) & set(test_values)),
        "tied_parameter_max_abs_errors": {
            "text_orientation": text_orientation_error,
            "image_orientation": image_orientation_error,
            "text_log_scale": text_scale_error,
            "image_log_scale": image_scale_error,
        },
        "operation_only_identity_checks": operation_errors,
    }


def modality_map_similarity(
    args: argparse.Namespace,
    spaces: dict[str, lsync.LSpace],
    edge_by_pair: dict[tuple[str, str], sync.Edge],
    *,
    seed: int,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
) -> list[dict]:
    specs = [
        ("text_to_image", ("text:addition", "image:addition"), ("text:subtraction", "image:subtraction")),
        ("image_to_text", ("image:addition", "text:addition"), ("image:subtraction", "text:subtraction")),
    ]
    rows = []
    for direction, add_pair, sub_pair in specs:
        add_edge = edge_by_pair[add_pair]
        sub_edge = edge_by_pair[sub_pair]
        q_delta = float((add_edge.q - sub_edge.q).norm() / add_edge.q.norm().clamp_min(1e-12))
        trace_similarity = float(torch.trace(add_edge.q.T @ sub_edge.q) / add_edge.q.shape[0])
        scale_log_difference = abs(math.log(max(add_edge.alpha, 1e-12)) - math.log(max(sub_edge.alpha, 1e-12)))
        item = {
            "direction": direction,
            "seed": seed,
            "value_split_seed": value_split_seed,
            "addition_relation": lsync.directed_key(*add_pair),
            "subtraction_relation": lsync.directed_key(*sub_pair),
            "rotation_normalized_frobenius_disagreement": q_delta,
            "rotation_trace_similarity": trace_similarity,
            "scale_addition": add_edge.alpha,
            "scale_subtraction": sub_edge.alpha,
            "scale_log_difference": scale_log_difference,
            "addition_self_heldout_transition_cosine": add_edge.fit_metrics.get("heldout_transition_cosine_mean"),
            "subtraction_self_heldout_transition_cosine": sub_edge.fit_metrics.get("heldout_transition_cosine_mean"),
        }
        for source_pair, q_name, q, alpha in [
            (sub_pair, "addition_map_on_subtraction", add_edge.q, add_edge.alpha),
            (add_pair, "subtraction_map_on_addition", sub_edge.q, sub_edge.alpha),
        ]:
            source, destination = source_pair
            _train_transitions, x_train = geom.transition_matrix(spaces[source].centroids, train_values)
            _train_transitions_b, y_train = geom.transition_matrix(spaces[destination].centroids, train_values)
            _test_transitions, x_test = geom.transition_matrix(spaces[source].centroids, test_values)
            _test_transitions_b, y_test = geom.transition_matrix(spaces[destination].centroids, test_values)
            item.update(lsync.metric_set(x_train, y_train, q, alpha, f"{q_name}_train_transition"))
            item.update(lsync.metric_set(x_test, y_test, q, alpha, f"{q_name}_heldout_transition"))
        rows.append(item)
    return rows


def run_for_seed_split(
    args: argparse.Namespace,
    tasks: list[str],
    spaces_by_key: dict[tuple[str, int], lsync.LSpace],
    *,
    seed: int,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    lsync.assert_value_split(train_values, test_values, f"seed={seed} value_split={value_split_seed}")
    spaces = {task: spaces_by_key[(task, seed)] for task in tasks}
    edge_args = sync_l_args(args)
    edge_by_pair = {}
    map_rows = []
    for source_task, destination_task in lsync.directed_pairs(tasks):
        edge_path = lsync.map_file(edge_args, spaces[source_task], spaces[destination_task], value_split_seed)
        if not edge_path.exists():
            raise FileNotFoundError(
                f"Missing existing exact-L pairwise map {edge_path}. "
                "Run src.experiments.global_geometry.synchronization.synchronization_L first."
            )
        edge = lsync.fit_or_load_edge(
            edge_args,
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
                "relation_type": directed_relation_type(source_task, destination_task),
                "seed": seed,
                "value_split_seed": value_split_seed,
                "path": str(edge.path),
                "alpha": edge.alpha,
                "weight": edge.weight,
                **edge.fit_metrics,
            }
        )

    modality_orientations, modality_log_scales, rotation_stats, scale_stats = fit_modality_only(
        tasks,
        list(edge_by_pair.values()),
        args.latent_dim,
    )
    modality_path = save_modality_payload(
        args,
        tasks,
        list(edge_by_pair.values()),
        modality_orientations,
        modality_log_scales,
        rotation_stats,
        scale_stats,
        seed=seed,
        value_split_seed=value_split_seed,
    )
    full_orientations, full_log_scales, full_path = load_full_sync(args, seed, value_split_seed)
    identity = torch.eye(args.latent_dim)
    sync_rows = [
        {
            "condition": "modality_only_sync",
            "seed": seed,
            "value_split_seed": value_split_seed,
            "n_cross_modal_fit_edges": sum(1 for edge in edge_by_pair.values() if modality(edge.source) != modality(edge.destination)),
            "n_total_scored_edges": len(edge_by_pair),
            "synchronization_path": str(modality_path),
            "rotation_residual_mean_cross_modal_fit": rotation_stats["rotation_residual_mean"],
            "rotation_residual_mean_all_edges": rotation_stats["all_edge_rotation_residual_mean"],
            "scale_residual_mean_abs_cross_modal_fit": scale_stats["scale_residual_mean_abs"],
            "scale_residual_mean_abs_all_edges": scale_stats["all_edge_scale_residual_mean_abs"],
        }
    ]

    result_rows = []
    for source_task, destination_task in lsync.directed_pairs(tasks):
        source = spaces[source_task]
        destination = spaces[destination_task]
        direct = edge_by_pair[(source_task, destination_task)]
        specs = [
            ("direct_reuse_identity", identity, 1.0, None, None, None),
            (
                "modality_only_sync",
                *sync.hub_map(source_task, destination_task, modality_orientations, modality_log_scales),
                modality_path,
                modality_orientations,
                modality_log_scales,
            ),
            (
                "full_task_specific_sync",
                *sync.hub_map(source_task, destination_task, full_orientations, full_log_scales),
                full_path,
                full_orientations,
                full_log_scales,
            ),
        ]
        for transport, q, alpha, path, orientations, log_scales in specs:
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
                direct_edge=direct,
                synchronization_path=path,
                heldout_relation=None,
                orientations=orientations,
                log_scales=log_scales,
            )
            row.update(
                {
                    "experiment": EXPERIMENT,
                    "relation": lsync.undirected_key(source_task, destination_task),
                    "relation_type": directed_relation_type(source_task, destination_task),
                }
            )
            result_rows.append(row)
    checks = [
        {
            "seed": seed,
            "value_split_seed": value_split_seed,
            **sanity_checks(args, spaces, train_values, test_values, modality_orientations, modality_log_scales),
        }
    ]
    params = [
        {
            "seed": seed,
            "value_split_seed": value_split_seed,
            "modality": mod,
            "log_scale": modality_log_scales[next(task for task in tasks if modality(task) == mod)],
            "scale": float(torch.exp(torch.tensor(modality_log_scales[next(task for task in tasks if modality(task) == mod)])).item()),
            "orientation_domain_to_hub": modality_orientations[next(task for task in tasks if modality(task) == mod)].tolist(),
        }
        for mod in ["text", "image"]
    ]
    diagnostic_rows = modality_map_similarity(
        args,
        spaces,
        edge_by_pair,
        seed=seed,
        value_split_seed=value_split_seed,
        train_values=train_values,
        test_values=test_values,
    )
    return result_rows, map_rows, sync_rows, checks, params, diagnostic_rows


def plot_main(args: argparse.Namespace, by_type_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    relation_types = ["operation_only", "modality_only", "modality_plus_operation"]
    labels = ["operation only", "modality only", "modality + operation"]
    lookup = {(row["transport"], row["relation_type"]): row for row in by_type_rows}
    colors = {
        "direct_reuse_identity": "#5f6c72",
        "modality_only_sync": "#27806b",
        "full_task_specific_sync": "#b65b3a",
    }
    x = np.arange(len(relation_types))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for offset, transport in zip([-width, 0, width], TRANSPORTS):
        values = [
            np.nan
            if (transport, relation_type) not in lookup
            else lookup[(transport, relation_type)].get("heldout_transition_cosine_mean_mean")
            for relation_type in relation_types
        ]
        ax.bar(x + offset, values, width, label=MODEL_LABELS[transport], color=colors[transport])
    ax.set_xticks(x, labels)
    ax.set_ylabel("held-out transition cosine")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_dir / "modality_only_synchronization_transition_cosine.png", dpi=240)
    fig.savefig(figure_dir / "modality_only_synchronization_transition_cosine.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = lsync.normalize_tasks(args.tasks)
    if args.latent_dim != 13:
        raise ValueError(f"This reevaluation is defined for exact-L dim=13, got --latent_dim={args.latent_dim}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_selection = selected_seeds(args)
    sync_seeds = sync.common_sync_seeds(seed_selection)
    value_splits = lsync.load_value_splits(args)
    common_values = lgeom.all_split_values(value_splits)

    value_split_check = saved_value_splits_check(args, value_splits)

    print("Modality-only constrained exact-L synchronization")
    print(f"  tasks={args.tasks}")
    print(f"  sync_seeds={sync_seeds}")
    print(f"  value_split_seeds={list(value_splits)}")
    print(f"  existing_sync_L_dir={args.sync_L_dir}")
    print("  model forwards=0")

    data_by_task = lsync.load_rows_and_data(args, args.tasks, seed_selection)
    spaces_by_key, diagnostics = lsync.build_l_spaces(args, args.tasks, seed_selection, data_by_task, common_values)

    result_rows = []
    map_rows = []
    sync_rows = []
    check_rows = []
    parameter_rows = []
    diagnostic_rows = []
    for split_seed_text, split in value_splits.items():
        value_split_seed = int(split_seed_text)
        train_values = split["train_values"]
        test_values = split["test_values"]
        print(f"\nVALUE_SPLIT seed={value_split_seed} train={len(train_values)} test={len(test_values)}")
        for seed in sync_seeds:
            print(f"  SYNC_SEED {seed}")
            rows, maps, sync_fit, checks, params, diagnostics_for_split = run_for_seed_split(
                args,
                args.tasks,
                spaces_by_key,
                seed=seed,
                value_split_seed=value_split_seed,
                train_values=train_values,
                test_values=test_values,
            )
            result_rows.extend(rows)
            map_rows.extend(maps)
            sync_rows.extend(sync_fit)
            check_rows.extend(checks)
            parameter_rows.extend(params)
            diagnostic_rows.extend(diagnostics_for_split)
            save_jsonl(result_rows, args.output_dir / "raw_results.jsonl")
            geom.write_csv(result_rows, args.output_dir / "raw_results.csv")

    metrics = [
        "heldout_transition_cosine_mean",
        "heldout_transition_relative_error_mean",
        "top1_value_retrieval",
        "top5_value_retrieval",
        "same_value_cosine_mean",
        "common_frame_same_value_cosine_mean",
        "Q_relative_frobenius_error",
        "scale_relative_error",
        "operator_relative_distance_to_direct",
    ]
    per_relation = summarize(result_rows, ["transport", "task_relation", "relation_type"], metrics)
    by_relation_type = summarize(result_rows, ["transport", "relation_type"], metrics)
    by_model_all = summarize(result_rows, ["transport"], metrics)
    for row in by_model_all:
        row["relation_type"] = "ALL"
    by_relation_type_all = by_relation_type + by_model_all
    model_comparison = comparison_rows(by_relation_type_all)
    compact_models = compact_model_rows(by_relation_type_all)
    reproduction = full_sync_reproduction_check(args, result_rows)

    geom.write_csv(per_relation, args.output_dir / "per_relation.csv")
    geom.write_csv(by_relation_type_all, args.output_dir / "by_relation_type.csv")
    geom.write_csv(model_comparison, args.output_dir / "model_comparison.csv")
    geom.write_csv(compact_models, args.output_dir / "model_scores_compact.csv")
    geom.write_csv(sync_rows, args.output_dir / "modality_sync_fit_summary.csv")
    geom.write_csv(map_rows, args.output_dir / "reused_pairwise_map_summary.csv")
    geom.write_csv(check_rows, args.output_dir / "sanity_checks.csv")
    geom.write_csv(diagnostics, args.output_dir / "L_space_diagnostics.csv")
    geom.write_csv(diagnostic_rows, args.output_dir / "modality_map_similarity.csv")
    save_json(
        {
            "experiment": EXPERIMENT,
            "description": (
                "Offline exact-L comparison of direct identity reuse, modality-only constrained "
                "synchronization, and existing full task-specific synchronization."
            ),
            "space_type": SPACE_TYPE,
            "tasks": args.tasks,
            "task_labels": geom.TASK_LABELS,
            "L_dimension": args.latent_dim,
            "seed_selection": seed_selection,
            "sync_seeds": sync_seeds,
            "value_splits": value_splits,
            "existing_value_splits_check": value_split_check,
            "common_values": common_values,
            "transports": TRANSPORTS,
            "relation_type_definitions": RELATION_TYPE_BY_RELATION,
            "n_result_rows": len(result_rows),
            "per_relation": per_relation,
            "by_relation_type": by_relation_type_all,
            "primary_comparison": model_comparison,
            "full_sync_reproduction_check": reproduction,
            "sanity_checks": check_rows,
            "modality_map_similarity_rows": len(diagnostic_rows),
            "controls_and_leakage_checks": {
                "offline_only": "passed: no model forward paths are invoked",
                "L_dimension_assertion": f"all spaces exactly {args.latent_dim}D",
                "value_split_protocol": "reused exact value_splits from previous_geometry_dir; existing sync_L value_splits presence recorded",
                "heldout_value_leakage": "pairwise maps are loaded only when their cached train/test values match this split",
                "full_sync_reuse": "existing exact-L full synchronization payloads are loaded from sync_L_dir",
                "modality_tying": "same orientation tensor and log-scale scalar assigned to both operations within each modality",
                "operation_only_prediction": "identity/unit scale under modality-only model, checked numerically",
            },
            "config": jsonable(vars(args)),
        },
        args.output_dir / "summary.json",
    )
    save_json(
        {
            "experiment": EXPERIMENT,
            "space_type": SPACE_TYPE,
            "parameters": parameter_rows,
            "note": "Matrices are modality-to-hub orientations; task-level parameters are tied exactly by modality.",
        },
        args.output_dir / "fitted_modality_parameters.json",
    )
    if not args.skip_plots:
        plot_main(args, by_relation_type_all)

    if any(row["status"] != "passed" for row in check_rows):
        raise RuntimeError("At least one modality-only sanity check failed; inspect sanity_checks.csv.")
    if reproduction["status"] == "warning":
        raise RuntimeError("Full-sync reproduction check exceeded tolerance; inspect summary.json.")

    print("\nModality-only synchronization reevaluation complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Summary: {args.output_dir / 'summary.json'}")
    for row in by_model_all:
        print(
            f"  ALL {row['transport']}: transition_cosine="
            f"{row.get('heldout_transition_cosine_mean_mean')} top1="
            f"{row.get('top1_value_retrieval_mean')} top5="
            f"{row.get('top5_value_retrieval_mean')}"
        )


if __name__ == "__main__":
    main()
