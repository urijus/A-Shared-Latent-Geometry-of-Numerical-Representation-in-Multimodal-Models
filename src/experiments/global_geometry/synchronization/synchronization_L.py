"""Synchronize the exact 13D readout-orthogonal L components.

This is the geometric analogue of the DAS synchronization experiment.  It
reuses the existing spectral rotation synchronization and log-scale
synchronization code, but fits the pairwise maps in the exact L coordinates
used by the closing readout-removal experiments.

No model forwards are run here.  The script loads saved activations, rebuilds
the exact C/L split from each K=22 DAS basis and the saved digit-readout span,
fits train-value transition maps in L, and evaluates full-graph and
leave-one-relation-out synchronization on held-out numerical values.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from src.experiments.global_geometry import causal_subspace_geometry as lgeom
from src.experiments.readout_latent_geometry import value_heldout_geometry as geom
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    first_result_row,
    jsonable,
    label,
    parse_task,
    save_json,
    save_jsonl,
    selected_seeds,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import clean_name, stable_seed
from src.experiments.global_geometry.synchronization import synchronization as sync


EXPERIMENT = "readout_orthogonal_L_synchronization"
SPACE_TYPE = "readout_orthogonal_L"
TASKS = [
    "text:addition",
    "text:subtraction",
    "image:addition",
    "image:subtraction",
]
UNDIRECTED_RELATIONS = [
    ("text:addition", "text:subtraction"),
    ("text:addition", "image:addition"),
    ("text:addition", "image:subtraction"),
    ("text:subtraction", "image:addition"),
    ("text:subtraction", "image:subtraction"),
    ("image:addition", "image:subtraction"),
]
TRANSPORTS = ["pairwise_direct", "synchronized_full", "synchronized_loro"]


@dataclass(frozen=True)
class LSpace:
    task: str
    modality: str
    operation: str
    seed: int
    basis: torch.Tensor
    c_basis: torch.Tensor
    centroids: dict[int, torch.Tensor]
    diagnostics: dict


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
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper/synchronization/readout_orthogonal_L"))
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
    parser.add_argument("--rsa_permutations", type=int, default=1000)
    parser.add_argument("--sanity_tolerance", type=float, default=1e-5)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_tasks(tasks: list[str]) -> list[str]:
    if any(task.lower() == "all" for task in tasks):
        return list(TASKS)
    return [task_key(*parse_task(task)) for task in tasks]


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.seeds == [0, 1, 2]:
        args.seeds = [0]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]
    args.rsa_permutations = min(args.rsa_permutations, 100)


def task_label(task: str) -> str:
    return geom.TASK_LABELS.get(task, label(task))


def directed_pairs(tasks: list[str]) -> list[tuple[str, str]]:
    return [(source, destination) for source in tasks for destination in tasks if source != destination]


def unique_relations(tasks: list[str]) -> list[tuple[str, str]]:
    task_set = set(tasks)
    preferred = [pair for pair in UNDIRECTED_RELATIONS if pair[0] in task_set and pair[1] in task_set]
    extras = [
        (a, b)
        for index, a in enumerate(tasks)
        for b in tasks[index + 1 :]
        if (a, b) not in preferred and (b, a) not in preferred
    ]
    return preferred + extras


def relation_key(source: str, destination: str) -> str:
    return f"{task_label(source)}<->{task_label(destination)}"


def directed_key(source: str, destination: str) -> str:
    return f"{task_label(source)}->{task_label(destination)}"


def undirected_key(source: str, destination: str) -> str:
    order = {task: index for index, task in enumerate(TASKS)}
    if order.get(destination, 999) < order.get(source, 999):
        source, destination = destination, source
    return relation_key(source, destination)


def mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(sum(clean) / len(clean))


def sample_std(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.stdev(clean)) if len(clean) > 1 else 0.0


def sem(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sample_std(clean) / math.sqrt(len(clean)) if len(clean) > 1 else 0.0


def assert_value_split(train_values: list[int], test_values: list[int], context: str) -> None:
    overlap = set(train_values) & set(test_values)
    if overlap:
        raise ValueError(f"{context} has train/test value leakage: {sorted(overlap)[:10]}")


def metric_set(x: torch.Tensor, y: torch.Tensor, q: torch.Tensor, alpha: float, prefix: str) -> dict:
    if x.numel() == 0 or y.numel() == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_cosine_mean": None,
            f"{prefix}_cosine_median": None,
            f"{prefix}_cosine_std": None,
            f"{prefix}_cosine_sem": None,
            f"{prefix}_relative_error_mean": None,
            f"{prefix}_relative_error_median": None,
            f"{prefix}_relative_error_std": None,
            f"{prefix}_relative_error_sem": None,
        }
    predicted = float(alpha) * (x @ q)
    cosine = torch.nn.functional.cosine_similarity(predicted, y, dim=1)
    relative_error = (predicted - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-12)
    cosine_values = [float(value) for value in cosine.tolist()]
    error_values = [float(value) for value in relative_error.tolist()]
    return {
        f"{prefix}_n": int(x.shape[0]),
        f"{prefix}_cosine_mean": mean(cosine_values),
        f"{prefix}_cosine_median": float(statistics.median(cosine_values)),
        f"{prefix}_cosine_std": sample_std(cosine_values),
        f"{prefix}_cosine_sem": sem(cosine_values),
        f"{prefix}_relative_error_mean": mean(error_values),
        f"{prefix}_relative_error_median": float(statistics.median(error_values)),
        f"{prefix}_relative_error_std": sample_std(error_values),
        f"{prefix}_relative_error_sem": sem(error_values),
    }


def fit_stats_for_values(
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
    train_values: list[int],
    test_values: list[int],
) -> tuple[torch.Tensor, float, dict]:
    train_transitions, x_train = geom.transition_matrix(source_centroids, train_values)
    train_transitions_b, y_train = geom.transition_matrix(destination_centroids, train_values)
    test_transitions, x_test = geom.transition_matrix(source_centroids, test_values)
    test_transitions_b, y_test = geom.transition_matrix(destination_centroids, test_values)
    if train_transitions != train_transitions_b:
        raise RuntimeError("Source and destination train transitions are not in the same order.")
    if test_transitions != test_transitions_b:
        raise RuntimeError("Source and destination test transitions are not in the same order.")
    q, alpha = geom.fit_scaled_procrustes(x_train, y_train)
    train_predicted = float(alpha) * (x_train @ q)
    train_residual = train_predicted - y_train
    train_cosine = torch.nn.functional.cosine_similarity(train_predicted, y_train, dim=1)
    stats = {
        "fit_mode": "value_heldout_transition_mean",
        "train_values": train_values,
        "test_values": test_values,
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "n_train_transitions": len(train_transitions),
        "n_test_transitions": len(test_transitions),
        "mean_cosine": float(train_cosine.mean()),
        "root_mean_squared_error": float(train_residual.square().mean().sqrt()),
        **metric_set(x_train, y_train, q, alpha, "train_transition"),
        **metric_set(x_test, y_test, q, alpha, "heldout_transition"),
    }
    return q, alpha, stats


def map_file(args: argparse.Namespace, source: LSpace, destination: LSpace, value_split_seed: int) -> Path:
    return (
        args.output_dir
        / "maps"
        / (
            f"L_transition_mean_scaled_map_{clean_name(source.task, source.seed)}"
            f"_to_{clean_name(destination.task, destination.seed)}"
            f"_valuesplit{value_split_seed}_layer{args.layer}_k{args.latent_dim}.pt"
        )
    )


def fit_or_load_edge(
    args: argparse.Namespace,
    source: LSpace,
    destination: LSpace,
    *,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
) -> sync.Edge:
    path = map_file(args, source, destination, value_split_seed)
    if path.exists() and not args.force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload.get("value_split_seed", -1)) != int(value_split_seed):
            raise ValueError(f"{path} value_split_seed does not match requested split {value_split_seed}.")
        if payload.get("source_task") != source.task or payload.get("destination_task") != destination.task:
            raise ValueError(f"{path} source/destination metadata does not match the requested edge.")
        if int(payload.get("source_seed", -1)) != int(source.seed) or int(payload.get("destination_seed", -1)) != int(destination.seed):
            raise ValueError(f"{path} source/destination seed metadata does not match the requested edge.")
        if int(payload.get("k", -1)) != int(args.latent_dim):
            raise ValueError(f"{path} was not fit in the requested L dimension {args.latent_dim}.")
        cached_metrics = payload.get("fit_metrics", {})
        if cached_metrics.get("train_values") != train_values or cached_metrics.get("test_values") != test_values:
            raise ValueError(
                f"{path} was fit with different train/test values; pass --force to regenerate."
            )
        q = torch.as_tensor(payload["Q_source_to_destination"]).float()
        alpha = float(payload["alpha"])
        fit_metrics = cached_metrics
    else:
        q, alpha, fit_metrics = fit_stats_for_values(source.centroids, destination.centroids, train_values, test_values)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "L_scaled_displacement_transition_mean_map",
                "experiment": EXPERIMENT,
                "space_type": SPACE_TYPE,
                "source_task": source.task,
                "source_seed": source.seed,
                "destination_task": destination.task,
                "destination_seed": destination.seed,
                "value_split_seed": value_split_seed,
                "Q_source_to_destination": q.cpu(),
                "alpha": alpha,
                "fit_metrics": fit_metrics,
                "layer": args.layer,
                "k_original_das": args.k,
                "k": args.latent_dim,
                "target": args.target,
                "config": jsonable(vars(args)),
            },
            path,
        )
    if q.shape != (args.latent_dim, args.latent_dim):
        raise ValueError(f"{path} has Q shape {tuple(q.shape)}, expected {(args.latent_dim, args.latent_dim)}.")
    return sync.Edge(
        source=source.task,
        destination=destination.task,
        q=q,
        alpha=alpha,
        weight=sync.edge_weight(args, fit_metrics),
        path=path,
        fit_metrics=fit_metrics,
    )


def load_value_splits(args: argparse.Namespace) -> dict[str, dict[str, list[int]]]:
    splits = lgeom.load_exact_value_splits(args)
    for seed_text, split in splits.items():
        train_values = [int(value) for value in split["train_values"]]
        test_values = [int(value) for value in split["test_values"]]
        assert_value_split(train_values, test_values, f"value split {seed_text}")
        split["train_values"] = train_values
        split["test_values"] = test_values
    return splits


def load_rows_and_data(args: argparse.Namespace, tasks: list[str], seed_selection: dict[str, list[int]]) -> dict[str, geom.TaskData]:
    data_by_task = {}
    for task in tasks:
        modality, operation = parse_task(task)
        row = first_result_row(args, modality, operation, args.condition, seed_selection[task][0])
        geom.validate_result_metadata(
            args,
            row,
            task=task,
            path=subspace_path(args, modality, operation, seed_selection[task][0]).with_name("results.jsonl"),
        )
        data_by_task[task] = geom.load_task_data(args, task, row)
    return data_by_task


def build_l_spaces(
    args: argparse.Namespace,
    tasks: list[str],
    seed_selection: dict[str, list[int]],
    data_by_task: dict[str, geom.TaskData],
    common_values: list[int],
) -> tuple[dict[tuple[str, int], LSpace], list[dict]]:
    spaces = {}
    diagnostics = []
    common_set = set(common_values)
    for task in tasks:
        data = data_by_task[task]
        digit_basis = geom.load_digit_basis(args.digit_readout_basis_path, data.hidden.shape[1])
        if digit_basis.shape[1] != args.m_readout:
            raise ValueError(
                f"{args.digit_readout_basis_path} rank is {digit_basis.shape[1]}, "
                f"expected m_readout={args.m_readout}."
            )
        for seed in seed_selection[task]:
            c_basis, l_basis, _ordered, diag = lgeom.construct_exact_CL(
                args,
                task=task,
                data=data,
                digit_basis=digit_basis,
                seed=seed,
            )
            if l_basis.shape[1] != args.latent_dim:
                raise ValueError(f"{task} seed={seed} L dimension {l_basis.shape[1]} != {args.latent_dim}.")
            if c_basis.shape[1] + l_basis.shape[1] != args.k:
                raise ValueError(
                    f"{task} seed={seed} C+L dimension {c_basis.shape[1]}+{l_basis.shape[1]} != k={args.k}."
                )
            centroids = geom.centroid_coordinates(data, l_basis, common_set, args.target)
            missing = sorted(common_set - set(centroids))
            if missing:
                raise ValueError(f"{task} seed={seed} missing L centroids for values {missing[:10]}.")
            item = LSpace(
                task=task,
                modality=data.modality,
                operation=data.operation,
                seed=seed,
                basis=l_basis,
                c_basis=c_basis,
                centroids=centroids,
                diagnostics=diag,
            )
            spaces[(task, seed)] = item
            diagnostics.append(
                {
                    **diag,
                    "experiment": EXPERIMENT,
                    "space_type": SPACE_TYPE,
                    "task": task,
                    "task_label": task_label(task),
                    "seed": seed,
                    "L_dimension": int(l_basis.shape[1]),
                    "C_dimension": int(c_basis.shape[1]),
                    "basis_source": "K=22 DAS subspace plus digit_readout_basis via build_ordered_split",
                }
            )
            print(
                f"L_SPACE task={task} seed={seed} L_dim={l_basis.shape[1]} "
                f"C_dim={c_basis.shape[1]} digit_overlap={diag['digit_overlap']:.3e} "
                f"C_L_overlap={diag['C_L_overlap']:.3e}"
            )
    return spaces, diagnostics


def sync_payload_file(args: argparse.Namespace, seed: int, value_split_seed: int, condition: str, heldout_relation: tuple[str, str] | None) -> Path:
    if heldout_relation is None:
        suffix = "full_graph"
    else:
        suffix = (
            "heldout_"
            + clean_name(heldout_relation[0], seed).replace(f"_seed{seed}", "")
            + "__"
            + clean_name(heldout_relation[1], seed).replace(f"_seed{seed}", "")
        )
    return (
        args.output_dir
        / "maps"
        / f"L_synchronized_hub_seed{seed}_valuesplit{value_split_seed}_{condition}_{suffix}_layer{args.layer}_k{args.latent_dim}.pt"
    )


def synchronize_and_save(
    args: argparse.Namespace,
    tasks: list[str],
    edges: list[sync.Edge],
    *,
    seed: int,
    value_split_seed: int,
    condition: str,
    heldout_relation: tuple[str, str] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict, dict, Path]:
    orientations, rotation_stats = sync.synchronize_rotations(tasks, edges, args.latent_dim)
    log_scales, scale_stats = sync.synchronize_scales(tasks, edges)
    path = sync_payload_file(args, seed, value_split_seed, condition, heldout_relation)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "L_orthogonal_synchronization_hub",
            "experiment": EXPERIMENT,
            "space_type": SPACE_TYPE,
            "seed": seed,
            "value_split_seed": value_split_seed,
            "condition": condition,
            "tasks": tasks,
            "heldout_relation": None if heldout_relation is None else list(heldout_relation),
            "orientations_domain_to_hub": {task: value.cpu() for task, value in orientations.items()},
            "log_scales": log_scales,
            "scales": {task: float(torch.exp(torch.tensor(scale)).item()) for task, scale in log_scales.items()},
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
    return orientations, log_scales, rotation_stats, scale_stats, path


def assert_loro_edges_absent(edges: list[sync.Edge], heldout_relation: tuple[str, str]) -> None:
    banned = {heldout_relation, (heldout_relation[1], heldout_relation[0])}
    leaked = [(edge.source, edge.destination) for edge in edges if (edge.source, edge.destination) in banned]
    if leaked:
        raise RuntimeError(f"Leave-one-relation-out leakage: held-out directed edge(s) present: {leaked}")


def map_prediction_metrics(q: torch.Tensor, alpha: float, direct_q: torch.Tensor, direct_alpha: float) -> dict:
    q_error = float((q - direct_q).norm() / direct_q.norm().clamp_min(1e-12))
    q_similarity = float(torch.trace(q.T @ direct_q) / q.shape[0])
    alpha_hat = max(float(alpha), 1e-12)
    alpha_direct = max(float(direct_alpha), 1e-12)
    return {
        "Q_relative_frobenius_error": q_error,
        "Q_trace_similarity": q_similarity,
        "scale_predicted": float(alpha),
        "scale_direct": float(direct_alpha),
        "scale_relative_error": abs(float(alpha) - float(direct_alpha)) / max(abs(float(direct_alpha)), 1e-12),
        "scale_log_error": abs(math.log(alpha_hat) - math.log(alpha_direct)),
        "operator_relative_distance_to_direct": sync.relative_operator_distance(q, alpha, direct_q, direct_alpha),
    }


def common_frame_cosines(
    source: LSpace,
    destination: LSpace,
    values: list[int],
    train_values: list[int],
    orientations: dict[str, torch.Tensor],
    log_scales: dict[str, float],
) -> dict:
    if not values:
        return {
            "common_frame_same_value_cosine_mean": None,
            "common_frame_same_value_cosine_median": None,
            "common_frame_same_value_cosine_std": None,
            "common_frame_same_value_cosine_sem": None,
            "common_frame_different_value_cosine_mean": None,
            "common_frame_different_value_cosine_median": None,
            "common_frame_different_value_cosine_std": None,
            "common_frame_different_value_cosine_sem": None,
        }
    source_center = torch.stack([source.centroids[value] for value in train_values]).mean(dim=0)
    destination_center = torch.stack([destination.centroids[value] for value in train_values]).mean(dim=0)
    source_scale = float(torch.exp(torch.tensor(log_scales[source.task])).item())
    destination_scale = float(torch.exp(torch.tensor(log_scales[destination.task])).item())
    source_shared = {
        value: ((source.centroids[value] - source_center) @ orientations[source.task]) / max(source_scale, 1e-12)
        for value in values
    }
    destination_shared = {
        value: ((destination.centroids[value] - destination_center) @ orientations[destination.task]) / max(destination_scale, 1e-12)
        for value in values
    }
    same = [
        float(torch.nn.functional.cosine_similarity(source_shared[value][None], destination_shared[value][None]))
        for value in values
    ]
    different = []
    for source_value in values:
        for destination_value in values:
            if source_value != destination_value:
                different.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            source_shared[source_value][None],
                            destination_shared[destination_value][None],
                        )
                    )
                )
    return {
        "common_frame_same_value_cosine_mean": mean(same),
        "common_frame_same_value_cosine_median": float(statistics.median(same)),
        "common_frame_same_value_cosine_std": sample_std(same),
        "common_frame_same_value_cosine_sem": sem(same),
        "common_frame_different_value_cosine_mean": mean(different),
        "common_frame_different_value_cosine_median": float(statistics.median(different)) if different else None,
        "common_frame_different_value_cosine_std": sample_std(different),
        "common_frame_different_value_cosine_sem": sem(different),
    }


def evaluate_transport(
    args: argparse.Namespace,
    *,
    source: LSpace,
    destination: LSpace,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
    transport: str,
    q: torch.Tensor,
    alpha: float,
    direct_edge: sync.Edge,
    synchronization_path: Path | None,
    heldout_relation: tuple[str, str] | None,
    orientations: dict[str, torch.Tensor] | None = None,
    log_scales: dict[str, float] | None = None,
) -> dict:
    train_transitions, x_train = geom.transition_matrix(source.centroids, train_values)
    train_transitions_b, y_train = geom.transition_matrix(destination.centroids, train_values)
    test_transitions, x_test = geom.transition_matrix(source.centroids, test_values)
    test_transitions_b, y_test = geom.transition_matrix(destination.centroids, test_values)
    if train_transitions != train_transitions_b or test_transitions != test_transitions_b:
        raise RuntimeError("Transition ordering mismatch during evaluation.")
    retrieval = geom.retrieval_metrics(source.centroids, destination.centroids, train_values, test_values, q, alpha)
    common = {}
    if orientations is not None and log_scales is not None:
        common = common_frame_cosines(source, destination, test_values, train_values, orientations, log_scales)
    else:
        common = {
            "common_frame_same_value_cosine_mean": None,
            "common_frame_same_value_cosine_median": None,
            "common_frame_same_value_cosine_std": None,
            "common_frame_same_value_cosine_sem": None,
            "common_frame_different_value_cosine_mean": None,
            "common_frame_different_value_cosine_median": None,
            "common_frame_different_value_cosine_std": None,
            "common_frame_different_value_cosine_sem": None,
        }
    return {
        "experiment": EXPERIMENT,
        "space_type": SPACE_TYPE,
        "transport": transport,
        "source_task": source.task,
        "source_label": task_label(source.task),
        "destination_task": destination.task,
        "destination_label": task_label(destination.task),
        "task_relation": directed_key(source.task, destination.task),
        "source_seed": source.seed,
        "destination_seed": destination.seed,
        "sync_seed": source.seed,
        "value_split_seed": value_split_seed,
        "relation_holdout": heldout_relation is not None,
        "heldout_relation": None if heldout_relation is None else relation_key(*heldout_relation),
        "heldout_relation_source": None if heldout_relation is None else heldout_relation[0],
        "heldout_relation_destination": None if heldout_relation is None else heldout_relation[1],
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "train_values": train_values,
        "test_values": test_values,
        "n_train_transitions": len(train_transitions),
        "n_test_transitions": len(test_transitions),
        "source_L_dim": source.basis.shape[1],
        "destination_L_dim": destination.basis.shape[1],
        "alpha": float(alpha),
        "direct_alpha": float(direct_edge.alpha),
        "direct_map_path": str(direct_edge.path),
        "synchronization_path": None if synchronization_path is None else str(synchronization_path),
        "fit_value_leakage_check": "passed_disjoint_train_test_values",
        **metric_set(x_train, y_train, q, alpha, "train_transition"),
        **metric_set(x_test, y_test, q, alpha, "heldout_transition"),
        **retrieval,
        **common,
        **map_prediction_metrics(q, alpha, direct_edge.q, direct_edge.alpha),
    }


def pairwise_inverse_checks(edges: dict[tuple[str, str], sync.Edge]) -> list[dict]:
    rows = []
    seen = set()
    for (source, destination), edge in sorted(edges.items()):
        if (destination, source) in seen or (destination, source) not in edges:
            continue
        reverse = edges[(destination, source)]
        rows.append(
            {
                "source_task": source,
                "destination_task": destination,
                "relation": relation_key(source, destination),
                "Q_inverse_relative_error": float((edge.q - reverse.q.T).norm() / edge.q.norm().clamp_min(1e-12)),
                "alpha_inverse_log_error": abs(math.log(max(edge.alpha, 1e-12)) + math.log(max(reverse.alpha, 1e-12))),
                "alpha_product": float(edge.alpha * reverse.alpha),
                "source_map_path": str(edge.path),
                "reverse_map_path": str(reverse.path),
            }
        )
        seen.add((source, destination))
    return rows


def summarize_results(rows: list[dict]) -> list[dict]:
    metrics = [
        "heldout_transition_cosine_mean",
        "heldout_transition_relative_error_mean",
        "top1_value_retrieval",
        "top5_value_retrieval",
        "same_value_cosine_mean",
        "common_frame_same_value_cosine_mean",
        "common_frame_different_value_cosine_mean",
        "Q_relative_frobenius_error",
        "Q_trace_similarity",
        "scale_relative_error",
        "scale_log_error",
        "operator_relative_distance_to_direct",
    ]
    groups = {}
    for row in rows:
        relation = row["heldout_relation"] if row["transport"] == "synchronized_loro" else undirected_key(row["source_task"], row["destination_task"])
        groups.setdefault((row["transport"], relation), []).append(row)
    output = []
    for (transport, relation), parts in sorted(groups.items()):
        item = {"transport": transport, "relation": relation, "n": len(parts)}
        for metric in metrics:
            values = [part.get(metric) for part in parts]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = sample_std(values)
            item[f"{metric}_sem"] = sem(values)
        output.append(item)
    for transport in sorted({row["transport"] for row in rows}):
        parts = [row for row in rows if row["transport"] == transport]
        item = {"transport": transport, "relation": "ALL", "n": len(parts)}
        for metric in metrics:
            values = [part.get(metric) for part in parts]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = sample_std(values)
            item[f"{metric}_sem"] = sem(values)
        output.append(item)
    return output


def save_summary_table(summary_rows: list[dict], path: Path) -> None:
    compact = []
    for row in summary_rows:
        compact.append(
            {
                "transport": row["transport"],
                "relation": row["relation"],
                "n": row["n"],
                "transition_cosine": row.get("heldout_transition_cosine_mean_mean"),
                "value_retrieval_top1": row.get("top1_value_retrieval_mean"),
                "mapped_same_value_cosine": row.get("same_value_cosine_mean_mean"),
                "common_frame_same_value_cosine": row.get("common_frame_same_value_cosine_mean_mean"),
                "common_frame_different_value_cosine": row.get("common_frame_different_value_cosine_mean_mean"),
                "Q_prediction_error": row.get("Q_relative_frobenius_error_mean"),
                "Q_trace_similarity": row.get("Q_trace_similarity_mean"),
                "scale_log_error": row.get("scale_log_error_mean"),
            }
        )
    geom.write_csv(compact, path)


def rsa_rows_for_split(
    args: argparse.Namespace,
    tasks: list[str],
    spaces: dict[tuple[str, int], LSpace],
    seed: int,
    value_split_seed: int,
    test_values: list[int],
) -> list[dict]:
    rows = []
    for task_a, task_b in unique_relations(tasks):
        source = spaces[(task_a, seed)]
        destination = spaces[(task_b, seed)]
        row = geom.rsa_row(
            geom.TaskSpace(source.task, source.seed, SPACE_TYPE, source.basis, source.centroids, source.diagnostics),
            geom.TaskSpace(destination.task, destination.seed, SPACE_TYPE, destination.basis, destination.centroids, destination.diagnostics),
            test_values,
            value_split_seed,
            args.rsa_permutations,
            stable_seed(EXPERIMENT, "rsa", task_a, task_b, seed, value_split_seed),
        )
        row["space_type"] = SPACE_TYPE
        row["task_relation"] = relation_key(task_a, task_b)
        rows.append(row)
    return rows


def plot_figures(args: argparse.Namespace, summary_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    relation_order = [relation_key(a, b) for a, b in unique_relations(args.tasks)] + ["ALL"]
    transports = ["pairwise_direct", "synchronized_full", "synchronized_loro"]
    labels = {
        "pairwise_direct": "direct",
        "synchronized_full": "full sync",
        "synchronized_loro": "LORO sync",
    }
    metrics = [
        ("heldout_transition_cosine_mean_mean", "held-out transition cosine", "figure1_L_sync_transition_cosine"),
        ("top1_value_retrieval_mean", "top-1 value retrieval", "figure2_L_sync_value_retrieval"),
        ("Q_relative_frobenius_error_mean", "Q prediction error", "figure3_L_sync_map_prediction"),
    ]
    lookup = {(row["transport"], row["relation"]): row for row in summary_rows}
    for metric, ylabel, stem in metrics:
        fig, ax = plt.subplots(figsize=(11.0, 4.2))
        x = np.arange(len(relation_order))
        width = 0.24
        offsets = np.linspace(-width, width, len(transports))
        for offset, transport in zip(offsets, transports):
            values = [
                np.nan if (transport, relation) not in lookup else lookup[(transport, relation)].get(metric)
                for relation in relation_order
            ]
            ax.bar(x + offset, values, width, label=labels[transport])
        ax.set_xticks(x, relation_order, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{stem}.png", dpi=240)
        fig.savefig(figure_dir / f"{stem}.pdf")
        plt.close(fig)


def run_for_seed_split(
    args: argparse.Namespace,
    tasks: list[str],
    spaces: dict[tuple[str, int], LSpace],
    *,
    seed: int,
    value_split_seed: int,
    train_values: list[int],
    test_values: list[int],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    assert_value_split(train_values, test_values, f"seed={seed} value_split={value_split_seed}")
    space_by_task = {task: spaces[(task, seed)] for task in tasks}
    edge_by_pair = {}
    map_rows = []
    for source_task, destination_task in directed_pairs(tasks):
        edge = fit_or_load_edge(
            args,
            space_by_task[source_task],
            space_by_task[destination_task],
            value_split_seed=value_split_seed,
            train_values=train_values,
            test_values=test_values,
        )
        edge_by_pair[(source_task, destination_task)] = edge
        map_rows.append(
            {
                "source_task": source_task,
                "destination_task": destination_task,
                "task_relation": directed_key(source_task, destination_task),
                "seed": seed,
                "value_split_seed": value_split_seed,
                "path": str(edge.path),
                "alpha": edge.alpha,
                "weight": edge.weight,
                **edge.fit_metrics,
            }
        )

    all_edges = list(edge_by_pair.values())
    full_orientations, full_log_scales, rotation_stats, scale_stats, full_path = synchronize_and_save(
        args,
        tasks,
        all_edges,
        seed=seed,
        value_split_seed=value_split_seed,
        condition="synchronized_full",
        heldout_relation=None,
    )
    sync_rows = [
        {
            "condition": "synchronized_full",
            "seed": seed,
            "value_split_seed": value_split_seed,
            "heldout_relation": None,
            "n_edges": len(all_edges),
            "components": sync.connected_components(tasks, all_edges),
            "synchronization_path": str(full_path),
            "rotation_residual_mean": rotation_stats["rotation_residual_mean"],
            "rotation_residual_std": rotation_stats["rotation_residual_std"],
            "scale_residual_mean_abs": scale_stats["scale_residual_mean_abs"],
            "scale_residual_std_abs": scale_stats["scale_residual_std_abs"],
        }
    ]
    result_rows = []
    for source_task, destination_task in directed_pairs(tasks):
        source = space_by_task[source_task]
        destination = space_by_task[destination_task]
        direct = edge_by_pair[(source_task, destination_task)]
        result_rows.append(
            evaluate_transport(
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
                synchronization_path=None,
                heldout_relation=None,
            )
        )
        q_full, alpha_full = sync.hub_map(source_task, destination_task, full_orientations, full_log_scales)
        result_rows.append(
            evaluate_transport(
                args,
                source=source,
                destination=destination,
                value_split_seed=value_split_seed,
                train_values=train_values,
                test_values=test_values,
                transport="synchronized_full",
                q=q_full,
                alpha=alpha_full,
                direct_edge=direct,
                synchronization_path=full_path,
                heldout_relation=None,
                orientations=full_orientations,
                log_scales=full_log_scales,
            )
        )

    for heldout in unique_relations(tasks):
        banned = {heldout, (heldout[1], heldout[0])}
        train_edges = [edge for pair, edge in edge_by_pair.items() if pair not in banned]
        assert_loro_edges_absent(train_edges, heldout)
        components = sync.connected_components(tasks, train_edges)
        if any(len(component) < len(tasks) for component in components):
            raise RuntimeError(f"LORO graph disconnected for {heldout}: {components}")
        orientations, log_scales, rotation_stats, scale_stats, path = synchronize_and_save(
            args,
            tasks,
            train_edges,
            seed=seed,
            value_split_seed=value_split_seed,
            condition="synchronized_loro",
            heldout_relation=heldout,
        )
        sync_rows.append(
            {
                "condition": "synchronized_loro",
                "seed": seed,
                "value_split_seed": value_split_seed,
                "heldout_relation": relation_key(*heldout),
                "heldout_source": heldout[0],
                "heldout_destination": heldout[1],
                "n_edges": len(train_edges),
                "components": components,
                "synchronization_path": str(path),
                "leakage_check": "passed_both_directed_edges_removed",
                "rotation_residual_mean": rotation_stats["rotation_residual_mean"],
                "rotation_residual_std": rotation_stats["rotation_residual_std"],
                "scale_residual_mean_abs": scale_stats["scale_residual_mean_abs"],
                "scale_residual_std_abs": scale_stats["scale_residual_std_abs"],
            }
        )
        for source_task, destination_task in [heldout, (heldout[1], heldout[0])]:
            source = space_by_task[source_task]
            destination = space_by_task[destination_task]
            direct = edge_by_pair[(source_task, destination_task)]
            q_loro, alpha_loro = sync.hub_map(source_task, destination_task, orientations, log_scales)
            result_rows.append(
                evaluate_transport(
                    args,
                    source=source,
                    destination=destination,
                    value_split_seed=value_split_seed,
                    train_values=train_values,
                    test_values=test_values,
                    transport="synchronized_loro",
                    q=q_loro,
                    alpha=alpha_loro,
                    direct_edge=direct,
                    synchronization_path=path,
                    heldout_relation=heldout,
                    orientations=orientations,
                    log_scales=log_scales,
                )
            )

    inverse_rows = pairwise_inverse_checks(edge_by_pair)
    for row in inverse_rows:
        row["seed"] = seed
        row["value_split_seed"] = value_split_seed
    return result_rows, map_rows, sync_rows, inverse_rows


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = normalize_tasks(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_selection = selected_seeds(args)
    sync_seeds = sync.common_sync_seeds(seed_selection)
    value_splits = load_value_splits(args)
    common_values = lgeom.all_split_values(value_splits)

    print("Readout-orthogonal L synchronization")
    print(f"  tasks={args.tasks}")
    print(f"  sync_seeds={sync_seeds}")
    print(f"  value_split_seeds={list(value_splits)}")
    print(f"  L dimension={args.latent_dim}")
    print("  model forwards=0")

    data_by_task = load_rows_and_data(args, args.tasks, seed_selection)
    for task, data in data_by_task.items():
        missing = [value for value in common_values if data.counts.get(value, 0) < 1]
        if missing:
            raise ValueError(f"{task} is missing value-split values: {missing[:10]}")

    spaces, diagnostics = build_l_spaces(args, args.tasks, seed_selection, data_by_task, common_values)
    geom.write_csv(diagnostics, args.output_dir / "L_space_diagnostics.csv")
    save_json(value_splits, args.output_dir / "value_splits.json")

    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "space_type": SPACE_TYPE,
                "tasks": args.tasks,
                "seed_selection": seed_selection,
                "sync_seeds": sync_seeds,
                "L_dimension": args.latent_dim,
                "common_values": common_values,
                "value_splits": value_splits,
                "diagnostics_rows": len(diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("\nARTIFACT_CHECK_ONLY complete")
        return

    result_rows = []
    map_rows = []
    sync_rows = []
    inverse_rows = []
    rsa_rows = []
    for split_seed_text, split in value_splits.items():
        value_split_seed = int(split_seed_text)
        train_values = split["train_values"]
        test_values = split["test_values"]
        print(f"\nVALUE_SPLIT seed={value_split_seed} train={len(train_values)} test={len(test_values)}")
        for seed in sync_seeds:
            print(f"  SYNC_SEED {seed}")
            rows, maps, sync_fit, inverses = run_for_seed_split(
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
            inverse_rows.extend(inverses)
            rsa_rows.extend(rsa_rows_for_split(args, args.tasks, spaces, seed, value_split_seed, test_values))
            save_jsonl(result_rows, args.output_dir / "L_synchronization_geometric_results.jsonl")
            geom.write_csv(result_rows, args.output_dir / "L_synchronization_geometric_results.csv")
            geom.write_csv(map_rows, args.output_dir / "L_pairwise_map_summary.csv")
            geom.write_csv(sync_rows, args.output_dir / "L_synchronization_fit_summary.csv")
            geom.write_csv(inverse_rows, args.output_dir / "L_pairwise_inverse_checks.csv")
            geom.write_csv(rsa_rows, args.output_dir / "L_synchronization_rsa.csv")

    summary_rows = summarize_results(result_rows)
    geom.write_csv(summary_rows, args.output_dir / "L_synchronization_relation_summary.csv")
    save_summary_table(summary_rows, args.output_dir / "L_synchronization_summary_table.csv")
    rsa_summary = geom.summarize(rsa_rows, ["spearman_rsa", "permutation_pvalue"], ["task_relation", "space_type"])
    save_json(
        {
            "experiment": EXPERIMENT,
            "description": (
                "Full-graph and leave-one-relation-out synchronization of the exact 13D "
                "readout-orthogonal L components, evaluated on held-out numerical values."
            ),
            "space_type": SPACE_TYPE,
            "tasks": args.tasks,
            "task_labels": geom.TASK_LABELS,
            "L_dimension": args.latent_dim,
            "seed_selection": seed_selection,
            "sync_seeds": sync_seeds,
            "value_splits": value_splits,
            "common_values": common_values,
            "transports": TRANSPORTS,
            "n_result_rows": len(result_rows),
            "n_map_rows": len(map_rows),
            "n_sync_rows": len(sync_rows),
            "n_rsa_rows": len(rsa_rows),
            "relation_summary": summary_rows,
            "rsa_summary": rsa_summary,
            "inverse_checks_summary": geom.summarize(
                inverse_rows,
                ["Q_inverse_relative_error", "alpha_inverse_log_error"],
                ["relation"],
            ),
            "controls_and_leakage_checks": {
                "L_dimension_assertion": f"all spaces exactly {args.latent_dim}D",
                "orthogonality_to_readout": "checked by construct_exact_CL diagnostics",
                "heldout_relation_leakage": "both directed edges removed and asserted absent for LORO",
                "heldout_value_leakage": "pairwise maps fit only on train_values; test_values asserted disjoint",
                "inverse_consistency": "reported, not enforced",
                "autoregressive_iia": "not run; this experiment is geometric only",
            },
            "config": jsonable(vars(args)),
        },
        args.output_dir / "L_synchronization_summary.json",
    )
    if not args.skip_plots:
        plot_figures(args, summary_rows)

    print("\nReadout-orthogonal L synchronization complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Result rows: {len(result_rows)}")
    print(f"Summary: {args.output_dir / 'L_synchronization_summary.json'}")
    for row in summary_rows:
        if row["relation"] == "ALL":
            print(
                f"  ALL {row['transport']}: transition_cosine="
                f"{row.get('heldout_transition_cosine_mean_mean')} top1="
                f"{row.get('top1_value_retrieval_mean')} Q_error="
                f"{row.get('Q_relative_frobenius_error_mean')}"
            )


if __name__ == "__main__":
    main()
