"""
Shared geometry of the exact causally active readout-orthogonal DAS component.

This is an offline experiment. It does not load Gemma, run forwards, generate,
or train. For each final DAS basis R, it reconstructs the exact split used by
latent_readout_causal_interaction:

    U_digit^T R = A Sigma V^T
    Q = R V
    C = Q[:, :9]
    L = Q[:, 9:22]

Then it evaluates whether the 13-D L coordinates preserve shared numerical
geometry across task, operation, and modality using the existing value-heldout
transition Procrustes, value retrieval, and no-fit RSA metrics.
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
from src.experiments.autoregressive_transfer.latent_identity_swap import build_ordered_split
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    first_result_row,
    jsonable,
    parse_task,
    save_json,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed


EXPERIMENT = "causal_L_shared_geometry"
TASK_ORDER = geom.TASK_ORDER
TASK_LABELS = geom.TASK_LABELS
TASK_PAIRS_UNIQUE = [
    (a, b)
    for index, a in enumerate(TASK_ORDER)
    for b in TASK_ORDER[index + 1 :]
]
TASK_PAIRS_DIRECTED = [(a, b) for a in TASK_ORDER for b in TASK_ORDER if a != b]


@dataclass
class SpaceItem:
    task: str
    space_type: str
    das_seed: int
    basis: torch.Tensor
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
        "--previous_specificity_dir",
        type=Path,
        default=Path("results/experiments/closing/readout_free_space_specificity"),
        help="Optional completed baseline/control directory loaded only for summary comparison.",
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
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/causal_L_shared_geometry"))
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=geom.DEFAULT_TASKS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--value_split_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--control_seeds", type=int, nargs="+", default=list(range(20)))
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
    parser.add_argument("--random_controls", type=int, default=20, help="Shuffled-label controls per relation/split.")
    parser.add_argument("--rsa_permutations", type=int, default=1000)
    parser.add_argument("--include_C_space", action="store_true")
    parser.add_argument("--sanity_tolerance", type=float, default=1e-5)
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_tasks(tasks: list[str]) -> list[str]:
    if any(task.lower() == "all" for task in tasks):
        return list(geom.DEFAULT_TASKS)
    return [task_key(*parse_task(task)) for task in tasks]


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.tasks == geom.DEFAULT_TASKS:
        args.tasks = ["text:addition", "image:addition"]
    if args.seeds == [0, 1, 2]:
        args.seeds = [0]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]
    if args.control_seeds == list(range(20)):
        args.control_seeds = [0, 1]
    args.random_controls = min(args.random_controls, 2)
    args.rsa_permutations = min(args.rsa_permutations, 100)


def load_exact_value_splits(args: argparse.Namespace) -> dict[str, dict[str, list[int]]]:
    path = args.previous_geometry_dir / "value_splits.json"
    selected = specificity.load_exact_value_splits(args)
    for seed_text, split in selected.items():
        seed = int(seed_text)
        train_values = split["train_values"]
        test_values = split["test_values"]
        if len(train_values) != 80 or len(test_values) != 20:
            raise ValueError(
                f"{path} split {seed} is not the required 80/20 value split: "
                f"train={len(train_values)} test={len(test_values)}."
            )
        if set(train_values) & set(test_values):
            raise ValueError(f"{path} split {seed} has overlapping train/test values.")
        if set(train_values) | set(test_values) != set(range(100)):
            raise ValueError(f"{path} split {seed} does not cover exactly values 0..99.")
    return selected


def all_split_values(value_splits: dict[str, dict[str, list[int]]]) -> list[int]:
    values = set()
    for split in value_splits.values():
        values.update(split["train_values"])
        values.update(split["test_values"])
    return sorted(values)


def load_task_rows_and_data(args: argparse.Namespace, tasks: list[str]) -> tuple[dict[str, dict], dict[str, geom.TaskData]]:
    rows_by_task = {}
    data_by_task = {}
    for task in tasks:
        modality, operation = parse_task(task)
        row = first_result_row(args, modality, operation, args.condition, args.seeds[0])
        geom.validate_result_metadata(
            args,
            row,
            task=task,
            path=subspace_path(args, modality, operation, args.seeds[0]).with_name("results.jsonl"),
        )
        rows_by_task[task] = row
        data_by_task[task] = geom.load_task_data(args, task, row)
    return rows_by_task, data_by_task


def rank_from_svd(matrix: torch.Tensor) -> int:
    singular_values = torch.linalg.svdvals(matrix.float())
    if singular_values.numel() == 0:
        return 0
    tolerance = max(matrix.shape) * torch.finfo(torch.float32).eps * singular_values.max()
    return int((singular_values > tolerance).sum().item())


def construct_exact_CL(
    args: argparse.Namespace,
    *,
    task: str,
    data: geom.TaskData,
    digit_basis: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    basis_path = subspace_path(args, data.modality, data.operation, seed)
    result_path = basis_path.with_name("results.jsonl")
    result_row = first_result_row(args, data.modality, data.operation, args.condition, seed)
    geom.validate_result_metadata(args, result_row, task=task, path=result_path)
    raw, _payload = geom.load_basis_file(basis_path, data.hidden.shape[1], args.k)
    basis, ordered, c_basis, l_basis, split_diag = build_ordered_split(
        args,
        raw_basis=raw,
        digit_basis=digit_basis,
        hidden_size=data.hidden.shape[1],
        m_readout=args.m_readout,
    )
    if l_basis.shape[1] != args.latent_dim:
        raise ValueError(f"{task} seed={seed} L rank is {l_basis.shape[1]}, expected {args.latent_dim}.")
    projector_split = c_basis @ c_basis.T + l_basis @ l_basis.T
    projector_basis = basis @ basis.T
    singular_values = torch.linalg.svdvals(digit_basis.T @ basis)
    if singular_values.numel() < basis.shape[1]:
        singular_values = torch.cat(
            [
                singular_values,
                torch.zeros(basis.shape[1] - singular_values.numel(), dtype=singular_values.dtype),
            ]
        )
    diagnostics = {
        "task": task,
        "task_label": TASK_LABELS[task],
        "das_seed": seed,
        "rank": int(l_basis.shape[1]),
        "rank_L": rank_from_svd(l_basis),
        "rank_C": rank_from_svd(c_basis),
        "orthogonality_error": float((l_basis.T @ l_basis - torch.eye(l_basis.shape[1])).norm()),
        "digit_overlap": float((digit_basis.T @ l_basis).norm()),
        "C_L_overlap": float((c_basis.T @ l_basis).norm()),
        "split_projector_error": float((projector_split - projector_basis).norm()),
        "basis_projector_error": split_diag["ordered_projector_error"],
        "m_readout": args.m_readout,
        "latent_dim": args.latent_dim,
        "singular_values": [float(value) for value in singular_values.tolist()],
        "basis_path": str(basis_path),
        "regression_check": "recomputed_with_latent_readout_causal_interaction.build_ordered_split",
        "causal_experiment_saved_L_basis": False,
        "projector_error_vs_causal_recompute": 0.0,
    }
    failures = []
    if diagnostics["rank_L"] != args.latent_dim:
        failures.append("rank_L")
    if diagnostics["orthogonality_error"] > args.sanity_tolerance:
        failures.append("orthogonality_error")
    if diagnostics["digit_overlap"] > args.sanity_tolerance:
        failures.append("digit_overlap")
    if diagnostics["C_L_overlap"] > args.sanity_tolerance:
        failures.append("C_L_overlap")
    if diagnostics["split_projector_error"] > args.sanity_tolerance:
        failures.append("split_projector_error")
    if failures:
        raise RuntimeError(f"L-space sanity failed for {task} seed={seed}: {failures}; {diagnostics}")
    return c_basis, l_basis, ordered, diagnostics


def random_readout_orthogonal_basis(hidden_size: int, rank: int, digit_basis: torch.Tensor, seed: int) -> tuple[torch.Tensor, dict]:
    raw = specificity.random_residual_basis(hidden_size, rank, seed)
    basis, meta = specificity.project_out_readout(raw, digit_basis, rank, "random 13D readout-orthogonal")
    if basis.shape[1] != rank:
        raise ValueError(f"Random readout-orthogonal basis rank {basis.shape[1]} != {rank}.")
    return basis, meta


def result_centroid_variance_fraction(data: geom.TaskData, basis: torch.Tensor, values: set[int], target: str) -> float:
    return specificity.result_centroid_variance_fraction(data, basis, values, target)


def build_spaces(
    args: argparse.Namespace,
    tasks: list[str],
    data_by_task: dict[str, geom.TaskData],
    common_values: list[int],
) -> tuple[dict[tuple[str, str], list[SpaceItem]], list[dict]]:
    spaces: dict[tuple[str, str], list[SpaceItem]] = {}
    diagnostics = []
    common_set = set(common_values)
    for task in tasks:
        data = data_by_task[task]
        digit_basis = geom.load_digit_basis(args.digit_readout_basis_path, data.hidden.shape[1])
        if digit_basis.shape[1] != args.m_readout:
            raise ValueError(
                f"{args.digit_readout_basis_path} rank is {digit_basis.shape[1]}, expected m_readout={args.m_readout}."
            )
        for seed in args.seeds:
            c_basis, l_basis, _ordered, diag = construct_exact_CL(args, task=task, data=data, digit_basis=digit_basis, seed=seed)
            centroids = geom.centroid_coordinates(data, l_basis, common_set, args.target)
            l_r2 = result_centroid_variance_fraction(data, l_basis, common_set, args.target)
            diag["space_type"] = "causal_L"
            diag["result_centroid_variance_fraction"] = l_r2
            diagnostics.append(diag)
            spaces.setdefault((task, "causal_L"), []).append(
                SpaceItem(task, "causal_L", seed, l_basis, centroids, diag)
            )
            print(
                f"L_SPACE task={task} seed={seed} rank={l_basis.shape[1]} "
                f"digit_overlap={diag['digit_overlap']:.3e} value_R2={l_r2:.4f}"
            )
            if args.include_C_space:
                c_diag = {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "das_seed": seed,
                    "space_type": "readout_coupled_C",
                    "rank_L": None,
                    "rank_C": rank_from_svd(c_basis),
                    "rank": c_basis.shape[1],
                    "orthogonality_error": float((c_basis.T @ c_basis - torch.eye(c_basis.shape[1])).norm()),
                    "digit_overlap": float((digit_basis.T @ c_basis).norm()),
                    "C_L_overlap": 0.0,
                    "split_projector_error": None,
                    "m_readout": args.m_readout,
                    "latent_dim": None,
                    "result_centroid_variance_fraction": result_centroid_variance_fraction(data, c_basis, common_set, args.target),
                }
                diagnostics.append(c_diag)
                spaces.setdefault((task, "readout_coupled_C"), []).append(
                    SpaceItem(task, "readout_coupled_C", seed, c_basis, geom.centroid_coordinates(data, c_basis, common_set, args.target), c_diag)
                )
        for control_seed in args.control_seeds:
            basis, meta = random_readout_orthogonal_basis(
                data.hidden.shape[1],
                args.latent_dim,
                digit_basis,
                stable_seed(EXPERIMENT, "random_readout_orthogonal_13d", task, control_seed),
            )
            centroids = geom.centroid_coordinates(data, basis, common_set, args.target)
            diag = {
                "task": task,
                "task_label": TASK_LABELS[task],
                "das_seed": control_seed,
                "space_type": "random_readout_orthogonal_13d",
                "rank_L": basis.shape[1],
                "rank_C": None,
                "rank": basis.shape[1],
                "orthogonality_error": float((basis.T @ basis - torch.eye(basis.shape[1])).norm()),
                "digit_overlap": float((digit_basis.T @ basis).norm()),
                "C_L_overlap": None,
                "split_projector_error": None,
                "m_readout": args.m_readout,
                "latent_dim": args.latent_dim,
                "control_seed": control_seed,
                "result_centroid_variance_fraction": result_centroid_variance_fraction(data, basis, common_set, args.target),
                **meta,
            }
            diagnostics.append(diag)
            spaces.setdefault((task, "random_readout_orthogonal_13d"), []).append(
                SpaceItem(task, "random_readout_orthogonal_13d", control_seed, basis, centroids, diag)
            )
    return spaces, diagnostics


def paired_items(
    spaces: dict[tuple[str, str], list[SpaceItem]],
    source_task: str,
    destination_task: str,
    space_type: str,
) -> list[tuple[SpaceItem, SpaceItem]]:
    source_items = spaces[(source_task, space_type)]
    destination_items = spaces[(destination_task, space_type)]
    if space_type in {"causal_L", "readout_coupled_C"}:
        return [(source, destination) for source in source_items for destination in destination_items]
    destination_by_seed = {item.das_seed: item for item in destination_items}
    return [(source, destination_by_seed[source.das_seed]) for source in source_items if source.das_seed in destination_by_seed]


def as_task_space(item: SpaceItem) -> geom.TaskSpace:
    return geom.TaskSpace(item.task, item.das_seed, item.space_type, item.basis, item.centroids, item.diagnostics)


def base_relation_row(source: SpaceItem, destination: SpaceItem, split_seed: int, train_values: list[int], test_values: list[int]) -> dict:
    return {
        "source_task": source.task,
        "source_task_label": TASK_LABELS[source.task],
        "destination_task": destination.task,
        "destination_task_label": TASK_LABELS[destination.task],
        "task_relation": f"{TASK_LABELS[source.task]}->{TASK_LABELS[destination.task]}",
        "source_seed": source.das_seed,
        "destination_seed": destination.das_seed,
        "value_split_seed": split_seed,
        "space_type": source.space_type,
        "space": source.space_type,
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "chance_top1": 1 / len(test_values),
        "source_rank": source.basis.shape[1],
        "destination_rank": destination.basis.shape[1],
    }


def analyze_primary_spaces(
    args: argparse.Namespace,
    tasks: list[str],
    spaces: dict[tuple[str, str], list[SpaceItem]],
    value_splits: dict[str, dict[str, list[int]]],
    space_types: list[str],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    transition_rows = []
    retrieval_rows = []
    rsa_rows = []
    control_rows = []
    task_pairs = [(a, b) for a in tasks for b in tasks if a != b]
    for split_seed_text, split in value_splits.items():
        split_seed = int(split_seed_text)
        train_values = split["train_values"]
        test_values = split["test_values"]
        print(f"\nVALUE_SPLIT seed={split_seed} train={len(train_values)} test={len(test_values)}")
        for space_type in space_types:
            for source_task, destination_task in task_pairs:
                for source, destination in paired_items(spaces, source_task, destination_task, space_type):
                    train_transitions, x_train = geom.transition_matrix(source.centroids, train_values)
                    _train_transitions_b, y_train = geom.transition_matrix(destination.centroids, train_values)
                    test_transitions, x_test = geom.transition_matrix(source.centroids, test_values)
                    _test_transitions_b, y_test = geom.transition_matrix(destination.centroids, test_values)
                    q, alpha = geom.fit_scaled_procrustes(x_train, y_train)
                    base = {
                        **base_relation_row(source, destination, split_seed, train_values, test_values),
                        "n_train_transitions": len(train_transitions),
                        "n_test_transitions": len(test_transitions),
                        "alpha": alpha,
                    }
                    transition_metrics = geom.transition_metrics(x_test, y_test, q, alpha)
                    retrieval_metrics = geom.retrieval_metrics(source.centroids, destination.centroids, train_values, test_values, q, alpha)
                    transition_rows.append({**base, **transition_metrics})
                    retrieval_rows.append({**base, **retrieval_metrics})
                    if space_type == "random_readout_orthogonal_13d":
                        control_rows.append(
                            {
                                **base,
                                "control_type": "random_readout_orthogonal_13d",
                                "control_seed": source.das_seed,
                                **transition_metrics,
                                **retrieval_metrics,
                            }
                        )
                    if space_type == "causal_L":
                        rng = random.Random(
                            stable_seed(EXPERIMENT, "shuffle", source_task, destination_task, source.das_seed, destination.das_seed, split_seed)
                        )
                        for control_seed in range(args.random_controls):
                            shuffled = list(train_values)
                            rng.shuffle(shuffled)
                            shuffled_destination = {
                                value: destination.centroids[shuffled[index]]
                                for index, value in enumerate(train_values)
                            }
                            _shuf_transitions, y_train_shuffled = geom.transition_matrix(shuffled_destination, train_values)
                            q_shuffle, alpha_shuffle = geom.fit_scaled_procrustes(x_train, y_train_shuffled)
                            retrieval_destination = dict(destination.centroids)
                            retrieval_destination.update(shuffled_destination)
                            shuffled_retrieval = geom.retrieval_metrics(
                                source.centroids,
                                retrieval_destination,
                                train_values,
                                test_values,
                                q_shuffle,
                                alpha_shuffle,
                            )
                            control_base = {
                                **base,
                                "control_type": "shuffled_numerical_correspondence",
                                "control_seed": control_seed,
                                "source_space_type": base["space_type"],
                                "space_type": "shuffled_causal_L",
                                "space": "shuffled_causal_L",
                                "alpha": alpha_shuffle,
                            }
                            control_rows.append(
                                {
                                    **control_base,
                                    **geom.transition_metrics(x_test, y_test, q_shuffle, alpha_shuffle),
                                    **shuffled_retrieval,
                                }
                            )
            for task_a, task_b in TASK_PAIRS_UNIQUE:
                if task_a not in tasks or task_b not in tasks:
                    continue
                for source, destination in paired_items(spaces, task_a, task_b, space_type):
                    rsa = geom.rsa_row(
                        as_task_space(source),
                        as_task_space(destination),
                        test_values,
                        split_seed,
                        args.rsa_permutations,
                        stable_seed(EXPERIMENT, "rsa", task_a, task_b, source.das_seed, destination.das_seed, split_seed, space_type),
                    )
                    rsa["space_type"] = space_type
                    rsa["space"] = space_type
                    rsa["task_relation"] = f"{TASK_LABELS[task_a]}-{TASK_LABELS[task_b]}"
                    rsa_rows.append(rsa)
    return transition_rows, retrieval_rows, rsa_rows, control_rows


def load_previous_baseline_summary(path: Path) -> dict:
    candidates = [
        path / "space_specificity_summary.json",
        path / "summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {"available": False, "path": str(path)}


def mean_metric(rows: list[dict], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]
    return None if not values else float(sum(values) / len(values))


def relation_summary(rows: list[dict], metrics: list[str], relation_key: str = "task_relation") -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row[relation_key], row["space_type"]), []).append(row)
    summary = []
    for (relation, space_type), parts in sorted(groups.items()):
        item = {"task_relation": relation, "space_type": space_type, "n": len(parts)}
        for metric in metrics:
            values = [float(row[metric]) for row in parts if row.get(metric) not in {None, ""}]
            item[f"{metric}_mean"] = None if not values else float(sum(values) / len(values))
            item[f"{metric}_sd"] = None if len(values) < 2 else float(statistics.stdev(values))
        summary.append(item)
    return summary


def aggregate_summary(args: argparse.Namespace, transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict], control_rows: list[dict]) -> dict:
    primary_spaces = sorted({row["space_type"] for row in transition_rows})
    primary = {}
    for space_type in primary_spaces:
        transition = [row for row in transition_rows if row["space_type"] == space_type]
        retrieval = [row for row in retrieval_rows if row["space_type"] == space_type]
        rsa = [row for row in rsa_rows if row["space_type"] == space_type]
        primary[space_type] = {
            "mean_heldout_transition_cosine": mean_metric(transition, "heldout_transition_cosine_mean"),
            "mean_heldout_relative_error": mean_metric(transition, "heldout_relative_error_mean"),
            "mean_top1_retrieval": mean_metric(retrieval, "top1_value_retrieval"),
            "mean_top5_retrieval": mean_metric(retrieval, "top5_value_retrieval"),
            "mean_same_value_cosine": mean_metric(retrieval, "same_value_cosine_mean"),
            "mean_cross_task_rsa": mean_metric(rsa, "spearman_rsa"),
            "mean_rsa_pvalue": mean_metric(rsa, "permutation_pvalue"),
        }
    shuffled = [row for row in control_rows if row["control_type"] == "shuffled_numerical_correspondence"]
    primary["shuffled_causal_L"] = {
        "mean_heldout_transition_cosine": mean_metric(shuffled, "heldout_transition_cosine_mean"),
        "mean_heldout_relative_error": mean_metric(shuffled, "heldout_relative_error_mean"),
        "mean_top1_retrieval": mean_metric(shuffled, "top1_value_retrieval"),
        "mean_top5_retrieval": mean_metric(shuffled, "top5_value_retrieval"),
        "mean_same_value_cosine": mean_metric(shuffled, "same_value_cosine_mean"),
    }
    return {
        "primary_metric_means": primary,
        "transition_relation_summary": relation_summary(transition_rows, ["heldout_transition_cosine_mean", "heldout_relative_error_mean"]),
        "retrieval_relation_summary": relation_summary(retrieval_rows, ["top1_value_retrieval", "top5_value_retrieval", "same_value_cosine_mean", "mean_absolute_value_error"]),
        "rsa_pair_summary": relation_summary(rsa_rows, ["spearman_rsa", "permutation_pvalue"]),
        "control_relation_summary": relation_summary(control_rows, ["heldout_transition_cosine_mean", "top1_value_retrieval", "same_value_cosine_mean"]),
        "chance_top1": 0.05,
        "previous_baselines": {
            "value_heldout_readout_free_geometry": load_previous_baseline_summary(args.previous_geometry_dir),
            "readout_free_space_specificity": load_previous_baseline_summary(args.previous_specificity_dir),
        },
    }


def baseline_comparison_rows(summary: dict) -> list[dict]:
    rows = []
    primary = summary["primary_metric_means"]
    for label, space_type in [
        ("L_13D_causal_latent", "causal_L"),
        ("random_13D_readout_orthogonal", "random_readout_orthogonal_13d"),
        ("shuffled_L_labels", "shuffled_causal_L"),
        ("C_9D_readout_coupled", "readout_coupled_C"),
    ]:
        if space_type in primary:
            rows.append({"comparison_item": label, "space_type": space_type, **primary[space_type]})

    specificity_summary = summary["previous_baselines"].get("readout_free_space_specificity", {})
    previous_primary = specificity_summary.get("primary_metric_means", {}) if isinstance(specificity_summary, dict) else {}
    for space_type in ["das_readout_free", "pca_leading_readout_free", "random_pca_readout_free", "random_residual_readout_free"]:
        if space_type not in previous_primary:
            continue
        old = previous_primary[space_type]
        rows.append(
            {
                "comparison_item": f"previous_{space_type}",
                "space_type": space_type,
                "mean_heldout_transition_cosine": old.get("mean_heldout_transition_cosine"),
                "mean_heldout_relative_error": old.get("mean_heldout_relative_error"),
                "mean_top1_retrieval": old.get("mean_top1_retrieval"),
                "mean_top5_retrieval": old.get("mean_top5_retrieval"),
                "mean_same_value_cosine": old.get("mean_same_value_cosine"),
                "mean_cross_task_rsa": old.get("mean_cross_task_rsa"),
                "mean_rsa_pvalue": old.get("mean_rsa_pvalue"),
            }
        )
    return rows


def plot_figures(args: argparse.Namespace, transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict], control_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    relations = [f"{TASK_LABELS[a]}->{TASK_LABELS[b]}" for a, b in TASK_PAIRS_DIRECTED if a in args.tasks and b in args.tasks]
    spaces = ["causal_L", "shuffled_causal_L", "random_readout_orthogonal_13d"]
    labels = {
        "causal_L": "L",
        "shuffled_causal_L": "shuffled-L",
        "random_readout_orthogonal_13d": "random 13D RF",
        "readout_coupled_C": "C",
    }

    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(relations))
    width = 0.24
    for offset, space in zip([-width, 0.0, width], spaces):
        means = []
        source = control_rows if space == "shuffled_causal_L" else transition_rows
        for relation in relations:
            values = [
                float(row["heldout_transition_cosine_mean"])
                for row in source
                if row["task_relation"] == relation
                and row.get("space_type") == space
            ]
            means.append(np.nan if not values else float(np.mean(values)))
        ax.bar(x + offset, means, width, label=labels[space])
    ax.set_xticks(x, relations, rotation=45, ha="right")
    ax.set_ylabel("held-out transition cosine")
    ax.set_title("Held-out transition geometry")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure1_L_heldout_transition_geometry.png", dpi=240)
    fig.savefig(figure_dir / "figure1_L_heldout_transition_geometry.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for offset, space in zip([-width, 0.0, width], spaces):
        means = []
        source = control_rows if space == "shuffled_causal_L" else retrieval_rows
        for relation in relations:
            values = [
                float(row["top1_value_retrieval"])
                for row in source
                if row["task_relation"] == relation
                and row.get("space_type") == space
            ]
            means.append(np.nan if not values else float(np.mean(values)))
        ax.bar(x + offset, means, width, label=labels[space])
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xticks(x, relations, rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("top-1 held-out value retrieval")
    ax.set_title("Held-out value retrieval")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure2_L_heldout_value_retrieval.png", dpi=240)
    fig.savefig(figure_dir / "figure2_L_heldout_value_retrieval.pdf")
    plt.close(fig)

    task_indices = {task: index for index, task in enumerate(TASK_ORDER)}
    matrix = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    pvalues = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    for task_a, task_b in TASK_PAIRS_UNIQUE:
        rows = [row for row in rsa_rows if row["space_type"] == "causal_L" and row["task_a"] == task_a and row["task_b"] == task_b]
        if not rows:
            continue
        i, j = task_indices[task_a], task_indices[task_b]
        matrix[i, j] = matrix[j, i] = float(np.mean([float(row["spearman_rsa"]) for row in rows]))
        pvalues[i, j] = pvalues[j, i] = float(np.mean([float(row["permutation_pvalue"]) for row in rows if row.get("permutation_pvalue") not in {None, ""}]))
    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_yticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_title("No-fit RSA in causal L")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure3_L_nofit_rsa_heatmap.png", dpi=240)
    fig.savefig(figure_dir / "figure3_L_nofit_rsa_heatmap.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.7))
    image = ax.imshow(pvalues, vmin=0, vmax=1, cmap="viridis_r")
    ax.set_xticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_yticks(range(len(TASK_ORDER)), [TASK_LABELS[task] for task in TASK_ORDER])
    ax.set_title("RSA permutation p-values")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure3_L_nofit_rsa_pvalues.png", dpi=240)
    fig.savefig(figure_dir / "figure3_L_nofit_rsa_pvalues.pdf")
    plt.close(fig)


def output_exists(args: argparse.Namespace) -> bool:
    return (args.output_dir / "L_geometry_summary.json").exists() and not args.force


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = normalize_tasks(args.tasks)
    if output_exists(args) and not args.artifact_check_only:
        raise FileExistsError(f"{args.output_dir} already has L_geometry_summary.json; pass --force to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Causal L shared geometry")
    print(f"  tasks={args.tasks}")
    print(f"  DAS seeds={args.seeds}")
    print(f"  value_split_seeds={args.value_split_seeds}")
    print(f"  control_seeds={args.control_seeds}")
    print("  model forwards=0")

    value_splits = load_exact_value_splits(args)
    common_values = all_split_values(value_splits)
    rows_by_task, data_by_task = load_task_rows_and_data(args, args.tasks)
    for task, data in data_by_task.items():
        missing = [value for value in common_values if data.counts.get(value, 0) < 1]
        if missing:
            raise ValueError(f"{task} is missing common split values from previous experiment: {missing[:10]}")

    spaces, diagnostics = build_spaces(args, args.tasks, data_by_task, common_values)
    geom.write_csv(diagnostics, args.output_dir / "L_space_diagnostics.csv")
    save_json(value_splits, args.output_dir / "value_splits.json")

    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "seeds": args.seeds,
                "value_splits": value_splits,
                "n_common_values": len(common_values),
                "diagnostics_rows": len(diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("\nARTIFACT_CHECK_ONLY complete")
        return

    space_types = ["causal_L", "random_readout_orthogonal_13d"]
    if args.include_C_space:
        space_types.append("readout_coupled_C")
    transition_rows, retrieval_rows, rsa_rows, control_rows = analyze_primary_spaces(args, args.tasks, spaces, value_splits, space_types)
    summary = aggregate_summary(args, transition_rows, retrieval_rows, rsa_rows, control_rows)
    comparison_rows = baseline_comparison_rows(summary)

    geom.write_csv(transition_rows, args.output_dir / "L_transition_procrustes.csv")
    geom.write_csv(retrieval_rows, args.output_dir / "L_value_retrieval.csv")
    geom.write_csv(rsa_rows, args.output_dir / "L_rsa.csv")
    geom.write_csv(control_rows, args.output_dir / "L_controls.csv")
    geom.write_csv(comparison_rows, args.output_dir / "L_final_comparison.csv")
    save_json(
        {
            "experiment": EXPERIMENT,
            "description": "Shared value-heldout geometry of exact 13-D causal L = span(q_10,...,q_22).",
            "tasks": args.tasks,
            "task_labels": TASK_LABELS,
            "seeds": args.seeds,
            "value_splits": value_splits,
            "n_common_values": len(common_values),
            "common_values": common_values,
            "diagnostics_rows": len(diagnostics),
            "transition_rows": len(transition_rows),
            "retrieval_rows": len(retrieval_rows),
            "rsa_rows": len(rsa_rows),
            "control_rows": len(control_rows),
            "comparison_rows": len(comparison_rows),
            "config": jsonable(vars(args)),
            **summary,
        },
        args.output_dir / "L_geometry_summary.json",
    )
    if not args.skip_plots:
        plot_figures(args, transition_rows, retrieval_rows, rsa_rows, control_rows)

    print("\nCausal L shared geometry complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Transition rows: {len(transition_rows)}")
    print(f"Retrieval rows: {len(retrieval_rows)}")
    print(f"RSA rows: {len(rsa_rows)}")
    print(f"Control rows: {len(control_rows)}")


if __name__ == "__main__":
    main()
