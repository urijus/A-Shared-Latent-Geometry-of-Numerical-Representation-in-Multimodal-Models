"""Factorized Procrustes paths across modality and operation.

The experiment tests whether cross-modal and cross-operation transformations
compose.  For direct cross pairs such as text:addition -> image:subtraction, it
compares three transports on the same held-out numerical transitions:

1. direct: fit source -> destination directly.
2. operation_first: source -> same-modality other-operation -> destination.
3. modality_first: source -> same-operation other-modality -> destination.

All edge maps are scaled displacement Procrustes maps.  Because this code uses
row-vector coordinates, a path source -> middle -> destination composes as
Q_path = Q_source_to_middle @ Q_middle_to_destination.

Maps are fit on transition-level means.  For each ordered result transition
a -> b, the training vector is mean_z(b) - mean_z(a), so each numerical
transition contributes once to the Procrustes objective.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from src.interventions.das import build_unique_pairs, target_answers
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    control_iia,
    first_result_row,
    heldout_pairs_path,
    jsonable,
    label,
    load_basis,
    load_jsonl,
    normalized_transfer,
    parse_task,
    position_for,
    results_path,
    save_json,
    save_jsonl,
    selected_seeds,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    autoregressive_iia_image_aligned,
    autoregressive_iia_text_aligned,
    cached_task_samples,
    centroid_coordinates,
    clean_name,
    coordinate_deltas_for_pairs,
    coordinate_metrics,
    data_root_for,
    gain,
    load_causal_rows,
    load_model_bundle,
    orthogonal_procrustes,
    sample_split,
    scaled_alpha,
    stable_seed,
)


EXPERIMENT = "factorized_paths"
TASKS = [
    "text:addition",
    "text:subtraction",
    "image:addition",
    "image:subtraction",
]
DEFAULT_DIRECT_TESTS = [
    "text:addition->image:subtraction",
    "image:subtraction->text:addition",
    "text:subtraction->image:addition",
    "image:addition->text:subtraction",
]
DEFAULT_SAME_OPERATION_TESTS = [
    "text:addition->image:addition",
    "image:addition->text:addition",
    "text:subtraction->image:subtraction",
    "image:subtraction->text:subtraction",
]
DEFAULT_OPERATION_ONLY_TESTS = [
    "text:addition->text:subtraction",
    "text:subtraction->text:addition",
    "image:addition->image:subtraction",
    "image:subtraction->image:addition",
]
TRANSPORTS = ["direct", "operation_first", "modality_first"]


@dataclass(frozen=True)
class TaskSpace:
    task: str
    modality: str
    operation: str
    seed: int
    row: dict
    basis: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--causal_transfer_rows", type=Path, default=Path("results/final_exps/causal_tranfer/transfer_results.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper/procrustes/factorized_paths"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--direct_tests", nargs="+", default=DEFAULT_DIRECT_TESTS)
    parser.add_argument("--same_operation_tests", nargs="+", default=DEFAULT_SAME_OPERATION_TESTS)
    parser.add_argument("--operation_only_tests", nargs="+", default=DEFAULT_OPERATION_ONLY_TESTS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--control_condition", default="random_subspace_in_pca_span")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--seed_map", nargs="*", default=[], metavar="TASK=SEEDS")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--alignment_seed", type=int, default=0)
    parser.add_argument("--max_alignment_samples", type=int, default=2048)
    parser.add_argument("--max_pair_bank", type=int, default=65536)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_direct_test(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Direct test must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def opposite_modality(modality: str) -> str:
    if modality == "text":
        return "image"
    if modality == "image":
        return "text"
    raise ValueError(f"Unsupported modality: {modality!r}.")


def opposite_operation(operation: str) -> str:
    if operation == "addition":
        return "subtraction"
    if operation == "subtraction":
        return "addition"
    raise ValueError(f"Unsupported operation: {operation!r}.")


def load_task_space(
    args: argparse.Namespace,
    space_cache: dict[tuple[str, int], TaskSpace],
    *,
    task: str,
    seed: int,
    hidden_size: int,
) -> TaskSpace:
    key = (task, seed)
    if key in space_cache:
        return space_cache[key]
    modality, operation = parse_task(task)
    row = first_result_row(args, modality, operation, args.condition, seed)
    basis = load_basis(subspace_path(args, modality, operation, seed), args.layer, hidden_size, args.k)
    space = TaskSpace(task, modality, operation, seed, row, basis)
    space_cache[key] = space
    return space


def transition_key(pair: dict, target: str) -> tuple[str, str]:
    base_answer, source_answer, *_ = target_answers(pair["base"], pair["source"], target)
    return base_answer, source_answer


def group_by_transition(pairs: list[dict], target: str) -> dict[tuple[str, str], list[dict]]:
    grouped = {}
    for pair in pairs:
        grouped.setdefault(transition_key(pair, target), []).append(pair)
    return grouped


def match_transition_pairs(
    args: argparse.Namespace,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    *,
    seed: int,
    max_pairs: int,
) -> tuple[list[dict], list[dict], dict]:
    source_groups = group_by_transition(source_pairs, args.target)
    destination_groups = group_by_transition(destination_pairs, args.target)
    rng = random.Random(seed)
    matched = []
    for transition in sorted(set(source_groups) & set(destination_groups)):
        source_group = list(source_groups[transition])
        destination_group = list(destination_groups[transition])
        rng.shuffle(source_group)
        rng.shuffle(destination_group)
        if max_pairs > 0:
            matched.append((source_group[0], destination_group[0]))
        else:
            matched.extend(zip(source_group, destination_group))
    rng.shuffle(matched)
    if max_pairs > 0:
        matched = matched[:max_pairs]
    source_matched = [source for source, _destination in matched]
    destination_matched = [destination for _source, destination in matched]
    stats = {
        "source_pairs": len(source_pairs),
        "destination_pairs": len(destination_pairs),
        "shared_transitions": len(set(source_groups) & set(destination_groups)),
        "source_transitions": len(source_groups),
        "destination_transitions": len(destination_groups),
        "matched_pairs": len(matched),
        "matching": "one_pair_per_transition" if max_pairs > 0 else "all_zipped_pairs_per_transition",
    }
    return source_matched, destination_matched, stats


def train_pairs_for_task(
    args: argparse.Namespace,
    sample_cache: dict,
    *,
    space: TaskSpace,
) -> list[dict]:
    samples = sample_split(args, cached_task_samples(sample_cache, space.task, space.row), "train")
    pairs, _stats = build_unique_pairs(samples, args.target, args.alignment_seed, args.max_pair_bank)
    return pairs


def test_pairs_for_task(
    args: argparse.Namespace,
    sample_cache: dict,
    *,
    space: TaskSpace,
) -> list[dict]:
    samples = sample_split(args, cached_task_samples(sample_cache, space.task, space.row), "test")
    pairs, _stats = build_unique_pairs(
        samples,
        args.target,
        stable_seed("test_pairs", space.task, space.seed, args.alignment_seed),
        args.max_pair_bank,
    )
    return pairs


def same_result_width(first: int, second: int) -> bool:
    return len(str(int(first))) == len(str(int(second)))


def transition_mean_deltas(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    source_train = sample_split(args, cached_task_samples(sample_cache, source.task, source.row), "train")
    destination_train = sample_split(
        args, cached_task_samples(sample_cache, destination.task, destination.row), "train"
    )
    source_centroids = centroid_coordinates(
        args,
        activation_cache,
        sample_cache,
        task=source.task,
        row=source.row,
        samples=source_train,
        basis=source.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    destination_centroids = centroid_coordinates(
        args,
        activation_cache,
        sample_cache,
        task=destination.task,
        row=destination.row,
        samples=destination_train,
        basis=destination.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    shared_values = sorted(set(source_centroids) & set(destination_centroids))
    transitions = [
        (base, donor)
        for base in shared_values
        for donor in shared_values
        if base != donor and same_result_width(base, donor)
    ]
    available_transitions = len(transitions)
    random.Random(
        stable_seed("transition_means", source.task, source.seed, destination.task, destination.seed, args.alignment_seed)
    ).shuffle(transitions)
    if args.max_alignment_samples > 0:
        transitions = transitions[: args.max_alignment_samples]
    if len(transitions) < 2:
        raise ValueError(
            f"Need at least two shared train transitions for "
            f"{source.task}[{source.seed}] -> {destination.task}[{destination.seed}]."
        )
    x_train = torch.stack([source_centroids[donor] - source_centroids[base] for base, donor in transitions])
    y_train = torch.stack(
        [destination_centroids[donor] - destination_centroids[base] for base, donor in transitions]
    )
    stats = {
        "fit_mode": "transition_mean",
        "source_train_samples": len(source_train),
        "destination_train_samples": len(destination_train),
        "source_result_values": len(source_centroids),
        "destination_result_values": len(destination_centroids),
        "shared_result_values": len(shared_values),
        "available_ordered_transitions": available_transitions,
        "selected_ordered_transitions": len(transitions),
        "transition_weighting": "one_mean_vector_per_ordered_transition",
    }
    return x_train, y_train, stats


def map_path(args: argparse.Namespace, source: TaskSpace, destination: TaskSpace) -> Path:
    return (
        args.output_dir
        / "maps"
        / (
            f"transition_mean_scaled_map_{clean_name(source.task, source.seed)}"
            f"_to_{clean_name(destination.task, destination.seed)}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    )


def fit_scaled_map(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    path = map_path(args, source, destination)
    if path.exists() and not args.force:
        payload = torch.load(path, map_location="cpu")
        if payload.get("fit_stats", {}).get("fit_mode") != "transition_mean":
            print(f"    ignoring stale non-transition-mean map {path}")
        else:
            print(f"    cached map {source.task}[{source.seed}] -> {destination.task}[{destination.seed}]")
            return {
                "q": payload["Q_source_to_destination"].float(),
                "alpha": float(payload["alpha"]),
                "path": path,
                "fit_metrics": payload["fit_metrics"],
                "fit_stats": payload["fit_stats"],
            }

    x_train, y_train, fit_stats = transition_mean_deltas(
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
    q = orthogonal_procrustes(x_train, y_train)
    alpha = scaled_alpha(x_train, y_train, q)
    fit_metrics = coordinate_metrics(x_train, y_train, q, alpha)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "scaled_displacement_transition_mean_map",
            "source_task": source.task,
            "source_seed": source.seed,
            "destination_task": destination.task,
            "destination_seed": destination.seed,
            "Q_source_to_destination": q.cpu(),
            "alpha": alpha,
            "fit_metrics": fit_metrics,
            "fit_stats": fit_stats,
            "layer": args.layer,
            "k": args.k,
            "hook": args.hook,
            "target": args.target,
            "config": jsonable(vars(args)),
        },
        path,
    )
    print(
        f"    fit map {source.task}[{source.seed}] -> {destination.task}[{destination.seed}]: "
        f"alpha={alpha:.4f}; transitions={fit_stats['selected_ordered_transitions']}/"
        f"{fit_stats['available_ordered_transitions']}; "
        f"train cosine={fit_metrics['mean_cosine']}; rmse={fit_metrics['root_mean_squared_error']}"
    )
    return {"q": q, "alpha": alpha, "path": path, "fit_metrics": fit_metrics, "fit_stats": fit_stats}


def path_specs(source_task: str, destination_task: str) -> dict[str, list[tuple[str, str]]]:
    source_modality, source_operation = parse_task(source_task)
    destination_modality, destination_operation = parse_task(destination_task)
    if source_operation == destination_operation or source_modality == destination_modality:
        return {"direct": [(source_task, destination_task)]}
    operation_middle = task_key(source_modality, destination_operation)
    modality_middle = task_key(destination_modality, source_operation)
    return {
        "direct": [(source_task, destination_task)],
        "operation_first": [(source_task, operation_middle), (operation_middle, destination_task)],
        "modality_first": [(source_task, modality_middle), (modality_middle, destination_task)],
    }


def edge_seed(
    edge_task: str,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    source_seed: int,
    destination_seed: int,
) -> int:
    if edge_task == source.task:
        return source_seed
    if edge_task == destination.task:
        return destination_seed
    edge_modality, _edge_operation = parse_task(edge_task)
    return source_seed if edge_modality == source.modality else destination_seed


def compose_maps(first: dict, second: dict) -> tuple[torch.Tensor, float]:
    return first["q"] @ second["q"], float(first["alpha"]) * float(second["alpha"])


def prediction_agreement(predicted: torch.Tensor, reference: torch.Tensor) -> dict:
    if predicted.numel() == 0:
        return {"mean_cosine_to_direct": None, "rmse_to_direct": None}
    cosine = torch.nn.functional.cosine_similarity(predicted, reference, dim=1)
    residual = predicted - reference
    return {
        "mean_cosine_to_direct": float(cosine.mean()),
        "rmse_to_direct": float(residual.square().mean().sqrt()),
    }


def operator_distance(q: torch.Tensor, alpha: float, q_ref: torch.Tensor, alpha_ref: float) -> float:
    operator = float(alpha) * q
    reference = float(alpha_ref) * q_ref
    return float((operator - reference).norm() / reference.norm().clamp_min(1e-12))


def hidden_deltas_from_source_pairs(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    source_pairs: list[dict],
    q: torch.Tensor,
    alpha: float,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    source_coordinate_deltas = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source.task,
        row=source.row,
        pairs=source_pairs,
        basis=source.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return (float(alpha) * (source_coordinate_deltas @ q)) @ destination.basis.T


def load_result_cache(args: argparse.Namespace) -> dict[tuple, dict]:
    path = args.output_dir / "factorized_paths_results.jsonl"
    if args.force or not path.exists():
        return {}
    cache = {}
    for row in load_jsonl(path):
        if row.get("fit_mode") != "transition_mean":
            continue
        cache[
            (
                row["source_task"],
                int(row["source_seed"]),
                row["destination_task"],
                int(row["destination_seed"]),
                row["transport"],
            )
        ] = row
    return cache


def row_key(row: dict) -> tuple:
    return (
        row["source_task"],
        int(row["source_seed"]),
        row["destination_task"],
        int(row["destination_seed"]),
        row["transport"],
    )


def evaluate_transport(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    causal_rows: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    transport: str,
    test_family: str,
    eval_stats: dict,
    q: torch.Tensor,
    alpha: float,
    direct_q: torch.Tensor,
    direct_alpha: float,
    direct_prediction: torch.Tensor,
    component_paths: list[Path],
    model,
    processor,
    tokenizer,
    blocks,
    model_name: str,
) -> dict:
    x_eval = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source.task,
        row=source.row,
        pairs=source_pairs,
        basis=source.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    y_eval = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=destination.task,
        row=destination.row,
        pairs=destination_pairs,
        basis=destination.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    predicted = float(alpha) * (x_eval @ q)
    geometry = coordinate_metrics(x_eval, y_eval, q, alpha)
    agreement = prediction_agreement(predicted, direct_prediction)
    hidden_deltas = predicted @ destination.basis.T

    description = (
        f"{EXPERIMENT} {transport} {source.task}[{source.seed}]"
        f" -> {destination.task}[{destination.seed}]"
    )
    if destination.modality == "text":
        aligned_iia = autoregressive_iia_text_aligned(
            model, tokenizer, blocks, destination_pairs, hidden_deltas, args, description
        )
    else:
        aligned_iia = autoregressive_iia_image_aligned(
            model,
            processor,
            tokenizer,
            blocks,
            destination_pairs,
            hidden_deltas,
            data_root_for(destination.row),
            args,
            description,
        )

    destination_control = control_iia(args, destination.modality, destination.operation, destination.seed, destination.row)
    destination_self = float(destination.row["autoregressive_iia"])
    normalized = normalized_transfer(aligned_iia, destination_control, destination_self)
    causal = causal_rows.get((source.task, source.seed, destination.task, destination.seed), {})
    causal_iia = causal.get("autoregressive_iia")
    causal_normalized = causal.get("destination_normalized_transfer")
    row = {
        "model": model_name,
        "experiment": EXPERIMENT,
        "test_family": test_family,
        "fit_mode": "transition_mean",
        "eval_pair_mode": eval_stats.get("matching"),
        "transport": transport,
        "source_task": source.task,
        "source_modality": source.modality,
        "source_operation": source.operation,
        "source_seed": source.seed,
        "destination_task": destination.task,
        "destination_modality": destination.modality,
        "destination_operation": destination.operation,
        "destination_seed": destination.seed,
        "source_subspace_path": str(subspace_path(args, source.modality, source.operation, source.seed)),
        "destination_results_path": str(
            results_path(args, destination.modality, destination.operation, args.condition, destination.seed)
        ),
        "eval_pair_source": "test_split_unique_pairs_matched_by_ordered_transition",
        "destination_audit_heldout_pairs_path": str(
            heldout_pairs_path(args, destination.modality, destination.operation, destination.seed)
        ),
        "component_map_paths": [str(path) for path in component_paths],
        "layer": args.layer,
        "k": args.k,
        "hook": args.hook,
        "source_position": position_for(args, source.modality),
        "destination_position": position_for(args, destination.modality),
        "target": args.target,
        "n_pairs": len(destination_pairs),
        "eval_stats": eval_stats,
        "alpha": float(alpha),
        "direct_alpha": float(direct_alpha),
        "operator_relative_distance_to_direct": operator_distance(q, alpha, direct_q, direct_alpha),
        "prediction_agreement_with_direct": agreement,
        "alignment_fit_test": geometry,
        "autoregressive_iia": aligned_iia,
        "destination_self_autoregressive_iia": destination_self,
        "control_condition": args.control_condition,
        "destination_control_autoregressive_iia": destination_control,
        "destination_normalized_transfer": normalized,
        "causal_autoregressive_iia": causal_iia,
        "causal_destination_normalized_transfer": causal_normalized,
        "autoregressive_iia_gain_over_causal": gain(aligned_iia, causal_iia),
        "destination_normalized_transfer_gain_over_causal": gain(normalized, causal_normalized),
    }
    print(
        f"    {transport}: alpha={alpha:.4f}; AR IIA={aligned_iia:.4f}; "
        f"normalized={normalized}; cosine-to-direct={agreement['mean_cosine_to_direct']}"
    )
    return row


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def nested(row: dict, path: str):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row.get("test_family"), row["source_task"], row["destination_task"], row["transport"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (test_family, source_task, destination_task, transport), parts in sorted(groups.items()):
        summary.append(
            {
                "test_family": test_family,
                "fit_mode": "transition_mean",
                "source_task": source_task,
                "source_label": label(source_task),
                "destination_task": destination_task,
                "destination_label": label(destination_task),
                "transport": transport,
                "n": len(parts),
                "n_pairs_mean": mean([row["n_pairs"] for row in parts]),
                "eval_shared_transitions_mean": mean(
                    [nested(row, "eval_stats.shared_transitions") for row in parts]
                ),
                "autoregressive_iia_mean": mean([row["autoregressive_iia"] for row in parts]),
                "autoregressive_iia_std": sample_std([row["autoregressive_iia"] for row in parts]),
                "destination_normalized_transfer_mean": mean(
                    [row["destination_normalized_transfer"] for row in parts]
                ),
                "destination_normalized_transfer_std": sample_std(
                    [row["destination_normalized_transfer"] for row in parts]
                ),
                "destination_normalized_transfer_gain_over_causal_mean": mean(
                    [row["destination_normalized_transfer_gain_over_causal"] for row in parts]
                ),
                "operator_relative_distance_to_direct_mean": mean(
                    [row["operator_relative_distance_to_direct"] for row in parts]
                ),
                "prediction_cosine_to_direct_mean": mean(
                    [nested(row, "prediction_agreement_with_direct.mean_cosine_to_direct") for row in parts]
                ),
                "alignment_test_cosine_mean": mean(
                    [nested(row, "alignment_fit_test.mean_cosine") for row in parts]
                ),
                "alignment_test_rmse_mean": mean(
                    [nested(row, "alignment_fit_test.root_mean_squared_error") for row in parts]
                ),
                "alpha_mean": mean([row["alpha"] for row in parts]),
                "alpha_std": sample_std([row["alpha"] for row in parts]),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, rows: list[dict], seed_selection: dict[str, list[int]]) -> None:
    rows.sort(key=row_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, args.output_dir / "factorized_paths_results.jsonl")
    save_jsonl(summarize(rows), args.output_dir / "factorized_paths_summary.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "description": (
                "Direct cross-operation/modality Procrustes transports compared "
                "with operation-first and modality-first composed paths."
            ),
            "transports": TRANSPORTS,
            "direct_tests": args.direct_tests,
            "same_operation_tests": args.same_operation_tests,
            "operation_only_tests": args.operation_only_tests,
            "seed_selection": seed_selection,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def merge_rows(cache: dict[tuple, dict], rows: list[dict]) -> list[dict]:
    merged = dict(cache)
    for row in rows:
        merged[row_key(row)] = row
    return list(merged.values())


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    direct_tests = [parse_direct_test(item) for item in args.direct_tests]
    same_operation_tests = [parse_direct_test(item) for item in args.same_operation_tests]
    operation_only_tests = [parse_direct_test(item) for item in args.operation_only_tests]
    all_tests = [
        ("cross_operation_factorized", source, destination)
        for source, destination in direct_tests
    ] + [
        ("same_operation_transition_mean", source, destination)
        for source, destination in same_operation_tests
    ] + [
        ("operation_only_transition_mean", source, destination)
        for source, destination in operation_only_tests
    ]
    args.tasks = list(dict.fromkeys(args.tasks + [task for _family, source, destination in all_tests for task in (source, destination)]))
    seed_selection = selected_seeds(args)
    causal_rows = load_causal_rows(args.causal_transfer_rows)
    result_cache = load_result_cache(args)

    print("Factorized Procrustes path experiment")
    print("Cross-operation factorization tests:")
    for source_task, destination_task in direct_tests:
        print(f"  {source_task} -> {destination_task}")
        specs = path_specs(source_task, destination_task)
        print(f"    operation_first: {' -> '.join(edge[0] for edge in specs['operation_first'])} -> {destination_task}")
        print(f"    modality_first: {' -> '.join(edge[0] for edge in specs['modality_first'])} -> {destination_task}")
    print("Same-operation transition-mean direct refits:")
    for source_task, destination_task in same_operation_tests:
        print(f"  {source_task} -> {destination_task}")
    print("Operation-only transition-mean direct refits:")
    for source_task, destination_task in operation_only_tests:
        print(f"  {source_task} -> {destination_task}")
    print("Selected seeds:")
    for task in args.tasks:
        print(f"  {task}: {seed_selection[task]}")

    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    activation_cache = {}
    sample_cache = {}
    space_cache = {}
    map_cache = {}
    rows = []

    for test_family, source_task, destination_task in all_tests:
        print(f"\n=== {test_family}: {source_task} -> {destination_task} ===")
        for source_seed in seed_selection[source_task]:
            for destination_seed in seed_selection[destination_task]:
                source = load_task_space(
                    args, space_cache, task=source_task, seed=source_seed, hidden_size=hidden_size
                )
                destination = load_task_space(
                    args, space_cache, task=destination_task, seed=destination_seed, hidden_size=hidden_size
                )
                source_test_pairs = test_pairs_for_task(args, sample_cache, space=source)
                destination_test_pairs = test_pairs_for_task(args, sample_cache, space=destination)
                source_pairs, destination_pairs, eval_stats = match_transition_pairs(
                    args,
                    source_test_pairs,
                    destination_test_pairs,
                    seed=stable_seed("eval", source_task, source_seed, destination_task, destination_seed),
                    max_pairs=args.max_autoregressive_pairs,
                )
                if len(source_pairs) < 1:
                    raise ValueError(
                        f"No matched held-out transitions for "
                        f"{source_task}[{source_seed}] -> {destination_task}[{destination_seed}]. "
                    f"Stats: {eval_stats}"
                    )
                print(
                    f"  seeds {source_seed}->{destination_seed}; "
                    f"matched test pairs={eval_stats['matched_pairs']}; "
                    f"shared test transitions={eval_stats['shared_transitions']}; "
                    f"source test transitions={eval_stats['source_transitions']}; "
                    f"destination test transitions={eval_stats['destination_transitions']}"
                )

                fitted = {}
                specs = path_specs(source_task, destination_task)
                for transport_edges in specs.values():
                    for edge_source_task, edge_destination_task in transport_edges:
                        edge_source_seed = edge_seed(
                            edge_source_task,
                            source=source,
                            destination=destination,
                            source_seed=source_seed,
                            destination_seed=destination_seed,
                        )
                        edge_destination_seed = edge_seed(
                            edge_destination_task,
                            source=source,
                            destination=destination,
                            source_seed=source_seed,
                            destination_seed=destination_seed,
                        )
                        edge_key = (
                            edge_source_task,
                            edge_source_seed,
                            edge_destination_task,
                            edge_destination_seed,
                        )
                        if edge_key in fitted:
                            continue
                        if edge_key not in map_cache:
                            edge_source = load_task_space(
                                args,
                                space_cache,
                                task=edge_source_task,
                                seed=edge_source_seed,
                                hidden_size=hidden_size,
                            )
                            edge_destination = load_task_space(
                                args,
                                space_cache,
                                task=edge_destination_task,
                                seed=edge_destination_seed,
                                hidden_size=hidden_size,
                            )
                            map_cache[edge_key] = fit_scaled_map(
                                args,
                                activation_cache,
                                sample_cache,
                                source=edge_source,
                                destination=edge_destination,
                                model=model,
                                processor=processor,
                                tokenizer=tokenizer,
                                blocks=blocks,
                            )
                        fitted[edge_key] = map_cache[edge_key]

                direct_edge = specs["direct"][0]
                direct_key = (direct_edge[0], source_seed, direct_edge[1], destination_seed)
                direct_map = fitted[direct_key]
                direct_prediction_source = coordinate_deltas_for_pairs(
                    args,
                    activation_cache,
                    sample_cache,
                    task=source.task,
                    row=source.row,
                    pairs=source_pairs,
                    basis=source.basis,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    blocks=blocks,
                )
                direct_prediction = float(direct_map["alpha"]) * (direct_prediction_source @ direct_map["q"])

                transports = {
                    "direct": (direct_map["q"], direct_map["alpha"], [direct_map["path"]]),
                }
                if "operation_first" in specs:
                    op_first_a, op_first_b = specs["operation_first"]
                    op_first_key_a = (op_first_a[0], source_seed, op_first_a[1], source_seed)
                    op_first_key_b = (op_first_b[0], source_seed, op_first_b[1], destination_seed)
                    q_op, alpha_op = compose_maps(fitted[op_first_key_a], fitted[op_first_key_b])
                    transports["operation_first"] = (
                        q_op,
                        alpha_op,
                        [fitted[op_first_key_a]["path"], fitted[op_first_key_b]["path"]],
                    )

                if "modality_first" in specs:
                    mod_first_a, mod_first_b = specs["modality_first"]
                    mod_first_key_a = (mod_first_a[0], source_seed, mod_first_a[1], destination_seed)
                    mod_first_key_b = (mod_first_b[0], destination_seed, mod_first_b[1], destination_seed)
                    q_mod, alpha_mod = compose_maps(fitted[mod_first_key_a], fitted[mod_first_key_b])
                    transports["modality_first"] = (
                        q_mod,
                        alpha_mod,
                        [fitted[mod_first_key_a]["path"], fitted[mod_first_key_b]["path"]],
                    )

                for transport, (q, alpha, component_paths) in transports.items():
                    key = (source.task, source.seed, destination.task, destination.seed, transport)
                    if key in result_cache:
                        rows.append(result_cache[key])
                        print(f"    cached {transport}")
                        continue
                    rows.append(
                        evaluate_transport(
                            args,
                            activation_cache,
                            sample_cache,
                            causal_rows,
                            source=source,
                            destination=destination,
                            source_pairs=source_pairs,
                            destination_pairs=destination_pairs,
                            transport=transport,
                            test_family=test_family,
                            eval_stats=eval_stats,
                            q=q,
                            alpha=alpha,
                            direct_q=direct_map["q"],
                            direct_alpha=direct_map["alpha"],
                            direct_prediction=direct_prediction,
                            component_paths=component_paths,
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            blocks=blocks,
                            model_name=model_name,
                        )
                    )
                    write_outputs(args, merge_rows(result_cache, rows), seed_selection)

    final_rows = merge_rows(result_cache, rows)
    write_outputs(args, final_rows, seed_selection)
    print(f"\nSaved rows: {args.output_dir / 'factorized_paths_results.jsonl'}")
    print(f"Saved summary: {args.output_dir / 'factorized_paths_summary.jsonl'}")
    print(f"Saved maps under: {args.output_dir / 'maps'}")


if __name__ == "__main__":
    main()
