"""Orthogonal synchronization over the four arithmetic DAS spaces.

The input graph is a set of fitted pairwise maps:

    z_destination ~= alpha_source_to_destination z_source Q_source_to_destination

For row-vector coordinates we estimate domain-to-hub orientations A_d such that

    Q_source_to_destination ~= A_source A_destination^T

and synchronize scales separately with

    log alpha_source_to_destination ~= log s_destination - log s_source.

The hub-derived map is then evaluated causally, not only geometrically.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from src.geometry import synchronization as geometry_sync

from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    DEFAULT_TASKS,
    jsonable,
    label,
    load_basis,
    load_jsonl,
    parse_task,
    position_for,
    results_path,
    row_matches,
    save_json,
    save_jsonl,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.factorized_paths.factorized_paths import (
    TaskSpace,
    evaluate_transport,
    hidden_deltas_from_source_pairs,
    match_transition_pairs,
    test_pairs_for_task,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    cached_task_samples,
    clean_name,
    coordinate_deltas_for_pairs,
    load_causal_rows,
    load_model_bundle,
    sample_key,
    stable_seed,
)


EXPERIMENT = "orthogonal_synchronization"
TASKS = [
    "text:addition",
    "image:addition",
    "text:subtraction",
    "image:subtraction",
]
DEFAULT_TESTS = [
    "text:addition->image:addition",
    "image:addition->text:addition",
    "text:subtraction->image:subtraction",
    "image:subtraction->text:subtraction",
    "text:addition->text:subtraction",
    "text:subtraction->text:addition",
    "image:addition->image:subtraction",
    "image:subtraction->image:addition",
    "text:addition->image:subtraction",
    "image:subtraction->text:addition",
    "text:subtraction->image:addition",
    "image:addition->text:subtraction",
]
TRANSPORTS = ["pairwise_direct", "synchronized_hub"]


@dataclass(frozen=True)
class Edge:
    source: str
    destination: str
    q: torch.Tensor
    alpha: float
    weight: float
    path: Path
    fit_metrics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--causal_transfer_rows", type=Path, default=Path("results/final_exps/causal_tranfer/transfer_results.jsonl"))
    parser.add_argument("--factorized_dir", type=Path, default=Path("results/paper/procrustes/factorized_paths"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/synchronization"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--tests", nargs="+", default=DEFAULT_TESTS)
    parser.add_argument("--transports", nargs="+", choices=TRANSPORTS, default=TRANSPORTS)
    parser.add_argument("--heldout_edges", nargs="*", default=[], metavar="SOURCE->DESTINATION")
    parser.add_argument("--drop_reverse_heldouts", action="store_true")
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
    parser.add_argument("--max_pair_bank", type=int, default=65536)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--weight_metric", choices=["mean_cosine", "inverse_rmse", "uniform"], default="mean_cosine")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_edge(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Edge must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def parse_seed_map(items: list[str]) -> dict[str, list[int]]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Seed-map item must be TASK=SEEDS, got {item!r}.")
        task, seeds = item.split("=", 1)
        parsed[task_key(*parse_task(task))] = [int(seed) for seed in re.split(r"[,\s]+", seeds) if seed]
    return parsed


def matching_result_row(args: argparse.Namespace, task: str, seed: int) -> dict | None:
    modality, operation = parse_task(task)
    path = results_path(args, modality, operation, args.condition, seed)
    if not path.exists():
        return None
    for row in load_jsonl(path):
        if row_matches(args, row, modality):
            return row
    return None


def discover_seed_rows(args: argparse.Namespace, task: str) -> list[dict]:
    modality, operation = parse_task(task)
    split_dir = args.audit_root / modality / operation
    if args.target != "result":
        split_dir = split_dir / args.target
    split_dir = split_dir / args.condition / f"split_{args.split_seed}"
    candidates = []
    for seed_dir in sorted(split_dir.glob("seed_*")):
        match = re.match(r"seed_(\d+)$", seed_dir.name)
        if match is None:
            continue
        seed = int(match.group(1))
        row = matching_result_row(args, task, seed)
        checkpoint = subspace_path(args, modality, operation, seed)
        if row is not None and checkpoint.exists():
            candidates.append({"seed": seed, "row": row})
    candidates.sort(key=lambda item: float(item["row"].get("autoregressive_iia", -1.0)), reverse=True)
    return candidates


def selected_matching_seeds(args: argparse.Namespace) -> dict[str, list[int]]:
    explicit = parse_seed_map(args.seed_map)
    selection = {}
    for task in args.tasks:
        modality, _operation = parse_task(task)
        if task in explicit:
            seeds = []
            for seed in explicit[task]:
                if matching_result_row(args, task, seed) is None:
                    raise ValueError(
                        f"{task} seed {seed} has no row matching layer={args.layer}, "
                        f"k={args.k}, position={position_for(args, modality)}, hook={args.hook}."
                    )
                seeds.append(seed)
            selection[task] = seeds
            continue
        candidates = discover_seed_rows(args, task)
        if args.top_n > 0:
            candidates = candidates[: args.top_n]
        else:
            wanted = set(args.seeds)
            candidates = [item for item in candidates if item["seed"] in wanted]
        if not candidates:
            raise FileNotFoundError(
                f"No matching DAS row for {task} under {args.audit_root} "
                f"(layer={args.layer}, k={args.k}, position={position_for(args, modality)}, hook={args.hook})."
            )
        selection[task] = [item["seed"] for item in candidates]
    return selection


def common_sync_seeds(seed_selection: dict[str, list[int]]) -> list[int]:
    seed_sets = [set(seeds) for seeds in seed_selection.values()]
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not common:
        raise ValueError(
            "Synchronization needs the same seed to exist for every task; "
            f"matching seeds by task were {seed_selection}."
        )
    return common


def map_path(args: argparse.Namespace, source: str, destination: str, seed: int) -> Path:
    return (
        args.factorized_dir
        / "maps"
        / (
            f"transition_mean_scaled_map_{clean_name(source, seed)}"
            f"_to_{clean_name(destination, seed)}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    )


def edge_weight(args: argparse.Namespace, metrics: dict) -> float:
    if args.weight_metric == "uniform":
        return 1.0
    if args.weight_metric == "inverse_rmse":
        rmse = metrics.get("root_mean_squared_error")
        return 1.0 / max(float(rmse), 1e-6) if rmse is not None else 1.0
    cosine = metrics.get("mean_cosine")
    if cosine is None:
        return 1.0
    return max(float(cosine), 1e-3)


def load_edge(args: argparse.Namespace, source: str, destination: str, seed: int) -> Edge | None:
    path = map_path(args, source, destination, seed)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metrics = payload.get("fit_metrics", {})
    return Edge(
        source=source,
        destination=destination,
        q=torch.as_tensor(payload["Q_source_to_destination"]).float(),
        alpha=float(payload["alpha"]),
        weight=edge_weight(args, metrics),
        path=path,
        fit_metrics=metrics,
    )


def load_edges(args: argparse.Namespace, tasks: list[str], seed: int) -> list[Edge]:
    heldout = set(parse_edge(edge) for edge in args.heldout_edges)
    if args.drop_reverse_heldouts:
        heldout |= {(destination, source) for source, destination in list(heldout)}
    edges = []
    for source in tasks:
        for destination in tasks:
            if source == destination or (source, destination) in heldout:
                continue
            edge = load_edge(args, source, destination, seed)
            if edge is not None:
                edges.append(edge)
    return edges


def connected_components(tasks: list[str], edges: list[Edge]) -> list[list[str]]:
    graph = {task: set() for task in tasks}
    for edge in edges:
        graph[edge.source].add(edge.destination)
        graph[edge.destination].add(edge.source)
    seen = set()
    components = []
    for task in tasks:
        if task in seen:
            continue
        stack = [task]
        part = []
        seen.add(task)
        while stack:
            item = stack.pop()
            part.append(item)
            for nxt in graph[item]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(part))
    return components


def project_orthogonal(matrix: torch.Tensor) -> torch.Tensor:
    return geometry_sync.project_orthogonal(matrix)


def synchronize_rotations(tasks: list[str], edges: list[Edge], k: int) -> tuple[dict[str, torch.Tensor], dict]:
    return geometry_sync.synchronize_rotations(tasks, edges, k)


def synchronize_scales(tasks: list[str], edges: list[Edge]) -> tuple[dict[str, float], dict]:
    return geometry_sync.synchronize_scales(tasks, edges)


def rotation_residuals(tasks: list[str], edges: list[Edge], orientations: dict[str, torch.Tensor]) -> list[dict]:
    return geometry_sync.rotation_residuals(tasks, edges, orientations)


def hub_map(source: str, destination: str, orientations: dict[str, torch.Tensor], log_scales: dict[str, float]) -> tuple[torch.Tensor, float]:
    return geometry_sync.hub_map(source, destination, orientations, log_scales)


def make_spaces(args: argparse.Namespace, tasks: list[str], seed: int, hidden_size: int) -> dict[str, TaskSpace]:
    spaces = {}
    for task in tasks:
        modality, operation = parse_task(task)
        row = matching_result_row(args, task, seed)
        if row is None:
            raise FileNotFoundError(f"No matching DAS result row for {task} seed {seed}.")
        basis = load_basis(subspace_path(args, modality, operation, seed), args.layer, hidden_size, args.k)
        spaces[task] = TaskSpace(task, modality, operation, seed, row, basis)
    return spaces


def paired_test_batches(args: argparse.Namespace, sample_cache: dict, source: TaskSpace, destination: TaskSpace) -> tuple[list[dict], list[dict], dict]:
    source_pairs = test_pairs_for_task(args, sample_cache, space=source)
    destination_pairs = test_pairs_for_task(args, sample_cache, space=destination)
    return match_transition_pairs(
        args,
        source_pairs,
        destination_pairs,
        seed=stable_seed("synchronization_eval", source.task, destination.task, source.seed, args.alignment_seed),
        max_pairs=args.max_autoregressive_pairs,
    )


def prediction_for_pairs(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source: TaskSpace,
    source_pairs: list[dict],
    q: torch.Tensor,
    alpha: float,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
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
    return float(alpha) * (x_eval @ q)


def sync_payload_path(args: argparse.Namespace, seed: int) -> Path:
    suffix = "full_graph" if not args.heldout_edges else "heldout_" + "__".join(
        edge.replace(":", "_").replace("->", "_to_") for edge in args.heldout_edges
    )
    return args.output_dir / "maps" / f"synchronized_hub_seed{seed}_{suffix}_layer{args.layer}_k{args.k}.pt"


def save_sync_payload(
    args: argparse.Namespace,
    seed: int,
    tasks: list[str],
    edges: list[Edge],
    orientations: dict[str, torch.Tensor],
    log_scales: dict[str, float],
    rotation_stats: dict,
    scale_stats: dict,
) -> Path:
    path = sync_payload_path(args, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "orthogonal_synchronization_hub",
            "seed": seed,
            "tasks": tasks,
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


def load_cache(args: argparse.Namespace) -> dict[tuple, dict]:
    path = args.output_dir / "synchronization_results.jsonl"
    if args.force or not path.exists():
        return {}
    return {row_key(row): row for row in load_jsonl(path)}


def row_key(row: dict) -> tuple:
    return (
        int(row["sync_seed"]),
        row["source_task"],
        row["destination_task"],
        row["transport"],
        row.get("heldout_edges_key", ""),
    )


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def relative_operator_distance(q: torch.Tensor, alpha: float, direct_q: torch.Tensor, direct_alpha: float) -> float:
    direct = float(direct_alpha) * direct_q
    predicted = float(alpha) * q
    return float((predicted - direct).norm() / max(direct.norm().item(), 1e-12))


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["source_task"], row["destination_task"], row["transport"], row["heldout_edges_key"]), []).append(row)
    summary = []
    for (source, destination, transport, heldout_key), parts in sorted(groups.items()):
        summary.append(
            {
                "source_task": source,
                "source_label": label(source),
                "destination_task": destination,
                "destination_label": label(destination),
                "transport": transport,
                "heldout_edges_key": heldout_key,
                "n": len(parts),
                "autoregressive_iia_mean": mean([row["autoregressive_iia"] for row in parts]),
                "autoregressive_iia_std": sample_std([row["autoregressive_iia"] for row in parts]),
                "destination_normalized_transfer_mean": mean([row["destination_normalized_transfer"] for row in parts]),
                "destination_normalized_transfer_std": sample_std([row["destination_normalized_transfer"] for row in parts]),
                "operator_relative_distance_to_direct_mean": mean([row["operator_relative_distance_to_direct"] for row in parts]),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, rows: list[dict], sync_rows: list[dict], seed_selection: dict[str, list[int]]) -> None:
    rows.sort(key=row_key)
    save_jsonl(rows, args.output_dir / "synchronization_results.jsonl")
    save_jsonl(summarize(rows), args.output_dir / "synchronization_summary.jsonl")
    save_jsonl(sync_rows, args.output_dir / "synchronization_fit_summary.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "transports": args.transports,
            "seed_selection": seed_selection,
            "heldout_edges": args.heldout_edges,
            "drop_reverse_heldouts": args.drop_reverse_heldouts,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def merge_rows(cache: dict[tuple, dict], rows: list[dict]) -> list[dict]:
    merged = dict(cache)
    for row in rows:
        merged[row_key(row)] = row
    return list(merged.values())


def evaluate_one(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    causal_rows: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    eval_stats: dict,
    transport: str,
    q: torch.Tensor,
    alpha: float,
    direct_q: torch.Tensor,
    direct_alpha: float,
    direct_prediction: torch.Tensor,
    component_paths: list[Path],
    sync_seed: int,
    sync_path: Path,
    model,
    processor,
    tokenizer,
    blocks,
    model_name: str,
) -> dict:
    row = evaluate_transport(
        args,
        activation_cache,
        sample_cache,
        causal_rows,
        source=source,
        destination=destination,
        source_pairs=source_pairs,
        destination_pairs=destination_pairs,
        transport=transport,
        test_family=EXPERIMENT,
        eval_stats=eval_stats,
        q=q,
        alpha=alpha,
        direct_q=direct_q,
        direct_alpha=direct_alpha,
        direct_prediction=direct_prediction,
        component_paths=component_paths,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        model_name=model_name,
    )
    row["experiment"] = EXPERIMENT
    row["sync_seed"] = sync_seed
    row["synchronization_path"] = str(sync_path)
    row["heldout_edges"] = args.heldout_edges
    row["heldout_edges_key"] = "none" if not args.heldout_edges else "|".join(args.heldout_edges)
    row["operator_relative_distance_to_direct"] = relative_operator_distance(q, alpha, direct_q, direct_alpha)
    return row


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    tests = [parse_edge(test) for test in args.tests]
    args.tasks = list(dict.fromkeys(args.tasks + [task for pair in tests for task in pair]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_selection = selected_matching_seeds(args)
    seeds = common_sync_seeds(seed_selection)
    cache = load_cache(args)
    rows = []
    sync_rows = []
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    causal_rows = load_causal_rows(args.causal_transfer_rows)
    activation_cache = {}
    sample_cache = {}

    print("Orthogonal synchronization")
    print(f"  tasks={args.tasks}")
    print(f"  tests={args.tests}")
    print(f"  heldout_edges={args.heldout_edges}")
    for seed in seeds:
        print(f"\nSeed {seed}")
        spaces = make_spaces(args, args.tasks, seed, hidden_size)
        edges = load_edges(args, args.tasks, seed)
        components = connected_components(args.tasks, edges)
        if any(len(component) < len(args.tasks) for component in components):
            print(f"  warning: synchronization graph has components={components}")
        if not edges:
            print("  no synchronization edges found; skipping")
            continue
        orientations, rotation_stats = synchronize_rotations(args.tasks, edges, args.k)
        log_scales, scale_stats = synchronize_scales(args.tasks, edges)
        sync_path = save_sync_payload(args, seed, args.tasks, edges, orientations, log_scales, rotation_stats, scale_stats)
        sync_rows.append(
            {
                "sync_seed": seed,
                "synchronization_path": str(sync_path),
                "n_edges": len(edges),
                "components": components,
                "heldout_edges": args.heldout_edges,
                "heldout_edges_key": "none" if not args.heldout_edges else "|".join(args.heldout_edges),
                "rotation_residual_mean": rotation_stats["rotation_residual_mean"],
                "rotation_residual_std": rotation_stats["rotation_residual_std"],
                "scale_residual_mean_abs": scale_stats["scale_residual_mean_abs"],
                "scale_residual_std_abs": scale_stats["scale_residual_std_abs"],
            }
        )
        print(
            f"  synchronized {len(edges)} edges; rotation residual={rotation_stats['rotation_residual_mean']}; "
            f"scale residual={scale_stats['scale_residual_mean_abs']}"
        )

        for source_task, destination_task in tests:
            direct_edge = load_edge(args, source_task, destination_task, seed)
            if direct_edge is None:
                print(f"  missing direct edge {source_task}[{seed}] -> {destination_task}[{seed}]; skipping causal comparison")
                continue
            source = spaces[source_task]
            destination = spaces[destination_task]
            source_pairs, destination_pairs, eval_stats = paired_test_batches(args, sample_cache, source, destination)
            direct_prediction = prediction_for_pairs(
                args,
                activation_cache,
                sample_cache,
                source=source,
                source_pairs=source_pairs,
                q=direct_edge.q,
                alpha=direct_edge.alpha,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
            )
            for transport in args.transports:
                key = (
                    seed,
                    source_task,
                    destination_task,
                    transport,
                    "none" if not args.heldout_edges else "|".join(args.heldout_edges),
                )
                if key in cache:
                    rows.append(cache[key])
                    print(f"  cached {transport} {source_task}[{seed}] -> {destination_task}[{seed}]")
                    continue
                if transport == "pairwise_direct":
                    q, alpha = direct_edge.q, direct_edge.alpha
                    component_paths = [direct_edge.path]
                else:
                    q, alpha = hub_map(source_task, destination_task, orientations, log_scales)
                    component_paths = [sync_path]
                rows.append(
                    evaluate_one(
                        args,
                        activation_cache,
                        sample_cache,
                        causal_rows,
                        source=source,
                        destination=destination,
                        source_pairs=source_pairs,
                        destination_pairs=destination_pairs,
                        eval_stats=eval_stats,
                        transport=transport,
                        q=q,
                        alpha=alpha,
                        direct_q=direct_edge.q,
                        direct_alpha=direct_edge.alpha,
                        direct_prediction=direct_prediction,
                        component_paths=component_paths,
                        sync_seed=seed,
                        sync_path=sync_path,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        model_name=model_name,
                    )
                )
                write_outputs(args, merge_rows(cache, rows), sync_rows, seed_selection)
    write_outputs(args, merge_rows(cache, rows), sync_rows, seed_selection)
    print(f"\nSaved rows: {args.output_dir / 'synchronization_results.jsonl'}")
    print(f"Saved summary: {args.output_dir / 'synchronization_summary.jsonl'}")
    print(f"Saved sync fit summary: {args.output_dir / 'synchronization_fit_summary.jsonl'}")


if __name__ == "__main__":
    main()
