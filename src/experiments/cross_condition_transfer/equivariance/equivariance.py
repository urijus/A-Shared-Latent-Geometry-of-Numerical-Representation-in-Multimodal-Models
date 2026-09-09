"""Fit a +1 arithmetic operator in the synchronized shared DAS space.

The experiment freezes:

1. the existing DAS subspaces;
2. the synchronized domain-to-hub orientations and scales.

It fits G_{+1} using text:addition centroids only, then evaluates the frozen
operator geometrically and causally in all four domains.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.interventions.das import format_prompt, resolve_position, target_answers
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    jsonable,
    label,
    load_basis,
    load_jsonl,
    normalized_transfer,
    parse_task,
    position_for,
    results_path,
    row_matches,
    save_json,
    save_jsonl,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    activations_for_ids,
    cached_task_samples,
    coordinate_metrics,
    data_root_for,
    load_model_bundle,
    orthogonal_procrustes,
    patched_forward_with_deltas,
    random_orthogonal,
    sample_key,
    sample_split,
    scaled_alpha,
    stable_seed,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template


EXPERIMENT = "shared_space_plus_one_equivariance"
TASKS = [
    "text:addition",
    "image:addition",
    "text:subtraction",
    "image:subtraction",
]
FIT_TASK = "text:addition"
BASE_MODELS = [
    "orthogonal",
    "scaled_orthogonal",
    "ridge",
    "affine_ridge",
]
CONTROL_MODELS = [
    "oracle_target_state",
    "identity",
    "random_orthogonal",
    "shuffled_scaled_orthogonal",
]


@dataclass(frozen=True)
class TaskSpace:
    task: str
    modality: str
    operation: str
    seed: int
    row: dict
    basis: torch.Tensor
    orientation: torch.Tensor
    scale: float


@dataclass
class PlusOneMap:
    name: str
    kind: str
    matrix: torch.Tensor | None = None
    alpha: float = 1.0
    bias: torch.Tensor | None = None
    lambda_: float | None = None
    fit_task: str = FIT_TASK
    fit_seed: int | None = None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--sync_dir", type=Path, default=Path("results/final_exps/synchronization"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/equivariance"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--fit_task", default=FIT_TASK)
    parser.add_argument("--models", nargs="+", default=BASE_MODELS + CONTROL_MODELS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
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
    parser.add_argument("--value_test_fraction", type=float, default=0.2)
    parser.add_argument("--value_validation_fraction", type=float, default=0.2)
    parser.add_argument("--value_split_seed", type=int, default=0)
    parser.add_argument("--ridge_lambdas", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
    parser.add_argument("--geometry_steps", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--causal_steps", type=int, nargs="+", default=[1])
    parser.add_argument("--max_causal_pairs_per_domain", type=int, default=32)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--include_domain_specific_upper", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def common_experiment_seeds(seed_selection: dict[str, list[int]]) -> list[int]:
    seed_sets = [set(seeds) for seeds in seed_selection.values()]
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not common:
        raise ValueError(
            "Equivariance needs the same seed to exist for every task; "
            f"matching seeds by task were {seed_selection}."
        )
    return common


def sync_map_path(args: argparse.Namespace, seed: int) -> Path:
    return args.sync_dir / "maps" / f"synchronized_hub_seed{seed}_full_graph_layer{args.layer}_k{args.k}.pt"


def load_sync(args: argparse.Namespace, seed: int) -> dict:
    path = sync_map_path(args, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing synchronized hub {path}. Run experiments/04_global_geometry/synchronize_frames.py first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": path,
        "orientations": {
            task_key(*parse_task(task)): torch.as_tensor(value).float()
            for task, value in payload["orientations_domain_to_hub"].items()
        },
        "scales": {task_key(*parse_task(task)): float(value) for task, value in payload["scales"].items()},
        "payload": payload,
    }


def make_spaces(args: argparse.Namespace, tasks: list[str], seed: int, hidden_size: int, sync: dict) -> dict[str, TaskSpace]:
    spaces = {}
    for task in tasks:
        modality, operation = parse_task(task)
        row = matching_result_row(args, task, seed)
        if row is None:
            raise FileNotFoundError(f"No matching DAS result row for {task} seed {seed}.")
        spaces[task] = TaskSpace(
            task=task,
            modality=modality,
            operation=operation,
            seed=seed,
            row=row,
            basis=load_basis(subspace_path(args, modality, operation, seed), args.layer, hidden_size, args.k),
            orientation=sync["orientations"][task],
            scale=sync["scales"][task],
        )
    return spaces


def result_value(sample: dict) -> int:
    return int(sample["result"])


def grouped_by_result(samples: list[dict]) -> dict[int, list[dict]]:
    grouped = {}
    for sample in samples:
        grouped.setdefault(result_value(sample), []).append(sample)
    return grouped


def shared_from_domain(space: TaskSpace, z: torch.Tensor) -> torch.Tensor:
    return (z @ space.orientation) / float(space.scale)


def domain_from_shared(space: TaskSpace, h: torch.Tensor) -> torch.Tensor:
    return float(space.scale) * (h @ space.orientation.T)


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


def shared_centroids(
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
) -> dict[int, torch.Tensor]:
    grouped = grouped_by_result(samples)
    ids = []
    for parts in grouped.values():
        ids.extend(sample_key(sample) for sample in parts)
    activations = activations_for_ids(
        args,
        activation_cache,
        sample_cache,
        task=space.task,
        row=space.row,
        ids=list(dict.fromkeys(ids)),
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    by_id = {item: activations[index] for index, item in enumerate(list(dict.fromkeys(ids)))}
    centroids = {}
    for value, parts in grouped.items():
        z = torch.stack([by_id[sample_key(sample)] @ space.basis for sample in parts])
        centroids[value] = shared_from_domain(space, z).mean(dim=0)
    return centroids


def available_plus_edges(centroids: dict[int, torch.Tensor], step: int = 1) -> list[int]:
    values = set(centroids)
    return sorted(value for value in values if value + step in values)


def value_split(args: argparse.Namespace, fit_centroids: dict[int, torch.Tensor]) -> dict:
    starts = available_plus_edges(fit_centroids, 1)
    rng = random.Random(stable_seed("plus_one_values", args.value_split_seed, args.split_seed))
    shuffled = list(starts)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * args.value_test_fraction))
    test_starts = sorted(shuffled[:n_test])
    test_vertices = set(test_starts) | {value + 1 for value in test_starts}
    remaining = [value for value in starts if value not in test_vertices and value + 1 not in test_vertices]
    rng.shuffle(remaining)
    n_val = max(1, round(len(remaining) * args.value_validation_fraction)) if remaining else 0
    validation_starts = sorted(remaining[:n_val])
    validation_vertices = set(validation_starts) | {value + 1 for value in validation_starts}
    train_starts = sorted(
        value
        for value in remaining[n_val:]
        if value not in validation_vertices and value + 1 not in validation_vertices
    )
    if len(train_starts) < 2 or len(validation_starts) < 1 or len(test_starts) < 1:
        raise ValueError(
            f"Insufficient value-disjoint +1 edges: train={len(train_starts)}, "
            f"validation={len(validation_starts)}, test={len(test_starts)}."
        )
    return {
        "train_starts": train_starts,
        "validation_starts": validation_starts,
        "test_starts": test_starts,
        "test_vertices": sorted(test_vertices),
        "n_available_edges": len(starts),
    }


def edge_tensors(centroids: dict[int, torch.Tensor], starts: list[int], step: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    usable = [value for value in starts if value in centroids and value + step in centroids]
    if not usable:
        k = next(iter(centroids.values())).shape[0]
        return torch.empty(0, k), torch.empty(0, k)
    return torch.stack([centroids[value] for value in usable]), torch.stack([centroids[value + step] for value in usable])


def fit_linear_ridge(x: torch.Tensor, y: torch.Tensor, lambda_: float) -> torch.Tensor:
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    return torch.linalg.solve(x.T @ x + float(lambda_) * eye, x.T @ y)


def fit_affine_ridge(x: torch.Tensor, y: torch.Tensor, lambda_: float) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype)
    x_aug = torch.cat([x, ones], dim=1)
    penalty = torch.diag(torch.tensor([float(lambda_)] * x.shape[1] + [0.0], dtype=x.dtype))
    params = torch.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y)
    return params[:-1], params[-1]


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("inf")
    return float((x - y).square().mean())


def choose_ridge_lambda(
    lambdas: list[float],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    affine: bool,
) -> tuple[float, torch.Tensor, torch.Tensor | None, list[dict]]:
    trials = []
    best = None
    for lambda_ in lambdas:
        if affine:
            matrix, bias = fit_affine_ridge(x_train, y_train, lambda_)
            pred = x_val @ matrix + bias
        else:
            matrix, bias = fit_linear_ridge(x_train, y_train, lambda_), None
            pred = x_val @ matrix
        score = mse(pred, y_val)
        trials.append({"lambda": lambda_, "validation_mse": score})
        if best is None or score < best[0]:
            best = (score, lambda_, matrix, bias)
    assert best is not None
    return best[1], best[2], best[3], trials


def apply_plus_map(h: torch.Tensor, plus_map: PlusOneMap, steps: int = 1) -> torch.Tensor:
    out = h
    for _ in range(steps):
        if plus_map.kind == "oracle":
            raise ValueError("Oracle maps are handled outside apply_plus_map.")
        if plus_map.kind == "identity":
            out = out
        elif plus_map.kind in {"orthogonal", "random_orthogonal"}:
            out = out @ plus_map.matrix
        elif plus_map.kind == "scaled_orthogonal":
            out = float(plus_map.alpha) * (out @ plus_map.matrix)
        elif plus_map.kind == "ridge":
            out = out @ plus_map.matrix
        elif plus_map.kind == "affine_ridge":
            out = out @ plus_map.matrix + plus_map.bias
        else:
            raise ValueError(f"Unknown +1 map kind: {plus_map.kind}")
    return out


def fit_plus_maps(
    args: argparse.Namespace,
    fit_centroids: dict[int, torch.Tensor],
    split: dict,
    *,
    seed: int,
    name_prefix: str = "",
    fit_task: str = FIT_TASK,
) -> tuple[dict[str, PlusOneMap], dict]:
    x_train, y_train = edge_tensors(fit_centroids, split["train_starts"])
    x_val, y_val = edge_tensors(fit_centroids, split["validation_starts"])
    q = orthogonal_procrustes(x_train, y_train)
    alpha = scaled_alpha(x_train, y_train, q)
    ridge_lambda, ridge_w, _ridge_b, ridge_trials = choose_ridge_lambda(
        args.ridge_lambdas, x_train, y_train, x_val, y_val, affine=False
    )
    affine_lambda, affine_w, affine_b, affine_trials = choose_ridge_lambda(
        args.ridge_lambdas, x_train, y_train, x_val, y_val, affine=True
    )
    rng = random.Random(stable_seed("shuffled_plus_one", seed, fit_task, args.value_split_seed))
    shuffled_indices = list(range(y_train.shape[0]))
    rng.shuffle(shuffled_indices)
    y_shuffled = y_train[shuffled_indices]
    q_shuffled = orthogonal_procrustes(x_train, y_shuffled)
    alpha_shuffled = scaled_alpha(x_train, y_shuffled, q_shuffled)
    random_q = random_orthogonal(args.k, stable_seed("random_plus_one", seed, fit_task))

    prefix = f"{name_prefix}_" if name_prefix else ""
    maps = {
        f"{prefix}orthogonal": PlusOneMap(f"{prefix}orthogonal", "orthogonal", matrix=q, fit_seed=seed, fit_task=fit_task),
        f"{prefix}scaled_orthogonal": PlusOneMap(
            f"{prefix}scaled_orthogonal", "scaled_orthogonal", matrix=q, alpha=alpha, fit_seed=seed, fit_task=fit_task
        ),
        f"{prefix}ridge": PlusOneMap(
            f"{prefix}ridge", "ridge", matrix=ridge_w, lambda_=ridge_lambda, fit_seed=seed, fit_task=fit_task
        ),
        f"{prefix}affine_ridge": PlusOneMap(
            f"{prefix}affine_ridge", "affine_ridge", matrix=affine_w, bias=affine_b, lambda_=affine_lambda, fit_seed=seed, fit_task=fit_task
        ),
        f"{prefix}identity": PlusOneMap(f"{prefix}identity", "identity", fit_seed=seed, fit_task=fit_task),
        f"{prefix}random_orthogonal": PlusOneMap(
            f"{prefix}random_orthogonal", "random_orthogonal", matrix=random_q, fit_seed=seed, fit_task=fit_task
        ),
        f"{prefix}shuffled_scaled_orthogonal": PlusOneMap(
            f"{prefix}shuffled_scaled_orthogonal",
            "scaled_orthogonal",
            matrix=q_shuffled,
            alpha=alpha_shuffled,
            fit_seed=seed,
            fit_task=fit_task,
            notes="fit with shuffled y->y+1 correspondences",
        ),
    }
    fit_stats = {
        "fit_task": fit_task,
        "train_edges": len(split["train_starts"]),
        "validation_edges": len(split["validation_starts"]),
        "test_edges": len(split["test_starts"]),
        "orthogonal_train": coordinate_metrics(x_train, y_train, q, 1.0),
        "scaled_orthogonal_train": coordinate_metrics(x_train, y_train, q, alpha),
        "ridge_lambda": ridge_lambda,
        "ridge_validation_trials": ridge_trials,
        "affine_ridge_lambda": affine_lambda,
        "affine_ridge_validation_trials": affine_trials,
        "shuffled_scaled_orthogonal_train": coordinate_metrics(x_train, y_shuffled, q_shuffled, alpha_shuffled),
    }
    return maps, fit_stats


def nearest_centroid_accuracy(predicted: torch.Tensor, expected_values: list[int], centroids: dict[int, torch.Tensor]) -> float | None:
    if predicted.numel() == 0:
        return None
    values = sorted(centroids)
    matrix = torch.stack([centroids[value] for value in values])
    distances = torch.cdist(predicted, matrix)
    nearest = [values[index] for index in distances.argmin(dim=1).tolist()]
    return sum(int(a == b) for a, b in zip(nearest, expected_values)) / len(expected_values)


def normalized_mse(predicted: torch.Tensor, target: torch.Tensor) -> float | None:
    if predicted.numel() == 0:
        return None
    denom = target.var(dim=0, unbiased=False).mean().clamp_min(1e-12)
    return float((predicted - target).square().mean() / denom)


def geometry_metrics(predicted: torch.Tensor, target: torch.Tensor, expected_values: list[int], centroids: dict[int, torch.Tensor]) -> dict:
    if predicted.numel() == 0:
        return {
            "n_edges": 0,
            "mean_cosine": None,
            "normalized_mse": None,
            "nearest_result_centroid_accuracy": None,
        }
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=1)
    return {
        "n_edges": int(predicted.shape[0]),
        "mean_cosine": float(cosine.mean()),
        "normalized_mse": normalized_mse(predicted, target),
        "nearest_result_centroid_accuracy": nearest_centroid_accuracy(predicted, expected_values, centroids),
    }


def carry_flag(value: int, step: int) -> str:
    return "carry" if len(str(int(value))) != len(str(int(value + step))) else "non_carry"


def evaluate_geometry(
    centroids_by_task: dict[str, dict[int, torch.Tensor]],
    plus_map: PlusOneMap,
    *,
    task: str,
    starts: list[int],
    step: int,
) -> list[dict]:
    centroids = centroids_by_task[task]
    usable = [value for value in starts if value in centroids and value + step in centroids]
    rows = []
    for subset_name, subset in [
        ("all", usable),
        ("carry", [value for value in usable if carry_flag(value, step) == "carry"]),
        ("non_carry", [value for value in usable if carry_flag(value, step) == "non_carry"]),
    ]:
        x, y = edge_tensors(centroids, subset, step)
        predicted = y if plus_map.kind == "oracle" else apply_plus_map(x, plus_map, step)
        rows.append(
            {
                "task": task,
                "task_label": label(task),
                "model": plus_map.name,
                "model_kind": plus_map.kind,
                "step": step,
                "subset": subset_name,
                "starts": subset,
                **geometry_metrics(predicted, y, [value + step for value in subset], centroids),
            }
        )
    return rows


def plus_pairs_for_task(
    args: argparse.Namespace,
    sample_cache: dict,
    *,
    space: TaskSpace,
    starts: list[int],
    step: int,
) -> list[dict]:
    samples = sample_split(args, cached_task_samples(sample_cache, space.task, space.row), "test")
    grouped = grouped_by_result(samples)
    rng = random.Random(stable_seed("plus_pairs", space.task, space.seed, step, args.value_split_seed))
    pairs = []
    for start in starts:
        if start not in grouped or start + step not in grouped:
            continue
        bases = list(grouped[start])
        sources = list(grouped[start + step])
        rng.shuffle(bases)
        rng.shuffle(sources)
        for index, base in enumerate(bases):
            pairs.append(
                {
                    "pair_id": len(pairs),
                    "base": base,
                    "source": sources[index % len(sources)],
                    "step": step,
                    "carry": carry_flag(start, step),
                }
            )
    rng.shuffle(pairs)
    if args.max_causal_pairs_per_domain > 0:
        pairs = pairs[: args.max_causal_pairs_per_domain]
    for index, pair in enumerate(pairs):
        pair["pair_id"] = index
    return pairs


def hidden_deltas_for_map(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    space: TaskSpace,
    pairs: list[dict],
    plus_map: PlusOneMap,
    step: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    ids = []
    for pair in pairs:
        ids.append(sample_key(pair["base"]))
        ids.append(sample_key(pair["source"]))
    ids = list(dict.fromkeys(ids))
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
    by_id = {item: activations[index] for index, item in enumerate(ids)}
    base_z = torch.stack([by_id[sample_key(pair["base"])] @ space.basis for pair in pairs])
    source_z = torch.stack([by_id[sample_key(pair["source"])] @ space.basis for pair in pairs])
    if plus_map.kind == "oracle":
        target_z = source_z
    else:
        base_h = shared_from_domain(space, base_z)
        target_h = apply_plus_map(base_h, plus_map, step)
        target_z = domain_from_shared(space, target_h)
    return (target_z - base_z) @ space.basis.T


def number_from_ids(tokenizer, ids: list[int]) -> str | None:
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    match = re.match(r"-?\d+", text)
    return match.group() if match is not None else None


def answer_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Answer {answer!r} produced no tokens.")
    return [int(item) for item in ids]


def topk_contains_target(logits: torch.Tensor, target_id: int, k: int) -> bool:
    top = torch.topk(logits.float(), k=min(k, logits.shape[-1])).indices.tolist()
    return int(target_id) in top


@torch.no_grad()
def text_delta_pair_metrics(model, tokenizer, blocks, pairs: list[dict], hidden_deltas: torch.Tensor, args: argparse.Namespace, description: str) -> list[dict]:
    rows = []
    use_chat = args.use_chat_template or uses_chat_template(args.model)
    for index, (pair, delta) in enumerate(tqdm(list(zip(pairs, hidden_deltas)), desc=description)):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        target_ids = answer_token_ids(tokenizer, expected)
        base_prompt = format_prompt(tokenizer, base, use_chat)
        base_position = resolve_position(tokenizer, base_prompt, args.text_position)
        base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            outputs = patched_forward_with_deltas(
                model,
                {"input_ids": base_ids[None]},
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            next_id = outputs.logits[0, base_ids.shape[0] - 1].argmax().reshape(1)
            generated.append(int(next_id.item()))
            base_ids = torch.cat([base_ids, next_id])
            if next_id.item() == tokenizer.eos_token_id:
                break
        prediction = number_from_ids(tokenizer, generated)

        base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        topk_hits = []
        for target_id in target_ids:
            outputs = patched_forward_with_deltas(
                model,
                {"input_ids": base_ids[None]},
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            topk_hits.append(topk_contains_target(outputs.logits[0, base_ids.shape[0] - 1], target_id, args.top_k))
            base_ids = torch.cat([base_ids, torch.tensor([target_id], device=model.device)])
        rows.append(
            {
                "pair_id": pair.get("pair_id", index),
                "expected": expected,
                "prediction": prediction,
                "exact_match": prediction == expected,
                "target_tokens_all_topk": len(topk_hits) == len(target_ids) and all(topk_hits),
                "target_tokens_topk_fraction": sum(topk_hits) / len(target_ids),
                "target_token_count": len(target_ids),
            }
        )
    return rows


@torch.no_grad()
def image_delta_pair_metrics(model, processor, tokenizer, blocks, pairs: list[dict], hidden_deltas: torch.Tensor, data_root: Path, args: argparse.Namespace, description: str) -> list[dict]:
    rows = []
    for index, (pair, delta) in enumerate(tqdm(list(zip(pairs, hidden_deltas)), desc=description)):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        target_ids = answer_token_ids(tokenizer, expected)
        prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(base, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.image_position)[0]
        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            length = int(inputs["attention_mask"][0].sum())
            outputs = patched_forward_with_deltas(
                model,
                inputs,
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            next_id = outputs.logits[0, length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = next_id.item()
            inputs["attention_mask"][0, length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        prediction = number_from_ids(tokenizer, generated)

        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        topk_hits = []
        for target_id in target_ids:
            length = int(inputs["attention_mask"][0].sum())
            outputs = patched_forward_with_deltas(
                model,
                inputs,
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            topk_hits.append(topk_contains_target(outputs.logits[0, length - 1], target_id, args.top_k))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = int(target_id)
            inputs["attention_mask"][0, length] = 1
        rows.append(
            {
                "pair_id": pair.get("pair_id", index),
                "expected": expected,
                "prediction": prediction,
                "exact_match": prediction == expected,
                "target_tokens_all_topk": len(topk_hits) == len(target_ids) and all(topk_hits),
                "target_tokens_topk_fraction": sum(topk_hits) / len(target_ids),
                "target_token_count": len(target_ids),
            }
        )
    return rows


def summarize_pair_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "autoregressive_exact_match": 0.0,
            "target_tokens_all_topk": 0.0,
            "target_tokens_topk_fraction": 0.0,
        }
    return {
        "autoregressive_exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "target_tokens_all_topk": sum(row["target_tokens_all_topk"] for row in rows) / len(rows),
        "target_tokens_topk_fraction": sum(row["target_tokens_topk_fraction"] for row in rows) / len(rows),
    }


def causal_metrics_for_deltas(
    args: argparse.Namespace,
    *,
    space: TaskSpace,
    pairs: list[dict],
    hidden_deltas: torch.Tensor,
    description: str,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    if space.modality == "text":
        rows = text_delta_pair_metrics(model, tokenizer, blocks, pairs, hidden_deltas, args, description)
    else:
        rows = image_delta_pair_metrics(
            model,
            processor,
            tokenizer,
            blocks,
            pairs,
            hidden_deltas,
            data_root_for(space.row),
            args,
            description,
        )
    return summarize_pair_metrics(rows)


def evaluate_causal(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    space: TaskSpace,
    pairs: list[dict],
    plus_map: PlusOneMap,
    step: int,
    oracle_iia: float | None,
    control_iia: float | None,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    hidden_deltas = hidden_deltas_for_map(
        args,
        activation_cache,
        sample_cache,
        space=space,
        pairs=pairs,
        plus_map=plus_map,
        step=step,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    description = f"{EXPERIMENT} {plus_map.name} {space.task}[{space.seed}] +{step}"
    metrics = causal_metrics_for_deltas(
        args,
        space=space,
        pairs=pairs,
        hidden_deltas=hidden_deltas,
        description=description,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    iia = metrics["autoregressive_exact_match"]
    recovery = normalized_transfer(iia, control_iia, oracle_iia) if oracle_iia is not None and control_iia is not None else None
    return {
        "task": space.task,
        "task_label": label(space.task),
        "seed": space.seed,
        "model": plus_map.name,
        "model_kind": plus_map.kind,
        "step": step,
        "n_pairs": len(pairs),
        "autoregressive_exact_match": iia,
        "target_tokens_all_topk": metrics["target_tokens_all_topk"],
        "target_tokens_topk_fraction": metrics["target_tokens_topk_fraction"],
        "oracle_autoregressive_exact_match": oracle_iia,
        "control_autoregressive_exact_match": control_iia,
        "equivariance_recovery": recovery,
        "carry_pairs": sum(1 for pair in pairs if pair["carry"] == "carry"),
        "non_carry_pairs": sum(1 for pair in pairs if pair["carry"] == "non_carry"),
    }


def spectral_stats(plus_map: PlusOneMap) -> dict:
    if plus_map.matrix is None:
        return {
            "model": plus_map.name,
            "model_kind": plus_map.kind,
            "available": False,
        }
    matrix = plus_map.matrix.float()
    analysis_matrix = matrix
    eigvals = torch.linalg.eigvals(analysis_matrix)
    angles = torch.angle(eigvals)
    periods = []
    for angle in angles.tolist():
        absolute = abs(float(angle))
        if absolute > 1e-8:
            periods.append(float(2 * torch.pi / absolute))
    orth_error = float((analysis_matrix.T @ analysis_matrix - torch.eye(analysis_matrix.shape[1])).norm())
    try:
        determinant = float(torch.linalg.det(analysis_matrix))
    except RuntimeError:
        determinant = None
    u, s, vh = torch.linalg.svd(analysis_matrix, full_matrices=False)
    polar_q = u @ vh
    polar_angles = torch.angle(torch.linalg.eigvals(polar_q))
    polar_periods = [
        float(2 * torch.pi / abs(float(angle)))
        for angle in polar_angles.tolist()
        if abs(float(angle)) > 1e-8
    ]
    return {
        "model": plus_map.name,
        "model_kind": plus_map.kind,
        "available": True,
        "alpha": plus_map.alpha,
        "lambda": plus_map.lambda_,
        "spectral_radius": float(eigvals.abs().max()),
        "orthogonality_error_fro": orth_error,
        "determinant": determinant,
        "singular_values": [float(value) for value in s.tolist()],
        "stretch_mean": float(s.mean()),
        "stretch_std": float(s.std()) if s.numel() > 1 else 0.0,
        "eig_angles": [float(value) for value in angles.tolist()],
        "rotation_periods": sorted(periods),
        "polar_rotation_periods": sorted(polar_periods),
    }


def sanity_alignment_rows(centroids_by_task: dict[str, dict[int, torch.Tensor]]) -> list[dict]:
    rows = []
    tasks = sorted(centroids_by_task)
    for i, source in enumerate(tasks):
        for destination in tasks[i + 1 :]:
            shared = sorted(set(centroids_by_task[source]) & set(centroids_by_task[destination]))
            if not shared:
                continue
            x = torch.stack([centroids_by_task[source][value] for value in shared])
            y = torch.stack([centroids_by_task[destination][value] for value in shared])
            rows.append(
                {
                    "source_task": source,
                    "destination_task": destination,
                    "n_shared_values": len(shared),
                    "mean_cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=1).mean()),
                    "root_mean_squared_error": float((x - y).square().mean().sqrt()),
                }
            )
    return rows


def save_maps(args: argparse.Namespace, seed: int, maps: dict[str, PlusOneMap], fit_stats: dict, split: dict, sync_path: Path) -> Path:
    path = args.output_dir / "maps" / f"plus_one_maps_seed{seed}_layer{args.layer}_k{args.k}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "shared_space_plus_one_maps",
            "seed": seed,
            "maps": {
                name: {
                    "name": item.name,
                    "kind": item.kind,
                    "matrix": None if item.matrix is None else item.matrix.cpu(),
                    "alpha": item.alpha,
                    "bias": None if item.bias is None else item.bias.cpu(),
                    "lambda": item.lambda_,
                    "fit_task": item.fit_task,
                    "fit_seed": item.fit_seed,
                    "notes": item.notes,
                }
                for name, item in maps.items()
            },
            "fit_stats": fit_stats,
            "value_split": split,
            "synchronization_path": str(sync_path),
            "config": jsonable(vars(args)),
        },
        path,
    )
    return path


def maps_for_task(maps: dict[str, PlusOneMap], task: str) -> list[PlusOneMap]:
    selected = []
    for plus_map in maps.values():
        if plus_map.name.startswith("upper_") and plus_map.fit_task != task:
            continue
        selected.append(plus_map)
    return selected


def row_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def row_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def summarize_geometry(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["model"], row["step"], row["subset"]), []).append(row)
    summary = []
    for (task, model, step, subset), parts in sorted(groups.items()):
        summary.append(
            {
                "task": task,
                "task_label": label(task),
                "model": model,
                "step": step,
                "subset": subset,
                "n": len(parts),
                "mean_cosine_mean": row_mean([row["mean_cosine"] for row in parts]),
                "normalized_mse_mean": row_mean([row["normalized_mse"] for row in parts]),
                "nearest_result_centroid_accuracy_mean": row_mean([row["nearest_result_centroid_accuracy"] for row in parts]),
            }
        )
    return summary


def summarize_causal(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["model"], row["step"]), []).append(row)
    summary = []
    for (task, model, step), parts in sorted(groups.items()):
        summary.append(
            {
                "task": task,
                "task_label": label(task),
                "model": model,
                "step": step,
                "n": len(parts),
                "autoregressive_exact_match_mean": row_mean([row["autoregressive_exact_match"] for row in parts]),
                "autoregressive_exact_match_std": row_std([row["autoregressive_exact_match"] for row in parts]),
                "target_tokens_all_topk_mean": row_mean([row.get("target_tokens_all_topk") for row in parts]),
                "target_tokens_all_topk_std": row_std([row.get("target_tokens_all_topk") for row in parts]),
                "target_tokens_topk_fraction_mean": row_mean([row.get("target_tokens_topk_fraction") for row in parts]),
                "target_tokens_topk_fraction_std": row_std([row.get("target_tokens_topk_fraction") for row in parts]),
                "equivariance_recovery_mean": row_mean([row["equivariance_recovery"] for row in parts]),
                "equivariance_recovery_std": row_std([row["equivariance_recovery"] for row in parts]),
                "equivariance_topk_recovery_mean": row_mean([row.get("equivariance_topk_recovery") for row in parts]),
                "equivariance_topk_fraction_recovery_mean": row_mean([row.get("equivariance_topk_fraction_recovery") for row in parts]),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, geometry_rows: list[dict], causal_rows: list[dict], fit_rows: list[dict], spectral_rows: list[dict], sanity_rows: list[dict], seed_selection: dict) -> None:
    save_jsonl(geometry_rows, args.output_dir / "geometry_results.jsonl")
    save_jsonl(summarize_geometry(geometry_rows), args.output_dir / "geometry_summary.jsonl")
    save_jsonl(causal_rows, args.output_dir / "causal_results.jsonl")
    save_jsonl(summarize_causal(causal_rows), args.output_dir / "causal_summary.jsonl")
    save_jsonl(fit_rows, args.output_dir / "fit_summary.jsonl")
    save_jsonl(spectral_rows, args.output_dir / "spectral_summary.jsonl")
    save_jsonl(sanity_rows, args.output_dir / "shared_space_sanity.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "seed_selection": seed_selection,
            "models": args.models,
            "fit_task": args.fit_task,
            "geometry_steps": args.geometry_steps,
            "causal_steps": args.causal_steps,
            "top_k": args.top_k,
            "top_k_definition": "teacher-forced target answer tokens; every target token must be in top-k for target_tokens_all_topk",
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    args.fit_task = task_key(*parse_task(args.fit_task))
    if args.fit_task not in args.tasks:
        args.tasks.append(args.fit_task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_selection = selected_matching_seeds(args)
    seeds = common_experiment_seeds(seed_selection)
    model, processor, tokenizer, blocks, hidden_size, _model_name = load_model_bundle(args, args.tasks)
    activation_cache = {}
    sample_cache = {}
    geometry_rows = []
    causal_rows = []
    fit_rows = []
    spectral_rows = []
    sanity_rows = []

    print("Shared-space +1 equivariance")
    print(f"  fit_task={args.fit_task}")
    print(f"  models={args.models}")
    for seed in seeds:
        print(f"\nSeed {seed}")
        sync = load_sync(args, seed)
        spaces = make_spaces(args, args.tasks, seed, hidden_size, sync)
        centroids_by_task = {}
        for task, space in spaces.items():
            samples = cached_task_samples(sample_cache, task, space.row)
            centroids_by_task[task] = shared_centroids(
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
        sanity_rows.extend({"seed": seed, **row} for row in sanity_alignment_rows(centroids_by_task))
        split = value_split(args, centroids_by_task[args.fit_task])
        plus_maps, fit_stats = fit_plus_maps(args, centroids_by_task[args.fit_task], split, seed=seed, fit_task=args.fit_task)

        if args.include_domain_specific_upper:
            for task in args.tasks:
                domain_maps, _domain_stats = fit_plus_maps(
                    args,
                    centroids_by_task[task],
                    split,
                    seed=seed,
                    name_prefix=f"upper_{label(task)}",
                    fit_task=task,
                )
                plus_maps[f"upper_{label(task)}_scaled_orthogonal"] = domain_maps[f"upper_{label(task)}_scaled_orthogonal"]

        selected_maps = {
            name: plus_map
            for name, plus_map in plus_maps.items()
            if name in args.models or any(name == f"upper_{label(task)}_scaled_orthogonal" for task in args.tasks)
        }
        selected_maps["oracle_target_state"] = PlusOneMap("oracle_target_state", "oracle", fit_seed=seed, fit_task="none")
        maps_path = save_maps(args, seed, selected_maps, fit_stats, split, sync["path"])
        fit_rows.append(
            {
                "seed": seed,
                "maps_path": str(maps_path),
                "synchronization_path": str(sync["path"]),
                "value_split": split,
                **fit_stats,
            }
        )
        spectral_rows.extend({"seed": seed, **spectral_stats(plus_map)} for plus_map in selected_maps.values())

        for step in args.geometry_steps:
            starts = split["test_starts"]
            for task in args.tasks:
                for plus_map in maps_for_task(selected_maps, task):
                    geometry_rows.extend(
                        {"seed": seed, "maps_path": str(maps_path), **row}
                        for row in evaluate_geometry(centroids_by_task, plus_map, task=task, starts=starts, step=step)
                    )

        for step in args.causal_steps:
            starts = split["test_starts"]
            for task, space in spaces.items():
                pairs = plus_pairs_for_task(args, sample_cache, space=space, starts=starts, step=step)
                if not pairs:
                    print(f"  no +{step} causal pairs for {task}[{seed}]")
                    continue
                oracle = selected_maps["oracle_target_state"]
                identity = selected_maps.get("identity", PlusOneMap("identity", "identity", fit_seed=seed))
                oracle_row = evaluate_causal(
                    args,
                    activation_cache,
                    sample_cache,
                    space=space,
                    pairs=pairs,
                    plus_map=oracle,
                    step=step,
                    oracle_iia=None,
                    control_iia=None,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    blocks=blocks,
                )
                identity_row = evaluate_causal(
                    args,
                    activation_cache,
                    sample_cache,
                    space=space,
                    pairs=pairs,
                    plus_map=identity,
                    step=step,
                    oracle_iia=None,
                    control_iia=None,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    blocks=blocks,
                )
                oracle_iia = oracle_row["autoregressive_exact_match"]
                control_iia = identity_row["autoregressive_exact_match"]
                oracle_topk = oracle_row["target_tokens_all_topk"]
                control_topk = identity_row["target_tokens_all_topk"]
                oracle_topk_fraction = oracle_row["target_tokens_topk_fraction"]
                control_topk_fraction = identity_row["target_tokens_topk_fraction"]
                oracle_row["oracle_autoregressive_exact_match"] = oracle_iia
                oracle_row["control_autoregressive_exact_match"] = control_iia
                oracle_row["oracle_target_tokens_all_topk"] = oracle_topk
                oracle_row["control_target_tokens_all_topk"] = control_topk
                oracle_row["oracle_target_tokens_topk_fraction"] = oracle_topk_fraction
                oracle_row["control_target_tokens_topk_fraction"] = control_topk_fraction
                oracle_row["equivariance_recovery"] = normalized_transfer(oracle_iia, control_iia, oracle_iia)
                oracle_row["equivariance_topk_recovery"] = normalized_transfer(oracle_topk, control_topk, oracle_topk)
                oracle_row["equivariance_topk_fraction_recovery"] = normalized_transfer(oracle_topk_fraction, control_topk_fraction, oracle_topk_fraction)
                identity_row["oracle_autoregressive_exact_match"] = oracle_iia
                identity_row["control_autoregressive_exact_match"] = control_iia
                identity_row["oracle_target_tokens_all_topk"] = oracle_topk
                identity_row["control_target_tokens_all_topk"] = control_topk
                identity_row["oracle_target_tokens_topk_fraction"] = oracle_topk_fraction
                identity_row["control_target_tokens_topk_fraction"] = control_topk_fraction
                identity_row["equivariance_recovery"] = normalized_transfer(control_iia, control_iia, oracle_iia)
                identity_row["equivariance_topk_recovery"] = normalized_transfer(control_topk, control_topk, oracle_topk)
                identity_row["equivariance_topk_fraction_recovery"] = normalized_transfer(control_topk_fraction, control_topk_fraction, oracle_topk_fraction)
                causal_rows.extend(
                    {"maps_path": str(maps_path), "value_split": split, **row}
                    for row in [oracle_row, identity_row]
                )
                for plus_map in maps_for_task(selected_maps, task):
                    if plus_map.name in {"oracle_target_state", "identity"}:
                        continue
                    row = evaluate_causal(
                        args,
                        activation_cache,
                        sample_cache,
                        space=space,
                        pairs=pairs,
                        plus_map=plus_map,
                        step=step,
                        oracle_iia=oracle_iia,
                        control_iia=control_iia,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                    )
                    row["oracle_target_tokens_all_topk"] = oracle_topk
                    row["control_target_tokens_all_topk"] = control_topk
                    row["oracle_target_tokens_topk_fraction"] = oracle_topk_fraction
                    row["control_target_tokens_topk_fraction"] = control_topk_fraction
                    row["equivariance_topk_recovery"] = normalized_transfer(
                        row["target_tokens_all_topk"], control_topk, oracle_topk
                    )
                    row["equivariance_topk_fraction_recovery"] = normalized_transfer(
                        row["target_tokens_topk_fraction"], control_topk_fraction, oracle_topk_fraction
                    )
                    causal_rows.append(
                        {
                            "maps_path": str(maps_path),
                            "value_split": split,
                            **row,
                        }
                    )
                write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, seed_selection)
        write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, seed_selection)

    write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, seed_selection)
    print(f"\nSaved geometry: {args.output_dir / 'geometry_results.jsonl'}")
    print(f"Saved causal: {args.output_dir / 'causal_results.jsonl'}")
    print(f"Saved fit summary: {args.output_dir / 'fit_summary.jsonl'}")


if __name__ == "__main__":
    main()
