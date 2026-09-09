"""Quick universality baselines over frozen text/image DAS spaces.

The key cross-operation test uses ambient operators:

    M_o = R_destination,o alpha_o Q_o R_source,o^T

so maps learned on one operation can be applied to donor-base interventions
from the other operation without comparing arbitrary DAS coordinate entries.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from src.interventions.das import build_unique_pairs
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
    clean_name,
    coordinate_deltas_for_pairs,
    coordinate_metrics,
    data_root_for,
    load_heldout_pairs,
    load_model_bundle,
    orthogonal_procrustes,
    sample_key,
    sample_split,
    scaled_alpha,
    stable_seed,
)


EXPERIMENT = "universality_baselines"
OPERATIONS = ["addition", "subtraction"]
DEFAULT_TASKS = [
    "text:addition",
    "image:addition",
    "text:subtraction",
    "image:subtraction",
]
DEFAULT_ALIGNMENTS = [
    "text:addition->image:addition",
    "image:addition->text:addition",
    "text:subtraction->image:subtraction",
    "image:subtraction->text:subtraction",
]
CONDITION_DESCRIPTIONS = {
    "destination_self_das": "destination DAS subspace on the evaluation operation",
    "raw_direct_transfer": "source DAS delta patched directly in destination residual space",
    "operation_specific_transport": "ambient operator learned on the same operation",
    "cross_operation_transport": "ambient operator learned on the opposite operation",
    "pooled_transport": "one shared ambient operator fit on addition and subtraction together",
}
CONDITIONS = list(CONDITION_DESCRIPTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--procrustes_dir", type=Path, default=Path("results/final_exps/procrustes"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/universality"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--alignments", nargs="+", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--transport_variant", default="scaled_displacement")
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
    parser.add_argument("--max_alignment_pairs_per_operation", type=int, default=1024)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_alignment(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Alignment must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def opposite_operation(operation: str) -> str:
    if operation == "addition":
        return "subtraction"
    if operation == "subtraction":
        return "addition"
    raise ValueError(f"Unsupported operation for universality: {operation!r}")


def operation_task(task: str, operation: str) -> str:
    modality, _old_operation = parse_task(task)
    return task_key(modality, operation)


def basis_for(args: argparse.Namespace, hidden_size: int, task: str, seed: int) -> torch.Tensor:
    modality, operation = parse_task(task)
    return load_basis(subspace_path(args, modality, operation, seed), args.layer, hidden_size, args.k)


def scaled_map_path(args: argparse.Namespace, source_task: str, source_seed: int, destination_task: str, destination_seed: int) -> Path:
    return (
        args.procrustes_dir
        / args.transport_variant
        / "matrices"
        / (
            f"{args.transport_variant}_{clean_name(source_task, source_seed)}"
            f"_to_{clean_name(destination_task, destination_seed)}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    )


def load_scaled_map(args: argparse.Namespace, source_task: str, source_seed: int, destination_task: str, destination_seed: int) -> dict:
    path = scaled_map_path(args, source_task, source_seed, destination_task, destination_seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing operation-specific transport map {path}. Run Procrustes scaled_displacement first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "source_task": source_task,
        "destination_task": destination_task,
        "source_basis": torch.as_tensor(payload["source_basis"]).float(),
        "destination_basis": torch.as_tensor(payload["destination_basis"]).float(),
        "matrix": torch.as_tensor(payload["Q_source_to_destination"]).float(),
        "alpha": float(payload["alpha"]),
        "path": path,
        "fit_kind": f"ambient_operator_from_{args.transport_variant}",
    }


def orthonormal_union(bases: list[torch.Tensor]) -> torch.Tensor:
    joined = torch.cat([basis.float() for basis in bases], dim=1)
    left, singular_values, _vh = torch.linalg.svd(joined, full_matrices=False)
    tolerance = max(joined.shape) * torch.finfo(joined.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    return left[:, :rank]


def train_pairs_for_task(args: argparse.Namespace, sample_cache: dict, task: str, row: dict, seed_suffix: str) -> list[dict]:
    samples = sample_split(args, cached_task_samples(sample_cache, task, row), "train")
    pairs, _stats = build_unique_pairs(
        samples,
        args.target,
        stable_seed(args.alignment_seed, task, seed_suffix),
        args.max_alignment_pairs_per_operation,
    )
    for index, pair in enumerate(pairs):
        pair["pair_id"] = index
    return pairs


def mirror_pairs(pairs: list[dict], source_samples: list[dict]) -> list[dict]:
    lookup = {sample_key(sample): sample for sample in source_samples}
    mirrored = []
    missing = []
    for index, pair in enumerate(pairs):
        base_key = sample_key(pair["base"])
        source_key = sample_key(pair["source"])
        if base_key not in lookup or source_key not in lookup:
            missing.append((base_key, source_key))
            continue
        mirrored.append(
            {
                "pair_id": pair.get("pair_id", index),
                "base": lookup[base_key],
                "source": lookup[source_key],
            }
        )
    if missing:
        raise KeyError(f"Could not mirror {len(missing)} pairs; first missing={missing[0]!r}")
    return mirrored


def fit_pooled_map(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_modality: str,
    destination_modality: str,
    source_seed: int,
    destination_seed: int,
    hidden_size: int,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict:
    source_tasks = [task_key(source_modality, operation) for operation in OPERATIONS]
    destination_tasks = [task_key(destination_modality, operation) for operation in OPERATIONS]
    source_bases = [basis_for(args, hidden_size, task, source_seed) for task in source_tasks]
    destination_bases = [basis_for(args, hidden_size, task, destination_seed) for task in destination_tasks]
    source_pool = orthonormal_union(source_bases)
    destination_pool = orthonormal_union(destination_bases)

    x_parts = []
    y_parts = []
    stats = []
    for operation in OPERATIONS:
        source_task = task_key(source_modality, operation)
        destination_task = task_key(destination_modality, operation)
        source_row = first_result_row(args, source_modality, operation, args.condition, source_seed)
        destination_row = first_result_row(args, destination_modality, operation, args.condition, destination_seed)
        destination_pairs = train_pairs_for_task(args, sample_cache, destination_task, destination_row, operation)
        source_pairs = mirror_pairs(destination_pairs, cached_task_samples(sample_cache, source_task, source_row))
        x = coordinate_deltas_for_pairs(
            args,
            activation_cache,
            sample_cache,
            task=source_task,
            row=source_row,
            pairs=source_pairs,
            basis=source_pool,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        y = coordinate_deltas_for_pairs(
            args,
            activation_cache,
            sample_cache,
            task=destination_task,
            row=destination_row,
            pairs=destination_pairs,
            basis=destination_pool,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        x_parts.append(x)
        y_parts.append(y)
        stats.append(
            {
                "operation": operation,
                "source_task": source_task,
                "destination_task": destination_task,
                "n_train_pairs": len(destination_pairs),
            }
        )

    x_train = torch.cat(x_parts, dim=0)
    y_train = torch.cat(y_parts, dim=0)
    matrix = orthogonal_procrustes(x_train, y_train)
    alpha = scaled_alpha(x_train, y_train, matrix)
    return {
        "source_task": f"{source_modality}:pooled_addition_subtraction",
        "destination_task": f"{destination_modality}:pooled_addition_subtraction",
        "source_basis": source_pool,
        "destination_basis": destination_pool,
        "matrix": matrix,
        "alpha": alpha,
        "path": None,
        "fit_kind": "pooled_addition_subtraction_ambient_operator",
        "fit_metrics": coordinate_metrics(x_train, y_train, matrix, alpha),
        "fit_stats": stats,
    }


def pooled_cache_path(args: argparse.Namespace, source_modality: str, destination_modality: str, source_seed: int, destination_seed: int) -> Path:
    return (
        args.output_dir
        / "maps"
        / "pooled_transport"
        / (
            f"pooled_{source_modality}_seed{source_seed}"
            f"_to_{destination_modality}_seed{destination_seed}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    )


def load_or_fit_pooled_map(args: argparse.Namespace, activation_cache: dict, sample_cache: dict, *, source_modality: str, destination_modality: str, source_seed: int, destination_seed: int, hidden_size: int, model, processor, tokenizer, blocks) -> dict:
    path = pooled_cache_path(args, source_modality, destination_modality, source_seed, destination_seed)
    if path.exists() and not args.force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "source_task": payload["source_task"],
            "destination_task": payload["destination_task"],
            "source_basis": torch.as_tensor(payload["source_basis"]).float(),
            "destination_basis": torch.as_tensor(payload["destination_basis"]).float(),
            "matrix": torch.as_tensor(payload["matrix_source_to_destination"]).float(),
            "alpha": float(payload["alpha"]),
            "path": path,
            "fit_kind": payload["fit_kind"],
            "fit_metrics": payload.get("fit_metrics"),
            "fit_stats": payload.get("fit_stats"),
        }
    fitted = fit_pooled_map(
        args,
        activation_cache,
        sample_cache,
        source_modality=source_modality,
        destination_modality=destination_modality,
        source_seed=source_seed,
        destination_seed=destination_seed,
        hidden_size=hidden_size,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "universality_pooled_ambient_operator",
            "source_task": fitted["source_task"],
            "destination_task": fitted["destination_task"],
            "source_seed": source_seed,
            "destination_seed": destination_seed,
            "source_basis": fitted["source_basis"].cpu(),
            "destination_basis": fitted["destination_basis"].cpu(),
            "matrix_source_to_destination": fitted["matrix"].cpu(),
            "alpha": fitted["alpha"],
            "fit_kind": fitted["fit_kind"],
            "fit_metrics": fitted["fit_metrics"],
            "fit_stats": fitted["fit_stats"],
            "config": jsonable(vars(args)),
        },
        path,
    )
    fitted["path"] = path
    return fitted


def hidden_deltas_from_map(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    source_row: dict,
    source_pairs: list[dict],
    transport: dict,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    coords = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source_task,
        row=source_row,
        pairs=source_pairs,
        basis=transport["source_basis"],
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return (float(transport["alpha"]) * (coords @ transport["matrix"])) @ transport["destination_basis"].T


def raw_direct_deltas(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    source_row: dict,
    source_pairs: list[dict],
    source_basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    coords = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source_task,
        row=source_row,
        pairs=source_pairs,
        basis=source_basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return coords @ source_basis.T


def destination_self_deltas(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    destination_task: str,
    destination_row: dict,
    destination_pairs: list[dict],
    destination_basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    coords = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=destination_task,
        row=destination_row,
        pairs=destination_pairs,
        basis=destination_basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return coords @ destination_basis.T


def pair_outputs_path(args: argparse.Namespace, row: dict) -> Path:
    return (
        args.output_dir
        / "autoregressive_outputs"
        / row["condition"]
        / (
            f"{row['source_label']}_seed{row['source_seed']}"
            f"_to_{row['destination_label']}_seed{row['destination_seed']}"
            f"_layer{args.layer}_k{args.k}.jsonl"
        )
    )


def attach_pair_outputs(rows: list[dict], pairs: list[dict]) -> list[dict]:
    outputs = []
    for pair, row in zip(pairs, rows):
        base, source = pair["base"], pair["source"]
        item = dict(row)
        item["base_result"] = int(base["result"])
        item["source_result"] = int(source["result"])
        outputs.append(item)
    return outputs


def evaluate_patched(
    args: argparse.Namespace,
    *,
    destination_task: str,
    destination_row: dict,
    destination_pairs: list[dict],
    hidden_deltas: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
    description: str,
) -> tuple[float, list[dict]]:
    modality, _operation = parse_task(destination_task)
    if modality == "text":
        score = autoregressive_iia_text_aligned(
            model, tokenizer, blocks, destination_pairs, hidden_deltas, args, description
        )
    else:
        score = autoregressive_iia_image_aligned(
            model,
            processor,
            tokenizer,
            blocks,
            destination_pairs,
            hidden_deltas,
            data_root_for(destination_row),
            args,
            description,
        )
    # The reused scorer returns only the aggregate. Store pair metadata so the
    # cached rows remain auditable, even without per-pair correctness.
    outputs = [
        {
            "pair_id": pair.get("pair_id", index),
            "base_result": int(pair["base"]["result"]),
            "source_result": int(pair["source"]["result"]),
        }
        for index, pair in enumerate(destination_pairs)
    ]
    return score, outputs


def row_key(row: dict) -> tuple:
    return (
        row["condition"],
        row["source_task"],
        int(row["source_seed"]),
        row["destination_task"],
        int(row["destination_seed"]),
    )


def load_cache(args: argparse.Namespace) -> dict[tuple, dict]:
    path = args.output_dir / "universality_results.jsonl"
    if args.force or not path.exists():
        return {}
    return {row_key(row): row for row in load_jsonl(path)}


def evaluate_condition(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    condition: str,
    source_task: str,
    destination_task: str,
    source_seed: int,
    destination_seed: int,
    source_row: dict,
    destination_row: dict,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    op_transport: dict,
    cross_transport: dict,
    pooled_transport: dict,
    model,
    processor,
    tokenizer,
    blocks,
    model_name: str,
) -> dict:
    source_modality, source_operation = parse_task(source_task)
    destination_modality, destination_operation = parse_task(destination_task)
    if condition == "destination_self_das":
        hidden_deltas = destination_self_deltas(
            args,
            activation_cache,
            sample_cache,
            destination_task=destination_task,
            destination_row=destination_row,
            destination_pairs=destination_pairs,
            destination_basis=destination_basis,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        transport = None
        fit_kind = "destination_self_das"
    elif condition == "raw_direct_transfer":
        hidden_deltas = raw_direct_deltas(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            source_row=source_row,
            source_pairs=source_pairs,
            source_basis=source_basis,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        transport = None
        fit_kind = "source_das_delta_direct_in_ambient_space"
    else:
        transport = {
            "operation_specific_transport": op_transport,
            "cross_operation_transport": cross_transport,
            "pooled_transport": pooled_transport,
        }[condition]
        hidden_deltas = hidden_deltas_from_map(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            source_row=source_row,
            source_pairs=source_pairs,
            transport=transport,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        fit_kind = transport["fit_kind"]

    description = f"{condition} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]"
    score, pair_outputs = evaluate_patched(
        args,
        destination_task=destination_task,
        destination_row=destination_row,
        destination_pairs=destination_pairs,
        hidden_deltas=hidden_deltas,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        description=description,
    )
    destination_control = control_iia(args, destination_modality, destination_operation, destination_seed, destination_row)
    destination_self = float(destination_row["autoregressive_iia"])
    normalized = normalized_transfer(score, destination_control, destination_self)
    row = {
        "model": model_name,
        "experiment": EXPERIMENT,
        "condition": condition,
        "condition_description": CONDITION_DESCRIPTIONS[condition],
        "fit_kind": fit_kind,
        "source_task": source_task,
        "source_label": label(source_task),
        "source_modality": source_modality,
        "source_operation": source_operation,
        "source_seed": source_seed,
        "source_subspace_path": str(subspace_path(args, source_modality, source_operation, source_seed)),
        "destination_task": destination_task,
        "destination_label": label(destination_task),
        "destination_modality": destination_modality,
        "destination_operation": destination_operation,
        "destination_seed": destination_seed,
        "destination_results_path": str(results_path(args, destination_modality, destination_operation, args.condition, destination_seed)),
        "destination_heldout_pairs_path": str(heldout_pairs_path(args, destination_modality, destination_operation, destination_seed)),
        "transport_path": None if transport is None or transport.get("path") is None else str(transport["path"]),
        "transport_train_source_task": None if transport is None else transport["source_task"],
        "transport_train_destination_task": None if transport is None else transport["destination_task"],
        "layer": args.layer,
        "k": args.k,
        "hook": args.hook,
        "source_position": position_for(args, source_modality),
        "destination_position": position_for(args, destination_modality),
        "target": args.target,
        "n_pairs": len(destination_pairs),
        "alpha": None if transport is None else transport["alpha"],
        "alignment_fit": None if transport is None else transport.get("fit_metrics"),
        "autoregressive_iia": score,
        "destination_self_autoregressive_iia": destination_self,
        "control_condition": args.control_condition,
        "destination_control_autoregressive_iia": destination_control,
        "destination_normalized_transfer": normalized,
    }
    output_path = pair_outputs_path(args, row)
    save_jsonl(pair_outputs, output_path)
    row["autoregressive_outputs_path"] = str(output_path)
    print(
        f"  {condition} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]: "
        f"AR IIA={score:.4f}; normalized={normalized}"
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


def matrix_payload(rows: list[dict], tasks: list[str], metric: str) -> dict:
    labels = [label(task) for task in tasks]
    matrix = []
    matrix_rows = []
    for source in tasks:
        row_values = []
        for destination in tasks:
            cell_rows = [
                row
                for row in rows
                if row["source_task"] == source and row["destination_task"] == destination
            ]
            values = [row.get(metric) for row in cell_rows]
            value = mean(values)
            row_values.append(value)
            matrix_rows.append(
                {
                    "source_task": source,
                    "source_label": label(source),
                    "destination_task": destination,
                    "destination_label": label(destination),
                    "metric": metric,
                    "mean": value,
                    "std": sample_std(values),
                    "n": len(cell_rows),
                    "values": values,
                }
            )
        matrix.append(row_values)
    return {"tasks": tasks, "labels": labels, "metric": metric, "matrix": matrix, "rows": matrix_rows}


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["source_task"], row["destination_task"], row["condition"]), []).append(row)
    summary = []
    for (source_task, destination_task, condition), parts in sorted(groups.items()):
        summary.append(
            {
                "source_task": source_task,
                "source_label": label(source_task),
                "destination_task": destination_task,
                "destination_label": label(destination_task),
                "condition": condition,
                "condition_description": CONDITION_DESCRIPTIONS[condition],
                "n": len(parts),
                "autoregressive_iia_mean": mean([row["autoregressive_iia"] for row in parts]),
                "autoregressive_iia_std": sample_std([row["autoregressive_iia"] for row in parts]),
                "destination_normalized_transfer_mean": mean([row["destination_normalized_transfer"] for row in parts]),
                "destination_normalized_transfer_std": sample_std([row["destination_normalized_transfer"] for row in parts]),
                "alpha_mean": mean([row["alpha"] for row in parts]),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, rows: list[dict], seed_selection: dict[str, list[int]]) -> None:
    rows.sort(key=row_key)
    save_jsonl(rows, args.output_dir / "universality_results.jsonl")
    save_jsonl(summarize(rows), args.output_dir / "universality_summary.jsonl")
    for condition in args.conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for metric in ["autoregressive_iia", "destination_normalized_transfer"]:
            payload = matrix_payload(condition_rows, args.tasks, metric)
            out = args.output_dir / "matrices" / condition
            save_json(payload, out / f"{metric}_matrix.json")
            save_jsonl(payload["rows"], out / f"{metric}_matrix.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "condition_descriptions": CONDITION_DESCRIPTIONS,
            "seed_selection": seed_selection,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def merge(cache: dict[tuple, dict], rows: list[dict]) -> list[dict]:
    merged = dict(cache)
    for row in rows:
        merged[row_key(row)] = row
    return list(merged.values())


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    alignments = [parse_alignment(item) for item in args.alignments]
    args.tasks = list(dict.fromkeys(args.tasks + [task for pair in alignments for task in pair]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_selection = selected_seeds(args)
    cache = load_cache(args)
    rows = []
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    activation_cache = {}
    sample_cache = {}
    pooled_cache = {}

    print("Universality baselines")
    print(f"  conditions={args.conditions}")
    print(f"  alignments={args.alignments}")
    for source_task, destination_task in alignments:
        source_modality, source_operation = parse_task(source_task)
        destination_modality, destination_operation = parse_task(destination_task)
        cross_source_task = task_key(source_modality, opposite_operation(source_operation))
        cross_destination_task = task_key(destination_modality, opposite_operation(destination_operation))
        for source_seed in seed_selection[source_task]:
            for destination_seed in seed_selection[destination_task]:
                source_row = first_result_row(args, source_modality, source_operation, args.condition, source_seed)
                destination_row = first_result_row(args, destination_modality, destination_operation, args.condition, destination_seed)
                source_basis = basis_for(args, hidden_size, source_task, source_seed)
                destination_basis = basis_for(args, hidden_size, destination_task, destination_seed)
                destination_pairs = load_heldout_pairs(
                    args,
                    destination_modality,
                    destination_operation,
                    destination_seed,
                    destination_row,
                    args.max_autoregressive_pairs,
                )
                source_pairs = mirror_pairs(destination_pairs, cached_task_samples(sample_cache, source_task, source_row))
                op_transport = load_scaled_map(args, source_task, source_seed, destination_task, destination_seed)
                cross_transport = load_scaled_map(args, cross_source_task, source_seed, cross_destination_task, destination_seed)
                pooled_key = (source_modality, destination_modality, source_seed, destination_seed)
                if pooled_key not in pooled_cache:
                    pooled_cache[pooled_key] = load_or_fit_pooled_map(
                        args,
                        activation_cache,
                        sample_cache,
                        source_modality=source_modality,
                        destination_modality=destination_modality,
                        source_seed=source_seed,
                        destination_seed=destination_seed,
                        hidden_size=hidden_size,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                    )
                for condition in args.conditions:
                    key = (condition, source_task, source_seed, destination_task, destination_seed)
                    if key in cache:
                        rows.append(cache[key])
                        print(f"  cached {condition} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]")
                        continue
                    rows.append(
                        evaluate_condition(
                            args,
                            activation_cache,
                            sample_cache,
                            condition=condition,
                            source_task=source_task,
                            destination_task=destination_task,
                            source_seed=source_seed,
                            destination_seed=destination_seed,
                            source_row=source_row,
                            destination_row=destination_row,
                            source_basis=source_basis,
                            destination_basis=destination_basis,
                            source_pairs=source_pairs,
                            destination_pairs=destination_pairs,
                            op_transport=op_transport,
                            cross_transport=cross_transport,
                            pooled_transport=pooled_cache[pooled_key],
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            blocks=blocks,
                            model_name=model_name,
                        )
                    )
                    write_outputs(args, merge(cache, rows), seed_selection)
    write_outputs(args, merge(cache, rows), seed_selection)
    print(f"\nSaved rows: {args.output_dir / 'universality_results.jsonl'}")
    print(f"Saved summary: {args.output_dir / 'universality_summary.jsonl'}")
    print(f"Saved matrices under: {args.output_dir / 'matrices'}")


if __name__ == "__main__":
    main()
