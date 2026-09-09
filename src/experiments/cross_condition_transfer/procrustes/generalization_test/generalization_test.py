"""Frozen-DAS value-held-out cross-modal Procrustes generalization.

This experiment does not retrain DAS. It treats the existing text/image DAS
subspaces as fixed causal coordinate systems, fits cross-modal maps using only
training result values, and evaluates causal patching on held-out result values.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from contextlib import ExitStack
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.interventions.das import (
    build_unique_pairs,
    format_prompt,
    target_answers,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    control_iia,
    first_result_row,
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
    activations_for_ids,
    cached_task_samples,
    clean_name,
    coordinate_deltas_for_pairs,
    coordinate_metrics,
    data_root_for,
    hidden,
    hook_module,
    load_model_bundle,
    orthogonal_procrustes,
    patched_forward_with_deltas,
    random_orthogonal,
    replace_hidden,
    resolve_position,
    sample_key,
    scaled_alpha,
    stable_seed,
    target_value,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template


EXPERIMENT = "frozen_das_value_heldout_generalization"
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
    "destination_self_das": "destination DAS subspace evaluated on held-out values",
    "raw_source_das_transfer": "source DAS subspace used directly in destination residual space",
    "rotation_only": "orthogonal map fit on train value centroids, alpha fixed to 1",
    "global_scaling_only": "global alpha from the learned map, Q fixed to identity",
    "rotation_plus_scaling": "orthogonal map plus one global scale fit on train value centroids",
    "unrestricted_linear_upper_bound": "unconstrained linear map fit on train value centroids",
    "random_orthogonal_map": "random orthogonal map control",
    "shuffled_value_correspondences": "scaled orthogonal map after shuffling train value correspondences",
}
CONDITIONS = list(CONDITION_DESCRIPTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper/procrustes/generalization_test"))
    parser.add_argument("--plot_dir", type=Path, default=Path("visualizations/main_paper/procrustes/generalization_test"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--alignments", nargs="+", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--control_condition", default="random_subspace_in_pca_span")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--das_split_seed", type=int, default=0)
    parser.add_argument("--value_split_seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train_value_fraction", type=float, default=0.8)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--seed_map", nargs="*", default=[], metavar="TASK=SEEDS")
    parser.add_argument("--max_eval_pairs", type=int, default=64)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--plot_first_seed_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_alignment(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Alignment must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def unrestricted_linear_map(source_z: torch.Tensor, destination_z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(source_z.float(), destination_z.float()).solution.to(dtype=source_z.dtype)


def split_result_values(values: list[int], train_fraction: float, split_seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train_value_fraction must be between 0 and 1.")
    shuffled = sorted(set(values))
    random.Random(split_seed).shuffle(shuffled)
    train_n = int(len(shuffled) * train_fraction)
    train_n = max(1, min(train_n, len(shuffled) - 1))
    return sorted(shuffled[:train_n]), sorted(shuffled[train_n:])


def values_in_samples(args: argparse.Namespace, samples: list[dict]) -> set[int]:
    return {target_value(args, sample) for sample in samples}


def task_rows_and_bases(args: argparse.Namespace, hidden_size: int, tasks: list[str], seed_selection: dict[str, list[int]]) -> dict:
    spaces = {}
    for task in tasks:
        modality, operation = parse_task(task)
        for seed in seed_selection[task]:
            row = first_result_row(args, modality, operation, args.condition, seed)
            basis = load_basis(subspace_path(args, modality, operation, seed), args.layer, hidden_size, args.k)
            spaces[(task, seed)] = {"row": row, "basis": basis}
    return spaces


def samples_for_values(args: argparse.Namespace, samples: list[dict], values: set[int]) -> list[dict]:
    return [sample for sample in samples if target_value(args, sample) in values]


def matched_pair_samples(pairs: list[dict], source_samples: list[dict]) -> list[dict]:
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
        raise KeyError(f"Could not mirror {len(missing)} destination pairs into the source task; first={missing[0]!r}.")
    return mirrored


def heldout_value_pairs(
    args: argparse.Namespace,
    destination_samples: list[dict],
    test_values: set[int],
    split_seed: int,
) -> tuple[list[dict], dict]:
    filtered = samples_for_values(args, destination_samples, test_values)
    pairs, stats = build_unique_pairs(filtered, args.target, split_seed + 1009, args.max_eval_pairs)
    for index, pair in enumerate(pairs):
        pair["pair_id"] = index
    stats["heldout_values"] = sorted(test_values)
    stats["n_heldout_samples"] = len(filtered)
    return pairs, stats


def centroids_for_values(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    task: str,
    row: dict,
    basis: torch.Tensor,
    values: set[int],
    model,
    processor,
    tokenizer,
    blocks,
) -> dict[int, torch.Tensor]:
    samples = samples_for_values(args, cached_task_samples(sample_cache, task, row), values)
    ids = [sample_key(sample) for sample in samples]
    activations = activations_for_ids(
        args,
        activation_cache,
        sample_cache,
        task=task,
        row=row,
        ids=ids,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    grouped: dict[int, list[torch.Tensor]] = {}
    for sample, coordinate in zip(samples, activations @ basis):
        grouped.setdefault(target_value(args, sample), []).append(coordinate)
    return {value: torch.stack(parts).mean(dim=0) for value, parts in grouped.items()}


def centroid_matrices(source_centroids: dict[int, torch.Tensor], destination_centroids: dict[int, torch.Tensor]) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    values = sorted(set(source_centroids) & set(destination_centroids))
    if len(values) < 2:
        raise ValueError("Need at least two shared result values for centroid fitting/projection.")
    source = torch.stack([source_centroids[value] for value in values])
    destination = torch.stack([destination_centroids[value] for value in values])
    return values, source, destination


def fit_maps(
    args: argparse.Namespace,
    *,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    split_seed: int,
    x_train_abs: torch.Tensor,
    y_train_abs: torch.Tensor,
) -> dict[str, dict]:
    x_origin = x_train_abs.mean(dim=0, keepdim=True)
    y_origin = y_train_abs.mean(dim=0, keepdim=True)
    x_train = x_train_abs - x_origin
    y_train = y_train_abs - y_origin
    q = orthogonal_procrustes(x_train, y_train)
    alpha = scaled_alpha(x_train, y_train, q)
    random_q = random_orthogonal(
        args.k,
        stable_seed(args.model, split_seed, source_task, source_seed, destination_task, destination_seed, "random_q"),
    ).to(dtype=x_train.dtype)
    order = list(range(y_train.shape[0]))
    random.Random(stable_seed(args.model, split_seed, source_task, destination_task, "shuffle_values")).shuffle(order)
    shuffled_y = y_train[order]
    shuffled_q = orthogonal_procrustes(x_train, shuffled_y)
    shuffled_alpha = scaled_alpha(x_train, shuffled_y, shuffled_q)
    linear = unrestricted_linear_map(x_train, y_train)
    identity = torch.eye(args.k, dtype=x_train.dtype)
    return {
        "rotation_only": {"matrix": q, "alpha": 1.0, "fit_kind": "train_value_centroids_orthogonal"},
        "global_scaling_only": {"matrix": identity, "alpha": alpha, "fit_kind": "train_value_centroids_alpha_identity"},
        "rotation_plus_scaling": {"matrix": q, "alpha": alpha, "fit_kind": "train_value_centroids_scaled_orthogonal"},
        "unrestricted_linear_upper_bound": {"matrix": linear, "alpha": 1.0, "fit_kind": "train_value_centroids_unrestricted_linear"},
        "random_orthogonal_map": {"matrix": random_q, "alpha": 1.0, "fit_kind": "random_orthogonal_control"},
        "shuffled_value_correspondences": {
            "matrix": shuffled_q,
            "alpha": shuffled_alpha,
            "fit_kind": "train_value_centroids_shuffled_correspondences",
            "shuffled_destination_order": order,
        },
        "_origins": {"source": x_origin, "destination": y_origin, "centered_x_train": x_train, "centered_y_train": y_train},
    }


def hidden_deltas_from_source_pairs(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    source_row: dict,
    source_pairs: list[dict],
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    matrix: torch.Tensor,
    alpha: float,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    source_deltas = coordinate_deltas_for_pairs(
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
    return (float(alpha) * (source_deltas @ matrix)) @ destination_basis.T


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
    deltas = coordinate_deltas_for_pairs(
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
    return deltas @ destination_basis.T


def raw_source_transfer_deltas(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    destination_task: str,
    destination_row: dict,
    destination_pairs: list[dict],
    source_basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    deltas = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=destination_task,
        row=destination_row,
        pairs=destination_pairs,
        basis=source_basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return deltas @ source_basis.T


@torch.no_grad()
def autoregressive_text_outputs(model, tokenizer, blocks, pairs: list[dict], hidden_deltas: torch.Tensor, args: argparse.Namespace, description: str) -> tuple[float, list[dict]]:
    correct = 0
    outputs = []
    for pair, delta in zip(pairs, hidden_deltas):
        base, source = pair["base"], pair["source"]
        base_prompt = format_prompt(tokenizer, base, args.use_chat_template or uses_chat_template(args.model))
        expected = target_answers(base, source, args.target)[1]
        base_position = resolve_position(tokenizer, base_prompt, args.text_position)
        base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            outputs_ = patched_forward_with_deltas(
                model,
                {"input_ids": base_ids[None]},
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            next_id = outputs_.logits[0, base_ids.shape[0] - 1].argmax().reshape(1)
            generated.append(int(next_id.item()))
            base_ids = torch.cat([base_ids, next_id])
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        is_correct = match is not None and match.group() == expected
        correct += int(is_correct)
        outputs.append(
            {
                "pair_id": pair.get("pair_id", len(outputs)),
                "expected": expected,
                "prediction": "" if match is None else match.group(),
                "raw_generation": text,
                "correct": bool(is_correct),
                "base_result": int(base["result"]),
                "source_result": int(source["result"]),
            }
        )
    return (correct / len(pairs) if pairs else 0.0), outputs


@torch.no_grad()
def autoregressive_image_outputs(model, processor, tokenizer, blocks, pairs: list[dict], hidden_deltas: torch.Tensor, data_root: Path, args: argparse.Namespace, description: str) -> tuple[float, list[dict]]:
    correct = 0
    outputs = []
    for pair, delta in zip(pairs, hidden_deltas):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(base, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.image_position)[0]
        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            length = int(inputs["attention_mask"][0].sum())
            outputs_ = patched_forward_with_deltas(
                model,
                inputs,
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            next_id = outputs_.logits[0, length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = next_id.item()
            inputs["attention_mask"][0, length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        is_correct = match is not None and match.group() == expected
        correct += int(is_correct)
        outputs.append(
            {
                "pair_id": pair.get("pair_id", len(outputs)),
                "expected": expected,
                "prediction": "" if match is None else match.group(),
                "raw_generation": text,
                "correct": bool(is_correct),
                "base_result": int(base["result"]),
                "source_result": int(source["result"]),
            }
        )
    return (correct / len(pairs) if pairs else 0.0), outputs


def evaluate_with_outputs(
    args: argparse.Namespace,
    *,
    model,
    processor,
    tokenizer,
    blocks,
    destination_task: str,
    destination_row: dict,
    destination_pairs: list[dict],
    hidden_deltas: torch.Tensor,
    description: str,
) -> tuple[float, list[dict]]:
    destination_modality, _operation = parse_task(destination_task)
    if destination_modality == "text":
        return autoregressive_text_outputs(model, tokenizer, blocks, destination_pairs, hidden_deltas, args, description)
    return autoregressive_image_outputs(
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


def pair_outputs_path(args: argparse.Namespace, row: dict) -> Path:
    return (
        args.output_dir
        / "autoregressive_outputs"
        / row["condition"]
        / (
            f"{row['source_label']}_seed{row['source_seed']}"
            f"_to_{row['destination_label']}_seed{row['destination_seed']}"
            f"_split{row['value_split_seed']}_layer{args.layer}_k{args.k}.jsonl"
        )
    )


def save_map_bundle(args: argparse.Namespace, row: dict, payload: dict) -> Path:
    path = (
        args.output_dir
        / "maps"
        / row["condition"]
        / (
            f"{row['source_label']}_seed{row['source_seed']}"
            f"_to_{row['destination_label']}_seed{row['destination_seed']}"
            f"_split{row['value_split_seed']}_layer{args.layer}_k{args.k}.pt"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


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
    split_seed: int,
    source_row: dict,
    destination_row: dict,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    maps: dict,
    train_values: list[int],
    test_values: list[int],
    pair_stats: dict,
    geometry: dict,
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
        fit_kind = "destination_self_das"
        alpha = 1.0
        alignment_path = None
        fit_metrics = None
        test_metrics = None
    elif condition == "raw_source_das_transfer":
        hidden_deltas = raw_source_transfer_deltas(
            args,
            activation_cache,
            sample_cache,
            destination_task=destination_task,
            destination_row=destination_row,
            destination_pairs=destination_pairs,
            source_basis=source_basis,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        fit_kind = "raw_source_das_basis_in_destination_space"
        alpha = 1.0
        alignment_path = None
        fit_metrics = None
        test_metrics = None
    else:
        fit = maps[condition]
        hidden_deltas = hidden_deltas_from_source_pairs(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            source_row=source_row,
            source_pairs=source_pairs,
            source_basis=source_basis,
            destination_basis=destination_basis,
            matrix=fit["matrix"],
            alpha=fit["alpha"],
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        fit_kind = fit["fit_kind"]
        alpha = fit["alpha"]
        fit_metrics = coordinate_metrics(geometry["x_train"], geometry["y_train"], fit["matrix"], alpha)
        test_metrics = coordinate_metrics(geometry["x_test"], geometry["y_test"], fit["matrix"], alpha)
        row_stub = {
            "condition": condition,
            "source_label": label(source_task),
            "source_seed": source_seed,
            "destination_label": label(destination_task),
            "destination_seed": destination_seed,
            "value_split_seed": split_seed,
        }
        alignment_path = save_map_bundle(
            args,
            row_stub,
            {
                "experiment": EXPERIMENT,
                "condition": condition,
                "condition_description": CONDITION_DESCRIPTIONS[condition],
                "fit_kind": fit_kind,
                "source_task": source_task,
                "source_seed": source_seed,
                "destination_task": destination_task,
                "destination_seed": destination_seed,
                "value_split_seed": split_seed,
                "train_values": train_values,
                "test_values": test_values,
                "matrix_source_to_destination": fit["matrix"].cpu(),
                "alpha": alpha,
                "source_basis": source_basis.cpu(),
                "destination_basis": destination_basis.cpu(),
                "config": jsonable(vars(args)),
            },
        )

    destination_control = control_iia(args, destination_modality, destination_operation, destination_seed, destination_row)
    destination_self = float(destination_row["autoregressive_iia"])
    description = f"{condition} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}] split={split_seed}"
    ar_iia, pair_outputs = evaluate_with_outputs(
        args,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        destination_task=destination_task,
        destination_row=destination_row,
        destination_pairs=destination_pairs,
        hidden_deltas=hidden_deltas,
        description=description,
    )
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
        "alignment_path": None if alignment_path is None else str(alignment_path),
        "layer": args.layer,
        "k": args.k,
        "hook": args.hook,
        "source_position": position_for(args, source_modality),
        "destination_position": position_for(args, destination_modality),
        "target": args.target,
        "value_split_seed": split_seed,
        "train_values": train_values,
        "test_values": test_values,
        "train_test_value_intersection": sorted(set(train_values) & set(test_values)),
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "pair_statistics": pair_stats,
        "n_pairs": len(destination_pairs),
        "alpha": alpha,
        "alignment_fit": fit_metrics,
        "alignment_fit_test": test_metrics,
        "autoregressive_iia": ar_iia,
        "destination_self_autoregressive_iia": destination_self,
        "control_condition": args.control_condition,
        "destination_control_autoregressive_iia": destination_control,
        "destination_normalized_transfer": normalized_transfer(ar_iia, destination_control, destination_self),
    }
    output_path = pair_outputs_path(args, row)
    save_jsonl(pair_outputs, output_path)
    row["autoregressive_outputs_path"] = str(output_path)
    print(
        f"    {condition}: AR IIA={ar_iia:.4f}; "
        f"norm={row['destination_normalized_transfer']}; pairs={len(destination_pairs)}"
    )
    return row


def value_projection(vectors: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> np.ndarray:
    return ((vectors.float() - mean) @ components).cpu().numpy()


def fit_pca(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    mean = vectors.float().mean(dim=0, keepdim=True)
    centered = vectors.float() - mean
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    variance = singular_values.square()
    ratio = variance / variance.sum().clamp_min(1e-12)
    return mean, components, {
        "explained_variance_ratio": [float(x) for x in ratio[:2]],
        "singular_values": [float(x) for x in singular_values[:2]],
    }


def save_generalization_plot(
    args: argparse.Namespace,
    *,
    source_task: str,
    destination_task: str,
    source_seed: int,
    destination_seed: int,
    split_seed: int,
    train_values: list[int],
    test_values: list[int],
    x_train_abs: torch.Tensor,
    y_train_abs: torch.Tensor,
    x_test_abs: torch.Tensor,
    y_test_abs: torch.Tensor,
    learned_map: dict,
) -> dict:
    x_origin = x_train_abs.mean(dim=0, keepdim=True)
    y_origin = y_train_abs.mean(dim=0, keepdim=True)
    x_train = x_train_abs - x_origin
    y_train = y_train_abs - y_origin
    x_test = x_test_abs - x_origin
    y_test = y_test_abs - y_origin
    transported_train = float(learned_map["alpha"]) * (x_train @ learned_map["matrix"])
    transported_test = float(learned_map["alpha"]) * (x_test @ learned_map["matrix"])

    raw_mean, raw_components, raw_pca = fit_pca(torch.cat([x_train, y_train], dim=0))
    aligned_mean, aligned_components, aligned_pca = fit_pca(torch.cat([transported_train, y_train], dim=0))

    raw_source_test = value_projection(x_test, raw_mean, raw_components)
    raw_destination_test = value_projection(y_test, raw_mean, raw_components)
    aligned_source_test = value_projection(transported_test, aligned_mean, aligned_components)
    aligned_destination_test = value_projection(y_test, aligned_mean, aligned_components)

    rows = []
    for panel, source_points, destination_points in [
        ("raw", raw_source_test, raw_destination_test),
        ("transported", aligned_source_test, aligned_destination_test),
    ]:
        for value, point in zip(test_values, source_points):
            rows.append(
                {
                    "panel": panel,
                    "kind": "source",
                    "source_task": source_task,
                    "destination_task": destination_task,
                    "source_seed": source_seed,
                    "destination_seed": destination_seed,
                    "value_split_seed": split_seed,
                    "result": value,
                    "pc1": float(point[0]),
                    "pc2": float(point[1]),
                }
            )
        for value, point in zip(test_values, destination_points):
            rows.append(
                {
                    "panel": panel,
                    "kind": "destination",
                    "source_task": source_task,
                    "destination_task": destination_task,
                    "source_seed": source_seed,
                    "destination_seed": destination_seed,
                    "value_split_seed": split_seed,
                    "result": value,
                    "pc1": float(point[0]),
                    "pc2": float(point[1]),
                }
            )

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"pca_{label(source_task)}_seed{source_seed}"
        f"_to_{label(destination_task)}_seed{destination_seed}_split{split_seed}"
    )
    json_path = args.output_dir / "pca_geometry" / f"{stem}.jsonl"
    save_jsonl(rows, json_path)

    values = np.asarray(test_values, dtype=float)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharex=False, sharey=False)
    fig.patch.set_facecolor("#fdfdfd")
    for ax, panel, title in zip(axes, ["raw", "transported"], ["Raw DAS coordinates", "Transported source into destination"]):
        panel_rows = [row for row in rows if row["panel"] == panel]
        source_rows = [row for row in panel_rows if row["kind"] == "source"]
        destination_rows = [row for row in panel_rows if row["kind"] == "destination"]
        for srow, drow in zip(source_rows, destination_rows):
            color = cmap(norm(srow["result"]))
            ax.plot([srow["pc1"], drow["pc1"]], [srow["pc2"], drow["pc2"]], color=color, alpha=0.38, linewidth=0.9)
            ax.scatter(srow["pc1"], srow["pc2"], marker="o", s=28, color=color, edgecolor="#202426", linewidth=0.55)
            ax.scatter(drow["pc1"], drow["pc2"], marker="^", s=32, color=color, edgecolor="#202426", linewidth=0.55)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, color="#d8ddde", linewidth=0.65, alpha=0.75)
        ax.set_facecolor("#f7f8f8")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#226a74", markeredgecolor="#202426", label="source"),
        plt.Line2D([], [], marker="^", linestyle="", color="#226a74", markeredgecolor="#202426", label="destination"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.035, pad=0.02)
    colorbar.set_label("held-out result")
    fig.suptitle(f"{source_task} -> {destination_task}, held-out value split {split_seed}", y=1.03)
    png_path = args.plot_dir / f"{stem}.png"
    pdf_path = args.plot_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {
        "source_task": source_task,
        "destination_task": destination_task,
        "source_seed": source_seed,
        "destination_seed": destination_seed,
        "value_split_seed": split_seed,
        "train_values": train_values,
        "test_values": test_values,
        "raw_pca": raw_pca,
        "transported_pca": aligned_pca,
        "rows_path": str(json_path),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(samples):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]


def summarize(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["source_task"], row["destination_task"], row["condition"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (source_task, destination_task, condition), parts in sorted(groups.items()):
        pair_values = []
        for row in parts:
            for item in load_jsonl(Path(row["autoregressive_outputs_path"])):
                pair_values.append(1.0 if item.get("correct") else 0.0)
        lo, hi = bootstrap_ci(
            pair_values,
            args.bootstrap_samples,
            stable_seed(args.model, source_task, destination_task, condition),
        )
        summaries.append(
            {
                "source_task": source_task,
                "source_label": label(source_task),
                "destination_task": destination_task,
                "destination_label": label(destination_task),
                "condition": condition,
                "condition_description": CONDITION_DESCRIPTIONS[condition],
                "n_rows": len(parts),
                "n_pair_outputs": len(pair_values),
                "autoregressive_iia_mean": mean([row["autoregressive_iia"] for row in parts]),
                "autoregressive_iia_std": sample_std([row["autoregressive_iia"] for row in parts]),
                "autoregressive_iia_pair_bootstrap_ci_low": lo,
                "autoregressive_iia_pair_bootstrap_ci_high": hi,
                "destination_normalized_transfer_mean": mean([row["destination_normalized_transfer"] for row in parts]),
                "destination_normalized_transfer_std": sample_std([row["destination_normalized_transfer"] for row in parts]),
                "alignment_test_cosine_mean": mean(
                    [
                        None if row["alignment_fit_test"] is None else row["alignment_fit_test"]["mean_cosine"]
                        for row in parts
                    ]
                ),
                "alignment_test_rmse_mean": mean(
                    [
                        None if row["alignment_fit_test"] is None else row["alignment_fit_test"]["root_mean_squared_error"]
                        for row in parts
                    ]
                ),
            }
        )
    return summaries


def cache_key(row: dict) -> tuple:
    return (
        row["condition"],
        row["source_task"],
        int(row["source_seed"]),
        row["destination_task"],
        int(row["destination_seed"]),
        int(row["value_split_seed"]),
    )


def load_cache(args: argparse.Namespace) -> dict[tuple, dict]:
    path = args.output_dir / "generalization_results.jsonl"
    if args.force or not path.exists():
        return {}
    return {cache_key(row): row for row in load_jsonl(path)}


def write_outputs(args: argparse.Namespace, rows: list[dict], plot_summaries: list[dict], seed_selection: dict) -> None:
    rows.sort(key=cache_key)
    save_jsonl(rows, args.output_dir / "generalization_results.jsonl")
    save_jsonl(summarize(rows, args), args.output_dir / "generalization_summary.jsonl")
    save_jsonl(plot_summaries, args.output_dir / "pca_geometry" / "pca_geometry_summary.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "claim_scope": (
                "DAS subspaces are frozen; value holdout tests generalization of "
                "the cross-modal transport map, not value-held-out DAS discovery."
            ),
            "condition_descriptions": CONDITION_DESCRIPTIONS,
            "seed_selection": seed_selection,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def main() -> None:
    args = parse_args()
    args.split_seed = args.das_split_seed
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    alignments = [parse_alignment(item) for item in args.alignments]
    args.tasks = list(dict.fromkeys(args.tasks + [task for pair in alignments for task in pair]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    seed_selection = selected_seeds(args)
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    spaces = task_rows_and_bases(args, hidden_size, args.tasks, seed_selection)
    activation_cache = {}
    sample_cache = {}
    cache = load_cache(args)
    rows = list(cache.values())
    plot_summaries = []

    print("Frozen-DAS value-held-out generalization")
    print(f"  tasks={args.tasks}")
    print(f"  alignments={args.alignments}")
    print(f"  conditions={args.conditions}")

    for split_seed in args.value_split_seed:
        print(f"\n=== value split seed {split_seed} ===")
        for source_task, destination_task in alignments:
            source_samples_all = cached_task_samples(sample_cache, source_task, spaces[(source_task, seed_selection[source_task][0])]["row"])
            destination_samples_all = cached_task_samples(
                sample_cache,
                destination_task,
                spaces[(destination_task, seed_selection[destination_task][0])]["row"],
            )
            shared_values = sorted(values_in_samples(args, source_samples_all) & values_in_samples(args, destination_samples_all))
            train_values, test_values = split_result_values(shared_values, args.train_value_fraction, split_seed)
            print(f"  {source_task} -> {destination_task}: train values={len(train_values)}, test values={len(test_values)}")

            for source_seed in seed_selection[source_task]:
                for destination_seed in seed_selection[destination_task]:
                    source_space = spaces[(source_task, source_seed)]
                    destination_space = spaces[(destination_task, destination_seed)]
                    source_row = source_space["row"]
                    destination_row = destination_space["row"]
                    source_basis = source_space["basis"]
                    destination_basis = destination_space["basis"]

                    source_train_centroids = centroids_for_values(
                        args,
                        activation_cache,
                        sample_cache,
                        task=source_task,
                        row=source_row,
                        basis=source_basis,
                        values=set(train_values),
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                    )
                    destination_train_centroids = centroids_for_values(
                        args,
                        activation_cache,
                        sample_cache,
                        task=destination_task,
                        row=destination_row,
                        basis=destination_basis,
                        values=set(train_values),
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                    )
                    source_test_centroids = centroids_for_values(
                        args,
                        activation_cache,
                        sample_cache,
                        task=source_task,
                        row=source_row,
                        basis=source_basis,
                        values=set(test_values),
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                    )
                    destination_test_centroids = centroids_for_values(
                        args,
                        activation_cache,
                        sample_cache,
                        task=destination_task,
                        row=destination_row,
                        basis=destination_basis,
                        values=set(test_values),
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                    )
                    train_values_present, x_train_abs, y_train_abs = centroid_matrices(
                        source_train_centroids, destination_train_centroids
                    )
                    test_values_present, x_test_abs, y_test_abs = centroid_matrices(
                        source_test_centroids, destination_test_centroids
                    )
                    maps = fit_maps(
                        args,
                        source_task=source_task,
                        source_seed=source_seed,
                        destination_task=destination_task,
                        destination_seed=destination_seed,
                        split_seed=split_seed,
                        x_train_abs=x_train_abs,
                        y_train_abs=y_train_abs,
                    )
                    geometry = {
                        "x_train": maps["_origins"]["centered_x_train"],
                        "y_train": maps["_origins"]["centered_y_train"],
                        "x_test": x_test_abs - maps["_origins"]["source"],
                        "y_test": y_test_abs - maps["_origins"]["destination"],
                    }

                    if not args.plot_first_seed_only or (source_seed == seed_selection[source_task][0] and destination_seed == seed_selection[destination_task][0]):
                        plot_summaries.append(
                            save_generalization_plot(
                                args,
                                source_task=source_task,
                                destination_task=destination_task,
                                source_seed=source_seed,
                                destination_seed=destination_seed,
                                split_seed=split_seed,
                                train_values=train_values_present,
                                test_values=test_values_present,
                                x_train_abs=x_train_abs,
                                y_train_abs=y_train_abs,
                                x_test_abs=x_test_abs,
                                y_test_abs=y_test_abs,
                                learned_map=maps["rotation_plus_scaling"],
                            )
                        )

                    destination_samples = cached_task_samples(sample_cache, destination_task, destination_row)
                    destination_pairs, pair_stats = heldout_value_pairs(args, destination_samples, set(test_values_present), split_seed)
                    source_pairs = matched_pair_samples(
                        destination_pairs,
                        cached_task_samples(sample_cache, source_task, source_row),
                    )

                    for condition in args.conditions:
                        key = (condition, source_task, source_seed, destination_task, destination_seed, split_seed)
                        if key in cache:
                            print(f"    cached {condition} {source_seed}->{destination_seed} split={split_seed}")
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
                                split_seed=split_seed,
                                source_row=source_row,
                                destination_row=destination_row,
                                source_basis=source_basis,
                                destination_basis=destination_basis,
                                source_pairs=source_pairs,
                                destination_pairs=destination_pairs,
                                maps=maps,
                                train_values=train_values_present,
                                test_values=test_values_present,
                                pair_stats=pair_stats,
                                geometry=geometry,
                                model=model,
                                processor=processor,
                                tokenizer=tokenizer,
                                blocks=blocks,
                                model_name=model_name,
                            )
                        )
                        write_outputs(args, rows, plot_summaries, seed_selection)

    write_outputs(args, rows, plot_summaries, seed_selection)
    print(f"\nSaved rows: {args.output_dir / 'generalization_results.jsonl'}")
    print(f"Saved summary: {args.output_dir / 'generalization_summary.jsonl'}")
    print(f"Saved PCA plots under: {args.plot_dir}")


if __name__ == "__main__":
    main()
