"""Backfill missing controls for the existing Procrustes result grid.

This keeps the old results intact and writes two additional variant folders
with the same row/matrix schema:

* unrestricted_linear
* shuffled_value_correspondences
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    DEFAULT_TASKS,
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
    DEFAULT_ALIGNMENTS,
    autoregressive_iia_image_aligned,
    autoregressive_iia_text_aligned,
    clean_name,
    coordinate_metrics,
    data_root_for,
    displacement_alignment_data,
    gain,
    hidden_deltas_for_pairs,
    load_causal_rows,
    load_heldout_pairs,
    load_model_bundle,
    matrix_payload,
    parse_alignment,
    orthogonal_procrustes,
    scaled_alpha,
    stable_seed,
)


VARIANT_DESCRIPTIONS = {
    "unrestricted_linear": "unconstrained least-squares linear map fit on base-to-donor displacements",
    "shuffled_value_correspondences": "scaled orthogonal map fit after shuffling train displacement correspondences",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--causal_transfer_rows", type=Path, default=Path("results/final_exps/causal_tranfer/transfer_results.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/procrustes"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--alignments", nargs="+", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_DESCRIPTIONS), default=list(VARIANT_DESCRIPTIONS))
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
    parser.add_argument("--max_alignment_eval_samples", type=int, default=2048)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def unrestricted_linear_map(source_z: torch.Tensor, destination_z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(source_z.float(), destination_z.float()).solution.to(dtype=source_z.dtype)


def variant_dir(args: argparse.Namespace, variant: str) -> Path:
    return args.output_dir / variant


def row_key(row: dict) -> tuple:
    return (
        row["source_task"],
        int(row["source_seed"]),
        row["destination_task"],
        int(row["destination_seed"]),
    )


def load_cache(args: argparse.Namespace, variant: str) -> dict[tuple, dict]:
    path = variant_dir(args, variant) / "procrustes_results.jsonl"
    if args.force or not path.exists():
        return {}
    return {row_key(row): row for row in load_jsonl(path)}


def save_map(args: argparse.Namespace, variant: str, row: dict, matrix: torch.Tensor, alpha: float, source_basis: torch.Tensor, destination_basis: torch.Tensor, fit_metrics: dict, test_metrics: dict) -> Path:
    path = (
        variant_dir(args, variant)
        / "matrices"
        / (
            f"{variant}_{clean_name(row['source_task'], int(row['source_seed']))}"
            f"_to_{clean_name(row['destination_task'], int(row['destination_seed']))}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "das_coordinate_alignment_backfilled_control",
            "variant": variant,
            "variant_description": VARIANT_DESCRIPTIONS[variant],
            "fit_kind": row["fit_kind"],
            "matrix_source_to_destination": matrix.cpu(),
            "alpha": alpha,
            "source_basis": source_basis.cpu(),
            "destination_basis": destination_basis.cpu(),
            "source_task": row["source_task"],
            "source_seed": row["source_seed"],
            "destination_task": row["destination_task"],
            "destination_seed": row["destination_seed"],
            "layer": args.layer,
            "k": args.k,
            "hook": args.hook,
            "target": args.target,
            "metrics": fit_metrics,
            "heldout_metrics": test_metrics,
            "config": jsonable(vars(args)),
        },
        path,
    )
    return path


def write_variant_outputs(args: argparse.Namespace, variant: str, rows: list[dict], seed_selection: dict[str, list[int]]) -> None:
    rows.sort(key=row_key)
    out = variant_dir(args, variant)
    save_jsonl(rows, out / "procrustes_results.jsonl")
    save_json(
        {
            "variant": variant,
            "variant_description": VARIANT_DESCRIPTIONS[variant],
            "seed_selection": seed_selection,
            "alignments": args.alignments,
            "config": jsonable(vars(args)),
        },
        out / "manifest.json",
    )
    metrics = [
        "autoregressive_iia",
        "destination_normalized_transfer",
        "autoregressive_iia_gain_over_causal",
        "destination_normalized_transfer_gain_over_causal",
        "alignment_fit_test.mean_cosine",
        "alignment_fit_test.root_mean_squared_error",
        "alpha",
    ]
    for metric in metrics:
        payload = matrix_payload(rows, args.tasks, metric)
        name = metric.replace(".", "_")
        save_json(payload, out / f"{name}_matrix.json")
        save_jsonl(payload["rows"], out / f"{name}_matrix.jsonl")


def fit_control(args: argparse.Namespace, variant: str, x_train: torch.Tensor, y_train: torch.Tensor) -> tuple[torch.Tensor, float, str]:
    if variant == "unrestricted_linear":
        return unrestricted_linear_map(x_train, y_train), 1.0, "base_to_donor_displacements_unrestricted_linear"
    order = list(range(y_train.shape[0]))
    random.Random(stable_seed(args.model, args.alignment_seed, variant, y_train.shape[0])).shuffle(order)
    shuffled_y = y_train[order]
    q = orthogonal_procrustes(x_train, shuffled_y)
    return q, scaled_alpha(x_train, shuffled_y, q), "base_to_donor_displacements_shuffled_correspondences"


def evaluate_one(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    causal_rows: dict,
    *,
    variant: str,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    model,
    processor,
    tokenizer,
    blocks,
    hidden_size: int,
    model_name: str,
) -> dict:
    source_modality, source_operation = parse_task(source_task)
    destination_modality, destination_operation = parse_task(destination_task)
    source_row = first_result_row(args, source_modality, source_operation, args.condition, source_seed)
    destination_row = first_result_row(args, destination_modality, destination_operation, args.condition, destination_seed)
    source_basis = load_basis(subspace_path(args, source_modality, source_operation, source_seed), args.layer, hidden_size, args.k)
    destination_basis = load_basis(subspace_path(args, destination_modality, destination_operation, destination_seed), args.layer, hidden_size, args.k)
    destination_pairs = load_heldout_pairs(
        args,
        destination_modality,
        destination_operation,
        destination_seed,
        destination_row,
        args.max_autoregressive_pairs,
    )
    x_train, y_train, x_test, y_test = displacement_alignment_data(
        args,
        activation_cache,
        sample_cache,
        source_task=source_task,
        destination_task=destination_task,
        source_row=source_row,
        destination_row=destination_row,
        source_basis=source_basis,
        destination_basis=destination_basis,
        destination_pairs=destination_pairs,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    matrix, alpha, fit_kind = fit_control(args, variant, x_train, y_train)
    fit_metrics = coordinate_metrics(x_train, y_train, matrix, alpha)
    test_metrics = coordinate_metrics(x_test, y_test, matrix, alpha)
    deltas = hidden_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        source_task=source_task,
        source_row=source_row,
        destination_pairs=destination_pairs,
        q=matrix,
        alpha=alpha,
        source_basis=source_basis,
        destination_basis=destination_basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    description = f"{variant} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]"
    if destination_modality == "text":
        ar_iia = autoregressive_iia_text_aligned(model, tokenizer, blocks, destination_pairs, deltas, args, description)
    else:
        ar_iia = autoregressive_iia_image_aligned(
            model,
            processor,
            tokenizer,
            blocks,
            destination_pairs,
            deltas,
            data_root_for(destination_row),
            args,
            description,
        )

    destination_control = control_iia(args, destination_modality, destination_operation, destination_seed, destination_row)
    destination_self = float(destination_row["autoregressive_iia"])
    causal = causal_rows.get((source_task, source_seed, destination_task, destination_seed), {})
    normalized = normalized_transfer(ar_iia, destination_control, destination_self)
    row = {
        "model": model_name,
        "variant": variant,
        "variant_description": VARIANT_DESCRIPTIONS[variant],
        "fit_kind": fit_kind,
        "fit_source_task": source_task,
        "fit_source_seed": source_seed,
        "fit_destination_task": destination_task,
        "fit_destination_seed": destination_seed,
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
        "layer": args.layer,
        "k": args.k,
        "hook": args.hook,
        "source_position": position_for(args, source_modality),
        "destination_position": position_for(args, destination_modality),
        "target": args.target,
        "n_pairs": len(destination_pairs),
        "alpha": alpha,
        "alignment_fit": fit_metrics,
        "alignment_fit_test": test_metrics,
        "autoregressive_iia": ar_iia,
        "destination_self_autoregressive_iia": destination_self,
        "control_condition": args.control_condition,
        "destination_control_autoregressive_iia": destination_control,
        "destination_normalized_transfer": normalized,
        "causal_autoregressive_iia": causal.get("autoregressive_iia"),
        "causal_destination_normalized_transfer": causal.get("destination_normalized_transfer"),
        "autoregressive_iia_gain_over_causal": gain(ar_iia, causal.get("autoregressive_iia")),
        "destination_normalized_transfer_gain_over_causal": gain(normalized, causal.get("destination_normalized_transfer")),
    }
    row["alignment_path"] = str(save_map(args, variant, row, matrix, alpha, source_basis, destination_basis, fit_metrics, test_metrics))
    print(f"  {variant} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]: AR IIA={ar_iia:.4f}; norm={normalized}")
    return row


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
    seed_selection = selected_seeds(args)
    causal_rows = load_causal_rows(args.causal_transfer_rows)
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    activation_cache = {}
    sample_cache = {}

    for variant in args.variants:
        print(f"\n=== Backfill {variant} ===")
        cache = load_cache(args, variant)
        rows = []
        for source_task, destination_task in alignments:
            for source_seed in seed_selection[source_task]:
                for destination_seed in seed_selection[destination_task]:
                    key = (source_task, source_seed, destination_task, destination_seed)
                    if key in cache:
                        rows.append(cache[key])
                        print(f"  cached {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]")
                        continue
                    rows.append(
                        evaluate_one(
                            args,
                            activation_cache,
                            sample_cache,
                            causal_rows,
                            variant=variant,
                            source_task=source_task,
                            source_seed=source_seed,
                            destination_task=destination_task,
                            destination_seed=destination_seed,
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            blocks=blocks,
                            hidden_size=hidden_size,
                            model_name=model_name,
                        )
                    )
                    write_variant_outputs(args, variant, merge(cache, rows), seed_selection)
        write_variant_outputs(args, variant, merge(cache, rows), seed_selection)
        print(f"Saved {variant} rows under {variant_dir(args, variant)}")


if __name__ == "__main__":
    main()
