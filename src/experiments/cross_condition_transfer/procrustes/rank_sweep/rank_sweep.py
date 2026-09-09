"""Causal rank sweep over matched Procrustes displacement modes.

Alpha is fit once on the full-rank scaled displacement map and reused for every
truncated rank. This keeps the curve faithful to information lost by removing
matched modes instead of letting each rank retune its intervention magnitude.
"""

from __future__ import annotations

import argparse
import statistics
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
    parse_alignment,
    scaled_alpha,
)


EXPERIMENT = "rank_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--causal_transfer_rows", type=Path, default=Path("results/final_exps/causal_tranfer/transfer_results.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/paper/procrustes/rank_sweep"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--alignments", nargs="+", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--ranks", type=int, nargs="*", default=None, help="Defaults to every rank 1..k.")
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


def rank_values(args: argparse.Namespace) -> list[int]:
    ranks = args.ranks if args.ranks is not None and len(args.ranks) else range(1, args.k + 1)
    ranks = sorted(set(int(rank) for rank in ranks))
    invalid = [rank for rank in ranks if rank < 1 or rank > args.k]
    if invalid:
        raise ValueError(f"Ranks must be between 1 and k={args.k}, got {invalid}.")
    return ranks


def procrustes_svd(source_z: torch.Tensor, destination_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cross_covariance = source_z.T @ destination_z
    u, singular_values, vh = torch.linalg.svd(cross_covariance, full_matrices=False)
    return u, singular_values, vh


def rank_transport(u: torch.Tensor, vh: torch.Tensor, rank: int) -> torch.Tensor:
    return u[:, :rank] @ vh[:rank, :]


def svd_fraction(singular_values: torch.Tensor, rank: int) -> tuple[float | None, float | None]:
    total = float(singular_values.sum())
    if total <= 1e-12:
        return None, None
    return float(singular_values[rank - 1] / total), float(singular_values[:rank].sum() / total)


def row_key(row: dict) -> tuple:
    return (
        row["source_task"],
        int(row["source_seed"]),
        row["destination_task"],
        int(row["destination_seed"]),
        int(row["rank"]),
    )


def load_cache(args: argparse.Namespace) -> dict[tuple, dict]:
    path = args.output_dir / "rank_sweep_results.jsonl"
    if args.force or not path.exists():
        return {}
    cache = {}
    for row in load_jsonl(path):
        if row.get("alpha_fit_kind") != "full_rank_scaled_displacement":
            continue
        cache[row_key(row)] = row
    return cache


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["source_task"], row["destination_task"], int(row["rank"]))
        groups.setdefault(key, []).append(row)

    summary = []
    for (source_task, destination_task, rank), parts in sorted(groups.items()):
        summary.append(
            {
                "source_task": source_task,
                "source_label": label(source_task),
                "destination_task": destination_task,
                "destination_label": label(destination_task),
                "rank": rank,
                "rank_fraction": mean([row["rank_fraction"] for row in parts]),
                "n": len(parts),
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
                "destination_normalized_transfer_gain_over_causal_std": sample_std(
                    [row["destination_normalized_transfer_gain_over_causal"] for row in parts]
                ),
                "alpha_mean": mean([row["alpha"] for row in parts]),
                "alignment_test_cosine_mean": mean(
                    [row["alignment_fit_test"]["mean_cosine"] for row in parts]
                ),
                "alignment_test_rmse_mean": mean(
                    [row["alignment_fit_test"]["root_mean_squared_error"] for row in parts]
                ),
                "cumulative_singular_value_fraction_mean": mean(
                    [row["cumulative_singular_value_fraction"] for row in parts]
                ),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, rows: list[dict], seed_selection: dict[str, list[int]], ranks: list[int]) -> None:
    rows.sort(key=row_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, args.output_dir / "rank_sweep_results.jsonl")
    save_jsonl(summarize(rows), args.output_dir / "rank_sweep_summary.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "description": (
                "Causal recovery as a function of matched Procrustes displacement rank. "
                "The scalar alpha is fit once at full rank and reused for every rank."
            ),
            "seed_selection": seed_selection,
            "alignments": args.alignments,
            "ranks": ranks,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def save_svd_bundle(
    args: argparse.Namespace,
    *,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    u: torch.Tensor,
    singular_values: torch.Tensor,
    vh: torch.Tensor,
    full_rank_alpha: float,
    rank_payloads: dict[int, dict],
) -> Path:
    path = (
        args.output_dir
        / "svd_transports"
        / (
            f"rank_sweep_{clean_name(source_task, source_seed)}"
            f"_to_{clean_name(destination_task, destination_seed)}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "procrustes_rank_sweep_svd_transport",
            "source_task": source_task,
            "source_seed": source_seed,
            "destination_task": destination_task,
            "destination_seed": destination_seed,
            "layer": args.layer,
            "k": args.k,
            "hook": args.hook,
            "target": args.target,
            "source_basis": source_basis.cpu(),
            "destination_basis": destination_basis.cpu(),
            "cross_covariance_left_singular_vectors": u.cpu(),
            "singular_values": singular_values.cpu(),
            "cross_covariance_right_singular_vectors_h": vh.cpu(),
            "alpha": full_rank_alpha,
            "alpha_fit_kind": "full_rank_scaled_displacement",
            "rank_transports": rank_payloads,
            "config": jsonable(vars(args)),
        },
        path,
    )
    return path


def evaluate_alignment(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    causal_rows: dict,
    cache: dict[tuple, dict],
    *,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    ranks: list[int],
    model,
    processor,
    tokenizer,
    blocks,
    hidden_size: int,
    model_name: str,
) -> list[dict]:
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
    destination_control = control_iia(args, destination_modality, destination_operation, destination_seed, destination_row)
    destination_self = float(destination_row["autoregressive_iia"])
    causal = causal_rows.get((source_task, source_seed, destination_task, destination_seed), {})
    causal_iia = causal.get("autoregressive_iia")
    causal_normalized = causal.get("destination_normalized_transfer")

    print(f"  fitting displacement SVD for {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]")
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
    u, singular_values, vh = procrustes_svd(x_train, y_train)
    full_rank_q = rank_transport(u, vh, args.k)
    full_rank_alpha = scaled_alpha(x_train, y_train, full_rank_q)

    rank_payloads = {}
    prepared = {}
    for rank in ranks:
        q = rank_transport(u, vh, rank)
        singular_fraction, cumulative_fraction = svd_fraction(singular_values, rank)
        fit_metrics = coordinate_metrics(x_train, y_train, q, full_rank_alpha)
        heldout_metrics = coordinate_metrics(x_test, y_test, q, full_rank_alpha)
        prepared[rank] = {
            "q": q,
            "alpha": full_rank_alpha,
            "alpha_fit_kind": "full_rank_scaled_displacement",
            "singular_value_fraction": singular_fraction,
            "cumulative_singular_value_fraction": cumulative_fraction,
            "fit_metrics": fit_metrics,
            "heldout_metrics": heldout_metrics,
        }
        rank_payloads[rank] = {
            "Q_rank_source_to_destination": q.cpu(),
            "alpha": full_rank_alpha,
            "alpha_fit_kind": "full_rank_scaled_displacement",
            "singular_value": float(singular_values[rank - 1]),
            "singular_value_fraction": singular_fraction,
            "cumulative_singular_value_fraction": cumulative_fraction,
            "metrics": fit_metrics,
            "heldout_metrics": heldout_metrics,
        }
    svd_path = save_svd_bundle(
        args,
        source_task=source_task,
        source_seed=source_seed,
        destination_task=destination_task,
        destination_seed=destination_seed,
        source_basis=source_basis,
        destination_basis=destination_basis,
        u=u,
        singular_values=singular_values,
        vh=vh,
        full_rank_alpha=full_rank_alpha,
        rank_payloads=rank_payloads,
    )

    rows = []
    for rank in ranks:
        key = (source_task, source_seed, destination_task, destination_seed, rank)
        if key in cache:
            rows.append(cache[key])
            print(f"    cached rank {rank}/{args.k}")
            continue

        item = prepared[rank]
        deltas = hidden_deltas_for_pairs(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            source_row=source_row,
            destination_pairs=destination_pairs,
            q=item["q"],
            alpha=item["alpha"],
            source_basis=source_basis,
            destination_basis=destination_basis,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        description = (
            f"rank {rank}/{args.k} {source_task}[{source_seed}]"
            f" -> {destination_task}[{destination_seed}]"
        )
        if destination_modality == "text":
            aligned_iia = autoregressive_iia_text_aligned(
                model, tokenizer, blocks, destination_pairs, deltas, args, description
            )
        else:
            aligned_iia = autoregressive_iia_image_aligned(
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

        normalized = normalized_transfer(aligned_iia, destination_control, destination_self)
        row = {
            "model": model_name,
            "experiment": EXPERIMENT,
            "fit_kind": "rank_truncated_scaled_base_to_donor_displacements",
            "source_task": source_task,
            "source_modality": source_modality,
            "source_operation": source_operation,
            "source_seed": source_seed,
            "source_subspace_path": str(subspace_path(args, source_modality, source_operation, source_seed)),
            "destination_task": destination_task,
            "destination_modality": destination_modality,
            "destination_operation": destination_operation,
            "destination_seed": destination_seed,
            "destination_results_path": str(
                results_path(args, destination_modality, destination_operation, args.condition, destination_seed)
            ),
            "destination_heldout_pairs_path": str(
                heldout_pairs_path(args, destination_modality, destination_operation, destination_seed)
            ),
            "svd_transport_path": str(svd_path),
            "layer": args.layer,
            "k": args.k,
            "rank": rank,
            "rank_fraction": rank / args.k,
            "hook": args.hook,
            "source_position": position_for(args, source_modality),
            "destination_position": position_for(args, destination_modality),
            "target": args.target,
            "n_pairs": len(destination_pairs),
            "alpha": item["alpha"],
            "alpha_fit_kind": item["alpha_fit_kind"],
            "singular_value": float(singular_values[rank - 1]),
            "singular_value_fraction": item["singular_value_fraction"],
            "cumulative_singular_value_fraction": item["cumulative_singular_value_fraction"],
            "alignment_fit": item["fit_metrics"],
            "alignment_fit_test": item["heldout_metrics"],
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
            f"    rank {rank:02d}/{args.k}: alpha={item['alpha']:.4f}; "
            f"AR IIA={aligned_iia:.4f}; normalized={normalized}; "
            f"cum sv={item['cumulative_singular_value_fraction']}"
        )
        rows.append(row)
    return rows


def merge_rows(cache: dict[tuple, dict], rows: list[dict]) -> list[dict]:
    merged = dict(cache)
    for row in rows:
        merged[row_key(row)] = row
    return list(merged.values())


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    alignments = [parse_alignment(item) for item in args.alignments]
    args.tasks = list(dict.fromkeys(args.tasks + [task for pair in alignments for task in pair]))
    ranks = rank_values(args)
    seed_selection = selected_seeds(args)
    causal_rows = load_causal_rows(args.causal_transfer_rows)

    print("Rank-sweep Procrustes experiment")
    print(f"Ranks: {ranks}")
    print("Alignment directions:")
    for source, destination in alignments:
        print(f"  {source} -> {destination}")
    print("Selected seeds:")
    for task in args.tasks:
        print(f"  {task}: {seed_selection[task]}")

    cache = load_cache(args)
    rows = []
    tasks_for_model = args.tasks
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, tasks_for_model)
    activation_cache = {}
    sample_cache = {}

    for source_task, destination_task in alignments:
        print(f"\nDirection: {source_task} -> {destination_task}")
        for source_seed in seed_selection[source_task]:
            for destination_seed in seed_selection[destination_task]:
                rows.extend(
                    evaluate_alignment(
                        args,
                        activation_cache,
                        sample_cache,
                        causal_rows,
                        cache,
                        source_task=source_task,
                        source_seed=source_seed,
                        destination_task=destination_task,
                        destination_seed=destination_seed,
                        ranks=ranks,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        hidden_size=hidden_size,
                        model_name=model_name,
                    )
                )
                write_outputs(args, merge_rows(cache, rows), seed_selection, ranks)

    final_rows = merge_rows(cache, rows)
    write_outputs(args, final_rows, seed_selection, ranks)
    print(f"\nSaved rows: {args.output_dir / 'rank_sweep_results.jsonl'}")
    print(f"Saved summary: {args.output_dir / 'rank_sweep_summary.jsonl'}")
    print(f"Saved SVD transports under: {args.output_dir / 'svd_transports'}")


if __name__ == "__main__":
    main()
