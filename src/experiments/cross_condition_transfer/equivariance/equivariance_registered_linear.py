"""Registered-hub linear +1 equivariance experiment.

This is the cleaned-up version of the original G_{+1} idea.  It freezes DAS,
freezes synchronization, fits one domain translation b_d from training values,
then fits a linear +1 operator using text:addition only in the registered hub:

    h_tilde_d(y) = (z_d(y) R_d) / s_d - b_d
    G_{+1} h_tilde_T+(y) ~= h_tilde_T+(y + 1).

The frozen operator is evaluated geometrically and causally in all domains on
held-out values, with the same top-k diagnostics used by the oracle audit.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    jsonable,
    label,
    normalized_transfer,
    parse_task,
    save_json,
    save_jsonl,
    task_key,
)
from src.experiments.cross_condition_transfer.equivariance.equivariance import (
    BASE_MODELS,
    CONTROL_MODELS,
    FIT_TASK,
    TASKS,
    PlusOneMap,
    apply_plus_map,
    carry_flag,
    causal_metrics_for_deltas,
    common_experiment_seeds,
    edge_tensors,
    fit_plus_maps,
    geometry_metrics,
    load_model_bundle,
    load_sync,
    make_spaces,
    plus_pairs_for_task,
    sample_key,
    selected_matching_seeds,
    spectral_stats,
    value_split,
)
from src.experiments.cross_condition_transfer.equivariance.equivariance_affine import (
    alignment_sanity,
    fit_domain_translations,
    raw_shared_centroids,
    translated_centroids,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    activations_for_ids,
    cached_task_samples,
)


EXPERIMENT = "registered_hub_linear_plus_one_equivariance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--sync_dir", type=Path, default=Path("results/final_exps/synchronization"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/equivariance_registered_linear"))
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
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def evaluate_geometry_registered(centroids_by_task, plus_map: PlusOneMap, *, task: str, starts: list[int], step: int) -> list[dict]:
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


def hidden_deltas_for_map_registered(args, activation_cache, sample_cache, *, space, pairs, plus_map, step, translation, model, processor, tokenizer, blocks):
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
        base_h = (base_z @ space.orientation) / float(space.scale) - translation
        target_h = apply_plus_map(base_h, plus_map, step)
        target_z = float(space.scale) * ((target_h + translation) @ space.orientation.T)
    return (target_z - base_z) @ space.basis.T


def evaluate_causal_registered(args, activation_cache, sample_cache, *, space, pairs, plus_map, step, translation, oracle_iia, control_iia, model, processor, tokenizer, blocks) -> dict:
    hidden_deltas = hidden_deltas_for_map_registered(
        args,
        activation_cache,
        sample_cache,
        space=space,
        pairs=pairs,
        plus_map=plus_map,
        step=step,
        translation=translation,
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
        "equivariance_recovery": normalized_transfer(iia, control_iia, oracle_iia) if oracle_iia is not None and control_iia is not None else None,
        "carry_pairs": sum(1 for pair in pairs if pair["carry"] == "carry"),
        "non_carry_pairs": sum(1 for pair in pairs if pair["carry"] == "non_carry"),
    }


def save_registered_maps(args, seed: int, maps: dict[str, PlusOneMap], fit_stats: dict, split: dict, sync_path: Path, translations: dict[str, torch.Tensor], translation_stats: dict) -> Path:
    path = args.output_dir / "maps" / f"registered_plus_one_maps_seed{seed}_layer{args.layer}_k{args.k}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "registered_hub_linear_plus_one_maps",
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
            "domain_translations": {task: offset.cpu() for task, offset in translations.items()},
            "translation_stats": translation_stats,
            "fit_stats": fit_stats,
            "value_split": split,
            "synchronization_path": str(sync_path),
            "config": jsonable(vars(args)),
        },
        path,
    )
    return path


def summarize_geometry(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["model"], row["step"], row["subset"]), []).append(row)
    return [
        {
            "task": task,
            "task_label": label(task),
            "model": model,
            "step": step,
            "subset": subset,
            "n": len(parts),
            "mean_cosine_mean": mean([row["mean_cosine"] for row in parts]),
            "normalized_mse_mean": mean([row["normalized_mse"] for row in parts]),
            "nearest_result_centroid_accuracy_mean": mean([row["nearest_result_centroid_accuracy"] for row in parts]),
        }
        for (task, model, step, subset), parts in sorted(groups.items())
    ]


def summarize_causal(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["model"], row["step"]), []).append(row)
    return [
        {
            "task": task,
            "task_label": label(task),
            "model": model,
            "step": step,
            "n": len(parts),
            "autoregressive_exact_match_mean": mean([row["autoregressive_exact_match"] for row in parts]),
            "autoregressive_exact_match_std": sample_std([row["autoregressive_exact_match"] for row in parts]),
            "target_tokens_all_topk_mean": mean([row.get("target_tokens_all_topk") for row in parts]),
            "target_tokens_topk_fraction_mean": mean([row.get("target_tokens_topk_fraction") for row in parts]),
            "equivariance_recovery_mean": mean([row["equivariance_recovery"] for row in parts]),
            "equivariance_topk_recovery_mean": mean([row.get("equivariance_topk_recovery") for row in parts]),
            "equivariance_topk_fraction_recovery_mean": mean([row.get("equivariance_topk_fraction_recovery") for row in parts]),
        }
        for (task, model, step), parts in sorted(groups.items())
    ]


def write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, translation_rows, seed_selection):
    save_jsonl(geometry_rows, args.output_dir / "geometry_results.jsonl")
    save_jsonl(summarize_geometry(geometry_rows), args.output_dir / "geometry_summary.jsonl")
    save_jsonl(causal_rows, args.output_dir / "causal_results.jsonl")
    save_jsonl(summarize_causal(causal_rows), args.output_dir / "causal_summary.jsonl")
    save_jsonl(fit_rows, args.output_dir / "fit_summary.jsonl")
    save_jsonl(spectral_rows, args.output_dir / "spectral_summary.jsonl")
    save_jsonl(sanity_rows, args.output_dir / "shared_space_sanity.jsonl")
    save_jsonl(translation_rows, args.output_dir / "domain_translation_summary.jsonl")
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


def selected_maps_from_args(maps: dict[str, PlusOneMap], args: argparse.Namespace) -> dict[str, PlusOneMap]:
    selected = {name: item for name, item in maps.items() if name in args.models}
    selected["oracle_target_state"] = PlusOneMap("oracle_target_state", "oracle", fit_task="none")
    return selected


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
    geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, translation_rows = [], [], [], [], [], []

    print("Registered-hub linear +1 equivariance")
    for seed in seeds:
        print(f"\nSeed {seed}")
        sync = load_sync(args, seed)
        spaces = make_spaces(args, args.tasks, seed, hidden_size, sync)
        raw_centroids = {}
        for task, space in spaces.items():
            raw_centroids[task] = raw_shared_centroids(
                args,
                activation_cache,
                sample_cache,
                space=space,
                samples=cached_task_samples(sample_cache, task, space.row),
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
            )

        split = value_split(args, raw_centroids[args.fit_task])
        train_values = sorted(set(split["train_starts"]) | {value + 1 for value in split["train_starts"]})
        translations, translation_stats = fit_domain_translations(raw_centroids, train_values)
        centroids_by_task = translated_centroids(raw_centroids, translations)
        heldout_values = sorted(set(split["test_starts"]) | {value + 1 for value in split["test_starts"]})
        sanity_rows.append(
            {
                "seed": seed,
                "value_split": split,
                "train_registration": translation_stats,
                "heldout_before_registration": alignment_sanity(raw_centroids, {task: torch.zeros_like(offset) for task, offset in translations.items()}, heldout_values),
                "heldout_after_registration": alignment_sanity(raw_centroids, translations, heldout_values),
            }
        )
        translation_rows.append(
            {
                "seed": seed,
                "synchronization_path": str(sync["path"]),
                "translation_stats": translation_stats,
                "translation_norms": {task: float(offset.norm()) for task, offset in translations.items()},
            }
        )

        maps, fit_stats = fit_plus_maps(args, centroids_by_task[args.fit_task], split, seed=seed, fit_task=args.fit_task)
        selected_maps = selected_maps_from_args(maps, args)
        maps_path = save_registered_maps(args, seed, selected_maps, fit_stats, split, sync["path"], translations, translation_stats)
        fit_rows.append(
            {
                "seed": seed,
                "maps_path": str(maps_path),
                "synchronization_path": str(sync["path"]),
                "value_split": split,
                "translation_stats": translation_stats,
                **fit_stats,
            }
        )
        spectral_rows.extend({"seed": seed, **spectral_stats(plus_map)} for plus_map in selected_maps.values())

        for step in args.geometry_steps:
            for task in args.tasks:
                for plus_map in selected_maps.values():
                    geometry_rows.extend(
                        {"seed": seed, "maps_path": str(maps_path), **row}
                        for row in evaluate_geometry_registered(centroids_by_task, plus_map, task=task, starts=split["test_starts"], step=step)
                    )

        for step in args.causal_steps:
            for task, space in spaces.items():
                pairs = plus_pairs_for_task(args, sample_cache, space=space, starts=split["test_starts"], step=step)
                if not pairs:
                    print(f"  no +{step} causal pairs for {task}[{seed}]")
                    continue
                oracle = selected_maps["oracle_target_state"]
                identity = selected_maps.get("identity", PlusOneMap("identity", "identity", fit_task=args.fit_task))
                oracle_row = evaluate_causal_registered(
                    args, activation_cache, sample_cache, space=space, pairs=pairs, plus_map=oracle,
                    step=step, translation=translations[task], oracle_iia=None, control_iia=None,
                    model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
                )
                identity_row = evaluate_causal_registered(
                    args, activation_cache, sample_cache, space=space, pairs=pairs, plus_map=identity,
                    step=step, translation=translations[task], oracle_iia=None, control_iia=None,
                    model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
                )
                oracle_iia = oracle_row["autoregressive_exact_match"]
                control_iia = identity_row["autoregressive_exact_match"]
                oracle_topk = oracle_row["target_tokens_all_topk"]
                control_topk = identity_row["target_tokens_all_topk"]
                oracle_topk_fraction = oracle_row["target_tokens_topk_fraction"]
                control_topk_fraction = identity_row["target_tokens_topk_fraction"]
                for row in [oracle_row, identity_row]:
                    row["oracle_autoregressive_exact_match"] = oracle_iia
                    row["control_autoregressive_exact_match"] = control_iia
                    row["oracle_target_tokens_all_topk"] = oracle_topk
                    row["control_target_tokens_all_topk"] = control_topk
                    row["oracle_target_tokens_topk_fraction"] = oracle_topk_fraction
                    row["control_target_tokens_topk_fraction"] = control_topk_fraction
                    row["equivariance_recovery"] = normalized_transfer(row["autoregressive_exact_match"], control_iia, oracle_iia)
                    row["equivariance_topk_recovery"] = normalized_transfer(row["target_tokens_all_topk"], control_topk, oracle_topk)
                    row["equivariance_topk_fraction_recovery"] = normalized_transfer(row["target_tokens_topk_fraction"], control_topk_fraction, oracle_topk_fraction)
                    causal_rows.append({"maps_path": str(maps_path), "value_split": split, **row})
                for plus_map in selected_maps.values():
                    if plus_map.name in {"oracle_target_state", "identity"}:
                        continue
                    row = evaluate_causal_registered(
                        args, activation_cache, sample_cache, space=space, pairs=pairs, plus_map=plus_map,
                        step=step, translation=translations[task], oracle_iia=oracle_iia, control_iia=control_iia,
                        model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
                    )
                    row["oracle_target_tokens_all_topk"] = oracle_topk
                    row["control_target_tokens_all_topk"] = control_topk
                    row["oracle_target_tokens_topk_fraction"] = oracle_topk_fraction
                    row["control_target_tokens_topk_fraction"] = control_topk_fraction
                    row["equivariance_topk_recovery"] = normalized_transfer(row["target_tokens_all_topk"], control_topk, oracle_topk)
                    row["equivariance_topk_fraction_recovery"] = normalized_transfer(row["target_tokens_topk_fraction"], control_topk_fraction, oracle_topk_fraction)
                    causal_rows.append({"maps_path": str(maps_path), "value_split": split, **row})
                write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, translation_rows, seed_selection)
        write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, translation_rows, seed_selection)

    write_outputs(args, geometry_rows, causal_rows, fit_rows, spectral_rows, sanity_rows, translation_rows, seed_selection)
    print(f"\nSaved registered linear equivariance results under: {args.output_dir}")


if __name__ == "__main__":
    main()
