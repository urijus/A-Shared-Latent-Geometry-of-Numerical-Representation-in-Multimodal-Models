"""Train and evaluate DAS on image-rendered arithmetic prompts."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    TARGET_COMPONENTS,
    build_unique_pairs,
    dataset_statistics,
    load_subspaces,
    pca_principal_space,
    random_subspaces,
    shuffled_donors,
    split_samples,
)
from src.experiments.arithmetic_reference.das.text.train_das import (
    file_fingerprint,
    load_existing_rows,
    pair_sample_ids,
    prefixed,
    unique_existing_values,
)
from src.experiments.arithmetic_reference.das.text.plot_results import plot_das_results
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    autoregressive_iia_image,
    collect_initialization_features_image,
    evaluate_teacher_forced_image,
    image_path_for,
    load_rgb_image,
    print_position_summary,
    sample_prompt,
    train_subspace_image,
)
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument(
        "--data_root",
        type=Path,
        help="Root for relative image_path values. Defaults to data_path.parent.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument(
        "--target",
        choices=["result", "c1_hat", "c1_hat_full", "c0_hat", "c0"],
        required=True,
    )
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--position",
        default="-1",
        help=(
            "Raw image-prompt token position to patch. Non-negative values count "
            "from the beginning; negative values count from the end, so -1 is "
            "the final input token."
        ),
    )
    parser.add_argument(
        "--position_strategy",
        choices=["last_input", "last_image_token"],
        default=None,
        help="Deprecated compatibility option. Prefer --position -1.",
    )
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--max_train_pairs", type=int, default=4096)
    parser.add_argument("--max_validation_pairs", type=int, default=512)
    parser.add_argument("--max_test_pairs", type=int, default=512)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=64)
    parser.add_argument("--min_train_pairs", type=int, default=1000)
    parser.add_argument(
        "--initialization",
        choices=["random_pca", "random"],
        default="random_pca",
    )
    parser.add_argument("--pca_max_samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument(
        "--prompt",
        default="Solve the arithmetic expression in the image. Output ONLY a number.",
    )
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def config_key(args, layer, dimension):
    config = {
        "model": args.model,
        "data_path": str(args.data_path.resolve()),
        "data_root": str(args.data_root.resolve()),
        "operation": args.operation,
        "target": args.target,
        "layer": layer,
        "position": str(args.position),
        "hook": args.hook,
        "dimension": dimension,
        "max_train_pairs": args.max_train_pairs,
        "max_validation_pairs": args.max_validation_pairs,
        "max_test_pairs": args.max_test_pairs,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "patience": args.patience,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_autoregressive_pairs": args.max_autoregressive_pairs,
        "data_fingerprint": args.data_fingerprint,
        "initialization": args.initialization,
        "pca_max_samples": args.pca_max_samples,
        "pca_variance_threshold": args.pca_variance_threshold,
        "prompt": args.prompt,
        "enable_thinking": args.enable_thinking,
    }
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def paths(args):
    run_dir = args.output_dir / "runs"
    subspace_dir = args.output_dir / "subspaces"
    plot_dir = args.output_dir / "plots"
    for directory in (run_dir, subspace_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.operation}_{args.target}_pos{args.position}_{args.hook}_"
        f"init{args.initialization}_seed{args.seed}"
    )
    return run_dir / f"{stem}.jsonl", subspace_dir, plot_dir / stem


def checkpoint_path(subspace_dir, args, layer, dimension, key):
    return subspace_dir / (
        f"{args.operation}_{args.target}_layer{layer}_pos{args.position}_"
        f"{args.hook}_init{args.initialization}_k{dimension}_seed{args.seed}_{key}.pt"
    )


def compatible_existing_row(existing, args, layer, dimension):
    candidates = []
    for row in existing.values():
        if (
            row.get("operation") == args.operation
            and row.get("target") == args.target
            and int(row.get("layer", -1)) == int(layer)
            and str(row.get("position")) == str(args.position)
            and row.get("hook") == args.hook
            and int(row.get("k", -1)) == int(dimension)
            and int(row.get("seed", -1)) == int(args.seed)
        ):
            checkpoint_value = row.get("checkpoint_path")
            if checkpoint_value and Path(checkpoint_value).exists():
                candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda row: str(row.get("run_id", "")))


def evaluate_with_controls(
    model,
    processor,
    tokenizer,
    blocks,
    learned,
    args,
    layer,
    dimension,
    hidden_size,
    test_pairs,
    experiment_seed,
):
    autoregressive_pairs = (
        test_pairs[: args.max_autoregressive_pairs]
        if args.max_autoregressive_pairs > 0
        else test_pairs
    )
    common = dict(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        layers=[layer],
        hook_name=args.hook,
        pairs=test_pairs,
        data_root=args.data_root,
        prompt=args.prompt,
        position_strategy=args.position,
        target=args.target,
        enable_thinking=args.enable_thinking,
        batch_size=args.batch_size,
    )
    learned_metrics = evaluate_teacher_forced_image(subspaces=learned, **common)
    learned_metrics["autoregressive_iia"] = autoregressive_iia_image(
        subspaces=learned,
        pairs=autoregressive_pairs,
        max_new_tokens=args.max_new_tokens,
        description=f"image learned layer={layer} k={dimension}",
        **{
            name: value
            for name, value in common.items()
            if name not in {"batch_size", "pairs"}
        },
    )

    random_space = random_subspaces(
        hidden_size, [layer], dimension, model.device, experiment_seed + 1
    )
    random_metrics = evaluate_teacher_forced_image(subspaces=random_space, **common)
    random_metrics["autoregressive_iia"] = autoregressive_iia_image(
        subspaces=random_space,
        pairs=autoregressive_pairs,
        max_new_tokens=args.max_new_tokens,
        description=f"image random layer={layer} k={dimension}",
        **{
            name: value
            for name, value in common.items()
            if name not in {"batch_size", "pairs"}
        },
    )

    shuffled_pairs = shuffled_donors(test_pairs, experiment_seed + 2)
    shuffled_common = dict(common, pairs=shuffled_pairs)
    shuffled_metrics = evaluate_teacher_forced_image(
        subspaces=learned, **shuffled_common
    )
    shuffled_metrics["autoregressive_iia"] = autoregressive_iia_image(
        subspaces=learned,
        pairs=shuffled_pairs[: len(autoregressive_pairs)],
        max_new_tokens=args.max_new_tokens,
        description=f"image shuffled layer={layer} k={dimension}",
        **{
            name: value
            for name, value in shuffled_common.items()
            if name not in {"batch_size", "pairs"}
        },
    )
    return learned_metrics, random_metrics, shuffled_metrics


def main():
    args = parse_args()
    if args.position_strategy is not None:
        args.position = args.position_strategy
    args.data_root = args.data_root or args.data_path.parent
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    args.data_fingerprint = file_fingerprint(args.data_path)

    metrics_path, subspace_dir, plot_stem = paths(args)
    existing = load_existing_rows(metrics_path, args)
    experiments = [
        (layer, dimension, config_key(args, layer, dimension))
        for layer in args.layers
        for dimension in args.dimensions
    ]
    experiment_rows = {}
    pending = []
    for layer, dimension, key in experiments:
        checkpoint_value = existing.get(key, {}).get("checkpoint_path")
        if key not in existing or not checkpoint_value or not Path(checkpoint_value).exists():
            fallback = compatible_existing_row(existing, args, layer, dimension)
            if fallback is None:
                pending.append((layer, dimension, key))
            else:
                existing[key] = fallback
                experiment_rows[key] = fallback
        else:
            experiment_rows[key] = existing[key]
    if not pending:
        print("All requested image DAS experiments already exist.")
        plot_das_results([existing[key] for _, _, key in experiments], plot_stem)
        return

    samples = load_jsonl(args.data_path)
    train_samples, validation_samples, test_samples = split_samples(
        samples, args.train_fraction, args.validation_fraction, args.seed
    )
    train_pairs, train_stats = build_unique_pairs(
        train_samples, args.target, args.seed, args.max_train_pairs
    )
    validation_pairs, validation_stats = build_unique_pairs(
        validation_samples, args.target, args.seed + 1, args.max_validation_pairs
    )
    test_pairs, test_stats = build_unique_pairs(
        test_samples, args.target, args.seed + 2, args.max_test_pairs
    )
    if len(train_pairs) < args.min_train_pairs:
        raise ValueError(
            f"Only {len(train_pairs)} unique training pairs are available; "
            f"requested minimum is {args.min_train_pairs}."
        )
    if not validation_pairs or not test_pairs:
        raise ValueError("No held-out image DAS pairs are available.")
    pair_stats = {
        "split_mode": "sample_disjoint_then_unique_pairs",
        "train": train_stats,
        "validation": validation_stats,
        "test": test_stats,
        "n_train_pairs": len(train_pairs),
        "n_validation_pairs": len(validation_pairs),
        "n_test_pairs": len(test_pairs),
        "n_autoregressive_test_pairs": min(
            len(test_pairs), args.max_autoregressive_pairs
        ) if args.max_autoregressive_pairs > 0 else len(test_pairs),
    }
    train_ids = pair_sample_ids(train_pairs)
    validation_ids = pair_sample_ids(validation_pairs)
    test_ids = pair_sample_ids(test_pairs)
    pair_stats.update(
        {
            "n_train_samples": len(train_ids),
            "n_validation_samples": len(validation_ids),
            "n_test_samples": len(test_ids),
            "train_validation_sample_overlap": len(train_ids & validation_ids),
            "train_test_sample_overlap": len(train_ids & test_ids),
        }
    )
    print("Dataset statistics:", dataset_statistics(samples))
    print("Pair statistics:", pair_stats)

    model_path, model_name = resolve_model_for_loading(args.model)
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, args.layers)
    hidden_size = get_hidden_size(model)
    first_sample = samples[0]
    print_position_summary(
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        text=sample_prompt(processor, first_sample, args.prompt, args.enable_thinking),
        image=load_rgb_image(image_path_for(first_sample, args.data_root)),
        position=args.position,
    )

    initialization_spaces = None
    if args.initialization == "random_pca":
        pca_samples = []
        seen_ids = set()
        for pair in train_pairs:
            for sample in (pair["base"], pair["source"]):
                identity = str(sample.get("sample_id", (sample.get("a"), sample.get("b"))))
                if identity not in seen_ids:
                    seen_ids.add(identity)
                    pca_samples.append(sample)
        pca_layers = sorted({layer for layer, _, _ in pending})
        n_pca_samples = (
            min(len(pca_samples), args.pca_max_samples)
            if args.pca_max_samples
            else len(pca_samples)
        )
        print(
            f"Collecting image PCA initialization from {n_pca_samples} "
            f"unique training samples for layers {pca_layers}."
        )
        initialization_features = collect_initialization_features_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            samples=pca_samples,
            layers=pca_layers,
            hook_name=args.hook,
            data_root=args.data_root,
            prompt=args.prompt,
            position_strategy=args.position,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            max_samples=args.pca_max_samples,
        )
        initialization_spaces = {
            layer: pca_principal_space(features, args.pca_variance_threshold)
            for layer, features in initialization_features.items()
        }
        del initialization_features

    for layer, dimension, key in pending:
        experiment_seed = args.seed + layer * 100 + dimension
        checkpoint = checkpoint_path(subspace_dir, args, layer, dimension, key)
        if checkpoint.exists():
            print(f"Loading existing image subspace: {checkpoint}")
            subspaces = load_subspaces(
                checkpoint, [layer], hidden_size, dimension, model.device
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            best_epoch = payload.get("best_epoch")
            history = payload.get("history")
        else:
            print(f"Training image DAS layer={layer}, k={dimension}")
            subspaces, best_epoch, history = train_subspace_image(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
                train_pairs=train_pairs,
                validation_pairs=validation_pairs,
                layer=layer,
                hook_name=args.hook,
                data_root=args.data_root,
                prompt=args.prompt,
                position_strategy=args.position,
                target=args.target,
                dimension=dimension,
                hidden_size=hidden_size,
                enable_thinking=args.enable_thinking,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                seed=experiment_seed,
                initialization=args.initialization,
                initialization_space=(
                    initialization_spaces[layer]
                    if initialization_spaces is not None
                    else None
                ),
                gradient_clip_norm=args.gradient_clip_norm,
            )
            torch.save(
                {
                    "run_id": key,
                    "bases": {layer: subspaces[str(layer)].basis().detach().cpu()},
                    "best_epoch": best_epoch,
                    "history": history,
                    "config": vars(args),
                    "pair_statistics": pair_stats,
                },
                checkpoint,
            )

        learned, random_control, shuffled_control = evaluate_with_controls(
            model, processor, tokenizer, blocks, subspaces, args, layer, dimension,
            hidden_size, test_pairs, experiment_seed,
        )
        row = {
            "run_id": key,
            "model": model_name,
            "data_path": str(args.data_path),
            "data_root": str(args.data_root),
            "data_fingerprint": args.data_fingerprint,
            "input_format": "image",
            "operation": args.operation,
            "target": args.target,
            "target_component": TARGET_COMPONENTS[args.target],
            "pairing_control": {
                "result": "different_result_same_answer_length",
                "c1_hat": "same_result_mod_100_different_c1_hat",
                "c1_hat_full": "mixed_answer_source_c1_hat_base_c0",
                "c0_hat": "same_p1_different_c0_hat",
                "c0": "same_c1_hat_different_c0",
            }[args.target],
            "layer": layer,
            "layers": [layer],
            "position": str(args.position),
            "position_strategy": args.position_strategy,
            "hook": args.hook,
            "subspace_dim": dimension,
            "k": dimension,
            "seed": args.seed,
            "experiment_seed": experiment_seed,
            "best_epoch": best_epoch,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_clip_norm": args.gradient_clip_norm,
            "max_autoregressive_pairs": args.max_autoregressive_pairs,
            "initialization": args.initialization,
            "pca_max_samples": args.pca_max_samples,
            "pca_variance_threshold": args.pca_variance_threshold,
            "prompt": args.prompt,
            "enable_thinking": args.enable_thinking,
            "checkpoint_path": str(checkpoint),
            "dataset_statistics": dataset_statistics(samples),
            "pair_statistics": pair_stats,
            "training_history": history,
        }
        row.update(learned)
        row.update(prefixed("random_", random_control))
        row.update(prefixed("shuffled_donor_", shuffled_control))
        existing[key] = row
        experiment_rows[key] = row
        save_jsonl(
            sorted(unique_existing_values(existing), key=lambda item: (item["layer"], item["k"])),
            metrics_path,
        )
        plot_das_results(
            [
                experiment_rows[item_key]
                for _, _, item_key in experiments
                if item_key in experiment_rows
            ],
            plot_stem,
        )
        print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
