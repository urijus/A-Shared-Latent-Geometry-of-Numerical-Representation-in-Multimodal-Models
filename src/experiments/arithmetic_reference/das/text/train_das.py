"""Train and evaluate the arithmetic-reference DAS experiments.

Reusable intervention logic lives in :mod:`src.interventions.das`; this module
defines the command-line experiment orchestration and preserves the historical
result schema.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    TARGET_COMPONENTS,
    autoregressive_iia,
    build_unique_pairs,
    collect_initialization_features,
    dataset_statistics,
    evaluate_teacher_forced,
    load_subspaces,
    pca_principal_space,
    print_patch_diagnostics,
    random_subspaces,
    shuffled_donors,
    split_samples,
    train_subspace,
)
from src.experiments.arithmetic_reference.das.text.plot_results import plot_das_results
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    uses_chat_template,
)
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    resolve_model_for_loading,
    validate_block_layers,
)


class FullActivationPatch:
    def patch(self, _base_vector, source_vector):
        return source_vector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument(
        "--target",
        choices=["result", "c1_hat", "c1_hat_full", "c0_hat", "c0"],
        required=True,
    )
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--position", default="17")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32])
    parser.add_argument(
        "--max_train_pairs", type=int, default=4096,
        help="Maximum unique training pairs; 0 means use every valid pair.",
    )
    parser.add_argument(
        "--max_validation_pairs", type=int, default=512,
        help="Maximum unique validation pairs; 0 means use every valid pair.",
    )
    parser.add_argument(
        "--max_test_pairs", type=int, default=512,
        help="Maximum unique test pairs; 0 means use every valid pair.",
    )
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument(
        "--split_mode",
        choices=["sample_disjoint", "pair_disjoint"],
        default="sample_disjoint",
        help=(
            "sample_disjoint holds prompts out before pairing. pair_disjoint "
            "samples counterfactual pairs first, closer to the GoodFire setup."
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="Clip DAS gradient norm; 0 disables clipping.",
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--max_autoregressive_pairs",
        type=int,
        default=128,
        help="Maximum test pairs for each autoregressive metric; 0 means all.",
    )
    parser.add_argument(
        "--run_full_activation_control",
        action="store_true",
        help="Also evaluate full residual activation patching as a diagnostic.",
    )
    parser.add_argument("--min_train_pairs", type=int, default=1000)
    parser.add_argument(
        "--initialization",
        choices=["random_pca", "random"],
        default="random_pca",
    )
    parser.add_argument("--pca_max_samples", type=int, default=3584)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument("--use_chat_template", action="store_true")
    return parser.parse_args()


def config_key(args, layer, dimension):
    config = {
        "model": args.model,
        "data_path": str(args.data_path.resolve()),
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
        "split_mode": args.split_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "patience": args.patience,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_autoregressive_pairs": args.max_autoregressive_pairs,
        "data_fingerprint": args.data_fingerprint,
        "use_chat_template": args.use_chat_template,
        "initialization": args.initialization,
        "pca_max_samples": args.pca_max_samples,
        "pca_variance_threshold": args.pca_variance_threshold,
    }
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


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
    return (
        run_dir / f"{stem}.jsonl",
        subspace_dir,
        plot_dir / stem,
    )


def checkpoint_path(subspace_dir, args, layer, dimension, key):
    return subspace_dir / (
        f"{args.operation}_{args.target}_layer{layer}_pos{args.position}_"
        f"{args.hook}_init{args.initialization}_k{dimension}_"
        f"seed{args.seed}_{key}.pt"
    )


def load_rows(path):
    if not path.exists():
        return {}
    return {
        row["run_id"]: row
        for row in load_jsonl(path)
        if isinstance(row, dict) and "run_id" in row
    }


def legacy_metrics_paths(metrics_path, args):
    paths = [metrics_path]
    clip_label = str(args.gradient_clip_norm).replace(".", "p")
    path_text = str(metrics_path)
    seed_segment = f"_seed{args.seed}"
    clip_segment = f"_clip{clip_label}_seed{args.seed}"
    if clip_segment in path_text:
        paths.append(Path(path_text.replace(f"_clip{clip_label}_", "_")))
    elif seed_segment in path_text:
        paths.append(Path(path_text.replace(seed_segment, clip_segment)))
    return list(dict.fromkeys(paths))


def load_existing_rows(metrics_path, args):
    rows = {}
    for path in legacy_metrics_paths(metrics_path, args):
        for run_id, row in load_rows(path).items():
            rows.setdefault(run_id, row)
    return rows


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
    return max(
        candidates,
        key=lambda row: (
            row.get("max_autoregressive_pairs") is not None,
            row.get("gradient_clip_norm") is not None,
            str(row.get("run_id", "")),
        ),
    )


def unique_existing_values(existing):
    rows = {}
    for row in existing.values():
        rows[row["run_id"]] = row
    return rows.values()


def prefixed(prefix, metrics):
    return {f"{prefix}{name}": value for name, value in metrics.items()}


def file_fingerprint(path):
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def pair_sample_ids(pairs):
    return {
        str(item.get("sample_id", (item.get("a"), item.get("b"))))
        for pair in pairs
        for item in (pair["base"], pair["source"])
    }


def make_pair_splits(samples, args):
    if args.split_mode == "sample_disjoint":
        train_samples, validation_samples, test_samples = split_samples(
            samples, args.train_fraction, args.validation_fraction, args.seed
        )
        train_pairs, train_stats = build_unique_pairs(
            train_samples, args.target, args.seed, args.max_train_pairs
        )
        validation_pairs, validation_stats = build_unique_pairs(
            validation_samples,
            args.target,
            args.seed + 1,
            args.max_validation_pairs,
        )
        test_pairs, test_stats = build_unique_pairs(
            test_samples, args.target, args.seed + 2, args.max_test_pairs
        )
    else:
        total_pairs = (
            max(0, args.max_train_pairs)
            + max(0, args.max_validation_pairs)
            + max(0, args.max_test_pairs)
        )
        if total_pairs <= 0:
            raise ValueError("pair_disjoint split requires positive pair caps.")
        pairs, all_stats = build_unique_pairs(
            samples, args.target, args.seed, total_pairs
        )
        train_end = min(args.max_train_pairs, len(pairs))
        validation_end = min(
            train_end + args.max_validation_pairs,
            len(pairs),
        )
        train_pairs = pairs[:train_end]
        validation_pairs = pairs[train_end:validation_end]
        test_pairs = pairs[validation_end:validation_end + args.max_test_pairs]
        train_stats = {
            **all_stats,
            "selected_unique_pairs": len(train_pairs),
            "pair_slice": "train",
        }
        validation_stats = {
            **all_stats,
            "selected_unique_pairs": len(validation_pairs),
            "pair_slice": "validation",
        }
        test_stats = {
            **all_stats,
            "selected_unique_pairs": len(test_pairs),
            "pair_slice": "test",
        }
    return train_pairs, validation_pairs, test_pairs, train_stats, validation_stats, test_stats


def evaluate_with_controls(
    model,
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
        tokenizer=tokenizer,
        blocks=blocks,
        layers=[layer],
        hook_name=args.hook,
        pairs=test_pairs,
        position=args.position,
        target=args.target,
        use_chat_template=args.use_chat_template,
        batch_size=args.batch_size,
    )
    learned_metrics = evaluate_teacher_forced(subspaces=learned, **common)
    learned_metrics["autoregressive_iia"] = autoregressive_iia(
        subspaces=learned,
        pairs=autoregressive_pairs,
        max_new_tokens=args.max_new_tokens,
        description=f"learned layer={layer} k={dimension}",
        **{
            name: value
            for name, value in common.items()
            if name not in {"batch_size", "pairs"}
        },
    )

    full_metrics = None
    if args.run_full_activation_control:
        full_patch = {str(layer): FullActivationPatch()}
        full_metrics = evaluate_teacher_forced(subspaces=full_patch, **common)
        full_metrics["autoregressive_iia"] = autoregressive_iia(
            subspaces=full_patch,
            pairs=autoregressive_pairs,
            max_new_tokens=args.max_new_tokens,
            description=f"full activation patch layer={layer}",
            **{
                name: value
                for name, value in common.items()
                if name not in {"batch_size", "pairs"}
            },
        )

    random_space = random_subspaces(
        hidden_size,
        [layer],
        dimension,
        model.device,
        experiment_seed + 1,
    )
    random_metrics = evaluate_teacher_forced(subspaces=random_space, **common)
    random_metrics["autoregressive_iia"] = autoregressive_iia(
        subspaces=random_space,
        pairs=autoregressive_pairs,
        max_new_tokens=args.max_new_tokens,
        description=f"random layer={layer} k={dimension}",
        **{
            name: value
            for name, value in common.items()
            if name not in {"batch_size", "pairs"}
        },
    )

    shuffled_pairs = shuffled_donors(test_pairs, experiment_seed + 2)
    shuffled_common = dict(common, pairs=shuffled_pairs)
    shuffled_metrics = evaluate_teacher_forced(subspaces=learned, **shuffled_common)
    shuffled_metrics["autoregressive_iia"] = autoregressive_iia(
        subspaces=learned,
        pairs=shuffled_pairs[: len(autoregressive_pairs)],
        max_new_tokens=args.max_new_tokens,
        description=f"shuffled donor layer={layer} k={dimension}",
        **{
            name: value
            for name, value in shuffled_common.items()
            if name not in {"batch_size", "pairs"}
        },
    )
    return learned_metrics, random_metrics, shuffled_metrics, full_metrics


def main():
    args = parse_args()
    if not 0 < args.pca_variance_threshold <= 1:
        raise ValueError("--pca_variance_threshold must be in (0, 1].")
    if args.pca_max_samples < 0:
        raise ValueError("--pca_max_samples must be non-negative; 0 means all.")
    if args.max_autoregressive_pairs < 0:
        raise ValueError("--max_autoregressive_pairs must be non-negative; 0 means all.")
    if args.gradient_clip_norm < 0:
        raise ValueError("--gradient_clip_norm must be non-negative; 0 disables it.")
    args.use_chat_template = args.use_chat_template or uses_chat_template(args.model)
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
    for item in experiments:
        layer, dimension = item[0], item[1]
        key = item[2]
        checkpoint_value = existing.get(key, {}).get("checkpoint_path")
        if key not in existing or not checkpoint_value or not Path(checkpoint_value).exists():
            fallback = compatible_existing_row(existing, args, layer, dimension)
            if fallback is None:
                pending.append(item)
            else:
                existing[key] = fallback
                experiment_rows[key] = fallback
                print(
                    f"Reusing compatible existing subspace for layer={layer}, "
                    f"k={dimension}: {fallback['checkpoint_path']}"
                )
        else:
            experiment_rows[key] = existing[key]
    if not pending:
        print("All requested experiments already exist.")
        plot_das_results([existing[key] for _, _, key in experiments], plot_stem)
        return

    samples = load_jsonl(args.data_path)
    (
        train_pairs,
        validation_pairs,
        test_pairs,
        train_stats,
        validation_stats,
        test_stats,
    ) = make_pair_splits(
        samples, args
    )
    if not validation_pairs or not test_pairs:
        raise ValueError(
            f"No held-out pairs: validation={len(validation_pairs)}, "
            f"test={len(test_pairs)}."
        )
    pair_stats = {
        "split_mode": args.split_mode,
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
    if len(train_pairs) < args.min_train_pairs:
        raise ValueError(
            f"Only {len(train_pairs)} unique training pairs are available; "
            f"the requested minimum is {args.min_train_pairs}."
        )
    print("Dataset statistics:", dataset_statistics(samples))
    print("Pair statistics:", pair_stats)

    model_path, model_name = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, args.layers)
    hidden_size = get_hidden_size(model)

    print_patch_diagnostics(
        tokenizer,
        [("base", train_pairs[0]["base"]), ("source", train_pairs[0]["source"])],
        args.use_chat_template,
        args.position,
    )

    initialization_spaces = None
    if args.initialization == "random_pca":
        pca_samples = []
        seen_ids = set()
        for pair in train_pairs:
            for sample in (pair["base"], pair["source"]):
                identity = str(
                    sample.get("sample_id", (sample.get("a"), sample.get("b")))
                )
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
            f"Collecting PCA initialization from {n_pca_samples} unique "
            f"training samples for layers {pca_layers}."
        )
        initialization_features = collect_initialization_features(
            model,
            tokenizer,
            blocks,
            pca_samples,
            pca_layers,
            args.hook,
            args.position,
            args.use_chat_template,
            args.batch_size,
            args.pca_max_samples,
        )
        initialization_spaces = {
            layer: pca_principal_space(
                features, args.pca_variance_threshold
            )
            for layer, features in initialization_features.items()
        }
        del initialization_features

    for layer, dimension, key in pending:
        experiment_seed = args.seed + layer * 100 + dimension
        checkpoint = checkpoint_path(
            subspace_dir, args, layer, dimension, key
        )
        history = None
        if checkpoint.exists():
            print(f"Loading existing subspace: {checkpoint}")
            subspaces = load_subspaces(
                checkpoint, [layer], hidden_size, dimension, model.device
            )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            best_epoch = payload.get("best_epoch")
            history = payload.get("history")
        else:
            print(f"Training layer={layer}, k={dimension}")
            subspaces, best_epoch, history = train_subspace(
                model,
                tokenizer,
                blocks,
                train_pairs,
                validation_pairs,
                layer,
                args.hook,
                args.position,
                args.target,
                dimension,
                hidden_size,
                args.use_chat_template,
                args.epochs,
                args.batch_size,
                args.learning_rate,
                args.patience,
                experiment_seed,
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
                    "bases": {
                        layer: subspaces[str(layer)].basis().detach().cpu()
                    },
                    "best_epoch": best_epoch,
                    "history": history,
                    "config": vars(args),
                    "pair_statistics": pair_stats,
                },
                checkpoint,
            )

        learned, random_control, shuffled_control, full_control = evaluate_with_controls(
            model,
            tokenizer,
            blocks,
            subspaces,
            args,
            layer,
            dimension,
            hidden_size,
            test_pairs,
            experiment_seed,
        )
        row = {
            "run_id": key,
            "model": model_name,
            "data_path": str(args.data_path),
            "data_fingerprint": args.data_fingerprint,
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
            "use_chat_template": args.use_chat_template,
            "checkpoint_path": str(checkpoint),
            "dataset_statistics": dataset_statistics(samples),
            "pair_statistics": pair_stats,
            "training_history": history,
        }
        row.update(learned)
        row.update(prefixed("random_", random_control))
        row.update(prefixed("shuffled_donor_", shuffled_control))
        if full_control is not None:
            row.update(prefixed("full_activation_", full_control))
        existing[key] = row
        experiment_rows[key] = row
        save_jsonl(
            sorted(unique_existing_values(existing), key=lambda item: (item["layer"], item["k"])),
            metrics_path,
        )
        plot_das_results(
            [experiment_rows[item_key] for _, _, item_key in experiments if item_key in experiment_rows],
            plot_stem,
        )
        print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
