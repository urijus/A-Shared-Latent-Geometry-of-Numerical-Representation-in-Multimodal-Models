import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.common.io import load_jsonl, save_jsonl
from src.experiments.arithmetic_reference.linear_probes.splitting import make_split_indices
from src.probes.linear import (
    TOP_K_VALUES,
    evaluate_probe,
    shuffle_labels,
    train_and_evaluate_probe,
)
from src.models import validate_saved_block_layers


MODALITIES = ["addition", "subtraction", "multiplication"]


def result_key(modality, target, layer, position_name, shuffled, split_mode, split_id):
    return (
        modality,
        target,
        int(layer),
        str(position_name),
        bool(shuffled),
        str(split_mode),
        str(split_id or ""),
    )


def row_result_key(row):
    return result_key(
        modality=row["modality"],
        target=row["target"],
        layer=row["layer"],
        position_name=row["position_name"],
        shuffled=row.get("shuffle_train_labels", False),
        split_mode=row.get("split_mode", "random"),
        split_id=row.get("split_id", ""),
    )


def position_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def sort_result_rows(rows, modality_names, target_names):
    modality_order = {name: index for index, name in enumerate(modality_names)}
    target_order = {name: index for index, name in enumerate(target_names)}
    return sorted(
        rows,
        key=lambda row: (
            modality_order.get(row["modality"], len(modality_order)),
            row["modality"],
            target_order.get(row["target"], len(target_order)),
            row["target"],
            int(row["layer"]),
            position_sort_key(row["position_name"]),
            bool(row.get("shuffle_train_labels", False)),
            row.get("split_mode", "random"),
            row.get("split_id", ""),
        ),
    )


def print_activation_position_token_examples(data):
    examples = data.get("position_token_examples") or []
    if not examples:
        return
    print("Activation token examples:")
    for example_index, example in enumerate(examples):
        print(f"  example {example_index}: {example.get('rendered_prompt')!r}")
        print("    policy:", example.get("multi_subtoken_policy", "unknown"))
        for selected in example.get("selected_positions", []):
            print(
                "    "
                f"{selected.get('position_name')}: "
                f"semantic {selected.get('semantic_index')}:"
                f"{selected.get('semantic_token')!r} -> "
                f"HF {selected.get('hf_position')}, "
                f"token={selected.get('hf_token')!r}"
            )


def labels_to_tensor(labels, target):
    return torch.tensor(
        [int(row[target]) for row in labels],
        dtype=torch.long,
    )


def majority_baseline(y):
    values, counts = torch.unique(y, return_counts=True)
    return counts.max().item() / y.numel()


def make_class_weight(y, n_classes):
    counts = torch.bincount(y, minlength=n_classes).float()
    weights = torch.zeros(n_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def parse_int_values(values):
    if values is None:
        return None
    parsed = []
    for value in values:
        parsed.extend(int(item) for item in str(value).replace(",", " ").split())
    return parsed


def split_indices_by_c(labels, train_c_values, val_c_values):
    train_c = set(parse_int_values(train_c_values) or [])
    val_c = set(parse_int_values(val_c_values) or [])
    if not train_c or not val_c:
        raise ValueError("c_holdout requires --train_c_values and --val_c_values.")
    overlap = train_c & val_c
    if overlap:
        raise ValueError(f"Train/validation c groups overlap: {sorted(overlap)}")

    train_idx = []
    val_idx = []
    ignored = 0
    for index, row in enumerate(labels):
        c = int(row["c"])
        if c in train_c:
            train_idx.append(index)
        elif c in val_c:
            val_idx.append(index)
        else:
            ignored += 1

    if not train_idx or not val_idx:
        raise ValueError(
            f"Empty c_holdout split: train={len(train_idx)}, val={len(val_idx)}."
        )

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        {
            "split": "c_holdout",
            "train_c_values": sorted(train_c),
            "val_c_values": sorted(val_c),
            "ignored_examples": ignored,
        },
    )


def make_probe_split(labels, split_mode, val_fraction, seed, train_c_values, val_c_values):
    if split_mode == "c_holdout":
        return split_indices_by_c(
            labels=labels,
            train_c_values=train_c_values,
            val_c_values=val_c_values,
        )
    return make_split_indices(
        labels=labels,
        split_mode=split_mode,
        val_fraction=val_fraction,
        seed=seed,
    )


def safe_name(value):
    return str(value).replace("=", "eq").replace("+", "plus").replace("-", "minus").replace("*", "times")


def save_probe_artifact(output_path, probe, metadata, metrics, train_idx, val_idx):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in probe.state_dict().items()
            },
            "weight": probe.linear.weight.detach().cpu(),
            "bias": probe.linear.bias.detach().cpu(),
            "metadata": metadata,
            "metrics": metrics,
            "train_idx": train_idx.cpu(),
            "val_idx": val_idx.cpu(),
        },
        output_path,
    )


def remember_best_probe(best_probes, probe, metadata, metrics, top_k):
    best_probes.append(
        {
            "probe": probe,
            "metadata": metadata,
            "metrics": metrics,
        }
    )
    best_probes.sort(
        key=lambda item: item["metrics"]["normalized_gain"],
        reverse=True,
    )
    del best_probes[top_k:]


def run_probe_grid(
    activations,
    labels,
    target,
    device,
    layer_indices,
    position_names,
    modality,
    val_fraction=0.2,
    seed=0,
    epochs=30,
    probe_batch_size=4096,
    lr=1e-3,
    shuffle_train_labels=False,
    shuffle_seed=0,
    early_stopping_patience=None,
    best_probe_output_dir=None,
    best_probe_top_k=1,
    completed_keys=None,
    split_mode="random",
    split_id="",
    train_c_values=None,
    val_c_values=None,
):
    """
    activations: [n_examples, n_layers, n_positions, d_model]
    """
    y = labels_to_tensor(labels, target)
    n_classes = int(y.max().item()) + 1

    train_idx, val_idx, split_info = make_probe_split(
        labels=labels,
        split_mode=split_mode,
        val_fraction=val_fraction,
        seed=seed,
        train_c_values=train_c_values,
        val_c_values=val_c_values,
    )

    y_train = y[train_idx]
    y_val = y[val_idx]
    class_weight = make_class_weight(y_train, n_classes)

    if shuffle_train_labels:
        y_train = shuffle_labels(y_train, seed=shuffle_seed)
        class_weight = make_class_weight(y_train, n_classes)

    n_layers = activations.shape[1]
    n_positions = activations.shape[2]

    rows = []
    best_probes = []
    if completed_keys is None:
        completed_keys = set()
    skipped = 0

    print(f"Modality: {modality}")
    print(f"Target: {target}")
    print(f"Classes: {n_classes}")
    print(f"Train examples: {len(train_idx)}")
    print(f"Val examples: {len(val_idx)}")
    print(f"Split mode: {split_mode}")
    print(f"Split id: {split_id or '<none>'}")
    print(f"Split info: {split_info}")
    print(f"Train majority baseline: {majority_baseline(y_train):.4f}")
    print(f"Val majority baseline: {majority_baseline(y_val):.4f}")
    print("Loss: weighted cross entropy")

    # Train probe for each layer and token position and evaluate the target
    for layer_idx in tqdm(range(n_layers), desc=f"layers for {modality}:{target}"):

        # Convert layer from 0...31 to 1...32 (as we defined the hugging face layers the activations!)
        layer = layer_indices[layer_idx]

        for pos_idx in range(n_positions):
            key = result_key(
                modality=modality,
                target=target,
                layer=layer,
                position_name=position_names[pos_idx],
                shuffled=shuffle_train_labels,
                split_mode=split_mode,
                split_id=split_id,
            )
            if key in completed_keys:
                skipped += 1
                continue

            X = activations[:, layer_idx, pos_idx, :].float()

            X_train = X[train_idx]
            X_val = X[val_idx]

            probe, metrics = train_and_evaluate_probe(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_classes=n_classes,
                device=device,
                epochs=epochs,
                batch_size=probe_batch_size,
                lr=lr,
                early_stopping_patience=early_stopping_patience,
                class_weight=class_weight,
            )
            train_metrics = evaluate_probe(
                probe=probe,
                X=X_train,
                y=y_train,
                n_classes=n_classes,
                device=device,
                batch_size=probe_batch_size,
            )

            row = {
                "modality": modality,
                "target": target,
                "layer": layer,
                "block_axis_idx": layer_idx,
                "position_names": position_names,
                "position_idx": pos_idx,
                "position_name": position_names[pos_idx],
                "split_mode": split_mode,
                "split_id": split_id,
                "split_info": split_info,
                "loss": "weighted_cross_entropy",
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "generalization_gap": (
                    train_metrics["balanced_accuracy"]
                    - metrics["balanced_accuracy"]
                ),
                "macro_f1": metrics["macro_f1"],
                "majority_baseline": metrics["majority_baseline"],
                "normalized_gain": metrics["normalized_gain"],
                "class_counts": metrics["class_counts"],
                "per_class_accuracy": metrics["per_class_accuracy"],
                "shuffle_train_labels": shuffle_train_labels,
                "shuffle_seed": shuffle_seed if shuffle_train_labels else None,
            }
            for k in TOP_K_VALUES:
                row[f"top_{k}_accuracy"] = metrics[f"top_{k}_accuracy"]
            rows.append(row)
            completed_keys.add(key)

            if best_probe_output_dir is not None:
                remember_best_probe(
                    best_probes=best_probes,
                    probe=probe,
                    metadata=row,
                    metrics=metrics,
                    top_k=best_probe_top_k,
                )

    if best_probe_output_dir is not None:
        for rank, item in enumerate(best_probes, start=1):
            metadata = dict(item["metadata"], rank=rank)
            stem = (
                f"{safe_name(modality)}_{safe_name(target)}"
                f"_rank{rank}_layer{metadata['layer']}"
                f"_pos{safe_name(metadata['position_name'])}"
            )
            probe_path = (
                Path(best_probe_output_dir)
                / modality
                / target
                / f"{stem}_probe.pt"
            )
            save_probe_artifact(
                output_path=probe_path,
                probe=item["probe"],
                metadata=metadata,
                metrics=item["metrics"],
                train_idx=train_idx,
                val_idx=val_idx,
            )
            print(
                "Saved best probe:",
                probe_path,
                f"(normalized_gain={item['metrics']['normalized_gain']:.4f})",
            )

    if skipped:
        print(f"Skipped {skipped} combinations already present in the output file.")

    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--modalities", nargs="+", default=MODALITIES)
    parser.add_argument("--targets", nargs="+", default=["result_mod_10"])
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument(
        "--split_mode",
        choices=["random", "operand", "result", "c_holdout"],
        default="random",
    )
    parser.add_argument("--split_id", default="")
    parser.add_argument("--train_c_values", nargs="+", default=None)
    parser.add_argument("--val_c_values", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--shuffle_train_labels", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--best_probe_output_dir", type=Path, default=None)
    parser.add_argument("--best_probe_top_k", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.best_probe_top_k < 1:
        raise ValueError("--best_probe_top_k must be at least 1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output_path)
    existing_rows = load_jsonl(output_path) if output_path.exists() else []
    rows_by_key = {row_result_key(row): row for row in existing_rows}
    completed_keys = set(rows_by_key)
    modality_names = list(dict.fromkeys(
        [row["modality"] for row in existing_rows] + args.modalities
    ))
    target_names = list(dict.fromkeys(
        [row["target"] for row in existing_rows] + args.targets
    ))

    if existing_rows:
        print(
            f"Loaded {len(existing_rows)} existing rows "
            f"({len(completed_keys)} unique combinations) from {output_path}."
        )

    for modality in args.modalities:
        activation_path = Path(args.activation_dir) / f"{modality}_baseline.pt"

        print("Loading activations...")
        data = torch.load(activation_path, map_location="cpu")

        activations = data["activations"]
        labels = data["labels"]
        layer_indices = validate_saved_block_layers(data["block_layers"])
        position_names = data["position_names"]

        print("Activation path:", activation_path)
        print("Activation shape:", activations.shape)
        print("Positions:", position_names)
        print_activation_position_token_examples(data)

        for target in args.targets:

            if target == "requires_carry" and modality == "subtraction":
                print("Skipping requires_carry as it is only for addition.")
                continue
            elif target == "requires_borrow" and modality == "addition":
                print("Skipping requires_borrow as it is only for subtraction.")
                continue

            rows = run_probe_grid(
                activations=activations,
                labels=labels,
                target=target,
                device=device,
                layer_indices=layer_indices,
                position_names=position_names,
                modality=modality,
                val_fraction=args.val_fraction,
                seed=args.seed,
                epochs=args.epochs,
                probe_batch_size=args.probe_batch_size,
                lr=args.lr,
                shuffle_train_labels=args.shuffle_train_labels,
                shuffle_seed=args.shuffle_seed,
                early_stopping_patience=args.early_stopping_patience,
                best_probe_output_dir=args.best_probe_output_dir,
                best_probe_top_k=args.best_probe_top_k,
                completed_keys=completed_keys,
                split_mode=args.split_mode,
                split_id=args.split_id,
                train_c_values=args.train_c_values,
                val_c_values=args.val_c_values,
            )
            for row in rows:
                rows_by_key[row_result_key(row)] = row

            all_rows = sort_result_rows(
                rows_by_key.values(),
                modality_names=modality_names,
                target_names=target_names,
            )
            save_jsonl(all_rows, output_path)
            print("Saved snapshot:", output_path)

    print("Done.")


if __name__ == "__main__":
    main()
