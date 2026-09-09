"""Train linear probes and measure transfer between arithmetic modalities."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.common.io import load_jsonl, save_jsonl
from src.experiments.arithmetic_reference.linear_probes.text.run_probes import make_class_weight
from src.probes.linear import evaluate_probe, train_probe
from src.experiments.arithmetic_reference.linear_probes.splitting import make_split_indices
from src.models import validate_saved_block_layers


TARGET_MODALITIES = {
    # Full multiplication results are much sparser and live on a larger label
    # support than addition/subtraction, so they are not a fair cross-format
    # reference. Compare full result only where the label space is shared.
    "result": ["addition", "subtraction"],
    "result_mod_10": ["addition", "subtraction", "multiplication"],
    "result_mod_100": ["addition", "subtraction", "multiplication"],
}

LABEL_ENCODING_VERSION = "contiguous_union_v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_dir", type=Path, required=True)
    parser.add_argument(
        "--multiplication_activation_dir",
        type=Path,
        help=(
            "Optional activation directory used only for multiplication. "
            "Use this for balanced multiplication caches while keeping "
            "addition/subtraction on --activation_dir."
        ),
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--plot_dir", type=Path, required=True)
    parser.add_argument("--position", required=True, help="Saved prompt-token position name.")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", default=list(TARGET_MODALITIES))
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    return parser.parse_args()


def activation_path_for_modality(args, modality):
    activation_dir = (
        args.multiplication_activation_dir
        if modality == "multiplication" and args.multiplication_activation_dir
        else args.activation_dir
    )
    return activation_dir / f"{modality}_baseline.pt"


def load_activations(args, modalities, layers, position):
    datasets = {}
    for modality in modalities:
        path = activation_path_for_modality(args, modality)
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        saved_layers = validate_saved_block_layers(data["block_layers"])
        saved_positions = [str(value) for value in data["position_names"]]

        # Select the same residual-stream layer and prompt position in every modality.
        missing_layers = [layer for layer in layers if layer not in saved_layers]
        if missing_layers:
            raise ValueError(f"{path} does not contain layers {missing_layers}.")
        if position not in saved_positions:
            raise ValueError(
                f"{path} does not contain position {position!r}; "
                f"available positions: {saved_positions}."
            )
        layer_indices = [saved_layers.index(layer) for layer in layers]
        position_idx = saved_positions.index(position)

        # Materialize only the requested cells instead of retaining the full checkpoint.
        selected_activations = data["activations"][
            :, layer_indices, position_idx, :
        ].unsqueeze(2).contiguous()
        datasets[modality] = {
            "path": path,
            "activations": selected_activations,
            "labels": data["labels"],
            "layer_to_idx": {layer: index for index, layer in enumerate(layers)},
            "position_idx": 0,
            "model_name": data.get("model_name"),
        }
        del data
    return datasets


def make_splits(datasets, val_fraction, seed):
    splits = {}
    for modality, data in datasets.items():
        train_idx, test_idx, split_info = make_split_indices(
            labels=data["labels"],
            split_mode="random",
            val_fraction=val_fraction,
            seed=seed,
        )
        splits[modality] = (train_idx, test_idx, split_info)
    return splits


def raw_target_labels(data, target):
    return torch.tensor([int(row[target]) for row in data["labels"]], dtype=torch.long)


def encode_labels(raw_labels, class_values):
    value_to_class = {int(value): index for index, value in enumerate(class_values)}
    return torch.tensor(
        [value_to_class[int(value)] for value in raw_labels.tolist()],
        dtype=torch.long,
    )


def metric_fields(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"class_counts", "per_class_accuracy"}
    }


def requested_activation_paths(args):
    modalities = list(dict.fromkeys(
        modality for target in args.targets for modality in TARGET_MODALITIES[target]
    ))
    return {
        modality: str(activation_path_for_modality(args, modality))
        for modality in modalities
    }


def rows_match_requested_activation_paths(rows, activation_paths):
    return all(
        row.get("train_activation_path") == activation_paths[row["train_modality"]]
        and row.get("test_activation_path") == activation_paths[row["test_modality"]]
        for row in rows
    )


def plot_target_metric(rows, target, modalities, layers, metric, title, output_path):
    ncols = min(3, len(layers))
    nrows = int(np.ceil(len(layers) / ncols))
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.7 * nrows),
        squeeze=False,
    )

    values = [row[metric] for row in rows if np.isfinite(row[metric])]
    if metric == "normalized_transfer":
        cmap = "coolwarm"
        limit = max(1.0, max((abs(value) for value in values), default=1.0))
        vmin, vmax = -limit, limit
    else:
        cmap = "viridis"
        vmin, vmax = 0.0, 1.0
    image = None
    for axis, layer in zip(axes.flat, layers):
        layer_rows = [row for row in rows if row["layer"] == layer]
        matrix = np.full((len(modalities), len(modalities)), np.nan)
        for row in layer_rows:
            test_idx = modalities.index(row["test_modality"])
            train_idx = modalities.index(row["train_modality"])
            matrix[test_idx, train_idx] = row[metric]

        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(modalities)), modalities, rotation=25, ha="right")
        axis.set_yticks(range(len(modalities)), modalities)
        axis.set_title(f"Layer {layer}")
        for row_idx in range(len(modalities)):
            for col_idx in range(len(modalities)):
                value = matrix[row_idx, col_idx]
                label = "n/a" if np.isnan(value) else f"{value:.2f}"
                axis.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9)

    for axis in axes.flat[len(layers):]:
        axis.set_visible(False)

    figure.suptitle(f"{title}: {target}")
    figure.supxlabel("Train modality")
    figure.supylabel("Test modality")

    # Adjust layout and add dedicated colorbar axis
    figure.subplots_adjust(left=0.09, right=0.88, bottom=0.12, top=0.90, wspace=0.42, hspace=0.45)
    if image is not None:
        cbar_ax = figure.add_axes([0.90, 0.15, 0.02, 0.7])
        figure.colorbar(image, cax=cbar_ax, label=title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_target(rows, target, modalities, layers, plot_dir):
    metrics = (
        ("normalized_transfer", "Normalized transfer", "normalized_transfer"),
        ("cross_score", "Raw accuracy", "accuracy"),
        ("normalized_gain", "Normalized gain", "normalized_gain"),
    )
    for metric, title, stem in metrics:
        plot_target_metric(
            rows,
            target,
            modalities,
            layers,
            metric,
            title,
            plot_dir / f"{target}_{stem}.png",
        )


def run_target(args, target, modalities, datasets, splits, device):
    # Use one shared, compact class encoding across modalities. For exact result,
    # the raw values are 100..900; using raw labels would create 100 impossible
    # classes (0..99) and make the probe harder for no scientific reason.
    raw_labels = {
        modality: raw_target_labels(datasets[modality], target)
        for modality in modalities
    }
    class_values = sorted(
        set().union(*(set(values.tolist()) for values in raw_labels.values()))
    )
    labels = {
        modality: encode_labels(raw_labels[modality], class_values)
        for modality in modalities
    }
    n_classes = len(class_values)
    probes = {}

    for train_modality in modalities:
        train_idx, test_idx, _ = splits[train_modality]
        y_train = labels[train_modality][train_idx]
        y_test = labels[train_modality][test_idx]
        class_weight = make_class_weight(y_train, n_classes)

        for layer in args.layers:
            data = datasets[train_modality]
            X = data["activations"][
                :, data["layer_to_idx"][layer], data["position_idx"], :
            ].float()
            label = f"{target}, {train_modality}, layer {layer}"
            print(f"Training {label}", flush=True)
            probes[(train_modality, layer)] = train_probe(
                X_train=X[train_idx],
                y_train=y_train,
                X_val=X[test_idx],
                y_val=y_test,
                n_classes=n_classes,
                device=device,
                epochs=args.epochs,
                batch_size=args.probe_batch_size,
                lr=args.lr,
                early_stopping_patience=args.early_stopping_patience,
                class_weight=class_weight,
                progress_label=label,
            )

    evaluations = {}
    for train_modality in modalities:
        for test_modality in modalities:
            _, test_idx, _ = splits[test_modality]
            y_test = labels[test_modality][test_idx]
            for layer in args.layers:
                data = datasets[test_modality]
                X_test = data["activations"][
                    test_idx, data["layer_to_idx"][layer], data["position_idx"], :
                ].float()
                evaluations[(train_modality, test_modality, layer)] = evaluate_probe(
                    probe=probes[(train_modality, layer)],
                    X=X_test,
                    y=y_test,
                    n_classes=n_classes,
                    device=device,
                    batch_size=args.probe_batch_size,
                )

    rows = []
    for (train_modality, test_modality, layer), metrics in evaluations.items():
        train_idx, _, train_split_info = splits[train_modality]
        _, test_idx, test_split_info = splits[test_modality]
        target_self_score = evaluations[(test_modality, test_modality, layer)]["accuracy"]
        majority_baseline = metrics["majority_baseline"]
        denominator = target_self_score - majority_baseline
        normalized_transfer = (
            (metrics["accuracy"] - majority_baseline) / denominator
            if denominator != 0
            else float("nan")
        )
        rows.append(
            {
                "target": target,
                "train_modality": train_modality,
                "test_modality": test_modality,
                "layer": layer,
                "position_name": args.position,
                "hook_name": "resid_post",
                "n_classes": n_classes,
                "class_values": class_values,
                "label_encoding": LABEL_ENCODING_VERSION,
                "model_name": datasets[test_modality]["model_name"],
                "train_activation_path": str(datasets[train_modality]["path"]),
                "test_activation_path": str(datasets[test_modality]["path"]),
                "train_dataset_n_examples": len(datasets[train_modality]["labels"]),
                "test_dataset_n_examples": len(datasets[test_modality]["labels"]),
                "train_split": train_split_info,
                "test_split": test_split_info,
                "n_train_examples": len(train_idx),
                "n_test_examples": len(test_idx),
                "epochs": args.epochs,
                "probe_batch_size": args.probe_batch_size,
                "learning_rate": args.lr,
                "early_stopping_patience": args.early_stopping_patience,
                "score_name": "accuracy",
                "cross_score": metrics["accuracy"],
                "target_self_score": target_self_score,
                "normalized_transfer": normalized_transfer,
                **metric_fields(metrics),
                "class_counts": metrics["class_counts"],
                "per_class_accuracy": metrics["per_class_accuracy"],
            }
        )
    return rows


def main():
    args = parse_args()
    unknown_targets = [target for target in args.targets if target not in TARGET_MODALITIES]
    if unknown_targets:
        raise ValueError(f"Unsupported targets: {unknown_targets}.")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val_fraction must be between 0 and 1.")

    # Plot directly from a complete existing run instead of retraining every probe.
    if args.output_path.exists():
        saved_rows = load_jsonl(args.output_path)
        activation_paths = requested_activation_paths(args)
        requested_rows = [
            row for row in saved_rows
            if row["target"] in args.targets
            and row["layer"] in args.layers
            and str(row["position_name"]) == str(args.position)
            and row.get("label_encoding") == LABEL_ENCODING_VERSION
        ]
        expected = sum(
            len(TARGET_MODALITIES[target]) ** 2 * len(args.layers)
            for target in args.targets
        )
        if (
            len(requested_rows) == expected
            and rows_match_requested_activation_paths(requested_rows, activation_paths)
        ):
            print(f"Loading {expected} existing metric rows from {args.output_path}")
            for target in args.targets:
                plot_target(
                    [row for row in requested_rows if row["target"] == target],
                    target,
                    TARGET_MODALITIES[target],
                    args.layers,
                    args.plot_dir,
                )
            print(f"Saved plots to {args.plot_dir}")
            return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_modalities = list(dict.fromkeys(
        modality for target in args.targets for modality in TARGET_MODALITIES[target]
    ))
    datasets = load_activations(args, all_modalities, args.layers, args.position)
    splits = make_splits(datasets, args.val_fraction, args.seed)

    rows = []
    for target in args.targets:
        modalities = TARGET_MODALITIES[target]
        target_rows = run_target(args, target, modalities, datasets, splits, device)
        rows.extend(target_rows)
        plot_target(
            target_rows,
            target,
            modalities,
            args.layers,
            args.plot_dir,
        )

    save_jsonl(rows, args.output_path)
    print(f"Saved {len(rows)} metric rows to {args.output_path}")
    print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
