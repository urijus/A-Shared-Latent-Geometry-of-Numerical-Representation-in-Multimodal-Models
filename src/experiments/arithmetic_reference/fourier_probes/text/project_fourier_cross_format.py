import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiments.arithmetic_reference.fourier_probes.text.fourier_probing import (
    MODALITIES,
    labels_to_residues,
    make_fourier_targets,
    make_residue_weights,
    train_and_evaluate_fourier_probe,
)
from src.experiments.arithmetic_reference.fourier_probes.text.project_fourier import (
    layer_to_saved_idx,
    project_activations,
    save_probe_artifact,
)
from src.experiments.arithmetic_reference.linear_probes.splitting import make_split_indices
from src.models import validate_saved_block_layers


def safe_position_name(position):
    return (
        position.replace("=", "eq")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace("*", "times")
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_activation_dir", type=Path, required=True)
    parser.add_argument("--project_activation_dir", type=Path, required=True)
    parser.add_argument("--probe_output_dir", type=Path, required=True)
    parser.add_argument("--projection_output_dir", type=Path, required=True)
    parser.add_argument("--train_format", type=str, default="digits")
    parser.add_argument("--project_format", type=str, default="words")
    parser.add_argument("--modalities", nargs="+", default=MODALITIES)
    parser.add_argument("--target", type=str, default="result")
    parser.add_argument("--periods", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--train_position", type=int, required=True)
    parser.add_argument("--project_position", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_ridge", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--weight_by_residue", action="store_true")
    parser.add_argument(
        "--split_mode",
        choices=["random", "operand", "sum"],
        default="random",
    )
    return parser.parse_args()


def load_activation_pair(args, modality):
    train_path = args.train_activation_dir / f"{modality}_baseline.pt"
    project_path = args.project_activation_dir / f"{modality}_baseline.pt"

    train_data = torch.load(train_path, map_location="cpu")
    project_data = torch.load(project_path, map_location="cpu")

    return train_path, project_path, train_data, project_data


def run_cross_format_projection(args, modality, device):
    train_path, project_path, train_data, project_data = load_activation_pair(args, modality)

    train_activations = train_data["activations"]
    project_activations_tensor = project_data["activations"]
    train_labels = train_data["labels"]
    project_labels = project_data["labels"]
    train_layer_indices = validate_saved_block_layers(train_data["block_layers"])
    project_layer_indices = validate_saved_block_layers(project_data["block_layers"])
    train_position_names = train_data["position_names"]
    project_position_names = project_data["position_names"]

    train_position = str(args.train_position)
    project_position = str(args.project_position)
    train_position_idx = train_position_names.index(train_position)
    project_position_idx = project_position_names.index(project_position)

    train_idx, val_idx, split_info = make_split_indices(
        labels=train_labels,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    output_name = f"{args.target}_{modality}"
    probe_dir = args.probe_output_dir / output_name
    projection_dir = args.projection_output_dir / output_name
    saved_paths = []

    for requested_layer in args.layers:
        train_saved_layer_idx = layer_to_saved_idx(train_layer_indices, requested_layer)
        project_saved_layer_idx = layer_to_saved_idx(project_layer_indices, requested_layer)
        layer = train_layer_indices[train_saved_layer_idx]

        X_train_format = train_activations[:, train_saved_layer_idx, train_position_idx, :].float()
        X_train = X_train_format[train_idx]
        X_val = X_train_format[val_idx]
        X_project = project_activations_tensor[:, project_saved_layer_idx, project_position_idx, :].float()

        desc = f"{args.train_format}->{args.project_format} {modality}:{args.target} layer {layer}"
        for period in tqdm(args.periods, desc=desc):
            y = make_fourier_targets(labels=train_labels, target=args.target, period=period)
            y_train = y[train_idx]
            y_val = y[val_idx]
            sample_weight_train = None

            if args.weight_by_residue:
                residues = labels_to_residues(train_labels, args.target, period)
                sample_weight_train = make_residue_weights(
                    residues=residues[train_idx],
                    period=period,
                )

            probe, metrics = train_and_evaluate_fourier_probe(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_grad_desc=not args.use_ridge,
                early_stopping_patience=args.early_stopping_patience,
                sample_weight=sample_weight_train,
            )

            method = "ridge" if args.use_ridge else "gd"
            loss_name = "residue_weighted_mse" if args.weight_by_residue else "mse"
            metadata = {
                "train_format": args.train_format,
                "project_format": args.project_format,
                "train_activation_path": str(train_path),
                "project_activation_path": str(project_path),
                "modality": modality,
                "target": args.target,
                "period": period,
                "layer": layer,
                "requested_layer": requested_layer,
                "train_block_axis_idx": train_saved_layer_idx,
                "project_block_axis_idx": project_saved_layer_idx,
                "train_position": train_position,
                "project_position": project_position,
                "train_position_idx": train_position_idx,
                "project_position_idx": project_position_idx,
                "val_fraction": args.val_fraction,
                "split_mode": args.split_mode,
                "split_info": split_info,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "optimization_method": method,
                "loss": loss_name,
                "weight_by_residue": args.weight_by_residue,
                "train_model_name": train_data.get("model_name"),
                "project_model_name": project_data.get("model_name"),
            }

            position_name = (
                f"train{safe_position_name(train_position)}_"
                f"project{safe_position_name(project_position)}"
            )
            stem = f"{args.target}_T{period}_layer{layer}_{position_name}_{method}"
            probe_path = probe_dir / f"{stem}_probe.pt"
            projection_path = projection_dir / f"{stem}_projections.pt"

            save_probe_artifact(
                output_path=probe_path,
                probe=probe,
                metadata=metadata,
                metrics=metrics,
                train_idx=train_idx,
                val_idx=val_idx,
            )

            projected = project_activations(X_project, probe)
            projected.update(
                {
                    "labels": project_labels,
                    "fourier_targets": make_fourier_targets(
                        labels=project_labels,
                        target=args.target,
                        period=period,
                    ),
                    "metadata": metadata,
                    "metrics": metrics,
                    "probe_path": str(probe_path),
                }
            )
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(projected, projection_path)

            saved_paths.append((probe_path, projection_path))
            print("Saved probe:", probe_path)
            print("Saved projection:", projection_path)

    print(
        f"Saved {len(saved_paths)} cross-format Fourier probes and projection files "
        f"for {modality}."
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for modality in args.modalities:
        run_cross_format_projection(args, modality, device)


if __name__ == "__main__":
    main()
