import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiments.arithmetic_reference.fourier_probes.text.fourier_probing import (
    FourierProbe,
    MODALITIES,
    labels_to_residues,
    make_fourier_targets,
    make_residue_weights,
    train_and_evaluate_fourier_probe,
)
from src.experiments.arithmetic_reference.linear_probes.splitting import make_split_indices
from src.models import validate_saved_block_layers



def layer_to_saved_idx(layer_indices, layer):
    if layer in layer_indices:
        return layer_indices.index(layer)
    raise ValueError(
        f"Transformer block layer {layer} was not saved. "
        f"Available 1-based block layers: {layer_indices}"
    )


def normalized_rows(weight):
    # Convert into normal basis
    return weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-8)


def orthonormal_plane(weight):
    # QR decomposition to turn cos and sin directions into orthonormal bases
    basis, _ = torch.linalg.qr(weight.T, mode="reduced")
    return basis.T


@torch.no_grad()
def project_activations(X, probe):
    weight = probe.linear.weight.detach().cpu()
    bias = probe.linear.bias.detach().cpu()
    X = X.detach().cpu().float()

    unit_weight = normalized_rows(weight)
    plane_basis = orthonormal_plane(weight)

    return {
        "probe_output": X @ weight.T + bias,
        "raw_direction_projection": X @ weight.T,
        "unit_direction_projection": X @ unit_weight.T,
        "orthonormal_plane_projection": X @ plane_basis.T,
        "weight": weight,
        "unit_weight": unit_weight,
        "orthonormal_plane_basis": plane_basis,
        "bias": bias,
    }


def save_probe_artifact(
    output_path,
    probe,
    metadata,
    metrics,
    train_idx,
    val_idx,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in probe.state_dict().items()
    }
    torch.save(
        {
            "state_dict": state_dict,
            "weight": probe.linear.weight.detach().cpu(),
            "bias": probe.linear.bias.detach().cpu(),
            "metadata": metadata,
            "metrics": metrics,
            "train_idx": train_idx.cpu(),
            "val_idx": val_idx.cpu(),
        },
        output_path,
    )


# Look if the probe was already stored when perfoming fourier probing
def load_saved_probe(
    saved_probe_dir,
    modality,
    target,
    period,
    layer,
    position,
    method,
    device,
):
    if saved_probe_dir is None:
        return None

    for path in sorted(Path(saved_probe_dir).rglob("*_probe.pt")):
        artifact = torch.load(path, map_location="cpu")
        metadata = artifact.get("metadata", {})
        if metadata.get("period") is None or metadata.get("layer") is None:
            continue
        if (
            metadata.get("modality") == modality
            and metadata.get("target") == target
            and int(metadata.get("period")) == int(period)
            and int(metadata.get("layer")) == int(layer)
            and metadata.get("position_name", metadata.get("position")) == position
            and metadata.get("optimization_method") == method
        ):
            probe = FourierProbe(artifact["weight"].shape[1]).to(device)
            probe.load_state_dict(artifact["state_dict"])
            probe.eval()
            return probe, artifact.get("metrics", {}), path

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", default=MODALITIES)
    parser.add_argument("--probe_output_dir", type=Path, required=True)
    parser.add_argument("--saved_probe_dir", type=Path, default=None)
    parser.add_argument("--projection_output_dir", type=Path, required=True)
    parser.add_argument("--target", type=str, default="result")
    parser.add_argument("--periods", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--position", type=int, required=True)
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


def run_projection_for_activation(
    args,
    activation_path,
    probe_dir,
    projection_dir,
    device,
    modality=None,
):
    data = torch.load(activation_path, map_location="cpu")
    activations = data["activations"]
    labels = data["labels"]
    layer_indices = validate_saved_block_layers(data["block_layers"])
    position_names = data["position_names"]

    position = str(args.position)
    if position not in position_names:
        raise ValueError(
            f"Position {args.position!r} was not saved. "
            f"Available positions: {position_names}"
        )
    position_idx = position_names.index(position)

    train_idx, val_idx, split_info = make_split_indices(
        labels=labels,
        split_mode=args.split_mode,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    saved_paths = []

    for requested_layer in args.layers:
        saved_layer_idx = layer_to_saved_idx(layer_indices, requested_layer)
        layer = layer_indices[saved_layer_idx]

        X = activations[:, saved_layer_idx, position_idx, :].float()
        X_train = X[train_idx]
        X_val = X[val_idx]

        for period in tqdm(args.periods, desc=f"{args.target} at layer {layer}, pos {position}"):
            y = make_fourier_targets(labels=labels, target=args.target, period=period)
            y_train = y[train_idx]
            y_val = y[val_idx]
            sample_weight_train = None

            if args.weight_by_residue:
                residues = labels_to_residues(labels, args.target, period)
                sample_weight_train = make_residue_weights(
                    residues=residues[train_idx],
                    period=period,
                )

            method = "ridge" if args.use_ridge else "gd"
            loaded = load_saved_probe(
                saved_probe_dir=args.saved_probe_dir,
                modality=modality,
                target=args.target,
                period=period,
                layer=layer,
                position=position,
                method=method,
                device=device,
            )
            if loaded is None:
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
                loaded_probe_path = None
            else:
                probe, metrics, loaded_probe_path = loaded
                print("Loaded saved probe:", loaded_probe_path)

            metadata = {
                "activation_path": str(activation_path),
                "modality": modality,
                "target": args.target,
                "period": period,
                "layer": layer,
                "requested_layer": requested_layer,
                "block_axis_idx": saved_layer_idx,
                "position": position,
                "position_idx": position_idx,
                "val_fraction": args.val_fraction,
                "split_mode": args.split_mode,
                "split_info": split_info,
                "seed": args.seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "optimization_method": method,
                "loss": "residue_weighted_mse" if args.weight_by_residue else "mse",
                "weight_by_residue": args.weight_by_residue,
                "model_name": data.get("model_name"),
                "loaded_probe_path": str(loaded_probe_path) if loaded_probe_path else None,
            }

            stem = f"{args.target}_T{period}_layer{layer}_pos{position}_{method}"
            stem = stem.replace("=", "eq").replace("+", "plus").replace("-", "minus").replace("*", "times").replace(":", "_")
            probe_path = probe_dir / f"{stem}_probe.pt"
            projection_path = projection_dir / f"{stem}_projections.pt"

            if loaded_probe_path is None:
                save_probe_artifact(
                    output_path=probe_path,
                    probe=probe,
                    metadata=metadata,
                    metrics=metrics,
                    train_idx=train_idx,
                    val_idx=val_idx,
                )
            else:
                save_probe_artifact(
                    output_path=probe_path,
                    probe=probe,
                    metadata=metadata,
                    metrics=metrics,
                    train_idx=train_idx,
                    val_idx=val_idx,
                )

            projection = project_activations(X, probe)
            projection.update(
                {
                    "labels": labels,
                    "fourier_targets": y,
                    "metadata": metadata,
                    "metrics": metrics,
                    "probe_path": str(probe_path),
                }
            )
            projection_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(projection, projection_path)

            saved_paths.append((probe_path, projection_path))
            print("Saved probe:", probe_path)
            print("Saved projection:", projection_path)

    print(f"Saved {len(saved_paths)} Fourier probes and projection files.")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for modality in args.modalities:
        activation_path = args.activation_dir / f"{modality}_baseline.pt"
        output_name = f"{args.target}_{modality}"

        run_projection_for_activation(
            args=args,
            activation_path=activation_path,
            probe_dir=Path(args.probe_output_dir) / output_name,
            projection_dir=Path(args.projection_output_dir) / output_name,
            device=device,
            modality=modality,
        )


if __name__ == "__main__":
    main()
