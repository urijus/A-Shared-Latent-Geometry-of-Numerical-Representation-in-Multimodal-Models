
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.common.io import load_jsonl, save_jsonl
from src.experiments.arithmetic_reference.linear_probes.splitting import make_split_indices
from src.models import validate_saved_block_layers
from torch.utils.data import DataLoader, TensorDataset


MODALITIES = ["addition", "subtraction", "multiplication"]


def result_key(
    modality,
    target,
    period,
    layer,
    position_name,
    shuffled,
    optimization_method,
    weight_by_residue,
):
    return (
        modality,
        target,
        int(period),
        int(layer),
        str(position_name),
        bool(shuffled),
        optimization_method,
        bool(weight_by_residue),
    )


def row_result_key(row):
    return result_key(
        modality=row["modality"],
        target=row["target"],
        period=row["period"],
        layer=row["layer"],
        position_name=row["position_name"],
        shuffled=row.get("shuffle_train_labels", False),
        optimization_method=row.get("optimization_method", "gd"),
        weight_by_residue=row.get("weight_by_residue", False),
    )


def sort_result_rows(rows):
    modality_order = {name: index for index, name in enumerate(MODALITIES)}

    def position_sort_key(value):
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    return sorted(
        rows,
        key=lambda row: (
            modality_order.get(row["modality"], len(modality_order)),
            row["modality"],
            row["target"],
            int(row["period"]),
            int(row["layer"]),
            position_sort_key(row["position_name"]),
            bool(row.get("shuffle_train_labels", False)),
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


class FourierProbe(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 2)

    def forward(self, x):
        return self.linear(x)


def safe_name(value):
    return (
        str(value)
        .replace("=", "eq")
        .replace("+", "plus")
        .replace("-", "minus")
        .replace("*", "times")
        .replace(":", "_")
    )


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
        key=lambda item: item["metrics"]["r2_mean"],
        reverse=True,
    )
    del best_probes[top_k:]


def make_fourier_targets(labels, target, period):
    """
    labels: list of dicts
    target: "a", "b", or "result"
    period: int, e.g. 2, 5, 10

    Returns:
        Y: [n_examples, 2]
    """
    values = torch.tensor(
        [int(row[target]) for row in labels],
        dtype=torch.float32,
    )

    theta = 2 * torch.pi * values / period

    Y = torch.stack(
        [
            torch.cos(theta),
            torch.sin(theta),
        ],
        dim=1,
    )

    return Y


def labels_to_residues(labels, target, period):
    values = torch.tensor(
        [int(row[target]) for row in labels],
        dtype=torch.long,
    )
    return values % period


def make_residue_weights(residues, period):
    counts = torch.bincount(residues, minlength=period).float()
    weights = torch.zeros(period, dtype=torch.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights[residues]


def shuffle_rows(y, weights=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(y.shape[0], generator=generator)
    if weights is None:
        return y[indices], None
    return y[indices], weights[indices]


def compute_r2(y_true, y_pred, loss_fn):
    r2_cos = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_sin = r2_score(y_true[:, 1], y_pred[:, 1])
    r2_mean = (r2_cos + r2_sin) / 2

    return {
        "r2_cos": float(r2_cos),
        "r2_sin": float(r2_sin),
        "r2_mean": float(r2_mean),
        "mse": loss_fn(y_pred, y_true).item(),
    }


def evaluate_probe(probe, X_val, y_val, batch_size, device):
    loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True)

    probe.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            preds = probe(xb).cpu()

            y_true.append(yb.cpu())
            y_pred.append(preds.cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    metrics = compute_r2(y_true, y_pred, nn.MSELoss())
    return metrics


def ridge_regression(X_train, y_train, X_val, y_val, sample_weight=None, alpha=5.0):
    ridge = Ridge(alpha=alpha, solver="lsqr")
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight.cpu().numpy()

    ridge.fit(X_train.cpu().numpy(), y_train.cpu().numpy(), **fit_kwargs)

    y_pred = torch.tensor(
        ridge.predict(X_val.cpu().numpy()),
        dtype=torch.float32,
    )

    metrics = compute_r2(y_val.float(), y_pred, nn.MSELoss())

    probe = FourierProbe(X_train.shape[1])
    with torch.no_grad():
        probe.linear.weight.copy_(torch.tensor(ridge.coef_, dtype=torch.float32))
        probe.linear.bias.copy_(torch.tensor(ridge.intercept_, dtype=torch.float32))

    return probe, metrics

def train_and_evaluate_fourier_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
    epochs=30,
    batch_size=4096,
    lr=1e-3,
    use_grad_desc=True,
    early_stopping_patience=None,
    sample_weight=None,
):
    loss_fn = nn.MSELoss()

    if use_grad_desc:
        if sample_weight is None:
            dataset = TensorDataset(X_train, y_train)
        else:
            dataset = TensorDataset(X_train, y_train, sample_weight)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        probe = FourierProbe(X_train.shape[1]).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
        use_early_stopping = (
            early_stopping_patience is not None
            and early_stopping_patience > 0
        )
        best_r2_mean = float("-inf")
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            probe.train()

            for batch in loader:
                if sample_weight is None:
                    xb, yb = batch
                    wb = None
                else:
                    xb, yb, wb = batch
                    wb = wb.to(device)

                xb = xb.to(device)
                yb = yb.to(device)

                y_pred = probe(xb)
                if wb is None:
                    loss = loss_fn(y_pred, yb)
                else:
                    # Use weighted mse if activated
                    per_example_loss = (y_pred - yb).pow(2).mean(dim=1)
                    loss = (per_example_loss * wb).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if use_early_stopping:
                metrics = evaluate_probe(probe, X_val, y_val, batch_size, device)
                if metrics["r2_mean"] > best_r2_mean:
                    best_r2_mean = metrics["r2_mean"]
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in probe.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    break

        if best_state_dict is not None:
            probe.load_state_dict(
                {
                    key: value.to(device)
                    for key, value in best_state_dict.items()
                }
            )

        metrics = evaluate_probe(probe, X_val, y_val, batch_size, device)
        return probe, metrics

    else:
        return ridge_regression(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weight=sample_weight,
        )


def run_fourier_probe_grid(
    activations,
    labels,
    layer_indices,
    target,
    periods,
    device,
    position_names,
    modality,
    val_fraction=0.2,
    split_mode="random",
    seed=0,
    epochs=30,
    probe_batch_size=4096,
    lr=1e-3,
    shuffle_train_labels=False,
    shuffle_seed=0,
    use_grad_desc=True,
    early_stopping_patience=None,
    weight_by_residue=False,
    best_probe_output_dir=None,
    best_probe_top_k=1,
    completed_keys=None,
):
    """
    activations: [n_examples, n_layers, n_positions, d_model]
    """

    n_layers = activations.shape[1]

    train_idx, val_idx, split_info = make_split_indices(
        labels=labels,
        split_mode=split_mode,
        val_fraction=val_fraction,
        seed=seed,
    )

    rows = []
    best_probes_by_period = {period: [] for period in periods}
    if completed_keys is None:
        completed_keys = set()
    skipped = 0

    print(f"Modality: {modality}")
    print(f"Train examples: {len(train_idx)}")
    print(f"Val examples: {len(val_idx)}")
    loss_name = "residue_weighted_mse" if weight_by_residue else "mse"
    print(f"Loss: {loss_name}")

    for period in periods:
        y = make_fourier_targets(
            labels=labels,
            target=target,
            period=period,
        )

        y_train = y[train_idx]
        y_val = y[val_idx]
        sample_weight_train = None

        if weight_by_residue:
            residues = labels_to_residues(labels, target, period)
            sample_weight_train = make_residue_weights(
                residues=residues[train_idx],
                period=period,
            )

        if shuffle_train_labels:
            y_train, sample_weight_train = shuffle_rows(
                y_train,
                weights=sample_weight_train,
                seed=shuffle_seed,
            )


        print(f"Target={target}, period={period}")

        for layer_idx in tqdm(range(n_layers), desc=f"layers for {modality}:{target}"):
            layer = layer_indices[layer_idx]

            for position_idx in range(len(position_names)):
                key = result_key(
                    modality=modality,
                    target=target,
                    period=period,
                    layer=layer,
                    position_name=position_names[position_idx],
                    shuffled=shuffle_train_labels,
                    optimization_method="gd" if use_grad_desc else "ridge",
                    weight_by_residue=weight_by_residue,
                )
                if key in completed_keys:
                    skipped += 1
                    continue

                X = activations[:, layer_idx, position_idx, :].float()

                X_train = X[train_idx]
                X_val = X[val_idx]

                probe, metrics = train_and_evaluate_fourier_probe(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    device=device,
                    epochs=epochs,
                    batch_size=probe_batch_size,
                    lr=lr,
                    use_grad_desc=use_grad_desc,
                    early_stopping_patience=early_stopping_patience,
                    sample_weight=sample_weight_train,
                )

                row = {
                    "modality": modality,
                    "target": target,
                    "period": period,
                    "layer": layer,
                    "layer_indexing": "1_based_transformer_blocks",
                    "block_axis_idx": layer_idx,
                    "position_names": position_names,
                    "position": position_names[position_idx],
                    "position_name": position_names[position_idx],
                    "position_idx": position_idx,
                    "r2_cos": metrics["r2_cos"],
                    "r2_sin": metrics["r2_sin"],
                    "r2_mean": metrics["r2_mean"],
                    "mse": metrics["mse"],
                    "val_fraction": val_fraction,
                    "split_mode": split_mode,
                    "split_info": split_info,
                    "loss": loss_name,
                    "weight_by_residue": weight_by_residue,
                    "seed": seed,
                    "shuffle_train_labels": shuffle_train_labels,
                    "shuffle_seed": shuffle_seed if shuffle_train_labels else None,
                    "optimization_method": "gd" if use_grad_desc else "ridge",
                }
                rows.append(row)
                completed_keys.add(key)

                if best_probe_output_dir is not None:
                    remember_best_probe(
                        best_probes=best_probes_by_period[period],
                        probe=probe,
                        metadata=row,
                        metrics=metrics,
                        top_k=best_probe_top_k,
                    )

    if best_probe_output_dir is not None:
        for period, best_probes in best_probes_by_period.items():
            for rank, item in enumerate(best_probes, start=1):
                metadata = dict(item["metadata"], rank=rank)
                stem = (
                    f"{safe_name(modality)}_{safe_name(target)}"
                    f"_T{metadata['period']}_rank{rank}"
                    f"_layer{metadata['layer']}"
                    f"_pos{safe_name(metadata['position_name'])}"
                )
                probe_path = (
                    Path(best_probe_output_dir)
                    / target
                    / f"T{period}"
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
                    "Saved best Fourier probe:",
                    probe_path,
                    f"(r2_mean={item['metrics']['r2_mean']:.4f})",
                )

    if skipped:
        print(f"Skipped {skipped} combinations already present in the output file.")

    return rows


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--activation_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--targets", nargs="+", default=["result"])
    parser.add_argument("--modalities", nargs="+", default=MODALITIES)

    parser.add_argument(
        "--periods",
        type=int,
        nargs="+",
        default=[2, 5, 10, 20, 50, 100],
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--split_mode", type=str, default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_train_labels", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=0)
    parser.add_argument("--use_ridge", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--weight_by_residue", action="store_true")
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

    if existing_rows:
        print(
            f"Loaded {len(existing_rows)} existing rows "
            f"({len(completed_keys)} unique combinations) from {output_path}."
        )

    for modality in args.modalities:
        activation_path = Path(args.activation_dir) / f"{modality}_baseline.pt"
        print("Loading activations...")
        print("Activation path:", activation_path)
        data = torch.load(activation_path, map_location="cpu")

        activations = data["activations"]
        labels = data["labels"]
        layer_indices = validate_saved_block_layers(data["block_layers"])
        position_names = data["position_names"]

        print("Activation shape:", activations.shape)
        print("Positions:", position_names)
        print_activation_position_token_examples(data)

        for target in args.targets:
            rows = run_fourier_probe_grid(
                activations=activations,
                labels=labels,
                layer_indices=layer_indices,
                target=target,
                periods=args.periods,
                device=device,
                position_names=position_names,
                modality=modality,
                val_fraction=args.val_fraction,
                split_mode=args.split_mode,
                seed=args.seed,
                epochs=args.epochs,
                probe_batch_size=args.batch_size,
                lr=args.lr,
                shuffle_train_labels=args.shuffle_train_labels,
                shuffle_seed=args.shuffle_seed,
                use_grad_desc=not args.use_ridge,
                early_stopping_patience=args.early_stopping_patience,
                weight_by_residue=args.weight_by_residue,
                best_probe_output_dir=args.best_probe_output_dir,
                best_probe_top_k=args.best_probe_top_k,
                completed_keys=completed_keys,
            )

            for row in rows:
                rows_by_key[row_result_key(row)] = row

            all_rows = sort_result_rows(rows_by_key.values())
            save_jsonl(all_rows, output_path)
            print("Saved snapshot:", output_path)

    print("Done.")


if __name__ == "__main__":
    main()
