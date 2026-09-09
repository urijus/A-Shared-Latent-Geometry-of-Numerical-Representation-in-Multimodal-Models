"""Measure overlap between learned DAS subspaces."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.common import load_jsonl, save_jsonl
from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subspace",
        nargs=3,
        action="append",
        metavar=("LABEL", "RESULTS_JSONL", "TARGET"),
        required=True,
        help=(
            "DAS result file to include. Example: "
            "--subspace add_result path/to/addition_result.jsonl result"
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--position", default="17")
    parser.add_argument("--hook", default="resid_post")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--stem",
        help="Optional output filename stem. Defaults to layer/k/position/seed.",
    )
    return parser.parse_args()


def find_row(path, target, layer, k, position, hook, seed):
    rows = load_jsonl(path)
    matches = [
        row
        for row in rows
        if row.get("target") == target
        and int(row.get("layer", -1)) == int(layer)
        and int(row.get("k", -1)) == int(k)
        and str(row.get("position")) == str(position)
        and row.get("hook") == hook
        and int(row.get("seed", -1)) == int(seed)
    ]
    if not matches:
        raise ValueError(
            f"No DAS row in {path} for target={target}, layer={layer}, "
            f"k={k}, position={position}, hook={hook}, seed={seed}."
        )
    return max(matches, key=lambda row: str(row.get("run_id", "")))


def load_basis(row, layer):
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch_load_portable(checkpoint)
    basis = payload["bases"].get(layer, payload["bases"].get(str(layer)))
    if basis is None:
        raise ValueError(f"Checkpoint {checkpoint} has no layer {layer} basis.")
    return torch.as_tensor(basis).detach().float(), checkpoint


def subspace_metrics(first_basis, second_basis):
    first_basis = torch.as_tensor(first_basis).detach().float().squeeze()
    second_basis = torch.as_tensor(second_basis).detach().float().squeeze()
    d_model = max(first_basis.shape)
    if max(second_basis.shape) != d_model:
        raise ValueError(
            f"Subspaces have different d_model axes: "
            f"{tuple(first_basis.shape)} vs {tuple(second_basis.shape)}."
        )
    first = orthonormal_columns(first_basis, d_model, "first DAS basis")
    second = orthonormal_columns(second_basis, d_model, "second DAS basis")
    singular_values = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    squared_sum = singular_values.square().sum()
    first_in_second = squared_sum / first.shape[1]
    second_in_first = squared_sum / second.shape[1]
    min_rank_containment = squared_sum / min(first.shape[1], second.shape[1])
    return {
        "first_in_second": float(first_in_second),
        "second_in_first": float(second_in_first),
        "symmetric_overlap": float((first_in_second + second_in_first) / 2),
        "min_rank_containment": float(min_rank_containment),
        "principal_cosines": [float(value) for value in singular_values],
        "first_rank": first.shape[1],
        "second_rank": second.shape[1],
        "d_model": d_model,
    }


def plot_matrix(rows, labels, metric, title, output_path):
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    matrix = torch.tensor(
        [[lookup[(first, second)][metric] for second in labels] for first in labels]
    ).numpy()
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.4, 1.2 * len(labels) + 2.0))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Second subspace")
    ax.set_ylabel("First subspace")
    ax.set_title(title)
    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = matrix[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    entries = []
    for label, path_text, target in args.subspace:
        row = find_row(
            Path(path_text),
            target,
            args.layer,
            args.k,
            args.position,
            args.hook,
            args.seed,
        )
        basis, checkpoint = load_basis(row, args.layer)
        entries.append(
            {
                "label": label,
                "target": target,
                "row": row,
                "basis": basis,
                "checkpoint": checkpoint,
            }
        )

    summary = []
    for first in entries:
        for second in entries:
            row = {
                "first_label": first["label"],
                "second_label": second["label"],
                "first_target": first["target"],
                "second_target": second["target"],
                "layer": args.layer,
                "k": args.k,
                "position": str(args.position),
                "hook": args.hook,
                "seed": args.seed,
                "first_checkpoint_path": str(first["checkpoint"]),
                "second_checkpoint_path": str(second["checkpoint"]),
            }
            row.update(subspace_metrics(first["basis"], second["basis"]))
            summary.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = [entry["label"] for entry in entries]
    stem = args.stem or (
        f"das_overlap_layer{args.layer}_k{args.k}_pos{args.position}_"
        f"{args.hook}_seed{args.seed}"
    )
    output_path = args.output_dir / f"{stem}.jsonl"
    save_jsonl(summary, output_path)
    plot_matrix(
        summary,
        labels,
        "symmetric_overlap",
        "DAS subspace overlap",
        args.output_dir / f"{stem}_symmetric_overlap.png",
    )
    plot_matrix(
        summary,
        labels,
        "min_rank_containment",
        "DAS min-rank containment",
        args.output_dir / f"{stem}_min_rank_containment.png",
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
