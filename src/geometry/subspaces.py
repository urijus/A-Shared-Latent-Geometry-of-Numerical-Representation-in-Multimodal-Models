"""Measure containment of Fourier probe planes in learned DAS subspaces."""

import argparse
import pathlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.common import load_jsonl, save_jsonl
from src.probes.fourier import probe_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--das_results", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--das_target", required=True)
    parser.add_argument(
        "--fourier_target",
        help="Defaults to --das_target; useful for intentional cross-variable tests.",
    )
    parser.add_argument("--periods", type=int, nargs="+", required=True)
    parser.add_argument("--k_dims", type=int, nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", type=int, default=17)
    parser.add_argument("--hook", default="resid_post")
    parser.add_argument("--method", choices=["ridge", "gd"], default="ridge")
    parser.add_argument("--seed", type=int, default=8)
    return parser.parse_args()


def torch_load_portable(path):
    """Load checkpoints that may contain Path objects saved on another OS."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except NotImplementedError as error:
        if "PosixPath" not in str(error) or not hasattr(pathlib, "WindowsPath"):
            raise
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            return torch.load(path, map_location="cpu", weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path


def orthonormal_columns(matrix, d_model, name):
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix; got {tuple(matrix.shape)}.")
    if matrix.shape[0] == d_model:
        columns = matrix
    elif matrix.shape[1] == d_model:
        columns = matrix.T
    else:
        raise ValueError(
            f"Neither axis of {name} has d_model={d_model}: {tuple(matrix.shape)}."
        )
    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    tolerance = (
        max(columns.shape)
        * torch.finfo(columns.dtype).eps
        * singular_values.max()
    )
    rank = int((singular_values > tolerance).sum().item())
    if rank == 0:
        raise ValueError(f"{name} has numerical rank zero.")
    return left[:, :rank]


def overlap_metrics(fourier_weight, das_basis):
    """Return basis-invariant subspace containment metrics."""
    das_basis = torch.as_tensor(das_basis).detach().float().squeeze()
    if das_basis.ndim != 2:
        raise ValueError(f"DAS basis must be 2D; got {tuple(das_basis.shape)}.")
    # A saved DAS basis is normally [d_model, k]; select the larger axis as d_model.
    d_model = max(das_basis.shape)
    das = orthonormal_columns(das_basis, d_model, "DAS basis")
    fourier = orthonormal_columns(fourier_weight, d_model, "Fourier weight")
    raw_fourier = torch.as_tensor(fourier_weight).detach().float().squeeze()
    if raw_fourier.shape[0] != d_model:
        raw_fourier = raw_fourier.T
    individual_projection_lengths = (
        (das.T @ raw_fourier).norm(dim=0)
        / raw_fourier.norm(dim=0).clamp_min(1e-8)
    )
    singular_values = torch.linalg.svdvals(das.T @ fourier).clamp(0, 1)
    squared_sum = singular_values.square().sum()
    fourier_in_das = squared_sum / fourier.shape[1]
    das_in_fourier = squared_sum / das.shape[1]
    random_baseline = das.shape[1] / d_model
    return {
        "fourier_in_das": float(fourier_in_das),
        "das_in_fourier": float(das_in_fourier),
        "goodfire_mean_projection_length": float(individual_projection_lengths.mean()),
        "individual_projection_lengths": [
            float(value) for value in individual_projection_lengths
        ],
        "principal_cosines": [float(value) for value in singular_values],
        "random_baseline": float(random_baseline),
        "fourier_in_das_over_random": float(fourier_in_das / random_baseline),
        "fourier_in_das_excess": float(fourier_in_das - random_baseline),
        "d_model": d_model,
        "das_rank": das.shape[1],
        "fourier_rank": fourier.shape[1],
    }


def find_das_rows(args):
    rows = load_jsonl(args.das_results)
    selected = {}
    for row in rows:
        if (
            row.get("target") == args.das_target
            and int(row.get("layer", -1)) == args.layer
            and str(row.get("position")) == str(args.position)
            and row.get("hook") == args.hook
            and int(row.get("seed", -1)) == args.seed
            and int(row.get("k", -1)) in args.k_dims
        ):
            selected[int(row["k"])] = row
    missing = sorted(set(args.k_dims) - selected.keys())
    if missing:
        raise ValueError(
            f"No matching DAS rows for k={missing} in {args.das_results}."
        )
    return selected


def load_das_basis(row, layer):
    checkpoint = Path(row["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    payload = torch_load_portable(checkpoint)
    bases = payload["bases"]
    basis = bases.get(layer, bases.get(str(layer)))
    if basis is None:
        raise ValueError(f"Checkpoint {checkpoint} has no layer {layer} basis.")
    return basis, checkpoint


def plot_heatmaps(rows, periods, dimensions, output_path):
    metrics = (
        ("fourier_in_das", "Fourier plane contained in DAS"),
        ("fourier_in_das_over_random", "Overlap / random expectation"),
    )
    lookup = {(row["period"], row["k"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        matrix = torch.tensor(
            [[lookup[(period, k)][metric] for k in dimensions] for period in periods]
        ).numpy()
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(dimensions)), labels=dimensions)
        ax.set_yticks(range(len(periods)), labels=periods)
        ax.set_xlabel("DAS dimension k")
        ax.set_ylabel("Fourier period T")
        ax.set_title(title)
        for row_index in range(len(periods)):
            for column_index in range(len(dimensions)):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() * 0.55 else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_metric_by_period(rows, periods, dimensions, metric, title, ylabel, output_path):
    lookup = {(row["period"], row["k"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for k in dimensions:
        values = [lookup[(period, k)][metric] for period in periods]
        ax.plot(
            periods,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5,
            label=f"k={k}",
        )

    ax.set_xscale("log")
    ax.set_xticks(periods, labels=[str(period) for period in periods])
    ax.set_xlabel("Fourier period T")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    fourier_target = args.fourier_target or args.das_target
    das_rows = find_das_rows(args)
    summary = []

    for period in args.periods:
        fourier_probe_path = probe_path(
            args.probe_root,
            args.modality,
            fourier_target,
            period,
            args.layer,
            args.position,
            args.method,
        )
        if not fourier_probe_path.exists():
            raise FileNotFoundError(fourier_probe_path)
        artifact = torch_load_portable(fourier_probe_path)
        for k in args.k_dims:
            das_basis, das_checkpoint = load_das_basis(das_rows[k], args.layer)
            row = {
                "modality": args.modality,
                "das_target": args.das_target,
                "fourier_target": fourier_target,
                "layer": args.layer,
                "position": args.position,
                "hook": args.hook,
                "period": period,
                "k": k,
                "method": args.method,
                "seed": args.seed,
                "fourier_probe_path": str(fourier_probe_path),
                "das_checkpoint_path": str(das_checkpoint),
                "fourier_probe_r2_mean": artifact.get("metrics", {}).get("r2_mean"),
            }
            row.update(overlap_metrics(artifact["weight"], das_basis))
            summary.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.modality}_{args.das_target}_fourier_{fourier_target}_"
        f"layer{args.layer}_pos{args.position}_{args.hook}_seed{args.seed}"
    )
    output_path = args.output_dir / f"{stem}.jsonl"
    save_jsonl(summary, output_path)
    plot_heatmaps(
        summary,
        args.periods,
        args.k_dims,
        args.output_dir / f"{stem}_heatmap.png",
    )
    plot_metric_by_period(
        summary,
        args.periods,
        args.k_dims,
        "fourier_in_das",
        "Fourier plane contained in DAS",
        "Containment",
        args.output_dir / f"{stem}_fourier_in_das_by_period.png",
    )
    plot_metric_by_period(
        summary,
        args.periods,
        args.k_dims,
        "fourier_in_das_over_random",
        "DAS-Fourier overlap over random expectation",
        "Overlap / random",
        args.output_dir / f"{stem}_over_random_by_period.png",
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
