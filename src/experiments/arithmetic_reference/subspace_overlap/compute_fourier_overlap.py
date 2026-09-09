"""Measure overlap between Fourier probe subspaces."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.common import save_jsonl
from src.probes.fourier import probe_path
from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        nargs=5,
        action="append",
        metavar=("LABEL", "MODALITY", "TARGET", "LAYER", "PERIOD"),
        required=True,
        help=(
            "Fourier probe to include. Example: "
            "--probe add_T10 addition result 44 10"
        ),
    )
    parser.add_argument(
        "--probe_root",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more Fourier probe roots. With one root, labels are kept "
            "unchanged. With multiple roots, each requested --probe is loaded "
            "from every root and labels are prefixed by a root label."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--position", type=int, default=17)
    parser.add_argument(
        "--root_position",
        nargs=2,
        action="append",
        metavar=("ROOT_LABEL", "POSITION"),
        default=[],
        help=(
            "Optional per-root saved probe position. Example: "
            "--root_position text 17 --root_position image -1. "
            "ROOT_LABEL is the inferred label printed for each probe root."
        ),
    )
    parser.add_argument("--method", choices=["ridge", "gd"], default="ridge")
    parser.add_argument(
        "--stem",
        help="Optional output filename stem. Defaults to position/method.",
    )
    return parser.parse_args()


def load_fourier_basis(probe_root, modality, target, layer, period, position, method):
    path = probe_path(probe_root, modality, target, period, layer, position, method)
    if not path.exists():
        raise FileNotFoundError(path)
    artifact = torch_load_portable(path)
    weight = torch.as_tensor(artifact["weight"]).detach().float().squeeze()
    return weight, path, artifact.get("metrics", {})


def root_label(path):
    parts = [part.lower() for part in Path(path).parts]
    if "baseline_images" in parts:
        return "image"
    if "baseline_bank" in parts or "baseline" in parts:
        return "text"
    meaningful = [
        part
        for part in Path(path).parts
        if part not in {".", "/", "\\"}
    ]
    return meaningful[-2] if len(meaningful) >= 2 and meaningful[-1] == "fourier_probes" else Path(path).name


def unique_root_labels(roots):
    counts = {}
    labels = []
    for root in roots:
        base = root_label(root)
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else f"{base}{counts[base]}")
    return labels


def position_by_root_label(root_labels, default_position, overrides):
    mapping = {label: int(default_position) for label in root_labels}
    for label, position_text in overrides:
        if label not in mapping:
            known = ", ".join(root_labels)
            raise ValueError(
                f"Unknown --root_position label {label!r}. Known labels: {known}."
            )
        mapping[label] = int(position_text)
    return mapping


def subspace_metrics(first_basis, second_basis):
    first_basis = torch.as_tensor(first_basis).detach().float().squeeze()
    second_basis = torch.as_tensor(second_basis).detach().float().squeeze()
    d_model = max(first_basis.shape)
    if max(second_basis.shape) != d_model:
        raise ValueError(
            f"Subspaces have different d_model axes: "
            f"{tuple(first_basis.shape)} vs {tuple(second_basis.shape)}."
        )
    first = orthonormal_columns(first_basis, d_model, "first Fourier basis")
    second = orthonormal_columns(second_basis, d_model, "second Fourier basis")
    singular_values = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    squared_sum = singular_values.square().sum()
    first_in_second = squared_sum / first.shape[1]
    second_in_first = squared_sum / second.shape[1]
    min_rank_containment = squared_sum / min(first.shape[1], second.shape[1])
    random_baseline = second.shape[1] / d_model
    return {
        "first_in_second": float(first_in_second),
        "second_in_first": float(second_in_first),
        "symmetric_overlap": float((first_in_second + second_in_first) / 2),
        "min_rank_containment": float(min_rank_containment),
        "first_in_second_over_random": float(first_in_second / random_baseline),
        "principal_cosines": [float(value) for value in singular_values],
        "first_rank": first.shape[1],
        "second_rank": second.shape[1],
        "d_model": d_model,
    }


def plot_matrix(rows, labels, metric, title, output_path, vmax=1.0):
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    matrix = torch.tensor(
        [[lookup[(first, second)][metric] for second in labels] for first in labels]
    ).numpy()
    fig, ax = plt.subplots(figsize=(1.35 * len(labels) + 2.8, 1.2 * len(labels) + 2.2))
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Second Fourier subspace")
    ax.set_ylabel("First Fourier subspace")
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
                color="white" if value > vmax * 0.45 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.82, label=metric)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    entries = []
    root_labels = unique_root_labels(args.probe_root)
    root_positions = position_by_root_label(
        root_labels=root_labels,
        default_position=args.position,
        overrides=args.root_position,
    )
    print("Fourier probe position mapping:")
    for probe_root, probe_root_label in zip(args.probe_root, root_labels):
        print(f"  {probe_root_label}: position {root_positions[probe_root_label]} ({probe_root})")

    multiple_roots = len(args.probe_root) > 1
    for probe_root, probe_root_label in zip(args.probe_root, root_labels):
        position = root_positions[probe_root_label]
        for label, modality, target, layer_text, period_text in args.probe:
            layer = int(layer_text)
            period = int(period_text)
            basis, path, metrics = load_fourier_basis(
                probe_root,
                modality,
                target,
                layer,
                period,
                position,
                args.method,
            )
            entries.append(
                {
                    "label": f"{probe_root_label}_{label}" if multiple_roots else label,
                    "modality": modality,
                    "target": target,
                    "layer": layer,
                    "period": period,
                    "position": position,
                    "root_label": probe_root_label,
                    "probe_root": str(probe_root),
                    "basis": basis,
                    "path": path,
                    "r2_mean": metrics.get("r2_mean"),
                }
            )

    summary = []
    for first in entries:
        for second in entries:
            row = {
                "first_label": first["label"],
                "second_label": second["label"],
                "first_modality": first["modality"],
                "second_modality": second["modality"],
                "first_target": first["target"],
                "second_target": second["target"],
                "first_layer": first["layer"],
                "second_layer": second["layer"],
                "first_period": first["period"],
                "second_period": second["period"],
                "first_position": first["position"],
                "second_position": second["position"],
                "first_root_label": first["root_label"],
                "second_root_label": second["root_label"],
                "first_probe_root": first["probe_root"],
                "second_probe_root": second["probe_root"],
                "method": args.method,
                "first_probe_path": str(first["path"]),
                "second_probe_path": str(second["path"]),
                "first_probe_r2_mean": first["r2_mean"],
                "second_probe_r2_mean": second["r2_mean"],
            }
            row.update(subspace_metrics(first["basis"], second["basis"]))
            summary.append(row)

    labels = [entry["label"] for entry in entries]
    stem = args.stem or f"fourier_overlap_pos{args.position}_{args.method}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{stem}.jsonl"
    save_jsonl(summary, output_path)
    plot_matrix(
        summary,
        labels,
        "symmetric_overlap",
        "Fourier subspace overlap",
        args.output_dir / f"{stem}_symmetric_overlap.png",
    )
    plot_matrix(
        summary,
        labels,
        "min_rank_containment",
        "Fourier min-rank containment",
        args.output_dir / f"{stem}_min_rank_containment.png",
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
