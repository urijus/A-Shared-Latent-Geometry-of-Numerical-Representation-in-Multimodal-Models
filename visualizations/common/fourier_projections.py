import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECTION_KEYS = [
    ("raw_direction_projection", "Raw probe directions"),
    ("unit_direction_projection", "Unit probe directions"),
    ("orthonormal_plane_projection", "Orthonormal plane"),
]


def safe_name(path):
    return path.stem.replace("_projections", "")


def plot_projection_file(projection_path, output_dir):
    data = torch.load(projection_path, map_location="cpu")
    metadata = data["metadata"]
    labels = data["labels"]
    target = metadata["target"]
    period = int(metadata["period"])

    colors = torch.tensor(
        [int(row[target]) % period for row in labels],
        dtype=torch.float32,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    scatter = None
    for ax, (key, title) in zip(axes, PROJECTION_KEYS):
        coords = data[key].float()
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            s=8,
            alpha=0.75,
            cmap="twilight",
        )
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(
        f"{target} mod {period} | layer {metadata['layer']} | position {metadata['position']}"
    )
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.85)
    cbar.set_label(f"{target} mod {period}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name(projection_path)}_bases.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def iter_projection_paths(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*_projections.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = iter_projection_paths(args.input_path)
    if not paths:
        raise FileNotFoundError(f"No *_projections.pt files found in {args.input_path}")

    for projection_path in paths:
        output_path = plot_projection_file(projection_path, args.output_dir)
        print("Saved plot:", output_path)


if __name__ == "__main__":
    main()
