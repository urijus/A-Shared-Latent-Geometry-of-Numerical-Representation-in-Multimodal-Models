import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.io import load_jsonl
from src.probes.linear import TOP_K_VALUES



def safe_name(value):
    return str(value).replace("/", "_").replace("*", "mul")


def plot_topk(rows, output_path, title):
    layers = sorted({int(row["layer"]) for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5))

    for k in TOP_K_VALUES:
        scores = {int(row["layer"]): row[f"top_{k}_accuracy"] for row in rows}
        ax.plot(
            layers,
            [scores[layer] for layer in layers],
            marker="o",
            markersize=3,
            label=f"top-{k}",
        )

    ax.set(title=title, xlabel="Layer", ylabel="Validation accuracy", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print("Saved:", output_path)


# Filter rows for the grouped plot
def row_matches(row, target=None, position_name=None, modality=None):
    if target is not None and str(row.get("target")) != str(target):
        return False

    if position_name is not None and str(row.get("position_name")) != str(position_name):
        return False

    if modality is not None and str(row.get("modality")) != str(modality):
        return False

    return True


def plot_grouped_topk(
    input_dir,
    output_path,
    title,
    seeds=None,
    prefix="result_seed",
    error="std",          # "std", "sem", or "ci95"
    top_k_values=None,
    show_markers=False,
    modality="multiplication",
    target="result",
    position_name="17",
):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if top_k_values is None:
        top_k_values = TOP_K_VALUES

    if seeds is None:
        files = sorted(input_dir.glob(f"{prefix}*.jsonl"))
    else:
        files = [input_dir / f"{prefix}{seed}.jsonl" for seed in seeds]

    if not files:
        raise ValueError(f"No files found in {input_dir} with prefix '{prefix}'.")

    values = {
        k: defaultdict(list)
        for k in top_k_values
    }

    for file in files:
        if not file.exists():
            raise ValueError(f"Missing file: {file}")

        rows = load_jsonl(file)
        rows = [
            row for row in rows
            if row_matches(
                row,
                target=target,
                position_name=position_name,
                modality=modality,
            )
        ]

        for row in rows:
            layer = int(row["layer"])

            for k in top_k_values:
                key = f"top_{k}_accuracy"
                values[k][layer].append(float(row[key]))

    layers = sorted({
        layer
        for k in top_k_values
        for layer in values[k].keys()
    })

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.0))

    for k in top_k_values:
        means = []
        bands = []

        for layer in layers:
            vals = np.array(values[k][layer], dtype=float)

            mean = vals.mean()
            means.append(mean)

            if len(vals) <= 1:
                band = 0.0
            elif error == "std":
                band = vals.std(ddof=1)
            elif error == "sem":
                band = vals.std(ddof=1) / np.sqrt(len(vals))
            elif error == "ci95":
                band = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
            else:
                raise ValueError("error must be one of: 'std', 'sem', 'ci95'")

            bands.append(band)

        means = np.array(means)
        bands = np.array(bands)

        line, = ax.plot(
            layers,
            means,
            linewidth=1.8,
            marker="o" if show_markers else None,
            markersize=2.5,
            label=f"top-{k}",
        )

        ax.fill_between(
            layers,
            means - bands,
            means + bands,
            alpha=0.15,
            linewidth=0,
            color=line.get_color(),
        )

    ax.set(
        xlabel="Layer",
        ylabel="Validation accuracy",
    )

    ax.set_ylim(0, 1.08)
    ax.set_title(title, pad=12)

    ax.grid(alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    # Also save vector version for paper/thesis
    if output_path.suffix.lower() != ".pdf":
        fig.savefig(output_path.with_suffix(".pdf"))

    plt.close(fig)

    print("Saved:", output_path)
    print("Saved:", output_path.with_suffix(".pdf"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Multiplication: Result at token 17")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--modality",
        choices=["addition", "subtraction", "multiplication"],
        default="multiplication",
    )
    parser.add_argument("--target", default="result")
    parser.add_argument("--position", default="17")
    parser.add_argument("--prefix", default="result_seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_grouped_topk(
        args.input_dir,
        args.output_path,
        args.title,
        args.seeds,
        prefix=args.prefix,
        modality=args.modality,
        target=args.target,
        position_name=args.position,
    )
