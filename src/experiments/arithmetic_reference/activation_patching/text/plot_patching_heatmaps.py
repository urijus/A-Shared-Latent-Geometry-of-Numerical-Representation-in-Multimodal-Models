import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.arithmetic_reference.activation_patching.text.common import OPERATIONS


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    for row in rows:
        if row["layer"] != "all":
            row["layer"] = int(row["layer"])
        row["recovery"] = None if row["recovery"] in {None, "", "None"} else float(row["recovery"])
    return rows


def mean_metric_matrix(rows, hook, metric):
    rows = [row for row in rows if row["hook"] == hook]
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    if all(isinstance(layer, int) for layer in layers):
        layers = sorted(layers)
    positions = list(dict.fromkeys(row["position"] for row in rows))
    values = np.full((len(positions), len(layers)), np.nan)
    grouped = defaultdict(list)

    for row in rows:
        if row.get(metric) is not None:
            grouped[(row["layer"], row["position"])].append(float(row[metric]))

    for i, position in enumerate(positions):
        for j, layer in enumerate(layers):
            scores = grouped[(layer, position)]
            if scores:
                values[i, j] = sum(scores) / len(scores)
    return values, layers, positions


def mean_recovery_matrix(rows, hook):
    return mean_metric_matrix(rows, hook, "recovery")


def plot(rows, operation, hook, metric, output_path):
    matrix, layers, positions = mean_metric_matrix(rows, hook, metric)
    fig, ax = plt.subplots(
        figsize=(max(8, len(layers) * 0.3), max(5, len(positions) * 0.5))
    )
    if metric == "recovery":
        image = ax.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        colorbar_label = "Mean recovery"
    else:
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        colorbar_label = "Exact clean-answer IIA"
    ax.set_title(f"{operation}: {hook} {metric}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Semantic prompt position")
    ax.set_xticks(range(len(layers)), layers, rotation=90)
    ax.set_yticks(range(len(positions)), positions)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def default_inputs(output_dir, operation):
    folder = output_dir / operation
    paths = sorted(folder.glob(f"{operation}_activation_patching_layers*.jsonl"))
    return paths or sorted(folder.glob(f"{operation}_activation_patching*.jsonl"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-dir", default="results/activation_patching")
    parser.add_argument("--operation", choices=sorted(OPERATIONS), default=None)
    parser.add_argument(
        "--layer-mode",
        choices=["individual", "together"],
        default="individual",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.input_path:
        paths = [Path(args.input_path)]
    elif args.operation:
        paths = default_inputs(output_dir, args.operation)
    else:
        paths = [
            path
            for operation in OPERATIONS
            for path in default_inputs(output_dir, operation)
        ]

    if not paths:
        raise ValueError("No result files found")

    for path in paths:
        rows = load_rows(path)
        if not rows:
            continue
        operation = rows[0]["operation"]
        layer_mode = rows[0].get("layer_mode", "individual")
        if layer_mode != args.layer_mode:
            continue
        for hook in sorted({row["hook"] for row in rows}):
            for metric in ("recovery", "patched_clean_iia"):
                output_path = (
                    output_dir
                    / operation
                    / "plots"
                    / f"{path.stem}_{hook}_{metric}_heatmap.png"
                )
                plot(rows, operation, hook, metric, output_path)
                print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
