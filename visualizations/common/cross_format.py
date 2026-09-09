"""
Plot cross-format probe accuracy matrices.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_config(metric_name):
    configs = {
        "accuracy": ("accuracy_matrix", "Accuracy", 0.0, 1.0),
        "balanced_accuracy": ("balanced_accuracy_matrix", "Balanced Accuracy", 0.0, 1.0),
        "macro_f1": ("macro_f1_matrix", "Macro F1", 0.0, 1.0),
        "normalized_gain": ("normalized_gain_matrix", "Normalized Gain", -1.0, 1.0),
    }
    return configs[metric_name]


def save_accuracy_matrix_plot(
    formats,
    accuracy_matrix,
    output_path,
    title=None,
    colorbar_label="Accuracy",
    vmin=0.0,
    vmax=1.0,
):

    values = [
        [accuracy_matrix[train_fmt][eval_fmt] for eval_fmt in formats]
        for train_fmt in formats
    ]

    fig, ax = plt.subplots(figsize=(1.5 * len(formats) + 3, 1.2 * len(formats) + 2.5))
    image = ax.imshow(values, vmin=vmin, vmax=vmax, cmap="viridis")

    ax.set_xticks(range(len(formats)))
    ax.set_yticks(range(len(formats)))
    ax.set_xticklabels(formats, rotation=35, ha="right")
    ax.set_yticklabels(formats)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Training task")

    if title is not None:
        ax.set_title(title)

    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            midpoint = ((vmin or 0.0) + (vmax or 1.0)) / 2
            text_color = "white" if value < midpoint else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_matrix_json", type=str)
    parser.add_argument("--output_dir", type=str, default="results/plots")
    parser.add_argument(
        "--metric",
        choices=("accuracy", "balanced_accuracy", "macro_f1", "normalized_gain"),
        default="accuracy",
    )
    args = parser.parse_args()

    with Path(args.probe_matrix_json).open() as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_paths = []
    matrix_key, metric_label, vmin, vmax = plot_config(args.metric)
    metric_suffix = "" if args.metric == "accuracy" else f"_{args.metric}"

    labels = None
    if "tasks" in data:
        labels = [task["label"] for task in data["tasks"]]

    for result in data["results"]:
        if matrix_key not in result:
            raise KeyError(
                f"{matrix_key} not found in {args.probe_matrix_json}. "
                "This metric is only available in newer probe-matrix outputs."
            )
        layer_idx = result["layer"]
        path = output_dir / f"probe_matrix_layer{layer_idx}{metric_suffix}.png"
        output_paths.append(save_accuracy_matrix_plot(
            formats=labels or result["formats"],
            accuracy_matrix=result[matrix_key],
            output_path=path,
            title=f"Probe {metric_label} Matrix, Layer {layer_idx}",
            colorbar_label=metric_label,
            vmin=vmin,
            vmax=vmax,
        ))

    print("Saved plots:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()