"""Replot a subset of an existing subspace-overlap JSONL."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.common import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metric",
        default="min_rank_containment",
        choices=[
            "symmetric_overlap",
            "min_rank_containment",
            "first_in_second",
            "second_in_first",
            "first_in_second_over_random",
        ],
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Exact labels to keep, in the desired order.",
    ) # exact labels add_result_T10 sub_result_T10 mul_result_T10
    parser.add_argument(
        "--include_regex",
        action="append",
        help="Regex for labels to keep. Can be passed multiple times.",
    ) # "^(add|sub|mul)_result_T", "_T10$", "^mul_"
    parser.add_argument("--title", default=None)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--annotate_threshold", type=float, default=0.05)
    return parser.parse_args()


def ordered_labels(rows):
    labels = []
    seen = set()
    for row in rows:
        for key in ("first_label", "second_label"):
            label = row[key]
            if label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def select_labels(rows, labels, regexes):
    available = ordered_labels(rows)
    if labels:
        missing = [label for label in labels if label not in available]
        if missing:
            raise ValueError(f"Labels not found in input: {missing}")
        return labels

    patterns = [re.compile(pattern) for pattern in regexes or []]
    if not patterns:
        return available
    selected = [
        label
        for label in available
        if any(pattern.search(label) for pattern in patterns)
    ]
    if not selected:
        raise ValueError("No labels matched --include_regex.")
    return selected


def plot_subset(rows, labels, metric, title, output, vmax, annotate_threshold):
    lookup = {(row["first_label"], row["second_label"]): row for row in rows}
    matrix = torch.tensor(
        [[lookup[(first, second)][metric] for second in labels] for first in labels]
    ).numpy()

    size = max(3.2, min(10.0, 0.55 * len(labels) + 2.4))
    fig, ax = plt.subplots(figsize=(size + 1.1, size))
    image = ax.imshow(matrix, cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Second subspace")
    ax.set_ylabel("First subspace")
    ax.set_title(title or metric.replace("_", " "))

    for row_index in range(len(labels)):
        for column_index in range(len(labels)):
            value = matrix[row_index, column_index]
            if value < annotate_threshold and row_index != column_index:
                continue
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
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_jsonl(args.input)
    labels = select_labels(rows, args.labels, args.include_regex)
    plot_subset(
        rows,
        labels,
        args.metric,
        args.title,
        args.output,
        args.vmax,
        args.annotate_threshold,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
