import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.io import load_jsonl


POSITION_NAMES = ["a", "+", "b", "="]


def rows_to_matrix(rows, target, metric, modality=None, period=None):
    target_rows = [r for r in rows if r["target"] == target]

    if modality is not None:
        target_rows = [r for r in target_rows if r.get("modality") == modality]

    if period is not None:
        target_rows = [r for r in target_rows if r.get("period") == period]

    if not target_rows:
        modality_msg = "" if modality is None else f", modality={modality}"
        period_msg = "" if period is None else f", period={period}"
        raise ValueError(f"No rows found for target={target}{modality_msg}{period_msg}")

    layers = sorted({int(r["layer"]) for r in target_rows})

    def position_sort_key(value):
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    def row_position_name(row):
        if "position_name" in row:
            return str(row["position_name"])
        if "position" in row:
            return str(row["position"])
        return str(row["position_idx"])

    positions = sorted(
        {row_position_name(r) for r in target_rows},
        key=position_sort_key,
    )
    layer_index = {layer: idx for idx, layer in enumerate(layers)}
    position_index = {position: idx for idx, position in enumerate(positions)}

    matrix = np.full((len(positions), len(layers)), np.nan)

    for r in target_rows:
        layer = int(r["layer"])
        position = row_position_name(r)
        matrix[position_index[position], layer_index[layer]] = r[metric]

    return matrix, layers, positions


def get_position_names(rows):
    if "position_names" in rows[0]:
        return rows[0]["position_names"]
    else:
        raise ValueError("position_names not in activation extraction .jsonl")


def plot_heatmap(matrix, target, metric, output_path, position_names, layer_names, modality=None, period=None):
    fig, ax = plt.subplots(figsize=(12, 3.5))

    vmin = -1.0 if metric.startswith("r2_") or metric == "normalized_gain" else 0.0

    im = ax.imshow(
        matrix,
        aspect="auto",
        vmin=vmin,
        vmax=1.0,
    )

    title = f"{target} probe - {metric}"
    if modality is not None:
        title = f"{modality} {title}"
    if period is not None:
        title += f" (T={period})"
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token position")

    ax.set_yticks(range(len(position_names)))
    ax.set_yticklabels(position_names)

    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=90)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def discover_targets(rows):
    return sorted({r["target"] for r in rows})


def discover_modalities(rows):
    modalities = sorted({r.get("modality") for r in rows if r.get("modality") is not None})
    return modalities or [None]


def filter_rows_by_modalities(rows, modalities):
    if modalities is None:
        return rows
    modalities = set(modalities)
    return [r for r in rows if r.get("modality") in modalities]


def discover_periods(rows):
    return sorted({r["period"] for r in rows if "period" in r})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        default="normalized_gain",
        choices=[
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "normalized_gain",
            "top_1_accuracy",
            "top_2_accuracy",
            "top_5_accuracy",
            "top_10_accuracy",
            "top_50_accuracy",
            "r2_cos",
            "r2_sin",
            "r2_mean",
        ],
    )
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--modalities", nargs="+", default=None)
    parser.add_argument("--periods", type=int, nargs="+", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    rows = load_jsonl(input_path)
    rows = filter_rows_by_modalities(rows, args.modalities)
    if not rows:
        raise ValueError(f"No rows left after filtering modalities={args.modalities}")

    targets = args.targets
    if targets is None:
        targets = discover_targets(rows)

    modalities = args.modalities
    if modalities is None:
        modalities = discover_modalities(rows)

    periods = args.periods
    if periods is None:
        periods = discover_periods(rows)
    if not periods:
        periods = [None]

    print("Targets:", targets)
    if modalities != [None]:
        print("Modalities:", modalities)
    if periods != [None]:
        print("Periods:", periods)

    for modality in modalities:
        modality_rows = (
            rows
            if modality is None
            else [r for r in rows if r.get("modality") == modality]
        )
        position_names = get_position_names(modality_rows)

        for target in targets:
            if (modality, target) in {
                ("addition", "requires_borrow"),
                ("subtraction", "requires_carry"),
            }:
                continue
            for period in periods:
                matrix, layer_names, position_names = rows_to_matrix(
                    rows=rows,
                    target=target,
                    metric=args.metric,
                    modality=modality,
                    period=period,
                )

                modality_suffix = "" if modality is None else f"_{modality}"
                period_suffix = "" if period is None else f"_T{period}"
                output_path = output_dir / f"{input_path.stem}{modality_suffix}_{target}{period_suffix}_{args.metric}.png"

                plot_heatmap(
                    matrix=matrix,
                    target=target,
                    metric=args.metric,
                    output_path=output_path,
                    position_names=position_names,
                    layer_names=layer_names,
                    modality=modality,
                    period=period,
                )

                print("Saved:", output_path)


if __name__ == "__main__":
    main()
