import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common import load_jsonl
from src.experiments.arithmetic_reference.das.text.plot_results import plot_das_results


def _save_lines(rows, metrics, ylabel, output_path):
    rows = sorted(rows, key=lambda row: row["layers"][0])
    layers = [row["layers"][0] for row in rows]
    plt.figure(figsize=(7, 4))
    for metric, label in metrics:
        plt.plot(layers, [row[metric] for row in rows], marker="o", label=label)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_das_plots(rows, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = sorted({row["subspace_dim"] for row in rows})
    gains = [
        sum(row["variable_specific_iia"] for row in rows if row["subspace_dim"] == dimension)
        / sum(row["subspace_dim"] == dimension for row in rows)
        for dimension in dimensions
    ]
    plt.figure(figsize=(6, 4))
    plt.plot(dimensions, gains, marker="o")
    plt.xlabel("DAS subspace dimension")
    plt.ylabel("Validation Variable Specific IIA")
    plt.tight_layout()
    plt.savefig(output_dir / f"{stem}_dimension_sweep.png", dpi=180)
    plt.close()

    individual = [row for row in rows if len(row["layers"]) == 1]
    for position in sorted({str(row["position"]) for row in individual}):
        for dimension in dimensions:
            selected = [
                row for row in individual
                if str(row["position"]) == position
                and row["subspace_dim"] == dimension
            ]
            if not selected:
                continue
            suffix = f"{stem}_pos{position}_k{dimension}"
            _save_lines(
                selected,
                [
                    ("candidate_top1_iia", "top-1"),
                    ("candidate_top5_iia", "top-5"),
                    ("candidate_top10_iia", "top-10"),
                ],
                "Candidate-set IIA",
                output_dir / f"{suffix}_candidate_iia.png",
            )
            _save_lines(
                selected,
                [
                    ("source_gain", "source gain"),
                    ("base_change", "base change"),
                    ("contrast_gain", "contrast gain"),
                ],
                "Mean log-probability change",
                output_dir / f"{suffix}_behavioral_effects.png",
            )


def get_layer(row):
    if "layer" in row:
        return int(row["layer"])
    elif "layers" in row:
        # your DAS rows often have "layers": [38]
        return int(row["layers"][0])
    else:
        raise KeyError("Row has neither 'layer' nor 'layers'")


def get_metric(row, metric):
    if metric == "variable_iia":
        for key in ("variable_teacher_forced_iia", "variable_specific_iia"):
            if key in row:
                return float(row[key])
        raise KeyError(
            "Row has neither 'variable_teacher_forced_iia' nor "
            "'variable_specific_iia'."
        )
    elif metric == "autoregressive_iia":
        return float(row["autoregressive_iia"])
    else:
        raise ValueError("metric must be 'variable_iia' or 'autoregressive_iia'")


def metric_label(metric):
    if metric == "variable_iia":
        return "Variable-specific IIA"
    if metric == "autoregressive_iia":
        return "Autoregressive IIA"
    raise ValueError("metric must be 'variable_iia' or 'autoregressive_iia'")


def save_pdf_and_png(fig, output_path, dpi=300):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), dpi=dpi)
    fig.savefig(output_path.with_suffix(".png"), dpi=dpi)


# Plot metrics vs layer for multiple das variables
def plot_temporal_das_from_files(
    file_dict,
    output_path,
    metric="variable_iia",
    layers=(38, 40, 42, 43, 44, 45, 46),
    k=16,
    hook="resid_post",
    position="17",   # use "last" if that is what your files contain
    title=None,
):
    """
    file_dict: dict like
        {
            "result": path1,
            "c1": path2,
            "c2": path3,
            "c3": path4,
        }
    """

    layers = list(layers)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    for label, file_path in file_dict.items():
        rows = load_jsonl(Path(file_path))

        # filter rows
        rows = [
            row for row in rows
            if get_layer(row) in layers
            and str(row.get("hook")) == str(hook)
            and int(row.get("k")) == int(k)
            and str(row.get("position")) == str(position)
        ]
        rows_by_layer = {}
        for row in rows:
            rows_by_layer[get_layer(row)] = row

        # sort by layer
        rows = sorted(rows_by_layer.values(), key=get_layer)

        x = [get_layer(row) for row in rows]
        y = [get_metric(row, metric) for row in rows]

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=label,
        )

    ylabel = metric_label(metric)

    if title is None:
        title = f"Temporal DAS ({ylabel})"

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, pad=10)

    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    output_path = Path(output_path)
    save_pdf_and_png(fig, output_path)

    plt.close(fig)
    print("Saved:", output_path.with_suffix(".pdf"))
    print("Saved:", output_path.with_suffix(".png"))


def plot_das_dimension_from_files(
    file_dict,
    output_path,
    metric="variable_iia",
    layer=44,
    ks=(16, 32),
    hook="resid_post",
    position="17",
    title=None,
):
    """Plot a DAS metric against subspace dimension k at one layer/position."""
    ks = list(ks)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    for label, file_path in file_dict.items():
        rows = load_jsonl(Path(file_path))
        rows = [
            row for row in rows
            if get_layer(row) == int(layer)
            and str(row.get("hook")) == str(hook)
            and int(row.get("k")) in ks
            and str(row.get("position")) == str(position)
        ]
        rows_by_k = {}
        for row in rows:
            rows_by_k[int(row["k"])] = row
        rows = [rows_by_k[k] for k in ks if k in rows_by_k]
        if not rows:
            continue

        x = [int(row["k"]) for row in rows]
        y = [get_metric(row, metric) for row in rows]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=label,
        )

    ylabel = metric_label(metric)
    if title is None:
        title = f"DAS dimension sweep: {ylabel}"

    ax.set_xlabel("DAS subspace dimension k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.02)
    ax.set_title(title, pad=10)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    output_path = Path(output_path)
    save_pdf_and_png(fig, output_path)

    plt.close(fig)
    print("Saved:", output_path.with_suffix(".pdf"))
    print("Saved:", output_path.with_suffix(".png"))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DAS metrics from existing JSONL files.")
    parser.add_argument(
        "--mode",
        choices=["temporal", "dimension", "both"],
        default="both",
        help="temporal plots metric vs layer; dimension plots metric vs k.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["variable_iia", "autoregressive_iia"],
        default=["variable_iia", "autoregressive_iia"],
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[30, 32, 34, 36, 38, 40, 42, 43, 44, 45, 46])
    parser.add_argument("--layer", type=int, default=44, help="Layer for dimension plots.")
    parser.add_argument("--k", type=int, default=None, help="Single DAS k for temporal plots.")
    parser.add_argument(
        "--temporal_ks",
        type=int,
        nargs="+",
        default=[16, 32],
        help="DAS k values for temporal plots.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128], help="DAS k values for dimension plots.")
    parser.add_argument("--position", default="17")
    parser.add_argument("--hook", default="resid_post")
    parser.add_argument(
        "--series",
        nargs="+",
        default=["result_add", "result_sub"],
        help="Subset of plotted series labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./results/baseline/gemma4_12b_it/digits/das_v2/plots"),
    )
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    files = {
        "result_add": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/addition_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "result_sub": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/subtraction_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "result_mul": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "c1_hat": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_c1_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "c1_hat_full": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_c1_hat_full_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "mul_T100": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "c0_c1_combined": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_combined_pos17_resid_post_seed8.jsonl",
        "c0_p10": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_c0_pos10_resid_post_initrandom_pca_seed8.jsonl",
        "c0_p17": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_c0_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "c0_p17": "./results/baseline/gemma4_12b_it/digits/das_v2/runs_bons/multiplication_c0_hat_pos17_resid_post_initrandom_pca_seed8.jsonl"
    }
    unknown = sorted(set(args.series) - set(files))
    if unknown:
        raise ValueError(f"Unknown series {unknown}. Available series: {sorted(files)}")
    files = {label: files[label] for label in args.series}

    metric_stems = {
        "variable_iia": "variable_iia",
        "autoregressive_iia": "autoregressive_iia",
    }
    metric_titles = {
        "variable_iia": "variable-specific IIA",
        "autoregressive_iia": "autoregressive IIA",
    }
    temporal_ks = [args.k] if args.k is not None else args.temporal_ks

    for metric in args.metrics:
        if args.mode in {"temporal", "both"} and metric == args.metrics[0]:
            for label, file_path in files.items():
                rows = [
                    row for row in load_jsonl(Path(file_path))
                    if get_layer(row) in args.layers
                    and str(row.get("hook")) == str(args.hook)
                    and str(row.get("position")) == str(args.position)
                    and int(row.get("k", -1)) in {16, 32}
                ]
                plot_das_results(rows, args.output_dir / label)
        if args.mode in {"dimension", "both"}:
            plot_das_dimension_from_files(
                file_dict=files,
                output_path=args.output_dir / (
                    f"dimension_das_layer{args.layer}_pos{args.position}_"
                    f"{metric_stems[metric]}.pdf"
                ),
                metric=metric,
                layer=args.layer,
                ks=args.ks,
                hook=args.hook,
                position=args.position,
                title=f"DAS dimension sweep: {metric_titles[metric]}",
            )
