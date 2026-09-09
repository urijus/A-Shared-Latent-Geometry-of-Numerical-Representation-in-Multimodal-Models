"""Appendix plots for DAS layer sweeps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "das_v2"
    / "runs"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "baseline" / "das"

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"

K_COLORS = {
    8: "#b85f5d",
    16: "#d08b57",
    32: "#5f9b92",
    64: "#6f7fae",
    128: "#8a6aa3",
}

METRICS = [
    ("variable_teacher_forced_iia", "Variable IIA"),
    ("full_answer_teacher_forced_iia", "Teacher-forced IIA"),
    ("autoregressive_iia", "Autoregressive IIA"),
]

VARIABLE_LABELS = {
    ("addition", "result"): "Addition result",
    ("subtraction", "result"): "Subtraction result",
    ("multiplication", "result"): "Multiplication result",
    ("multiplication", "c0"): r"Multiplication $c_0$",
    ("multiplication", "c0_hat"): r"Multiplication $\hat{c}_0$",
    ("multiplication", "c1_hat"): r"Multiplication simple $\hat{c}_1$",
    ("multiplication", "c1_hat_full"): r"Multiplication $\hat{c}_1$",
}


def long_path(path):
    path = Path(path)
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_jsonl(path):
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib():
    plt.rcParams.update(
        {
            "font.family": [
                "Palatino Linotype",
                "Georgia",
                "Cambria",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.6,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.68)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.28)


def variable_label(operation, target):
    return VARIABLE_LABELS.get(
        (operation, target),
        f"{operation.replace('_', ' ').title()} {target.replace('_', ' ')}",
    )


def output_stem(operation, target):
    if operation == "multiplication" and target == "c1_hat":
        return "das_multiplication_simple_c1_hat"
    if operation == "multiplication" and target == "c1_hat_full":
        return "das_multiplication_c1_hat"
    return f"das_{operation}_{target}".replace("__", "_")


def deduplicate_rows(rows):
    best = {}
    for row in rows:
        if "layer" not in row or "k" not in row:
            continue
        key = (int(row["layer"]), int(row["k"]))
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        current_score = float(current.get("autoregressive_iia", -1.0))
        candidate_score = float(row.get("autoregressive_iia", -1.0))
        if candidate_score >= current_score:
            best[key] = row
    return list(best.values())


def discover_run_files(run_dir, position, hook, seed):
    pattern = f"*_pos{position}_{hook}_initrandom_pca_seed{seed}.jsonl"
    files = sorted(Path(run_dir).glob(pattern))
    return [
        path
        for path in files
        if path.name.count("_") >= 2 and "combined" not in path.name
    ]


def filter_rows(rows, position, hook, layers, ks):
    layer_set = {int(layer) for layer in layers} if layers else None
    k_set = {int(k) for k in ks} if ks else None
    filtered = []
    for row in rows:
        if str(row.get("position")) != str(position):
            continue
        if str(row.get("hook")) != str(hook):
            continue
        if layer_set is not None and int(row.get("layer", -1)) not in layer_set:
            continue
        if k_set is not None and int(row.get("k", -1)) not in k_set:
            continue
        if not all(metric in row for metric, _ in METRICS):
            continue
        filtered.append(row)
    return deduplicate_rows(filtered)


def sorted_layers(rows):
    return sorted({int(row["layer"]) for row in rows})


def sorted_ks(rows, requested_ks):
    available = sorted({int(row["k"]) for row in rows})
    if requested_ks:
        requested = [int(k) for k in requested_ks]
        return [k for k in requested if k in available]
    return available


def plot_variable(rows, operation, target, output_dir, requested_ks=None):
    rows = deduplicate_rows(rows)
    if not rows:
        return []

    layers = sorted_layers(rows)
    ks = sorted_ks(rows, requested_ks)
    if not layers or not ks:
        return []

    style_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), sharex=True, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)

    handles = []
    labels = []
    for ax, (metric, title) in zip(axes, METRICS):
        setup_axis(ax)
        for k in ks:
            selected = sorted(
                (row for row in rows if int(row["k"]) == k),
                key=lambda row: int(row["layer"]),
            )
            if not selected:
                continue
            x = [int(row["layer"]) for row in selected]
            y = [float(row[metric]) for row in selected]
            line = ax.plot(
                x,
                y,
                color=K_COLORS.get(k, "#5d6264"),
                marker="o",
                markersize=3.5,
                linewidth=1.45,
                label=f"k={k}",
            )[0]
            if metric == METRICS[0][0]:
                handles.append(line)
                labels.append(f"k={k}")
        ax.set_title(title, pad=7, fontweight="normal")
        ax.set_xlabel("Layer")
        ax.set_xticks(layers)
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("IIA")
    fig.suptitle(f"DAS for {variable_label(operation, target)}", y=0.965)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 5),
        handlelength=1.7,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.26, wspace=0.18)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(operation, target)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def discover_position_sweep_files(run_dir, hook, seed):
    pattern = f"*_pos*_{hook}_initrandom_pca_seed{seed}.jsonl"
    files = sorted(Path(run_dir).glob(pattern))
    return [
        path
        for path in files
        if path.name.count("_") >= 2 and "combined" not in path.name
    ]


def filter_position_rows(rows, hook, layer, k):
    filtered = []
    for row in rows:
        if str(row.get("hook")) != str(hook):
            continue
        if int(row.get("layer", -1)) != int(layer):
            continue
        if int(row.get("k", -1)) != int(k):
            continue
        if "position" not in row:
            continue
        if not all(metric in row for metric, _ in METRICS):
            continue
        filtered.append(row)
    return filtered


def position_sort_key(position):
    text = str(position)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def plot_position_sweep(rows_by_variable, output_dir, layer, k):
    rows_by_variable = {
        key: sorted(rows, key=lambda row: position_sort_key(row["position"]))
        for key, rows in rows_by_variable.items()
        if rows
    }
    if not rows_by_variable:
        return []

    style_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.35), sharex=False, sharey=True)
    fig.patch.set_facecolor(BACKGROUND)

    variable_colors = [
        "#b85f5d",
        "#5f9b92",
        "#6f7fae",
        "#8a6aa3",
        "#d08b57",
    ]
    handles = []
    labels = []
    all_positions = sorted(
        {str(row["position"]) for rows in rows_by_variable.values() for row in rows},
        key=position_sort_key,
    )
    position_to_x = {position: index for index, position in enumerate(all_positions)}
    for ax, (metric, title) in zip(axes, METRICS):
        setup_axis(ax)
        for color, ((operation, target), rows) in zip(variable_colors, rows_by_variable.items()):
            positions = [str(row["position"]) for row in rows]
            x = [position_to_x[position] for position in positions]
            y = [float(row[metric]) for row in rows]
            line = ax.plot(
                x,
                y,
                color=color,
                marker="o",
                markersize=4.0,
                linewidth=1.55,
                label=variable_label(operation, target),
            )[0]
            if metric == METRICS[0][0]:
                handles.append(line)
                labels.append(variable_label(operation, target))
        ax.set_xticks(range(len(all_positions)), all_positions)
        ax.set_title(title, pad=7, fontweight="normal")
        ax.set_xlabel("Position")
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("IIA")
    fig.suptitle(f"DAS position sweep at layer {layer}, k={k}", y=0.965)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 3),
        handlelength=1.7,
        columnspacing=1.3,
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.28, wspace=0.18)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"das_position_sweep_layer{layer}_k{k}.png"
    pdf_path = output_dir / f"das_position_sweep_layer{layer}_k{k}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return [png_path, pdf_path]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--position", default="17")
    parser.add_argument("--hook", default="resid_post")
    parser.add_argument("--seed", default="8")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional layer subset. Defaults to every layer found in each file.",
    )
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="DAS k values to draw when available.",
    )
    parser.add_argument(
        "--files",
        type=Path,
        nargs="+",
        help="Optional explicit DAS JSONL files. Defaults to discovered run files.",
    )
    parser.add_argument(
        "--plot_position_sweep",
        action="store_true",
        help="Also plot IIA against position for a fixed layer and k.",
    )
    parser.add_argument("--position_layer", type=int, default=43)
    parser.add_argument("--position_k", type=int, default=32)
    parser.add_argument(
        "--position_files",
        type=Path,
        nargs="+",
        help="Optional explicit JSONL files for the position-sweep plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    files = args.files or discover_run_files(args.run_dir, args.position, args.hook, args.seed)
    if not files:
        raise FileNotFoundError(f"No DAS JSONL files found in {args.run_dir}")

    outputs = []
    for path in files:
        rows = load_jsonl(path)
        rows = filter_rows(rows, args.position, args.hook, args.layers, args.ks)
        if not rows:
            print(f"Skipping {path}: no rows after filtering")
            continue
        operation = rows[0].get("operation", "unknown")
        target = rows[0].get("target", path.stem)
        outputs.extend(plot_variable(rows, operation, target, args.output_dir, args.ks))

    if args.plot_position_sweep:
        rows_by_variable = {}
        position_files = args.position_files or discover_position_sweep_files(
            args.run_dir,
            args.hook,
            args.seed,
        )
        for path in position_files:
            position_rows = filter_position_rows(
                load_jsonl(path),
                hook=args.hook,
                layer=args.position_layer,
                k=args.position_k,
            )
            if not position_rows:
                continue
            operation = position_rows[0].get("operation", "unknown")
            target = position_rows[0].get("target", path.stem)
            rows_by_variable.setdefault((operation, target), []).extend(position_rows)
        outputs.extend(
            plot_position_sweep(
                rows_by_variable,
                args.output_dir,
                layer=args.position_layer,
                k=args.position_k,
            )
        )

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
