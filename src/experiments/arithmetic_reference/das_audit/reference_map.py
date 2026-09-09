"""Plot DAS seed reliability and cross-variable reference overlap maps."""

from pathlib import Path
import argparse
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)


TASKS = [
    ("text", "addition", "result", "Text add\nresult"),
    ("text", "subtraction", "result", "Text sub\nresult"),
    ("image", "addition", "result", "Image add\nresult"),
    ("image", "subtraction", "result", "Image sub\nresult"),
    ("text", "multiplication", "result", "Text mul\nresult"),
    ("text", "multiplication", "c0_hat", r"Text mul $\hat{c}_0$"),
    ("text", "multiplication", "c1_hat_full", r"Text mul $\hat{c}_1$"),
]

TEXT_REFERENCE = [
    ("text", "addition", "result", "Add\nresult"),
    ("text", "subtraction", "result", "Sub\nresult"),
    ("text", "multiplication", "result", "Mul\nresult"),
    ("text", "multiplication", "c1_hat_full", r"Mul $\hat{c}_1$"),
    ("text", "multiplication", "c0_hat", r"Mul $\hat{c}_0$"),
]

BACKGROUND = "#fdfdfd"
AX_FACE = "#f7f8f8"
SPINE = "#b9c0c2"
TEXT = "#171717"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--output_dir", type=Path, default=Path("visualizations/main_paper/DAS_audit"))
    parser.add_argument("--n_seeds", type=int, default=3)
    return parser.parse_args()


def style():
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def folder(root, modality, operation, target, condition):
    path = root / modality / operation
    if target != "result":
        path = path / target
    return path / condition


def basis_path(root, modality, operation, target, condition, seed):
    return folder(root, modality, operation, target, condition) / "split_0" / f"seed_{seed}" / "subspace.pt"


def load_basis(path):
    payload = torch_load_portable(path)
    basis = torch.as_tensor(payload["basis"]).detach().float().squeeze()
    d_model = max(basis.shape)
    return orthonormal_columns(basis, d_model, f"DAS basis {path}")


def load_task(root, modality, operation, target, condition, n_seeds):
    bases = []
    for seed in range(n_seeds):
        path = basis_path(root, modality, operation, target, condition, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        bases.append(load_basis(path))
    return bases


def overlap(first, second):
    singular = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    return float(singular.square().sum() / min(first.shape[1], second.shape[1]))


def seed_matrix(bases):
    n = len(bases)
    matrix = torch.eye(n).tolist()
    for i in range(n):
        for j in range(i + 1, n):
            value = overlap(bases[i], bases[j])
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def off_diagonal_mean(matrix):
    values = [matrix[i][j] for i in range(len(matrix)) for j in range(i + 1, len(matrix))]
    return sum(values) / len(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def heatmap(ax, matrix, labels, title, vmax=0.8, annotate=True, std_matrix=None):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "das_overlap", ["#d8dee0", "#c7d7d4", "#96c8af", "#59aa84", "#2f7f75"]
    )
    ax.set_facecolor(AX_FACE)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    image = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap)
    ax.set_title(title, pad=8)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(length=0)
    if annotate:
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                color = "white" if value > vmax * 0.52 else "#181818"
                if std_matrix is None:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)
                else:
                    ax.text(j, i - 0.09, f"{value:.2f}", ha="center", va="center", fontsize=7.4, color=color)
                    ax.text(
                        j,
                        i + 0.15,
                        fr"$\pm${std_matrix[i][j]:.3f}",
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color=color,
                    )
    return image


def plot_seed_grid(seed_rows, output_dir):
    fig = plt.figure(figsize=(8.5, 4.75), constrained_layout=True)
    fig.patch.set_facecolor(BACKGROUND)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 1])
    axes = list(subfigs[0].subplots(1, 4)) + list(subfigs[1].subplots(1, 3))
    for subfig in subfigs:
        subfig.set_facecolor(BACKGROUND)
    for ax, row in zip(axes, seed_rows):
        mean = row["mean_off_diagonal"]
        heatmap(ax, row["matrix"], ["0", "1", "2"], f"{row['label']}\nmean={mean:.2f}", vmax=1.0)
    fig.savefig(output_dir / "das_seed_reliability_grid.png", dpi=350)
    fig.savefig(output_dir / "das_seed_reliability_grid.pdf")
    plt.close(fig)


def cross_matrix(items):
    labels = [item["label"] for item in items]
    matrix = []
    std_matrix = []
    rows = []
    for first in items:
        matrix_row = []
        std_row = []
        for second in items:
            if first is second:
                value = first["within"]
                values = first.get("within_values", first["off_diagonal_values"])
            else:
                values = [overlap(a, b) for a in first["bases"] for b in second["bases"]]
                value = sum(values) / len(values)
            std = sample_std(values)
            matrix_row.append(value)
            std_row.append(std)
            rows.append({
                "first": first["key"],
                "second": second["key"],
                "symmetric_overlap_mean": value,
                "symmetric_overlap_std": std,
                "n_seed_pairs": len(values),
                "symmetric_overlap_values": values,
            })
        matrix.append(matrix_row)
        std_matrix.append(std_row)
    return labels, matrix, std_matrix, rows


def plot_cross_reference(labels, matrix, std_matrix, output_dir):
    fig, ax = plt.subplots(figsize=(4.85, 4.35))
    fig.patch.set_facecolor(BACKGROUND)
    image = heatmap(ax, matrix, labels, "DAS reference map, text L43, k=22", vmax=0.8, std_matrix=std_matrix)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Symmetric overlap")
    fig.savefig(output_dir / "das_text_reference_map.png", dpi=350)
    fig.savefig(output_dir / "das_text_reference_map.pdf")
    plt.close(fig)


def save_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()

    seed_rows = []
    loaded = {}
    for modality, operation, target, label in TASKS:
        bases = load_task(args.root, modality, operation, target, args.condition, args.n_seeds)
        matrix = seed_matrix(bases)
        values = [
            matrix[i][j]
            for i in range(len(matrix))
            for j in range(i + 1, len(matrix))
        ]
        key = f"{modality}/{operation}/{target}"
        row = {
            "key": key,
            "label": label,
            "matrix": matrix,
            "mean_off_diagonal": off_diagonal_mean(matrix),
            "std_off_diagonal": sample_std(values),
            "off_diagonal_values": values,
        }
        seed_rows.append(row)
        loaded[key] = {**row, "bases": bases}

    save_jsonl(seed_rows, args.output_dir / "das_seed_reliability_matrices.jsonl")
    plot_seed_grid(seed_rows, args.output_dir)

    text_items = []
    for modality, operation, target, label in TEXT_REFERENCE:
        key = f"{modality}/{operation}/{target}"
        text_items.append(
            {
                **loaded[key],
                "key": key,
                "label": label,
                "within": loaded[key]["mean_off_diagonal"],
                "within_values": loaded[key]["off_diagonal_values"],
            }
        )
    labels, matrix, std_matrix, rows = cross_matrix(text_items)
    save_jsonl(rows, args.output_dir / "das_text_reference_map.jsonl")
    plot_cross_reference(labels, matrix, std_matrix, args.output_dir)

    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
