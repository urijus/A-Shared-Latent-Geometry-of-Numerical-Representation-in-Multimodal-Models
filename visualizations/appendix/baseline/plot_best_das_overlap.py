import json
import pathlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "results" / "baseline" / "gemma4_12b_it" / "digits" / "das_v2" / "runs"
OUTPUT_DIR = REPO_ROOT / "visualizations" / "01_arithmetic_reference" / "reference_outputs"

BACKGROUND = "#f3f5f5"
SPINE = "#b9c0c2"
TEXT = "#171717"

SUBSPACES = [
    {
        "key": "add_result",
        "label": "Add\nresult\nL44",
        "run_file": "addition_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "target": "result",
        "layer": 44,
    },
    {
        "key": "sub_result",
        "label": "Sub\nresult\nL44",
        "run_file": "subtraction_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "target": "result",
        "layer": 44,
    },
    {
        "key": "mul_result",
        "label": "Mul\nresult\nL42",
        "run_file": "multiplication_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "target": "result",
        "layer": 42,
    },
    {
        "key": "mul_c1_hat",
        "label": r"Mul $\hat{c}_1$" + "\nL44",
        "run_file": "multiplication_c1_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "target": "c1_hat",
        "layer": 44,
    },
    {
        "key": "mul_c0_hat",
        "label": r"Mul $\hat{c}_0$" + "\nL36",
        "run_file": "multiplication_c0_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
        "target": "c0_hat",
        "layer": 36,
    },
]


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def torch_load_portable(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except NotImplementedError as error:
        if "PosixPath" not in str(error) or not hasattr(pathlib, "WindowsPath"):
            raise
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            return torch.load(path, map_location="cpu", weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path


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
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.titlesize": 16,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
        }
    )


def find_row(spec):
    rows = load_jsonl(RUN_DIR / spec["run_file"])
    matches = [
        row
        for row in rows
        if row.get("target") == spec["target"]
        and int(row.get("layer", -1)) == int(spec["layer"])
        and int(row.get("k", -1)) == 32
        and str(row.get("position")) == "17"
        and row.get("hook") == "resid_post"
        and int(row.get("seed", -1)) == 8
    ]
    if not matches:
        raise ValueError(f"No matching DAS row for {spec}")
    return matches[0]


def load_basis(spec):
    row = find_row(spec)
    checkpoint = REPO_ROOT / row["checkpoint_path"]
    payload = torch_load_portable(checkpoint)
    bases = payload["bases"]
    basis = bases.get(spec["layer"], bases.get(str(spec["layer"])))
    if basis is None:
        raise ValueError(f"No basis for layer {spec['layer']} in {checkpoint}")
    return torch.as_tensor(basis).detach().float(), row, checkpoint


def orthonormal_columns(matrix, d_model):
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    if matrix.shape[0] == d_model:
        columns = matrix
    elif matrix.shape[1] == d_model:
        columns = matrix.T
    else:
        raise ValueError(f"Cannot infer d_model={d_model} axis from {tuple(matrix.shape)}")
    left, singular_values, _ = torch.linalg.svd(columns, full_matrices=False)
    tolerance = max(columns.shape) * torch.finfo(columns.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    return left[:, :rank]


def symmetric_overlap(first_basis, second_basis):
    first_basis = torch.as_tensor(first_basis).detach().float().squeeze()
    second_basis = torch.as_tensor(second_basis).detach().float().squeeze()
    d_model = max(first_basis.shape)
    first = orthonormal_columns(first_basis, d_model)
    second = orthonormal_columns(second_basis, d_model)
    singular_values = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    squared_sum = singular_values.square().sum()
    first_in_second = squared_sum / first.shape[1]
    second_in_first = squared_sum / second.shape[1]
    return float((first_in_second + second_in_first) / 2)


def make_matrix():
    entries = []
    for spec in SUBSPACES:
        basis, row, checkpoint = load_basis(spec)
        entries.append({**spec, "basis": basis, "row": row, "checkpoint": checkpoint})

    matrix = np.zeros((len(entries), len(entries)), dtype=float)
    for row_index, first in enumerate(entries):
        for col_index, second in enumerate(entries):
            matrix[row_index, col_index] = symmetric_overlap(first["basis"], second["basis"])
    return entries, matrix


def plot_matrix(entries, matrix):
    style_matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)

    plot_matrix = matrix.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8dee0")
    image = ax.imshow(plot_matrix, vmin=0, vmax=0.45, cmap=cmap)

    labels = [entry["label"] for entry in entries]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Best-layer DAS overlap", pad=12)

    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            value = matrix[row_index, col_index]
            if row_index == col_index:
                text = "1"
                color = "#5e6668"
            else:
                text = f"{value:.2f}"
                color = "white" if value > 0.25 else "#181818"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=9, color=color)

    highlights = [(0, 1), (1, 0), (3, 4), (4, 3), (2, 3), (3, 2), (2, 4), (4, 2)]
    for row_index, col_index in highlights:
        ax.add_patch(
            Rectangle(
                (col_index - 0.5, row_index - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#1f1f1f",
                linewidth=1.45,
            )
        )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.035)
    colorbar.outline.set_edgecolor(SPINE)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Symmetric overlap", fontsize=10)

    fig.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.23)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "best_layer_das_overlap.png"
    pdf_path = OUTPUT_DIR / "best_layer_das_overlap.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path


def main():
    entries, matrix = make_matrix()
    for output_path in plot_matrix(entries, matrix):
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
