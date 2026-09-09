import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import math


def plot_matrix(rows, output_values, output_path):
    if output_path is None:
        return

    matrix_rows = sorted(
        [row for row in rows if row["row_type"] == "matrix"],
        key=lambda row: row["steering_target"],
    )
    if not matrix_rows:
        raise ValueError("No matrix rows found to plot.")

    matrix = [
        [row["avg_steered_probs"][str(value)] for value in output_values]
        for row in matrix_rows
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Output token")
    ax.set_ylabel("Steering target")
    ax.set_xticks(range(len(output_values)))
    ax.set_xticklabels(output_values)
    ax.set_yticks(range(len(matrix_rows)))
    ax.set_yticklabels([row["steering_target"] for row in matrix_rows])
    metric = matrix_rows[0].get("probability_metric") or ""
    colorbar_label = (
        "average normalized candidate probability"
        if "candidate_sequence_probability" in metric
        else "average next-token probability"
    )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_counterfactual_summary(rows, output_path):
    if output_path is None:
        return

    example_rows = [row for row in rows if row["row_type"] == "example"]
    if not example_rows or "counterfactual_preferred" not in example_rows[0]:
        return

    steering_targets = sorted({row["steering_target"] for row in example_rows})
    preference_rates = []
    mean_logprob_gains = []

    for steering_target in steering_targets:
        target_rows = [
            row for row in example_rows
            if row["steering_target"] == steering_target
        ]
        preference_rates.append(
            sum(bool(row["counterfactual_preferred"]) for row in target_rows)
            / len(target_rows)
        )
        gains = [
            row["counterfactual_output_logprob"] - row["original_output_logprob"]
            for row in target_rows
            if (
                math.isfinite(row["counterfactual_output_logprob"])
                and math.isfinite(row["original_output_logprob"])
            )
        ]
        mean_logprob_gains.append(
            sum(gains) / len(gains) if gains else float("nan")
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(steering_targets, preference_rates, color="#2a9d8f")
    axes[0].set_ylabel("Counterfactual preferred")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(steering_targets, mean_logprob_gains, color="#e76f51")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Steering target")
    axes[1].set_ylabel("Mean logprob gain")
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Counterfactual final-answer effect")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_add_sub_gains(summary_rows, output_path):
    alphas = sorted({row["alpha"] for row in summary_rows})
    target_gain = []
    contrast_gain = []
    for alpha in alphas:
        rows = [row for row in summary_rows if row["alpha"] == alpha]
        target_gain.append(
            sum(row["mean_target_logprob_gain"] for row in rows) / len(rows)
        )
        contrast_gain.append(
            sum(row["mean_target_vs_base_contrast_gain"] for row in rows)
            / len(rows)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, target_gain, marker="o", label="Target logprob gain")
    ax.plot(alphas, contrast_gain, marker="o", label="Target-vs-base contrast gain")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Alpha", ylabel="Mean logprob gain")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gain_matrices(
    summary_rows,
    output_values,
    output_dir,
    stem,
    row_label,
    column_label,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for alpha in sorted({row["alpha"] for row in summary_rows}):
        rows = sorted(
            [row for row in summary_rows if row["alpha"] == alpha],
            key=lambda row: row["steering_target"],
        )
        matrix = np.array([
            [row["mean_candidate_logprob_gains"][str(value)] for value in output_values]
            for row in rows
        ])
        scale = max(float(np.abs(matrix).max()), 1e-8)

        fig, ax = plt.subplots(
            figsize=(max(7, len(output_values) * 0.25), max(5, len(rows) * 0.25))
        )
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
        )
        ax.set_title(f"Candidate logprob gain (alpha={alpha})")
        ax.set_xlabel(column_label)
        ax.set_ylabel(row_label)
        ax.set_xticks(range(len(output_values)), output_values, rotation=90)
        ax.set_yticks(
            range(len(rows)), [row["steering_target"] for row in rows]
        )
        fig.colorbar(image, ax=ax, label="Mean logprob gain")
        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}_alpha{alpha}_matrix.png", dpi=200)
        plt.close(fig)


def plot_mul_alpha(summary_rows, output_path):
    rows = sorted(summary_rows, key=lambda row: row["alpha"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [row["alpha"] for row in rows],
        [row["mean_source_mod100_contrast"] for row in rows],
        marker="o",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel="Alpha",
        ylabel="log P(source suffix) - log P(base suffix)",
        title="Multiplication source mod-100 contrast",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_c1_hat_alpha(summary_rows, output_path):
    rows = sorted(summary_rows, key=lambda row: row["alpha"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [row["alpha"] for row in rows],
        [row["mean_source_hundreds_contrast"] for row in rows],
        marker="o",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel="Alpha",
        ylabel="log P(source hundreds) - log P(base hundreds)",
        title="c1_hat steering: carry-sensitive hundreds digit",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
