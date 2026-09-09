"""Focused plots for the simplified DAS JSONL schema."""

from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {16: "#2563eb", 32: "#dc2626"}
LAYER_SWEEP_DIMS = (16, 32)


def _metric_plot(rows, metric, ylabel, output_path, dimensions=LAYER_SWEEP_DIMS):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for dimension in dimensions:
        selected = sorted(
            (row for row in rows if int(row["k"]) == dimension),
            key=lambda row: int(row["layer"]),
        )
        if not selected:
            continue
        color = COLORS.get(dimension)
        layers = [int(row["layer"]) for row in selected]
        ax.plot(
            layers,
            [float(row[metric]) for row in selected],
            color=color,
            marker="o",
            linewidth=2.2,
            label=f"learned k={dimension}",
        )
        for prefix, style, alpha, label in (
            ("random_", "--", 0.55, "random"),
            ("shuffled_donor_", ":", 0.75, "shuffled donor"),
        ):
            control = prefix + metric
            if all(control in row for row in selected):
                ax.plot(
                    layers,
                    [float(row[control]) for row in selected],
                    color=color,
                    linestyle=style,
                    linewidth=1.4,
                    alpha=alpha,
                    label=f"{label} k={dimension}",
                )

    ax.set_xlabel("Transformer block (1-based)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(sorted({int(row["layer"]) for row in rows}))
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_das_results(rows, output_stem):
    if not rows:
        return
    output_stem = Path(output_stem)
    _metric_plot(
        rows,
        "variable_teacher_forced_iia",
        "Variable IIA (teacher-forced)",
        output_stem.with_name(output_stem.name + "_variable_iia.png"),
    )
    _metric_plot(
        rows,
        "full_answer_teacher_forced_iia",
        "Full-answer IIA (teacher-forced)",
        output_stem.with_name(output_stem.name + "_full_answer_iia.png"),
    )
    _metric_plot(
        rows,
        "autoregressive_iia",
        "Autoregressive IIA",
        output_stem.with_name(output_stem.name + "_autoregressive_iia.png"),
    )
