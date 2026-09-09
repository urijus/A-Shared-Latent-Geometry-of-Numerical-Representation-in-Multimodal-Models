from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_best_das_overlap import (
    BACKGROUND,
    OUTPUT_DIR,
    SPINE,
    TEXT,
    load_basis,
    style_matplotlib,
    symmetric_overlap,
)


C0_FIXED = {
    "key": "mul_c0_hat_l36",
    "label": r"Fixed $\hat{c}_0$, L36",
    "run_file": "multiplication_c0_hat_pos17_resid_post_initrandom_pca_seed8.jsonl",
    "target": "c0_hat",
    "layer": 36,
}

COMPARISON_LAYERS = [36, 38, 40, 42, 44, 46]
C1_RUN_FILE = "multiplication_c1_hat_pos17_resid_post_initrandom_pca_seed8.jsonl"
RESULT_RUN_FILE = "multiplication_result_pos17_resid_post_initrandom_pca_seed8.jsonl"
C0_RUN_FILE = "multiplication_c0_hat_pos17_resid_post_initrandom_pca_seed8.jsonl"


def c1_spec(layer):
    return {
        "key": f"mul_c1_hat_l{layer}",
        "label": rf"$\hat{{c}}_1$, L{layer}",
        "run_file": C1_RUN_FILE,
        "target": "c1_hat",
        "layer": layer,
    }


def result_spec(layer):
    return {
        "key": f"mul_result_l{layer}",
        "label": f"result, L{layer}",
        "run_file": RESULT_RUN_FILE,
        "target": "result",
        "layer": layer,
    }


def c0_spec(layer):
    return {
        "key": f"mul_c0_hat_l{layer}",
        "label": rf"$\hat{{c}}_0$, L{layer}",
        "run_file": C0_RUN_FILE,
        "target": "c0_hat",
        "layer": layer,
    }


def compute_overlaps():
    fixed_basis, fixed_row, fixed_checkpoint = load_basis(C0_FIXED)
    series = {
        r"$\hat{c}_0$": {"spec_fn": c0_spec, "records": []},
        r"$\hat{c}_1$": {"spec_fn": c1_spec, "records": []},
        "result": {"spec_fn": result_spec, "records": []},
    }
    for comparison in series.values():
        for layer in COMPARISON_LAYERS:
            basis, row, checkpoint = load_basis(comparison["spec_fn"](layer))
            comparison["records"].append(
                {
                    "layer": layer,
                    "overlap": symmetric_overlap(fixed_basis, basis),
                    "row": row,
                    "checkpoint": checkpoint,
                }
            )
    return fixed_row, fixed_checkpoint, series


def plot_overlaps(series):
    style_matplotlib()
    all_overlaps = [
        record["overlap"]
        for comparison in series.values()
        for record in comparison["records"]
    ]

    fig, ax = plt.subplots(figsize=(5.75, 3.35))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)

    ax.axvspan(35.55, 36.45, color="#c5ced1", alpha=0.55, linewidth=0)
    ax.axhline(0, color="#9fa8aa", linewidth=0.8, zorder=0)
    ax.grid(axis="y", color="#c7d0d2", linewidth=0.6, alpha=0.55)
    ax.grid(axis="x", visible=False)

    styles = {
        r"$\hat{c}_0$": {"line": "#2f7f75", "marker": "#96c8af"},
        r"$\hat{c}_1$": {"line": "#496f82", "marker": "#f0c36a"},
        "result": {"line": "#8a5f68", "marker": "#d9b0a3"},
    }
    for label, comparison in series.items():
        layers = np.array([record["layer"] for record in comparison["records"]], dtype=int)
        overlaps = np.array([record["overlap"] for record in comparison["records"]], dtype=float)
        ax.plot(
            layers,
            overlaps,
            color=styles[label]["line"],
            linewidth=2.25,
            label=label,
            zorder=2,
        )
        ax.scatter(
            layers,
            overlaps,
            s=52,
            color=styles[label]["marker"],
            edgecolor=styles[label]["line"],
            linewidth=1.05,
            zorder=3,
        )

    y_max = max(0.12, float(max(all_overlaps)) * 1.28)
    ax.set_ylim(0, y_max)
    ax.set_xlim(35.4, 46.6)
    ax.set_xticks(COMPARISON_LAYERS)
    ax.tick_params(axis="both", length=0, pad=5)
    ax.set_xlabel("Comparison layer", labelpad=7, fontsize=11.5)
    ax.set_ylabel("Symmetric overlap", labelpad=8, fontsize=11.5)
    ax.set_title(r"Fixed $\hat{c}_0$ L36 vs multiplication subspaces", pad=11, fontsize=13.5)
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=SPINE,
        framealpha=0.9,
        fontsize=9.5,
        handlelength=1.7,
    )
    legend.get_frame().set_linewidth(0.7)

    ax.text(
        36,
        y_max * 0.965,
        r"$\hat{c}_0$ fixed",
        ha="center",
        va="top",
        fontsize=9,
        color=TEXT,
    )

    fig.subplots_adjust(left=0.16, right=0.98, top=0.84, bottom=0.20)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "c0_l36_vs_mul_subspaces_das_overlap.png"
    pdf_path = OUTPUT_DIR / "c0_l36_vs_mul_subspaces_das_overlap.pdf"
    legacy_png_path = OUTPUT_DIR / "c0_l36_vs_c1_layers_das_overlap.png"
    legacy_pdf_path = OUTPUT_DIR / "c0_l36_vs_c1_layers_das_overlap.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    fig.savefig(legacy_png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(legacy_pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path, pdf_path, legacy_png_path, legacy_pdf_path


def main():
    _, _, series = compute_overlaps()
    print("c0_hat L36 vs multiplication subspaces")
    for label, comparison in series.items():
        print(label)
        for record in comparison["records"]:
            print(f"  layer {record['layer']}: {record['overlap']:.4f}")
    for output_path in plot_overlaps(series):
        print(f"Saved {Path(output_path)}")


if __name__ == "__main__":
    main()
