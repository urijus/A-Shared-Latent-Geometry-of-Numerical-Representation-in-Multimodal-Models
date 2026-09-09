"""Plot the four-task DAS dimension-choice audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "final_exps"
DEFAULT_BASELINE_TEXT_RUN_DIR = REPO_ROOT / "results" / "baseline" / "gemma4_12b_it" / "digits" / "das_v2" / "runs_bons"
DEFAULT_BASELINE_IMAGE_RUN_DIR = REPO_ROOT / "results" / "baseline_images" / "gemma4_12b_it" / "digits" / "das_v2" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "das_dimension_choice"
DEFAULT_STEM = "das_dimension_choice_autoregressive_iia"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
MUTED = "#5e696d"
SPINE = "#202426"
GRID = "#d8ddde"
TEAL = "#226a74"
RUST = "#a95642"
PURPLE = "#7a5b98"
GREEN = "#5d8f7b"

TASKS = [
    ("text", "addition", "T+", TEAL, "o"),
    ("text", "subtraction", "T-", RUST, "^"),
    ("image", "addition", "I+", PURPLE, "s"),
    ("image", "subtraction", "I-", GREEN, "D"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--baseline-text-run-dir", type=Path, default=DEFAULT_BASELINE_TEXT_RUN_DIR)
    parser.add_argument("--baseline-image-run-dir", type=Path, default=DEFAULT_BASELINE_IMAGE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 22, 32])
    parser.add_argument("--appendix-k32-layer", type=int, default=44)
    return parser.parse_args()


def style_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino Linotype", "Palatino", "Georgia", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "figure.dpi": 120,
            "savefig.dpi": 350,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.0,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.9,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_row(
    raw_rows: list[dict],
    values: dict[tuple[int, str, str], list[float]],
    *,
    k: int,
    modality: str,
    operation: str,
    label: str,
    seed: int,
    value: float,
    source: str,
    path: Path,
) -> None:
    raw_rows.append(
        {
            "k": k,
            "modality": modality,
            "operation": operation,
            "task": label,
            "seed": seed,
            "autoregressive_iia": value,
            "source": source,
            "path": str(path),
        }
    )
    values[(k, modality, operation)].append(value)


def appendix_k32_paths(baseline_text_run_dir: Path, baseline_image_run_dir: Path) -> dict[tuple[str, str], Path]:
    return {
        ("text", "addition"): baseline_text_run_dir / "addition_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        ("text", "subtraction"): baseline_text_run_dir / "subtraction_result_pos17_resid_post_initrandom_pca_seed8.jsonl",
        ("image", "addition"): baseline_image_run_dir / "addition_result_pos-1_resid_post_initrandom_pca_seed8.jsonl",
        ("image", "subtraction"): baseline_image_run_dir / "subtraction_result_pos-1_resid_post_initrandom_pca_seed8.jsonl",
    }


def collect_rows(
    result_root: Path,
    ks: list[int],
    baseline_text_run_dir: Path,
    baseline_image_run_dir: Path,
    appendix_k32_layer: int,
) -> tuple[list[dict], list[dict]]:
    raw_rows = []
    summary_rows = []
    values: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    audit_keys: set[tuple[int, str, str]] = set()
    for k in ks:
        audit_root = result_root / f"DAS_audit_k_{k}"
        if not audit_root.exists():
            continue
        for modality, operation, label, _, _ in TASKS:
            pattern = f"{modality}/{operation}/das_pca_initialized/split_0/seed_*/results.jsonl"
            for path in sorted(audit_root.glob(pattern)):
                for row in load_jsonl(path):
                    if row.get("condition") != "das_pca_initialized" or row.get("target") != "result":
                        continue
                    value = float(row["autoregressive_iia"])
                    seed = int(row.get("seed", path.parent.name.replace("seed_", "")))
                    audit_keys.add((k, modality, operation))
                    add_row(
                        raw_rows,
                        values,
                        k=k,
                        modality=modality,
                        operation=operation,
                        label=label,
                        seed=seed,
                        value=value,
                        source="DAS_audit",
                        path=path,
                    )

    sweep_roots = [
        result_root / "why_image_worse" / "k_sweep" / "layer43" / "text_pos17",
        result_root / "why_image_worse" / "k_sweep" / "layer43" / "image_pos-1",
    ]
    task_labels = {(modality, operation): label for modality, operation, label, _, _ in TASKS}
    for k in ks:
        for sweep_root in sweep_roots:
            root = sweep_root / f"k{k}"
            if not root.exists():
                continue
            for path in sorted(root.glob("*/*/das_pca_initialized/split_0/seed_*/results.jsonl")):
                rel = path.relative_to(root).parts
                modality, operation = rel[0], rel[1]
                label = task_labels.get((modality, operation))
                if label is None or (k, modality, operation) in audit_keys:
                    continue
                for row in load_jsonl(path):
                    if row.get("condition") != "das_pca_initialized" or row.get("target") != "result":
                        continue
                    if int(row.get("layer", -1)) != 43 or int(row.get("k", -1)) != int(k):
                        continue
                    seed = int(row.get("seed", path.parent.name.replace("seed_", "")))
                    add_row(
                        raw_rows,
                        values,
                        k=k,
                        modality=modality,
                        operation=operation,
                        label=label,
                        seed=seed,
                        value=float(row["autoregressive_iia"]),
                        source="layer43_k_sweep",
                        path=path,
                    )

    if 32 in {int(k) for k in ks}:
        for modality, operation, label, _, _ in TASKS:
            if values.get((32, modality, operation)):
                continue
            path = appendix_k32_paths(baseline_text_run_dir, baseline_image_run_dir)[(modality, operation)]
            if not path.exists():
                continue
            for row in load_jsonl(path):
                if row.get("target") != "result" or int(row.get("k", -1)) != 32:
                    continue
                if int(row.get("layer", -1)) != int(appendix_k32_layer):
                    continue
                if str(row.get("hook")) != "resid_post":
                    continue
                seed = int(row.get("seed", 8))
                add_row(
                    raw_rows,
                    values,
                    k=32,
                    modality=modality,
                    operation=operation,
                    label=label,
                    seed=seed,
                    value=float(row["autoregressive_iia"]),
                    source=f"appendix_das_sweep_layer{appendix_k32_layer}",
                    path=path,
                )

    for k in ks:
        for modality, operation, label, _, _ in TASKS:
            vals = np.array(values.get((k, modality, operation), []), dtype=float)
            if vals.size == 0:
                continue
            summary_rows.append(
                {
                    "k": k,
                    "modality": modality,
                    "operation": operation,
                    "task": label,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)),
                    "n": int(vals.size),
                    "sources": sorted({row["source"] for row in raw_rows if row["k"] == k and row["modality"] == modality and row["operation"] == operation}),
                }
            )
    return raw_rows, summary_rows


def setup_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.grid(axis="y", color=GRID, linewidth=0.60, alpha=0.72)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.30)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.8, pad=2)


def draw_dimension_choice(ax: plt.Axes, raw_rows: list[dict], summary_rows: list[dict], ks: list[int]) -> None:
    setup_axis(ax)
    by_summary = {(row["task"], row["k"]): row for row in summary_rows}

    for _, _, label, color, marker in TASKS:
        xs = []
        ys = []
        yerr = []
        for k in ks:
            row = by_summary.get((label, k))
            if row is None:
                continue
            xs.append(k)
            ys.append(row["mean"])
            yerr.append(row["std"])
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=4.7,
            linewidth=1.20,
            elinewidth=0.80,
            capsize=2.6,
            capthick=0.8,
            zorder=4,
        )

    ax.axvline(22, color=SPINE, linewidth=0.85, linestyle=(0, (3, 2.4)), alpha=0.62, zorder=1)
    ax.text(22.25, 0.065, "used in main\nanalyses", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xlabel(r"DAS dimension $k$")
    ax.set_ylabel("Autoregressive IIA")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlim(min(ks) - 1.7, max(ks) + 2.4)
    ax.set_ylim(0.0, 1.05)
    ax.text(0.5, 1.035, "DAS dimension choice", transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5)
    handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=1.20, markersize=4.8, label=label)
        for _, _, label, color, marker in TASKS
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.988, 0.990),
        ncol=4,
        frameon=False,
        columnspacing=0.95,
        handletextpad=0.28,
        borderpad=0.1,
    )


def save_outputs(fig: plt.Figure, output_dir: Path, stem: str, summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    summary_path = output_dir / f"{stem}_summary.json"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.035)
    fig.savefig(pdf_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.035)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {summary_path}")


def main() -> None:
    args = parse_args()
    style_matplotlib()
    raw_rows, summary_rows = collect_rows(
        args.result_root,
        args.ks,
        args.baseline_text_run_dir,
        args.baseline_image_run_dir,
        args.appendix_k32_layer,
    )
    available = sorted({int(row["k"]) for row in summary_rows})
    if not summary_rows:
        raise RuntimeError(f"No DAS audit rows found for ks={args.ks} under {args.result_root}.")
    missing = [
        {"k": k, "task": label}
        for k in args.ks
        for _, _, label, _, _ in TASKS
        if not any(int(row["k"]) == int(k) and row["task"] == label for row in summary_rows)
    ]
    print("DAS dimension-choice source roots:")
    for k in available:
        audit_root = args.result_root / f"DAS_audit_k_{k}"
        if audit_root.exists():
            print(f"  {audit_root}")
    sweep_root = args.result_root / "why_image_worse" / "k_sweep" / "layer43"
    if any("layer43_k_sweep" in row.get("sources", []) for row in summary_rows):
        print(f"  {sweep_root}")
    if any("appendix_das_sweep" in source for row in summary_rows for source in row.get("sources", [])):
        print(f"  {args.baseline_text_run_dir}")
        print(f"  {args.baseline_image_run_dir}")
    if missing:
        print(f"Missing task/k combinations: {missing}")

    fig, ax = plt.subplots(1, 1, figsize=(4.25, 3.05), constrained_layout=False)
    fig.patch.set_facecolor(BACKGROUND)
    draw_dimension_choice(ax, raw_rows, summary_rows, available)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.820, bottom=0.170)
    save_outputs(
        fig,
        args.output_dir,
        args.stem,
        {
            "source_result_root": str(args.result_root),
            "baseline_text_run_dir": str(args.baseline_text_run_dir),
            "baseline_image_run_dir": str(args.baseline_image_run_dir),
            "appendix_k32_layer": args.appendix_k32_layer,
            "requested_ks": args.ks,
            "available_ks": available,
            "note": "DAS audit rows are used where available; layer-43 k-sweep rows fill additional comparable k values where present; remaining k=32 task values are taken from the older appendix DAS sweep at the requested layer.",
            "raw_rows": raw_rows,
            "summary_rows": summary_rows,
            "missing": missing,
        },
    )


if __name__ == "__main__":
    main()
