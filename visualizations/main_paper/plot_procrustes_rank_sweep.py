"""Plot Procrustes rank-sweep causal recovery curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "final_exps" / "procrustes" / "rank_sweep"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "main_paper" / "procrustes" / "rank_sweep"

BACKGROUND = "#fdfdfd"
TEXT = "#171717"
GRID = "#d8ddde"
TRANSFER = "#226a74"
GAIN = "#8f343f"

TASK_LABELS = {
    "text_add": "Text add",
    "text_sub": "Text sub",
    "image_add": "Image add",
    "image_sub": "Image sub",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="procrustes_rank_sweep")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
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


def display(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " "))


def task_label(row: dict, prefix: str) -> str:
    return row.get(f"{prefix}_label") or display(row[f"{prefix}_task"])


def as_float(value) -> float:
    return np.nan if value is None else float(value)


def grouped(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    groups = {}
    for row in rows:
        key = (row["source_task"], row["destination_task"])
        groups.setdefault(key, []).append(row)
    for parts in groups.values():
        parts.sort(key=lambda row: int(row["rank"]))
    return groups


def draw_panel(ax, rows: list[dict]) -> None:
    x = np.array([as_float(row["rank_fraction"]) for row in rows])
    transfer = np.array([as_float(row["destination_normalized_transfer_mean"]) for row in rows])
    transfer_std = np.array([as_float(row["destination_normalized_transfer_std"]) for row in rows])
    gain = np.array(
        [as_float(row["destination_normalized_transfer_gain_over_causal_mean"]) for row in rows]
    )
    gain_std = np.array(
        [as_float(row["destination_normalized_transfer_gain_over_causal_std"]) for row in rows]
    )
    sv_fraction = np.array([as_float(row["cumulative_singular_value_fraction_mean"]) for row in rows])

    ax.set_facecolor("#f7f8f8")
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.75)
    ax.axhline(0.0, color="#9aa1a3", linewidth=0.8)
    ax.axhline(1.0, color="#9aa1a3", linewidth=0.8, linestyle=":")
    ax.plot(x, transfer, color=TRANSFER, linewidth=2.0, label="Transfer")
    ax.fill_between(x, transfer - transfer_std, transfer + transfer_std, color=TRANSFER, alpha=0.16, linewidth=0)
    ax.plot(x, gain, color=GAIN, linewidth=1.6, linestyle="--", label="Gain over causal")
    ax.fill_between(x, gain - gain_std, gain + gain_std, color=GAIN, alpha=0.12, linewidth=0)
    ax.plot(x, sv_fraction, color="#5a6265", linewidth=1.1, alpha=0.75, label="Cumulative SV")

    source = task_label(rows[0], "source")
    destination = task_label(rows[0], "destination")
    ax.set_title(f"{source} -> {destination}")
    ax.set_xlim(0.0, 1.02)
    finite = np.concatenate([transfer[np.isfinite(transfer)], gain[np.isfinite(gain)]])
    if finite.size:
        low = min(-0.15, float(finite.min()) - 0.1)
        high = max(1.1, float(finite.max()) + 0.1)
        ax.set_ylim(low, high)


def main() -> None:
    args = parse_args()
    style_matplotlib()
    rows = load_rows(args.input_dir / "rank_sweep_summary.jsonl")
    groups = grouped(rows)
    if not groups:
        raise ValueError(f"No rank-sweep rows found under {args.input_dir}.")

    n_panels = len(groups)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.15 * n_rows), squeeze=False)
    fig.patch.set_facecolor(BACKGROUND)

    for ax, (_key, parts) in zip(axes.ravel(), groups.items()):
        draw_panel(ax, parts)
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.supxlabel("Rank fraction m/k", y=0.035, fontsize=10.5)
    fig.supylabel("Score", x=0.025, fontsize=10.5)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.13, hspace=0.34, wspace=0.24)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
