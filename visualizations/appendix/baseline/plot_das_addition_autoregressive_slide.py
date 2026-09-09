"""PowerPoint-friendly single-panel DAS autoregressive IIA plot for addition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "results"
    / "baseline"
    / "gemma4_12b_it"
    / "digits"
    / "das_v2"
    / "runs"
    / "addition_result_pos17_resid_post_initrandom_pca_seed8.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "visualizations" / "appendix" / "baseline" / "das"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
SPINE = "#202426"
TEXT = "#171717"
GRID = "#d8ddde"
K_COLORS = {
    8: "#b85f5d",
    16: "#d08b57",
    32: "#5f9b92",
    64: "#6f7fae",
    128: "#8a6aa3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="das_addition_result_autoregressive_iia_ppt")
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    return parser.parse_args()


def long_path(path: Path) -> str:
    value = str(path)
    if os.name == "nt" and path.is_absolute() and not value.startswith("\\\\?\\"):
        return "\\\\?\\" + value
    return value


def load_jsonl(path: Path) -> list[dict]:
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Palatino Linotype", "Georgia", "Cambria", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10.6,
            "axes.titlesize": 14.0,
            "axes.labelsize": 12.2,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.4,
            "legend.fontsize": 10.0,
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


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    best = {}
    for row in rows:
        if "layer" not in row or "k" not in row or "autoregressive_iia" not in row:
            continue
        key = (int(row["layer"]), int(row["k"]))
        current = best.get(key)
        if current is None or float(row["autoregressive_iia"]) >= float(current["autoregressive_iia"]):
            best[key] = row
    return list(best.values())


def setup_axis(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.78, alpha=0.78)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.42)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(1.0)
    ax.tick_params(width=0.9, length=3.4)


def main() -> None:
    args = parse_args()
    style_matplotlib()

    rows = deduplicate_rows(load_jsonl(args.input))
    layers = sorted({int(row["layer"]) for row in rows})
    lookup = {(int(row["layer"]), int(row["k"])): row for row in rows}

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.05), facecolor=BACKGROUND)
    fig.patch.set_facecolor(BACKGROUND)
    setup_axis(ax)

    summary = {}
    for k in args.ks:
        available_layers = [layer for layer in layers if (layer, k) in lookup]
        if not available_layers:
            continue
        values = [float(lookup[(layer, k)]["autoregressive_iia"]) for layer in available_layers]
        ax.plot(
            available_layers,
            values,
            color=K_COLORS.get(k, "#5d6264"),
            linewidth=1.95,
            marker="o",
            markersize=4.9,
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.65,
            label=fr"$k={k}$",
            zorder=3,
        )
        summary[str(k)] = {
            "layers": available_layers,
            "autoregressive_iia": values,
            "best_layer": int(available_layers[int(np.argmax(values))]),
            "max_iia": float(np.max(values)),
        }

    ax.set_title("Addition DAS: autoregressive IIA", pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Autoregressive IIA")
    ax.set_xlim(min(layers) - 0.35, max(layers) + 0.35)
    ax.set_ylim(-0.025, 1.025)
    ax.set_xticks(layers)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        framealpha=0.94,
        ncol=1,
        borderpad=0.42,
        labelspacing=0.28,
        handlelength=1.55,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.17)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    pdf_path = args.output_dir / f"{args.stem}.pdf"
    summary_path = args.output_dir / f"{args.stem}_summary.json"
    fig.savefig(png_path, dpi=350, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
