"""Main-paper figure for repeat activations projected into arithmetic exact L.

This script reads the saved repeat centroid-noise control and full-centroid
repeat-vs-arithmetic geometry outputs, validates that the representation is the
13D readout-orthogonal arithmetic exact-L component, and produces the
three-panel thesis figure for Section VIII-B.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


NOISE_DIR = Path("results/experiments/closing/repeat_centroid_noise_control")
FULL_DIR = Path("results/experiments/closing/repeat_vs_causal_L_geometry")
DEFAULT_FIGURE_DIR = FULL_DIR / "figures"
FIGURE_BASENAME = "repeat_vs_causal_L_geometry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise_dir", type=Path, default=NOISE_DIR)
    parser.add_argument("--full_dir", type=Path, default=FULL_DIR)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_header_and_rows(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        n_rows = sum(1 for _ in reader)
    return header, n_rows


def inspect_pt(path: Path) -> list[str]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return [f"type={type(obj).__name__}, shape={getattr(obj, 'shape', None)}"]
    lines = [f"keys={sorted(obj.keys())}"]
    for key, value in obj.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            detail = f"shape={tuple(shape)}"
        elif hasattr(value, "__len__") and not isinstance(value, str):
            detail = f"len={len(value)}"
        else:
            detail = f"value={value}"
        lines.append(f"{key}: {type(value).__name__}, {detail}")
    return lines


def discovered_files_summary(result_dir: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(result_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(result_dir)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            header, n_rows = csv_header_and_rows(path)
            lines.append(f"{rel} | csv rows={n_rows}, cols={len(header)}")
        elif suffix == ".json":
            data = read_json(path)
            keys = sorted(data.keys()) if isinstance(data, dict) else []
            lines.append(f"{rel} | json keys={keys[:10]}")
        elif suffix == ".pt":
            details = "; ".join(inspect_pt(path))
            lines.append(f"{rel} | pt {details}")
        else:
            lines.append(f"{rel} | bytes={path.stat().st_size}")
    return lines


def load_csv_frame(path: Path):
    import pandas as pd

    return pd.read_csv(path)


def per_replicate_distribution(frame, family: str, metric: str):
    values = (
        frame.loc[frame["comparison_family"].eq(family)]
        .groupby("replicate", sort=True)[metric]
        .mean()
        .sort_index()
    )
    return values


def series_stats(values) -> dict[str, float]:
    return {
        "n": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def full_centroid_metrics(full_dir: Path, noise_summary: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    final_comparison_path = full_dir / "repeat_vs_arithmetic_L_final_comparison.csv"
    final = load_csv_frame(final_comparison_path)
    from_second: dict[str, dict[str, float]] = {}
    for _, row in final.iterrows():
        item = str(row["comparison_item"])
        if item in {"arithmetic<->arithmetic L", "repeat<->arithmetic L"}:
            from_second[item] = {
                "transition": float(row["transition"]),
                "top1": float(row["top1"]),
                "RSA": float(row["RSA"]),
            }

    ref = noise_summary.get("reference_means", {})
    from_noise_reference = {
        "arithmetic<->arithmetic L": {
            "transition": float(ref["arithmetic_arithmetic_transition"]),
            "top1": float(ref["arithmetic_arithmetic_top1"]),
            "RSA": float(ref["arithmetic_arithmetic_RSA"]),
        },
        "repeat<->arithmetic L": {
            "transition": float(ref["repeat_arithmetic_transition"]),
            "top1": float(ref["repeat_arithmetic_top1"]),
            "RSA": float(ref["repeat_arithmetic_RSA"]),
        },
    }
    return from_second, from_noise_reference


def validate_and_load(args: argparse.Namespace) -> dict[str, Any]:
    noise_summary = read_json(args.noise_dir / "summary.json")
    full_summary = read_json(args.full_dir / "repeat_vs_arithmetic_L_summary.json")

    noise_diag = load_csv_frame(args.noise_dir / "L_space_diagnostics.csv")
    full_diag = load_csv_frame(args.full_dir / "repeat_L_space_diagnostics.csv")

    latent_dims = set(noise_diag["latent_dim"].dropna().astype(int)) | set(full_diag["latent_dim"].dropna().astype(int))
    space_types = set(noise_diag["space_type"].dropna().astype(str)) | set(full_diag["space_type"].dropna().astype(str))
    causal_latent_dims = set(noise_diag.loc[noise_diag["space_type"].eq("causal_L"), "latent_dim"].dropna().astype(int))
    causal_latent_dims |= set(full_diag.loc[full_diag["space_type"].eq("causal_L"), "latent_dim"].dropna().astype(int))
    rank_l = set(noise_diag.loc[noise_diag["space_type"].eq("causal_L"), "rank_L"].dropna().astype(int))
    rank_l |= set(full_diag.loc[full_diag["space_type"].eq("causal_L"), "rank_L"].dropna().astype(int))

    config_dims = {
        int(noise_summary["config"]["latent_dim"]),
        int(full_summary["config"]["latent_dim"]),
    }
    layers = {int(noise_summary["config"]["layer"]), int(full_summary["config"]["layer"])}
    repeat_positions = {noise_summary["config"]["repeat_position"], full_summary["config"]["repeat_position"]}

    if config_dims != {13} or causal_latent_dims != {13} or rank_l != {13}:
        raise ValueError(
            "Saved representation is not confirmed as exact 13D arithmetic L: "
            f"config_dims={config_dims}, causal_latent_dims={causal_latent_dims}, rank_L={rank_l}"
        )
    if layers != {43}:
        raise ValueError(f"Expected layer 43, found layers={layers}")
    if repeat_positions != {"equals"}:
        raise ValueError(f"Expected repeat final pre-generation position 'equals', found {repeat_positions}")
    if "random_readout_orthogonal_13d" in space_types:
        pass
    if "causal_L" not in space_types:
        raise ValueError(f"Could not find causal_L diagnostics; space_types={space_types}")

    geom = load_csv_frame(args.noise_dir / "one_example_geometry.csv")
    rsa = load_csv_frame(args.noise_dir / "one_example_rsa.csv")
    transition_ar = per_replicate_distribution(geom, "arithmetic_arithmetic", "heldout_transition_cosine")
    transition_repeat = per_replicate_distribution(geom, "repeat_arithmetic", "heldout_transition_cosine")
    rsa_ar = per_replicate_distribution(rsa, "arithmetic_arithmetic", "spearman_rsa")
    rsa_repeat = per_replicate_distribution(rsa, "repeat_arithmetic", "spearman_rsa")

    n_expected = int(noise_summary["config"]["n_replicates"])
    for name, values in {
        "transition arithmetic->arithmetic": transition_ar,
        "transition repeat->arithmetic": transition_repeat,
        "RSA arithmetic->arithmetic": rsa_ar,
        "RSA repeat->arithmetic": rsa_repeat,
    }.items():
        if values.shape[0] != n_expected:
            raise ValueError(f"{name} has {values.shape[0]} resamples, expected {n_expected}")
    if n_expected != 200:
        raise ValueError(f"Noise-control resample count is {n_expected}, expected 200")

    full_second, full_reference = full_centroid_metrics(args.full_dir, noise_summary)

    return {
        "noise_summary": noise_summary,
        "full_summary": full_summary,
        "noise_diag": noise_diag,
        "full_diag": full_diag,
        "latent_dims": latent_dims,
        "space_types": space_types,
        "transition_ar": transition_ar,
        "transition_repeat": transition_repeat,
        "rsa_ar": rsa_ar,
        "rsa_repeat": rsa_repeat,
        "full_second": full_second,
        "full_reference": full_reference,
    }


def print_validation(args: argparse.Namespace, loaded: dict[str, Any]) -> None:
    print("Validation summary")
    print("=" * 72)
    for label, result_dir in [
        ("repeat_centroid_noise_control", args.noise_dir),
        ("repeat_vs_causal_L_geometry", args.full_dir),
    ]:
        print(f"\nFiles discovered in {label}:")
        for line in discovered_files_summary(result_dir):
            print(f"  - {line}")

    noise_config = loaded["noise_summary"]["config"]
    full_config = loaded["full_summary"]["config"]
    print("\nRepresentation:")
    print(f"  exact representation dimension: {noise_config['latent_dim']} (noise control), {full_config['latent_dim']} (full-centroid)")
    print("  confirmed representation: arithmetic exact L, 13D readout-orthogonal component of k=22 DAS space")
    print(f"  readout dimensions removed: m_readout={noise_config['m_readout']}; layer={noise_config['layer']}; hook={noise_config['hook']}")
    print(f"  repeat position: {noise_config['repeat_position']} / {full_config['repeat_position']} (final pre-generation)")

    stats = {
        "transition arithmetic -> arithmetic": series_stats(loaded["transition_ar"]),
        "transition repeat -> arithmetic": series_stats(loaded["transition_repeat"]),
        "RSA arithmetic -> arithmetic": series_stats(loaded["rsa_ar"]),
        "RSA repeat -> arithmetic": series_stats(loaded["rsa_repeat"]),
    }
    n_resamples = stats["transition arithmetic -> arithmetic"]["n"]
    print(f"\nEstimator-matched resamples: {n_resamples}")
    for label, stat in stats.items():
        print(f"  {label}: mean={stat['mean']:.6f}, std={stat['std']:.6f}, min={stat['min']:.6f}, max={stat['max']:.6f}")

    print("\nFull-centroid metrics from second directory:")
    for label, vals in loaded["full_second"].items():
        print(f"  {label}: transition={vals['transition']:.6f}, top1={vals['top1']:.6f}, RSA={vals['RSA']:.6f}")

    print("\nFull-centroid reference means recorded by noise-control summary:")
    for label, vals in loaded["full_reference"].items():
        print(f"  {label}: transition={vals['transition']:.6f}, top1={vals['top1']:.6f}, RSA={vals['RSA']:.6f}")


def draw_schematic(ax) -> None:
    import matplotlib.patches as patches

    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    arith_color = "#4C78A8"
    repeat_color = "#D65F5F"
    neutral = "#30343B"
    light = "#F6F7F9"

    def box(x: float, y: float, w: float, h: float, text: str, color: str, size: float = 8.5) -> None:
        patch = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.1,
            edgecolor=color,
            facecolor=light,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=neutral)

    def arrow(x0: float, y0: float, x1: float, y1: float, color: str) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.15, shrinkA=3, shrinkB=3),
        )

    ax.text(0.02, 0.97, "(a)", ha="left", va="top", fontsize=11, fontweight="bold")
    ax.set_title("Experimental comparison", fontsize=12, pad=8)

    ax.text(0.09, 0.82, "Arithmetic route", color=arith_color, fontsize=9.5, fontweight="bold")
    ax.text(0.09, 0.48, "Repeat route", color=repeat_color, fontsize=9.5, fontweight="bold")

    box(0.08, 0.69, 0.23, 0.095, '"38 + 5 ="', arith_color)
    box(0.39, 0.69, 0.20, 0.095, r"$h_{43}^{arith}(43)$", arith_color)
    box(0.68, 0.69, 0.25, 0.095, r"$L_i^T h_{43}^{arith}$", arith_color)

    box(0.08, 0.35, 0.23, 0.095, '"Repeat 43"', repeat_color)
    box(0.39, 0.35, 0.20, 0.095, r"$h_{43}^{repeat}(43)$", repeat_color)
    box(0.68, 0.35, 0.25, 0.095, r"$L_i^T h_{43}^{repeat}$", repeat_color)

    arrow(0.31, 0.737, 0.39, 0.737, arith_color)
    arrow(0.59, 0.737, 0.68, 0.737, arith_color)
    arrow(0.31, 0.397, 0.39, 0.397, repeat_color)
    arrow(0.59, 0.397, 0.68, 0.397, repeat_color)

    ax.plot([0.805, 0.805], [0.445, 0.69], color="#8B8F98", linewidth=1.0, linestyle=":")
    ax.text(0.84, 0.565, "compare\nby value\nidentity", ha="center", va="center", fontsize=8.2, color=neutral)

    ax.text(
        0.5,
        0.21,
        r"$L_i$: arithmetic exact-L basis (13D), layer 43 final pre-generation position",
        ha="center",
        va="center",
        fontsize=8.2,
        color=neutral,
    )
    ax.text(
        0.5,
        0.11,
        "No repeat DAS and no intervention",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=neutral,
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.02", facecolor="#FFF8E8", edgecolor="#D8B35A", linewidth=0.9),
    )


def draw_violin(ax, series_a, series_b, ylabel: str, panel_label: str, title: str, ylim: tuple[float, float]) -> None:
    import numpy as np

    colors = ["#4C78A8", "#D65F5F"]
    labels = ["Arith. -> arith.", "Repeat -> arith."]
    data = [series_a.to_numpy(), series_b.to_numpy()]
    positions = [1, 2]

    violins = ax.violinplot(data, positions=positions, widths=0.72, showmeans=False, showmedians=True, showextrema=False)
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
        body.set_linewidth(0.9)
    violins["cmedians"].set_color("#202020")
    violins["cmedians"].set_linewidth(1.0)

    rng = np.random.default_rng(43)
    for x, values, color in zip(positions, data, colors):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(np.full_like(values, x, dtype=float) + jitter, values, s=6, color=color, alpha=0.22, linewidths=0)
        mean = float(np.mean(values))
        ax.scatter([x], [mean], s=46, facecolor="white", edgecolor="#111111", linewidth=1.1, zorder=5)
        ax.text(x, mean + 0.018 * (ylim[1] - ylim[0]), f"{mean:.3f}", ha="center", va="bottom", fontsize=8.2)

    ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8DCE2", linewidth=0.55, alpha=0.45)
    ax.set_axisbelow(True)
    ax.text(
        0.5,
        0.045,
        "200 estimator-matched resamples",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#545961",
    )


def plot_figure(args: argparse.Namespace, loaded: dict[str, Any]) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.figure_dir / f"{FIGURE_BASENAME}.pdf"
    png_path = args.figure_dir / f"{FIGURE_BASENAME}.png"

    fig = plt.figure(figsize=(11.8, 3.35), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.34)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    draw_schematic(axes[0])

    transition_all = np.concatenate([loaded["transition_ar"].to_numpy(), loaded["transition_repeat"].to_numpy()])
    transition_pad = 0.06 * (float(transition_all.max()) - float(transition_all.min()))
    transition_ylim = (max(0.0, float(transition_all.min()) - transition_pad - 0.015), min(1.0, float(transition_all.max()) + transition_pad + 0.015))
    draw_violin(
        axes[1],
        loaded["transition_ar"],
        loaded["transition_repeat"],
        "Transition cosine",
        "(b)",
        "Transition geometry",
        transition_ylim,
    )

    rsa_all = np.concatenate([loaded["rsa_ar"].to_numpy(), loaded["rsa_repeat"].to_numpy()])
    rsa_pad = 0.08 * (float(rsa_all.max()) - float(rsa_all.min()))
    rsa_ylim = (max(-1.0, float(rsa_all.min()) - rsa_pad - 0.02), min(1.0, float(rsa_all.max()) + rsa_pad + 0.02))
    draw_violin(
        axes[2],
        loaded["rsa_ar"],
        loaded["rsa_repeat"],
        "Spearman RSA",
        "(c)",
        "Relational geometry",
        rsa_ylim,
    )

    fig.subplots_adjust(left=0.045, right=0.992, bottom=0.17, top=0.87, wspace=0.34)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    loaded = validate_and_load(args)
    print_validation(args, loaded)
    pdf_path, png_path = plot_figure(args, loaded)
    print("\nSaved figure outputs:")
    print(f"  {pdf_path}")
    print(f"  {png_path}")


if __name__ == "__main__":
    main()
