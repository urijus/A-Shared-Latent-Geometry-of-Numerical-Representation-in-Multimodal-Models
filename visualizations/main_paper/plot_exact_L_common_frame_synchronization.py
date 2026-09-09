from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.global_geometry.synchronization import synchronization_L as sync_l


DEFAULT_INPUT_DIR = Path("results/paper/synchronization/readout_orthogonal_L")
DEFAULT_FACTORIZATION_DIR = Path("results/paper/procrustes/factorized_paths/readout_orthogonal_L")
DEFAULT_OUTPUT_DIR = Path("visualizations/main_paper/exact_L_common_frame")
DEFAULT_STEM = "exact_L_common_frame_synchronization"

BACKGROUND = "#fdfdfd"
PANEL_BG = "#f7f8f8"
TEXT = "#171717"
MUTED = "#5e696d"
SECONDARY_TEXT = "#4f5a5e"
SPINE = "#202426"
GRID = "#d8ddde"
TEAL = "#226a74"
RUST = "#a95642"
PURPLE = "#7a5b98"
GREEN = "#5d8f7b"

TASK_ORDER = ["text:addition", "text:subtraction", "image:addition", "image:subtraction"]
TASK_LABELS = {
    "text:addition": "T+",
    "text:subtraction": "T-",
    "image:addition": "I+",
    "image:subtraction": "I-",
}
TASK_MARKERS = {
    "text:addition": "o",
    "text:subtraction": "^",
    "image:addition": "s",
    "image:subtraction": "D",
}
CONTROL_LABELS = {
    "pairwise_direct": "Direct",
    "synchronized_full": "Full sync",
    "synchronized_loro": "LORO",
}
CONTROL_COLORS = {
    "pairwise_direct": TEAL,
    "synchronized_full": GREEN,
    "synchronized_loro": PURPLE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--factorization-dir", type=Path, default=DEFAULT_FACTORIZATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--panel-a-seed", type=int, default=0)
    parser.add_argument("--panel-a-value-split", type=int, default=0)
    parser.add_argument("--activation-dir-text", type=Path, default=None)
    parser.add_argument("--activation-dir-image", type=Path, default=None)
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
            "axes.labelsize": 8.1,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.4,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.9,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_panel(ax: plt.Axes, heading: str, panel: str, *, grid: bool = False) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.9)
    ax.text(0.018, 0.985, panel, transform=ax.transAxes, ha="left", va="top", fontsize=8.9)
    ax.text(0.5, 1.030, heading, transform=ax.transAxes, ha="center", va="bottom", fontsize=9.5)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, linewidth=0.60, alpha=0.72)


def pathify_config(config: dict, input_dir: Path, factorization_dir: Path) -> SimpleNamespace:
    converted = dict(config)
    path_keys = {
        "audit_root",
        "previous_geometry_dir",
        "digit_readout_basis_path",
        "activation_dir_text",
        "activation_dir_image",
        "output_dir",
        "figure_dir",
        "old_das_summary",
    }
    for key in path_keys:
        if converted.get(key) is not None:
            converted[key] = Path(converted[key])
    converted["output_dir"] = input_dir
    converted.setdefault("force", False)
    args = SimpleNamespace(**converted)
    args.factorization_dir = factorization_dir
    return args


def has_activation_layer(root: Path, operation: str, layer: int) -> bool:
    path = root / f"{operation}_baseline.pt"
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return int(layer) in [int(value) for value in payload.get("block_layers", [])]


def apply_activation_overrides(plot_args: argparse.Namespace, exp_args: SimpleNamespace) -> None:
    if plot_args.activation_dir_text is not None:
        exp_args.activation_dir_text = plot_args.activation_dir_text
    if plot_args.activation_dir_image is not None:
        exp_args.activation_dir_image = plot_args.activation_dir_image

    if not sync_l.geom.activation_path_for(exp_args, "text", "addition").exists():
        text_root = Path(".local/banks/baseline_bank") / exp_args.model / "digits"
        for fallback in [text_root / "activations", text_root / "activations_odd_layers", text_root / "activations_fourier_layers_32_48"]:
            if has_activation_layer(fallback, "addition", exp_args.layer):
                exp_args.activation_dir_text = fallback
                break
    if not sync_l.geom.activation_path_for(exp_args, "image", "addition").exists():
        image_root = Path(".local/banks/baseline_images_bank") / exp_args.model / "digits"
        for fallback in [image_root / "activations", image_root / "activations_odd_layers", image_root / "activations_fourier_layers_32_48"]:
            if has_activation_layer(fallback, "addition", exp_args.layer):
                exp_args.activation_dir_image = fallback
                break


def assert_exact_l_summary(summary: dict, source: Path) -> None:
    if summary.get("space_type") != "readout_orthogonal_L":
        raise RuntimeError(f"{source} is not readout_orthogonal_L: {summary.get('space_type')!r}")
    if int(summary.get("L_dimension", -1)) != 13:
        raise RuntimeError(f"{source} does not report exact L dimension 13.")
    config = summary.get("config", {})
    if int(config.get("latent_dim", -1)) != 13 or int(config.get("k", -1)) != 22 or int(config.get("m_readout", -1)) != 9:
        raise RuntimeError(f"{source} config is not the expected exact C/L split: {config}")
    marker = " ".join(str(value).lower() for value in [summary.get("experiment"), summary.get("space_type"), summary.get("description")])
    if any(term in marker for term in ["full_das", "22d", "shared_frame"]):
        raise RuntimeError(f"{source} looks like an old/full-DAS output, refusing to plot.")


def selected_sync_payload(input_dir: Path, seed: int, value_split: int) -> tuple[Path, dict]:
    preferred = input_dir / "maps" / f"L_synchronized_hub_seed{seed}_valuesplit{value_split}_synchronized_full_full_graph_layer43_k13.pt"
    candidates = sorted((input_dir / "maps").glob("L_synchronized_hub_seed*_valuesplit*_synchronized_full_full_graph_layer43_k13.pt"))
    path = preferred if preferred.exists() else (candidates[0] if candidates else None)
    if path is None:
        raise FileNotFoundError(f"No exact-L synchronized hub payload found under {input_dir / 'maps'}.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "L_orthogonal_synchronization_hub" or payload.get("space_type") != "readout_orthogonal_L":
        raise RuntimeError(f"{path} is not an exact-L synchronized hub payload.")
    if int(payload.get("config", {}).get("latent_dim", -1)) != 13:
        raise RuntimeError(f"{path} was not saved with latent_dim=13.")
    return path, payload


def synchronized_centroids(
    spaces: dict[tuple[str, int], sync_l.LSpace],
    payload: dict,
    train_values: list[int],
    test_values: list[int],
    seed: int,
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    orientations = payload["orientations_domain_to_hub"]
    log_scales = payload["log_scales"]
    train_rows = []
    test_rows = []
    validation_pairs = []

    for task in TASK_ORDER:
        item = spaces[(task, seed)]
        center = torch.stack([item.centroids[value] for value in train_values]).mean(dim=0)
        scale = float(torch.exp(torch.tensor(log_scales[task])).item())
        orientation = torch.as_tensor(orientations[task]).float()
        if tuple(orientation.shape) != (13, 13):
            raise RuntimeError(f"Orientation for {task} has shape {tuple(orientation.shape)}, expected (13, 13).")
        for value in train_values:
            common = ((item.centroids[value] - center) @ orientation) / max(scale, 1e-12)
            train_rows.append({"task": task, "label": TASK_LABELS[task], "value": value, "common": common.numpy()})
        for value in test_values:
            common = ((item.centroids[value] - center) @ orientation) / max(scale, 1e-12)
            test_rows.append({"task": task, "label": TASK_LABELS[task], "value": value, "common": common.numpy()})

    for i, source in enumerate(TASK_ORDER):
        for destination in TASK_ORDER[i + 1 :]:
            a = {row["value"]: row["common"] for row in test_rows if row["task"] == source}
            b = {row["value"]: row["common"] for row in test_rows if row["task"] == destination}
            same = []
            for value in test_values:
                av = torch.tensor(a[value])[None]
                bv = torch.tensor(b[value])[None]
                same.append(float(torch.nn.functional.cosine_similarity(av, bv)))
            validation_pairs.append(float(np.mean(same)))

    train_matrix = np.stack([row["common"] for row in train_rows], axis=0)
    test_matrix = np.stack([row["common"] for row in test_rows], axis=0)
    validation = {
        "panel_a_common_frame_same_value_cosine_mean": float(np.mean(validation_pairs)),
        "panel_a_pair_means": validation_pairs,
    }
    return test_rows, train_matrix, test_matrix, validation


def averaged_by_value(rows: list[dict], values: list[int]) -> np.ndarray:
    averaged = []
    for value in values:
        points = [row["common"] for row in rows if int(row["value"]) == int(value)]
        if len(points) != len(TASK_ORDER):
            raise RuntimeError(f"Expected {len(TASK_ORDER)} task centroids for value {value}, found {len(points)}.")
        averaged.append(np.mean(np.stack(points, axis=0), axis=0))
    return np.stack(averaged, axis=0)


def pca_from_train(train_matrix: np.ndarray, test_matrix: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    train_mean = train_matrix.mean(axis=0, keepdims=True)
    centered_train = train_matrix - train_mean
    _, _, vh = np.linalg.svd(centered_train, full_matrices=False)
    components = vh[:n_components].T
    projected_test = (test_matrix - train_mean) @ components
    total_var = float(np.var(centered_train, axis=0).sum())
    explained = np.var(centered_train @ components, axis=0) / max(total_var, 1e-12)
    return projected_test, explained


def quantile_subset(values: list[int], n_values: int = 9) -> list[int]:
    unique = sorted({int(value) for value in values})
    if len(unique) <= n_values:
        return unique
    indices = np.rint(np.linspace(0, len(unique) - 1, n_values)).astype(int)
    return [unique[int(index)] for index in sorted(set(indices))]


def load_panel_a(plot_args: argparse.Namespace, sync_summary: dict, sync_payload: dict) -> tuple[list[dict], dict]:
    exp_args = pathify_config(sync_summary["config"], plot_args.input_dir, plot_args.factorization_dir)
    exp_args.tasks = list(TASK_ORDER)
    exp_args.force = False
    apply_activation_overrides(plot_args, exp_args)

    seed = int(sync_payload["seed"])
    value_split = int(sync_payload["value_split_seed"])
    if seed != plot_args.panel_a_seed or value_split != plot_args.panel_a_value_split:
        print(f"Panel (a) using deterministic fallback payload seed={seed}, value_split={value_split}.")
    splits = sync_summary["value_splits"][str(value_split)]
    train_values = [int(v) for v in splits["train_values"]]
    test_values = [int(v) for v in splits["test_values"]]
    seed_selection = {task: [seed] for task in TASK_ORDER}

    data_by_task = sync_l.load_rows_and_data(exp_args, TASK_ORDER, seed_selection)
    spaces, diagnostics = sync_l.build_l_spaces(exp_args, TASK_ORDER, seed_selection, data_by_task, sync_summary["common_values"])
    l_dims = {int(row["L_dimension"]) for row in diagnostics}
    if l_dims != {13}:
        raise RuntimeError(f"Panel (a) reconstructed non-13D L dimensions: {sorted(l_dims)}")
    test_rows, _, _, validation = synchronized_centroids(spaces, sync_payload, train_values, test_values, seed)
    train_rows_for_pca = []
    for task in TASK_ORDER:
        item = spaces[(task, seed)]
        orientation = torch.as_tensor(sync_payload["orientations_domain_to_hub"][task]).float()
        scale = float(torch.exp(torch.tensor(sync_payload["log_scales"][task])).item())
        center = torch.stack([item.centroids[value] for value in train_values]).mean(dim=0)
        for value in train_values:
            common = ((item.centroids[value] - center) @ orientation) / max(scale, 1e-12)
            train_rows_for_pca.append({"task": task, "value": value, "common": common.numpy()})
    train_avg_matrix = averaged_by_value(train_rows_for_pca, train_values)
    selected_values = quantile_subset(test_values, n_values=9)
    shown_rows = [row for row in test_rows if int(row["value"]) in selected_values]
    shown_matrix = np.stack([row["common"] for row in shown_rows], axis=0)
    shown_avg_matrix = averaged_by_value(shown_rows, selected_values)
    combined_matrix = np.concatenate([shown_matrix, shown_avg_matrix], axis=0)
    projected, explained = pca_from_train(train_avg_matrix, combined_matrix, n_components=3)
    projected_rows = projected[: len(shown_rows)]
    projected_shared = projected[len(shown_rows) :]
    shared_by_value = {
        int(value): {"mean_pc1": float(point[0]), "mean_pc2": float(point[1]), "mean_pc3": float(point[2])}
        for value, point in zip(selected_values, projected_shared)
    }
    for row, point in zip(shown_rows, projected_rows):
        row["pc1"] = float(point[0])
        row["pc2"] = float(point[1])
        row["pc3"] = float(point[2])
        row.update(shared_by_value[int(row["value"])])
    validation.update(
        {
            "seed": seed,
            "value_split_seed": value_split,
            "train_values": train_values,
            "test_values": test_values,
            "shown_values": selected_values,
            "n_heldout_values": len(selected_values),
            "L_dimension": 13,
            "max_digit_overlap": max(float(row.get("digit_overlap", 0.0)) for row in diagnostics),
            "pca_explained_variance_ratio": [float(explained[0]), float(explained[1]), float(explained[2])],
            "pca_cumulative_pc1_pc2": float(explained[0] + explained[1]),
            "pca_fit_object": "task-averaged synchronized exact-L centroids from training values",
            "activation_dir_text": str(exp_args.activation_dir_text),
            "activation_dir_image": str(exp_args.activation_dir_image),
        }
    )
    return shown_rows, validation


def draw_panel_a(ax: plt.Axes, rows: list[dict]) -> None:
    style_panel(ax, "Synchronized L geometry", "(a)")
    values = np.array([row["value"] for row in rows], dtype=float)
    norm = mpl.colors.Normalize(vmin=float(values.min()), vmax=float(values.max()))
    cmap = mpl.colormaps["viridis"]

    by_value: dict[int, tuple[float, float]] = {}
    for row in rows:
        by_value.setdefault(int(row["value"]), (row["mean_pc1"], row["mean_pc2"]))
    means = {value: np.array(point) for value, point in by_value.items()}
    for row in rows:
        mean_xy = means[int(row["value"])]
        ax.plot([row["pc1"], mean_xy[0]], [row["pc2"], mean_xy[1]], color=GRID, linewidth=0.35, alpha=0.32, zorder=1)

    for value, xy in means.items():
        ax.scatter(
            [xy[0]],
            [xy[1]],
            c=[value],
            cmap=cmap,
            norm=norm,
            marker="o",
            s=72,
            alpha=0.96,
            edgecolors=SPINE,
            linewidths=0.55,
            zorder=4,
        )

    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        ax.scatter(
            [row["pc1"] for row in task_rows],
            [row["pc2"] for row in task_rows],
            c=[row["value"] for row in task_rows],
            cmap=cmap,
            norm=norm,
            marker=TASK_MARKERS[task],
            s=24,
            alpha=0.78,
            edgecolors="#ffffff",
            linewidths=0.35,
            label=TASK_LABELS[task],
            zorder=3,
        )

    label_values = quantile_subset([int(v) for v in values], n_values=5)
    for value in label_values:
        xy = means[value]
        ax.text(xy[0], xy[1], str(value), fontsize=5.7, color=TEXT, ha="center", va="center", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.tick_params(length=2.2, pad=1.5)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.text(
        0.035,
        0.055,
        "held-out values;\nsynchronized exact L (13D)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.9,
        color=SECONDARY_TEXT,
        linespacing=1.05,
    )
    handles = [
        Line2D([0], [0], marker=TASK_MARKERS[task], color="none", markerfacecolor="#b9c8c8", markeredgecolor=SPINE, markersize=4.4, label=TASK_LABELS[task])
        for task in TASK_ORDER
    ]
    handles.insert(0, Line2D([0], [0], marker="o", color="none", markerfacecolor="#b9c8c8", markeredgecolor=SPINE, markersize=5.6, label=r"$\bar{u}(y)$"))
    ax.legend(handles=handles, loc="upper right", frameon=False, ncol=2, columnspacing=0.8, handletextpad=0.25, borderpad=0.1)
    cax = inset_axes(ax, width="36%", height="4.7%", loc="lower right", borderpad=1.05)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Result value", fontsize=5.8, labelpad=-18, color=TEXT)
    cbar.ax.tick_params(labelsize=5.4, length=1.8, pad=1)
    cbar.outline.set_linewidth(0.45)


def synchronization_all_rows(rows: list[dict]) -> list[dict]:
    order = ["pairwise_direct", "synchronized_full", "synchronized_loro"]
    selected = {row["transport"]: row for row in rows if row["relation"] == "ALL"}
    missing = [transport for transport in order if transport not in selected]
    if missing:
        raise RuntimeError(f"Missing aggregate synchronization rows: {missing}")
    return [selected[transport] for transport in order]


def draw_sync_bars(ax: plt.Axes, rows: list[dict]) -> list[dict]:
    style_panel(ax, "Synchronization generalization", "(b)", grid=True)
    summary_rows = synchronization_all_rows(rows)
    x = np.arange(len(summary_rows), dtype=float)
    means = [float(row["transition_cosine"]) for row in summary_rows]
    errors = [float(row.get("heldout_transition_cosine_mean_sem", 0.0) or 0.0) for row in summary_rows]
    labels = [CONTROL_LABELS[row["transport"]] for row in summary_rows]
    colors = [CONTROL_COLORS[row["transport"]] for row in summary_rows]

    bars = ax.bar(
        x,
        means,
        yerr=errors,
        color=colors,
        edgecolor=SPINE,
        linewidth=0.72,
        width=0.54,
        capsize=2.0,
        error_kw={"elinewidth": 0.72, "ecolor": TEXT, "capthick": 0.72},
        zorder=3,
    )
    for bar, mean_value, error in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, mean_value + error + 0.005, f"{mean_value:.3f}", ha="center", va="bottom", fontsize=6.9)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Held-out transition cosine")
    ax.set_ylim(0.74, 0.855)
    ax.set_yticks([0.75, 0.80, 0.85])
    ax.margins(x=0.10)
    return summary_rows


def draw_status_cell(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    text: str,
    facecolor: str,
    edgecolor: str,
    dashed: bool = False,
    bold: bool = False,
) -> None:
    rect = Rectangle((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.82, linestyle=(0, (3, 2)) if dashed else "solid")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.45, fontweight="bold" if bold else "normal")


def draw_metric_box(ax: plt.Axes, x: float, y: float, label: str, value: str, color: str, *, width: float = 0.285, height: float = 0.122) -> None:
    rect = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.012",
        linewidth=0.72,
        edgecolor=GRID,
        facecolor="#ffffff",
        alpha=0.96,
    )
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height * 0.72, label, ha="center", va="center", fontsize=5.8, color=SECONDARY_TEXT)
    ax.text(x + width / 2, y + height * 0.24, value, ha="center", va="center", fontsize=7.0, color=color)


def factorization_metrics(summary: dict) -> dict:
    rows = summary.get("fit3_aggregate_summary", [])
    by_control = {row["control_type"]: row for row in rows}
    fit3 = by_control.get("none")
    shuffled = by_control.get("shuffled_mod_neighbor_values")
    if fit3 is None or shuffled is None:
        raise RuntimeError("Missing fit-3 or shuffled aggregate rows in L_factorization_summary.json.")
    return {
        "fit3_transition": float(fit3["heldout_transition_cosine_mean_mean"]),
        "fit3_transition_sem": float(fit3["heldout_transition_cosine_mean_sem"]),
        "shuffled_transition": float(shuffled["heldout_transition_cosine_mean_mean"]),
        "shuffled_transition_sem": float(shuffled["heldout_transition_cosine_mean_sem"]),
        "fit3_top1": float(fit3["heldout_top1_value_retrieval_mean"]),
        "fit3_top1_sem": float(fit3["heldout_top1_value_retrieval_sem"]),
        "chance": 0.05,
    }


def draw_unseen_condition(ax: plt.Axes, metrics: dict) -> dict:
    style_panel(ax, "Held-out condition", "(c)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    left = 0.055
    bottom = 0.455
    cell_w = 0.205
    cell_h = 0.145
    label_w = 0.082
    header_h = 0.070

    ax.text(left + label_w + cell_w / 2, bottom + 2 * cell_h + header_h * 0.66, "Addition", ha="center", va="center", fontsize=6.8)
    ax.text(left + label_w + 1.5 * cell_w, bottom + 2 * cell_h + header_h * 0.66, "Subtraction", ha="center", va="center", fontsize=6.8)
    ax.text(left + label_w * 0.35, bottom + 1.5 * cell_h, "Text", ha="center", va="center", fontsize=6.7)
    ax.text(left + label_w * 0.35, bottom + 0.5 * cell_h, "Image", ha="center", va="center", fontsize=6.7)

    observed_face = "#edf3f1"
    heldout_face = "#f1e6e2"
    for text, dashed, col, row in [
        ("observed", False, 0, 1),
        ("observed", False, 1, 1),
        ("observed", False, 0, 0),
        ("held out", True, 1, 0),
    ]:
        draw_status_cell(
            ax,
            left + label_w + col * cell_w,
            bottom + row * cell_h,
            cell_w,
            cell_h,
            text=text,
            facecolor=heldout_face if dashed else observed_face,
            edgecolor=RUST if dashed else GRID,
            dashed=dashed,
            bold=dashed,
        )

    ax.text(left + label_w + cell_w, 0.350, "example: I- held out", ha="center", va="center", fontsize=6.1, color=SECONDARY_TEXT)
    ax.text(
        left + label_w + cell_w,
        0.220,
        "fit on 3 conditions\nevaluate on $I-$",
        ha="center",
        va="center",
        fontsize=5.75,
        color=SECONDARY_TEXT,
        linespacing=1.05,
    )
    ax.text(
        left + label_w + cell_w,
        0.102,
        "held-out maps are\nevaluation references only",
        ha="center",
        va="center",
        fontsize=5.55,
        color=SECONDARY_TEXT,
        linespacing=1.05,
    )

    metric_x = 0.665
    draw_metric_box(ax, metric_x, 0.630, "fit-3 cosine", f"{metrics['fit3_transition']:.3f}", TEAL)
    draw_metric_box(ax, metric_x, 0.475, "shuffled", f"{metrics['shuffled_transition']:.3f}", RUST)
    draw_metric_box(ax, metric_x, 0.285, "fit-3 top-1", f"{100 * metrics['fit3_top1']:.1f}%", TEAL)
    draw_metric_box(ax, metric_x, 0.130, "chance", f"{100 * metrics['chance']:.0f}%", SECONDARY_TEXT)
    return metrics


def validation_report(
    *,
    sync_summary_path: Path,
    factorization_summary_path: Path,
    sync_payload_path: Path,
    panel_a_validation: dict,
    panel_b_rows: list[dict],
    panel_c_metrics: dict,
) -> dict:
    sync_values = {row["transport"]: float(row["transition_cosine"]) for row in panel_b_rows}
    report = {
        "exact_l_verified": True,
        "exact_representation_dimension": panel_a_validation["L_dimension"],
        "sync_summary_path": str(sync_summary_path),
        "factorization_summary_path": str(factorization_summary_path),
        "synchronization_payload_path": str(sync_payload_path),
        "panel_a_seed": panel_a_validation["seed"],
        "panel_a_value_split": panel_a_validation["value_split_seed"],
        "selected_seed_split_config": f"seed={panel_a_validation['seed']}, value_split={panel_a_validation['value_split_seed']}",
        "heldout_values_plotted": panel_a_validation["n_heldout_values"],
        "heldout_values_shown": panel_a_validation["shown_values"],
        "readout_orthogonality_max_digit_overlap": panel_a_validation["max_digit_overlap"],
        "panel_a_pca_explained_variance_ratio_pc1_pc2_pc3": panel_a_validation["pca_explained_variance_ratio"],
        "panel_a_pca_cumulative_pc1_pc2": panel_a_validation["pca_cumulative_pc1_pc2"],
        "panel_a_pca_fit_object": panel_a_validation["pca_fit_object"],
        "panel_a_uses_task_averaged_centroids_for_pca": True,
        "direct_transition_cosine": sync_values["pairwise_direct"],
        "full_sync_transition_cosine": sync_values["synchronized_full"],
        "loro_transition_cosine": sync_values["synchronized_loro"],
        "fit3_transition_cosine": panel_c_metrics["fit3_transition"],
        "shuffled_transition_cosine": panel_c_metrics["shuffled_transition"],
        "fit3_top1": panel_c_metrics["fit3_top1"],
        "panel_a_common_frame_same_value_cosine_mean": panel_a_validation["panel_a_common_frame_same_value_cosine_mean"],
        "activation_dir_text": panel_a_validation["activation_dir_text"],
        "activation_dir_image": panel_a_validation["activation_dir_image"],
    }
    if report["exact_representation_dimension"] != 13:
        raise RuntimeError("ASSERT failed: L dimension != 13")
    print("\nVALIDATION REPORT")
    for key, value in report.items():
        print(f"  {key}: {value}")
    return report


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

    sync_summary_path = args.input_dir / "L_synchronization_summary.json"
    factorization_summary_path = args.factorization_dir / "L_factorization_summary.json"
    sync_summary = load_json(sync_summary_path)
    factorization_summary = load_json(factorization_summary_path)
    assert_exact_l_summary(sync_summary, sync_summary_path)
    assert_exact_l_summary(factorization_summary, factorization_summary_path)

    sync_payload_path, sync_payload = selected_sync_payload(args.input_dir, args.panel_a_seed, args.panel_a_value_split)
    panel_a_rows, panel_a_validation = load_panel_a(args, sync_summary, sync_payload)
    panel_b_rows = synchronization_all_rows(read_csv(args.input_dir / "L_synchronization_summary_table.csv"))
    panel_c_metrics = factorization_metrics(factorization_summary)
    report = validation_report(
        sync_summary_path=sync_summary_path,
        factorization_summary_path=factorization_summary_path,
        sync_payload_path=sync_payload_path,
        panel_a_validation=panel_a_validation,
        panel_b_rows=panel_b_rows,
        panel_c_metrics=panel_c_metrics,
    )

    fig = plt.figure(figsize=(9.9, 3.15), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    fig.patch.set_facecolor(BACKGROUND)

    draw_panel_a(axes[0], panel_a_rows)
    panel_b = draw_sync_bars(axes[1], panel_b_rows)
    panel_c = draw_unseen_condition(axes[2], panel_c_metrics)

    fig.subplots_adjust(left=0.050, right=0.985, top=0.835, bottom=0.185, wspace=0.275)
    save_outputs(
        fig,
        args.output_dir,
        args.stem,
        {
            "input_dir": str(args.input_dir),
            "factorization_dir": str(args.factorization_dir),
            "validation_report": report,
            "panel_a_points": [
                {
                    "task": row["task"],
                    "label": row["label"],
                    "value": int(row["value"]),
                    "pc1": float(row["pc1"]),
                    "pc2": float(row["pc2"]),
                    "pc3": float(row["pc3"]),
                    "mean_pc1": float(row["mean_pc1"]),
                    "mean_pc2": float(row["mean_pc2"]),
                    "mean_pc3": float(row["mean_pc3"]),
                }
                for row in panel_a_rows
            ],
            "panel_b_synchronization": panel_b,
            "panel_c_unseen_condition": panel_c,
        },
    )


if __name__ == "__main__":
    main()
