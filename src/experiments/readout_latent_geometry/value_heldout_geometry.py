"""Held-out numerical-value geometry after removing digit readout directions.

This offline experiment asks whether the cross-task Procrustes geometry that
remains after readout ablation is organized by numerical value.  It never runs
the model by default: it uses saved final-position activations, final DAS
subspaces, and optionally already-materialized readout-ablated subspaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from scipy.stats import rankdata

from src.common import load_jsonl
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    first_result_row,
    jsonable,
    parse_task,
    save_json,
    selected_seeds,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    orthogonal_procrustes,
    scaled_alpha,
    stable_seed,
)
from src.geometry.subspaces import torch_load_portable
from src.models import validate_saved_block_layers


EXPERIMENT = "value_heldout_readout_free_geometry"
DEFAULT_TASKS = [
    "text:addition",
    "text:subtraction",
    "image:addition",
    "image:subtraction",
]
TASK_ORDER = ["text:addition", "text:subtraction", "image:addition", "image:subtraction"]
TASK_LABELS = {
    "text:addition": "T+",
    "text:subtraction": "T-",
    "image:addition": "I+",
    "image:subtraction": "I-",
}


@dataclass
class TaskData:
    task: str
    modality: str
    operation: str
    row: dict
    labels: list[dict]
    hidden: torch.Tensor
    counts: dict[int, int]
    activation_path: Path


@dataclass
class TaskSpace:
    task: str
    seed: int
    space: str
    basis: torch.Tensor
    centroids: dict[int, torch.Tensor]
    diagnostics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--readout_free_audit_root",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43"),
        help="Optional audit-shaped root containing already readout-ablated subspace.pt files.",
    )
    parser.add_argument(
        "--digit_readout_basis_path",
        type=Path,
        default=Path("results/experiments/closing/readout_causal_audit/digit_readout_basis.pt"),
        help="Optional torch file with centered digit readout basis for strict projection-out reconstruction.",
    )
    parser.add_argument(
        "--readout_free_source",
        choices=["auto", "project", "load_ablated"],
        default="auto",
        help="auto uses --digit_readout_basis_path if present, otherwise --readout_free_audit_root.",
    )
    parser.add_argument(
        "--activation_dir_text",
        type=Path,
        default=Path("outputs/activations/baseline/gemma4_12b_it/digits"),
    )
    parser.add_argument(
        "--activation_dir_image",
        type=Path,
        default=Path("outputs/activations/baseline_images/gemma4_12b_it/digits"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/value_heldout_readout_free_geometry"))
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--seed_map", nargs="*", default=[], metavar="TASK=SEEDS")
    parser.add_argument("--value_split_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train_value_fraction", type=float, default=0.8)
    parser.add_argument("--min_examples_per_value", type=int, default=1)
    parser.add_argument("--random_controls", type=int, default=20)
    parser.add_argument("--rsa_permutations", type=int, default=1000)
    parser.add_argument("--spaces", nargs="+", choices=["original", "readout_free"], default=["original", "readout_free"])
    parser.add_argument(
        "--artifact_check_only",
        action="store_true",
        help="Load and validate datasets, activation slices, and DAS/readout-free bases, then exit before Procrustes.",
    )
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_tasks(tasks: list[str]) -> list[str]:
    if any(task.lower() == "all" for task in tasks):
        return list(DEFAULT_TASKS)
    return [task_key(*parse_task(task)) for task in tasks]


def position_for_task(args: argparse.Namespace, modality: str) -> str:
    return str(args.image_position if modality == "image" else args.text_position)


def run_dir(root: Path, args: argparse.Namespace, modality: str, operation: str, condition: str, seed: int) -> Path:
    base = root / modality / operation
    if args.target != "result":
        base = base / args.target
    return base / condition / f"split_{args.split_seed}" / f"seed_{seed}"


def load_basis_file(path: Path, hidden_size: int, k: int | None = None) -> tuple[torch.Tensor, dict]:
    payload = torch_load_portable(path)
    if "basis" in payload:
        basis = payload["basis"]
    elif "bases" in payload:
        saved = payload["bases"]
        basis = next(iter(saved.values())) if isinstance(saved, dict) else saved
    else:
        raise ValueError(f"{path} does not contain a DAS basis.")
    basis = torch.as_tensor(basis).detach().float().squeeze()
    if basis.ndim != 2:
        raise ValueError(f"{path} basis must be a rank-2 tensor; got {tuple(basis.shape)}.")
    if basis.shape[0] != hidden_size and basis.shape[1] == hidden_size:
        basis = basis.T
    if basis.shape[0] != hidden_size:
        raise ValueError(f"{path} basis hidden size {tuple(basis.shape)} does not match activations d={hidden_size}.")
    if k is not None and basis.shape[1] < k:
        raise ValueError(f"{path} basis rank {basis.shape[1]} is smaller than requested k={k}.")
    return basis[:, :k] if k is not None else basis, payload


def result_row(path: Path) -> dict:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}.")
    return rows[0]


def validate_result_metadata(
    args: argparse.Namespace,
    row: dict,
    *,
    task: str,
    path: Path,
) -> None:
    modality, _operation = parse_task(task)
    expected_position = position_for_task(args, modality)
    mismatches = []
    checks = {
        "layer": (row.get("layer"), args.layer),
        "k": (row.get("k"), args.k),
        "position": (row.get("position"), expected_position),
        "hook": (row.get("hook"), args.hook),
    }
    for key, (actual, expected) in checks.items():
        if str(actual) != str(expected):
            mismatches.append(f"{key}={actual!r} expected {expected!r}")
    if mismatches:
        raise ValueError(
            f"{path} is not the requested site for {task}: "
            + ", ".join(mismatches)
        )


def orthonormal_basis(matrix: torch.Tensor, name: str, rank: int | None = None) -> torch.Tensor:
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        raise ValueError(f"{name} has no singular values.")
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values.max()
    numerical_rank = int((singular_values > tolerance).sum().item())
    if numerical_rank == 0:
        raise ValueError(f"{name} has numerical rank zero.")
    if rank is not None:
        numerical_rank = min(rank, numerical_rank)
    return left[:, :numerical_rank]


def orthogonality_error(basis: torch.Tensor) -> float:
    eye = torch.eye(basis.shape[1], dtype=basis.dtype)
    return float((basis.T @ basis - eye).abs().max())


def load_digit_basis(path: Path, hidden_size: int) -> torch.Tensor:
    payload = torch_load_portable(path)
    if isinstance(payload, torch.Tensor):
        basis = payload
    elif "basis" in payload:
        basis = payload["basis"]
    elif "digit_basis" in payload:
        basis = payload["digit_basis"]
    elif "digit_readout_basis" in payload:
        basis = payload["digit_readout_basis"]
    else:
        raise ValueError(f"{path} has no basis/digit_basis/digit_readout_basis tensor.")
    basis = torch.as_tensor(basis).detach().float().squeeze()
    if basis.shape[0] != hidden_size and basis.shape[1] == hidden_size:
        basis = basis.T
    if basis.shape[0] != hidden_size:
        raise ValueError(f"{path} basis shape {tuple(basis.shape)} does not match hidden size {hidden_size}.")
    return orthonormal_basis(basis, "centered digit readout span")


def project_out_readout(basis: torch.Tensor, digit_basis: torch.Tensor) -> tuple[torch.Tensor, dict]:
    original = orthonormal_basis(basis, "original DAS basis", rank=basis.shape[1])
    projected = original - digit_basis @ (digit_basis.T @ original)
    readout_free = orthonormal_basis(projected, "readout-free DAS projection")
    return readout_free, {
        "readout_free_source": "project",
        "rank_R": int(original.shape[1]),
        "rank_L": int(readout_free.shape[1]),
        "orthogonality_error_L": orthogonality_error(readout_free),
        "digit_overlap_norm": float((digit_basis.T @ readout_free).norm()),
        "digit_overlap_squared_before": float((digit_basis.T @ original).square().sum()),
        "digit_overlap_squared_after": float((digit_basis.T @ readout_free).square().sum()),
    }


def sample_key(sample: dict, fallback: int | None = None):
    if "sample_id" in sample:
        return sample["sample_id"]
    if "id" in sample:
        return sample["id"]
    if "expr" in sample:
        return sample["expr"]
    if fallback is not None:
        return fallback
    raise KeyError("Sample has no sample_id/id/expr.")


def activation_path_for(args: argparse.Namespace, modality: str, operation: str) -> Path:
    root = args.activation_dir_image if modality == "image" else args.activation_dir_text
    candidates = [
        root / f"{operation}_baseline.pt",
        root / operation / f"{operation}_baseline.pt",
        root / f"{operation}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def select_axis_index(values: list, wanted, axis_name: str, path: Path) -> int:
    wanted_text = str(wanted)
    for index, value in enumerate(values):
        if str(value) == wanted_text:
            return index
    raise ValueError(f"{path} has no {axis_name}={wanted_text!r}; available={values}.")


def load_task_data(args: argparse.Namespace, task: str, row: dict) -> TaskData:
    modality, operation = parse_task(task)
    activation_path = activation_path_for(args, modality, operation)
    if not activation_path.exists():
        raise FileNotFoundError(
            f"Missing saved activations for {task}: {activation_path}. "
            "Set ACTIVATION_DIR_TEXT/ACTIVATION_DIR_IMAGE or pass --activation_dir_*."
        )
    payload = torch.load(activation_path, map_location="cpu", weights_only=False)
    activations = payload["activations"]
    labels = list(payload["labels"])
    layers = validate_saved_block_layers(payload["block_layers"])
    positions = payload.get("position_names") or [str(x) for x in payload.get("position_indices", [])]
    layer_idx = select_axis_index(layers, args.layer, "layer", activation_path)
    position_idx = select_axis_index(positions, position_for_task(args, modality), "position", activation_path)
    hidden_by_label = activations[:, layer_idx, position_idx, :].float()

    data_path = Path(row["data_path"])
    samples = load_jsonl(data_path)
    allowed_keys = {sample_key(sample, index) for index, sample in enumerate(samples)}
    allowed_exprs = {sample.get("expr") for sample in samples if "expr" in sample}
    filtered_labels = []
    filtered_hidden = []
    for index, label_row in enumerate(labels):
        key = sample_key(label_row, index)
        if key in allowed_keys or label_row.get("expr") in allowed_exprs:
            filtered_labels.append(label_row)
            filtered_hidden.append(hidden_by_label[index])
    if not filtered_labels:
        raise ValueError(
            f"No activation labels in {activation_path} matched correct-example dataset {data_path}."
        )
    hidden = torch.stack(filtered_hidden)
    counts: dict[int, int] = {}
    for item in filtered_labels:
        value = int(item[args.target])
        counts[value] = counts.get(value, 0) + 1
    print(
        f"Loaded {task}: activations={activation_path} layer={args.layer} "
        f"position={position_for_task(args, modality)} examples={len(filtered_labels)} "
        f"values={len(counts)}"
    )
    return TaskData(task, modality, operation, row, filtered_labels, hidden, counts, activation_path)


def centroid_coordinates(data: TaskData, basis: torch.Tensor, values: set[int], target: str) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    coords = data.hidden @ basis
    for item, coordinate in zip(data.labels, coords):
        value = int(item[target])
        if value in values:
            grouped.setdefault(value, []).append(coordinate)
    return {value: torch.stack(parts).mean(dim=0) for value, parts in grouped.items()}


def split_values(values: list[int], train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--train_value_fraction must be between 0 and 1.")
    shuffled = list(sorted(values))
    random.Random(seed).shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_fraction))
    n_train = max(1, min(n_train, len(shuffled) - 1))
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:])


def transition_matrix(centroids: dict[int, torch.Tensor], values: list[int]) -> tuple[list[tuple[int, int]], torch.Tensor]:
    transitions = [(a, b) for a in values for b in values if a != b]
    if not transitions:
        return [], torch.empty(0, next(iter(centroids.values())).shape[0])
    matrix = torch.stack([centroids[b] - centroids[a] for a, b in transitions])
    return transitions, matrix


def rectangular_random_orthogonal(source_dim: int, destination_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if source_dim >= destination_dim:
        q, _ = torch.linalg.qr(torch.randn(source_dim, destination_dim, generator=generator), mode="reduced")
        return q.float()
    q, _ = torch.linalg.qr(torch.randn(destination_dim, source_dim, generator=generator), mode="reduced")
    return q.T.float()


def fit_scaled_procrustes(x_train: torch.Tensor, y_train: torch.Tensor) -> tuple[torch.Tensor, float]:
    if x_train.shape[1] == y_train.shape[1]:
        q = orthogonal_procrustes(x_train, y_train)
    else:
        u, _s, vh = torch.linalg.svd(x_train.T @ y_train, full_matrices=False)
        q = u @ vh
    alpha = scaled_alpha(x_train, y_train, q)
    return q.float(), float(alpha)


def transition_metrics(x: torch.Tensor, y: torch.Tensor, q: torch.Tensor, alpha: float) -> dict:
    pred = float(alpha) * (x @ q)
    cosine = torch.nn.functional.cosine_similarity(pred, y, dim=1)
    relative_error = (pred - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-12)
    return {
        "heldout_transition_cosine_mean": float(cosine.mean()),
        "heldout_transition_cosine_median": float(cosine.median()),
        "heldout_transition_cosine_std": float(cosine.std(unbiased=True)) if cosine.numel() > 1 else 0.0,
        "heldout_relative_error_mean": float(relative_error.mean()),
        "heldout_relative_error_median": float(relative_error.median()),
        "heldout_relative_error_std": float(relative_error.std(unbiased=True)) if relative_error.numel() > 1 else 0.0,
    }


def retrieval_metrics(
    source_centroids: dict[int, torch.Tensor],
    destination_centroids: dict[int, torch.Tensor],
    train_values: list[int],
    test_values: list[int],
    q: torch.Tensor,
    alpha: float,
) -> dict:
    train_offsets = [
        destination_centroids[value] - float(alpha) * (source_centroids[value] @ q)
        for value in train_values
    ]
    offset = torch.stack(train_offsets).mean(dim=0)
    destination_matrix = torch.stack([destination_centroids[value] for value in test_values])
    destination_center = destination_matrix.mean(dim=0)
    exact = 0
    top5 = 0
    abs_errors = []
    cosine_exact = 0
    cosine_top5 = 0
    cosines = []
    for value in test_values:
        mapped = offset + float(alpha) * (source_centroids[value] @ q)
        true = destination_centroids[value]
        cosines.append(float(torch.nn.functional.cosine_similarity((mapped - destination_center)[None], (true - destination_center)[None])))
        distances = (destination_matrix - mapped).norm(dim=1)
        order = torch.argsort(distances).tolist()
        predicted = test_values[order[0]]
        exact += int(predicted == value)
        top5 += int(value in [test_values[i] for i in order[:5]])
        abs_errors.append(abs(predicted - value))

        centered_destination = destination_matrix - destination_center
        mapped_centered = mapped - destination_center
        cosine_scores = torch.nn.functional.cosine_similarity(mapped_centered[None], centered_destination, dim=1)
        cosine_order = torch.argsort(cosine_scores, descending=True).tolist()
        cosine_predicted = test_values[cosine_order[0]]
        cosine_exact += int(cosine_predicted == value)
        cosine_top5 += int(value in [test_values[i] for i in cosine_order[:5]])
    n = len(test_values)
    return {
        "same_value_cosine_mean": float(sum(cosines) / n),
        "same_value_cosine_median": float(statistics.median(cosines)),
        "top1_value_retrieval": exact / n,
        "top5_value_retrieval": top5 / n,
        "mean_absolute_value_error": float(sum(abs_errors) / n),
        "median_absolute_value_error": float(statistics.median(abs_errors)),
        "cosine_top1_value_retrieval": cosine_exact / n,
        "cosine_top5_value_retrieval": cosine_top5 / n,
    }


def distance_vector(centroids: dict[int, torch.Tensor], values: list[int]) -> torch.Tensor:
    matrix = torch.stack([centroids[value] for value in values])
    distances = torch.cdist(matrix, matrix, p=2)
    upper = torch.triu_indices(len(values), len(values), offset=1)
    return distances[upper[0], upper[1]]


def spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    xr = torch.tensor(rankdata(x.detach().cpu().numpy()), dtype=torch.float32)
    yr = torch.tensor(rankdata(y.detach().cpu().numpy()), dtype=torch.float32)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = xr.norm() * yr.norm()
    if float(denom) < 1e-12:
        return float("nan")
    return float((xr * yr).sum() / denom)


def rsa_row(
    task_a: TaskSpace,
    task_b: TaskSpace,
    test_values: list[int],
    value_split_seed: int,
    permutations: int,
    seed: int,
) -> dict:
    values = sorted(test_values)
    observed = spearman(distance_vector(task_a.centroids, values), distance_vector(task_b.centroids, values))
    rng = random.Random(seed)
    null = []
    for _ in range(permutations):
        shuffled = list(values)
        rng.shuffle(shuffled)
        paired = {value: task_b.centroids[shuffled[index]] for index, value in enumerate(values)}
        null.append(spearman(distance_vector(task_a.centroids, values), distance_vector(paired, values)))
    pvalue = (1 + sum(item >= observed for item in null)) / (permutations + 1) if permutations else None
    return {
        "task_a": task_a.task,
        "task_b": task_b.task,
        "task_a_label": TASK_LABELS[task_a.task],
        "task_b_label": TASK_LABELS[task_b.task],
        "seed_a": task_a.seed,
        "seed_b": task_b.seed,
        "value_split_seed": value_split_seed,
        "space": task_a.space,
        "n_test_values": len(values),
        "spearman_rsa": observed,
        "permutation_pvalue": pvalue,
        "rsa_permutations": permutations,
        "permutation_null_mean": None if not null else float(sum(null) / len(null)),
        "permutation_null_std": None if len(null) < 2 else float(statistics.stdev(null)),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def load_cache(path: Path, key_fields: list[str], force: bool) -> dict[tuple, dict]:
    if force or not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {tuple(row[field] for field in key_fields): row for row in csv.DictReader(handle)}


def summarize(rows: list[dict], metrics: list[str], group_keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for key, parts in sorted(groups.items()):
        item = {group_keys[index]: key[index] for index in range(len(group_keys))}
        item["n"] = len(parts)
        for metric in metrics:
            values = [float(row[metric]) for row in parts if row.get(metric) not in {None, ""}]
            item[f"{metric}_mean"] = None if not values else float(sum(values) / len(values))
            item[f"{metric}_std"] = None if len(values) < 2 else float(statistics.stdev(values))
        output.append(item)
    return output


def plot_figures(args: argparse.Namespace, transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    relations = [f"{TASK_LABELS[a]}->{TASK_LABELS[b]}" for a in TASK_ORDER for b in TASK_ORDER if a != b]

    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    x = np.arange(len(relations))
    width = 0.25
    for offset, space, title in [(-width, "original", "Original DAS"), (0.0, "readout_free", "Readout-free DAS")]:
        means = []
        for relation in relations:
            source_label, dest_label = relation.split("->")
            values = [
                float(row["heldout_transition_cosine_mean"])
                for row in transition_rows
                if row.get("control_type") in {None, ""}
                and TASK_LABELS[row["source_task"]] == source_label
                and TASK_LABELS[row["destination_task"]] == dest_label
                and row["space"] == space
            ]
            means.append(np.nan if not values else float(np.mean(values)))
        ax.bar(x + offset, means, width, label=title)
    shuffled = []
    for relation in relations:
        source_label, dest_label = relation.split("->")
        values = [
            float(row["heldout_transition_cosine_mean"])
            for row in transition_rows
            if TASK_LABELS[row["source_task"]] == source_label
            and TASK_LABELS[row["destination_task"]] == dest_label
            and row.get("control_type") == "shuffled_values"
        ]
        shuffled.append(np.nan if not values else float(np.mean(values)))
    ax.bar(x + width, shuffled, width, label="Shuffled-value fit")
    ax.set_ylabel("held-out transition cosine")
    ax.set_title("Value-held-out transition Procrustes (test values unseen during fit)")
    ax.set_xticks(x, relations, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure1_heldout_transition_geometry.png", dpi=240)
    fig.savefig(figure_dir / "figure1_heldout_transition_geometry.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    means = []
    top5 = []
    chances = []
    for relation in relations:
        source_label, dest_label = relation.split("->")
        rows = [
            row
            for row in retrieval_rows
            if TASK_LABELS[row["source_task"]] == source_label
            and TASK_LABELS[row["destination_task"]] == dest_label
            and row["space"] == "readout_free"
        ]
        means.append(np.nan if not rows else float(np.mean([float(row["top1_value_retrieval"]) for row in rows])))
        top5.append(np.nan if not rows else float(np.mean([float(row["top5_value_retrieval"]) for row in rows])))
        chances.append(np.nan if not rows else float(np.mean([1.0 / float(row["n_test_values"]) for row in rows])))
    ax.bar(x - width / 2, means, width, label="top-1")
    ax.bar(x + width / 2, top5, width, label="top-5")
    ax.plot(x, chances, color="black", linestyle="--", linewidth=1.0, label="chance top-1")
    ax.set_ylabel("held-out value retrieval")
    ax.set_title("Readout-free held-out same-value retrieval")
    ax.set_xticks(x, relations, rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure2_heldout_value_retrieval.png", dpi=240)
    fig.savefig(figure_dir / "figure2_heldout_value_retrieval.pdf")
    plt.close(fig)

    matrix = np.full((len(TASK_ORDER), len(TASK_ORDER)), np.nan)
    pvalues = np.full_like(matrix, np.nan)
    for i, task_a in enumerate(TASK_ORDER):
        for j, task_b in enumerate(TASK_ORDER):
            rows = [row for row in rsa_rows if row["task_a"] == task_a and row["task_b"] == task_b and row["space"] == "readout_free"]
            if rows:
                matrix[i, j] = np.mean([float(row["spearman_rsa"]) for row in rows])
                finite_p = [float(row["permutation_pvalue"]) for row in rows if row.get("permutation_pvalue") not in {None, ""}]
                pvalues[i, j] = np.mean(finite_p) if finite_p else np.nan
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, values, title, cmap, vmin, vmax in [
        (axes[0], matrix, "Mean readout-free RSA", "viridis", -1, 1),
        (axes[1], pvalues, "Mean permutation p-value", "magma_r", 0, 1),
    ]:
        image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(TASK_ORDER)), [TASK_LABELS[t] for t in TASK_ORDER])
        ax.set_yticks(range(len(TASK_ORDER)), [TASK_LABELS[t] for t in TASK_ORDER])
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure3_nofit_rsa.png", dpi=240)
    fig.savefig(figure_dir / "figure3_nofit_rsa.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tasks = normalize_tasks(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_selection = selected_seeds(args)
    print("Value-held-out readout-free geometry")
    print(f"  tasks={args.tasks}")
    print(f"  spaces={args.spaces}")
    print(f"  value_split_seeds={args.value_split_seeds}")

    rows_by_task = {}
    data_by_task = {}
    for task in args.tasks:
        modality, operation = parse_task(task)
        row = first_result_row(args, modality, operation, args.condition, seed_selection[task][0])
        rows_by_task[task] = row
        data_by_task[task] = load_task_data(args, task, row)

    common_values = sorted(
        set.intersection(
            *[
                {value for value, count in data_by_task[task].counts.items() if count >= args.min_examples_per_value}
                for task in args.tasks
            ]
        )
    )
    if len(common_values) < 3:
        raise ValueError(f"Need at least three common result values after filtering; got {len(common_values)}.")
    per_task_counts = {
        task: [data_by_task[task].counts[value] for value in common_values]
        for task in args.tasks
    }
    print(f"Common result values: {len(common_values)}")
    for task in args.tasks:
        counts = per_task_counts[task]
        print(
            f"  {task}: min/median/max examples per common value = "
            f"{min(counts)}/{statistics.median(counts)}/{max(counts)}"
        )

    value_splits = {}
    for split_seed in args.value_split_seeds:
        train_values, test_values = split_values(common_values, args.train_value_fraction, split_seed)
        value_splits[str(split_seed)] = {"train_values": train_values, "test_values": test_values}
    save_json(value_splits, args.output_dir / "value_splits.json")

    spaces: dict[tuple[str, int, str], TaskSpace] = {}
    space_diagnostics = []
    for task in args.tasks:
        modality, operation = parse_task(task)
        data = data_by_task[task]
        hidden_size = data.hidden.shape[1]
        digit_basis = None
        if args.readout_free_source in {"auto", "project"} and args.digit_readout_basis_path.exists():
            digit_basis = load_digit_basis(args.digit_readout_basis_path, hidden_size)
        elif args.readout_free_source == "project":
            raise FileNotFoundError(
                f"--readout_free_source project requires {args.digit_readout_basis_path}. "
                "Use --readout_free_source load_ablated to reuse materialized ablated subspaces."
            )
        for seed in seed_selection[task]:
            original_path = subspace_path(args, modality, operation, seed)
            original_basis_raw, _payload = load_basis_file(original_path, hidden_size, args.k)
            original_basis = orthonormal_basis(original_basis_raw, f"{task} seed={seed} original", rank=args.k)
            if "original" in args.spaces:
                centroids = centroid_coordinates(data, original_basis, set(common_values), args.target)
                spaces[(task, seed, "original")] = TaskSpace(
                    task,
                    seed,
                    "original",
                    original_basis,
                    centroids,
                    {
                        "readout_free_source": None,
                        "rank_R": int(original_basis.shape[1]),
                        "rank_L": None,
                        "orthogonality_error_R": orthogonality_error(original_basis),
                    },
                )
            if "readout_free" in args.spaces:
                if digit_basis is not None:
                    readout_basis, diagnostics = project_out_readout(original_basis, digit_basis)
                    readout_basis_path = args.digit_readout_basis_path
                else:
                    ablated_dir = run_dir(args.readout_free_audit_root, args, modality, operation, args.condition, seed)
                    ablated_path = ablated_dir / "subspace.pt"
                    ablated_results = ablated_dir / "results.jsonl"
                    if not ablated_path.exists() or not ablated_results.exists():
                        raise FileNotFoundError(
                            f"Missing materialized readout-free DAS for {task} seed={seed} at {ablated_dir}. "
                            "With the currently checked local artifacts, "
                            "results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43 "
                            "contains das_pca_initialized seeds 1 and 2, but not seed 0. "
                            "Either run with SEEDS='1 2', generate the missing readout-ablated seed 0 "
                            "subspaces, or provide --digit_readout_basis_path so this script can project "
                            "the original DAS basis directly."
                        )
                    validate_result_metadata(args, result_row(ablated_results), task=task, path=ablated_results)
                    readout_basis_raw, ablated_payload = load_basis_file(ablated_path, hidden_size, None)
                    readout_basis = orthonormal_basis(readout_basis_raw, f"{task} seed={seed} materialized readout-free")
                    ablation_meta = ablated_payload.get("readout_ablation", {})
                    if float(ablation_meta.get("readout_in_ablated_subspace", 0.0)) > 1e-6:
                        raise ValueError(
                            f"{ablated_path} is not sufficiently readout-free: "
                            f"readout_in_ablated_subspace={ablation_meta.get('readout_in_ablated_subspace')}"
                        )
                    diagnostics = {
                        "readout_free_source": "load_ablated",
                        "rank_R": int(original_basis.shape[1]),
                        "rank_L": int(readout_basis.shape[1]),
                        "orthogonality_error_L": orthogonality_error(readout_basis),
                        "digit_overlap_norm": None,
                        "readout_in_original_subspace": ablation_meta.get("readout_in_original_subspace"),
                        "readout_in_ablated_subspace": ablation_meta.get("readout_in_ablated_subspace"),
                    }
                    readout_basis_path = ablated_path
                centroids = centroid_coordinates(data, readout_basis, set(common_values), args.target)
                spaces[(task, seed, "readout_free")] = TaskSpace(task, seed, "readout_free", readout_basis, centroids, diagnostics)
                print(
                    f"READOUT_FREE_SPACE task={task} das_seed={seed} "
                    f"rank(R)={diagnostics.get('rank_R')} rank(L)={diagnostics.get('rank_L')} "
                    f"||L^T L-I||={diagnostics.get('orthogonality_error_L')} "
                    f"||U^T L||={diagnostics.get('digit_overlap_norm')}"
                )
            for space_name in args.spaces:
                item = spaces[(task, seed, space_name)]
                space_diagnostics.append(
                    {
                        "task": task,
                        "task_label": TASK_LABELS[task],
                        "das_seed": seed,
                        "space": space_name,
                        "rank": int(item.basis.shape[1]),
                        "orthogonality_error": orthogonality_error(item.basis),
                        "basis_path": str(original_path if space_name == "original" else readout_basis_path),
                        **item.diagnostics,
                    }
                )

    write_csv(space_diagnostics, args.output_dir / "readout_free_space_diagnostics.csv")
    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "seed_selection": seed_selection,
                "n_common_values": len(common_values),
                "common_values": common_values,
                "per_task_common_value_counts": per_task_counts,
                "value_splits": value_splits,
                "space_diagnostics_rows": len(space_diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("\nARTIFACT_CHECK_ONLY complete")
        print(f"Validated tasks: {args.tasks}")
        print(f"Validated seeds: {seed_selection}")
        print(f"Common values: {len(common_values)}")
        print(f"Saved diagnostics: {args.output_dir / 'readout_free_space_diagnostics.csv'}")
        print(f"Saved summary: {args.output_dir / 'artifact_check_summary.json'}")
        return

    transition_rows = []
    retrieval_rows = []
    control_rows = []
    rsa_rows = []
    task_pairs = [(a, b) for a in args.tasks for b in args.tasks if a != b]

    for split_seed_text, split in value_splits.items():
        split_seed = int(split_seed_text)
        train_values = split["train_values"]
        test_values = split["test_values"]
        print(f"\nVALUE_SPLIT seed={split_seed} train={len(train_values)} test={len(test_values)}")
        for space_name in args.spaces:
            for source_task, destination_task in task_pairs:
                for source_seed in seed_selection[source_task]:
                    for destination_seed in seed_selection[destination_task]:
                        source = spaces[(source_task, source_seed, space_name)]
                        destination = spaces[(destination_task, destination_seed, space_name)]
                        train_transitions, x_train = transition_matrix(source.centroids, train_values)
                        _train_transitions_b, y_train = transition_matrix(destination.centroids, train_values)
                        test_transitions, x_test = transition_matrix(source.centroids, test_values)
                        _test_transitions_b, y_test = transition_matrix(destination.centroids, test_values)
                        q, alpha = fit_scaled_procrustes(x_train, y_train)
                        metrics = transition_metrics(x_test, y_test, q, alpha)
                        base_row = {
                            "source_task": source_task,
                            "source_task_label": TASK_LABELS[source_task],
                            "destination_task": destination_task,
                            "destination_task_label": TASK_LABELS[destination_task],
                            "source_seed": source_seed,
                            "destination_seed": destination_seed,
                            "value_split_seed": split_seed,
                            "space": space_name,
                            "n_train_values": len(train_values),
                            "n_test_values": len(test_values),
                            "n_train_transitions": len(train_transitions),
                            "n_test_transitions": len(test_transitions),
                            "source_rank": source.basis.shape[1],
                            "destination_rank": destination.basis.shape[1],
                            "alpha": alpha,
                        }
                        transition_rows.append({**base_row, **metrics})
                        retrieval_rows.append(
                            {
                                **base_row,
                                **retrieval_metrics(
                                    source.centroids,
                                    destination.centroids,
                                    train_values,
                                    test_values,
                                    q,
                                    alpha,
                                ),
                            }
                        )
                        rng = random.Random(stable_seed(EXPERIMENT, "shuffle", source_task, destination_task, source_seed, destination_seed, split_seed, space_name))
                        for control_seed in range(args.random_controls):
                            shuffled_train_values = list(train_values)
                            rng.shuffle(shuffled_train_values)
                            shuffled_destination = {
                                value: destination.centroids[shuffled_train_values[index]]
                                for index, value in enumerate(train_values)
                            }
                            _transitions, y_train_shuffled = transition_matrix(shuffled_destination, train_values)
                            q_shuffle, alpha_shuffle = fit_scaled_procrustes(x_train, y_train_shuffled)
                            control_metrics = transition_metrics(x_test, y_test, q_shuffle, alpha_shuffle)
                            control_rows.append(
                                {
                                    **base_row,
                                    "control_type": "shuffled_values",
                                    "control_seed": control_seed,
                                    "alpha": alpha_shuffle,
                                    **control_metrics,
                                }
                            )
                            q_random = rectangular_random_orthogonal(
                                source.basis.shape[1],
                                destination.basis.shape[1],
                                stable_seed(EXPERIMENT, "random_Q", source_task, destination_task, source_seed, destination_seed, split_seed, space_name, control_seed),
                            )
                            control_rows.append(
                                {
                                    **base_row,
                                    "control_type": "random_Q",
                                    "control_seed": control_seed,
                                    "alpha": 1.0,
                                    **transition_metrics(x_test, y_test, q_random, 1.0),
                                }
                            )
            for task_a in args.tasks:
                for task_b in args.tasks:
                    for seed_a in seed_selection[task_a]:
                        for seed_b in seed_selection[task_b]:
                            rsa_rows.append(
                                rsa_row(
                                    spaces[(task_a, seed_a, space_name)],
                                    spaces[(task_b, seed_b, space_name)],
                                    test_values,
                                    split_seed,
                                    args.rsa_permutations,
                                    stable_seed(EXPERIMENT, "rsa", task_a, task_b, seed_a, seed_b, split_seed, space_name),
                                )
                            )

        write_csv(transition_rows, args.output_dir / "readout_free_transition_procrustes.csv")
        write_csv(retrieval_rows, args.output_dir / "readout_free_value_retrieval.csv")
        write_csv(rsa_rows, args.output_dir / "readout_free_rsa.csv")
        write_csv(control_rows, args.output_dir / "readout_free_controls.csv")
        write_csv(space_diagnostics, args.output_dir / "readout_free_space_diagnostics.csv")
        print(f"CHECKPOINT_VALUE_HELDOUT_GEOMETRY saved split={split_seed} output_dir={args.output_dir}")

    transition_summary = summarize(
        transition_rows,
        ["heldout_transition_cosine_mean", "heldout_relative_error_mean"],
        ["source_task", "destination_task", "space"],
    )
    retrieval_summary = summarize(
        retrieval_rows,
        ["top1_value_retrieval", "top5_value_retrieval", "mean_absolute_value_error", "same_value_cosine_mean"],
        ["source_task", "destination_task", "space"],
    )
    rsa_summary = summarize(rsa_rows, ["spearman_rsa", "permutation_pvalue"], ["task_a", "task_b", "space"])
    control_summary = summarize(
        control_rows,
        ["heldout_transition_cosine_mean", "heldout_relative_error_mean"],
        ["source_task", "destination_task", "space", "control_type"],
    )
    save_json(
        {
            "experiment": EXPERIMENT,
            "description": (
                "Value-disjoint transition Procrustes, same-value retrieval, and no-fit RSA "
                "for original and readout-free final DAS coordinates."
            ),
            "tasks": args.tasks,
            "task_labels": TASK_LABELS,
            "seed_selection": seed_selection,
            "n_common_values": len(common_values),
            "common_values": common_values,
            "min_examples_per_value": args.min_examples_per_value,
            "per_task_common_value_counts": per_task_counts,
            "transition_summary": transition_summary,
            "retrieval_summary": retrieval_summary,
            "rsa_summary": rsa_summary,
            "control_summary": control_summary,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "summary.json",
    )
    if not args.skip_plots:
        plot_figures(args, transition_rows + control_rows, retrieval_rows, rsa_rows)

    print("\nValue-heldout readout-free geometry complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Transition rows: {len(transition_rows)}")
    print(f"Retrieval rows: {len(retrieval_rows)}")
    print(f"Control rows: {len(control_rows)}")
    print(f"RSA rows: {len(rsa_rows)}")


if __name__ == "__main__":
    main()
