"""One-example/value control for the repeat-vs-arithmetic causal-L gap.

Experiment 7A asks whether the repeat-vs-arithmetic gap is an artifact of
arithmetic centroids averaging many examples while repeat has one prompt per
value.  This script reuses saved arithmetic activations, saved repeat
activations, exact causal L bases, and the saved value splits.  It samples one
arithmetic example per value and recomputes the same held-out geometry metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.experiments.global_geometry import causal_subspace_geometry as causal_l
from src.experiments.autoregressive_transfer import repeat_geometry as repeat_l
from src.experiments.readout_latent_geometry import value_heldout_geometry as geom
from src.interventions.das import format_prompt
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import jsonable, save_json
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template


EXPERIMENT = "repeat_centroid_noise_control"
TASK_ORDER = geom.TASK_ORDER
TASK_LABELS = geom.TASK_LABELS
VALUE_MIN = 10
VALUE_MAX = 89


@dataclass
class Space:
    task: str
    task_label: str
    das_seed: int
    basis: torch.Tensor
    full_centroids: dict[int, torch.Tensor]
    repeat_centroids: dict[int, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--previous_geometry_dir", type=Path, default=Path("results/experiments/closing/causal_L_shared_geometry"))
    parser.add_argument("--repeat_geometry_dir", type=Path, default=Path("results/experiments/closing/repeat_vs_causal_L_geometry"))
    parser.add_argument("--digit_readout_basis_path", type=Path, default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43/digit_readout_basis.pt"))
    parser.add_argument("--activation_dir_text", type=Path, default=Path("outputs/activations/baseline/gemma4_12b_it/digits"))
    parser.add_argument("--activation_dir_image", type=Path, default=Path("outputs/activations/baseline_images/gemma4_12b_it/digits"))
    parser.add_argument("--repeat_transfer_dir", type=Path, default=Path("results/experiments/check_unembeeding/repeat_transfer"))
    parser.add_argument("--repeat_activation_path", type=Path, default=Path("results/experiments/closing/repeat_vs_causal_L_geometry/repeat_activations_layer43_resid_post.pt"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/repeat_centroid_noise_control"))
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_ORDER))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--value_split_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_replicates", type=int, default=200)
    parser.add_argument("--replicate_seed", type=int, default=1729)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--m_readout", type=int, default=9)
    parser.add_argument("--latent_dim", type=int, default=13)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--repeat_position", default="equals")
    parser.add_argument("--prompt_template", default="Output ONLY a number. Repeat this number: {n} =")
    parser.add_argument("--answer_separator", default=" ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--run_ar_check", action="store_true", default=True)
    parser.add_argument("--skip_ar_check", action="store_false", dest="run_ar_check")
    parser.add_argument("--nontrivial_ar_failures", type=int, default=1)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sanity_tolerance", type=float, default=1e-5)
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.tasks == list(TASK_ORDER):
        args.tasks = ["text:addition", "text:subtraction"]
    if args.seeds == [0, 1, 2]:
        args.seeds = [0]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]
    args.n_replicates = min(args.n_replicates, 3)
    args.bootstrap_samples = min(args.bootstrap_samples, 100)
    args.skip_plots = True


def output_exists(args: argparse.Namespace) -> bool:
    return (args.output_dir / "summary.json").exists() and not args.force


def normalize_tasks(tasks: list[str]) -> list[str]:
    return repeat_l.normalize_tasks(tasks)


def write_csv(rows: list[dict], path: Path) -> None:
    repeat_l.write_csv(rows, path)


def mean_metric(rows: list[dict], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]
    return None if not values else float(sum(values) / len(values))


def filtered_splits(args: argparse.Namespace) -> dict[str, dict[str, list[int]]]:
    all_splits = repeat_l.load_exact_value_splits(args)
    output = {}
    for split_seed, split in all_splits.items():
        train_values = [value for value in split["train_values"] if VALUE_MIN <= int(value) <= VALUE_MAX]
        test_values = [value for value in split["test_values"] if VALUE_MIN <= int(value) <= VALUE_MAX]
        if len(train_values) < 2 or len(test_values) < 2:
            raise ValueError(f"Filtered split {split_seed} is too small: train={len(train_values)} test={len(test_values)}")
        output[split_seed] = {"train_values": train_values, "test_values": test_values}
    return output


def selected_values(value_splits: dict[str, dict[str, list[int]]]) -> list[int]:
    values = set()
    for split in value_splits.values():
        values.update(split["train_values"])
        values.update(split["test_values"])
    return sorted(values)


def load_repeat_data_from_cache(args: argparse.Namespace, values: list[int]) -> repeat_l.RepeatData:
    hidden_by_value, labels, inventory = repeat_l.load_repeat_activation_cache(args.repeat_activation_path, args)
    missing = [value for value in values if value not in hidden_by_value]
    if missing:
        raise FileNotFoundError(
            f"Repeat activation cache {args.repeat_activation_path} is missing values {missing[:20]}. "
            "Run repeat_vs_causal_L_geometry once first so this experiment remains offline."
        )
    label_by_value = {int(row["result"]): row for row in labels if "result" in row}
    kept_labels = [label_by_value.get(value, {"result": value, "sample_id": value}) for value in values]
    hidden = torch.stack([hidden_by_value[value] for value in values]).float()
    print(f"Loaded repeat activation cache: {args.repeat_activation_path} values={len(values)}")
    return repeat_l.RepeatData(kept_labels, hidden, {value: 1 for value in values}, Path(inventory["path"]), {value: True for value in values})


def value_index_groups(data: geom.TaskData, values: list[int], target: str) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {value: [] for value in values}
    for index, label in enumerate(data.labels):
        value = int(label[target])
        if value in groups:
            groups[value].append(index)
    missing = [value for value, indices in groups.items() if not indices]
    if missing:
        raise ValueError(f"{data.task} has zero examples for selected values: {missing[:20]}")
    return groups


def sampling_inventory(data_by_task: dict[str, geom.TaskData], groups_by_task: dict[str, dict[int, list[int]]], values: list[int]) -> list[dict]:
    rows = []
    for task in TASK_ORDER:
        if task not in data_by_task:
            continue
        counts = [len(groups_by_task[task][value]) for value in values]
        print(
            f"{task}: examples/value over {VALUE_MIN}..{VALUE_MAX} "
            f"min/median/max = {min(counts)}/{statistics.median(counts)}/{max(counts)}"
        )
        for value in values:
            rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "value": value,
                    "n_examples": len(groups_by_task[task][value]),
                    "min_examples_over_range": min(counts),
                    "median_examples_over_range": statistics.median(counts),
                    "max_examples_over_range": max(counts),
                }
            )
    return rows


def build_spaces(
    args: argparse.Namespace,
    data_by_task: dict[str, geom.TaskData],
    repeat_data: repeat_l.RepeatData,
    values: list[int],
) -> tuple[list[Space], list[dict]]:
    spaces = []
    diagnostics = []
    value_set = set(values)
    for task in args.tasks:
        data = data_by_task[task]
        digit_basis = geom.load_digit_basis(args.digit_readout_basis_path, data.hidden.shape[1])
        for seed in args.seeds:
            _c_basis, l_basis, _ordered, diag = causal_l.construct_exact_CL(
                args,
                task=task,
                data=data,
                digit_basis=digit_basis,
                seed=seed,
            )
            full_centroids = geom.centroid_coordinates(data, l_basis, value_set, args.target)
            repeat_centroids = repeat_l.repeat_centroids_for_basis(repeat_data, l_basis, value_set)
            missing_repeat = sorted(value_set - set(repeat_centroids))
            if missing_repeat:
                raise ValueError(f"Repeat through {task} seed={seed} missing values: {missing_repeat[:20]}")
            diagnostics.append(
                {
                    **diag,
                    "space_type": "causal_L",
                    "value_min": VALUE_MIN,
                    "value_max": VALUE_MAX,
                    "n_values": len(values),
                }
            )
            spaces.append(Space(task, TASK_LABELS[task], seed, l_basis, full_centroids, repeat_centroids))
            print(f"L_SPACE task={task} seed={seed} rank={l_basis.shape[1]} digit_overlap={diag['digit_overlap']:.3e}")
    return spaces, diagnostics


def sample_indices_for_replicate(
    args: argparse.Namespace,
    groups_by_task: dict[str, dict[int, list[int]]],
    values: list[int],
    replicate: int,
) -> dict[str, dict[int, int]]:
    selected = {}
    for task in args.tasks:
        selected[task] = {}
        for value in values:
            candidates = groups_by_task[task][value]
            rng = random.Random(stable_seed(EXPERIMENT, args.replicate_seed, replicate, task, value))
            selected[task][value] = candidates[rng.randrange(len(candidates))]
    return selected


def one_example_centroids(data: geom.TaskData, basis: torch.Tensor, selected_indices: dict[int, int]) -> dict[int, torch.Tensor]:
    values = sorted(selected_indices)
    indices = torch.tensor([selected_indices[value] for value in values], dtype=torch.long)
    coords = data.hidden[indices] @ basis
    return {value: coord for value, coord in zip(values, coords)}


def procrustes_row(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int], test_values: list[int]) -> dict:
    metrics, _confusions = repeat_l.procrustes_metrics(source, destination, train_values, test_values)
    return metrics


def scale_row(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int], test_values: list[int]) -> dict:
    metrics, _confusions = repeat_l.scale_translation_metrics(source, destination, train_values, test_values)
    return metrics


def rsa_value(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], test_values: list[int]) -> float:
    values = sorted(test_values)
    return geom.spearman(geom.distance_vector(source, values), geom.distance_vector(destination, values))


def arithmetic_pairs(spaces: list[Space]) -> list[tuple[Space, Space]]:
    return [(a, b) for a in spaces for b in spaces if a.task != b.task]


def arithmetic_pairs_unique(spaces: list[Space]) -> list[tuple[Space, Space]]:
    order = {task: index for index, task in enumerate(TASK_ORDER)}
    return [(a, b) for a in spaces for b in spaces if order[a.task] < order[b.task]]


def reference_rows(
    args: argparse.Namespace,
    spaces: list[Space],
    value_splits: dict[str, dict[str, list[int]]],
    *,
    repeat_value_filter: set[int] | None = None,
    value_filter_name: str = "all_values",
) -> tuple[list[dict], list[dict]]:
    geometry_rows = []
    rsa_rows = []
    for split_seed_text, split in value_splits.items():
        train_values = [value for value in split["train_values"] if repeat_value_filter is None or value in repeat_value_filter]
        test_values = [value for value in split["test_values"] if repeat_value_filter is None or value in repeat_value_filter]
        if len(train_values) < 2 or len(test_values) < 2:
            continue
        split_seed = int(split_seed_text)
        for source, destination in arithmetic_pairs(spaces):
            metrics = procrustes_row(source.full_centroids, destination.full_centroids, train_values, test_values)
            geometry_rows.append(
                {
                    "reference_type": "full_centroid",
                    "comparison_family": "arithmetic_arithmetic",
                    "value_filter": value_filter_name,
                    "source_task": source.task,
                    "source_task_label": source.task_label,
                    "destination_task": destination.task,
                    "destination_task_label": destination.task_label,
                    "task_relation": f"{source.task_label}->{destination.task_label}",
                    "source_seed": source.das_seed,
                    "destination_seed": destination.das_seed,
                    "value_split_seed": split_seed,
                    "n_train_values": len(train_values),
                    "n_test_values": len(test_values),
                    **metrics,
                }
            )
        for destination in spaces:
            proc = procrustes_row(destination.repeat_centroids, destination.full_centroids, train_values, test_values)
            scale = scale_row(destination.repeat_centroids, destination.full_centroids, train_values, test_values)
            base = {
                "reference_type": "full_centroid",
                "comparison_family": "repeat_arithmetic",
                "value_filter": value_filter_name,
                "source_task": "repeat:number",
                "source_task_label": "Repeat",
                "destination_task": destination.task,
                "destination_task_label": destination.task_label,
                "task_relation": f"Repeat->{destination.task_label}",
                "source_seed": None,
                "destination_seed": destination.das_seed,
                "value_split_seed": split_seed,
                "n_train_values": len(train_values),
                "n_test_values": len(test_values),
            }
            geometry_rows.append({**base, "alignment": "procrustes", **proc})
            geometry_rows.append({**base, "alignment": "scale_translation", **scale})
        for source, destination in arithmetic_pairs_unique(spaces):
            rsa_rows.append(
                {
                    "reference_type": "full_centroid",
                    "comparison_family": "arithmetic_arithmetic",
                    "value_filter": value_filter_name,
                    "task_a": source.task,
                    "task_a_label": source.task_label,
                    "task_b": destination.task,
                    "task_b_label": destination.task_label,
                    "task_relation": f"{source.task_label}-{destination.task_label}",
                    "seed_a": source.das_seed,
                    "seed_b": destination.das_seed,
                    "value_split_seed": split_seed,
                    "n_test_values": len(test_values),
                    "spearman_rsa": rsa_value(source.full_centroids, destination.full_centroids, test_values),
                }
            )
        for destination in spaces:
            rsa_rows.append(
                {
                    "reference_type": "full_centroid",
                    "comparison_family": "repeat_arithmetic",
                    "value_filter": value_filter_name,
                    "task_a": "repeat:number",
                    "task_a_label": "Repeat",
                    "task_b": destination.task,
                    "task_b_label": destination.task_label,
                    "task_relation": f"Repeat-{destination.task_label}",
                    "seed_a": None,
                    "seed_b": destination.das_seed,
                    "value_split_seed": split_seed,
                    "n_test_values": len(test_values),
                    "spearman_rsa": rsa_value(destination.repeat_centroids, destination.full_centroids, test_values),
                }
            )
    return geometry_rows, rsa_rows


def one_replicate_rows(
    args: argparse.Namespace,
    replicate: int,
    spaces: list[Space],
    data_by_task: dict[str, geom.TaskData],
    selected_indices: dict[str, dict[int, int]],
    value_splits: dict[str, dict[str, list[int]]],
    *,
    value_filter: set[int] | None = None,
    value_filter_name: str = "all_values",
) -> tuple[list[dict], list[dict]]:
    sampled_by_space = {
        (space.task, space.das_seed): one_example_centroids(data_by_task[space.task], space.basis, selected_indices[space.task])
        for space in spaces
    }
    geometry_rows = []
    rsa_rows = []
    for split_seed_text, split in value_splits.items():
        split_seed = int(split_seed_text)
        train_values = [value for value in split["train_values"] if value_filter is None or value in value_filter]
        test_values = [value for value in split["test_values"] if value_filter is None or value in value_filter]
        if len(train_values) < 2 or len(test_values) < 2:
            continue
        for source, destination in arithmetic_pairs(spaces):
            source_centroids = sampled_by_space[(source.task, source.das_seed)]
            destination_centroids = sampled_by_space[(destination.task, destination.das_seed)]
            metrics = procrustes_row(source_centroids, destination_centroids, train_values, test_values)
            geometry_rows.append(
                {
                    "replicate": replicate,
                    "comparison_family": "arithmetic_arithmetic",
                    "value_filter": value_filter_name,
                    "alignment": "procrustes",
                    "source_task": source.task,
                    "source_task_label": source.task_label,
                    "destination_task": destination.task,
                    "destination_task_label": destination.task_label,
                    "task_relation": f"{source.task_label}->{destination.task_label}",
                    "source_seed": source.das_seed,
                    "destination_seed": destination.das_seed,
                    "value_split_seed": split_seed,
                    "n_train_values": len(train_values),
                    "n_test_values": len(test_values),
                    **metrics,
                }
            )
        for destination in spaces:
            destination_centroids = sampled_by_space[(destination.task, destination.das_seed)]
            proc = procrustes_row(destination.repeat_centroids, destination_centroids, train_values, test_values)
            scale = scale_row(destination.repeat_centroids, destination_centroids, train_values, test_values)
            base = {
                "replicate": replicate,
                "comparison_family": "repeat_arithmetic",
                "value_filter": value_filter_name,
                "source_task": "repeat:number",
                "source_task_label": "Repeat",
                "destination_task": destination.task,
                "destination_task_label": destination.task_label,
                "task_relation": f"Repeat->{destination.task_label}",
                "source_seed": None,
                "destination_seed": destination.das_seed,
                "value_split_seed": split_seed,
                "n_train_values": len(train_values),
                "n_test_values": len(test_values),
            }
            geometry_rows.append({**base, "alignment": "procrustes", **proc})
            geometry_rows.append({**base, "alignment": "scale_translation", **scale})
        for source, destination in arithmetic_pairs_unique(spaces):
            source_centroids = sampled_by_space[(source.task, source.das_seed)]
            destination_centroids = sampled_by_space[(destination.task, destination.das_seed)]
            rsa_rows.append(
                {
                    "replicate": replicate,
                    "comparison_family": "arithmetic_arithmetic",
                    "value_filter": value_filter_name,
                    "task_a": source.task,
                    "task_a_label": source.task_label,
                    "task_b": destination.task,
                    "task_b_label": destination.task_label,
                    "task_relation": f"{source.task_label}-{destination.task_label}",
                    "seed_a": source.das_seed,
                    "seed_b": destination.das_seed,
                    "value_split_seed": split_seed,
                    "n_test_values": len(test_values),
                    "spearman_rsa": rsa_value(source_centroids, destination_centroids, test_values),
                }
            )
        for destination in spaces:
            destination_centroids = sampled_by_space[(destination.task, destination.das_seed)]
            rsa_rows.append(
                {
                    "replicate": replicate,
                    "comparison_family": "repeat_arithmetic",
                    "value_filter": value_filter_name,
                    "task_a": "repeat:number",
                    "task_a_label": "Repeat",
                    "task_b": destination.task,
                    "task_b_label": destination.task_label,
                    "task_relation": f"Repeat-{destination.task_label}",
                    "seed_a": None,
                    "seed_b": destination.das_seed,
                    "value_split_seed": split_seed,
                    "n_test_values": len(test_values),
                    "spearman_rsa": rsa_value(destination.repeat_centroids, destination_centroids, test_values),
                }
            )
    return geometry_rows, rsa_rows


def aggregate_replicates(geometry_rows: list[dict], rsa_rows: list[dict], value_filter: str = "all_values") -> list[dict]:
    replicates = sorted({int(row["replicate"]) for row in geometry_rows if row.get("value_filter", "all_values") == value_filter})
    rows = []
    for replicate in replicates:
        arith_geom = [
            row for row in geometry_rows
            if int(row["replicate"]) == replicate
            and row.get("value_filter", "all_values") == value_filter
            and row["comparison_family"] == "arithmetic_arithmetic"
            and row.get("alignment") == "procrustes"
        ]
        repeat_geom = [
            row for row in geometry_rows
            if int(row["replicate"]) == replicate
            and row.get("value_filter", "all_values") == value_filter
            and row["comparison_family"] == "repeat_arithmetic"
            and row.get("alignment") == "procrustes"
        ]
        arith_rsa = [
            row for row in rsa_rows
            if int(row["replicate"]) == replicate
            and row.get("value_filter", "all_values") == value_filter
            and row["comparison_family"] == "arithmetic_arithmetic"
        ]
        repeat_rsa = [
            row for row in rsa_rows
            if int(row["replicate"]) == replicate
            and row.get("value_filter", "all_values") == value_filter
            and row["comparison_family"] == "repeat_arithmetic"
        ]
        rows.append(
            {
                "replicate": replicate,
                "value_filter": value_filter,
                "arithmetic_arithmetic_transition": mean_metric(arith_geom, "heldout_transition_cosine"),
                "repeat_arithmetic_transition": mean_metric(repeat_geom, "heldout_transition_cosine"),
                "gap_transition": (mean_metric(arith_geom, "heldout_transition_cosine") or 0.0) - (mean_metric(repeat_geom, "heldout_transition_cosine") or 0.0),
                "arithmetic_arithmetic_top1": mean_metric(arith_geom, "top1"),
                "repeat_arithmetic_top1": mean_metric(repeat_geom, "top1"),
                "gap_top1": (mean_metric(arith_geom, "top1") or 0.0) - (mean_metric(repeat_geom, "top1") or 0.0),
                "arithmetic_arithmetic_RSA": mean_metric(arith_rsa, "spearman_rsa"),
                "repeat_arithmetic_RSA": mean_metric(repeat_rsa, "spearman_rsa"),
                "gap_RSA": (mean_metric(arith_rsa, "spearman_rsa") or 0.0) - (mean_metric(repeat_rsa, "spearman_rsa") or 0.0),
            }
        )
    return rows


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(samples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return float(lo), float(hi)


def gap_summary_rows(args: argparse.Namespace, replicate_summary: list[dict], reference_geometry: list[dict], reference_rsa: list[dict]) -> list[dict]:
    rows = []
    reference_arith = [row for row in reference_geometry if row["comparison_family"] == "arithmetic_arithmetic" and row.get("value_filter") == "all_values"]
    reference_repeat = [row for row in reference_geometry if row["comparison_family"] == "repeat_arithmetic" and row.get("alignment") == "procrustes" and row.get("value_filter") == "all_values"]
    reference_arith_rsa = [row for row in reference_rsa if row["comparison_family"] == "arithmetic_arithmetic" and row.get("value_filter") == "all_values"]
    reference_repeat_rsa = [row for row in reference_rsa if row["comparison_family"] == "repeat_arithmetic" and row.get("value_filter") == "all_values"]
    full = {
        "transition": (mean_metric(reference_arith, "heldout_transition_cosine") or 0.0) - (mean_metric(reference_repeat, "heldout_transition_cosine") or 0.0),
        "top1": (mean_metric(reference_arith, "top1") or 0.0) - (mean_metric(reference_repeat, "top1") or 0.0),
        "RSA": (mean_metric(reference_arith_rsa, "spearman_rsa") or 0.0) - (mean_metric(reference_repeat_rsa, "spearman_rsa") or 0.0),
    }
    metric_specs = [
        ("transition", "gap_transition", "arithmetic_arithmetic_transition", "repeat_arithmetic_transition", "heldout_transition_cosine"),
        ("top1", "gap_top1", "arithmetic_arithmetic_top1", "repeat_arithmetic_top1", "top1"),
        ("RSA", "gap_RSA", "arithmetic_arithmetic_RSA", "repeat_arithmetic_RSA", "spearman_rsa"),
    ]
    for label, gap_key, ar_key, repeat_key, _metric in metric_specs:
        gap_values = [float(row[gap_key]) for row in replicate_summary if row.get(gap_key) is not None]
        ar_values = [float(row[ar_key]) for row in replicate_summary if row.get(ar_key) is not None]
        rep_values = [float(row[repeat_key]) for row in replicate_summary if row.get(repeat_key) is not None]
        lo, hi = bootstrap_ci(gap_values, args.bootstrap_samples, stable_seed(EXPERIMENT, "bootstrap", label, args.replicate_seed))
        rows.append(
            {
                "metric": label,
                "one_example_gap_mean": None if not gap_values else float(sum(gap_values) / len(gap_values)),
                "one_example_gap_median": None if not gap_values else float(statistics.median(gap_values)),
                "one_example_gap_std": None if len(gap_values) < 2 else float(statistics.stdev(gap_values)),
                "one_example_gap_bootstrap_ci_low": lo,
                "one_example_gap_bootstrap_ci_high": hi,
                "full_centroid_gap": full[label],
                "one_example_arithmetic_arithmetic_mean": None if not ar_values else float(sum(ar_values) / len(ar_values)),
                "one_example_repeat_arithmetic_mean": None if not rep_values else float(sum(rep_values) / len(rep_values)),
                "centroiding_gain_arithmetic_arithmetic": None if not ar_values else reference_mean_for_metric(reference_geometry, reference_rsa, "arithmetic_arithmetic", label, "all_values") - float(sum(ar_values) / len(ar_values)),
                "centroiding_gain_repeat_arithmetic": None if not rep_values else reference_mean_for_metric(reference_geometry, reference_rsa, "repeat_arithmetic", label, "all_values") - float(sum(rep_values) / len(rep_values)),
                "n_replicates": len(gap_values),
            }
        )
    return rows


def reference_mean_for_metric(reference_geometry: list[dict], reference_rsa: list[dict], family: str, metric: str, value_filter: str = "all_values") -> float:
    if metric == "transition":
        rows = [row for row in reference_geometry if row["comparison_family"] == family and row.get("alignment", "procrustes") == "procrustes" and row.get("value_filter") == value_filter]
        return mean_metric(rows, "heldout_transition_cosine") or 0.0
    if metric == "top1":
        rows = [row for row in reference_geometry if row["comparison_family"] == family and row.get("alignment", "procrustes") == "procrustes" and row.get("value_filter") == value_filter]
        return mean_metric(rows, "top1") or 0.0
    rows = [row for row in reference_rsa if row["comparison_family"] == family and row.get("value_filter") == value_filter]
    return mean_metric(rows, "spearman_rsa") or 0.0


@torch.no_grad()
def repeat_autoregressive_accuracy(args: argparse.Namespace) -> list[dict]:
    samples, _inventory = repeat_l.find_repeat_dataset(argparse.Namespace(**vars(args), min_value=0, max_value=99))
    model, tokenizer, _blocks, _resolved_model = repeat_l.load_model_for_repeat(args)
    use_chat_template = bool(args.use_chat_template) or uses_chat_template(args.model)
    rows = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for sample in tqdm(samples, desc="minimal repeat AR accuracy"):
            value = repeat_l.result_value(sample)
            expected = str(value)
            prompt = format_prompt(tokenizer, sample, use_chat_template)
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            answer_token_count = max(1, len(tokenizer(expected, add_special_tokens=False)["input_ids"]))
            generated_ids = []
            for _ in range(answer_token_count):
                outputs = model(input_ids=input_ids, use_cache=False)
                next_id = outputs.logits[0, input_ids.shape[1] - 1].argmax().reshape(1, 1)
                generated_ids.append(int(next_id.item()))
                input_ids = torch.cat([input_ids, next_id.to(model.device)], dim=1)
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            match = re.match(r"-?\d+", generated)
            parsed = match.group() if match else None
            expected_digits = expected
            parsed_digits = parsed or ""
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "value": value,
                    "prompt": sample["expr"],
                    "expected": expected,
                    "generated": generated,
                    "parsed": parsed,
                    "answer_token_count": answer_token_count,
                    "correct_full_answer": parsed == expected,
                    "first_digit_correct": bool(parsed_digits) and parsed_digits[0] == expected_digits[0],
                    "second_digit_correct": None if len(expected_digits) < 2 else (len(parsed_digits) >= 2 and parsed_digits[1] == expected_digits[1]),
                }
            )
    finally:
        tokenizer.padding_side = old_padding_side
    return rows


def summarize_ar_accuracy(rows: list[dict]) -> dict:
    if not rows:
        return {"available": False}
    full = [bool(row["correct_full_answer"]) for row in rows]
    first = [bool(row["first_digit_correct"]) for row in rows]
    second = [bool(row["second_digit_correct"]) for row in rows if row["second_digit_correct"] is not None]
    failures = [int(row["value"]) for row in rows if not bool(row["correct_full_answer"])]
    return {
        "available": True,
        "exact_full_answer_ar_accuracy": sum(full) / len(full),
        "first_digit_accuracy": sum(first) / len(first),
        "second_digit_accuracy_two_digit": None if not second else sum(second) / len(second),
        "n_failures": len(failures),
        "failed_values": failures,
    }


def plot_figures(args: argparse.Namespace, replicate_summary: list[dict], gap_rows: list[dict], reference_geometry: list[dict], reference_rsa: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    specs = [
        (axes[0], "transition", "arithmetic_arithmetic_transition", "repeat_arithmetic_transition", "heldout transition cosine"),
        (axes[1], "RSA", "arithmetic_arithmetic_RSA", "repeat_arithmetic_RSA", "RSA"),
    ]
    for ax, metric, ar_key, repeat_key, title in specs:
        ar = [float(row[ar_key]) for row in replicate_summary if row.get(ar_key) is not None]
        rep = [float(row[repeat_key]) for row in replicate_summary if row.get(repeat_key) is not None]
        ax.hist(ar, bins=30, alpha=0.6, label="arith <-> arith one-example")
        ax.hist(rep, bins=30, alpha=0.6, label="repeat <-> arith one-example")
        if metric == "transition":
            ar_ref = reference_mean_for_metric(reference_geometry, reference_rsa, "arithmetic_arithmetic", "transition")
            rep_ref = reference_mean_for_metric(reference_geometry, reference_rsa, "repeat_arithmetic", "transition")
        else:
            ar_ref = reference_mean_for_metric(reference_geometry, reference_rsa, "arithmetic_arithmetic", "RSA")
            rep_ref = reference_mean_for_metric(reference_geometry, reference_rsa, "repeat_arithmetic", "RSA")
        ax.axvline(ar_ref, color="black", linestyle="--", linewidth=1.0, label="arith full centroid")
        ax.axvline(rep_ref, color="tab:red", linestyle="--", linewidth=1.0, label="repeat full centroid")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure1_one_example_distributions.png", dpi=240)
    fig.savefig(figure_dir / "figure1_one_example_distributions.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = [row["metric"] for row in gap_rows]
    x = list(range(len(labels)))
    full = [row["full_centroid_gap"] for row in gap_rows]
    one = [row["one_example_gap_mean"] for row in gap_rows]
    lo = [row["one_example_gap_mean"] - row["one_example_gap_bootstrap_ci_low"] for row in gap_rows]
    hi = [row["one_example_gap_bootstrap_ci_high"] - row["one_example_gap_mean"] for row in gap_rows]
    width = 0.35
    ax.bar([i - width / 2 for i in x], full, width, label="full centroid gap")
    ax.bar([i + width / 2 for i in x], one, width, yerr=[lo, hi], capsize=3, label="one-example gap")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("arith-arith minus repeat-arith")
    ax.set_title("Centroid gap vs matched-noise gap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure2_gap_comparison.png", dpi=240)
    fig.savefig(figure_dir / "figure2_gap_comparison.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = normalize_tasks(args.tasks)
    if output_exists(args) and not args.artifact_check_only:
        raise FileExistsError(f"{args.output_dir} already has summary.json; pass --force to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Repeat centroid-noise control")
    print(f"  tasks={args.tasks}")
    print(f"  seeds={args.seeds}")
    print(f"  replicates={args.n_replicates}")
    print(f"  value range={VALUE_MIN}..{VALUE_MAX}")

    repeat_l.apply_local_activation_fallbacks(args)
    value_splits = filtered_splits(args)
    values = selected_values(value_splits)
    if values != list(range(VALUE_MIN, VALUE_MAX + 1)):
        raise ValueError(f"Filtered value splits do not cover exactly {VALUE_MIN}..{VALUE_MAX}: {values[:5]}..{values[-5:]}")
    save_json(value_splits, args.output_dir / "filtered_value_splits.json")

    data_by_task = repeat_l.load_arithmetic_data(args, args.tasks)
    groups_by_task = {task: value_index_groups(data_by_task[task], values, args.target) for task in args.tasks}
    inventory_rows = sampling_inventory(data_by_task, groups_by_task, values)
    write_csv(inventory_rows, args.output_dir / "arithmetic_sampling_inventory.csv")

    repeat_data = load_repeat_data_from_cache(args, values)
    spaces, diagnostics = build_spaces(args, data_by_task, repeat_data, values)
    write_csv(diagnostics, args.output_dir / "L_space_diagnostics.csv")

    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "seeds": args.seeds,
                "n_values": len(values),
                "filtered_value_splits": value_splits,
                "sampling_inventory_rows": len(inventory_rows),
                "L_space_diagnostics_rows": len(diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("ARTIFACT_CHECK_ONLY complete")
        return

    ar_rows = []
    ar_summary = {"available": False}
    ar_correct_filter = None
    if args.run_ar_check:
        ar_rows = repeat_autoregressive_accuracy(args)
        write_csv(ar_rows, args.output_dir / "repeat_autoregressive_accuracy.csv")
        ar_summary = summarize_ar_accuracy(ar_rows)
        if ar_summary["n_failures"] >= args.nontrivial_ar_failures:
            ar_correct_filter = {int(row["value"]) for row in ar_rows if row["correct_full_answer"] and VALUE_MIN <= int(row["value"]) <= VALUE_MAX}

    reference_geometry, reference_rsa = reference_rows(args, spaces, value_splits)
    if ar_correct_filter is not None:
        extra_geometry, extra_rsa = reference_rows(
            args,
            spaces,
            value_splits,
            repeat_value_filter=ar_correct_filter,
            value_filter_name="AR_correct_only",
        )
        reference_geometry.extend(extra_geometry)
        reference_rsa.extend(extra_rsa)
    write_csv(reference_geometry + reference_rsa, args.output_dir / "centroid_reference.csv")

    all_geometry_rows = []
    all_rsa_rows = []
    for replicate in tqdm(range(args.n_replicates), desc="one-example replicates"):
        selected = sample_indices_for_replicate(args, groups_by_task, values, replicate)
        geometry_rows, rsa_rows = one_replicate_rows(args, replicate, spaces, data_by_task, selected, value_splits)
        all_geometry_rows.extend(geometry_rows)
        all_rsa_rows.extend(rsa_rows)
        if ar_correct_filter is not None:
            filtered_geometry, filtered_rsa = one_replicate_rows(
                args,
                replicate,
                spaces,
                data_by_task,
                selected,
                value_splits,
                value_filter=ar_correct_filter,
                value_filter_name="AR_correct_only",
            )
            all_geometry_rows.extend(filtered_geometry)
            all_rsa_rows.extend(filtered_rsa)
    write_csv(all_geometry_rows, args.output_dir / "one_example_geometry.csv")
    write_csv(all_rsa_rows, args.output_dir / "one_example_rsa.csv")

    replicate_summary = aggregate_replicates(all_geometry_rows, all_rsa_rows, "all_values")
    gap_rows = gap_summary_rows(args, replicate_summary, reference_geometry, reference_rsa)
    write_csv(gap_rows, args.output_dir / "gap_summary.csv")

    summary = {
        "experiment": EXPERIMENT,
        "description": "One arithmetic example per value control for repeat-vs-arithmetic causal-L geometry.",
        "tasks": args.tasks,
        "task_labels": TASK_LABELS,
        "seeds": args.seeds,
        "value_min": VALUE_MIN,
        "value_max": VALUE_MAX,
        "n_values": len(values),
        "n_replicates": args.n_replicates,
        "filtered_value_splits": value_splits,
        "repeat_autoregressive_accuracy": ar_summary,
        "reference_means": {
            "arithmetic_arithmetic_transition": reference_mean_for_metric(reference_geometry, reference_rsa, "arithmetic_arithmetic", "transition"),
            "repeat_arithmetic_transition": reference_mean_for_metric(reference_geometry, reference_rsa, "repeat_arithmetic", "transition"),
            "arithmetic_arithmetic_top1": reference_mean_for_metric(reference_geometry, reference_rsa, "arithmetic_arithmetic", "top1"),
            "repeat_arithmetic_top1": reference_mean_for_metric(reference_geometry, reference_rsa, "repeat_arithmetic", "top1"),
            "arithmetic_arithmetic_RSA": reference_mean_for_metric(reference_geometry, reference_rsa, "arithmetic_arithmetic", "RSA"),
            "repeat_arithmetic_RSA": reference_mean_for_metric(reference_geometry, reference_rsa, "repeat_arithmetic", "RSA"),
        },
        "gap_summary": gap_rows,
        "rows": {
            "centroid_reference": len(reference_geometry) + len(reference_rsa),
            "one_example_geometry": len(all_geometry_rows),
            "one_example_rsa": len(all_rsa_rows),
            "arithmetic_sampling_inventory": len(inventory_rows),
            "repeat_autoregressive_accuracy": len(ar_rows),
        },
        "config": jsonable(vars(args)),
    }
    save_json(summary, args.output_dir / "summary.json")
    if not args.skip_plots:
        plot_figures(args, replicate_summary, gap_rows, reference_geometry, reference_rsa)

    print("\nRepeat centroid-noise control complete")
    print(f"Output directory: {args.output_dir}")
    print(f"One-example geometry rows: {len(all_geometry_rows)}")
    print(f"One-example RSA rows: {len(all_rsa_rows)}")


if __name__ == "__main__":
    main()
