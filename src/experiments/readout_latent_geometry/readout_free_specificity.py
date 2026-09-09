"""Compare readout-free DAS geometry against PCA and random subspace controls.

This is an offline specificity control for value-heldout numerical geometry.
It reuses the completed value_heldout_readout_free_geometry implementation for
activation loading, correct-example filtering, centroids, Procrustes,
retrieval, and RSA. The only scientific variable changed here is the
readout-free observation subspace.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from src.common import load_jsonl
from src.interventions.das import (
    build_unique_pairs,
    sample_id as das_sample_id,
    split_samples,
)
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    jsonable,
    parse_task,
    save_json,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed
from src.experiments.readout_latent_geometry import value_heldout_geometry as geom


EXPERIMENT = "readout_free_space_specificity"
PRIMARY_SPACES = [
    "das_readout_free",
    "pca_leading_readout_free",
    "random_pca_readout_free",
    "random_residual_readout_free",
]
RANDOM_SPACES = {"random_pca_readout_free", "random_residual_readout_free"}


@dataclass
class SpaceItem:
    task: str
    space_type: str
    space_seed: int
    basis: torch.Tensor
    centroids: dict[int, torch.Tensor]
    result_centroid_variance_fraction: float
    diagnostics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--readout_free_audit_root",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43"),
    )
    parser.add_argument(
        "--previous_geometry_dir",
        type=Path,
        default=Path("results/experiments/closing/value_heldout_readout_free_geometry"),
        help="Completed value-heldout geometry directory; value_splits.json is loaded from here.",
    )
    parser.add_argument(
        "--digit_readout_basis_path",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43/digit_readout_basis.pt"),
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
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/readout_free_space_specificity"))
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=geom.DEFAULT_TASKS)
    parser.add_argument("--space_types", nargs="+", default=PRIMARY_SPACES)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="DAS seeds.")
    parser.add_argument("--control_seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--value_split_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rsa_permutations", type=int, default=1000)
    parser.add_argument("--readout_overlap_tolerance", type=float, default=1e-6)
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_das_regression_check", action="store_true")
    parser.add_argument("--das_regression_min_transition", type=float, default=0.75)
    parser.add_argument("--das_regression_min_top1", type=float, default=0.75)
    parser.add_argument("--das_regression_min_rsa", type=float, default=0.20)
    return parser.parse_args()


def normalize_tasks(tasks: list[str]) -> list[str]:
    if any(task.lower() == "all" for task in tasks):
        return list(geom.DEFAULT_TASKS)
    return [task_key(*parse_task(task)) for task in tasks]


def normalize_spaces(spaces: list[str]) -> list[str]:
    if any(space.lower() == "all" for space in spaces):
        return list(PRIMARY_SPACES)
    unknown = sorted(set(spaces) - set(PRIMARY_SPACES) - {"digit_readout"})
    if unknown:
        raise ValueError(f"Unknown --space_types: {unknown}")
    return spaces


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.tasks = ["text:addition", "image:addition"] if args.tasks == geom.DEFAULT_TASKS else args.tasks
    if args.space_types == PRIMARY_SPACES:
        args.space_types = ["das_readout_free", "pca_leading_readout_free", "random_pca_readout_free"]
    if args.control_seeds == list(range(20)):
        args.control_seeds = [0, 1]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]
    args.rsa_permutations = min(args.rsa_permutations, 100)


def load_exact_value_splits(args: argparse.Namespace) -> dict[str, dict[str, list[int]]]:
    path = args.previous_geometry_dir / "value_splits.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing exact previous value splits: {path}. "
            "Run the completed value_heldout_readout_free_geometry experiment first; "
            "this control experiment must not create new splits."
        )
    with path.open("r", encoding="utf-8") as handle:
        all_splits = json.load(handle)
    selected = {}
    for seed in args.value_split_seeds:
        key = str(seed)
        if key not in all_splits:
            raise ValueError(f"{path} has no value split seed {seed}; available={sorted(all_splits)}")
        selected[key] = {
            "train_values": [int(x) for x in all_splits[key]["train_values"]],
            "test_values": [int(x) for x in all_splits[key]["test_values"]],
        }
    return selected


def all_split_values(value_splits: dict[str, dict[str, list[int]]]) -> list[int]:
    values = set()
    for split in value_splits.values():
        values.update(split["train_values"])
        values.update(split["test_values"])
    return sorted(values)


def unique_samples_from_pairs(pairs: list[dict]) -> list[dict]:
    samples = []
    seen = set()
    for pair in pairs:
        for sample in (pair["base"], pair["source"]):
            key = das_sample_id(sample)
            if key not in seen:
                seen.add(key)
                samples.append(sample)
    return samples


def label_keys(row: dict, fallback: int | None = None) -> list:
    keys = []
    for key in ("sample_id", "id", "expr", "image_text"):
        if key in row and row[key] is not None:
            keys.append(row[key])
            keys.append(str(row[key]))
    if "a" in row and "b" in row:
        try:
            pair_key = (int(row["a"]), int(row["b"]))
            keys.append(pair_key)
            keys.append(str(pair_key))
        except Exception:
            pass
    if fallback is not None:
        keys.append(fallback)
        keys.append(str(fallback))
    return keys


def sample_lookup_keys(sample: dict) -> list:
    keys = label_keys(sample)
    try:
        sid = das_sample_id(sample)
        keys.extend([sid, str(sid)])
    except Exception:
        pass
    return keys


def task_hidden_lookup(data: geom.TaskData) -> dict:
    lookup = {}
    for index, (label, hidden) in enumerate(zip(data.labels, data.hidden)):
        for key in label_keys(label, index):
            lookup.setdefault(key, hidden)
    return lookup


def pca_training_features(args: argparse.Namespace, task: str, data: geom.TaskData, row: dict) -> tuple[torch.Tensor, dict]:
    source_path = subspace_path(args, data.modality, data.operation, args.seeds[0])
    _basis, payload = geom.load_basis_file(source_path, data.hidden.shape[1], args.k)
    config = payload.get("config", {})
    train_fraction = float(config.get("train_fraction", 0.7))
    validation_fraction = float(config.get("validation_fraction", 0.15))
    max_train_pairs = int(config.get("max_train_pairs", row.get("pair_statistics", {}).get("n_train_pairs", 4096)))
    pca_max_samples = int(config.get("pca_max_samples", 1024))

    samples = load_jsonl(Path(row["data_path"]))
    train_samples, _validation_samples, _test_samples = split_samples(
        samples, train_fraction, validation_fraction, args.split_seed
    )
    train_pairs, train_stats = build_unique_pairs(train_samples, args.target, args.split_seed, max_train_pairs)
    pca_samples = unique_samples_from_pairs(train_pairs)
    selected = list(pca_samples[:pca_max_samples]) if pca_max_samples else list(pca_samples)
    lookup = task_hidden_lookup(data)
    rows = []
    missing = []
    for sample in selected:
        vector = None
        for key in sample_lookup_keys(sample):
            if key in lookup:
                vector = lookup[key]
                break
        if vector is None:
            missing.append(sample.get("sample_id", sample.get("expr", sample.get("image_text"))))
        else:
            rows.append(vector)
    if missing:
        raise ValueError(
            f"Could not find saved activation rows for {len(missing)} PCA training samples in {task}; "
            f"first missing={missing[:5]}"
        )
    if len(rows) < args.k + 1:
        raise ValueError(f"Need at least k+1 PCA rows for {task}; got {len(rows)}.")
    return torch.stack(rows).float(), {
        "pca_source": "reconstructed_from_saved_train_activations",
        "pca_train_fraction": train_fraction,
        "pca_validation_fraction": validation_fraction,
        "pca_max_samples": pca_max_samples,
        "pca_selected_samples": len(rows),
        "pca_train_pair_stats": train_stats,
        "pca_source_subspace_path": str(source_path),
    }


def pca_space_with_at_least_k(features: torch.Tensor, k: int, variance_threshold: float) -> tuple[torch.Tensor, dict]:
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    max_components = min(centered.shape[0] - 1, centered.shape[1])
    if max_components < k:
        raise ValueError(f"Need k={k} PCA components, but only {max_components} exist.")
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh.T[:, :max_components]
    variance = singular_values[:max_components].square()
    total_variance = centered.square().sum().clamp_min(1e-12)
    cumulative = variance.cumsum(0) / total_variance
    threshold_k = int(torch.searchsorted(cumulative, torch.tensor(variance_threshold, dtype=cumulative.dtype)).item()) + 1
    n_components = max(k, min(threshold_k, components.shape[1]))
    return components[:, :n_components].float(), {
        "pca_variance_threshold": variance_threshold,
        "pca_retained_rank": int(n_components),
        "pca_explained_variance": float(cumulative[n_components - 1]),
    }


def project_out_readout(matrix: torch.Tensor, digit_basis: torch.Tensor, rank: int | None, name: str) -> tuple[torch.Tensor, dict]:
    original = geom.orthonormal_basis(matrix, f"{name} pre-readout", rank=rank)
    projected = original - digit_basis @ (digit_basis.T @ original)
    readout_free = geom.orthonormal_basis(projected, f"{name} readout-free", rank=rank)
    return readout_free, {
        "rank_before_readout_removal": int(original.shape[1]),
        "rank_after_readout_removal": int(readout_free.shape[1]),
        "readout_overlap_before": float((digit_basis.T @ original).square().sum()),
        "readout_overlap": float((digit_basis.T @ readout_free).square().sum()),
    }


def random_residual_basis(hidden_size: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden_size, k, generator=generator), mode="reduced")
    return q.float()


def random_pca_basis(pca_space: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    coords = torch.randn(pca_space.shape[1], k, generator=generator)
    q, _ = torch.linalg.qr(pca_space @ coords, mode="reduced")
    return q[:, :k].float()


def basis_diagnostics(
    *,
    task: str,
    space_type: str,
    space_seed: int,
    basis: torch.Tensor,
    digit_basis: torch.Tensor,
    extra: dict,
) -> dict:
    eye = torch.eye(basis.shape[1], dtype=basis.dtype)
    return {
        "task": task,
        "task_label": geom.TASK_LABELS.get(task, task),
        "space_type": space_type,
        "space_seed": space_seed,
        "rank": int(basis.shape[1]),
        "orthonormality_error": float((basis.T @ basis - eye).norm()),
        "readout_overlap": float((digit_basis.T @ basis).square().sum()),
        **extra,
    }


def result_centroid_variance_fraction(data: geom.TaskData, basis: torch.Tensor, values: set[int], target: str) -> float:
    coords = data.hidden @ basis
    kept_coords = []
    kept_values = []
    for label, coord in zip(data.labels, coords):
        value = int(label[target])
        if value in values:
            kept_values.append(value)
            kept_coords.append(coord)
    matrix = torch.stack(kept_coords)
    grand_mean = matrix.mean(dim=0)
    grouped: dict[int, list[torch.Tensor]] = {}
    for value, coord in zip(kept_values, kept_coords):
        grouped.setdefault(value, []).append(coord)
    centroids = {value: torch.stack(parts).mean(dim=0) for value, parts in grouped.items()}
    residual = torch.stack([coord - centroids[value] for value, coord in zip(kept_values, kept_coords)])
    total = matrix - grand_mean
    return float(1.0 - residual.square().sum() / total.square().sum().clamp_min(1e-12))


def load_das_readout_free_basis(args: argparse.Namespace, task: str, data: geom.TaskData, seed: int) -> tuple[torch.Tensor, dict]:
    run = geom.run_dir(args.readout_free_audit_root, args, data.modality, data.operation, args.condition, seed)
    subspace = run / "subspace.pt"
    results = run / "results.jsonl"
    if not subspace.exists() or not results.exists():
        raise FileNotFoundError(f"Missing materialized readout-free DAS for {task} seed={seed}: {run}")
    geom.validate_result_metadata(args, geom.result_row(results), task=task, path=results)
    basis, payload = geom.load_basis_file(subspace, data.hidden.shape[1], None)
    basis = geom.orthonormal_basis(basis, f"{task} DAS readout-free seed={seed}", rank=args.k)
    meta = payload.get("readout_ablation", {})
    return basis, {"basis_path": str(subspace), "readout_ablation": meta}


def build_spaces(
    args: argparse.Namespace,
    data_by_task: dict[str, geom.TaskData],
    rows_by_task: dict[str, dict],
    common_values: list[int],
) -> tuple[dict[tuple[str, str], list[SpaceItem]], list[dict]]:
    spaces: dict[tuple[str, str], list[SpaceItem]] = {}
    diagnostics = []
    pca_cache = {}
    common_set = set(common_values)

    for task in args.tasks:
        data = data_by_task[task]
        if not args.digit_readout_basis_path.exists():
            raise FileNotFoundError(
                f"Missing digit readout basis: {args.digit_readout_basis_path}. "
                "Regenerate/update the readout-ablated artifact root with "
                    "run src.experiments.readout_latent_geometry.controls.readout_ablated_subspaces first; "
                "the generator now saves digit_readout_basis.pt even when subspaces already exist."
            )
        digit_basis = geom.load_digit_basis(args.digit_readout_basis_path, data.hidden.shape[1])
        row = rows_by_task[task]
        if any(space in args.space_types for space in ("pca_leading_readout_free", "random_pca_readout_free")):
            features, feature_meta = pca_training_features(args, task, data, row)
            config = geom.load_basis_file(subspace_path(args, data.modality, data.operation, args.seeds[0]), data.hidden.shape[1], args.k)[1].get("config", {})
            pca_space, pca_meta = pca_space_with_at_least_k(
                features,
                args.k,
                float(config.get("pca_variance_threshold", 0.9)),
            )
            pca_cache[task] = (pca_space, {**feature_meta, **pca_meta})
            print(
                f"PCA_TRAIN_SPACE task={task} retained_rank={pca_meta['pca_retained_rank']} "
                f"explained={pca_meta['pca_explained_variance']:.4f} samples={feature_meta['pca_selected_samples']}"
            )

        for space_type in args.space_types:
            items = []
            if space_type == "das_readout_free":
                seeds = args.seeds
                for seed in seeds:
                    raw_basis, extra = load_das_readout_free_basis(args, task, data, seed)
                    basis, rf_meta = project_out_readout(raw_basis, digit_basis, args.k, f"{task} das_readout_free seed={seed}")
                    items.append((seed, basis, {**extra, **rf_meta}))
            elif space_type == "pca_leading_readout_free":
                pca_space, pca_meta = pca_cache[task]
                basis, rf_meta = project_out_readout(pca_space[:, : args.k], digit_basis, args.k, f"{task} leading PCA")
                items.append((0, basis, {**pca_meta, **rf_meta}))
            elif space_type == "random_pca_readout_free":
                pca_space, pca_meta = pca_cache[task]
                for seed in args.control_seeds:
                    raw = random_pca_basis(pca_space, args.k, stable_seed(EXPERIMENT, task, space_type, seed))
                    basis, rf_meta = project_out_readout(raw, digit_basis, args.k, f"{task} random PCA seed={seed}")
                    items.append((seed, basis, {**pca_meta, **rf_meta}))
            elif space_type == "random_residual_readout_free":
                for seed in args.control_seeds:
                    raw = random_residual_basis(data.hidden.shape[1], args.k, stable_seed(EXPERIMENT, task, space_type, seed))
                    basis, rf_meta = project_out_readout(raw, digit_basis, args.k, f"{task} random residual seed={seed}")
                    items.append((seed, basis, rf_meta))
            elif space_type == "digit_readout":
                items.append((0, digit_basis, {"supplementary": True, "rank_after_readout_removal": int(digit_basis.shape[1])}))
            else:
                raise ValueError(f"Unhandled space type {space_type}")

            for seed, basis, extra in items:
                diag = basis_diagnostics(
                    task=task,
                    space_type=space_type,
                    space_seed=seed,
                    basis=basis,
                    digit_basis=digit_basis,
                    extra=extra,
                )
                if space_type.endswith("_readout_free") and diag["readout_overlap"] > args.readout_overlap_tolerance:
                    raise ValueError(
                        f"{task} {space_type} seed={seed} is not readout-free enough: "
                        f"readout_overlap={diag['readout_overlap']:.3e} > {args.readout_overlap_tolerance:.3e}"
                    )
                if space_type in PRIMARY_SPACES and basis.shape[1] < args.k:
                    print(f"RANK_DROP task={task} space={space_type} seed={seed} rank={basis.shape[1]} expected={args.k}")
                centroids = geom.centroid_coordinates(data, basis, common_set, args.target)
                r2 = result_centroid_variance_fraction(data, basis, common_set, args.target)
                diag["result_centroid_variance_fraction"] = r2
                diagnostics.append(diag)
                spaces.setdefault((task, space_type), []).append(
                    SpaceItem(task, space_type, seed, basis, centroids, r2, diag)
                )
                print(
                    f"SPACE task={task} space={space_type} seed={seed} rank={basis.shape[1]} "
                    f"readout_overlap={diag['readout_overlap']:.3e} value_R2={r2:.4f}"
                )
    return spaces, diagnostics


def paired_space_items(
    spaces: dict[tuple[str, str], list[SpaceItem]],
    source_task: str,
    destination_task: str,
    space_type: str,
) -> list[tuple[SpaceItem, SpaceItem]]:
    source_items = spaces[(source_task, space_type)]
    destination_items = spaces[(destination_task, space_type)]
    if space_type == "das_readout_free":
        return [(source, destination) for source in source_items for destination in destination_items]
    if space_type in RANDOM_SPACES:
        destination_by_seed = {item.space_seed: item for item in destination_items}
        return [(source, destination_by_seed[source.space_seed]) for source in source_items if source.space_seed in destination_by_seed]
    return [(source_items[0], destination_items[0])]


def as_task_space(item: SpaceItem) -> geom.TaskSpace:
    return geom.TaskSpace(
        task=item.task,
        seed=item.space_seed,
        space=item.space_type,
        basis=item.basis,
        centroids=item.centroids,
        diagnostics=item.diagnostics,
    )


def analyze_spaces(
    args: argparse.Namespace,
    spaces: dict[tuple[str, str], list[SpaceItem]],
    value_splits: dict[str, dict[str, list[int]]],
    space_types: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    transition_rows = []
    retrieval_rows = []
    rsa_rows = []
    task_pairs = [(a, b) for a in args.tasks for b in args.tasks if a != b]
    for space_type in space_types:
        if space_type not in args.space_types:
            continue
        print(f"\nANALYZE_SPACE {space_type}")
        for split_seed_text, split in value_splits.items():
            split_seed = int(split_seed_text)
            train_values = split["train_values"]
            test_values = split["test_values"]
            for source_task, destination_task in task_pairs:
                for source, destination in paired_space_items(spaces, source_task, destination_task, space_type):
                    train_transitions, x_train = geom.transition_matrix(source.centroids, train_values)
                    _train_transitions_b, y_train = geom.transition_matrix(destination.centroids, train_values)
                    test_transitions, x_test = geom.transition_matrix(source.centroids, test_values)
                    _test_transitions_b, y_test = geom.transition_matrix(destination.centroids, test_values)
                    q, alpha = geom.fit_scaled_procrustes(x_train, y_train)
                    base = {
                        "source_task": source_task,
                        "source_task_label": geom.TASK_LABELS[source_task],
                        "destination_task": destination_task,
                        "destination_task_label": geom.TASK_LABELS[destination_task],
                        "task_relation": f"{geom.TASK_LABELS[source_task]}->{geom.TASK_LABELS[destination_task]}",
                        "source_space_seed": source.space_seed,
                        "destination_space_seed": destination.space_seed,
                        "value_split_seed": split_seed,
                        "space_type": space_type,
                        "space": space_type,
                        "n_train_values": len(train_values),
                        "n_test_values": len(test_values),
                        "n_train_transitions": len(train_transitions),
                        "n_test_transitions": len(test_transitions),
                        "source_rank": source.basis.shape[1],
                        "destination_rank": destination.basis.shape[1],
                        "alpha": alpha,
                        "source_result_centroid_variance_fraction": source.result_centroid_variance_fraction,
                        "destination_result_centroid_variance_fraction": destination.result_centroid_variance_fraction,
                    }
                    transition_rows.append({**base, **geom.transition_metrics(x_test, y_test, q, alpha)})
                    retrieval_rows.append(
                        {
                            **base,
                            **geom.retrieval_metrics(source.centroids, destination.centroids, train_values, test_values, q, alpha),
                        }
                    )
            for task_a in args.tasks:
                for task_b in args.tasks:
                    for source, destination in paired_space_items(spaces, task_a, task_b, space_type):
                        rsa = geom.rsa_row(
                            as_task_space(source),
                            as_task_space(destination),
                            test_values,
                            split_seed,
                            args.rsa_permutations,
                            stable_seed(EXPERIMENT, "rsa", task_a, task_b, source.space_seed, destination.space_seed, split_seed, space_type),
                        )
                        rsa["space_type"] = space_type
                        rsa["space"] = space_type
                        rsa_rows.append(rsa)
        geom.write_csv(transition_rows, args.output_dir / "transition_procrustes_by_space.csv")
        geom.write_csv(retrieval_rows, args.output_dir / "value_retrieval_by_space.csv")
        geom.write_csv(rsa_rows, args.output_dir / "rsa_by_space.csv")
        print(f"CHECKPOINT_SPACE_SPECIFICITY space={space_type} output_dir={args.output_dir}")
    return transition_rows, retrieval_rows, rsa_rows


def mean_metric(rows: list[dict], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]
    return None if not values else float(sum(values) / len(values))


def regression_check(args: argparse.Namespace, transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict]) -> dict:
    das_transition = [row for row in transition_rows if row["space_type"] == "das_readout_free"]
    das_retrieval = [row for row in retrieval_rows if row["space_type"] == "das_readout_free"]
    das_rsa = [
        row for row in rsa_rows
        if row["space_type"] == "das_readout_free" and row["task_a"] != row["task_b"]
    ]
    result = {
        "das_mean_heldout_transition_cosine": mean_metric(das_transition, "heldout_transition_cosine_mean"),
        "das_mean_top1_value_retrieval": mean_metric(das_retrieval, "top1_value_retrieval"),
        "das_mean_cross_task_rsa": mean_metric(das_rsa, "spearman_rsa"),
        "thresholds": {
            "transition": args.das_regression_min_transition,
            "top1": args.das_regression_min_top1,
            "rsa": args.das_regression_min_rsa,
        },
    }
    if args.skip_das_regression_check or args.smoke_test:
        result["passed"] = None
        result["skipped"] = True
        return result
    failures = []
    if result["das_mean_heldout_transition_cosine"] is None or result["das_mean_heldout_transition_cosine"] < args.das_regression_min_transition:
        failures.append("heldout transition cosine")
    if result["das_mean_top1_value_retrieval"] is None or result["das_mean_top1_value_retrieval"] < args.das_regression_min_top1:
        failures.append("top1 retrieval")
    if result["das_mean_cross_task_rsa"] is None or result["das_mean_cross_task_rsa"] < args.das_regression_min_rsa:
        failures.append("cross-task RSA")
    result["passed"] = not failures
    result["failures"] = failures
    if failures:
        raise RuntimeError(f"DAS readout-free regression check failed: {failures}; metrics={result}")
    return result


def aggregate_summary(args: argparse.Namespace, transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict]) -> dict:
    transition_summary = geom.summarize(
        transition_rows,
        ["heldout_transition_cosine_mean", "heldout_relative_error_mean"],
        ["source_task", "destination_task", "space_type"],
    )
    retrieval_summary = geom.summarize(
        retrieval_rows,
        ["top1_value_retrieval", "top5_value_retrieval", "same_value_cosine_mean", "mean_absolute_value_error"],
        ["source_task", "destination_task", "space_type"],
    )
    rsa_summary = geom.summarize(
        rsa_rows,
        ["spearman_rsa", "permutation_pvalue"],
        ["task_a", "task_b", "space_type"],
    )
    primary = {}
    order = {task: index for index, task in enumerate(geom.TASK_ORDER)}
    for space_type in args.space_types:
        cross_transition = [row for row in transition_rows if row["space_type"] == space_type]
        cross_retrieval = [row for row in retrieval_rows if row["space_type"] == space_type]
        cross_rsa = [
            row for row in rsa_rows
            if row["space_type"] == space_type and order.get(row["task_a"], 999) < order.get(row["task_b"], 999)
        ]
        primary[space_type] = {
            "mean_heldout_transition_cosine": mean_metric(cross_transition, "heldout_transition_cosine_mean"),
            "mean_top1_retrieval": mean_metric(cross_retrieval, "top1_value_retrieval"),
            "mean_same_value_cosine": mean_metric(cross_retrieval, "same_value_cosine_mean"),
            "mean_cross_task_rsa": mean_metric(cross_rsa, "spearman_rsa"),
        }
    return {
        "transition_summary": transition_summary,
        "retrieval_summary": retrieval_summary,
        "rsa_summary": rsa_summary,
        "primary_metric_means": primary,
    }


def control_comparisons(transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict]) -> list[dict]:
    specs = [
        ("transition", transition_rows, "heldout_transition_cosine_mean", ["source_task", "destination_task", "value_split_seed"]),
        ("retrieval", retrieval_rows, "top1_value_retrieval", ["source_task", "destination_task", "value_split_seed"]),
        ("rsa", rsa_rows, "spearman_rsa", ["task_a", "task_b", "value_split_seed"]),
    ]
    rows = []
    for family, source_rows, metric, keys in specs:
        groups: dict[tuple, list[dict]] = {}
        for row in source_rows:
            groups.setdefault(tuple(row[key] for key in keys), []).append(row)
        for key, parts in groups.items():
            das_values = [float(row[metric]) for row in parts if row["space_type"] == "das_readout_free"]
            if not das_values:
                continue
            das_mean = float(sum(das_values) / len(das_values))
            item_base = {keys[index]: key[index] for index in range(len(keys))}
            for control in ["random_pca_readout_free", "random_residual_readout_free"]:
                control_values = [float(row[metric]) for row in parts if row["space_type"] == control]
                if control_values:
                    rows.append(
                        {
                            **item_base,
                            "metric_family": family,
                            "metric": metric,
                            "control_space_type": control,
                            "das_mean": das_mean,
                            "control_mean": float(sum(control_values) / len(control_values)),
                            "difference_from_control_mean": das_mean - float(sum(control_values) / len(control_values)),
                            "empirical_control_percentile": (1 + sum(value >= das_mean for value in control_values)) / (1 + len(control_values)),
                            "n_control": len(control_values),
                        }
                    )
            pca_values = [float(row[metric]) for row in parts if row["space_type"] == "pca_leading_readout_free"]
            if pca_values:
                pca_mean = float(sum(pca_values) / len(pca_values))
                rows.append(
                    {
                        **item_base,
                        "metric_family": family,
                        "metric": metric,
                        "control_space_type": "pca_leading_readout_free",
                        "das_mean": das_mean,
                        "control_mean": pca_mean,
                        "difference_from_control_mean": das_mean - pca_mean,
                        "empirical_control_percentile": None,
                        "n_control": len(pca_values),
                    }
                )
    return rows


def plot_figures(args: argparse.Namespace, transition_rows: list[dict], retrieval_rows: list[dict], rsa_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    space_order = [space for space in PRIMARY_SPACES if space in args.space_types]
    labels = {
        "das_readout_free": "DAS RF",
        "pca_leading_readout_free": "PCA lead RF",
        "random_pca_readout_free": "Rand PCA RF",
        "random_residual_readout_free": "Rand resid RF",
    }
    relations = [f"{geom.TASK_LABELS[a]}->{geom.TASK_LABELS[b]}" for a in geom.TASK_ORDER for b in geom.TASK_ORDER if a in args.tasks and b in args.tasks and a != b]

    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    x = np.arange(len(relations))
    width = 0.18 if len(space_order) > 3 else 0.24
    offsets = np.linspace(-width * (len(space_order) - 1) / 2, width * (len(space_order) - 1) / 2, len(space_order))
    for offset, space in zip(offsets, space_order):
        means, stds = [], []
        for relation in relations:
            values = [float(row["heldout_transition_cosine_mean"]) for row in transition_rows if row["task_relation"] == relation and row["space_type"] == space]
            means.append(np.nan if not values else float(np.mean(values)))
            stds.append(0.0 if len(values) < 2 else float(np.std(values, ddof=1)))
        ax.bar(x + offset, means, width, yerr=stds, label=labels.get(space, space), capsize=2)
    ax.set_ylabel("held-out transition cosine")
    ax.set_title("Readout-free numerical transition geometry by observation space")
    ax.set_xticks(x, relations, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure1_space_specificity_transition_geometry.png", dpi=240)
    fig.savefig(figure_dir / "figure1_space_specificity_transition_geometry.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    means, stds = [], []
    for space in space_order:
        values = [float(row["top1_value_retrieval"]) for row in retrieval_rows if row["space_type"] == space]
        means.append(np.nan if not values else float(np.mean(values)))
        stds.append(0.0 if len(values) < 2 else float(np.std(values, ddof=1)))
    ax.bar(np.arange(len(space_order)), means, yerr=stds, capsize=3)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, label="chance top-1")
    ax.set_xticks(np.arange(len(space_order)), [labels.get(space, space) for space in space_order], rotation=20, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("mean top-1 held-out value retrieval")
    ax.set_title("Held-out same-value retrieval")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure2_space_specificity_value_retrieval.png", dpi=240)
    fig.savefig(figure_dir / "figure2_space_specificity_value_retrieval.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    means, stds = [], []
    for space in space_order:
        values = [
            float(row["spearman_rsa"]) for row in rsa_rows
            if row["space_type"] == space and geom.TASK_ORDER.index(row["task_a"]) < geom.TASK_ORDER.index(row["task_b"])
        ]
        means.append(np.nan if not values else float(np.mean(values)))
        stds.append(0.0 if len(values) < 2 else float(np.std(values, ddof=1)))
    ax.bar(np.arange(len(space_order)), means, yerr=stds, capsize=3)
    ax.set_xticks(np.arange(len(space_order)), [labels.get(space, space) for space in space_order], rotation=20, ha="right")
    ax.set_ylabel("mean cross-task RSA")
    ax.set_title("No-fit readout-free value geometry")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure3_space_specificity_nofit_rsa.png", dpi=240)
    fig.savefig(figure_dir / "figure3_space_specificity_nofit_rsa.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = normalize_tasks(args.tasks)
    args.space_types = normalize_spaces(args.space_types)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Readout-free space specificity")
    print(f"  tasks={args.tasks}")
    print(f"  space_types={args.space_types}")
    print(f"  value_splits={args.value_split_seeds}")
    print(f"  control_seeds={args.control_seeds}")

    value_splits = load_exact_value_splits(args)
    common_values = all_split_values(value_splits)

    rows_by_task = {}
    data_by_task = {}
    for task in args.tasks:
        modality, operation = parse_task(task)
        row = geom.first_result_row(args, modality, operation, args.condition, args.seeds[0])
        geom.validate_result_metadata(args, row, task=task, path=subspace_path(args, modality, operation, args.seeds[0]))
        rows_by_task[task] = row
        data_by_task[task] = geom.load_task_data(args, task, row)

    for task, data in data_by_task.items():
        missing = [value for value in common_values if data.counts.get(value, 0) < 1]
        if missing:
            raise ValueError(f"{task} is missing common split values from previous experiment: {missing[:10]}")

    spaces, diagnostics = build_spaces(args, data_by_task, rows_by_task, common_values)
    geom.write_csv(diagnostics, args.output_dir / "space_diagnostics.csv")
    save_json(value_splits, args.output_dir / "value_splits.json")

    if args.artifact_check_only:
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "space_types": args.space_types,
                "n_common_values": len(common_values),
                "common_values": common_values,
                "value_splits": value_splits,
                "diagnostics_rows": len(diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("\nARTIFACT_CHECK_ONLY complete")
        return

    transition_rows, retrieval_rows, rsa_rows = analyze_spaces(args, spaces, value_splits, ["das_readout_free"])
    regression = regression_check(args, transition_rows, retrieval_rows, rsa_rows)
    remaining = [space for space in args.space_types if space != "das_readout_free"]
    more_transition, more_retrieval, more_rsa = analyze_spaces(args, spaces, value_splits, remaining)
    transition_rows += more_transition
    retrieval_rows += more_retrieval
    rsa_rows += more_rsa

    geom.write_csv(transition_rows, args.output_dir / "transition_procrustes_by_space.csv")
    geom.write_csv(retrieval_rows, args.output_dir / "value_retrieval_by_space.csv")
    geom.write_csv(rsa_rows, args.output_dir / "rsa_by_space.csv")
    comparisons = control_comparisons(transition_rows, retrieval_rows, rsa_rows)
    geom.write_csv(comparisons, args.output_dir / "interpretation_safe_comparisons.csv")
    summary = aggregate_summary(args, transition_rows, retrieval_rows, rsa_rows)
    summary.update(
        {
            "experiment": EXPERIMENT,
            "description": "Readout-free DAS value geometry compared to train-PCA and random readout-free subspace controls.",
            "tasks": args.tasks,
            "task_labels": geom.TASK_LABELS,
            "space_types": args.space_types,
            "n_common_values": len(common_values),
            "common_values": common_values,
            "value_splits": value_splits,
            "das_regression_check": regression,
            "interpretation_safe_comparisons_rows": len(comparisons),
            "config": jsonable(vars(args)),
        }
    )
    save_json(summary, args.output_dir / "space_specificity_summary.json")
    if not args.skip_plots:
        plot_figures(args, transition_rows, retrieval_rows, rsa_rows)

    print("\nReadout-free space specificity complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Transition rows: {len(transition_rows)}")
    print(f"Retrieval rows: {len(retrieval_rows)}")
    print(f"RSA rows: {len(rsa_rows)}")
    print(f"Diagnostics rows: {len(diagnostics)}")


if __name__ == "__main__":
    main()
