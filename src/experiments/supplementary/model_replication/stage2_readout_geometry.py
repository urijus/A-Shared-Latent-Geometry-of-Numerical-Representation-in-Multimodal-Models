"""Stage 2 readout decomposition and value-held-out L-space geometry for Ministral.

This runner assumes the final Stage 1 DAS spaces already exist. It does not
train DAS or run autoregressive interventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from src.geometry.readout import get_output_weight
from src.interventions.das import (
    format_prompt,
    hidden,
    hook_module,
    resolve_position,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    orthogonal_procrustes,
    scaled_alpha,
    stable_seed,
)
from src.experiments.global_geometry.synchronization.synchronization import (
    Edge,
    connected_components,
    hub_map,
    synchronize_rotations,
    synchronize_scales,
)
from src.experiments.supplementary.model_replication import stage1_das
from src.models import get_blocks, get_hidden_size, load_hf_model_and_processor, resolve_model_for_loading


TASKS = ["T+", "T-", "I+", "I-"]
DEFAULT_OUTPUT_DIR = Path("src/experiments/supplementary/model_replication/results/stage2_readout_geometry")
DEFAULT_DAS_ROOT = Path("src/experiments/supplementary/model_replication/results/das")


@dataclass
class DASArtifact:
    task: str
    seed: int
    digest: str
    run_dir: Path
    basis: torch.Tensor
    metrics: dict
    metadata: dict


@dataclass
class Space:
    task: str
    seed: int
    basis: torch.Tensor
    c_basis: torch.Tensor
    l_basis: torch.Tensor
    centroids: dict[int, torch.Tensor]
    digest: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=stage1_das.MODEL_ALIAS)
    parser.add_argument("--model_slug", default=stage1_das.MODEL_SLUG)
    parser.add_argument("--das_root", type=Path, default=DEFAULT_DAS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--position", default="-1")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--target", default="result")
    parser.add_argument("--format", default="digits")
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true", default=True)

    # These must match the final Stage 1 config, because they are in the hash.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--max_train_pairs", type=int, default=4096)
    parser.add_argument("--max_validation_pairs", type=int, default=512)
    parser.add_argument("--max_test_pairs", type=int, default=512)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=128)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pca_max_samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=8)

    parser.add_argument("--activation_batch_size", type=int, default=1)
    parser.add_argument("--value_split_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train_value_fraction", type=float, default=0.8)
    parser.add_argument("--rsa_permutations", type=int, default=1000)
    parser.add_argument("--sanity_tolerance", type=float, default=1e-4)
    parser.add_argument("--svd_tolerance", type=float, default=1e-6)
    parser.add_argument("--weight_metric", choices=["mean_cosine", "inverse_rmse", "uniform"], default="mean_cosine")
    parser.add_argument("--force_activations", action="store_true")
    parser.add_argument("--skip_heatmaps", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    return parser.parse_args()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def std(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    m = sum(finite) / len(finite)
    return (sum((value - m) ** 2 for value in finite) / (len(finite) - 1)) ** 0.5


def stage1_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        model_slug=args.model_slug,
        output_root=args.das_root,
        target=args.target,
        hook=args.hook,
        position=args.position,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        split_seed=args.split_seed,
        max_train_pairs=args.max_train_pairs,
        max_validation_pairs=args.max_validation_pairs,
        max_test_pairs=args.max_test_pairs,
        max_autoregressive_pairs=args.max_autoregressive_pairs,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        batch_size=args.batch_size,
        pca_max_samples=args.pca_max_samples,
        pca_variance_threshold=args.pca_variance_threshold,
        max_new_tokens=args.max_new_tokens,
        format=args.format,
        prompt=args.prompt,
        enable_thinking=args.enable_thinking,
        use_chat_template=args.use_chat_template,
    )


def load_basis(path: Path, hidden_size: int, k: int) -> torch.Tensor:
    basis = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(basis, dict):
        basis = basis.get("basis", basis.get(str(k)))
    basis = torch.as_tensor(basis).detach().float().cpu()
    if tuple(basis.shape) == (k, hidden_size):
        basis = basis.T.contiguous()
    if tuple(basis.shape) != (hidden_size, k):
        raise ValueError(f"Expected DAS basis shape {(hidden_size, k)} in {path}, got {tuple(basis.shape)}.")
    return basis


def artifact_hash_from_metadata(metadata: dict, run_dir: Path) -> str:
    return str(metadata.get("artifact_hash") or metadata.get("config_hash") or run_dir.name.split("_")[-1])


def dataset_identity(config: dict) -> dict:
    pair_sampling = config.get("pair_sampling_seeds", {})
    return {
        "model_id": config.get("model_id"),
        "model_slug": config.get("model_slug"),
        "task": config.get("task"),
        "modality": config.get("modality"),
        "operation": config.get("operation"),
        "target": config.get("target"),
        "layer": config.get("layer"),
        "hook": config.get("hook"),
        "intervention_position": config.get("intervention_position"),
        "k": config.get("k"),
        "split_seed": config.get("split_seed"),
        "pair_sampling_seeds": {
            "train": pair_sampling.get("train"),
            "validation": pair_sampling.get("validation"),
            "test": pair_sampling.get("test"),
        },
        "train_pair_count": config.get("train_pair_count"),
        "validation_pair_count": config.get("validation_pair_count"),
        "test_pair_count": config.get("test_pair_count"),
        "train_fraction": config.get("train_fraction"),
        "validation_fraction": config.get("validation_fraction"),
        "format": config.get("format"),
    }


def assert_fixed_datasets_across_das_seeds(artifacts: dict[tuple[str, int], DASArtifact]) -> None:
    by_task: dict[str, dict | None] = defaultdict(lambda: None)
    for (task, seed), artifact in sorted(artifacts.items()):
        identity = dataset_identity(load_json(artifact.run_dir / "config.json"))
        if by_task[task] is None:
            by_task[task] = identity
            continue
        if identity != by_task[task]:
            raise RuntimeError(
                "Dataset split or pair-sampling fields changed across DAS seeds "
                f"for {task}; first identity={by_task[task]}, seed{seed} identity={identity}"
            )


def find_final_artifacts(args: argparse.Namespace, hidden_size: int | None = None) -> dict[tuple[str, int], DASArtifact]:
    finder_args = stage1_args(args)
    missing = []
    artifacts: dict[tuple[str, int], DASArtifact] = {}
    for task in args.tasks:
        for seed in args.seeds:
            _config, digest, run_dir = stage1_das.das_config(finder_args, task, args.layer, args.k, seed)
            if not stage1_das.completed_run(run_dir):
                missing.append((task, seed, digest, run_dir))
                continue
            metadata = load_json(run_dir / "metadata.json")
            metrics = load_json(run_dir / "metrics.json")
            basis = torch.empty(0)
            if hidden_size is not None:
                basis = load_basis(run_dir / "das_basis.pt", hidden_size, args.k)
            artifacts[(task, seed)] = DASArtifact(
                task=task,
                seed=seed,
                digest=digest,
                run_dir=run_dir,
                basis=basis,
                metrics=metrics,
                metadata=metadata,
            )
            print(f"[FOUND] {task} S{seed} -> {digest} {run_dir}")
    if missing:
        lines = [
            "Missing required completed Stage 1 DAS artifacts for exact final config:",
            f"  layer={args.layer} k={args.k} position={args.position} epochs={args.epochs}",
        ]
        for task, seed, digest, run_dir in missing:
            lines.append(f"  {task} seed={seed} hash={digest} expected={run_dir}")
        raise FileNotFoundError("\n".join(lines))
    assert_fixed_datasets_across_das_seeds(artifacts)
    return artifacts


def digit_readout_basis(model, tokenizer, args: argparse.Namespace) -> tuple[torch.Tensor, dict]:
    output_weight = get_output_weight(model).double()
    digit_ids = {}
    digit_vectors = []
    for digit in range(10):
        token_ids = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Digit {digit!r} is not a single token: {token_ids}")
        token_id = int(token_ids[0])
        digit_ids[str(digit)] = {
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens([token_id])[0],
        }
        digit_vectors.append(output_weight[token_id])
    matrix = torch.stack(digit_vectors, dim=0)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    tol = args.svd_tolerance * max(centered.shape) * float(singular_values.max().item())
    rank = int((singular_values > tol).sum().item())
    if rank < 1:
        raise ValueError("Centered digit readout span has numerical rank 0.")
    u_digit = vh[:rank].T.contiguous()
    info = {
        "digit_ids": digit_ids,
        "digit_singular_values": [float(value) for value in singular_values],
        "digit_rank": rank,
        "digit_rank_tolerance": tol,
        "output_weight_shape": list(output_weight.shape),
    }
    return u_digit.float(), info


def decompose_artifacts(
    args: argparse.Namespace,
    artifacts: dict[tuple[str, int], DASArtifact],
    u_digit: torch.Tensor,
) -> tuple[dict[tuple[str, int], Space], list[dict]]:
    spaces: dict[tuple[str, int], Space] = {}
    rows = []
    c_payload = {"basis": {}, "config": jsonable(vars(args))}
    l_payload = {"basis": {}, "config": jsonable(vars(args))}
    for (task, seed), artifact in sorted(artifacts.items()):
        r = artifact.basis.double()
        u = u_digit.double()
        rr = r.T @ r
        b = u.T @ r
        _ub, singular_values, vh = torch.linalg.svd(b, full_matrices=True)
        rank_tol = args.svd_tolerance * max(b.shape) * float(singular_values.max().item()) if singular_values.numel() else 0.0
        rank_digit = int((singular_values > rank_tol).sum().item())
        v = vh.T
        q = r @ v
        c_basis = q[:, :rank_digit].contiguous().float()
        l_basis = q[:, rank_digit:].contiguous().float()
        projection_sum = c_basis.double() @ c_basis.double().T + l_basis.double() @ l_basis.double().T
        original_projection = r @ r.T
        row = {
            "task": task,
            "seed": seed,
            "hash": artifact.digest,
            "rank_digit": rank_digit,
            "dimC": int(c_basis.shape[1]),
            "dimL": int(l_basis.shape[1]),
            "E_digit": float((u.T @ r).square().sum() / args.k),
            "singular_values": [float(value) for value in singular_values],
            "R_orthonormality_error": float((rr - torch.eye(args.k, dtype=rr.dtype)).abs().max()),
            "U_T_L_norm": float((u.T @ l_basis.double()).norm()),
            "C_T_L_norm": float((c_basis.double().T @ l_basis.double()).norm()),
            "recon_err": float((projection_sum - original_projection).norm()),
        }
        bad = [
            name
            for name in ("R_orthonormality_error", "U_T_L_norm", "C_T_L_norm", "recon_err")
            if row[name] > args.sanity_tolerance
        ]
        if bad:
            raise RuntimeError(f"Poor decomposition tolerance for {task} seed={seed}: {bad} metrics={row}")
        c_payload["basis"].setdefault(task, {})[str(seed)] = c_basis
        l_payload["basis"].setdefault(task, {})[str(seed)] = l_basis
        spaces[(task, seed)] = Space(
            task=task,
            seed=seed,
            basis=artifact.basis,
            c_basis=c_basis,
            l_basis=l_basis,
            centroids={},
            digest=artifact.digest,
        )
        rows.append(row)
    decomp_dir = args.output_dir / "decomposition"
    decomp_dir.mkdir(parents=True, exist_ok=True)
    torch.save(c_payload, decomp_dir / "C_basis.pt")
    torch.save(l_payload, decomp_dir / "L_basis.pt")
    write_json(decomp_dir / "decomposition_metrics.json", rows)
    print("\nDECOMPOSITION")
    print_table(
        ["Task", "Seed", "rank_digit", "dimC", "dimL", "E_digit", "||U^T L||", "recon_err"],
        [
            [
                row["task"],
                row["seed"],
                row["rank_digit"],
                row["dimC"],
                row["dimL"],
                fmt(row["E_digit"]),
                fmt(row["U_T_L_norm"]),
                fmt(row["recon_err"]),
            ]
            for row in rows
        ],
    )
    return spaces, rows


def task_data_path(task: str, artifacts: dict[tuple[str, int], DASArtifact], args: argparse.Namespace) -> Path:
    metrics = artifacts[(task, args.seeds[0])].metrics
    path = metrics.get("data_path")
    if path:
        return Path(path)
    return stage1_das.task_data_path(task, args.model_slug, args.format)


def resolve_data_root(data_path: Path, task: str) -> Path:
    if task.startswith("I"):
        return data_path.parent
    return data_path.parent


def activation_cache_path(args: argparse.Namespace, task: str) -> Path:
    clean = task.replace("+", "plus").replace("-", "minus")
    return args.output_dir / "activations" / f"{clean}_L{args.layer}_pos{args.position}.pt"


def capture_batch(model, blocks, encoding: dict, positions: list[int], layer: int, hook_name: str) -> torch.Tensor:
    captured = {}
    with ExitStack() as stack:
        module, pre_hook = hook_module(blocks[layer - 1], hook_name)
        if pre_hook:
            def capture_pre(_module, inputs):
                captured["value"] = hidden(inputs[0]).detach()

            handle = module.register_forward_pre_hook(capture_pre)
        else:
            def capture_post(_module, _inputs, output):
                captured["value"] = hidden(output).detach()

            handle = module.register_forward_hook(capture_post)
        stack.callback(handle.remove)
        model(**encoding, use_cache=False)
    rows = torch.arange(len(positions), device=model.device)
    cols = torch.tensor(positions, device=model.device)
    return captured["value"][rows, cols].float().cpu()


@torch.no_grad()
def collect_text_activations(args, task: str, samples: list[dict], model, tokenizer, blocks) -> tuple[torch.Tensor, list[dict]]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    parts = []
    labels = []
    try:
        for start in tqdm(range(0, len(samples), args.activation_batch_size), desc=f"Cache {task} activations"):
            batch = samples[start : start + args.activation_batch_size]
            prompts = [format_prompt(tokenizer, sample, args.use_chat_template) for sample in batch]
            positions = [resolve_position(tokenizer, prompt, args.position) for prompt in prompts]
            encoding = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
            parts.append(capture_batch(model, blocks, encoding, positions, args.layer, args.hook))
            for sample, prompt, position in zip(batch, prompts, positions):
                labels.append(
                    {
                        "expr": sample.get("expr"),
                        "result": int(sample["result"]),
                        "a": sample.get("a"),
                        "b": sample.get("b"),
                        "model_expr": prompt,
                        "hf_position": int(position),
                    }
                )
    finally:
        tokenizer.padding_side = old_padding_side
    return torch.cat(parts, dim=0), labels


@torch.no_grad()
def collect_image_activations(args, task: str, samples: list[dict], data_root: Path, model, processor, tokenizer, blocks) -> tuple[torch.Tensor, list[dict]]:
    parts = []
    labels = []
    for start in tqdm(range(0, len(samples), args.activation_batch_size), desc=f"Cache {task} activations"):
        batch = samples[start : start + args.activation_batch_size]
        prompts = [sample_prompt(processor, sample, args.prompt, args.enable_thinking) for sample in batch]
        images = [load_rgb_image(image_path_for(sample, data_root)) for sample in batch]
        positions = resolve_batch_positions(processor, tokenizer, model, prompts, images, args.position)
        encoding = inputs_to_device(
            make_inputs(processor, prompts, images),
            model.device,
            dtype=getattr(model, "dtype", None),
        )
        parts.append(capture_batch(model, blocks, encoding, positions, args.layer, args.hook))
        for sample, prompt, position in zip(batch, prompts, positions):
            labels.append(
                {
                    "expr": sample.get("expr", sample.get("image_text")),
                    "image_path": sample.get("image_path"),
                    "result": int(sample["result"]),
                    "a": sample.get("a"),
                    "b": sample.get("b"),
                    "model_expr": prompt,
                    "hf_position": int(position),
                }
            )
    return torch.cat(parts, dim=0), labels


def load_or_collect_activations(args, task: str, artifacts, model, processor, tokenizer, blocks) -> dict:
    path = activation_cache_path(args, task)
    if path.exists() and not args.force_activations:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        print(f"[CACHE] activations {task} -> {path}")
        return payload
    data_path = task_data_path(task, artifacts, args)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing correctly answered dataset for {task}: {data_path}")
    samples = load_jsonl(data_path)
    if not samples:
        raise ValueError(f"No samples in {data_path}")
    if task.startswith("T"):
        activations, labels = collect_text_activations(args, task, samples, model, tokenizer, blocks)
    else:
        activations, labels = collect_image_activations(
            args,
            task,
            samples,
            resolve_data_root(data_path, task),
            model,
            processor,
            tokenizer,
            blocks,
        )
    counts = defaultdict(int)
    for label in labels:
        counts[int(label["result"])] += 1
    payload = {
        "task": task,
        "layer": args.layer,
        "position": args.position,
        "hook": args.hook,
        "data_path": str(data_path),
        "activations": activations.to(torch.float16),
        "labels": labels,
        "result_counts": dict(sorted(counts.items())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[SAVED] activations {task} -> {path} shape={tuple(activations.shape)}")
    return payload


def compute_centroids_for_space(space: Space, activation_payload: dict) -> dict[int, torch.Tensor]:
    activations = activation_payload["activations"].float()
    labels = activation_payload["labels"]
    z = activations @ space.l_basis.float()
    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    for row, label in zip(z, labels):
        groups[int(label["result"])].append(row)
    return {value: torch.stack(rows).mean(dim=0) for value, rows in groups.items()}


def add_centroids(args, spaces: dict[tuple[str, int], Space], activations: dict[str, dict]) -> tuple[list[int], dict]:
    all_sets = []
    missing_by_space = {}
    for key, space in spaces.items():
        space.centroids = compute_centroids_for_space(space, activations[space.task])
        values = set(space.centroids)
        all_sets.append(values)
        missing_by_space[f"{space.task}_seed{space.seed}"] = sorted(set(range(100)) - values)
    common_values = sorted(set.intersection(*all_sets))
    write_json(
        args.output_dir / "activations" / "centroid_value_coverage.json",
        {
            "common_values": common_values,
            "n_common_values": len(common_values),
            "missing_0_99_by_space": missing_by_space,
        },
    )
    if len(common_values) < 3:
        raise ValueError(f"Too few common values for geometry: {common_values}")
    missing_common_0_99 = sorted(set(range(100)) - set(common_values))
    print(f"Common centroid values: n={len(common_values)} missing_0_99={missing_common_0_99}")
    return common_values, missing_by_space


def value_splits(values: list[int], seeds: list[int], fraction: float) -> list[dict]:
    splits = []
    n_train = int(round(len(values) * fraction))
    for seed in seeds:
        shuffled = list(values)
        random.Random(seed).shuffle(shuffled)
        train = sorted(shuffled[:n_train])
        test = sorted(shuffled[n_train:])
        if len(test) < 2:
            raise ValueError("Held-out value split must contain at least two values.")
        splits.append({"split_seed": seed, "train_values": train, "test_values": test})
    return splits


def transitions(values: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for a in values for b in values if a != b]


def delta_matrix(centroids: dict[int, torch.Tensor], pairs: list[tuple[int, int]]) -> torch.Tensor:
    return torch.stack([centroids[b] - centroids[a] for a, b in pairs]).float()


def fit_scaled_map(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int]) -> tuple[torch.Tensor, float, dict]:
    pairs = transitions(train_values)
    source_deltas = delta_matrix(source, pairs)
    destination_deltas = delta_matrix(destination, pairs)
    q = orthogonal_procrustes(source_deltas, destination_deltas).float()
    alpha = scaled_alpha(source_deltas, destination_deltas, q)
    aligned = float(alpha) * (source_deltas @ q)
    residual = aligned - destination_deltas
    cosine = F.cosine_similarity(aligned, destination_deltas, dim=1)
    return q, alpha, {
        "n_alignment_samples": int(source_deltas.shape[0]),
        "root_mean_squared_error": float(residual.square().mean().sqrt()),
        "mean_cosine": float(cosine.mean()),
    }


def train_offset(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int], q: torch.Tensor, alpha: float) -> torch.Tensor:
    mapped = torch.stack([float(alpha) * (source[value] @ q) for value in train_values])
    target = torch.stack([destination[value] for value in train_values])
    return (target - mapped).mean(dim=0)


def evaluate_map(
    source: dict[int, torch.Tensor],
    destination: dict[int, torch.Tensor],
    q: torch.Tensor,
    alpha: float,
    offset: torch.Tensor,
    test_values: list[int],
) -> dict:
    test_pairs = transitions(test_values)
    source_deltas = delta_matrix(source, test_pairs)
    destination_deltas = delta_matrix(destination, test_pairs)
    mapped_deltas = float(alpha) * (source_deltas @ q)
    transition_cosines = F.cosine_similarity(mapped_deltas, destination_deltas, dim=1)
    mapped_values = torch.stack([float(alpha) * (source[value] @ q) + offset for value in test_values])
    destination_values = torch.stack([destination[value] for value in test_values])
    same_value_cosines = F.cosine_similarity(mapped_values, destination_values, dim=1)
    distances = torch.cdist(mapped_values, destination_values)
    topk = torch.argsort(distances, dim=1)
    ids = torch.arange(len(test_values)).unsqueeze(1)
    return {
        "heldout_transition_cosine": float(transition_cosines.mean()),
        "same_value_aligned_cosine": float(same_value_cosines.mean()),
        "top1": float((topk[:, :1] == ids).any(dim=1).float().mean()),
        "top5": float((topk[:, : min(5, len(test_values))] == ids).any(dim=1).float().mean()),
        "n_test_values": len(test_values),
        "n_test_transitions": len(test_pairs),
    }


def shuffle_destination(destination: dict[int, torch.Tensor], train_values: list[int], seed: int) -> dict[int, torch.Tensor]:
    shuffled = list(train_values)
    random.Random(seed).shuffle(shuffled)
    remapped = dict(destination)
    for original, replacement in zip(train_values, shuffled):
        remapped[original] = destination[replacement]
    return remapped


def random_readout_free_basis(hidden_size: int, dim: int, u_digit: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    u = u_digit.float()
    for _attempt in range(20):
        raw = torch.randn(hidden_size, dim, generator=generator)
        raw = raw - u @ (u.T @ raw)
        q, r = torch.linalg.qr(raw, mode="reduced")
        if int((r.diag().abs() > 1e-7).sum()) == dim:
            return q.float()
    raise RuntimeError("Could not sample a full-rank random readout-orthogonal basis.")


def run_pairwise_geometry(args, spaces: dict[tuple[str, int], Space], splits: list[dict]) -> list[dict]:
    rows = []
    for split in splits:
        train_values = split["train_values"]
        test_values = split["test_values"]
        for source_task in args.tasks:
            for destination_task in args.tasks:
                if source_task == destination_task:
                    continue
                for source_seed in args.seeds:
                    for destination_seed in args.seeds:
                        source = spaces[(source_task, source_seed)]
                        destination = spaces[(destination_task, destination_seed)]
                        q, alpha, fit_metrics = fit_scaled_map(source.centroids, destination.centroids, train_values)
                        offset = train_offset(source.centroids, destination.centroids, train_values, q, alpha)
                        metrics = evaluate_map(source.centroids, destination.centroids, q, alpha, offset, test_values)
                        rows.append(
                            {
                                "space_type": "L",
                                "split_seed": split["split_seed"],
                                "source_task": source_task,
                                "destination_task": destination_task,
                                "source_seed": source_seed,
                                "destination_seed": destination_seed,
                                "alpha": alpha,
                                **fit_metrics,
                                **metrics,
                            }
                        )
    return rows


def run_shuffled_control(args, spaces: dict[tuple[str, int], Space], splits: list[dict]) -> list[dict]:
    rows = []
    for split in splits:
        train_values = split["train_values"]
        test_values = split["test_values"]
        for source_task in args.tasks:
            for destination_task in args.tasks:
                if source_task == destination_task:
                    continue
                for source_seed in args.seeds:
                    for destination_seed in args.seeds:
                        source = spaces[(source_task, source_seed)]
                        destination = spaces[(destination_task, destination_seed)]
                        control_seed = stable_seed("shuffle", split["split_seed"], source_task, destination_task, source_seed, destination_seed)
                        shuffled_destination = shuffle_destination(destination.centroids, train_values, control_seed)
                        q, alpha, fit_metrics = fit_scaled_map(source.centroids, shuffled_destination, train_values)
                        offset = train_offset(source.centroids, shuffled_destination, train_values, q, alpha)
                        metrics = evaluate_map(source.centroids, destination.centroids, q, alpha, offset, test_values)
                        rows.append(
                            {
                                "space_type": "shuffled",
                                "split_seed": split["split_seed"],
                                "source_task": source_task,
                                "destination_task": destination_task,
                                "source_seed": source_seed,
                                "destination_seed": destination_seed,
                                "control_seed": control_seed,
                                "alpha": alpha,
                                **fit_metrics,
                                **metrics,
                            }
                        )
    return rows


def make_random_spaces(args, activations: dict[str, dict], u_digit: torch.Tensor, hidden_size: int, dim: int) -> dict[tuple[str, int], Space]:
    spaces = {}
    for task in args.tasks:
        for seed in args.seeds:
            basis = random_readout_free_basis(hidden_size, dim, u_digit, stable_seed("random_readout_free", task, seed))
            space = Space(task, seed, basis, torch.empty(hidden_size, 0), basis, {}, f"random_{task}_{seed}")
            space.centroids = compute_centroids_for_space(space, activations[task])
            spaces[(task, seed)] = space
    return spaces


def run_random_control(args, random_spaces: dict[tuple[str, int], Space], splits: list[dict]) -> list[dict]:
    rows = run_pairwise_geometry(args, random_spaces, splits)
    for row in rows:
        row["space_type"] = "random_readout_free"
    return rows


def distance_vector(centroids: dict[int, torch.Tensor], values: list[int]) -> list[float]:
    matrix = torch.stack([centroids[value] for value in values]).float()
    distances = torch.cdist(matrix, matrix)
    return [float(distances[i, j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def rsa_metric(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], values: list[int], permutations: int, seed: int, shuffled_observed: bool = False) -> dict:
    source_vec = distance_vector(source, values)
    observed_values = list(values)
    if shuffled_observed:
        random.Random(seed).shuffle(observed_values)
    destination_vec = distance_vector(destination, observed_values)
    observed = float(spearmanr(source_vec, destination_vec).correlation)
    if math.isnan(observed):
        observed = 0.0
    rng = random.Random(seed + 17)
    extreme = 0
    for _ in range(permutations):
        permuted = list(values)
        rng.shuffle(permuted)
        permuted_vec = distance_vector(destination, permuted)
        score = float(spearmanr(source_vec, permuted_vec).correlation)
        if math.isnan(score):
            score = 0.0
        if abs(score) >= abs(observed):
            extreme += 1
    return {
        "rsa_spearman": observed,
        "rsa_permutation_p": (extreme + 1) / (permutations + 1),
    }


def run_rsa(args, spaces_by_type: dict[str, dict[tuple[str, int], Space]], splits: list[dict]) -> list[dict]:
    rows = []
    for split in splits:
        test_values = split["test_values"]
        for space_type, spaces in spaces_by_type.items():
            for source_task in args.tasks:
                for destination_task in args.tasks:
                    if source_task == destination_task:
                        continue
                    for source_seed in args.seeds:
                        for destination_seed in args.seeds:
                            source = spaces[(source_task, source_seed)]
                            destination = spaces[(destination_task, destination_seed)]
                            seed = stable_seed("rsa", space_type, split["split_seed"], source_task, destination_task, source_seed, destination_seed)
                            metrics = rsa_metric(
                                source.centroids,
                                destination.centroids,
                                test_values,
                                args.rsa_permutations,
                                seed,
                                shuffled_observed=(space_type == "shuffled"),
                            )
                            rows.append(
                                {
                                    "space_type": space_type,
                                    "split_seed": split["split_seed"],
                                    "source_task": source_task,
                                    "destination_task": destination_task,
                                    "source_seed": source_seed,
                                    "destination_seed": destination_seed,
                                    **metrics,
                                }
                            )
    return rows


def edge_from_fit(args, source: str, destination: str, q: torch.Tensor, alpha: float, fit_metrics: dict, tag: str) -> Edge:
    if args.weight_metric == "uniform":
        weight = 1.0
    elif args.weight_metric == "inverse_rmse":
        weight = 1.0 / max(float(fit_metrics.get("root_mean_squared_error") or 1.0), 1e-6)
    else:
        weight = max(float(fit_metrics.get("mean_cosine") or 0.0), 1e-3)
    return Edge(source, destination, q.float(), float(alpha), weight, Path(tag), fit_metrics)


def sync_edges_for_seed(spaces: dict[tuple[str, int], Space], tasks: list[str], seed: int, train_values: list[int], args, drop_pair: tuple[str, str] | None = None) -> list[Edge]:
    edges = []
    drop = set()
    if drop_pair is not None:
        drop = {drop_pair, (drop_pair[1], drop_pair[0])}
    for source in tasks:
        for destination in tasks:
            if source == destination or (source, destination) in drop:
                continue
            q, alpha, fit_metrics = fit_scaled_map(spaces[(source, seed)].centroids, spaces[(destination, seed)].centroids, train_values)
            edges.append(edge_from_fit(args, source, destination, q, alpha, fit_metrics, f"split{train_values[0]}_{source}_to_{destination}_seed{seed}"))
    return edges


def evaluate_relation_row(
    source_space: Space,
    destination_space: Space,
    q: torch.Tensor,
    alpha: float,
    train_values: list[int],
    test_values: list[int],
    extra: dict,
) -> dict:
    offset = train_offset(source_space.centroids, destination_space.centroids, train_values, q, alpha)
    return {**extra, **evaluate_map(source_space.centroids, destination_space.centroids, q, alpha, offset, test_values)}


def run_synchronization(args, spaces: dict[tuple[str, int], Space], splits: list[dict]) -> list[dict]:
    rows = []
    dim_l = next(iter(spaces.values())).l_basis.shape[1]
    undirected_relations = [
        (args.tasks[i], args.tasks[j])
        for i in range(len(args.tasks))
        for j in range(i + 1, len(args.tasks))
    ]
    for split in splits:
        train_values = split["train_values"]
        test_values = split["test_values"]
        for seed in args.seeds:
            direct_edges = sync_edges_for_seed(spaces, args.tasks, seed, train_values, args)
            full_orientations, _rotation_stats = synchronize_rotations(args.tasks, direct_edges, dim_l)
            full_log_scales, _scale_stats = synchronize_scales(args.tasks, direct_edges)
            for edge in direct_edges:
                source_space = spaces[(edge.source, seed)]
                destination_space = spaces[(edge.destination, seed)]
                rows.append(
                    evaluate_relation_row(
                        source_space,
                        destination_space,
                        edge.q,
                        edge.alpha,
                        train_values,
                        test_values,
                        {
                            "transport": "direct_pairwise",
                            "split_seed": split["split_seed"],
                            "sync_seed": seed,
                            "source_task": edge.source,
                            "destination_task": edge.destination,
                            "heldout_relation": "none",
                            "n_edges": len(direct_edges),
                        },
                    )
                )
                q, alpha = hub_map(edge.source, edge.destination, full_orientations, full_log_scales)
                rows.append(
                    evaluate_relation_row(
                        source_space,
                        destination_space,
                        q,
                        alpha,
                        train_values,
                        test_values,
                        {
                            "transport": "full_synchronization",
                            "split_seed": split["split_seed"],
                            "sync_seed": seed,
                            "source_task": edge.source,
                            "destination_task": edge.destination,
                            "heldout_relation": "none",
                            "n_edges": len(direct_edges),
                        },
                    )
                )
            for relation in undirected_relations:
                loro_edges = sync_edges_for_seed(spaces, args.tasks, seed, train_values, args, drop_pair=relation)
                components = connected_components(args.tasks, loro_edges)
                if len(components) != 1:
                    raise RuntimeError(f"LORO graph disconnected for relation={relation} seed={seed}: {components}")
                loro_orientations, _ = synchronize_rotations(args.tasks, loro_edges, dim_l)
                loro_log_scales, _ = synchronize_scales(args.tasks, loro_edges)
                for source_task, destination_task in (relation, (relation[1], relation[0])):
                    q, alpha = hub_map(source_task, destination_task, loro_orientations, loro_log_scales)
                    rows.append(
                        evaluate_relation_row(
                            spaces[(source_task, seed)],
                            spaces[(destination_task, seed)],
                            q,
                            alpha,
                            train_values,
                            test_values,
                            {
                                "transport": "loro_synchronization",
                                "split_seed": split["split_seed"],
                                "sync_seed": seed,
                                "source_task": source_task,
                                "destination_task": destination_task,
                                "heldout_relation": f"{relation[0]}<->{relation[1]}",
                                "n_edges": len(loro_edges),
                            },
                        )
                    )
    return rows


def summarize_space(rows: list[dict], space_type: str, rsa_rows: list[dict]) -> dict:
    selected = [row for row in rows if row.get("space_type") == space_type]
    rsa_selected = [row for row in rsa_rows if row.get("space_type") == space_type]
    return {
        "transition_cosine": mean([row["heldout_transition_cosine"] for row in selected]),
        "top1": mean([row["top1"] for row in selected]),
        "top5": mean([row["top5"] for row in selected]),
        "rsa": mean([row["rsa_spearman"] for row in rsa_selected]),
    }


def aggregate_by(rows: list[dict], key: str) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return {
        group: {
            "heldout_transition_cosine_mean": mean([row["heldout_transition_cosine"] for row in parts]),
            "heldout_transition_cosine_std": std([row["heldout_transition_cosine"] for row in parts]),
            "top1_mean": mean([row["top1"] for row in parts]),
            "top5_mean": mean([row["top5"] for row in parts]),
            "n": len(parts),
        }
        for group, parts in sorted(grouped.items())
    }


def save_heatmap(path: Path, tasks: list[str], rows: list[dict], value_key: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = torch.full((len(tasks), len(tasks)), float("nan"))
    for i, source in enumerate(tasks):
        for j, destination in enumerate(tasks):
            if source == destination:
                continue
            values = [
                float(row[value_key])
                for row in rows
                if row.get("source_task") == source and row.get("destination_task") == destination and row.get(value_key) is not None
            ]
            if values:
                matrix[i, j] = sum(values) / len(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(matrix.numpy(), vmin=-1.0 if "cosine" in value_key or "rsa" in value_key else 0.0, vmax=1.0)
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_yticks(range(len(tasks)), tasks)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Source")
    ax.set_title(title)
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if i != j and math.isfinite(float(matrix[i, j])):
                ax.text(j, i, f"{float(matrix[i, j]):.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def final_tables(pairwise_rows, control_rows, rsa_rows, sync_rows) -> None:
    all_geometry = pairwise_rows + control_rows
    summaries = {
        "L": summarize_space(all_geometry, "L", rsa_rows),
        "random-readout-free": summarize_space(all_geometry, "random_readout_free", rsa_rows),
        "shuffled": summarize_space(all_geometry, "shuffled", rsa_rows),
    }
    print("\nL GEOMETRY")
    print_table(
        ["Metric", "L", "random-readout-free", "shuffled"],
        [
            ["transition cosine", fmt(summaries["L"]["transition_cosine"]), fmt(summaries["random-readout-free"]["transition_cosine"]), fmt(summaries["shuffled"]["transition_cosine"])],
            ["top1", fmt(summaries["L"]["top1"]), fmt(summaries["random-readout-free"]["top1"]), fmt(summaries["shuffled"]["top1"])],
            ["top5", fmt(summaries["L"]["top5"]), fmt(summaries["random-readout-free"]["top5"]), fmt(summaries["shuffled"]["top5"])],
            ["RSA", fmt(summaries["L"]["rsa"]), fmt(summaries["random-readout-free"]["rsa"]), fmt(summaries["shuffled"]["rsa"])],
        ],
    )
    by_transport = aggregate_by(sync_rows, "transport")
    print("\nCOMMON FRAME")
    print_table(
        ["Transport", "transition cosine", "top1", "top5"],
        [
            [
                key,
                fmt(value["heldout_transition_cosine_mean"]),
                fmt(value["top1_mean"]),
                fmt(value["top5_mean"]),
            ]
            for key, value in by_transport.items()
        ],
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 2 Ministral readout geometry")
    print(f"  model={args.model}")
    print(f"  DAS root={args.das_root}")
    print(f"  output={args.output_dir}")
    print(f"  layer={args.layer} k={args.k} position={args.position}")

    if args.artifact_check_only:
        find_final_artifacts(args, hidden_size=None)
        return 0

    find_final_artifacts(args, hidden_size=None)

    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print(f"Loading model from {model_path}")
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    blocks = get_blocks(model)
    hidden_size = get_hidden_size(model)
    print(f"Loaded {saved_model_name}; hidden_size={hidden_size}; blocks={len(blocks)}")

    artifacts = find_final_artifacts(args, hidden_size=hidden_size)
    u_digit, readout_info = digit_readout_basis(model, tokenizer, args)
    write_json(args.output_dir / "decomposition" / "digit_readout.json", readout_info)
    torch.save(u_digit.cpu(), args.output_dir / "decomposition" / "digit_readout_basis.pt")
    print(f"Digit readout rank={readout_info['digit_rank']} ids={readout_info['digit_ids']}")

    spaces, decomp_rows = decompose_artifacts(args, artifacts, u_digit)

    activations = {}
    for task in args.tasks:
        activations[task] = load_or_collect_activations(args, task, artifacts, model, processor, tokenizer, blocks)
    common_values, missing_by_space = add_centroids(args, spaces, activations)

    splits = value_splits(common_values, args.value_split_seeds, args.train_value_fraction)
    write_json(args.output_dir / "value_splits.json", splits)
    print(f"Value splits: {[(split['split_seed'], len(split['train_values']), len(split['test_values'])) for split in splits]}")

    print("\nRunning value-held-out pairwise L geometry")
    pairwise_rows = run_pairwise_geometry(args, spaces, splits)
    write_csv(args.output_dir / "pairwise_geometry.csv", pairwise_rows)
    write_json(args.output_dir / "pairwise_geometry.json", pairwise_rows)

    dim_l = next(iter(spaces.values())).l_basis.shape[1]
    print("Running random readout-free control")
    random_spaces = make_random_spaces(args, activations, u_digit, hidden_size, dim_l)
    random_rows = run_random_control(args, random_spaces, splits)
    print("Running shuffled numerical correspondence control")
    shuffled_rows = run_shuffled_control(args, spaces, splits)
    control_rows = random_rows + shuffled_rows
    write_csv(args.output_dir / "controls.csv", control_rows)
    write_json(args.output_dir / "controls.json", control_rows)

    print("Running no-fit RSA")
    spaces_by_type = {"L": spaces, "random_readout_free": random_spaces, "shuffled": spaces}
    rsa_rows = run_rsa(args, spaces_by_type, splits)
    write_csv(args.output_dir / "rsa.csv", rsa_rows)
    write_json(args.output_dir / "rsa.json", rsa_rows)

    print("Running matched-seed common-frame synchronization and LORO")
    sync_rows = run_synchronization(args, spaces, splits)
    write_csv(args.output_dir / "synchronization.csv", sync_rows)
    write_json(args.output_dir / "synchronization.json", sync_rows)

    if not args.skip_heatmaps:
        save_heatmap(args.output_dir / "heatmaps" / "pairwise_transition_cosine.png", args.tasks, pairwise_rows, "heldout_transition_cosine", "Held-out transition cosine")
        save_heatmap(args.output_dir / "heatmaps" / "pairwise_top1.png", args.tasks, pairwise_rows, "top1", "Held-out top1")
        l_rsa_rows = [row for row in rsa_rows if row.get("space_type") == "L"]
        save_heatmap(args.output_dir / "heatmaps" / "nofit_rsa.png", args.tasks, l_rsa_rows, "rsa_spearman", "No-fit RSA")

    summary = {
        "config": vars(args),
        "model": {"saved_model_name": saved_model_name, "hidden_size": hidden_size, "n_blocks": len(blocks)},
        "digit_readout": readout_info,
        "decomposition": decomp_rows,
        "common_values": common_values,
        "missing_0_99_by_space": missing_by_space,
        "pairwise_geometry": aggregate_by(pairwise_rows, "space_type"),
        "controls": aggregate_by(control_rows, "space_type"),
        "rsa": {
            space_type: {
                "rsa_spearman_mean": mean([row["rsa_spearman"] for row in rsa_rows if row["space_type"] == space_type]),
                "rsa_spearman_std": std([row["rsa_spearman"] for row in rsa_rows if row["space_type"] == space_type]),
                "rsa_p_mean": mean([row["rsa_permutation_p"] for row in rsa_rows if row["space_type"] == space_type]),
            }
            for space_type in ("L", "random_readout_free", "shuffled")
        },
        "synchronization": aggregate_by(sync_rows, "transport"),
        "artifact_hashes": {
            f"{artifact.task}_seed{artifact.seed}": {
                "config_hash": artifact.digest,
                "artifact_hash": artifact_hash_from_metadata(artifact.metadata, artifact.run_dir),
                "run_dir": str(artifact.run_dir),
            }
            for artifact in artifacts.values()
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    final_tables(pairwise_rows, control_rows, rsa_rows, sync_rows)
    print(f"\nSaved Stage 2 outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
