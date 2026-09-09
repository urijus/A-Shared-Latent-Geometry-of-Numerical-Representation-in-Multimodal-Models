"""Experiment 2A: natural donor patching in existing Fourier result subspaces.

For text addition/subtraction, this script loads already-trained Fourier ridge
probes, builds the combined Fourier subspace at each requested layer and
semantic position, and evaluates donor interchange interventions:

    h_base <- h_base + F F^T (h_donor - h_base)

It reuses the repository's DAS intervention/evaluation machinery by wrapping
each Fourier basis in ``DASSubspace``. No probes or DAS spaces are retrained.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from types import SimpleNamespace
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import load_jsonl
from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
    format_prompt,
    patched_forward,
    resolve_position,
    teacher_forced_batch,
)
from src.experiments.arithmetic_reference.das_audit.audit_das import clean_autoregressive_iia_text
from src.experiments.arithmetic_reference.fourier_probes.text.project_fourier import run_projection_for_activation
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.probes.fourier import probe_path, safe_name
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    model_slug,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_TASKS = ("text_add", "text_sub")
TASKS = {
    "text_add": ("addition", "T+"),
    "text_addition": ("addition", "T+"),
    "text:addition": ("addition", "T+"),
    "addition": ("addition", "T+"),
    "text_sub": ("subtraction", "T-"),
    "text_subtraction": ("subtraction", "T-"),
    "text:subtraction": ("subtraction", "T-"),
    "subtraction": ("subtraction", "T-"),
}
DEFAULT_LAYERS = [34, 35, 36, 38, 40, 43, 44]
DEFAULT_POSITIONS = list(range(10, 18))
SMOKE_POSITIONS = [11, 13, 15, 17]
PERIODS = [2, 5, 10, 20, 50, 100]
POSITION_LABELS = {
    10: "second_operand",
    11: "equals",
    12: "<turn|>",
    13: "<|turn>",
    14: "model",
    15: "<|channel>",
    16: "thought",
    17: "final_pre_generation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--probe_root", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/fourier_donor_patch_2a"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--positions", type=int, nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--periods", type=int, nargs="+", default=PERIODS)
    parser.add_argument("--method", choices=["ridge", "gd"], default="ridge")
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--pair_seed", type=int, default=0, help="DAS seed directory whose held-out pairs are reused.")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--max_pairs", type=int, default=0, help="0 means use all saved held-out pairs.")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--random_controls", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=2718)
    parser.add_argument(
        "--random_control_space",
        choices=["residual"],
        default="residual",
        help="Currently uses deterministic rank-matched random residual-stream subspaces.",
    )
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--skip_autoregressive", action="store_true")
    parser.add_argument("--skip_teacher_forced", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--skip_missing_probes", action="store_true")
    parser.add_argument(
        "--auto_train_missing_probes",
        action="store_true",
        help="Materialize missing Fourier probe weight files with the existing project_fourier procedure before evaluation.",
    )
    parser.add_argument("--activation_dir", type=Path)
    parser.add_argument("--probe_epochs", type=int, default=300)
    parser.add_argument("--probe_batch_size", type=int, default=4096)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_val_fraction", type=float, default=0.2)
    parser.add_argument("--probe_split_mode", choices=["random", "operand", "sum"], default="random")
    parser.add_argument("--probe_seed", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--no_weight_by_residue", action="store_false", dest="weight_by_residue")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--sanity_pairs", type=int, default=2)
    parser.set_defaults(weight_by_residue=True)
    return parser.parse_args()


def normalize_tasks(raw_tasks: list[str]) -> list[dict]:
    if any(item.lower() == "all" for item in raw_tasks):
        raw_tasks = list(DEFAULT_TASKS)
    tasks = []
    seen = set()
    for raw in raw_tasks:
        key = raw.lower()
        if key not in TASKS:
            raise ValueError(f"Unknown task {raw!r}; choose text_add, text_sub, or all.")
        operation, label = TASKS[key]
        if operation in seen:
            continue
        seen.add(operation)
        tasks.append({"task": f"text_{operation}", "operation": operation, "task_label": label})
    return tasks


def default_probe_root(model_name: str) -> Path:
    return Path("results") / "baseline" / model_slug(model_name) / "digits" / "fourier_probes"


def default_activation_dir(model_name: str) -> Path:
    return Path("outputs") / "activations" / "baseline" / model_slug(model_name) / "digits"


def audit_run_dir(args: argparse.Namespace, operation: str) -> Path:
    root = args.audit_root / "text" / operation
    if args.target != "result":
        root = root / args.target
    return root / args.condition / f"split_{args.split_seed}" / f"seed_{args.pair_seed}"


def sample_lookup(data_path: Path) -> dict:
    samples = {}
    for index, sample in enumerate(load_jsonl(data_path)):
        key = sample.get("sample_id", index)
        samples[key] = sample
        samples[str(key)] = sample
    return samples


def load_pairs(run_dir: Path, data_path: Path, max_pairs: int) -> list[dict]:
    heldout = run_dir / "heldout_pairs.jsonl"
    if not heldout.exists():
        raise FileNotFoundError(heldout)
    samples = sample_lookup(data_path)
    pairs = []
    for row in load_jsonl(heldout):
        pairs.append(
            {
                "pair_id": row.get("pair_id", len(pairs)),
                "base": samples[row["base_sample_id"]],
                "source": samples[row["source_sample_id"]],
            }
        )
    return pairs[:max_pairs] if max_pairs and max_pairs > 0 else pairs


def read_first_jsonl(path: Path) -> dict:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}.")
    return rows[0]


def orthonormal_columns(matrix: torch.Tensor, name: str) -> torch.Tensor:
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be rank-2; got {tuple(matrix.shape)}.")
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    if rank == 0:
        raise ValueError(f"{name} has numerical rank zero.")
    return left[:, :rank]


def orthogonality_error(basis: torch.Tensor) -> float:
    eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    return float((basis.T @ basis - eye).abs().max())


def load_probe_artifact(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def flat_probe_path(
    probe_root: Path,
    operation: str,
    target: str,
    period: int,
    layer: int,
    position: int,
    method: str,
) -> Path:
    stem = safe_name(f"{target}_T{period}_layer{layer}_pos{position}_{method}_probe.pt")
    return (
        probe_root
        / operation
        / "fourier_projections"
        / f"{target}_pos{position}_{method}"
        / "probes"
        / stem
    )


def resolve_probe_path(
    probe_root: Path,
    operation: str,
    target: str,
    period: int,
    layer: int,
    position: int,
    method: str,
) -> Path:
    nested = probe_path(probe_root, operation, target, period, layer, position, method)
    if nested.exists():
        return nested
    flat = flat_probe_path(probe_root, operation, target, period, layer, position, method)
    if flat.exists():
        return flat
    return nested


def valid_probe_rows(weight: torch.Tensor, period: int) -> torch.Tensor:
    weight = torch.as_tensor(weight).detach().float().squeeze()
    if weight.ndim == 1:
        weight = weight[None, :]
    if period == 2:
        return weight[:1]
    return weight


def combined_fourier_basis(
    probe_root: Path,
    operation: str,
    target: str,
    periods: list[int],
    layer: int,
    position: int,
    method: str,
) -> tuple[torch.Tensor, dict]:
    rows = []
    paths = []
    period_metrics = {}
    for period in periods:
        path = resolve_probe_path(probe_root, operation, target, period, layer, position, method)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Fourier probe {path}. Existing probes must be generated before Experiment 2A."
            )
        artifact = load_probe_artifact(path)
        weight_rows = valid_probe_rows(artifact["weight"], period)
        rows.append(weight_rows)
        paths.append(str(path))
        period_metrics[str(period)] = artifact.get("metrics", {})
    stacked = torch.cat(rows, dim=0)
    basis = orthonormal_columns(stacked.T, f"combined Fourier {operation} layer={layer} pos={position}")
    return basis, {"probe_paths": paths, "period_metrics": period_metrics}


def missing_probe_paths(args: argparse.Namespace, probe_root: Path, tasks: list[dict]) -> list[Path]:
    missing = []
    for task in tasks:
        for layer in args.layers:
            for position in args.positions:
                for period in args.periods:
                    path = resolve_probe_path(
                        probe_root,
                        task["operation"],
                        args.target,
                        period,
                        layer,
                        position,
                        args.method,
                    )
                    if not path.exists():
                        missing.append(path)
    return missing


def missing_sites(args: argparse.Namespace, probe_root: Path, tasks: list[dict]) -> dict[tuple[str, int], set[int]]:
    sites = defaultdict(set)
    for task in tasks:
        for layer in args.layers:
            for position in args.positions:
                if any(
                    not resolve_probe_path(
                        probe_root,
                        task["operation"],
                        args.target,
                        period,
                        layer,
                        position,
                        args.method,
                    ).exists()
                    for period in args.periods
                ):
                    sites[(task["operation"], position)].add(layer)
    return sites


def train_missing_probe_weights(args: argparse.Namespace, probe_root: Path, tasks: list[dict]) -> None:
    activation_dir = args.activation_dir or default_activation_dir(args.model)
    sites = missing_sites(args, probe_root, tasks)
    if not sites:
        return
    print("\nTRAINING_MISSING_FOURIER_PROBES")
    print("Reason: one or more requested Fourier probe weight .pt files were absent.")
    print("Procedure: existing src.experiments.arithmetic_reference.fourier_probes.text.project_fourier")
    print(
        "Config: "
        f"activation_dir={activation_dir} probe_root={probe_root} "
        f"method={args.method} periods={args.periods} "
        f"epochs={args.probe_epochs} batch_size={args.probe_batch_size} "
        f"lr={args.probe_lr} split_mode={args.probe_split_mode} "
        f"weight_by_residue={args.weight_by_residue}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for (operation, position), layer_set in sorted(sites.items()):
        activation_path = activation_dir / f"{operation}_baseline.pt"
        if not activation_path.exists():
            raise FileNotFoundError(
                f"Cannot train missing Fourier probes because activation file is missing: {activation_path}. "
                "Run the existing activation extraction first."
            )
        layers = sorted(layer_set)
        projection_root = probe_root / operation / "fourier_projections" / f"{args.target}_pos{position}_{args.method}"
        # Match project_fourier.py's normal output layout:
        #   probes/result_addition/*.pt
        #   projections/result_addition/*.pt
        output_name = f"{args.target}_{operation}"
        probe_dir = projection_root / "probes" / output_name
        projection_dir = projection_root / "projections" / output_name
        saved_probe_dir = probe_root / operation / "best_probes"
        print(
            "TRAINING_MISSING_FOURIER_PROBES site: "
            f"task={operation} position={position} layers={layers} "
            f"activation_path={activation_path}"
        )
        project_args = SimpleNamespace(
            activation_dir=activation_dir,
            modalities=[operation],
            probe_output_dir=probe_dir,
            saved_probe_dir=saved_probe_dir,
            projection_output_dir=projection_dir,
            target=args.target,
            periods=args.periods,
            layers=layers,
            position=position,
            epochs=args.probe_epochs,
            batch_size=args.probe_batch_size,
            lr=args.probe_lr,
            val_fraction=args.probe_val_fraction,
            seed=args.probe_seed,
            use_ridge=args.method == "ridge",
            early_stopping_patience=args.early_stopping_patience,
            weight_by_residue=args.weight_by_residue,
            split_mode=args.probe_split_mode,
        )
        run_projection_for_activation(
            args=project_args,
            activation_path=activation_path,
            probe_dir=probe_dir,
            projection_dir=projection_dir,
            device=device,
            modality=operation,
        )
        print(
            "FINISHED_TRAINING_MISSING_FOURIER_PROBES site: "
            f"task={operation} position={position} layers={layers}"
        )
    print("FINISHED_TRAINING_MISSING_FOURIER_PROBES\n")


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_probe_score_rows(probe_root: Path, operation: str, method: str) -> dict:
    path = probe_root / operation / f"add_sub_fourier_results_{method}.jsonl"
    if not path.exists():
        return {}
    lookup = {}
    for row in load_jsonl(path):
        if row.get("target") != "result":
            continue
        if row.get("optimization_method") != method:
            continue
        key = (int(row["layer"]), int(row.get("position_name", row["position"])), int(row["period"]))
        lookup[key] = row
    return lookup


def probe_r2_fields(score_lookup: dict, periods: list[int], layer: int, position: int) -> dict:
    values = []
    result = {}
    for period in periods:
        row = score_lookup.get((layer, position, period))
        if row is None:
            continue
        value = float(row["r2_cos"]) if period == 2 else float(row["r2_mean"])
        values.append(value)
        if period == 10:
            result["fourier_T10_r2"] = float(row["r2_mean"])
    result["mean_fourier_r2"] = None if not values else float(sum(values) / len(values))
    result.setdefault("fourier_T10_r2", None)
    return result


def subspace_from_basis(layer: int, basis: torch.Tensor, device: torch.device) -> dict:
    basis = basis.detach().float()
    return {str(layer): DASSubspace(basis.shape[0], basis.shape[1], initial_basis=basis).to(device)}


def evaluate_teacher_forced_iia(
    args: argparse.Namespace,
    model,
    tokenizer,
    blocks,
    pairs: list[dict],
    layer: int,
    position: int,
    basis: torch.Tensor,
) -> float | None:
    if args.skip_teacher_forced:
        return None
    metrics = teacher_forced_metrics(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=subspace_from_basis(layer, basis, model.device),
        layer=layer,
        hook_name=args.hook,
        pairs=pairs,
        position=str(position),
        target=args.target,
        use_chat_template=args.use_chat_template or uses_chat_template(args.model),
        batch_size=args.batch_size,
        patch=True,
    )
    return metrics["variable_teacher_forced_iia"]


@torch.no_grad()
def teacher_forced_metrics(
    model,
    tokenizer,
    blocks,
    subspaces,
    layer: int,
    hook_name: str,
    pairs: list[dict],
    position: str,
    target: str,
    use_chat_template: bool,
    batch_size: int,
    patch: bool,
) -> dict:
    names = ("variable_teacher_forced_iia", "full_answer_teacher_forced_iia")
    parts = {name: [] for name in names}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        result = teacher_forced_batch(
            model,
            tokenizer,
            blocks,
            subspaces,
            [layer],
            hook_name,
            batch,
            position,
            target,
            use_chat_template,
            patch=patch,
        )
        for name in names:
            parts[name].append(result[name].detach().cpu())
    return {name: float(torch.cat(items).mean()) for name, items in parts.items()}


def evaluate_ar_iia(
    args: argparse.Namespace,
    model,
    tokenizer,
    blocks,
    pairs: list[dict],
    layer: int,
    position: int,
    basis: torch.Tensor,
    description: str,
) -> float | None:
    if args.skip_autoregressive:
        return None
    return autoregressive_iia(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=subspace_from_basis(layer, basis, model.device),
        layers=[layer],
        hook_name=args.hook,
        pairs=pairs,
        position=str(position),
        target=args.target,
        use_chat_template=args.use_chat_template or uses_chat_template(args.model),
        max_new_tokens=args.max_new_tokens,
        description=description,
    )


def clean_tf_iia(args, model, tokenizer, blocks, pairs: list[dict], layer: int, position: int) -> float | None:
    if args.skip_teacher_forced:
        return None
    dummy = subspace_from_basis(layer, torch.eye(get_hidden_size(model), 1), model.device)
    metrics = teacher_forced_metrics(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=dummy,
        layer=layer,
        hook_name=args.hook,
        pairs=pairs,
        position=str(position),
        target=args.target,
        use_chat_template=args.use_chat_template or uses_chat_template(args.model),
        batch_size=args.batch_size,
        patch=False,
    )
    return metrics["variable_teacher_forced_iia"]


def random_basis(hidden_size: int, rank: int, seed: int) -> torch.Tensor:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        basis, _ = torch.linalg.qr(torch.randn(hidden_size, rank), mode="reduced")
    finally:
        torch.random.set_rng_state(state)
    return basis[:, :rank].float()


@torch.no_grad()
def capture_pair_activations(args, model, tokenizer, blocks, pair: dict, layer: int, position: int) -> tuple[torch.Tensor, torch.Tensor]:
    base = pair["base"]
    donor = pair.get("donor", pair["source"])
    use_chat = args.use_chat_template or uses_chat_template(args.model)
    prompts = [
        format_prompt(tokenizer, base, use_chat),
        format_prompt(tokenizer, donor, use_chat),
    ]
    positions = [resolve_position(tokenizer, prompt, str(position)) for prompt in prompts]
    encoding = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    captured = {}

    def capture(_module, _inputs, output):
        captured["hidden"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = blocks[layer - 1].register_forward_hook(capture)
    try:
        model(**encoding, use_cache=False)
    finally:
        handle.remove()
    hidden = captured["hidden"]
    return hidden[0, positions[0]].float().cpu(), hidden[1, positions[1]].float().cpu()


def patch_sanity_checks(args, model, tokenizer, blocks, pairs: list[dict], basis: torch.Tensor, layer: int, position: int) -> dict:
    subspace = DASSubspace(basis.shape[0], basis.shape[1], initial_basis=basis)
    coord_errors = []
    orth_errors = []
    noop_errors = []
    for pair in pairs[: max(0, args.sanity_pairs)]:
        base, donor = capture_pair_activations(args, model, tokenizer, blocks, pair, layer, position)
        patched = subspace.patch(base, donor)
        coord_errors.append(float((basis.T @ patched - basis.T @ donor).norm()))
        orth_base = base - basis @ (basis.T @ base)
        orth_patched = patched - basis @ (basis.T @ patched)
        orth_errors.append(float((orth_patched - orth_base).norm()))
        noop_errors.append(float((subspace.patch(base, base) - base).norm()))
    return {
        "coord_replacement_error": max(coord_errors) if coord_errors else None,
        "orthogonal_preservation_error": max(orth_errors) if orth_errors else None,
        "noop_error": max(noop_errors) if noop_errors else None,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def condition_values(rows: list[dict], task: str, condition: str, layer: int, position: int) -> list[float]:
    return [
        float(row["ar_iia"])
        for row in rows
        if row["task"] == task
        and row["condition"] == condition
        and int(row["layer"]) == layer
        and int(row["position"]) == position
        and row.get("ar_iia") is not None
        and row.get("ar_iia") != ""
    ]


def summarize(rows: list[dict], tasks: list[dict]) -> dict:
    summary = {}
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task["task"]]
        fourier_rows = [row for row in task_rows if row["condition"] == "fourier" and row.get("ar_iia") is not None]
        tf_rows = [row for row in task_rows if row["condition"] == "fourier" and row.get("tf_iia") is not None]
        item = {}
        if fourier_rows:
            best = max(fourier_rows, key=lambda row: float(row["ar_iia"]))
            item["max_fourier_ar_iia"] = float(best["ar_iia"])
            item["max_fourier_ar_site"] = {"layer": int(best["layer"]), "position": int(best["position"])}
        if tf_rows:
            best_tf = max(tf_rows, key=lambda row: float(row["tf_iia"]))
            item["max_fourier_tf_iia"] = float(best_tf["tf_iia"])
            item["max_fourier_tf_site"] = {"layer": int(best_tf["layer"]), "position": int(best_tf["position"])}
        for layer, position in [(43, 17), (35, 11), (36, 11), (38, 11)]:
            vals = condition_values(rows, task["task"], "fourier", layer, position)
            random_vals = condition_values(rows, task["task"], "random", layer, position)
            item[f"fourier_ar_iia_L{layer}_P{position}"] = vals[0] if vals else None
            item[f"random_ar_iia_L{layer}_P{position}_mean"] = None if not random_vals else sum(random_vals) / len(random_vals)
        summary[task["task"]] = item
    return summary


def matrix_for(rows: list[dict], task: str, condition: str, metric: str, layers: list[int], positions: list[int], random_mean: bool = False):
    import numpy as np

    matrix = np.full((len(positions), len(layers)), np.nan)
    for row_index, position in enumerate(positions):
        for col_index, layer in enumerate(layers):
            values = [
                float(row[metric])
                for row in rows
                if row["task"] == task
                and row["condition"] == condition
                and int(row["layer"]) == layer
                and int(row["position"]) == position
                and row.get(metric) is not None
                and row.get(metric) != ""
            ]
            if values:
                matrix[row_index, col_index] = sum(values) / len(values) if random_mean else values[0]
    return matrix


def plot_heatmap(matrix, layers: list[int], positions: list[int], title: str, cbar_label: str, output_path: Path, vmin: float = 0.0, vmax: float = 1.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    image = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Semantic token position")
    ax.set_xticks(range(len(layers)), [str(layer) for layer in layers])
    ax.set_yticks(range(len(positions)), [str(position) for position in positions])
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_all(rows: list[dict], tasks: list[dict], layers: list[int], positions: list[int], output_dir: Path) -> None:
    import numpy as np

    if not rows:
        return
    for task in tasks:
        task_name = task["task"]
        label = task["task_label"]
        ar = matrix_for(rows, task_name, "fourier", "ar_iia", layers, positions)
        tf = matrix_for(rows, task_name, "fourier", "tf_iia", layers, positions)
        random_ar = matrix_for(rows, task_name, "random", "ar_iia", layers, positions, random_mean=True)
        plot_heatmap(ar, layers, positions, f"{label} Fourier donor patch AR IIA", "AR IIA", output_dir / f"{task_name}_fourier_donor_patch_ar_iia.png")
        plot_heatmap(tf, layers, positions, f"{label} Fourier donor patch TF IIA", "TF IIA", output_dir / f"{task_name}_fourier_donor_patch_tf_iia.png")
        if not np.isnan(random_ar).all():
            diff = ar - random_ar
            finite = diff[~np.isnan(diff)]
            limit = max(abs(float(finite.min())), abs(float(finite.max()))) if finite.size else 1.0
            plot_heatmap(
                diff,
                layers,
                positions,
                f"{label} Fourier minus random AR IIA",
                "AR IIA difference",
                output_dir / f"{task_name}_fourier_minus_random_ar_iia.png",
                vmin=-limit,
                vmax=limit,
            )


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        if args.positions == DEFAULT_POSITIONS:
            args.positions = SMOKE_POSITIONS
        if args.max_pairs == 0:
            args.max_pairs = 8
    tasks = normalize_tasks(args.tasks)
    probe_root = args.probe_root or default_probe_root(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = missing_probe_paths(args, probe_root, tasks)
    if missing and args.auto_train_missing_probes:
        train_missing_probe_weights(args, probe_root, tasks)
        missing = missing_probe_paths(args, probe_root, tasks)
    if missing and not args.skip_missing_probes:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} Fourier probe weight files for the requested grid. "
            "Run/restore the matching Fourier probes first, narrow --layers/--positions, "
            "or pass --skip_missing_probes to evaluate only available sites.\n"
            f"First missing paths:\n{preview}"
        )
    if missing:
        print(f"Warning: skipping sites with missing probe weights; missing files={len(missing)}")

    print("\nSTARTING_FOURIER_DONOR_PATCH_CAUSAL_EVALUATION")
    print("All requested probe weight files are available; loading Gemma for causal patching.")

    model_path, resolved_model = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, args.layers)
    hidden_size = get_hidden_size(model)

    all_rows = []
    print(
        "fourier_donor_patch config: "
        f"model={resolved_model} tasks={[task['task'] for task in tasks]} "
        f"layers={args.layers} positions={args.positions} max_pairs={args.max_pairs} "
        f"random_controls={args.random_controls}"
    )

    for task in tasks:
        run_dir = audit_run_dir(args, task["operation"])
        result_row = read_first_jsonl(run_dir / "results.jsonl")
        data_path = Path(result_row["data_path"])
        pairs = load_pairs(run_dir, data_path, args.max_pairs)
        score_lookup = load_probe_score_rows(probe_root, task["operation"], args.method)
        print(f"\nLoaded {task['task']} pairs from {run_dir}: n_pairs={len(pairs)}")

        baseline_ar = None
        if not args.skip_autoregressive:
            baseline_ar = clean_autoregressive_iia_text(
                model,
                tokenizer,
                pairs,
                args.target,
                args.use_chat_template or uses_chat_template(args.model),
                args.max_new_tokens,
            )
        baseline_tf_by_position = {}
        if not args.skip_teacher_forced:
            # Clean TF depends on answer-token alignment but not on layer. Use the
            # first requested layer for the no-hook forward pass helper.
            for position in args.positions:
                baseline_tf_by_position[position] = clean_tf_iia(
                    args, model, tokenizer, blocks, pairs, args.layers[0], position
                )

        for layer in args.layers:
            for position in args.positions:
                try:
                    basis, probe_meta = combined_fourier_basis(
                        probe_root,
                        task["operation"],
                        args.target,
                        args.periods,
                        layer,
                        position,
                        args.method,
                    )
                except FileNotFoundError as error:
                    if not args.skip_missing_probes:
                        raise
                    r2 = probe_r2_fields(score_lookup, args.periods, layer, position)
                    all_rows.append(
                        {
                            "task": task["task"],
                            "task_label": task["task_label"],
                            "layer": layer,
                            "position": position,
                            "position_label": POSITION_LABELS.get(position, str(position)),
                            "condition": "missing_probe",
                            "control_seed": None,
                            "fourier_rank": None,
                            "n_pairs": len(pairs),
                            "ar_iia": None,
                            "tf_iia": None,
                            "missing_probe_error": str(error),
                            **r2,
                        }
                    )
                    print(f"  skipping missing probe site: {task['task']} layer={layer} position={position}")
                    continue
                if basis.shape[0] != hidden_size:
                    raise ValueError(
                        f"Fourier basis d_model={basis.shape[0]} does not match model hidden_size={hidden_size}."
                    )
                orth_error = orthogonality_error(basis)
                if orth_error > 1e-4:
                    raise RuntimeError(f"Fourier basis not orthonormal: error={orth_error:.3e}")
                sanity = patch_sanity_checks(args, model, tokenizer, blocks, pairs, basis, layer, position)
                print(
                    f"  {task['task']} layer={layer} position={position} "
                    f"rank={basis.shape[1]} orth_error={orth_error:.3e} "
                    f"coord_err={sanity['coord_replacement_error']} "
                    f"orth_preserve_err={sanity['orthogonal_preservation_error']}"
                )
                r2 = probe_r2_fields(score_lookup, args.periods, layer, position)
                ar_iia = evaluate_ar_iia(
                    args,
                    model,
                    tokenizer,
                    blocks,
                    pairs,
                    layer,
                    position,
                    basis,
                    f"{task['task']} Fourier donor patch L{layer} P{position}",
                )
                tf_iia = evaluate_teacher_forced_iia(
                    args, model, tokenizer, blocks, pairs, layer, position, basis
                )
                site_common = {
                    "task": task["task"],
                    "task_label": task["task_label"],
                    "layer": layer,
                    "position": position,
                    "position_label": POSITION_LABELS.get(position, str(position)),
                    "fourier_rank": basis.shape[1],
                    "n_pairs": len(pairs),
                    "target": args.target,
                    "hook": args.hook,
                    "method": args.method,
                    "random_control_space": None,
                    "basis_orthonormality_error": orth_error,
                    **sanity,
                    **r2,
                }
                all_rows.append(
                    {
                        **site_common,
                        "condition": "fourier",
                        "control_seed": None,
                        "ar_iia": ar_iia,
                        "tf_iia": tf_iia,
                    }
                )
                all_rows.append(
                    {
                        **site_common,
                        "condition": "no_intervention",
                        "control_seed": None,
                        "ar_iia": baseline_ar,
                        "tf_iia": baseline_tf_by_position.get(position),
                    }
                )
                for control_index in range(args.random_controls):
                    control_seed = args.random_seed + 10000 * layer + 100 * position + control_index
                    control_basis = random_basis(hidden_size, basis.shape[1], control_seed)
                    control_ar = evaluate_ar_iia(
                        args,
                        model,
                        tokenizer,
                        blocks,
                        pairs,
                        layer,
                        position,
                        control_basis,
                        f"{task['task']} random donor patch L{layer} P{position} seed={control_seed}",
                    )
                    control_tf = evaluate_teacher_forced_iia(
                        args, model, tokenizer, blocks, pairs, layer, position, control_basis
                    )
                    all_rows.append(
                        {
                            **site_common,
                            "condition": "random",
                            "control_seed": control_seed,
                            "ar_iia": control_ar,
                            "tf_iia": control_tf,
                            "random_control_space": args.random_control_space,
                        }
                    )

    result_path = args.output_dir / "fourier_donor_patch_results.csv"
    write_csv(result_path, all_rows)
    summary = {
        "model": resolved_model,
        "audit_root": str(args.audit_root),
        "probe_root": str(probe_root),
        "layers": args.layers,
        "positions": args.positions,
        "periods": args.periods,
        "random_controls": args.random_controls,
        "summary_by_task": summarize(all_rows, tasks),
        "config": jsonable(vars(args)),
    }
    write_json(args.output_dir / "summary.json", summary)
    if not args.skip_plots:
        plot_all(all_rows, tasks, args.layers, args.positions, args.output_dir)

    print(f"\nSaved {result_path}")
    print(f"Saved {args.output_dir / 'summary.json'}")
    print("FINISHED_FOURIER_DONOR_PATCH_CAUSAL_EVALUATION")


if __name__ == "__main__":
    main()
