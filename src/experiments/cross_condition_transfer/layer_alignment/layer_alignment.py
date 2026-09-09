"""Selected-layer causal alignment and hub-consistency study.

This is intentionally compact: it reuses existing DAS audit spaces and the
transition-mean Procrustes fitting code.  For each selected layer it reports:

* text/image self-DAS IIA from the audit rows;
* raw cross-space patching;
* pairwise transported patching;
* synchronized-hub transported patching;
* alignment error, cycle consistency, and synchronization residuals.

If a layer is missing from the DAS audit artifacts, it is recorded in the
manifest and skipped.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from pathlib import Path

import torch

from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    heldout_pairs_path,
    jsonable,
    label,
    load_basis,
    load_jsonl,
    normalized_transfer,
    parse_task,
    position_for,
    results_path,
    row_matches,
    save_json,
    save_jsonl,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.factorized_paths.factorized_paths import (
    TaskSpace,
    evaluate_transport,
    fit_scaled_map,
    match_transition_pairs,
    test_pairs_for_task,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    autoregressive_iia_image_aligned,
    autoregressive_iia_text_aligned,
    coordinate_deltas_for_pairs,
    coordinate_metrics,
    data_root_for,
    load_causal_rows,
    load_model_bundle,
    sample_split,
    stable_seed,
)
from src.experiments.global_geometry.synchronization.synchronization import (
    Edge,
    hub_map,
    synchronize_rotations,
    synchronize_scales,
)


EXPERIMENT = "selected_layer_alignment"
TASKS = [
    "text:addition",
    "image:addition",
    "text:subtraction",
    "image:subtraction",
]
DEFAULT_ALIGNMENTS = [
    "text:addition->image:addition",
    "image:addition->text:addition",
    "text:subtraction->image:subtraction",
    "image:subtraction->text:subtraction",
]
TRANSPORTS = [
    "raw_source_ambient",
    "pairwise_transport",
    "hub_transport",
    "pairwise_transport_from_layer43",
    "hub_transport_from_layer43",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--causal_transfer_rows", type=Path, default=Path("results/final_exps/causal_tranfer/transfer_results.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/layer_alignment"))
    parser.add_argument("--plot_dir", type=Path, default=Path("visualizations/main_paper/layer_alignment"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--alignments", nargs="+", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--layers", type=int, nargs="+", default=[38, 40, 42, 43, 44, 46])
    parser.add_argument("--full_seed_layer", type=int, default=43)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--control_condition", default="random_subspace_in_pca_span")
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--alignment_seed", type=int, default=0)
    parser.add_argument("--max_alignment_samples", type=int, default=1024)
    parser.add_argument("--max_pair_bank", type=int, default=65536)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--reuse_source_layer", type=int, default=43)
    parser.add_argument("--reuse_target_layers", type=int, nargs="+", default=[42, 44])
    parser.add_argument("--skip_causal", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_alignment(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Alignment must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def seeds_for_layer(args: argparse.Namespace, layer: int) -> list[int]:
    return list(args.seeds) if layer == args.full_seed_layer else [int(args.base_seed)]


def with_audit_root(args: argparse.Namespace, audit_root: Path) -> argparse.Namespace:
    patched = argparse.Namespace(**vars(args))
    patched.audit_root = audit_root
    return patched


def layer_audit_args(args: argparse.Namespace) -> argparse.Namespace:
    """Point at the per-layer audit folder created by launch_layer_alignment_k22.sh."""
    layer_root = args.audit_root / f"layer_{args.layer}"
    return with_audit_root(args, layer_root)


def matching_result_row(args: argparse.Namespace, task: str, seed: int, condition: str | None = None) -> dict | None:
    modality, operation = parse_task(task)
    lookup_args = layer_audit_args(args)
    path = results_path(lookup_args, modality, operation, condition or args.condition, seed)
    if not path.exists():
        lookup_args = args
        path = results_path(lookup_args, modality, operation, condition or args.condition, seed)
    if not path.exists():
        return None
    for row in load_jsonl(path):
        if row_matches(args, row, modality):
            return row
    return None


def layer_control_iia(args: argparse.Namespace, space: TaskSpace) -> float | None:
    if args.control_condition == "clean":
        return float(space.row["clean_autoregressive_iia"])
    control_row = matching_result_row(args, space.task, space.seed, args.control_condition)
    if control_row is None:
        return None
    return float(control_row["autoregressive_iia"])


def make_space(args: argparse.Namespace, task: str, seed: int, hidden_size: int) -> TaskSpace | None:
    row = matching_result_row(args, task, seed)
    if row is None:
        return None
    modality, operation = parse_task(task)
    lookup_args = layer_audit_args(args)
    path = subspace_path(lookup_args, modality, operation, seed)
    if not path.exists():
        path = subspace_path(args, modality, operation, seed)
    basis = load_basis(path, args.layer, hidden_size, args.k)
    return TaskSpace(task, modality, operation, seed, row, basis)


def make_spaces(args: argparse.Namespace, tasks: list[str], seed: int, hidden_size: int) -> tuple[dict[str, TaskSpace], list[dict]]:
    spaces = {}
    missing = []
    for task in tasks:
        try:
            space = make_space(args, task, seed, hidden_size)
        except Exception as exc:
            space = None
            missing.append({"layer": args.layer, "seed": seed, "task": task, "reason": repr(exc)})
        if space is None:
            missing.append({"layer": args.layer, "seed": seed, "task": task, "reason": "missing matching audit row"})
        else:
            spaces[task] = space
    return spaces, missing


def pair_batches(args: argparse.Namespace, sample_cache: dict, source: TaskSpace, destination: TaskSpace) -> tuple[list[dict], list[dict], dict]:
    source_pairs = test_pairs_for_task(args, sample_cache, space=source)
    destination_pairs = test_pairs_for_task(args, sample_cache, space=destination)
    return match_transition_pairs(
        args,
        source_pairs,
        destination_pairs,
        seed=stable_seed("layer_alignment_eval", args.layer, source.task, destination.task, source.seed, args.alignment_seed),
        max_pairs=args.max_autoregressive_pairs,
    )


def evaluate_raw_source_ambient(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source: TaskSpace,
    destination: TaskSpace,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    eval_stats: dict,
    model,
    processor,
    tokenizer,
    blocks,
    model_name: str,
) -> dict:
    source_coords = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source.task,
        row=source.row,
        pairs=source_pairs,
        basis=source.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    destination_coords = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=destination.task,
        row=destination.row,
        pairs=destination_pairs,
        basis=destination.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    hidden_deltas = source_coords @ source.basis.T
    projected_destination_coords = hidden_deltas @ destination.basis
    geometry = coordinate_metrics(projected_destination_coords, destination_coords, torch.eye(args.k), 1.0)
    description = f"{EXPERIMENT} raw {source.task}[{source.seed}] -> {destination.task}[{destination.seed}] layer {args.layer}"
    if destination.modality == "text":
        iia = autoregressive_iia_text_aligned(model, tokenizer, blocks, destination_pairs, hidden_deltas, args, description)
    else:
        iia = autoregressive_iia_image_aligned(
            model,
            processor,
            tokenizer,
            blocks,
            destination_pairs,
            hidden_deltas,
            data_root_for(destination.row),
            args,
            description,
        )
    destination_control = layer_control_iia(args, destination)
    destination_self = float(destination.row["autoregressive_iia"])
    return {
        "model": model_name,
        "experiment": EXPERIMENT,
        "layer": args.layer,
        "seed": source.seed,
        "transport": "raw_source_ambient",
        "source_task": source.task,
        "source_label": label(source.task),
        "destination_task": destination.task,
        "destination_label": label(destination.task),
        "source_seed": source.seed,
        "destination_seed": destination.seed,
        "n_pairs": len(destination_pairs),
        "eval_stats": eval_stats,
        "alignment_fit_test": geometry,
        "autoregressive_iia": iia,
        "destination_self_autoregressive_iia": destination_self,
        "destination_control_autoregressive_iia": destination_control,
        "destination_normalized_transfer": None
        if destination_control is None
        else normalized_transfer(iia, destination_control, destination_self),
    }


def fit_graph_maps(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    spaces: dict[str, TaskSpace],
    *,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict[tuple[str, str], dict]:
    maps = {}
    for source_task, destination_task in itertools.permutations(sorted(spaces), 2):
        source, destination = spaces[source_task], spaces[destination_task]
        maps[(source_task, destination_task)] = fit_scaled_map(
            args,
            activation_cache,
            sample_cache,
            source=source,
            destination=destination,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
    return maps


def edge_weight(fit_metrics: dict) -> float:
    cosine = fit_metrics.get("mean_cosine")
    return max(float(cosine), 1e-3) if cosine is not None else 1.0


def sync_from_maps(args: argparse.Namespace, maps: dict[tuple[str, str], dict], tasks: list[str]) -> tuple[dict[str, torch.Tensor], dict[str, float], dict]:
    edges = []
    for (source, destination), fitted in maps.items():
        edges.append(
            Edge(
                source=source,
                destination=destination,
                q=fitted["q"],
                alpha=fitted["alpha"],
                weight=edge_weight(fitted.get("fit_metrics", {})),
                path=fitted["path"],
                fit_metrics=fitted.get("fit_metrics", {}),
            )
        )
    orientations, rotation_stats = synchronize_rotations(tasks, edges, args.k)
    log_scales, scale_stats = synchronize_scales(tasks, edges)
    return orientations, log_scales, {"rotation": rotation_stats, "scale": scale_stats}


def cycle_consistency(args: argparse.Namespace, maps: dict[tuple[str, str], dict], tasks: list[str]) -> dict:
    rotation_errors = []
    scale_errors = []
    for a, b, c in itertools.permutations(tasks, 3):
        if (a, b) not in maps or (b, c) not in maps or (c, a) not in maps:
            continue
        q_cycle = maps[(a, b)]["q"] @ maps[(b, c)]["q"] @ maps[(c, a)]["q"]
        alpha_cycle = float(maps[(a, b)]["alpha"]) * float(maps[(b, c)]["alpha"]) * float(maps[(c, a)]["alpha"])
        rotation_errors.append(float((q_cycle - torch.eye(args.k)).norm() / (args.k ** 0.5)))
        scale_errors.append(abs(alpha_cycle - 1.0))
    return {
        "n_directed_triangles": len(rotation_errors),
        "cycle_rotation_error_mean": mean(rotation_errors),
        "cycle_rotation_error_std": sample_std(rotation_errors),
        "cycle_scale_error_mean_abs": mean(scale_errors),
        "cycle_scale_error_std_abs": sample_std(scale_errors),
    }


def prediction_for_pairs(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source: TaskSpace,
    source_pairs: list[dict],
    q: torch.Tensor,
    alpha: float,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    x_eval = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source.task,
        row=source.row,
        pairs=source_pairs,
        basis=source.basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return float(alpha) * (x_eval @ q)


def evaluate_pairwise_or_hub(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    causal_rows: dict,
    *,
    transport: str,
    source: TaskSpace,
    destination: TaskSpace,
    source_pairs: list[dict],
    destination_pairs: list[dict],
    eval_stats: dict,
    q: torch.Tensor,
    alpha: float,
    direct_q: torch.Tensor,
    direct_alpha: float,
    component_paths: list[Path],
    model,
    processor,
    tokenizer,
    blocks,
    model_name: str,
) -> dict:
    direct_prediction = prediction_for_pairs(
        args,
        activation_cache,
        sample_cache,
        source=source,
        source_pairs=source_pairs,
        q=direct_q,
        alpha=direct_alpha,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    row = evaluate_transport(
        args,
        activation_cache,
        sample_cache,
        causal_rows,
        source=source,
        destination=destination,
        source_pairs=source_pairs,
        destination_pairs=destination_pairs,
        transport=transport,
        test_family=EXPERIMENT,
        eval_stats=eval_stats,
        q=q,
        alpha=alpha,
        direct_q=direct_q,
        direct_alpha=direct_alpha,
        direct_prediction=direct_prediction,
        component_paths=component_paths,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        model_name=model_name,
    )
    row["experiment"] = EXPERIMENT
    row["layer"] = args.layer
    row["seed"] = source.seed
    destination_control = layer_control_iia(args, destination)
    row["destination_control_autoregressive_iia"] = destination_control
    row["destination_normalized_transfer"] = (
        None
        if destination_control is None
        else normalized_transfer(
            row["autoregressive_iia"],
            destination_control,
            row["destination_self_autoregressive_iia"],
        )
    )
    return row


def self_rows(args: argparse.Namespace, spaces: dict[str, TaskSpace], model_name: str) -> list[dict]:
    rows = []
    for space in spaces.values():
        rows.append(
            {
                "model": model_name,
                "experiment": EXPERIMENT,
                "layer": args.layer,
                "seed": space.seed,
                "task": space.task,
                "task_label": label(space.task),
                "modality": space.modality,
                "operation": space.operation,
                "self_autoregressive_iia": float(space.row["autoregressive_iia"]),
                "variable_teacher_forced_iia": space.row.get("variable_teacher_forced_iia"),
                "full_answer_teacher_forced_iia": space.row.get("full_answer_teacher_forced_iia"),
            }
        )
    return rows


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def nested(row: dict, path: str):
    value = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def summarize_transport(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["layer"], row["transport"], row["source_task"], row["destination_task"]), []).append(row)
    summary = []
    for (layer, transport, source, destination), parts in sorted(groups.items()):
        summary.append(
            {
                "layer": layer,
                "transport": transport,
                "source_task": source,
                "source_label": label(source),
                "destination_task": destination,
                "destination_label": label(destination),
                "n": len(parts),
                "autoregressive_iia_mean": mean([row["autoregressive_iia"] for row in parts]),
                "autoregressive_iia_std": sample_std([row["autoregressive_iia"] for row in parts]),
                "destination_normalized_transfer_mean": mean([row["destination_normalized_transfer"] for row in parts]),
                "alignment_rmse_mean": mean([nested(row, "alignment_fit_test.root_mean_squared_error") for row in parts]),
                "alignment_cosine_mean": mean([nested(row, "alignment_fit_test.mean_cosine") for row in parts]),
            }
        )
    return summary


def summarize_self(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["layer"], row["task"]), []).append(row)
    summary = []
    for (layer, task), parts in sorted(groups.items()):
        summary.append(
            {
                "layer": layer,
                "task": task,
                "task_label": label(task),
                "n": len(parts),
                "self_autoregressive_iia_mean": mean([row["self_autoregressive_iia"] for row in parts]),
                "self_autoregressive_iia_std": sample_std([row["self_autoregressive_iia"] for row in parts]),
            }
        )
    return summary


def summarize_sync(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["layer"], []).append(row)
    summary = []
    for layer, parts in sorted(groups.items()):
        summary.append(
            {
                "layer": layer,
                "n": len(parts),
                "sync_rotation_residual_mean": mean([row["sync_rotation_residual_mean"] for row in parts]),
                "sync_scale_residual_mean_abs": mean([row["sync_scale_residual_mean_abs"] for row in parts]),
                "cycle_rotation_error_mean": mean([row["cycle_rotation_error_mean"] for row in parts]),
                "cycle_scale_error_mean_abs": mean([row["cycle_scale_error_mean_abs"] for row in parts]),
            }
        )
    return summary


def write_outputs(args: argparse.Namespace, self_iia: list[dict], transport: list[dict], sync: list[dict], missing: list[dict], model_name: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(self_iia, args.output_dir / "self_iia_by_layer.jsonl")
    save_jsonl(summarize_self(self_iia), args.output_dir / "self_iia_summary.jsonl")
    save_jsonl(transport, args.output_dir / "transport_by_layer.jsonl")
    save_jsonl(summarize_transport(transport), args.output_dir / "transport_summary.jsonl")
    save_jsonl(sync, args.output_dir / "sync_cycle_by_layer.jsonl")
    save_jsonl(summarize_sync(sync), args.output_dir / "sync_cycle_summary.jsonl")
    save_jsonl(missing, args.output_dir / "missing_layers.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "model": model_name,
            "layers": args.layers,
            "tasks": args.tasks,
            "alignments": args.alignments,
            "transports": TRANSPORTS,
            "seed_policy": {
                "one_seed_layers": [layer for layer in args.layers if layer != args.full_seed_layer],
                "base_seed": args.base_seed,
                "full_seed_layer": args.full_seed_layer,
                "full_seed_layer_seeds": args.seeds,
            },
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def plot_outputs(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots; matplotlib unavailable: {exc}")
        return
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    self_summary = load_jsonl(args.output_dir / "self_iia_summary.jsonl") if (args.output_dir / "self_iia_summary.jsonl").exists() else []
    transport_summary = load_jsonl(args.output_dir / "transport_summary.jsonl") if (args.output_dir / "transport_summary.jsonl").exists() else []
    sync_summary = load_jsonl(args.output_dir / "sync_cycle_summary.jsonl") if (args.output_dir / "sync_cycle_summary.jsonl").exists() else []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for task in sorted({row["task"] for row in self_summary}):
        rows = sorted([row for row in self_summary if row["task"] == task], key=lambda row: row["layer"])
        ax.plot([row["layer"] for row in rows], [row["self_autoregressive_iia_mean"] for row in rows], marker="o", label=f"self {label(task)}")
    for transport in TRANSPORTS:
        rows = sorted([row for row in transport_summary if row["transport"] == transport], key=lambda row: row["layer"])
        by_layer = {}
        for row in rows:
            by_layer.setdefault(row["layer"], []).append(row["autoregressive_iia_mean"])
        if by_layer:
            ax.plot(sorted(by_layer), [mean(by_layer[layer]) for layer in sorted(by_layer)], marker="s", label=transport)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Autoregressive IIA")
    ax.set_title("Causality and Cross-Space Transport by Layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.plot_dir / "causal_iia_by_layer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for transport in ["pairwise_transport", "hub_transport", "raw_source_ambient"]:
        rows = sorted([row for row in transport_summary if row["transport"] == transport], key=lambda row: row["layer"])
        by_layer = {}
        for row in rows:
            by_layer.setdefault(row["layer"], []).append(row["alignment_rmse_mean"])
        if by_layer:
            ax.plot(sorted(by_layer), [mean(by_layer[layer]) for layer in sorted(by_layer)], marker="o", label=transport)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment RMSE")
    ax.set_title("Alignment Error by Layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.plot_dir / "alignment_error_by_layer.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    if sync_summary:
        rows = sorted(sync_summary, key=lambda row: row["layer"])
        ax.plot([row["layer"] for row in rows], [row["sync_rotation_residual_mean"] for row in rows], marker="o", label="sync residual")
        ax.plot([row["layer"] for row in rows], [row["cycle_rotation_error_mean"] for row in rows], marker="s", label="cycle error")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Rotation Error")
    ax.set_title("Hub and Cycle Consistency by Layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.plot_dir / "hub_cycle_consistency_by_layer.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    alignments = [parse_alignment(item) for item in args.alignments]
    args.tasks = list(dict.fromkeys(args.tasks + [task for pair in alignments for task in pair]))
    args.layer = int(args.reuse_source_layer)
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)
    causal_rows = load_causal_rows(args.causal_transfer_rows)
    activation_cache = {}
    sample_cache = {}
    self_iia_rows = []
    transport_rows = []
    sync_rows = []
    missing_rows = []
    layer_maps = {}
    layer_sync = {}

    print("Selected-layer alignment")
    print(f"  layers={args.layers}")
    print(f"  alignments={args.alignments}")
    layer_order = [args.reuse_source_layer] + [layer for layer in args.layers if layer != args.reuse_source_layer]
    for layer in layer_order:
        args.layer = int(layer)
        for seed in seeds_for_layer(args, layer):
            print(f"\nLayer {layer}, seed {seed}")
            spaces, missing = make_spaces(args, args.tasks, seed, hidden_size)
            missing_rows.extend(missing)
            if len(spaces) < len(args.tasks):
                print(f"  skipping incomplete layer/seed; found {len(spaces)}/{len(args.tasks)} spaces")
                write_outputs(args, self_iia_rows, transport_rows, sync_rows, missing_rows, model_name)
                continue
            self_iia_rows.extend(self_rows(args, spaces, model_name))
            maps = fit_graph_maps(
                args,
                activation_cache,
                sample_cache,
                spaces,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
            )
            layer_maps[(layer, seed)] = maps
            orientations, log_scales, sync_stats = sync_from_maps(args, maps, args.tasks)
            layer_sync[(layer, seed)] = (orientations, log_scales, sync_stats)
            cycle = cycle_consistency(args, maps, args.tasks)
            sync_rows.append(
                {
                    "layer": layer,
                    "seed": seed,
                    "n_edges": len(maps),
                    "sync_rotation_residual_mean": sync_stats["rotation"]["rotation_residual_mean"],
                    "sync_rotation_residual_std": sync_stats["rotation"]["rotation_residual_std"],
                    "sync_scale_residual_mean_abs": sync_stats["scale"]["scale_residual_mean_abs"],
                    "sync_scale_residual_std_abs": sync_stats["scale"]["scale_residual_std_abs"],
                    **cycle,
                }
            )
            if args.skip_causal:
                write_outputs(args, self_iia_rows, transport_rows, sync_rows, missing_rows, model_name)
                continue
            for source_task, destination_task in alignments:
                source, destination = spaces[source_task], spaces[destination_task]
                source_pairs, destination_pairs, eval_stats = pair_batches(args, sample_cache, source, destination)
                direct = maps[(source_task, destination_task)]
                transport_rows.append(
                    evaluate_raw_source_ambient(
                        args,
                        activation_cache,
                        sample_cache,
                        source=source,
                        destination=destination,
                        source_pairs=source_pairs,
                        destination_pairs=destination_pairs,
                        eval_stats=eval_stats,
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        model_name=model_name,
                    )
                )
                transport_rows.append(
                    evaluate_pairwise_or_hub(
                        args,
                        activation_cache,
                        sample_cache,
                        causal_rows,
                        transport="pairwise_transport",
                        source=source,
                        destination=destination,
                        source_pairs=source_pairs,
                        destination_pairs=destination_pairs,
                        eval_stats=eval_stats,
                        q=direct["q"],
                        alpha=direct["alpha"],
                        direct_q=direct["q"],
                        direct_alpha=direct["alpha"],
                        component_paths=[direct["path"]],
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        model_name=model_name,
                    )
                )
                hub_q, hub_alpha = hub_map(source_task, destination_task, orientations, log_scales)
                transport_rows.append(
                    evaluate_pairwise_or_hub(
                        args,
                        activation_cache,
                        sample_cache,
                        causal_rows,
                        transport="hub_transport",
                        source=source,
                        destination=destination,
                        source_pairs=source_pairs,
                        destination_pairs=destination_pairs,
                        eval_stats=eval_stats,
                        q=hub_q,
                        alpha=hub_alpha,
                        direct_q=direct["q"],
                        direct_alpha=direct["alpha"],
                        component_paths=[direct["path"]],
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        model_name=model_name,
                    )
                )
                reuse_key = (args.reuse_source_layer, seed)
                if layer in args.reuse_target_layers and reuse_key in layer_maps and reuse_key in layer_sync:
                    reuse_maps = layer_maps[reuse_key]
                    if (source_task, destination_task) in reuse_maps:
                        reuse_direct = reuse_maps[(source_task, destination_task)]
                        transport_rows.append(
                            evaluate_pairwise_or_hub(
                                args,
                                activation_cache,
                                sample_cache,
                                causal_rows,
                                transport=f"pairwise_transport_from_layer{args.reuse_source_layer}",
                                source=source,
                                destination=destination,
                                source_pairs=source_pairs,
                                destination_pairs=destination_pairs,
                                eval_stats=eval_stats,
                                q=reuse_direct["q"],
                                alpha=reuse_direct["alpha"],
                                direct_q=direct["q"],
                                direct_alpha=direct["alpha"],
                                component_paths=[reuse_direct["path"]],
                                model=model,
                                processor=processor,
                                tokenizer=tokenizer,
                                blocks=blocks,
                                model_name=model_name,
                            )
                        )
                    reuse_orientations, reuse_log_scales, _reuse_stats = layer_sync[reuse_key]
                    reuse_hub_q, reuse_hub_alpha = hub_map(source_task, destination_task, reuse_orientations, reuse_log_scales)
                    transport_rows.append(
                        evaluate_pairwise_or_hub(
                            args,
                            activation_cache,
                            sample_cache,
                            causal_rows,
                            transport=f"hub_transport_from_layer{args.reuse_source_layer}",
                            source=source,
                            destination=destination,
                            source_pairs=source_pairs,
                            destination_pairs=destination_pairs,
                            eval_stats=eval_stats,
                            q=reuse_hub_q,
                            alpha=reuse_hub_alpha,
                            direct_q=direct["q"],
                            direct_alpha=direct["alpha"],
                            component_paths=[reuse_maps[(source_task, destination_task)]["path"]] if (source_task, destination_task) in reuse_maps else [],
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            blocks=blocks,
                            model_name=model_name,
                        )
                    )
                write_outputs(args, self_iia_rows, transport_rows, sync_rows, missing_rows, model_name)

    write_outputs(args, self_iia_rows, transport_rows, sync_rows, missing_rows, model_name)
    plot_outputs(args)
    print(f"\nSaved tables under: {args.output_dir}")
    print(f"Saved plots under: {args.plot_dir}")


if __name__ == "__main__":
    main()
