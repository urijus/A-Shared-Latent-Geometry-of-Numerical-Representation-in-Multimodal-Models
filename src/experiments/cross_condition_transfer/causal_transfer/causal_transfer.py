"""Causal transfer between DAS subspaces across operation and modality.

For each source task A and destination task B, load the selected source DAS
subspace R_A and evaluate it on base/donor pairs from B:

    h'_b = h_b + R_A (R_A^T h_d - R_A^T h_b)

The script reports raw autoregressive IIA and destination-normalized transfer:

    T_{A->B} = (IIA(R_A -> B) - IIA_control)
               / (IIA(R_B -> B) - IIA_control)

By default, every requested source seed is crossed with every requested
destination seed for each task pair.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import torch

from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import autoregressive_iia_image
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_TASKS = [
    "text:addition",
    "text:subtraction",
    "image:addition",
    "image:subtraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/causal_tranfer"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--control_condition", default="random_subspace_in_pca_span")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--top_n",
        type=int,
        default=0,
        help="Use top N seeds by source-task AR IIA. Default 0 uses --seeds.",
    )
    parser.add_argument(
        "--seed_map",
        nargs="*",
        default=[],
        metavar="TASK=SEEDS",
        help=(
            "Override automatic seed selection. Example: "
            "text:addition=2 image:addition=0,1"
        ),
    )
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--prompt",
        default="Output ONLY a number.",
        help="Only used for image destinations.",
    )
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parse_task(task: str) -> tuple[str, str]:
    parts = task.split(":")
    if len(parts) != 2:
        raise ValueError(f"Task must be modality:operation, got {task!r}.")
    modality, operation = parts
    if modality not in {"text", "image"}:
        raise ValueError(f"Unsupported modality in task {task!r}.")
    if operation not in {"addition", "subtraction"}:
        raise ValueError(f"Unsupported operation in task {task!r}.")
    return modality, operation


def task_key(modality: str, operation: str) -> str:
    return f"{modality}:{operation}"


def label(task: str) -> str:
    modality, operation = parse_task(task)
    short_op = "add" if operation == "addition" else "sub"
    return f"{modality}_{short_op}"


def position_for(args: argparse.Namespace, modality: str) -> str:
    return args.image_position if modality == "image" else args.text_position


def task_root(args: argparse.Namespace, modality: str, operation: str) -> Path:
    root = args.audit_root / modality / operation
    return root if args.target == "result" else root / args.target


def run_dir(
    args: argparse.Namespace,
    modality: str,
    operation: str,
    condition: str,
    seed: int,
) -> Path:
    return task_root(args, modality, operation) / condition / f"split_{args.split_seed}" / f"seed_{seed}"


def results_path(
    args: argparse.Namespace,
    modality: str,
    operation: str,
    condition: str,
    seed: int,
) -> Path:
    return run_dir(args, modality, operation, condition, seed) / "results.jsonl"


def subspace_path(args: argparse.Namespace, modality: str, operation: str, seed: int) -> Path:
    return run_dir(args, modality, operation, args.condition, seed) / "subspace.pt"


def heldout_pairs_path(
    args: argparse.Namespace, modality: str, operation: str, seed: int
) -> Path:
    return run_dir(args, modality, operation, args.condition, seed) / "heldout_pairs.jsonl"


def first_result_row(
    args: argparse.Namespace,
    modality: str,
    operation: str,
    condition: str,
    seed: int,
) -> dict:
    path = results_path(args, modality, operation, condition, seed)
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No result rows in {path}.")
    return rows[0]


def row_matches(args: argparse.Namespace, row: dict, modality: str) -> bool:
    return (
        int(row.get("layer", -1)) == args.layer
        and int(row.get("k", -1)) == args.k
        and str(row.get("position")) == str(position_for(args, modality))
        and row.get("hook") == args.hook
    )


def parse_seed_map(items: list[str]) -> dict[str, list[int]]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Seed-map item must be TASK=SEEDS, got {item!r}.")
        task, seeds = item.split("=", 1)
        parse_task(task)
        parsed[task] = [int(seed) for seed in re.split(r"[,\s]+", seeds) if seed]
    return parsed


def discover_seed_rows(args: argparse.Namespace, task: str) -> list[dict]:
    modality, operation = parse_task(task)
    split_dir = task_root(args, modality, operation) / args.condition / f"split_{args.split_seed}"
    candidates = []
    for seed_dir in sorted(split_dir.glob("seed_*")):
        match = re.match(r"seed_(\d+)$", seed_dir.name)
        if not match:
            continue
        seed = int(match.group(1))
        path = seed_dir / "results.jsonl"
        checkpoint = seed_dir / "subspace.pt"
        if not path.exists() or not checkpoint.exists():
            continue
        row = load_jsonl(path)[0]
        if row_matches(args, row, modality):
            candidates.append({"seed": seed, "row": row, "path": path, "subspace_path": checkpoint})
    candidates.sort(key=lambda item: float(item["row"].get("autoregressive_iia", -1.0)), reverse=True)
    return candidates


def validate_seed(args: argparse.Namespace, task: str, seed: int) -> None:
    modality, operation = parse_task(task)
    checkpoint = subspace_path(args, modality, operation, seed)
    result = results_path(args, modality, operation, args.condition, seed)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not result.exists():
        raise FileNotFoundError(result)
    row = load_jsonl(result)[0]
    if not row_matches(args, row, modality):
        raise ValueError(
            f"{result} does not match layer={args.layer}, k={args.k}, "
            f"position={position_for(args, modality)}, hook={args.hook}."
        )


def selected_seeds(args: argparse.Namespace) -> dict[str, list[int]]:
    explicit = parse_seed_map(args.seed_map)
    seeds = {}
    for task in args.tasks:
        parse_task(task)
        if task in explicit:
            for seed in explicit[task]:
                validate_seed(args, task, seed)
            seeds[task] = explicit[task]
            continue
        if args.top_n <= 0:
            for seed in args.seeds:
                validate_seed(args, task, seed)
            seeds[task] = list(args.seeds)
            continue
        candidates = discover_seed_rows(args, task)
        if not candidates:
            raise FileNotFoundError(
                f"No matching {args.condition} seed found for {task} under {args.audit_root}."
            )
        seeds[task] = [item["seed"] for item in candidates[: args.top_n]]
    return seeds


def load_basis(path: Path, layer: int, hidden_size: int, k: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "basis" in payload:
        basis = torch.as_tensor(payload["basis"]).detach().float()
    elif "bases" in payload:
        saved = payload["bases"]
        saved_basis = saved.get(layer, saved.get(str(layer)))
        if saved_basis is None:
            raise ValueError(f"{path} has no basis for layer {layer}.")
        basis = torch.as_tensor(saved_basis).detach().float()
    else:
        raise ValueError(f"{path} does not contain a DAS basis.")
    basis = basis.squeeze()
    if basis.shape == (k, hidden_size):
        basis = basis.T
    if basis.shape != (hidden_size, k):
        raise ValueError(
            f"Basis at {path} has shape {tuple(basis.shape)}, expected {(hidden_size, k)}."
        )
    return basis


def subspace_from_basis(layer: int, basis: torch.Tensor, device: torch.device) -> dict[str, DASSubspace]:
    return {
        str(layer): DASSubspace(
            basis.shape[0], basis.shape[1], initial_basis=basis
        ).to(device)
    }


def sample_lookup(data_path: Path) -> dict:
    samples = load_jsonl(data_path)
    lookup = {}
    for index, sample in enumerate(samples):
        lookup[sample.get("sample_id", index)] = sample
    return lookup


def load_destination_pairs(
    args: argparse.Namespace, modality: str, operation: str, seed: int, result_row: dict
) -> list[dict]:
    heldout = heldout_pairs_path(args, modality, operation, seed)
    if not heldout.exists():
        raise FileNotFoundError(heldout)
    samples = sample_lookup(Path(result_row["data_path"]))
    pairs = []
    for row in load_jsonl(heldout):
        pairs.append(
            {
                "pair_id": row.get("pair_id", len(pairs)),
                "base": samples[row["base_sample_id"]],
                "source": samples[row["source_sample_id"]],
            }
        )
    return pairs[: args.max_autoregressive_pairs] if args.max_autoregressive_pairs > 0 else pairs


def control_iia(
    args: argparse.Namespace, modality: str, operation: str, seed: int, self_row: dict
) -> float:
    if args.control_condition == "clean":
        return float(self_row["clean_autoregressive_iia"])
    row = first_result_row(args, modality, operation, args.control_condition, seed)
    return float(row["autoregressive_iia"])


def normalized_transfer(raw: float, control: float, self_iia: float) -> float | None:
    denominator = self_iia - control
    if abs(denominator) < 1e-12:
        return None
    return (raw - control) / denominator


def cached_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "transfer_results.jsonl"


def load_cached(args: argparse.Namespace) -> dict[tuple, dict]:
    path = cached_path(args)
    if args.force or not path.exists():
        return {}
    rows = load_jsonl(path)
    return {
        (
            row["source_task"],
            int(row["source_seed"]),
            row["destination_task"],
            int(row["destination_seed"]),
        ): row
        for row in rows
    }


def evaluate_transfer_for_destination(
    args: argparse.Namespace,
    destination_task: str,
    destination_seed: int,
    source_tasks: list[str],
    seed_selection: dict[str, list[int]],
    cache: dict[tuple, dict],
) -> list[dict]:
    modality, operation = parse_task(destination_task)
    self_row = first_result_row(args, modality, operation, args.condition, destination_seed)
    pairs = load_destination_pairs(args, modality, operation, destination_seed, self_row)
    destination_control = control_iia(args, modality, operation, destination_seed, self_row)
    destination_self = float(self_row["autoregressive_iia"])

    model_path, model_name = resolve_model_for_loading(args.model)
    if modality == "text":
        model, tokenizer = load_hf_model(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        processor = None
        use_chat = args.use_chat_template or uses_chat_template(args.model)
    else:
        model, processor, tokenizer = load_hf_model_and_processor(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        use_chat = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    hidden_size = get_hidden_size(model)

    rows = []
    for source_task in source_tasks:
        source_modality, source_operation = parse_task(source_task)
        for source_seed in seed_selection[source_task]:
            key = (source_task, source_seed, destination_task, destination_seed)
            if key in cache:
                rows.append(cache[key])
                continue

            if source_task == destination_task and source_seed == destination_seed:
                raw_iia = destination_self
                source_path = subspace_path(args, source_modality, source_operation, source_seed)
                used_saved_self = True
            else:
                source_path = subspace_path(args, source_modality, source_operation, source_seed)
                basis = load_basis(source_path, args.layer, hidden_size, args.k)
                subspaces = subspace_from_basis(args.layer, basis, model.device)
                if modality == "text":
                    raw_iia = autoregressive_iia(
                        model=model,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        subspaces=subspaces,
                        layers=[args.layer],
                        hook_name=args.hook,
                        pairs=pairs,
                        position=args.text_position,
                        target=args.target,
                        use_chat_template=use_chat,
                        max_new_tokens=args.max_new_tokens,
                        description=f"{source_task}[{source_seed}] -> {destination_task}[{destination_seed}]",
                    )
                else:
                    raw_iia = autoregressive_iia_image(
                        model=model,
                        processor=processor,
                        tokenizer=tokenizer,
                        blocks=blocks,
                        subspaces=subspaces,
                        layers=[args.layer],
                        hook_name=args.hook,
                        pairs=pairs,
                        data_root=Path(self_row.get("data_root") or Path(self_row["data_path"]).parent),
                        prompt=args.prompt,
                        position_strategy=args.image_position,
                        target=args.target,
                        enable_thinking=args.enable_thinking,
                        max_new_tokens=args.max_new_tokens,
                        description=f"{source_task}[{source_seed}] -> {destination_task}[{destination_seed}]",
                    )
                used_saved_self = False

            rows.append(
                {
                    "model": model_name,
                    "source_task": source_task,
                    "source_modality": source_modality,
                    "source_operation": source_operation,
                    "source_seed": source_seed,
                    "source_subspace_path": str(source_path),
                    "destination_task": destination_task,
                    "destination_modality": modality,
                    "destination_operation": operation,
                    "destination_seed": destination_seed,
                    "destination_results_path": str(
                        results_path(args, modality, operation, args.condition, destination_seed)
                    ),
                    "destination_heldout_pairs_path": str(
                        heldout_pairs_path(args, modality, operation, destination_seed)
                    ),
                    "layer": args.layer,
                    "k": args.k,
                    "hook": args.hook,
                    "source_position": position_for(args, source_modality),
                    "destination_position": position_for(args, modality),
                    "target": args.target,
                    "n_pairs": len(pairs),
                    "autoregressive_iia": raw_iia,
                    "destination_self_autoregressive_iia": destination_self,
                    "control_condition": args.control_condition,
                    "destination_control_autoregressive_iia": destination_control,
                    "destination_normalized_transfer": normalized_transfer(
                        raw_iia, destination_control, destination_self
                    ),
                    "used_saved_self_score": used_saved_self,
                }
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return sum(finite) / len(finite)


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def matrix_payload(rows: list[dict], tasks: list[str], metric: str) -> dict:
    labels = [label(task) for task in tasks]
    matrix = []
    matrix_rows = []
    for source in tasks:
        row_values = []
        for destination in tasks:
            cell_rows = [
                row
                for row in rows
                if row["source_task"] == source and row["destination_task"] == destination
            ]
            values = [row.get(metric) for row in cell_rows]
            value = mean(values)
            row_values.append(value)
            matrix_rows.append(
                {
                    "source_task": source,
                    "source_label": label(source),
                    "destination_task": destination,
                    "destination_label": label(destination),
                    "metric": metric,
                    "mean": value,
                    "std": sample_std(values),
                    "n": len(cell_rows),
                    "values": values,
                }
            )
        matrix.append(row_values)
    return {"tasks": tasks, "labels": labels, "metric": metric, "matrix": matrix, "rows": matrix_rows}


def write_outputs(args: argparse.Namespace, rows: list[dict], seed_selection: dict[str, list[int]]) -> None:
    rows.sort(
        key=lambda row: (
            row["source_task"],
            int(row["source_seed"]),
            row["destination_task"],
            int(row["destination_seed"]),
        )
    )
    save_jsonl(rows, args.output_dir / "transfer_results.jsonl")
    save_json(
        {"seed_selection": seed_selection, "config": jsonable(vars(args))},
        args.output_dir / "manifest.json",
    )

    raw = matrix_payload(rows, args.tasks, "autoregressive_iia")
    normalized = matrix_payload(rows, args.tasks, "destination_normalized_transfer")
    save_json(raw, args.output_dir / "raw_autoregressive_iia_matrix.json")
    save_json(normalized, args.output_dir / "destination_normalized_transfer_matrix.json")
    save_jsonl(raw["rows"], args.output_dir / "raw_autoregressive_iia_matrix.jsonl")
    save_jsonl(normalized["rows"], args.output_dir / "destination_normalized_transfer_matrix.jsonl")


def merge_rows(cache: dict[tuple, dict], rows: list[dict]) -> list[dict]:
    merged = dict(cache)
    for row in rows:
        merged[
            (
                row["source_task"],
                int(row["source_seed"]),
                row["destination_task"],
                int(row["destination_seed"]),
            )
        ] = row
    return list(merged.values())


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    seed_selection = selected_seeds(args)
    print("Selected seeds:")
    for task in args.tasks:
        print(f"  {task}: {seed_selection[task]}")

    cache = load_cached(args)
    rows = []
    for destination_task in args.tasks:
        for destination_seed in seed_selection[destination_task]:
            print(f"\nDestination {destination_task} seed={destination_seed}")
            rows.extend(
                evaluate_transfer_for_destination(
                    args,
                    destination_task,
                    destination_seed,
                    args.tasks,
                    seed_selection,
                    cache,
                )
            )
            write_outputs(args, merge_rows(cache, rows), seed_selection)

    final_rows = merge_rows(cache, rows)
    write_outputs(args, final_rows, seed_selection)
    print(f"Saved transfer rows: {args.output_dir / 'transfer_results.jsonl'}")
    print(f"Saved raw matrix: {args.output_dir / 'raw_autoregressive_iia_matrix.json'}")
    print(
        "Saved normalized matrix: "
        f"{args.output_dir / 'destination_normalized_transfer_matrix.json'}"
    )


if __name__ == "__main__":
    main()
