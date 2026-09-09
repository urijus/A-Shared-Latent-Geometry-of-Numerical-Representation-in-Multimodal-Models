"""Compare repeat-number representations with the exact causal arithmetic L.

This is a representational closing experiment.  It reuses the repository's
existing repeat-number task and the exact causal-L reconstruction from
``causal_L_shared_geometry``:

    U_digit^T R = A Sigma V^T
    Q = R V
    L = Q[:, 9:22]

For each arithmetic task/seed, the same 13-D arithmetic L basis is used as the
coordinate lens for both arithmetic final-position activations and repeat-task
final pre-generation activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import load_jsonl, save_jsonl
from src.experiments.readout_latent_geometry.controls import repeat_transfer
from src.experiments.global_geometry import causal_subspace_geometry as causal_l
from src.experiments.readout_latent_geometry import value_heldout_geometry as geom
from src.interventions.das import (
    format_prompt,
    hidden,
    hook_module,
    resolve_position,
    sequence_scores,
)
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    first_result_row,
    jsonable,
    parse_task,
    save_json,
    subspace_path,
    task_key,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.models import get_blocks, load_hf_model, resolve_model_for_loading, validate_block_layers


EXPERIMENT = "repeat_vs_causal_L_geometry"
TASK_ORDER = geom.TASK_ORDER
TASK_LABELS = geom.TASK_LABELS
REPEAT_TASK = "repeat:number"
REPEAT_LABEL = "Repeat"
PRIMARY_DOMAIN = "two_digit"


@dataclass
class RepeatData:
    labels: list[dict]
    hidden: torch.Tensor
    counts: dict[int, int]
    activation_path: Path | None
    correctness: dict[int, bool]


@dataclass
class LensItem:
    task: str
    task_label: str
    das_seed: int
    space_type: str
    basis: torch.Tensor
    arith_centroids: dict[int, torch.Tensor]
    repeat_centroids: dict[int, torch.Tensor]
    diagnostics: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--previous_geometry_dir", type=Path, default=Path("results/experiments/closing/causal_L_shared_geometry"))
    parser.add_argument(
        "--digit_readout_basis_path",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43/digit_readout_basis.pt"),
    )
    parser.add_argument("--activation_dir_text", type=Path, default=Path("outputs/activations/baseline/gemma4_12b_it/digits"))
    parser.add_argument("--activation_dir_image", type=Path, default=Path("outputs/activations/baseline_images/gemma4_12b_it/digits"))
    parser.add_argument("--repeat_transfer_dir", type=Path, default=Path("results/experiments/check_unembeeding/repeat_transfer"))
    parser.add_argument(
        "--repeat_activation_path",
        type=Path,
        default=Path("results/experiments/closing/repeat_vs_causal_L_geometry/repeat_activations_layer43_resid_post.pt"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/repeat_vs_causal_L_geometry"))
    parser.add_argument("--figure_dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_ORDER))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--value_split_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--control_seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--n_shuffles", type=int, default=20)
    parser.add_argument("--rsa_permutations", type=int, default=1000)
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
    parser.add_argument("--min_value", type=int, default=0)
    parser.add_argument("--max_value", type=int, default=99)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--accuracy_mode", choices=["autoregressive", "teacher_forced", "none"], default="teacher_forced")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true", help="Recorded in metadata only; repeat text prompts keep thinking disabled by default.")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sanity_tolerance", type=float, default=1e-5)
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    if args.tasks == list(TASK_ORDER):
        args.tasks = ["text:addition"]
    if args.seeds == [0, 1, 2]:
        args.seeds = [0]
    if args.value_split_seeds == [0, 1, 2]:
        args.value_split_seeds = [0]
    if args.control_seeds == list(range(20)):
        args.control_seeds = [0, 1]
    args.n_shuffles = min(args.n_shuffles, 2)
    args.rsa_permutations = min(args.rsa_permutations, 100)
    args.max_value = min(args.max_value, 19)
    args.batch_size = min(args.batch_size, 8)


def normalize_tasks(tasks: list[str]) -> list[str]:
    if any(task.lower() == "all" for task in tasks):
        return list(TASK_ORDER)
    return [task_key(*parse_task(task)) for task in tasks]


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def maybe_stdev(values: list[float]) -> float | None:
    return None if len(values) < 2 else float(statistics.stdev(values))


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def output_exists(args: argparse.Namespace) -> bool:
    return (args.output_dir / "repeat_vs_arithmetic_L_summary.json").exists() and not args.force


def split_domains(train_values: list[int], test_values: list[int]) -> dict[str, tuple[list[int], list[int]]]:
    domains = {
        "all_0_99": (list(train_values), list(test_values)),
        "two_digit": ([v for v in train_values if v >= 10], [v for v in test_values if v >= 10]),
        "one_digit": ([v for v in train_values if 0 <= v <= 9], [v for v in test_values if 0 <= v <= 9]),
    }
    return {
        name: (train, test)
        for name, (train, test) in domains.items()
        if len(train) >= 2 and len(test) >= 1
    }


def result_value(row: dict) -> int:
    return int(row["result"])


def sample_values(rows: list[dict]) -> set[int]:
    return {result_value(row) for row in rows if "result" in row}


def find_repeat_dataset(args: argparse.Namespace) -> tuple[list[dict], dict]:
    requested = set(range(args.min_value, args.max_value + 1))
    candidates = sorted(args.repeat_transfer_dir.glob("*dataset.jsonl"))
    inventory = []
    best_rows = None
    best_path = None
    best_coverage = set()
    for path in candidates:
        rows = load_jsonl(path)
        coverage = sample_values(rows)
        inventory.append(
            {
                "path": str(path),
                "n_rows": len(rows),
                "min_value": min(coverage) if coverage else None,
                "max_value": max(coverage) if coverage else None,
                "n_requested_values_present": len(coverage & requested),
                "covers_requested_range": requested.issubset(coverage),
            }
        )
        if len(coverage & requested) > len(best_coverage):
            best_rows = rows
            best_path = path
            best_coverage = coverage

    if best_rows is not None and requested.issubset(best_coverage):
        rows_by_value = {result_value(row): row for row in best_rows if result_value(row) in requested}
        return [rows_by_value[value] for value in sorted(requested)], {
            "dataset_source": "existing_repeat_transfer_dataset",
            "dataset_path": str(best_path),
            "extended_values_with_existing_generator": False,
            "dataset_candidates": inventory,
        }

    generated = repeat_transfer.make_repeat_samples(args)
    generated = [row for row in generated if result_value(row) in requested]
    return generated, {
        "dataset_source": "repeat_transfer.make_repeat_samples",
        "dataset_path": None,
        "best_existing_dataset_path": None if best_path is None else str(best_path),
        "best_existing_coverage": sorted(best_coverage),
        "extended_values_with_existing_generator": bool(best_rows is not None),
        "dataset_candidates": inventory,
    }


def cache_labels_by_value(payload: dict) -> dict[int, int]:
    return {int(row["result"]): index for index, row in enumerate(payload.get("labels", [])) if "result" in row}


def load_repeat_activation_cache(path: Path, args: argparse.Namespace) -> tuple[dict[int, torch.Tensor], list[dict], dict]:
    if not path.exists():
        return {}, [], {"exists": False, "path": str(path)}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    labels = list(payload.get("labels", []))
    by_value = cache_labels_by_value(payload)
    if "hidden" in payload:
        hidden_tensor = torch.as_tensor(payload["hidden"]).float()
    else:
        activations = torch.as_tensor(payload["activations"])
        layers = [int(x) for x in payload["block_layers"]]
        positions = payload.get("position_names") or [str(x) for x in payload.get("position_indices", [])]
        layer_idx = geom.select_axis_index(layers, args.layer, "layer", path)
        position_idx = geom.select_axis_index(positions, args.repeat_position, "position", path)
        hidden_tensor = activations[:, layer_idx, position_idx, :].float()
    hidden_by_value = {value: hidden_tensor[index].float() for value, index in by_value.items()}
    return hidden_by_value, labels, {
        "exists": True,
        "path": str(path),
        "n_rows": len(labels),
        "values": sorted(hidden_by_value),
        "layer": payload.get("layer", args.layer),
        "hook": payload.get("hook", args.hook),
        "position": payload.get("position", args.repeat_position),
        "model_name": payload.get("model_name"),
    }


def save_repeat_activation_cache(
    path: Path,
    *,
    hidden_by_value: dict[int, torch.Tensor],
    labels_by_value: dict[int, dict],
    args: argparse.Namespace,
    model_name: str,
    resolved_position: str,
    use_chat_template: bool,
) -> None:
    values = sorted(hidden_by_value)
    hidden_tensor = torch.stack([hidden_by_value[value].half() for value in values])
    labels = [labels_by_value[value] for value in values]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hidden": hidden_tensor,
            "labels": labels,
            "layer": args.layer,
            "hook": args.hook,
            "position": args.repeat_position,
            "resolved_position": resolved_position,
            "block_layers": [args.layer],
            "position_names": [args.repeat_position],
            "model_name": model_name,
            "use_chat_template": use_chat_template,
            "prompt_template": args.prompt_template,
            "answer_separator": args.answer_separator,
            "note": "Repeat-number final pre-generation activations at the prompt '=' site.",
        },
        path,
    )


@torch.no_grad()
def teacher_forced_correctness(model, tokenizer, samples: list[dict], use_chat_template: bool, batch_size: int) -> list[dict]:
    rows = []
    for start in tqdm(range(0, len(samples), batch_size), desc="repeat teacher-forced accuracy"):
        batch = samples[start : start + batch_size]
        prompts = [format_prompt(tokenizer, sample, use_chat_template) for sample in batch]
        answers = [str(result_value(sample)) for sample in batch]
        encoding, positions = repeat_transfer.prompt_answer_encoding(tokenizer, prompts, answers, model.device)
        outputs = model(**encoding, use_cache=False)
        _score, exact = sequence_scores(outputs.logits, encoding["input_ids"], positions)
        for sample, ok in zip(batch, exact.detach().cpu().tolist()):
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "value": result_value(sample),
                    "prompt": sample["expr"],
                    "mode": "teacher_forced",
                    "expected": str(result_value(sample)),
                    "generated": None,
                    "parsed": None,
                    "correct": bool(ok),
                }
            )
    return rows


@torch.no_grad()
def autoregressive_correctness(model, tokenizer, samples: list[dict], use_chat_template: bool, max_new_tokens: int) -> list[dict]:
    rows = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for sample in tqdm(samples, desc="repeat autoregressive accuracy"):
            prompt = format_prompt(tokenizer, sample, use_chat_template)
            expected = str(result_value(sample))
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            generated_ids = []
            for _ in range(max_new_tokens):
                outputs = model(input_ids=input_ids, use_cache=False)
                next_id = outputs.logits[0, input_ids.shape[1] - 1].argmax().reshape(1, 1)
                generated_ids.append(int(next_id.item()))
                input_ids = torch.cat([input_ids, next_id.to(model.device)], dim=1)
                if tokenizer.eos_token_id is not None and int(next_id.item()) == int(tokenizer.eos_token_id):
                    break
            generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            match = re.match(r"-?\d+", generated)
            parsed = match.group() if match else None
            rows.append(
                {
                    "sample_id": sample.get("sample_id"),
                    "value": result_value(sample),
                    "prompt": sample["expr"],
                    "mode": "autoregressive",
                    "expected": expected,
                    "generated": generated,
                    "parsed": parsed,
                    "correct": parsed == expected,
                }
            )
    finally:
        tokenizer.padding_side = old_padding_side
    return rows


@torch.no_grad()
def collect_repeat_activations(
    *,
    model,
    tokenizer,
    blocks,
    samples: list[dict],
    args: argparse.Namespace,
    use_chat_template: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, dict], str]:
    hidden_by_value = {}
    labels_by_value = {}
    first_resolved_position = None
    for start in tqdm(range(0, len(samples), args.batch_size), desc="collect repeat activations"):
        batch = samples[start : start + args.batch_size]
        prompts = [format_prompt(tokenizer, sample, use_chat_template) for sample in batch]
        if first_resolved_position is None:
            first_resolved_position = repeat_transfer.repeat_position_spec(tokenizer, prompts[0], args.repeat_position)
        positions = [resolve_position(tokenizer, prompt, repeat_transfer.repeat_position_spec(tokenizer, prompt, args.repeat_position)) for prompt in prompts]
        encoding = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
        captured = {}
        with ExitStack() as stack:
            module, pre_hook = hook_module(blocks[args.layer - 1], args.hook)
            if pre_hook:
                def capture_pre(_module, inputs):
                    captured[args.layer] = hidden(inputs[0]).detach()
                handle = module.register_forward_pre_hook(capture_pre)
            else:
                def capture_post(_module, _inputs, output):
                    captured[args.layer] = hidden(output).detach()
                handle = module.register_forward_hook(capture_post)
            stack.callback(handle.remove)
            model(**encoding, use_cache=False)
        rows = torch.arange(len(batch), device=model.device)
        token_positions = torch.tensor(positions, device=model.device)
        selected = captured[args.layer][rows, token_positions].float().cpu()
        for sample, vector, prompt, position in zip(batch, selected, prompts, positions):
            value = result_value(sample)
            label = dict(sample)
            label.update({"model_expr": prompt, "repeat_hf_position": int(position)})
            hidden_by_value[value] = vector
            labels_by_value[value] = label
    return hidden_by_value, labels_by_value, str(first_resolved_position)


def load_model_for_repeat(args: argparse.Namespace):
    model_path, resolved_model = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    return model, tokenizer, blocks, resolved_model


def prepare_repeat_data(args: argparse.Namespace, required_values: list[int]) -> tuple[RepeatData, dict, list[dict]]:
    samples, dataset_inventory = find_repeat_dataset(args)
    sample_by_value = {result_value(row): row for row in samples}
    requested = set(required_values)
    cache_hidden, cache_labels, cache_inventory = load_repeat_activation_cache(args.repeat_activation_path, args)
    accuracy_path = args.output_dir / "repeat_accuracy.csv"
    accuracy_rows = []
    if accuracy_path.exists() and not args.force:
        with accuracy_path.open("r", newline="", encoding="utf-8") as handle:
            accuracy_rows = list(csv.DictReader(handle))
            for row in accuracy_rows:
                row["value"] = int(row["value"])
                row["correct"] = str(row["correct"]).lower() == "true"
        if args.accuracy_mode != "none":
            cached_modes = {row.get("mode") for row in accuracy_rows}
            if cached_modes != {args.accuracy_mode}:
                print(
                    f"Ignoring cached repeat accuracy at {accuracy_path} because "
                    f"cached_modes={sorted(cached_modes)} requested={args.accuracy_mode}."
                )
                accuracy_rows = []
    cached_values = set(cache_hidden)
    missing_values = sorted(requested - cached_values)
    accuracy_values_present = {int(row["value"]) for row in accuracy_rows}
    needs_accuracy = args.accuracy_mode != "none" and not requested.issubset(accuracy_values_present)
    inventory = {
        "reused_files_and_functions": [
            "src.experiments.readout_latent_geometry.controls.repeat_transfer.make_repeat_samples",
            "src.experiments.readout_latent_geometry.controls.repeat_transfer.render_repeat_prompt",
            "src.experiments.readout_latent_geometry.controls.repeat_transfer.repeat_position_spec",
            "src.experiments.readout_latent_geometry.controls.repeat_transfer.prompt_answer_encoding",
            "src.experiments.global_geometry.causal_subspace_geometry.construct_exact_CL",
            "src.experiments.global_geometry.causal_subspace_geometry.random_readout_orthogonal_basis",
            "src.experiments.readout_latent_geometry.value_heldout_geometry centroid/Procrustes/RSA helpers",
        ],
        **dataset_inventory,
        "activation_cache": cache_inventory,
        "requested_values": sorted(requested),
        "missing_activation_values_before_extraction": missing_values,
        "missing_accuracy_values_before_check": sorted(requested - accuracy_values_present) if args.accuracy_mode != "none" else [],
        "model_forwards_needed": len(missing_values) > 0 or needs_accuracy,
        "repeat_prompt_template": args.prompt_template,
        "answer_separator": args.answer_separator,
        "repeat_position": args.repeat_position,
        "thinking_disabled": not args.enable_thinking,
    }
    save_json(inventory, args.output_dir / "repeat_artifact_inventory.json")
    print("Repeat artifact inventory")
    print(f"  dataset_source={inventory['dataset_source']}")
    print(f"  activation_cache={args.repeat_activation_path} exists={cache_inventory['exists']}")
    print(f"  missing_values={missing_values}")

    if args.artifact_check_only:
        correctness = {value: True for value in requested if value in cache_hidden}
        labels = [sample_by_value[value] for value in sorted(requested) if value in cache_hidden]
        hidden = torch.stack([cache_hidden[value] for value in sorted(requested) if value in cache_hidden]) if labels else torch.empty(0, 0)
        return RepeatData(labels, hidden, {}, args.repeat_activation_path if cache_hidden else None, correctness), inventory, accuracy_rows

    model = tokenizer = blocks = resolved_model = None
    if missing_values or needs_accuracy:
        model, tokenizer, blocks, resolved_model = load_model_for_repeat(args)
    use_chat_template = bool(args.use_chat_template) or uses_chat_template(args.model)

    if missing_values:
        missing_samples = [sample_by_value[value] for value in missing_values]
        new_hidden, new_labels, resolved_position = collect_repeat_activations(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            samples=missing_samples,
            args=args,
            use_chat_template=use_chat_template,
        )
        cache_hidden.update(new_hidden)
        labels_by_value = {int(row["result"]): row for row in cache_labels if "result" in row}
        labels_by_value.update(new_labels)
        save_repeat_activation_cache(
            args.repeat_activation_path,
            hidden_by_value=cache_hidden,
            labels_by_value=labels_by_value,
            args=args,
            model_name=resolved_model,
            resolved_position=resolved_position,
            use_chat_template=use_chat_template,
        )

    if needs_accuracy:
        if args.accuracy_mode == "autoregressive":
            accuracy_rows = autoregressive_correctness(model, tokenizer, samples, use_chat_template, args.max_new_tokens)
        else:
            accuracy_rows = teacher_forced_correctness(model, tokenizer, samples, use_chat_template, args.batch_size)
        write_csv(accuracy_rows, accuracy_path)

    if args.accuracy_mode == "none":
        correctness = {value: True for value in requested}
    else:
        correctness = {int(row["value"]): bool(row["correct"]) for row in accuracy_rows}
    kept_values = [value for value in sorted(requested) if value in cache_hidden and correctness.get(value, False)]
    if not kept_values:
        raise ValueError("No correct repeat examples with cached activations were available.")
    labels = [sample_by_value[value] for value in kept_values]
    hidden_tensor = torch.stack([cache_hidden[value] for value in kept_values]).float()
    counts = {value: 1 for value in kept_values}
    return RepeatData(labels, hidden_tensor, counts, args.repeat_activation_path, correctness), inventory, accuracy_rows


def load_exact_value_splits(args: argparse.Namespace) -> dict[str, dict[str, list[int]]]:
    proxy = argparse.Namespace(**vars(args))
    proxy.previous_geometry_dir = args.previous_geometry_dir
    return causal_l.load_exact_value_splits(proxy)


def all_split_values(value_splits: dict[str, dict[str, list[int]]]) -> list[int]:
    values = set()
    for split in value_splits.values():
        values.update(split["train_values"])
        values.update(split["test_values"])
    return sorted(values)


def load_arithmetic_data(args: argparse.Namespace, tasks: list[str]) -> dict[str, geom.TaskData]:
    data_by_task = {}
    for task in tasks:
        modality, operation = parse_task(task)
        row = first_result_row(args, modality, operation, args.condition, args.seeds[0])
        data_path = Path(row["data_path"])
        if not data_path.exists() and modality == "text":
            fallback = Path("dataset/baseline") / f"{operation}_baseline.jsonl"
            if fallback.exists():
                row = dict(row)
                row["data_path"] = str(fallback)
                print(f"Using local text dataset fallback for {task}: {fallback}")
        geom.validate_result_metadata(args, row, task=task, path=subspace_path(args, modality, operation, args.seeds[0]).with_name("results.jsonl"))
        data_by_task[task] = geom.load_task_data(args, task, row)
    return data_by_task


def apply_local_activation_fallbacks(args: argparse.Namespace) -> None:
    def has_layer(directory: Path, operation: str) -> bool:
        path = directory / f"{operation}_baseline.pt"
        if not path.exists():
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        return int(args.layer) in [int(value) for value in payload.get("block_layers", [])]

    text_probe = geom.activation_path_for(args, "text", "addition")
    if not text_probe.exists():
        root = Path(".local/banks/baseline_bank") / args.model / "digits"
        for fallback in [root / "activations", root / "activations_odd_layers", root / "activations_fourier_layers_32_48"]:
            if has_layer(fallback, "addition"):
                print(f"Using local text activation fallback: {fallback}")
                args.activation_dir_text = fallback
                break
    image_probe = geom.activation_path_for(args, "image", "addition")
    if not image_probe.exists():
        root = Path(".local/banks/baseline_images_bank") / args.model / "digits"
        for fallback in [root / "activations", root / "activations_odd_layers", root / "activations_fourier_layers_32_48"]:
            if has_layer(fallback, "addition"):
                print(f"Using local image activation fallback: {fallback}")
                args.activation_dir_image = fallback
                break


def repeat_centroids_for_basis(repeat_data: RepeatData, basis: torch.Tensor, values: set[int]) -> dict[int, torch.Tensor]:
    coords = repeat_data.hidden @ basis
    centroids = {}
    for label, coord in zip(repeat_data.labels, coords):
        value = int(label["result"])
        if value in values:
            centroids[value] = coord
    return centroids


def build_lens_items(
    args: argparse.Namespace,
    tasks: list[str],
    data_by_task: dict[str, geom.TaskData],
    repeat_data: RepeatData,
    common_values: list[int],
) -> tuple[list[LensItem], list[LensItem], list[dict]]:
    items = []
    random_items = []
    diagnostics = []
    common_set = set(common_values)
    for task in tasks:
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
            arith_centroids = geom.centroid_coordinates(data, l_basis, common_set, args.target)
            repeat_centroids = repeat_centroids_for_basis(repeat_data, l_basis, common_set)
            row = {
                **diag,
                "space_type": "causal_L",
                "repeat_values_available": sorted(repeat_centroids),
                "n_repeat_values_available": len(repeat_centroids),
            }
            diagnostics.append(row)
            items.append(
                LensItem(task, TASK_LABELS[task], seed, "causal_L", l_basis, arith_centroids, repeat_centroids, row)
            )
            print(
                f"LENS task={task} seed={seed} rank={l_basis.shape[1]} "
                f"repeat_values={len(repeat_centroids)} digit_overlap={diag['digit_overlap']:.3e}"
            )
        for control_seed in args.control_seeds:
            basis, meta = causal_l.random_readout_orthogonal_basis(
                data.hidden.shape[1],
                args.latent_dim,
                digit_basis,
                stable_seed(EXPERIMENT, "random_readout_orthogonal_13d", task, control_seed),
            )
            arith_centroids = geom.centroid_coordinates(data, basis, common_set, args.target)
            repeat_centroids = repeat_centroids_for_basis(repeat_data, basis, common_set)
            row = {
                "task": task,
                "task_label": TASK_LABELS[task],
                "das_seed": control_seed,
                "control_seed": control_seed,
                "space_type": "random_readout_orthogonal_13d",
                "rank": int(basis.shape[1]),
                "rank_L": int(basis.shape[1]),
                "orthogonality_error": float((basis.T @ basis - torch.eye(basis.shape[1])).norm()),
                "digit_overlap": float((digit_basis.T @ basis).norm()),
                "repeat_values_available": sorted(repeat_centroids),
                "n_repeat_values_available": len(repeat_centroids),
                **meta,
            }
            diagnostics.append(row)
            random_items.append(
                LensItem(task, TASK_LABELS[task], control_seed, "random_readout_orthogonal_13d", basis, arith_centroids, repeat_centroids, row)
            )
    return items, random_items, diagnostics


def build_l_space_diagnostics_only(
    args: argparse.Namespace,
    tasks: list[str],
    data_by_task: dict[str, geom.TaskData],
) -> list[dict]:
    diagnostics = []
    for task in tasks:
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
            diagnostics.append(
                {
                    **diag,
                    "space_type": "causal_L",
                    "artifact_check_only": True,
                    "repeat_values_available": [],
                    "n_repeat_values_available": 0,
                    "note": "Exact L reconstructed without repeat activations.",
                }
            )
            print(
                f"L_CHECK task={task} seed={seed} rank={l_basis.shape[1]} "
                f"digit_overlap={diag['digit_overlap']:.3e}"
            )
    return diagnostics


def centered_matrix(centroids: dict[int, torch.Tensor], values: list[int], train_values: list[int]) -> torch.Tensor:
    mean_vector = torch.stack([centroids[value] for value in train_values]).mean(dim=0)
    return torch.stack([centroids[value] - mean_vector for value in values])


def nearest_metrics(source: torch.Tensor, destination: torch.Tensor, source_values: list[int], destination_values: list[int]) -> tuple[dict, list[dict]]:
    exact = 0
    top5 = 0
    abs_errors = []
    confusions = []
    for row_index, value in enumerate(source_values):
        distances = (destination - source[row_index]).norm(dim=1)
        order = torch.argsort(distances).tolist()
        predicted = destination_values[order[0]]
        exact += int(predicted == value)
        top5 += int(value in [destination_values[i] for i in order[:5]])
        abs_errors.append(abs(predicted - value))
        if predicted != value:
            confusions.append({"target_value": value, "retrieved_value": predicted, **digit_confusion(value, predicted)})
    n = len(source_values)
    return {
        "top1": exact / n,
        "top5": top5 / n,
        "MAE": float(sum(abs_errors) / n),
        "median_absolute_value_error": float(statistics.median(abs_errors)),
    }, confusions


def digit_confusion(target: int, retrieved: int) -> dict:
    target_text = f"{int(target):02d}" if 0 <= int(target) <= 99 else str(target)
    retrieved_text = f"{int(retrieved):02d}" if 0 <= int(retrieved) <= 99 else str(retrieved)
    same_first = len(target_text) >= 2 and len(retrieved_text) >= 2 and target_text[0] == retrieved_text[0]
    same_second = len(target_text) >= 2 and len(retrieved_text) >= 2 and target_text[1] == retrieved_text[1]
    return {
        "same_first_digit": bool(same_first),
        "same_second_digit": bool(same_second),
        "neither_digit": not same_first and not same_second,
    }


def confusion_summary(confusions: list[dict]) -> dict:
    n = len(confusions)
    if n == 0:
        return {
            "n_errors": 0,
            "error_same_first_digit": None,
            "error_same_second_digit": None,
            "error_neither": None,
        }
    return {
        "n_errors": n,
        "error_same_first_digit": sum(bool(row["same_first_digit"]) for row in confusions) / n,
        "error_same_second_digit": sum(bool(row["same_second_digit"]) for row in confusions) / n,
        "error_neither": sum(bool(row["neither_digit"]) for row in confusions) / n,
    }


def direct_alignment(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int], test_values: list[int]) -> tuple[dict, list[dict]]:
    source_test = centered_matrix(source, test_values, train_values)
    destination_test = centered_matrix(destination, test_values, train_values)
    cosines = torch.nn.functional.cosine_similarity(source_test, destination_test, dim=1)
    nearest, confusions = nearest_metrics(source_test, destination_test, test_values, test_values)
    return {
        "direct_same_value_cosine": float(cosines.mean()),
        "direct_same_value_cosine_median": float(cosines.median()),
        "direct_top1_retrieval": nearest["top1"],
        "direct_top5_retrieval": nearest["top5"],
        "direct_numeric_MAE": nearest["MAE"],
        "direct_median_absolute_value_error": nearest["median_absolute_value_error"],
    }, confusions


def fit_scale_translation(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int]) -> tuple[float, torch.Tensor]:
    source_train = torch.stack([source[value] for value in train_values])
    destination_train = torch.stack([destination[value] for value in train_values])
    source_mean = source_train.mean(dim=0)
    destination_mean = destination_train.mean(dim=0)
    xs = source_train - source_mean
    ys = destination_train - destination_mean
    alpha = float((xs * ys).sum() / xs.square().sum().clamp_min(1e-12))
    offset = destination_mean - alpha * source_mean
    return alpha, offset


def mapped_retrieval(
    source: dict[int, torch.Tensor],
    destination: dict[int, torch.Tensor],
    test_values: list[int],
    map_fn,
) -> tuple[dict, list[dict]]:
    mapped = torch.stack([map_fn(source[value]) for value in test_values])
    destination_matrix = torch.stack([destination[value] for value in test_values])
    destination_center = destination_matrix.mean(dim=0)
    cosines = torch.nn.functional.cosine_similarity(mapped - destination_center, destination_matrix - destination_center, dim=1)
    nearest, confusions = nearest_metrics(mapped, destination_matrix, test_values, test_values)
    return {
        "same_value_cosine_mean": float(cosines.mean()),
        "same_value_cosine_median": float(cosines.median()),
        "top1": nearest["top1"],
        "top5": nearest["top5"],
        "MAE": nearest["MAE"],
        "median_absolute_value_error": nearest["median_absolute_value_error"],
    }, confusions


def scale_translation_metrics(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int], test_values: list[int]) -> tuple[dict, list[dict]]:
    alpha, offset = fit_scale_translation(source, destination, train_values)
    _train_transitions, _x_train = geom.transition_matrix(source, train_values)
    test_transitions, x_test = geom.transition_matrix(source, test_values)
    _test_transitions_b, y_test = geom.transition_matrix(destination, test_values)
    q = torch.eye(x_test.shape[1])
    transition = geom.transition_metrics(x_test, y_test, q, alpha)
    retrieval, confusions = mapped_retrieval(source, destination, test_values, lambda vector: offset + alpha * vector)
    return {
        "alpha": alpha,
        "n_test_transitions": len(test_transitions),
        "transition_cosine": transition["heldout_transition_cosine_mean"],
        "relative_error": transition["heldout_relative_error_mean"],
        "same_value_cosine": retrieval["same_value_cosine_mean"],
        "top1": retrieval["top1"],
        "top5": retrieval["top5"],
        "MAE": retrieval["MAE"],
    }, confusions


def procrustes_metrics(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], train_values: list[int], test_values: list[int]) -> tuple[dict, list[dict]]:
    train_transitions, x_train = geom.transition_matrix(source, train_values)
    test_transitions, x_test = geom.transition_matrix(source, test_values)
    _train_transitions_b, y_train = geom.transition_matrix(destination, train_values)
    _test_transitions_b, y_test = geom.transition_matrix(destination, test_values)
    q, alpha = geom.fit_scaled_procrustes(x_train, y_train)
    transition = geom.transition_metrics(x_test, y_test, q, alpha)
    retrieval = geom.retrieval_metrics(source, destination, train_values, test_values, q, alpha)
    offset = torch.stack([destination[value] - float(alpha) * (source[value] @ q) for value in train_values]).mean(dim=0)
    _retrieval2, confusions = mapped_retrieval(source, destination, test_values, lambda vector: offset + float(alpha) * (vector @ q))
    return {
        "alpha": alpha,
        "n_train_transitions": len(train_transitions),
        "n_test_transitions": len(test_transitions),
        "heldout_transition_cosine": transition["heldout_transition_cosine_mean"],
        "heldout_relative_error": transition["heldout_relative_error_mean"],
        "top1": retrieval["top1_value_retrieval"],
        "top5": retrieval["top5_value_retrieval"],
        "same_value_cosine": retrieval["same_value_cosine_mean"],
        "MAE": retrieval["mean_absolute_value_error"],
    }, confusions


def rsa_metrics(source: dict[int, torch.Tensor], destination: dict[int, torch.Tensor], test_values: list[int], permutations: int, seed: int) -> dict:
    values = sorted(test_values)
    observed = geom.spearman(geom.distance_vector(source, values), geom.distance_vector(destination, values))
    rng = random.Random(seed)
    null = []
    for _ in range(permutations):
        shuffled = list(values)
        rng.shuffle(shuffled)
        paired = {value: destination[shuffled[index]] for index, value in enumerate(values)}
        null.append(geom.spearman(geom.distance_vector(source, values), geom.distance_vector(paired, values)))
    pvalue = (1 + sum(item >= observed for item in null)) / (permutations + 1) if permutations else None
    return {
        "spearman_rsa": observed,
        "permutation_pvalue": pvalue,
        "rsa_permutations": permutations,
        "permutation_null_mean": None if not null else float(sum(null) / len(null)),
        "permutation_null_std": None if len(null) < 2 else float(statistics.stdev(null)),
    }


def common_available_values(item: LensItem, values: list[int]) -> list[int]:
    return [value for value in values if value in item.arith_centroids and value in item.repeat_centroids]


def base_row(item: LensItem, split_seed: int, domain: str, train_values: list[int], test_values: list[int], direction: str) -> dict:
    return {
        "task": item.task,
        "task_label": item.task_label,
        "das_seed": item.das_seed,
        "space_type": item.space_type,
        "value_split_seed": split_seed,
        "value_domain": domain,
        "direction": direction,
        "source_task": REPEAT_TASK if direction == "repeat_to_arithmetic" else item.task,
        "destination_task": item.task if direction == "repeat_to_arithmetic" else REPEAT_TASK,
        "task_relation": f"{REPEAT_LABEL}->{item.task_label}" if direction == "repeat_to_arithmetic" else f"{item.task_label}->{REPEAT_LABEL}",
        "n_train_values": len(train_values),
        "n_test_values": len(test_values),
        "chance_top1": 1 / len(test_values),
    }


def analyze_items(args: argparse.Namespace, items: list[LensItem], value_splits: dict[str, dict[str, list[int]]], *, control: bool = False):
    direct_rows = []
    scale_rows = []
    procrustes_rows = []
    retrieval_rows = []
    rsa_rows = []
    controls = []
    confusion_rows = []
    for split_seed_text, split in value_splits.items():
        split_seed = int(split_seed_text)
        for domain, (raw_train, raw_test) in split_domains(split["train_values"], split["test_values"]).items():
            for item in items:
                train_values = common_available_values(item, raw_train)
                test_values = common_available_values(item, raw_test)
                if len(train_values) < 2 or len(test_values) < 1:
                    continue
                if domain == "one_digit" and len(test_values) < 2:
                    continue
                pairs = [
                    ("repeat_to_arithmetic", item.repeat_centroids, item.arith_centroids),
                    ("arithmetic_to_repeat", item.arith_centroids, item.repeat_centroids),
                ]
                for direction, source, destination in pairs:
                    row_base = base_row(item, split_seed, domain, train_values, test_values, direction)
                    direct, direct_confusions = direct_alignment(source, destination, train_values, test_values)
                    direct_row = {**row_base, **direct}
                    direct_rows.append(direct_row)
                    retrieval_rows.append({**row_base, "analysis": "direct", "top1": direct["direct_top1_retrieval"], "top5": direct["direct_top5_retrieval"], "MAE": direct["direct_numeric_MAE"], "same_value_cosine": direct["direct_same_value_cosine"]})
                    scale, scale_confusions = scale_translation_metrics(source, destination, train_values, test_values)
                    scale_row = {**row_base, **scale}
                    scale_rows.append(scale_row)
                    retrieval_rows.append({**row_base, "analysis": "scale_translation", "top1": scale["top1"], "top5": scale["top5"], "MAE": scale["MAE"], "same_value_cosine": scale["same_value_cosine"]})
                    proc, proc_confusions = procrustes_metrics(source, destination, train_values, test_values)
                    proc_row = {**row_base, **proc}
                    procrustes_rows.append(proc_row)
                    retrieval_rows.append({**row_base, "analysis": "procrustes", "top1": proc["top1"], "top5": proc["top5"], "MAE": proc["MAE"], "same_value_cosine": proc["same_value_cosine"]})
                    for analysis, confusions in [
                        ("direct", direct_confusions),
                        ("scale_translation", scale_confusions),
                        ("procrustes", proc_confusions),
                    ]:
                        if domain != PRIMARY_DOMAIN:
                            continue
                        confusion_rows.append({**row_base, "analysis": analysis, **confusion_summary(confusions)})
                    if control:
                        controls.append({**proc_row, "control_type": item.space_type, "analysis": "procrustes"})
                        controls.append({**direct_row, "control_type": item.space_type, "analysis": "direct"})
                if len(test_values) >= 3:
                    rsa = rsa_metrics(
                        item.repeat_centroids,
                        item.arith_centroids,
                        test_values,
                        args.rsa_permutations,
                        stable_seed(EXPERIMENT, "rsa", item.task, item.das_seed, split_seed, domain, item.space_type),
                    )
                    rsa_row = {
                        "task": item.task,
                        "task_label": item.task_label,
                        "das_seed": item.das_seed,
                        "space_type": item.space_type,
                        "value_split_seed": split_seed,
                        "value_domain": domain,
                        "task_relation": f"{REPEAT_LABEL}-{item.task_label}",
                        "n_test_values": len(test_values),
                        **rsa,
                    }
                    rsa_rows.append(rsa_row)
                    if control:
                        controls.append({**rsa_row, "control_type": item.space_type, "analysis": "rsa"})
                if not control and item.space_type == "causal_L":
                    rng = random.Random(stable_seed(EXPERIMENT, "shuffle", item.task, item.das_seed, split_seed, domain))
                    _train_transitions, x_train = geom.transition_matrix(item.repeat_centroids, train_values)
                    _test_transitions, x_test = geom.transition_matrix(item.repeat_centroids, test_values)
                    _test_dest, y_test = geom.transition_matrix(item.arith_centroids, test_values)
                    for shuffle_index in range(args.n_shuffles):
                        shuffled = list(train_values)
                        rng.shuffle(shuffled)
                        shuffled_destination = {value: item.arith_centroids[shuffled[index]] for index, value in enumerate(train_values)}
                        _st, y_train_shuffled = geom.transition_matrix(shuffled_destination, train_values)
                        q, alpha = geom.fit_scaled_procrustes(x_train, y_train_shuffled)
                        transition = geom.transition_metrics(x_test, y_test, q, alpha)
                        retrieval = geom.retrieval_metrics(item.repeat_centroids, item.arith_centroids, train_values, test_values, q, alpha)
                        controls.append(
                            {
                                **base_row(item, split_seed, domain, train_values, test_values, "repeat_to_arithmetic"),
                                "control_type": "shuffled_repeat_train_labels",
                                "analysis": "procrustes",
                                "control_seed": shuffle_index,
                                "alpha": alpha,
                                "heldout_transition_cosine": transition["heldout_transition_cosine_mean"],
                                "heldout_relative_error": transition["heldout_relative_error_mean"],
                                "top1": retrieval["top1_value_retrieval"],
                                "top5": retrieval["top5_value_retrieval"],
                                "same_value_cosine": retrieval["same_value_cosine_mean"],
                                "MAE": retrieval["mean_absolute_value_error"],
                            }
                        )
    return direct_rows, scale_rows, procrustes_rows, retrieval_rows, rsa_rows, controls, confusion_rows


def mean_metric(rows: list[dict], metric: str) -> float | None:
    values = [float(row[metric]) for row in rows if row.get(metric) not in {None, ""}]
    return mean(values)


def summarize_group(rows: list[dict], metrics: list[str], keys: list[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output = []
    for key, parts in sorted(groups.items()):
        item = {keys[index]: key[index] for index in range(len(keys))}
        item["n"] = len(parts)
        for metric in metrics:
            values = [float(row[metric]) for row in parts if row.get(metric) not in {None, ""}]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = maybe_stdev(values)
        output.append(item)
    return output


def load_arithmetic_baseline(path: Path) -> dict:
    summary_path = path / "L_geometry_summary.json"
    if not summary_path.exists():
        return {"available": False, "path": str(summary_path)}
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    primary = summary.get("primary_metric_means", {}).get("causal_L", {})
    return {
        "available": True,
        "path": str(summary_path),
        "transition": primary.get("mean_heldout_transition_cosine"),
        "top1": primary.get("mean_top1_retrieval"),
        "RSA": primary.get("mean_cross_task_rsa"),
        "raw_primary_metric_means": primary,
    }


def comparison_table(args: argparse.Namespace, procrustes_rows: list[dict], rsa_rows: list[dict], controls: list[dict]) -> list[dict]:
    baseline = load_arithmetic_baseline(args.previous_geometry_dir)
    rows = []
    if baseline.get("available"):
        rows.append(
            {
                "comparison_item": "arithmetic<->arithmetic L",
                "transition": baseline["transition"],
                "top1": baseline["top1"],
                "RSA": baseline["RSA"],
                "source": baseline["path"],
            }
        )
    primary_proc = [row for row in procrustes_rows if row["value_domain"] == PRIMARY_DOMAIN and row["direction"] == "repeat_to_arithmetic" and row["space_type"] == "causal_L"]
    primary_rsa = [row for row in rsa_rows if row["value_domain"] == PRIMARY_DOMAIN and row["space_type"] == "causal_L"]
    rows.append(
        {
            "comparison_item": "repeat<->arithmetic L",
            "transition": mean_metric(primary_proc, "heldout_transition_cosine"),
            "top1": mean_metric(primary_proc, "top1"),
            "RSA": mean_metric(primary_rsa, "spearman_rsa"),
            "source": "this_experiment_primary_two_digit",
        }
    )
    shuffled = [row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "shuffled_repeat_train_labels" and row.get("analysis") == "procrustes"]
    rows.append(
        {
            "comparison_item": "shuffled repeat",
            "transition": mean_metric(shuffled, "heldout_transition_cosine"),
            "top1": mean_metric(shuffled, "top1"),
            "RSA": None,
            "source": "this_experiment_control_two_digit",
        }
    )
    random_proc = [row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "random_readout_orthogonal_13d" and row.get("analysis") == "procrustes" and row.get("direction") == "repeat_to_arithmetic"]
    random_rsa = [row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "random_readout_orthogonal_13d" and row.get("analysis") == "rsa"]
    rows.append(
        {
            "comparison_item": "random 13D RF",
            "transition": mean_metric(random_proc, "heldout_transition_cosine"),
            "top1": mean_metric(random_proc, "top1"),
            "RSA": mean_metric(random_rsa, "spearman_rsa"),
            "source": "this_experiment_control_two_digit",
        }
    )
    return rows


def nested_task_summary(direct_rows: list[dict], scale_rows: list[dict], procrustes_rows: list[dict], rsa_rows: list[dict]) -> dict:
    output = {}
    for row in direct_rows:
        if row["value_domain"] != PRIMARY_DOMAIN or row["space_type"] != "causal_L":
            continue
        slot = output.setdefault(row["task"], {}).setdefault(str(row["das_seed"]), {})
        if row["direction"] == "repeat_to_arithmetic":
            slot["DIRECT"] = {
                "direct_same_value_cosine": row["direct_same_value_cosine"],
                "direct_top1": row["direct_top1_retrieval"],
                "direct_top5": row["direct_top5_retrieval"],
                "direct_MAE": row["direct_numeric_MAE"],
            }
    for row in scale_rows:
        if row["value_domain"] != PRIMARY_DOMAIN or row["space_type"] != "causal_L" or row["direction"] != "repeat_to_arithmetic":
            continue
        output.setdefault(row["task"], {}).setdefault(str(row["das_seed"]), {})["SCALE_TRANSLATION"] = {
            "top1": row["top1"],
            "same_value_cosine": row["same_value_cosine"],
        }
    for row in procrustes_rows:
        if row["value_domain"] != PRIMARY_DOMAIN or row["space_type"] != "causal_L" or row["direction"] != "repeat_to_arithmetic":
            continue
        output.setdefault(row["task"], {}).setdefault(str(row["das_seed"]), {})["PROCRUSTES"] = {
            "transition_cosine": row["heldout_transition_cosine"],
            "relative_error": row["heldout_relative_error"],
            "top1": row["top1"],
            "top5": row["top5"],
            "same_value_cosine": row["same_value_cosine"],
            "MAE": row["MAE"],
        }
    for row in rsa_rows:
        if row["value_domain"] != PRIMARY_DOMAIN or row["space_type"] != "causal_L":
            continue
        output.setdefault(row["task"], {}).setdefault(str(row["das_seed"]), {})["RSA"] = {
            "spearman": row["spearman_rsa"],
            "permutation_pvalue": row["permutation_pvalue"],
        }
    return output


def aggregate_primary(direct_rows: list[dict], scale_rows: list[dict], procrustes_rows: list[dict], rsa_rows: list[dict], controls: list[dict]) -> dict:
    primary_filter = lambda row: row.get("value_domain") == PRIMARY_DOMAIN and row.get("direction") == "repeat_to_arithmetic" and row.get("space_type") == "causal_L"
    return {
        "direct": {
            "direct_same_value_cosine": mean_metric([row for row in direct_rows if primary_filter(row)], "direct_same_value_cosine"),
            "direct_top1": mean_metric([row for row in direct_rows if primary_filter(row)], "direct_top1_retrieval"),
            "direct_top5": mean_metric([row for row in direct_rows if primary_filter(row)], "direct_top5_retrieval"),
            "direct_MAE": mean_metric([row for row in direct_rows if primary_filter(row)], "direct_numeric_MAE"),
        },
        "scale_translation": {
            "top1": mean_metric([row for row in scale_rows if primary_filter(row)], "top1"),
            "same_value_cosine": mean_metric([row for row in scale_rows if primary_filter(row)], "same_value_cosine"),
        },
        "procrustes": {
            "transition_cosine": mean_metric([row for row in procrustes_rows if primary_filter(row)], "heldout_transition_cosine"),
            "relative_error": mean_metric([row for row in procrustes_rows if primary_filter(row)], "heldout_relative_error"),
            "top1": mean_metric([row for row in procrustes_rows if primary_filter(row)], "top1"),
            "top5": mean_metric([row for row in procrustes_rows if primary_filter(row)], "top5"),
            "same_value_cosine": mean_metric([row for row in procrustes_rows if primary_filter(row)], "same_value_cosine"),
            "MAE": mean_metric([row for row in procrustes_rows if primary_filter(row)], "MAE"),
        },
        "rsa": {
            "spearman": mean_metric([row for row in rsa_rows if row.get("value_domain") == PRIMARY_DOMAIN and row.get("space_type") == "causal_L"], "spearman_rsa"),
            "permutation_pvalue": mean_metric([row for row in rsa_rows if row.get("value_domain") == PRIMARY_DOMAIN and row.get("space_type") == "causal_L"], "permutation_pvalue"),
        },
        "controls": {
            "shuffled_top1": mean_metric([row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "shuffled_repeat_train_labels"], "top1"),
            "shuffled_transition": mean_metric([row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "shuffled_repeat_train_labels"], "heldout_transition_cosine"),
            "random_top1": mean_metric([row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "random_readout_orthogonal_13d" and row.get("analysis") == "procrustes"], "top1"),
            "random_transition": mean_metric([row for row in controls if row.get("value_domain") == PRIMARY_DOMAIN and row.get("control_type") == "random_readout_orthogonal_13d" and row.get("analysis") == "procrustes"], "heldout_transition_cosine"),
        },
    }


def plot_figures(args: argparse.Namespace, direct_rows: list[dict], procrustes_rows: list[dict], rsa_rows: list[dict], controls: list[dict], comparison_rows: list[dict], confusion_rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = args.figure_dir or (args.output_dir / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    task_labels = [TASK_LABELS[task] for task in TASK_ORDER if task in args.tasks]
    x = np.arange(len(task_labels))
    width = 0.22

    def task_mean(rows, metric, task_label, **filters):
        values = [
            float(row[metric])
            for row in rows
            if row.get("task_label") == task_label
            and row.get("value_domain") == PRIMARY_DOMAIN
            and all(row.get(key) == value for key, value in filters.items())
            and row.get(metric) not in {None, ""}
        ]
        return np.nan if not values else float(np.mean(values))

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    direct = [task_mean(direct_rows, "direct_top1_retrieval", label, direction="repeat_to_arithmetic", space_type="causal_L") for label in task_labels]
    proc = [task_mean(procrustes_rows, "top1", label, direction="repeat_to_arithmetic", space_type="causal_L") for label in task_labels]
    random_values = [task_mean(controls, "top1", label, direction="repeat_to_arithmetic", control_type="random_readout_orthogonal_13d", analysis="procrustes") for label in task_labels]
    shuffled = [task_mean(controls, "top1", label, direction="repeat_to_arithmetic", control_type="shuffled_repeat_train_labels", analysis="procrustes") for label in task_labels]
    ax.bar(x - 1.5 * width, direct, width, label="direct")
    ax.bar(x - 0.5 * width, proc, width, label="Procrustes")
    ax.bar(x + 0.5 * width, random_values, width, label="random 13D RF")
    ax.bar(x + 1.5 * width, shuffled, width, label="shuffled")
    ax.axhline(1 / 18, color="black", linestyle="--", linewidth=1.0, label="chance")
    ax.set_xticks(x, task_labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("held-out top-1 value retrieval")
    ax.set_title("Repeat -> arithmetic retrieval in causal L")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure1_repeat_vs_arithmetic_retrieval.png", dpi=240)
    fig.savefig(figure_dir / "figure1_repeat_vs_arithmetic_retrieval.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    repeat_values = [task_mean(procrustes_rows, "heldout_transition_cosine", label, direction="repeat_to_arithmetic", space_type="causal_L") for label in task_labels]
    ax.bar(x, repeat_values, width=0.5, label="Repeat -> arithmetic L")
    baseline = next((row for row in comparison_rows if row["comparison_item"] == "arithmetic<->arithmetic L"), None)
    if baseline and baseline.get("transition") is not None:
        ax.axhline(float(baseline["transition"]), color="black", linestyle="--", linewidth=1.2, label="arith <-> arith L mean")
    ax.set_xticks(x, task_labels)
    ax.set_ylabel("held-out transition cosine")
    ax.set_title("Repeat geometry vs arithmetic reference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure2_geometry_comparison.png", dpi=240)
    fig.savefig(figure_dir / "figure2_geometry_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    rsa_values = [task_mean(rsa_rows, "spearman_rsa", label, space_type="causal_L") for label in task_labels]
    ax.bar(x, rsa_values, width=0.5)
    baseline = next((row for row in comparison_rows if row["comparison_item"] == "arithmetic<->arithmetic L"), None)
    if baseline and baseline.get("RSA") is not None:
        ax.axhline(float(baseline["RSA"]), color="black", linestyle="--", linewidth=1.2, label="arith <-> arith L mean")
        ax.legend(frameon=False)
    ax.set_xticks(x, task_labels)
    ax.set_ylabel("Spearman RSA")
    ax.set_title("No-fit repeat/arithmetic RSA")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure3_nofit_rsa.png", dpi=240)
    fig.savefig(figure_dir / "figure3_nofit_rsa.pdf")
    plt.close(fig)

    interpretable = [row for row in confusion_rows if row.get("analysis") == "procrustes" and row.get("n_errors", 0)]
    if interpretable:
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        categories = ["error_same_first_digit", "error_same_second_digit", "error_neither"]
        labels = ["same tens", "same units", "neither"]
        means = [np.mean([float(row[category]) for row in interpretable if row.get(category) not in {None, ""}]) for category in categories]
        ax.bar(np.arange(len(categories)), means)
        ax.set_xticks(np.arange(len(categories)), labels)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of retrieval errors")
        ax.set_title("Digit structure in incorrect retrievals")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / "figure4_digit_confusion.png", dpi=240)
        fig.savefig(figure_dir / "figure4_digit_confusion.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    args.tasks = normalize_tasks(args.tasks)
    if output_exists(args) and not args.artifact_check_only:
        raise FileExistsError(f"{args.output_dir} already has repeat_vs_arithmetic_L_summary.json; pass --force to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Repeat vs causal L geometry")
    print(f"  tasks={args.tasks}")
    print(f"  DAS seeds={args.seeds}")
    print(f"  value_split_seeds={args.value_split_seeds}")
    print(f"  repeat template={args.prompt_template!r}")
    print("  reused repeat task from src.experiments.readout_latent_geometry.controls.repeat_transfer")

    apply_local_activation_fallbacks(args)
    value_splits = load_exact_value_splits(args)
    common_values = all_split_values(value_splits)
    required_values = [value for value in common_values if args.min_value <= value <= args.max_value]
    repeat_data, inventory, accuracy_rows = prepare_repeat_data(args, required_values)
    if args.artifact_check_only:
        data_by_task = load_arithmetic_data(args, args.tasks)
        diagnostics = build_l_space_diagnostics_only(args, args.tasks, data_by_task)
        write_csv(diagnostics, args.output_dir / "repeat_L_space_diagnostics.csv")
        save_json(
            {
                "experiment": EXPERIMENT,
                "artifact_check_only": True,
                "tasks": args.tasks,
                "seeds": args.seeds,
                "value_splits": value_splits,
                "repeat_inventory": inventory,
                "repeat_activation_rows_loaded": len(repeat_data.labels),
                "L_space_diagnostics_rows": len(diagnostics),
                "config": jsonable(vars(args)),
            },
            args.output_dir / "artifact_check_summary.json",
        )
        print("ARTIFACT_CHECK_ONLY complete")
        return

    if args.accuracy_mode != "none":
        write_csv(accuracy_rows, args.output_dir / "repeat_accuracy.csv")
    accuracy_values = [bool(row["correct"]) for row in accuracy_rows] if accuracy_rows else [True for _ in repeat_data.labels]
    repeat_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
    print(f"Repeat accuracy ({args.accuracy_mode}): {repeat_accuracy}")

    data_by_task = load_arithmetic_data(args, args.tasks)
    for task, data in data_by_task.items():
        missing = [value for value in required_values if data.counts.get(value, 0) < 1]
        if missing:
            raise ValueError(f"{task} is missing required arithmetic values: {missing[:10]}")
    repeat_missing = [value for value in required_values if repeat_data.counts.get(value, 0) < 1]
    if repeat_missing:
        raise ValueError(f"Repeat task is missing correct cached values: {repeat_missing[:20]}")

    lens_items, random_items, diagnostics = build_lens_items(args, args.tasks, data_by_task, repeat_data, required_values)
    write_csv(diagnostics, args.output_dir / "repeat_L_space_diagnostics.csv")
    save_json(value_splits, args.output_dir / "value_splits.json")

    direct_rows, scale_rows, procrustes_rows, retrieval_rows, rsa_rows, control_rows, confusion_rows = analyze_items(args, lens_items, value_splits, control=False)
    random_direct, random_scale, random_proc, random_retrieval, random_rsa, random_controls, random_confusions = analyze_items(args, random_items, value_splits, control=True)
    control_rows.extend(random_controls)
    confusion_rows.extend(random_confusions)

    write_csv(direct_rows, args.output_dir / "repeat_direct_alignment.csv")
    write_csv(scale_rows, args.output_dir / "repeat_scale_translation.csv")
    write_csv(procrustes_rows, args.output_dir / "repeat_procrustes.csv")
    write_csv(retrieval_rows, args.output_dir / "repeat_retrieval.csv")
    write_csv(rsa_rows, args.output_dir / "repeat_rsa.csv")
    write_csv(control_rows, args.output_dir / "repeat_controls.csv")
    write_csv(confusion_rows, args.output_dir / "repeat_digit_confusions.csv")

    comparison_rows = comparison_table(args, procrustes_rows, rsa_rows, control_rows)
    write_csv(comparison_rows, args.output_dir / "repeat_vs_arithmetic_L_final_comparison.csv")
    summary = {
        "experiment": EXPERIMENT,
        "description": "Repeat-number activations viewed through exact 13-D causal arithmetic L bases.",
        "tasks": args.tasks,
        "task_labels": TASK_LABELS,
        "seeds": args.seeds,
        "value_splits": value_splits,
        "primary_value_domain": PRIMARY_DOMAIN,
        "repeat_accuracy": repeat_accuracy,
        "repeat_accuracy_mode": args.accuracy_mode,
        "repeat_inventory": inventory,
        "per_task_seed": nested_task_summary(direct_rows, scale_rows, procrustes_rows, rsa_rows),
        "aggregate_primary_two_digit": aggregate_primary(direct_rows, scale_rows, procrustes_rows, rsa_rows, control_rows),
        "final_comparison_table": comparison_rows,
        "direct_summary": summarize_group(direct_rows, ["direct_same_value_cosine", "direct_top1_retrieval", "direct_top5_retrieval", "direct_numeric_MAE"], ["task", "task_label", "value_domain", "direction", "space_type"]),
        "scale_translation_summary": summarize_group(scale_rows, ["transition_cosine", "top1", "same_value_cosine", "MAE"], ["task", "task_label", "value_domain", "direction", "space_type"]),
        "procrustes_summary": summarize_group(procrustes_rows, ["heldout_transition_cosine", "heldout_relative_error", "top1", "top5", "same_value_cosine", "MAE"], ["task", "task_label", "value_domain", "direction", "space_type"]),
        "rsa_summary": summarize_group(rsa_rows, ["spearman_rsa", "permutation_pvalue"], ["task", "task_label", "value_domain", "space_type"]),
        "control_summary": summarize_group(control_rows, ["heldout_transition_cosine", "top1", "same_value_cosine", "direct_top1_retrieval", "direct_same_value_cosine"], ["task", "task_label", "value_domain", "control_type", "analysis"]),
        "rows": {
            "direct": len(direct_rows),
            "scale_translation": len(scale_rows),
            "procrustes": len(procrustes_rows),
            "retrieval": len(retrieval_rows),
            "rsa": len(rsa_rows),
            "controls": len(control_rows),
            "digit_confusions": len(confusion_rows),
            "diagnostics": len(diagnostics),
        },
        "config": jsonable(vars(args)),
    }
    save_json(summary, args.output_dir / "repeat_vs_arithmetic_L_summary.json")
    if not args.skip_plots:
        plot_figures(args, direct_rows, procrustes_rows, rsa_rows, control_rows, comparison_rows, confusion_rows)

    print("\nRepeat vs causal L geometry complete")
    print(f"Output directory: {args.output_dir}")
    print(f"Direct rows: {len(direct_rows)}")
    print(f"Procrustes rows: {len(procrustes_rows)}")
    print(f"RSA rows: {len(rsa_rows)}")
    print(f"Control rows: {len(control_rows)}")


if __name__ == "__main__":
    main()
