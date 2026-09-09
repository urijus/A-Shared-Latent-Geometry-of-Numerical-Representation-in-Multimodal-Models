"""Coordinate-aligned causal transport between DAS spaces.

The experiment evaluates interpretable alignment variants:

1. basic: fit an orthogonal Procrustes map on absolute, centered DAS
   coordinates z(x).
2. random_orthogonal: use a random orthogonal map as a causal and geometric
   control.
3. displacement: fit Procrustes directly on base-to-donor DAS displacements.
4. centroid_displacement: fit on value-to-value centroid displacements.
5. scaled_displacement: fit the displacement map plus one scalar alpha.
6. scaled_random_orthogonal: use the learned alpha with random Q.
7. alpha_identity: use the learned alpha with Q fixed to identity.
8. cross_operation_scaled: fit scaled Q on the opposite operation.

Each variant is saved in its own output folder.
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from tqdm import tqdm

from src.geometry.alignment import (
    coordinate_metrics,
    orthogonal_procrustes,
    random_orthogonal,
    scaled_alpha,
    stable_seed,
)

from src.interventions.das import (
    build_unique_pairs,
    format_prompt,
    hidden,
    hook_module,
    replace_hidden,
    resolve_position,
    sample_id as das_sample_id,
    split_samples,
    target_answers,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
    DEFAULT_TASKS,
    control_iia,
    first_result_row,
    heldout_pairs_path,
    jsonable,
    label,
    load_basis,
    load_jsonl,
    normalized_transfer,
    parse_task,
    position_for,
    results_path,
    save_json,
    save_jsonl,
    selected_seeds,
    subspace_path,
    task_key,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_ALIGNMENTS = [
    "text:addition->image:addition",
    "image:addition->text:addition",
    "text:subtraction->image:subtraction",
    "image:subtraction->text:subtraction",
]

ALL_VARIANTS = [
    "basic",
    "random_orthogonal",
    "displacement",
    "centroid_displacement",
    "scaled_displacement",
    "scaled_random_orthogonal",
    "alpha_identity",
    "cross_operation_scaled",
]

DEFAULT_VARIANTS = [
    "scaled_displacement",
    "scaled_random_orthogonal",
    "alpha_identity",
    "cross_operation_scaled",
]

VARIANT_DESCRIPTIONS = {
    "basic": "orthogonal map fit on centered absolute DAS coordinates",
    "random_orthogonal": "random orthogonal map control",
    "displacement": "orthogonal map fit on matched base-to-donor DAS displacements",
    "centroid_displacement": "orthogonal map fit on value-to-value centroid displacements",
    "scaled_displacement": "displacement map with one learned scalar alpha",
    "scaled_random_orthogonal": "random orthogonal control using the alpha from the learned scaled map",
    "alpha_identity": "alpha-only control with Q fixed to identity",
    "cross_operation_scaled": "scaled map fit on the opposite operation and tested on this operation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--causal_transfer_rows", type=Path, default=Path("results/final_exps/causal_tranfer/transfer_results.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/procrustes"))
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--alignments", nargs="+", default=DEFAULT_ALIGNMENTS)
    parser.add_argument("--variants", nargs="+", choices=ALL_VARIANTS, default=DEFAULT_VARIANTS)
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
    parser.add_argument("--top_n", type=int, default=0)
    parser.add_argument("--seed_map", nargs="*", default=[], metavar="TASK=SEEDS")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--alignment_seed", type=int, default=0)
    parser.add_argument("--max_alignment_samples", type=int, default=2048)
    parser.add_argument("--max_alignment_eval_samples", type=int, default=2048)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_alignment(spec: str) -> tuple[str, str]:
    if "->" not in spec:
        raise ValueError(f"Alignment must be SOURCE->DESTINATION, got {spec!r}.")
    source, destination = spec.split("->", 1)
    return task_key(*parse_task(source)), task_key(*parse_task(destination))


def opposite_operation(operation: str) -> str:
    if operation == "addition":
        return "subtraction"
    if operation == "subtraction":
        return "addition"
    raise ValueError(f"Cross-operation Procrustes is defined for addition/subtraction, got {operation!r}.")


def sample_key(sample: dict):
    if "sample_id" in sample:
        return sample["sample_id"]
    return das_sample_id(sample)


def sample_lookup(samples: list[dict]) -> dict:
    return {sample_key(sample): sample for sample in samples}


def data_root_for(row: dict) -> Path:
    return Path(row.get("data_root") or Path(row["data_path"]).parent)


def variant_dir(args: argparse.Namespace, variant: str) -> Path:
    return args.output_dir / variant


def clean_name(task: str, seed: int) -> str:
    return f"{label(task)}_seed{seed}"


def alignment_file_name(
    variant: str,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    args: argparse.Namespace,
    *,
    fit_source_task: str | None = None,
    fit_destination_task: str | None = None,
) -> str:
    if fit_source_task is None or fit_destination_task is None:
        return (
            f"{variant}_{clean_name(source_task, source_seed)}"
            f"_to_{clean_name(destination_task, destination_seed)}"
            f"_layer{args.layer}_k{args.k}.pt"
        )
    return (
        f"{variant}_fit_{clean_name(fit_source_task, source_seed)}"
        f"_to_{clean_name(fit_destination_task, destination_seed)}"
        f"_test_{clean_name(source_task, source_seed)}"
        f"_to_{clean_name(destination_task, destination_seed)}"
        f"_layer{args.layer}_k{args.k}.pt"
    )


def load_model_bundle(args: argparse.Namespace, tasks: list[str]):
    needs_image = any(parse_task(task)[0] == "image" for task in tasks)
    model_path, model_name = resolve_model_for_loading(args.model)
    if needs_image:
        model, processor, tokenizer = load_hf_model_and_processor(model_path)
    else:
        model, tokenizer = load_hf_model(model_path)
        processor = None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    return model, processor, tokenizer, blocks, get_hidden_size(model), model_name


@torch.no_grad()
def collect_activations(
    *,
    model,
    processor,
    tokenizer,
    blocks,
    samples: list[dict],
    modality: str,
    layer: int,
    hook_name: str,
    position: str,
    data_root: Path | None,
    prompt: str,
    enable_thinking: bool,
    use_chat: bool,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in tqdm(range(0, len(samples), batch_size), desc=f"Collect {modality} activations"):
        batch = samples[start : start + batch_size]
        if modality == "text":
            prompts = [format_prompt(tokenizer, sample, use_chat) for sample in batch]
            positions = [resolve_position(tokenizer, text, position) for text in prompts]
            encoding = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(model.device)
        else:
            if processor is None or data_root is None:
                raise ValueError("Image activation collection requires a processor and data_root.")
            prompts = [sample_prompt(processor, sample, prompt, enable_thinking) for sample in batch]
            images = [load_rgb_image(image_path_for(sample, data_root)) for sample in batch]
            positions = resolve_batch_positions(processor, tokenizer, model, prompts, images, position)
            encoding = inputs_to_device(make_inputs(processor, prompts, images), model.device)

        captured = {}
        with ExitStack() as stack:
            module, pre_hook = hook_module(blocks[layer - 1], hook_name)
            if pre_hook:
                def capture_pre(_module, inputs):
                    captured[layer] = hidden(inputs[0]).detach()

                handle = module.register_forward_pre_hook(capture_pre)
            else:
                def capture_post(_module, _inputs, output):
                    captured[layer] = hidden(output).detach()

                handle = module.register_forward_hook(capture_post)
            stack.callback(handle.remove)
            model(**encoding, use_cache=False)

        rows = torch.arange(len(batch), device=model.device)
        token_positions = torch.tensor(positions, device=model.device)
        parts.append(captured[layer][rows, token_positions].float().cpu())
    return torch.cat(parts, dim=0)


def cached_task_samples(sample_cache: dict, task: str, row: dict) -> list[dict]:
    if task not in sample_cache:
        sample_cache[task] = load_jsonl(Path(row["data_path"]))
    return sample_cache[task]


def sample_split_ids(args: argparse.Namespace, samples: list[dict], part: str) -> set:
    train, validation, test = split_samples(
        samples, args.train_fraction, args.validation_fraction, args.split_seed
    )
    split = {"train": train, "validation": validation, "test": test}[part]
    return {sample_key(sample) for sample in split}


def sample_split(args: argparse.Namespace, samples: list[dict], part: str) -> list[dict]:
    train, validation, test = split_samples(
        samples, args.train_fraction, args.validation_fraction, args.split_seed
    )
    return {"train": train, "validation": validation, "test": test}[part]


def activations_for_ids(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    task: str,
    row: dict,
    ids: list,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    key = (task, tuple(ids))
    if key in activation_cache:
        return activation_cache[key]
    modality, _operation = parse_task(task)
    samples = cached_task_samples(sample_cache, task, row)
    lookup = sample_lookup(samples)
    missing = [item for item in ids if item not in lookup]
    if missing:
        raise KeyError(f"{task} is missing {len(missing)} paired sample ids, first={missing[0]!r}.")
    features = collect_activations(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        samples=[lookup[item] for item in ids],
        modality=modality,
        layer=args.layer,
        hook_name=args.hook,
        position=position_for(args, modality),
        data_root=data_root_for(row) if modality == "image" else None,
        prompt=args.prompt,
        enable_thinking=args.enable_thinking,
        use_chat=args.use_chat_template or uses_chat_template(args.model),
        batch_size=args.activation_batch_size,
    )
    activation_cache[key] = features
    return features


def load_heldout_pairs(
    args: argparse.Namespace,
    modality: str,
    operation: str,
    seed: int,
    result_row: dict,
    max_pairs: int,
) -> list[dict]:
    samples = sample_lookup(load_jsonl(Path(result_row["data_path"])))
    rows = load_jsonl(heldout_pairs_path(args, modality, operation, seed))
    if max_pairs > 0:
        rows = rows[:max_pairs]
    return [
        {
            "pair_id": row.get("pair_id", index),
            "base": samples[row["base_sample_id"]],
            "source": samples[row["source_sample_id"]],
        }
        for index, row in enumerate(rows)
    ]


def coordinate_deltas_for_pairs(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    task: str,
    row: dict,
    pairs: list[dict],
    basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    ids = []
    for pair in pairs:
        ids.extend([sample_key(pair["base"]), sample_key(pair["source"])])
    unique_ids = list(dict.fromkeys(ids))
    activations = activations_for_ids(
        args,
        activation_cache,
        sample_cache,
        task=task,
        row=row,
        ids=unique_ids,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    by_id = {item: activations[index] for index, item in enumerate(unique_ids)}
    deltas = [
        (by_id[sample_key(pair["source"])] - by_id[sample_key(pair["base"])]) @ basis
        for pair in pairs
    ]
    return torch.stack(deltas) if deltas else torch.empty(0, basis.shape[1])


def basic_alignment_data(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    destination_task: str,
    source_row: dict,
    destination_row: dict,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_samples = cached_task_samples(sample_cache, source_task, source_row)
    destination_samples = cached_task_samples(sample_cache, destination_task, destination_row)
    train_ids = list(sample_split_ids(args, source_samples, "train") & sample_split_ids(args, destination_samples, "train"))
    test_ids = list(sample_split_ids(args, source_samples, "test") & sample_split_ids(args, destination_samples, "test"))
    random.Random(args.alignment_seed).shuffle(train_ids)
    random.Random(args.alignment_seed + 1).shuffle(test_ids)
    if args.max_alignment_samples > 0:
        train_ids = train_ids[: args.max_alignment_samples]
    if args.max_alignment_eval_samples > 0:
        test_ids = test_ids[: args.max_alignment_eval_samples]
    if len(train_ids) < 2:
        raise ValueError(f"Need at least two paired train samples for {source_task}->{destination_task}.")

    source_train = activations_for_ids(
        args, activation_cache, sample_cache, task=source_task, row=source_row,
        ids=train_ids, model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
    )
    destination_train = activations_for_ids(
        args, activation_cache, sample_cache, task=destination_task, row=destination_row,
        ids=train_ids, model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
    )
    source_mean = source_train.mean(dim=0)
    destination_mean = destination_train.mean(dim=0)
    x_train = (source_train - source_mean) @ source_basis
    y_train = (destination_train - destination_mean) @ destination_basis

    if len(test_ids) >= 2:
        source_test = activations_for_ids(
            args, activation_cache, sample_cache, task=source_task, row=source_row,
            ids=test_ids, model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
        )
        destination_test = activations_for_ids(
            args, activation_cache, sample_cache, task=destination_task, row=destination_row,
            ids=test_ids, model=model, processor=processor, tokenizer=tokenizer, blocks=blocks,
        )
        x_test = (source_test - source_mean) @ source_basis
        y_test = (destination_test - destination_mean) @ destination_basis
    else:
        x_test = torch.empty(0, source_basis.shape[1])
        y_test = torch.empty(0, destination_basis.shape[1])
    return x_train, y_train, x_test, y_test, source_mean, destination_mean


def displacement_alignment_data(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    destination_task: str,
    source_row: dict,
    destination_row: dict,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    destination_pairs: list[dict],
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train, y_train = displacement_train_data(
        args,
        activation_cache,
        sample_cache,
        source_task=source_task,
        destination_task=destination_task,
        source_row=source_row,
        destination_row=destination_row,
        source_basis=source_basis,
        destination_basis=destination_basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    eval_pairs = destination_pairs[: args.max_alignment_eval_samples] if args.max_alignment_eval_samples > 0 else destination_pairs
    x_test = coordinate_deltas_for_pairs(
        args, activation_cache, sample_cache, task=source_task, row=source_row,
        pairs=eval_pairs, basis=source_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    y_test = coordinate_deltas_for_pairs(
        args, activation_cache, sample_cache, task=destination_task, row=destination_row,
        pairs=eval_pairs, basis=destination_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    return x_train, y_train, x_test, y_test


def displacement_train_data(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    destination_task: str,
    source_row: dict,
    destination_row: dict,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_samples = cached_task_samples(sample_cache, source_task, source_row)
    train_samples = sample_split(args, source_samples, "train")
    train_pairs, _stats = build_unique_pairs(
        train_samples,
        args.target,
        args.alignment_seed,
        args.max_alignment_samples,
    )
    if len(train_pairs) < 2:
        raise ValueError(f"Need at least two train displacement pairs for {source_task}->{destination_task}.")

    x_train = coordinate_deltas_for_pairs(
        args, activation_cache, sample_cache, task=source_task, row=source_row,
        pairs=train_pairs, basis=source_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    y_train = coordinate_deltas_for_pairs(
        args, activation_cache, sample_cache, task=destination_task, row=destination_row,
        pairs=train_pairs, basis=destination_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    return x_train, y_train


def target_value(args: argparse.Namespace, sample: dict) -> int:
    if args.target in sample:
        return int(sample[args.target])
    return int(sample["result"])


def centroid_coordinates(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    task: str,
    row: dict,
    samples: list[dict],
    basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> dict[int, torch.Tensor]:
    ids = [sample_key(sample) for sample in samples]
    activations = activations_for_ids(
        args,
        activation_cache,
        sample_cache,
        task=task,
        row=row,
        ids=ids,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    grouped = {}
    for sample, coordinate in zip(samples, activations @ basis):
        grouped.setdefault(target_value(args, sample), []).append(coordinate)
    return {value: torch.stack(parts).mean(dim=0) for value, parts in grouped.items()}


def centroid_alignment_data(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    destination_task: str,
    source_row: dict,
    destination_row: dict,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    destination_pairs: list[dict],
    model,
    processor,
    tokenizer,
    blocks,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source_train = sample_split(args, cached_task_samples(sample_cache, source_task, source_row), "train")
    destination_train = sample_split(args, cached_task_samples(sample_cache, destination_task, destination_row), "train")
    source_centroids = centroid_coordinates(
        args, activation_cache, sample_cache, task=source_task, row=source_row,
        samples=source_train, basis=source_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    destination_centroids = centroid_coordinates(
        args, activation_cache, sample_cache, task=destination_task, row=destination_row,
        samples=destination_train, basis=destination_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    values = sorted(set(source_centroids) & set(destination_centroids))
    transitions = [(first, second) for first in values for second in values if first != second]
    random.Random(args.alignment_seed).shuffle(transitions)
    if args.max_alignment_samples > 0:
        transitions = transitions[: args.max_alignment_samples]
    if len(transitions) < 2:
        raise ValueError(f"Need at least two centroid transitions for {source_task}->{destination_task}.")
    x_train = torch.stack([source_centroids[end] - source_centroids[start] for start, end in transitions])
    y_train = torch.stack([destination_centroids[end] - destination_centroids[start] for start, end in transitions])

    eval_pairs = destination_pairs[: args.max_alignment_eval_samples] if args.max_alignment_eval_samples > 0 else destination_pairs
    x_test = coordinate_deltas_for_pairs(
        args, activation_cache, sample_cache, task=source_task, row=source_row,
        pairs=eval_pairs, basis=source_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    y_test = coordinate_deltas_for_pairs(
        args, activation_cache, sample_cache, task=destination_task, row=destination_row,
        pairs=eval_pairs, basis=destination_basis, model=model, processor=processor,
        tokenizer=tokenizer, blocks=blocks,
    )
    return x_train, y_train, x_test, y_test


def fit_variant(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    variant: str,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    destination_pairs: list[dict],
    model,
    processor,
    tokenizer,
    blocks,
    hidden_size: int,
) -> dict:
    source_modality, source_operation = parse_task(source_task)
    destination_modality, destination_operation = parse_task(destination_task)
    source_row = first_result_row(args, source_modality, source_operation, args.condition, source_seed)
    destination_row = first_result_row(args, destination_modality, destination_operation, args.condition, destination_seed)
    source_basis = load_basis(subspace_path(args, source_modality, source_operation, source_seed), args.layer, hidden_size, args.k)
    destination_basis = load_basis(subspace_path(args, destination_modality, destination_operation, destination_seed), args.layer, hidden_size, args.k)
    fit_source_task = source_task
    fit_destination_task = destination_task
    fit_source_row = source_row
    fit_destination_row = destination_row
    fit_source_basis = source_basis
    fit_destination_basis = destination_basis

    source_mean = None
    destination_mean = None
    if variant == "cross_operation_scaled":
        fit_source_task = task_key(source_modality, opposite_operation(source_operation))
        fit_destination_task = task_key(destination_modality, opposite_operation(destination_operation))
        fit_source_modality, fit_source_operation = parse_task(fit_source_task)
        fit_destination_modality, fit_destination_operation = parse_task(fit_destination_task)
        fit_source_row = first_result_row(args, fit_source_modality, fit_source_operation, args.condition, source_seed)
        fit_destination_row = first_result_row(
            args, fit_destination_modality, fit_destination_operation, args.condition, destination_seed
        )
        fit_source_basis = load_basis(
            subspace_path(args, fit_source_modality, fit_source_operation, source_seed),
            args.layer,
            hidden_size,
            args.k,
        )
        fit_destination_basis = load_basis(
            subspace_path(args, fit_destination_modality, fit_destination_operation, destination_seed),
            args.layer,
            hidden_size,
            args.k,
        )
        fit_kind = "base_to_donor_displacements_fit_on_opposite_operation"
        x_train, y_train = displacement_train_data(
            args,
            activation_cache,
            sample_cache,
            source_task=fit_source_task,
            destination_task=fit_destination_task,
            source_row=fit_source_row,
            destination_row=fit_destination_row,
            source_basis=fit_source_basis,
            destination_basis=fit_destination_basis,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        eval_pairs = destination_pairs[: args.max_alignment_eval_samples] if args.max_alignment_eval_samples > 0 else destination_pairs
        x_test = coordinate_deltas_for_pairs(
            args, activation_cache, sample_cache, task=source_task, row=source_row,
            pairs=eval_pairs, basis=source_basis, model=model, processor=processor,
            tokenizer=tokenizer, blocks=blocks,
        )
        y_test = coordinate_deltas_for_pairs(
            args, activation_cache, sample_cache, task=destination_task, row=destination_row,
            pairs=eval_pairs, basis=destination_basis, model=model, processor=processor,
            tokenizer=tokenizer, blocks=blocks,
        )
        print(f"    fitting on {fit_source_task}[{source_seed}] -> {fit_destination_task}[{destination_seed}]")
        print(f"    testing on {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]")
    elif variant in {"basic", "random_orthogonal"}:
        fit_kind = "centered_absolute_coordinates"
        x_train, y_train, x_test, y_test, source_mean, destination_mean = basic_alignment_data(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            destination_task=destination_task,
            source_row=source_row,
            destination_row=destination_row,
            source_basis=source_basis,
            destination_basis=destination_basis,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
    elif variant in {"displacement", "scaled_displacement", "scaled_random_orthogonal", "alpha_identity"}:
        fit_kind = "base_to_donor_displacements"
        x_train, y_train, x_test, y_test = displacement_alignment_data(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            destination_task=destination_task,
            source_row=source_row,
            destination_row=destination_row,
            source_basis=source_basis,
            destination_basis=destination_basis,
            destination_pairs=destination_pairs,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
    elif variant == "centroid_displacement":
        fit_kind = "value_centroid_displacements"
        x_train, y_train, x_test, y_test = centroid_alignment_data(
            args,
            activation_cache,
            sample_cache,
            source_task=source_task,
            destination_task=destination_task,
            source_row=source_row,
            destination_row=destination_row,
            source_basis=source_basis,
            destination_basis=destination_basis,
            destination_pairs=destination_pairs,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
    else:
        raise ValueError(f"Unsupported Procrustes variant: {variant}")

    random_seed = stable_seed(args.alignment_seed, variant, source_task, source_seed, destination_task, destination_seed)
    random_q = random_orthogonal(args.k, random_seed).to(dtype=x_train.dtype)
    learned_q = None
    learned_alpha = 1.0
    if variant == "random_orthogonal":
        q = random_q
        alpha = 1.0
    elif variant == "scaled_random_orthogonal":
        learned_q = orthogonal_procrustes(x_train, y_train)
        learned_alpha = scaled_alpha(x_train, y_train, learned_q)
        q = random_q
        alpha = learned_alpha
    elif variant == "alpha_identity":
        learned_q = orthogonal_procrustes(x_train, y_train)
        learned_alpha = scaled_alpha(x_train, y_train, learned_q)
        q = torch.eye(args.k, dtype=x_train.dtype)
        alpha = learned_alpha
    else:
        learned_q = orthogonal_procrustes(x_train, y_train)
        learned_alpha = (
            scaled_alpha(x_train, y_train, learned_q)
            if variant in {"scaled_displacement", "cross_operation_scaled"}
            else 1.0
        )
        q = learned_q
        alpha = learned_alpha

    metrics = coordinate_metrics(x_train, y_train, q, alpha)
    heldout_metrics = coordinate_metrics(x_test, y_test, q, alpha)
    random_alpha = learned_alpha if variant in {
        "scaled_displacement",
        "scaled_random_orthogonal",
        "alpha_identity",
        "cross_operation_scaled",
    } else 1.0
    random_metrics = coordinate_metrics(x_train, y_train, random_q, random_alpha)
    random_heldout_metrics = coordinate_metrics(x_test, y_test, random_q, random_alpha)

    path = (
        variant_dir(args, variant)
        / "matrices"
        / alignment_file_name(
            variant,
            source_task,
            source_seed,
            destination_task,
            destination_seed,
            args,
            fit_source_task=fit_source_task if variant == "cross_operation_scaled" else None,
            fit_destination_task=fit_destination_task if variant == "cross_operation_scaled" else None,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "das_coordinate_alignment",
            "variant": variant,
            "variant_description": VARIANT_DESCRIPTIONS[variant],
            "fit_kind": fit_kind,
            "Q_source_to_destination": q.cpu(),
            "learned_Q_source_to_destination": None if learned_q is None else learned_q.cpu(),
            "random_Q_source_to_destination": random_q.cpu(),
            "alpha": alpha,
            "learned_alpha": learned_alpha,
            "random_control_alpha": random_alpha,
            "source_basis": source_basis.cpu(),
            "destination_basis": destination_basis.cpu(),
            "source_hidden_mean": None if source_mean is None else source_mean.cpu(),
            "destination_hidden_mean": None if destination_mean is None else destination_mean.cpu(),
            "source_task": source_task,
            "source_seed": source_seed,
            "destination_task": destination_task,
            "destination_seed": destination_seed,
            "fit_source_task": fit_source_task,
            "fit_source_seed": source_seed,
            "fit_destination_task": fit_destination_task,
            "fit_destination_seed": destination_seed,
            "fit_source_subspace_path": str(
                subspace_path(args, *parse_task(fit_source_task), source_seed)
            ),
            "fit_destination_subspace_path": str(
                subspace_path(args, *parse_task(fit_destination_task), destination_seed)
            ),
            "layer": args.layer,
            "k": args.k,
            "hook": args.hook,
            "target": args.target,
            "metrics": metrics,
            "heldout_metrics": heldout_metrics,
            "random_control_metrics": random_metrics,
            "random_control_heldout_metrics": random_heldout_metrics,
            "config": jsonable(vars(args)),
        },
        path,
    )
    print(
        f"    fit kind={fit_kind}; alpha={alpha:.4f}; "
        f"test cosine={heldout_metrics['mean_cosine']}; "
        f"test rmse={heldout_metrics['root_mean_squared_error']}; "
        f"random test cosine={random_heldout_metrics['mean_cosine']}"
    )
    return {
        "q": q,
        "alpha": alpha,
        "learned_alpha": learned_alpha,
        "random_control_alpha": random_alpha,
        "source_basis": source_basis,
        "destination_basis": destination_basis,
        "source_row": source_row,
        "destination_row": destination_row,
        "fit_source_task": fit_source_task,
        "fit_source_seed": source_seed,
        "fit_destination_task": fit_destination_task,
        "fit_destination_seed": destination_seed,
        "alignment_path": path,
        "fit_kind": fit_kind,
        "metrics": metrics,
        "heldout_metrics": heldout_metrics,
        "random_metrics": random_metrics,
        "random_heldout_metrics": random_heldout_metrics,
    }


def patched_forward_with_deltas(
    model,
    encoding,
    blocks,
    layer: int,
    hook_name: str,
    positions: list[int],
    hidden_deltas: torch.Tensor,
):
    deltas = hidden_deltas.to(model.device)

    def patch_tensor(value):
        activations = hidden(value)
        updated = activations.clone()
        for row, position in enumerate(positions):
            updated[row, position] = activations[row, position] + deltas[row].to(dtype=activations.dtype)
        return replace_hidden(value, updated)

    with ExitStack() as stack:
        module, pre_hook = hook_module(blocks[layer - 1], hook_name)
        if pre_hook:
            def hook(_module, inputs):
                return (patch_tensor(inputs[0]), *inputs[1:])

            handle = module.register_forward_pre_hook(hook)
        else:
            def hook(_module, _inputs, output):
                return patch_tensor(output)

            handle = module.register_forward_hook(hook)
        stack.callback(handle.remove)
        return model(**encoding, use_cache=False)


@torch.no_grad()
def autoregressive_iia_text_aligned(
    model,
    tokenizer,
    blocks,
    pairs: list[dict],
    hidden_deltas: torch.Tensor,
    args: argparse.Namespace,
    description: str,
) -> float:
    correct = 0
    for pair, delta in tqdm(list(zip(pairs, hidden_deltas)), desc=description):
        base, source = pair["base"], pair["source"]
        base_prompt = format_prompt(tokenizer, base, args.use_chat_template or uses_chat_template(args.model))
        expected = target_answers(base, source, args.target)[1]
        base_position = resolve_position(tokenizer, base_prompt, args.text_position)
        base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            outputs = patched_forward_with_deltas(
                model,
                {"input_ids": base_ids[None]},
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            next_id = outputs.logits[0, base_ids.shape[0] - 1].argmax().reshape(1)
            generated.append(int(next_id.item()))
            base_ids = torch.cat([base_ids, next_id])
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs) if pairs else 0.0


@torch.no_grad()
def autoregressive_iia_image_aligned(
    model,
    processor,
    tokenizer,
    blocks,
    pairs: list[dict],
    hidden_deltas: torch.Tensor,
    data_root: Path,
    args: argparse.Namespace,
    description: str,
) -> float:
    correct = 0
    for pair, delta in tqdm(list(zip(pairs, hidden_deltas)), desc=description):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(base, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.image_position)[0]
        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            length = int(inputs["attention_mask"][0].sum())
            outputs = patched_forward_with_deltas(
                model,
                inputs,
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            next_id = outputs.logits[0, length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = next_id.item()
            inputs["attention_mask"][0, length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs) if pairs else 0.0


def hidden_deltas_for_pairs(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    *,
    source_task: str,
    source_row: dict,
    destination_pairs: list[dict],
    q: torch.Tensor,
    alpha: float,
    source_basis: torch.Tensor,
    destination_basis: torch.Tensor,
    model,
    processor,
    tokenizer,
    blocks,
) -> torch.Tensor:
    source_coordinate_deltas = coordinate_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        task=source_task,
        row=source_row,
        pairs=destination_pairs,
        basis=source_basis,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    return (float(alpha) * (source_coordinate_deltas @ q)) @ destination_basis.T


def load_causal_rows(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    return {
        (
            row["source_task"],
            int(row["source_seed"]),
            row["destination_task"],
            int(row["destination_seed"]),
        ): row
        for row in load_jsonl(path)
    }


def load_cached_rows(args: argparse.Namespace, variant: str) -> dict[tuple, dict]:
    path = variant_dir(args, variant) / "procrustes_results.jsonl"
    if args.force or not path.exists():
        return {}
    cache = {}
    for row in load_jsonl(path):
        if variant in DEFAULT_VARIANTS and (
            "fit_source_task" not in row or "random_control_alpha" not in row
        ):
            continue
        cache[
            (
                row["source_task"],
                int(row["source_seed"]),
                row["destination_task"],
                int(row["destination_seed"]),
            )
        ] = row
    return cache


def gain(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def evaluate_one(
    args: argparse.Namespace,
    activation_cache: dict,
    sample_cache: dict,
    causal_rows: dict,
    *,
    variant: str,
    source_task: str,
    source_seed: int,
    destination_task: str,
    destination_seed: int,
    model,
    processor,
    tokenizer,
    blocks,
    hidden_size: int,
    model_name: str,
) -> dict:
    source_modality, source_operation = parse_task(source_task)
    destination_modality, destination_operation = parse_task(destination_task)
    destination_row = first_result_row(args, destination_modality, destination_operation, args.condition, destination_seed)
    destination_pairs = load_heldout_pairs(
        args,
        destination_modality,
        destination_operation,
        destination_seed,
        destination_row,
        args.max_autoregressive_pairs,
    )
    destination_control = control_iia(args, destination_modality, destination_operation, destination_seed, destination_row)
    destination_self = float(destination_row["autoregressive_iia"])

    print(f"  {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]")
    fit = fit_variant(
        args,
        activation_cache,
        sample_cache,
        variant=variant,
        source_task=source_task,
        source_seed=source_seed,
        destination_task=destination_task,
        destination_seed=destination_seed,
        destination_pairs=destination_pairs,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        hidden_size=hidden_size,
    )
    deltas = hidden_deltas_for_pairs(
        args,
        activation_cache,
        sample_cache,
        source_task=source_task,
        source_row=fit["source_row"],
        destination_pairs=destination_pairs,
        q=fit["q"],
        alpha=fit["alpha"],
        source_basis=fit["source_basis"],
        destination_basis=fit["destination_basis"],
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
    )
    description = f"{variant} {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]"
    if destination_modality == "text":
        aligned_iia = autoregressive_iia_text_aligned(model, tokenizer, blocks, destination_pairs, deltas, args, description)
    else:
        aligned_iia = autoregressive_iia_image_aligned(
            model, processor, tokenizer, blocks, destination_pairs, deltas,
            data_root_for(destination_row), args, description,
        )

    aligned_normalized = normalized_transfer(aligned_iia, destination_control, destination_self)
    causal = causal_rows.get((source_task, source_seed, destination_task, destination_seed), {})
    causal_iia = causal.get("autoregressive_iia")
    causal_normalized = causal.get("destination_normalized_transfer")
    print(
        f"    AR IIA={aligned_iia:.4f}; normalized={aligned_normalized}; "
        f"gain over causal={gain(aligned_normalized, causal_normalized)}"
    )
    return {
        "model": model_name,
        "variant": variant,
        "variant_description": VARIANT_DESCRIPTIONS[variant],
        "fit_kind": fit["fit_kind"],
        "fit_source_task": fit["fit_source_task"],
        "fit_source_seed": fit["fit_source_seed"],
        "fit_destination_task": fit["fit_destination_task"],
        "fit_destination_seed": fit["fit_destination_seed"],
        "source_task": source_task,
        "source_modality": source_modality,
        "source_operation": source_operation,
        "source_seed": source_seed,
        "source_subspace_path": str(subspace_path(args, source_modality, source_operation, source_seed)),
        "destination_task": destination_task,
        "destination_modality": destination_modality,
        "destination_operation": destination_operation,
        "destination_seed": destination_seed,
        "destination_results_path": str(results_path(args, destination_modality, destination_operation, args.condition, destination_seed)),
        "destination_heldout_pairs_path": str(heldout_pairs_path(args, destination_modality, destination_operation, destination_seed)),
        "alignment_path": str(fit["alignment_path"]),
        "layer": args.layer,
        "k": args.k,
        "hook": args.hook,
        "source_position": position_for(args, source_modality),
        "destination_position": position_for(args, destination_modality),
        "target": args.target,
        "n_pairs": len(destination_pairs),
        "alpha": fit["alpha"],
        "learned_alpha": fit["learned_alpha"],
        "random_control_alpha": fit["random_control_alpha"],
        "alignment_fit": fit["metrics"],
        "alignment_fit_test": fit["heldout_metrics"],
        "random_control_fit": fit["random_metrics"],
        "random_control_test": fit["random_heldout_metrics"],
        "autoregressive_iia": aligned_iia,
        "destination_self_autoregressive_iia": destination_self,
        "control_condition": args.control_condition,
        "destination_control_autoregressive_iia": destination_control,
        "destination_normalized_transfer": aligned_normalized,
        "causal_autoregressive_iia": causal_iia,
        "causal_destination_normalized_transfer": causal_normalized,
        "autoregressive_iia_gain_over_causal": gain(aligned_iia, causal_iia),
        "destination_normalized_transfer_gain_over_causal": gain(aligned_normalized, causal_normalized),
    }


def mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def sample_std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return statistics.stdev(finite) if len(finite) > 1 else 0.0


def nested_metric(row: dict, metric: str):
    value = row
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def row_metric(row: dict, metric: str):
    return nested_metric(row, metric) if "." in metric else row.get(metric)


def clean_metric_name(metric: str) -> str:
    return metric.replace(".", "_")


def matrix_payload(rows: list[dict], tasks: list[str], metric: str) -> dict:
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
            values = [row_metric(row, metric) for row in cell_rows]
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
    return {"tasks": tasks, "labels": [label(task) for task in tasks], "metric": metric, "matrix": matrix, "rows": matrix_rows}


def write_outputs(args: argparse.Namespace, rows: list[dict], seed_selection: dict[str, list[int]], variant: str) -> None:
    rows.sort(
        key=lambda row: (
            row["source_task"],
            int(row["source_seed"]),
            row["destination_task"],
            int(row["destination_seed"]),
        )
    )
    out = variant_dir(args, variant)
    save_jsonl(rows, out / "procrustes_results.jsonl")
    save_json(
        {
            "variant": variant,
            "variant_description": VARIANT_DESCRIPTIONS[variant],
            "seed_selection": seed_selection,
            "alignments": args.alignments,
            "config": jsonable(vars(args)),
        },
        out / "manifest.json",
    )
    metrics = [
        "autoregressive_iia",
        "destination_normalized_transfer",
        "autoregressive_iia_gain_over_causal",
        "destination_normalized_transfer_gain_over_causal",
        "alignment_fit_test.mean_cosine",
        "alignment_fit_test.root_mean_squared_error",
        "random_control_test.mean_cosine",
        "random_control_test.root_mean_squared_error",
        "alignment_fit_test.mean_norm_ratio",
        "alpha",
    ]
    for metric in metrics:
        payload = matrix_payload(rows, args.tasks, metric)
        name = clean_metric_name(metric)
        save_json(payload, out / f"{name}_matrix.json")
        save_jsonl(payload["rows"], out / f"{name}_matrix.jsonl")


def write_root_manifest(args: argparse.Namespace, seed_selection: dict[str, list[int]]) -> None:
    save_json(
        {
            "variants": args.variants,
            "variant_descriptions": VARIANT_DESCRIPTIONS,
            "seed_selection": seed_selection,
            "alignments": args.alignments,
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def merge_rows(cache: dict[tuple, dict], rows: list[dict]) -> list[dict]:
    merged = dict(cache)
    for row in rows:
        merged[(row["source_task"], int(row["source_seed"]), row["destination_task"], int(row["destination_seed"]))] = row
    return list(merged.values())


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    alignments = [parse_alignment(item) for item in args.alignments]
    args.tasks = list(dict.fromkeys(args.tasks + [task for pair in alignments for task in pair]))
    seed_selection = selected_seeds(args)
    causal_rows = load_causal_rows(args.causal_transfer_rows)
    model, processor, tokenizer, blocks, hidden_size, model_name = load_model_bundle(args, args.tasks)

    print("Selected seeds:")
    for task in args.tasks:
        print(f"  {task}: {seed_selection[task]}")
    print("Alignment directions:")
    for source, destination in alignments:
        print(f"  {source} -> {destination}")
    print("Variants:")
    for variant in args.variants:
        print(f"  {variant}: {VARIANT_DESCRIPTIONS[variant]}")
    write_root_manifest(args, seed_selection)

    activation_cache = {}
    sample_cache = {}
    for variant in args.variants:
        print(f"\n=== Variant: {variant} ===")
        print(f"Description: {VARIANT_DESCRIPTIONS[variant]}")
        cache = load_cached_rows(args, variant)
        rows = []
        for source_task, destination_task in alignments:
            print(f"\nDirection: {source_task} -> {destination_task}")
            for source_seed in seed_selection[source_task]:
                for destination_seed in seed_selection[destination_task]:
                    key = (source_task, source_seed, destination_task, destination_seed)
                    if key in cache:
                        rows.append(cache[key])
                        print(f"  cached {source_task}[{source_seed}] -> {destination_task}[{destination_seed}]")
                        continue
                    rows.append(
                        evaluate_one(
                            args,
                            activation_cache,
                            sample_cache,
                            causal_rows,
                            variant=variant,
                            source_task=source_task,
                            source_seed=source_seed,
                            destination_task=destination_task,
                            destination_seed=destination_seed,
                            model=model,
                            processor=processor,
                            tokenizer=tokenizer,
                            blocks=blocks,
                            hidden_size=hidden_size,
                            model_name=model_name,
                        )
                    )
                    write_outputs(args, merge_rows(cache, rows), seed_selection, variant)
        final_rows = merge_rows(cache, rows)
        write_outputs(args, final_rows, seed_selection, variant)
        print(f"Saved {variant} rows: {variant_dir(args, variant) / 'procrustes_results.jsonl'}")
        print(f"Saved {variant} matrices under: {variant_dir(args, variant) / 'matrices'}")


if __name__ == "__main__":
    main()
