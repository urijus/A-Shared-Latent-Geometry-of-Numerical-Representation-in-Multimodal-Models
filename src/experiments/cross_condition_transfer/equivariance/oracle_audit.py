"""Audit the equivariance oracle against the original self-DAS patch.

The shared-space +1 experiment has an ``oracle_target_state`` control.  This
script checks that this oracle is really equivalent to normal DAS source-state
patching on the exact same y -> y+1 pairs.

It reports, per task/seed:

* original joint DAS patch greedy exact match;
* equivariance fixed-delta oracle greedy exact match;
* a teacher-forced top-k target-token diagnostic for both paths.

If the two patch paths agree, a low oracle is a property of the strict +1 pair
distribution.  If they diverge, the equivariance scorer needs fixing before the
G_{+1} causal result is interpretable.
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import torch
from tqdm import tqdm

from src.interventions.das import (
    format_prompt,
    patched_forward as das_patched_forward_text,
    resolve_position,
    target_answers,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    patched_forward as das_patched_forward_image,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import (
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
    subspace_from_basis,
    task_key,
)
from src.experiments.cross_condition_transfer.equivariance.equivariance import (
    FIT_TASK,
    TASKS,
    PlusOneMap,
    TaskSpace,
    data_root_for,
    hidden_deltas_for_map,
    load_model_bundle,
    load_sync,
    plus_pairs_for_task,
    shared_centroids,
    value_split,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import (
    cached_task_samples,
    patched_forward_with_deltas,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template


EXPERIMENT = "equivariance_oracle_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--sync_dir", type=Path, default=Path("results/final_exps/synchronization"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/equivariance_oracle_audit"))
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--fit_task", default=FIT_TASK)
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
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--value_test_fraction", type=float, default=0.2)
    parser.add_argument("--value_validation_fraction", type=float, default=0.2)
    parser.add_argument("--value_split_seed", type=int, default=0)
    parser.add_argument("--max_causal_pairs_per_domain", type=int, default=32)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    return parser.parse_args()


def parse_seed_map(items: list[str]) -> dict[str, list[int]]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Seed-map item must be TASK=SEEDS, got {item!r}.")
        task, seeds = item.split("=", 1)
        parsed[task_key(*parse_task(task))] = [int(seed) for seed in re.split(r"[,\s]+", seeds) if seed]
    return parsed


def matching_result_row(args: argparse.Namespace, task: str, seed: int) -> dict | None:
    modality, operation = parse_task(task)
    path = results_path(args, modality, operation, args.condition, seed)
    if not path.exists():
        return None
    for row in load_jsonl(path):
        if row_matches(args, row, modality):
            return row
    return None


def discover_seed_rows(args: argparse.Namespace, task: str) -> list[dict]:
    modality, operation = parse_task(task)
    split_dir = args.audit_root / modality / operation
    if args.target != "result":
        split_dir = split_dir / args.target
    split_dir = split_dir / args.condition / f"split_{args.split_seed}"
    candidates = []
    for seed_dir in sorted(split_dir.glob("seed_*")):
        match = re.match(r"seed_(\d+)$", seed_dir.name)
        if match is None:
            continue
        seed = int(match.group(1))
        row = matching_result_row(args, task, seed)
        checkpoint = subspace_path(args, modality, operation, seed)
        if row is not None and checkpoint.exists():
            candidates.append({"seed": seed, "row": row})
    candidates.sort(key=lambda item: float(item["row"].get("autoregressive_iia", -1.0)), reverse=True)
    return candidates


def selected_matching_seeds(args: argparse.Namespace) -> dict[str, list[int]]:
    explicit = parse_seed_map(args.seed_map)
    selection = {}
    for task in args.tasks:
        if task in explicit:
            usable = []
            for seed in explicit[task]:
                if matching_result_row(args, task, seed) is None:
                    raise ValueError(
                        f"{task} seed {seed} has no row matching layer={args.layer}, "
                        f"k={args.k}, position={position_for(args, parse_task(task)[0])}, hook={args.hook}."
                    )
                usable.append(seed)
            selection[task] = usable
            continue
        candidates = discover_seed_rows(args, task)
        if args.top_n > 0:
            candidates = candidates[: args.top_n]
        else:
            wanted = set(args.seeds)
            candidates = [item for item in candidates if item["seed"] in wanted]
        if not candidates:
            raise FileNotFoundError(
                f"No matching DAS row for {task} under {args.audit_root} "
                f"(layer={args.layer}, k={args.k}, position={position_for(args, parse_task(task)[0])}, hook={args.hook})."
            )
        selection[task] = [item["seed"] for item in candidates]
    return selection


def common_experiment_seeds(seed_selection: dict[str, list[int]]) -> list[int]:
    seed_sets = [set(seeds) for seeds in seed_selection.values()]
    common = sorted(set.intersection(*seed_sets)) if seed_sets else []
    if not common:
        raise ValueError(
            "Oracle audit needs the same seed to exist for every task; "
            f"matching seeds by task were {seed_selection}."
        )
    return common


def make_matching_spaces(args: argparse.Namespace, tasks: list[str], seed: int, hidden_size: int, sync: dict) -> dict[str, TaskSpace]:
    spaces = {}
    for task in tasks:
        modality, operation = parse_task(task)
        row = matching_result_row(args, task, seed)
        if row is None:
            raise FileNotFoundError(f"No matching DAS result row for {task} seed {seed}.")
        spaces[task] = TaskSpace(
            task=task,
            modality=modality,
            operation=operation,
            seed=seed,
            row=row,
            basis=load_basis(subspace_path(args, modality, operation, seed), args.layer, hidden_size, args.k),
            orientation=sync["orientations"][task],
            scale=sync["scales"][task],
        )
    return spaces


def number_from_ids(tokenizer, ids: list[int]) -> str | None:
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    match = re.match(r"-?\d+", text)
    return match.group() if match is not None else None


def answer_token_ids(tokenizer, answer: str) -> list[int]:
    ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Answer {answer!r} produced no tokens.")
    return [int(item) for item in ids]


def topk_contains_target(logits: torch.Tensor, target_id: int, k: int) -> bool:
    top = torch.topk(logits.float(), k=min(k, logits.shape[-1])).indices.tolist()
    return int(target_id) in top


@torch.no_grad()
def text_greedy_and_topk_joint_das(model, tokenizer, blocks, subspaces, pairs, args, description: str) -> list[dict]:
    rows = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    use_chat = args.use_chat_template or uses_chat_template(args.model)
    try:
        for index, pair in enumerate(tqdm(pairs, desc=description)):
            base, source = pair["base"], pair["source"]
            expected = target_answers(base, source, args.target)[1]
            target_ids = answer_token_ids(tokenizer, expected)
            base_prompt = format_prompt(tokenizer, base, use_chat)
            donor_prompt = format_prompt(tokenizer, source, use_chat)
            base_position = resolve_position(tokenizer, base_prompt, args.text_position)
            donor_position = resolve_position(tokenizer, donor_prompt, args.text_position)
            base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            donor_ids = tokenizer(donor_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            generated = []
            for step in range(args.max_new_tokens):
                base_length, donor_length = len(base_ids), len(donor_ids)
                length = max(base_length, donor_length)
                input_ids = torch.full((2, length), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
                attention_mask = torch.zeros_like(input_ids)
                input_ids[0, :base_length] = base_ids
                input_ids[1, :donor_length] = donor_ids
                attention_mask[0, :base_length] = 1
                attention_mask[1, :donor_length] = 1
                outputs = das_patched_forward_text(
                    model,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    blocks,
                    subspaces,
                    [args.layer],
                    args.hook,
                    [base_position],
                    [donor_position],
                    n_base_groups=1,
                )
                logits = outputs.logits[0, base_length - 1]
                next_id = logits.argmax().reshape(1)
                generated.append(int(next_id.item()))
                base_ids = torch.cat([base_ids, next_id])
                if next_id.item() == tokenizer.eos_token_id:
                    break
            prediction = number_from_ids(tokenizer, generated)
            base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            topk_hits = []
            for target_id in target_ids:
                base_length, donor_length = len(base_ids), len(donor_ids)
                length = max(base_length, donor_length)
                input_ids = torch.full((2, length), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
                attention_mask = torch.zeros_like(input_ids)
                input_ids[0, :base_length] = base_ids
                input_ids[1, :donor_length] = donor_ids
                attention_mask[0, :base_length] = 1
                attention_mask[1, :donor_length] = 1
                outputs = das_patched_forward_text(
                    model,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    blocks,
                    subspaces,
                    [args.layer],
                    args.hook,
                    [base_position],
                    [donor_position],
                    n_base_groups=1,
                )
                topk_hits.append(topk_contains_target(outputs.logits[0, base_length - 1], target_id, args.top_k))
                base_ids = torch.cat([base_ids, torch.tensor([target_id], device=model.device)])
            rows.append(
                {
                    "pair_id": pair.get("pair_id", index),
                    "expected": expected,
                    "prediction": prediction,
                    "exact_match": prediction == expected,
                    "target_tokens_all_topk": len(topk_hits) == len(target_ids) and all(topk_hits),
                    "target_tokens_topk_fraction": sum(topk_hits) / len(target_ids),
                    "target_token_count": len(target_ids),
                }
            )
    finally:
        tokenizer.padding_side = old_padding_side
    return rows


@torch.no_grad()
def text_greedy_and_topk_delta(model, tokenizer, blocks, pairs, hidden_deltas, args, description: str) -> list[dict]:
    rows = []
    use_chat = args.use_chat_template or uses_chat_template(args.model)
    for index, (pair, delta) in enumerate(tqdm(list(zip(pairs, hidden_deltas)), desc=description)):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        target_ids = answer_token_ids(tokenizer, expected)
        base_prompt = format_prompt(tokenizer, base, use_chat)
        base_position = resolve_position(tokenizer, base_prompt, args.text_position)
        base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        generated = []
        for step in range(args.max_new_tokens):
            outputs = patched_forward_with_deltas(
                model,
                {"input_ids": base_ids[None]},
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            logits = outputs.logits[0, base_ids.shape[0] - 1]
            next_id = logits.argmax().reshape(1)
            generated.append(int(next_id.item()))
            base_ids = torch.cat([base_ids, next_id])
            if next_id.item() == tokenizer.eos_token_id:
                break
        prediction = number_from_ids(tokenizer, generated)
        base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
        topk_hits = []
        for target_id in target_ids:
            outputs = patched_forward_with_deltas(
                model,
                {"input_ids": base_ids[None]},
                blocks,
                args.layer,
                args.hook,
                [base_position],
                delta[None],
            )
            topk_hits.append(topk_contains_target(outputs.logits[0, base_ids.shape[0] - 1], target_id, args.top_k))
            base_ids = torch.cat([base_ids, torch.tensor([target_id], device=model.device)])
        rows.append(
            {
                "pair_id": pair.get("pair_id", index),
                "expected": expected,
                "prediction": prediction,
                "exact_match": prediction == expected,
                "target_tokens_all_topk": len(topk_hits) == len(target_ids) and all(topk_hits),
                "target_tokens_topk_fraction": sum(topk_hits) / len(target_ids),
                "target_token_count": len(target_ids),
            }
        )
    return rows


@torch.no_grad()
def image_greedy_and_topk_joint_das(model, processor, tokenizer, blocks, subspaces, pairs, data_root, args, description: str) -> list[dict]:
    rows = []
    for index, pair in enumerate(tqdm(pairs, desc=description)):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        target_ids = answer_token_ids(tokenizer, expected)
        base_prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
        donor_prompt = sample_prompt(processor, source, args.prompt, args.enable_thinking)
        base_image = load_rgb_image(image_path_for(base, data_root))
        donor_image = load_rgb_image(image_path_for(source, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [base_prompt], [base_image], args.image_position)[0]
        donor_position = resolve_batch_positions(processor, tokenizer, model, [donor_prompt], [donor_image], args.image_position)[0]
        inputs = inputs_to_device(make_inputs(processor, [base_prompt, donor_prompt], [base_image, donor_image]), model.device)
        generated = []
        for step in range(args.max_new_tokens):
            base_length = int(inputs["attention_mask"][0].sum())
            outputs = das_patched_forward_image(
                model,
                inputs,
                blocks,
                subspaces,
                [args.layer],
                args.hook,
                [base_position],
                [donor_position],
                n_base_groups=1,
            )
            logits = outputs.logits[0, base_length - 1]
            next_id = logits.argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, base_length] = next_id.item()
            inputs["attention_mask"][0, base_length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        prediction = number_from_ids(tokenizer, generated)
        inputs = inputs_to_device(make_inputs(processor, [base_prompt, donor_prompt], [base_image, donor_image]), model.device)
        topk_hits = []
        for target_id in target_ids:
            base_length = int(inputs["attention_mask"][0].sum())
            outputs = das_patched_forward_image(
                model,
                inputs,
                blocks,
                subspaces,
                [args.layer],
                args.hook,
                [base_position],
                [donor_position],
                n_base_groups=1,
            )
            topk_hits.append(topk_contains_target(outputs.logits[0, base_length - 1], target_id, args.top_k))
            pad = torch.full((inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, base_length] = int(target_id)
            inputs["attention_mask"][0, base_length] = 1
        rows.append(
            {
                "pair_id": pair.get("pair_id", index),
                "expected": expected,
                "prediction": prediction,
                "exact_match": prediction == expected,
                "target_tokens_all_topk": len(topk_hits) == len(target_ids) and all(topk_hits),
                "target_tokens_topk_fraction": sum(topk_hits) / len(target_ids),
                "target_token_count": len(target_ids),
            }
        )
    return rows


@torch.no_grad()
def image_greedy_and_topk_delta(model, processor, tokenizer, blocks, pairs, hidden_deltas, data_root, args, description: str) -> list[dict]:
    rows = []
    for index, (pair, delta) in enumerate(tqdm(list(zip(pairs, hidden_deltas)), desc=description)):
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, args.target)[1]
        target_ids = answer_token_ids(tokenizer, expected)
        prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(base, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.image_position)[0]
        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        generated = []
        for step in range(args.max_new_tokens):
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
            logits = outputs.logits[0, length - 1]
            next_id = logits.argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = next_id.item()
            inputs["attention_mask"][0, length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        prediction = number_from_ids(tokenizer, generated)
        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        topk_hits = []
        for target_id in target_ids:
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
            topk_hits.append(topk_contains_target(outputs.logits[0, length - 1], target_id, args.top_k))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = int(target_id)
            inputs["attention_mask"][0, length] = 1
        rows.append(
            {
                "pair_id": pair.get("pair_id", index),
                "expected": expected,
                "prediction": prediction,
                "exact_match": prediction == expected,
                "target_tokens_all_topk": len(topk_hits) == len(target_ids) and all(topk_hits),
                "target_tokens_topk_fraction": sum(topk_hits) / len(target_ids),
                "target_token_count": len(target_ids),
            }
        )
    return rows


def summarize_pair_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_pairs": 0,
            "autoregressive_exact_match": 0.0,
            "target_tokens_all_topk": 0.0,
            "target_tokens_topk_fraction": 0.0,
        }
    return {
        "n_pairs": len(rows),
        "autoregressive_exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "target_tokens_all_topk": sum(row["target_tokens_all_topk"] for row in rows) / len(rows),
        "target_tokens_topk_fraction": sum(row["target_tokens_topk_fraction"] for row in rows) / len(rows),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def summarize(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row["task"], row["method"]), []).append(row)
    summary = []
    for (task, method), parts in sorted(grouped.items()):
        summary.append(
            {
                "task": task,
                "task_label": label(task),
                "method": method,
                "n": len(parts),
                "autoregressive_exact_match_mean": mean([row["autoregressive_exact_match"] for row in parts]),
                "autoregressive_exact_match_std": std([row["autoregressive_exact_match"] for row in parts]),
                "target_tokens_all_topk_mean": mean([row["target_tokens_all_topk"] for row in parts]),
                "target_tokens_topk_fraction_mean": mean([row["target_tokens_topk_fraction"] for row in parts]),
            }
        )
    return summary


def write_outputs(args, result_rows, pair_rows, seed_selection) -> None:
    save_jsonl(result_rows, args.output_dir / "oracle_audit_results.jsonl")
    save_jsonl(summarize(result_rows), args.output_dir / "oracle_audit_summary.jsonl")
    save_jsonl(pair_rows, args.output_dir / "oracle_audit_pair_details.jsonl")
    save_json(
        {
            "experiment": EXPERIMENT,
            "seed_selection": seed_selection,
            "top_k": args.top_k,
            "top_k_definition": "teacher-forced target answer tokens; every target token must be in top-k for target_tokens_all_topk",
            "config": jsonable(vars(args)),
        },
        args.output_dir / "manifest.json",
    )


def main() -> None:
    args = parse_args()
    args.tasks = [task_key(*parse_task(task)) for task in args.tasks]
    args.fit_task = task_key(*parse_task(args.fit_task))
    if args.fit_task not in args.tasks:
        args.tasks.append(args.fit_task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_selection = selected_matching_seeds(args)
    seeds = common_experiment_seeds(seed_selection)

    model, processor, tokenizer, blocks, hidden_size, _model_name = load_model_bundle(args, args.tasks)
    activation_cache = {}
    sample_cache = {}
    result_rows = []
    pair_rows = []
    oracle = PlusOneMap("oracle_target_state", "oracle", fit_task="none")

    print("Equivariance oracle audit")
    print(f"  top_k={args.top_k}")
    for seed in seeds:
        print(f"\nSeed {seed}")
        sync = load_sync(args, seed)
        spaces = make_matching_spaces(args, args.tasks, seed, hidden_size, sync)
        fit_space = spaces[args.fit_task]
        fit_centroids = shared_centroids(
            args,
            activation_cache,
            sample_cache,
            space=fit_space,
            samples=cached_task_samples(sample_cache, args.fit_task, fit_space.row),
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
        )
        split = value_split(args, fit_centroids)
        for task, space in spaces.items():
            pairs = plus_pairs_for_task(args, sample_cache, space=space, starts=split["test_starts"], step=1)
            if not pairs:
                print(f"  no +1 pairs for {task}[{seed}]")
                continue
            subspaces = subspace_from_basis(args.layer, space.basis, model.device)
            hidden_deltas = hidden_deltas_for_map(
                args,
                activation_cache,
                sample_cache,
                space=space,
                pairs=pairs,
                plus_map=oracle,
                step=1,
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
            )
            if space.modality == "text":
                joint = text_greedy_and_topk_joint_das(
                    model, tokenizer, blocks, subspaces, pairs, args, f"joint DAS {task}[{seed}]"
                )
                delta = text_greedy_and_topk_delta(
                    model, tokenizer, blocks, pairs, hidden_deltas, args, f"delta oracle {task}[{seed}]"
                )
            else:
                data_root = data_root_for(space.row)
                joint = image_greedy_and_topk_joint_das(
                    model, processor, tokenizer, blocks, subspaces, pairs, data_root, args, f"joint DAS {task}[{seed}]"
                )
                delta = image_greedy_and_topk_delta(
                    model, processor, tokenizer, blocks, pairs, hidden_deltas, data_root, args, f"delta oracle {task}[{seed}]"
                )

            joint_summary = summarize_pair_rows(joint)
            delta_summary = summarize_pair_rows(delta)
            exact_gap = delta_summary["autoregressive_exact_match"] - joint_summary["autoregressive_exact_match"]
            topk_gap = delta_summary["target_tokens_all_topk"] - joint_summary["target_tokens_all_topk"]
            for method, summary in [("joint_self_das", joint_summary), ("delta_oracle", delta_summary)]:
                result_rows.append(
                    {
                        "task": task,
                        "task_label": label(task),
                        "seed": seed,
                        "method": method,
                        "value_split": split,
                        "top_k": args.top_k,
                        "delta_minus_joint_exact_match": exact_gap,
                        "delta_minus_joint_topk_all": topk_gap,
                        **summary,
                    }
                )
            by_delta_pair_id = {row["pair_id"]: row for row in delta}
            for row in joint:
                delta_row = by_delta_pair_id[row["pair_id"]]
                pair = pairs[int(row["pair_id"])]
                pair_rows.append(
                    {
                        "task": task,
                        "task_label": label(task),
                        "seed": seed,
                        "pair_id": row["pair_id"],
                        "base_result": int(pair["base"]["result"]),
                        "source_result": int(pair["source"]["result"]),
                        "carry": pair["carry"],
                        "expected": row["expected"],
                        "joint_prediction": row["prediction"],
                        "delta_prediction": delta_row["prediction"],
                        "joint_exact_match": row["exact_match"],
                        "delta_exact_match": delta_row["exact_match"],
                        "joint_target_tokens_all_topk": row["target_tokens_all_topk"],
                        "delta_target_tokens_all_topk": delta_row["target_tokens_all_topk"],
                        "joint_target_tokens_topk_fraction": row["target_tokens_topk_fraction"],
                        "delta_target_tokens_topk_fraction": delta_row["target_tokens_topk_fraction"],
                    }
                )
            print(
                f"  {task}[{seed}]: joint={joint_summary['autoregressive_exact_match']:.4f}, "
                f"delta={delta_summary['autoregressive_exact_match']:.4f}, "
                f"joint_top{args.top_k}={joint_summary['target_tokens_all_topk']:.4f}, "
                f"delta_top{args.top_k}={delta_summary['target_tokens_all_topk']:.4f}"
            )
            write_outputs(args, result_rows, pair_rows, seed_selection)

    write_outputs(args, result_rows, pair_rows, seed_selection)
    print(f"\nSaved oracle audit under: {args.output_dir}")


if __name__ == "__main__":
    main()
