"""
Test whether readout-aligned and latent DAS components functionally interact.

For each task/seed we rotate the learned DAS basis by alignment to the
centered digit-readout span, split it into a readout-aligned causal_component C
and the remaining latent_component L, then run consistent and inconsistent
base/donor interventions:

    full_das:       P_R(h_d - h_b)
    matched_Cd_Ld:  P_C(h_d - h_b) + P_L(h_d - h_b)
    C_only_d:       P_C(h_d - h_b)
    L_only_d:       P_L(h_d - h_b)
    mismatch_Cd_Lc: P_C(h_d - h_b) + P_L(h_c - h_b)
    mismatch_Cc_Ld: P_C(h_c - h_b) + P_L(h_d - h_b)

The donor d is the causal target; c supplies an inconsistent latent target.
Rows are saved per intervention so later analysis can aggregate by task, seed,
condition, or digit-conflict category.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import load_jsonl
from src.experiments.readout_latent_geometry.readout_causal_audit import (
    build_digit_readout_span,
    metadata_for_run,
    order_das_by_reference,
    orthonormal_basis,
    position_for,
    write_csv,
)
from src.interventions.das import (
    format_prompt,
    hidden,
    hook_module,
    replace_hidden,
    resolve_position,
    sequence_scores,
    target_answers,
    tokenize_answers,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    answer_token_positions,
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    resolve_batch_positions,
    sample_prompt,
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


DEFAULT_TASKS = ("text:addition", "text:subtraction", "image:addition", "image:subtraction")
DEFAULT_CONDITIONS = (
    "full_das",
    "matched_Cd_Ld",
    "C_only_d",
    "L_only_d",
    "mismatch_Cd_Lc",
    "mismatch_Cc_Ld",
    "random_latent_mismatch",
)
DEFAULT_MISMATCH_CATEGORIES = (
    "different_first_digit",
    "same_first_diff_second",
    "diff_first_same_second",
    "one_digit_targets",
)
TASK_LABELS = {
    "text:addition": "T+",
    "text:subtraction": "T-",
    "image:addition": "I+",
    "image:subtraction": "I-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/latent_readout_causal_interaction"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--m_readout", type=int, default=5)
    parser.add_argument(
        "--m_mode",
        choices=["fixed", "global_from_audit", "per_task_from_audit", "per_seed_from_audit"],
        default="fixed",
        help="How to choose m_readout. *_from_audit reads cumulative_readout_results.csv from --readout_audit_dir.",
    )
    parser.add_argument("--readout_audit_dir", type=Path, default=Path("results/experiments/closing/readout_causal_audit"))
    parser.add_argument("--m_recovery_threshold", type=float, default=0.90)
    parser.add_argument("--m_selection_metric", default="relative_ar_recovery")
    parser.add_argument("--m_selection_ordering", default="aligned_first")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--target", default="result")
    parser.add_argument("--n_triplets", type=int, default=256)
    parser.add_argument("--n_random_latent", type=int, default=5)
    parser.add_argument("--triplet_seed", type=int, default=1729)
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--mismatch_categories", nargs="+", default=list(DEFAULT_MISMATCH_CATEGORIES))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--activation_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--skip_teacher_forced", action="store_true")
    parser.add_argument("--skip_autoregressive", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--sanity_tolerance", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def parse_task(raw: str) -> tuple[str, str]:
    aliases = {
        "text_add": "text:addition",
        "text_addition": "text:addition",
        "t+": "text:addition",
        "text_sub": "text:subtraction",
        "text_subtraction": "text:subtraction",
        "t-": "text:subtraction",
        "image_add": "image:addition",
        "image_addition": "image:addition",
        "i+": "image:addition",
        "image_sub": "image:subtraction",
        "image_subtraction": "image:subtraction",
        "i-": "image:subtraction",
    }
    key = aliases.get(raw.lower(), raw.lower())
    if key == "all":
        raise ValueError("'all' is handled before parse_task.")
    parts = key.split(":")
    if len(parts) != 2 or parts[0] not in {"text", "image"} or parts[1] not in {"addition", "subtraction"}:
        raise ValueError(f"Unknown task {raw!r}.")
    return parts[0], parts[1]


def normalize_tasks(raw_tasks: list[str]) -> list[dict]:
    if any(item.lower() == "all" for item in raw_tasks):
        raw_tasks = list(DEFAULT_TASKS)
    tasks = []
    seen = set()
    for raw in raw_tasks:
        modality, operation = parse_task(raw)
        key = f"{modality}:{operation}"
        if key in seen:
            continue
        seen.add(key)
        tasks.append(
            {
                "task": f"{modality}_{operation}",
                "task_key": key,
                "task_label": TASK_LABELS[key],
                "modality": modality,
                "operation": operation,
            }
        )
    return tasks


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2)
        handle.write("\n")


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row)) + "\n")


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def maybe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(sum(clean) / len(clean))


def select_m_from_rows(args: argparse.Namespace, rows: list[dict], *, task_name: str | None, seed: int | None) -> tuple[int, dict]:
    filtered = [
        row
        for row in rows
        if row.get("ordering") == args.m_selection_ordering
        and (task_name is None or row.get("task") == task_name)
        and (seed is None or int(row.get("das_seed", -1)) == int(seed))
        and maybe_float(row.get(args.m_selection_metric)) is not None
    ]
    if not filtered:
        scope = "global" if task_name is None else task_name
        if seed is not None:
            scope += f" seed={seed}"
        raise ValueError(
            f"No cumulative readout rows for {scope} with ordering={args.m_selection_ordering!r} "
            f"and metric={args.m_selection_metric!r} in {args.readout_audit_dir}."
        )
    by_m = {}
    for row in filtered:
        m = int(row["m"])
        by_m.setdefault(m, []).append(maybe_float(row.get(args.m_selection_metric)))
    aggregate = [
        {"m": m, "metric_mean": mean(values), "n": len(values)}
        for m, values in sorted(by_m.items())
        if mean(values) is not None
    ]
    eligible = [row for row in aggregate if row["metric_mean"] >= args.m_recovery_threshold]
    selected = min(eligible, key=lambda row: row["m"]) if eligible else max(aggregate, key=lambda row: row["metric_mean"])
    return int(selected["m"]), {
        "m_mode": args.m_mode,
        "m_selected": int(selected["m"]),
        "m_selection_scope_task": task_name,
        "m_selection_scope_seed": seed,
        "m_selection_metric": args.m_selection_metric,
        "m_selection_metric_mean": selected["metric_mean"],
        "m_selection_threshold": args.m_recovery_threshold,
        "m_selection_n_rows": selected["n"],
        "m_selection_reached_threshold": bool(selected in eligible),
    }


def build_m_selector(args: argparse.Namespace, tasks: list[dict]) -> dict[tuple[str | None, int | None], tuple[int, dict]]:
    if args.m_mode == "fixed":
        return {
            (task["task"], seed): (
                int(args.m_readout),
                {
                    "m_mode": "fixed",
                    "m_selected": int(args.m_readout),
                    "m_selection_scope_task": task["task"],
                    "m_selection_scope_seed": seed,
                },
            )
            for task in tasks
            for seed in args.seeds
        }
    path = args.readout_audit_dir / "cumulative_readout_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv_rows(path)
    selector = {}
    global_selection = None
    if args.m_mode == "global_from_audit":
        global_selection = select_m_from_rows(args, rows, task_name=None, seed=None)
    for task in tasks:
        task_selection = None
        if args.m_mode == "per_task_from_audit":
            task_selection = select_m_from_rows(args, rows, task_name=task["task"], seed=None)
        for seed in args.seeds:
            if args.m_mode == "global_from_audit":
                selector[(task["task"], seed)] = global_selection
            elif args.m_mode == "per_task_from_audit":
                selector[(task["task"], seed)] = task_selection
            elif args.m_mode == "per_seed_from_audit":
                selector[(task["task"], seed)] = select_m_from_rows(args, rows, task_name=task["task"], seed=seed)
            else:
                raise ValueError(f"Unknown m_mode: {args.m_mode}")
    return selector


def sample_key(sample: dict):
    if "sample_id" in sample:
        return sample["sample_id"]
    if "id" in sample:
        return sample["id"]
    return json.dumps(sample, sort_keys=True)


def stable_seed(*items) -> int:
    text = "|".join(str(item) for item in items)
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (2**31)


def target_value(sample: dict, target: str) -> int:
    return int(sample[target] if target in sample else sample["result"])


def answer_text(sample: dict, target: str) -> str:
    return str(target_value(sample, target))


def digits(value: int) -> str:
    return str(abs(int(value)))


def mismatch_category(causal_value: int, latent_value: int) -> str | None:
    d = digits(causal_value)
    c = digits(latent_value)
    if 0 <= causal_value <= 9 and 0 <= latent_value <= 9:
        return "one_digit_targets"
    if not d or not c:
        return None
    if d[0] != c[0]:
        if len(d) >= 2 and len(c) >= 2 and d[1] == c[1]:
            return "diff_first_same_second"
        return "different_first_digit"
    if len(d) >= 2 and len(c) >= 2 and d[1] != c[1]:
        return "same_first_diff_second"
    return None


def category_matches(causal_value: int, latent_value: int, category: str) -> bool:
    return mismatch_category(causal_value, latent_value) == category


def load_pairs_for_task(directory: Path, data_path: Path) -> list[dict]:
    lookup = {}
    for index, sample in enumerate(load_jsonl(data_path)):
        key = sample.get("sample_id", index)
        lookup[key] = sample
        lookup[str(key)] = sample
    pairs = []
    for index, row in enumerate(load_jsonl(directory / "heldout_pairs.jsonl")):
        pairs.append(
            {
                "pair_id": row.get("pair_id", index),
                "base": lookup[row["base_sample_id"]],
                "source": lookup[row["source_sample_id"]],
            }
        )
    return pairs


def pick_latent_candidate(
    samples: list[dict],
    *,
    base: dict,
    donor: dict,
    category: str,
    rng: random.Random,
    target: str,
) -> dict | None:
    base_id = sample_key(base)
    donor_id = sample_key(donor)
    base_value = target_value(base, target)
    donor_value = target_value(donor, target)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    preferred = []
    fallback = []
    for sample in shuffled:
        sid = sample_key(sample)
        if sid in {base_id, donor_id}:
            continue
        value = target_value(sample, target)
        if value == donor_value:
            continue
        if not category_matches(donor_value, value, category):
            continue
        if value != base_value:
            preferred.append(sample)
        else:
            fallback.append(sample)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def build_triplets(
    args: argparse.Namespace,
    *,
    task: dict,
    seed: int,
    pairs: list[dict],
    samples: list[dict],
) -> list[dict]:
    rng = random.Random(stable_seed(args.triplet_seed, task["task_key"], seed))
    pair_order = list(pairs)
    rng.shuffle(pair_order)
    categories = [item for item in args.mismatch_categories if item in DEFAULT_MISMATCH_CATEGORIES]
    if not categories:
        raise ValueError("No valid mismatch categories requested.")
    triplets = []
    category_cursor = 0
    passes_without_add = 0
    while len(triplets) < args.n_triplets and passes_without_add < len(pair_order) * len(categories) + 1:
        pair = pair_order[passes_without_add % len(pair_order)]
        passes_without_add += 1
        base, donor = pair["base"], pair["source"]
        if target_value(base, args.target) == target_value(donor, args.target):
            continue
        category = categories[category_cursor % len(categories)]
        category_cursor += 1
        latent = pick_latent_candidate(
            samples,
            base=base,
            donor=donor,
            category=category,
            rng=rng,
            target=args.target,
        )
        if latent is None:
            continue
        latent_value = target_value(latent, args.target)
        donor_value = target_value(donor, args.target)
        actual_category = mismatch_category(donor_value, latent_value)
        triplet_id = f"{task['task_key']}_seed{seed}_{len(triplets):04d}"
        triplets.append(
            {
                "triplet_id": triplet_id,
                "pair_id": pair.get("pair_id"),
                "task": task["task"],
                "task_key": task["task_key"],
                "task_label": task["task_label"],
                "modality": task["modality"],
                "operation": task["operation"],
                "das_seed": seed,
                "base_sample_id": sample_key(base),
                "causal_donor_sample_id": sample_key(donor),
                "latent_donor_sample_id": sample_key(latent),
                "base_value": target_value(base, args.target),
                "causal_donor_value": donor_value,
                "latent_donor_value": latent_value,
                "mismatch_category": actual_category,
                "preferred_latent_not_base": latent_value != target_value(base, args.target),
                "base": base,
                "causal_donor": donor,
                "latent_donor": latent,
            }
        )
    if not triplets:
        raise RuntimeError(f"No valid triplets for {task['task_key']} seed={seed}.")
    if len(triplets) < args.n_triplets:
        print(f"WARNING: requested {args.n_triplets} triplets for {task['task_key']} seed={seed}, got {len(triplets)}.")
    return triplets


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
    for start in tqdm(range(0, len(samples), batch_size), desc=f"collect {modality} activations"):
        batch = samples[start : start + batch_size]
        if modality == "text":
            prompts = [format_prompt(tokenizer, sample, use_chat) for sample in batch]
            positions = [resolve_position(tokenizer, text, position) for text in prompts]
            encoding = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
        else:
            if processor is None or data_root is None:
                raise ValueError("Image activation collection requires processor and data_root.")
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


def project_delta(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[1] == 0:
        return torch.zeros_like(delta)
    return (delta @ basis) @ basis.T


def build_ordered_split(
    args: argparse.Namespace,
    *,
    raw_basis: torch.Tensor,
    digit_basis: torch.Tensor,
    hidden_size: int,
    m_readout: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    if raw_basis.shape[0] != hidden_size and raw_basis.shape[1] == hidden_size:
        raw_basis = raw_basis.T
    basis = orthonormal_basis(raw_basis, "DAS basis", rank=args.k)
    ordered, singular_values, projector_rotation_error = order_das_by_reference(basis, digit_basis)
    m_readout = min(m_readout, ordered.shape[1])
    causal_component = ordered[:, :m_readout]
    latent_component = ordered[:, m_readout:]
    projector = basis @ basis.T
    split_projector = causal_component @ causal_component.T + latent_component @ latent_component.T
    diagnostics = {
        "k": int(basis.shape[1]),
        "m_readout": int(m_readout),
        "latent_dim": int(latent_component.shape[1]),
        "digit_span_rank": int(digit_basis.shape[1]),
        "ordered_projector_error": float((ordered @ ordered.T - projector).norm()),
        "projector_rotation_error": projector_rotation_error,
        "split_projector_error": float((split_projector - projector).norm()),
        "component_orthogonality": float((causal_component.T @ latent_component).norm()),
        "top_readout_singular_values": [float(value) for value in singular_values[: min(10, singular_values.numel())]],
    }
    return basis, ordered, causal_component, latent_component, diagnostics


def random_latent_samples(
    args: argparse.Namespace,
    *,
    samples: list[dict],
    triplet: dict,
    task_key: str,
    seed: int,
) -> list[dict]:
    rng = random.Random(stable_seed(args.triplet_seed, "random_latent", task_key, seed, triplet["triplet_id"]))
    base_id = triplet["base_sample_id"]
    donor_id = triplet["causal_donor_sample_id"]
    latent_id = triplet["latent_donor_sample_id"]
    donor_value = triplet["causal_donor_value"]
    base_value = triplet["base_value"]
    candidates = [
        sample
        for sample in samples
        if sample_key(sample) not in {base_id, donor_id, latent_id}
        and target_value(sample, args.target) != donor_value
        and target_value(sample, args.target) != base_value
    ]
    if len(candidates) < args.n_random_latent:
        candidates = [
            sample
            for sample in samples
            if sample_key(sample) not in {base_id, donor_id, latent_id}
            and target_value(sample, args.target) != donor_value
        ]
    rng.shuffle(candidates)
    return candidates[: args.n_random_latent]


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


def first_digit_metric(parsed: str | None, expected: int) -> bool:
    if parsed is None:
        return False
    return digits(int(parsed))[0:1] == digits(expected)[0:1]


def second_digit_metric(parsed: str | None, expected: int) -> bool | None:
    expected_digits = digits(expected)
    if len(expected_digits) < 2:
        return None
    if parsed is None:
        return False
    parsed_digits = digits(int(parsed))
    if len(parsed_digits) < 2:
        return False
    return parsed_digits[1] == expected_digits[1]


def parse_int_prefix(text: str) -> str | None:
    match = re.match(r"\s*(-?\d+)", text)
    return match.group(1) if match else None


def classify_generation(parsed: str | None, row: dict) -> str:
    if parsed is None:
        return "other"
    try:
        value = int(parsed)
    except ValueError:
        return "other"
    if value == row["base_value"]:
        return "equals_base"
    if value == row["causal_donor_value"]:
        return "equals_causal_donor"
    if value == row["latent_donor_value"]:
        return "equals_latent_donor"
    return "other"


@torch.no_grad()
def teacher_forced_scores_text(args, model, tokenizer, blocks, rows: list[dict]) -> list[dict]:
    outputs_rows = []
    use_chat = args.use_chat_template or uses_chat_template(args.model)
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [format_prompt(tokenizer, row["base"], use_chat) for row in batch]
        positions = [resolve_position(tokenizer, prompt, args.text_position) for prompt in prompts]
        for label in ("d", "c", "base"):
            answers = []
            for row in batch:
                if label == "d":
                    answers.append(str(row["causal_donor_value"]))
                elif label == "c":
                    answers.append(str(row["latent_donor_value"]))
                else:
                    answers.append(str(row["base_value"]))
            encoding, full_positions, _ = tokenize_answers(
                tokenizer,
                prompts,
                answers,
                [(0, len(answer)) for answer in answers],
                model.device,
            )
            deltas = torch.stack([row["hidden_delta"] for row in batch])
            result = patched_forward_with_deltas(model, encoding, blocks, args.layer, args.hook, positions, deltas)
            scores, exact = sequence_scores(result.logits, encoding["input_ids"], full_positions)
            for index, row in enumerate(batch):
                first_pos = full_positions[index][0]
                first_id = encoding["input_ids"][index, first_pos]
                logits = result.logits[index, first_pos - 1].float()
                probs = logits.softmax(dim=-1)
                if label == "d":
                    row["tf_logprob_d"] = float(scores[index])
                    row["tf_iia_to_d"] = bool(exact[index])
                    row["logit_d_first"] = float(logits[first_id])
                    row["prob_d_first"] = float(probs[first_id])
                elif label == "c":
                    row["tf_logprob_c"] = float(scores[index])
                    row["tf_iia_to_c"] = bool(exact[index])
                    row["logit_c_first"] = float(logits[first_id])
                    row["prob_c_first"] = float(probs[first_id])
                else:
                    row["tf_logprob_base"] = float(scores[index])
                    row["tf_iia_to_base"] = bool(exact[index])
                    row["logit_base_first"] = float(logits[first_id])
                    row["prob_base_first"] = float(probs[first_id])
        outputs_rows.extend(batch)
    return outputs_rows


@torch.no_grad()
def teacher_forced_scores_image(args, model, processor, tokenizer, blocks, rows: list[dict], data_root: Path) -> list[dict]:
    outputs_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [sample_prompt(processor, row["base"], args.prompt, args.enable_thinking) for row in batch]
        images = [load_rgb_image(image_path_for(row["base"], data_root)) for row in batch]
        positions = resolve_batch_positions(processor, tokenizer, model, prompts, images, args.image_position)
        for label in ("d", "c", "base"):
            answers = []
            for row in batch:
                if label == "d":
                    answers.append(str(row["causal_donor_value"]))
                elif label == "c":
                    answers.append(str(row["latent_donor_value"]))
                else:
                    answers.append(str(row["base_value"]))
            encoding, full_positions, _ = answer_token_positions(
                processor,
                prompts,
                answers,
                [(0, len(answer)) for answer in answers],
                images,
                model.device,
            )
            deltas = torch.stack([row["hidden_delta"] for row in batch])
            result = patched_forward_with_deltas(model, encoding, blocks, args.layer, args.hook, positions, deltas)
            scores, exact = sequence_scores(result.logits, encoding["input_ids"], full_positions)
            for index, row in enumerate(batch):
                first_pos = full_positions[index][0]
                first_id = encoding["input_ids"][index, first_pos]
                logits = result.logits[index, first_pos - 1].float()
                probs = logits.softmax(dim=-1)
                if label == "d":
                    row["tf_logprob_d"] = float(scores[index])
                    row["tf_iia_to_d"] = bool(exact[index])
                    row["logit_d_first"] = float(logits[first_id])
                    row["prob_d_first"] = float(probs[first_id])
                elif label == "c":
                    row["tf_logprob_c"] = float(scores[index])
                    row["tf_iia_to_c"] = bool(exact[index])
                    row["logit_c_first"] = float(logits[first_id])
                    row["prob_c_first"] = float(probs[first_id])
                else:
                    row["tf_logprob_base"] = float(scores[index])
                    row["tf_iia_to_base"] = bool(exact[index])
                    row["logit_base_first"] = float(logits[first_id])
                    row["prob_base_first"] = float(probs[first_id])
        outputs_rows.extend(batch)
    return outputs_rows


@torch.no_grad()
def autoregressive_text(args, model, tokenizer, blocks, rows: list[dict]) -> list[dict]:
    use_chat = args.use_chat_template or uses_chat_template(args.model)
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for row in tqdm(rows, desc="text AR latent/readout interactions"):
            prompt = format_prompt(tokenizer, row["base"], use_chat)
            position = resolve_position(tokenizer, prompt, args.text_position)
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            generated = []
            for _ in range(args.max_new_tokens):
                result = patched_forward_with_deltas(
                    model,
                    {"input_ids": input_ids[None]},
                    blocks,
                    args.layer,
                    args.hook,
                    [position],
                    row["hidden_delta"][None],
                )
                next_id = result.logits[0, input_ids.shape[0] - 1].argmax().reshape(1)
                generated.append(int(next_id.item()))
                input_ids = torch.cat([input_ids, next_id])
                if next_id.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed = parse_int_prefix(text)
            row["generated_answer"] = text
            row["parsed_answer"] = parsed
            row["generation_class"] = classify_generation(parsed, row)
            row["ar_iia_to_d"] = parsed == str(row["causal_donor_value"])
            row["ar_iia_to_c"] = parsed == str(row["latent_donor_value"])
            row["ar_iia_to_base"] = parsed == str(row["base_value"])
    finally:
        tokenizer.padding_side = old_padding_side
    return rows


@torch.no_grad()
def autoregressive_image(args, model, processor, tokenizer, blocks, rows: list[dict], data_root: Path) -> list[dict]:
    for row in tqdm(rows, desc="image AR latent/readout interactions"):
        prompt = sample_prompt(processor, row["base"], args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(row["base"], data_root))
        position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.image_position)[0]
        inputs = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            length = int(inputs["attention_mask"][0].sum())
            result = patched_forward_with_deltas(
                model,
                inputs,
                blocks,
                args.layer,
                args.hook,
                [position],
                row["hidden_delta"][None],
            )
            next_id = result.logits[0, length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((1, 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, length] = next_id.item()
            inputs["attention_mask"][0, length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = parse_int_prefix(text)
        row["generated_answer"] = text
        row["parsed_answer"] = parsed
        row["generation_class"] = classify_generation(parsed, row)
        row["ar_iia_to_d"] = parsed == str(row["causal_donor_value"])
        row["ar_iia_to_c"] = parsed == str(row["latent_donor_value"])
        row["ar_iia_to_base"] = parsed == str(row["base_value"])
    return rows


def strip_heavy(row: dict) -> dict:
    excluded = {"base", "causal_donor", "latent_donor", "random_latent", "hidden_delta"}
    return {key: value for key, value in row.items() if key not in excluded}


def add_vector_diagnostics(row: dict, full_delta: torch.Tensor, c_delta: torch.Tensor, l_delta: torch.Tensor, combined: torch.Tensor) -> dict:
    full_norm = float(full_delta.norm())
    c_norm = float(c_delta.norm())
    l_norm = float(l_delta.norm())
    combined_norm = float(combined.norm())
    dot = float((c_delta * l_delta).sum())
    denom = c_norm * l_norm
    return {
        **strip_heavy(row),
        "norm_full_delta": full_norm,
        "norm_C": c_norm,
        "norm_L": l_norm,
        "norm_combined": combined_norm,
        "combined_over_full_norm": None if full_norm < 1e-12 else combined_norm / full_norm,
        "C_L_dot": dot,
        "C_L_cosine": None if denom < 1e-12 else dot / denom,
    }


def build_intervention_rows(
    args: argparse.Namespace,
    *,
    task: dict,
    seed: int,
    triplets: list[dict],
    samples: list[dict],
    activation_by_id: dict,
    basis: torch.Tensor,
    causal_component: torch.Tensor,
    latent_component: torch.Tensor,
) -> tuple[list[dict], list[dict], list[dict]]:
    rows = []
    diagnostics = []
    generated_random_rows = []
    for triplet in triplets:
        h_b = activation_by_id[triplet["base_sample_id"]]
        h_d = activation_by_id[triplet["causal_donor_sample_id"]]
        h_c = activation_by_id[triplet["latent_donor_sample_id"]]
        delta_bd = h_d - h_b
        delta_bc = h_c - h_b
        full_bd = project_delta(delta_bd, basis)
        c_bd = project_delta(delta_bd, causal_component)
        l_bd = project_delta(delta_bd, latent_component)
        c_bc = project_delta(delta_bc, causal_component)
        l_bc = project_delta(delta_bc, latent_component)
        if float((c_bd + l_bd - full_bd).norm()) > args.sanity_tolerance:
            raise RuntimeError(f"Component split failed for {triplet['triplet_id']}.")
        condition_deltas = {
            "full_das": (full_bd, c_bd, l_bd),
            "matched_Cd_Ld": (c_bd + l_bd, c_bd, l_bd),
            "C_only_d": (c_bd, c_bd, torch.zeros_like(l_bd)),
            "L_only_d": (l_bd, torch.zeros_like(c_bd), l_bd),
            "mismatch_Cd_Lc": (c_bd + l_bc, c_bd, l_bc),
            "mismatch_Cc_Ld": (c_bc + l_bd, c_bc, l_bd),
        }
        for condition, (combined, c_part, l_part) in condition_deltas.items():
            if condition not in args.conditions:
                continue
            row = {
                **triplet,
                "condition": condition,
                "random_latent_index": None,
                "hidden_delta": combined,
            }
            rows.append(row)
            diagnostics.append(add_vector_diagnostics(row, full_bd, c_part, l_part, combined))
        if "random_latent_mismatch" in args.conditions and args.n_random_latent > 0:
            for random_index, random_sample in enumerate(
                random_latent_samples(args, samples=samples, triplet=triplet, task_key=task["task_key"], seed=seed)
            ):
                random_id = sample_key(random_sample)
                random_value = target_value(random_sample, args.target)
                h_r = activation_by_id[random_id]
                l_br = project_delta(h_r - h_b, latent_component)
                combined = c_bd + l_br
                row = {
                    **triplet,
                    "condition": "random_latent_mismatch",
                    "random_latent_index": random_index,
                    "primary_latent_donor_sample_id": triplet["latent_donor_sample_id"],
                    "primary_latent_donor_value": triplet["latent_donor_value"],
                    "latent_donor_sample_id": random_id,
                    "latent_donor_value": random_value,
                    "latent_donor": random_sample,
                    "random_latent_sample_id": random_id,
                    "random_latent_value": random_value,
                    "random_latent": random_sample,
                    "hidden_delta": combined,
                }
                rows.append(row)
                diagnostics.append(add_vector_diagnostics(row, full_bd, c_bd, l_br, combined))
                generated_random_rows.append(row)
    return rows, diagnostics, generated_random_rows


def digit_rows_from_metrics(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        parsed = row.get("parsed_answer")
        parsed_int = int(parsed) if parsed is not None and re.fullmatch(r"-?\d+", str(parsed)) else None
        parsed_text = None if parsed_int is None else str(parsed_int)
        output.append(
            {
                "task": row["task"],
                "task_key": row["task_key"],
                "task_label": row["task_label"],
                "modality": row["modality"],
                "operation": row["operation"],
                "das_seed": row["das_seed"],
                "triplet_id": row["triplet_id"],
                "condition": row["condition"],
                "random_latent_index": row.get("random_latent_index"),
                "mismatch_category": row["mismatch_category"],
                "base_value": row["base_value"],
                "causal_donor_value": row["causal_donor_value"],
                "latent_donor_value": row["latent_donor_value"],
                "parsed_answer": parsed,
                "generated_first_digit": None if parsed_text is None else digits(int(parsed_text))[:1],
                "generated_second_digit": None if parsed_text is None or len(digits(int(parsed_text))) < 2 else digits(int(parsed_text))[1],
                "first_digit_ar_to_d": first_digit_metric(parsed, row["causal_donor_value"]) if parsed is not None else False,
                "first_digit_ar_to_c": first_digit_metric(parsed, row["latent_donor_value"]) if parsed is not None else False,
                "first_digit_ar_to_base": first_digit_metric(parsed, row["base_value"]) if parsed is not None else False,
                "second_digit_ar_to_d": second_digit_metric(parsed, row["causal_donor_value"]),
                "second_digit_ar_to_c": second_digit_metric(parsed, row["latent_donor_value"]),
                "second_digit_ar_to_base": second_digit_metric(parsed, row["base_value"]),
                "prob_d_first": row.get("prob_d_first"),
                "prob_c_first": row.get("prob_c_first"),
                "prob_base_first": row.get("prob_base_first"),
            }
        )
    return output


def summarize_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped = {}
    for row in rows:
        key = (row["task_key"], row["task_label"], row["das_seed"], row["mismatch_category"], row["condition"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for key, parts in sorted(grouped.items()):
        task_key, task_label, seed, category, condition = key
        summary_rows.append(
            {
                "task_key": task_key,
                "task_label": task_label,
                "das_seed": seed,
                "mismatch_category": category,
                "condition": condition,
                "n": len(parts),
                "tf_iia_to_d": mean([row.get("tf_iia_to_d") for row in parts]),
                "tf_iia_to_c": mean([row.get("tf_iia_to_c") for row in parts]),
                "tf_iia_to_base": mean([row.get("tf_iia_to_base") for row in parts]),
                "ar_iia_to_d": mean([row.get("ar_iia_to_d") for row in parts]),
                "ar_iia_to_c": mean([row.get("ar_iia_to_c") for row in parts]),
                "ar_iia_to_base": mean([row.get("ar_iia_to_base") for row in parts]),
                "prob_d_first": mean([row.get("prob_d_first") for row in parts]),
                "prob_c_first": mean([row.get("prob_c_first") for row in parts]),
                "prob_base_first": mean([row.get("prob_base_first") for row in parts]),
            }
        )
    lookup = {
        (row["task_key"], row["das_seed"], row["mismatch_category"], row["condition"]): row
        for row in summary_rows
    }
    contrast_rows = []
    for task_key, task_label, seed, category in sorted(
        {(row["task_key"], row["task_label"], row["das_seed"], row["mismatch_category"]) for row in summary_rows}
    ):
        matched = lookup.get((task_key, seed, category, "matched_Cd_Ld"))
        mismatch = lookup.get((task_key, seed, category, "mismatch_Cd_Lc"))
        c_only = lookup.get((task_key, seed, category, "C_only_d"))
        l_only = lookup.get((task_key, seed, category, "L_only_d"))
        if not any([matched, mismatch, c_only, l_only]):
            continue
        contrast_rows.append(
            {
                "task_key": task_key,
                "task_label": task_label,
                "das_seed": seed,
                "mismatch_category": category,
                "condition": "contrasts",
                "matched_minus_mismatch_ar_to_d": None
                if not matched or not mismatch or matched.get("ar_iia_to_d") is None or mismatch.get("ar_iia_to_d") is None
                else matched["ar_iia_to_d"] - mismatch["ar_iia_to_d"],
                "matched_minus_mismatch_tf_to_d": None
                if not matched or not mismatch or matched.get("tf_iia_to_d") is None or mismatch.get("tf_iia_to_d") is None
                else matched["tf_iia_to_d"] - mismatch["tf_iia_to_d"],
                "latent_pull_ar_to_c_minus_C_only": None
                if not mismatch or not c_only or mismatch.get("ar_iia_to_c") is None or c_only.get("ar_iia_to_c") is None
                else mismatch["ar_iia_to_c"] - c_only["ar_iia_to_c"],
                "C_only_recovery_ar_to_d": None if not c_only else c_only.get("ar_iia_to_d"),
                "L_only_recovery_ar_to_d": None if not l_only else l_only.get("ar_iia_to_d"),
                "component_dominance_ar": None
                if not c_only or not l_only or c_only.get("ar_iia_to_d") is None or l_only.get("ar_iia_to_d") is None
                else c_only["ar_iia_to_d"] - l_only["ar_iia_to_d"],
            }
        )
    compact = {}
    for row in summary_rows:
        compact.setdefault(row["task_key"], {}).setdefault(str(row["das_seed"]), {})[
            f"{row['mismatch_category']}::{row['condition']}"
        ] = row
    return summary_rows + contrast_rows, compact


def plot_results(output_dir: Path, summary_rows: list[dict]) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "ar_iia_to_d"
    conditions = ["full_das", "matched_Cd_Ld", "C_only_d", "L_only_d", "mismatch_Cd_Lc", "mismatch_Cc_Ld"]
    tasks = list(TASK_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    axes = axes.ravel()
    for ax, task_key in zip(axes, tasks):
        ax.set_title(TASK_LABELS[task_key])
        rows = [row for row in summary_rows if row.get("task_key") == task_key and row.get("condition") in conditions]
        values = []
        labels = []
        for condition in conditions:
            condition_values = [row.get(metric) for row in rows if row["condition"] == condition and row.get(metric) is not None]
            values.append(mean(condition_values) or 0.0)
            labels.append(condition.replace("_", "\n"))
        ax.bar(range(len(labels)), values, color=["#4c78a8", "#72b7b2", "#54a24b", "#eeca3b", "#f58518", "#b279a2"])
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[::2]:
        ax.set_ylabel("AR IIA to causal donor")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "matched_vs_mismatched_causality.png", dpi=220)
    fig.savefig(output_dir / "matched_vs_mismatched_causality.pdf")
    plt.close(fig)


def output_exists(args: argparse.Namespace) -> bool:
    return (args.output_dir / "summary.json").exists() and not args.force


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.n_triplets = min(args.n_triplets, 16)
        args.n_random_latent = min(args.n_random_latent, 1)
    tasks = normalize_tasks(args.tasks)
    if args.artifact_check_only:
        missing = []
        for task in tasks:
            for seed in args.seeds:
                try:
                    metadata_for_run(args, task, seed)
                except FileNotFoundError as error:
                    missing.append(str(error))
        if missing:
            raise FileNotFoundError("\n".join(missing[:20]))
        print("ARTIFACT_CHECK_OK latent_readout_causal_interaction")
        return
    if output_exists(args):
        raise FileExistsError(f"{args.output_dir} already has summary.json; pass --force to overwrite.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m_selector = build_m_selector(args, tasks)

    model_path, resolved_model = resolve_model_for_loading(args.model)
    needs_image = any(task["modality"] == "image" for task in tasks)
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
    hidden_size = get_hidden_size(model)
    digit_basis, digit_rows = build_digit_readout_span(model, tokenizer)

    triplet_rows = []
    intervention_rows = []
    vector_rows = []
    generated_rows = []
    sanity_rows = []

    for task in tasks:
        for seed in args.seeds:
            directory, result_row, _payload, raw_basis = metadata_for_run(args, task, seed)
            data_path = Path(result_row["data_path"])
            data_root = data_path.parent
            pairs = load_pairs_for_task(directory, data_path)
            samples = load_jsonl(data_path)
            basis, ordered, causal_component, latent_component, split_diag = build_ordered_split(
                args,
                raw_basis=raw_basis,
                digit_basis=digit_basis,
                hidden_size=hidden_size,
                m_readout=m_selector[(task["task"], seed)][0],
            )
            m_selection_diag = m_selector[(task["task"], seed)][1]
            if split_diag["ordered_projector_error"] > args.sanity_tolerance:
                raise RuntimeError(f"Ordered DAS projector changed too much: {split_diag}")
            if split_diag["component_orthogonality"] > args.sanity_tolerance:
                raise RuntimeError(f"C/L components are not orthogonal enough: {split_diag}")
            print(
                f"\n{task['task_key']} seed={seed}: triplets={args.n_triplets} "
                f"k={args.k} m_readout={split_diag['m_readout']} "
                f"m_mode={args.m_mode} "
                f"split_projector_error={split_diag['split_projector_error']:.3e}"
            )
            triplets = build_triplets(args, task=task, seed=seed, pairs=pairs, samples=samples)
            random_samples = []
            for triplet in triplets:
                random_samples.extend(random_latent_samples(args, samples=samples, triplet=triplet, task_key=task["task_key"], seed=seed))
            unique_samples = {}
            for triplet in triplets:
                for sample in (triplet["base"], triplet["causal_donor"], triplet["latent_donor"]):
                    unique_samples[sample_key(sample)] = sample
            for sample in random_samples:
                unique_samples[sample_key(sample)] = sample
            ids = list(unique_samples.keys())
            activations = collect_activations(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
                samples=[unique_samples[item] for item in ids],
                modality=task["modality"],
                layer=args.layer,
                hook_name=args.hook,
                position=position_for(args, task["modality"]),
                data_root=data_root if task["modality"] == "image" else None,
                prompt=args.prompt,
                enable_thinking=args.enable_thinking,
                use_chat=args.use_chat_template or uses_chat_template(args.model),
                batch_size=args.activation_batch_size,
            )
            activation_by_id = {item: activations[index] for index, item in enumerate(ids)}
            rows, diagnostics, _random_rows = build_intervention_rows(
                args,
                task=task,
                seed=seed,
                triplets=triplets,
                samples=samples,
                activation_by_id=activation_by_id,
                basis=basis,
                causal_component=causal_component,
                latent_component=latent_component,
            )
            max_matched_delta_error = 0.0
            by_triplet_condition = {(row["triplet_id"], row["condition"], row.get("random_latent_index")): row for row in rows}
            for triplet in triplets:
                full = by_triplet_condition.get((triplet["triplet_id"], "full_das", None))
                matched = by_triplet_condition.get((triplet["triplet_id"], "matched_Cd_Ld", None))
                if full and matched:
                    max_matched_delta_error = max(max_matched_delta_error, float((full["hidden_delta"] - matched["hidden_delta"]).norm()))
            if max_matched_delta_error > args.sanity_tolerance:
                raise RuntimeError(f"matched_Cd_Ld does not reproduce full_das deltas: {max_matched_delta_error}")
            sanity_rows.append(
                {
                    "task_key": task["task_key"],
                    "task_label": task["task_label"],
                    "das_seed": seed,
                    **split_diag,
                    **m_selection_diag,
                    "n_triplets": len(triplets),
                    "n_interventions": len(rows),
                    "max_matched_delta_error": max_matched_delta_error,
                }
            )
            if not args.skip_teacher_forced:
                if task["modality"] == "text":
                    rows = teacher_forced_scores_text(args, model, tokenizer, blocks, rows)
                else:
                    rows = teacher_forced_scores_image(args, model, processor, tokenizer, blocks, rows, data_root)
            if not args.skip_autoregressive:
                if task["modality"] == "text":
                    rows = autoregressive_text(args, model, tokenizer, blocks, rows)
                else:
                    rows = autoregressive_image(args, model, processor, tokenizer, blocks, rows, data_root)
            triplet_rows.extend([strip_heavy(row) for row in triplets])
            intervention_rows.extend([strip_heavy(row) for row in rows])
            vector_rows.extend(diagnostics)
            generated_rows.extend([strip_heavy(row) for row in rows if "generated_answer" in row])

            write_csv(args.output_dir / "triplets.csv", triplet_rows)
            write_csv(args.output_dir / "intervention_metrics.csv", intervention_rows)
            write_csv(args.output_dir / "vector_diagnostics.csv", vector_rows)
            write_csv(args.output_dir / "digit_level_metrics.csv", digit_rows_from_metrics(intervention_rows))
            save_jsonl(args.output_dir / "generated_outputs.jsonl", generated_rows)
            save_json(args.output_dir / "sanity_checks.json", {"rows": sanity_rows})

    digit_metric_rows = digit_rows_from_metrics(intervention_rows)
    summary_rows, compact_summary = summarize_rows(intervention_rows)
    write_csv(args.output_dir / "triplets.csv", triplet_rows)
    write_csv(args.output_dir / "intervention_metrics.csv", intervention_rows)
    write_csv(args.output_dir / "digit_level_metrics.csv", digit_metric_rows)
    write_csv(args.output_dir / "vector_diagnostics.csv", vector_rows)
    write_csv(args.output_dir / "mismatch_summary.csv", summary_rows)
    save_jsonl(args.output_dir / "generated_outputs.jsonl", generated_rows)
    save_json(
        args.output_dir / "summary.json",
        {
            "model": args.model,
            "resolved_model": resolved_model,
            "audit_root": args.audit_root,
            "output_dir": args.output_dir,
            "tasks": [task["task_key"] for task in tasks],
            "seeds": args.seeds,
            "conditions": args.conditions,
            "mismatch_categories": args.mismatch_categories,
            "m_mode": args.m_mode,
            "readout_audit_dir": args.readout_audit_dir,
            "m_recovery_threshold": args.m_recovery_threshold,
            "m_selection_metric": args.m_selection_metric,
            "m_selection_ordering": args.m_selection_ordering,
            "n_triplets_requested": args.n_triplets,
            "n_intervention_rows": len(intervention_rows),
            "digit_readout_rows": digit_rows,
            "sanity_checks": sanity_rows,
            "summary": compact_summary,
        },
    )
    if not args.skip_plots:
        plot_results(args.output_dir, summary_rows)
    print("\nlatent_readout_causal_interaction complete")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
