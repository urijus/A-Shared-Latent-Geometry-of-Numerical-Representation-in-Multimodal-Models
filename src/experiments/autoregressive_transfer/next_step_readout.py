"""Test latent-to-readout conversion after the first autoregressive digit.

This experiment reuses the same-first/different-second triplets from
``latent_readout_causal_interaction`` and asks whether replacing the latent DAS
component at the pre-generation position changes the next-token state, after a
teacher-forced shared first digit, in the natural second-digit readout direction.

The critical implementation detail is that patched prompt ``past_key_values``
are reused for the first-digit step. The prompt is never rerun with the first
digit appended for the primary measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import load_jsonl
from src.experiments.autoregressive_transfer.latent_identity_swap import (
    DEFAULT_TASKS,
    TASK_LABELS,
    build_m_selector,
    build_ordered_split,
    build_triplets,
    digits,
    load_pairs_for_task,
    mismatch_category,
    normalize_tasks,
    project_delta,
    sample_key,
    stable_seed,
    target_value,
)
from src.experiments.readout_latent_geometry.readout_causal_audit import (
    build_digit_readout_span,
    metadata_for_run,
    position_for,
    write_csv,
)
from src.interventions.das import (
    format_prompt,
    hidden,
    hook_module,
    replace_hidden,
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
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


CONDITIONS = ("matched_Cd_Ld", "mismatch_Cd_Lc", "C_only_d", "full_original_das")
PRIMARY_MISMATCH_CATEGORY = "same_first_diff_second"


@dataclass
class PromptForward:
    logits_t0: torch.Tensor
    past_key_values: object | None
    h_t0: torch.Tensor | None
    prompt_length: int
    prompt_position: int
    prompt: str


@dataclass
class TwoStepRun:
    logits_t0: torch.Tensor
    logits_t1: torch.Tensor
    h_t0: torch.Tensor | None
    h_t1_by_layer: dict[int, torch.Tensor]
    t0_argmax_token_id: int
    t0_argmax_text: str
    first_digit_token_id: int
    first_digit_text: str
    prompt_length: int
    prompt_position: int
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--latent_interaction_dir",
        type=Path,
        default=Path("results/experiments/closing/latent_readout_causal_interaction"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/latent_to_next_digit_readout"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--capture_layers", type=int, nargs="+", default=[40, 41, 42, 43, 44, 45, 46, 47])
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--m_readout", type=int, default=9)
    parser.add_argument(
        "--m_mode",
        choices=["fixed", "global_from_audit", "per_task_from_audit", "per_seed_from_audit"],
        default="fixed",
    )
    parser.add_argument("--readout_audit_dir", type=Path, default=Path("results/experiments/closing/readout_causal_audit"))
    parser.add_argument("--m_recovery_threshold", type=float, default=0.90)
    parser.add_argument("--m_selection_metric", default="relative_ar_recovery")
    parser.add_argument("--m_selection_ordering", default="aligned_first")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--target", default="result")
    parser.add_argument("--max_triplets", type=int, default=128)
    parser.add_argument("--n_random_same_tens", type=int, default=3)
    parser.add_argument("--triplet_seed", type=int, default=1729)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--skip_random_controls", action="store_true")
    parser.add_argument("--skip_full_original_das", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sanity_tolerance", type=float, default=1e-4)
    parser.add_argument("--recovery_denominator_floor", type=float, default=1e-4)
    return parser.parse_args()


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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")


def mean(values) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not clean else float(sum(clean) / len(clean))


def median(values) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float | None:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = float(a.norm() * b.norm())
    if denom < eps:
        return None
    return float((a * b).sum() / denom)


def euclidean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float().flatten() - b.float().flatten()).norm())


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_lookup(samples: list[dict]) -> dict:
    lookup = {}
    for index, sample in enumerate(samples):
        keys = {sample.get("sample_id", index), str(sample.get("sample_id", index)), sample_key(sample), str(sample_key(sample))}
        for key in keys:
            lookup[key] = sample
    return lookup


def lookup_sample(lookup: dict, sample_id):
    if sample_id in lookup:
        return lookup[sample_id]
    text = str(sample_id)
    if text in lookup:
        return lookup[text]
    try:
        numeric = int(text)
    except (TypeError, ValueError):
        numeric = None
    if numeric is not None and numeric in lookup:
        return lookup[numeric]
    raise KeyError(f"Could not find sample_id={sample_id!r} in task data.")


def clean_triplet_row(row: dict) -> dict:
    out = dict(row)
    for key in ("das_seed", "base_value", "causal_donor_value", "latent_donor_value"):
        out[key] = int(out[key])
    out["preferred_latent_not_base"] = str(out.get("preferred_latent_not_base", "")).lower() == "true"
    return out


def is_same_first_diff_second(d_value: int, c_value: int) -> bool:
    d = digits(d_value)
    c = digits(c_value)
    return len(d) == 2 and len(c) == 2 and d[0] == c[0] and d[1] != c[1]


def load_exact_triplets(args, task: dict, seed: int, samples: list[dict], pairs: list[dict]) -> tuple[list[dict], str]:
    lookup = sample_lookup(samples)
    path = args.latent_interaction_dir / "triplets.csv"
    rows = []
    if path.exists():
        for raw in read_csv_rows(path):
            if raw.get("task_key") != task["task_key"]:
                continue
            if int(raw.get("das_seed", -1)) != int(seed):
                continue
            if raw.get("mismatch_category") != PRIMARY_MISMATCH_CATEGORY:
                continue
            row = clean_triplet_row(raw)
            if not is_same_first_diff_second(row["causal_donor_value"], row["latent_donor_value"]):
                continue
            row["base"] = lookup_sample(lookup, row["base_sample_id"])
            row["causal_donor"] = lookup_sample(lookup, row["causal_donor_sample_id"])
            row["latent_donor"] = lookup_sample(lookup, row["latent_donor_sample_id"])
            row["triplet_source"] = str(path)
            rows.append(row)
            if len(rows) >= args.max_triplets:
                return rows, "reused_triplets_csv"
    if rows:
        return rows, "reused_triplets_csv_partial"

    fallback_args = argparse.Namespace(**vars(args))
    fallback_args.n_triplets = args.max_triplets
    fallback_args.mismatch_categories = [PRIMARY_MISMATCH_CATEGORY]
    rows = build_triplets(fallback_args, task=task, seed=seed, pairs=pairs, samples=samples)
    return rows[: args.max_triplets], "resampled_fallback"


def encode_sample(args, model, processor, tokenizer, task: dict, sample: dict, data_root: Path) -> tuple[dict, int, int, str]:
    if task["modality"] == "text":
        prompt = format_prompt(tokenizer, sample, args.use_chat_template or uses_chat_template(args.model))
        position = resolve_position(tokenizer, prompt, args.text_position)
        encoding = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    else:
        if processor is None:
            raise ValueError("Image task requires processor.")
        prompt = sample_prompt(processor, sample, args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(sample, data_root))
        position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.image_position)[0]
        encoding = inputs_to_device(make_inputs(processor, [prompt], [image]), model.device)
    length = int(encoding["attention_mask"][0].sum().item()) if "attention_mask" in encoding else int(encoding["input_ids"].shape[1])
    return encoding, position, length, prompt


def add_patch_hook(stack: ExitStack, model, blocks, layer: int, hook_name: str, position: int, delta: torch.Tensor) -> None:
    delta = delta.detach().to(model.device)

    def patch_tensor(value):
        activations = hidden(value)
        updated = activations.clone()
        updated[0, position] = activations[0, position] + delta.to(dtype=activations.dtype)
        return replace_hidden(value, updated)

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


def add_capture_hooks(stack: ExitStack, blocks, layers: list[int], hook_name: str, store: dict[int, torch.Tensor]) -> None:
    for layer in layers:
        module, pre_hook = hook_module(blocks[layer - 1], hook_name)
        if pre_hook:
            def capture_pre(_module, inputs, layer=layer):
                store[layer] = hidden(inputs[0]).detach()

            handle = module.register_forward_pre_hook(capture_pre)
        else:
            def capture_post(_module, _inputs, output, layer=layer):
                store[layer] = hidden(output).detach()

            handle = module.register_forward_hook(capture_post)
        stack.callback(handle.remove)


@torch.no_grad()
def run_prompt_forward(
    args,
    model,
    processor,
    tokenizer,
    blocks,
    task: dict,
    sample: dict,
    data_root: Path,
    *,
    delta: torch.Tensor | None,
    capture_t0: bool,
    keep_cache: bool = True,
) -> PromptForward:
    encoding, position, prompt_length, prompt = encode_sample(args, model, processor, tokenizer, task, sample, data_root)
    captured = {}
    with ExitStack() as stack:
        if delta is not None:
            add_patch_hook(stack, model, blocks, args.layer, args.hook, position, delta)
        if capture_t0:
            add_capture_hooks(stack, blocks, [args.layer], args.hook, captured)
        outputs = model(**encoding, use_cache=keep_cache)
    h_t0 = None
    if capture_t0:
        h_t0 = captured[args.layer][0, position].float().cpu()
    return PromptForward(
        logits_t0=outputs.logits[0, prompt_length - 1].float().detach().cpu(),
        past_key_values=outputs.past_key_values if keep_cache else None,
        h_t0=h_t0,
        prompt_length=prompt_length,
        prompt_position=position,
        prompt=prompt,
    )


@torch.no_grad()
def run_cached_digit_step(args, model, tokenizer, blocks, prompt_forward: PromptForward, digit_token_id: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    if prompt_forward.past_key_values is None:
        raise ValueError("Cached digit step requires patched prompt past_key_values.")
    input_ids = torch.tensor([[digit_token_id]], dtype=torch.long, device=model.device)
    attention_mask = torch.ones((1, prompt_forward.prompt_length + 1), dtype=torch.long, device=model.device)
    captured = {}
    with ExitStack() as stack:
        add_capture_hooks(stack, blocks, args.capture_layers, args.hook, captured)
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=prompt_forward.past_key_values,
                use_cache=True,
            )
        except (TypeError, RuntimeError) as error:
            if "attention_mask" not in str(error) and "shape" not in str(error) and "cache" not in str(error).lower():
                raise
            outputs = model(input_ids=input_ids, past_key_values=prompt_forward.past_key_values, use_cache=True)
    h_t1 = {layer: value[0, 0].float().cpu() for layer, value in captured.items()}
    return outputs.logits[0, 0].float().detach().cpu(), h_t1


def digit_token_table(tokenizer) -> dict[str, int]:
    table = {}
    for digit in range(10):
        token_ids = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(f"Digit {digit} is not a single token: {token_ids}")
        table[str(digit)] = int(token_ids[0])
    return table


def run_two_step(
    args,
    model,
    processor,
    tokenizer,
    blocks,
    task: dict,
    sample: dict,
    data_root: Path,
    *,
    delta: torch.Tensor | None,
    first_digit: str,
    first_digit_token_id: int,
    capture_t0: bool,
) -> TwoStepRun:
    prompt_forward = run_prompt_forward(
        args,
        model,
        processor,
        tokenizer,
        blocks,
        task,
        sample,
        data_root,
        delta=delta,
        capture_t0=capture_t0,
        keep_cache=True,
    )
    logits_t1, h_t1 = run_cached_digit_step(args, model, tokenizer, blocks, prompt_forward, first_digit_token_id)
    t0_argmax = int(prompt_forward.logits_t0.argmax().item())
    return TwoStepRun(
        logits_t0=prompt_forward.logits_t0,
        logits_t1=logits_t1,
        h_t0=prompt_forward.h_t0,
        h_t1_by_layer=h_t1,
        t0_argmax_token_id=t0_argmax,
        t0_argmax_text=tokenizer.decode([t0_argmax], skip_special_tokens=False),
        first_digit_token_id=first_digit_token_id,
        first_digit_text=first_digit,
        prompt_length=prompt_forward.prompt_length,
        prompt_position=prompt_forward.prompt_position,
        prompt=prompt_forward.prompt,
    )


def readout_coords(hidden_state: torch.Tensor, digit_basis: torch.Tensor) -> torch.Tensor:
    return hidden_state.float() @ digit_basis.float()


def projection_norm(delta: torch.Tensor, digit_basis: torch.Tensor) -> float:
    return float((delta.float() @ digit_basis.float()).norm())


def digit_scores(logits: torch.Tensor, digit_token_ids: dict[str, int], digit: str) -> tuple[float, float]:
    token_id = digit_token_ids[digit]
    probs = logits.float().softmax(dim=-1)
    return float(logits[token_id]), float(probs[token_id])


def margin_c_over_d(logits: torch.Tensor, digit_token_ids: dict[str, int], d_unit: str, c_unit: str) -> float:
    return float(logits[digit_token_ids[c_unit]] - logits[digit_token_ids[d_unit]])


def condition_digit_row(base: dict, condition: str, run: TwoStepRun, digit_token_ids: dict[str, int]) -> dict:
    meta = base["meta"]
    d_digits = digits(meta["causal_donor_value"])
    c_digits = digits(meta["latent_donor_value"])
    d_unit, c_unit = d_digits[1], c_digits[1]
    logit_d, prob_d = digit_scores(run.logits_t1, digit_token_ids, d_unit)
    logit_c, prob_c = digit_scores(run.logits_t1, digit_token_ids, c_unit)
    argmax_id = int(run.logits_t1.argmax().item())
    argmax_text = base["tokenizer"].decode([argmax_id], skip_special_tokens=False)
    return {
        **meta,
        "condition": condition,
        "first_digit": run.first_digit_text,
        "first_digit_token_id": run.first_digit_token_id,
        "second_digit_d": d_unit,
        "second_digit_c": c_unit,
        "logit_units_d": logit_d,
        "prob_units_d": prob_d,
        "logit_units_c": logit_c,
        "prob_units_c": prob_c,
        "margin_c_over_d": logit_c - logit_d,
        "t1_argmax_token_id": argmax_id,
        "t1_argmax_text": argmax_text,
        "t1_argmax_is_units_d": argmax_id == digit_token_ids[d_unit],
        "t1_argmax_is_units_c": argmax_id == digit_token_ids[c_unit],
        "patched_kv_cache_reused": True,
        "t1_patch_applied": False,
    }


def tensor_json(value: torch.Tensor) -> str:
    return json.dumps([float(item) for item in value.float().tolist()])


def pick_random_same_tens(args, samples: list[dict], triplet: dict, n: int) -> list[dict]:
    rng = random.Random(stable_seed(args.triplet_seed, "same_tens_control", triplet["task_key"], triplet["das_seed"], triplet["triplet_id"]))
    d_value = int(triplet["causal_donor_value"])
    c_value = int(triplet["latent_donor_value"])
    d_digits, c_digits = digits(d_value), digits(c_value)
    excluded = {str(triplet["base_sample_id"]), str(triplet["causal_donor_sample_id"]), str(triplet["latent_donor_sample_id"])}
    candidates = []
    for sample in samples:
        value = target_value(sample, args.target)
        value_digits = digits(value)
        if len(value_digits) != 2:
            continue
        if str(sample_key(sample)) in excluded:
            continue
        if value_digits[0] != d_digits[0]:
            continue
        if value_digits[1] in {d_digits[1], c_digits[1]}:
            continue
        candidates.append(sample)
    rng.shuffle(candidates)
    return candidates[:n]


def base_meta(task: dict, seed: int, triplet: dict) -> dict:
    return {
        "task": task["task"],
        "task_key": task["task_key"],
        "task_label": task["task_label"],
        "modality": task["modality"],
        "operation": task["operation"],
        "das_seed": seed,
        "triplet_id": triplet["triplet_id"],
        "pair_id": triplet.get("pair_id"),
        "base_sample_id": triplet["base_sample_id"],
        "causal_donor_sample_id": triplet["causal_donor_sample_id"],
        "latent_donor_sample_id": triplet["latent_donor_sample_id"],
        "base_value": int(triplet["base_value"]),
        "causal_donor_value": int(triplet["causal_donor_value"]),
        "latent_donor_value": int(triplet["latent_donor_value"]),
        "mismatch_category": triplet["mismatch_category"],
    }


def summarize(conversion_rows: list[dict], logits_rows: list[dict], final_layer: int) -> tuple[list[dict], dict]:
    logits_lookup = {
        (row["task_key"], row["das_seed"], row["triplet_id"], row["condition"]): row
        for row in logits_rows
    }
    grouped = {}
    for row in conversion_rows:
        key = (row["task_key"], row["task_label"], row["das_seed"], row["layer"])
        grouped.setdefault(key, []).append(row)
    summary_rows = []
    for (task_key, task_label, seed, layer), rows in sorted(grouped.items()):
        matched_probs_d = []
        mismatch_probs_d = []
        matched_probs_c = []
        mismatch_probs_c = []
        for row in rows:
            m = logits_lookup.get((task_key, seed, row["triplet_id"], "matched_Cd_Ld"))
            mm = logits_lookup.get((task_key, seed, row["triplet_id"], "mismatch_Cd_Lc"))
            if m:
                matched_probs_d.append(m.get("prob_units_d"))
                matched_probs_c.append(m.get("prob_units_c"))
            if mm:
                mismatch_probs_d.append(mm.get("prob_units_d"))
                mismatch_probs_c.append(mm.get("prob_units_c"))
        final_flips = None
        if int(layer) == int(final_layer):
            paired = []
            for row in rows:
                m = logits_lookup.get((task_key, seed, row["triplet_id"], "matched_Cd_Ld"))
                mm = logits_lookup.get((task_key, seed, row["triplet_id"], "mismatch_Cd_Lc"))
                if m and mm:
                    paired.append(bool(m.get("t1_argmax_is_units_d")) and bool(mm.get("t1_argmax_is_units_c")))
            final_flips = mean(paired)
        summary_rows.append(
            {
                "task_key": task_key,
                "task_label": task_label,
                "das_seed": seed,
                "layer": layer,
                "n": len(rows),
                "mean_conversion_cosine": mean(row.get("conversion_cosine") for row in rows),
                "median_conversion_cosine": median(row.get("conversion_cosine") for row in rows),
                "matched_prob_units_d": mean(matched_probs_d),
                "mismatch_prob_units_d": mean(mismatch_probs_d),
                "matched_prob_units_c": mean(matched_probs_c),
                "mismatch_prob_units_c": mean(mismatch_probs_c),
                "mean_margin_shift": mean(row.get("intervention_margin_shift") for row in rows),
                "mean_conversion_recovery": mean(row.get("conversion_recovery") for row in rows),
                "fraction_mismatch_flips_argmax_d_to_c": final_flips,
            }
        )
    compact = {}
    for row in summary_rows:
        compact.setdefault(row["task_key"], {}).setdefault(str(row["das_seed"]), {})[str(row["layer"])] = row
    return summary_rows, compact


def plot_results(output_dir: Path, summary_rows: list[dict], logits_rows: list[dict], conversion_rows: list[dict]) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for task_key, label in TASK_LABELS.items():
        rows = [row for row in summary_rows if row["task_key"] == task_key]
        if not rows:
            continue
        by_layer = {}
        for row in rows:
            by_layer.setdefault(int(row["layer"]), []).append(row["mean_conversion_cosine"])
        layers = sorted(by_layer)
        values = [mean(by_layer[layer]) for layer in layers]
        ax.plot(layers, values, marker="o", label=label)
    ax.axhline(0, color="#777777", linewidth=1, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Conversion cosine")
    ax.set_title("Latent-to-readout conversion after shared first digit")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "figure1_causal_readout_conversion.png", dpi=220)
    fig.savefig(output_dir / "figure1_causal_readout_conversion.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    conditions = ["matched_Cd_Ld", "mismatch_Cd_Lc", "C_only_d"]
    labels = ["matched", "mismatch", "C-only"]
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for ax, metric, title in zip(axes, ["prob_units_d", "prob_units_c"], ["P(units d)", "P(units c)"]):
        vals = [mean(row.get(metric) for row in logits_rows if row.get("condition") == condition) or 0.0 for condition in conditions]
        ax.bar(range(len(conditions)), vals, color=colors)
        ax.set_xticks(range(len(conditions)), labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Probability")
    fig.tight_layout()
    fig.savefig(output_dir / "figure2_second_digit_probability.png", dpi=220)
    fig.savefig(output_dir / "figure2_second_digit_probability.pdf")
    plt.close(fig)

    candidates = [
        row for row in conversion_rows
        if row.get("layer") == max(int(item["layer"]) for item in conversion_rows)
        and row.get("conversion_cosine") is not None
    ]
    if candidates:
        example = max(candidates, key=lambda row: (row.get("conversion_cosine") or -999, row.get("intervention_margin_shift") or -999))
        rows = [
            row for row in logits_rows
            if row["task_key"] == example["task_key"]
            and int(row["das_seed"]) == int(example["das_seed"])
            and row["triplet_id"] == example["triplet_id"]
            and row["condition"] in conditions
        ]
        if rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            x = range(len(rows))
            ax.bar([i - 0.18 for i in x], [row["prob_units_d"] for row in rows], width=0.36, label="units d", color="#4c78a8")
            ax.bar([i + 0.18 for i in x], [row["prob_units_c"] for row in rows], width=0.36, label="units c", color="#f58518")
            ax.set_xticks(list(x), [row["condition"].replace("_", "\n") for row in rows])
            ax.set_ylabel("Probability at t1")
            ax.set_title(
                f"Example {example['task_label']} seed {example['das_seed']}: "
                f"C_{example['causal_donor_value']} + L_{example['latent_donor_value']}"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(output_dir / "figure3_representative_example.png", dpi=220)
            fig.savefig(output_dir / "figure3_representative_example.pdf")
            plt.close(fig)


def output_exists(args) -> bool:
    return (args.output_dir / "summary.json").exists() and not args.force


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.tasks = ["text:addition"]
        args.seeds = [0]
        args.max_triplets = min(args.max_triplets, 16)
        args.capture_layers = [43, 47]
        args.n_random_same_tens = min(args.n_random_same_tens, 1)
    args.capture_layers = sorted(set(int(layer) for layer in args.capture_layers))
    tasks = normalize_tasks(args.tasks)
    if args.artifact_check_only:
        missing = []
        triplet_counts = {}
        for task in tasks:
            for seed in args.seeds:
                try:
                    metadata_for_run(args, task, seed)
                except FileNotFoundError as error:
                    missing.append(str(error))
        triplet_path = args.latent_interaction_dir / "triplets.csv"
        if not triplet_path.exists():
            missing.append(str(triplet_path))
        else:
            rows = read_csv_rows(triplet_path)
            for task in tasks:
                for seed in args.seeds:
                    count = sum(
                        1
                        for row in rows
                        if row.get("task_key") == task["task_key"]
                        and int(row.get("das_seed", -1)) == int(seed)
                        and row.get("mismatch_category") == PRIMARY_MISMATCH_CATEGORY
                        and is_same_first_diff_second(int(row["causal_donor_value"]), int(row["latent_donor_value"]))
                    )
                    triplet_counts[(task["task_key"], seed)] = count
                    if count <= 0:
                        missing.append(f"{triplet_path} has no {PRIMARY_MISMATCH_CATEGORY} triplets for {task['task_key']} seed={seed}")
        if missing:
            raise FileNotFoundError("\n".join(missing[:20]))
        print(f"ARTIFACT_CHECK_OK latent_to_next_digit_readout triplet_counts={triplet_counts}")
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
    validate_block_layers(model, [args.layer, *args.capture_layers])
    hidden_size = get_hidden_size(model)
    digit_basis, digit_rows = build_digit_readout_span(model, tokenizer)
    digit_basis = digit_basis.float().cpu()
    digit_token_ids = digit_token_table(tokenizer)

    triplet_inventory = []
    t0_rows = []
    t1_rows = []
    second_digit_rows = []
    natural_rows = []
    conversion_rows = []
    random_rows = []
    sanity_rows = []

    for task in tasks:
        for seed in args.seeds:
            directory, result_row, _payload, raw_basis = metadata_for_run(args, task, seed)
            data_path = Path(result_row["data_path"])
            data_root = data_path.parent
            samples = load_jsonl(data_path)
            pairs = load_pairs_for_task(directory, data_path)
            basis, _ordered, causal_component, latent_component, split_diag = build_ordered_split(
                args,
                raw_basis=raw_basis,
                digit_basis=digit_basis,
                hidden_size=hidden_size,
                m_readout=m_selector[(task["task"], seed)][0],
            )
            if causal_component.shape[1] != 9 or latent_component.shape[1] != 13:
                raise ValueError(
                    f"Expected C dim 9 and L dim 13, got C={causal_component.shape[1]} L={latent_component.shape[1]}."
                )
            latent_digit_overlap = float((digit_basis.T @ latent_component.float()).norm())
            if latent_digit_overlap > args.sanity_tolerance:
                print(f"WARNING: U^T L overlap is {latent_digit_overlap:.3e}, above tolerance {args.sanity_tolerance:.1e}.")

            triplets, triplet_source = load_exact_triplets(args, task, seed, samples, pairs)
            print(
                f"\n{task['task_key']} seed={seed}: triplets={len(triplets)} source={triplet_source} "
                f"layers={args.capture_layers} U^T L={latent_digit_overlap:.3e}"
            )
            natural_cache: dict[tuple[str, str], TwoStepRun] = {}
            prompt_cache: dict[str, PromptForward] = {}

            def prompt_state(sample: dict) -> PromptForward:
                key = str(sample_key(sample))
                if key not in prompt_cache:
                    prompt_cache[key] = run_prompt_forward(
                        args,
                        model,
                        processor,
                        tokenizer,
                        blocks,
                        task,
                        sample,
                        data_root,
                        delta=None,
                        capture_t0=True,
                        keep_cache=False,
                    )
                return prompt_cache[key]

            def natural_two_step(sample: dict, first_digit: str) -> TwoStepRun:
                key = (str(sample_key(sample)), first_digit)
                if key not in natural_cache:
                    natural_cache[key] = run_two_step(
                        args,
                        model,
                        processor,
                        tokenizer,
                        blocks,
                        task,
                        sample,
                        data_root,
                        delta=None,
                        first_digit=first_digit,
                        first_digit_token_id=digit_token_ids[first_digit],
                        capture_t0=True,
                    )
                return natural_cache[key]

            for triplet in tqdm(triplets, desc=f"7B {task['task_label']} seed={seed}"):
                d_value = int(triplet["causal_donor_value"])
                c_value = int(triplet["latent_donor_value"])
                if mismatch_category(d_value, c_value) != PRIMARY_MISMATCH_CATEGORY or not is_same_first_diff_second(d_value, c_value):
                    raise ValueError(f"Triplet {triplet['triplet_id']} is not same-first/different-second.")
                first_digit = digits(d_value)[0]
                first_digit_token_id = digit_token_ids[first_digit]
                meta = base_meta(task, seed, triplet)
                triplet_inventory.append(
                    {
                        **meta,
                        "triplet_source": triplet_source,
                        "first_digit": first_digit,
                        "first_digit_token_id": first_digit_token_id,
                        "same_first_digit": digits(d_value)[0] == digits(c_value)[0],
                        "different_second_digit": digits(d_value)[1] != digits(c_value)[1],
                    }
                )

                h_b = prompt_state(triplet["base"]).h_t0
                h_d = prompt_state(triplet["causal_donor"]).h_t0
                h_c = prompt_state(triplet["latent_donor"]).h_t0
                delta_bd = h_d - h_b
                delta_bc = h_c - h_b
                full_bd = project_delta(delta_bd, basis.float())
                c_bd = project_delta(delta_bd, causal_component.float())
                l_bd = project_delta(delta_bd, latent_component.float())
                l_bc = project_delta(delta_bc, latent_component.float())
                matched_delta = c_bd + l_bd
                matched_full_error = float((matched_delta - full_bd).norm())
                if matched_full_error > args.sanity_tolerance:
                    raise RuntimeError(f"matched Cd+Ld no longer reproduces full DAS for {triplet['triplet_id']}: {matched_full_error}")

                natural_d = natural_two_step(triplet["causal_donor"], first_digit)
                natural_c = natural_two_step(triplet["latent_donor"], first_digit)
                run_payload = {"meta": meta, "tokenizer": tokenizer}
                condition_deltas = {
                    "matched_Cd_Ld": matched_delta,
                    "mismatch_Cd_Lc": c_bd + l_bc,
                    "C_only_d": c_bd,
                }
                if not args.skip_full_original_das:
                    condition_deltas["full_original_das"] = full_bd
                condition_runs = {}
                for condition, delta in condition_deltas.items():
                    run = run_two_step(
                        args,
                        model,
                        processor,
                        tokenizer,
                        blocks,
                        task,
                        triplet["base"],
                        data_root,
                        delta=delta,
                        first_digit=first_digit,
                        first_digit_token_id=first_digit_token_id,
                        capture_t0=False,
                    )
                    condition_runs[condition] = run
                    logit_first, prob_first = digit_scores(run.logits_t0, digit_token_ids, first_digit)
                    if condition == "matched_Cd_Ld":
                        l_part = l_bd
                    elif condition == "mismatch_Cd_Lc":
                        l_part = l_bc
                    else:
                        l_part = torch.zeros_like(l_bd)
                    t0_rows.append(
                        {
                            **meta,
                            "condition": condition,
                            "first_digit": first_digit,
                            "first_digit_token_id": first_digit_token_id,
                            "t0_L_readout_norm": projection_norm(l_part, digit_basis),
                            "t0_delta_readout_norm": projection_norm(delta, digit_basis),
                            "t0_first_digit_logit": logit_first,
                            "t0_first_digit_prob": prob_first,
                            "t0_argmax_token_id": run.t0_argmax_token_id,
                            "t0_argmax_text": run.t0_argmax_text,
                            "t0_argmax_is_shared_first_digit": run.t0_argmax_token_id == first_digit_token_id,
                            "matched_full_delta_error": matched_full_error,
                        }
                    )
                    second_digit_rows.append(condition_digit_row(run_payload, condition, run, digit_token_ids))
                    for layer in args.capture_layers:
                        coords = readout_coords(run.h_t1_by_layer[layer], digit_basis)
                        t1_rows.append(
                            {
                                **meta,
                                "condition": condition,
                                "layer": layer,
                                "readout_norm": float(coords.norm()),
                                "readout_coords": tensor_json(coords),
                                "patched_kv_cache_reused": True,
                                "t1_patch_applied": False,
                            }
                        )

                natural_margins = {}
                for label, natural in (("natural_d", natural_d), ("natural_c", natural_c)):
                    second_digit_rows.append(condition_digit_row(run_payload, label, natural, digit_token_ids))
                    natural_margins[label] = margin_c_over_d(natural.logits_t1, digit_token_ids, digits(d_value)[1], digits(c_value)[1])
                    for layer in args.capture_layers:
                        coords = readout_coords(natural.h_t1_by_layer[layer], digit_basis)
                        natural_rows.append(
                            {
                                **meta,
                                "reference": label,
                                "layer": layer,
                                "readout_norm": float(coords.norm()),
                                "readout_coords": tensor_json(coords),
                                "margin_c_over_d": natural_margins[label],
                            }
                        )

                matched = condition_runs["matched_Cd_Ld"]
                mismatch = condition_runs["mismatch_Cd_Lc"]
                matched_margin = margin_c_over_d(matched.logits_t1, digit_token_ids, digits(d_value)[1], digits(c_value)[1])
                mismatch_margin = margin_c_over_d(mismatch.logits_t1, digit_token_ids, digits(d_value)[1], digits(c_value)[1])
                natural_margin = natural_margins["natural_c"] - natural_margins["natural_d"]
                margin_shift = mismatch_margin - matched_margin
                recovery = None if abs(natural_margin) < args.recovery_denominator_floor else margin_shift / natural_margin
                for layer in args.capture_layers:
                    r_matched = readout_coords(matched.h_t1_by_layer[layer], digit_basis)
                    r_mismatch = readout_coords(mismatch.h_t1_by_layer[layer], digit_basis)
                    r_nat_d = readout_coords(natural_d.h_t1_by_layer[layer], digit_basis)
                    r_nat_c = readout_coords(natural_c.h_t1_by_layer[layer], digit_basis)
                    delta_mismatch = r_mismatch - r_matched
                    delta_natural = r_nat_c - r_nat_d
                    conversion_rows.append(
                        {
                            **meta,
                            "layer": layer,
                            "delta_readout_mismatch_norm": float(delta_mismatch.norm()),
                            "delta_readout_natural_norm": float(delta_natural.norm()),
                            "delta_readout_mismatch_coords": tensor_json(delta_mismatch),
                            "delta_readout_natural_coords": tensor_json(delta_natural),
                            "conversion_cosine": cosine(delta_mismatch, delta_natural),
                            "matched_margin_c_over_d": matched_margin,
                            "mismatch_margin_c_over_d": mismatch_margin,
                            "natural_d_margin_c_over_d": natural_margins["natural_d"],
                            "natural_c_margin_c_over_d": natural_margins["natural_c"],
                            "natural_margin": natural_margin,
                            "intervention_margin_shift": margin_shift,
                            "conversion_recovery": recovery,
                            "conversion_recovery_denominator_stable": abs(natural_margin) >= args.recovery_denominator_floor,
                            "cos_mismatch_to_natural_c": cosine(r_mismatch, r_nat_c),
                            "cos_mismatch_to_natural_d": cosine(r_mismatch, r_nat_d),
                            "cos_matched_to_natural_c": cosine(r_matched, r_nat_c),
                            "cos_matched_to_natural_d": cosine(r_matched, r_nat_d),
                            "dist_mismatch_to_natural_c": euclidean(r_mismatch, r_nat_c),
                            "dist_mismatch_to_natural_d": euclidean(r_mismatch, r_nat_d),
                            "dist_matched_to_natural_c": euclidean(r_matched, r_nat_c),
                            "dist_matched_to_natural_d": euclidean(r_matched, r_nat_d),
                        }
                    )

                if not args.skip_random_controls and args.n_random_same_tens > 0:
                    for random_index, random_sample in enumerate(pick_random_same_tens(args, samples, triplet, args.n_random_same_tens)):
                        random_value = target_value(random_sample, args.target)
                        h_r = prompt_state(random_sample).h_t0
                        l_br = project_delta(h_r - h_b, latent_component.float())
                        random_delta = c_bd + l_br
                        random_run = run_two_step(
                            args,
                            model,
                            processor,
                            tokenizer,
                            blocks,
                            task,
                            triplet["base"],
                            data_root,
                            delta=random_delta,
                            first_digit=first_digit,
                            first_digit_token_id=first_digit_token_id,
                            capture_t0=False,
                        )
                        random_margin = margin_c_over_d(random_run.logits_t1, digit_token_ids, digits(d_value)[1], digits(c_value)[1])
                        random_base = {
                            **meta,
                            "condition": "random_Cd_Lr",
                            "random_latent_index": random_index,
                            "random_latent_sample_id": sample_key(random_sample),
                            "random_latent_value": random_value,
                            "random_latent_units_digit": digits(random_value)[1],
                            "t0_L_readout_norm": projection_norm(l_br, digit_basis),
                            "margin_c_over_d": random_margin,
                            "random_minus_matched_margin": random_margin - matched_margin,
                        }
                        logit_d, prob_d = digit_scores(random_run.logits_t1, digit_token_ids, digits(d_value)[1])
                        logit_c, prob_c = digit_scores(random_run.logits_t1, digit_token_ids, digits(c_value)[1])
                        for layer in args.capture_layers:
                            r_random = readout_coords(random_run.h_t1_by_layer[layer], digit_basis)
                            r_matched = readout_coords(matched.h_t1_by_layer[layer], digit_basis)
                            r_nat_d = readout_coords(natural_d.h_t1_by_layer[layer], digit_basis)
                            r_nat_c = readout_coords(natural_c.h_t1_by_layer[layer], digit_basis)
                            random_rows.append(
                                {
                                    **random_base,
                                    "layer": layer,
                                    "prob_units_d": prob_d,
                                    "prob_units_c": prob_c,
                                    "logit_units_d": logit_d,
                                    "logit_units_c": logit_c,
                                    "conversion_cosine_to_primary_c": cosine(r_random - r_matched, r_nat_c - r_nat_d),
                                    "cos_random_to_natural_c": cosine(r_random, r_nat_c),
                                    "cos_random_to_natural_d": cosine(r_random, r_nat_d),
                                    "dist_random_to_natural_c": euclidean(r_random, r_nat_c),
                                    "dist_random_to_natural_d": euclidean(r_random, r_nat_d),
                                }
                            )

            sanity_rows.append(
                {
                    "task_key": task["task_key"],
                    "task_label": task["task_label"],
                    "das_seed": seed,
                    "n_triplets": len(triplets),
                    "triplet_source": triplet_source,
                    "layer": args.layer,
                    "capture_layers": args.capture_layers,
                    "k": args.k,
                    "m_readout": causal_component.shape[1],
                    "latent_dim": latent_component.shape[1],
                    "U_T_L_norm": latent_digit_overlap,
                    "split_projector_error": split_diag["split_projector_error"],
                    "component_orthogonality": split_diag["component_orthogonality"],
                    "patched_kv_cache_reused": True,
                    "t1_patch_applied": False,
                }
            )
            summary_rows, compact_summary = summarize(conversion_rows, second_digit_rows, max(args.capture_layers))
            write_csv(args.output_dir / "triplet_inventory.csv", triplet_inventory)
            write_csv(args.output_dir / "t0_readout_diagnostics.csv", t0_rows)
            write_csv(args.output_dir / "t1_layerwise_readout.csv", t1_rows)
            write_csv(args.output_dir / "second_digit_logits.csv", second_digit_rows)
            write_csv(args.output_dir / "natural_reference_metrics.csv", natural_rows)
            write_csv(args.output_dir / "conversion_metrics.csv", conversion_rows)
            write_csv(args.output_dir / "random_latent_controls.csv", random_rows)
            write_csv(args.output_dir / "summary_by_layer.csv", summary_rows)
            save_json(args.output_dir / "sanity_checks.json", {"rows": sanity_rows})
            save_json(
                args.output_dir / "summary.json",
                {
                    "model": args.model,
                    "resolved_model": resolved_model,
                    "output_dir": args.output_dir,
                    "tasks": [task["task_key"] for task in tasks],
                    "seeds": args.seeds,
                    "layer": args.layer,
                    "capture_layers": args.capture_layers,
                    "k": args.k,
                    "m_readout": args.m_readout,
                    "latent_dim": args.k - args.m_readout,
                    "hook": args.hook,
                    "max_triplets": args.max_triplets,
                    "n_random_same_tens": args.n_random_same_tens,
                    "primary_metric": "conversion_cosine",
                    "cache_requirement": "patched prompt past_key_values reused for teacher-forced first digit; no t1 patch",
                    "n_triplets": len(triplet_inventory),
                    "n_conversion_rows": len(conversion_rows),
                    "summary": compact_summary,
                    "sanity_checks": sanity_rows,
                    "digit_tokens": digit_token_ids,
                    "digit_readout_rows": digit_rows,
                },
            )

    summary_rows, compact_summary = summarize(conversion_rows, second_digit_rows, max(args.capture_layers))
    write_csv(args.output_dir / "triplet_inventory.csv", triplet_inventory)
    write_csv(args.output_dir / "t0_readout_diagnostics.csv", t0_rows)
    write_csv(args.output_dir / "t1_layerwise_readout.csv", t1_rows)
    write_csv(args.output_dir / "second_digit_logits.csv", second_digit_rows)
    write_csv(args.output_dir / "natural_reference_metrics.csv", natural_rows)
    write_csv(args.output_dir / "conversion_metrics.csv", conversion_rows)
    write_csv(args.output_dir / "random_latent_controls.csv", random_rows)
    write_csv(args.output_dir / "summary_by_layer.csv", summary_rows)
    save_json(args.output_dir / "sanity_checks.json", {"rows": sanity_rows})
    if not args.skip_plots:
        plot_results(args.output_dir, summary_rows, second_digit_rows, conversion_rows)
    save_json(
        args.output_dir / "summary.json",
        {
            "model": args.model,
            "resolved_model": resolved_model,
            "output_dir": args.output_dir,
            "tasks": [task["task_key"] for task in tasks],
            "seeds": args.seeds,
            "layer": args.layer,
            "capture_layers": args.capture_layers,
            "k": args.k,
            "m_readout": args.m_readout,
            "latent_dim": args.k - args.m_readout,
            "hook": args.hook,
            "max_triplets": args.max_triplets,
            "n_random_same_tens": args.n_random_same_tens,
            "primary_metric": "conversion_cosine",
            "cache_requirement": "patched prompt past_key_values reused for teacher-forced first digit; no t1 patch",
            "n_triplets": len(triplet_inventory),
            "n_conversion_rows": len(conversion_rows),
            "summary": compact_summary,
            "sanity_checks": sanity_rows,
            "digit_tokens": digit_token_ids,
            "digit_readout_rows": digit_rows,
        },
    )
    print(f"Saved latent-to-next-digit readout results to {args.output_dir}")


if __name__ == "__main__":
    main()
