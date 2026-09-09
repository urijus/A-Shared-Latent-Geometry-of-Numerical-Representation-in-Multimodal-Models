"""Evaluate an existing arithmetic DAS subspace on a repeat-number task.

If an addition DAS subspace transfers strongly to "repeat this number" prompts,
that supports the worry that the intervention is mostly a digit output channel.
If full activation patching works but the DAS subspace does not, the worry is
weakened.
"""

import argparse
import re
from pathlib import Path

import torch

from src.common import save_jsonl
from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
    build_unique_pairs,
    format_prompt,
    patched_forward,
    resolve_position,
    sequence_scores,
    target_answers,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    build_probe_context,
    uses_chat_template,
)
from src.geometry.subspaces import torch_load_portable
from src.models import get_blocks, load_hf_model, resolve_model_for_loading, validate_block_layers


DEFAULT_SUBSPACE = (
    "results/final_exps/DAS_audit_k_22/text/addition/"
    "das_pca_initialized/split_0/seed_0/subspace.pt"
)


class FullActivationPatch:
    def patch(self, _base_vector, source_vector):
        return source_vector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subspace", type=Path, default=Path(DEFAULT_SUBSPACE))
    parser.add_argument("--model", default=None, help="Defaults to the saved subspace config model.")
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/check_unembeeding/repeat_transfer"))
    parser.add_argument("--prompt_template", default="Output ONLY a number. Repeat this number: {n} =")
    parser.add_argument(
        "--answer_separator",
        default=" ",
        help="Appended to each rendered prompt if absent, so prompt+answer tokenization is prefix-stable.",
    )
    parser.add_argument("--min_value", type=int, default=0)
    parser.add_argument("--max_value", type=int, default=99)
    parser.add_argument("--max_pairs", type=int, default=128)
    parser.add_argument("--pair_seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--run_autoregressive", action="store_true")
    parser.add_argument(
        "--position",
        default="equals",
        help="Patch position in repeat prompts. Use 'equals'/'=' for the last '=' token, or a numeric DAS position.",
    )
    parser.add_argument("--hook", default=None, choices=["resid_pre", "resid_post"])
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--stem", default="repeat_transfer")
    return parser.parse_args()


def render_repeat_prompt(template, value):
    """Render repeat prompt templates while tolerating common brace typos."""
    rendered = re.sub(r"\{\s*(n|number)\s*\}", str(value), template)
    if rendered != template:
        return rendered

    # Common accidental spelling: "{n =}" instead of "{n} =".
    rendered = re.sub(r"\{\s*(n|number)\s*=\s*\}", f"{value} =", template)
    if rendered != template:
        return rendered

    try:
        return template.format(n=value, number=value)
    except KeyError as error:
        raise ValueError(
            "Prompt template must contain '{n}' or '{number}', for example "
            "'Output ONLY a number. Repeat this number: {n} ='. "
            f"Got: {template!r}"
        ) from error


def add_answer_separator(prompt, separator):
    if not separator or prompt.endswith(separator):
        return prompt
    return prompt + separator


def make_repeat_samples(args):
    rows = []
    for sample_id, value in enumerate(range(args.min_value, args.max_value + 1)):
        expr = add_answer_separator(
            render_repeat_prompt(args.prompt_template, value),
            args.answer_separator,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "task": "repeat_number",
                "operation": "repeat",
                "expr": expr,
                "tokens": ["Repeat", str(value), "="],
                "a": value,
                "b": 0,
                "result": value,
                "result_mod_2": value % 2,
                "result_mod_5": value % 5,
                "result_mod_10": value % 10,
                "result_mod_20": value % 20,
                "result_mod_50": value % 50,
                "result_mod_100": value % 100,
            }
        )
    return rows


def load_saved_subspace(path, layer, device):
    payload = torch_load_portable(path)
    basis = torch.as_tensor(payload["basis"]).detach().float()
    return {str(layer): DASSubspace(basis.shape[0], basis.shape[1], initial_basis=basis).to(device)}


def repeat_position_spec(tokenizer, prompt, requested):
    if str(requested).lower() not in {"equals", "=", "eq"}:
        return str(requested)
    context = build_probe_context(tokenizer, prompt)
    matches = [index for index, token in enumerate(context["tokens"]) if token == "="]
    if not matches:
        raise ValueError(f"Could not find '=' in repeat prompt: {prompt!r}")
    return str(matches[-1])


def full_patch_subspace(layer):
    return {str(layer): FullActivationPatch()}


def metric_row(name, metrics, clean, full, ar=None):
    denominator = full["variable_teacher_forced_iia"] - clean
    eta = None if abs(denominator) < 1e-12 else (
        metrics["variable_teacher_forced_iia"] - clean
    ) / denominator
    return {
        "condition": name,
        "variable_teacher_forced_iia": metrics["variable_teacher_forced_iia"],
        "full_answer_teacher_forced_iia": metrics["full_answer_teacher_forced_iia"],
        "source_variable_logprob_gain": metrics.get("source_variable_logprob_gain"),
        "source_answer_logprob_gain": metrics.get("source_answer_logprob_gain"),
        "eta_vs_full_patch_teacher_forced": eta,
        "autoregressive_iia": ar,
    }


def prompt_answer_encoding(tokenizer, prompts, answers, device):
    prompt_ids = [
        tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for prompt in prompts
    ]
    answer_ids = [
        tokenizer(answer, add_special_tokens=False)["input_ids"]
        for answer in answers
    ]
    if any(not ids for ids in answer_ids):
        raise ValueError("At least one answer encoded to zero tokens.")

    lengths = [len(prompt) + len(answer) for prompt, answer in zip(prompt_ids, answer_ids)]
    max_length = max(lengths)
    input_ids = torch.full(
        (len(prompts), max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    answer_positions = []
    for row, (prefix, answer) in enumerate(zip(prompt_ids, answer_ids)):
        ids = prefix + answer
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1
        answer_positions.append(list(range(len(prefix), len(ids))))
    return {"input_ids": input_ids, "attention_mask": attention_mask}, answer_positions


def repeat_teacher_forced_batch(
    model,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    position,
    use_chat_template,
    patch=True,
):
    base_prompts, donor_prompts = [], []
    base_answers, source_answers = [], []
    base_positions, donor_positions = [], []

    for pair in pairs:
        base, source = pair["base"], pair["source"]
        donor = pair.get("donor", source)
        base_prompt = format_prompt(tokenizer, base, use_chat_template)
        donor_prompt = format_prompt(tokenizer, donor, use_chat_template)
        base_prompts.append(base_prompt)
        donor_prompts.append(donor_prompt)
        base_answers.append(str(int(base["result"])))
        source_answers.append(str(int(source["result"])))
        base_positions.append(resolve_position(tokenizer, base_prompt, position))
        donor_positions.append(resolve_position(tokenizer, donor_prompt, position))

    prompts = base_prompts + base_prompts + donor_prompts
    answers = source_answers + base_answers + source_answers
    encoding, answer_positions = prompt_answer_encoding(
        tokenizer, prompts, answers, model.device
    )
    outputs = (
        patched_forward(
            model,
            encoding,
            blocks,
            subspaces,
            layers,
            hook_name,
            base_positions,
            donor_positions,
            n_base_groups=2,
        )
        if patch
        else model(**encoding, use_cache=False)
    )

    n = len(pairs)
    source_score, source_exact = sequence_scores(
        outputs.logits[:n], encoding["input_ids"][:n], answer_positions[:n]
    )
    base_score, _base_exact = sequence_scores(
        outputs.logits[n : 2 * n],
        encoding["input_ids"][n : 2 * n],
        answer_positions[n : 2 * n],
    )
    return {
        "source_variable_logprob": source_score,
        "source_answer_logprob": source_score,
        "base_variable_logprob": base_score,
        "base_answer_logprob": base_score,
        "variable_teacher_forced_iia": source_exact,
        "full_answer_teacher_forced_iia": source_exact,
    }


def tensor_mean(parts):
    return float(torch.cat(parts).mean())


@torch.no_grad()
def evaluate_repeat_teacher_forced(
    model,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    position,
    use_chat_template,
    batch_size,
    include_clean=True,
):
    names = (
        "source_variable_logprob",
        "source_answer_logprob",
        "base_variable_logprob",
        "base_answer_logprob",
        "variable_teacher_forced_iia",
        "full_answer_teacher_forced_iia",
    )
    patched = {name: [] for name in names}
    clean = {name: [] for name in names}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        result = repeat_teacher_forced_batch(
            model,
            tokenizer,
            blocks,
            subspaces,
            layers,
            hook_name,
            batch,
            position,
            use_chat_template,
            patch=True,
        )
        for name in names:
            patched[name].append(result[name].detach().cpu())
        if include_clean:
            baseline = repeat_teacher_forced_batch(
                model,
                tokenizer,
                blocks,
                subspaces,
                layers,
                hook_name,
                batch,
                position,
                use_chat_template,
                patch=False,
            )
            for name in names:
                clean[name].append(baseline[name].detach().cpu())

    metrics = {name: tensor_mean(parts) for name, parts in patched.items()}
    if include_clean:
        clean_metrics = {name: tensor_mean(parts) for name, parts in clean.items()}
        metrics.update(
            {
                "source_variable_logprob_gain": metrics["source_variable_logprob"]
                - clean_metrics["source_variable_logprob"],
                "source_answer_logprob_gain": metrics["source_answer_logprob"]
                - clean_metrics["source_answer_logprob"],
                "base_variable_logprob_change": metrics["base_variable_logprob"]
                - clean_metrics["base_variable_logprob"],
                "base_answer_logprob_change": metrics["base_answer_logprob"]
                - clean_metrics["base_answer_logprob"],
                "clean_counterfactual_variable_teacher_forced_iia": clean_metrics[
                    "variable_teacher_forced_iia"
                ],
                "clean_counterfactual_full_answer_teacher_forced_iia": clean_metrics[
                    "full_answer_teacher_forced_iia"
                ],
            }
        )
    return metrics


@torch.no_grad()
def clean_autoregressive_iia(model, tokenizer, pairs, use_chat_template, max_new_tokens):
    correct = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for pair in pairs:
            prompt = format_prompt(tokenizer, pair["base"], use_chat_template)
            expected = target_answers(pair["base"], pair["source"], "result")[1]
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device)
            generated = []
            for _ in range(max_new_tokens):
                outputs = model(input_ids=input_ids, use_cache=False)
                next_id = outputs.logits[0, input_ids.shape[1] - 1].argmax().reshape(1, 1)
                generated.append(int(next_id.item()))
                input_ids = torch.cat([input_ids, next_id.to(model.device)], dim=1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            match = re.match(r"-?\d+", text)
            correct += int(match is not None and match.group() == expected)
    finally:
        tokenizer.padding_side = old_padding_side
    return correct / len(pairs) if pairs else 0.0


@torch.no_grad()
def repeat_task_teacher_forced_accuracy(
    model,
    tokenizer,
    samples,
    use_chat_template,
    batch_size,
):
    exact = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        prompts = [format_prompt(tokenizer, sample, use_chat_template) for sample in batch]
        answers = [str(int(sample["result"])) for sample in batch]
        encoding, full_positions = prompt_answer_encoding(
            tokenizer, prompts, answers, model.device
        )
        outputs = model(**encoding, use_cache=False)
        _scores, batch_exact = sequence_scores(
            outputs.logits, encoding["input_ids"], full_positions
        )
        exact.append(batch_exact.detach().cpu())
    return float(torch.cat(exact).mean()) if exact else 0.0


@torch.no_grad()
def repeat_task_autoregressive_accuracy(
    model,
    tokenizer,
    samples,
    use_chat_template,
    max_new_tokens,
):
    correct = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for sample in samples:
            prompt = format_prompt(tokenizer, sample, use_chat_template)
            expected = str(int(sample["result"]))
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(model.device)
            generated = []
            for _ in range(max_new_tokens):
                outputs = model(input_ids=input_ids, use_cache=False)
                next_id = outputs.logits[0, input_ids.shape[1] - 1].argmax().reshape(1, 1)
                generated.append(int(next_id.item()))
                input_ids = torch.cat([input_ids, next_id.to(model.device)], dim=1)
                if next_id.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            match = re.match(r"-?\d+", text)
            correct += int(match is not None and match.group() == expected)
    finally:
        tokenizer.padding_side = old_padding_side
    return correct / len(samples) if samples else 0.0


def print_summary(rows):
    first = rows[0]
    print("\nRepeat-task sanity check")
    print(
        "  base task teacher-forced accuracy: "
        f"{first['repeat_task_teacher_forced_accuracy']:.4f}"
    )
    if first["repeat_task_autoregressive_accuracy"] is not None:
        print(
            "  base task autoregressive accuracy: "
            f"{first['repeat_task_autoregressive_accuracy']:.4f}"
        )

    print("\nRepeat-task DAS transfer")
    print("Question: does an arithmetic DAS subspace steer a non-arithmetic repeat task?\n")
    print(
        f"{'condition':26} {'tf_iia':>10} {'logprob_gain':>13} "
        f"{'eta_full':>10} {'ar_iia':>10}"
    )
    print("-" * 75)
    for row in rows:
        gain = (
            "NA"
            if row["source_variable_logprob_gain"] is None
            else f"{row['source_variable_logprob_gain']:.4f}"
        )
        eta = (
            "NA"
            if row["eta_vs_full_patch_teacher_forced"] is None
            else f"{row['eta_vs_full_patch_teacher_forced']:.3f}"
        )
        ar = "NA" if row["autoregressive_iia"] is None else f"{row['autoregressive_iia']:.4f}"
        print(
            f"{row['condition']:26} "
            f"{row['variable_teacher_forced_iia']:10.4f} "
            f"{gain:>13} {eta:>10} {ar:>10}"
        )
    print("\nInterpretation:")
    print("  full patch high, DAS low: evidence against a pure final-readout explanation.")
    print("  full patch high, DAS high: supports the final-readout/output-channel concern.")
    print("  full patch low: inconclusive; patch position or prompt format may not transfer.")


def main():
    args = parse_args()
    payload = torch_load_portable(args.subspace)
    config = payload.get("config", {})
    layer = args.layer or int(config["layer"])
    requested_position = args.position
    hook = args.hook or config.get("hook", "resid_post")
    model_name = args.model or config.get("model")
    if model_name is None:
        raise ValueError("Pass --model because the saved subspace has no config model.")

    samples = make_repeat_samples(args)
    pairs, pair_stats = build_unique_pairs(samples, "result", args.pair_seed, args.max_pairs)
    if not pairs:
        raise ValueError("No valid repeat-task pairs were created.")

    model_path, resolved_model = resolve_model_for_loading(model_name)
    model, tokenizer = load_hf_model(model_path)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = get_blocks(model)
    validate_block_layers(model, [layer])
    use_chat_template = bool(config.get("use_chat_template")) or uses_chat_template(model_name)
    first_prompt = format_prompt(tokenizer, pairs[0]["base"], use_chat_template)
    position = repeat_position_spec(tokenizer, first_prompt, requested_position)

    print("Repeat-transfer diagnostic")
    print(f"  subspace: {args.subspace}")
    print(f"  model: {resolved_model}")
    print(
        f"  layer={layer}, requested_position={requested_position!r}, "
        f"resolved_repeat_position={position!r}, hook={hook}, "
        f"use_chat_template={use_chat_template}"
    )
    for label, sample in [("base", pairs[0]["base"]), ("source", pairs[0]["source"])]:
        prompt = format_prompt(tokenizer, sample, use_chat_template)
        print(f"  {label} prompt: {prompt!r}")
        print(f"  {label} patch HF position: {resolve_position(tokenizer, prompt, position)}")

    full_patch = full_patch_subspace(layer)
    das_subspace = load_saved_subspace(args.subspace, layer, model.device)
    task_teacher_forced_accuracy = repeat_task_teacher_forced_accuracy(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        use_chat_template=use_chat_template,
        batch_size=args.batch_size,
    )
    task_autoregressive_accuracy = None

    full_metrics = evaluate_repeat_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=full_patch,
        layers=[layer],
        hook_name=hook,
        pairs=pairs,
        position=position,
        use_chat_template=use_chat_template,
        batch_size=args.batch_size,
        include_clean=True,
    )
    das_metrics = evaluate_repeat_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=das_subspace,
        layers=[layer],
        hook_name=hook,
        pairs=pairs,
        position=position,
        use_chat_template=use_chat_template,
        batch_size=args.batch_size,
        include_clean=True,
    )

    clean_tf = full_metrics["clean_counterfactual_variable_teacher_forced_iia"]
    rows = [
        {
            "condition": "clean_no_patch",
            "variable_teacher_forced_iia": clean_tf,
            "full_answer_teacher_forced_iia": full_metrics[
                "clean_counterfactual_full_answer_teacher_forced_iia"
            ],
            "source_variable_logprob_gain": 0.0,
            "source_answer_logprob_gain": 0.0,
            "eta_vs_full_patch_teacher_forced": 0.0,
            "autoregressive_iia": None,
        },
        metric_row("full_activation_patch", full_metrics, clean_tf, full_metrics),
        metric_row("arithmetic_das_subspace", das_metrics, clean_tf, full_metrics),
    ]

    if args.run_autoregressive:
        ar_pairs = pairs[: min(len(pairs), args.max_pairs)]
        task_autoregressive_accuracy = repeat_task_autoregressive_accuracy(
            model, tokenizer, samples, use_chat_template, args.max_new_tokens
        )
        rows[0]["autoregressive_iia"] = clean_autoregressive_iia(
            model, tokenizer, ar_pairs, use_chat_template, args.max_new_tokens
        )
        rows[1]["autoregressive_iia"] = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[layer],
            hook_name=hook,
            pairs=ar_pairs,
            position=position,
            target="result",
            use_chat_template=use_chat_template,
            max_new_tokens=args.max_new_tokens,
            description="repeat full patch autoregressive",
        )
        rows[2]["autoregressive_iia"] = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=das_subspace,
            layers=[layer],
            hook_name=hook,
            pairs=ar_pairs,
            position=position,
            target="result",
            use_chat_template=use_chat_template,
            max_new_tokens=args.max_new_tokens,
            description="repeat DAS autoregressive",
        )

    metadata = {
        "subspace_path": str(args.subspace),
        "model": resolved_model,
        "layer": layer,
        "position": position,
        "requested_position": requested_position,
        "saved_das_position": config.get("position"),
        "hook": hook,
        "use_chat_template": use_chat_template,
        "prompt_template": args.prompt_template,
        "answer_separator": args.answer_separator,
        "repeat_task_teacher_forced_accuracy": task_teacher_forced_accuracy,
        "repeat_task_autoregressive_accuracy": task_autoregressive_accuracy,
        "pair_statistics": pair_stats,
        "n_pairs": len(pairs),
    }
    for row in rows:
        row.update(metadata)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(samples, args.output_dir / f"{args.stem}_dataset.jsonl")
    save_jsonl(
        [
            {
                "pair_id": i,
                "base_sample_id": pair["base"]["sample_id"],
                "source_sample_id": pair["source"]["sample_id"],
                "base_expr": pair["base"]["expr"],
                "source_expr": pair["source"]["expr"],
                "base_result": pair["base"]["result"],
                "source_result": pair["source"]["result"],
            }
            for i, pair in enumerate(pairs)
        ],
        args.output_dir / f"{args.stem}_pairs.jsonl",
    )
    jsonl_path = args.output_dir / f"{args.stem}.jsonl"
    save_jsonl(rows, jsonl_path)
    print_summary(rows)
    print(f"Saved {jsonl_path}")


if __name__ == "__main__":
    main()
