"""Steer one arithmetic variable in Fourier space and measure causal IIA."""

import argparse
import hashlib
import re
from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    TARGET_COMPONENTS,
    build_unique_pairs,
    dataset_statistics,
    format_prompt,
    resolve_position,
    sequence_scores,
    target_answers,
    tokenize_answers,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.probes.fourier import (
    load_probe_grid,
    steer_hidden,
)
from src.models import (
    get_blocks,
    load_hf_model,
    resolve_model_for_loading,
    validate_block_layers,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--modality", required=True)
    parser.add_argument("--target", choices=["result", "c1_hat", "c0"], required=True)
    parser.add_argument("--periods", type=int, nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", type=int, default=17)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--method", choices=["ridge", "gd"], default="ridge")
    parser.add_argument("--max_pairs", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--use_chat_template", action="store_true")
    return parser.parse_args()


def steering_values(pairs, target):
    field = {"result": "result", "c1_hat": "c1_hat", "c0": "c0"}[target]
    return [int(pair["source"][field]) for pair in pairs]


def steered_forward(
    model,
    encoding,
    blocks,
    tokenizer,
    prompts,
    steering_targets,
    layer,
    position,
    probes,
    periods,
    alpha,
    resolved_positions=None,
):
    if alpha == 0:
        return model(**encoding, use_cache=False)
    if resolved_positions is None:
        resolved_positions = [
            resolve_position(tokenizer, prompt, position) for prompt in prompts
        ]
    token_positions = torch.tensor(resolved_positions, device=model.device)
    targets = torch.tensor(
        steering_targets, device=model.device, dtype=torch.float32
    )
    batch_indices = torch.arange(len(prompts), device=model.device)

    def hook(_module, _inputs, output):
        hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
        selected = hidden[batch_indices, token_positions]
        hidden[batch_indices, token_positions] = steer_hidden(
            selected,
            targets,
            probes[layer][position],
            periods,
            alpha,
        )
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    with ExitStack() as stack:
        stack.callback(blocks[layer - 1].register_forward_hook(hook).remove)
        return model(**encoding, use_cache=False)


@torch.no_grad()
def teacher_forced_batch(
    model,
    tokenizer,
    blocks,
    pairs,
    target,
    layer,
    position,
    probes,
    periods,
    alpha,
    use_chat_template,
):
    prompts, source_answers, base_answers = [], [], []
    source_spans, base_spans = [], []
    for pair in pairs:
        prompts.append(format_prompt(tokenizer, pair["base"], use_chat_template))
        base_answer, source_answer, base_span, source_span = target_answers(
            pair["base"], pair["source"], target
        )
        base_answers.append(base_answer)
        source_answers.append(source_answer)
        base_spans.append(base_span)
        source_spans.append(source_span)

    all_prompts = prompts + prompts
    answers = source_answers + base_answers
    spans = source_spans + base_spans
    encoding, full_positions, variable_positions = tokenize_answers(
        tokenizer, all_prompts, answers, spans, model.device
    )
    values = steering_values(pairs, target)
    outputs = steered_forward(
        model,
        encoding,
        blocks,
        tokenizer,
        all_prompts,
        values + values,
        layer,
        position,
        probes,
        periods,
        alpha,
    )
    n = len(pairs)
    source_variable, variable_exact = sequence_scores(
        outputs.logits[:n], encoding["input_ids"][:n], variable_positions[:n]
    )
    source_full, full_exact = sequence_scores(
        outputs.logits[:n], encoding["input_ids"][:n], full_positions[:n]
    )
    base_variable, _ = sequence_scores(
        outputs.logits[n:], encoding["input_ids"][n:], variable_positions[n:]
    )
    base_full, _ = sequence_scores(
        outputs.logits[n:], encoding["input_ids"][n:], full_positions[n:]
    )
    return {
        "source_variable_logprob": source_variable.cpu(),
        "source_answer_logprob": source_full.cpu(),
        "base_variable_logprob": base_variable.cpu(),
        "base_answer_logprob": base_full.cpu(),
        "variable_teacher_forced_iia": variable_exact.cpu(),
        "full_answer_teacher_forced_iia": full_exact.cpu(),
    }


def concatenate(parts):
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}


@torch.no_grad()
def teacher_forced_metrics(
    model,
    tokenizer,
    blocks,
    pairs,
    target,
    layer,
    position,
    probes,
    periods,
    alpha,
    use_chat_template,
    batch_size,
):
    parts = []
    for start in range(0, len(pairs), batch_size):
        parts.append(
            teacher_forced_batch(
                model,
                tokenizer,
                blocks,
                pairs[start : start + batch_size],
                target,
                layer,
                position,
                probes,
                periods,
                alpha,
                use_chat_template,
            )
        )
    return concatenate(parts)


@torch.no_grad()
def autoregressive_iia(
    model,
    tokenizer,
    blocks,
    pairs,
    target,
    layer,
    position,
    probes,
    periods,
    alpha,
    use_chat_template,
    max_new_tokens,
):
    correct = 0
    for pair in tqdm(pairs, desc=f"autoregressive alpha={alpha}"):
        prompt = format_prompt(tokenizer, pair["base"], use_chat_template)
        expected = target_answers(pair["base"], pair["source"], target)[1]
        steering_target = steering_values([pair], target)
        resolved_position = resolve_position(tokenizer, prompt, position)
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        generated = []
        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(input_ids)
            outputs = steered_forward(
                model,
                {"input_ids": input_ids, "attention_mask": attention_mask},
                blocks,
                tokenizer,
                [prompt],
                steering_target,
                layer,
                position,
                probes,
                periods,
                alpha,
                resolved_positions=[resolved_position],
            )
            next_id = outputs.logits[0, -1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs)


def summarize(alpha, steered, clean, autoregressive, clean_autoregressive):
    return {
        "row_type": "summary",
        "alpha": alpha,
        "variable_teacher_forced_iia": float(
            steered["variable_teacher_forced_iia"].mean()
        ),
        "full_answer_teacher_forced_iia": float(
            steered["full_answer_teacher_forced_iia"].mean()
        ),
        "autoregressive_iia": autoregressive,
        "source_variable_logprob_gain": float(
            (steered["source_variable_logprob"] - clean["source_variable_logprob"]).mean()
        ),
        "source_answer_logprob_gain": float(
            (steered["source_answer_logprob"] - clean["source_answer_logprob"]).mean()
        ),
        "base_variable_logprob_change": float(
            (steered["base_variable_logprob"] - clean["base_variable_logprob"]).mean()
        ),
        "base_answer_logprob_change": float(
            (steered["base_answer_logprob"] - clean["base_answer_logprob"]).mean()
        ),
        "clean_counterfactual_variable_teacher_forced_iia": float(
            clean["variable_teacher_forced_iia"].mean()
        ),
        "clean_counterfactual_full_answer_teacher_forced_iia": float(
            clean["full_answer_teacher_forced_iia"].mean()
        ),
        "clean_counterfactual_autoregressive_iia": clean_autoregressive,
    }


def plot_metrics(rows, output_path):
    rows = sorted(rows, key=lambda row: row["alpha"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    metrics = (
        ("variable_teacher_forced_iia", "Variable IIA\n(teacher-forced)"),
        ("full_answer_teacher_forced_iia", "Full-answer IIA\n(teacher-forced)"),
        ("autoregressive_iia", "Autoregressive IIA"),
    )
    for ax, (metric, title) in zip(axes, metrics):
        ax.plot(
            [row["alpha"] for row in rows],
            [row[metric] for row in rows],
            marker="o",
            linewidth=2,
        )
        baseline_name = {
            "variable_teacher_forced_iia": "clean_counterfactual_variable_teacher_forced_iia",
            "full_answer_teacher_forced_iia": "clean_counterfactual_full_answer_teacher_forced_iia",
            "autoregressive_iia": "clean_counterfactual_autoregressive_iia",
        }[metric]
        ax.axhline(rows[0][baseline_name], color="gray", linestyle="--", label="no steering")
        ax.set_title(title)
        ax.set_xlabel("Steering scale alpha")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("IIA")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def file_fingerprint(path):
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def main():
    args = parse_args()
    args.use_chat_template = args.use_chat_template or uses_chat_template(args.model)
    samples = load_jsonl(args.data_path)
    data_fingerprint = file_fingerprint(args.data_path)
    pairs, pair_stats = build_unique_pairs(
        samples, args.target, args.seed, args.max_pairs
    )
    if not pairs:
        raise ValueError("No valid steering pairs were found.")
    print("Pair statistics:", pair_stats)

    stem = (
        f"{args.modality}_{args.target}_layer{args.layer}_pos{args.position}_"
        f"periods{'-'.join(map(str, args.periods))}_{args.method}_seed{args.seed}"
    )
    output_path = args.output_dir / f"{stem}.jsonl"
    plot_path = args.output_dir / "plots" / f"{stem}_iia.png"
    if output_path.exists():
        existing = load_jsonl(output_path)
        if (
            {float(row["alpha"]) for row in existing} == set(args.alphas)
            and all(row.get("data_fingerprint") == data_fingerprint for row in existing)
        ):
            print(f"All requested steering rows already exist: {output_path}")
            plot_metrics(existing, plot_path)
            return

    model_path, model_name = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    probes, probe_paths = load_probe_grid(
        args.probe_root,
        args.modality,
        args.target,
        args.periods,
        [args.layer],
        [args.position],
        args.method,
    )

    common = dict(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        pairs=pairs,
        target=args.target,
        layer=args.layer,
        position=args.position,
        probes=probes,
        periods=args.periods,
        use_chat_template=args.use_chat_template,
    )
    clean = teacher_forced_metrics(
        alpha=0,
        batch_size=args.batch_size,
        **common,
    )
    clean_autoregressive = autoregressive_iia(
        alpha=0,
        max_new_tokens=args.max_new_tokens,
        **common,
    )
    rows = []
    for alpha in args.alphas:
        steered = teacher_forced_metrics(
            alpha=alpha,
            batch_size=args.batch_size,
            **common,
        )
        autoregressive = autoregressive_iia(
            alpha=alpha,
            max_new_tokens=args.max_new_tokens,
            **common,
        )
        row = summarize(
            alpha, steered, clean, autoregressive, clean_autoregressive
        )
        row.update(
            {
                "model": model_name,
                "modality": args.modality,
                "target": args.target,
                "target_component": TARGET_COMPONENTS[args.target],
                "layer": args.layer,
                "position": args.position,
                "periods": args.periods,
                "method": args.method,
                "n_pairs": len(pairs),
                "seed": args.seed,
                "data_path": str(args.data_path),
                "data_fingerprint": data_fingerprint,
                "dataset_statistics": dataset_statistics(samples),
                "pair_statistics": pair_stats,
                "probe_paths": [
                    str(path)
                    for path in probe_paths[args.layer][args.position]
                ],
            }
        )
        rows.append(row)
        print(
            f"alpha={alpha}: variable={row['variable_teacher_forced_iia']:.3f} "
            f"full={row['full_answer_teacher_forced_iia']:.3f} "
            f"autoregressive={row['autoregressive_iia']:.3f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, output_path)
    plot_metrics(rows, plot_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
