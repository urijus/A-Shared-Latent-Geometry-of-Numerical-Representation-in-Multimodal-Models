"""GoodFire-style Fourier steering matrix for baseline arithmetic prompts.

For every steering target n, this script steers the selected hidden state so
the Fourier probes encode n. It then scores candidate answers and averages the
candidate distribution into a matrix:

    rows    = steered Fourier target values
    columns = candidate output values

A successful intervention should put high probability near the diagonal.
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from src.common.io import load_jsonl, save_jsonl
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    uses_chat_template,
)
from src.probes.fourier import (
    batches,
    load_probe_grid,
    score_candidates,
)
from src.models import load_hf_model, resolve_model_for_loading, validate_block_layers
from visualizations.common.fourier_steering import plot_matrix


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--plot_path", type=Path, default=None)
    parser.add_argument("--modality", default="addition")
    parser.add_argument("--target", default="result")
    parser.add_argument("--periods", type=int, nargs="+", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--method", choices=["gd", "ridge"], default="ridge")
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--max_operand", type=int, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--output_values",
        type=int,
        nargs="+",
        default=list(range(21)),
        help="Candidate answers to score as matrix columns.",
    )
    parser.add_argument(
        "--steering_targets",
        type=int,
        nargs="+",
        default=list(range(21)),
        help="Fourier target values to steer toward as matrix rows.",
    )
    return parser.parse_args()


def select_samples(samples, max_samples, max_operand):
    """Keep the prompt distribution simple and deterministic for comparisons."""
    selected = []
    for sample in samples:
        if max_operand is not None and (
            int(sample["a"]) > max_operand or int(sample["b"]) > max_operand
        ):
            continue
        selected.append(sample)
        if max_samples > 0 and len(selected) >= max_samples:
            break
    if not selected:
        raise ValueError("No samples left after filtering.")
    return selected


def candidate_distribution(logprobs):
    """Turn candidate sequence logprobs into a normalized candidate distribution."""
    return torch.softmax(logprobs.float(), dim=1)


def matrix_row_from_batches(prob_parts):
    probs = torch.cat(prob_parts, dim=0)
    return probs.mean(dim=0)


def best_value(probabilities, output_values):
    index = int(probabilities.argmax().item())
    return int(output_values[index]), float(probabilities[index])


@torch.no_grad()
def run_matrix(args, model, tokenizer, model_name, probes, probe_paths, samples):
    common = {
        "layers": [args.layer],
        "positions": [args.position],
        "use_chat_template": args.use_chat_template,
        "probes": probes,
        "periods": args.periods,
    }
    candidates = [str(value) for value in args.output_values]
    rows = []
    matrix_rows = []

    for steering_target in tqdm(args.steering_targets, desc="steering targets"):
        prob_parts = []
        for batch in batches(samples, args.batch_size):
            # Every prompt in this batch is steered toward the same target value.
            logprobs = score_candidates(
                model=model,
                tokenizer=tokenizer,
                samples=batch,
                candidates=candidates,
                steering_targets=[steering_target] * len(batch),
                alpha=args.alpha,
                **common,
            )
            probs = candidate_distribution(logprobs)
            prob_parts.append(probs)

            for sample, sample_probs, sample_logprobs in zip(batch, probs, logprobs):
                predicted_value, predicted_prob = best_value(
                    sample_probs, args.output_values
                )
                rows.append(
                    {
                        "row_type": "example",
                        "model": model_name,
                        "expr": sample["expr"],
                        "a": int(sample["a"]),
                        "b": int(sample["b"]),
                        "gold_result": int(sample[args.target]),
                        "steering_target": int(steering_target),
                        "predicted_candidate": predicted_value,
                        "predicted_candidate_probability": predicted_prob,
                        "target_candidate_probability": float(
                            sample_probs[args.output_values.index(steering_target)]
                        )
                        if steering_target in args.output_values
                        else None,
                        "candidate_logprobs": {
                            str(value): float(logprob)
                            for value, logprob in zip(
                                args.output_values, sample_logprobs
                            )
                        },
                        "candidate_probabilities": {
                            str(value): float(prob)
                            for value, prob in zip(args.output_values, sample_probs)
                        },
                    }
                )

        avg_probs = matrix_row_from_batches(prob_parts)
        predicted_value, predicted_prob = best_value(avg_probs, args.output_values)
        target_prob = (
            float(avg_probs[args.output_values.index(steering_target)])
            if steering_target in args.output_values
            else None
        )
        matrix_rows.append(
            {
                "row_type": "matrix",
                "steering_target": int(steering_target),
                "avg_steered_probs": {
                    str(value): float(prob)
                    for value, prob in zip(args.output_values, avg_probs)
                },
                "predicted_candidate": predicted_value,
                "predicted_candidate_probability": predicted_prob,
                "target_candidate_probability": target_prob,
                "n_examples": len(samples),
            }
        )

    diagonal_hits = [
        row["predicted_candidate"] == row["steering_target"]
        for row in matrix_rows
        if row["steering_target"] in args.output_values
    ]
    summary = {
        "row_type": "summary",
        "model": model_name,
        "modality": args.modality,
        "target": args.target,
        "layer": args.layer,
        "position": args.position,
        "periods": args.periods,
        "alpha": args.alpha,
        "method": args.method,
        "n_prompts": len(samples),
        "output_values": args.output_values,
        "steering_targets": args.steering_targets,
        "matrix_top1_diagonal_accuracy": sum(diagonal_hits) / len(diagonal_hits)
        if diagonal_hits
        else None,
        "probability_metric": "normalized_candidate_sequence_probability",
        "probe_paths": [
            str(path) for path in probe_paths[args.layer][args.position]
        ],
    }

    for row in rows + matrix_rows:
        row.update(
            {
                "modality": args.modality,
                "target": args.target,
                "layer": args.layer,
                "position": args.position,
                "periods": args.periods,
                "alpha": args.alpha,
                "method": args.method,
                "probability_metric": summary["probability_metric"],
            }
        )

    return rows + matrix_rows + [summary], summary


def main():
    args = parse_args()
    args.use_chat_template = args.use_chat_template or uses_chat_template(args.model)

    samples = select_samples(
        load_jsonl(args.data_path),
        max_samples=args.max_samples,
        max_operand=args.max_operand,
    )
    print(f"Using {len(samples)} prompts from {args.data_path}")

    model_path, model_name = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    model.eval()
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

    rows, summary = run_matrix(
        args=args,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        probes=probes,
        probe_paths=probe_paths,
        samples=samples,
    )

    save_jsonl(rows, args.output_path)
    plot_matrix(rows, args.output_values, args.plot_path)

    print("Saved:", args.output_path)
    if args.plot_path is not None:
        print("Saved plot:", args.plot_path)
    print(
        "Matrix top-1 diagonal accuracy:",
        summary["matrix_top1_diagonal_accuracy"],
    )


if __name__ == "__main__":
    main()
