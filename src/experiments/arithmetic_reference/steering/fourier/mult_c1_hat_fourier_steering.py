import argparse
import random
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.common.io import load_jsonl, save_jsonl
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.probes.fourier import (
    batches,
    load_probe_grid,
    teacher_forced_logprobs,
)
from src.models import load_hf_model, resolve_model_for_loading, validate_block_layers
from visualizations.common.fourier_steering import plot_c1_hat_alpha


def build_pairs(samples, n_pairs, seed):
    """Hold result mod 100 fixed while changing the c1_hat hundreds component."""
    samples = [sample for sample in samples if int(sample["c1_hat"]) // 10 > 0]
    groups = defaultdict(list)
    for sample in samples:
        groups[int(sample["c1_hat"]) % 10].append(sample)

    valid = [
        sample
        for sample in samples
        if any(
            int(source["c1_hat"]) // 10 != int(sample["c1_hat"]) // 10
            for source in groups[int(sample["c1_hat"]) % 10]
        )
    ]
    rng = random.Random(seed)
    bases = rng.sample(valid, min(n_pairs, len(valid)))
    pairs = []
    for base in bases:
        sources = [
            source
            for source in groups[int(base["c1_hat"]) % 10]
            if int(source["c1_hat"]) // 10 != int(base["c1_hat"]) // 10
        ]
        pairs.append({"base": base, "source": rng.choice(sources)})
    return pairs


def compatible_answer(base, c1_hat):
    # Multiplication decomposition: result = c0 + 10 * c1_hat.
    return str(int(base["c0"]) + 10 * int(c1_hat))


def summarize(rows):
    summaries = []
    for alpha in sorted({row["alpha"] for row in rows}):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        summaries.append(
            {
                "row_type": "summary",
                "alpha": alpha,
                "n_pairs": len(alpha_rows),
                "mean_source_hundreds_contrast": sum(
                    row["source_hundreds_contrast"] for row in alpha_rows
                ) / len(alpha_rows),
                "mean_source_hundreds_contrast_gain": sum(
                    row["source_hundreds_contrast_gain"] for row in alpha_rows
                ) / len(alpha_rows),
                "mean_source_hundreds_logprob_gain": sum(
                    row["source_hundreds_logprob_gain"] for row in alpha_rows
                ) / len(alpha_rows),
                "mean_source_compatible_answer_logprob_gain": sum(
                    row["source_compatible_answer_logprob_gain"]
                    for row in alpha_rows
                ) / len(alpha_rows),
            }
        )
    return summaries


def output_paths(root, layers, positions, periods, method, alphas):
    config = (
        f"c1_hat_layers{'-'.join(map(str, layers))}"
        f"_pos{'-'.join(map(str, positions))}_{method}"
    )
    folder = Path(root) / "multiplication" / "fourier_steering" / config
    stem = (
        f"periods_{'-'.join(map(str, periods))}"
        f"_alphas_{'-'.join(map(str, alphas))}"
    )
    return folder / f"{stem}.jsonl", folder / "plots" / f"{stem}.png"


def score(model, tokenizer, samples, answers, common, **kwargs):
    return teacher_forced_logprobs(
        model, tokenizer, samples, answers, **common, **kwargs
    )


def run(args, model, tokenizer, model_name):
    layers = validate_block_layers(model, args.layers)
    samples = load_jsonl(args.data_dir / "multiplication_baseline.jsonl")
    pairs = build_pairs(samples, args.n_pairs, args.seed)
    if not pairs:
        raise ValueError("No controlled c1_hat pairs were found.")

    probes, probe_paths = load_probe_grid(
        args.probe_root,
        "multiplication",
        "c1_hat",
        args.periods,
        layers,
        args.positions,
        args.method,
    )
    output_path, plot_path = output_paths(
        args.output_root,
        layers,
        args.positions,
        args.periods,
        args.method,
        args.alphas,
    )
    common = {
        "layers": layers,
        "positions": args.positions,
        "use_chat_template": args.use_chat_template,
        "probes": probes,
        "periods": args.periods,
    }

    rows = []
    for pair_batch in tqdm(list(batches(pairs, args.batch_size)), desc="c1_hat"):
        base_samples = [pair["base"] for pair in pair_batch]
        base_answers = [
            compatible_answer(pair["base"], pair["base"]["c1_hat"])
            for pair in pair_batch
        ]
        source_answers = [
            compatible_answer(pair["base"], pair["source"]["c1_hat"])
            for pair in pair_batch
        ]
        source_targets = [int(pair["source"]["c1_hat"]) for pair in pair_batch]

        baseline_base_hundreds = score(
            model, tokenizer, base_samples, base_answers, common,
            steering_targets=None, alpha=0, prefix_chars=1,
        )
        baseline_source_hundreds = score(
            model, tokenizer, base_samples, source_answers, common,
            steering_targets=None, alpha=0, prefix_chars=1,
        )
        baseline_source_answer = score(
            model, tokenizer, base_samples, source_answers, common,
            steering_targets=None, alpha=0,
        )

        for alpha in args.alphas:
            steering = {"steering_targets": source_targets, "alpha": alpha}
            steered_base_hundreds = score(
                model, tokenizer, base_samples, base_answers, common,
                prefix_chars=1, **steering,
            )
            steered_source_hundreds = score(
                model, tokenizer, base_samples, source_answers, common,
                prefix_chars=1, **steering,
            )
            steered_source_answer = score(
                model, tokenizer, base_samples, source_answers, common, **steering
            )

            for index, pair in enumerate(pair_batch):
                baseline_contrast = float(
                    baseline_source_hundreds[index] - baseline_base_hundreds[index]
                )
                contrast = float(
                    steered_source_hundreds[index] - steered_base_hundreds[index]
                )
                rows.append(
                    {
                        "row_type": "example",
                        "model": model_name,
                        "base_sample_id": pair["base"].get("sample_id"),
                        "source_sample_id": pair["source"].get("sample_id"),
                        "base_expr": pair["base"]["expr"],
                        "base_result": int(pair["base"]["result"]),
                        "base_c1_hat": int(pair["base"]["c1_hat"]),
                        "source_c1_hat": int(pair["source"]["c1_hat"]),
                        "controlled_result_mod100": int(pair["base"]["result"]) % 100,
                        "base_hundreds": int(pair["base"]["c1_hat"]) // 10,
                        "source_hundreds": int(pair["source"]["c1_hat"]) // 10,
                        "base_compatible_answer": base_answers[index],
                        "source_compatible_answer": source_answers[index],
                        "alpha": alpha,
                        "baseline_source_hundreds_contrast": baseline_contrast,
                        "source_hundreds_contrast": contrast,
                        "source_hundreds_contrast_gain": contrast - baseline_contrast,
                        "source_hundreds_logprob_gain": float(
                            steered_source_hundreds[index]
                            - baseline_source_hundreds[index]
                        ),
                        "source_compatible_answer_logprob_gain": float(
                            steered_source_answer[index]
                            - baseline_source_answer[index]
                        ),
                    }
                )

    summary_rows = summarize(rows)
    metadata = {
        "steering_variable": "c1_hat",
        "pairing_control": "same_c1_hat_mod10_different_c1_hat_div10",
        "layers": layers,
        "positions": args.positions,
        "periods": args.periods,
        "alphas": args.alphas,
        "method": args.method,
        "probe_paths": {
            str(layer): {
                str(position): [str(path) for path in paths]
                for position, paths in positions.items()
            }
            for layer, positions in probe_paths.items()
        },
    }
    for row in rows + summary_rows:
        row.update(metadata)
    save_jsonl(rows + summary_rows, output_path)
    print(f"Saved: {output_path}")
    for row in summary_rows:
        print(
            f"alpha={row['alpha']}: "
            f"source_hundreds_contrast={row['mean_source_hundreds_contrast']:.3f}, "
            f"gain={row['mean_source_hundreds_contrast_gain']:.3f}"
        )
    plot_c1_hat_alpha(summary_rows, plot_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--periods", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    parser.add_argument("--method", choices=["gd", "ridge"], default="ridge")
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_chat_template", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.use_chat_template = args.use_chat_template or uses_chat_template(args.model)
    model_path, model_name = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    model.eval()
    run(args, model, tokenizer, model_name)


if __name__ == "__main__":
    main()
