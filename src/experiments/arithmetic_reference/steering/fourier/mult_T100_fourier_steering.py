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
from visualizations.common.fourier_steering import (
    plot_gain_matrices,
    plot_mul_alpha,
)


def build_pairs(samples, n_pairs, seed):
    groups = defaultdict(list)
    for sample in samples:
        groups[len(str(int(sample["result"])))].append(sample)

    valid = [
        sample
        for sample in samples
        if any(
            int(source["result"]) % 100 != int(sample["result"]) % 100
            for source in groups[len(str(int(sample["result"])))]
        )
    ]
    rng = random.Random(seed)
    bases = rng.sample(valid, min(n_pairs, len(valid)))
    pairs = []
    for base in bases:
        candidates = [
            source
            for source in groups[len(str(int(base["result"])))]
            if int(source["result"]) % 100 != int(base["result"]) % 100
        ]
        pairs.append({"base": base, "source": rng.choice(candidates)})
    return pairs


# Replace base result's last two digits
def answer_with_suffix(result, suffix):
    text = str(int(result))
    if len(text) <= 2:
        return str(int(suffix)).zfill(len(text))
    return text[:-2] + f"{int(suffix):02d}"


def output_paths(root, layers, positions, method, alphas):
    config = (
        f"result_T100_layers{'-'.join(map(str, layers))}"
        f"_pos{'-'.join(map(str, positions))}_{method}"
    )
    folder = Path(root) / "multiplication" / "fourier_steering" / config
    stem = f"alphas_{'-'.join(map(str, alphas))}"
    return folder / f"{stem}.jsonl", folder / "plots" / stem


def summarize(rows, matrix_suffixes):
    summaries = []
    for alpha in sorted({row["alpha"] for row in rows}):
        alpha_rows = [row for row in rows if row["alpha"] == alpha]
        summaries.append(
            {
                "row_type": "summary",
                "alpha": alpha,
                "n_pairs": len(alpha_rows),
                "mean_source_mod100_contrast": sum(
                    row["source_mod100_contrast"] for row in alpha_rows
                ) / len(alpha_rows),
                "mean_source_mod100_contrast_gain": sum(
                    row["source_mod100_contrast_gain"] for row in alpha_rows
                ) / len(alpha_rows),
                "mean_source_suffix_logprob_gain": sum(
                    row["source_suffix_logprob_gain"] for row in alpha_rows
                ) / len(alpha_rows),
                "mean_full_source_answer_logprob_gain": sum(
                    row["full_source_answer_logprob_gain"] for row in alpha_rows
                ) / len(alpha_rows),
            }
        )

        if matrix_suffixes:
            targets = sorted({row["source_suffix"] for row in alpha_rows})
            for target in targets:
                target_rows = [
                    row for row in alpha_rows if row["source_suffix"] == target
                ]
                summaries.append(
                    {
                        "row_type": "matrix",
                        "alpha": alpha,
                        "steering_target": target,
                        "n_examples": len(target_rows),
                        "mean_candidate_logprob_gains": {
                            str(suffix): sum(
                                row["candidate_suffix_logprob_gains"][str(suffix)]
                                for row in target_rows
                            ) / len(target_rows)
                            for suffix in matrix_suffixes
                        },
                    }
                )
    return summaries


def score_answers(model, tokenizer, samples, answers, common, **kwargs):
    return teacher_forced_logprobs(
        model, tokenizer, samples, answers, **common, **kwargs
    )


def score_suffix_grid(model, tokenizer, pairs, suffixes, common, **kwargs):
    samples = [pair["base"] for pair in pairs]
    return {
        suffix: score_answers(
            model,
            tokenizer,
            samples,
            [answer_with_suffix(pair["base"]["result"], suffix) for pair in pairs],
            common,
            suffix_chars=2,
            **kwargs,
        )
        for suffix in suffixes
    }


def run(args, model, tokenizer, model_name):
    layers = validate_block_layers(model, args.layers)
    samples = load_jsonl(args.data_dir / "multiplication_baseline.jsonl")
    if args.max_operand is not None:
        samples = [
            sample for sample in samples
            if int(sample["a"]) <= args.max_operand
            and int(sample["b"]) <= args.max_operand
        ]
    pairs = build_pairs(samples, args.n_pairs, args.seed)
    if not pairs:
        raise ValueError("No multiplication pairs with different mod-100 results found.")

    # We just use result (T=100) as we foudn evidence that this could be vital for multiplication
    probes, probe_paths = load_probe_grid(
        args.probe_root,
        "multiplication",
        "result",
        [100],
        layers,
        args.positions,
        args.method,
    )
    output_path, plot_stem = output_paths(
        args.output_root, layers, args.positions, args.method, args.alphas
    )
    matrix_suffixes = sorted(set(args.matrix_suffixes)) if args.plot_matrix else []
    common = {
        "layers": layers,
        "positions": args.positions,
        "use_chat_template": args.use_chat_template,
        "probes": probes,
        "periods": [100],
    }

    rows = []
    for pair_batch in tqdm(list(batches(pairs, args.batch_size)), desc="multiplication"):
        base_samples = [pair["base"] for pair in pair_batch]
        base_answers = [str(int(pair["base"]["result"])) for pair in pair_batch]
        source_answers = [str(int(pair["source"]["result"])) for pair in pair_batch]
        source_suffixes = [int(pair["source"]["result"]) % 100 for pair in pair_batch]
        suffix_answers = [
            answer_with_suffix(pair["base"]["result"], suffix)
            for pair, suffix in zip(pair_batch, source_suffixes)
        ]

        baseline_base_suffix = score_answers(
            model, tokenizer, base_samples, base_answers, common,
            steering_targets=None, alpha=0, suffix_chars=2,
        )
        baseline_source_suffix = score_answers(
            model, tokenizer, base_samples, suffix_answers, common,
            steering_targets=None, alpha=0, suffix_chars=2,
        )
        baseline_full_source = score_answers(
            model, tokenizer, base_samples, source_answers, common,
            steering_targets=None, alpha=0,
        )
        baseline_grid = (
            score_suffix_grid(
                model, tokenizer, pair_batch, matrix_suffixes, common,
                steering_targets=None, alpha=0,
            )
            if matrix_suffixes else {}
        )

        for alpha in args.alphas:
            steering = {
                "steering_targets": source_suffixes,
                "alpha": alpha,
            }
            steered_base_suffix = score_answers(
                model, tokenizer, base_samples, base_answers, common,
                suffix_chars=2, **steering,
            )
            steered_source_suffix = score_answers(
                model, tokenizer, base_samples, suffix_answers, common,
                suffix_chars=2, **steering,
            )
            steered_full_source = score_answers(
                model, tokenizer, base_samples, source_answers, common, **steering
            )
            steered_grid = (
                score_suffix_grid(
                    model, tokenizer, pair_batch, matrix_suffixes, common, **steering
                )
                if matrix_suffixes else {}
            )

            for index, pair in enumerate(pair_batch):
                baseline_contrast = float(
                    baseline_source_suffix[index] - baseline_base_suffix[index]
                )
                contrast = float(
                    steered_source_suffix[index] - steered_base_suffix[index]
                )
                row = {
                    "row_type": "example",
                    "model": model_name,
                    "base_sample_id": pair["base"].get("sample_id"),
                    "source_sample_id": pair["source"].get("sample_id"),
                    "base_expr": pair["base"]["expr"],
                    "base_result": int(pair["base"]["result"]),
                    "source_result": int(pair["source"]["result"]),
                    "base_suffix": int(pair["base"]["result"]) % 100,
                    "source_suffix": source_suffixes[index],
                    "suffix_candidate_answer": suffix_answers[index],
                    "alpha": alpha,
                    "baseline_source_mod100_contrast": baseline_contrast,
                    "source_mod100_contrast": contrast,
                    "source_mod100_contrast_gain": contrast - baseline_contrast,
                    "source_suffix_logprob_gain": float(
                        steered_source_suffix[index] - baseline_source_suffix[index]
                    ),
                    "full_source_answer_logprob_gain": float(
                        steered_full_source[index] - baseline_full_source[index]
                    ),
                }
                if matrix_suffixes:
                    row["candidate_suffix_logprob_gains"] = {
                        str(suffix): float(
                            steered_grid[suffix][index] - baseline_grid[suffix][index]
                        )
                        for suffix in matrix_suffixes
                    }
                rows.append(row)

    summary_rows = summarize(rows, matrix_suffixes)
    metadata = {
        "steering_variable": "result_T100",
        "layers": layers,
        "positions": args.positions,
        "periods": [100],
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

    primary_summaries = [row for row in summary_rows if row["row_type"] == "summary"]
    for row in primary_summaries:
        print(
            f"alpha={row['alpha']}: "
            f"source_mod100_contrast={row['mean_source_mod100_contrast']:.3f}, "
            f"gain={row['mean_source_mod100_contrast_gain']:.3f}, "
            f"full_source_gain={row['mean_full_source_answer_logprob_gain']:.3f}"
        )
    plot_mul_alpha(
        primary_summaries, plot_stem.with_name(plot_stem.name + "_mod100.png")
    )
    if matrix_suffixes:
        plot_gain_matrices(
            [row for row in summary_rows if row["row_type"] == "matrix"],
            matrix_suffixes,
            plot_stem.parent,
            plot_stem.name + "_suffix",
            row_label="Source mod-100 target",
            column_label="Output suffix",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--probe_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    parser.add_argument("--method", choices=["gd", "ridge"], default="ridge")
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_operand", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--plot_matrix", action="store_true")
    parser.add_argument("--matrix_suffixes", type=int, nargs="+", default=list(range(100)))
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
