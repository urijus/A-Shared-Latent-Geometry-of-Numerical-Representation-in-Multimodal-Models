import argparse
import random
import pandas as pd
from pathlib import Path

from src.common.io import save_jsonl
from src.common.seed import set_seed
from src.data.simple_addition.addition_mapping import addition_expr_to_words, addition_tokens_to_words



def pairs_to_samples(pairs, use_digits=True, prompt="Output ONLY a number."):
    samples = []

    for sample_id, (a, b) in enumerate(pairs):
        if use_digits:
            if prompt:
                expr = f"{prompt} {a} + {b} ="
            else:
                expr = f"{a} + {b} ="

            tokens = [str(a), "+", str(b), "="]
        else:
            expr = addition_expr_to_words(a, b, prompt)
            tokens = addition_tokens_to_words(a, b)


        samples.append({
            "sample_id": sample_id,
            "expr": expr,
            "tokens": tokens,
            "a": a,
            "b": b,
            "sum": a + b,
            "a_mod_2": a % 2,
            "a_mod_5": a % 5,
            "a_mod_10": a % 10,
            "b_mod_2": b % 2,
            "b_mod_5": b % 5,
            "b_mod_10": b % 10,
            "sum_mod_2": (a + b) % 2,
            "sum_mod_5": (a + b) % 5,
            "sum_mod_10": (a + b) % 10,
        })

    return samples


def make_operand_balanced_addition(
        min_value,
        max_value,
        n_samples = None,
        output_dir = None,
        use_digits=True,
        prompt="Output ONLY a number."):

    all_pairs = [
        (a,b)
        for a in range(min_value, max_value+1)
        for b in range(min_value, max_value+1)
    ]

    random.shuffle(all_pairs)

    if n_samples is not None:
        if n_samples > len(all_pairs):
            raise ValueError(f"n_samples={n_samples} is larger than the total number of possible pairs ({len(all_pairs)})")

        all_pairs = all_pairs[:n_samples]

    samples = pairs_to_samples(all_pairs, use_digits, prompt)

    if output_dir:
        save_jsonl(samples, Path(output_dir) / "addition_balanced_operands.jsonl")

    return pd.DataFrame(samples)


def make_result_balanced_addition(
        min_value,
        max_value,
        min_sum = 40,
        max_sum = 158,
        examples_per_sum = 40,
        output_dir = None,
        use_digits=True,
        prompt="Output ONLY a number."):

    all_sampled_pairs = []

    min_possible_sum = min_value + min_value
    max_possible_sum = max_value + max_value

    if min_sum < min_possible_sum or max_sum > max_possible_sum:
        raise ValueError(
            f"Invalid sum range [{min_sum}, {max_sum}] for operands in "
            f"[{min_value}, {max_value}]. Possible sums are "
            f"[{min_possible_sum}, {max_possible_sum}]."
        )

    for target_sum in range(min_sum, max_sum+1):
        possible_pairs = []

        for a in range(min_value, max_value+1):
            b = target_sum - a

            if min_value <= b <= max_value:
                possible_pairs.append((a, b))

        if len(possible_pairs) < examples_per_sum:
            raise ValueError(
                f"Cannot sample {examples_per_sum} examples for sum={target_sum}. "
                f"Only {len(possible_pairs)} pairs are available. "
                f"Choose a smaller examples_per_sum or a narrower sum range."
            )

        sampled_pairs = random.sample(possible_pairs, examples_per_sum)
        all_sampled_pairs.extend(sampled_pairs)

    random.shuffle(all_sampled_pairs)
    samples = pairs_to_samples(all_sampled_pairs, use_digits, prompt)

    if output_dir:
        save_jsonl(samples, Path(output_dir) / "addition_balanced_sums.jsonl")

    return pd.DataFrame(samples)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_value", type=int, default=0)
    parser.add_argument("--max_value", type=int, default=99)
    parser.add_argument("--min_sum", type=int, default=45)
    parser.add_argument("--max_sum", type=int, default=158)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="dataset/simple_addition")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--examples_per_sum", type=int, default=40)
    parser.add_argument("--use_words", action="store_true")
    parser.add_argument("--prompt", type=str, default="Output ONLY a number.")

    return parser.parse_args()


def print_dataset_stats(df):
    print(f"Number of samples: {len(df)}")
    print(f"a range: {df['a'].min()} to {df['a'].max()}")
    print(f"b range: {df['b'].min()} to {df['b'].max()}")
    print(f"sum range: {df['sum'].min()} to {df['sum'].max()}")

    print("\na counts:")
    print(f"min: {df['a'].value_counts().min()}, max: {df['a'].value_counts().max()}")

    print("\nb counts:")
    print(f"min: {df['b'].value_counts().min()}, max: {df['b'].value_counts().max()}")

    print("\nsum counts:")
    print(f"min: {df['sum'].value_counts().min()}, max: {df['sum'].value_counts().max()}")

    n_duplicates = df.duplicated(subset=["a", "b"]).sum()
    print(f"\nDuplicate (a, b) pairs: {n_duplicates}")


def generate_addition_datasets():
    args = parse_args()
    set_seed(args.seed)

    use_digits = not args.use_words

    balanced_operand_df = make_operand_balanced_addition(
        min_value=args.min_value,
        max_value=args.max_value,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        use_digits=use_digits,
        prompt=args.prompt
    )

    print_dataset_stats(balanced_operand_df)

    balanced_result_df = make_result_balanced_addition(
        min_value=args.min_value,
        max_value=args.max_value,
        min_sum=args.min_sum,
        max_sum=args.max_sum,
        examples_per_sum=args.examples_per_sum,
        output_dir=args.output_dir,
        use_digits=use_digits,
        prompt=args.prompt
    )

    print_dataset_stats(balanced_result_df)

    print("Done.")



if __name__=="__main__":
    generate_addition_datasets()
