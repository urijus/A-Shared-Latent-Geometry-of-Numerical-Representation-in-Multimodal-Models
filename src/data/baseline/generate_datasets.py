import argparse
import random
import pandas as pd
from pathlib import Path

from src.common.io import save_jsonl
from src.common.seed import set_seed
from src.data.simple_addition.addition_mapping import (
    addition_expr_to_words,
    subtraction_expr_to_words,
    multiplication_expr_to_words,
    tokens_to_words
)



def addition_pairs_to_samples(pairs, use_digits=True, prompt="Output ONLY a number."):
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
            tokens = tokens_to_words(a, b, "+")

        samples.append({
            "sample_id": sample_id,
            "task": "baseline_add",
            "operation": "+",
            "expr": expr,
            "tokens": tokens,

            "a": a,
            "b": b,
            "result": a + b,

            "a_mod_2": a % 2,
            "a_mod_5": a % 5,
            "a_mod_10": a % 10,

            "b_mod_2": b % 2,
            "b_mod_5": b % 5,
            "b_mod_10": b % 10,

            "result_mod_2": (a + b) % 2,
            "result_mod_5": (a + b) % 5,
            "result_mod_10": (a + b) % 10,
            "result_mod_20": (a + b) % 20,
            "result_mod_50": (a + b) % 50,
            "result_mod_100": (a + b) % 100,

            "requires_carry": int((a % 10) + (b % 10) >= 10)
    })

    return samples


def subtraction_pairs_to_samples(pairs, use_digits=True, prompt="Output ONLY a number."):
    samples = []

    for sample_id, (a, b) in enumerate(pairs):
        if use_digits:
            if prompt:
                expr = f"{prompt} {a} - {b} ="
            else:
                expr = f"{a} - {b} ="

            tokens = [str(a), "-", str(b), "="]
        else:
            expr = subtraction_expr_to_words(a, b, prompt)
            tokens = tokens_to_words(a, b, "-")

        samples.append({
            "sample_id": sample_id,
            "task": "baseline_sub",
            "operation": "-",
            "expr": expr,
            "tokens": tokens,

            "a": a,
            "b": b,
            "result": a - b,

            "a_mod_2": a % 2,
            "a_mod_5": a % 5,
            "a_mod_10": a % 10,

            "b_mod_2": b % 2,
            "b_mod_5": b % 5,
            "b_mod_10": b % 10,

            "result_mod_2": (a - b) % 2,
            "result_mod_5": (a - b) % 5,
            "result_mod_10": (a - b) % 10,
            "result_mod_20": (a - b) % 20,
            "result_mod_50": (a - b) % 50,
            "result_mod_100": (a - b) % 100,

            "requires_borrow": int((a % 10) < (b % 10)),
    })

    return samples


def multiplication_pairs_to_samples(pairs, use_digits=True, prompt="Output ONLY a number."):
    samples = []

    for sample_id, (a, b) in enumerate(pairs):
        if use_digits:
            if prompt:
                expr = f"{prompt} {a} * {b} ="
            else:
                expr = f"{a} * {b} ="

            tokens = [str(a), "*", str(b), "="]
        else:
            expr = multiplication_expr_to_words(a, b, prompt)
            tokens = tokens_to_words(a, b, "*")

        # Use it for the intermediate computations in multiplication
        a0 = a % 10
        a1 = a // 10
        b0 = b

        samples.append({
            "sample_id": sample_id,
            "task": "baseline_mul",
            "operation": "*",
            "expr": expr,
            "tokens": tokens,

            "a": a,
            "b": b,
            "result": a * b,

            "a_mod_2": a % 2,
            "a_mod_5": a % 5,
            "a_mod_10": a % 10,

            "b_mod_2": b % 2,
            "b_mod_5": b % 5,
            "b_mod_10": b % 10,

            "result_mod_2": (a * b) % 2,
            "result_mod_5": (a * b) % 5,
            "result_mod_10": (a * b) % 10,
            "result_mod_20": (a * b) % 20,
            "result_mod_50": (a * b) % 50,
            "result_mod_100": (a * b) % 100,

            "requires_carry": int((a % 10) * b >= 10),

            "result_ones": (a * b) % 10,
            "result_tens": ((a * b) // 10) % 10,
            "result_hundreds": (a * b) // 100,

            "p0": a0 * b0,
            "p1": a1 * b0,
            "product_tens": (a1 * b0) % 10,
            "product_hundreds": (a1 * b0) // 10,

            "c0_hat": a0 * b0,
            "c0": a0 * b0 % 10,
            "r0": a0 * b0 // 10,

            "c1_hat": a1 * b0 + (a0 * b0 // 10),
            "c1": (a1 * b0 + (a0 * b0 // 10)) % 10,
            "r1": (a1 * b0 + (a0 * b0 // 10)) // 10,

            "c2": (a1 * b0 + (a0 * b0 // 10)) // 10
    })

    return samples


def make_addition_dataset(
        output_dir = None,
        use_digits=True,
        prompt="Output ONLY a number.",
        min_value=0,
        max_value=99,
        max_sum=99):

    all_pairs = [
        (a,b)
        for a in range(min_value, max_value + 1)
        for b in range(min_value, max_value + 1)
        if max_sum is None or a + b <= max_sum
    ]

    random.shuffle(all_pairs)
    samples = addition_pairs_to_samples(all_pairs, use_digits, prompt)

    if output_dir:
        save_jsonl(samples, Path(output_dir) / "addition_baseline.jsonl")

    return pd.DataFrame(samples)


def make_subtraction_dataset(
        output_dir = None,
        use_digits=True,
        prompt="Output ONLY a number."):

    all_pairs = [
        (a,b)
        for a in range(0, 100)
        for b in range(0, 100)
        if a >= b
    ]

    random.shuffle(all_pairs)
    samples = subtraction_pairs_to_samples(all_pairs, use_digits, prompt)

    if output_dir:
        save_jsonl(samples, Path(output_dir) / "subtraction_baseline.jsonl")

    return pd.DataFrame(samples)


def make_multiplication_dataset(
        output_dir = None,
        use_digits=True,
        prompt="Output ONLY a number."):

    all_pairs = [
        (a,b)
        for a in range(0, 100)
        for b in range(0, 10)
    ]

    random.shuffle(all_pairs)
    samples = multiplication_pairs_to_samples(all_pairs, use_digits, prompt)

    if output_dir:
        save_jsonl(samples, Path(output_dir) / "multiplication_baseline.jsonl")

    return pd.DataFrame(samples)


# Just used for testing accuracy
def make_mixed_arithmetic_dataset(output_dir=None, prompt=""):
    examples = [
        ("(2 + 4) * 3 =", 18),
        ("(3 * 4) - 2 * (3 + (2 + 1)) =", 0),
        ("(8 - 3) * (4 + 2) =", 30),
        ("7 * (5 + 3) - 9 =", 47),
        ("2 * (6 + (4 * 3)) =", 36),
        ("(9 * 9) - (7 * 8) =", 25),
        ("(15 - 6) * (2 + 5) =", 63),
        ("4 * (3 * (2 + 1)) - 5 =", 31),
        ("(20 - 3) * (2 + 2) =", 68),
        ("(6 + 7) * (8 - 5) =", 39),
    ]
    samples = [
        {
            "sample_id": i,
            "task": "mixed_arithmetic",
            "expr": f"{prompt} {expr}" if prompt else expr,
            "result": result,
        }
        for i, (expr, result) in enumerate(examples)
    ]
    if output_dir:
        save_jsonl(samples, Path(output_dir) / "mixed_arithmetic_baseline.jsonl")
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="dataset/baseline")
    parser.add_argument("--use_words", action="store_true")
    parser.add_argument("--prompt", type=str, default="Output ONLY a number.")
    parser.add_argument("--addition_min", type=int, default=0)
    parser.add_argument("--addition_max", type=int, default=99)
    parser.add_argument(
        "--addition_max_sum",
        type=int,
        default=99,
        help="Use a negative value to disable the sum cap.",
    )

    return parser.parse_args()


def print_dataset_stats(df):
    print(f"Number of samples: {len(df)}")
    print(f"a range: {df['a'].min()} to {df['a'].max()}")
    print(f"b range: {df['b'].min()} to {df['b'].max()}")
    print(f"result range: {df['result'].min()} to {df['result'].max()}")

    print("\na counts:")
    print(f"min: {df['a'].value_counts().min()}, max: {df['a'].value_counts().max()}")

    print("\nb counts:")
    print(f"min: {df['b'].value_counts().min()}, max: {df['b'].value_counts().max()}")

    print("\nresult counts:")
    print(f"min: {df['result'].value_counts().min()}, max: {df['result'].value_counts().max()}")

    n_duplicates = df.duplicated(subset=["a", "b"]).sum()
    print(f"\nDuplicate (a, b) pairs: {n_duplicates}")


def generate_baseline_datasets():
    args = parse_args()
    set_seed(args.seed)

    use_digits = not args.use_words

    addition = make_addition_dataset(
        output_dir=args.output_dir,
        use_digits=use_digits,
        prompt=args.prompt,
        min_value=args.addition_min,
        max_value=args.addition_max,
        max_sum=None if args.addition_max_sum < 0 else args.addition_max_sum,
    )

    subtraction = make_subtraction_dataset(
        output_dir=args.output_dir,
        use_digits=use_digits,
        prompt=args.prompt
    )

    multiplication = make_multiplication_dataset(
        output_dir=args.output_dir,
        use_digits=use_digits,
        prompt=args.prompt
    )
    mixed_arithmetic = make_mixed_arithmetic_dataset(args.output_dir, args.prompt)

    print("Stats for addition dataset")
    print_dataset_stats(addition)

    print("Stats for subtraction dataset")
    print_dataset_stats(subtraction)

    print("Stats for multiplication dataset")
    print_dataset_stats(multiplication)

    print(f"Mixed arithmetic samples: {len(mixed_arithmetic)}")

    print("Done.")



if __name__=="__main__":
    generate_baseline_datasets()
