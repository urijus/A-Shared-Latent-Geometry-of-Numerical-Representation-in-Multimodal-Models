"""Generate balanced nested arithmetic datasets with intermediate values."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INNER_OPERATIONS = {
    "addition": "+",
    "subtraction": "-",
}

ORDER_TO_OPERATION = {
    ("addition", "inner_left"): "add_then_multiply",
    ("addition", "inner_right"): "multiply_add",
    ("subtraction", "inner_left"): "sub_then_multiply",
    ("subtraction", "inner_right"): "multiply_sub",
}

MODULI = (2, 5, 10, 20, 50, 100)
ORDERS = ("inner_left", "inner_right")


def save_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")


def candidate_pairs(inner_operation, max_value=99):
    if inner_operation == "addition":
        return [
            (a, b)
            for a in range(max_value + 1)
            for b in range(max_value + 1)
            if a + b <= max_value
        ]

    if inner_operation == "subtraction":
        return [
            (a, b)
            for a in range(max_value + 1)
            for b in range(max_value + 1)
            if a >= b and a - b <= max_value
        ]

    raise ValueError(f"Unknown inner operation: {inner_operation}")


def intermediate_value(inner_operation, a, b):
    if inner_operation == "addition":
        return a + b
    if inner_operation == "subtraction":
        return a - b
    raise ValueError(f"Unknown inner operation: {inner_operation}")


def count_exhaustive_examples(
    max_value=99,
    max_c=9,
):
    n_c = max_c + 1
    return sum(
        len(candidate_pairs(inner_operation, max_value)) * n_c * len(ORDERS)
        for inner_operation in INNER_OPERATIONS
    )


def max_exact_balanced_examples(
    max_value=99,
    max_c=9,
):
    total = 0
    for inner_operation in INNER_OPERATIONS:
        counts = Counter(
            intermediate_value(inner_operation, a, b)
            for a, b in candidate_pairs(inner_operation, max_value)
        )
        total += min(counts.values()) * len(counts) * (max_c + 1) * len(ORDERS)
    return total


def pairs_by_intermediate(inner_operation, max_value):
    grouped = defaultdict(list)
    for a, b in candidate_pairs(inner_operation, max_value):
        grouped[intermediate_value(inner_operation, a, b)].append((a, b))
    return grouped


def make_expr_and_tokens(inner_operation, order, a, b, c, prompt):
    symbol = INNER_OPERATIONS[inner_operation]
    inner_tokens = ["(", str(a), symbol, str(b), ")"]
    inner_expr = f"({a} {symbol} {b})"

    if order == "inner_left":
        tokens = [*inner_tokens, "*", str(c), "="]
        expr = f"{inner_expr} * {c} ="
    elif order == "inner_right":
        tokens = [str(c), "*", *inner_tokens, "="]
        expr = f"{c} * {inner_expr} ="
    else:
        raise ValueError(f"Unknown expression order: {order}")

    if prompt:
        expr = f"{prompt} {expr}"
    return expr, tokens


def add_mod_fields(sample, prefix, value):
    for modulus in MODULI:
        sample[f"{prefix}_mod_{modulus}"] = value % modulus


def make_sample(sample_id, inner_operation, order, a, b, c, prompt):
    inner = intermediate_value(inner_operation, a, b)
    result = inner * c
    expr, tokens = make_expr_and_tokens(inner_operation, order, a, b, c, prompt)
    left_factor = inner if order == "inner_left" else c
    right_factor = c if order == "inner_left" else inner

    sample = {
        "sample_id": sample_id,
        "task": f"intermediate_{inner_operation}_multiply",
        "operation": ORDER_TO_OPERATION[(inner_operation, order)],
        "inner_operation": inner_operation,
        "inner_symbol": INNER_OPERATIONS[inner_operation],
        "order": order,
        "expr": expr,
        "tokens": tokens,
        "a": a,
        "b": b,
        "c": c,
        "x": a,
        "y": b,
        "inner_value": inner,
        "intermediate": inner,
        "left_factor": left_factor,
        "right_factor": right_factor,
        "result": result,
        "requires_carry": int(
            inner_operation == "addition" and (a % 10) + (b % 10) >= 10
        ),
        "requires_borrow": int(
            inner_operation == "subtraction" and (a % 10) < (b % 10)
        ),
        "result_ones": result % 10,
        "result_tens": (result // 10) % 10,
        "result_hundreds": result // 100,
        "c0_hat": (inner % 10) * c,
        "c0": ((inner % 10) * c) % 10,
        "r0": ((inner % 10) * c) // 10,
        "c1_hat": (inner // 10) * c + (((inner % 10) * c) // 10),
        "c1": ((inner // 10) * c + (((inner % 10) * c) // 10)) % 10,
        "r1": ((inner // 10) * c + (((inner % 10) * c) // 10)) // 10,
        "c2": ((inner // 10) * c + (((inner % 10) * c) // 10)) // 10,
    }

    if inner_operation == "addition":
        sample["x_plus_y"] = inner
        sample["x_plus_y_mod_10"] = inner % 10
        sample["x_plus_y_mod_100"] = inner % 100
    else:
        sample["x_minus_y"] = inner
        sample["x_minus_y_mod_10"] = inner % 10
        sample["x_minus_y_mod_100"] = inner % 100

    add_mod_fields(sample, "a", a)
    add_mod_fields(sample, "b", b)
    add_mod_fields(sample, "c", c)
    add_mod_fields(sample, "inner_value", inner)
    add_mod_fields(sample, "intermediate", inner)
    add_mod_fields(sample, "result", result)
    return sample


def make_balanced_dataset(
    samples_per_cell=1,
    max_value=99,
    max_c=9,
    prompt="Output ONLY a number.",
    max_samples=None,
):
    samples = []
    sample_id = 0
    for inner_operation in INNER_OPERATIONS:
        grouped = pairs_by_intermediate(inner_operation, max_value)
        for inner in range(max_value + 1):
            pairs = grouped[inner]
            n = min(samples_per_cell, len(pairs))
            for c in range(max_c + 1):
                for order in ORDERS:
                    selected_pairs = random.sample(pairs, n)
                    for a, b in selected_pairs:
                        samples.append(
                            make_sample(
                                sample_id=sample_id,
                                inner_operation=inner_operation,
                                order=order,
                                a=a,
                                b=b,
                                c=c,
                                prompt=prompt,
                            )
                        )
                        sample_id += 1

    random.shuffle(samples)
    if max_samples is not None and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    for sample_id, sample in enumerate(samples):
        sample["sample_id"] = sample_id
    return samples


def print_dataset_stats(samples, max_value, max_c):
    print("Samples:", len(samples))
    print(
        "Exhaustive possible samples:",
        count_exhaustive_examples(max_value, max_c),
    )
    print(
        "Maximum exactly balanced samples:",
        max_exact_balanced_examples(max_value, max_c),
    )
    for field in ("inner_operation", "order", "inner_value", "c"):
        counts = Counter(sample[field] for sample in samples)
        print(f"{field} counts: min={min(counts.values())}, max={max(counts.values())}")
    if samples:
        print("First sample:", samples[0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/intermediate"))
    parser.add_argument(
        "--output_name",
        default="intermediate_add_sub_multiply_baseline.jsonl",
    )
    parser.add_argument("--samples_per_cell", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_value", type=int, default=99)
    parser.add_argument("--max_c", type=int, default=9)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_cell < 1:
        raise ValueError("--samples_per_cell must be at least 1.")
    if args.max_value < 0:
        raise ValueError("--max_value must be non-negative.")
    if args.max_c < 0 or args.max_c > 9:
        raise ValueError("--max_c must be in 0..9 to keep multiplication one digit.")

    random.seed(args.seed)
    samples = make_balanced_dataset(
        samples_per_cell=args.samples_per_cell,
        max_value=args.max_value,
        max_c=args.max_c,
        prompt=args.prompt,
        max_samples=args.max_samples,
    )

    output_path = args.output_dir / args.output_name
    save_jsonl(samples, output_path)
    print("Saved:", output_path)
    print_dataset_stats(samples, args.max_value, args.max_c)


if __name__ == "__main__":
    main()
