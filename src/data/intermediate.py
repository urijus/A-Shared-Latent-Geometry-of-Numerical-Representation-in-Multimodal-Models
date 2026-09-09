"""Generate tiny add-then-multiply examples for intermediate-value tests."""

import argparse
import random
from pathlib import Path

from src.common.io import save_jsonl
from src.common.seed import set_seed


def make_sample(sample_id, x, y, c, prompt):
    x_plus_y = x + y
    result = x_plus_y * c
    expr = f"({x} + {y}) * {c} ="
    if prompt:
        expr = f"{prompt} {expr}"

    return {
        "sample_id": sample_id,
        "task": "intermediate_add_multiply",
        "operation": "add_then_multiply",
        "expr": expr,
        "tokens": ["(", str(x), "+", str(y), ")", "*", str(c), "="],
        "x": x,
        "y": y,
        "c": c,
        "x_plus_y": x_plus_y,
        "result": result,
        "x_plus_y_mod_10": x_plus_y % 10,
        "x_plus_y_mod_100": x_plus_y % 100,
        "result_mod_10": result % 10,
        "result_mod_100": result % 100,
    }


def make_dataset(n_examples, max_x, max_y, max_c, prompt, max_result=None):
    triples = [
        (x, y, c)
        for x in range(max_x + 1)
        for y in range(max_y + 1)
        for c in range(max_c + 1)
        if x + y <= 99 and (max_result is None or (x + y) * c <= max_result)
    ]
    random.shuffle(triples)
    return [
        make_sample(sample_id, x, y, c, prompt)
        for sample_id, (x, y, c) in enumerate(triples[:n_examples])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--max_x", type=int, default=49)
    parser.add_argument("--max_y", type=int, default=49)
    parser.add_argument("--max_c", type=int, default=49)
    parser.add_argument("--max_result", type=int, default=None)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    samples = make_dataset(
        n_examples=args.n_examples,
        max_x=args.max_x,
        max_y=args.max_y,
        max_c=args.max_c,
        prompt=args.prompt,
        max_result=args.max_result,
    )

    output_path = args.output_dir / "intermediate_add_multiply_baseline.jsonl"
    save_jsonl(samples, output_path)

    print("Saved:", output_path)
    print("Samples:", len(samples))
    if samples:
        print("First sample:", samples[0])


if __name__ == "__main__":
    main()
