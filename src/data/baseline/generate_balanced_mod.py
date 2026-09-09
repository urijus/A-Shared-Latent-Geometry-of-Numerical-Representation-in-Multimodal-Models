"""Generate arithmetic data balanced by target modulo a period."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from src.common.io import save_jsonl
from src.common.seed import set_seed
from src.data.baseline.generate_datasets import (
    addition_pairs_to_samples,
    multiplication_pairs_to_samples,
    subtraction_pairs_to_samples,
)


BUILDERS = {
    "addition": addition_pairs_to_samples,
    "subtraction": subtraction_pairs_to_samples,
    "multiplication": multiplication_pairs_to_samples,
}


def candidate_pairs(modality, max_a, max_b, cap_addition_result=True):
    pairs = [(a, b) for a in range(max_a + 1) for b in range(max_b + 1)]
    if modality == "addition" and cap_addition_result:
        max_result = max(max_a, max_b)
        pairs = [(a, b) for a, b in pairs if a + b <= max_result]
    if modality == "subtraction":
        pairs = [(a, b) for a, b in pairs if a >= b]
    return pairs


def make_balanced_mod_dataset(
    modality,
    target,
    modulus,
    examples_per_class,
    max_a,
    max_b,
    prompt="Output ONLY a number.",
    use_digits=True,
    allow_fewer=True,
    min_target_value=None,
    max_target_value=None,
    cap_addition_result=True,
    allow_missing_residues=False,
):
    samples = BUILDERS[modality](
        candidate_pairs(
            modality,
            max_a,
            max_b,
            cap_addition_result=cap_addition_result,
        ),
        use_digits,
        prompt,
    )
    if min_target_value is not None:
        samples = [
            sample for sample in samples
            if int(sample[target]) >= min_target_value
        ]
    if max_target_value is not None:
        samples = [
            sample for sample in samples
            if int(sample[target]) <= max_target_value
        ]
    if not samples:
        raise ValueError("No samples remain after target-value filtering.")

    by_residue = defaultdict(list)
    for sample in samples:
        by_residue[int(sample[target]) % modulus].append(sample)

    available_residues = [residue for residue in range(modulus) if by_residue[residue]]
    missing = [residue for residue in range(modulus) if not by_residue[residue]]
    if missing and not allow_fewer and not allow_missing_residues:
        raise ValueError(f"No examples for residues: {missing}")

    min_available = min(len(by_residue[residue]) for residue in available_residues)
    n_per_class = min(examples_per_class, min_available) if allow_fewer else examples_per_class
    if examples_per_class > min_available and not allow_fewer:
        raise ValueError(
            f"Requested {examples_per_class} examples per class, "
            f"but the smallest class has {min_available}."
        )

    balanced = []
    for residue in available_residues:
        n = min(examples_per_class, len(by_residue[residue])) if allow_fewer else n_per_class
        balanced.extend(random.sample(by_residue[residue], n))
    random.shuffle(balanced)
    return balanced, min_available, n_per_class, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--modality", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--modulus", type=int, required=True)
    parser.add_argument("--examples_per_class", type=int, default=50)
    parser.add_argument("--max_a", type=int, default=99)
    parser.add_argument("--max_b", type=int, default=99)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--use_words", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--min_target_value", type=int, default=None)
    parser.add_argument("--max_target_value", type=int, default=None)
    parser.add_argument(
        "--allow_missing_residues",
        action="store_true",
        help="Allow absent residue ids while still enforcing examples_per_class for present ids.",
    )
    parser.add_argument(
        "--allow_addition_overflow",
        action="store_true",
        help="For addition, keep all a,b pairs instead of requiring a + b <= max(max_a, max_b).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    samples, min_available, n_per_class, missing = make_balanced_mod_dataset(
        modality=args.modality,
        target=args.target,
        modulus=args.modulus,
        examples_per_class=args.examples_per_class,
        max_a=args.max_a,
        max_b=args.max_b,
        prompt=args.prompt,
        use_digits=not args.use_words,
        allow_fewer=not args.strict,
        min_target_value=args.min_target_value,
        max_target_value=args.max_target_value,
        cap_addition_result=not args.allow_addition_overflow,
        allow_missing_residues=args.allow_missing_residues,
    )

    output_path = args.output_dir / f"{args.modality}_baseline.jsonl"
    save_jsonl(samples, output_path)
    print("Saved:", output_path)
    print("Modality:", args.modality)
    print("Target:", args.target)
    print("Modulus:", args.modulus)
    print("Samples:", len(samples))
    print("Requested examples per residue:", args.examples_per_class)
    print("Smallest used residue count:", n_per_class)
    print("Smallest available residue class:", min_available)
    if args.min_target_value is not None or args.max_target_value is not None:
        print(
            "Target value filter:",
            f"{args.min_target_value if args.min_target_value is not None else '-inf'}"
            ".."
            f"{args.max_target_value if args.max_target_value is not None else 'inf'}",
        )
    if missing:
        print("Missing residues:", missing)


if __name__ == "__main__":
    main()