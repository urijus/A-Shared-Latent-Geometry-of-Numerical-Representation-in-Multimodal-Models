"""Compare learned DAS bases to digit-token unembedding directions.

This is a cheap readout-channel audit. If the DAS subspace is mostly the
final digit readout, digit-token unembedding vectors should project into it
well above the random expectation k / d_model.
"""

import argparse
import glob
from pathlib import Path

import torch

from src.common import save_jsonl
from src.geometry.readout import get_output_weight
from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)
from src.models import resolve_model_for_loading


DEFAULT_SUBSPACE_GLOBS = (
    "results/final_exps/DAS_audit_k_22/text/addition/"
    "das_pca_initialized/split_0/seed_*/subspace.pt",
    "results/final_exps/DAS_audit_k_22/text/addition/"
    "untrained_pca_initialized/split_0/seed_*/subspace.pt",
    "results/final_exps/DAS_audit_k_22/text/addition/"
    "random_subspace_in_pca_span/split_0/seed_*/subspace.pt",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Defaults to the first subspace config model.")
    parser.add_argument(
        "--load_device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Where to load the model for reading unembedding weights. CPU avoids GPU OOM.",
    )
    parser.add_argument("--subspace", type=Path, nargs="*", help="One or more subspace.pt files.")
    parser.add_argument(
        "--subspace_glob",
        action="append",
        help=(
            "Glob for subspace.pt files. Can be repeated. "
            "Defaults to DAS, untrained-PCA, and random-PCA controls."
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/check_unembeeding"))
    parser.add_argument(
        "--token_text",
        action="append",
        help=(
            "Text whose single token unembedding should be included. "
            "Defaults to bare numbers 0..99. Can be repeated."
        ),
    )
    parser.add_argument(
        "--max_token_number",
        type=int,
        default=99,
        help="When --token_text is absent, audit single-token strings from 0 through this value.",
    )
    parser.add_argument(
        "--include_space_prefixed",
        action="store_true",
        help="Also try token texts like ' 7'. Multi-token texts are skipped and reported.",
    )
    parser.add_argument("--stem", default="unembedding_overlap")
    return parser.parse_args()


def load_model_for_unembedding(model_path, load_device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device_map = {"": 0} if load_device == "cuda" else {"": "cpu"}
    kwargs = {
        "dtype": "auto",
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "use_safetensors": True,
    }

    if any(name in str(model_path).lower() for name in ("gemma-4", "glm-4.1v", "smolvlm", "qwen3.5")):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = AutoModelForMultimodalLM.from_pretrained(model_path, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def load_basis(path):
    payload = torch_load_portable(path)
    if "basis" not in payload:
        raise ValueError(f"{path} does not contain a final audit 'basis' tensor.")
    return torch.as_tensor(payload["basis"]).detach().float(), payload


def token_specs(tokenizer, texts):
    specs, skipped = [], []
    seen = set()
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            skipped.append({"text": text, "token_ids": token_ids, "reason": "not_single_token"})
            continue
        token_id = int(token_ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        specs.append(
            {
                "text": text,
                "token_id": token_id,
                "token": tokenizer.convert_ids_to_tokens([token_id])[0],
            }
        )
    if not specs:
        raise ValueError("No requested token texts encoded as a single token.")
    return specs, skipped


def projection_rows(basis, digit_vectors):
    d_model = max(basis.shape)
    das = orthonormal_columns(basis, d_model, "DAS basis")
    digit_subspace = orthonormal_columns(digit_vectors.T, d_model, "digit unembedding vectors")

    projection = das.T @ digit_vectors.T
    vector_norms = digit_vectors.norm(dim=1).clamp_min(1e-12)
    per_token_squared = projection.square().sum(dim=0) / vector_norms.square()
    per_token_length = per_token_squared.sqrt()

    singular_values = torch.linalg.svdvals(das.T @ digit_subspace).clamp(0, 1)
    squared_sum = singular_values.square().sum()
    random_baseline = das.shape[1] / d_model
    digit_subspace_in_das = squared_sum / digit_subspace.shape[1]

    return {
        "d_model": d_model,
        "k": das.shape[1],
        "digit_subspace_rank": digit_subspace.shape[1],
        "n_digit_tokens": digit_vectors.shape[0],
        "random_baseline_k_over_d_model": float(random_baseline),
        "mean_digit_token_projection_squared": float(per_token_squared.mean()),
        "mean_digit_token_projection_over_random": float(per_token_squared.mean() / random_baseline),
        "max_digit_token_projection_squared": float(per_token_squared.max()),
        "digit_subspace_in_das": float(digit_subspace_in_das),
        "digit_subspace_in_das_over_random": float(digit_subspace_in_das / random_baseline),
        "das_in_digit_subspace": float(squared_sum / das.shape[1]),
        "principal_cosines": [float(value) for value in singular_values],
        "per_token_projection_squared": [float(value) for value in per_token_squared],
        "per_token_projection_length": [float(value) for value in per_token_length],
    }


def prefixed(prefix, values):
    return {f"{prefix}{key}": value for key, value in values.items()}


def flag(over_random):
    if over_random < 2.0:
        return "near_random"
    if over_random < 5.0:
        return "elevated"
    return "strongly_elevated"


def print_summary(rows, skipped):
    print("\nUnembedding overlap audit")
    print("Question: does the learned DAS subspace coincide with number-token readout directions?\n")
    print(
        f"{'condition':28} {'seed':>4} {'k':>4} {'random':>10} "
        f"{'raw_x':>8} {'center_x':>9} {'center_span':>12} flag"
    )
    print("-" * 100)
    for row in rows:
        condition = str(row.get("condition") or Path(row["subspace_path"]).parents[2].name)
        if len(condition) > 28:
            condition = condition[:25] + "..."
        print(
            f"{condition:28} {str(row.get('seed')):>4} {row['k']:4d} "
            f"{row['random_baseline_k_over_d_model']:10.6f} "
            f"{row['mean_digit_token_projection_over_random']:8.2f} "
            f"{row['centered_mean_digit_token_projection_over_random']:9.2f} "
            f"{row['centered_digit_subspace_in_das']:12.6f} "
            f"{row['centered_readout_channel_flag']}"
        )
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    print("\nCondition means:")
    print(f"{'condition':28} {'n':>3} {'raw_x':>8} {'center_x':>9} {'center_span':>12}")
    print("-" * 66)
    for condition, condition_rows in sorted(by_condition.items()):
        raw = sum(row["mean_digit_token_projection_over_random"] for row in condition_rows)
        centered = sum(
            row["centered_mean_digit_token_projection_over_random"]
            for row in condition_rows
        )
        span = sum(row["centered_digit_subspace_in_das"] for row in condition_rows)
        n = len(condition_rows)
        print(f"{condition:28} {n:3d} {raw / n:8.2f} {centered / n:9.2f} {span / n:12.6f}")
    print("\nHeuristic: near_random weakens the final-readout concern; elevated overlap supports checking further.")
    print("The table uses centered number-token directions for the flag; JSONL includes raw and centered metrics.")
    if skipped:
        print("\nSkipped token texts because they were not single tokens:")
        for item in skipped:
            print(f"  {item['text']!r} -> {item['token_ids']}")


def main():
    args = parse_args()
    if args.subspace:
        subspace_paths = list(args.subspace)
    else:
        subspace_paths = []
        for pattern in args.subspace_glob or DEFAULT_SUBSPACE_GLOBS:
            subspace_paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    if not subspace_paths:
        raise FileNotFoundError("No subspaces matched the requested globs.")

    first_basis, first_payload = load_basis(subspace_paths[0])
    del first_basis
    model_name = args.model or first_payload.get("config", {}).get("model")
    if model_name is None:
        raise ValueError("Pass --model because the first subspace has no config model.")

    model_path, resolved_model = resolve_model_for_loading(model_name)
    model, tokenizer = load_model_for_unembedding(model_path, args.load_device)
    weight = get_output_weight(model)

    texts = args.token_text or [str(i) for i in range(args.max_token_number + 1)]
    if args.include_space_prefixed:
        texts = texts + [f" {i}" for i in range(args.max_token_number + 1)]
    specs, skipped = token_specs(tokenizer, texts)
    digit_vectors = torch.stack([weight[item["token_id"]] for item in specs])
    centered_digit_vectors = digit_vectors - digit_vectors.mean(dim=0, keepdim=True)

    rows = []
    for path in subspace_paths:
        basis, payload = load_basis(path)
        row = {
            "subspace_path": str(path),
            "model": resolved_model,
            "condition": payload.get("condition"),
            "kind": payload.get("kind"),
            "layer": payload.get("config", {}).get("layer"),
            "position": payload.get("config", {}).get("position"),
            "hook": payload.get("config", {}).get("hook"),
            "seed": payload.get("config", {}).get("seed"),
            "token_specs": specs,
            "skipped_token_texts": skipped,
        }
        row.update(projection_rows(basis, digit_vectors))
        row.update(prefixed("centered_", projection_rows(basis, centered_digit_vectors)))
        row["readout_channel_flag"] = flag(row["mean_digit_token_projection_over_random"])
        row["centered_readout_channel_flag"] = flag(
            row["centered_mean_digit_token_projection_over_random"]
        )
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.stem}.jsonl"
    save_jsonl(rows, jsonl_path)
    print_summary(rows, skipped)
    print(f"Saved {jsonl_path}")


if __name__ == "__main__":
    main()
