"""Create DAS subspaces with number-token readout directions projected out.

The output mirrors the DAS audit folder structure, so existing final_exps
scripts can be rerun by pointing AUDIT_ROOT at --output_root.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from src.common import save_jsonl
from src.experiments.readout_latent_geometry.controls.unembedding_overlap import (
    load_model_for_unembedding,
    token_specs,
)
from src.geometry.readout import get_output_weight
from src.experiments.cross_condition_transfer.causal_transfer.causal_transfer import parse_task
from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)
from src.models import resolve_model_for_loading


DEFAULT_TASKS = ["text:addition", "text:subtraction", "image:addition", "image:subtraction"]
DEFAULT_CONDITIONS = [
    "das_pca_initialized",
    "random_subspace_in_pca_span",
    "untrained_pca_initialized",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/experiments/check_unembeeding/readout_ablated_audit_k_22_layer43"),
    )
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--load_device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--target", default="result")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--token_text",
        action="append",
        help="Single-token readout direction to remove. Defaults to number tokens 0..9.",
    )
    parser.add_argument("--max_token_number", type=int, default=9)
    parser.add_argument("--include_space_prefixed", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output subspace folders.",
    )
    return parser.parse_args()


def task_root(root, modality, operation, target):
    base = root / modality / operation
    return base if target == "result" else base / target


def run_dir(root, modality, operation, target, condition, split_seed, seed):
    return task_root(root, modality, operation, target) / condition / f"split_{split_seed}" / f"seed_{seed}"


def load_basis(path):
    payload = torch_load_portable(path)
    if "basis" not in payload:
        raise ValueError(f"{path} does not contain a final audit basis.")
    return torch.as_tensor(payload["basis"]).detach().float(), payload


def result_row(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No rows in {path}")


def position_for(args, modality):
    return args.image_position if modality == "image" else args.text_position


def row_matches(args, row, modality):
    return (
        int(row.get("layer", -1)) == args.layer
        and int(row.get("k", -1)) == args.k
        and str(row.get("position")) == str(position_for(args, modality))
        and row.get("hook") == args.hook
    )


def readout_basis(args):
    model_path, resolved_model = resolve_model_for_loading(args.model)
    model, tokenizer = load_model_for_unembedding(model_path, args.load_device)
    weight = get_output_weight(model)
    texts = args.token_text or [str(i) for i in range(args.max_token_number + 1)]
    if args.include_space_prefixed:
        texts = texts + [f" {i}" for i in range(args.max_token_number + 1)]
    specs, skipped = token_specs(tokenizer, texts)
    vectors = torch.stack([weight[item["token_id"]] for item in specs])
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    basis = orthonormal_columns(centered.T, centered.shape[1], "centered readout vectors")
    return basis, specs, skipped, resolved_model


def project_out_readout(basis, readout):
    d_model = max(basis.shape)
    original = orthonormal_columns(basis, d_model, "source basis")
    projected = original - readout @ (readout.T @ original)
    residual_energy = float(projected.square().sum() / original.shape[1])
    left, singular_values, _ = torch.linalg.svd(projected, full_matrices=False)
    tolerance = max(projected.shape) * torch.finfo(projected.dtype).eps * singular_values.max()
    rank = int((singular_values > tolerance).sum().item())
    if rank < original.shape[1]:
        raise ValueError(
            f"Readout ablation reduced rank from {original.shape[1]} to {rank}; "
            "this script keeps k fixed for downstream Procrustes."
        )
    ablated = left[:, : original.shape[1]]
    after_overlap = float(torch.linalg.svdvals(ablated.T @ readout).square().sum() / readout.shape[1])
    before_overlap = float(torch.linalg.svdvals(original.T @ readout).square().sum() / readout.shape[1])
    return ablated, {
        "k": original.shape[1],
        "d_model": d_model,
        "readout_rank": readout.shape[1],
        "retained_energy_after_projection": residual_energy,
        "removed_energy_fraction": 1.0 - residual_energy,
        "readout_in_original_subspace": before_overlap,
        "readout_in_ablated_subspace": after_overlap,
    }


def copy_support_files(source_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("results.jsonl", "heldout_pairs.jsonl", "autoregressive_outputs.jsonl"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def write_subspace(output_path, payload, ablated_basis, metrics, source_path, args, token_specs):
    new_payload = dict(payload)
    new_payload["basis"] = ablated_basis.cpu()
    new_payload["kind"] = f"readout_ablated_{payload.get('kind', 'subspace')}"
    new_payload["readout_ablation"] = {
        **metrics,
        "source_subspace_path": str(source_path),
        "token_specs": token_specs,
        "centered_readout": True,
        "model": args.model,
    }
    torch.save(new_payload, output_path)


def main():
    args = parse_args()
    readout, specs, skipped, resolved_model = readout_basis(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    readout_basis_path = args.output_root / "digit_readout_basis.pt"
    torch.save(
        {
            "basis": readout.cpu(),
            "digit_readout_basis": readout.cpu(),
            "token_specs": specs,
            "skipped_token_texts": skipped,
            "centered_readout": True,
            "model": args.model,
            "resolved_model": resolved_model,
            "max_token_number": args.max_token_number,
            "include_space_prefixed": args.include_space_prefixed,
        },
        readout_basis_path,
    )
    rows = []

    for task in args.tasks:
        modality, operation = parse_task(task)
        for condition in args.conditions:
            for seed in args.seeds:
                source_dir = run_dir(
                    args.audit_root, modality, operation, args.target, condition, args.split_seed, seed
                )
                source_path = source_dir / "subspace.pt"
                source_results = source_dir / "results.jsonl"
                if not source_path.exists():
                    print(f"Skipping missing {source_path}")
                    continue
                if not source_results.exists():
                    print(f"Skipping missing {source_results}")
                    continue
                source_row = result_row(source_results)
                if not row_matches(args, source_row, modality):
                    print(
                        "Skipping metadata mismatch "
                        f"{source_results}: layer={source_row.get('layer')} "
                        f"k={source_row.get('k')} position={source_row.get('position')} "
                        f"hook={source_row.get('hook')} expected layer={args.layer} "
                        f"k={args.k} position={position_for(args, modality)} hook={args.hook}"
                    )
                    continue
                output_dir = run_dir(
                    args.output_root, modality, operation, args.target, condition, args.split_seed, seed
                )
                output_path = output_dir / "subspace.pt"
                if output_path.exists() and not args.force:
                    print(f"Skipping existing {output_path}; pass --force to overwrite.")
                    continue

                basis, payload = load_basis(source_path)
                ablated, metrics = project_out_readout(basis, readout)
                copy_support_files(source_dir, output_dir)
                write_subspace(output_path, payload, ablated, metrics, source_path, args, specs)
                row = {
                    "task": task,
                    "modality": modality,
                    "operation": operation,
                    "condition": condition,
                    "seed": seed,
                    "source_subspace_path": str(source_path),
                    "output_subspace_path": str(output_path),
                    "model": resolved_model,
                    "skipped_token_texts": skipped,
                    **metrics,
                }
                rows.append(row)

    summary_path = args.output_root / "readout_ablation_summary.jsonl"
    if rows or not summary_path.exists():
        save_jsonl(rows, summary_path)
    print("\nReadout-ablated subspaces")
    print(f"Output root: {args.output_root}")
    print(f"Saved readout basis: {readout_basis_path}")
    print(f"Readout rank removed: {readout.shape[1]}")
    print(f"Saved rows: {len(rows)}")
    print(f"{'task':18} {'condition':28} {'seed':>4} {'removed':>9} {'before':>9} {'after':>9}")
    print("-" * 86)
    for row in rows:
        print(
            f"{row['task']:18} {row['condition']:28} {row['seed']:4d} "
            f"{row['removed_energy_fraction']:9.4f} "
            f"{row['readout_in_original_subspace']:9.4f} "
            f"{row['readout_in_ablated_subspace']:9.4f}"
        )
    print(f"Saved {summary_path}")
    if skipped:
        print(f"Skipped {len(skipped)} non-single-token requested token texts.")


if __name__ == "__main__":
    main()
