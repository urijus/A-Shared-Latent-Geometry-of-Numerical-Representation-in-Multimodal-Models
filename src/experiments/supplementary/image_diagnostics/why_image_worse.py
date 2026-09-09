"""Run the targeted DAS checks for why image causality is weaker than text.

This script is an orchestrator around ``DAS_audit.audit_das``.  It runs two
small grids:

1. k sweep: image addition, with text addition as a reference.
2. image position sweep: layer 43, k=22, positions -6..-1 from the prompt end.

Each grid point gets its own audit output directory so changing only k or
position cannot overwrite another run.  A flat JSONL aggregate is written under
``results/final_exps/why_image_worse`` for quick plotting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


AUDIT_MODULE = "src.experiments.arithmetic_reference.das_audit.audit_das"
CONDITIONS = [
    "full_activation_patching",
    "untrained_pca_initialized",
    "random_subspace_in_pca_span",
    "das_pca_initialized",
    "das_random_initialized",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", "k_sweep", "position_sweep"],
        default="all",
    )
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--operations", nargs="+", default=["addition"])
    parser.add_argument("--target", default="result")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/why_image_worse"))

    parser.add_argument("--k_values", type=int, nargs="+", default=[16, 22, 32, 64, 128])
    parser.add_argument("--k_sweep_modalities", nargs="+", default=["image", "text"])
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")

    parser.add_argument("--position_values", nargs="+", default=["-6", "-5", "-4", "-3", "-2", "-1"])
    parser.add_argument("--position_sweep_modalities", nargs="+", default=["image"])
    parser.add_argument("--position_sweep_k", type=int, default=22)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--random_init_epochs", type=int, default=40)
    parser.add_argument("--run_random_init", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--text_batch_size", type=int, default=16)
    parser.add_argument("--image_batch_size", type=int, default=2)
    parser.add_argument("--max_train_pairs", type=int, default=4096)
    parser.add_argument("--max_validation_pairs", type=int, default=512)
    parser.add_argument("--max_test_pairs", type=int, default=128)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--pca_max_samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=8)

    parser.add_argument(
        "--prompt",
        default="Output ONLY a number.",
        help="Only used for image experiments.",
    )
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--das_only",
        action="store_true",
        help="Run only the PCA-initialized DAS condition inside each audit.",
    )

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun grid points even when a DAS result already exists.",
    )
    parser.add_argument(
        "--run_index",
        type=int,
        help="Run only this 0-indexed grid point; useful for Slurm arrays.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Run grid points whose index modulo num_shards equals run_index.",
    )
    parser.add_argument(
        "--collect_only",
        action="store_true",
        help="Only rebuild manifest.jsonl and aggregate_results.jsonl from existing runs.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    valid_modalities = {"text", "image"}
    for name in ("k_sweep_modalities", "position_sweep_modalities"):
        values = getattr(args, name)
        invalid = sorted(set(values) - valid_modalities)
        if invalid:
            raise ValueError(f"{name} contains invalid modalities: {invalid}")
    if "image" in args.position_sweep_modalities:
        integer_positions = []
        for value in args.position_values:
            try:
                integer_positions.append(int(value))
            except ValueError:
                continue
        if integer_positions and all(position >= 0 for position in integer_positions):
            print(
                "WARNING: image position sweep received only non-negative "
                "positions. Image DAS positions are raw input-token indices; "
                "for the prompt tail use negative offsets such as -6 -5 -4 -3 -2 -1.",
                file=sys.stderr,
            )


def batch_size(args: argparse.Namespace, modality: str) -> int:
    return args.image_batch_size if modality == "image" else args.text_batch_size


def default_position(args: argparse.Namespace, modality: str) -> str:
    return args.image_position if modality == "image" else args.text_position


def audit_output_dir(
    args: argparse.Namespace,
    experiment: str,
    modality: str,
    position: str,
    k: int,
) -> Path:
    if experiment == "k_sweep":
        return (
            args.output_dir
            / "k_sweep"
            / f"layer{args.layer}"
            / f"{modality}_pos{position}"
            / f"k{k}"
        )
    if experiment == "position_sweep":
        return (
            args.output_dir
            / "position_sweep"
            / f"layer{args.layer}"
            / f"k{k}"
            / f"{modality}_pos{position}"
        )
    raise ValueError(f"Unknown experiment: {experiment}")


def condition_dir(
    output_dir: Path,
    modality: str,
    operation: str,
    target: str,
    condition: str,
    split_seed: int,
    seed: int,
) -> Path:
    task_dir = output_dir / modality / operation
    if target != "result":
        task_dir = task_dir / target
    return task_dir / condition / f"split_{split_seed}" / f"seed_{seed}"


def result_path(
    output_dir: Path,
    modality: str,
    operation: str,
    target: str,
    split_seed: int,
    seed: int,
    condition: str = "das_pca_initialized",
) -> Path:
    return condition_dir(
        output_dir, modality, operation, target, condition, split_seed, seed
    ) / "results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def audit_command(
    args: argparse.Namespace,
    output_dir: Path,
    modality: str,
    operation: str,
    position: str,
    k: int,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        AUDIT_MODULE,
        "--model",
        args.model,
        "--modality",
        modality,
        "--operation",
        operation,
        "--target",
        args.target,
        "--output_dir",
        str(output_dir),
        "--layer",
        str(args.layer),
        "--position",
        str(position),
        "--hook",
        args.hook,
        "--seed",
        str(seed),
        "--split_seed",
        str(args.split_seed),
        "--k",
        str(k),
        "--epochs",
        str(args.epochs),
        "--random_init_epochs",
        str(args.random_init_epochs),
        "--patience",
        str(args.patience),
        "--learning_rate",
        str(args.learning_rate),
        "--batch_size",
        str(batch_size(args, modality)),
        "--max_train_pairs",
        str(args.max_train_pairs),
        "--max_validation_pairs",
        str(args.max_validation_pairs),
        "--max_test_pairs",
        str(args.max_test_pairs),
        "--max_autoregressive_pairs",
        str(args.max_autoregressive_pairs),
        "--pca_max_samples",
        str(args.pca_max_samples),
        "--pca_variance_threshold",
        str(args.pca_variance_threshold),
        "--max_new_tokens",
        str(args.max_new_tokens),
    ]
    if modality == "image":
        command.extend(["--prompt", args.prompt])
        if args.enable_thinking:
            command.append("--enable_thinking")
    if args.use_chat_template:
        command.append("--use_chat_template")
    if args.das_only:
        command.append("--das_only")
    if args.run_random_init:
        command.append("--run_random_init")
    return command


def expected_runs(args: argparse.Namespace) -> list[dict]:
    runs = []
    if args.mode in {"all", "k_sweep"}:
        for modality in args.k_sweep_modalities:
            position = default_position(args, modality)
            for operation in args.operations:
                for k in args.k_values:
                    for seed in args.seeds:
                        runs.append(
                            {
                                "experiment": "k_sweep",
                                "modality": modality,
                                "operation": operation,
                                "position": position,
                                "k": k,
                                "seed": seed,
                            }
                        )

    if args.mode in {"all", "position_sweep"}:
        for modality in args.position_sweep_modalities:
            for operation in args.operations:
                for position in args.position_values:
                    for seed in args.seeds:
                        runs.append(
                            {
                                "experiment": "position_sweep",
                                "modality": modality,
                                "operation": operation,
                                "position": str(position),
                                "k": args.position_sweep_k,
                                "seed": seed,
                            }
                        )
    return runs


def completed(run: dict, args: argparse.Namespace, output_dir: Path) -> bool:
    return result_path(
        output_dir=output_dir,
        modality=run["modality"],
        operation=run["operation"],
        target=args.target,
        split_seed=args.split_seed,
        seed=run["seed"],
    ).exists()


def collect_rows(args: argparse.Namespace, runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        output_dir = audit_output_dir(
            args,
            run["experiment"],
            run["modality"],
            run["position"],
            run["k"],
        )
        for condition in CONDITIONS:
            path = result_path(
                output_dir=output_dir,
                modality=run["modality"],
                operation=run["operation"],
                target=args.target,
                split_seed=args.split_seed,
                seed=run["seed"],
                condition=condition,
            )
            if not path.exists():
                continue
            for row in load_jsonl(path):
                rows.append(
                    {
                        "experiment": run["experiment"],
                        "grid_output_dir": str(output_dir),
                        "results_path": str(path),
                        "requested_k": run["k"],
                        "requested_position": str(run["position"]),
                        **row,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["experiment"],
            row["modality"],
            row["operation"],
            int(row["requested_k"]),
            str(row["requested_position"]),
            int(row["seed"]),
            row["condition"],
        )
    )
    return rows


def write_manifest(args: argparse.Namespace, runs: list[dict], rows: list[dict]) -> None:
    manifest_rows = []
    for run in runs:
        output_dir = audit_output_dir(
            args,
            run["experiment"],
            run["modality"],
            run["position"],
            run["k"],
        )
        manifest_rows.append(
            {
                **run,
                "target": args.target,
                "split_seed": args.split_seed,
                "layer": args.layer,
                "hook": args.hook,
                "output_dir": str(output_dir),
                "das_result_exists": completed(run, args, output_dir),
            }
        )
    save_jsonl(manifest_rows, args.output_dir / "manifest.jsonl")
    save_jsonl(rows, args.output_dir / "aggregate_results.jsonl")


def run_one(args: argparse.Namespace, run: dict) -> None:
    output_dir = audit_output_dir(
        args,
        run["experiment"],
        run["modality"],
        run["position"],
        run["k"],
    )
    if not args.force and completed(run, args, output_dir):
        print(
            "Skipping existing "
            f"{run['experiment']} {run['modality']} {run['operation']} "
            f"pos={run['position']} k={run['k']} seed={run['seed']}"
        )
        return
    command = audit_command(
        args=args,
        output_dir=output_dir,
        modality=run["modality"],
        operation=run["operation"],
        position=run["position"],
        k=run["k"],
        seed=run["seed"],
    )
    printable = " ".join(command)
    print(f"Running: {printable}")
    if args.dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    runs = expected_runs(args)
    if args.collect_only:
        rows = collect_rows(args, runs)
        write_manifest(args, runs, rows)
        print(f"Saved manifest: {args.output_dir / 'manifest.jsonl'}")
        print(f"Saved aggregate results: {args.output_dir / 'aggregate_results.jsonl'}")
        return

    indexed_runs = list(enumerate(runs))
    if args.run_index is not None:
        if args.run_index < 0:
            raise ValueError("--run_index must be non-negative.")
        if args.num_shards < 1:
            raise ValueError("--num_shards must be at least 1.")
        indexed_runs = [
            (index, run)
            for index, run in indexed_runs
            if index % args.num_shards == args.run_index
        ]
    selected_runs = [run for _, run in indexed_runs]
    print(f"Scheduled {len(selected_runs)} of {len(runs)} audit runs.")
    for index, run in indexed_runs:
        print(f"Grid index {index}/{len(runs) - 1}")
        run_one(args, run)
    if args.run_index is not None:
        print("Skipped aggregate writes for indexed run. Use --collect_only after all shards finish.")
        return
    rows = collect_rows(args, runs)
    write_manifest(args, runs, rows)
    print(f"Saved manifest: {args.output_dir / 'manifest.jsonl'}")
    print(f"Saved aggregate results: {args.output_dir / 'aggregate_results.jsonl'}")


if __name__ == "__main__":
    main()
