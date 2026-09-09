"""Extend the lightweight Ministral DAS rank sweep and run a final-budget check.

This runner intentionally reuses the cache layout from ``stage1_das.py``.
It only trains:

1. lightweight K128 for T+ and I+ at L37, seed 0;
2. a final-budget comparison for the two selected ranks, again only T+/I+.

Existing lightweight K22/K32/K64 runs are loaded from cache and never rerun.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from src.experiments.supplementary.model_replication import stage1_das as stage1


TASKS = ("T+", "I+")
LIGHT_RANKS = (22, 32, 64, 128)
LAYER = 37
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=stage1.MODEL_ALIAS)
    parser.add_argument("--model_slug", "--model-slug", default=stage1.MODEL_SLUG)
    parser.add_argument("--format", default="digits")
    parser.add_argument("--output_root", "--output-root", type=Path, default=stage1.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--position", default="-1")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--target", default="result")
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pca_variance_threshold", "--pca-variance-threshold", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=8)
    parser.add_argument("--train_fraction", "--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", "--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split_seed", "--split-seed", type=int, default=0)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", "--enable-thinking", action="store_true")
    parser.add_argument("--use_chat_template", "--use-chat-template", action="store_true", default=True)
    parser.add_argument("--force_k128", "--force-k128", action="store_true")
    parser.add_argument("--force_final", "--force-final", action="store_true")
    parser.add_argument(
        "--final_only",
        "--final-only",
        action="store_true",
        help="Skip the lightweight K128/cache step and run only final-budget ranks.",
    )
    parser.add_argument(
        "--final_ranks",
        "--final-ranks",
        type=int,
        nargs=2,
        default=(32, 64),
        help="Two ranks to compare when --final_only is set.",
    )
    parser.add_argument("--rank_tolerance", "--rank-tolerance", type=float, default=0.03)
    parser.add_argument("--final_retention_ratio", "--final-retention-ratio", type=float, default=0.95)
    return parser.parse_args()


def make_stage_args(
    args: argparse.Namespace,
    *,
    phase: str,
    epochs: int,
    patience: int,
    batch_size: int,
    pca_max_samples: int,
    max_train_pairs: int,
    max_validation_pairs: int,
    max_test_pairs: int,
    max_autoregressive_pairs: int,
    force: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        model=args.model,
        model_slug=args.model_slug,
        format=args.format,
        output_root=args.output_root,
        selected_layer=None,
        selected_k=None,
        position=args.position,
        hook=args.hook,
        target=args.target,
        epochs=epochs,
        patience=patience,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        pca_max_samples=pca_max_samples,
        pca_variance_threshold=args.pca_variance_threshold,
        max_train_pairs=max_train_pairs,
        max_validation_pairs=max_validation_pairs,
        max_test_pairs=max_test_pairs,
        max_autoregressive_pairs=max_autoregressive_pairs,
        max_new_tokens=args.max_new_tokens,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        prompt=args.prompt,
        enable_thinking=args.enable_thinking,
        use_chat_template=args.use_chat_template,
        force=force,
    )


def cached_metrics(args: SimpleNamespace, task: str, k: int) -> dict:
    _config, digest, run_dir = stage1.das_config(args, task, LAYER, k, SEED)
    if not stage1.completed_run(run_dir):
        raise FileNotFoundError(
            f"Missing required cached lightweight run for {task} L{LAYER} K{k}: {run_dir}\n"
            "This script will not rerun K22/K32/K64. Run the lightweight Phase C "
            "first, or check that its settings match this script."
        )
    row = stage1.load_json(run_dir / "metrics.json")
    row["config_hash"] = row.get("config_hash", digest)
    row["run_dir"] = row.get("run_dir", str(run_dir))
    return row


def print_metric_table(title: str, rows_by_task_k: dict[tuple[str, int], dict], metric: str) -> None:
    table = []
    for task in TASKS:
        table.append(
            [
                task,
                *(stage1.fmt(rows_by_task_k[(task, k)].get(metric)) for k in LIGHT_RANKS),
            ]
        )
    print(f"\n{title}", flush=True)
    stage1.print_table(["task", "K22", "K32", "K64", "K128"], table)


def rank_means(rows_by_task_k: dict[tuple[str, int], dict]) -> dict[int, float]:
    means = {}
    for k in LIGHT_RANKS:
        values = [
            rows_by_task_k[(task, k)].get("autoregressive_iia")
            for task in TASKS
        ]
        if any(value is None for value in values):
            raise RuntimeError(f"Cannot choose ranks: missing AR-IIA for K{k}.")
        means[k] = sum(float(value) for value in values) / len(values)
    return means


def choose_final_ranks(means: dict[int, float], tolerance: float) -> tuple[int, int] | None:
    spread = max(means.values()) - min(means.values())
    if spread <= tolerance:
        return None
    if means[128] > means[64] + tolerance:
        return 64, 128
    return 32, 64


def recommend_rank(final_rows: list[dict], selected_ranks: tuple[int, int], retention_ratio: float) -> int:
    low, high = selected_ranks
    by_k = {low: [], high: []}
    for row in final_rows:
        by_k[int(row["k"])].append(float(row["autoregressive_iia"]))
    low_mean = sum(by_k[low]) / len(by_k[low])
    high_mean = sum(by_k[high]) / len(by_k[high])
    return low if low_mean >= retention_ratio * high_mean else high


def run_final_budget(args: argparse.Namespace, selected_ranks: tuple[int, int]) -> list[dict]:
    print(
        f"\nFinal-budget comparison K{selected_ranks[0]} vs K{selected_ranks[1]}",
        flush=True,
    )
    final_args = make_stage_args(
        args,
        phase="rank_extension_final_budget",
        epochs=15,
        patience=3,
        batch_size=1,
        pca_max_samples=1024,
        max_train_pairs=4096,
        max_validation_pairs=512,
        max_test_pairs=512,
        max_autoregressive_pairs=128,
        force=args.force_final,
    )

    final_rows = []
    for task in TASKS:
        for k in selected_ranks:
            final_rows.append(stage1.run_one(final_args, task, LAYER, k, SEED))

    stage1.print_table(
        ["task", "k", "pair_budget", "best_epoch", "TF-IIA", "AR-IIA", "hash"],
        [
            [
                row["task"],
                row["k"],
                "4096/512/512/128",
                row.get("best_epoch"),
                stage1.fmt(row.get("variable_teacher_forced_iia")),
                stage1.fmt(row.get("autoregressive_iia")),
                row["config_hash"],
            ]
            for row in final_rows
        ],
    )
    return final_rows


def main() -> int:
    args = parse_args()
    if args.position != "-1":
        raise ValueError("This check is fixed to final pre-generation [/INST], position -1.")
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.final_only:
        selected_ranks = tuple(sorted(args.final_ranks))
        final_rows = run_final_budget(args, selected_ranks)
        recommended = recommend_rank(
            final_rows,
            selected_ranks,
            args.final_retention_ratio,
        )
        print(
            f"\nRecommendation: use K{recommended} as the smallest rank that captures "
            f"essentially all useful AR-IIA under this final-budget comparison.",
            flush=True,
        )
        stage1.write_phase_summary(
            args,
            "rank_extension_final_budget",
            final_rows,
            {
                "final_only": True,
                "selected_layer": LAYER,
                "selected_final_ranks": list(selected_ranks),
                "recommended_k": recommended,
                "final_retention_ratio": args.final_retention_ratio,
            },
        )
        stage1.write_json(
            args.output_root / "rank_extension_final_budget_recommendation.json",
            {
                "selected_layer": LAYER,
                "selected_final_ranks": list(selected_ranks),
                "recommended_k": recommended,
            },
        )
        return 0

    light_args = make_stage_args(
        args,
        phase="rank_extension_light",
        epochs=4,
        patience=1,
        batch_size=2,
        pca_max_samples=256,
        max_train_pairs=512,
        max_validation_pairs=128,
        max_test_pairs=128,
        max_autoregressive_pairs=32,
        force=args.force_k128,
    )

    print("STEP 1: lightweight K128 rank extension", flush=True)
    print(
        "fixed config: tasks=T+,I+ layer=37 position=-1 seed=0 "
        "train/val/test/ar=512/128/128/32",
        flush=True,
    )
    for task in TASKS:
        stage1.run_one(light_args, task, LAYER, 128, SEED)

    rows_by_task_k = {}
    for task in TASKS:
        for k in LIGHT_RANKS:
            if k == 128:
                rows_by_task_k[(task, k)] = cached_metrics(light_args, task, k)
            else:
                rows_by_task_k[(task, k)] = cached_metrics(light_args, task, k)

    print_metric_table("TF-IIA", rows_by_task_k, "variable_teacher_forced_iia")
    print_metric_table("AR-IIA", rows_by_task_k, "autoregressive_iia")

    means = rank_means(rows_by_task_k)
    print("\nMean AR-IIA across T+/I+:", flush=True)
    stage1.print_table(
        ["K", "mean AR-IIA"],
        [[k, stage1.fmt(means[k])] for k in LIGHT_RANKS],
    )

    selected_ranks = choose_final_ranks(means, args.rank_tolerance)
    summary = {
        "lightweight_rows": deepcopy(list(rows_by_task_k.values())),
        "lightweight_ar_means": means,
        "rank_tolerance": args.rank_tolerance,
        "selected_final_ranks": selected_ranks,
    }
    if selected_ranks is None:
        print(
            "\nSTOP: lightweight AR-IIA is essentially flat across ranks; "
            "rank does not appear to be the main limitation.",
            flush=True,
        )
        stage1.write_phase_summary(args, "rank_extension_final_budget", [], summary)
        return 0

    print("\nSTEP 2", flush=True)
    final_rows = run_final_budget(args, selected_ranks)

    recommended = recommend_rank(final_rows, selected_ranks, args.final_retention_ratio)
    print(
        f"\nRecommendation: use K{recommended} as the smallest rank that captures "
        f"essentially all useful AR-IIA under this final-budget comparison.",
        flush=True,
    )

    summary["final_rows"] = final_rows
    summary["recommended_k"] = recommended
    summary["final_retention_ratio"] = args.final_retention_ratio
    stage1.write_phase_summary(args, "rank_extension_final_budget", final_rows, summary)
    stage1.write_json(
        args.output_root / "rank_extension_final_budget_recommendation.json",
        {
            "selected_layer": LAYER,
            "selected_final_ranks": list(selected_ranks),
            "recommended_k": recommended,
            "lightweight_ar_means": means,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
