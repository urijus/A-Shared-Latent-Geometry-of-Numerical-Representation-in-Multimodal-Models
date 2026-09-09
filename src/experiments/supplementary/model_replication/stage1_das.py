"""Cache-aware Ministral Stage 1 DAS runner.

This file orchestrates Phase B/C/D only.  It reuses the existing DAS audit
trainer/evaluator and normalizes each finished run into a reusable artifact
directory under ``src/experiments/supplementary/model_replication/results/das``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import torch


MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
MODEL_ALIAS = "ministral3_14b_it_bf16"
MODEL_SLUG = "ministral3_14b_it_bf16"
DEFAULT_OUTPUT_ROOT = Path("src/experiments/supplementary/model_replication/results/das")
TASKS = {
    "T+": ("text", "addition", "Tplus"),
    "T-": ("text", "subtraction", "Tminus"),
    "I+": ("image", "addition", "Iplus"),
    "I-": ("image", "subtraction", "Iminus"),
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_data_path(task: str, model_slug: str, fmt: str) -> Path:
    modality, operation, _stem = TASKS[task]
    if modality == "text":
        return Path(
            f"dataset/baseline/{model_slug}/{fmt}/"
            f"model_correct_with_prompt/{operation}_baseline.jsonl"
        )
    return Path(
        f"dataset/baseline_images/{model_slug}/{fmt}/"
        f"model_correct_with_prompt/{operation}/{operation}_images.jsonl"
    )


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:10]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir_name(task: str, layer: int, k: int, epochs: int, seed: int, digest: str) -> str:
    return f"{TASKS[task][2]}_L{layer}_K{k}_E{epochs}_S{seed}_{digest}"


def das_config(args, task: str, layer: int, k: int, seed: int) -> tuple[dict, str, Path]:
    modality, operation, _stem = TASKS[task]
    pair_sampling_seeds = {
        "train": args.split_seed,
        "validation": args.split_seed + 1,
        "test": args.split_seed + 2,
        "optimizer_init": seed + layer * 100 + k,
    }
    config = {
        "model_id": MODEL_ID,
        "model_alias": args.model,
        "model_slug": args.model_slug,
        "task": task,
        "modality": modality,
        "operation": operation,
        "target": args.target,
        "layer": layer,
        "hook": args.hook,
        "intervention_position": args.position,
        "position_note": "final pre-generation [/INST] token via raw offset -1",
        "k": k,
        "optimizer": "adam",
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "seed": seed,
        "split_seed": args.split_seed,
        "pair_sampling_seeds": pair_sampling_seeds,
        "train_pair_count": args.max_train_pairs,
        "validation_pair_count": args.max_validation_pairs,
        "test_pair_count": args.max_test_pairs,
        "max_autoregressive_pairs": args.max_autoregressive_pairs,
        "train_fraction": args.train_fraction,
        "validation_fraction": args.validation_fraction,
        "batch_size": args.batch_size,
        "pca_max_samples": args.pca_max_samples,
        "pca_variance_threshold": args.pca_variance_threshold,
        "max_new_tokens": args.max_new_tokens,
        "format": args.format,
        "image_prompt": args.prompt,
        "enable_thinking": args.enable_thinking,
        "use_chat_template": args.use_chat_template,
    }
    digest = config_hash(config)
    run_dir = args.output_root / run_dir_name(task, layer, k, args.epochs, seed, digest)
    return config, digest, run_dir


def completed_run(run_dir: Path) -> bool:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
    except json.JSONDecodeError:
        return False
    required = ["config.json", "metadata.json", "das_basis.pt", "metrics.json", "training_history.json"]
    return metadata.get("status") == "completed" and all(
        (run_dir / name).exists() for name in required
    )


def find_audit_outputs(run_dir: Path) -> tuple[Path, Path]:
    audit_root = run_dir / "_audit"
    result_paths = sorted(audit_root.glob("**/das_pca_initialized/**/results.jsonl"))
    subspace_paths = sorted(audit_root.glob("**/das_pca_initialized/**/subspace.pt"))
    if len(result_paths) != 1 or len(subspace_paths) != 1:
        raise FileNotFoundError(
            f"Expected one DAS audit result/subspace under {audit_root}; "
            f"found {len(result_paths)} result files and {len(subspace_paths)} subspaces."
        )
    return result_paths[0], subspace_paths[0]


def load_basis(subspace_path: Path, layer: int) -> tuple[torch.Tensor, dict]:
    payload = torch.load(subspace_path, map_location="cpu", weights_only=False)
    basis = payload.get("basis")
    if basis is None and "bases" in payload:
        basis = payload["bases"].get(str(layer), payload["bases"].get(layer))
    if basis is None:
        raise ValueError(f"No DAS basis found in {subspace_path}.")
    return basis.detach().float().cpu(), payload


def orthonormality_error(basis: torch.Tensor) -> float:
    identity = torch.eye(basis.shape[1], dtype=basis.dtype)
    return float((basis.T @ basis - identity).abs().max())


def audit_command(args, task: str, layer: int, k: int, seed: int, run_dir: Path) -> list[str]:
    modality, operation, _stem = TASKS[task]
    data_path = task_data_path(task, args.model_slug, args.format)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing dataset for {task}: {data_path}. Run the Stage 0 accuracy "
            "scripts first so model_correct_with_prompt exists."
        )
    command = [
        sys.executable,
        "-u",
        "-m",
        "src.experiments.arithmetic_reference.das_audit.audit_das",
        "--model",
        args.model,
        "--modality",
        modality,
        "--operation",
        operation,
        "--target",
        args.target,
        "--data_path",
        str(data_path),
        "--data_root",
        str(data_path.parent),
        "--output_dir",
        str(run_dir / "_audit"),
        "--layer",
        str(layer),
        "--position",
        str(args.position),
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
        "--patience",
        str(args.patience),
        "--learning_rate",
        str(args.learning_rate),
        "--pca_max_samples",
        str(args.pca_max_samples),
        "--pca_variance_threshold",
        str(args.pca_variance_threshold),
        "--max_train_pairs",
        str(args.max_train_pairs),
        "--max_validation_pairs",
        str(args.max_validation_pairs),
        "--max_test_pairs",
        str(args.max_test_pairs),
        "--train_fraction",
        str(args.train_fraction),
        "--validation_fraction",
        str(args.validation_fraction),
        "--max_autoregressive_pairs",
        str(args.max_autoregressive_pairs),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--batch_size",
        str(args.batch_size),
        "--prompt",
        args.prompt,
        "--das_only",
    ]
    if args.use_chat_template:
        command.append("--use_chat_template")
    if args.enable_thinking:
        command.append("--enable_thinking")
    return command


def compact_audit_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("[stage]")
        or stripped.startswith("epoch=")
        or stripped.startswith("PCA space:")
        or stripped.startswith("Saved condition folders under:")
        or stripped.startswith("Loading pre-loaded model.")
    )


def run_audit(command: list[str]) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    tail = []
    assert process.stdout is not None
    for line in process.stdout:
        tail.append(line.rstrip())
        tail = tail[-40:]
        if compact_audit_line(line):
            print(line.rstrip(), flush=True)
    return_code = process.wait()
    if return_code != 0:
        print("\nLast audit output before failure:")
        for line in tail:
            print(line)
        raise subprocess.CalledProcessError(return_code, command)


def normalize_artifacts(
    args,
    task: str,
    layer: int,
    k: int,
    seed: int,
    config: dict,
    digest: str,
    run_dir: Path,
) -> dict:
    result_path, subspace_path = find_audit_outputs(run_dir)
    rows = load_jsonl(result_path)
    if len(rows) != 1:
        raise ValueError(f"Expected one DAS result row in {result_path}; got {len(rows)}.")
    metrics = rows[0]
    basis, payload = load_basis(subspace_path, layer)
    history = payload.get("history", [])
    best_epoch = payload.get("best_epoch", metrics.get("best_epoch"))
    basis_path = run_dir / "das_basis.pt"
    torch.save(basis, basis_path)
    shutil.copy2(result_path.parent / "heldout_pairs.jsonl", run_dir / "heldout_pairs.jsonl")
    write_json(run_dir / "config.json", config)
    orth_error = orthonormality_error(basis)
    metrics = {
        **metrics,
        "task": task,
        "config_hash": digest,
        "run_dir": str(run_dir),
        "das_basis_path": str(basis_path),
        "orthonormality_max_error": orth_error,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "training_history.json", {"history": history})
    artifact_hash = file_sha256(basis_path)
    metadata = {
        "status": "completed",
        "phase": args.phase,
        "task": task,
        "layer": layer,
        "k": k,
        "seed": seed,
        "config_hash": digest,
        "artifact_hash": artifact_hash,
        "epochs_completed": len(history) - 1 if history else best_epoch,
        "best_epoch": best_epoch,
        "source_audit_results": str(result_path),
        "source_audit_subspace": str(subspace_path),
        "required_artifacts": [
            "config.json",
            "metadata.json",
            "das_basis.pt",
            "metrics.json",
            "training_history.json",
        ],
    }
    write_json(run_dir / "metadata.json", metadata)
    return metrics


def run_one(args, task: str, layer: int, k: int, seed: int) -> dict:
    config, digest, run_dir = das_config(args, task, layer, k, seed)
    label = f"{task} L{layer} K{k} S{seed}"
    if completed_run(run_dir) and not args.force:
        print(f"[CACHE] {label} -> {digest}")
        return load_json(run_dir / "metrics.json")

    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(
        run_dir / "metadata.json",
        {
            "status": "running",
            "phase": args.phase,
            "task": task,
            "layer": layer,
            "k": k,
            "seed": seed,
            "config_hash": digest,
        },
    )
    print(f"[RUN] {label} -> {digest}", flush=True)
    print(
        "[stage] phase B/C/D audit config: "
        f"epochs={args.epochs}, patience={args.patience}, "
        f"batch_size={args.batch_size}, pca_max_samples={args.pca_max_samples}, "
        f"train_pairs={args.max_train_pairs}, validation_pairs={args.max_validation_pairs}, "
        f"test_pairs={args.max_test_pairs}, autoregressive_pairs={args.max_autoregressive_pairs}",
        flush=True,
    )
    run_audit(audit_command(args, task, layer, k, seed, run_dir))
    metrics = normalize_artifacts(args, task, layer, k, seed, config, digest, run_dir)
    print(
        f"[DONE] {label} TF-IIA={fmt(metrics.get('variable_teacher_forced_iia'))} "
        f"AR-IIA={fmt(metrics.get('autoregressive_iia'))} "
        f"epochs={metrics.get('best_epoch')} hash={digest}"
    )
    return metrics


def fmt(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    return str(value)


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        if rows
        else len(str(header))
        for i, header in enumerate(headers)
    ]
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def write_phase_summary(args, name: str, rows: list[dict], extra: dict) -> None:
    summary_dir = args.output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": name,
        "rows": rows,
        **extra,
    }
    write_json(summary_dir / f"{name}.json", payload)


def dataset_identity_config(config: dict) -> dict:
    """Fields that must stay fixed across DAS optimization seeds."""
    return {
        key: config[key]
        for key in (
            "model_id",
            "model_slug",
            "task",
            "modality",
            "operation",
            "target",
            "layer",
            "hook",
            "intervention_position",
            "k",
            "split_seed",
            "train_pair_count",
            "validation_pair_count",
            "test_pair_count",
            "train_fraction",
            "validation_fraction",
            "format",
        )
    } | {
        "pair_sampling_seeds": {
            "train": config["pair_sampling_seeds"]["train"],
            "validation": config["pair_sampling_seeds"]["validation"],
            "test": config["pair_sampling_seeds"]["test"],
        }
    }


def assert_fixed_dataset_across_seeds(args, task: str, layer: int, k: int) -> None:
    reference = None
    for seed in args.seeds:
        config, _digest, _run_dir = das_config(args, task, layer, k, seed)
        identity = dataset_identity_config(config)
        if reference is None:
            reference = identity
            continue
        if identity != reference:
            raise AssertionError(
                f"Dataset split/pair sampling changed across DAS seeds for {task}."
            )


def phase_b(args) -> int:
    print("PHASE B")
    rows = []
    for task in ("T+", "I+"):
        for layer in (37, 39):
            rows.append(run_one(args, task, layer, 32, 0))

    by_task_layer = {(row["task"], int(row["layer"])): row for row in rows}
    table_rows = []
    for task in ("T+", "I+"):
        for layer in (37, 39):
            row = by_task_layer[(task, layer)]
            table_rows.append(
                [
                    task,
                    layer,
                    fmt(row.get("variable_teacher_forced_iia")),
                    fmt(row.get("autoregressive_iia")),
                    row.get("best_epoch"),
                ]
            )
    print_table(["task", "layer", "TF-IIA", "AR-IIA", "epochs"], table_rows)
    print_table(
        ["Task", "L37 AR-IIA", "L39 AR-IIA"],
        [
            [
                task,
                fmt(by_task_layer[(task, 37)].get("autoregressive_iia")),
                fmt(by_task_layer[(task, 39)].get("autoregressive_iia")),
            ]
            for task in ("T+", "I+")
        ],
    )

    comparable = True
    for task in ("T+", "I+"):
        ar37 = by_task_layer[(task, 37)].get("autoregressive_iia")
        ar39 = by_task_layer[(task, 39)].get("autoregressive_iia")
        if ar37 is None or ar39 is None:
            comparable = False
            continue
        if ar39 > 0 and ar37 < args.layer_comparable_ratio * ar39:
            comparable = False
    selected_layer = 37 if comparable else 39
    print(f"selected_layer = {selected_layer}")
    write_phase_summary(args, "phase_b", rows, {"selected_layer": selected_layer})
    write_json(args.output_root / "selected_layer.json", {"selected_layer": selected_layer})
    return 0


def load_selected_layer(args) -> int:
    if args.selected_layer is not None:
        return args.selected_layer
    path = args.output_root / "selected_layer.json"
    if path.exists():
        return int(load_json(path)["selected_layer"])
    raise ValueError("No selected layer found. Run Phase B or pass --selected-layer.")


def choose_k(args, rows: list[dict]) -> int:
    by_task_k = {(row["task"], int(row["k"])): row for row in rows if row["task"] in {"T+", "I+"}}
    choices = {}
    for task in ("T+", "I+"):
        ref = by_task_k[(task, 64)].get("autoregressive_iia")
        if ref is None:
            raise RuntimeError(f"Cannot choose k: missing K64 AR-IIA for {task}.")
        for k in (22, 32, 64):
            ar = by_task_k[(task, k)].get("autoregressive_iia")
            if ar is not None and ar >= args.k_retention_ratio * ref:
                choices[task] = k
                break
    if len(set(choices.values())) != 1:
        raise RuntimeError(
            f"T+ and I+ disagree on k selection: {choices}. Stopping without expanding grid."
        )
    return choices["T+"]


def phase_c(args) -> int:
    print("PHASE C")
    layer = load_selected_layer(args)
    rows = []
    for k in (22, 32, 64):
        for task in ("T+", "I+"):
            rows.append(run_one(args, task, layer, k, 0))

    by_task_k = {(row["task"], int(row["k"])): row for row in rows}
    table_rows = []
    for task in ("T+", "I+"):
        table_rows.append(
            [
                task,
                fmt(by_task_k[(task, 22)].get("autoregressive_iia")),
                fmt(by_task_k[(task, 32)].get("autoregressive_iia")),
                fmt(by_task_k[(task, 64)].get("autoregressive_iia")),
            ]
        )
    print_table(["task", "K22", "K32", "K64"], table_rows)
    selected_k = args.selected_k or choose_k(args, rows)
    print(f"selected_k = {selected_k}")

    confirmations = []
    for task in ("T-", "I-"):
        confirmations.append(run_one(args, task, layer, selected_k, 0))
    for row in confirmations:
        print(
            f"{row['task']} confirmation = TF-IIA {fmt(row.get('variable_teacher_forced_iia'))}, "
            f"AR-IIA {fmt(row.get('autoregressive_iia'))}"
        )
    all_rows = rows + confirmations
    write_phase_summary(
        args,
        "phase_c",
        all_rows,
        {
            "selected_layer": layer,
            "selected_k": selected_k,
            "confirmations": confirmations,
        },
    )
    write_json(
        args.output_root / "selected_k.json",
        {"selected_layer": layer, "selected_k": selected_k},
    )
    return 0


def load_selected_k(args) -> tuple[int, int]:
    layer = args.selected_layer
    selected_k = args.selected_k
    path = args.output_root / "selected_k.json"
    if path.exists():
        payload = load_json(path)
        layer = layer or int(payload["selected_layer"])
        selected_k = selected_k or int(payload["selected_k"])
    if layer is None or selected_k is None:
        raise ValueError("No selected layer/k found. Run Phase C or pass both values.")
    return int(layer), int(selected_k)


def phase_d(args) -> int:
    print("PHASE D")
    layer, selected_k = load_selected_k(args)
    for task in args.tasks:
        assert_fixed_dataset_across_seeds(args, task, layer, selected_k)
    print(
        "Dataset split/pair seeds fixed across DAS seeds: "
        f"split={args.split_seed}, train={args.split_seed}, "
        f"validation={args.split_seed + 1}, test={args.split_seed + 2}",
        flush=True,
    )
    rows = []
    for task in args.tasks:
        for seed in args.seeds:
            rows.append(run_one(args, task, layer, selected_k, seed))

    table_rows = [
        [
            row["task"],
            row["seed"],
            row["layer"],
            row["k"],
            fmt(row.get("variable_teacher_forced_iia")),
            fmt(row.get("autoregressive_iia")),
            row["config_hash"],
        ]
        for row in rows
    ]
    print_table(["Task", "Seed", "Layer", "K", "TF-IIA", "AR-IIA", "Hash"], table_rows)

    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    for task in args.tasks:
        values = [
            row.get("autoregressive_iia")
            for row in by_task[task]
            if row.get("autoregressive_iia") is not None
        ]
        if values:
            sigma = stdev(values) if len(values) > 1 else 0.0
            print(f"{task} AR-IIA mean +/- std = {mean(values):.3f} +/- {sigma:.3f}")
        else:
            print(f"{task} AR-IIA mean +/- std = NA")

    print("\nFinal DAS artifacts:")
    for row in rows:
        print(f"{row['task']} S{row['seed']} {row['config_hash']} {row['run_dir']}")
        print(f"  orthonormality_max_error = {row['orthonormality_max_error']:.3e}")

    write_phase_summary(
        args,
        "phase_d",
        rows,
        {"selected_layer": layer, "selected_k": selected_k},
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["B", "C", "D"], required=True)
    parser.add_argument("--model", default=MODEL_ALIAS)
    parser.add_argument("--model_slug", "--model-slug", default=MODEL_SLUG)
    parser.add_argument("--format", default="digits")
    parser.add_argument("--output_root", "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selected_layer", "--selected-layer", type=int)
    parser.add_argument("--selected_k", "--selected-k", type=int)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=["T+", "T-", "I+", "I-"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--position", default="-1")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--target", default="result")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning_rate", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
    parser.add_argument("--pca_max_samples", "--pca-max-samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", "--pca-variance-threshold", type=float, default=0.9)
    parser.add_argument("--max_train_pairs", "--max-train-pairs", type=int, default=4096)
    parser.add_argument("--max_validation_pairs", "--max-validation-pairs", type=int, default=512)
    parser.add_argument("--max_test_pairs", "--max-test-pairs", type=int, default=512)
    parser.add_argument("--max_autoregressive_pairs", "--max-autoregressive-pairs", type=int, default=0)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=8)
    parser.add_argument("--train_fraction", "--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", "--validation-fraction", type=float, default=0.15)
    parser.add_argument("--split_seed", "--split-seed", type=int, default=0)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", "--enable-thinking", action="store_true")
    parser.add_argument("--use_chat_template", "--use-chat-template", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--layer_comparable_ratio", "--layer-comparable-ratio", type=float, default=0.9)
    parser.add_argument("--k_retention_ratio", "--k-retention-ratio", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.position != "-1":
        raise ValueError("Stage 1 Phase B/C/D must use final pre-generation position -1.")
    if args.phase == "B":
        return phase_b(args)
    if args.phase == "C":
        return phase_c(args)
    if args.phase == "D":
        return phase_d(args)
    raise AssertionError(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
