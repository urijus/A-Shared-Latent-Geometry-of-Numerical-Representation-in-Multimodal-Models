"""Audit whether final L43 DAS causality is concentrated near digit readout geometry.

This script runs three compact analyses on already-trained final DAS subspaces:

1. cumulative_readout_ordering:
   rotate each DAS basis by principal alignment to the centered digit-readout
   span, then evaluate cumulative prefixes/suffixes.
2. readout_ablation_controls:
   compare original DAS, centered-digit-readout ablation, and random
   dimensionality-matched PCA-span ablations.
3. two_digit_diagnostics:
   evaluate first-digit and second-digit behavior on held-out pairs whose
   counterfactual result is exactly two decimal digits.

The code intentionally reuses the repository's existing DAS pair loading,
intervention, teacher-forced evaluation, autoregressive evaluation, and image
helpers. It never retrains DAS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import load_jsonl
from src.geometry.readout import get_output_weight, order_subspace_by_reference
from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
    build_unique_pairs,
    collect_initialization_features,
    evaluate_teacher_forced,
    format_prompt,
    patched_forward,
    random_subspace_from_pca,
    resolve_position,
    split_samples,
    target_answers,
    tokenize_answers,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    answer_token_positions,
    autoregressive_iia_image,
    collect_initialization_features_image,
    evaluate_teacher_forced_image,
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    patched_forward as patched_forward_image,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.arithmetic_reference.das_audit.audit_das import (
    pca_space_with_at_least_k,
    unique_samples_from_pairs,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.geometry.subspaces import torch_load_portable
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_TASKS = ("text_add", "text_sub", "image_add", "image_sub")
TASK_SPECS = {
    "text_add": ("text", "addition", "T+"),
    "text_addition": ("text", "addition", "T+"),
    "text:addition": ("text", "addition", "T+"),
    "t+": ("text", "addition", "T+"),
    "text_sub": ("text", "subtraction", "T-"),
    "text_subtraction": ("text", "subtraction", "T-"),
    "text:subtraction": ("text", "subtraction", "T-"),
    "t-": ("text", "subtraction", "T-"),
    "image_add": ("image", "addition", "I+"),
    "image_addition": ("image", "addition", "I+"),
    "image:addition": ("image", "addition", "I+"),
    "i+": ("image", "addition", "I+"),
    "image_sub": ("image", "subtraction", "I-"),
    "image_subtraction": ("image", "subtraction", "I-"),
    "image:subtraction": ("image", "subtraction", "I-"),
    "i-": ("image", "subtraction", "I-"),
}
EXPERIMENTS = (
    "cumulative_readout_ordering",
    "readout_ablation_controls",
    "two_digit_diagnostics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--audit_root", type=Path, default=Path("results/final_exps/DAS_audit_k_22"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/closing/readout_causal_audit"))
    parser.add_argument("--tasks", nargs="+", default=["all"], help="all, or any of text_add/text_sub/image_add/image_sub")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Final DAS seeds.")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument("--control_condition", default="random_subspace_in_pca_span")
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--k", type=int, default=22)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--text_position", default="17")
    parser.add_argument("--image_position", default="-1")
    parser.add_argument("--target", default="result")
    parser.add_argument("--max_pairs", type=int, default=0, help="0 means use the saved held-out evaluation size.")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--random_controls", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=1729)
    parser.add_argument("--pca_max_samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument("--m_values", type=int, nargs="+")
    parser.add_argument("--smoke_test", action="store_true", help="Use small cumulative m grid and a small pair cap unless explicitly overridden.")
    parser.add_argument("--skip_autoregressive", action="store_true")
    parser.add_argument("--skip_teacher_forced", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Load existing CSVs and skip completed task/seed chunks.")
    parser.add_argument("--iia_tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--reproduction_pair_tolerance",
        type=float,
        default=2.0,
        help=(
            "At m=k, allow this many pair-equivalents of AR IIA difference "
            "between the original basis and a projector-equivalent rotated basis."
        ),
    )
    parser.add_argument("--prompt", default="Output ONLY a number.", help="Image prompt, matching DAS audit default.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--force_cpu_digit_span", action="store_true", help="Kept for compatibility; digit span uses the loaded model output head.")
    return parser.parse_args()


def normalize_tasks(raw_tasks: list[str]) -> list[dict]:
    if any(item.lower() == "all" for item in raw_tasks):
        raw_tasks = list(DEFAULT_TASKS)
    tasks = []
    seen = set()
    for raw in raw_tasks:
        key = raw.lower()
        if key not in TASK_SPECS:
            raise ValueError(f"Unknown task {raw!r}; choose all or one of {DEFAULT_TASKS}.")
        modality, operation, label = TASK_SPECS[key]
        canonical = f"{modality}_{operation}"
        if canonical in seen:
            continue
        seen.add(canonical)
        tasks.append(
            {
                "task": canonical,
                "task_label": label,
                "modality": modality,
                "operation": operation,
            }
        )
    return tasks


def normalize_experiments(raw: list[str]) -> set[str]:
    if "all" in raw:
        return set(EXPERIMENTS)
    return set(raw)


def position_for(args: argparse.Namespace, modality: str) -> str:
    return str(args.image_position if modality == "image" else args.text_position)


def run_dir(args: argparse.Namespace, modality: str, operation: str, condition: str, seed: int) -> Path:
    root = args.audit_root / modality / operation
    if args.target != "result":
        root = root / args.target
    return root / condition / f"split_{args.split_seed}" / f"seed_{seed}"


def read_first_jsonl(path: Path) -> dict:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}.")
    return rows[0]


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_basis(path: Path, hidden_size: int | None = None) -> tuple[torch.Tensor, dict]:
    payload = torch_load_portable(path)
    if "basis" in payload:
        basis = payload["basis"]
    elif "bases" in payload:
        basis = payload["bases"]
        if isinstance(basis, dict):
            if len(basis) != 1:
                raise ValueError(f"{path} has multiple bases; pass a subspace.pt with one final basis.")
            basis = next(iter(basis.values()))
    else:
        raise ValueError(f"{path} does not contain a saved DAS basis.")
    basis = torch.as_tensor(basis).detach().float().squeeze()
    if basis.ndim != 2:
        raise ValueError(f"{path} basis must be rank-2; got {tuple(basis.shape)}.")
    if hidden_size is not None and basis.shape[0] != hidden_size and basis.shape[1] == hidden_size:
        basis = basis.T
    return basis, payload


def orthonormal_basis(matrix: torch.Tensor, name: str, rank: int | None = None) -> torch.Tensor:
    matrix = torch.as_tensor(matrix).detach().float().squeeze()
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix; got {tuple(matrix.shape)}.")
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    if singular_values.numel() == 0:
        raise ValueError(f"{name} has no singular values.")
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular_values.max()
    numerical_rank = int((singular_values > tolerance).sum().item())
    if numerical_rank == 0:
        raise ValueError(f"{name} has numerical rank zero.")
    if rank is not None:
        if numerical_rank < rank:
            raise ValueError(f"{name} has rank {numerical_rank}, expected at least {rank}.")
        numerical_rank = rank
    return left[:, :numerical_rank]


def orthogonality_error(basis: torch.Tensor) -> float:
    eye = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    return float((basis.T @ basis - eye).abs().max())


def projector_error(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first @ first.T - second @ second.T).norm())


def subspace_from_basis(layer: int, basis: torch.Tensor, device: torch.device) -> dict:
    basis = basis.detach().float()
    return {str(layer): DASSubspace(basis.shape[0], basis.shape[1], initial_basis=basis).to(device)}


def sample_lookup(data_path: Path) -> dict:
    samples = {}
    for index, sample in enumerate(load_jsonl(data_path)):
        key = sample.get("sample_id", index)
        samples[key] = sample
        samples[str(key)] = sample
    return samples


def load_heldout_pairs(run_directory: Path, data_path: Path, max_pairs: int) -> list[dict]:
    heldout = run_directory / "heldout_pairs.jsonl"
    if not heldout.exists():
        raise FileNotFoundError(heldout)
    samples = sample_lookup(data_path)
    pairs = []
    for row in load_jsonl(heldout):
        pairs.append(
            {
                "pair_id": row.get("pair_id", len(pairs)),
                "base": samples[row["base_sample_id"]],
                "source": samples[row["source_sample_id"]],
            }
        )
    return pairs[:max_pairs] if max_pairs and max_pairs > 0 else pairs


def metadata_for_run(args: argparse.Namespace, task: dict, seed: int) -> tuple[Path, dict, dict, torch.Tensor]:
    directory = run_dir(args, task["modality"], task["operation"], args.condition, seed)
    result_path = directory / "results.jsonl"
    subspace_path = directory / "subspace.pt"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    if not subspace_path.exists():
        raise FileNotFoundError(subspace_path)
    result_row = read_first_jsonl(result_path)
    if int(result_row.get("layer", -1)) != args.layer:
        raise ValueError(f"{result_path} has layer={result_row.get('layer')}, expected {args.layer}.")
    if int(result_row.get("k", -1)) != args.k:
        raise ValueError(f"{result_path} has k={result_row.get('k')}, expected {args.k}.")
    if str(result_row.get("position")) != position_for(args, task["modality"]):
        raise ValueError(
            f"{result_path} has position={result_row.get('position')}, "
            f"expected {position_for(args, task['modality'])}."
        )
    if result_row.get("hook") != args.hook:
        raise ValueError(f"{result_path} has hook={result_row.get('hook')}, expected {args.hook}.")
    basis, payload = load_basis(subspace_path)
    return directory, result_row, payload, basis


def build_digit_readout_span(model, tokenizer) -> tuple[torch.Tensor, list[dict]]:
    output_weight = get_output_weight(model)
    rows = []
    vectors = []
    for digit in range(10):
        text = str(digit)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        token = tokenizer.convert_ids_to_tokens(token_ids) if len(token_ids) == 1 else []
        rows.append(
            {
                "text": text,
                "token_ids": [int(item) for item in token_ids],
                "token": token[0] if token else None,
                "decoded": decoded,
                "is_single_token": len(token_ids) == 1,
            }
        )
        if len(token_ids) != 1:
            raise ValueError(f"Digit {text!r} is not a single token: {token_ids}.")
        vectors.append(output_weight[int(token_ids[0])])
    digit_vectors = torch.stack(vectors).float()
    centered = digit_vectors - digit_vectors.mean(dim=0, keepdim=True)
    basis = orthonormal_basis(centered.T, "centered digit readout span")
    return basis, rows


def print_digit_readout_table(digit_rows: list[dict], digit_basis: torch.Tensor) -> None:
    print("\nCentered digit-readout span")
    print(f"{'digit':>5} {'token_id':>10} {'token':>18} {'decoded':>12}")
    print("-" * 52)
    for row in digit_rows:
        token_id = row["token_ids"][0] if row["token_ids"] else None
        print(f"{row['text']:>5} {str(token_id):>10} {str(row['token']):>18} {row['decoded']!r:>12}")
    print(f"digit-span rank={digit_basis.shape[1]} orthonormality_error={orthogonality_error(digit_basis):.3e}")


def answer_tokens_text(tokenizer, prompt: str, answer: str, device) -> list[dict]:
    encoding, full_positions, _ = tokenize_answers(
        tokenizer,
        [prompt],
        [answer],
        [(0, len(answer))],
        device,
    )
    ids = encoding["input_ids"][0].detach().cpu().tolist()
    return [
        {
            "position": int(pos),
            "token_id": int(ids[pos]),
            "token": tokenizer.convert_ids_to_tokens([int(ids[pos])])[0],
            "decoded": tokenizer.decode([int(ids[pos])], skip_special_tokens=False),
        }
        for pos in full_positions[0]
    ]


def answer_tokens_image(processor, tokenizer, prompt: str, image, answer: str, device) -> list[dict]:
    encoding, full_positions, _ = answer_token_positions(
        processor,
        [prompt],
        [answer],
        [(0, len(answer))],
        [image],
        device,
    )
    ids = encoding["input_ids"][0].detach().cpu().tolist()
    return [
        {
            "position": int(pos),
            "token_id": int(ids[pos]),
            "token": tokenizer.convert_ids_to_tokens([int(ids[pos])])[0],
            "decoded": tokenizer.decode([int(ids[pos])], skip_special_tokens=False),
        }
        for pos in full_positions[0]
    ]


def print_continuation_tokenization(args, task, tokenizer, processor, model, pairs, data_path) -> None:
    if not pairs:
        return
    base = pairs[0]["base"]
    examples = ["7", "10", "42"]
    print(f"\nContinuation tokenization sanity for {task['task']} using existing answer helper")
    for answer in examples:
        if task["modality"] == "text":
            prompt = format_prompt(tokenizer, base, args.use_chat_template or uses_chat_template(args.model))
            rows = answer_tokens_text(tokenizer, prompt, answer, model.device)
        else:
            data_root = data_path.parent
            prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
            image = load_rgb_image(image_path_for(base, data_root))
            rows = answer_tokens_image(processor, tokenizer, prompt, image, answer, model.device)
        compact = ", ".join(f"{row['token_id']}:{row['decoded']!r}" for row in rows)
        print(f"  answer {answer!r}: {compact}")


def order_das_by_reference(das_basis: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    return order_subspace_by_reference(das_basis, reference)


def project_out(basis: torch.Tensor, remove_span: torch.Tensor, name: str) -> tuple[torch.Tensor, dict]:
    source = orthonormal_basis(basis, f"{name} source", rank=basis.shape[1])
    remove = orthonormal_basis(remove_span, f"{name} remove span")
    projected = source - remove @ (remove.T @ source)
    projected_basis = orthonormal_basis(projected, f"{name} projected basis")
    before = float((remove.T @ source).square().sum())
    after = float((remove.T @ projected_basis).square().sum())
    return projected_basis, {
        "source_rank": int(source.shape[1]),
        "removed_span_rank": int(remove.shape[1]),
        "projected_rank": int(projected_basis.shape[1]),
        "overlap_removed_or_before": before,
        "overlap_after": after,
        "orthogonality_after": float((remove.T @ projected_basis).norm()),
    }


def evaluate_candidate(args, task, model, processor, tokenizer, blocks, pairs, data_path, basis, description: str) -> tuple[float | None, float | None]:
    subspaces = subspace_from_basis(args.layer, basis, model.device)
    tf_iia = None
    ar_iia = None
    if not args.skip_teacher_forced:
        if task["modality"] == "text":
            metrics = evaluate_teacher_forced(
                model=model,
                tokenizer=tokenizer,
                blocks=blocks,
                subspaces=subspaces,
                layers=[args.layer],
                hook_name=args.hook,
                pairs=pairs,
                position=position_for(args, "text"),
                target=args.target,
                use_chat_template=args.use_chat_template or uses_chat_template(args.model),
                batch_size=args.batch_size,
                include_clean=False,
            )
        else:
            metrics = evaluate_teacher_forced_image(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
                subspaces=subspaces,
                layers=[args.layer],
                hook_name=args.hook,
                pairs=pairs,
                data_root=data_path.parent,
                prompt=args.prompt,
                position_strategy=position_for(args, "image"),
                target=args.target,
                enable_thinking=args.enable_thinking,
                batch_size=args.batch_size,
                include_clean=False,
            )
        tf_iia = float(metrics["variable_teacher_forced_iia"])
    if not args.skip_autoregressive:
        if task["modality"] == "text":
            ar_iia = autoregressive_iia(
                model=model,
                tokenizer=tokenizer,
                blocks=blocks,
                subspaces=subspaces,
                layers=[args.layer],
                hook_name=args.hook,
                pairs=pairs,
                position=position_for(args, "text"),
                target=args.target,
                use_chat_template=args.use_chat_template or uses_chat_template(args.model),
                max_new_tokens=args.max_new_tokens,
                description=description,
            )
        else:
            ar_iia = autoregressive_iia_image(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                blocks=blocks,
                subspaces=subspaces,
                layers=[args.layer],
                hook_name=args.hook,
                pairs=pairs,
                data_root=data_path.parent,
                prompt=args.prompt,
                position_strategy=position_for(args, "image"),
                target=args.target,
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
                description=description,
            )
    return ar_iia, tf_iia


def load_control_basis(args, task, seed: int, hidden_size: int) -> torch.Tensor | None:
    directory = run_dir(args, task["modality"], task["operation"], args.control_condition, seed)
    path = directory / "subspace.pt"
    if not path.exists():
        print(f"Control subspace missing, using saved/evaluated control as unavailable: {path}")
        return None
    basis, _payload = load_basis(path, hidden_size)
    return basis


def relative_recovery(value: float | None, control: float | None, full: float | None) -> float | None:
    if value is None or control is None or full is None:
        return None
    denominator = full - control
    if abs(denominator) < 1e-12:
        return None
    return (value - control) / denominator


def m_grid(args, k: int) -> list[int]:
    if args.m_values:
        values = sorted({int(value) for value in args.m_values if 1 <= int(value) <= k})
    elif args.smoke_test:
        values = [1, min(2, k), max(1, k // 2), k]
    else:
        values = list(range(1, k + 1))
    if k not in values:
        values.append(k)
    return sorted(set(values))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def save_readout_outputs(
    args,
    resolved_model,
    digit_basis,
    digit_rows,
    cumulative_rows,
    singular_value_rows,
    ablation_rows,
    diagnostic_rows,
    diagnostic_pair_rows,
    checkpoint_label,
):
    write_csv(args.output_dir / "cumulative_readout_results.csv", cumulative_rows)
    write_csv(args.output_dir / "cumulative_singular_values.csv", singular_value_rows)
    write_csv(args.output_dir / "ablation_control_results.csv", ablation_rows)
    write_csv(args.output_dir / "digit_diagnostic_results.csv", diagnostic_rows)
    write_csv(args.output_dir / "digit_diagnostic_pairs.csv", diagnostic_pair_rows)

    summary = summarize(cumulative_rows, ablation_rows, diagnostic_rows, digit_basis.shape[1])
    write_json(
        args.output_dir / "summary.json",
        {
            "model": resolved_model,
            "layer": args.layer,
            "hook": args.hook,
            "k": args.k,
            "digit_span_rank": digit_basis.shape[1],
            "digit_tokenization": digit_rows,
            "tasks": summary,
            "checkpoint_label": checkpoint_label,
            "config": vars(args),
        },
    )
    print(
        f"CHECKPOINT_READOUT_CAUSAL_AUDIT saved {checkpoint_label}: "
        f"cumulative_rows={len(cumulative_rows)} "
        f"ablation_rows={len(ablation_rows)} "
        f"diagnostic_rows={len(diagnostic_rows)} "
        f"output_dir={args.output_dir}"
    )
    return summary


def existing_rows(args):
    return {
        "cumulative": read_csv_rows(args.output_dir / "cumulative_readout_results.csv"),
        "singular": read_csv_rows(args.output_dir / "cumulative_singular_values.csv"),
        "ablation": read_csv_rows(args.output_dir / "ablation_control_results.csv"),
        "diagnostic": read_csv_rows(args.output_dir / "digit_diagnostic_results.csv"),
        "diagnostic_pairs": read_csv_rows(args.output_dir / "digit_diagnostic_pairs.csv"),
    }


def row_matches_task_seed(row: dict, task_name: str, seed: int) -> bool:
    try:
        return row.get("task") == task_name and int(row.get("das_seed", -1)) == int(seed)
    except (TypeError, ValueError):
        return False


def task_seed_completed(rows_by_kind: dict, experiments: set[str], task_name: str, seed: int) -> bool:
    if "cumulative_readout_ordering" in experiments and not any(
        row_matches_task_seed(row, task_name, seed) for row in rows_by_kind["cumulative"]
    ):
        return False
    if "readout_ablation_controls" in experiments and not any(
        row_matches_task_seed(row, task_name, seed) for row in rows_by_kind["ablation"]
    ):
        return False
    if "two_digit_diagnostics" in experiments and not any(
        row_matches_task_seed(row, task_name, seed) for row in rows_by_kind["diagnostic"]
    ):
        return False
    return True


def run_cumulative_readout_ordering(args, task, model, processor, tokenizer, blocks, pairs, data_path, basis, digit_basis, seed, full_ar, full_tf, control_ar):
    ordered, singular_values, proj_error = order_das_by_reference(basis, digit_basis)
    k = ordered.shape[1]
    q_orth_error = orthogonality_error(ordered)
    if q_orth_error > 1e-4:
        raise RuntimeError(f"Ordered basis is not orthonormal enough: {q_orth_error:.3e}")
    if proj_error > 1e-3:
        raise RuntimeError(f"Rotated DAS projector changed: {proj_error:.3e}")
    print(f"  cumulative basis: Q_orth_error={q_orth_error:.3e} projector_error={proj_error:.3e}")

    rows = []
    spectrum_rows = []
    for index, value in enumerate(singular_values.tolist(), start=1):
        spectrum_rows.append(
            {
                "task": task["task"],
                "task_label": task["task_label"],
                "das_seed": seed,
                "direction_index": index,
                "singular_value": float(value),
                "digit_span_rank": digit_basis.shape[1],
                "k": k,
            }
        )

    for ordering in ("aligned_first", "orthogonal_first"):
        for m in m_grid(args, k):
            if ordering == "aligned_first":
                subset = ordered[:, :m]
                principal_value = float(singular_values[m - 1])
            else:
                subset = ordered[:, k - m :]
                principal_value = float(singular_values[k - m])
            ar_iia, tf_iia = evaluate_candidate(
                args,
                task,
                model,
                processor,
                tokenizer,
                blocks,
                pairs,
                data_path,
                subset,
                f"{task['task']} seed={seed} {ordering} m={m}",
            )
            if m == k and full_ar is not None and ar_iia is not None:
                difference = abs(ar_iia - full_ar)
                one_pair = 1.0 / len(pairs) if pairs else 0.0
                pair_tolerance = args.reproduction_pair_tolerance * one_pair
                material_tolerance = max(args.iia_tolerance, pair_tolerance + 1e-12)
                if difference > material_tolerance:
                    raise RuntimeError(
                        f"{task['task']} seed={seed} {ordering} m={k} AR IIA={ar_iia} "
                        f"does not reproduce full DAS AR IIA={full_ar}; "
                        f"difference={difference:.6f} exceeds tolerance={material_tolerance:.6f}."
                    )
                if difference > args.iia_tolerance:
                    print(
                        f"  warning: {task['task']} seed={seed} {ordering} m={k} "
                        f"differs from full DAS by {difference:.6f} "
                        f"({difference / one_pair:.1f} pair-equivalents); continuing."
                    )
            rows.append(
                {
                    "task": task["task"],
                    "task_label": task["task_label"],
                    "modality": task["modality"],
                    "operation": task["operation"],
                    "das_seed": seed,
                    "ordering": ordering,
                    "m": m,
                    "fraction_m": m / k,
                    "principal_value_m": principal_value,
                    "raw_ar_iia": ar_iia,
                    "raw_tf_iia": tf_iia,
                    "control_ar_iia": control_ar,
                    "full_das_ar_iia": full_ar,
                    "full_das_tf_iia": full_tf,
                    "relative_ar_recovery": relative_recovery(ar_iia, control_ar, full_ar),
                    "n_pairs": len(pairs),
                    "layer": args.layer,
                    "position": position_for(args, task["modality"]),
                    "hook": args.hook,
                    "k": k,
                    "digit_span_rank": digit_basis.shape[1],
                    "basis_projector_error": proj_error,
                }
            )
    return rows, spectrum_rows


def reconstruct_train_pairs(args, result_row: dict, seed: int) -> list[dict]:
    data_path = Path(result_row["data_path"])
    samples = load_jsonl(data_path)
    config = result_row.get("config", {})
    train_fraction = float(config.get("train_fraction", 0.7))
    validation_fraction = float(config.get("validation_fraction", 0.15))
    max_train_pairs = int(
        result_row.get("pair_statistics", {})
        .get("train", {})
        .get("selected_unique_pairs", config.get("max_train_pairs", 4096))
    )
    train_samples, _validation_samples, _test_samples = split_samples(
        samples, train_fraction, validation_fraction, args.split_seed
    )
    train_pairs, stats = build_unique_pairs(train_samples, args.target, args.split_seed, max_train_pairs)
    print(f"  reconstructed train pairs for PCA span: n={len(train_pairs)} stats={stats}")
    return train_pairs


def collect_pca_space(args, task, model, processor, tokenizer, blocks, result_row, train_pairs, hidden_size):
    pca_samples = unique_samples_from_pairs(train_pairs)
    min_components = max(args.k, 10)
    if task["modality"] == "text":
        features = collect_initialization_features(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            samples=pca_samples,
            layers=[args.layer],
            hook_name=args.hook,
            position=position_for(args, "text"),
            use_chat_template=args.use_chat_template or uses_chat_template(args.model),
            batch_size=args.batch_size,
            max_samples=args.pca_max_samples,
        )[args.layer]
    else:
        features = collect_initialization_features_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            samples=pca_samples,
            layers=[args.layer],
            hook_name=args.hook,
            data_root=Path(result_row["data_path"]).parent,
            prompt=args.prompt,
            position_strategy=position_for(args, "image"),
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            max_samples=args.pca_max_samples,
        )[args.layer]
    pca_space = pca_space_with_at_least_k(features, min_components, args.pca_variance_threshold)
    if pca_space.shape[0] != hidden_size:
        raise ValueError(f"PCA space d_model={pca_space.shape[0]} != hidden_size={hidden_size}.")
    print(f"  random ablation control space: retained PCA span with rank {pca_space.shape[1]}")
    return pca_space


def random_span_in_pca(pca_space: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        span = random_subspace_from_pca(pca_space, rank)
    finally:
        torch.random.set_rng_state(state)
    return orthonormal_basis(span, "random PCA ablation span", rank=rank)


def run_readout_ablation_controls(args, task, model, processor, tokenizer, blocks, pairs, data_path, result_row, basis, digit_basis, seed, full_ar, full_tf, hidden_size):
    rows = []
    original = orthonormal_basis(basis, "original DAS", rank=args.k)
    digit_ablated, digit_metrics = project_out(original, digit_basis, "digit readout ablation")
    if digit_metrics["orthogonality_after"] > 1e-3:
        raise RuntimeError(f"Digit ablation did not remove readout span: {digit_metrics['orthogonality_after']:.3e}")

    conditions = [
        ("original", None, original, {"removed_span_rank": None, "overlap_removed_or_before": float((digit_basis.T @ original).square().sum())}),
        ("digit_ablated", None, digit_ablated, digit_metrics),
    ]
    for condition, random_seed, candidate, metrics in conditions:
        ar_iia, tf_iia = (full_ar, full_tf) if condition == "original" else evaluate_candidate(
            args,
            task,
            model,
            processor,
            tokenizer,
            blocks,
            pairs,
            data_path,
            candidate,
            f"{task['task']} seed={seed} {condition}",
        )
        rows.append(
            {
                "task": task["task"],
                "task_label": task["task_label"],
                "modality": task["modality"],
                "operation": task["operation"],
                "das_seed": seed,
                "condition": condition,
                "random_seed": random_seed,
                "removed_span_rank": metrics.get("removed_span_rank"),
                "raw_ar_iia": ar_iia,
                "raw_tf_iia": tf_iia,
                "original_ar_iia": full_ar,
                "overlap_removed_or_before": metrics.get("overlap_removed_or_before"),
                "overlap_after": metrics.get("overlap_after"),
                "orthogonality_after": metrics.get("orthogonality_after"),
                "projected_rank": metrics.get("projected_rank", original.shape[1]),
                "n_pairs": len(pairs),
                "layer": args.layer,
                "position": position_for(args, task["modality"]),
                "hook": args.hook,
                "k": original.shape[1],
                "digit_span_rank": digit_basis.shape[1],
            }
        )

    train_pairs = reconstruct_train_pairs(args, result_row, seed)
    pca_space = collect_pca_space(args, task, model, processor, tokenizer, blocks, result_row, train_pairs, hidden_size)
    for index in range(args.random_controls):
        random_seed = args.random_seed + 1000 * seed + index
        random_remove = random_span_in_pca(pca_space, digit_basis.shape[1], random_seed)
        random_ablated, random_metrics = project_out(original, random_remove, "random PCA ablation")
        if random_metrics["orthogonality_after"] > 1e-3:
            raise RuntimeError(f"Random ablation did not remove sampled span: {random_metrics['orthogonality_after']:.3e}")
        ar_iia, tf_iia = evaluate_candidate(
            args,
            task,
            model,
            processor,
            tokenizer,
            blocks,
            pairs,
            data_path,
            random_ablated,
            f"{task['task']} seed={seed} random ablation {index + 1}/{args.random_controls}",
        )
        rows.append(
            {
                "task": task["task"],
                "task_label": task["task_label"],
                "modality": task["modality"],
                "operation": task["operation"],
                "das_seed": seed,
                "condition": "random_ablated",
                "random_seed": random_seed,
                "removed_span_rank": random_metrics["removed_span_rank"],
                "raw_ar_iia": ar_iia,
                "raw_tf_iia": tf_iia,
                "original_ar_iia": full_ar,
                "overlap_removed_or_before": random_metrics["overlap_removed_or_before"],
                "overlap_after": random_metrics["overlap_after"],
                "orthogonality_after": random_metrics["orthogonality_after"],
                "projected_rank": random_metrics["projected_rank"],
                "n_pairs": len(pairs),
                "layer": args.layer,
                "position": position_for(args, task["modality"]),
                "hook": args.hook,
                "k": original.shape[1],
                "digit_span_rank": digit_basis.shape[1],
                "control_space": "retained_pca_span_recomputed_from_training_pairs",
            }
        )
    return rows, digit_ablated


def parse_int_prefix(text: str) -> str | None:
    match = re.match(r"\s*(-?\d+)", text)
    return match.group(1) if match else None


@torch.no_grad()
def autoregressive_outputs_text(args, model, tokenizer, blocks, pairs, basis, description):
    subspaces = subspace_from_basis(args.layer, basis, model.device)
    rows = []
    correct = 0
    first_correct = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for pair in tqdm(pairs, desc=description):
            base, source = pair["base"], pair["source"]
            donor = pair.get("donor", source)
            expected = target_answers(base, source, args.target)[1]
            base_prompt = format_prompt(tokenizer, base, args.use_chat_template or uses_chat_template(args.model))
            donor_prompt = format_prompt(tokenizer, donor, args.use_chat_template or uses_chat_template(args.model))
            base_position = resolve_position(tokenizer, base_prompt, position_for(args, "text"))
            donor_position = resolve_position(tokenizer, donor_prompt, position_for(args, "text"))
            base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            donor_ids = tokenizer(donor_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            generated = []
            for _ in range(args.max_new_tokens):
                base_length, donor_length = len(base_ids), len(donor_ids)
                length = max(base_length, donor_length)
                input_ids = torch.full((2, length), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
                attention_mask = torch.zeros_like(input_ids)
                input_ids[0, :base_length] = base_ids
                input_ids[1, :donor_length] = donor_ids
                attention_mask[0, :base_length] = 1
                attention_mask[1, :donor_length] = 1
                outputs = patched_forward(
                    model,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    blocks,
                    subspaces,
                    [args.layer],
                    args.hook,
                    [base_position],
                    [donor_position],
                    n_base_groups=1,
                )
                next_id = outputs.logits[0, base_length - 1].argmax().reshape(1)
                generated.append(int(next_id.item()))
                base_ids = torch.cat([base_ids, next_id])
                if next_id.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            parsed = parse_int_prefix(text)
            full_ok = parsed == expected
            first_ok = parsed is not None and parsed.lstrip("-")[:1] == expected[:1]
            correct += int(full_ok)
            first_correct += int(first_ok)
            rows.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "base_result": int(base["result"]),
                    "donor_result": int(source["result"]),
                    "generated_answer": text,
                    "parsed_answer": parsed,
                    "target_first_digit": expected[0],
                    "target_second_digit": expected[1] if len(expected) > 1 else None,
                    "generated_first_digit": None if parsed is None else parsed.lstrip("-")[:1],
                    "first_digit_ar_correct": first_ok,
                    "full_ar_correct": full_ok,
                }
            )
    finally:
        tokenizer.padding_side = old_padding_side
    n = len(pairs)
    return (first_correct / n if n else 0.0), (correct / n if n else 0.0), rows


@torch.no_grad()
def autoregressive_outputs_image(args, model, processor, tokenizer, blocks, pairs, data_root, basis, description):
    subspaces = subspace_from_basis(args.layer, basis, model.device)
    rows = []
    correct = 0
    first_correct = 0
    for pair in tqdm(pairs, desc=description):
        base, source = pair["base"], pair["source"]
        donor = pair.get("donor", source)
        expected = target_answers(base, source, args.target)[1]
        base_prompt = sample_prompt(processor, base, args.prompt, args.enable_thinking)
        donor_prompt = sample_prompt(processor, donor, args.prompt, args.enable_thinking)
        base_image = load_rgb_image(image_path_for(base, data_root))
        donor_image = load_rgb_image(image_path_for(donor, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [base_prompt], [base_image], position_for(args, "image"))[0]
        donor_position = resolve_batch_positions(processor, tokenizer, model, [donor_prompt], [donor_image], position_for(args, "image"))[0]
        inputs = inputs_to_device(make_inputs(processor, [base_prompt, donor_prompt], [base_image, donor_image]), model.device)
        generated = []
        for _ in range(args.max_new_tokens):
            base_length = int(inputs["attention_mask"][0].sum())
            outputs = patched_forward_image(
                model,
                inputs,
                blocks,
                subspaces,
                [args.layer],
                args.hook,
                [base_position],
                [donor_position],
                n_base_groups=1,
            )
            next_id = outputs.logits[0, base_length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=model.device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, base_length] = next_id.item()
            inputs["attention_mask"][0, base_length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        parsed = parse_int_prefix(text)
        full_ok = parsed == expected
        first_ok = parsed is not None and parsed.lstrip("-")[:1] == expected[:1]
        correct += int(full_ok)
        first_correct += int(first_ok)
        rows.append(
            {
                "pair_id": pair.get("pair_id"),
                "base_result": int(base["result"]),
                "donor_result": int(source["result"]),
                "generated_answer": text,
                "parsed_answer": parsed,
                "target_first_digit": expected[0],
                "target_second_digit": expected[1] if len(expected) > 1 else None,
                "generated_first_digit": None if parsed is None else parsed.lstrip("-")[:1],
                "first_digit_ar_correct": first_ok,
                "full_ar_correct": full_ok,
            }
        )
    n = len(pairs)
    return (first_correct / n if n else 0.0), (correct / n if n else 0.0), rows


@torch.no_grad()
def digit_teacher_forced_text(args, model, tokenizer, blocks, pairs, basis):
    subspaces = subspace_from_basis(args.layer, basis, model.device)
    first_exact = []
    second_exact = []
    pair_rows = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        base_prompts, donor_prompts, answers = [], [], []
        base_positions, donor_positions = [], []
        for pair in batch:
            base, source = pair["base"], pair["source"]
            donor = pair.get("donor", source)
            answer = target_answers(base, source, args.target)[1]
            if len(answer) != 2:
                raise ValueError(f"Digit diagnostic expected two-digit answer, got {answer!r}.")
            base_prompt = format_prompt(tokenizer, base, args.use_chat_template or uses_chat_template(args.model))
            donor_prompt = format_prompt(tokenizer, donor, args.use_chat_template or uses_chat_template(args.model))
            base_prompts.append(base_prompt)
            donor_prompts.append(donor_prompt)
            answers.append(answer)
            base_positions.append(resolve_position(tokenizer, base_prompt, position_for(args, "text")))
            donor_positions.append(resolve_position(tokenizer, donor_prompt, position_for(args, "text")))
        prompts = base_prompts + donor_prompts
        full_answers = answers + answers
        spans = [(0, len(answer)) for answer in full_answers]
        encoding, full_positions, _ = tokenize_answers(tokenizer, prompts, full_answers, spans, model.device)
        for answer, positions in zip(answers, full_positions[: len(batch)]):
            if len(positions) != 2:
                raise ValueError(f"Answer {answer!r} did not tokenize as two answer tokens: {positions}.")
        outputs = patched_forward(
            model,
            encoding,
            blocks,
            subspaces,
            [args.layer],
            args.hook,
            base_positions,
            donor_positions,
            n_base_groups=1,
        )
        for row, positions in enumerate(full_positions[: len(batch)]):
            first_pos, second_pos = positions
            first_target = encoding["input_ids"][row, first_pos]
            second_target = encoding["input_ids"][row, second_pos]
            first_pred = outputs.logits[row, first_pos - 1].argmax()
            second_pred = outputs.logits[row, second_pos - 1].argmax()
            first_ok = bool(first_pred == first_target)
            second_ok = bool(second_pred == second_target)
            first_exact.append(float(first_ok))
            second_exact.append(float(second_ok))
            pair_rows.append(
                {
                    "pair_id": batch[row].get("pair_id"),
                    "first_digit_tf_correct": first_ok,
                    "second_digit_tf_correct": second_ok,
                }
            )
    return mean(first_exact), mean(second_exact), pair_rows


@torch.no_grad()
def digit_teacher_forced_image(args, model, processor, tokenizer, blocks, pairs, data_root, basis):
    subspaces = subspace_from_basis(args.layer, basis, model.device)
    first_exact = []
    second_exact = []
    pair_rows = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        base_prompts, donor_prompts, answers = [], [], []
        base_images, donor_images = [], []
        for pair in batch:
            base, source = pair["base"], pair["source"]
            donor = pair.get("donor", source)
            answer = target_answers(base, source, args.target)[1]
            if len(answer) != 2:
                raise ValueError(f"Digit diagnostic expected two-digit answer, got {answer!r}.")
            base_prompts.append(sample_prompt(processor, base, args.prompt, args.enable_thinking))
            donor_prompts.append(sample_prompt(processor, donor, args.prompt, args.enable_thinking))
            answers.append(answer)
            base_images.append(load_rgb_image(image_path_for(base, data_root)))
            donor_images.append(load_rgb_image(image_path_for(donor, data_root)))
        base_positions = resolve_batch_positions(processor, tokenizer, model, base_prompts, base_images, position_for(args, "image"))
        donor_positions = resolve_batch_positions(processor, tokenizer, model, donor_prompts, donor_images, position_for(args, "image"))
        prompts = base_prompts + donor_prompts
        images = base_images + donor_images
        full_answers = answers + answers
        spans = [(0, len(answer)) for answer in full_answers]
        encoding, full_positions, _ = answer_token_positions(processor, prompts, full_answers, spans, images, model.device)
        for answer, positions in zip(answers, full_positions[: len(batch)]):
            if len(positions) != 2:
                raise ValueError(f"Answer {answer!r} did not tokenize as two answer tokens: {positions}.")
        outputs = patched_forward_image(
            model,
            encoding,
            blocks,
            subspaces,
            [args.layer],
            args.hook,
            base_positions,
            donor_positions,
            n_base_groups=1,
        )
        for row, positions in enumerate(full_positions[: len(batch)]):
            first_pos, second_pos = positions
            first_target = encoding["input_ids"][row, first_pos]
            second_target = encoding["input_ids"][row, second_pos]
            first_pred = outputs.logits[row, first_pos - 1].argmax()
            second_pred = outputs.logits[row, second_pos - 1].argmax()
            first_ok = bool(first_pred == first_target)
            second_ok = bool(second_pred == second_target)
            first_exact.append(float(first_ok))
            second_exact.append(float(second_ok))
            pair_rows.append(
                {
                    "pair_id": batch[row].get("pair_id"),
                    "first_digit_tf_correct": first_ok,
                    "second_digit_tf_correct": second_ok,
                }
            )
    return mean(first_exact), mean(second_exact), pair_rows


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def maybe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def run_two_digit_diagnostics(args, task, model, processor, tokenizer, blocks, pairs, data_path, basis, digit_ablated, seed):
    two_digit_pairs = [
        pair for pair in pairs if re.fullmatch(r"\d\d", target_answers(pair["base"], pair["source"], args.target)[1])
    ]
    if not two_digit_pairs:
        print(f"  no two-digit held-out pairs for {task['task']} seed={seed}; skipping diagnostics")
        return [], []
    rows = []
    pair_rows = []
    conditions = [("original", basis), ("digit_ablated", digit_ablated)]
    for condition, candidate in conditions:
        if task["modality"] == "text":
            first_ar, full_ar, generated_rows = autoregressive_outputs_text(
                args,
                model,
                tokenizer,
                blocks,
                two_digit_pairs,
                candidate,
                f"{task['task']} seed={seed} {condition} two-digit AR",
            )
            first_tf, second_tf, tf_rows = digit_teacher_forced_text(args, model, tokenizer, blocks, two_digit_pairs, candidate)
        else:
            first_ar, full_ar, generated_rows = autoregressive_outputs_image(
                args,
                model,
                processor,
                tokenizer,
                blocks,
                two_digit_pairs,
                data_path.parent,
                candidate,
                f"{task['task']} seed={seed} {condition} two-digit AR",
            )
            first_tf, second_tf, tf_rows = digit_teacher_forced_image(args, model, processor, tokenizer, blocks, two_digit_pairs, data_path.parent, candidate)
        rows.append(
            {
                "task": task["task"],
                "task_label": task["task_label"],
                "modality": task["modality"],
                "operation": task["operation"],
                "das_seed": seed,
                "condition": condition,
                "n_two_digit_pairs": len(two_digit_pairs),
                "first_digit_ar_iia": first_ar,
                "full_answer_ar_iia": full_ar,
                "first_digit_tf_iia": first_tf,
                "second_digit_tf_iia": second_tf,
                "layer": args.layer,
                "position": position_for(args, task["modality"]),
                "hook": args.hook,
                "k": candidate.shape[1],
            }
        )
        tf_lookup = {row["pair_id"]: row for row in tf_rows}
        for row in generated_rows:
            pair_rows.append(
                {
                    "task": task["task"],
                    "task_label": task["task_label"],
                    "das_seed": seed,
                    "condition": condition,
                    **row,
                    **tf_lookup.get(row.get("pair_id"), {}),
                }
            )
    return rows, pair_rows


def aggregate_mean_sd(rows: list[dict], value_key: str, group_keys: list[str]) -> list[dict]:
    grouped = {}
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        key = tuple(row.get(item) for item in group_keys)
        grouped.setdefault(key, []).append(float(value))
    output = []
    for key, values in grouped.items():
        item = {name: key[index] for index, name in enumerate(group_keys)}
        item[f"{value_key}_mean"] = mean(values)
        item[f"{value_key}_sd"] = 0.0 if len(values) == 1 else float(statistics.stdev(values))
        item["n"] = len(values)
        output.append(item)
    return output


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_cumulative(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    plt = pyplot()
    tasks = ["text_addition", "text_subtraction", "image_addition", "image_subtraction"]
    labels = {"text_addition": "T+", "text_subtraction": "T-", "image_addition": "I+", "image_subtraction": "I-"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, task_name in zip(axes, tasks):
        task_rows = [row for row in rows if row["task"] == task_name and row.get("relative_ar_recovery") is not None]
        ax.set_title(labels[task_name])
        for ordering, label in (("aligned_first", "readout-aligned first"), ("orthogonal_first", "readout-orthogonal first")):
            grouped = {}
            for row in task_rows:
                if row["ordering"] == ordering:
                    grouped.setdefault(row["m"], []).append(float(row["relative_ar_recovery"]))
            if not grouped:
                continue
            x_values = []
            means = []
            lows = []
            highs = []
            for m in sorted(grouped):
                values = grouped[m]
                k = next(row["k"] for row in task_rows if row["m"] == m)
                x_values.append(m / k)
                mu = mean(values)
                sd = 0.0 if len(values) == 1 else statistics.stdev(values)
                means.append(mu)
                lows.append(mu - sd)
                highs.append(mu + sd)
            line = ax.plot(x_values, means, marker="o", linewidth=1.5, markersize=3.5, label=label)[0]
            ax.fill_between(x_values, lows, highs, color=line.get_color(), alpha=0.15, linewidth=0)
        full_values = [float(row["full_das_ar_iia"]) for row in task_rows if row.get("full_das_ar_iia") is not None]
        if full_values:
            ax.text(0.03, 0.92, f"full AR IIA {mean(full_values):.2f}", transform=ax.transAxes, fontsize=8)
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.25)
    for ax in axes[2:]:
        ax.set_xlabel("m / 22")
    for ax in axes[::2]:
        ax.set_ylabel("fraction of full DAS effect")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "cumulative_readout_recovery.png", dpi=220)
    fig.savefig(output_dir / "cumulative_readout_recovery.pdf")
    plt.close(fig)


def plot_ablation(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    plt = pyplot()
    tasks = ["text_addition", "text_subtraction", "image_addition", "image_subtraction"]
    labels = {"text_addition": "T+", "text_subtraction": "T-", "image_addition": "I+", "image_subtraction": "I-"}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), sharey=True)
    axes = axes.ravel()
    conditions = ["original", "digit_ablated", "random_ablated"]
    for ax, task_name in zip(axes, tasks):
        task_rows = [row for row in rows if row["task"] == task_name and row.get("raw_ar_iia") is not None]
        ax.set_title(labels[task_name])
        for index, condition in enumerate(conditions):
            values = [float(row["raw_ar_iia"]) for row in task_rows if row["condition"] == condition]
            if not values:
                continue
            if condition == "random_ablated":
                jitter = [index + random.Random(i).uniform(-0.12, 0.12) for i in range(len(values))]
                ax.scatter(jitter, values, s=14, alpha=0.45, label="random PCA-span ablations")
                ax.hlines(mean(values), index - 0.25, index + 0.25, color="0.25", linewidth=1.5)
            else:
                ax.scatter([index] * len(values), values, s=28, label=condition.replace("_", " "))
                ax.hlines(mean(values), index - 0.25, index + 0.25, color="0.25", linewidth=1.5)
        ax.set_xticks(range(len(conditions)), ["original", "digit\nablated", "random\nablated"])
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[::2]:
        ax.set_ylabel("raw AR IIA")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "readout_ablation_control.png", dpi=220)
    fig.savefig(output_dir / "readout_ablation_control.pdf")
    plt.close(fig)


def plot_digit_diagnostics(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    plt = pyplot()
    metrics = [
        ("first_digit_ar_iia", "first digit AR"),
        ("full_answer_ar_iia", "full answer AR"),
        ("first_digit_tf_iia", "first digit TF"),
        ("second_digit_tf_iia", "second digit TF"),
    ]
    tasks = ["text_addition", "text_subtraction", "image_addition", "image_subtraction"]
    labels = {"text_addition": "T+", "text_subtraction": "T-", "image_addition": "I+", "image_subtraction": "I-"}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0), sharey=True)
    axes = axes.ravel()
    for ax, task_name in zip(axes, tasks):
        task_rows = [row for row in rows if row["task"] == task_name]
        ax.set_title(labels[task_name])
        width = 0.35
        for cond_index, condition in enumerate(("original", "digit_ablated")):
            values = []
            for metric, _label in metrics:
                metric_values = [float(row[metric]) for row in task_rows if row["condition"] == condition and row.get(metric) is not None]
                values.append(mean(metric_values) if metric_values else 0.0)
            offsets = [i + (cond_index - 0.5) * width for i in range(len(metrics))]
            ax.bar(offsets, values, width=width, label=condition.replace("_", " "))
        ax.set_xticks(range(len(metrics)), [label for _metric, label in metrics], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[::2]:
        ax.set_ylabel("IIA / success")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "digit_diagnostics.png", dpi=220)
    fig.savefig(output_dir / "digit_diagnostics.pdf")
    plt.close(fig)


def summarize(cumulative_rows, ablation_rows, diagnostic_rows, digit_rank: int) -> dict:
    summary = {}
    tasks = sorted({row["task"] for row in cumulative_rows + ablation_rows + diagnostic_rows})
    for task in tasks:
        task_summary = {}
        full_values = [
            maybe_float(row.get("full_das_ar_iia"))
            for row in cumulative_rows
            if row["task"] == task
            and maybe_float(row.get("full_das_ar_iia")) is not None
            and int(row["m"]) == int(row["k"])
        ]
        full_values = [value for value in full_values if value is not None]
        task_summary["full_das_ar_iia_mean"] = mean(full_values)
        task_summary["full_das_ar_iia_sd"] = None if len(full_values) < 2 else statistics.stdev(full_values)
        for selected_m in (digit_rank, 22):
            selected = [
                maybe_float(row.get("relative_ar_recovery"))
                for row in cumulative_rows
                if row["task"] == task
                and row["ordering"] == "aligned_first"
                and int(row["m"]) == selected_m
                and maybe_float(row.get("relative_ar_recovery")) is not None
            ]
            selected = [value for value in selected if value is not None]
            task_summary[f"aligned_recovery_m_{selected_m}_mean"] = mean(selected)
            task_summary[f"aligned_recovery_m_{selected_m}_sd"] = None if len(selected) < 2 else statistics.stdev(selected)
        digit_ablated = [
            maybe_float(row.get("raw_ar_iia"))
            for row in ablation_rows
            if row["task"] == task and row["condition"] == "digit_ablated" and maybe_float(row.get("raw_ar_iia")) is not None
        ]
        digit_ablated = [value for value in digit_ablated if value is not None]
        random_ablated = [
            maybe_float(row.get("raw_ar_iia"))
            for row in ablation_rows
            if row["task"] == task and row["condition"] == "random_ablated" and maybe_float(row.get("raw_ar_iia")) is not None
        ]
        random_ablated = [value for value in random_ablated if value is not None]
        task_summary["digit_ablated_ar_iia_mean"] = mean(digit_ablated)
        task_summary["digit_ablated_ar_iia_sd"] = None if len(digit_ablated) < 2 else statistics.stdev(digit_ablated)
        task_summary["random_ablation_ar_iia_mean"] = mean(random_ablated)
        task_summary["random_ablation_ar_iia_sd"] = None if len(random_ablated) < 2 else statistics.stdev(random_ablated)
        for condition in ("original", "digit_ablated"):
            for metric in ("first_digit_ar_iia", "full_answer_ar_iia", "first_digit_tf_iia", "second_digit_tf_iia"):
                values = [
                    maybe_float(row.get(metric))
                    for row in diagnostic_rows
                    if row["task"] == task and row["condition"] == condition and maybe_float(row.get(metric)) is not None
                ]
                values = [value for value in values if value is not None]
                task_summary[f"{condition}_{metric}_mean"] = mean(values)
                task_summary[f"{condition}_{metric}_sd"] = None if len(values) < 2 else statistics.stdev(values)
        summary[task] = task_summary
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke_test and args.max_pairs == 0:
        args.max_pairs = 8
    experiments = normalize_experiments(args.experiments)
    tasks = normalize_tasks(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path, resolved_model = resolve_model_for_loading(args.model)
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    hidden_size = get_hidden_size(model)
    digit_basis, digit_rows = build_digit_readout_span(model, tokenizer)
    print_digit_readout_table(digit_rows, digit_basis)

    loaded_rows = existing_rows(args) if args.resume else {
        "cumulative": [],
        "singular": [],
        "ablation": [],
        "diagnostic": [],
        "diagnostic_pairs": [],
    }
    cumulative_rows = loaded_rows["cumulative"]
    singular_value_rows = loaded_rows["singular"]
    ablation_rows = loaded_rows["ablation"]
    diagnostic_rows = loaded_rows["diagnostic"]
    diagnostic_pair_rows = loaded_rows["diagnostic_pairs"]
    if args.resume:
        print(
            "RESUME_READOUT_CAUSAL_AUDIT: "
            f"loaded cumulative={len(cumulative_rows)} "
            f"ablation={len(ablation_rows)} "
            f"diagnostic={len(diagnostic_rows)} rows from {args.output_dir}"
        )

    for task in tasks:
        for seed in args.seeds:
            if args.resume and task_seed_completed(loaded_rows, experiments, task["task"], seed):
                print(f"RESUME_READOUT_CAUSAL_AUDIT skipping completed {task['task']} seed={seed}")
                continue
            directory, result_row, payload, raw_basis = metadata_for_run(args, task, seed)
            data_path = Path(result_row["data_path"])
            pairs = load_heldout_pairs(directory, data_path, args.max_pairs)
            if raw_basis.shape[0] != hidden_size and raw_basis.shape[1] == hidden_size:
                raw_basis = raw_basis.T
            if raw_basis.shape[0] != hidden_size:
                raise ValueError(f"{directory / 'subspace.pt'} basis shape {tuple(raw_basis.shape)} != hidden_size={hidden_size}.")
            basis = orthonormal_basis(raw_basis, "final DAS basis", rank=args.k)
            print(
                f"\nLoaded {task['task']} ({task['task_label']}) seed={seed}: "
                f"layer={args.layer} position={position_for(args, task['modality'])} "
                f"hook={args.hook} DAS shape={tuple(basis.shape)} "
                f"R_orth_error={orthogonality_error(basis):.3e} "
                f"digit_rank={digit_basis.shape[1]} n_pairs={len(pairs)} "
                f"saved_full_AR={result_row.get('autoregressive_iia')}"
            )
            print_continuation_tokenization(args, task, tokenizer, processor, model, pairs, data_path)

            full_ar, full_tf = evaluate_candidate(
                args,
                task,
                model,
                processor,
                tokenizer,
                blocks,
                pairs,
                data_path,
                basis,
                f"{task['task']} seed={seed} original full DAS",
            )
            print(f"  full DAS raw IIA: AR={full_ar} TF={full_tf}")

            control_ar = None
            control_basis = load_control_basis(args, task, seed, hidden_size)
            if control_basis is not None and "cumulative_readout_ordering" in experiments:
                control_basis = orthonormal_basis(control_basis, "random PCA control basis")
                control_ar, _control_tf = evaluate_candidate(
                    args,
                    task,
                    model,
                    processor,
                    tokenizer,
                    blocks,
                    pairs,
                    data_path,
                    control_basis,
                    f"{task['task']} seed={seed} random PCA control",
                )
                print(f"  control raw AR IIA ({args.control_condition})={control_ar}")

            digit_ablated_basis = None
            if "cumulative_readout_ordering" in experiments:
                rows, spectra = run_cumulative_readout_ordering(
                    args,
                    task,
                    model,
                    processor,
                    tokenizer,
                    blocks,
                    pairs,
                    data_path,
                    basis,
                    digit_basis,
                    seed,
                    full_ar,
                    full_tf,
                    control_ar,
                )
                cumulative_rows.extend(rows)
                singular_value_rows.extend(spectra)

            if "readout_ablation_controls" in experiments:
                rows, digit_ablated_basis = run_readout_ablation_controls(
                    args,
                    task,
                    model,
                    processor,
                    tokenizer,
                    blocks,
                    pairs,
                    data_path,
                    result_row,
                    basis,
                    digit_basis,
                    seed,
                    full_ar,
                    full_tf,
                    hidden_size,
                )
                ablation_rows.extend(rows)
            else:
                digit_ablated_basis, metrics = project_out(basis, digit_basis, "digit readout ablation")
                print(f"  digit ablation orthogonality_after={metrics['orthogonality_after']:.3e}")

            if "two_digit_diagnostics" in experiments:
                rows, pair_rows = run_two_digit_diagnostics(
                    args,
                    task,
                    model,
                    processor,
                    tokenizer,
                    blocks,
                    pairs,
                    data_path,
                    basis,
                    digit_ablated_basis,
                    seed,
                )
                diagnostic_rows.extend(rows)
                diagnostic_pair_rows.extend(pair_rows)

            summary = save_readout_outputs(
                args,
                resolved_model,
                digit_basis,
                digit_rows,
                cumulative_rows,
                singular_value_rows,
                ablation_rows,
                diagnostic_rows,
                diagnostic_pair_rows,
                checkpoint_label=f"after_{task['task']}_seed_{seed}",
            )

    summary = save_readout_outputs(
        args,
        resolved_model,
        digit_basis,
        digit_rows,
        cumulative_rows,
        singular_value_rows,
        ablation_rows,
        diagnostic_rows,
        diagnostic_pair_rows,
        checkpoint_label="final",
    )

    if not args.skip_plots:
        plot_cumulative(args.output_dir, cumulative_rows)
        plot_ablation(args.output_dir, ablation_rows)
        plot_digit_diagnostics(args.output_dir, diagnostic_rows)

    print("\nClosing readout-causality audit complete")
    print(f"Output directory: {args.output_dir}")
    for task_name, task_summary in summary.items():
        full = task_summary.get("full_das_ar_iia_mean")
        digit = task_summary.get("digit_ablated_ar_iia_mean")
        random_mean = task_summary.get("random_ablation_ar_iia_mean")
        print(f"  {task_name}: full_AR={full} digit_ablated_AR={digit} random_ablated_AR_mean={random_mean}")


if __name__ == "__main__":
    main()
