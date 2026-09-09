"""Evaluate DAS directions ordered by Fourier alignment.

For a trained DAS basis D and a Fourier probe space F, this script computes

    F^T D = U S V^T

and evaluates autoregressive IIA after replacing the DAS basis by cumulative
prefixes and suffixes of D V.  Prefixes add directions from most Fourier-aligned
to least aligned; suffixes add them in the opposite order.
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from tqdm import tqdm

from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    DASSubspace,
    build_unique_pairs,
    format_prompt,
    patched_forward,
    resolve_position,
    split_samples,
    target_answers,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.probes.fourier import probe_path
from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    model_slug,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_DAS_ROOT = Path("results/final_exps/DAS_audit")
DEFAULT_DAS_BANK_NAME = "das_v2"
DEFAULT_OUTPUT_ROOT = Path("results/final_exps/fourier_not_causal")

BACKGROUND = "#fdfdfd"
SPINE = "#b9c0c2"
TEXT = "#171717"
GRID = "#d8dddd"
VISIBLE = "#5f9b92"
NULL = "#8f4f59"
FULL = "#2f3437"
FOURIER_ALIGNED_DIRECTIONS = 11


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style_matplotlib(plt):
    plt.rcParams.update(
        {
            "font.family": [
                "Palatino Linotype",
                "Georgia",
                "Cambria",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.2,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 7.1,
            "savefig.bbox": "tight",
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def setup_axis(ax):
    ax.set_facecolor(BACKGROUND)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.68)
    ax.grid(axis="x", color=GRID, linewidth=0.45, alpha=0.28)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--operation", default="addition")
    parser.add_argument("--modality", default="text")
    parser.add_argument("--target", default="result")
    parser.add_argument("--fourier_target", help="Defaults to --target.")
    parser.add_argument(
        "--fourier_modality",
        help="Fourier probe modality folder. Defaults to --operation, e.g. addition.",
    )
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--position", default="17")
    parser.add_argument("--fourier_position", help="Defaults to --position.")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--k", type=int, help="DAS dimension; used for bank lookup.")
    parser.add_argument("--condition", default="das_pca_initialized")
    parser.add_argument(
        "--artifact_source",
        choices=["bank", "audit", "explicit"],
        default="bank",
        help="Where to find DAS artifacts. Bank uses results/baseline/<model>/digits/das_v2.",
    )
    parser.add_argument("--das_root", type=Path, default=DEFAULT_DAS_ROOT)
    parser.add_argument("--bank_root", type=Path)
    parser.add_argument("--subspace_path", type=Path)
    parser.add_argument("--results_path", type=Path)
    parser.add_argument("--heldout_pairs_path", type=Path)
    parser.add_argument("--probe_root", type=Path)
    parser.add_argument("--periods", type=int, nargs="+", default=[2, 5, 10, 20, 50, 100])
    parser.add_argument("--method", choices=["ridge", "gd"], default="ridge")
    parser.add_argument("--m_values", type=int, nargs="+")
    parser.add_argument("--max_m", type=int, default=32)
    parser.add_argument("--m_step", type=int, default=4)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=128)
    parser.add_argument("--max_test_pairs", type=int, default=128)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Only build a two-panel comparison plot from --compare inputs.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("LABEL", "RESULT_JSONL"),
        help="Add a model curve file to the comparison plot.",
    )
    return parser.parse_args()


def default_run_dir(args):
    return (
        args.das_root
        / args.modality
        / args.operation
        / args.condition
        / f"split_{args.split_seed}"
        / f"seed_{args.seed}"
    )


def default_bank_root(args):
    return Path("results") / "baseline" / model_slug(args.model) / "digits" / DEFAULT_DAS_BANK_NAME


def row_layer(row):
    if "layer" in row:
        return int(row["layer"])
    layers = row.get("layers")
    if isinstance(layers, list) and len(layers) == 1:
        return int(layers[0])
    return None


def row_k(row):
    for key in ("k", "subspace_dim"):
        if row.get(key) is not None:
            return int(row[key])
    return None


def row_checkpoint_path(row):
    value = row.get("checkpoint_path") or row.get("basis_path")
    return Path(value) if value else None


def optional_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_bank_row(args):
    root = args.bank_root or default_bank_root(args)
    run_files = sorted((root / "runs").glob("*.jsonl"))
    if not run_files:
        raise FileNotFoundError(f"No DAS bank run files found under {root / 'runs'}.")
    candidates = []
    for path in run_files:
        for row in load_jsonl(path):
            checkpoint = row_checkpoint_path(row)
            if checkpoint is not None and not checkpoint.exists():
                checkpoint = root / "subspaces" / checkpoint.name
            if (
                row.get("operation", row.get("modality")) == args.operation
                and row.get("target") == args.target
                and row_layer(row) == args.layer
                and str(row.get("position")) == str(args.position)
                and row.get("hook") == args.hook
                and int(row.get("seed", -1)) == int(args.seed)
                and (args.k is None or row_k(row) == args.k)
                and checkpoint is not None
                and checkpoint.exists()
            ):
                candidates.append((row_k(row) or 0, path, checkpoint, row))
    if not candidates:
        raise FileNotFoundError(
            "No matching DAS bank row. Either train it first, pass explicit "
            "--subspace_path/--results_path, or adjust --layer/--position/--k/--seed."
        )
    _k, path, checkpoint, row = max(candidates, key=lambda item: item[0])
    row = dict(row)
    row["checkpoint_path"] = str(checkpoint)
    row["_bank_results_path"] = str(path)
    return row, path, checkpoint


def resolve_artifact_paths(args):
    if args.artifact_source == "bank" and args.subspace_path is None:
        row, results_path, subspace_path = find_bank_row(args)
        return {
            "subspace": subspace_path,
            "results": results_path,
            "heldout": args.heldout_pairs_path,
            "bank_row": row,
        }

    run_dir = default_run_dir(args)
    return {
        "subspace": args.subspace_path or run_dir / "subspace.pt",
        "results": args.results_path or run_dir / "results.jsonl",
        "heldout": args.heldout_pairs_path or run_dir / "heldout_pairs.jsonl",
        "bank_row": None,
    }


def default_probe_root(args):
    slug = model_slug(args.model)
    return Path("results") / "baseline" / slug / "digits" / "fourier_probes"


def load_basis(path, layer):
    payload = torch_load_portable(path)
    if "basis" in payload:
        basis = payload["basis"]
    elif "bases" in payload:
        basis = payload["bases"].get(layer, payload["bases"].get(str(layer)))
        if basis is None:
            raise ValueError(f"{path} has no saved DAS basis for layer {layer}.")
    else:
        raise ValueError(f"{path} does not contain a saved DAS basis.")
    return torch.as_tensor(basis).detach().float(), payload


def load_run_row(path):
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}.")
    return rows[0]


def sample_lookup(data_path):
    samples = load_jsonl(Path(data_path))
    by_id = {}
    for index, sample in enumerate(samples):
        key = sample.get("sample_id", index)
        by_id[key] = sample
    return by_id


def load_pairs(path, data_path):
    samples = sample_lookup(data_path)
    pairs = []
    for row in load_jsonl(path):
        base_id = row["base_sample_id"]
        source_id = row["source_sample_id"]
        if base_id not in samples or source_id not in samples:
            raise KeyError(f"Held-out pair references missing samples: {row}")
        pairs.append(
            {
                "pair_id": row.get("pair_id", len(pairs)),
                "base": samples[base_id],
                "source": samples[source_id],
            }
        )
    return pairs


def reconstruct_pairs(args, data_path, row, payload):
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    samples = load_jsonl(Path(data_path))
    seed = int(row.get("seed", config.get("seed", args.seed)))
    train_fraction = float(config.get("train_fraction", args.train_fraction))
    validation_fraction = float(
        config.get("validation_fraction", args.validation_fraction)
    )
    max_test_pairs = int(
        row.get(
            "max_test_pairs",
            config.get(
                "max_test_pairs",
                config.get("n_validation_pairs", args.max_test_pairs),
            ),
        )
    )
    _, _, test_samples = split_samples(
        samples, train_fraction, validation_fraction, seed
    )
    test_pairs, stats = build_unique_pairs(
        test_samples, args.target, seed + 2, max_test_pairs
    )
    print(
        "Reconstructed held-out pairs from DAS bank config: "
        f"n={len(test_pairs)}, stats={stats}"
    )
    for pair_id, pair in enumerate(test_pairs):
        pair["pair_id"] = pair_id
    return test_pairs


def load_fourier_space(args, d_model):
    root = args.probe_root or default_probe_root(args)
    target = args.fourier_target or args.target
    fourier_modality = args.fourier_modality or args.operation
    position = args.fourier_position or args.position
    if root.name == fourier_modality and not (root / fourier_modality).exists():
        root = root.parent
    bases = []
    paths = []
    metrics = {}
    for period in args.periods:
        path = probe_path(root, fourier_modality, target, period, args.layer, position, args.method)
        if not path.exists():
            raise FileNotFoundError(
                f"Fourier probe not found: {path}. Run project_fourier first, "
                "or pass --probe_root/--fourier_modality/--fourier_position "
                "matching existing probes."
            )
        artifact = torch_load_portable(path)
        weight = torch.as_tensor(artifact["weight"]).detach().float().squeeze()
        bases.append(orthonormal_columns(weight, d_model, f"Fourier T{period}"))
        paths.append(str(path))
        metrics[str(period)] = artifact.get("metrics", {})
    fourier = torch.cat(bases, dim=1)
    fourier = orthonormal_columns(fourier, d_model, "combined Fourier space")
    return fourier, paths, metrics


def ordered_das_basis(das_basis, fourier_basis):
    d_model = max(das_basis.shape)
    das = orthonormal_columns(das_basis, d_model, "DAS basis")
    _, singular_values, vh = torch.linalg.svd(fourier_basis.T @ das, full_matrices=True)
    principal = das @ vh.T
    if singular_values.numel() < principal.shape[1]:
        singular_values = torch.cat(
            [
                singular_values,
                torch.zeros(
                    principal.shape[1] - singular_values.numel(),
                    dtype=singular_values.dtype,
                    device=singular_values.device,
                ),
            ]
        )
    return principal, singular_values.clamp(0, 1)


def effective_k(k):
    return min(FOURIER_ALIGNED_DIRECTIONS, k)


def null_complement_k(k):
    visible = effective_k(k)
    return max(1, k - visible)


def landmark_ms(k):
    values = {effective_k(k)}
    complement = null_complement_k(k)
    if complement < k:
        values.add(complement)
    return sorted(value for value in values if 1 <= value < k)


def m_grid(k, args):
    if args.m_values:
        values = sorted({value for value in args.m_values if 1 <= value <= k})
    else:
        max_m = min(args.max_m, k)
        values = list(range(1, max_m + 1, max(1, args.m_step)))
        values.extend(value for value in landmark_ms(k) if value <= max_m)
        if max_m not in values:
            values.append(max_m)
    if k not in values:
        values.append(k)
    return sorted(set(values))


def subspace_from_columns(layer, basis, device):
    return {
        str(layer): DASSubspace(
            basis.shape[0], basis.shape[1], initial_basis=basis
        ).to(device)
    }


@torch.no_grad()
def clean_autoregressive_iia(model, tokenizer, pairs, target, use_chat_template, max_new_tokens):
    correct = 0
    for pair in tqdm(pairs, desc="clean autoregressive"):
        base, source = pair["base"], pair["source"]
        prompt = format_prompt(tokenizer, base, use_chat_template)
        expected = target_answers(base, source, target)[1]
        input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(model.device)
        generated = []
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, use_cache=False)
            next_id = outputs.logits[0, input_ids.shape[1] - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            input_ids = torch.cat([input_ids, next_id.to(model.device)], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs) if pairs else 0.0


@torch.no_grad()
def autoregressive_iia_with_outputs(
    model,
    tokenizer,
    blocks,
    subspaces,
    layer,
    hook_name,
    pairs,
    position,
    target,
    use_chat_template,
    max_new_tokens,
    description,
):
    rows = []
    correct = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for pair in tqdm(pairs, desc=description):
            base, source = pair["base"], pair["source"]
            donor = pair.get("donor", source)
            base_prompt = format_prompt(tokenizer, base, use_chat_template)
            donor_prompt = format_prompt(tokenizer, donor, use_chat_template)
            expected = target_answers(base, source, target)[1]
            base_position = resolve_position(tokenizer, base_prompt, position)
            donor_position = resolve_position(tokenizer, donor_prompt, position)
            base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0].to(model.device)
            donor_ids = tokenizer(donor_prompt, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0].to(model.device)
            generated = []
            for _ in range(max_new_tokens):
                base_length, donor_length = len(base_ids), len(donor_ids)
                length = max(base_length, donor_length)
                input_ids = torch.full(
                    (2, length),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=model.device,
                )
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
                    [layer],
                    hook_name,
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
            match = re.match(r"-?\d+", text)
            prediction = match.group() if match else text
            is_correct = prediction == expected
            correct += int(is_correct)
            rows.append(
                {
                    "pair_id": pair.get("pair_id"),
                    "expected": expected,
                    "prediction": prediction,
                    "correct": is_correct,
                }
            )
    finally:
        tokenizer.padding_side = old_padding_side
    return (correct / len(pairs) if pairs else 0.0), rows


def curve_output_dir(args, model_name):
    slug = model_slug(model_name)
    return (
        args.output_dir
        / slug
        / args.modality
        / args.operation
        / args.condition
        / f"layer{args.layer}_pos{args.position}_{args.hook}"
        / f"split_{args.split_seed}"
        / f"seed_{args.seed}"
    )


def eta(value, clean, full):
    denominator = full - clean
    if abs(denominator) < 1e-12:
        return None
    return (value - clean) / denominator


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


def enrich_source_das_fields(rows):
    for row in rows:
        source_row = None
        source_path = row.get("das_results_path")
        if source_path and row.get("source_das_autoregressive_iia") is None:
            path = Path(source_path)
            if path.exists():
                try:
                    source_row = load_run_row(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    source_row = None
        if source_row is not None:
            row["source_das_autoregressive_iia"] = optional_float(
                source_row.get("autoregressive_iia")
            )
            row["source_das_variable_teacher_forced_iia"] = optional_float(
                source_row.get("variable_teacher_forced_iia")
            )
            row["source_das_k"] = row_k(source_row)
        saved_config = row.get("saved_das_config") or {}
        saved_basis_dim = row.get("saved_basis_dim") or row.get("source_das_k")
        saved_basis_dim = saved_basis_dim or saved_config.get("k")
        if saved_basis_dim is not None:
            row["saved_basis_dim"] = int(saved_basis_dim)
            row["truncated_saved_basis"] = int(saved_basis_dim) != int(row["k"])


def add_relative_iia(rows):
    enrich_source_das_fields(rows)
    denominators = {}
    for row in rows:
        if int(row["m"]) == int(row["k"]):
            denominators[row["curve"]] = row["autoregressive_iia"]
    denominator_values = [
        value for value in denominators.values() if value is not None
    ]
    full_reference = max(denominator_values) if denominator_values else None
    for row in rows:
        denominator = full_reference
        row["full_curve_autoregressive_iia"] = denominator
        if full_reference is not None:
            row["full_das_autoregressive_iia"] = full_reference
        row["m_over_k"] = row["m"] / row["k"]
        relative = None
        if denominator is not None and abs(denominator) >= 1e-12:
            relative = row["autoregressive_iia"] / denominator
        row["relative_autoregressive_iia_raw"] = relative
        row["relative_autoregressive_iia"] = (
            None if relative is None else min(relative, 1.0)
        )
        row["eta_vs_full_das"] = (
            eta(
                row["autoregressive_iia"],
                row.get("clean_autoregressive_iia"),
                full_reference,
            )
            if full_reference is not None
            else None
        )


def row_for_curve_m(rows, curve, m):
    candidates = [row for row in rows if row["curve"] == curve and int(row["m"]) == int(m)]
    return candidates[0] if candidates else None


def relative_value(row):
    if row is None:
        return None
    if row.get("relative_autoregressive_iia") is not None:
        return row["relative_autoregressive_iia"]
    denominator = row.get("full_curve_autoregressive_iia")
    if denominator:
        return row["autoregressive_iia"] / denominator
    return None


def run_experiment(args):
    paths = resolve_artifact_paths(args)
    das_basis, payload = load_basis(paths["subspace"], args.layer)
    result_row = paths["bank_row"] or load_run_row(paths["results"])
    data_path = result_row.get("data_path") or payload.get("config", {}).get("data_path")
    if data_path is None:
        raise ValueError("Could not infer data_path from DAS result row or checkpoint config.")
    if paths["heldout"] is not None and Path(paths["heldout"]).exists():
        pairs = load_pairs(paths["heldout"], data_path)
    else:
        pairs = reconstruct_pairs(args, data_path, result_row, payload)
    if args.max_autoregressive_pairs < 0:
        raise ValueError("--max_autoregressive_pairs must be non-negative; 0 means all.")
    if args.max_autoregressive_pairs > 0:
        pairs = pairs[: args.max_autoregressive_pairs]

    model_path, model_name = resolve_model_for_loading(args.model)
    model, tokenizer = load_hf_model(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    hidden_size = get_hidden_size(model)
    if max(das_basis.shape) != hidden_size:
        raise ValueError(
            f"DAS basis d_model={max(das_basis.shape)} does not match "
            f"loaded model hidden size={hidden_size}."
        )

    use_chat = args.use_chat_template or uses_chat_template(args.model)
    fourier_basis, fourier_paths, fourier_metrics = load_fourier_space(args, hidden_size)
    principal, singular_values = ordered_das_basis(das_basis, fourier_basis)
    saved_basis_dim = principal.shape[1]
    source_das_k = row_k(result_row) or payload.get("config", {}).get("k") or saved_basis_dim
    source_das_iia = optional_float(result_row.get("autoregressive_iia"))
    source_das_tf_iia = optional_float(result_row.get("variable_teacher_forced_iia"))
    truncated_saved_basis = False
    if args.k is not None:
        if args.k > principal.shape[1]:
            raise ValueError(
                f"Requested k={args.k}, but the DAS basis only has "
                f"{principal.shape[1]} directions."
            )
        if args.k < principal.shape[1]:
            print(
                f"Using the first {args.k} Fourier-ordered DAS directions "
                f"from a saved {principal.shape[1]}-direction basis."
            )
            principal = principal[:, : args.k]
            singular_values = singular_values[: args.k]
            truncated_saved_basis = True
    values = m_grid(principal.shape[1], args)

    clean_iia = clean_autoregressive_iia(
        model, tokenizer, pairs, args.target, use_chat, args.max_new_tokens
    )
    full_das_iia = None

    output_dir = curve_output_dir(args, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions = []
    for direction_order in ("top", "bottom"):
        for m in values:
            if direction_order == "top":
                basis = principal[:, :m]
                label = "most_to_least"
            else:
                basis = principal[:, principal.shape[1] - m :]
                label = "least_to_most"
            subspace = subspace_from_columns(args.layer, basis, model.device)
            iia, outputs = autoregressive_iia_with_outputs(
                model=model,
                tokenizer=tokenizer,
                blocks=blocks,
                subspaces=subspace,
                layer=args.layer,
                hook_name=args.hook,
                pairs=pairs,
                position=args.position,
                target=args.target,
                use_chat_template=use_chat,
                max_new_tokens=args.max_new_tokens,
                description=f"{label} m={m}",
            )
            row = {
                "model": model_name,
                "model_slug": model_slug(model_name),
                "modality": args.modality,
                "operation": args.operation,
                "target": args.target,
                "fourier_target": args.fourier_target or args.target,
                "fourier_modality": args.fourier_modality or args.operation,
                "condition": args.condition,
                "layer": args.layer,
                "position": str(args.position),
                "fourier_position": str(args.fourier_position or args.position),
                "hook": args.hook,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "m": m,
                "k": principal.shape[1],
                "curve": label,
                "autoregressive_iia": iia,
                "clean_autoregressive_iia": clean_iia,
                "full_das_autoregressive_iia": full_das_iia,
                "source_das_autoregressive_iia": source_das_iia,
                "source_das_variable_teacher_forced_iia": source_das_tf_iia,
                "source_das_k": int(source_das_k) if source_das_k is not None else None,
                "saved_basis_dim": int(saved_basis_dim),
                "truncated_saved_basis": truncated_saved_basis,
                "eta_vs_full_das": (
                    eta(iia, clean_iia, full_das_iia)
                    if full_das_iia is not None
                    else None
                ),
                "singular_values": [float(value) for value in singular_values],
                "mean_sigma_in_subset": float(
                    singular_values[:m].mean()
                    if direction_order == "top"
                    else singular_values[-m:].mean()
                ),
                "min_sigma_in_subset": float(
                    singular_values[:m].min()
                    if direction_order == "top"
                    else singular_values[-m:].min()
                ),
                "max_sigma_in_subset": float(
                    singular_values[:m].max()
                    if direction_order == "top"
                    else singular_values[-m:].max()
                ),
                "das_subspace_path": str(paths["subspace"]),
                "das_results_path": str(paths["results"]),
                "heldout_pairs_path": str(paths["heldout"]) if paths["heldout"] else None,
                "artifact_source": args.artifact_source,
                "fourier_probe_paths": fourier_paths,
                "fourier_probe_metrics": fourier_metrics,
                "config": jsonable(vars(args)),
                "saved_das_config": jsonable(payload.get("config")),
            }
            rows.append(row)
            for output in outputs:
                predictions.append(
                    {
                        "curve": label,
                        "m": m,
                        **output,
                    }
                )

    add_relative_iia(rows)
    result_path = output_dir / "fourier_ordered_autoregressive_iia.jsonl"
    prediction_path = output_dir / "fourier_ordered_predictions.jsonl"
    save_jsonl(rows, result_path)
    save_jsonl(predictions, prediction_path)
    try:
        plot_single_curve(rows, output_dir / "fourier_ordered_autoregressive_iia.png")
    except ModuleNotFoundError as error:
        if error.name != "matplotlib":
            raise
        print("matplotlib is not installed; saved JSONL results without plots.")
    print(json.dumps(rows, indent=2))
    print(f"Saved {result_path}")
    print(f"Saved {prediction_path}")
    return result_path


def add_landmarks(ax, k):
    for m in landmark_ms(k):
        x = m / k
        ax.axvline(x, color=SPINE, linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.9)
        ax.text(
            x,
            0.935,
            f"{m}/{k}",
            transform=ax.get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=6.9,
            color="#5b6366",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": BACKGROUND,
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )


def plot_summary_bars(ax, rows):
    k = int(rows[0]["k"])
    visible_m = effective_k(k)
    null_m = null_complement_k(k)
    full_m = k
    values = [
        relative_value(row_for_curve_m(rows, "most_to_least", visible_m)),
        relative_value(row_for_curve_m(rows, "least_to_most", null_m)),
        relative_value(row_for_curve_m(rows, "most_to_least", full_m)),
    ]
    labels = [f"top {visible_m}", f"bottom {null_m}", f"all {full_m}"]
    colors = [VISIBLE, NULL, FULL]
    y_values = list(range(len(labels)))
    ax.barh(
        y_values,
        [0 if value is None else value for value in values],
        color=colors,
        height=0.58,
    )
    for index, value in enumerate(values):
        if value is None:
            ax.text(0.02, index, "n/a", ha="left", va="center", fontsize=7.3)
        else:
            if value >= 0.92:
                ax.text(
                    value - 0.035,
                    index,
                    f"{value:.2f}",
                    ha="right",
                    va="center",
                    fontsize=7.3,
                    color=BACKGROUND,
                )
            else:
                ax.text(
                    value + 0.035,
                    index,
                    f"{value:.2f}",
                    ha="left",
                    va="center",
                    fontsize=7.3,
                    color=TEXT,
                )
    setup_axis(ax)
    ax.set_title("Subspace slices", fontsize=8.9, pad=4)
    ax.set_xlabel("relative IIA")
    ax.set_yticks(y_values, labels)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.65, len(labels) - 0.35)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def absolute_note(rows):
    if not rows:
        return None
    row = rows[0]
    plotted_full = row.get("full_curve_autoregressive_iia")
    source_full = row.get("source_das_autoregressive_iia")
    source_k = row.get("source_das_k")
    saved_basis_dim = row.get("saved_basis_dim")
    k = row.get("k")
    lines = []
    if plotted_full is not None:
        lines.append(f"absolute AR IIA, all {k}: {plotted_full:.2f}")
    if (
        source_full is not None
        and plotted_full is not None
        and abs(source_full - plotted_full) > 0.015
    ):
        lines.append(f"source DAS k={source_k}: {source_full:.2f}")
    if saved_basis_dim is not None and k is not None and int(saved_basis_dim) != int(k):
        lines.append(f"plotted {k}/{saved_basis_dim} saved dirs")
    return "\n".join(lines) if lines else None


def plot_single_curve(rows, output_path, title=None, ax=None, summary_ax=None):
    plt = pyplot()
    style_matplotlib(plt)
    own_fig = ax is None
    if own_fig:
        fig, (ax, summary_ax) = plt.subplots(
            1,
            2,
            figsize=(6.8, 3.45),
            gridspec_kw={"width_ratios": [4.45, 1.25], "wspace": 0.28},
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.082, right=0.975, bottom=0.18, top=0.9, wspace=0.28)
    setup_axis(ax)
    labels = {
        "most_to_least": "Fourier-aligned first",
        "least_to_most": "Fourier-orthogonal first",
    }
    colors = {"most_to_least": VISIBLE, "least_to_most": NULL}
    relative_handles = []
    for curve in ("most_to_least", "least_to_most"):
        curve_rows = sorted(
            [row for row in rows if row["curve"] == curve], key=lambda row: row["m"]
        )
        if not curve_rows:
            continue
        relative = [relative_value(row) for row in curve_rows]
        x_values = [row.get("m_over_k", row["m"] / row["k"]) for row in curve_rows]
        handle = ax.plot(
            x_values,
            relative,
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            label=labels[curve],
            color=colors[curve],
        )[0]
        relative_handles.append(handle)
    full = rows[0].get("full_curve_autoregressive_iia") if rows else None
    k = int(rows[0]["k"]) if rows else 32
    add_landmarks(ax, k)
    if full is not None:
        ax.axhline(1.0, color=FULL, linestyle="--", linewidth=1.0, alpha=0.88)
    ax.set_xlabel("fraction of DAS subspace used (m/k)")
    ax.set_ylabel("relative autoregressive IIA")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.08)
    if title:
        ax.set_title(title)
    note = absolute_note(rows)
    if note:
        ax.text(
            0.985,
            0.08,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.9,
            color="#3f4648",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": BACKGROUND,
                "edgecolor": SPINE,
                "linewidth": 0.45,
                "alpha": 0.9,
            },
        )
    legend = ax.legend(
        handles=relative_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.91),
        fontsize=6.2,
        handlelength=1.15,
        labelspacing=0.18,
        borderpad=0.22,
        borderaxespad=0.0,
        markerscale=0.72,
        frameon=True,
    )
    legend.get_frame().set_facecolor(BACKGROUND)
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.92)
    if summary_ax is not None and rows:
        plot_summary_bars(summary_ax, rows)
    if own_fig:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=350, facecolor=fig.get_facecolor())
        fig.savefig(output_path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
        plt.close(fig)


def long_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = path.resolve()
    open_path = str(path)
    if os.name == "nt" and path.is_absolute() and not open_path.startswith("\\\\?\\"):
        open_path = "\\\\?\\" + open_path
    return open_path


def load_curve_rows(path):
    path = Path(path)
    with open(long_path(path), "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No curve rows in {path}.")
    return rows


def plot_comparison(compare, output_dir):
    plt = pyplot()
    style_matplotlib(plt)
    if not compare:
        raise ValueError("--plot_only requires at least one --compare LABEL RESULT_JSONL.")
    n = len(compare)
    fig, axes = plt.subplots(1, n, figsize=(4.75 * n, 3.7), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, compare):
        rows = load_curve_rows(path)
        add_relative_iia(rows)
        plot_single_curve(rows, Path(path).with_suffix(".png"), title=label, ax=ax)
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.supxlabel("fraction of DAS subspace used (m/k)", y=0.035, fontsize=10.2)
    fig.supylabel("relative autoregressive IIA", x=0.01, fontsize=10.2)
    fig.tight_layout(rect=(0.035, 0.075, 1.0, 1.0), pad=0.55, w_pad=0.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llama_gemma_fourier_ordered_iia.png"
    fig.savefig(output_path, dpi=240)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    args = parse_args()
    if args.plot_only:
        plot_comparison(args.compare, args.output_dir)
    else:
        result_path = run_experiment(args)
        if args.compare:
            compare = args.compare + [(model_slug(args.model), str(result_path))]
            plot_comparison(compare, args.output_dir)


if __name__ == "__main__":
    main()
