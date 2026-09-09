"""Layer-wise diagnostic for the arithmetic-to-readout transition.

For each existing DAS subspace.pt:
  1. measure overlap with digit unembedding directions;
  2. test transfer to a non-arithmetic one-digit repeat task;
  3. project out the centered digit-readout span;
  4. recompute self arithmetic IIA with the original and ablated bases.

This script does not train DAS. It is meant to reuse already-trained layer
sweep folders and answer whether earlier layers are causal before becoming a
generic digit readout/writeout channel.
"""

import argparse
import glob
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.common import load_jsonl, save_jsonl
from src.experiments.readout_latent_geometry.controls.readout_ablated_subspaces import project_out_readout
from src.experiments.readout_latent_geometry.controls.repeat_transfer import (
    clean_autoregressive_iia as repeat_clean_autoregressive_iia,
    evaluate_repeat_teacher_forced,
    make_repeat_samples,
    repeat_position_spec,
    repeat_task_autoregressive_accuracy,
    repeat_task_teacher_forced_accuracy,
)
from src.experiments.readout_latent_geometry.controls.unembedding_overlap import (
    projection_rows,
    token_specs,
)
from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
    build_unique_pairs,
    evaluate_teacher_forced,
    format_prompt,
)
from src.experiments.arithmetic_reference.das_audit.audit_das import clean_autoregressive_iia_text
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.geometry.subspaces import torch_load_portable
from src.geometry.subspaces import orthonormal_columns
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_GLOBS = [
    "results/final_exps/DAS_audit_k_22/text/addition/das_pca_initialized/split_0/seed_*/subspace.pt",
]


class FullActivationPatch:
    def patch(self, _base_vector, source_vector):
        return source_vector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subspace", type=Path, action="append", default=[])
    parser.add_argument("--subspace_glob", action="append", default=[])
    parser.add_argument("--layers", type=int, nargs="*", default=[])
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument("--model", default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/check_unembeeding/readout_transition_sweep"))
    parser.add_argument("--stem", default="readout_transition_sweep")
    parser.add_argument("--max_token_number", type=int, default=9)
    parser.add_argument("--include_space_prefixed", action="store_true")
    parser.add_argument("--repeat_min_value", type=int, default=0)
    parser.add_argument("--repeat_max_value", type=int, default=9)
    parser.add_argument("--repeat_max_pairs", type=int, default=90)
    parser.add_argument("--repeat_pair_seed", type=int, default=0)
    parser.add_argument("--repeat_position", default="-1")
    parser.add_argument("--repeat_prompt_template", default="Output ONLY a number. Repeat this number: {n} =")
    parser.add_argument("--repeat_answer_separator", default=" ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_self_pairs", type=int, default=0, help="Cap original arithmetic heldout pairs; 0 means all.")
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--skip_autoregressive", action="store_true")
    parser.add_argument("--skip_original", action="store_true", help="Only evaluate readout-ablated self/repeat.")
    parser.add_argument("--save_ablated_subspaces", action="store_true")
    return parser.parse_args()


def candidate_paths(args):
    paths = list(args.subspace)
    for pattern in args.subspace_glob or DEFAULT_GLOBS:
        paths.extend(Path(item) for item in sorted(glob.glob(pattern)))
    seen, unique = set(), []
    for path in paths:
        text = str(path)
        if text not in seen:
            seen.add(text)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("No subspace.pt files matched the requested inputs.")
    return unique


def first_jsonl(path):
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def metadata(path, payload):
    row_path = path.parent / "results.jsonl"
    row = first_jsonl(row_path) if row_path.exists() else {}
    config = payload.get("config", {})
    return {
        "row": row,
        "config": config,
        "model": row.get("model") or config.get("model"),
        "modality": row.get("modality") or config.get("modality"),
        "operation": row.get("operation") or config.get("operation"),
        "layer": int(row.get("layer", config.get("layer"))),
        "position": str(row.get("position", config.get("position"))),
        "hook": row.get("hook") or config.get("hook") or "resid_post",
        "target": row.get("target") or config.get("target") or "result",
        "seed": int(row.get("seed", config.get("seed", path.parent.name.removeprefix("seed_")))),
        "k": int(row.get("k", config.get("k", torch.as_tensor(payload["basis"]).shape[-1]))),
        "data_path": Path(row.get("data_path") or config.get("data_path")),
        "saved_autoregressive_iia": row.get("autoregressive_iia"),
        "saved_variable_teacher_forced_iia": row.get("variable_teacher_forced_iia"),
    }


def load_candidates(args):
    rows = []
    for path in candidate_paths(args):
        payload = torch_load_portable(path)
        if "basis" not in payload:
            print(f"Skipping {path}: no basis.")
            continue
        meta = metadata(path, payload)
        if meta["modality"] != "text":
            print(f"Skipping {path}: this sweep script currently supports text DAS only.")
            continue
        if args.layers and meta["layer"] not in set(args.layers):
            continue
        if args.seeds and meta["seed"] not in set(args.seeds):
            continue
        rows.append({"path": path, "payload": payload, "meta": meta})
    rows.sort(key=lambda item: (item["meta"]["layer"], item["meta"]["operation"], item["meta"]["seed"], str(item["path"])))
    if not rows:
        raise FileNotFoundError("No compatible text DAS candidates remained after filtering.")
    return rows


def sample_lookup(data_path):
    lookup = {}
    for index, sample in enumerate(load_jsonl(data_path)):
        key = sample.get("sample_id", index)
        lookup[key] = sample
        lookup[str(key)] = sample
    return lookup


def load_self_pairs(path, data_path, max_pairs):
    heldout = path.parent / "heldout_pairs.jsonl"
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


def normalize_basis(basis, hidden_size):
    basis = torch.as_tensor(basis).detach().float().squeeze()
    if basis.ndim != 2:
        raise ValueError(f"Basis must be rank-2, got {tuple(basis.shape)}.")
    if basis.shape[0] != hidden_size and basis.shape[1] == hidden_size:
        basis = basis.T
    if basis.shape[0] != hidden_size:
        raise ValueError(f"Basis shape {tuple(basis.shape)} does not match hidden_size={hidden_size}.")
    return basis


def subspace_from_basis(layer, basis, device):
    return {
        str(layer): DASSubspace(
            basis.shape[0], basis.shape[1], initial_basis=basis
        ).to(device)
    }


def digit_vectors_from_model(model, tokenizer, args):
    texts = [str(i) for i in range(args.max_token_number + 1)]
    if args.include_space_prefixed:
        texts += [f" {i}" for i in range(args.max_token_number + 1)]
    specs, skipped = token_specs(tokenizer, texts)
    token_ids = torch.tensor([item["token_id"] for item in specs], device=model.device)
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise ValueError("Could not find output embedding weight.")
    vectors = output.weight.detach()[token_ids].float().cpu()
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    return vectors, centered, specs, skipped


def metric_eta(value, clean, full):
    if value is None or clean is None or full is None:
        return None
    denominator = full - clean
    if abs(denominator) < 1e-12:
        return None
    return (value - clean) / denominator


def repeat_args(args):
    return SimpleNamespace(
        prompt_template=args.repeat_prompt_template,
        answer_separator=args.repeat_answer_separator,
        min_value=args.repeat_min_value,
        max_value=args.repeat_max_value,
        max_pairs=args.repeat_max_pairs,
        pair_seed=args.repeat_pair_seed,
    )


def repeat_baselines(model, tokenizer, blocks, layer, hook, position, use_chat, args):
    samples = make_repeat_samples(repeat_args(args))
    pairs, pair_stats = build_unique_pairs(samples, "result", args.repeat_pair_seed, args.repeat_max_pairs)
    if not pairs:
        raise ValueError("No repeat pairs were created.")
    first_prompt = format_prompt(tokenizer, pairs[0]["base"], use_chat)
    resolved_position = repeat_position_spec(tokenizer, first_prompt, position)
    full_patch = {str(layer): FullActivationPatch()}
    full_metrics = evaluate_repeat_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=full_patch,
        layers=[layer],
        hook_name=hook,
        pairs=pairs,
        position=resolved_position,
        use_chat_template=use_chat,
        batch_size=args.batch_size,
        include_clean=True,
    )
    clean_tf = full_metrics["clean_counterfactual_variable_teacher_forced_iia"]
    ar_pairs = pairs[: args.max_autoregressive_pairs] if args.max_autoregressive_pairs > 0 else pairs
    result = {
        "samples": samples,
        "pairs": pairs,
        "ar_pairs": ar_pairs,
        "pair_stats": pair_stats,
        "position": resolved_position,
        "task_teacher_forced_accuracy": repeat_task_teacher_forced_accuracy(
            model, tokenizer, samples, use_chat, args.batch_size
        ),
        "task_autoregressive_accuracy": None,
        "clean_tf_iia": clean_tf,
        "full_tf_iia": full_metrics["variable_teacher_forced_iia"],
        "clean_ar_iia": None,
        "full_ar_iia": None,
    }
    if not args.skip_autoregressive:
        result["task_autoregressive_accuracy"] = repeat_task_autoregressive_accuracy(
            model, tokenizer, samples, use_chat, args.max_new_tokens
        )
        result["clean_ar_iia"] = repeat_clean_autoregressive_iia(
            model, tokenizer, ar_pairs, use_chat, args.max_new_tokens
        )
        result["full_ar_iia"] = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[layer],
            hook_name=hook,
            pairs=ar_pairs,
            position=resolved_position,
            target="result",
            use_chat_template=use_chat,
            max_new_tokens=args.max_new_tokens,
            description=f"repeat full patch layer={layer}",
        )
    return result


def evaluate_repeat(model, tokenizer, blocks, subspaces, layer, hook, use_chat, baselines, args, label):
    metrics = evaluate_repeat_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=subspaces,
        layers=[layer],
        hook_name=hook,
        pairs=baselines["pairs"],
        position=baselines["position"],
        use_chat_template=use_chat,
        batch_size=args.batch_size,
        include_clean=True,
    )
    ar = None
    if not args.skip_autoregressive:
        ar = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspaces,
            layers=[layer],
            hook_name=hook,
            pairs=baselines["ar_pairs"],
            position=baselines["position"],
            target="result",
            use_chat_template=use_chat,
            max_new_tokens=args.max_new_tokens,
            description=f"repeat {label} layer={layer}",
        )
    return {
        "tf_iia": metrics["variable_teacher_forced_iia"],
        "tf_eta": metric_eta(metrics["variable_teacher_forced_iia"], baselines["clean_tf_iia"], baselines["full_tf_iia"]),
        "ar_iia": ar,
        "ar_eta": metric_eta(ar, baselines["clean_ar_iia"], baselines["full_ar_iia"]),
        "logprob_gain": metrics.get("source_variable_logprob_gain"),
    }


def self_baselines(model, tokenizer, blocks, meta, pairs, ar_pairs, use_chat, args):
    full_patch = {str(meta["layer"]): FullActivationPatch()}
    metrics = evaluate_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=full_patch,
        layers=[meta["layer"]],
        hook_name=meta["hook"],
        pairs=pairs,
        position=meta["position"],
        target=meta["target"],
        use_chat_template=use_chat,
        batch_size=args.batch_size,
        include_clean=True,
    )
    clean_tf = metrics["clean_counterfactual_variable_teacher_forced_iia"]
    result = {
        "clean_tf_iia": clean_tf,
        "full_tf_iia": metrics["variable_teacher_forced_iia"],
        "clean_ar_iia": None,
        "full_ar_iia": None,
    }
    if not args.skip_autoregressive:
        result["clean_ar_iia"] = clean_autoregressive_iia_text(
            model, tokenizer, ar_pairs, meta["target"], use_chat, args.max_new_tokens
        )
        result["full_ar_iia"] = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[meta["layer"]],
            hook_name=meta["hook"],
            pairs=ar_pairs,
            position=meta["position"],
            target=meta["target"],
            use_chat_template=use_chat,
            max_new_tokens=args.max_new_tokens,
            description=f"self full patch layer={meta['layer']}",
        )
    return result


def evaluate_self(model, tokenizer, blocks, subspaces, meta, pairs, ar_pairs, use_chat, baselines, args, label):
    metrics = evaluate_teacher_forced(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=subspaces,
        layers=[meta["layer"]],
        hook_name=meta["hook"],
        pairs=pairs,
        position=meta["position"],
        target=meta["target"],
        use_chat_template=use_chat,
        batch_size=args.batch_size,
        include_clean=True,
    )
    ar = None
    if not args.skip_autoregressive:
        ar = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspaces,
            layers=[meta["layer"]],
            hook_name=meta["hook"],
            pairs=ar_pairs,
            position=meta["position"],
            target=meta["target"],
            use_chat_template=use_chat,
            max_new_tokens=args.max_new_tokens,
            description=f"self {label} layer={meta['layer']}",
        )
    return {
        "tf_iia": metrics["variable_teacher_forced_iia"],
        "tf_eta": metric_eta(metrics["variable_teacher_forced_iia"], baselines["clean_tf_iia"], baselines["full_tf_iia"]),
        "ar_iia": ar,
        "ar_eta": metric_eta(ar, baselines["clean_ar_iia"], baselines["full_ar_iia"]),
        "logprob_gain": metrics.get("source_variable_logprob_gain"),
    }


def save_ablated(payload, meta, ablated_basis, metrics, args):
    output_dir = (
        args.output_dir
        / "ablated_subspaces"
        / f"layer_{meta['layer']}"
        / str(meta["operation"])
        / f"seed_{meta['seed']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subspace.pt"
    new_payload = dict(payload)
    new_payload["basis"] = ablated_basis.cpu()
    new_payload["kind"] = f"readout_ablated_{payload.get('kind', payload.get('condition', 'subspace'))}"
    new_payload["readout_ablation"] = metrics
    torch.save(new_payload, output_path)
    return output_path


def print_table(rows):
    print("\nReadout transition sweep")
    print("Looking for layers where self arithmetic stays high but repeat/readout effects are weak.\n")
    print(
        f"{'layer':>5} {'seed':>4} {'readout_x':>9} {'removed':>8} "
        f"{'repeat':>8} {'rep_ab':>8} {'self':>8} {'self_ab':>8} {'saved_ar':>8}"
    )
    print("-" * 82)
    for row in rows:
        def fmt(value):
            return "NA" if value is None else f"{float(value):.4f}"
        print(
            f"{row['layer']:5d} {row['seed']:4d} "
            f"{fmt(row['centered_mean_digit_token_projection_over_random']):>9} "
            f"{fmt(row['removed_energy_fraction']):>8} "
            f"{fmt(row.get('repeat_original_ar_iia')):>8} "
            f"{fmt(row.get('repeat_ablated_ar_iia')):>8} "
            f"{fmt(row.get('self_original_ar_iia')):>8} "
            f"{fmt(row.get('self_ablated_ar_iia')):>8} "
            f"{fmt(row.get('saved_autoregressive_iia')):>8}"
        )
    print("\nUse TF columns in the JSONL too; this table emphasizes AR IIA.")
    print("Promising layer: self_ablated remains high while repeat_original/repeat_ablated stay low.")


def main():
    args = parse_args()
    candidates = load_candidates(args)
    model_name = args.model or candidates[0]["meta"]["model"]
    if model_name is None:
        raise ValueError("Pass --model because the first subspace has no saved model.")
    model_path, resolved_model = resolve_model_for_loading(model_name)
    model, tokenizer = load_hf_model(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    hidden_size = get_hidden_size(model)

    layers = sorted({item["meta"]["layer"] for item in candidates})
    validate_block_layers(model, layers)
    digit_vectors, centered_digit_vectors, specs, skipped = digit_vectors_from_model(model, tokenizer, args)
    del digit_vectors
    readout_basis = orthonormal_columns(
        centered_digit_vectors.T, hidden_size, "centered digit readout"
    )
    use_chat = uses_chat_template(model_name)

    print(
        "readout_transition_sweep config: "
        f"model={resolved_model} candidates={len(candidates)} layers={layers} "
        f"repeat_range={args.repeat_min_value}..{args.repeat_max_value} "
        f"skip_ar={args.skip_autoregressive}"
    )

    rows = []
    repeat_cache = {}
    self_baseline_cache = {}
    for item in candidates:
        path, payload, meta = item["path"], item["payload"], item["meta"]
        basis = normalize_basis(payload["basis"], hidden_size)
        original_subspace = subspace_from_basis(meta["layer"], basis, model.device)
        ablated_basis, ablation_metrics = project_out_readout(basis, readout_basis)
        ablated_subspace = subspace_from_basis(meta["layer"], ablated_basis, model.device)

        overlap = projection_rows(basis, centered_digit_vectors)
        self_pairs = load_self_pairs(path, meta["data_path"], args.max_self_pairs)
        ar_pairs = self_pairs[: args.max_autoregressive_pairs] if args.max_autoregressive_pairs > 0 else self_pairs

        repeat_key = (meta["layer"], meta["hook"], args.repeat_position)
        if repeat_key not in repeat_cache:
            repeat_cache[repeat_key] = repeat_baselines(
                model, tokenizer, blocks, meta["layer"], meta["hook"], args.repeat_position, use_chat, args
            )
        repeat_base = repeat_cache[repeat_key]

        self_key = (str(path.parent), meta["layer"], meta["position"], meta["hook"], meta["target"])
        if self_key not in self_baseline_cache:
            self_baseline_cache[self_key] = self_baselines(
                model, tokenizer, blocks, meta, self_pairs, ar_pairs, use_chat, args
            )
        self_base = self_baseline_cache[self_key]

        repeat_original = None
        self_original = None
        if not args.skip_original:
            repeat_original = evaluate_repeat(
                model, tokenizer, blocks, original_subspace, meta["layer"], meta["hook"], use_chat, repeat_base, args, "original"
            )
            self_original = evaluate_self(
                model, tokenizer, blocks, original_subspace, meta, self_pairs, ar_pairs, use_chat, self_base, args, "original"
            )
        repeat_ablated = evaluate_repeat(
            model, tokenizer, blocks, ablated_subspace, meta["layer"], meta["hook"], use_chat, repeat_base, args, "ablated"
        )
        self_ablated = evaluate_self(
            model, tokenizer, blocks, ablated_subspace, meta, self_pairs, ar_pairs, use_chat, self_base, args, "ablated"
        )

        ablated_path = None
        if args.save_ablated_subspaces:
            ablated_path = save_ablated(payload, meta, ablated_basis, ablation_metrics, args)

        row = {
            "model": resolved_model,
            "subspace_path": str(path),
            "ablated_subspace_path": None if ablated_path is None else str(ablated_path),
            "modality": meta["modality"],
            "operation": meta["operation"],
            "layer": meta["layer"],
            "position": meta["position"],
            "hook": meta["hook"],
            "target": meta["target"],
            "seed": meta["seed"],
            "k": meta["k"],
            "n_self_pairs": len(self_pairs),
            "n_self_ar_pairs": len(ar_pairs),
            "n_repeat_pairs": len(repeat_base["pairs"]),
            "n_repeat_ar_pairs": len(repeat_base["ar_pairs"]),
            "token_specs": specs,
            "skipped_token_texts": skipped,
            "saved_autoregressive_iia": meta["saved_autoregressive_iia"],
            "saved_variable_teacher_forced_iia": meta["saved_variable_teacher_forced_iia"],
            "repeat_task_teacher_forced_accuracy": repeat_base["task_teacher_forced_accuracy"],
            "repeat_task_autoregressive_accuracy": repeat_base["task_autoregressive_accuracy"],
            "repeat_clean_tf_iia": repeat_base["clean_tf_iia"],
            "repeat_full_tf_iia": repeat_base["full_tf_iia"],
            "repeat_clean_ar_iia": repeat_base["clean_ar_iia"],
            "repeat_full_ar_iia": repeat_base["full_ar_iia"],
            "self_clean_tf_iia": self_base["clean_tf_iia"],
            "self_full_tf_iia": self_base["full_tf_iia"],
            "self_clean_ar_iia": self_base["clean_ar_iia"],
            "self_full_ar_iia": self_base["full_ar_iia"],
            **{f"centered_{key}": value for key, value in overlap.items()},
            **ablation_metrics,
        }
        for prefix, values in (
            ("repeat_original", repeat_original),
            ("self_original", self_original),
            ("repeat_ablated", repeat_ablated),
            ("self_ablated", self_ablated),
        ):
            if values is None:
                continue
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(rows, args.output_dir / f"{args.stem}.jsonl")

    print_table(rows)
    for row in rows:
        print(json.dumps(row))
    print(f"Saved {args.output_dir / f'{args.stem}.jsonl'}")


if __name__ == "__main__":
    main()
