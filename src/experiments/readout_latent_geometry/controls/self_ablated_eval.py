"""Recompute self-DAS performance for original or readout-ablated subspaces.

The readout ablation script mirrors the audit tree and copies results.jsonl,
so copied autoregressive_iia values are stale. This script reloads the actual
subspace.pt candidates and evaluates them on their own heldout pairs.
"""

import argparse
import json
from pathlib import Path

import torch

from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
    evaluate_teacher_forced,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    autoregressive_iia_image,
    evaluate_teacher_forced_image,
)
from src.experiments.arithmetic_reference.das_audit.audit_das import (
    clean_autoregressive_iia_image,
    clean_autoregressive_iia_text,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.geometry.subspaces import torch_load_portable
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_CANDIDATE = (
    "ablated=results/experiments/check_unembeeding/"
    "readout_ablated_audit_k_22_layer43/text/addition/"
    "das_pca_initialized/split_0/seed_1/subspace.pt"
)


class FullActivationPatch:
    def patch(self, _base_vector, source_vector):
        return source_vector


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Labeled subspace as label=/path/subspace.pt. Can be repeated.",
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        help="Directory containing results.jsonl and heldout_pairs.jsonl. Defaults to first candidate parent.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments/check_unembeeding/self_ablated_eval"))
    parser.add_argument("--stem", default="self_ablated_eval")
    parser.add_argument("--model", default=None)
    parser.add_argument("--modality", choices=["text", "image"], default=None)
    parser.add_argument("--operation", default=None)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--position", default=None)
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--prompt", default=None, help="Image prompt; defaults to saved config/result row.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_pairs", type=int, default=0, help="Teacher-forced heldout pair cap; 0 means all.")
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--skip_autoregressive", action="store_true")
    parser.add_argument("--skip_full_patch", action="store_true")
    return parser.parse_args()


def parse_candidate(text):
    if "=" in text:
        label, path = text.split("=", 1)
    else:
        path = text
        label = Path(path).parent.name
    return label, Path(path)


def first_jsonl(path):
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def first_payload(candidates):
    for _label, path in candidates:
        if path.exists():
            return torch_load_portable(path)
    raise FileNotFoundError("None of the candidate subspace.pt files exist.")


def metadata(args, run_dir, payload):
    row = first_jsonl(run_dir / "results.jsonl")
    config = payload.get("config", {})
    return {
        "model": args.model or row.get("model") or config.get("model") or "gemma4_12b_it",
        "modality": args.modality or row.get("modality") or config.get("modality"),
        "operation": args.operation or row.get("operation") or config.get("operation"),
        "data_path": args.data_path or Path(row.get("data_path") or config.get("data_path")),
        "data_root": args.data_root or (Path(row["data_root"]) if row.get("data_root") else None),
        "layer": args.layer if args.layer is not None else int(row.get("layer", config.get("layer"))),
        "position": args.position if args.position is not None else str(row.get("position", config.get("position"))),
        "hook": args.hook or row.get("hook") or config.get("hook") or "resid_post",
        "target": args.target or row.get("target") or config.get("target") or "result",
        "prompt": args.prompt or row.get("prompt") or config.get("prompt") or "Output ONLY a number.",
        "enable_thinking": bool(args.enable_thinking or row.get("enable_thinking") or config.get("enable_thinking")),
        "saved_condition": row.get("condition"),
        "saved_autoregressive_iia": row.get("autoregressive_iia"),
        "saved_clean_autoregressive_iia": row.get("clean_autoregressive_iia"),
        "saved_full_patch_autoregressive_iia": row.get("full_patch_autoregressive_iia"),
    }


def sample_lookup(data_path):
    lookup = {}
    for index, sample in enumerate(load_jsonl(data_path)):
        key = sample.get("sample_id", index)
        lookup[key] = sample
        lookup[str(key)] = sample
    return lookup


def load_pairs(run_dir, data_path, max_pairs):
    heldout = run_dir / "heldout_pairs.jsonl"
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


def load_subspace(path, layer, hidden_size, device):
    payload = torch_load_portable(path)
    if "basis" not in payload:
        raise ValueError(f"{path} does not contain a basis.")
    basis = torch.as_tensor(payload["basis"]).detach().float().squeeze()
    if basis.ndim != 2:
        raise ValueError(f"{path} basis must be rank-2; got {tuple(basis.shape)}.")
    if basis.shape[0] != hidden_size and basis.shape[1] == hidden_size:
        basis = basis.T
    if basis.shape[0] != hidden_size:
        raise ValueError(
            f"{path} basis shape {tuple(basis.shape)} is incompatible with hidden_size={hidden_size}."
        )
    return {str(layer): DASSubspace(hidden_size, basis.shape[1], initial_basis=basis).to(device)}, payload


def eta(value, clean, full):
    if value is None or clean is None or full is None:
        return None
    denom = full - clean
    if abs(denom) < 1e-12:
        return None
    return (value - clean) / denom


def mean_or_none(values):
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(float(value) for value in present) / len(present)


def evaluate_text_candidate(model, tokenizer, blocks, meta, pairs, ar_pairs, subspaces, args):
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
        use_chat_template=meta["use_chat_template"],
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
            use_chat_template=meta["use_chat_template"],
            max_new_tokens=args.max_new_tokens,
        )
    return metrics, ar


def evaluate_image_candidate(model, processor, tokenizer, blocks, meta, pairs, ar_pairs, subspaces, args):
    data_root = meta["data_root"] or meta["data_path"].parent
    metrics = evaluate_teacher_forced_image(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        subspaces=subspaces,
        layers=[meta["layer"]],
        hook_name=meta["hook"],
        pairs=pairs,
        data_root=data_root,
        prompt=meta["prompt"],
        position_strategy=meta["position"],
        target=meta["target"],
        enable_thinking=meta["enable_thinking"],
        batch_size=args.batch_size,
        include_clean=True,
    )
    ar = None
    if not args.skip_autoregressive:
        ar = autoregressive_iia_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspaces,
            layers=[meta["layer"]],
            hook_name=meta["hook"],
            pairs=ar_pairs,
            data_root=data_root,
            prompt=meta["prompt"],
            position_strategy=meta["position"],
            target=meta["target"],
            enable_thinking=meta["enable_thinking"],
            max_new_tokens=args.max_new_tokens,
        )
    return metrics, ar


def evaluate_full_and_clean(model, processor, tokenizer, blocks, meta, pairs, ar_pairs, args):
    if args.skip_full_patch:
        return {
            "clean_tf": None,
            "full_tf": None,
            "clean_ar": meta["saved_clean_autoregressive_iia"],
            "full_ar": meta["saved_full_patch_autoregressive_iia"],
        }
    full = {str(meta["layer"]): FullActivationPatch()}
    if meta["modality"] == "text":
        metrics, full_ar = evaluate_text_candidate(
            model, tokenizer, blocks, meta, pairs, ar_pairs, full, args
        )
        clean_ar = None
        if not args.skip_autoregressive:
            clean_ar = clean_autoregressive_iia_text(
                model,
                tokenizer,
                ar_pairs,
                meta["target"],
                meta["use_chat_template"],
                args.max_new_tokens,
            )
    else:
        metrics, full_ar = evaluate_image_candidate(
            model, processor, tokenizer, blocks, meta, pairs, ar_pairs, full, args
        )
        clean_ar = None
        if not args.skip_autoregressive:
            clean_ar = clean_autoregressive_iia_image(
                model,
                processor,
                tokenizer,
                ar_pairs,
                meta["data_root"] or meta["data_path"].parent,
                meta["prompt"],
                meta["target"],
                meta["enable_thinking"],
                args.max_new_tokens,
            )
    return {
        "clean_tf": metrics["clean_counterfactual_variable_teacher_forced_iia"],
        "full_tf": metrics["variable_teacher_forced_iia"],
        "clean_ar": clean_ar,
        "full_ar": full_ar,
    }


def make_row(label, path, meta, payload, metrics, ar, baselines, n_pairs, n_ar_pairs):
    condition = payload.get("kind") or payload.get("condition") or label
    clean_tf = metrics.get("clean_counterfactual_variable_teacher_forced_iia")
    return {
        "label": label,
        "condition": condition,
        "subspace_path": str(path),
        "source_subspace_path": payload.get("readout_ablation", {}).get("source_subspace_path"),
        "model": meta["resolved_model"],
        "modality": meta["modality"],
        "operation": meta["operation"],
        "layer": meta["layer"],
        "position": str(meta["position"]),
        "hook": meta["hook"],
        "target": meta["target"],
        "k": int(torch.as_tensor(payload["basis"]).squeeze().shape[-1]),
        "n_pairs": n_pairs,
        "n_autoregressive_pairs": n_ar_pairs,
        "variable_teacher_forced_iia": metrics["variable_teacher_forced_iia"],
        "full_answer_teacher_forced_iia": metrics["full_answer_teacher_forced_iia"],
        "source_variable_logprob_gain": metrics.get("source_variable_logprob_gain"),
        "source_answer_logprob_gain": metrics.get("source_answer_logprob_gain"),
        "clean_variable_teacher_forced_iia": clean_tf,
        "full_patch_variable_teacher_forced_iia": baselines["full_tf"],
        "eta_variable_teacher_forced": eta(metrics["variable_teacher_forced_iia"], clean_tf, baselines["full_tf"]),
        "autoregressive_iia": ar,
        "clean_autoregressive_iia": baselines["clean_ar"],
        "full_patch_autoregressive_iia": baselines["full_ar"],
        "eta_autoregressive": eta(ar, baselines["clean_ar"], baselines["full_ar"]),
        "saved_condition": meta["saved_condition"],
        "saved_autoregressive_iia": meta["saved_autoregressive_iia"],
        "readout_ablation": payload.get("readout_ablation"),
    }


def print_table(rows):
    print("\nSelf-DAS readout-ablation audit")
    print("Question: does the passed subspace still steer its own arithmetic task?")
    print()
    print(f"{'label':24} {'tf_iia':>8} {'tf_eta':>8} {'ar_iia':>8} {'ar_eta':>8} {'saved_ar':>9} {'logp_gain':>11}")
    print("-" * 86)
    for row in rows:
        def fmt(value):
            return "NA" if value is None else f"{float(value):.4f}"
        print(
            f"{row['label'][:24]:24} "
            f"{fmt(row['variable_teacher_forced_iia']):>8} "
            f"{fmt(row['eta_variable_teacher_forced']):>8} "
            f"{fmt(row['autoregressive_iia']):>8} "
            f"{fmt(row['eta_autoregressive']):>8} "
            f"{fmt(row['saved_autoregressive_iia']):>9} "
            f"{fmt(row['source_variable_logprob_gain']):>11}"
        )
    print()
    print("Interpretation:")
    print("  ablated near original: arithmetic-causal effect survives readout removal.")
    print("  ablated near control/clean: behavioral steering relied on readout directions.")
    print("  high TF but low AR: subspace shifts logits but does not robustly drive generation.")


def main():
    args = parse_args()
    candidates = [parse_candidate(item) for item in (args.candidate or [DEFAULT_CANDIDATE])]
    run_dir = args.run_dir or candidates[0][1].parent
    payload0 = first_payload(candidates)
    meta = metadata(args, run_dir, payload0)
    model_path, resolved_model = resolve_model_for_loading(meta["model"])
    meta["resolved_model"] = resolved_model

    pairs = load_pairs(run_dir, meta["data_path"], args.max_pairs)
    ar_pairs = pairs[: args.max_autoregressive_pairs] if args.max_autoregressive_pairs > 0 else pairs

    if meta["modality"] == "text":
        model, tokenizer = load_hf_model(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        processor = None
        meta["use_chat_template"] = bool(args.use_chat_template or uses_chat_template(meta["model"]))
    else:
        model, processor, tokenizer = load_hf_model_and_processor(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        meta["use_chat_template"] = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [meta["layer"]])
    hidden_size = get_hidden_size(model)

    print(
        "self_ablated_eval config: "
        f"run_dir={run_dir} model={resolved_model} task={meta['modality']}:{meta['operation']} "
        f"layer={meta['layer']} position={meta['position']} hook={meta['hook']} "
        f"target={meta['target']} n_pairs={len(pairs)} n_ar_pairs={len(ar_pairs)}"
    )

    baselines = evaluate_full_and_clean(model, processor, tokenizer, blocks, meta, pairs, ar_pairs, args)
    rows = []
    for label, path in candidates:
        subspaces, payload = load_subspace(path, meta["layer"], hidden_size, model.device)
        if meta["modality"] == "text":
            metrics, ar = evaluate_text_candidate(
                model, tokenizer, blocks, meta, pairs, ar_pairs, subspaces, args
            )
        else:
            metrics, ar = evaluate_image_candidate(
                model, processor, tokenizer, blocks, meta, pairs, ar_pairs, subspaces, args
            )
        rows.append(make_row(label, path, meta, payload, metrics, ar, baselines, len(pairs), len(ar_pairs)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.stem}.jsonl"
    save_jsonl(rows, output_path)
    print_table(rows)
    for row in rows:
        print(json.dumps(row))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
