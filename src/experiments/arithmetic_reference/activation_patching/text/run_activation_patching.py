import argparse

import torch
from tqdm import tqdm
from pathlib import Path

from src.experiments.arithmetic_reference.activation_patching.text.common import (
    HOOK_DESCRIPTIONS,
    OPERATIONS,
    answer_token_ids,
    append_jsonl,
    build_pairs_from_dataset,
    cache_clean_activations,
    contrast_metrics,
    encode_prompt,
    prompt_with_template,
    recovery,
    valid_hooks,
    write_json,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    build_probe_context,
    select_probe_positions,
)
from src.models import load_hf_model, resolve_model_for_loading, validate_block_layers

def output_paths(output_dir, operation, layer_mode, layers):
    folder = output_dir / operation
    layer_suffix = "-".join(str(layer) for layer in (layers or ["all"]))
    mode_suffix = "" if layer_mode == "individual" else "_together"
    stem = f"{operation}_activation_patching_layers{layer_suffix}{mode_suffix}"
    return {
        "folder": folder,
        "jsonl": folder / f"{stem}.jsonl",
        "metadata": folder / f"{stem}_metadata.json",
    }


def resolve_hf_positions(tokenizer, prompt, position_indices):
    context = build_probe_context(tokenizer, prompt)
    hf_positions, _ = select_probe_positions(context, position_indices)
    return hf_positions



@torch.no_grad()
def run_pair(
    model,
    tokenizer,
    model_name,
    pair,
    layers,
    hooks,
    position_indices,
    layer_mode,
):
    clean_prompt = prompt_with_template(tokenizer, pair["clean_expr"], model_name)
    corrupt_prompt = prompt_with_template(tokenizer, pair["corrupt_expr"], model_name)
    clean_encoding = encode_prompt(tokenizer, clean_prompt, model.device)
    corrupt_encoding = encode_prompt(tokenizer, corrupt_prompt, model.device)

    # Locate each semantic token
    clean_hf_positions = resolve_hf_positions(
        tokenizer, clean_prompt, position_indices
    )
    corrupt_hf_positions = resolve_hf_positions(
        tokenizer, corrupt_prompt, position_indices
    )

    # Tokenize the result
    clean_ids = answer_token_ids(tokenizer, pair["clean_answer"])
    corrupt_ids = answer_token_ids(tokenizer, pair["corrupt_answer"])

    # Run clean cache at each layer/hook
    clean_cache = cache_clean_activations(model, clean_encoding, layers, hooks)

    # Compute contrast without patching just yet
    clean_metrics = contrast_metrics(model, clean_encoding, clean_ids, corrupt_ids)
    corrupt_metrics = contrast_metrics(model, corrupt_encoding, clean_ids, corrupt_ids)
    clean_contrast = clean_metrics["contrast"]
    corrupt_contrast = corrupt_metrics["contrast"]

    rows = []
    layer_groups = [[layer] for layer in layers]
    if layer_mode == "together":
        layer_groups = [layers]

    for hook_name in hooks:
        for layer_group in layer_groups:
            for position, clean_hf_position, corrupt_hf_position in zip(
                position_indices,
                clean_hf_positions,
                corrupt_hf_positions,
            ):
                patches = [
                    {
                        "clean_activation": clean_cache[(layer, hook_name)],
                        "layer": layer,
                        "hook_name": hook_name,
                        "clean_position": clean_hf_position,
                        "corrupt_position": corrupt_hf_position,
                    }
                    for layer in layer_group
                ]
                patch = patches[0] if layer_mode == "individual" else patches
                patched_metrics = contrast_metrics(
                    model,
                    corrupt_encoding,
                    clean_ids,
                    corrupt_ids,
                    patch,
                )
                patched_contrast = patched_metrics["contrast"]
                recovered = recovery(clean_contrast, corrupt_contrast, patched_contrast)
                rows.append(
                    {
                        **pair,
                        "model": model_name,
                        "layer": layer_group[0] if layer_mode == "individual" else "all",
                        "patched_layers": layer_group,
                        "layer_mode": layer_mode,
                        "hook": hook_name,
                        "position": position,
                        "clean_hf_position": clean_hf_position,
                        "corrupt_hf_position": corrupt_hf_position,
                        "clean_prompt_length": clean_encoding["input_ids"].shape[1],
                        "corrupt_prompt_length": corrupt_encoding["input_ids"].shape[1],
                        "clean_prompt_token_id": clean_encoding["input_ids"][
                            0, clean_hf_position
                        ].item(),
                        "corrupt_prompt_token_id": corrupt_encoding["input_ids"][
                            0, corrupt_hf_position
                        ].item(),
                        "clean_answer_token_ids": clean_ids,
                        "corrupt_answer_token_ids": corrupt_ids,
                        "clean_contrast": clean_contrast,
                        "corrupt_contrast": corrupt_contrast,
                        "patched_contrast": patched_contrast,
                        "recovery": recovered,
                        "clean_answer_iia": clean_metrics["clean_answer"]["iia"],
                        "clean_answer_token_accuracy": clean_metrics["clean_answer"][
                            "token_accuracy"
                        ],
                        "corrupt_answer_iia": corrupt_metrics["corrupt_answer"]["iia"],
                        "corrupt_answer_token_accuracy": corrupt_metrics[
                            "corrupt_answer"
                        ]["token_accuracy"],
                        "patched_clean_iia": patched_metrics["clean_answer"]["iia"],
                        "patched_clean_token_accuracy": patched_metrics[
                            "clean_answer"
                        ]["token_accuracy"],
                    }
                )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=sorted(OPERATIONS), required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--model-name", default="gemma4_12b_it")
    parser.add_argument("--output-dir", default="results/activation_patching")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layer-mode",
        choices=["individual", "together"],
        default="individual",
        help="Patch each selected layer separately or all selected layers together.",
    )
    parser.add_argument("--hooks", nargs="+", default=None)
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        required=True,
        help="Required non-negative 0-based prompt-token indices.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    positions = args.positions
    if any(position < 0 for position in positions):
        raise ValueError("Prompt-token positions must be non-negative 0-based indices.")

    model_path, model_name = resolve_model_for_loading(args.model_name)
    model, tokenizer = load_hf_model(model_path)
    model.eval()

    layers = validate_block_layers(model, args.layers)
    hooks = valid_hooks(args.hooks)
    paths = output_paths(Path(args.output_dir), args.operation, args.layer_mode, layers)
    paths["folder"].mkdir(parents=True, exist_ok=True)
    if paths["jsonl"].exists():
        paths["jsonl"].unlink()
        print(f"Removed already existing file {paths['jsonl']}")
    if args.data_path:
        pairs = build_pairs_from_dataset(
            args.data_path,
            args.operation,
            args.n_pairs,
            args.seed,
        )
    else:
        raise ValueError("Please provide data_path.")

    for pair in tqdm(pairs, desc=f"{args.operation} patching"):
        rows = run_pair(
            model,
            tokenizer,
            model_name,
            pair,
            layers,
            hooks,
            positions,
            args.layer_mode,
        )
        append_jsonl(paths["jsonl"], rows)

    write_json(
        paths["metadata"],
        {
            "operation": args.operation,
            "model": model_name,
            "data_path": args.data_path,
            "n_pairs": args.n_pairs,
            "layers": layers,
            "layer_mode": args.layer_mode,
            "hooks": hooks,
            "positions": positions,
            "hook_descriptions": HOOK_DESCRIPTIONS,
            "patch": (
                "Resolve each numeric position separately in the clean and corrupt "
                "prompts, then copy clean[:, clean_hf_position, :] into "
                "corrupt[:, corrupt_hf_position, :] at each layer and hook."
            ),
            "layer_indexing": (
                "Layers are one-based transformer blocks. resid_pre at layer L "
                "equals hidden_states[L-1]; resid_post equals hidden_states[L]."
            ),
            "metric": (
                "contrast = mean log P(all clean-answer tokens) - mean log P(all "
                "corrupt-answer tokens). Recovery measures how far patching moves "
                "the corrupt contrast toward the clean contrast."
            ),
        },
    )
    print(f"Saved results: {paths['jsonl']}")


if __name__ == "__main__":
    main()
