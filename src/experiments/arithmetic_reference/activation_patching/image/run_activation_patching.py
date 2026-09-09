"""Activation patching for image-rendered arithmetic prompts."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from src.common.io import load_jsonl
from src.experiments.arithmetic_reference.activation_patching.text.common import (
    HOOK_DESCRIPTIONS,
    OPERATIONS,
    answer_token_ids,
    append_jsonl,
    cache_clean_activations,
    patched_forward,
    patched_forward_many,
    recovery,
    valid_hooks,
    write_json,
)
from src.experiments.arithmetic_reference.activation_patching.text.run_activation_patching import (
    output_paths,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    load_rgb_image,
    print_position_summary,
)
from src.experiments.arithmetic_reference.linear_probes.image.extract_image_activations import (
    make_inputs,
    render_prompt,
)
from src.models import (
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


def inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def build_pairs_from_image_dataset(data_path, operation, n_pairs, seed=0):
    samples = load_jsonl(Path(data_path))
    symbol = OPERATIONS[operation]
    samples = [sample for sample in samples if sample.get("operation") == symbol]
    if not samples:
        raise ValueError(f"No {symbol!r} image samples found in {data_path}")

    required = {"image_path", "a", "b", "result"}
    missing = required - samples[0].keys()
    if missing:
        raise ValueError(f"Dataset {data_path} is missing fields: {sorted(missing)}")

    groups = {}
    for sample in samples:
        key = (
            len(str(sample["a"])),
            len(str(sample["b"])),
            len(str(sample["result"])),
        )
        groups.setdefault(key, []).append(sample)

    valid_samples = [
        sample
        for group in groups.values()
        if len({item["result"] for item in group}) > 1
        for sample in group
    ]
    if len(valid_samples) < n_pairs:
        raise ValueError(
            f"Requested {n_pairs} pairs, but only {len(valid_samples)} valid clean "
            f"image samples are available."
        )

    rng = random.Random(seed)
    pairs = []
    clean_samples = rng.sample(valid_samples, n_pairs)
    for pair_id, clean in enumerate(clean_samples):
        key = (len(str(clean["a"])), len(str(clean["b"])), len(str(clean["result"])))
        corrupt = rng.choice(
            [item for item in groups[key] if item["result"] != clean["result"]]
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "operation": operation,
                "clean_sample_id": clean.get("sample_id"),
                "clean_x": clean["a"],
                "clean_y": clean["b"],
                "clean_expr": clean.get("image_text", clean.get("expr")),
                "clean_answer": clean["result"],
                "clean_image_path": clean["image_path"],
                "corrupt_sample_id": corrupt.get("sample_id"),
                "corrupt_x": corrupt["a"],
                "corrupt_y": corrupt["b"],
                "corrupt_expr": corrupt.get("image_text", corrupt.get("expr")),
                "corrupt_answer": corrupt["result"],
                "corrupt_image_path": corrupt["image_path"],
                "clean_sample": clean,
                "corrupt_sample": corrupt,
            }
        )
    return pairs


def resolve_position(processor, text, image, position):
    inputs = make_inputs(processor, [text], [image])
    length = (
        int(inputs["attention_mask"][0].sum())
        if "attention_mask" in inputs
        else int(inputs["input_ids"].shape[1])
    )
    resolved = length + position if position < 0 else position
    if resolved < 0 or resolved >= length:
        raise ValueError(
            f"Position {position} resolves to {resolved}, outside 0..{length - 1}."
        )
    return resolved, length


def encode_image_prompt(processor, text, image, device):
    return inputs_to_device(make_inputs(processor, [text], [image]), device)


def answer_metrics_image(
    model,
    processor,
    tokenizer,
    prompt_text,
    image,
    answer_ids,
    patch=None,
):
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
    prompt_encoding = encode_image_prompt(processor, prompt_text, image, model.device)
    prompt_length = int(prompt_encoding["attention_mask"][0].sum())
    full_encoding = encode_image_prompt(
        processor, prompt_text + answer_text, image, model.device
    )

    if patch is None:
        outputs = model(**full_encoding, use_cache=False)
    elif isinstance(patch, list):
        outputs = patched_forward_many(model, full_encoding, patch)
    else:
        outputs = patched_forward(model, full_encoding, **patch)

    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length + len(answer_ids) - 1,
        device=outputs.logits.device,
    )
    log_probs = outputs.logits[0, prediction_positions].float().log_softmax(dim=-1)
    target_ids = torch.tensor(answer_ids, device=log_probs.device)
    correct = log_probs.argmax(dim=-1) == target_ids
    return {
        "score": float(log_probs.gather(1, target_ids[:, None]).mean()),
        "token_accuracy": float(correct.float().mean()),
        "iia": bool(correct.all()),
    }


def contrast_metrics_image(
    model,
    processor,
    tokenizer,
    prompt_text,
    image,
    clean_ids,
    corrupt_ids,
    patch=None,
):
    clean = answer_metrics_image(
        model, processor, tokenizer, prompt_text, image, clean_ids, patch
    )
    corrupt = answer_metrics_image(
        model, processor, tokenizer, prompt_text, image, corrupt_ids, patch
    )
    return {
        "contrast": clean["score"] - corrupt["score"],
        "clean_answer": clean,
        "corrupt_answer": corrupt,
    }


@torch.no_grad()
def run_pair(
    model,
    processor,
    tokenizer,
    model_name,
    pair,
    data_root,
    prompt,
    enable_thinking,
    layers,
    hooks,
    position_indices,
    layer_mode,
):
    clean_text = render_prompt(processor, prompt, enable_thinking)
    corrupt_text = render_prompt(processor, prompt, enable_thinking)
    clean_image = load_rgb_image(image_path_for(pair["clean_sample"], data_root))
    corrupt_image = load_rgb_image(image_path_for(pair["corrupt_sample"], data_root))
    clean_encoding = encode_image_prompt(processor, clean_text, clean_image, model.device)
    corrupt_encoding = encode_image_prompt(
        processor, corrupt_text, corrupt_image, model.device
    )

    clean_positions = []
    corrupt_positions = []
    clean_length = int(clean_encoding["attention_mask"][0].sum())
    corrupt_length = int(corrupt_encoding["attention_mask"][0].sum())
    for position in position_indices:
        clean_position, _ = resolve_position(
            processor, clean_text, clean_image, position
        )
        corrupt_position, _ = resolve_position(
            processor, corrupt_text, corrupt_image, position
        )
        clean_positions.append(clean_position)
        corrupt_positions.append(corrupt_position)

    clean_ids = answer_token_ids(tokenizer, pair["clean_answer"])
    corrupt_ids = answer_token_ids(tokenizer, pair["corrupt_answer"])
    clean_cache = cache_clean_activations(model, clean_encoding, layers, hooks)

    clean_metrics = contrast_metrics_image(
        model, processor, tokenizer, clean_text, clean_image, clean_ids, corrupt_ids
    )
    corrupt_metrics = contrast_metrics_image(
        model,
        processor,
        tokenizer,
        corrupt_text,
        corrupt_image,
        clean_ids,
        corrupt_ids,
    )
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
                clean_positions,
                corrupt_positions,
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
                patched_metrics = contrast_metrics_image(
                    model,
                    processor,
                    tokenizer,
                    corrupt_text,
                    corrupt_image,
                    clean_ids,
                    corrupt_ids,
                    patch,
                )
                patched_contrast = patched_metrics["contrast"]
                recovered = recovery(clean_contrast, corrupt_contrast, patched_contrast)
                rows.append(
                    {
                        **{
                            key: value
                            for key, value in pair.items()
                            if key not in {"clean_sample", "corrupt_sample"}
                        },
                        "model": model_name,
                        "input_format": "image",
                        "prompt": prompt,
                        "enable_thinking": enable_thinking,
                        "layer": (
                            layer_group[0]
                            if layer_mode == "individual"
                            else "all"
                        ),
                        "patched_layers": layer_group,
                        "layer_mode": layer_mode,
                        "hook": hook_name,
                        "position": position,
                        "clean_hf_position": clean_hf_position,
                        "corrupt_hf_position": corrupt_hf_position,
                        "clean_prompt_length": clean_length,
                        "corrupt_prompt_length": corrupt_length,
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
                        "corrupt_answer_iia": corrupt_metrics["corrupt_answer"][
                            "iia"
                        ],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=sorted(OPERATIONS), required=True)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--model-name", default="gemma4_12b_it")
    parser.add_argument("--output-dir", default="results/activation_patching_images")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root for relative image_path values. Defaults to data_path.parent.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--layer-mode",
        choices=["individual", "together"],
        default="individual",
    )
    parser.add_argument("--hooks", nargs="+", default=None)
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[-1],
        help="Raw image-prompt token positions. Negative values count from the end.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Read the arithmetic expression in the image, solve it, and output "
            "only the final number."
        ),
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root = args.data_root or args.data_path.parent

    model_path, model_name = resolve_model_for_loading(args.model_name)
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    model.eval()

    layers = validate_block_layers(model, args.layers)
    hooks = valid_hooks(args.hooks)
    paths = output_paths(Path(args.output_dir), args.operation, args.layer_mode, layers)
    paths["folder"].mkdir(parents=True, exist_ok=True)
    if paths["jsonl"].exists():
        paths["jsonl"].unlink()
        print(f"Removed already existing file {paths['jsonl']}")

    pairs = build_pairs_from_image_dataset(
        args.data_path,
        args.operation,
        args.n_pairs,
        args.seed,
    )

    first_text = render_prompt(processor, args.prompt, args.enable_thinking)
    first_image = load_rgb_image(image_path_for(pairs[0]["clean_sample"], args.data_root))
    for position in args.positions:
        print_position_summary(
            processor=processor,
            tokenizer=tokenizer,
            model=model,
            text=first_text,
            image=first_image,
            position=position,
        )

    for pair in tqdm(pairs, desc=f"{args.operation} image patching"):
        rows = run_pair(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            model_name=model_name,
            pair=pair,
            data_root=args.data_root,
            prompt=args.prompt,
            enable_thinking=args.enable_thinking,
            layers=layers,
            hooks=hooks,
            position_indices=args.positions,
            layer_mode=args.layer_mode,
        )
        append_jsonl(paths["jsonl"], rows)

    write_json(
        paths["metadata"],
        {
            "operation": args.operation,
            "model": model_name,
            "input_format": "image",
            "data_path": str(args.data_path),
            "data_root": str(args.data_root),
            "n_pairs": args.n_pairs,
            "layers": layers,
            "layer_mode": args.layer_mode,
            "hooks": hooks,
            "positions": args.positions,
            "prompt": args.prompt,
            "enable_thinking": args.enable_thinking,
            "hook_descriptions": HOOK_DESCRIPTIONS,
            "patch": (
                "For each clean/corrupt image pair, copy the clean activation at "
                "the resolved clean raw token position into the corrupt run at "
                "the resolved corrupt raw token position."
            ),
            "metric": (
                "contrast = mean log P(all clean-answer tokens) - mean log "
                "P(all corrupt-answer tokens). Recovery measures how far "
                "patching moves the corrupt contrast toward the clean contrast."
            ),
        },
    )
    print(f"Saved results: {paths['jsonl']}")


if __name__ == "__main__":
    main()
