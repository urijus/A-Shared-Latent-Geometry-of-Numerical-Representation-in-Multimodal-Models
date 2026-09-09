"""Coarse Ministral result localization with activation patching.

This is a cheap, no-training localization pass. It patches only post-block
residual activations at offsets from the final pre-generation token.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.experiments.arithmetic_reference.activation_patching.text.common import (
    answer_token_ids,
    append_jsonl,
    build_pairs_from_dataset,
    cache_clean_activations,
    patched_forward,
    recovery,
    write_json,
)
from src.experiments.arithmetic_reference.activation_patching.image.run_activation_patching import (
    build_pairs_from_image_dataset,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs as make_image_inputs,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    apply_chat_template,
)
from src.models import (
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


TASKS = {
    "T+": ("text", "addition"),
    "I+": ("image", "addition"),
}
DEFAULT_LAYERS = [24, 27, 30, 33, 35, 37, 38, 39, 40]
DEFAULT_OFFSETS = [-2, -1, 0]


def task_data_path(task: str, model_slug: str, fmt: str) -> Path:
    modality, operation = TASKS[task]
    if modality == "text":
        return Path(
            f"dataset/baseline/{model_slug}/{fmt}/"
            f"model_correct_with_prompt/{operation}_baseline.jsonl"
        )
    return Path(
        f"dataset/baseline_images/{model_slug}/{fmt}/"
        f"model_correct_with_prompt/{operation}/{operation}_images.jsonl"
    )


def safe_repr(value) -> str:
    return ascii(value)


def token_label(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    except (AttributeError, TypeError):
        return tokenizer.decode([token_id], skip_special_tokens=False)


def encode_text_prompt(tokenizer, prompt: str, device):
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return {name: tensor.to(device) for name, tensor in encoding.items()}


def image_messages(prompt: str, image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def image_messages_without_payload(prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_image_prompt(processor, prompt: str, image, model):
    try:
        inputs = processor.apply_chat_template(
            image_messages(prompt, image),
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    except Exception:
        rendered = processor.apply_chat_template(
            image_messages_without_payload(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = make_image_inputs(processor, [rendered], [image])
    return inputs_to_device(
        inputs,
        model.device,
        dtype=getattr(model, "dtype", None),
    )


def append_answer(prompt_encoding: dict, answer_ids: list[int]) -> dict:
    answer = torch.tensor(
        [answer_ids],
        dtype=prompt_encoding["input_ids"].dtype,
        device=prompt_encoding["input_ids"].device,
    )
    attention = prompt_encoding.get(
        "attention_mask",
        torch.ones_like(prompt_encoding["input_ids"]),
    )
    full = {}
    for key, value in prompt_encoding.items():
        if key == "input_ids":
            full[key] = torch.cat([value, answer], dim=1)
        elif key == "attention_mask":
            full[key] = torch.cat([value, torch.ones_like(answer)], dim=1)
        else:
            full[key] = value
    if "attention_mask" not in full:
        full["attention_mask"] = torch.cat(
            [attention, torch.ones_like(answer)],
            dim=1,
        )
    return full


@torch.no_grad()
def answer_metrics(model, prompt_encoding: dict, answer_ids: list[int], patch=None):
    prompt_length = (
        int(prompt_encoding["attention_mask"][0].sum())
        if "attention_mask" in prompt_encoding
        else int(prompt_encoding["input_ids"].shape[1])
    )
    full_encoding = append_answer(prompt_encoding, answer_ids)
    if patch is None:
        outputs = model(**full_encoding, use_cache=False)
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


def contrast_metrics(model, prompt_encoding, clean_ids, corrupt_ids, patch=None):
    clean = answer_metrics(model, prompt_encoding, clean_ids, patch)
    corrupt = answer_metrics(model, prompt_encoding, corrupt_ids, patch)
    return {
        "contrast": clean["score"] - corrupt["score"],
        "clean_answer": clean,
        "corrupt_answer": corrupt,
    }


def final_position(encoding: dict) -> int:
    if "attention_mask" in encoding:
        return int(encoding["attention_mask"][0].sum()) - 1
    return int(encoding["input_ids"].shape[1]) - 1


def resolve_offsets(encoding: dict, offsets: list[int]) -> dict[int, int]:
    final = final_position(encoding)
    positions = {offset: final + offset for offset in offsets}
    for offset, position in positions.items():
        if position < 0 or position > final:
            raise ValueError(
                f"Offset {offset} resolves to {position}, outside 0..{final}."
            )
    return positions


def print_position_check(task, tokenizer, encoding, offsets):
    positions = resolve_offsets(encoding, offsets)
    ids = encoding["input_ids"][0].detach().cpu().tolist()
    print(f"[{task}] Resolved positions for one example")
    for offset in offsets:
        position = positions[offset]
        token_id = ids[position]
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        print(
            f"[{task}] offset {offset:>2} -> position {position} -> "
            f"id {token_id} -> token {safe_repr(token_label(tokenizer, token_id))} "
            f"-> decoded {safe_repr(decoded)}"
        )


def existing_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            keys.add((row["task"], int(row["layer"]), int(row["relative_position"])))
    return keys


def summarize_site(rows: list[dict]) -> dict:
    recoveries = [
        row["recovery"]
        for row in rows
        if row.get("recovery") is not None and math.isfinite(row["recovery"])
    ]
    if not recoveries:
        mean_value = None
        std_value = None
    else:
        mean_value = sum(recoveries) / len(recoveries)
        if len(recoveries) > 1:
            variance = sum((value - mean_value) ** 2 for value in recoveries)
            std_value = math.sqrt(variance / (len(recoveries) - 1))
        else:
            std_value = 0.0
    first = rows[0]
    return {
        "task": first["task"],
        "layer": int(first["layer"]),
        "relative_position": int(first["relative_position"]),
        "mean_recovery": mean_value,
        "std_recovery": std_value,
        "n_pairs": len(rows),
    }


def save_heatmap(summary_rows: list[dict], task: str, output_dir: Path) -> None:
    task_rows = [row for row in summary_rows if row["task"] == task]
    if not task_rows:
        return
    layers = sorted({int(row["layer"]) for row in task_rows})
    offsets = sorted({int(row["relative_position"]) for row in task_rows})
    values = {
        (int(row["layer"]), int(row["relative_position"])): (
            float(row["mean_recovery"])
            if row["mean_recovery"] is not None
            else float("nan")
        )
        for row in task_rows
    }
    matrix = [
        [values.get((layer, offset), float("nan")) for offset in offsets]
        for layer in layers
    ]
    fig, ax = plt.subplots(figsize=(4.5, 5.0))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(f"{task} recovery")
    ax.set_xlabel("offset from final token")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(offsets)), offsets)
    ax.set_yticks(range(len(layers)), layers)
    fig.colorbar(image, ax=ax, label="mean recovery")
    fig.tight_layout()
    path = output_dir / f"{task.replace('+', 'plus')}_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_best_layer_plot(summary_rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for task in sorted({row["task"] for row in summary_rows}):
        task_rows = [row for row in summary_rows if row["task"] == task]
        best_by_layer = {}
        for row in task_rows:
            value = row["mean_recovery"]
            if value is None:
                continue
            layer = int(row["layer"])
            best_by_layer[layer] = max(value, best_by_layer.get(layer, value))
        if best_by_layer:
            layers = sorted(best_by_layer)
            ax.plot(layers, [best_by_layer[layer] for layer in layers], marker="o", label=task)
    ax.set_xlabel("layer")
    ax.set_ylabel("best-position mean recovery")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "best_position_by_layer.png", dpi=200)
    plt.close(fig)


def task_pairs(task: str, data_path: Path, n_pairs: int, seed: int):
    modality, operation = TASKS[task]
    if modality == "text":
        return build_pairs_from_dataset(data_path, operation, n_pairs, seed)
    return build_pairs_from_image_dataset(data_path, operation, n_pairs, seed)


def task_stem(task: str) -> str:
    return task.replace("+", "plus").replace("-", "minus")


def serializable_pair(pair: dict) -> dict:
    return {
        key: value
        for key, value in pair.items()
        if key not in {"clean_sample", "corrupt_sample"}
    }


def save_pair_set(task: str, pairs: list[dict], output_dir: Path, overwrite: bool) -> None:
    pair_path = output_dir / f"{task_stem(task)}_pairs.jsonl"
    if overwrite and pair_path.exists():
        pair_path.unlink()
    if pair_path.exists():
        return
    append_jsonl(pair_path, [serializable_pair(pair) for pair in pairs])


def prepare_pair(task, pair, model, processor, tokenizer, image_prompt, data_root):
    modality, _operation = TASKS[task]
    clean_ids = answer_token_ids(tokenizer, pair["clean_answer"])
    corrupt_ids = answer_token_ids(tokenizer, pair["corrupt_answer"])
    if modality == "text":
        clean_prompt = apply_chat_template(tokenizer, pair["clean_expr"])
        corrupt_prompt = apply_chat_template(tokenizer, pair["corrupt_expr"])
        clean_encoding = encode_text_prompt(tokenizer, clean_prompt, model.device)
        corrupt_encoding = encode_text_prompt(tokenizer, corrupt_prompt, model.device)
    else:
        clean_image = load_rgb_image(image_path_for(pair["clean_sample"], data_root))
        corrupt_image = load_rgb_image(image_path_for(pair["corrupt_sample"], data_root))
        clean_encoding = encode_image_prompt(processor, image_prompt, clean_image, model)
        corrupt_encoding = encode_image_prompt(
            processor,
            image_prompt,
            corrupt_image,
            model,
        )
    return {
        "clean_encoding": clean_encoding,
        "corrupt_encoding": corrupt_encoding,
        "clean_ids": clean_ids,
        "corrupt_ids": corrupt_ids,
    }


@torch.no_grad()
def run_task(
    task,
    model,
    processor,
    tokenizer,
    layers,
    offsets,
    n_pairs,
    seed,
    image_prompt,
    data_path,
    output_path,
    overwrite,
):
    data_root = data_path.parent
    print(f"[{task}] Preparing {n_pairs} clean/corrupt pairs")
    pairs = task_pairs(task, data_path, n_pairs, seed)
    save_pair_set(task, pairs, output_path.parent, overwrite)
    prepared = [
        prepare_pair(task, pair, model, processor, tokenizer, image_prompt, data_root)
        for pair in pairs
    ]
    print_position_check(task, tokenizer, prepared[0]["clean_encoding"], offsets)

    done = set() if overwrite else existing_keys(output_path)
    for layer_index, layer in enumerate(layers, start=1):
        print(f"[{task}] Patching layer {layer} ({layer_index}/{len(layers)})")
        active_offsets = [
            offset for offset in offsets if (task, layer, offset) not in done
        ]
        if not active_offsets:
            continue
        rows_by_offset = {offset: [] for offset in active_offsets}
        for pair_id, item in enumerate(prepared):
            clean_cache = cache_clean_activations(
                model,
                item["clean_encoding"],
                [layer],
                ["resid_post"],
            )
            clean_metrics = contrast_metrics(
                model,
                item["clean_encoding"],
                item["clean_ids"],
                item["corrupt_ids"],
            )
            corrupt_metrics = contrast_metrics(
                model,
                item["corrupt_encoding"],
                item["clean_ids"],
                item["corrupt_ids"],
            )
            clean_positions = resolve_offsets(item["clean_encoding"], active_offsets)
            corrupt_positions = resolve_offsets(item["corrupt_encoding"], active_offsets)
            for offset in active_offsets:
                clean_position = clean_positions[offset]
                corrupt_position = corrupt_positions[offset]
                patch = {
                    "clean_activation": clean_cache[(layer, "resid_post")],
                    "layer": layer,
                    "hook_name": "resid_post",
                    "clean_position": clean_position,
                    "corrupt_position": corrupt_position,
                }
                patched_metrics = contrast_metrics(
                    model,
                    item["corrupt_encoding"],
                    item["clean_ids"],
                    item["corrupt_ids"],
                    patch=patch,
                )
                recovered = recovery(
                    clean_metrics["contrast"],
                    corrupt_metrics["contrast"],
                    patched_metrics["contrast"],
                )
                rows_by_offset[offset].append(
                    {
                        "task": task,
                        "pair_id": pair_id,
                        "layer": layer,
                        "relative_position": offset,
                        "hook": "resid_post",
                        "clean_hf_position": clean_position,
                        "corrupt_hf_position": corrupt_position,
                        "clean_prompt_length": final_position(item["clean_encoding"]) + 1,
                        "corrupt_prompt_length": final_position(item["corrupt_encoding"]) + 1,
                        "clean_answer_token_ids": item["clean_ids"],
                        "corrupt_answer_token_ids": item["corrupt_ids"],
                        "clean_contrast": clean_metrics["contrast"],
                        "corrupt_contrast": corrupt_metrics["contrast"],
                        "patched_contrast": patched_metrics["contrast"],
                        "recovery": recovered,
                        "clean_answer_iia": clean_metrics["clean_answer"]["iia"],
                        "corrupt_answer_iia": corrupt_metrics["corrupt_answer"]["iia"],
                        "patched_clean_iia": patched_metrics["clean_answer"]["iia"],
                    }
                )
        for offset, pair_rows in rows_by_offset.items():
            summary = summarize_site(pair_rows)
            append_jsonl(output_path, [summary])
            done.add((task, layer, offset))
    print(f"[{task}] Complete")


def print_top_sites(summary_rows: list[dict]) -> None:
    for task in sorted({row["task"] for row in summary_rows}):
        rows = [
            row
            for row in summary_rows
            if row["task"] == task and row["mean_recovery"] is not None
        ]
        rows.sort(key=lambda row: row["mean_recovery"], reverse=True)
        print(f"\n{task}:")
        print("layer | offset | recovery")
        for row in rows[:5]:
            print(
                f"{row['layer']:>5} | {row['relative_position']:>6} | "
                f"{row['mean_recovery']:.4f}"
            )


def common_region(summary_rows: list[dict]) -> None:
    by_task = defaultdict(list)
    for row in summary_rows:
        if row["mean_recovery"] is not None:
            by_task[row["task"]].append(row)
    if not all(task in by_task for task in ("T+", "I+")):
        return
    for rows in by_task.values():
        rows.sort(key=lambda row: row["mean_recovery"], reverse=True)
    text_top = {(row["layer"], row["relative_position"]) for row in by_task["T+"][:5]}
    image_top = {(row["layer"], row["relative_position"]) for row in by_task["I+"][:5]}
    overlap = sorted(text_top & image_top)
    if overlap:
        print(f"\nCommon top-5 layer/offset sites: {overlap}")
        return
    text_layers = {row["layer"] for row in by_task["T+"][:5]}
    image_layers = {row["layer"] for row in by_task["I+"][:5]}
    layer_overlap = sorted(text_layers & image_layers)
    print(f"\nCommon top-5 layers across text/image: {layer_overlap or 'none'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ministral3_14b_it_bf16")
    parser.add_argument("--model_slug", "--model-slug", default="ministral3_14b_it_bf16")
    parser.add_argument("--format", default="digits")
    parser.add_argument("--tasks", nargs="+", default=["T+", "I+"], choices=sorted(TASKS))
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--offsets", type=int, nargs="+", default=DEFAULT_OFFSETS)
    parser.add_argument("--n_pairs", "--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=Path("results/mistral/coarse_activation_patching"),
    )
    parser.add_argument("--image_prompt", "--image-prompt", default="Output ONLY a number.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.jsonl"
    if args.overwrite and output_path.exists():
        output_path.unlink()

    model_path, model_name = resolve_model_for_loading(args.model)
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    validate_block_layers(model, args.layers)

    print(f"Model: {model_name}")
    print(f"Layers: {args.layers}")
    print(f"Relative offsets: {args.offsets}")
    print("Hook: resid_post")

    for task in args.tasks:
        data_path = task_data_path(task, args.model_slug, args.format)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Missing data for {task}: {data_path}. Run Stage 0 accuracy first."
            )
        run_task(
            task=task,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            layers=args.layers,
            offsets=args.offsets,
            n_pairs=args.n_pairs,
            seed=args.seed,
            image_prompt=args.image_prompt,
            data_path=data_path,
            output_path=output_path,
            overwrite=args.overwrite,
        )

    summary_rows = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            summary_rows = [json.loads(line) for line in handle]
    for task in ("T+", "I+"):
        save_heatmap(summary_rows, task, args.output_dir)
    save_best_layer_plot(summary_rows, args.output_dir)
    print_top_sites(summary_rows)
    common_region(summary_rows)
    write_json(
        args.output_dir / "metadata.json",
        {
            "model": model_name,
            "tasks": args.tasks,
            "layers": args.layers,
            "relative_offsets": args.offsets,
            "n_pairs": args.n_pairs,
            "seed": args.seed,
            "hook": "resid_post",
            "position_rule": "final_prompt_token_position + relative_offset",
            "metric": (
                "Teacher-forced contrast recovery using complete clean/corrupt "
                "answer token sequences."
            ),
        },
    )
    print(f"\nSaved summary: {output_path}")
    print(f"Saved plots: {args.output_dir}")


if __name__ == "__main__":
    main()
