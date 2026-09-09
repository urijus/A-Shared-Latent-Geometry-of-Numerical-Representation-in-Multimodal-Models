"""Extract multimodal activations from arithmetic expression images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.common.io import load_jsonl
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import LABEL_FIELDS
from src.models import (
    get_hidden_size,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


MODEL_NAME = "gemma4_12b_it"
DEFAULT_MODALITIES = ("addition", "subtraction", "multiplication")


class ImageArithmeticDataset(Dataset):
    def __init__(self, rows: list[dict], data_root: Path, prompt: str):
        self.rows = rows
        self.data_root = data_root
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.data_root / image_path

        item = {
            "prompt": self.prompt,
            "image_path": str(image_path),
            "image_text": str(row.get("image_text", "")),
            "expr": str(row.get("image_text", row.get("expr", ""))),
            "source_expr": str(row.get("source_expr", row.get("expr", ""))),
            "tokens": list(row.get("tokens", [])),
        }
        for field in LABEL_FIELDS:
            if field not in row:
                continue
            value = row[field]
            item[field] = str(value) if field in {"task", "operation"} else int(value)
        return item


def collate_image_examples(batch: list[dict]) -> dict:
    collated = {
        "prompt": [row["prompt"] for row in batch],
        "image_path": [row["image_path"] for row in batch],
        "image_text": [row["image_text"] for row in batch],
        "expr": [row["expr"] for row in batch],
        "source_expr": [row["source_expr"] for row in batch],
        "tokens": [row["tokens"] for row in batch],
    }
    for field in LABEL_FIELDS:
        if field in batch[0]:
            collated[field] = [row[field] for row in batch]
    return collated


def make_loader(rows: list[dict], data_root: Path, prompt: str, batch_size: int):
    return DataLoader(
        ImageArithmeticDataset(rows=rows, data_root=data_root, prompt=prompt),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_image_examples,
    )


def load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def make_messages(prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def render_prompt(processor, prompt: str, enable_thinking: bool) -> str:
    return processor.apply_chat_template(
        make_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def make_inputs(processor, prompts: list[str], images: list[Image.Image]):
    image_batches = [[image] for image in images]
    return processor(
        text=prompts,
        images=image_batches,
        return_tensors="pt",
        padding=True,
    )


def inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def final_input_positions(inputs) -> list[int]:
    if "attention_mask" not in inputs:
        return [inputs["input_ids"].shape[1] - 1] * inputs["input_ids"].shape[0]
    return (inputs["attention_mask"].sum(dim=1) - 1).tolist()


def image_token_id_candidates(model, processor, tokenizer) -> list[int]:
    candidates = []
    for owner in (processor, tokenizer, getattr(model, "config", None)):
        if owner is None:
            continue
        for attr in ("image_token_id", "boi_token_id"):
            value = getattr(owner, attr, None)
            if isinstance(value, int):
                candidates.append(value)
    image_token = getattr(tokenizer, "image_token", None)
    if image_token is not None:
        token_id = tokenizer.convert_tokens_to_ids(image_token)
        if isinstance(token_id, int) and token_id >= 0:
            candidates.append(token_id)
    return list(dict.fromkeys(candidates))


def last_image_token_positions(inputs, image_token_ids: list[int]) -> list[int]:
    if not image_token_ids:
        raise ValueError(
            "Could not infer an image token id for --position_strategy last_image_token. "
            "Use --position_strategy last_input for model-agnostic extraction."
        )
    input_ids = inputs["input_ids"]
    image_ids = torch.tensor(image_token_ids, device=input_ids.device)
    matches = (input_ids[..., None] == image_ids).any(dim=-1)
    positions = []
    for row_idx in range(matches.shape[0]):
        found = torch.nonzero(matches[row_idx], as_tuple=False).flatten()
        if found.numel() == 0:
            raise ValueError(f"No image tokens found in example {row_idx}.")
        positions.append(int(found[-1].item()))
    return positions


def select_positions(inputs, position_specs: list[int]) -> tuple[list[list[int]], list[str]]:
    lengths = (
        inputs["attention_mask"].sum(dim=1).tolist()
        if "attention_mask" in inputs
        else [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]
    )
    selected = []
    for length in lengths:
        row_positions = []
        for spec in position_specs:
            position = int(length) + spec if spec < 0 else spec
            if position < 0 or position >= int(length):
                raise ValueError(
                    f"Position {spec} resolves to {position}, outside 0..{int(length) - 1}."
                )
            row_positions.append(position)
        selected.append(row_positions)
    return selected, [str(spec) for spec in position_specs]


def positions_from_strategy(
    inputs,
    strategy: str,
    image_token_ids: list[int],
) -> tuple[list[list[int]], list[str]]:
    if strategy == "last_input":
        return [[position] for position in final_input_positions(inputs)], ["-1"]
    if strategy == "last_image_token":
        return [[position] for position in last_image_token_positions(inputs, image_token_ids)], ["last_image_token"]
    raise ValueError(f"Unknown position strategy: {strategy}")


def print_position_summary(tokenizer, inputs, selected_positions, selected_names):
    row_ids = inputs["input_ids"][0]
    length = (
        int(inputs["attention_mask"][0].sum())
        if "attention_mask" in inputs
        else int(row_ids.shape[0])
    )
    tokens = tokenizer.convert_ids_to_tokens(row_ids[:length].tolist())
    print("\nImage position summary:")
    print(f"  input length: {length}")
    for name, position in zip(selected_names, selected_positions[0]):
        token = tokens[position] if 0 <= position < len(tokens) else "<out>"
        print(f"  {name}: raw token {position}, token={token!r}")
    print()


@torch.no_grad()
def collect_image_activations(
    model,
    processor,
    tokenizer,
    loader,
    output_path: Path,
    model_name: str,
    modality: str,
    prompt: str,
    layer_indices: list[int] | None = None,
    position_specs: list[int] | None = None,
    position_strategy: str | None = None,
    enable_thinking: bool = False,
):
    model.eval()
    layer_indices = validate_block_layers(model, layer_indices)
    image_token_ids = image_token_id_candidates(model, processor, tokenizer)
    if position_specs is None and position_strategy is None:
        position_specs = [-1]

    n_examples = len(loader.dataset)
    n_layers = len(layer_indices)
    n_positions = len(position_specs) if position_specs is not None else 1
    hidden_dim = get_hidden_size(model)
    activations = torch.empty(
        n_examples,
        n_layers,
        n_positions,
        hidden_dim,
        dtype=torch.float16,
        device="cpu",
    )

    labels = []
    offset = 0
    position_names = None

    for batch in tqdm(loader):
        images = [load_rgb_image(path) for path in batch["image_path"]]
        prompts = [
            render_prompt(
                processor=processor,
                prompt=prompt,
                enable_thinking=enable_thinking,
            )
            for prompt in batch["prompt"]
        ]
        inputs = make_inputs(processor, prompts, images)
        if position_specs is not None:
            positions, selected_names = select_positions(inputs, position_specs)
        else:
            positions, selected_names = positions_from_strategy(
                inputs=inputs,
                strategy=position_strategy,
                image_token_ids=image_token_ids,
            )
        if position_names is None:
            position_names = selected_names
            print_position_summary(tokenizer, inputs, positions, selected_names)

        inputs = inputs_to_device(inputs, model.device)
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        batch_acts = []
        for layer in layer_indices:
            hidden = outputs.hidden_states[layer]
            selected = torch.stack(
                [
                    hidden[row_idx, row_positions, :]
                    for row_idx, row_positions in enumerate(positions)
                ],
                dim=0,
            )
            batch_acts.append(selected.cpu().to(torch.float16))

        batch_size = len(images)
        batch_acts = torch.stack(batch_acts, dim=1)
        activations[offset:offset + batch_size] = batch_acts

        for i in range(batch_size):
            label = {
                "expr": batch["expr"][i],
                "source_expr": batch["source_expr"][i],
                "image_text": batch["image_text"][i],
                "image_path": batch["image_path"][i],
                "tokens": batch["tokens"][i],
                "prompt": batch["prompt"][i],
            }
            for field in LABEL_FIELDS:
                if field in batch:
                    label[field] = batch[field][i]
            labels.append(label)
        offset += batch_size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "activations": activations,
            "labels": labels,
            "block_layers": layer_indices,
            "position_names": position_names,
            "position_indices": (
                position_specs if position_specs is not None else [position_strategy]
            ),
            "modality": modality,
            "model_name": model_name,
            "input_format": "image",
            "prompt": prompt,
            "position_strategy": position_strategy,
            "position_specs": position_specs,
            "enable_thinking": enable_thinking,
            "note": (
                "Layers are 1-based transformer blocks; embeddings are not saved. "
                "Image samples use a chat-template image placeholder, with PIL "
                "images passed separately through the processor. Non-negative "
                "positions are raw input-token indices from the beginning; "
                "negative positions are raw offsets from the end, e.g. -1 is "
                "the final input token."
            ),
        },
        output_path,
    )
    print("Saved:", output_path)
    print("Activation shape:", activations.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--prompt",
        default="Solve the arithmetic expression in the image. Output ONLY a number.",
    )
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Raw input-token positions to save. Non-negative values count from "
            "the beginning; negative values count from the end, so -1 is the "
            "final input token. Defaults to -1."
        ),
    )
    parser.add_argument(
        "--position_strategy",
        choices=("last_input", "last_image_token"),
        default=None,
        help="Deprecated compatibility option. Prefer --positions -1.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional 1-based transformer block numbers to save.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading multimodal model...")
    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print("Model load path:", model_path)
    print("Saved model name:", saved_model_name)
    model, processor, tokenizer = load_hf_model_and_processor(model_path)

    for modality in args.modalities:
        data_path = args.data_dir / modality / f"{modality}_images.jsonl"
        print(f"Extracting image activations for {modality}: {data_path}")
        rows = load_jsonl(data_path)
        if not rows:
            raise ValueError(f"No samples found in {data_path}.")

        loader = make_loader(
            rows=rows,
            data_root=data_path.parent,
            prompt=args.prompt,
            batch_size=args.batch_size,
        )
        output_path = args.output_dir / f"{modality}_baseline.pt"
        collect_image_activations(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            loader=loader,
            output_path=output_path,
            model_name=saved_model_name,
            modality=modality,
            prompt=args.prompt,
            layer_indices=args.layers,
            position_specs=args.positions,
            position_strategy=args.position_strategy,
            enable_thinking=args.enable_thinking,
        )

        saved = torch.load(output_path, map_location="cpu")
        print(saved["activations"].shape)
        print(saved["labels"][0])
        print(saved["position_names"])
        print(saved["block_layers"][:5], saved["block_layers"][-5:])


if __name__ == "__main__":
    main()
