"""Evaluate arithmetic accuracy on image-rendered expressions."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.common.io import load_jsonl, save_jsonl
from src.models import load_hf_model_and_processor, resolve_model_for_loading


MODEL_NAME = "gemma4_12b_it"


def parse_generated_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


def batched(samples: list[dict], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def image_path_for(sample: dict, data_root: Path) -> Path:
    image_path = Path(sample["image_path"])
    return image_path if image_path.is_absolute() else data_root / image_path


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def is_pixtral_processor(processor) -> bool:
    return processor.__class__.__name__ == "PixtralProcessor"


def make_messages(prompt: str, image: Image.Image | None = None) -> list[dict]:
    image_block = {"type": "image"} if image is None else {"type": "image", "image": image}
    return [
        {
            "role": "user",
            "content": [
                image_block,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def render_prompt(processor, prompt: str, enable_thinking: bool) -> str:
    kwargs = {}
    if enable_thinking:
        kwargs["enable_thinking"] = True
    return processor.apply_chat_template(
        make_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def make_inputs(
    processor,
    prompts: list[str],
    images: list[Image.Image],
    prompt: str,
    enable_thinking: bool,
):
    if is_pixtral_processor(processor):
        messages = [make_messages(prompt, image) for image in images]
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    image_batches = [[image] for image in images]
    return processor(
        text=prompts,
        images=image_batches,
        return_tensors="pt",
        padding=True,
    )


def decode_answer(processor, token_ids: torch.Tensor) -> str:
    raw_text = processor.decode(token_ids, skip_special_tokens=False)
    try:
        parsed = processor.parse_response(raw_text)
    except Exception:
        return raw_text

    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("answer", "content", "final", "text"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return str(parsed)


def move_inputs_to_model(inputs, model):
    inputs = inputs.to(model.device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None and torch.is_floating_point(pixel_values):
        dtype = getattr(model, "dtype", None)
        if dtype is not None:
            inputs["pixel_values"] = pixel_values.to(dtype=dtype)
    return inputs


def requires_carry_or_borrow(sample: dict) -> bool:
    return bool(sample.get("requires_carry", False)) or bool(
        sample.get("requires_borrow", False)
    )


@torch.no_grad()
def evaluate_image_accuracy(
    model,
    processor,
    samples: list[dict],
    data_root: Path,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    print_examples: int,
    enable_thinking: bool,
) -> tuple[dict, list[dict]]:
    correct = 0
    total = 0
    examples_printed = 0
    correct_samples = []

    for batch in tqdm(list(batched(samples, batch_size))):
        image_paths = [image_path_for(sample, data_root) for sample in batch]
        images = [load_image(path) for path in image_paths]
        prompts = [render_prompt(processor, prompt, enable_thinking) for _ in batch]

        inputs = move_inputs_to_model(
            make_inputs(processor, prompts, images, prompt, enable_thinking),
            model,
        )
        input_len = inputs["input_ids"].shape[-1]
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        for idx, sample in enumerate(batch):
            new_tokens = generated[idx][input_len:]
            answer_text = decode_answer(processor, new_tokens)
            generated_numbers = parse_generated_ints(answer_text)
            expected = int(sample["result"])
            is_correct = expected in generated_numbers

            correct += int(is_correct)
            total += 1
            if is_correct:
                row = dict(sample)
                row["image_path"] = str(image_paths[idx].resolve())
                correct_samples.append(row)

            if examples_printed < print_examples:
                print(
                    {
                        "image_text": sample.get("image_text"),
                        "image_path": str(image_paths[idx]),
                        "expected": expected,
                        "generated": answer_text.strip(),
                        "generated_numbers": generated_numbers,
                        "correct": is_correct,
                    }
                )
                examples_printed += 1

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
    }, correct_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--prompt",
        default="Solve the arithmetic expression in the image. Output ONLY a number.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--print_examples", type=int, default=5)
    parser.add_argument("--test_carry_borrow", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.label is not None:
        print(f"Evaluation label: {args.label}")

    samples = load_jsonl(args.data_path)
    if args.limit is not None:
        samples = samples[:args.limit]
    if args.test_carry_borrow:
        print("Using only operations with requires_carry or requires_borrow.")
        samples = [sample for sample in samples if requires_carry_or_borrow(sample)]
    if not samples:
        raise ValueError(f"No samples to evaluate from {args.data_path}.")

    print("Loading multimodal model...")
    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print("Model load path:", model_path)
    print("Saved model name:", saved_model_name)
    model, processor, _tokenizer = load_hf_model_and_processor(model_path)

    metrics, correct_samples = evaluate_image_accuracy(
        model=model,
        processor=processor,
        samples=samples,
        data_root=args.data_path.parent,
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        print_examples=args.print_examples,
        enable_thinking=args.enable_thinking,
    )

    print("Correct:", metrics["correct"])
    print("Total:", metrics["total"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    save_jsonl(correct_samples, args.output_path)
    print("Saved correct-only image dataset:", args.output_path)


if __name__ == "__main__":
    main()
