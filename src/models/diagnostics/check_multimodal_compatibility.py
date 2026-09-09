"""Smoke-test a multimodal HF model before representation experiments."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoConfig, AutoProcessor

from src.models import (
    get_blocks,
    get_hidden_size,
    get_num_hidden_layers,
    is_ministral3_model_name,
    load_hf_model_and_processor,
    resolve_model_for_loading,
)


def render_expression_image(expression: str, image_size: int = 384) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), expression, font=font)
    x = (image_size - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (image_size - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), expression, fill="black", font=font)
    return image


def token_label(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    except (AttributeError, TypeError):
        return tokenizer.decode([token_id], skip_special_tokens=False)


def safe_repr(value) -> str:
    return ascii(value)


def print_token_sequence(tokenizer, input_ids: torch.Tensor, title: str, max_tokens: int) -> None:
    ids = input_ids.detach().cpu().tolist()
    print(f"\n{title}")
    for position, token_id in enumerate(ids[:max_tokens]):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        print(
            f"  {position:>3} id={token_id:<8} "
            f"token={safe_repr(token_label(tokenizer, token_id)):<18} "
            f"decoded={safe_repr(decoded)}"
        )
    if len(ids) > max_tokens:
        print(f"  ... {len(ids) - max_tokens} more tokens")
    final_position = len(ids) - 1
    final_id = ids[final_position]
    print(
        "  final pre-generation token: "
        f"position={final_position}, id={final_id}, "
        f"token={safe_repr(token_label(tokenizer, final_id))}, "
        f"decoded={safe_repr(tokenizer.decode([final_id], skip_special_tokens=False))}"
    )


def text_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def image_messages(prompt: str, image: Image.Image) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def apply_template(processor, messages: list[dict], *, tokenize: bool):
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": tokenize,
    }
    if tokenize:
        kwargs.update(
            {
                "return_dict": True,
                "return_tensors": "pt",
            }
        )
    return processor.apply_chat_template(messages, **kwargs)


def build_text_inputs(processor, prompt: str):
    messages = text_messages(prompt)
    rendered = apply_template(processor, messages, tokenize=False)
    try:
        inputs = apply_template(processor, messages, tokenize=True)
    except Exception:
        inputs = processor(text=rendered, return_tensors="pt")
    return rendered, inputs


def build_image_inputs(processor, prompt: str, image: Image.Image):
    messages = image_messages(prompt, image)
    rendered = apply_template(processor, messages, tokenize=False)
    try:
        inputs = apply_template(processor, messages, tokenize=True)
    except Exception:
        inputs = processor(text=rendered, images=[image], return_tensors="pt")
    return rendered, inputs


def move_inputs_to_model(inputs, model):
    inputs = inputs.to(model.device)
    pixel_values = inputs.get("pixel_values")
    if pixel_values is not None and torch.is_floating_point(pixel_values):
        dtype = getattr(model, "dtype", None)
        if dtype is not None:
            inputs["pixel_values"] = pixel_values.to(dtype=dtype)
    return inputs


def load_processor_only(model_path: str, model_name: str):
    kwargs = {}
    if is_ministral3_model_name(model_name) or is_ministral3_model_name(model_path):
        kwargs["fix_mistral_regex"] = True
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
    tokenizer = getattr(processor, "tokenizer", processor)
    config = AutoConfig.from_pretrained(model_path)
    return processor, tokenizer, config


def decode_new_tokens(processor, generated: torch.Tensor, input_len: int) -> str:
    new_tokens = generated[0, input_len:]
    return processor.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


@torch.no_grad()
def run_generation(model, processor, inputs, max_new_tokens: int) -> str:
    inputs = move_inputs_to_model(inputs, model)
    input_len = inputs["input_ids"].shape[-1]
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return decode_new_tokens(processor, generated, input_len)


@torch.no_grad()
def check_hidden_states(model, inputs) -> None:
    inputs = move_inputs_to_model(inputs, model)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    print("\nHidden-state exposure")
    print(f"  hidden_states returned: {hidden_states is not None}")
    if hidden_states is not None:
        print(f"  hidden_state tensors: {len(hidden_states)}")
        print(f"  first shape: {tuple(hidden_states[0].shape)}")
        print(f"  final shape: {tuple(hidden_states[-1].shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ministral3_14b_it_bf16")
    parser.add_argument("--text_prompt", default="Output ONLY a number. 10 - 6 =")
    parser.add_argument("--image_expression", default="10 - 6 =")
    parser.add_argument("--image_prompt", default="Output ONLY a number.")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_print_tokens", type=int, default=96)
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Only load/tokenize and inspect config; do not run model.generate.",
    )
    parser.add_argument(
        "--skip_model_load",
        action="store_true",
        help="Load only config/processor/tokenizer; useful before weights are downloaded.",
    )
    parser.add_argument(
        "--skip_hidden_states",
        action="store_true",
        help="Do not run the output_hidden_states=True forward pass.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print("Model load path:", model_path)
    print("Saved model name:", saved_model_name)

    if args.skip_model_load:
        processor, tokenizer, config = load_processor_only(model_path, args.model)
        model = None
        config_proxy = SimpleNamespace(config=config)
        print("Model class: <not loaded>")
        print("Transformer layers from config:", get_num_hidden_layers(config_proxy))
        print("Hidden size from config:", get_hidden_size(config_proxy))
    else:
        model, processor, tokenizer = load_hf_model_and_processor(model_path)
        print("Model class:", model.__class__.__name__)
        print("Transformer layers from config:", get_num_hidden_layers(model))
        print("Hidden size from config:", get_hidden_size(model))
        print("Hookable block modules:", len(get_blocks(model)))
    print("Processor class:", processor.__class__.__name__)
    print("Tokenizer class:", tokenizer.__class__.__name__)

    text_rendered, text_inputs = build_text_inputs(processor, args.text_prompt)
    print("\nRendered text prompt:")
    print(safe_repr(text_rendered))
    print_token_sequence(
        tokenizer,
        text_inputs["input_ids"][0],
        "Text prompt token sequence",
        args.max_print_tokens,
    )

    image = render_expression_image(args.image_expression)
    image_rendered, image_inputs = build_image_inputs(processor, args.image_prompt, image)
    print("\nRendered image prompt:")
    print(safe_repr(image_rendered))
    print_token_sequence(
        tokenizer,
        image_inputs["input_ids"][0],
        "Image prompt token sequence",
        args.max_print_tokens,
    )

    if args.skip_model_load or args.skip_generation:
        return

    text_answer = run_generation(model, processor, text_inputs, args.max_new_tokens)
    print("\nText generation:", safe_repr(text_answer))

    image_answer = run_generation(model, processor, image_inputs, args.max_new_tokens)
    print("Image generation:", safe_repr(image_answer))

    if not args.skip_hidden_states:
        check_hidden_states(model, text_inputs)


if __name__ == "__main__":
    main()
