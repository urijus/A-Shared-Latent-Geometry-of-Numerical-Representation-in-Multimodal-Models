from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_PATH = "models/gemma-4-12B-it"
IMAGE_PATH = "dataset/baseline_images/gemma4_12b_it/digits/accuracy_with_prompt/multiplication/png/000000.png"


def main():
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
    )

    print("Processor:", type(processor))
    print("Tokenizer:", type(processor.tokenizer))
    print("Model:", type(model))
    print("Model device:", model.device)
    print("Model dtype:", next(model.parameters()).dtype)
    print("Pad token:", processor.tokenizer.pad_token, processor.tokenizer.pad_token_id)
    print("EOS token:", processor.tokenizer.eos_token, processor.tokenizer.eos_token_id)

    image = Image.open(IMAGE_PATH).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Answer only with the result of the mathematical expression.",
                },
            ],
        }
    ]

    # IMPORTANT: start with thinking ON for Gemma 4 12B.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    print("\n=== RENDERED PROMPT ===")
    print(text)
    print("=======================\n")

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    print("input_ids shape:", inputs["input_ids"].shape)
    print("attention_mask shape:", inputs["attention_mask"].shape)
    print("Decoded input:")
    print(processor.decode(inputs["input_ids"][0], skip_special_tokens=False))

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    new_tokens = outputs[0][input_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=False)

    print("\n=== RAW OUTPUT ===")
    print(raw)
    print("==================\n")

    try:
        parsed = processor.parse_response(raw)
        print("=== PARSED OUTPUT ===")
        print(parsed)
        print("=====================")
    except Exception as e:
        print("parse_response failed:", repr(e))


if __name__ == "__main__":
    main()