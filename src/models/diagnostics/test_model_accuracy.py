import argparse
import re
from pathlib import Path

import torch
from tqdm import tqdm

from src.common.io import load_jsonl, save_jsonl
from src.models import load_hf_model, resolve_model_for_loading


MODEL_NAME = "pythia"


def parse_generated_ints(text):
    return [int(value) for value in re.findall(r"-?\d+", text)]


def batched(samples, batch_size):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def apply_chat_template(tokenizer, expr):
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("Tokenizer does not contain chat template option.")

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": expr}],
        tokenize=False,
        add_generation_prompt=True,
    )


def uses_chat_template(model_name):
    return model_name.lower() in {
        "gemma3_12b",
        "google/gemma-3-12b-it",
        "gemma4_12b_it",
        "google/gemma-4-12b-it",
        "gemma4_e4b_it",
        "google/gemma-4-e4b-it",
        "ministral3_14b_it_bf16",
        "mistralai/ministral-3-14b-instruct-2512-bf16",
    }


def configure_padding(tokenizer):
    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except (AttributeError, TypeError):
        pass
    try:
        tokenizer.padding_side = "left"
    except (AttributeError, TypeError):
        pass


def generation_kwargs(tokenizer):
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    return {"pad_token_id": pad_token_id} if pad_token_id is not None else {}


def requires_carry_or_borrow(sample):
    return bool(sample.get("requires_carry", False)) or bool(sample.get("requires_borrow", False))


@torch.no_grad()
def evaluate_accuracy(
    model,
    tokenizer,
    samples,
    batch_size,
    max_new_tokens,
    print_examples,
    use_chat_template,
):
    configure_padding(tokenizer)

    n_correct = 0
    n_total = 0
    examples_printed = 0
    correct_samples = []

    for batch in tqdm(list(batched(samples, batch_size))):
        exprs = [str(sample["expr"]) for sample in batch]
        model_inputs = [
            apply_chat_template(tokenizer, expr) if use_chat_template else expr
            for expr in exprs
        ]
        expected = [int(sample["result"]) for sample in batch]

        encoding = tokenizer(
            model_inputs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)

        generated = model.generate(
            **encoding,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **generation_kwargs(tokenizer),
        )

        prompt_length = encoding["input_ids"].shape[1]

        for i, sample in enumerate(batch):
            new_tokens = generated[i, prompt_length:]
            answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_numbers = parse_generated_ints(answer_text)

            is_correct = expected[i] in generated_numbers
            n_correct += int(is_correct)
            n_total += 1

            if is_correct:
                correct_samples.append(dict(sample))

            if examples_printed < print_examples:
                print(
                    {
                        "expr": sample["expr"],
                        "model_input": model_inputs[i],
                        "expected": expected[i],
                        "generated": answer_text.strip(),
                        "generated_numbers": generated_numbers,
                        "correct": is_correct,
                    }
                )
                examples_printed += 1

    accuracy = n_correct / n_total if n_total else 0.0
    metrics = {
        "correct": n_correct,
        "total": n_total,
        "accuracy": accuracy,
    }
    return metrics, correct_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--print_examples", type=int, default=5)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--test_carry_borrow", action="store_true") # test for just subset that uses a carry or borrow

    return parser.parse_args()


def main():
    args = parse_args()

    use_chat_template = args.use_chat_template or uses_chat_template(args.model)


    if args.label is not None:
        print(f"Evaluation label: {args.label}")

    samples = load_jsonl(Path(args.data_path))
    if args.limit is not None:
        samples = samples[:args.limit]

    if args.test_carry_borrow:
        print("Using only operations with requires_carry or requires_borrow.")
        samples = [sample for sample in samples if requires_carry_or_borrow(sample)]

    print("Loading model...")
    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print("Model load path:", model_path)
    print("Saved model name:", saved_model_name)
    print("Use chat template:", use_chat_template)

    model, tokenizer = load_hf_model(model_path)

    metrics, correct_samples = evaluate_accuracy(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        print_examples=args.print_examples,
        use_chat_template=use_chat_template,
    )

    print("Correct:", metrics["correct"])
    print("Total:", metrics["total"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    save_jsonl(correct_samples, args.output_path)
    print("Saved correct-only dataset:", args.output_path)


if __name__ == "__main__":
    main()
