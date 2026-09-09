"""Evaluate arithmetic accuracy on audio-rendered expressions."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly
from tqdm import tqdm

from src.common.io import load_jsonl, save_jsonl
from src.models import load_hf_model_and_processor, resolve_model_for_loading


MODEL_NAME = "gemma4_12b_it"


def parse_generated_ints(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"-?\d+", text)]


def batched(samples: list[dict], batch_size: int):
    for start in range(0, len(samples), batch_size):
        yield samples[start:start + batch_size]


def audio_path_for(sample: dict, data_root: Path) -> Path:
    audio_path = Path(sample["audio_path"])
    return audio_path if audio_path.is_absolute() else data_root / audio_path


def load_audio(path: Path, target_sample_rate: int) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    audio = data.astype(np.float32)
    audio /= 32768.0 # map back to [-1, 1] range

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        audio = resample_poly(
            audio,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

    return audio


def make_messages(prompt: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def render_prompt(processor, prompt: str) -> str:
    return processor.apply_chat_template(
        make_messages(prompt),
        tokenize=False,
        add_generation_prompt=True
    )


def make_inputs(
    processor,
    prompts: list[str],
    audios: list[np.ndarray],
    sample_rate: int,
):
    return processor(
        text=prompts,
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )


def processor_sample_rate(processor) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sample_rate = getattr(feature_extractor, "sampling_rate", None)
    if sample_rate is None:
        raise ValueError("Processor does not expose feature_extractor.sampling_rate.")
    return int(sample_rate)


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


def requires_carry_or_borrow(sample: dict) -> bool:
    return bool(sample.get("requires_carry", False)) or bool(
        sample.get("requires_borrow", False)
    )


@torch.no_grad()
def evaluate_audio_accuracy(
    model,
    processor,
    samples: list[dict],
    data_root: Path,
    prompt: str,
    batch_size: int,
    max_new_tokens: int,
    print_examples: int,
    sample_rate: int,
) -> tuple[dict, list[dict]]:
    correct = 0
    total = 0
    examples_printed = 0
    correct_samples = []

    for batch in tqdm(list(batched(samples, batch_size))):
        audio_paths = [audio_path_for(sample, data_root) for sample in batch]
        audios = [load_audio(path, sample_rate) for path in audio_paths]
        prompts = [render_prompt(processor, prompt) for _ in batch]

        inputs = make_inputs(processor, prompts, audios, sample_rate).to(model.device)
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
                row["audio_path"] = str(audio_paths[idx].resolve())
                correct_samples.append(row)

            if examples_printed < print_examples:
                print(
                    {
                        "audio_text": sample.get("audio_text"),
                        "audio_path": str(audio_paths[idx]),
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
        default="Output ONLY a number.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--print_examples", type=int, default=5)
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="Audio sample rate sent to the processor. Defaults to the processor rate.",
    )
    parser.add_argument("--test_carry_borrow", action="store_true")
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
    sample_rate = args.sample_rate or processor_sample_rate(processor)

    metrics, correct_samples = evaluate_audio_accuracy(
        model=model,
        processor=processor,
        samples=samples,
        data_root=args.data_path.parent,
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        print_examples=args.print_examples,
        sample_rate=sample_rate,
    )

    print("Correct:", metrics["correct"])
    print("Total:", metrics["total"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    save_jsonl(correct_samples, args.output_path)
    print("Saved correct-only audio dataset:", args.output_path)


if __name__ == "__main__":
    main()
