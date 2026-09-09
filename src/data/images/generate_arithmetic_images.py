"""Render arithmetic expressions as centered black text on white images.

The generated metadata keeps the original arithmetic labels and adds image
fields so text and image versions can be matched by ``sample_id``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from src.common.io import load_jsonl, save_jsonl
from src.common.seed import set_seed
from src.data.baseline.generate_datasets import (
    addition_pairs_to_samples,
    multiplication_pairs_to_samples,
    subtraction_pairs_to_samples,
)


OPERATIONS = ("addition", "subtraction", "multiplication")
OP_SYMBOLS = {
    "baseline_add": "+",
    "baseline_sub": "-",
    "baseline_mul": "*",
}


def expression_for_image(sample: dict) -> str:
    """Return only the arithmetic expression, without instruction prompts."""
    tokens = sample.get("tokens")
    if isinstance(tokens, list) and tokens:
        return " ".join(str(token) for token in tokens)

    a = sample.get("a")
    b = sample.get("b")
    operation = sample.get("operation") or OP_SYMBOLS.get(sample.get("task"))
    if a is not None and b is not None and operation:
        return f"{a} {operation} {b} ="

    expr = str(sample["expr"])
    return expr.split(". ", 1)[-1] if ". " in expr else expr


def load_font(font_path: str | None, font_size: int) -> ImageFont.ImageFont:
    if font_path:
        return ImageFont.truetype(font_path, font_size)

    for candidate in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_expression_image(
    expression: str,
    output_path: Path,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    image = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), expression, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size[0] - text_width) / 2 - bbox[0]
    y = (image_size[1] - text_height) / 2 - bbox[1]
    draw.text((x, y), expression, fill="black", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_operation_samples(
    operation: str,
    prompt: str,
    limit: int | None,
) -> list[dict]:
    if operation == "addition":
        pairs = [(a, b) for a in range(100) for b in range(100) if a + b <= 99]
        sample_fn = addition_pairs_to_samples
    elif operation == "subtraction":
        pairs = [(a, b) for a in range(100) for b in range(100) if a >= b]
        sample_fn = subtraction_pairs_to_samples
    elif operation == "multiplication":
        pairs = [(a, b) for a in range(100) for b in range(10)]
        sample_fn = multiplication_pairs_to_samples
    else:
        raise ValueError(f"Unknown operation: {operation}")

    random.shuffle(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    return sample_fn(pairs, use_digits=True, prompt=prompt)


def iter_generated_samples(
    operations: Iterable[str],
    prompt: str,
    limit_per_operation: int | None,
) -> Iterable[tuple[str, list[dict]]]:
    for operation in operations:
        yield operation, make_operation_samples(
            operation=operation,
            prompt=prompt,
            limit=limit_per_operation,
        )


def convert_samples_to_images(
    samples: list[dict],
    output_dir: Path,
    metadata_name: str,
    image_size: tuple[int, int],
    font_size: int,
    font_path: str | None,
) -> Path:
    font = load_font(font_path, font_size)
    image_dir = output_dir / "png"
    metadata_rows = []

    for index, sample in enumerate(samples):
        expression = expression_for_image(sample)
        sample_id = sample.get("sample_id", index)
        image_name = (
            f"{sample_id:06d}.png"
            if isinstance(sample_id, int)
            else f"{index:06d}.png"
        )
        image_path = image_dir / image_name
        render_expression_image(
            expression=expression,
            output_path=image_path,
            image_size=image_size,
            font=font,
        )

        row = {
            **sample,
            "format": "image_arithmetic",
            "source_format": sample.get("format", "text_arithmetic"),
            "source_expr": sample.get("expr"),
            "image_text": expression,
            "image_path": image_path.relative_to(output_dir).as_posix(),
            "image_width": image_size[0],
            "image_height": image_size[1],
            "font_size": font_size,
            "background_color": "white",
            "text_color": "black",
        }
        metadata_rows.append(row)

    metadata_path = output_dir / metadata_name
    save_jsonl(metadata_rows, metadata_path)
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        help="Existing arithmetic JSONL to render. If omitted, baseline samples are generated.",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/images"))
    parser.add_argument(
        "--operation",
        choices=(*OPERATIONS, "all"),
        default="all",
        help="Baseline operation to generate when --input_jsonl is omitted.",
    )
    parser.add_argument("--limit_per_operation", type=int)
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_width", type=int, default=384)
    parser.add_argument("--image_height", type=int, default=384)
    parser.add_argument("--font_size", type=int, default=64)
    parser.add_argument("--font_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    image_size = (args.image_width, args.image_height)
    if args.input_jsonl:
        samples = load_jsonl(args.input_jsonl)
        input_stem = args.input_jsonl.stem
        metadata_stem = (
            input_stem.removesuffix("_baseline")
            if input_stem.endswith("_baseline")
            else input_stem
        )
        metadata_name = f"{metadata_stem}_images.jsonl"
        metadata_path = convert_samples_to_images(
            samples=samples,
            output_dir=args.output_dir,
            metadata_name=metadata_name,
            image_size=image_size,
            font_size=args.font_size,
            font_path=args.font_path,
        )
        print(f"Rendered {len(samples)} images")
        print(f"Metadata: {metadata_path}")
        return

    operations = OPERATIONS if args.operation == "all" else (args.operation,)
    total = 0
    for operation, samples in iter_generated_samples(
        operations=operations,
        prompt=args.prompt,
        limit_per_operation=args.limit_per_operation,
    ):
        metadata_path = convert_samples_to_images(
            samples=samples,
            output_dir=args.output_dir / operation,
            metadata_name=f"{operation}_images.jsonl",
            image_size=image_size,
            font_size=args.font_size,
            font_path=args.font_path,
        )
        total += len(samples)
        print(f"{operation}: rendered {len(samples)} images")
        print(f"{operation}: metadata {metadata_path}")

    print(f"Done. Rendered {total} images.")


if __name__ == "__main__":
    main()
