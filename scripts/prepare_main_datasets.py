"""Generate and correctness-filter the four matched main-paper datasets.

Run from any directory. The final paired text/image JSONL files have identical
sample_id order, so split_seed=0 selects the same examples and intervention pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("addition", "subtraction")
PROMPT = "Output ONLY a number."


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def rows_by_id(rows: list[dict], path: Path) -> dict:
    indexed = {}
    for row in rows:
        if "sample_id" not in row:
            raise ValueError(f"Missing sample_id in {path}.")
        sample_id = row["sample_id"]
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}.")
        indexed[sample_id] = row
    return indexed


def intersect_correct_rows(text_path: Path, image_path: Path, text_output: Path, image_output: Path,
                           image_root: Path) -> dict:
    text_rows = load_jsonl(text_path)
    image_rows = load_jsonl(image_path)
    text_by_id = rows_by_id(text_rows, text_path)
    image_by_id = rows_by_id(image_rows, image_path)
    unexpected = image_by_id.keys() - text_by_id.keys()
    if unexpected:
        raise ValueError(f"Image-correct rows absent from text-correct input: {len(unexpected)}.")
    paired_text, paired_image = [], []
    for text_row in text_rows:
        image_row = image_by_id.get(text_row["sample_id"])
        if image_row is None:
            continue
        for field in ("a", "b", "result"):
            if int(text_row[field]) != int(image_row[field]):
                raise ValueError(f"Sample {text_row['sample_id']} differs on {field}.")
        paired_text.append(text_row)
        image_file = (image_root / image_row["image_path"]).resolve()
        if not image_file.is_file():
            raise FileNotFoundError(image_file)
        paired_image.append({**image_row, "image_path": Path(os.path.relpath(
            image_file, image_output.parent.resolve()
        )).as_posix()})
    if not paired_text:
        raise ValueError("The text/image correctness intersection is empty.")
    ids = [row["sample_id"] for row in paired_text]
    if ids != [row["sample_id"] for row in paired_image]:
        raise AssertionError("Paired text and image order diverged.")
    write_jsonl(text_output, paired_text)
    write_jsonl(image_output, paired_image)
    return {
        "text_correct_count": len(text_rows),
        "image_correct_count": len(image_rows),
        "intersection_count": len(ids),
        "removed_after_image_filter": len(text_rows) - len(ids),
        "ordered_sample_ids_sha256": hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "text_input": str(text_path),
        "image_input": str(image_path),
        "text_output": str(text_output),
        "image_output": str(image_output),
        "text_output_sha256": sha256(text_output),
        "image_output_sha256": sha256(image_output),
    }


def run_module(module: str, *options: str) -> None:
    command = [sys.executable, "-m", module, *map(str, options)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/corrected_128"))
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--only-intersect", action="store_true",
                        help="Reuse existing correctness-filtered files under --output-root.")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not args.only_intersect:
        if args.font_path is None or not args.font_path.is_file():
            parser.error("--font-path must name an existing Arial font file.")
        run_module("src.data.baseline.generate_datasets", "--seed", "0", "--output_dir", root / "raw")
        for operation in OPERATIONS:
            text_correct = root / "text_correct" / f"{operation}.jsonl"
            image_dir = root / "rendered_images" / operation
            image_correct = root / "image_correct" / f"{operation}.jsonl"
            run_module("src.models.diagnostics.test_model_accuracy",
                       "--model", args.model,
                       "--data_path", root / "raw" / f"{operation}_baseline.jsonl",
                       "--output_path", text_correct)
            run_module("src.data.images.generate_arithmetic_images",
                       "--input_jsonl", text_correct, "--output_dir", image_dir,
                       "--font_path", args.font_path.resolve())
            run_module("src.models.diagnostics.test_image_model_accuracy",
                       "--model", args.model,
                       "--data_path", image_dir / f"{operation}_images.jsonl",
                       "--output_path", image_correct, "--prompt", PROMPT)
    manifest = {"model": args.model, "prompt": PROMPT,
                "font_path": str(args.font_path.resolve()) if args.font_path else None,
                "operations": {}}
    for operation in OPERATIONS:
        manifest["operations"][operation] = intersect_correct_rows(
            root / "text_correct" / f"{operation}.jsonl",
            root / "image_correct" / f"{operation}.jsonl",
            root / "paired" / f"{operation}_text.jsonl",
            root / "paired" / f"{operation}_image.jsonl",
            root / "rendered_images" / operation,
        )
        print(operation, manifest["operations"][operation])
    (root / "paired" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
