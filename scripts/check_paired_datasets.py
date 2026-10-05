"""Small CPU check for correctness intersection and deterministic matched pairs."""

import ast
import random
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_main_datasets import intersect_correct_rows, load_jsonl, write_jsonl


SOURCE = Path(__file__).resolve().parents[1] / "src/interventions/das.py"
NAMES = {"sample_id", "valid_pair", "_pair_groups", "validate_samples",
         "build_unique_pairs", "split_samples"}
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in NAMES]
scope = {"random": random, "defaultdict": defaultdict}
exec(compile(tree, str(SOURCE), "exec"), scope)


with TemporaryDirectory() as directory:
    root = Path(directory)
    text_rows = [{"sample_id": index, "a": index, "b": 10,
                  "result": index + 10, "expr": f"{index} + 10 ="}
                 for index in range(20)]
    # One text-correct expression failed the image filter; image order is reversed.
    image_rows = [{**row, "image_path": f"{row['sample_id']}.png"}
                  for row in reversed(text_rows) if row["sample_id"] != 3]
    image_root = root / "rendered"
    image_root.mkdir()
    for row in image_rows:
        (image_root / row["image_path"]).write_bytes(b"test image")
    text_input, image_input = root / "text.jsonl", root / "image.jsonl"
    text_output, image_output = root / "paired_text.jsonl", root / "paired_image.jsonl"
    write_jsonl(text_input, text_rows)
    write_jsonl(image_input, image_rows)
    summary = intersect_correct_rows(text_input, image_input, text_output, image_output, image_root)
    text_paired, image_paired = load_jsonl(text_output), load_jsonl(image_output)
    assert summary["intersection_count"] == 19
    assert summary["removed_after_image_filter"] == 1
    assert [row["sample_id"] for row in text_paired] == [row["sample_id"] for row in image_paired]
    assert all((image_output.parent / row["image_path"]).is_file() for row in image_paired)
    for left, right in zip(scope["split_samples"](text_paired, 0.7, 0.15, 0),
                           scope["split_samples"](image_paired, 0.7, 0.15, 0)):
        assert [row["sample_id"] for row in left] == [row["sample_id"] for row in right]
        text_pairs, _ = scope["build_unique_pairs"](left, "result", 2, 10)
        image_pairs, _ = scope["build_unique_pairs"](right, "result", 2, 10)
        ids = lambda pairs: [(pair["base"]["sample_id"], pair["source"]["sample_id"])
                             for pair in pairs]
        assert ids(text_pairs) == ids(image_pairs)

print("Intersection preserves matched sample order, splits, and sampled pair IDs.")
