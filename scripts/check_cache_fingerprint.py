"""Check causal-cache resume and invalidation without loading the model."""

import ast
import argparse
import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

helpers = run_path(str(Path(__file__).resolve().parents[1] / "src/common/fingerprint.py"))
code_identity, dataset_hash, digest, file_hash, model_identity, ordered_pairs_hash = (
    helpers[name] for name in (
        "code_identity", "dataset_hash", "digest", "file_hash", "model_identity", "ordered_pairs_hash"
    )
)


SOURCE = Path(__file__).resolve().parents[1] / "src/experiments/cross_condition_transfer/causal_transfer/causal_transfer.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
names = {"evaluation_fingerprint", "load_cached", "cached_path", "_matches_fingerprint"}
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
scope = {"__file__": str(SOURCE), "Path": Path, "argparse": argparse,
         "code_identity": code_identity, "dataset_hash": dataset_hash,
         "digest": digest, "file_hash": file_hash, "model_identity": model_identity,
         "ordered_pairs_hash": ordered_pairs_hash}
exec(compile(tree, str(SOURCE), "exec"), scope)


with TemporaryDirectory() as directory:
    root = Path(directory)
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"fake"}', encoding="utf-8")
    source_data = root / "source.jsonl"
    destination_data = root / "destination.jsonl"
    image = root / "image.png"
    image.write_bytes(b"image A")
    source_data.write_text('{"sample_id":0}\n', encoding="utf-8")
    destination_data.write_text(json.dumps({"sample_id": 1, "image_path": str(image)}) + "\n", encoding="utf-8")
    pairs = root / "pairs.jsonl"
    pairs.write_text('{"base_sample_id":1,"source_sample_id":2}\n', encoding="utf-8")
    source_basis = root / "source.pt"
    destination_basis = root / "destination.pt"
    source_basis.write_bytes(b"basis A")
    destination_basis.write_bytes(b"basis B")
    self_result = root / "self.jsonl"
    control_result = root / "control.jsonl"
    self_result.write_text('{"autoregressive_iia":0.8}\n', encoding="utf-8")
    control_result.write_text('{"autoregressive_iia":0.0}\n', encoding="utf-8")
    args = SimpleNamespace(model=str(model), output_dir=root, force=False,
                           condition="das_pca_initialized",
                           control_condition="random_subspace_in_pca_span", max_autoregressive_pairs=128,
                           target="result", layer=43, k=22, hook="resid_post", split_seed=0,
                           text_position="17", image_position="-1", max_new_tokens=8,
                           prompt="Output ONLY a number.", enable_thinking=False,
                           use_chat_template=False)
    scope.update({
        "parse_task": lambda task: tuple(task.split(":")),
        "resolve_model_for_loading": lambda name: (model, "fake"),
        "first_result_row": lambda args, modality, operation, condition, seed:
            {"data_path": str(source_data if operation == "addition" else destination_data)},
        "subspace_path": lambda args, modality, operation, seed:
            source_basis if operation == "addition" else destination_basis,
        "results_path": lambda args, modality, operation, condition, seed:
            control_result if condition == args.control_condition else self_result,
        "heldout_pairs_path": lambda args, modality, operation, seed: pairs,
        "load_jsonl": lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()],
    })
    row = {"source_task": "text:addition", "source_seed": 0,
           "destination_task": "text:subtraction", "destination_seed": 1}
    row["fingerprint"] = scope["evaluation_fingerprint"](args, row)
    cache_path = root / "transfer_results.jsonl"
    cache_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(scope["load_cached"](args)) == 1
    source_basis.write_bytes(b"changed basis")
    assert not scope["load_cached"](args)
    source_basis.write_bytes(b"basis A")
    pairs.write_text('{"base_sample_id":2,"source_sample_id":1}\n', encoding="utf-8")
    assert not scope["load_cached"](args)
    pairs.write_text('{"base_sample_id":1,"source_sample_id":2}\n', encoding="utf-8")
    assert len(scope["load_cached"](args)) == 1
    image.write_bytes(b"image B")
    assert not scope["load_cached"](args)
    image.write_bytes(b"image A")
    assert len(scope["load_cached"](args)) == 1
    cache_path.write_text(json.dumps({**row, "fingerprint": None}) + "\n", encoding="utf-8")
    assert not scope["load_cached"](args)
    transport = root / "transport.pt"
    transport.write_bytes(b"fitted map")
    causal = root / "causal.jsonl"
    causal.write_text(json.dumps(row) + "\n", encoding="utf-8")
    args.causal_transfer_rows = causal
    transported = {**row, "variant": "scaled_displacement", "fit_kind": "base_to_donor_displacements",
                   "fit_source_task": row["source_task"], "fit_source_seed": row["source_seed"],
                   "fit_destination_task": row["destination_task"],
                   "fit_destination_seed": row["destination_seed"],
                   "alignment_path": str(transport)}
    code_file = SOURCE.parents[1] / "procrustes/procrustes.py"
    transported["fingerprint"] = scope["evaluation_fingerprint"](args, transported, code_file=code_file)
    assert scope["_matches_fingerprint"](args, transported, code_file=code_file)
    transport.write_bytes(b"different fitted map")
    assert not scope["_matches_fingerprint"](args, transported, code_file=code_file)

print("Cache resumes unchanged inputs and rejects changed basis, pairs, image, fitted map, and missing fingerprint.")
