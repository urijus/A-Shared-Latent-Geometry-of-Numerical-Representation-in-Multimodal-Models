from pathlib import Path
import os
import re
import torch
from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    DASSubspace,
    format_prompt,
    patched_forward as patched_forward_text,
    resolve_position,
    target_answers,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    patched_forward as patched_forward_image,
    resolve_batch_positions,
    sample_prompt,
)
from src.geometry.subspaces import (
    orthonormal_columns,
    torch_load_portable,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import uses_chat_template
from src.models import get_blocks, load_hf_model, load_hf_model_and_processor, resolve_model_for_loading


DEFAULT_TASKS = [
    ("text", "addition", "result"),
    ("text", "subtraction", "result"),
    ("image", "addition", "result"),
    ("image", "subtraction", "result"),
    ("text", "multiplication", "result"),
    ("text", "multiplication", "c0_hat"),
    ("text", "multiplication", "c1_hat_full"),
]


def configured_tasks():
    value = os.environ.get("TASKS")
    if not value:
        return DEFAULT_TASKS
    tasks = []
    for item in value.split(","):
        parts = item.split(":")
        if len(parts) == 2:
            modality, operation = parts
            target = "result"
        elif len(parts) == 3:
            modality, operation, target = parts
        else:
            raise ValueError("TASKS items must be modality:operation or modality:operation:target")
        tasks.append((modality, operation, target))
    return tasks


ROOT = Path(os.environ.get("DAS_AUDIT_ROOT", "./results/final_exps/DAS_audit_k_22"))
CONDITION = os.environ.get("CONDITION", "das_pca_initialized")
OUTPUT = ROOT / os.environ.get("RELIABILITY_OUTPUT", "subspace_reliability.jsonl")
MATRIX_OUTPUT = ROOT / os.environ.get("RELIABILITY_MATRIX_OUTPUT", "subspace_reliability_matrices.jsonl")
FUNCTIONAL_OUTPUT = ROOT / os.environ.get("RELIABILITY_FUNCTIONAL_OUTPUT", "functional_reliability.jsonl")


def task_dir(modality, operation, target):
    folder = ROOT / modality / operation
    return folder if target == "result" else folder / target


def load_basis(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    subspace_payload = torch_load_portable(path)
    if "basis" not in subspace_payload:
        raise ValueError(f"{path} does not contain a learned/stored basis.")

    return torch.as_tensor(subspace_payload["basis"]).detach().float()


def subspace_metrics(first_basis, second_basis):
    first_basis = torch.as_tensor(first_basis).detach().float().squeeze()
    second_basis = torch.as_tensor(second_basis).detach().float().squeeze()
    d_model = max(first_basis.shape)
    if max(second_basis.shape) != d_model:
        raise ValueError(
            f"Subspaces have different d_model axes: "
            f"{tuple(first_basis.shape)} vs {tuple(second_basis.shape)}."
        )
    first = orthonormal_columns(first_basis, d_model, "first DAS basis")
    second = orthonormal_columns(second_basis, d_model, "second DAS basis")
    singular_values = torch.linalg.svdvals(first.T @ second).clamp(0, 1)
    squared_sum = singular_values.square().sum()
    first_in_second = squared_sum / first.shape[1]
    second_in_first = squared_sum / second.shape[1]
    min_rank_containment = squared_sum / min(first.shape[1], second.shape[1])
    return {
        "first_in_second": float(first_in_second),
        "second_in_first": float(second_in_first),
        "symmetric_overlap": float((first_in_second + second_in_first) / 2),
        "min_rank_containment": float(min_rank_containment),
        "principal_cosines": [float(value) for value in singular_values],
        "first_rank": first.shape[1],
        "second_rank": second.shape[1],
        "d_model": d_model,
    }


def load_task_bases(task_path, n_seeds=3):
    subspaces = {}
    for seed in range(n_seeds):
        basis_path = find_subspace_path(task_path, seed)
        if basis_path is not None:
            subspaces[seed] = {
                "basis": load_basis(basis_path),
                "path": basis_path,
            }

    return subspaces


def find_subspace_path(task_path, seed, split_seed=0):
    candidates = [
        task_path / f"split_{split_seed}" / f"seed_{seed}" / "subspace.pt",
        task_path / f"seed_{seed}" / "subspace.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_results_path(task_path, seed, split_seed=0):
    candidates = [
        task_path / f"split_{split_seed}" / f"seed_{seed}" / "results.jsonl",
        task_path / f"seed_{seed}" / "results.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_predictions_path(task_path, seed, split_seed=0):
    candidates = [
        task_path / f"split_{split_seed}" / f"seed_{seed}" / "autoregressive_outputs.jsonl",
        task_path / f"split_{split_seed}" / f"seed_{seed}" / "predictions.jsonl",
        task_path / f"seed_{seed}" / "autoregressive_outputs.jsonl",
        task_path / f"seed_{seed}" / "predictions.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def pairwise_rows(modality, operation, target, subspaces):
    rows = []
    seeds = sorted(subspaces)
    for i, first_seed in enumerate(seeds):
        for second_seed in seeds[i + 1:]:
            first = subspaces[first_seed]
            second = subspaces[second_seed]
            row = {
                "modality": modality,
                "operation": operation,
                "target": target,
                "condition": CONDITION,
                "first_seed": first_seed,
                "second_seed": second_seed,
                "first_path": str(first["path"]),
                "second_path": str(second["path"]),
            }
            row.update(subspace_metrics(first["basis"], second["basis"]))
            rows.append(row)
    return rows


def matrix_from_pairwise(seeds, rows, metric):
    lookup = {}
    for row in rows:
        lookup[(row["first_seed"], row["second_seed"])] = row[metric]
        lookup[(row["second_seed"], row["first_seed"])] = row[metric]
    matrix = []
    for first in seeds:
        matrix_row = []
        for second in seeds:
            matrix_row.append(1.0 if first == second else lookup[(first, second)])
        matrix.append(matrix_row)
    return matrix


def off_diagonal_mean(matrix):
    values = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i < j:
                values.append(value)
    return sum(values) / len(values) if values else None


def load_seed_results(task_path, n_seeds=3):
    results = {}
    for seed in range(n_seeds):
        path = find_results_path(task_path, seed)
        if path is None:
            continue
        rows = load_jsonl(path)
        if rows:
            results[seed] = {"row": rows[0], "path": path}
    return results


def load_predictions(task_path, n_seeds=3):
    predictions = {}
    for seed in range(n_seeds):
        path = find_predictions_path(task_path, seed)
        if path is None:
            continue
        rows = load_jsonl(path)
        seed_predictions = {}
        for index, row in enumerate(rows):
            pair_id = row.get("pair_id", index)
            prediction = (
                row.get("prediction")
                or row.get("generated")
                or row.get("generated_text")
                or row.get("output")
            )
            if prediction is not None:
                seed_predictions[pair_id] = str(prediction)
        if seed_predictions:
            predictions[seed] = {"predictions": seed_predictions, "path": path}
    return predictions


def seed_folder(task_path, seed, split_seed=0):
    for folder in [task_path / f"split_{split_seed}" / f"seed_{seed}", task_path / f"seed_{seed}"]:
        if folder.exists():
            return folder
    return None


def load_pairs_for_seed(task_path, seed, result_row):
    folder = seed_folder(task_path, seed)
    heldout = folder / "heldout_pairs.jsonl"
    samples = {row["sample_id"]: row for row in load_jsonl(Path(result_row["data_path"]))}
    pairs = []
    for row in load_jsonl(heldout):
        pairs.append(
            {
                "pair_id": row["pair_id"],
                "base": samples[row["base_sample_id"]],
                "source": samples[row["source_sample_id"]],
            }
        )
    return pairs


def subspace_from_file(path, device):
    payload = torch_load_portable(path)
    basis = torch.as_tensor(payload["basis"]).detach().float()
    return {
        str(payload["config"]["layer"]): DASSubspace(
            basis.shape[0], basis.shape[1], initial_basis=basis
        ).to(device)
    }


@torch.no_grad()
def generate_text_outputs(model, tokenizer, blocks, subspaces, pairs, config):
    rows = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for pair in pairs:
            base, source = pair["base"], pair["source"]
            donor = pair.get("donor", source)
            use_chat = config.get("use_chat_template") or uses_chat_template(config["model"])
            base_prompt = format_prompt(tokenizer, base, use_chat)
            donor_prompt = format_prompt(tokenizer, donor, use_chat)
            expected = target_answers(base, source, config["target"])[1]
            base_position = resolve_position(tokenizer, base_prompt, config["position"])
            donor_position = resolve_position(tokenizer, donor_prompt, config["position"])
            base_ids = tokenizer(base_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            donor_ids = tokenizer(donor_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(model.device)
            generated = []
            for _ in range(config["max_new_tokens"]):
                base_length, donor_length = len(base_ids), len(donor_ids)
                length = max(base_length, donor_length)
                input_ids = torch.full((2, length), tokenizer.pad_token_id, dtype=torch.long, device=model.device)
                attention_mask = torch.zeros_like(input_ids)
                input_ids[0, :base_length] = base_ids
                input_ids[1, :donor_length] = donor_ids
                attention_mask[0, :base_length] = 1
                attention_mask[1, :donor_length] = 1
                outputs = patched_forward_text(
                    model, {"input_ids": input_ids, "attention_mask": attention_mask},
                    blocks, subspaces, [config["layer"]], config["hook"],
                    [base_position], [donor_position], n_base_groups=1,
                )
                next_id = outputs.logits[0, base_length - 1].argmax().reshape(1)
                generated.append(int(next_id.item()))
                base_ids = torch.cat([base_ids, next_id])
                if next_id.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            match = re.match(r"-?\d+", text)
            prediction = match.group() if match else text
            rows.append({"pair_id": pair["pair_id"], "expected": expected, "prediction": prediction, "correct": prediction == expected})
    finally:
        tokenizer.padding_side = old_padding_side
    return rows


@torch.no_grad()
def generate_image_outputs(
    model, processor, tokenizer, blocks, subspaces, pairs, config, result_row
):
    rows = []
    data_path = config.get("data_path") or result_row.get("data_path")
    data_root = config.get("data_root") or result_row.get("data_root")
    data_root = Path(data_root) if data_root else Path(data_path).parent
    for pair in pairs:
        base, source = pair["base"], pair["source"]
        donor = pair.get("donor", source)
        expected = target_answers(base, source, config["target"])[1]
        base_prompt = sample_prompt(processor, base, config["prompt"], config.get("enable_thinking", False))
        donor_prompt = sample_prompt(processor, donor, config["prompt"], config.get("enable_thinking", False))
        base_image = load_rgb_image(image_path_for(base, data_root))
        donor_image = load_rgb_image(image_path_for(donor, data_root))
        base_position = resolve_batch_positions(processor, tokenizer, model, [base_prompt], [base_image], config["position"])[0]
        donor_position = resolve_batch_positions(processor, tokenizer, model, [donor_prompt], [donor_image], config["position"])[0]
        inputs = inputs_to_device(make_inputs(processor, [base_prompt, donor_prompt], [base_image, donor_image]), model.device)
        generated = []
        for _ in range(config["max_new_tokens"]):
            base_length = int(inputs["attention_mask"][0].sum())
            outputs = patched_forward_image(
                model, inputs, blocks, subspaces, [config["layer"]], config["hook"],
                [base_position], [donor_position], n_base_groups=1,
            )
            next_id = outputs.logits[0, base_length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            pad = torch.full((inputs["input_ids"].shape[0], 1), tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype, device=model.device)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], pad], dim=1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros_like(pad)], dim=1)
            inputs["input_ids"][0, base_length] = next_id.item()
            inputs["attention_mask"][0, base_length] = 1
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        prediction = match.group() if match else text
        rows.append({"pair_id": pair["pair_id"], "expected": expected, "prediction": prediction, "correct": prediction == expected})
    return rows


def ensure_predictions(modality, task_path, seed_results):
    if len(load_predictions(task_path, len(seed_results))) >= len(seed_results):
        return
    first_config = torch_load_portable(find_subspace_path(task_path, next(iter(seed_results))))["config"]
    model_path, _ = resolve_model_for_loading(first_config["model"])
    if modality == "text":
        model, tokenizer = load_hf_model(model_path)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        blocks = get_blocks(model)
        for seed, result in seed_results.items():
            folder = seed_folder(task_path, seed)
            output = folder / "autoregressive_outputs.jsonl"
            if output.exists():
                continue
            payload = torch_load_portable(find_subspace_path(task_path, seed))
            pairs = load_pairs_for_seed(task_path, seed, result["row"])
            subspaces = subspace_from_file(find_subspace_path(task_path, seed), model.device)
            save_jsonl(generate_text_outputs(model, tokenizer, blocks, subspaces, pairs, payload["config"]), output)
    else:
        model, processor, tokenizer = load_hf_model_and_processor(model_path)
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        blocks = get_blocks(model)
        for seed, result in seed_results.items():
            folder = seed_folder(task_path, seed)
            output = folder / "autoregressive_outputs.jsonl"
            if output.exists():
                continue
            payload = torch_load_portable(find_subspace_path(task_path, seed))
            pairs = load_pairs_for_seed(task_path, seed, result["row"])
            subspaces = subspace_from_file(find_subspace_path(task_path, seed), model.device)
            save_jsonl(
                generate_image_outputs(
                    model,
                    processor,
                    tokenizer,
                    blocks,
                    subspaces,
                    pairs,
                    payload["config"],
                    result["row"],
                ),
                output,
            )


def agreement_matrix(predictions):
    seeds = sorted(predictions)
    if len(seeds) < 2:
        return seeds, None
    matrix = []
    for first_seed in seeds:
        row = []
        first = predictions[first_seed]["predictions"]
        for second_seed in seeds:
            second = predictions[second_seed]["predictions"]
            shared = sorted(set(first) & set(second))
            if first_seed == second_seed:
                row.append(1.0)
            elif not shared:
                row.append(None)
            else:
                agree = sum(first[pair_id] == second[pair_id] for pair_id in shared)
                row.append(agree / len(shared))
        matrix.append(row)
    return seeds, matrix


def functional_row(modality, operation, target, task_path):
    seed_results = load_seed_results(task_path)
    agreement_note = None
    predictions = load_predictions(task_path)
    if seed_results and len(predictions) < len(seed_results):
        missing = []
        for seed, result in sorted(seed_results.items()):
            data_path = Path(result["row"].get("data_path", ""))
            folder = seed_folder(task_path, seed)
            heldout = None if folder is None else folder / "heldout_pairs.jsonl"
            if not data_path.exists():
                missing.append(f"seed_{seed} data_path={data_path}")
            if heldout is None or not heldout.exists():
                missing.append(f"seed_{seed} heldout_pairs={heldout}")
        if missing:
            agreement_note = (
                "Could not generate per-pair autoregressive predictions because "
                "required saved artifacts are missing: "
                + "; ".join(missing)
            )
        else:
            try:
                ensure_predictions(modality, task_path, seed_results)
            except (FileNotFoundError, KeyError, ValueError, OSError) as error:
                agreement_note = (
                    "Could not generate per-pair autoregressive predictions from "
                    f"saved artifacts: {type(error).__name__}: {error}"
                )
            predictions = load_predictions(task_path)
    agreement_seeds, agreement = agreement_matrix(predictions)
    if agreement_note is None and agreement is None:
        agreement_note = (
            "Per-pair autoregressive predictions were not found, so A_ij "
            "cannot be computed from current saved outputs."
        )
    return {
        "modality": modality,
        "operation": operation,
        "target": target,
        "condition": CONDITION,
        "seed_autoregressive_iia": {
            str(seed): result["row"].get("autoregressive_iia")
            for seed, result in sorted(seed_results.items())
        },
        "result_paths": {
            str(seed): str(result["path"])
            for seed, result in sorted(seed_results.items())
        },
        "agreement_seeds": agreement_seeds,
        "output_agreement_matrix": agreement,
        "mean_output_agreement": (
            off_diagonal_mean(agreement) if agreement is not None else None
        ),
        "agreement_note": agreement_note,
    }



if __name__=="__main__":
    all_rows = []
    matrix_rows = []
    functional_rows = []
    for modality, operation, target in configured_tasks():
        task_path = task_dir(modality, operation, target) / CONDITION
        subspaces = load_task_bases(task_path)
        rows = pairwise_rows(modality, operation, target, subspaces)
        all_rows.extend(rows)
        seeds = sorted(subspaces)
        if len(seeds) >= 2:
            matrix = matrix_from_pairwise(seeds, rows, "symmetric_overlap")
            matrix_rows.append(
                {
                    "modality": modality,
                    "operation": operation,
                    "target": target,
                    "condition": CONDITION,
                    "seeds": seeds,
                    "metric": "symmetric_overlap",
                    "matrix": matrix,
                    "mean_off_diagonal": off_diagonal_mean(matrix),
                }
            )
        functional_rows.append(functional_row(modality, operation, target, task_path))
        print(f"{modality}/{operation}/{target}: {len(rows)} overlaps")

    save_jsonl(all_rows, OUTPUT)
    save_jsonl(matrix_rows, MATRIX_OUTPUT)
    save_jsonl(functional_rows, FUNCTIONAL_OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Saved {MATRIX_OUTPUT}")
    print(f"Saved {FUNCTIONAL_OUTPUT}")
