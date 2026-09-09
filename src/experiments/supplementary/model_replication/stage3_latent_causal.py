"""Stage 3 nulls and latent numerical identity causal tests for Ministral.

The null-distribution section uses only Stage 2 cached activations/centroids.
The causal section loads the model, patches the final pre-generation token
after layer 37, and reuses the intervention-affected KV cache for the direct
second-digit test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import random
import re
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from src.interventions.das import (
    format_prompt,
    hidden,
    hook_module,
    replace_hidden,
    resolve_position,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    resolve_batch_positions,
    sample_prompt,
)
from src.experiments.cross_condition_transfer.procrustes.procrustes import stable_seed
from src.experiments.supplementary.model_replication import stage1_das
from src.experiments.supplementary.model_replication import stage2_readout_geometry as s2
from src.models import get_blocks, get_hidden_size, load_hf_model_and_processor, resolve_model_for_loading


TASKS = ["T+", "T-", "I+", "I-"]
PILOT_TASKS = ["T+", "I+"]
PRIMARY_CONDITIONS = ["matched", "mismatch", "C-only", "random_latent", "value_specificity"]


@dataclass
class Stage3Space:
    task: str
    seed: int
    c_basis: torch.Tensor
    l_basis: torch.Tensor
    centroids: dict[int, torch.Tensor]
    activation_by_key: dict[str, torch.Tensor]


@dataclass
class PromptForward:
    logits_t0: torch.Tensor
    past_key_values: object | None
    prompt_length: int
    patch_position: int
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=stage1_das.MODEL_ALIAS)
    parser.add_argument("--model_slug", default=stage1_das.MODEL_SLUG)
    parser.add_argument("--das_root", type=Path, default=Path("src/experiments/supplementary/model_replication/results/das"))
    parser.add_argument("--stage2_dir", type=Path, default=Path("src/experiments/supplementary/model_replication/results/stage2_readout_geometry"))
    parser.add_argument("--output_dir", type=Path, default=Path("src/experiments/supplementary/model_replication/results/stage3_latent_causal"))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--pilot_tasks", nargs="+", default=PILOT_TASKS, choices=TASKS)
    parser.add_argument("--pilot_seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--position", default="-1")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--target", default="result")
    parser.add_argument("--format", default="digits")
    parser.add_argument("--prompt", default="Output ONLY a number.")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true", default=True)

    # Must match Stage 1 final DAS hash config.
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--max_train_pairs", type=int, default=4096)
    parser.add_argument("--max_validation_pairs", type=int, default=512)
    parser.add_argument("--max_test_pairs", type=int, default=512)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=128)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pca_max_samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=8)

    parser.add_argument("--max_triplets", type=int, default=128)
    parser.add_argument("--triplet_seed", type=int, default=1729)
    parser.add_argument("--random_draws", type=int, default=200)
    parser.add_argument("--label_permutations", type=int, default=500)
    parser.add_argument("--pilot_second_prob_margin", type=float, default=0.02)
    parser.add_argument("--pilot_first_prob_tolerance", type=float, default=0.10)
    parser.add_argument("--svd_tolerance", type=float, default=1e-6)
    parser.add_argument("--force_full", action="store_true")
    parser.add_argument("--skip_nulls", action="store_true")
    parser.add_argument("--skip_figures", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--artifact_check_only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s2.jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    s2.print_table(headers, rows)


def fmt(value) -> str:
    return s2.fmt(value)


def mean(values) -> float | None:
    return s2.mean(list(values))


def std(values) -> float | None:
    return s2.std(list(values))


def quantile(values: list[float], q: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = q * (len(clean) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - position) + clean[hi] * (position - lo)


def sample_key(sample: dict) -> str:
    if sample.get("image_path") is not None:
        return str(sample["image_path"])
    if sample.get("expr") is not None:
        return str(sample["expr"])
    if sample.get("image_text") is not None:
        return str(sample["image_text"])
    if sample.get("a") is not None and sample.get("b") is not None:
        return f"{sample['a']}:{sample['b']}"
    if sample.get("sample_id") is not None:
        return str(sample["sample_id"])
    if sample.get("id") is not None:
        return str(sample["id"])
    return json.dumps(sample, sort_keys=True)


def target_value(sample: dict, target: str = "result") -> int:
    return int(sample[target] if target in sample else sample["result"])


def digits(value: int) -> str:
    return str(abs(int(value)))


def two_digit(value: int) -> bool:
    return 10 <= int(value) <= 99


def parse_int_prefix(text: str) -> str | None:
    match = re.match(r"\s*(-?\d+)", text or "")
    return match.group(1) if match else None


def load_stage2_tensor(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 artifact: {path}")
    if os.name != "nt":
        return torch.load(path, map_location="cpu", weights_only=False)
    original = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        pathlib.PosixPath = original


def load_stage2_bases(args: argparse.Namespace) -> tuple[dict, dict, torch.Tensor | None]:
    decomp = args.stage2_dir / "decomposition"
    c_payload = load_stage2_tensor(decomp / "C_basis.pt")
    l_payload = load_stage2_tensor(decomp / "L_basis.pt")
    u_path = decomp / "digit_readout_basis.pt"
    if not u_path.exists():
        print(f"WARNING: missing {u_path}; Stage 3 will reconstruct U_digit from output weights without model forwards.")
        return c_payload["basis"], l_payload["basis"], None
    return c_payload["basis"], l_payload["basis"], load_stage2_tensor(u_path).float()


def activation_path(args: argparse.Namespace, task: str) -> Path:
    clean = task.replace("+", "plus").replace("-", "minus")
    return args.stage2_dir / "activations" / f"{clean}_L{args.layer}_pos{args.position}.pt"


def load_activation_payload(args: argparse.Namespace, task: str) -> dict:
    path = activation_path(args, task)
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 activation cache for {task}: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def label_key(label: dict) -> str:
    return sample_key(label)


def load_stage2_spaces(args: argparse.Namespace) -> tuple[dict[tuple[str, int], Stage3Space], dict[str, dict], torch.Tensor | None, list[dict]]:
    c_bases, l_bases, u_digit = load_stage2_bases(args)
    activations = {task: load_activation_payload(args, task) for task in args.tasks}
    splits_path = args.stage2_dir / "value_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 value splits: {splits_path}")
    splits = load_json(splits_path)
    spaces = {}
    for task in args.tasks:
        payload = activations[task]
        hidden = payload["activations"].float()
        labels = payload["labels"]
        by_key = {label_key(label): hidden[index] for index, label in enumerate(labels)}
        for seed in args.seeds:
            c_basis = c_bases[task][str(seed)].float()
            l_basis = l_bases[task][str(seed)].float()
            dummy = s2.Space(task, seed, torch.empty(0), c_basis, l_basis, {}, "")
            centroids = s2.compute_centroids_for_space(dummy, payload)
            spaces[(task, seed)] = Stage3Space(task, seed, c_basis, l_basis, centroids, by_key)
    return spaces, activations, u_digit, splits


def load_model_bundle(args: argparse.Namespace):
    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print(f"Loading model from {model_path}")
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor, tokenizer, get_blocks(model), get_hidden_size(model), saved_model_name


def metric_means(pairwise_rows: list[dict], rsa_rows: list[dict]) -> dict:
    return {
        "heldout_transition_cosine": mean(row["heldout_transition_cosine"] for row in pairwise_rows),
        "top1": mean(row["top1"] for row in pairwise_rows),
        "top5": mean(row["top5"] for row in pairwise_rows),
        "rsa": mean(row["rsa_spearman"] for row in rsa_rows),
    }


def rsa_no_permutation(args, spaces: dict[tuple[str, int], s2.Space], splits: list[dict], space_type: str) -> list[dict]:
    rows = []
    for split in splits:
        test_values = split["test_values"]
        for source_task in args.tasks:
            for destination_task in args.tasks:
                if source_task == destination_task:
                    continue
                for source_seed in args.seeds:
                    for destination_seed in args.seeds:
                        source = spaces[(source_task, source_seed)]
                        destination = spaces[(destination_task, destination_seed)]
                        rows.append(
                            {
                                "space_type": space_type,
                                "split_seed": split["split_seed"],
                                "source_task": source_task,
                                "destination_task": destination_task,
                                "source_seed": source_seed,
                                "destination_seed": destination_seed,
                                "rsa_spearman": s2.rsa_metric(
                                    source.centroids,
                                    destination.centroids,
                                    test_values,
                                    0,
                                    stable_seed("stage3_rsa", space_type, split["split_seed"], source_task, destination_task, source_seed, destination_seed),
                                )["rsa_spearman"],
                            }
                        )
    return rows


def to_s2_spaces(spaces: dict[tuple[str, int], Stage3Space]) -> dict[tuple[str, int], s2.Space]:
    out = {}
    for key, space in spaces.items():
        wrapped = s2.Space(space.task, space.seed, torch.empty(0), space.c_basis, space.l_basis, dict(space.centroids), "")
        out[key] = wrapped
    return out


def permuted_spaces(args, spaces: dict[tuple[str, int], Stage3Space], values: list[int], draw: int) -> dict[tuple[str, int], s2.Space]:
    out = {}
    for key, space in spaces.items():
        permuted = list(values)
        random.Random(stable_seed("label_permutation", draw, space.task, space.seed)).shuffle(permuted)
        centroids = {value: space.centroids[other] for value, other in zip(values, permuted)}
        wrapped = s2.Space(space.task, space.seed, torch.empty(0), space.c_basis, space.l_basis, centroids, "")
        out[key] = wrapped
    return out


def random_null_spaces(args, activations: dict[str, dict], u_digit: torch.Tensor, hidden_size: int, dim_l: int, draw: int) -> dict[tuple[str, int], s2.Space]:
    out = {}
    for task in args.tasks:
        for seed in args.seeds:
            basis = s2.random_readout_free_basis(
                hidden_size,
                dim_l,
                u_digit,
                stable_seed("stage3_random_readout_free", draw, task, seed),
            )
            wrapped = s2.Space(task, seed, basis, torch.empty(hidden_size, 0), basis, {}, "")
            wrapped.centroids = s2.compute_centroids_for_space(wrapped, activations[task])
            out[(task, seed)] = wrapped
    return out


def null_global_row(args, spaces: dict[tuple[str, int], s2.Space], splits: list[dict], draw: int, null_type: str) -> dict:
    pairwise = s2.run_pairwise_geometry(args, spaces, splits)
    for row in pairwise:
        row["space_type"] = null_type
    rsa = rsa_no_permutation(args, spaces, splits, null_type)
    return {"draw": draw, "null_type": null_type, **metric_means(pairwise, rsa)}


def load_observed_stage2(args) -> dict:
    pairwise_path = args.stage2_dir / "pairwise_geometry.json"
    rsa_path = args.stage2_dir / "rsa.json"
    if not pairwise_path.exists() or not rsa_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 geometry rows: {pairwise_path} or {rsa_path}")
    pairwise = [row for row in load_json(pairwise_path) if row.get("space_type") == "L"]
    rsa = [row for row in load_json(rsa_path) if row.get("space_type") == "L"]
    return metric_means(pairwise, rsa)


def run_null_distributions(args, spaces: dict[tuple[str, int], Stage3Space], activations: dict[str, dict], u_digit: torch.Tensor, splits: list[dict]) -> dict:
    null_dir = args.output_dir / "null_distributions"
    summary_path = null_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[CACHE] null distributions -> {summary_path}")
        return load_json(summary_path)

    observed = load_observed_stage2(args)
    dim_l = next(iter(spaces.values())).l_basis.shape[1]
    hidden_size = next(iter(activations.values()))["activations"].shape[1]
    all_values = sorted(set.intersection(*(set(space.centroids) for space in spaces.values())))

    random_rows = []
    for draw in tqdm(range(args.random_draws), desc="random readout-free null"):
        random_spaces = random_null_spaces(args, activations, u_digit, hidden_size, dim_l, draw)
        random_rows.append(null_global_row(args, random_spaces, splits, draw, "random_readout_free"))

    perm_rows = []
    for draw in tqdm(range(args.label_permutations), desc="label permutation null"):
        label_spaces = permuted_spaces(args, spaces, all_values, draw)
        perm_rows.append(null_global_row(args, label_spaces, splits, draw, "label_permutation"))

    write_csv(null_dir / "random_readout_free.csv", random_rows)
    write_csv(null_dir / "label_permutation.csv", perm_rows)

    metrics = ["heldout_transition_cosine", "top1", "top5", "rsa"]
    rows = []
    payload = {"observed": observed, "metrics": {}}
    for metric in metrics:
        random_values = [row[metric] for row in random_rows]
        perm_values = [row[metric] for row in perm_rows]
        obs = observed[metric]
        row = {
            "metric": metric,
            "observed_L": obs,
            "random_mean": mean(random_values),
            "random_std": std(random_values),
            "random_95_low": quantile(random_values, 0.025),
            "random_95_high": quantile(random_values, 0.975),
            "perm_mean": mean(perm_values),
            "perm_std": std(perm_values),
            "perm_95_low": quantile(perm_values, 0.025),
            "perm_95_high": quantile(perm_values, 0.975),
            "p_random": (1 + sum(value >= obs for value in random_values)) / (len(random_values) + 1),
            "p_perm": (1 + sum(value >= obs for value in perm_values)) / (len(perm_values) + 1),
        }
        rows.append(row)
        payload["metrics"][metric] = row
    write_csv(null_dir / "summary.csv", rows)
    write_json(summary_path, payload)
    if not args.skip_figures:
        save_null_histograms(args, random_rows, perm_rows, observed)
    print("\nNULL DISTRIBUTIONS")
    print_table(
        ["Metric", "observed L", "random mean+/-std", "random 95%", "perm mean+/-std", "p_random", "p_perm"],
        [
            [
                row["metric"],
                fmt(row["observed_L"]),
                f"{fmt(row['random_mean'])}+/-{fmt(row['random_std'])}",
                f"[{fmt(row['random_95_low'])}, {fmt(row['random_95_high'])}]",
                f"{fmt(row['perm_mean'])}+/-{fmt(row['perm_std'])}",
                fmt(row["p_random"]),
                fmt(row["p_perm"]),
            ]
            for row in rows
        ],
    )
    return payload


def save_null_histograms(args, random_rows: list[dict], perm_rows: list[dict], observed: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for metric in ("heldout_transition_cosine", "top1", "top5", "rsa"):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        ax.hist([row[metric] for row in random_rows], bins=30, alpha=0.6, label="random readout-free")
        ax.hist([row[metric] for row in perm_rows], bins=30, alpha=0.6, label="label permutation")
        ax.axvline(observed[metric], color="black", linewidth=2, label="observed L")
        ax.set_title(metric)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"null_{metric}.png", dpi=180)
        plt.close(fig)


def load_samples_from_artifact(artifact: s2.DASArtifact) -> list[dict]:
    data_path = Path(artifact.metrics["data_path"])
    return s2.load_jsonl(data_path)


def sample_lookup(samples: list[dict]) -> dict[str, dict]:
    lookup = {}
    for index, sample in enumerate(samples):
        keys = {str(index), sample_key(sample)}
        if sample.get("sample_id") is not None:
            keys.add(str(sample["sample_id"]))
        for key in keys:
            lookup[key] = sample
    return lookup


def load_heldout_pairs(artifact: s2.DASArtifact, samples: list[dict]) -> list[dict]:
    path = artifact.run_dir / "heldout_pairs.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing heldout pairs for {artifact.task} seed={artifact.seed}: {path}")
    lookup = sample_lookup(samples)
    rows = []
    for index, row in enumerate(s2.load_jsonl(path)):
        base = lookup[str(row["base_sample_id"])]
        source = lookup[str(row["source_sample_id"])]
        rows.append({"pair_id": row.get("pair_id", index), "base": base, "source": source})
    return rows


def build_task_triplets(args, task: str, artifact: s2.DASArtifact) -> tuple[list[dict], list[dict]]:
    samples = load_samples_from_artifact(artifact)
    heldout_pairs = load_heldout_pairs(artifact, samples)
    heldout_samples = {}
    for pair in heldout_pairs:
        heldout_samples[sample_key(pair["base"])] = pair["base"]
        heldout_samples[sample_key(pair["source"])] = pair["source"]
    candidates = list(heldout_samples.values())
    by_tens: dict[str, list[dict]] = defaultdict(list)
    for sample in candidates:
        value = target_value(sample, args.target)
        if two_digit(value):
            by_tens[digits(value)[0]].append(sample)
    rng = random.Random(stable_seed("stage3_triplets", args.triplet_seed, task))
    pairs = list(heldout_pairs)
    rng.shuffle(pairs)
    triplets = []
    for pair in pairs:
        if len(triplets) >= args.max_triplets:
            break
        base = pair["base"]
        donor = pair["source"]
        y = target_value(donor, args.target)
        if not two_digit(y) or target_value(base, args.target) == y:
            continue
        tens = digits(y)[0]
        options = [
            sample for sample in by_tens[tens]
            if sample_key(sample) not in {sample_key(base), sample_key(donor)}
            and digits(target_value(sample, args.target))[1] != digits(y)[1]
        ]
        if not options:
            continue
        y_tilde = rng.choice(options)
        random_options = [
            sample for sample in candidates
            if sample_key(sample) not in {sample_key(base), sample_key(donor), sample_key(y_tilde)}
            and target_value(sample, args.target) not in {target_value(base, args.target), y}
        ]
        if not random_options:
            continue
        random_latent = rng.choice(random_options)
        specificity_options = [
            sample for sample in by_tens[tens]
            if sample_key(sample) not in {sample_key(base), sample_key(donor), sample_key(y_tilde), sample_key(random_latent)}
            and digits(target_value(sample, args.target))[1] not in {digits(y)[1], digits(target_value(y_tilde, args.target))[1]}
        ]
        value_specificity = rng.choice(specificity_options) if specificity_options else random_latent
        triplets.append(
            {
                "triplet_id": f"{task}_{len(triplets):04d}",
                "task": task,
                "base_sample_id": sample_key(base),
                "causal_donor_sample_id": sample_key(donor),
                "latent_donor_sample_id": sample_key(y_tilde),
                "random_latent_sample_id": sample_key(random_latent),
                "value_specificity_sample_id": sample_key(value_specificity),
                "base_value": target_value(base, args.target),
                "causal_donor_value": y,
                "latent_donor_value": target_value(y_tilde, args.target),
                "random_latent_value": target_value(random_latent, args.target),
                "value_specificity_value": target_value(value_specificity, args.target),
                "base": base,
                "causal_donor": donor,
                "latent_donor": y_tilde,
                "random_latent": random_latent,
                "value_specificity": value_specificity,
            }
        )
    if len(triplets) < args.max_triplets:
        print(f"WARNING: {task} has {len(triplets)} eligible same-first/diff-second triplets, requested {args.max_triplets}.")
    if not triplets:
        raise RuntimeError(f"No eligible Stage 3 triplets for {task}.")
    return triplets, samples


def project_delta(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0 or basis.shape[1] == 0:
        return torch.zeros_like(delta)
    return (delta.float() @ basis.float()) @ basis.float().T


def encode_sample(args, model, processor, tokenizer, sample: dict, task: str, data_root: Path) -> tuple[dict, int, int, str]:
    if task.startswith("T"):
        prompt = format_prompt(tokenizer, sample, args.use_chat_template)
        position = resolve_position(tokenizer, prompt, args.position)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    else:
        prompt = sample_prompt(processor, sample, args.prompt, args.enable_thinking)
        image = load_rgb_image(image_path_for(sample, data_root))
        position = resolve_batch_positions(processor, tokenizer, model, [prompt], [image], args.position)[0]
        inputs = inputs_to_device(
            make_inputs(processor, [prompt], [image]),
            model.device,
            dtype=getattr(model, "dtype", None),
        )
    length = int(inputs["attention_mask"][0].sum()) if "attention_mask" in inputs else int(inputs["input_ids"].shape[1])
    return inputs, position, length, prompt


def add_delta_hook(stack: ExitStack, model, blocks, layer: int, hook_name: str, position: int, delta: torch.Tensor) -> None:
    delta = delta.detach().to(model.device)

    def patch_tensor(value):
        activations = hidden(value)
        updated = activations.clone()
        updated[0, position] = activations[0, position] + delta.to(dtype=activations.dtype)
        return replace_hidden(value, updated)

    module, pre_hook = hook_module(blocks[layer - 1], hook_name)
    if pre_hook:
        def hook(_module, inputs):
            return (patch_tensor(inputs[0]), *inputs[1:])

        handle = module.register_forward_pre_hook(hook)
    else:
        def hook(_module, _inputs, output):
            return patch_tensor(output)

        handle = module.register_forward_hook(hook)
    stack.callback(handle.remove)


@torch.no_grad()
def patched_prompt_forward(args, model, processor, tokenizer, blocks, sample: dict, task: str, data_root: Path, delta: torch.Tensor) -> PromptForward:
    inputs, position, length, prompt = encode_sample(args, model, processor, tokenizer, sample, task, data_root)
    with ExitStack() as stack:
        add_delta_hook(stack, model, blocks, args.layer, args.hook, position, delta)
        outputs = model(**inputs, use_cache=True)
    return PromptForward(
        logits_t0=outputs.logits[0, length - 1].float().detach().cpu(),
        past_key_values=outputs.past_key_values,
        prompt_length=length,
        patch_position=position,
        prompt=prompt,
    )


@torch.no_grad()
def cached_digit_logits(model, prompt_forward: PromptForward, digit_token_id: int) -> torch.Tensor:
    input_ids = torch.tensor([[digit_token_id]], dtype=torch.long, device=model.device)
    attention_mask = torch.ones((1, prompt_forward.prompt_length + 1), dtype=torch.long, device=model.device)
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=prompt_forward.past_key_values,
            use_cache=True,
        )
    except (TypeError, RuntimeError) as error:
        if "cache" not in str(error).lower() and "mask" not in str(error).lower() and "shape" not in str(error).lower():
            raise
        outputs = model(input_ids=input_ids, past_key_values=prompt_forward.past_key_values, use_cache=True)
    return outputs.logits[0, 0].float().detach().cpu()


@torch.no_grad()
def generate_from_patch(args, model, processor, tokenizer, blocks, sample: dict, task: str, data_root: Path, delta: torch.Tensor) -> tuple[str, list[dict]]:
    prompt = patched_prompt_forward(args, model, processor, tokenizer, blocks, sample, task, data_root, delta)
    logits = prompt.logits_t0
    past = prompt.past_key_values
    generated = []
    token_rows = []
    for step in range(args.max_new_tokens):
        next_id = int(logits.argmax().item())
        generated.append(next_id)
        token_rows.append({"step": step, "token_id": next_id, "token_text": tokenizer.decode([next_id], skip_special_tokens=False)})
        if next_id == tokenizer.eos_token_id:
            break
        input_ids = torch.tensor([[next_id]], dtype=torch.long, device=model.device)
        attention_mask = torch.ones((1, prompt.prompt_length + step + 1), dtype=torch.long, device=model.device)
        try:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        except (TypeError, RuntimeError) as error:
            if "cache" not in str(error).lower() and "mask" not in str(error).lower() and "shape" not in str(error).lower():
                raise
            outputs = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, 0].float().detach().cpu()
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), token_rows


def digit_token_ids(tokenizer) -> dict[str, int]:
    out = {}
    for digit in range(10):
        ids = tokenizer(str(digit), add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"Digit {digit} is not one token: {ids}")
        out[str(digit)] = int(ids[0])
    return out


def prob_for_token(logits: torch.Tensor, token_id: int) -> float:
    return float(logits.float().softmax(dim=-1)[token_id])


def condition_deltas(space: Stage3Space, triplet: dict) -> dict[str, torch.Tensor]:
    h_b = space.activation_by_key[triplet["base_sample_id"]]
    h_y = space.activation_by_key[triplet["causal_donor_sample_id"]]
    h_tilde = space.activation_by_key[triplet["latent_donor_sample_id"]]
    h_random = space.activation_by_key[triplet["random_latent_sample_id"]]
    h_specific = space.activation_by_key[triplet["value_specificity_sample_id"]]
    delta_y = h_y - h_b
    c_y = project_delta(delta_y, space.c_basis)
    l_y = project_delta(delta_y, space.l_basis)
    l_tilde = project_delta(h_tilde - h_b, space.l_basis)
    l_random = project_delta(h_random - h_b, space.l_basis)
    l_specific = project_delta(h_specific - h_b, space.l_basis)
    return {
        "matched": c_y + l_y,
        "mismatch": c_y + l_tilde,
        "C-only": c_y,
        "random_latent": c_y + l_random,
        "value_specificity": c_y + l_specific,
    }


def task_data_root(artifact: s2.DASArtifact) -> Path:
    return Path(artifact.metrics["data_path"]).parent


def run_causal_subset(
    args,
    *,
    tasks: list[str],
    seeds: list[int],
    spaces: dict[tuple[str, int], Stage3Space],
    artifacts: dict[tuple[str, int], s2.DASArtifact],
    task_triplets: dict[str, list[dict]],
    model,
    processor,
    tokenizer,
    blocks,
    digit_ids: dict[str, int],
    label: str,
) -> tuple[list[dict], list[dict]]:
    rows = []
    token_rows = []
    for task in tasks:
        triplets = task_triplets[task][: args.max_triplets]
        data_root = task_data_root(artifacts[(task, seeds[0])])
        for seed in seeds:
            space = spaces[(task, seed)]
            print(f"{label}: {task} seed={seed} triplets={len(triplets)}")
            for triplet in tqdm(triplets, desc=f"{label} {task} S{seed}"):
                y = int(triplet["causal_donor_value"])
                y_tilde = int(triplet["latent_donor_value"])
                y_digits = digits(y)
                tilde_digits = digits(y_tilde)
                if not (len(y_digits) == 2 and len(tilde_digits) == 2 and y_digits[0] == tilde_digits[0] and y_digits[1] != tilde_digits[1]):
                    raise ValueError(f"Bad Stage 3 triplet: {triplet}")
                first_digit = y_digits[0]
                y2 = y_digits[1]
                y_tilde2 = tilde_digits[1]
                deltas = condition_deltas(space, triplet)
                matched_first_prob = None
                for condition, delta in deltas.items():
                    prompt_forward = patched_prompt_forward(args, model, processor, tokenizer, blocks, triplet["base"], task, data_root, delta)
                    p_first = prob_for_token(prompt_forward.logits_t0, digit_ids[first_digit])
                    if condition == "matched":
                        matched_first_prob = p_first
                    logits_t1 = cached_digit_logits(model, prompt_forward, digit_ids[first_digit])
                    p_y2 = prob_for_token(logits_t1, digit_ids[y2])
                    p_tilde2 = prob_for_token(logits_t1, digit_ids[y_tilde2])
                    greedy_t1 = int(logits_t1.argmax().item())
                    row = {
                        "run_stage": label,
                        "task": task,
                        "das_seed": seed,
                        "triplet_id": triplet["triplet_id"],
                        "condition": condition,
                        "base_value": triplet["base_value"],
                        "causal_donor_value": y,
                        "latent_donor_value": y_tilde,
                        "random_latent_value": triplet["random_latent_value"],
                        "value_specificity_value": triplet["value_specificity_value"],
                        "first_digit": first_digit,
                        "y_second_digit": y2,
                        "y_tilde_second_digit": y_tilde2,
                        "p_shared_first_digit": p_first,
                        "p_first_mismatch_minus_matched": None,
                        "p_y_second": p_y2,
                        "p_y_tilde_second": p_tilde2,
                        "greedy_second_token_id": greedy_t1,
                        "greedy_second_token_text": tokenizer.decode([greedy_t1], skip_special_tokens=False),
                        "greedy_second_is_y": greedy_t1 == digit_ids[y2],
                        "greedy_second_is_y_tilde": greedy_t1 == digit_ids[y_tilde2],
                        "patched_kv_cache_reused_for_t1": True,
                        "t1_patch_applied": False,
                    }
                    if condition in {"matched", "mismatch", "C-only", "random_latent"}:
                        generated, generated_tokens = generate_from_patch(
                            args, model, processor, tokenizer, blocks, triplet["base"], task, data_root, delta
                        )
                        parsed = parse_int_prefix(generated)
                        row.update(
                            {
                                "generated_answer": generated,
                                "parsed_answer": parsed,
                                "ar_toward_y": parsed == str(y),
                                "ar_toward_y_tilde": parsed == str(y_tilde),
                                "ar_toward_random": parsed == str(triplet["random_latent_value"]),
                            }
                        )
                        for token in generated_tokens:
                            token_rows.append({**{key: row[key] for key in ("run_stage", "task", "das_seed", "triplet_id", "condition")}, **token})
                    rows.append(row)
                if matched_first_prob is not None:
                    for row in rows:
                        if row["task"] == task and row["das_seed"] == seed and row["triplet_id"] == triplet["triplet_id"] and row["condition"] == "mismatch":
                            row["p_first_mismatch_minus_matched"] = row["p_shared_first_digit"] - matched_first_prob
    return rows, token_rows


def attach_pair_diffs(rows: list[dict]) -> None:
    by_triplet = defaultdict(dict)
    for row in rows:
        by_triplet[(row["run_stage"], row["task"], row["das_seed"], row["triplet_id"])][row["condition"]] = row
    for parts in by_triplet.values():
        matched = parts.get("matched")
        mismatch = parts.get("mismatch")
        if matched and mismatch:
            diff = mismatch["p_shared_first_digit"] - matched["p_shared_first_digit"]
            matched["p_first_mismatch_minus_matched"] = diff
            mismatch["p_first_mismatch_minus_matched"] = diff


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_stage"], row["task"], row["das_seed"], row["condition"])].append(row)
    output = []
    for (stage, task, seed, condition), parts in sorted(grouped.items()):
        output.append(
            {
                "run_stage": stage,
                "task": task,
                "das_seed": seed,
                "condition": condition,
                "n": len(parts),
                "toward_y": mean(row.get("ar_toward_y") for row in parts),
                "toward_y_tilde": mean(row.get("ar_toward_y_tilde") for row in parts),
                "toward_random": mean(row.get("ar_toward_random") for row in parts),
                "p_shared_first_digit": mean(row["p_shared_first_digit"] for row in parts),
                "p_first_mismatch_minus_matched": mean(row.get("p_first_mismatch_minus_matched") for row in parts),
                "p_y_second": mean(row["p_y_second"] for row in parts),
                "p_y_tilde_second": mean(row["p_y_tilde_second"] for row in parts),
                "greedy_y_second": mean(row["greedy_second_is_y"] for row in parts),
                "greedy_y_tilde_second": mean(row["greedy_second_is_y_tilde"] for row in parts),
            }
        )
    return output


def condition_summary(rows: list[dict], stage: str) -> dict[str, dict]:
    selected = [row for row in rows if row["run_stage"] == stage]
    grouped = defaultdict(list)
    for row in selected:
        grouped[row["condition"]].append(row)
    return {
        condition: {
            "toward_y": mean(row.get("ar_toward_y") for row in parts),
            "toward_y_tilde": mean(row.get("ar_toward_y_tilde") for row in parts),
            "toward_random": mean(row.get("ar_toward_random") for row in parts),
            "p_shared_first_digit": mean(row["p_shared_first_digit"] for row in parts),
            "p_y_second": mean(row["p_y_second"] for row in parts),
            "p_y_tilde_second": mean(row["p_y_tilde_second"] for row in parts),
            "greedy_y_second": mean(row["greedy_second_is_y"] for row in parts),
            "greedy_y_tilde_second": mean(row["greedy_second_is_y_tilde"] for row in parts),
        }
        for condition, parts in grouped.items()
    }


def pilot_passed(args, rows: list[dict]) -> tuple[bool, dict]:
    summary = condition_summary(rows, "pilot")
    matched = summary.get("matched", {})
    mismatch = summary.get("mismatch", {})
    random_latent = summary.get("random_latent", {})
    value_specificity = summary.get("value_specificity", {})
    p_shift = (mismatch.get("p_y_tilde_second") or 0.0) - (matched.get("p_y_tilde_second") or 0.0)
    random_gap = (mismatch.get("p_y_tilde_second") or 0.0) - (random_latent.get("p_y_tilde_second") or 0.0)
    specific_gap = (mismatch.get("p_y_tilde_second") or 0.0) - (value_specificity.get("p_y_tilde_second") or 0.0)
    first_delta = abs((mismatch.get("p_shared_first_digit") or 0.0) - (matched.get("p_shared_first_digit") or 0.0))
    decision = {
        "p_y_tilde_second_mismatch_minus_matched": p_shift,
        "p_y_tilde_second_mismatch_minus_random": random_gap,
        "p_y_tilde_second_mismatch_minus_value_specificity": specific_gap,
        "first_digit_abs_delta": first_delta,
        "second_prob_margin": args.pilot_second_prob_margin,
        "first_prob_tolerance": args.pilot_first_prob_tolerance,
    }
    passed = (
        p_shift >= args.pilot_second_prob_margin
        and random_gap >= 0.0
        and specific_gap >= 0.0
        and first_delta <= args.pilot_first_prob_tolerance
    )
    return passed, decision


def strip_triplets(triplets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    heavy = {"base", "causal_donor", "latent_donor", "random_latent", "value_specificity"}
    return {task: [{key: value for key, value in row.items() if key not in heavy} for row in rows] for task, rows in triplets.items()}


def save_causal_outputs(args, triplets, per_example_rows, token_rows, aggregates, null_summary, pilot_decision, completed_full: bool) -> None:
    attach_pair_diffs(per_example_rows)
    aggregates[:] = aggregate_rows(per_example_rows)
    write_json(args.output_dir / "triplets.json", strip_triplets(triplets))
    write_csv(args.output_dir / "per_example.csv", per_example_rows)
    write_csv(args.output_dir / "generated_tokens.csv", token_rows)
    write_csv(args.output_dir / "aggregate_by_task_seed.csv", aggregates)
    summary = {
        "config": vars(args),
        "completed_full_run": completed_full,
        "pilot_decision": pilot_decision,
        "null_distributions": null_summary,
        "causal_swap": condition_summary(per_example_rows, "pilot" if not completed_full else "full"),
        "aggregates": aggregates,
        "means_by_task": summarize_dimension(per_example_rows, "task"),
        "means_by_seed": summarize_dimension(per_example_rows, "das_seed"),
        "means_by_task_seed": summarize_dimension(per_example_rows, ("task", "das_seed")),
    }
    write_json(args.output_dir / "summary.json", summary)


def summarize_dimension(rows: list[dict], dimension) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if isinstance(dimension, tuple):
            key = "|".join(str(row[item]) for item in dimension)
        else:
            key = str(row[dimension])
        grouped[key].append(row)
    return {
        key: {
            "p_shared_first_digit_mean": mean(row["p_shared_first_digit"] for row in parts),
            "p_y_second_mean": mean(row["p_y_second"] for row in parts),
            "p_y_tilde_second_mean": mean(row["p_y_tilde_second"] for row in parts),
            "toward_y_mean": mean(row.get("ar_toward_y") for row in parts),
            "toward_y_tilde_mean": mean(row.get("ar_toward_y_tilde") for row in parts),
        }
        for key, parts in sorted(grouped.items())
    }


def print_causal_tables(rows: list[dict], stage: str) -> None:
    summary = condition_summary(rows, stage)
    print("\nCAUSAL SWAP")
    print_table(
        ["Condition", "toward y", "toward y_tilde", "P(shared first digit)"],
        [
            [condition, fmt(values.get("toward_y")), fmt(values.get("toward_y_tilde")), fmt(values.get("p_shared_first_digit"))]
            for condition, values in summary.items()
            if condition in {"matched", "mismatch", "C-only", "random_latent"}
        ],
    )
    print("\nNEXT DIGIT")
    print_table(
        ["Condition", "P(y2)", "P(y_tilde2)", "greedy y2", "greedy y_tilde2"],
        [
            [condition, fmt(values.get("p_y_second")), fmt(values.get("p_y_tilde_second")), fmt(values.get("greedy_y_second")), fmt(values.get("greedy_y_tilde_second"))]
            for condition, values in summary.items()
        ],
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.artifact_check_only and args.output_dir.joinpath("summary.json").exists() and not args.force:
        raise FileExistsError(f"{args.output_dir / 'summary.json'} exists. Pass --force to overwrite.")

    print("Stage 3 Ministral latent causal experiment")
    print(f"  stage2_dir={args.stage2_dir}")
    print(f"  output_dir={args.output_dir}")
    print(f"  layer={args.layer} k={args.k} position={args.position}")

    artifacts = s2.find_final_artifacts(args, hidden_size=None)
    spaces, activations, u_digit, splits = load_stage2_spaces(args)
    dim_c = next(iter(spaces.values())).c_basis.shape[1]
    dim_l = next(iter(spaces.values())).l_basis.shape[1]
    if dim_c != 9 or dim_l != 23:
        raise ValueError(f"Expected dimC=9 and dimL=23, got dimC={dim_c}, dimL={dim_l}.")
    if args.artifact_check_only:
        print("ARTIFACT_CHECK_OK stage3_latent_causal")
        return 0

    model_bundle = None
    if not args.skip_nulls and u_digit is None:
        model_bundle = load_model_bundle(args)
        model, _processor, tokenizer, _blocks, _hidden_size, _saved_model_name = model_bundle
        u_digit, _readout_info = s2.digit_readout_basis(model, tokenizer, args)

    null_summary = None
    if not args.skip_nulls:
        if u_digit is None:
            raise RuntimeError("U_digit is required for random readout-free nulls.")
        null_summary = run_null_distributions(args, spaces, activations, u_digit, splits)

    task_triplets = {}
    for task in args.tasks:
        triplets, _samples = build_task_triplets(args, task, artifacts[(task, args.seeds[0])])
        task_triplets[task] = triplets

    if model_bundle is None:
        model_bundle = load_model_bundle(args)
    model, processor, tokenizer, blocks, hidden_size, saved_model_name = model_bundle
    if hidden_size != next(iter(activations.values()))["activations"].shape[1]:
        raise ValueError("Stage 2 activations and loaded model hidden size disagree.")
    digit_ids = digit_token_ids(tokenizer)

    per_example_rows, token_rows = run_causal_subset(
        args,
        tasks=args.pilot_tasks,
        seeds=args.pilot_seeds,
        spaces=spaces,
        artifacts=artifacts,
        task_triplets=task_triplets,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        digit_ids=digit_ids,
        label="pilot",
    )
    attach_pair_diffs(per_example_rows)
    passed, pilot_decision = pilot_passed(args, per_example_rows)
    print_causal_tables(per_example_rows, "pilot")
    print(f"pilot_passed = {passed} {pilot_decision}")

    completed_full = False
    if passed or args.force_full:
        remaining_tasks_seeds = [
            (task, seed)
            for task in args.tasks
            for seed in args.seeds
            if not (task in args.pilot_tasks and seed in args.pilot_seeds)
        ]
        if remaining_tasks_seeds:
            full_tasks_by_seed = defaultdict(list)
            for task, seed in remaining_tasks_seeds:
                full_tasks_by_seed[seed].append(task)
            for seed, tasks in sorted(full_tasks_by_seed.items()):
                rows, tokens = run_causal_subset(
                    args,
                    tasks=tasks,
                    seeds=[seed],
                    spaces=spaces,
                    artifacts=artifacts,
                    task_triplets=task_triplets,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    blocks=blocks,
                    digit_ids=digit_ids,
                    label="full",
                )
                per_example_rows.extend(rows)
                token_rows.extend(tokens)
        for row in per_example_rows:
            if row["run_stage"] == "pilot":
                row["run_stage"] = "full"
        completed_full = True
        attach_pair_diffs(per_example_rows)
        print_causal_tables(per_example_rows, "full")
    else:
        print("Stopping after pilot because the adaptive causal signature did not pass.")

    aggregates = aggregate_rows(per_example_rows)
    save_causal_outputs(args, task_triplets, per_example_rows, token_rows, aggregates, null_summary, pilot_decision, completed_full)
    print("\nMeans by task")
    print_table(
        ["task", "P(first)", "P(y2)", "P(y_tilde2)", "toward y", "toward y_tilde"],
        [
            [task, fmt(values["p_shared_first_digit_mean"]), fmt(values["p_y_second_mean"]), fmt(values["p_y_tilde_second_mean"]), fmt(values["toward_y_mean"]), fmt(values["toward_y_tilde_mean"])]
            for task, values in summarize_dimension(per_example_rows, "task").items()
        ],
    )
    print(f"\nSaved Stage 3 outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
