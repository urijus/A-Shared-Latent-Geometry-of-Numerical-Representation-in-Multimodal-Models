"""Minimal DAS audit: full patching, PCA controls, and one trained DAS run.

Run exactly one condition at a time:
  - one modality: text or image
  - one operation: addition, subtraction, or multiplication
  - one layer, position, and seed

The script uses ``split_seed`` for the dataset split and held-out pairs, and
``seed`` for optimization randomness.  This lets us repeat DAS runs with
different optimizer/minibatch seeds while keeping the data fixed.

    eta = (IIA_condition - IIA_clean) / (IIA_full_patch - IIA_clean)
"""

import argparse
import json
import re
from pathlib import Path

import torch

from src.common import load_jsonl, save_jsonl
from src.interventions.das import (
    DASSubspace,
    autoregressive_iia,
    build_unique_pairs,
    collect_initialization_features,
    dataset_statistics,
    evaluate_teacher_forced,
    random_subspace_from_pca,
    split_samples,
    target_answers,
    format_prompt,
    print_patch_diagnostics,
    train_subspace,
)
from src.experiments.arithmetic_reference.das.image.das_image_core import (
    autoregressive_iia_image,
    collect_initialization_features_image,
    evaluate_teacher_forced_image,
    image_path_for,
    inputs_to_device,
    load_rgb_image,
    make_inputs,
    print_position_summary,
    sample_prompt,
    train_subspace_image,
)
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    uses_chat_template,
)
from src.models import (
    get_blocks,
    get_hidden_size,
    load_hf_model,
    load_hf_model_and_processor,
    resolve_model_for_loading,
    validate_block_layers,
)


DEFAULT_DATA = {
    ("text", "addition"): Path("dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/addition_baseline.jsonl"),
    ("text", "subtraction"): Path("dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/subtraction_baseline.jsonl"),
    ("text", "multiplication"): Path("dataset/baseline/gemma4_12b_it/digits/model_correct_with_prompt/multiplication_baseline.jsonl"),
    ("image", "addition"): Path("dataset/baseline_images/gemma4_12b_it/digits/model_correct_with_prompt/addition/addition_images.jsonl"),
    ("image", "subtraction"): Path("dataset/baseline_images/gemma4_12b_it/digits/model_correct_with_prompt/subtraction/subtraction_images.jsonl"),
}


class FullActivationPatch:
    """DAS-compatible object that copies the full donor vector."""

    def patch(self, _base_vector, source_vector):
        return source_vector


def log_stage(message):
    print(f"[stage] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4_12b_it")
    parser.add_argument("--modality", choices=["text", "image"], required=True)
    parser.add_argument("--operation", choices=["addition", "subtraction", "multiplication"], required=True)
    parser.add_argument("--data_path", type=Path)
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("results/final_exps/DAS_audit"))
    parser.add_argument("--layer", type=int, default=43)
    parser.add_argument("--position", default="17")
    parser.add_argument("--hook", choices=["resid_pre", "resid_post"], default="resid_post")
    parser.add_argument("--seed", type=int, default=0, help="Optimization/run seed.")
    parser.add_argument("--split_seed", type=int, default=0, help="Dataset split seed.")
    parser.add_argument(
        "--target",
        choices=["result", "c0_hat", "c1_hat", "c1_hat_full"],
        default="result",
    )
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--random_init_epochs", type=int, default=40)
    parser.add_argument("--run_random_init", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--pca_max_samples", type=int, default=1024)
    parser.add_argument("--pca_variance_threshold", type=float, default=0.9)
    parser.add_argument("--max_train_pairs", type=int, default=4096)
    parser.add_argument("--max_validation_pairs", type=int, default=512)
    parser.add_argument("--max_test_pairs", type=int, default=128)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_autoregressive_pairs", type=int, default=32)
    parser.add_argument(
        "--prompt",
        default="Output ONLY a number.",
        help="Only used for image experiments.",
    )
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--das_only",
        action="store_true",
        help="Only train/evaluate the PCA-initialized DAS condition; skip full patch and control conditions.",
    )
    return parser.parse_args()


def resolve_data_path(args):
    if args.data_path is not None:
        return args.data_path
    key = (args.modality, args.operation)
    if key not in DEFAULT_DATA:
        raise ValueError(
            f"No default dataset for {args.modality} {args.operation}; "
            "pass --data_path explicitly."
        )
    return DEFAULT_DATA[key]


def make_pairs(samples, args):
    train_samples, validation_samples, test_samples = split_samples(
        samples, args.train_fraction, args.validation_fraction, args.split_seed
    )
    train_pairs, train_stats = build_unique_pairs(
        train_samples, args.target, args.split_seed, args.max_train_pairs
    )
    validation_pairs, validation_stats = build_unique_pairs(
        validation_samples, args.target, args.split_seed + 1, args.max_validation_pairs
    )
    test_pairs, test_stats = build_unique_pairs(
        test_samples, args.target, args.split_seed + 2, args.max_test_pairs
    )
    pair_stats = {
        "split_mode": "sample_disjoint_then_unique_pairs",
        "split_seed": args.split_seed,
        "run_seed": args.seed,
        "train": train_stats,
        "validation": validation_stats,
        "test": test_stats,
        "n_train_pairs": len(train_pairs),
        "n_validation_pairs": len(validation_pairs),
        "n_test_pairs": len(test_pairs),
    }
    return train_pairs, validation_pairs, test_pairs, pair_stats


def save_pairs(path, pairs):
    rows = []
    for index, pair in enumerate(pairs):
        rows.append(
            {
                "pair_id": index,
                "base_sample_id": pair["base"].get("sample_id"),
                "source_sample_id": pair["source"].get("sample_id"),
                "base_expr": pair["base"].get("expr", pair["base"].get("image_text")),
                "source_expr": pair["source"].get("expr", pair["source"].get("image_text")),
                "base_result": pair["base"]["result"],
                "source_result": pair["source"]["result"],
            }
        )
    save_jsonl(rows, path)


def unique_samples_from_pairs(pairs):
    samples = []
    seen = set()
    for pair in pairs:
        for sample in (pair["base"], pair["source"]):
            sample_id = sample.get("sample_id", (sample.get("a"), sample.get("b")))
            if sample_id not in seen:
                seen.add(sample_id)
                samples.append(sample)
    return samples


def eta(condition_iia, clean_iia, full_iia):
    if condition_iia is None or clean_iia is None or full_iia is None:
        return None
    denominator = full_iia - clean_iia
    if abs(denominator) < 1e-12:
        return None
    return (condition_iia - clean_iia) / denominator


def full_patch_subspace(layer):
    return {str(layer): FullActivationPatch()}


def subspace_from_basis(layer, hidden_size, basis, device):
    return {
        str(layer): DASSubspace(
            hidden_size, basis.shape[1], initial_basis=basis
        ).to(device)
    }


def pca_basis(pca_space, k, seed):
    old_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    basis = random_subspace_from_pca(pca_space, k)
    torch.random.set_rng_state(old_state)
    return basis


def pca_space_with_at_least_k(features, k, variance_threshold):
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    max_components = min(centered.shape[0] - 1, centered.shape[1], 500)
    _, singular_values, components = torch.pca_lowrank(
        centered, q=max_components, center=False, niter=2
    )
    total_variance = centered.square().sum().clamp_min(1e-12)
    cumulative = singular_values.square().cumsum(0) / total_variance
    threshold_k = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(variance_threshold, dtype=cumulative.dtype),
        ).item()
    ) + 1
    if max_components < k:
        raise ValueError(f"Need k={k} PCA components, but only {max_components} exist.")
    n_components = max(k, threshold_k)
    print(
        f"PCA space: using {n_components} components, "
        f"explained variance={float(cumulative[n_components - 1]):.3f}"
    )
    return components[:, :n_components]


def task_dir(args):
    folder = args.output_dir / args.modality / args.operation
    return folder if args.target == "result" else folder / args.target


def condition_dir(args, condition):
    return (
        task_dir(args)
        / condition
        / f"split_{args.split_seed}"
        / f"seed_{args.seed}"
    )


def save_condition(args, condition, row, subspace_payload, test_pairs):
    folder = condition_dir(args, condition)
    folder.mkdir(parents=True, exist_ok=True)
    save_jsonl([row], folder / "results.jsonl")
    save_pairs(folder / "heldout_pairs.jsonl", test_pairs)
    torch.save(subspace_payload, folder / "subspace.pt")


def summarize_condition(args, condition, metrics, clean_ar, condition_ar, clean, full):
    variable = metrics["variable_teacher_forced_iia"]
    full_answer = metrics["full_answer_teacher_forced_iia"]
    return {
        "condition": condition,
        "modality": args.modality,
        "operation": args.operation,
        "layer": args.layer,
        "position": str(args.position),
        "hook": args.hook,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "k": None if condition == "full_activation_patching" else args.k,
        "variable_teacher_forced_iia": variable,
        "full_answer_teacher_forced_iia": full_answer,
        "autoregressive_iia": condition_ar,
        "clean_variable_teacher_forced_iia": clean["variable"],
        "full_patch_variable_teacher_forced_iia": full["variable"],
        "eta_variable_teacher_forced": eta(variable, clean["variable"], full["variable"]),
        "clean_full_answer_teacher_forced_iia": clean["full_answer"],
        "full_patch_full_answer_teacher_forced_iia": full["full_answer"],
        "eta_full_answer_teacher_forced": eta(full_answer, clean["full_answer"], full["full_answer"]),
        "clean_autoregressive_iia": clean_ar,
        "full_patch_autoregressive_iia": full["autoregressive"],
        "eta_autoregressive": eta(condition_ar, clean_ar, full["autoregressive"]),
    }


@torch.no_grad()
def clean_autoregressive_iia_text(
    model,
    tokenizer,
    pairs,
    target,
    use_chat_template,
    max_new_tokens,
):
    correct = 0
    for pair in pairs:
        base, source = pair["base"], pair["source"]
        prompt = format_prompt(tokenizer, base, use_chat_template)
        expected = target_answers(base, source, target)[1]
        input_ids = tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(model.device)
        generated = []
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, use_cache=False)
            next_id = outputs.logits[0, input_ids.shape[1] - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            input_ids = torch.cat([input_ids, next_id.to(model.device)], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs) if pairs else 0.0


@torch.no_grad()
def clean_autoregressive_iia_image(
    model,
    processor,
    tokenizer,
    pairs,
    data_root,
    prompt,
    target,
    enable_thinking,
    max_new_tokens,
):
    correct = 0
    for pair in pairs:
        base, source = pair["base"], pair["source"]
        expected = target_answers(base, source, target)[1]
        text = sample_prompt(processor, base, prompt, enable_thinking)
        image = load_rgb_image(image_path_for(base, data_root))
        inputs = inputs_to_device(
            make_inputs(processor, [text], [image]),
            model.device,
            dtype=getattr(model, "dtype", None),
        )
        generated = []
        for _ in range(max_new_tokens):
            length = int(inputs["attention_mask"][0].sum())
            outputs = model(**inputs, use_cache=False)
            next_id = outputs.logits[0, length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_id.to(model.device)], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(next_id).to(model.device)],
                dim=1,
            )
            if next_id.item() == tokenizer.eos_token_id:
                break
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", decoded)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs) if pairs else 0.0


def evaluate_text(args, train_pairs, validation_pairs, test_pairs):
    log_stage(
        f"text {args.operation} L{args.layer} K{args.k}: resolving model path"
    )
    model_path, model_name = resolve_model_for_loading(args.model)
    log_stage(f"text {args.operation} L{args.layer} K{args.k}: loading model")
    model, tokenizer = load_hf_model(model_path)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    hidden_size = get_hidden_size(model)
    use_chat_template = args.use_chat_template or uses_chat_template(args.model)
    log_stage(
        f"text {args.operation} L{args.layer} K{args.k}: model ready "
        f"(hidden_size={hidden_size})"
    )
    print_patch_diagnostics(
        tokenizer=tokenizer,
        samples=[
            ("train base", train_pairs[0]["base"]),
            ("train source", train_pairs[0]["source"]),
            ("test base", test_pairs[0]["base"]),
            ("test source", test_pairs[0]["source"]),
        ],
        use_chat_template=use_chat_template,
        position=args.position,
    )

    ar_pairs = (
        test_pairs[: args.max_autoregressive_pairs]
        if args.max_autoregressive_pairs > 0
        else test_pairs
    )
    full_metrics = None
    clean_ar = None
    full_ar = None
    if not args.das_only:
        log_stage(
            f"text {args.operation} L{args.layer} K{args.k}: evaluating full patch controls"
        )
        full_patch = full_patch_subspace(args.layer)
        full_metrics = evaluate_teacher_forced(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=test_pairs,
            position=args.position,
            target=args.target,
            use_chat_template=use_chat_template,
            batch_size=args.batch_size,
            include_clean=True,
        )
        clean_ar = clean_autoregressive_iia_text(
            model, tokenizer, ar_pairs, args.target, use_chat_template, args.max_new_tokens
        )
        full_ar = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=ar_pairs,
            position=args.position,
            target=args.target,
            use_chat_template=use_chat_template,
            max_new_tokens=args.max_new_tokens,
            description="full patch autoregressive",
        )

    pca_samples = unique_samples_from_pairs(train_pairs)
    log_stage(
        f"text {args.operation} L{args.layer} K{args.k}: collecting PCA features "
        f"(samples<= {args.pca_max_samples}, unique_train_samples={len(pca_samples)})"
    )
    features = collect_initialization_features(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        samples=pca_samples,
        layers=[args.layer],
        hook_name=args.hook,
        position=args.position,
        use_chat_template=use_chat_template,
        batch_size=args.batch_size,
        max_samples=args.pca_max_samples,
    )[args.layer]
    log_stage(
        f"text {args.operation} L{args.layer} K{args.k}: PCA features collected "
        f"(n={features.shape[0]}, d={features.shape[1]})"
    )
    pca_space = pca_space_with_at_least_k(
        features, args.k, args.pca_variance_threshold
    )

    def evaluate_subspace(name, subspace):
        log_stage(
            f"text {args.operation} L{args.layer} K{args.k}: evaluating {name} "
            f"(test_pairs={len(test_pairs)}, ar_pairs={len(ar_pairs)})"
        )
        metrics = evaluate_teacher_forced(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspace,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=test_pairs,
            position=args.position,
            target=args.target,
            use_chat_template=use_chat_template,
            batch_size=args.batch_size,
            include_clean=False,
        )
        ar = autoregressive_iia(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspace,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=ar_pairs,
            position=args.position,
            target=args.target,
            use_chat_template=use_chat_template,
            max_new_tokens=args.max_new_tokens,
            description=f"{name} autoregressive",
        )
        return metrics, ar

    init_seed = args.seed + args.layer * 100 + args.k
    conditions = []
    if not args.das_only:
        initial_basis = pca_basis(pca_space, args.k, init_seed)
        untrained = subspace_from_basis(args.layer, hidden_size, initial_basis, model.device)
        untrained_metrics, untrained_ar = evaluate_subspace(
            "untrained PCA init", untrained
        )

        random_basis = pca_basis(pca_space, args.k, init_seed + 1)
        random_pca = subspace_from_basis(args.layer, hidden_size, random_basis, model.device)
        random_metrics, random_ar = evaluate_subspace(
            "random PCA subspace", random_pca
        )
        conditions.extend(
            [
                {
                    "condition": "full_activation_patching",
                    "metrics": full_metrics,
                    "autoregressive_iia": full_ar,
                    "best_epoch": None,
                    "subspace": {"kind": "full_activation_patch"},
                },
                {
                    "condition": "untrained_pca_initialized",
                    "metrics": untrained_metrics,
                    "autoregressive_iia": untrained_ar,
                    "best_epoch": 0,
                    "subspace": {
                        "basis": untrained[str(args.layer)].basis().detach().cpu(),
                        "kind": "untrained_pca_initialized",
                    },
                },
                {
                    "condition": "random_subspace_in_pca_span",
                    "metrics": random_metrics,
                    "autoregressive_iia": random_ar,
                    "best_epoch": 0,
                    "subspace": {
                        "basis": random_pca[str(args.layer)].basis().detach().cpu(),
                        "kind": "random_subspace_in_pca_span",
                    },
                },
            ]
        )

    log_stage(
        f"text {args.operation} L{args.layer} K{args.k}: training DAS "
        f"(epochs={args.epochs}, train_pairs={len(train_pairs)}, "
        f"validation_pairs={len(validation_pairs)}, batch_size={args.batch_size})"
    )
    das, best_epoch, history = train_subspace(
        model=model,
        tokenizer=tokenizer,
        blocks=blocks,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        layer=args.layer,
        hook_name=args.hook,
        position=args.position,
        target=args.target,
        dimension=args.k,
        hidden_size=hidden_size,
        use_chat_template=use_chat_template,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed + args.layer * 100 + args.k,
        initialization="random_pca",
        initialization_space=pca_space,
    )
    log_stage(
        f"text {args.operation} L{args.layer} K{args.k}: training finished "
        f"(best_epoch={best_epoch})"
    )
    das_metrics, das_ar = evaluate_subspace("DAS PCA init", das)

    conditions.append(
        {
            "condition": "das_pca_initialized",
            "metrics": das_metrics,
            "autoregressive_iia": das_ar,
            "best_epoch": best_epoch,
            "subspace": {
                "basis": das[str(args.layer)].basis().detach().cpu(),
                "history": history,
                "kind": "das_pca_initialized",
            },
        }
    )
    if args.run_random_init and not args.das_only:
        random_init_das, random_best_epoch, random_history = train_subspace(
            model=model,
            tokenizer=tokenizer,
            blocks=blocks,
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            layer=args.layer,
            hook_name=args.hook,
            position=args.position,
            target=args.target,
            dimension=args.k,
            hidden_size=hidden_size,
            use_chat_template=use_chat_template,
            epochs=args.random_init_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=init_seed,
            initialization="random",
            initialization_space=None,
        )
        random_init_metrics, random_init_ar = evaluate_subspace(
            "DAS random init", random_init_das
        )
        conditions.append(
            {
                "condition": "das_random_initialized",
                "metrics": random_init_metrics,
                "autoregressive_iia": random_init_ar,
                "best_epoch": random_best_epoch,
                "subspace": {
                    "basis": random_init_das[str(args.layer)].basis().detach().cpu(),
                    "history": random_history,
                    "kind": "das_random_initialized",
                },
            }
        )
    return model_name, full_metrics, clean_ar, full_ar, conditions


def evaluate_image(args, data_path, train_pairs, validation_pairs, test_pairs):
    log_stage(
        f"image {args.operation} L{args.layer} K{args.k}: resolving model path"
    )
    model_path, model_name = resolve_model_for_loading(args.model)
    log_stage(f"image {args.operation} L{args.layer} K{args.k}: loading model")
    model, processor, tokenizer = load_hf_model_and_processor(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    blocks = get_blocks(model)
    validate_block_layers(model, [args.layer])
    hidden_size = get_hidden_size(model)
    data_root = args.data_root or data_path.parent
    log_stage(
        f"image {args.operation} L{args.layer} K{args.k}: model ready "
        f"(hidden_size={hidden_size})"
    )

    first = test_pairs[0]["base"]
    print_position_summary(
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        text=sample_prompt(processor, first, args.prompt, args.enable_thinking),
        image=load_rgb_image(image_path_for(first, data_root)),
        position=args.position,
    )
    ar_pairs = (
        test_pairs[: args.max_autoregressive_pairs]
        if args.max_autoregressive_pairs > 0
        else test_pairs
    )
    full_metrics = None
    clean_ar = None
    full_ar = None
    if not args.das_only:
        log_stage(
            f"image {args.operation} L{args.layer} K{args.k}: evaluating full patch controls"
        )
        full_patch = full_patch_subspace(args.layer)
        full_metrics = evaluate_teacher_forced_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=test_pairs,
            data_root=data_root,
            prompt=args.prompt,
            position_strategy=args.position,
            target=args.target,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            include_clean=True,
        )
        clean_ar = clean_autoregressive_iia_image(
            model,
            processor,
            tokenizer,
            ar_pairs,
            data_root,
            args.prompt,
            args.target,
            args.enable_thinking,
            args.max_new_tokens,
        )
        full_ar = autoregressive_iia_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=full_patch,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=ar_pairs,
            data_root=data_root,
            prompt=args.prompt,
            position_strategy=args.position,
            target=args.target,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            description="image full patch autoregressive",
        )

    pca_samples = unique_samples_from_pairs(train_pairs)
    log_stage(
        f"image {args.operation} L{args.layer} K{args.k}: collecting PCA features "
        f"(samples<= {args.pca_max_samples}, unique_train_samples={len(pca_samples)})"
    )
    features = collect_initialization_features_image(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        samples=pca_samples,
        layers=[args.layer],
        hook_name=args.hook,
        data_root=data_root,
        prompt=args.prompt,
        position_strategy=args.position,
        enable_thinking=args.enable_thinking,
        batch_size=args.batch_size,
        max_samples=args.pca_max_samples,
    )[args.layer]
    log_stage(
        f"image {args.operation} L{args.layer} K{args.k}: PCA features collected "
        f"(n={features.shape[0]}, d={features.shape[1]})"
    )
    pca_space = pca_space_with_at_least_k(
        features, args.k, args.pca_variance_threshold
    )

    def evaluate_subspace(name, subspace):
        log_stage(
            f"image {args.operation} L{args.layer} K{args.k}: evaluating {name} "
            f"(test_pairs={len(test_pairs)}, ar_pairs={len(ar_pairs)})"
        )
        metrics = evaluate_teacher_forced_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspace,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=test_pairs,
            data_root=data_root,
            prompt=args.prompt,
            position_strategy=args.position,
            target=args.target,
            enable_thinking=args.enable_thinking,
            batch_size=args.batch_size,
            include_clean=False,
        )
        ar = autoregressive_iia_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            subspaces=subspace,
            layers=[args.layer],
            hook_name=args.hook,
            pairs=ar_pairs,
            data_root=data_root,
            prompt=args.prompt,
            position_strategy=args.position,
            target=args.target,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
            description=f"image {name} autoregressive",
        )
        return metrics, ar

    init_seed = args.seed + args.layer * 100 + args.k
    conditions = []
    if not args.das_only:
        initial_basis = pca_basis(pca_space, args.k, init_seed)
        untrained = subspace_from_basis(args.layer, hidden_size, initial_basis, model.device)
        untrained_metrics, untrained_ar = evaluate_subspace(
            "untrained PCA init", untrained
        )

        random_basis = pca_basis(pca_space, args.k, init_seed + 1)
        random_pca = subspace_from_basis(args.layer, hidden_size, random_basis, model.device)
        random_metrics, random_ar = evaluate_subspace(
            "random PCA subspace", random_pca
        )
        conditions.extend(
            [
                {
                    "condition": "full_activation_patching",
                    "metrics": full_metrics,
                    "autoregressive_iia": full_ar,
                    "best_epoch": None,
                    "subspace": {"kind": "full_activation_patch"},
                },
                {
                    "condition": "untrained_pca_initialized",
                    "metrics": untrained_metrics,
                    "autoregressive_iia": untrained_ar,
                    "best_epoch": 0,
                    "subspace": {
                        "basis": untrained[str(args.layer)].basis().detach().cpu(),
                        "kind": "untrained_pca_initialized",
                    },
                },
                {
                    "condition": "random_subspace_in_pca_span",
                    "metrics": random_metrics,
                    "autoregressive_iia": random_ar,
                    "best_epoch": 0,
                    "subspace": {
                        "basis": random_pca[str(args.layer)].basis().detach().cpu(),
                        "kind": "random_subspace_in_pca_span",
                    },
                },
            ]
        )

    log_stage(
        f"image {args.operation} L{args.layer} K{args.k}: training DAS "
        f"(epochs={args.epochs}, train_pairs={len(train_pairs)}, "
        f"validation_pairs={len(validation_pairs)}, batch_size={args.batch_size})"
    )
    das, best_epoch, history = train_subspace_image(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        blocks=blocks,
        train_pairs=train_pairs,
        validation_pairs=validation_pairs,
        layer=args.layer,
        hook_name=args.hook,
        data_root=data_root,
        prompt=args.prompt,
        position_strategy=args.position,
        target=args.target,
        dimension=args.k,
        hidden_size=hidden_size,
        enable_thinking=args.enable_thinking,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed + args.layer * 100 + args.k,
        initialization="random_pca",
        initialization_space=pca_space,
    )
    log_stage(
        f"image {args.operation} L{args.layer} K{args.k}: training finished "
        f"(best_epoch={best_epoch})"
    )
    das_metrics, das_ar = evaluate_subspace("DAS PCA init", das)

    conditions.append(
        {
            "condition": "das_pca_initialized",
            "metrics": das_metrics,
            "autoregressive_iia": das_ar,
            "best_epoch": best_epoch,
            "subspace": {
                "basis": das[str(args.layer)].basis().detach().cpu(),
                "history": history,
                "kind": "das_pca_initialized",
            },
        }
    )
    if args.run_random_init and not args.das_only:
        random_init_das, random_best_epoch, random_history = train_subspace_image(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            blocks=blocks,
            train_pairs=train_pairs,
            validation_pairs=validation_pairs,
            layer=args.layer,
            hook_name=args.hook,
            data_root=data_root,
            prompt=args.prompt,
            position_strategy=args.position,
            target=args.target,
            dimension=args.k,
            hidden_size=hidden_size,
            enable_thinking=args.enable_thinking,
            epochs=args.random_init_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=init_seed,
            initialization="random",
            initialization_space=None,
        )
        random_init_metrics, random_init_ar = evaluate_subspace(
            "DAS random init", random_init_das
        )
        conditions.append(
            {
                "condition": "das_random_initialized",
                "metrics": random_init_metrics,
                "autoregressive_iia": random_init_ar,
                "best_epoch": random_best_epoch,
                "subspace": {
                    "basis": random_init_das[str(args.layer)].basis().detach().cpu(),
                    "history": random_history,
                    "kind": "das_random_initialized",
                },
            }
        )
    return model_name, full_metrics, clean_ar, full_ar, conditions


def main():
    args = parse_args()
    data_path = resolve_data_path(args)
    samples = load_jsonl(data_path)
    train_pairs, validation_pairs, test_pairs, pair_stats = make_pairs(samples, args)
    dataset_stats = dataset_statistics(samples)
    log_stage(
        f"{args.modality} {args.operation} L{args.layer} K{args.k}: data ready "
        f"(samples={len(samples)}, train_pairs={len(train_pairs)}, "
        f"validation_pairs={len(validation_pairs)}, test_pairs={len(test_pairs)}, "
        f"batch_size={args.batch_size})"
    )

    if args.modality == "text":
        model_name, full_metrics, clean_ar, full_ar, conditions = evaluate_text(
            args, train_pairs, validation_pairs, test_pairs
        )
    else:
        model_name, full_metrics, clean_ar, full_ar, conditions = evaluate_image(
            args, data_path, train_pairs, validation_pairs, test_pairs
        )

    if full_metrics is None:
        clean = {"variable": None, "full_answer": None, "autoregressive": None}
        full = {"variable": None, "full_answer": None, "autoregressive": None}
    else:
        clean = {
            "variable": full_metrics["clean_counterfactual_variable_teacher_forced_iia"],
            "full_answer": full_metrics["clean_counterfactual_full_answer_teacher_forced_iia"],
            "autoregressive": clean_ar,
        }
        full = {
            "variable": full_metrics["variable_teacher_forced_iia"],
            "full_answer": full_metrics["full_answer_teacher_forced_iia"],
            "autoregressive": full_ar,
        }
    rows = []
    for item in conditions:
        row = summarize_condition(
            args,
            item["condition"],
            item["metrics"],
            clean_ar,
            item["autoregressive_iia"],
            clean,
            full,
        )
        row.update(
            {
                "model": model_name,
                "data_path": str(data_path),
                "target": args.target,
                "best_epoch": item["best_epoch"],
                "pair_statistics": pair_stats,
                "dataset_statistics": dataset_stats,
            }
        )
        subspace_payload = {
            **item["subspace"],
            "condition": item["condition"],
            "best_epoch": item["best_epoch"],
            "config": vars(args),
        }
        save_condition(args, item["condition"], row, subspace_payload, test_pairs)
        rows.append(row)

    summary_path = task_dir(args) / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, summary_path)
    print(json.dumps(rows, indent=2))
    print(f"Saved condition folders under: {summary_path.parent}")


if __name__ == "__main__":
    main()
