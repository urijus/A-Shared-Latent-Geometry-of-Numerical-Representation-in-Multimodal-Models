"""Small, reusable building blocks for the baseline DAS experiments.

The causal variables used by the multiplication experiments are deliberately
explicit:

* ``c1_hat`` changes ``r1 = c1_hat // 10`` while holding ``result % 100``.
  It therefore tests control of the hundreds digit, not all of ``c1_hat``.
* ``c0_hat`` changes the raw ones-product ``a0*b0`` while holding
  ``p1 = a1*b0`` fixed. This changes both ``c0`` and the carry ``r0``.
* ``c0`` changes the ones digit while holding ``c1_hat`` fixed.
* ``combined`` changes both ``r1`` and ``c0`` while holding ``c1`` fixed.

All reported IIA values are either explicitly teacher-forced or explicitly
autoregressive. PCA initialization is supported as a training aid without
introducing paper-specific dataset assumptions.
"""

import copy
import math
import random
import re
from collections import Counter, defaultdict
from contextlib import ExitStack

import torch
import torch.nn as nn
from tqdm import tqdm

from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    apply_chat_template,
    build_probe_context,
    select_probe_positions,
)


TARGET_COMPONENTS = {
    "result": "full_result",
    "c1_hat": "r1_equals_c1_hat_div_10",
    "c1_hat_full": "full_c1_hat_equals_result_div_10",
    "c0_hat": "raw_c0_hat_equals_a0_times_b0",
    "c0": "c0_equals_result_ones",
    "combined": "c0_and_r1_with_c1_fixed",
}


class DASSubspace(nn.Module):
    """An orthonormal low-rank residual-stream subspace."""

    def __init__(self, hidden_size, dimension, initial_basis=None):
        super().__init__()
        basis = (
            torch.randn(hidden_size, dimension)
            if initial_basis is None
            else initial_basis.detach().float().clone()
        )
        if tuple(basis.shape) != (hidden_size, dimension):
            raise ValueError(
                f"Basis shape {tuple(basis.shape)} does not match "
                f"{(hidden_size, dimension)}."
            )
        self.raw_basis = nn.Parameter(basis)

    def basis(self):
        return torch.linalg.qr(self.raw_basis, mode="reduced").Q

    def patch(self, base, source):
        basis = self.basis().to(base.dtype)
        return base + ((source - base) @ basis) @ basis.T


def sample_id(sample):
    return sample.get("sample_id", (int(sample["a"]), int(sample["b"])))


def target_answers(base, source, target):
    """Return base answer, counterfactual answer, and their character spans."""
    base_answer = str(int(base["result"]))
    source_answer = str(int(source["result"]))

    if target in {"result", "combined", "c0_hat"}:
        return (
            base_answer,
            source_answer,
            (0, len(base_answer)),
            (0, len(source_answer)),
        )
    if target == "c1_hat":
        # Pair construction holds the final two digits fixed and changes r1.
        return base_answer, source_answer, (0, len(base_answer) - 2), (
            0,
            len(source_answer) - 2,
        )
    if target == "c1_hat_full":
        base_answer = str(int(base["result"]))

        # Build counterfactual answer
        cf_result = 10 * int(source["c1_hat"]) + int(base["c0"])
        cf_answer = str(cf_result)

        # Score all digits except the final c0 digit.
        # This prefix is exactly source_c1_hat.
        source_span = (0, len(cf_answer) - 1)

        # For the base answer, the corresponding variable is base_c1_hat.
        base_span = (0, len(base_answer) - 1)

        return base_answer, cf_answer, base_span, source_span

    if target == "c0":
        return (
            base_answer,
            source_answer,
            (len(base_answer) - 1, len(base_answer)),
            (len(source_answer) - 1, len(source_answer)),
        )
    raise ValueError(f"Unsupported DAS target: {target}")


def valid_pair(base, source, target):
    """Define the intervention pairs for one causal variable."""
    if sample_id(base) == sample_id(source):
        return False

    if target == "c1_hat_full":
        base_answer = str(int(base["result"]))
        cf_answer = str(10 * int(source["c1_hat"]) + int(base["c0"]))
        return (
            sample_id(base) != sample_id(source)
            and len(base_answer) == len(cf_answer)
            and int(base["c1_hat"]) != int(source["c1_hat"])
            and int(source["c1_hat"]) > 0
        )

    if len(str(int(base["result"]))) != len(str(int(source["result"]))):
        return False

    if target == "result":
        return int(base["result"]) != int(source["result"])
    if target == "c1_hat":
        return (
            int(base["result_mod_100"]) == int(source["result_mod_100"])
            and int(base["c1_hat"]) != int(source["c1_hat"])
            and int(base["result"]) >= 100
            and int(source["result"]) >= 100
        )

    if target == "c0":
        return (
            int(base["c1_hat"]) == int(source["c1_hat"])
            and int(base["c0"]) != int(source["c0"])
        )
    if target == "c0_hat":
        return (
            int(base["p1"]) == int(source["p1"])
            and int(base["c0_hat"]) != int(source["c0_hat"])
        )
    if target == "combined":
        return (
            int(base["c1"]) == int(source["c1"])
            and int(base["c0"]) != int(source["c0"])
            and int(base["r1"]) != int(source["r1"])
        )
    raise ValueError(f"Unsupported DAS target: {target}")


def _pair_groups(samples, target):
    groups = defaultdict(list)
    for sample in samples:
        if target == "c1_hat":
            key = int(sample["result_mod_100"])
        elif target == "c1_hat_full":
            key = 0
        elif target == "c0":
            key = int(sample["c1_hat"])
        elif target == "c0_hat":
            key = int(sample["p1"])
        elif target == "combined":
            key = int(sample["c1"])
        else:
            key = len(str(int(sample["result"])))
        groups[key].append(sample)
    return groups


def validate_samples(samples, target):
    if target not in {"c1_hat", "c1_hat_full", "c0_hat", "c0", "combined"}:
        return
    required = {
        "result",
        "result_mod_100",
        "p1",
        "c0_hat",
        "c0",
        "r0",
        "c1",
        "r1",
        "c1_hat",
    }
    for index, sample in enumerate(samples):
        missing = required - sample.keys()
        if missing:
            raise ValueError(f"Multiplication sample {index} is missing {sorted(missing)}.")
        c1_hat = int(sample["c1_hat"])
        result = int(sample["result"])
        if result != int(sample["c0"]) + 10 * c1_hat:
            raise ValueError(f"Sample {index} violates result = c0 + 10*c1_hat.")
        if int(sample["c1"]) != c1_hat % 10 or int(sample["r1"]) != c1_hat // 10:
            raise ValueError(f"Sample {index} has inconsistent c1/r1 fields.")
        if int(sample["result_mod_100"]) != result % 100:
            raise ValueError(f"Sample {index} has inconsistent result_mod_100.")


def build_unique_pairs(samples, target, seed, max_pairs=0):
    """Uniformly sample valid ordered pairs using bounded memory when capped."""
    validate_samples(samples, target)
    rng = random.Random(seed)
    pairs = []
    available = 0
    for group in _pair_groups(samples, target).values():
        for base in group:
            for source in group:
                if valid_pair(base, source, target):
                    available += 1
                    pair = {"base": base, "source": source}
                    if max_pairs <= 0 or len(pairs) < max_pairs:
                        pairs.append(pair)
                    else:
                        # Reservoir sampling keeps an exactly uniform sample
                        # without materializing every valid ordered pair.
                        replacement = rng.randrange(available)
                        if replacement < max_pairs:
                            pairs[replacement] = pair

    rng.shuffle(pairs)
    eligible_ids = {
        sample_id(item)
        for pair in pairs
        for item in (pair["base"], pair["source"])
    }
    stats = {
        "available_unique_pairs": available,
        "selected_unique_pairs": len(pairs),
        "eligible_samples_in_selected_pairs": len(eligible_ids),
    }
    return pairs, stats


def split_samples(samples, train_fraction, validation_fraction, seed):
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be below one.")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    parts = (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )
    if any(not part for part in parts):
        raise ValueError(f"Sample split is too small: {[len(part) for part in parts]}.")
    return parts


def dataset_statistics(samples):
    result_lengths = Counter(len(str(int(row["result"]))) for row in samples)
    stats = {
        "n_samples": len(samples),
        "result_length_counts": dict(sorted(result_lengths.items())),
    }
    if samples:
        results = [int(row["result"]) for row in samples]
        stats["result_min"] = min(results)
        stats["result_max"] = max(results)
    if samples and all("c0" in row for row in samples):
        for field in ("c0", "c1", "r1", "c0_hat", "c1_hat", "result_mod_100"):
            values = [int(row[field]) for row in samples]
            stats[f"{field}_n_values"] = len(set(values))
            stats[f"{field}_min"] = min(values)
            stats[f"{field}_max"] = max(values)
    return stats


def format_prompt(tokenizer, sample, use_chat_template):
    expression = str(sample["expr"])
    return (
        apply_chat_template(tokenizer, expression)
        if use_chat_template
        else expression
    )


def resolve_position(tokenizer, prompt, position):
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Cannot select a position of an empty prompt.")
    try:
        numeric_position = int(position)
    except ValueError:
        if str(position).lower() in {"last", "last_token"}:
            return len(ids) - 1
        raise
    if numeric_position < 0:
        raw_position = len(ids) + numeric_position
        if raw_position < 0 or raw_position >= len(ids):
            raise ValueError(
                f"Raw token offset {position!r} is outside the prompt token range."
            )
        return raw_position
    context = build_probe_context(tokenizer, prompt)
    return select_probe_positions(context, [str(position)])[0][0]


def print_patch_diagnostics(tokenizer, samples, use_chat_template, position):
    """Print representative prompts and the exact HF token being patched."""
    print(f"Use chat template: {use_chat_template}")
    print(f"Requested patch position: {position!r}")
    seen = set()
    for label, sample in samples:
        identity = sample_id(sample)
        if identity in seen:
            continue
        seen.add(identity)
        prompt = format_prompt(tokenizer, sample, use_chat_template)
        context = build_probe_context(tokenizer, prompt)
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        hf_position = resolve_position(tokenizer, prompt, position)
        hf_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        semantic = [
            f"{index}:{token!r}->hf{hf_position_}"
            for index, (token, hf_position_) in enumerate(
                zip(context["tokens"], context["positions"])
            )
        ]
        rendered_hf = [
            f"{index}:{token!r}{' <-- PATCH' if index == hf_position else ''}"
            for index, token in enumerate(hf_tokens)
        ]
        print(f"\nPatch diagnostic ({label}, sample_id={identity}):")
        print(f"  expression: {sample['expr']!r}")
        print(f"  rendered prompt: {prompt!r}")
        print(f"  semantic tokens: {' | '.join(semantic)}")
        print(f"  HF tokens: {' | '.join(rendered_hf)}")
        print(
            f"  resolved HF position={hf_position}, token_id={input_ids[hf_position]}, "
            f"token={hf_tokens[hf_position]!r}, final_HF_position={len(input_ids) - 1}, "
            f"is_final={hf_position == len(input_ids) - 1}"
        )


@torch.no_grad()
def collect_initialization_features(
    model,
    tokenizer,
    blocks,
    samples,
    layers,
    hook_name,
    position,
    use_chat_template,
    batch_size,
    max_samples,
):
    """Collect train-only residual features for PCA initialization."""
    selected = list(samples[:max_samples]) if max_samples else list(samples)
    if len(selected) < 2:
        raise ValueError("PCA initialization requires at least two samples.")
    features = {layer: [] for layer in layers}
    for start in tqdm(
        range(0, len(selected), batch_size), desc="Collecting PCA features"
    ):
        batch = selected[start : start + batch_size]
        prompts = [format_prompt(tokenizer, sample, use_chat_template) for sample in batch]
        positions = [resolve_position(tokenizer, prompt, position) for prompt in prompts]
        encoding = tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)
        captured = {}
        with ExitStack() as stack:
            for layer in layers:
                module, pre_hook = hook_module(blocks[layer - 1], hook_name)
                if pre_hook:
                    def capture_pre(_module, inputs, layer=layer):
                        captured[layer] = hidden(inputs[0]).detach()

                    handle = module.register_forward_pre_hook(capture_pre)
                else:
                    def capture_post(_module, _inputs, output, layer=layer):
                        captured[layer] = hidden(output).detach()

                    handle = module.register_forward_hook(capture_post)
                stack.callback(handle.remove)
            model(**encoding, use_cache=False)

        rows = torch.arange(len(batch), device=model.device)
        token_positions = torch.tensor(positions, device=model.device)
        for layer in layers:
            features[layer].append(
                captured[layer][rows, token_positions].float().cpu()
            )
    return {layer: torch.cat(parts) for layer, parts in features.items()}


def pca_principal_space(features, variance_threshold=0.9):
    """Fit PCA once and retain the space reaching the variance threshold."""
    if not torch.isfinite(features).all():
        bad = int((~torch.isfinite(features)).sum())
        raise ValueError(f"PCA features contain {bad} non-finite values.")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    max_components = min(centered.shape[0] - 1, centered.shape[1], 500)
    if max_components < 1:
        raise ValueError("PCA initialization requires at least two feature rows.")
    _, singular_values, components = torch.pca_lowrank(
        centered, q=max_components, center=False, niter=2
    )
    total_variance = centered.square().sum().clamp_min(1e-12)
    cumulative = singular_values.square().cumsum(0) / total_variance
    cutoff = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(variance_threshold, dtype=cumulative.dtype),
        ).item()
    ) + 1
    cutoff = min(cutoff, components.shape[1])
    principal_space = components[:, :cutoff]
    if not torch.isfinite(principal_space).all():
        raise ValueError("PCA produced a non-finite principal space.")
    print(
        f"PCA initialization: {cutoff} PCs explain "
        f"{float(cumulative[cutoff - 1]):.3%} of train activation variance"
    )
    return principal_space


def random_subspace_from_pca(principal_space, dimension):
    """Sample an orthonormal basis from a previously fitted PCA space."""
    cutoff = principal_space.shape[1]
    if dimension <= cutoff:
        rotation, _ = torch.linalg.qr(torch.randn(cutoff, dimension))
        return principal_space @ rotation

    extra = torch.randn(principal_space.shape[0], dimension - cutoff)
    extra -= principal_space @ (principal_space.T @ extra)
    extra, _ = torch.linalg.qr(extra, mode="reduced")
    return torch.cat([principal_space, extra], dim=1)


def random_subspace_in_pca(features, dimension, variance_threshold=0.9):
    """Fit PCA and sample a basis; useful for standalone callers and tests."""
    principal_space = pca_principal_space(features, variance_threshold)
    return random_subspace_from_pca(principal_space, dimension)


def hook_module(block, hook_name):
    if hook_name == "resid_post":
        return block, False
    if hook_name == "resid_pre":
        return block, True
    raise ValueError("The simplified DAS supports resid_pre and resid_post only.")


def hidden(value):
    return value[0] if isinstance(value, tuple) else value


def replace_hidden(value, new_hidden):
    return (new_hidden, *value[1:]) if isinstance(value, tuple) else new_hidden


def patched_forward(
    model,
    encoding,
    blocks,
    subspaces,
    layers,
    hook_name,
    base_positions,
    source_positions,
    n_base_groups,
):
    """Patch donor projections into each base group in a joint forward pass."""
    batch_size = len(base_positions)

    def patch_tensor(value, layer):
        activations = hidden(value)
        updated = activations.clone()
        donor_offset = n_base_groups * batch_size
        for group in range(n_base_groups):
            for row, (base_position, source_position) in enumerate(
                zip(base_positions, source_positions)
            ):
                base_row = group * batch_size + row
                donor_row = donor_offset + row
                updated[base_row, base_position] = subspaces[str(layer)].patch(
                    activations[base_row, base_position],
                    activations[donor_row, source_position],
                )
        return replace_hidden(value, updated)

    with ExitStack() as stack:
        for layer in layers:
            module, pre_hook = hook_module(blocks[layer - 1], hook_name)
            if pre_hook:
                def hook(_module, inputs, layer=layer):
                    return (patch_tensor(inputs[0], layer), *inputs[1:])

                handle = module.register_forward_pre_hook(hook)
            else:
                def hook(_module, _inputs, output, layer=layer):
                    return patch_tensor(output, layer)

                handle = module.register_forward_hook(hook)
            stack.callback(handle.remove)
        return model(**encoding, use_cache=False)


def tokenize_answers(tokenizer, prompts, answers, spans, device):
    texts = [prompt + answer for prompt, answer in zip(prompts, answers)]
    encoding = tokenizer(
        texts,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    offsets = encoding.pop("offset_mapping")
    full_positions, variable_positions = [], []

    for prompt, answer, span, row_offsets, row_ids in zip(
        prompts, answers, spans, offsets, encoding["input_ids"]
    ):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if row_ids[: len(prompt_ids)].tolist() != prompt_ids:
            raise ValueError(
                "Tokenization of prompt+answer is not prefix-stable. Add an "
                "explicit answer separator before using this prompt format."
            )
        answer_start = len(prompt)
        answer_end = answer_start + len(answer)
        variable_start = answer_start + span[0]
        variable_end = answer_start + span[1]
        full, variable = [], []
        for token_index, (start, end) in enumerate(row_offsets.tolist()):
            if start < variable_start < end or start < variable_end < end:
                raise ValueError(
                    "A tokenizer token crosses the requested variable boundary."
                )
            if start < answer_end and end > answer_start:
                full.append(token_index)
            if start < variable_end and end > variable_start:
                variable.append(token_index)
        if not full or not variable:
            raise ValueError(f"Could not locate scored answer tokens in {texts!r}.")
        full_positions.append(full)
        variable_positions.append(variable)

    return (
        {name: tensor.to(device) for name, tensor in encoding.items()},
        full_positions,
        variable_positions,
    )


def sequence_scores(logits, input_ids, positions):
    scores, exact = [], []
    for row, row_positions in enumerate(positions):
        indices = torch.tensor(row_positions, device=logits.device)
        targets = input_ids[row, indices]
        predictions = logits[row, indices - 1].float().log_softmax(dim=-1)
        scores.append(predictions.gather(1, targets[:, None]).sum())
        exact.append((predictions.argmax(dim=-1) == targets).all().float())
    return torch.stack(scores), torch.stack(exact)


def teacher_forced_batch(
    model,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    position,
    target,
    use_chat_template,
    patch=True,
):
    base_prompts, donor_prompts = [], []
    base_answers, source_answers = [], []
    base_spans, source_spans = [], []
    base_positions, donor_positions = [], []

    for pair in pairs:
        base, source = pair["base"], pair["source"]
        donor = pair.get("donor", source)
        base_prompt = format_prompt(tokenizer, base, use_chat_template)
        donor_prompt = format_prompt(tokenizer, donor, use_chat_template)
        base_answer, source_answer, base_span, source_span = target_answers(
            base, source, target
        )
        base_prompts.append(base_prompt)
        donor_prompts.append(donor_prompt)
        base_answers.append(base_answer)
        source_answers.append(source_answer)
        base_spans.append(base_span)
        source_spans.append(source_span)
        base_positions.append(resolve_position(tokenizer, base_prompt, position))
        donor_positions.append(resolve_position(tokenizer, donor_prompt, position))

    prompts = base_prompts + base_prompts + donor_prompts
    answers = source_answers + base_answers + source_answers
    spans = source_spans + base_spans + source_spans
    encoding, full_positions, variable_positions = tokenize_answers(
        tokenizer, prompts, answers, spans, model.device
    )
    outputs = (
        patched_forward(
            model,
            encoding,
            blocks,
            subspaces,
            layers,
            hook_name,
            base_positions,
            donor_positions,
            n_base_groups=2,
        )
        if patch
        else model(**encoding, use_cache=False)
    )

    n = len(pairs)
    source_variable_score, variable_exact = sequence_scores(
        outputs.logits[:n], encoding["input_ids"][:n], variable_positions[:n]
    )
    source_full_score, full_exact = sequence_scores(
        outputs.logits[:n], encoding["input_ids"][:n], full_positions[:n]
    )
    base_variable_score, _ = sequence_scores(
        outputs.logits[n : 2 * n],
        encoding["input_ids"][n : 2 * n],
        variable_positions[n : 2 * n],
    )
    base_full_score, _ = sequence_scores(
        outputs.logits[n : 2 * n],
        encoding["input_ids"][n : 2 * n],
        full_positions[n : 2 * n],
    )
    return {
        "loss": -source_variable_score.mean(),
        "source_variable_logprob": source_variable_score,
        "source_answer_logprob": source_full_score,
        "base_variable_logprob": base_variable_score,
        "base_answer_logprob": base_full_score,
        "variable_teacher_forced_iia": variable_exact,
        "full_answer_teacher_forced_iia": full_exact,
    }


def _mean(parts):
    return float(torch.cat(parts).mean())


@torch.no_grad()
def evaluate_teacher_forced(
    model,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    position,
    target,
    use_chat_template,
    batch_size,
    include_clean=True,
):
    names = (
        "source_variable_logprob",
        "source_answer_logprob",
        "base_variable_logprob",
        "base_answer_logprob",
        "variable_teacher_forced_iia",
        "full_answer_teacher_forced_iia",
    )
    patched = {name: [] for name in names}
    clean = {name: [] for name in names}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        result = teacher_forced_batch(
            model,
            tokenizer,
            blocks,
            subspaces,
            layers,
            hook_name,
            batch,
            position,
            target,
            use_chat_template,
            patch=True,
        )
        for name in names:
            patched[name].append(result[name].detach().cpu())
        if include_clean:
            baseline = teacher_forced_batch(
                model,
                tokenizer,
                blocks,
                subspaces,
                layers,
                hook_name,
                batch,
                position,
                target,
                use_chat_template,
                patch=False,
            )
            for name in names:
                clean[name].append(baseline[name].detach().cpu())

    metrics = {name: _mean(parts) for name, parts in patched.items()}
    metrics["loss"] = -metrics["source_variable_logprob"]
    if include_clean:
        clean_metrics = {name: _mean(parts) for name, parts in clean.items()}
        metrics.update(
            {
                "source_variable_logprob_gain": metrics["source_variable_logprob"]
                - clean_metrics["source_variable_logprob"],
                "source_answer_logprob_gain": metrics["source_answer_logprob"]
                - clean_metrics["source_answer_logprob"],
                "base_variable_logprob_change": metrics["base_variable_logprob"]
                - clean_metrics["base_variable_logprob"],
                "base_answer_logprob_change": metrics["base_answer_logprob"]
                - clean_metrics["base_answer_logprob"],
                "clean_counterfactual_variable_teacher_forced_iia": clean_metrics[
                    "variable_teacher_forced_iia"
                ],
                "clean_counterfactual_full_answer_teacher_forced_iia": clean_metrics[
                    "full_answer_teacher_forced_iia"
                ],
            }
        )
    return metrics


@torch.no_grad()
def autoregressive_iia(
    model,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    position,
    target,
    use_chat_template,
    max_new_tokens,
    description="Autoregressive IIA",
):
    correct = 0
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for pair in tqdm(pairs, desc=description):
            base, source = pair["base"], pair["source"]
            donor = pair.get("donor", source)
            base_prompt = format_prompt(tokenizer, base, use_chat_template)
            donor_prompt = format_prompt(tokenizer, donor, use_chat_template)
            expected = target_answers(base, source, target)[1]
            base_position = resolve_position(tokenizer, base_prompt, position)
            donor_position = resolve_position(tokenizer, donor_prompt, position)
            base_ids = tokenizer(
                base_prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].to(model.device)
            donor_ids = tokenizer(
                donor_prompt, add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].to(model.device)
            generated = []

            for _ in range(max_new_tokens):
                base_length, donor_length = len(base_ids), len(donor_ids)
                length = max(base_length, donor_length)
                input_ids = torch.full(
                    (2, length),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=model.device,
                )
                attention_mask = torch.zeros_like(input_ids)
                input_ids[0, :base_length] = base_ids
                input_ids[1, :donor_length] = donor_ids
                attention_mask[0, :base_length] = 1
                attention_mask[1, :donor_length] = 1
                outputs = patched_forward(
                    model,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                    blocks,
                    subspaces,
                    layers,
                    hook_name,
                    [base_position],
                    [donor_position],
                    n_base_groups=1,
                )
                next_id = outputs.logits[0, base_length - 1].argmax().reshape(1)
                generated.append(int(next_id.item()))
                base_ids = torch.cat([base_ids, next_id])
                if next_id.item() == tokenizer.eos_token_id:
                    break

            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            match = re.match(r"-?\d+", text)
            correct += int(match is not None and match.group() == expected)
    finally:
        tokenizer.padding_side = old_padding_side
    return correct / len(pairs) if pairs else 0.0


def train_subspace(
    model,
    tokenizer,
    blocks,
    train_pairs,
    validation_pairs,
    layer,
    hook_name,
    position,
    target,
    dimension,
    hidden_size,
    use_chat_template,
    epochs,
    batch_size,
    learning_rate,
    patience,
    seed,
    initialization="random_pca",
    initialization_space=None,
    gradient_clip_norm=1.0,
):
    torch.manual_seed(seed)
    if initialization == "random_pca":
        if initialization_space is None:
            raise ValueError("random_pca requires a fitted PCA space.")
        initial_basis = random_subspace_from_pca(initialization_space, dimension)
    elif initialization == "random":
        initial_basis = None
    else:
        raise ValueError(f"Unknown initialization: {initialization}")
    subspaces = nn.ModuleDict(
        {
            str(layer): DASSubspace(
                hidden_size, dimension, initial_basis=initial_basis
            ).to(model.device)
        }
    )
    optimizer = torch.optim.Adam(subspaces.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    print(
        f"[stage] text DAS initial validation: layer={layer}, k={dimension}, "
        f"validation_pairs={len(validation_pairs)}, batch_size={batch_size}",
        flush=True,
    )
    initial_validation = evaluate_teacher_forced(
        model,
        tokenizer,
        blocks,
        subspaces,
        [layer],
        hook_name,
        validation_pairs,
        position,
        target,
        use_chat_template,
        batch_size,
        include_clean=False,
    )
    initial_loss = initial_validation["loss"]
    if not math.isfinite(initial_loss):
        raise RuntimeError(
            f"Non-finite validation loss before training for layer={layer}, "
            f"k={dimension}. Check the PCA features and patched position."
        )
    best_loss, best_epoch, stale = initial_loss, 0, 0
    best_state = copy.deepcopy(subspaces.state_dict())
    history = [{
        "epoch": 0,
        "train_loss": None,
        "validation_loss": initial_loss,
        "validation_variable_teacher_forced_iia": initial_validation[
            "variable_teacher_forced_iia"
        ],
    }]
    print(
        f"epoch=0 train_loss=n/a val_loss={initial_loss:.4f} "
        f"val_variable_iia="
        f"{initial_validation['variable_teacher_forced_iia']:.3f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        print(
            f"[stage] text DAS epoch {epoch}/{epochs}: "
            f"train_pairs={len(train_pairs)}, batch_size={batch_size}",
            flush=True,
        )
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        losses = []
        divergence_reason = None
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[index] for index in order[start : start + batch_size]]
            optimizer.zero_grad()
            result = teacher_forced_batch(
                model,
                tokenizer,
                blocks,
                subspaces,
                [layer],
                hook_name,
                batch,
                position,
                target,
                use_chat_template,
                patch=True,
            )
            if not torch.isfinite(result["loss"]):
                divergence_reason = (
                    f"Non-finite training loss at layer={layer}, k={dimension}, "
                    f"epoch={epoch}, batch_start={start}."
                )
                break
            result["loss"].backward()
            if gradient_clip_norm > 0:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    subspaces.parameters(), gradient_clip_norm, error_if_nonfinite=False
                )
                if not torch.isfinite(gradient_norm):
                    divergence_reason = (
                        f"Non-finite DAS gradient at layer={layer}, k={dimension}, "
                        f"epoch={epoch}, batch_start={start}. Try a lower learning rate."
                    )
                    break
            optimizer.step()
            # Keep the optimized matrix well-conditioned. The forward pass uses
            # this same QR basis; retracting prevents unstable QR gradients after
            # many Adam updates, especially for larger k.
            with torch.no_grad():
                for subspace in subspaces.values():
                    retracted = subspace.basis()
                    if not torch.isfinite(retracted).all():
                        divergence_reason = (
                            f"Non-finite DAS basis after optimizer step at "
                            f"layer={layer}, k={dimension}, epoch={epoch}."
                        )
                        break
                    subspace.raw_basis.copy_(retracted)
            if divergence_reason is not None:
                break
            losses.append(float(result["loss"].detach()))

        if divergence_reason is not None:
            print(f"{divergence_reason} Restoring best epoch {best_epoch}.", flush=True)
            break

        validation = evaluate_teacher_forced(
            model,
            tokenizer,
            blocks,
            subspaces,
            [layer],
            hook_name,
            validation_pairs,
            position,
            target,
            use_chat_template,
            batch_size,
            include_clean=False,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "validation_loss": validation["loss"],
            "validation_variable_teacher_forced_iia": validation[
                "variable_teacher_forced_iia"
            ],
        }
        history.append(epoch_row)
        print(
            f"epoch={epoch} train_loss={epoch_row['train_loss']:.4f} "
            f"val_loss={epoch_row['validation_loss']:.4f} "
            f"val_variable_iia="
            f"{epoch_row['validation_variable_teacher_forced_iia']:.3f}",
            flush=True,
        )
        if not math.isfinite(validation["loss"]):
            print(
                f"Stopping after non-finite validation loss at layer={layer}, "
                f"k={dimension}, epoch={epoch}; restoring epoch {best_epoch}.",
                flush=True,
            )
            break
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(subspaces.state_dict())
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break

    subspaces.load_state_dict(best_state)
    return subspaces, best_epoch, history


def random_subspaces(hidden_size, layers, dimension, device, seed):
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    modules = nn.ModuleDict(
        {
            str(layer): DASSubspace(hidden_size, dimension).to(device)
            for layer in layers
        }
    )
    torch.random.set_rng_state(state)
    return modules


def shuffled_donors(pairs, seed):
    donors = [pair["source"] for pair in pairs]
    rng = random.Random(seed)
    controlled = []
    for pair in pairs:
        excluded = {sample_id(pair["base"]), sample_id(pair["source"])}
        donor = None
        # Almost every donor is eligible, so rejection sampling avoids the
        # previous O(n^2) construction of a candidate list for every pair.
        for _ in range(20):
            candidate = rng.choice(donors)
            if sample_id(candidate) not in excluded:
                donor = candidate
                break
        if donor is None:
            donor = next(
                (
                    candidate
                    for candidate in donors
                    if sample_id(candidate) not in excluded
                ),
                pair["source"],
            )
        controlled.append(dict(pair, donor=donor))
    return controlled


def load_subspaces(path, layers, hidden_size, dimension, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload["bases"]
    modules = nn.ModuleDict()
    for layer in layers:
        basis = saved.get(layer, saved.get(str(layer)))
        if basis is None:
            raise ValueError(f"Checkpoint {path} has no basis for layer {layer}.")
        modules[str(layer)] = DASSubspace(
            hidden_size, dimension, initial_basis=basis
        ).to(device)
    return modules


def union_subspaces(first, second, layers, device):
    """Return the numerical union without adding directions from rank deficiency."""
    joined = nn.ModuleDict()
    for layer in layers:
        concatenated = torch.cat(
            [first[str(layer)].basis().detach(), second[str(layer)].basis().detach()],
            dim=1,
        ).float()
        left, singular_values, _ = torch.linalg.svd(
            concatenated, full_matrices=False
        )
        tolerance = (
            max(concatenated.shape)
            * torch.finfo(concatenated.dtype).eps
            * singular_values.max()
        )
        rank = int((singular_values > tolerance).sum().item())
        basis = left[:, :rank]
        joined[str(layer)] = DASSubspace(
            basis.shape[0], basis.shape[1], initial_basis=basis
        ).to(device)
    return joined
