import math
from pathlib import Path

import torch

from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    apply_chat_template,
    build_probe_context,
    select_probe_positions,
)
from src.models import get_blocks


def safe_name(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("*", "mul")
        .replace(":", "_")
        .replace("-", "minus")
    )


def probe_path(probe_root, modality, target, period, layer, position, method):
    folder = (
        Path(probe_root)
        / modality
        / "fourier_projections"
        / f"{target}_pos{position}_{method}"
        / "probes"
        / f"{target}_{modality}"
    )
    name = safe_name(
        f"{target}_T{period}_layer{layer}_pos{position}_{method}_probe.pt"
    )
    return folder / name


def load_probe_grid(probe_root, modality, target, periods, layers, positions, method):
    print("Loading probes...")
    paths = {}
    probes = {}
    for layer in layers:
        paths[layer] = {}
        probes[layer] = {}
        for position in positions:
            position_paths = [
                probe_path(
                    probe_root, modality, target, period, layer, position, method
                )
                for period in periods
            ]
            missing = [path for path in position_paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Fourier probe not found: {missing[0]}. "
                    "Run project_fourier with matching settings first."
                )
            paths[layer][position] = position_paths
            probes[layer][position] = [
                {
                    "weight": saved["weight"].float(),
                    "bias": saved["bias"].float(),
                }
                for saved in (
                    torch.load(path, map_location="cpu") for path in position_paths
                )
            ]
    return probes, paths


def steer_hidden(hidden, targets, probes, periods, alpha):
    """Apply the Fourier steering intervention from Eq. 4 of the paper."""
    original = hidden.float()
    steered = original.clone()
    targets = torch.as_tensor(targets, device=hidden.device, dtype=torch.float32)

    for period, probe in zip(periods, probes):
        weight = probe["weight"].to(hidden.device)  # [cos/sin, hidden]
        bias = probe["bias"].to(hidden.device)
        current = original @ weight.T + bias
        radius = current.norm(dim=-1).clamp_min(1e-8)
        theta = targets * (2 * math.pi / period)
        target_coordinates = alpha * radius[:, None] * torch.stack(
            (torch.cos(theta), torch.sin(theta)), dim=-1
        )
        # Minimum-norm hidden-state update whose probe-coordinate change is
        # exactly target_coordinates - current. Row-normalizing the weights is
        # insufficient when cosine/sine directions differ in scale or are not
        # orthogonal.
        decoder = torch.linalg.pinv(weight.T)  # [2, hidden]
        steered = steered + (target_coordinates - current) @ decoder

    return steered.to(hidden.dtype)


def resolve_positions(tokenizer, prompt, positions):
    context = build_probe_context(tokenizer, prompt)
    hf_positions, _ = select_probe_positions(context, positions)
    return hf_positions


def prompts_for_samples(tokenizer, samples, use_chat_template):
    if use_chat_template:
        return [apply_chat_template(tokenizer, sample["expr"]) for sample in samples]
    return [sample["expr"] for sample in samples]


@torch.no_grad()
def teacher_forced_logprobs(
    model,
    tokenizer,
    samples,
    answers,
    layers,
    positions,
    use_chat_template,
    probes,
    periods,
    steering_targets=None,
    alpha=0.0,
    suffix_chars=None,
    prefix_chars=None,
):
    """Sum answer-token log probabilities, optionally over a prefix or suffix."""
    if suffix_chars is not None and prefix_chars is not None:
        raise ValueError("Choose either suffix_chars or prefix_chars, not both.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompts = prompts_for_samples(tokenizer, samples, use_chat_template)
    answers = [str(answer) for answer in answers]
    texts = [prompt + answer for prompt, answer in zip(prompts, answers)]
    encoding = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoding.pop("offset_mapping")
    scored_positions = []

    for prompt, answer, token_offsets in zip(prompts, answers, offsets):
        answer_start = len(prompt)
        score_start = answer_start
        score_end = answer_start + len(answer)

        # If we use suffix_chars, we take the last two (or as many as we want)
        if suffix_chars is not None:
            score_start += max(0, len(answer) - suffix_chars)
        if prefix_chars is not None:
            score_end = answer_start + min(len(answer), prefix_chars)

        positions_for_answer = []
        for token_index, (start, end) in enumerate(token_offsets.tolist()):
            if suffix_chars is not None and start < score_start < end:
                raise ValueError(
                    "A tokenizer token crosses the requested suffix boundary; "
                    "the mod-100 score cannot be isolated for this tokenizer."
                )
            if prefix_chars is not None and start < score_end < end:
                raise ValueError(
                    "A tokenizer token crosses the requested prefix boundary; "
                    "the hundreds-digit score cannot be isolated for this tokenizer."
                )
            if start < score_end and end > score_start:
                positions_for_answer.append(token_index)
        if not positions_for_answer:
            raise ValueError(f"No answer tokens found in {prompt + answer!r}.")
        scored_positions.append(positions_for_answer)

    encoding = encoding.to(model.device)
    handles = []
    if steering_targets is not None and alpha != 0:
        steering_positions = torch.tensor(
            [resolve_positions(tokenizer, prompt, positions) for prompt in prompts],
            device=model.device,
        )
        targets = torch.as_tensor(
            steering_targets, device=model.device, dtype=torch.float32
        )
        batch_indices = torch.arange(len(samples), device=model.device)
        blocks = get_blocks(model)

        def make_hook(layer):
            def hook(_module, _inputs, output):
                hidden = output[0].clone() if isinstance(output, tuple) else output.clone()
                for position_index, position in enumerate(positions):
                    token_positions = steering_positions[:, position_index]
                    selected = hidden[batch_indices, token_positions]
                    hidden[batch_indices, token_positions] = steer_hidden(
                        selected,
                        targets,
                        probes[layer][position],
                        periods,
                        alpha,
                    )
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden

            return hook

        handles = [
            blocks[layer - 1].register_forward_hook(make_hook(layer))
            for layer in layers
        ]

    try:
        logits = model(**encoding, use_cache=False).logits.float()
    finally:
        for handle in handles:
            handle.remove()

    log_probs = logits.log_softmax(dim=-1)
    scores = []
    for sample_index, token_positions in enumerate(scored_positions):
        scores.append(
            torch.stack(
                [
                    log_probs[
                        sample_index,
                        token_position - 1,
                        encoding["input_ids"][sample_index, token_position],
                    ]
                    for token_position in token_positions
                ]
            ).sum()
        )
    return torch.stack(scores).cpu()

# Returns [number of prompts, number of candidates] (with raw sequence logprobs)
def score_candidates(model, tokenizer, samples, candidates, **kwargs):
    return torch.stack(
        [
            teacher_forced_logprobs(
                model,
                tokenizer,
                samples,
                [candidate] * len(samples), # the answers
                **kwargs,
            )
            for candidate in candidates
        ],
        dim=1,
    )


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]
