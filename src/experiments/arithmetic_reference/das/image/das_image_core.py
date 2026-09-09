"""Image-aware DAS helpers mirroring the baseline text DAS core."""

import copy
import math
import random
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from src.interventions.das import (
    DASSubspace,
    hidden,
    hook_module,
    pca_principal_space,
    random_subspace_from_pca,
    replace_hidden,
    target_answers,
)
from src.experiments.arithmetic_reference.linear_probes.image.extract_image_activations import (
    image_token_id_candidates,
    last_image_token_positions,
    render_prompt,
)


def image_path_for(sample, data_root):
    image_path = Path(sample["image_path"])
    return image_path if image_path.is_absolute() else data_root / image_path


def load_rgb_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def sample_prompt(processor, sample, prompt, enable_thinking):
    return render_prompt(processor, prompt, enable_thinking)


def make_inputs(processor, texts, images):
    if processor.__class__.__name__ == "PixtralProcessor":
        return processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
    return processor(
        text=texts,
        images=[[image] for image in images],
        return_tensors="pt",
        padding=True,
    )


def inputs_to_device(inputs, device, dtype=None):
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    pixel_values = inputs.get("pixel_values")
    if (
        dtype is not None
        and pixel_values is not None
        and torch.is_floating_point(pixel_values)
    ):
        inputs["pixel_values"] = pixel_values.to(dtype=dtype)
    return inputs


def input_lengths(processor, texts, images, device):
    inputs = inputs_to_device(make_inputs(processor, texts, images), device)
    if "attention_mask" in inputs:
        return [int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()]
    return [inputs["input_ids"].shape[1]] * len(texts)


def patched_forward(model, encoding, blocks, subspaces, layers, hook_name,
                    base_positions, source_positions, n_base_groups):
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


def resolve_position_spec(position):
    if isinstance(position, int):
        return position
    try:
        return int(position)
    except (TypeError, ValueError):
        return position


def resolve_batch_positions(processor, tokenizer, model, texts, images, position):
    inputs = make_inputs(processor, texts, images)
    spec = resolve_position_spec(position)
    if spec == "last_input":
        if "attention_mask" in inputs:
            return [int(value) - 1 for value in inputs["attention_mask"].sum(dim=1)]
        return [inputs["input_ids"].shape[1] - 1] * len(texts)
    if spec == "last_image_token":
        image_token_ids = image_token_id_candidates(model, processor, tokenizer)
        return last_image_token_positions(inputs, image_token_ids)
    if not isinstance(spec, int):
        raise ValueError(
            f"Image DAS position must be an integer, 'last_input', or "
            f"'last_image_token'; got {position!r}."
        )
    lengths = (
        inputs["attention_mask"].sum(dim=1).tolist()
        if "attention_mask" in inputs
        else [inputs["input_ids"].shape[1]] * len(texts)
    )
    positions = []
    for length in lengths:
        resolved = int(length) + spec if spec < 0 else spec
        if resolved < 0 or resolved >= int(length):
            raise ValueError(
                f"Position {position} resolves to {resolved}, outside "
                f"0..{int(length) - 1}."
            )
        positions.append(resolved)
    return positions


def print_position_summary(processor, tokenizer, model, text, image, position):
    inputs = make_inputs(processor, [text], [image])
    resolved = resolve_batch_positions(
        processor, tokenizer, model, [text], [image], position
    )[0]
    length = (
        int(inputs["attention_mask"][0].sum())
        if "attention_mask" in inputs
        else int(inputs["input_ids"].shape[1])
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0, :length].tolist())
    token = tokens[resolved] if 0 <= resolved < len(tokens) else "<out>"
    print("\nImage DAS position summary:")
    print(f"  requested position: {position}")
    print(f"  input length: {length}")
    print(f"  resolved raw token: {resolved}, token={token!r}")
    print()


def answer_token_positions(
    processor,
    prompt_texts,
    answers,
    spans,
    images,
    device,
    dtype=None,
):
    full_texts = [prompt + answer for prompt, answer in zip(prompt_texts, answers)]
    encoding = inputs_to_device(
        make_inputs(processor, full_texts, images),
        device,
        dtype=dtype,
    )

    full_end = input_lengths(processor, full_texts, images, device)
    variable_start_texts = [
        prompt + answer[: span[0]]
        for prompt, answer, span in zip(prompt_texts, answers, spans)
    ]
    variable_end_texts = [
        prompt + answer[: span[1]]
        for prompt, answer, span in zip(prompt_texts, answers, spans)
    ]
    prompt_end = input_lengths(processor, prompt_texts, images, device)
    variable_start = input_lengths(processor, variable_start_texts, images, device)
    variable_end = input_lengths(processor, variable_end_texts, images, device)

    full_positions = [
        list(range(start, end)) for start, end in zip(prompt_end, full_end)
    ]
    variable_positions = [
        list(range(start, end)) for start, end in zip(variable_start, variable_end)
    ]
    if any(not positions for positions in full_positions):
        raise ValueError("Could not locate full answer tokens.")
    if any(not positions for positions in variable_positions):
        raise ValueError("Could not locate variable answer tokens.")
    return encoding, full_positions, variable_positions


def sequence_scores(logits, input_ids, positions):
    scores, exact = [], []
    for row, row_positions in enumerate(positions):
        indices = torch.tensor(row_positions, device=logits.device)
        targets = input_ids[row, indices]
        predictions = logits[row, indices - 1].float().log_softmax(dim=-1)
        scores.append(predictions.gather(1, targets[:, None]).sum())
        exact.append((predictions.argmax(dim=-1) == targets).all().float())
    return torch.stack(scores), torch.stack(exact)


def teacher_forced_batch_image(
    model,
    processor,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    data_root,
    prompt,
    position_strategy,
    target,
    enable_thinking,
    patch=True,
):
    base_prompts, donor_prompts = [], []
    base_images, donor_images = [], []
    base_answers, source_answers = [], []
    base_spans, source_spans = [], []

    for pair in pairs:
        base, source = pair["base"], pair["source"]
        donor = pair.get("donor", source)
        base_prompt = sample_prompt(processor, base, prompt, enable_thinking)
        donor_prompt = sample_prompt(processor, donor, prompt, enable_thinking)
        base_answer, source_answer, base_span, source_span = target_answers(
            base, source, target
        )
        base_prompts.append(base_prompt)
        donor_prompts.append(donor_prompt)
        base_images.append(load_rgb_image(image_path_for(base, data_root)))
        donor_images.append(load_rgb_image(image_path_for(donor, data_root)))
        base_answers.append(base_answer)
        source_answers.append(source_answer)
        base_spans.append(base_span)
        source_spans.append(source_span)

    base_positions = resolve_batch_positions(
        processor, tokenizer, model, base_prompts, base_images, position_strategy
    )
    donor_positions = resolve_batch_positions(
        processor, tokenizer, model, donor_prompts, donor_images, position_strategy
    )

    prompts = base_prompts + base_prompts + donor_prompts
    images = base_images + base_images + donor_images
    answers = source_answers + base_answers + source_answers
    spans = source_spans + base_spans + source_spans
    encoding, full_positions, variable_positions = answer_token_positions(
        processor,
        prompts,
        answers,
        spans,
        images,
        model.device,
        dtype=getattr(model, "dtype", None),
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
def evaluate_teacher_forced_image(
    model,
    processor,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    data_root,
    prompt,
    position_strategy,
    target,
    enable_thinking,
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
        result = teacher_forced_batch_image(
            model, processor, tokenizer, blocks, subspaces, layers, hook_name,
            batch, data_root, prompt, position_strategy, target, enable_thinking,
            patch=True,
        )
        for name in names:
            patched[name].append(result[name].detach().cpu())
        if include_clean:
            baseline = teacher_forced_batch_image(
                model, processor, tokenizer, blocks, subspaces, layers, hook_name,
                batch, data_root, prompt, position_strategy, target,
                enable_thinking, patch=False,
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
def collect_initialization_features_image(
    model,
    processor,
    tokenizer,
    blocks,
    samples,
    layers,
    hook_name,
    data_root,
    prompt,
    position_strategy,
    enable_thinking,
    batch_size,
    max_samples,
):
    selected = list(samples[:max_samples]) if max_samples else list(samples)
    if len(selected) < 2:
        raise ValueError("PCA initialization requires at least two samples.")
    features = {layer: [] for layer in layers}
    for start in tqdm(range(0, len(selected), batch_size), desc="Collecting image PCA"):
        batch = selected[start : start + batch_size]
        prompts = [
            sample_prompt(processor, sample, prompt, enable_thinking)
            for sample in batch
        ]
        images = [load_rgb_image(image_path_for(sample, data_root)) for sample in batch]
        positions = resolve_batch_positions(
            processor, tokenizer, model, prompts, images, position_strategy
        )
        encoding = inputs_to_device(
            make_inputs(processor, prompts, images),
            model.device,
            dtype=getattr(model, "dtype", None),
        )
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


def train_subspace_image(
    model,
    processor,
    tokenizer,
    blocks,
    train_pairs,
    validation_pairs,
    layer,
    hook_name,
    data_root,
    prompt,
    position_strategy,
    target,
    dimension,
    hidden_size,
    enable_thinking,
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
    subspaces = torch.nn.ModuleDict(
        {
            str(layer): DASSubspace(
                hidden_size, dimension, initial_basis=initial_basis
            ).to(model.device)
        }
    )
    optimizer = torch.optim.Adam(subspaces.parameters(), lr=learning_rate)
    rng = random.Random(seed)
    print(
        f"[stage] image DAS initial validation: layer={layer}, k={dimension}, "
        f"validation_pairs={len(validation_pairs)}, batch_size={batch_size}",
        flush=True,
    )
    initial_validation = evaluate_teacher_forced_image(
        model, processor, tokenizer, blocks, subspaces, [layer], hook_name,
        validation_pairs, data_root, prompt, position_strategy, target,
        enable_thinking, batch_size, include_clean=False,
    )
    best_loss, best_epoch, stale = initial_validation["loss"], 0, 0
    if not math.isfinite(best_loss):
        raise RuntimeError("Non-finite validation loss before image DAS training.")
    best_state = copy.deepcopy(subspaces.state_dict())
    history = [{
        "epoch": 0,
        "train_loss": None,
        "validation_loss": best_loss,
        "validation_variable_teacher_forced_iia": initial_validation[
            "variable_teacher_forced_iia"
        ],
    }]
    print(
        f"epoch=0 train_loss=n/a val_loss={best_loss:.4f} "
        f"val_variable_iia={initial_validation['variable_teacher_forced_iia']:.3f}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        print(
            f"[stage] image DAS epoch {epoch}/{epochs}: "
            f"train_pairs={len(train_pairs)}, batch_size={batch_size}",
            flush=True,
        )
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        losses = []
        batch_starts = range(0, len(order), batch_size)
        for start in tqdm(
            batch_starts,
            desc=f"Image DAS train epoch {epoch}",
            leave=False,
        ):
            batch = [train_pairs[index] for index in order[start : start + batch_size]]
            optimizer.zero_grad()
            result = teacher_forced_batch_image(
                model, processor, tokenizer, blocks, subspaces, [layer], hook_name,
                batch, data_root, prompt, position_strategy, target,
                enable_thinking, patch=True,
            )
            if not torch.isfinite(result["loss"]):
                raise RuntimeError(
                    f"Non-finite image DAS loss at layer={layer}, k={dimension}."
                )
            result["loss"].backward()
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    subspaces.parameters(), gradient_clip_norm,
                    error_if_nonfinite=False,
                )
            optimizer.step()
            with torch.no_grad():
                for subspace in subspaces.values():
                    subspace.raw_basis.copy_(subspace.basis())
            losses.append(float(result["loss"].detach()))

        validation = evaluate_teacher_forced_image(
            model, processor, tokenizer, blocks, subspaces, [layer], hook_name,
            validation_pairs, data_root, prompt, position_strategy, target,
            enable_thinking, batch_size, include_clean=False,
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


@torch.no_grad()
def autoregressive_iia_image(
    model,
    processor,
    tokenizer,
    blocks,
    subspaces,
    layers,
    hook_name,
    pairs,
    data_root,
    prompt,
    position_strategy,
    target,
    enable_thinking,
    max_new_tokens,
    description="Image autoregressive IIA",
):
    correct = 0
    for pair in tqdm(pairs, desc=description):
        base, source = pair["base"], pair["source"]
        donor = pair.get("donor", source)
        expected = target_answers(base, source, target)[1]
        base_prompt = sample_prompt(processor, base, prompt, enable_thinking)
        donor_prompt = sample_prompt(processor, donor, prompt, enable_thinking)
        base_image = load_rgb_image(image_path_for(base, data_root))
        donor_image = load_rgb_image(image_path_for(donor, data_root))
        base_position = resolve_batch_positions(
            processor, tokenizer, model, [base_prompt], [base_image], position_strategy
        )[0]
        donor_position = resolve_batch_positions(
            processor, tokenizer, model, [donor_prompt], [donor_image], position_strategy
        )[0]
        inputs = inputs_to_device(
            make_inputs(processor, [base_prompt, donor_prompt], [base_image, donor_image]),
            model.device,
            dtype=getattr(model, "dtype", None),
        )
        generated = []
        for _ in range(max_new_tokens):
            base_length = int(inputs["attention_mask"][0].sum())
            outputs = patched_forward(
                model,
                inputs,
                blocks,
                subspaces,
                layers,
                hook_name,
                [base_position],
                [donor_position],
                n_base_groups=1,
            )
            next_id = outputs.logits[0, base_length - 1].argmax().reshape(1, 1)
            generated.append(int(next_id.item()))
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pad = torch.full(
                (input_ids.shape[0], 1),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros_like(pad)], dim=1)
            input_ids[0, base_length] = next_id.item()
            attention_mask[0, base_length] = 1
            inputs["input_ids"] = input_ids
            inputs["attention_mask"] = attention_mask
            if next_id.item() == tokenizer.eos_token_id:
                break

        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        match = re.match(r"-?\d+", text)
        correct += int(match is not None and match.group() == expected)
    return correct / len(pairs) if pairs else 0.0
