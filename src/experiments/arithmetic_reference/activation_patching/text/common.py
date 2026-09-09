import json
import random
from pathlib import Path

import torch

from src.common.io import load_jsonl
from src.experiments.arithmetic_reference.linear_probes.text.extract_activations import (
    apply_chat_template,
    uses_chat_template
)

from src.models import get_blocks


OPERATIONS = {
    "addition": "+",
    "subtraction": "-",
    "multiplication": "*"
}

DEFAULT_HOOKS = ("resid_pre", "attn_out", "mlp_input", "resid_post")
HOOK_DESCRIPTIONS = {
    "resid_pre": "Block input; for layer L this equals hidden_states[L-1].",
    "attn_out": "Raw self-attention output before Gemma 4 post-attention normalization.",
    "mlp_input": "Residual after attention addition, before the MLP input layernorm.",
    "resid_post": "Block output; for layer L this equals hidden_states[L].",
}


# utilities

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def valid_hooks(hooks):
    hooks = hooks or list(DEFAULT_HOOKS)
    unknown = set(hooks) - set(HOOK_DESCRIPTIONS)
    if unknown:
        raise ValueError(f"Unknown hooks {sorted(unknown)}; use {list(DEFAULT_HOOKS)}")
    return hooks


# arithmetic pairs
def build_pairs_from_dataset(data_path, operation, n_pairs, seed=0):
    """Pair dataset rows while preserving expressions verbatim."""
    samples = load_jsonl(Path(data_path))
    symbol = OPERATIONS[operation]
    samples = [sample for sample in samples if sample.get("operation") == symbol]
    if not samples:
        raise ValueError(f"No {symbol!r} samples found in {data_path}")

    required = {"expr", "a", "b", "result"}
    missing = required - samples[0].keys()
    if missing:
        raise ValueError(f"Dataset {data_path} is missing fields: {sorted(missing)}")

    groups = {}
    for sample in samples:
        # Organize by the numbr of digits per operand
        key = (
            len(str(sample["a"])),
            len(str(sample["b"])),
            len(str(sample["result"])),
        )
        groups.setdefault(key, []).append(sample)

    valid_samples = [
        sample
        for group in groups.values()
        if len({item["result"] for item in group}) > 1
        for sample in group
    ]
    rng = random.Random(seed)
    pairs = []

    clean_samples = rng.sample(valid_samples, n_pairs)
    for pair_id, clean in enumerate(clean_samples):
        key = (len(str(clean["a"])), len(str(clean["b"])), len(str(clean["result"])))
        corrupt = rng.choice(
            [item for item in groups[key] if item["result"] != clean["result"]]
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "operation": operation,
                "clean_sample_id": clean.get("sample_id"),
                "clean_x": clean["a"],
                "clean_y": clean["b"],
                "clean_expr": clean["expr"],
                "clean_answer": clean["result"],
                "corrupt_sample_id": corrupt.get("sample_id"),
                "corrupt_x": corrupt["a"],
                "corrupt_y": corrupt["b"],
                "corrupt_expr": corrupt["expr"],
                "corrupt_answer": corrupt["result"],
            }
        )
    return pairs


def prompt_with_template(tokenizer, expr, model_name):
    if uses_chat_template(model_name):
        return apply_chat_template(tokenizer, expr)
    return expr


def encode_prompt(tokenizer, prompt, device):
    encoding = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return {name: tensor.to(device) for name, tensor in encoding.items()}


def answer_token_ids(tokenizer, answer):
    ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Answer {answer!r} produced no tokens")
    return [int(token_id) for token_id in ids]


# activation patching

def hook_module(block, hook_name):
    """Return (module, use_pre_hook) for one clearly named patch location."""
    if hook_name == "resid_pre":
        return block, True
    if hook_name == "attn_out":
        return block.self_attn, False
    if hook_name == "mlp_input":
        module = getattr(
            block,
            "pre_feedforward_layernorm",
            getattr(block, "post_attention_layernorm", None),
        )
        if module is None:
            raise AttributeError("Block has no MLP input layernorm")
        return module, True
    if hook_name == "resid_post":
        return block, False
    raise ValueError(f"Unknown hook: {hook_name}")


def _hidden(value):
    return value[0] if isinstance(value, tuple) else value


def cache_clean_activations(model, clean_encoding, layers, hooks):
    """Run the clean prompt once and save [batch, sequence, hidden] at each site."""
    cache = {}
    handles = []
    blocks = get_blocks(model)

    for layer in layers:
        for hook_name in hooks:
            module, use_pre_hook = hook_module(blocks[layer - 1], hook_name)
            key = (layer, hook_name)

            if use_pre_hook:
                def save_pre(_module, inputs, key=key):
                    cache[key] = _hidden(inputs).detach().clone()

                handles.append(module.register_forward_pre_hook(save_pre))
            else:
                def save_output(_module, _inputs, output, key=key):
                    cache[key] = _hidden(output).detach().clone()

                handles.append(module.register_forward_hook(save_output))

    try:
        model(**clean_encoding, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return cache


def _register_patch(model, clean_activation, layer, hook_name, clean_position, corrupt_position):
    module, use_pre_hook = hook_module(get_blocks(model)[layer - 1], hook_name)

    def patch_token(value):
        hidden = _hidden(value).clone()
        if clean_position >= clean_activation.shape[1]:
            raise ValueError(f"Clean position {clean_position} is outside the activation")
        if corrupt_position >= hidden.shape[1]:
            raise ValueError(f"Corrupt position {corrupt_position} is outside the activation")
        if hidden.shape[2] != clean_activation.shape[2]:
            raise ValueError("Clean and corrupt activation shapes are incompatible")
        hidden[:, corrupt_position, :] = clean_activation[:, clean_position, :].to(
            hidden.device,
            hidden.dtype,
        )
        return (hidden,) + value[1:] if isinstance(value, tuple) else hidden

    if use_pre_hook:
        return module.register_forward_pre_hook(
            lambda _module, inputs: patch_token(inputs)
        )
    return module.register_forward_hook(
        lambda _module, _inputs, output: patch_token(output)
    )


def patched_forward(
    model,
    encoding,
    clean_activation,
    layer,
    hook_name,
    clean_position,
    corrupt_position,
):
    """Copy one clean token-position vector into the corrupt run."""
    handle = _register_patch(
        model,
        clean_activation,
        layer,
        hook_name,
        clean_position,
        corrupt_position,
    )

    try:
        return model(**encoding, use_cache=False)
    finally:
        handle.remove()


def patched_forward_many(model, encoding, patches):
    """Apply several layer patches during the same forward pass."""
    handles = [_register_patch(model, **patch) for patch in patches]
    try:
        return model(**encoding, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()


def answer_metrics(model, prompt_encoding, answer_ids, patch=None):
    """Score all answer tokens and report token and exact-match accuracy."""
    answer = torch.tensor([answer_ids], device=prompt_encoding["input_ids"].device)
    prompt_length = prompt_encoding["input_ids"].shape[1]
    prompt_mask = prompt_encoding.get(
        "attention_mask",
        torch.ones_like(prompt_encoding["input_ids"]),
    )
    encoding = {
        "input_ids": torch.cat([prompt_encoding["input_ids"], answer], dim=1),
        "attention_mask": torch.cat(
            [prompt_mask, torch.ones_like(answer)], dim=1
        ),
    }

    if patch is None:
        outputs = model(**encoding, use_cache=False)
    elif isinstance(patch, list):
        outputs = patched_forward_many(model, encoding, patch)
    else:
        outputs = patched_forward(model, encoding, **patch)

    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length + len(answer_ids) - 1,
        device=outputs.logits.device,
    )
    log_probs = outputs.logits[0, prediction_positions].float().log_softmax(dim=-1)
    target_ids = answer[0].to(log_probs.device)
    correct = log_probs.argmax(dim=-1) == target_ids
    return {
        "score": float(log_probs.gather(1, target_ids[:, None]).mean()),
        "token_accuracy": float(correct.float().mean()),
        "iia": bool(correct.all()),
    }


def contrast_metrics(model, prompt_encoding, clean_ids, corrupt_ids, patch=None):
    clean = answer_metrics(model, prompt_encoding, clean_ids, patch)
    corrupt = answer_metrics(model, prompt_encoding, corrupt_ids, patch)
    return {
        "contrast": clean["score"] - corrupt["score"],
        "clean_answer": clean,
        "corrupt_answer": corrupt,
    }


def recovery(clean_score, corrupt_score, patched_score, eps=1e-6):
    denominator = clean_score - corrupt_score
    if abs(denominator) < eps:
        return None
    return (patched_score - corrupt_score) / denominator
