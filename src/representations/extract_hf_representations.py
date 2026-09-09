import torch


def get_dataset_token_spans(expr, tokens):
    """
    Finds where each dataset token appears in the raw expression string.

    Example:
        expr = "5 + 8"
        tokens = ["5", "+", "8"]

    Returns:
        [(0, 1), (2, 3), (4, 5)]
    """

    spans = []
    cursor = 0

    for tok in tokens:
        start = expr.index(tok, cursor)
        end = start + len(tok)

        spans.append((start, end))
        cursor = end

    return spans


def encode_expr_with_offsets(tokenizer, expr, tokens):
    """
    Tokenizes the exact expression string.

    Then maps each dataset token to the Hugging Face token positions
    that overlap with it.
    """

    encoding = tokenizer(
        expr,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    input_ids = encoding["input_ids"]
    hf_offsets = encoding["offset_mapping"]

    dataset_spans = get_dataset_token_spans(expr, tokens)

    token_to_hf_positions = []

    for dataset_start, dataset_end in dataset_spans:
        matching_positions = []

        for hf_position, (hf_start, hf_end) in enumerate(hf_offsets):
            overlaps = hf_start < dataset_end and hf_end > dataset_start

            if overlaps:
                matching_positions.append(hf_position)

        if not matching_positions:
            raise ValueError(
                f"No HF token found for dataset span {(dataset_start, dataset_end)} "
                f"in expression: {expr}"
            )

        token_to_hf_positions.append(matching_positions)

    return {
        "input_ids": input_ids,
        "hf_offsets": hf_offsets,
        "dataset_spans": dataset_spans,
        "token_to_hf_positions": token_to_hf_positions,
    }


@torch.no_grad()
def extract_token_representations(model, tokenizer, expr, tokens, layer_idx, use_attn_out=False):
    """
    Returns one vector per original dataset token.

    If one dataset token maps to multiple HF tokens, we take the last one.
    """

    alignment = encode_expr_with_offsets(
        tokenizer=tokenizer,
        expr=expr,
        tokens=tokens,
    )

    input_ids = torch.tensor(
        [alignment["input_ids"]],
        dtype=torch.long,
        device=model.device,
    )


    # Define internal hook so we can access representations right after attn block
    captured = {}
    def save_attn_output(module, inputs, output):
        if isinstance(output, tuple):
            captured["attn_out"] = output[0].detach().cpu()
        else:
            captured["attn_out"] = output.detach().cpu()

    handle = None

    if use_attn_out:
        handle = model.model.layers[layer_idx].self_attn.register_forward_hook(save_attn_output)

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
    )

    # both are shape: [hf_seq_len, hidden_dim]
    if not use_attn_out:
        hidden = outputs.hidden_states[layer_idx][0]
    else:
        hidden = captured["attn_out"][0]

    if handle is not None:
        handle.remove()


    # Align representations with data tokens
    token_reps = []

    for hf_positions in alignment["token_to_hf_positions"]:
        subtok_reps = hidden[hf_positions]
        if not use_attn_out and layer_idx == 0:
            tok_rep = subtok_reps.mean(dim=0)
            print("Since layer_idx is 0, use the mean of HF vectors (raw embeddings) to form \
                  token vectors.")
        else:
            tok_rep = subtok_reps[-1]

        token_reps.append(tok_rep)

    token_reps = torch.stack(token_reps, dim=0)

    return token_reps.float().cpu(), alignment
