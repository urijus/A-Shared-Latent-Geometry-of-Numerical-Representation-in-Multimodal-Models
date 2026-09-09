import argparse
import re
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from src.models import (
    get_hidden_size,
    load_hf_model,
    resolve_model_for_loading,
    validate_block_layers,
)
from src.common.io import load_jsonl
from src.representations.extract_hf_representations import encode_expr_with_offsets


MODEL_NAME = "gemma4_12b_it"
DEFAULT_MODALITIES = ("addition", "subtraction", "multiplication")
# The chat template format we use to probe the tokens we want. Match for baseline oeprations only!!!
PROMPT_TOKEN_PATTERN = re.compile(r"<[^>]+>|[A-Za-z_]+|\d+|[^\sA-Za-z_\d]")
LABEL_FIELDS = (
    "sample_id",
    "task",
    "operation",
    "a",
    "b",
    "result",
    "a_mod_2",
    "a_mod_5",
    "a_mod_10",
    "b_mod_2",
    "b_mod_5",
    "b_mod_10",
    "result_mod_2",
    "result_mod_5",
    "result_mod_10",
    "result_mod_20",
    "result_mod_50",
    "result_mod_100",
    "requires_carry",
    "requires_borrow",
    "p0",
    "p1",
    "product_tens",
    "product_hundreds",
    "c0_hat",
    "c0",
    "r0",
    "c1_hat",
    "c1",
    "r1",
    "c2",
    "result_ones",
    "result_tens",
    "result_hundreds"
)


class BaselineDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        item = {
            "expr": str(ex["expr"]),
            "tokens": list(ex["tokens"]),
        }

        for field in LABEL_FIELDS:
            if field not in ex:
                continue
            value = ex[field]
            item[field] = str(value) if field in {"task", "operation"} else int(value)

        return item


def collate_examples(batch):
    collated = {
        "expr": [ex["expr"] for ex in batch],
        "tokens": [ex["tokens"] for ex in batch],
    }

    for field in LABEL_FIELDS:
        if field in batch[0]:
            collated[field] = [ex[field] for ex in batch]

    return collated


def make_loader(data, batch_size):
    return DataLoader(
        BaselineDataset(data),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )


def apply_chat_template(tokenizer, expr):
    if tokenizer.chat_template is None:
        raise ValueError("The tokenizer does not contain the chat template option.")

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": expr}],
        tokenize=False,
        add_generation_prompt=True,
    )


def uses_chat_template(model_name):
    return model_name.lower() in {
        "gemma4_12b_it",
        "google/gemma-4-12b-it",
        "gemma4_e4b_it",
        "google/gemma-4-e4b-it",
        "ministral3_14b_it_bf16",
        "mistralai/ministral-3-14b-instruct-2512-bf16",
    }


def final_hf_positions(alignment):
    # If a semantic token splits into multiple HF subtokens, probe the last one.
    return [
        positions[-1]
        for positions in alignment["token_to_hf_positions"]
    ]


def build_probe_context(tokenizer, expr):
    probe_tokens = PROMPT_TOKEN_PATTERN.findall(expr)
    probe_alignment = encode_expr_with_offsets(
        tokenizer=tokenizer,
        expr=expr,
        tokens=probe_tokens,
    )
    probe_positions = final_hf_positions(probe_alignment)

    return {
        "tokens": probe_tokens,
        "positions": probe_positions,
    }


def parse_position_index(spec, n_tokens):
    try:
        index = int(spec)
    except ValueError:
        raise ValueError(
            f"Position {spec!r} must be an integer."
        ) from None
    if index < 0:
        return index
    if index >= n_tokens:
        raise ValueError(
            f"Probe position index {spec!r} is outside the semantic token "
            f"range 0..{n_tokens - 1}."
        )
    return index


def raw_position_from_negative(index, n_raw_tokens):
    if n_raw_tokens is None:
        raise ValueError("Negative positions require n_raw_tokens.")
    raw_index = n_raw_tokens + index
    if raw_index < 0 or raw_index >= n_raw_tokens:
        raise ValueError(
            f"Raw token offset {index} is outside the token range "
            f"for a prompt with {n_raw_tokens} HF tokens."
        )
    return raw_index


def select_probe_positions(probe_context, position_specs, n_raw_tokens=None):
    selected_positions = []
    selected_names = []
    tokens = probe_context["tokens"]
    positions = probe_context["positions"]

    for spec in position_specs:
        index = parse_position_index(spec, len(tokens))
        if index < 0:
            selected_positions.append(raw_position_from_negative(index, n_raw_tokens))
            selected_names.append(str(index))
        else:
            selected_positions.append(positions[index])
            selected_names.append(str(index))

    return selected_positions, selected_names


def print_position_summary(tokenizer, expr, probe_context, selected_positions, selected_names):
    input_ids = tokenizer(expr, add_special_tokens=False)["input_ids"]
    hf_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print("\nPosition summary:")
    print(f"  rendered prompt: {expr!r}")
    print("  semantic tokens:")
    for index, (token, hf_position) in enumerate(
        zip(probe_context["tokens"], probe_context["positions"])
    ):
        print(f"    {index}: {token!r} -> HF {hf_position}")
    print("  selected positions:")
    for name, hf_position in zip(selected_names, selected_positions):
        token = hf_tokens[hf_position] if 0 <= hf_position < len(hf_tokens) else "<out>"
        print(f"    {name}: HF {hf_position}, token={token!r}")
    print()


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    loader,
    output_path,
    model_name,
    modality,
    layer_indices=None,
    use_chat_template=False,
    position_specs=None,
):
    model.eval()

    if position_specs is None:
        raise ValueError("Provide at least one 0-based prompt-token position.")

    layer_indices = validate_block_layers(model, layer_indices)

    n_examples = len(loader.dataset)
    n_layers = len(layer_indices)
    n_positions = len(position_specs)
    hidden_dim = get_hidden_size(model)

    activations = torch.empty(
        n_examples,
        n_layers,
        n_positions,
        hidden_dim,
        dtype=torch.float16,
        device="cpu",
    )

    labels = []
    position_names = None
    offset = 0
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch in tqdm(loader):
        exprs = batch["expr"]
        model_exprs = [
            apply_chat_template(tokenizer, expr) if use_chat_template else expr
            for expr in exprs
        ]

        # Tokens is just the arithmetic operation
        batch_size = len(exprs)

        encoding = tokenizer(
            model_exprs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        batch_token_positions = []
        for expr in model_exprs:
            probe_context = build_probe_context(
                tokenizer=tokenizer,
                expr=expr,
            )
            n_raw_tokens = len(encoding["input_ids"][len(batch_token_positions)])
            if "attention_mask" in encoding:
                n_raw_tokens = int(encoding["attention_mask"][len(batch_token_positions)].sum())
            selected_positions, selected_names = select_probe_positions(
                probe_context=probe_context,
                position_specs=position_specs,
                n_raw_tokens=n_raw_tokens,
            )
            if position_names is None:
                position_names = selected_names
                print_position_summary(
                    tokenizer,
                    expr,
                    probe_context,
                    selected_positions,
                    selected_names,
                )
            batch_token_positions.append(
                selected_positions
            )

        if any(len(positions) != n_positions for positions in batch_token_positions):
            raise ValueError(
                f"Expected {n_positions} token positions per example. "
                f"Got: {[len(positions) for positions in batch_token_positions]}"
            )

        encoding = encoding.to(model.device)

        outputs = model(
            **encoding,
            output_hidden_states=True,
            use_cache=False,
        )

        batch_acts = []

        for h_idx in layer_indices:
            hidden = outputs.hidden_states[h_idx]       # [batch, seq, dim]
            selected = torch.stack(
                [
                    hidden[i, positions, :]
                    for i, positions in enumerate(batch_token_positions)
                ],
                dim=0,
            )                                           # [batch, positions, dim]
            batch_acts.append(selected.cpu().to(torch.float16))

        batch_acts = torch.stack(batch_acts, dim=1)     # [batch, layers, positions, dim]

        activations[offset:offset + batch_size] = batch_acts

        for i in range(batch_size):
            label = {
                "expr": batch["expr"][i],
                "model_expr": model_exprs[i] if use_chat_template else batch["expr"][i],
                "tokens": batch["tokens"][i],
            }
            for field in LABEL_FIELDS:
                if field in batch:
                    label[field] = batch[field][i]
            labels.append(label)

        offset += batch_size

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "activations": activations,
            "labels": labels,
            "block_layers": layer_indices,
            "position_names": position_names,
            "position_indices": [int(position) for position in position_specs],
            "modality": modality,
            "model_name": model_name,
            "use_chat_template": use_chat_template,
            "note": (
                "Layers are 1-based transformer blocks; embeddings are not saved. "
                "Non-negative positions are 0-based semantic prompt tokens. "
                "Negative positions are raw HF token offsets from the end, "
                "e.g. -1 is the final input token."
            ),
        },
        output_path,
    )

    print("Saved:", output_path)
    print("Activation shape:", activations.shape)

    # Activations are (n_examples, n_layers, n_selected_tokens, d_model).


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Required non-negative 0-based prompt-token indices, e.g. 11 12 13."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional 1-based transformer block numbers to save. "
            "Defaults to every block."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # For any gemma model, use chat template
    use_chat_template = args.use_chat_template

    print("Loading model...")
    model_path, saved_model_name = resolve_model_for_loading(args.model)
    print("Model load path:", model_path)
    print("Saved model name:", saved_model_name)
    print("Use chat template:", use_chat_template)
    model, tokenizer = load_hf_model(model_path)

    for modality in args.modalities:
        print(f"Extracting activations for modality {modality}.")
        data_path = Path(args.data_dir) / f"{modality.lower()}_baseline.jsonl"
        data = load_jsonl(data_path)

        if not data:
            raise ValueError(f"No samples found in {data_path}.")

        loader = make_loader(data, batch_size=args.batch_size)
        output_path = Path(args.output_dir) / f"{modality.lower()}_baseline.pt"

        collect_activations(
            model=model,
            tokenizer=tokenizer,
            loader=loader,
            output_path=output_path,
            model_name=saved_model_name,
            modality=modality,
            layer_indices=args.layers,
            use_chat_template=use_chat_template,
            position_specs=args.positions,
        )

        saved = torch.load(output_path, map_location="cpu")
        print(saved["activations"].shape)
        print(saved["labels"][0])
        print(saved["position_names"])
        print(saved["block_layers"][:5], saved["block_layers"][-5:])


if __name__ == "__main__":
    main()
