from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"

PAD_ID = 0
CLS_ID = 1

SPECIAL_TOKENS = {
    PAD_TOKEN: PAD_ID,
    CLS_TOKEN: CLS_ID,
}


def get_device(config):
    requested = config["training"].get("device", "auto")

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def build_vocab(datasets, label_field="result_mod"):
    tokens = set()
    results = set()

    for data in datasets:
        for ex in data:
            tokens.update(ex["tokens"])
            results.add(ex[label_field])

    token_to_id = dict(SPECIAL_TOKENS)

    next_id = len(token_to_id)
    for tok in sorted(tokens):
        if tok not in token_to_id:
            token_to_id[tok] = next_id
            next_id += 1

    result_vocab = sorted(results)
    result_to_id = {r: i for i, r in enumerate(result_vocab)}
    id_to_result = {i: r for r, i in result_to_id.items()}

    assert token_to_id[PAD_TOKEN] == PAD_ID
    assert token_to_id[CLS_TOKEN] == CLS_ID

    return token_to_id, result_to_id, id_to_result


class ArithmeticDataset(Dataset):
    def __init__(self, data, token_to_id, result_to_id, max_len=64, label_field="result_mod"):
        self.data = data
        self.token_to_id = token_to_id
        self.result_to_id = result_to_id
        self.max_len = max_len
        self.label_field = label_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        tokens = [CLS_TOKEN] + ex["tokens"]
        input_ids = [self.token_to_id[t] for t in tokens]

        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        attn_mask = [1] * len(input_ids)

        pad_len = self.max_len - len(input_ids)
        input_ids += [self.token_to_id[PAD_TOKEN]] * pad_len
        attn_mask += [0] * pad_len

        label = self.result_to_id[ex[self.label_field]]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SmallTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_classes,
        max_len=64,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        pad_id=0,
        causal=False,
    ):
        super().__init__()

        self.pad_id = pad_id
        self.causal = causal
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attn_mask, return_hidden=False):
        batch_size, seq_len = input_ids.shape

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        hidden_states = [x]

        key_padding_mask = ~attn_mask
        causal_mask = None
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
                diagonal=1,
            )

        for layer in self.layers:
            x = layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=key_padding_mask,
            )
            hidden_states.append(x)

        x = self.ln(x)

        if self.causal:
            last_token_idx = attn_mask.long().sum(dim=1) - 1
            batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
            pooled = x[batch_idx, last_token_idx]
        else:
            pooled = x[:, 0]

        logits = self.classifier(pooled)

        if return_hidden:
            return logits, hidden_states

        return logits


def make_model(token_to_id, result_to_id, config):
    return SmallTransformer(
        vocab_size=len(token_to_id),
        n_classes=len(result_to_id),
        max_len=config["data"]["max_len"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        pad_id=token_to_id[PAD_TOKEN],
        causal=config["model"].get("causal", False),
    )


def load_model(path, device):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)

    token_to_id = checkpoint["token_to_id"]
    result_to_id = checkpoint["result_to_id"]

    id_to_result = checkpoint["id_to_result"]
    id_to_result = {int(k): v for k, v in id_to_result.items()}

    config = checkpoint["config"]

    model = make_model(token_to_id, result_to_id, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, token_to_id, result_to_id, id_to_result, config
