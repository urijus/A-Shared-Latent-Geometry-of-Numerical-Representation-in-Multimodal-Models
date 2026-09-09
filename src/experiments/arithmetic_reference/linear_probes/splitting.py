import torch


# Split functions
def split_indices_random(n, val_fraction=0.2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)

    n_val = int(n * val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return train_idx, val_idx, {
        "split": "random",
        "val_fraction": val_fraction,
        "seed": seed,
    }


def split_indices_by_operand(labels, val_fraction=0.1, seed=0):
    operands = sorted({
        int(row["a"]) for row in labels
    } | {
        int(row["b"]) for row in labels
    })

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(operands), generator=generator).tolist()

    n_val = int(val_fraction * len(operands))
    val_operands = {operands[i] for i in perm[:n_val]}
    train_operands = set(operands) - val_operands

    train_idx = []
    val_idx = []

    for i, row in enumerate(labels):
        a = int(row["a"])
        b = int(row["b"])

        if a in val_operands or b in val_operands:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        {
            "split": "operand_holdout",
            "val_fraction": val_fraction,
            "seed": seed,
            "train_operands": sorted(train_operands),
            "val_operands": sorted(val_operands),
        },
    )


def split_indices_by_result(labels, val_fraction=0.15, seed=0):
    sums = sorted({int(row["result"]) for row in labels})

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(sums), generator=generator).tolist()

    n_val = int(val_fraction * len(sums))
    val_sums = {sums[i] for i in perm[:n_val]}
    train_sums = set(sums) - val_sums

    train_idx = []
    val_idx = []

    for i, row in enumerate(labels):
        s = int(row["result"])

        if s in val_sums:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        {
            "split": "sum_holdout",
            "val_fraction": val_fraction,
            "seed": seed,
            "train_sums": sorted(train_sums),
            "val_sums": sorted(val_sums),
        },
    )


def make_split_indices(labels, split_mode, val_fraction, seed):
    if split_mode == "random":
        return split_indices_random(
            n=len(labels),
            val_fraction=val_fraction,
            seed=seed,
        )

    if split_mode == "operand":
        return split_indices_by_operand(
            labels=labels,
            val_fraction=val_fraction,
            seed=seed,
        )

    if split_mode == "result":
        return split_indices_by_result(
            labels=labels,
            val_fraction=val_fraction,
            seed=seed,
        )

    raise ValueError(f"Unknown split_mode: {split_mode}")



if __name__=="__main__":
    from pathlib import Path
    from src.common.io import load_jsonl

    data = load_jsonl(
        Path("./dataset/simple_addition/pythia_6_9b/digits/addition_balanced_operands.jsonl")
    )
    train_idx, val_idx, info = split_indices_by_result(data)
    print(info["train_sums"], info["val_sums"])