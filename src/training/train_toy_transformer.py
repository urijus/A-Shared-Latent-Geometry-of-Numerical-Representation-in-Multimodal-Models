"""Train the supplementary toy arithmetic Transformer."""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import load_jsonl, set_seed
from src.models.toy_transformer import (
    ArithmeticDataset,
    build_vocab,
    get_device,
    load_model,
    make_model,
)


def load_config(path):
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def save_config(config, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/toy_transformer.yaml")
    parser.add_argument("--run_dir", type=str, default="runs/debug")
    return parser.parse_args()


def plot_results(train_seq, val_seq, ylabel="Loss", path="loss_curve.png"):
    plt.figure()
    plt.plot(train_seq, label=f"train {ylabel.lower()}")
    plt.plot(val_seq, label=f"val {ylabel.lower()}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f"Training vs Validation {ylabel}")
    plt.savefig(path)
    plt.close()


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            loss_sum += loss.item() * labels.numel()

    return loss_sum / total, correct / total


def train(config, device):
    train_data = load_jsonl(config["data"]["train_path"])
    val_data = load_jsonl(config["data"]["val_path"])
    test_data = load_jsonl(config["data"]["test_path"])
    label_field = config["data"].get("label_field", "result_mod")

    # Include test_data only to make sure all possible result labels are known.
    # For stricter research, build result vocab analytically instead.
    token_to_id, result_to_id, id_to_result = build_vocab(
        [train_data, val_data, test_data],
        label_field=label_field,
    )

    train_ds = ArithmeticDataset(
        train_data,
        token_to_id,
        result_to_id,
        config["data"]["max_len"],
        label_field=label_field,
    )
    val_ds = ArithmeticDataset(
        val_data,
        token_to_id,
        result_to_id,
        config["data"]["max_len"],
        label_field=label_field,
    )

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    model = make_model(token_to_id, result_to_id, config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    scheduler = None
    use_scheduler = bool(config["training"]["use_scheduler"])
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=1e-5,
        )

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_eval_accs = []

    patience = 0
    max_patience = config["training"]["patience"]
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()

        total_loss = 0.0
        total = 0
        correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attn_mask)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total += labels.numel()

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)
        train_eval_loss, train_eval_acc = evaluate(model, train_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience += 1

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        train_eval_accs.append(train_eval_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc:.4f} | "
            f"train_eval_acc {train_eval_acc:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f}"
        )

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

        if use_scheduler:
            scheduler.step()

    plot_results(train_losses, val_losses, ylabel="Loss", path=Path(config["run_dir"]) / "loss_curve.png")
    plot_results(train_eval_accs, val_accs, ylabel="Accuracy", path=Path(config["run_dir"]) / "acc_curve.png")

    save = {
        "model_state_dict": best_state_dict,
        "token_to_id": token_to_id,
        "result_to_id": result_to_id,
        "id_to_result": id_to_result,
        "config": config,
        "metrics": {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "train_eval_accs": train_eval_accs,
            "val_accs": val_accs,
        },
    }

    save_path = Path(config["run_dir"]) / "transformer.pt"
    torch.save(save, save_path)
    print(f"Saved model to {save_path}")


def test(config, device):
    model, token_to_id, result_to_id, id_to_result, config = load_model(
        Path(config["run_dir"]) / "transformer.pt",
        device,
    )

    test_data = load_jsonl(config["data"]["test_path"])
    label_field = config["data"].get("label_field", "result_mod")
    test_ds = ArithmeticDataset(
        test_data,
        token_to_id,
        result_to_id,
        config["data"]["max_len"],
        label_field=label_field,
    )
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"])

    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)
    config["run_dir"] = args.run_dir

    set_seed(config.get("seed", 0))
    save_config(config, args.run_dir)
    print(config)

    device = get_device(config)

    train(config, device)
    test(config, device)


if __name__ == "__main__":
    main()
