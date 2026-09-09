import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset


NATIVE_DEPTH_FIELDS = ("composition_depths", "nesting_depths")
STANDARD_DEPTH_FIELDS = (
    "standard_composition_depths",
    "standard_nesting_depths",
)
DEPTH_FIELDS = STANDARD_DEPTH_FIELDS
ALL_DEPTH_FIELDS = NATIVE_DEPTH_FIELDS + STANDARD_DEPTH_FIELDS
TOP_K_VALUES = (1, 2, 5, 10, 50)


class LinearProbe(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.linear(x)


class HFDepthProbeDataset(Dataset):
    def __init__(self, data, depth_field="standard_composition_depths"):
        self.data = data
        self.depth_field = depth_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        tokens = ex["tokens"]
        depths = ex[self.depth_field]
        roles = ex["roles"]

        return {
            "expr": ex["text"],
            "tokens": tokens,
            "depth": torch.tensor(depths, dtype=torch.long),
            "roles": roles
        }


def collate_examples(batch):
    """
    Keeps examples as lists instead of stacking tensors.

    Why?
    HF tokenization happens inside extract_token_representations,
    and expressions may have different lengths. This way we process each
    one later, dont tensorize all in the loader.
    """
    return {
        "expr": [ex["expr"] for ex in batch],
        "tokens": [ex["tokens"] for ex in batch],
        "depth": [ex["depth"] for ex in batch],
        "roles": [ex["roles"] for ex in batch]
    }


def make_loader(data, batch_size, depth_field):
    ds = HFDepthProbeDataset(data=data, depth_field=depth_field)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )


def max_labeled_depth(data, depth_field):
    depths = [
        depth
        for row in data
        for depth in row[depth_field]
        if depth >= 0
    ]
    if not depths:
        raise ValueError(f"No labeled depths found for {depth_field}.")
    return max(depths)


def label_summary(y):
    unique, counts = torch.unique(y, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    majority_baseline = counts.max().item() / counts.sum().item()
    return class_counts, majority_baseline


def classification_metrics(y_true, y_pred, n_classes):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    accuracy = correct / total

    class_counts = {}
    per_class_accuracy = {}
    f1_scores = []

    for cls in range(n_classes):
        true_mask = y_true == cls
        pred_mask = y_pred == cls
        cls_total = true_mask.sum().item()
        class_counts[cls] = cls_total

        true_positive = (true_mask & pred_mask).sum().item()
        false_positive = (~true_mask & pred_mask).sum().item()
        false_negative = (true_mask & ~pred_mask).sum().item()

        if cls_total > 0:
            per_class_accuracy[cls] = true_positive / cls_total

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        if precision_denominator > 0 and recall_denominator > 0:
            precision = true_positive / precision_denominator
            recall = true_positive / recall_denominator
            f1_scores.append(
                0.0
                if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
        elif cls_total > 0:
            f1_scores.append(0.0)

    class_recalls = list(per_class_accuracy.values())
    balanced_accuracy = (
        sum(class_recalls) / len(class_recalls)
        if class_recalls
        else 0.0
    )
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    majority_baseline = (
        max(class_counts.values()) / total
        if total > 0
        else 0.0
    )
    normalized_gain = (
        (accuracy - majority_baseline) / (1 - majority_baseline)
        if majority_baseline < 1
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "majority_baseline": majority_baseline,
        "normalized_gain": normalized_gain,
        "per_class_accuracy": per_class_accuracy,
        "class_counts": class_counts,
    }


def evaluate_probe(probe, X, y, n_classes, device, batch_size=4096):
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)

    probe.eval()

    y_true = []
    y_pred = []
    all_logits = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)

            logits = probe(xb).cpu()
            preds = logits.argmax(dim=-1)

            y_true.append(yb.cpu())
            y_pred.append(preds)
            all_logits.append(logits)

    y_true = torch.cat(y_true, dim=0)
    metrics = classification_metrics(
        y_true=y_true,
        y_pred=torch.cat(y_pred, dim=0),
        n_classes=n_classes,
    )
    logits = torch.cat(all_logits, dim=0)
    for k in TOP_K_VALUES:
        top_ids = logits.topk(min(k, n_classes), dim=-1).indices
        metrics[f"top_{k}_accuracy"] = float(
            (top_ids == y_true[:, None]).any(dim=1).float().mean()
        )
    return metrics


def shuffle_labels(y, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(y.numel(), generator=generator)
    return y[indices]


def train_probe(
    X_train,
    y_train,
    n_classes,
    device,
    epochs=30,
    batch_size=4096,
    lr=1e-3,
    progress_label=None,
    X_val=None,
    y_val=None,
    early_stopping_patience=None,
    class_weight=None,
):
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    probe = LinearProbe(X_train.shape[1], n_classes).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    use_early_stopping = (
        early_stopping_patience is not None
        and early_stopping_patience > 0
        and X_val is not None
        and y_val is not None
    )
    best_val_accuracy = float("-inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        probe.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            loss = criterion(probe(xb), yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_num = epoch + 1

        if use_early_stopping:
            val_metrics = evaluate_probe(
                probe=probe,
                X=X_val,
                y=y_val,
                n_classes=n_classes,
                device=device,
                batch_size=batch_size,
            )
            val_accuracy = val_metrics["accuracy"]
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch = epoch_num
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in probe.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        should_report = epoch_num == 1 or epoch_num == epochs or epoch_num % 5 == 0
        if progress_label is not None and should_report:
            print(f"    {progress_label}: epoch {epoch_num}/{epochs}", flush=True)

        if use_early_stopping and epochs_without_improvement >= early_stopping_patience:
            if progress_label is not None:
                print(
                    f"    {progress_label}: early stopping at epoch {epoch_num}; "
                    f"best epoch {best_epoch}",
                    flush=True,
                )
            break

    if best_state_dict is not None:
        probe.load_state_dict(
            {
                key: value.to(device)
                for key, value in best_state_dict.items()
            }
        )

    return probe


def train_and_evaluate_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    n_classes,
    device,
    epochs=30,
    batch_size=4096,
    lr=1e-3,
    early_stopping_patience=None,
    class_weight=None,
):
    probe = train_probe(
        X_train=X_train,
        y_train=y_train,
        n_classes=n_classes,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        X_val=X_val,
        y_val=y_val,
        early_stopping_patience=early_stopping_patience,
        class_weight=class_weight,
    )
    metrics = evaluate_probe(
        probe=probe,
        X=X_val,
        y=y_val,
        n_classes=n_classes,
        device=device,
        batch_size=batch_size,
    )
    return probe, metrics
