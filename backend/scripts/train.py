from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import DEFAULT_MODEL_PATH, DEFAULT_PREPARED_PATH
from app.model import EEGNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_for(x, y, mask, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(x[mask]).float(), torch.from_numpy(y[mask]).long()),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.inference_mode():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            preds.extend(pred.tolist())
    return np.array(ys), np.array(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    if not args.prepared.exists():
        raise SystemExit(
            f"Prepared public EEG not found: {args.prepared}. "
            "Run download_nemar.py and prepare_dataset.py first."
        )

    archive = np.load(args.prepared, allow_pickle=False)
    x = archive["x"].astype(np.float32)
    y = archive["y"].astype(np.int64)
    split = archive["split"]
    labels = archive["labels"].tolist()
    dataset_id = str(archive["dataset_id"])

    train_mask, val_mask, test_mask = split == 0, split == 1, split == 2
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise SystemExit("Train/validation/test participant splits are required.")

    train_loader = loader_for(x, y, train_mask, args.batch_size, True)
    val_loader = loader_for(x, y, val_mask, args.batch_size, False)
    test_loader = loader_for(x, y, test_mask, args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet(channels=x.shape[1], classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            seen += len(xb)

        val_y, val_pred = evaluate(model, val_loader, device)
        val_bal = balanced_accuracy_score(val_y, val_pred)
        print(f"epoch={epoch:03d} loss={running_loss/max(seen,1):.4f} val_balanced_accuracy={val_bal:.4f}")
        if val_bal > best_val:
            best_val = float(val_bal)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training never produced a model state.")

    model.load_state_dict(best_state)
    model.to(device)
    test_y, test_pred = evaluate(model, test_loader, device)

    metrics = {
        "dataset_id": dataset_id,
        "split": "participant-held-out",
        "seed": args.seed,
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(test_y, test_pred)),
        "classification_report": classification_report(
            test_y, test_pred, labels=list(range(len(labels))), target_names=labels,
            zero_division=0, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(test_y, test_pred, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "channels": int(x.shape[1]),
            "samples": int(x.shape[2]),
            "dataset_id": dataset_id,
            "model_version": "eegnet_words_v1",
            "preprocessing": {
                "bandpass_hz": [1.0, 40.0],
                "target_sfreq": 128.0,
                "normalization": "per-trial per-channel z-score",
            },
            "metrics": metrics,
        },
        args.output,
    )

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"model={args.output}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
