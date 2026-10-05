from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ARCHITECTURES, build_model
from .provenance import sha256_file


Progress = Callable[[dict], None]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(x, y, mask, batch_size, shuffle):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x[mask]).float(),
            torch.from_numpy(y[mask]).long(),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _evaluate(model, loader, device):
    model.eval()
    ys = []
    preds = []
    with torch.inference_mode():
        for xb, yb in loader:
            prediction = model(xb.to(device)).argmax(dim=1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            preds.extend(prediction.tolist())
    return np.asarray(ys), np.asarray(preds)


def _class_weights(y: np.ndarray, classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split is missing class indices: {missing}")
    return len(y) / (classes * counts)


def read_prepared_metadata(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared public EEG not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "x",
            "y",
            "split",
            "labels",
            "dataset_id",
            "provenance_sha256",
            "sfreq",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                "Prepared archive is missing required fields: "
                + ", ".join(missing)
            )

        provenance = str(archive["provenance_sha256"])
        if len(provenance) != 64:
            raise ValueError("Prepared archive provenance hash is invalid.")

        split = archive["split"]
        groups = archive["groups"] if "groups" in archive.files else None
        return {
            "path": str(path),
            "dataset_id": str(archive["dataset_id"]),
            "task": str(archive["task"]) if "task" in archive.files else "words",
            "provenance_sha256": provenance,
            "labels": archive["labels"].tolist(),
            "trials": int(len(archive["x"])),
            "channels": int(archive["x"].shape[1]),
            "samples": int(archive["x"].shape[2]),
            "sfreq": float(archive["sfreq"]),
            "train_trials": int((split == 0).sum()),
            "val_trials": int((split == 1).sum()),
            "test_trials": int((split == 2).sum()),
            "subjects": (
                sorted(set(groups.tolist()))
                if groups is not None
                else []
            ),
            "channel_names": (
                archive["channel_names"].tolist()
                if "channel_names" in archive.files
                else []
            ),
        }


def train_model(
    prepared_path: Path,
    output_path: Path,
    *,
    task: str | None = None,
    architecture: str = "eegnet",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    progress: Progress | None = None,
) -> dict:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if not 1 <= epochs <= 500:
        raise ValueError("epochs must be between 1 and 500")
    if not 1 <= batch_size <= 1024:
        raise ValueError("batch_size must be between 1 and 1024")
    if not 1e-7 <= lr <= 1.0:
        raise ValueError("lr is outside the supported range")

    seed_everything(seed)
    prepared_path = Path(prepared_path)
    output_path = Path(output_path)

    with np.load(prepared_path, allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.int64)
        split = archive["split"]
        labels = archive["labels"].tolist()
        dataset_id = str(archive["dataset_id"])
        provenance = str(archive["provenance_sha256"])
        archive_task = (
            str(archive["task"]) if "task" in archive.files else "words"
        )
        sfreq = float(archive["sfreq"])

    requested_task = task or archive_task
    if requested_task != archive_task:
        raise ValueError(
            f"Requested task {requested_task!r} does not match prepared "
            f"archive task {archive_task!r}."
        )
    if len(provenance) != 64:
        raise ValueError("Prepared archive provenance hash is invalid.")

    train_mask = split == 0
    val_mask = split == 1
    test_mask = split == 2
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Train/validation/test participant-held-out splits are required."
        )

    train_loader = _loader(x, y, train_mask, batch_size, True)
    val_loader = _loader(x, y, val_mask, batch_size, False)
    test_loader = _loader(x, y, test_mask, batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        architecture,
        channels=x.shape[1],
        classes=len(labels),
    ).to(device)

    weights_np = _class_weights(y[train_mask], len(labels))
    weights = torch.tensor(
        weights_np,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            seen += len(xb)

        val_y, val_pred = _evaluate(model, val_loader, device)
        val_balanced = float(
            balanced_accuracy_score(val_y, val_pred)
        )

        if val_balanced > best_val:
            best_val = val_balanced
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if progress:
            progress(
                {
                    "phase": "training",
                    "architecture": architecture,
                    "epoch": epoch,
                    "epochs": epochs,
                    "loss": running_loss / max(seen, 1),
                    "validation_balanced_accuracy": val_balanced,
                    "best_validation_balanced_accuracy": best_val,
                }
            )

    if best_state is None:
        raise RuntimeError("Training never produced a model state.")

    model.load_state_dict(best_state)
    model.to(device)
    test_y, test_pred = _evaluate(model, test_loader, device)

    metrics = {
        "dataset_id": dataset_id,
        "task": requested_task,
        "architecture": architecture,
        "split": "participant-held-out",
        "provenance_sha256": provenance,
        "seed": seed,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_val,
        "training_class_weights": {
            label: float(weights_np[index])
            for index, label in enumerate(labels)
        },
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_y, test_pred)
        ),
        "classification_report": classification_report(
            test_y,
            test_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            test_y,
            test_pred,
            labels=list(range(len(labels))),
        ).tolist(),
        "labels": labels,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_version = (
        f"{dataset_id}_{architecture}_{requested_task}_"
        f"{provenance[:8]}_v1"
    )
    tmp = output_path.with_suffix(output_path.suffix + ".part")
    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "channels": int(x.shape[1]),
            "samples": int(x.shape[2]),
            "dataset_id": dataset_id,
            "provenance_sha256": provenance,
            "task": requested_task,
            "architecture": architecture,
            "model_version": model_version,
            "preprocessing": {
                "target_sfreq": sfreq,
                "normalization": "per-trial per-channel z-score",
            },
            "metrics": metrics,
        },
        tmp,
    )
    tmp.replace(output_path)
    metrics["checkpoint_sha256"] = sha256_file(output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if progress:
        progress(
            {
                "phase": "trained",
                "architecture": architecture,
                "model_path": str(output_path),
                "metrics_path": str(metrics_path),
                "best_validation_balanced_accuracy": best_val,
                "test_balanced_accuracy": metrics["test_balanced_accuracy"],
            }
        )

    return {
        "model_path": str(output_path),
        "metrics_path": str(metrics_path),
        "model_version": model_version,
        "metrics": metrics,
    }


def benchmark_models(
    prepared_path: Path,
    output_dir: Path,
    *,
    architectures: Iterable[str] = ARCHITECTURES,
    epochs: int = 15,
    batch_size: int = 32,
    seed: int = 42,
    progress: Progress | None = None,
) -> dict:
    metadata = read_prepared_metadata(prepared_path)
    architectures = list(architectures)
    if not architectures:
        raise ValueError("At least one architecture must be benchmarked.")

    unknown = sorted(set(architectures) - set(ARCHITECTURES))
    if unknown:
        raise ValueError(
            "Unsupported architectures: " + ", ".join(unknown)
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for index, architecture in enumerate(architectures, start=1):
        if progress:
            progress(
                {
                    "phase": "benchmark",
                    "architecture": architecture,
                    "architecture_index": index,
                    "architecture_total": len(architectures),
                }
            )

        trained = train_model(
            prepared_path,
            output_dir / f"{architecture}.pt",
            task=metadata["task"],
            architecture=architecture,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            progress=progress,
        )
        metrics = trained["metrics"]
        results.append(
            {
                "architecture": architecture,
                "best_epoch": metrics["best_epoch"],
                "validation_balanced_accuracy": metrics[
                    "best_validation_balanced_accuracy"
                ],
                "test_accuracy": metrics["test_accuracy"],
                "test_balanced_accuracy": metrics[
                    "test_balanced_accuracy"
                ],
                "model_path": trained["model_path"],
                "metrics_path": trained["metrics_path"],
                "checkpoint_sha256": metrics["checkpoint_sha256"],
            }
        )

    results.sort(
        key=lambda item: item["validation_balanced_accuracy"],
        reverse=True,
    )
    report = {
        "dataset_id": metadata["dataset_id"],
        "task": metadata["task"],
        "provenance_sha256": metadata["provenance_sha256"],
        "seed": seed,
        "epochs": epochs,
        "ranking_metric": "validation_balanced_accuracy",
        "selection_note": (
            "Test metrics are reported only after each validation-selected "
            "checkpoint is fixed and are not used to rank architectures."
        ),
        "winner": results[0]["architecture"],
        "winner_model_path": results[0]["model_path"],
        "results": results,
    }
    report_path = output_dir / "benchmark.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    if progress:
        progress(
            {
                "phase": "benchmark_completed",
                "winner": report["winner"],
                "winner_model_path": report["winner_model_path"],
            }
        )
    return report
