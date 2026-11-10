"""Train one controlled CNN experiment and save validation-only results.

Examples:
    python src/train.py --experiment baseline
    python src/train.py --experiment dropout_05 --dropout 0.5
    python src/train.py --experiment learning_rate_0005 --lr 0.0005
    python src/train.py --experiment batch_32 --batch-size 32
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CustomCNN


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DURIAN_DATA_DIR", PROJECT_DIR / "data" / "durian_leaf_dataset"))
RUNS_DIR = Path(os.environ.get("MODEL_OUTPUT_DIR", PROJECT_DIR / "training_runs"))
IMAGE_SIZE = (128, 128)
SEED = 42
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.3)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def confusion_and_report(labels, predictions, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    report = {}
    total = int(matrix.sum())
    weighted_f1 = 0.0
    for i, name in enumerate(classes):
        tp = int(matrix[i, i])
        support = int(matrix[i, :].sum())
        predicted_count = int(matrix[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        weighted_f1 += f1 * support
        report[name] = {
            "precision": precision, "recall": recall, "f1-score": f1, "support": support,
        }
    weighted_f1 = weighted_f1 / total if total else 0.0
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    report["accuracy"] = accuracy
    report["weighted avg"] = {
        "precision": sum(report[n]["precision"] * report[n]["support"] for n in classes) / total,
        "recall": sum(report[n]["recall"] * report[n]["support"] for n in classes) / total,
        "f1-score": weighted_f1,
        "support": total,
    }
    return matrix.tolist(), report, weighted_f1


def validate_images(split_path):
    """Check training/validation files only; leave test data untouched until final evaluation."""
    failures = []
    for path in split_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                failures.append((str(path), str(exc)))
    if failures:
        examples = "\n".join(f"  {p}: {err}" for p, err in failures[:10])
        raise RuntimeError(f"Found {len(failures)} unreadable images in {split_path}:\n{examples}")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total = 0.0, 0, 0
    labels_all, preds_all = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(predictions.cpu().tolist())
    matrix, _, weighted_f1 = confusion_and_report(labels_all, preds_all, range(len(loader.dataset.classes)))
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "weighted_f1": weighted_f1,
        "labels": labels_all,
        "predictions": preds_all,
        "confusion_matrix": matrix,
    }


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must be in [0, 1)")

    set_seed(SEED)
    train_path, val_path = DATA_DIR / "train", DATA_DIR / "val"
    if not train_path.is_dir() or not val_path.is_dir():
        raise FileNotFoundError(f"Expected train and val folders under {DATA_DIR}")
    # Validate only data used for fitting/model selection. The held-out test split is not loaded.
    validate_images(train_path)
    validate_images(val_path)

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_transform)
    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Training and validation class folders do not match")

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}; classes: {train_dataset.classes}")
    print(f"Samples: train={len(train_dataset)}, validation={len(val_dataset)}; test split not loaded")

    model = CustomCNN(len(train_dataset.classes), args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    run_dir = RUNS_DIR / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "epoch_metrics.csv"
    best_f1 = -1.0
    best_epoch = None
    best_val = None

    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_accuracy", "train_weighted_f1",
            "val_loss", "val_accuracy", "val_weighted_f1",
        ])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, criterion, device, optimizer)
            val = run_epoch(model, val_loader, criterion, device)
            writer.writerow({
                "epoch": epoch,
                "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                "train_weighted_f1": train["weighted_f1"],
                "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                "val_weighted_f1": val["weighted_f1"],
            })
            f.flush()
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss {train['loss']:.4f}, acc {train['accuracy']:.4f} | "
                f"val loss {val['loss']:.4f}, acc {val['accuracy']:.4f}, "
                f"weighted F1 {val['weighted_f1']:.4f}", flush=True,
            )
            if val["weighted_f1"] > best_f1:
                best_f1, best_epoch, best_val = val["weighted_f1"], epoch, val
                torch.save({
                    "state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "train_class_counts": [sum(y == i for y in train_dataset.targets) for i in range(len(train_dataset.classes))],
                    "settings": {"dropout": args.dropout, "image_size": list(IMAGE_SIZE)},
                }, run_dir / "best_model.pth")

    matrix, report, _ = confusion_and_report(
        best_val["labels"], best_val["predictions"], train_dataset.classes
    )
    results = {
        "experiment": args.experiment,
        "settings": {
            "epochs": args.epochs, "learning_rate": args.lr,
            "batch_size": args.batch_size, "dropout": args.dropout,
            "image_size": list(IMAGE_SIZE), "seed": SEED,
            "optimizer": "Adam", "loss": "CrossEntropyLoss",
        },
        "samples": {"train": len(train_dataset), "validation": len(val_dataset)},
        "classes": train_dataset.classes,
        "best_epoch_by_validation_weighted_f1": best_epoch,
        "best_validation": {k: best_val[k] for k in ("loss", "accuracy", "weighted_f1")},
        "validation_classification_report": report,
        "validation_confusion_matrix": matrix,
        "test_evaluated": False,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Best validation epoch: {best_epoch}; results saved to {run_dir}")


if __name__ == "__main__":
    main()
