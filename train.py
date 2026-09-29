from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from models.model_b import MobileNetStyleCNN
from preprocessing.augmentation import AugmentedDataset, SpecAugment
from preprocessing.dataset import LABELS, MelSpectrogramDataset, cache_mel_spectrograms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset: Dataset[tuple[Tensor, int]], seed: int) -> tuple[Subset, Subset, Subset]:
    total = len(dataset)
    train_len = int(total * 0.70)
    val_len = int(total * 0.15)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def create_loaders(
    dataset: Dataset[tuple[Tensor, int]],
    batch_size: int,
    seed: int,
    augment_train: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = split_dataset(dataset, seed)

    if augment_train:
        train_augmented = AugmentedDataset(train_set, transform=SpecAugment())
    else:
        train_augmented = train_set

    train_loader = DataLoader(train_augmented, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def confusion_matrix(preds: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true, pred in zip(labels, preds):
        cm[int(true), int(pred)] += 1
    return cm


def macro_f1_from_cm(cm: Tensor) -> float:
    f1_scores: list[float] = []
    for i in range(cm.shape[0]):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2 * precision * recall) / (precision + recall))
    return float(sum(f1_scores) / len(f1_scores))


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float | Tensor]:
    model.eval()
    total_loss = 0.0
    all_preds: list[Tensor] = []
    all_labels: list[Tensor] = []

    with torch.no_grad():
        for specs, labels in dataloader:
            specs = specs.to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * specs.size(0)
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    cm = confusion_matrix(preds=preds, labels=labels, num_classes=len(LABELS))

    accuracy = (preds == labels).float().mean().item()
    f1 = macro_f1_from_cm(cm)
    avg_loss = total_loss / len(dataloader.dataset)

    return {"loss": avg_loss, "accuracy": accuracy, "f1": f1, "cm": cm, "preds": preds, "labels": labels}


def save_checkpoint(model: nn.Module, path: Path, epoch: int, config: dict[str, int | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config, "labels": LABELS}, path)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.build_cache:
        cache_mel_spectrograms(data_dir=args.data_dir, cache_dir=args.cache_dir, n_mels=args.n_mels)

    dataset = MelSpectrogramDataset(args.cache_dir)
    train_loader, val_loader, test_loader = create_loaders(dataset, args.batch_size, args.seed, args.augment_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetStyleCNN(num_classes=len(LABELS), width_mult=args.width_mult, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run = wandb.init(
        project=args.wandb_project,
        config=vars(args),
        mode=args.wandb_mode,
    )

    best_val_f1 = -1.0
    checkpoint_path = Path(args.checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for specs, labels in train_loader:
            specs = specs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * specs.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_metrics = evaluate(model, val_loader, criterion, device)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "val/f1": val_metrics["f1"],
            }
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            checkpoint_config = {
                "width_mult": args.width_mult,
                "dropout": args.dropout,
                "n_mels": args.n_mels,
                "num_classes": len(LABELS),
            }
            save_checkpoint(model, checkpoint_path, epoch, checkpoint_config)

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f} val_f1: {val_metrics['f1']:.4f}"
        )

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    wandb.log(
        {
            "test/loss": test_metrics["loss"],
            "test/accuracy": test_metrics["accuracy"],
            "test/f1": test_metrics["f1"],
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_metrics["labels"].numpy(),
                preds=test_metrics["preds"].numpy(),
                class_names=LABELS,
            ),
        }
    )

    print(
        f"Test - loss: {test_metrics['loss']:.4f} "
        f"acc: {test_metrics['accuracy']:.4f} f1: {test_metrics['f1']:.4f}"
    )

    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train speech command classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/speech_commands_10"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./data/spectrogram_cache"))
    parser.add_argument("--build-cache", action="store_true", help="Build mel-spectrogram cache before training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--augment-train", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints/best_model.pt"))
    parser.add_argument("--wandb-project", type=str, default="speech-commands-10")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
