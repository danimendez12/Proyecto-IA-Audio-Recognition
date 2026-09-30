from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from models.model_b import MobileNetStyleCNN
from preprocessing.augmentation import AugmentedDataset, SpecAugment, WaveformAugmentedDataset
from preprocessing.dataset import LABELS, MelSpectrogramDataset, cache_mel_spectrograms


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(dataset: Dataset[tuple[Tensor, int]], seed: int) -> tuple[Subset, Subset, Subset]:
    """Return deterministic 70/15/15 subsets of original clips."""
    total = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def create_loaders(
    dataset: MelSpectrogramDataset,
    batch_size: int,
    seed: int,
    augment_mode: str = "none",
    noise_dir: Path | None = None,
    snr_min: float = 5.0,
    snr_max: float = 20.0,
    max_shift_ms: float = 100.0,
    p_augment: float = 1.0,
    freq_mask_param: int = 8,
    time_mask_param: int = 16,
    num_masks: int = 2,
    use_pitch_shift: bool = False,
    use_time_stretch: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build loaders while applying augmentation only to original train indices."""
    train_set, val_set, test_set = split_dataset(dataset, seed)
    train_indices = train_set.indices

    if augment_mode == "specaugment":
        train_data: Dataset[tuple[Tensor, int]] = AugmentedDataset(
            train_set,
            transform=SpecAugment(freq_mask_param, time_mask_param, num_masks),
        )
    elif augment_mode == "full":
        train_data = WaveformAugmentedDataset(
            dataset,
            train_indices,
            noise_dir=noise_dir,
            snr_min=snr_min,
            snr_max=snr_max,
            max_shift_ms=max_shift_ms,
            p_augment=p_augment,
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_masks=num_masks,
            use_pitch_shift=use_pitch_shift,
            use_time_stretch=use_time_stretch,
        )
    elif augment_mode == "none":
        train_data = train_set
    else:
        raise ValueError(f"Unknown augment mode: {augment_mode}")

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def confusion_matrix(preds: Tensor, labels: Tensor, num_classes: int) -> Tensor:
    """Compute a rows-by-true-class confusion matrix."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for true, pred in zip(labels, preds):
        cm[int(true), int(pred)] += 1
    return cm


def per_class_f1_from_cm(cm: Tensor) -> list[float]:
    """Compute one-vs-rest F1 scores from a confusion matrix."""
    scores: list[float] = []
    for index in range(cm.shape[0]):
        tp = cm[index, index].item()
        fp = cm[:, index].sum().item() - tp
        fn = cm[index, :].sum().item() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return scores


def macro_f1_from_cm(cm: Tensor) -> float:
    """Compute macro F1 across all classes."""
    return float(sum(per_class_f1_from_cm(cm)) / cm.shape[0])


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    """Evaluate loss, accuracy, F1 and predictions for one split."""
    model.eval()
    total_loss = 0.0
    all_preds: list[Tensor] = []
    all_labels: list[Tensor] = []
    with torch.no_grad():
        for specs, labels in dataloader:
            specs, labels = specs.to(device), labels.to(device)
            logits = model(specs)
            total_loss += criterion(logits, labels).item() * specs.size(0)
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    cm = confusion_matrix(preds=preds, labels=labels, num_classes=len(LABELS))
    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": (preds == labels).float().mean().item(),
        "f1": macro_f1_from_cm(cm),
        "per_class_f1": per_class_f1_from_cm(cm),
        "cm": cm,
        "preds": preds,
        "labels": labels,
    }


def _checkpoint_config(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI values into a serializable complete checkpoint config."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    config: dict[str, Any],
    normalization: dict[str, float],
) -> None:
    """Save model state while retaining experiment and normalization metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "normalization": normalization,
            "labels": LABELS,
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train one run and return its best-validation and test summary."""
    set_seed(args.seed)
    if args.build_cache:
        cache_mel_spectrograms(data_dir=args.data_dir, cache_dir=args.cache_dir, n_mels=args.n_mels)
    dataset = MelSpectrogramDataset(args.cache_dir)
    noise_dir = args.noise_dir
    if args.augment_mode == "full" and noise_dir is None:
        noise_dir = args.data_dir / "_background_noise_"
    train_loader, val_loader, test_loader = create_loaders(
        dataset, args.batch_size, args.seed, args.augment_mode, noise_dir, args.snr_min, args.snr_max,
        args.max_shift_ms, args.p_augment, args.freq_mask_param, args.time_mask_param, args.num_masks,
        args.use_pitch_shift, args.use_time_stretch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetStyleCNN(num_classes=len(LABELS), width_mult=args.width_mult, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    config = _checkpoint_config(args)
    config["num_classes"] = len(LABELS)
    run = wandb.init(
        project=args.wandb_project, name=args.run_name, group=args.wandb_group, tags=args.wandb_tags,
        config=config, mode=args.wandb_mode,
    )
    wandb.log({"model/parameters": model.count_parameters(), "model/macs": model.estimate_macs()})
    best_val_f1 = -1.0
    best_gap = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = Path(args.checkpoint_path)
    normalization = {"mean": float(dataset.metadata["mean"]), "std": float(dataset.metadata["std"])}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(specs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * specs.size(0)
            running_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            running_total += labels.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = evaluate(model, val_loader, criterion, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics["f1"])
        elif scheduler is not None:
            scheduler.step()
        learning_rate = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch": epoch, "train/loss": train_loss, "train/accuracy": train_acc,
            "val/loss": val_metrics["loss"], "val/accuracy": val_metrics["accuracy"], "val/f1": val_metrics["f1"],
            "gap/accuracy": train_acc - val_metrics["accuracy"],
            "gap/loss": train_loss - val_metrics["loss"], "learning_rate": learning_rate,
        })
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["f1"])
            best_gap = train_acc - float(val_metrics["accuracy"])
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path, epoch, config, normalization)
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_metrics['loss']:.4f} val_acc: {val_metrics['accuracy']:.4f} val_f1: {val_metrics['f1']:.4f}"
        )
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            break

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    wandb.log({
        "test/loss": test_metrics["loss"], "test/accuracy": test_metrics["accuracy"], "test/f1": test_metrics["f1"],
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=test_metrics["labels"].numpy(), preds=test_metrics["preds"].numpy(), class_names=LABELS,
        ),
        "test/f1_per_class": dict(zip(LABELS, test_metrics["per_class_f1"])),
    })
    run.finish()
    return {
        "mode": args.augment_mode, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
        "gap": best_gap,
        "test_loss": test_metrics["loss"], "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1"], "parameters": model.count_parameters(), "config": config,
    }


def parse_args() -> argparse.Namespace:
    """Parse individual training options."""
    parser = argparse.ArgumentParser(description="Train speech command classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("./data/speech_commands_10"))
    parser.add_argument("--cache-dir", type=Path, default=Path("./data/spectrogram_cache"))
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="none")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--augment-mode", choices=["none", "specaugment", "full"], default="none")
    parser.add_argument("--augment-train", action="store_true", help="Deprecated alias for specaugment")
    parser.add_argument("--noise-dir", type=Path, default=None)
    parser.add_argument("--snr-min", type=float, default=5.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--max-shift-ms", type=float, default=100.0)
    parser.add_argument("--p-augment", type=float, default=1.0)
    parser.add_argument("--freq-mask-param", type=int, default=8)
    parser.add_argument("--time-mask-param", type=int, default=16)
    parser.add_argument("--num-masks", type=int, default=2)
    parser.add_argument("--use-pitch-shift", action="store_true")
    parser.add_argument("--use-time-stretch", action="store_true")
    parser.add_argument("--checkpoint-path", type=Path, default=Path("./checkpoints/best_model.pt"))
    parser.add_argument("--wandb-project", type=str, default="speech-commands-10")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()
    if args.augment_train and args.augment_mode == "none":
        args.augment_mode = "specaugment"
    return args


if __name__ == "__main__":
    train(parse_args())