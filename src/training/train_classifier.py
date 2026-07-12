from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import ImageClassificationDataset
from src.data.transforms import get_classifier_transforms
from src.models.classifier import ResNet18Classifier
from src.utils.device import get_device
from src.utils.reproducibility import set_seed


def _resolve_path(project_root: Path, value: str) -> Path:
    """Resolve a configuration path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def build_classification_loaders(
    split_dir: Path,
    class_names: Sequence[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation loaders from the committed split CSVs."""
    transform_map = get_classifier_transforms(image_size=image_size)
    train_dataset = ImageClassificationDataset(
        split_csv=split_dir / "train_split.csv",
        class_names=class_names,
        transform=transform_map["train"],
    )
    validation_dataset = ImageClassificationDataset(
        split_csv=split_dir / "val_split.csv",
        class_names=class_names,
        transform=transform_map["validation"],
    )

    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader


def _binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute thresholded binary metrics for one epoch."""
    predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
    targets = labels.to(torch.int64)
    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (targets == 0)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())
    total = max(int(targets.numel()), 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (true_positive + true_negative) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None
    model.train(is_training)
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=amp_enabled)
        labels = labels.to(device, non_blocking=amp_enabled)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if is_training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.detach().item() * labels.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = _binary_metrics(logits, labels)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_loss: float,
    history: list[Dict[str, float]],
) -> None:
    """Save all state needed to resume classifier training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "history": history,
        },
        path,
    )


def train_classifier(
    model: ResNet18Classifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_dir: Path,
    resume_from: Optional[Path] = None,
    freeze_backbone_epochs: int = 0,
) -> list[Dict[str, float]]:
    """Train Model A using training data and validation-only checkpoint selection."""
    model.to(device)
    if freeze_backbone_epochs > 0:
        model.freeze_backbone()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[Dict[str, float]] = []
    start_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    if resume_from is not None and resume_from.exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])
        if history:
            best_epoch_index = min(
                range(len(history)),
                key=lambda index: history[index]["validation_loss"],
            )
            epochs_without_improvement = len(history) - best_epoch_index - 1

    if start_epoch >= freeze_backbone_epochs:
        model.unfreeze_backbone()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        if epoch == freeze_backbone_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.trainable_parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=2,
            )

        train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler
        )
        validation_metrics = _run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_metrics["loss"])
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": 0.0,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        epoch_metrics["epoch_seconds"] = time.perf_counter() - epoch_start
        history.append(epoch_metrics)
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_validation_loss,
                history,
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_validation_loss,
            history,
        )

        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train accuracy {train_metrics['accuracy']:.4f} | "
            f"validation loss {validation_metrics['loss']:.4f} | "
            f"validation accuracy {validation_metrics['accuracy']:.4f} | "
            f"validation F1 {validation_metrics['f1']:.4f} | "
            f"time {epoch_metrics['epoch_seconds']:.1f}s"
        )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    best_checkpoint = checkpoint_dir / "best.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    return history


def train_from_config(
    config_path: Path,
    project_root: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> Tuple[ResNet18Classifier, list[Dict[str, float]]]:
    """Build and train Model A from the project YAML configuration."""
    config_path = Path(config_path).resolve()
    project_root = (project_root or config_path.parent.parent).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    set_seed(int(config["seed"]))
    class_names = config["dataset"]["classes"]
    training = config["training"]
    classifier_config = config["model"]["classifier"]
    train_loader, validation_loader = build_classification_loaders(
        split_dir=_resolve_path(project_root, config["paths"]["splits_dir"]),
        class_names=class_names,
        image_size=int(training["image_size"]),
        batch_size=int(training["batch_size"]),
        num_workers=int(training["num_workers"]),
    )
    model = ResNet18Classifier(
        pretrained=bool(classifier_config["pretrained"]),
        dropout=float(classifier_config["classifier_dropout"]),
    )
    history = train_classifier(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=get_device(),
        epochs=int(training["classifier_epochs"]),
        learning_rate=float(classifier_config["learning_rate"]),
        weight_decay=float(classifier_config["weight_decay"]),
        patience=int(training["early_stopping_patience"]),
        checkpoint_dir=_resolve_path(project_root, config["paths"]["model_dir"]) / "model_a",
        resume_from=resume_from,
        freeze_backbone_epochs=int(training.get("classifier_freeze_epochs", 0)),
    )
    history_path = _resolve_path(project_root, config["paths"]["logs_dir"]) / "model_a_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model A on original images.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(
        config_path=args.config,
        project_root=args.project_root,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
