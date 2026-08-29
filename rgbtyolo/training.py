from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW


@dataclass(frozen=True)
class TrainingStage:
    epochs: int
    learning_rate: float
    patience: int = 10


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        if value < self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def create_optimizer(model, *, learning_rate, weight_decay=5e-4):
    return AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    count = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        thermal = batch["thermal"].to(device)
        targets = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"rgb", "thermal"}
        }

        optimizer.zero_grad(set_to_none=True)
        predictions = model(rgb, thermal)
        loss, _ = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total += float(loss.detach()) * rgb.shape[0]
        count += rgb.shape[0]

    return total / max(count, 1)


@torch.inference_mode()
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0

    for batch in loader:
        rgb = batch["rgb"].to(device)
        thermal = batch["thermal"].to(device)
        targets = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in {"rgb", "thermal"}
        }

        predictions = model(rgb, thermal)
        loss, _ = criterion(predictions, targets)

        total += float(loss.detach()) * rgb.shape[0]
        count += rgb.shape[0]

    return total / max(count, 1)


def fit_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    *,
    device,
    stage,
    checkpoint_path,
    weight_decay=5e-4,
):
    optimizer = create_optimizer(
        model,
        learning_rate=stage.learning_rate,
        weight_decay=weight_decay,
    )
    stopper = EarlyStopping(stage.patience)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, stage.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss = validate_epoch(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        if val_loss < stopper.best:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)

        if stopper.update(val_loss):
            break

    return history
