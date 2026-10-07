"""
Módulo de entrenamiento, optimización y validación cruzada para modelos CRNN de EEG.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from complejos_k.config import DEVICE, LEARNING_RATE, PATIENCE


def train_epoch_detectar(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Entrena una época para la tarea de detección binaria."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_detectar(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evalúa el modelo de detección binaria."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def train_epoch_localizar(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Entrena una época para localización secuencia a secuencia."""
    model.train()
    running_loss = 0.0
    correct = 0
    total_elements = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total_elements += labels.numel()

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    epoch_acc = correct / max(1, total_elements)
    return epoch_loss, epoch_acc


def evaluate_localizar(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evalúa el modelo de localización temporal."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total_elements = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total_elements += labels.numel()

    epoch_loss = running_loss / max(1, len(dataloader.dataset))
    epoch_acc = correct / max(1, total_elements)
    return epoch_loss, epoch_acc


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    is_localization: bool = False,
    epochs: int = 10,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """Ejecuta un ciclo completo de entrenamiento con Early Stopping."""
    if device is None:
        device = DEVICE

    model = model.to(device)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    train_fn = train_epoch_localizar if is_localization else train_epoch_detectar
    eval_fn = evaluate_localizar if is_localization else evaluate_detectar

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_fn(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = eval_fn(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return history
