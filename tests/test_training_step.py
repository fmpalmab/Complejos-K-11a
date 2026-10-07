"""Pruebas para pasos de entrenamiento y CLI."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from complejos_k.trainer import train_epoch_detectar, train_epoch_localizar
from complejos_k.models import CNNDETECTAR, CRNN_DETECTAR_LOCALIZAR
from complejos_k.cli import main


def test_train_epoch_detectar_step():
    model = CNNDETECTAR(in_channels=1, Nf=8, N1=16, N2=16)
    x = torch.randn(4, 1, 4000)
    y = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, acc = train_epoch_detectar(model, loader, criterion, optimizer, torch.device("cpu"))
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0


def test_train_epoch_localizar_step():
    model = CRNN_DETECTAR_LOCALIZAR(num_classes=1, in_channels=1, Nf=8, N1=16, N2=16)
    x = torch.randn(4, 1, 4000)
    y = torch.zeros(4, 1, 500)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, acc = train_epoch_localizar(model, loader, criterion, optimizer, torch.device("cpu"))
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0


def test_cli_synth():
    exit_code = main(["--synth", "--samples", "4"])
    assert exit_code == 0
