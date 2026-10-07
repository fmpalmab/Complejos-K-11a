"""Pruebas unitarias para arquitecturas neuronales de PyTorch."""

import pytest
import torch
from complejos_k.models import (
    CNNDETECTAR,
    CNNDETECTAR_MLP,
    CRNN_DETECTAR_LOCALIZAR,
    CWT_CRNN_LOCALIZAR,
    ZETA_CRNN_LOCALIZAR,
    ENSEMBLE_LOCALIZAR,
)


def test_cnn_detectar_forward_shape():
    model = CNNDETECTAR(in_channels=1, Nf=8, N1=16, N2=16)
    x = torch.randn(2, 1, 4000)
    out = model(x)
    assert out.shape == (2, 1)


def test_cnn_detectar_mlp_forward_shape():
    model = CNNDETECTAR_MLP(in_channels=1, Nf=8, N1=16, N2=16)
    x = torch.randn(2, 1, 4000)
    out = model(x)
    assert out.shape == (2, 1)


def test_crnn_localizar_forward_shape():
    model = CRNN_DETECTAR_LOCALIZAR(num_classes=1, in_channels=1, Nf=8, N1=16, N2=16)
    x = torch.randn(2, 1, 4000)
    out = model(x)
    assert out.shape == (2, 1, 500)


def test_cwt_and_zeta_localizar_forward_shape():
    model_cwt = CWT_CRNN_LOCALIZAR(num_classes=1, Nf=8, N1=16, N2=16)
    x_2ch = torch.randn(2, 2, 4000)
    out_cwt = model_cwt(x_2ch)
    assert out_cwt.shape == (2, 1, 500)

    model_zeta = ZETA_CRNN_LOCALIZAR(num_classes=1, Nf=8, N1=16, N2=16)
    out_zeta = model_zeta(x_2ch)
    assert out_zeta.shape == (2, 1, 500)


def test_ensemble_localizar_forward_shape():
    model_ens = ENSEMBLE_LOCALIZAR(num_classes=1, Nf=8, N1=16, N2=16)
    x_3ch = torch.randn(2, 3, 4000)
    out_ens = model_ens(x_3ch)
    assert out_ens.shape == (2, 1, 500)
