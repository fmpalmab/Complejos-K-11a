"""Pruebas unitarias para datasets de PyTorch."""

import pytest
import numpy as np
import pandas as pd
from complejos_k.datasets import (
    SignalDatasetDetectar,
    SignalDatasetLocalizar,
    SignalDatasetLocalizar_CWT,
    SignalDatasetLocalizar_ZETA,
    SignalDatasetLocalizar_ALL,
)


@pytest.fixture
def dummy_dataframe():
    sig = [np.zeros(4000).tolist(), np.ones(4000).tolist()]
    lbl = [np.zeros(4000).tolist(), np.ones(4000).tolist()]
    ex = [0, 1]
    return pd.DataFrame(
        {
            "signal": sig,
            "labels": lbl,
            "existeK": ex,
            "cwt": sig,
            "zeta": sig,
        }
    )


def test_signal_dataset_detectar(dummy_dataframe):
    ds = SignalDatasetDetectar(dummy_dataframe)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (1, 4000)
    assert y.shape == (1,)


def test_signal_dataset_localizar_pooling(dummy_dataframe):
    ds = SignalDatasetLocalizar(dummy_dataframe)
    assert len(ds) == 2
    x, y = ds[0]
    assert x.shape == (1, 4000)
    assert y.shape == (1, 500)  # Reducido por MaxPool


def test_signal_dataset_multichannel(dummy_dataframe):
    ds_cwt = SignalDatasetLocalizar_CWT(dummy_dataframe)
    x_cwt, y_cwt = ds_cwt[0]
    assert x_cwt.shape == (2, 4000)
    assert y_cwt.shape == (1, 500)

    ds_ens = SignalDatasetLocalizar_ALL(dummy_dataframe)
    x_ens, y_ens = ds_ens[0]
    assert x_ens.shape == (3, 4000)
    assert y_ens.shape == (1, 500)
