"""Pruebas unitarias para el generador fisiológico de EEG y Complejos-K."""

import pytest
import numpy as np
from complejos_k.synthetic import generate_single_eeg_epoch, generate_synthetic_eeg_dataset


def test_generate_single_eeg_epoch_with_k():
    sig, lbl, ex = generate_single_eeg_epoch(fs=200, length=4000, inject_k_complex=True)
    assert len(sig) == 4000
    assert len(lbl) == 4000
    assert ex == 1
    assert lbl.sum() > 0.0  # Debe contener puntos con 1.0


def test_generate_single_eeg_epoch_without_k():
    sig, lbl, ex = generate_single_eeg_epoch(fs=200, length=4000, inject_k_complex=False)
    assert len(sig) == 4000
    assert len(lbl) == 4000
    assert ex == 0
    assert lbl.sum() == 0.0


def test_generate_synthetic_eeg_dataset_structure():
    df = generate_synthetic_eeg_dataset(n_samples=4, compute_features=True, output_path=None)
    assert len(df) == 4
    for col in ["signal", "labels", "existeK", "zeta", "cwt"]:
        assert col in df.columns
    assert len(df["signal"].iloc[0]) == 4000
    assert len(df["cwt"].iloc[0]) == 4000
