"""Pruebas unitarias para cálculo de wavelets y CWT."""

import pytest
import numpy as np
from complejos_k.wavelets import ricker, cwt, compute_cwt_feature, compute_zscore_feature


def test_ricker_wavelet():
    points = 101
    w = ricker(points, a=5.0)
    assert len(w) == points
    # Simetría par alrededor del centro
    center = points // 2
    assert w[center - 10] == pytest.approx(w[center + 10], rel=1e-4)


def test_cwt_output_shape():
    data = np.sin(np.linspace(0, 10, 200))
    widths = [2.0, 4.0, 8.0]
    res = cwt(data, ricker, widths)
    assert res.shape == (3, 200)


def test_compute_cwt_and_zscore_features():
    signal = np.random.randn(500).astype(np.float32)
    cwt_feat = compute_cwt_feature(signal, widths=[1.0, 2.0, 4.0])
    assert len(cwt_feat) == len(signal)
    assert not np.isnan(cwt_feat).any()

    zeta_feat = compute_zscore_feature(signal)
    assert len(zeta_feat) == len(signal)
    assert abs(float(np.mean(zeta_feat))) < 1e-4
    assert abs(float(np.std(zeta_feat)) - 1.0) < 1e-3
