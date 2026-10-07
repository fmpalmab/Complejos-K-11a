"""Pruebas unitarias para métricas de detección y localización."""

import pytest
import numpy as np
from complejos_k.metrics import (
    compute_point_metrics,
    find_events_from_binary_mask,
    compute_event_overlap_iou,
    compute_event_based_metrics,
)


def test_compute_point_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_point_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_find_events_from_binary_mask():
    mask = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    events = find_events_from_binary_mask(mask)
    assert events == [(1, 4), (6, 7)]


def test_compute_event_overlap_iou():
    ev_a = (10, 20)
    ev_b = (15, 25)
    iou = compute_event_overlap_iou(ev_a, ev_b)
    # Intersección: 15..20 (5 puntos). Unión: 10..25 (15 puntos) -> 5/15 = 0.3333
    assert iou == pytest.approx(1.0 / 3.0)


def test_compute_event_based_metrics_perfect_match():
    y_true = np.zeros((2, 500))
    y_pred = np.zeros((2, 500))
    y_true[0, 50:100] = 1.0
    y_pred[0, 50:100] = 1.0

    ev_m = compute_event_based_metrics(y_true, y_pred, iou_threshold=0.5)
    assert ev_m["event_f1"] == 1.0
    assert ev_m["tp"] == 1.0
