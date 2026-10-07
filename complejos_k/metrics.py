"""
Métricas de evaluación punto a punto y basadas en eventos para Complejos-K.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas binarias estándar punto a punto."""
    y_t = (np.asarray(y_true).flatten() > 0.5).astype(int)
    y_p = (np.asarray(y_pred).flatten() > 0.5).astype(int)

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def find_events_from_binary_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Encuentra intervalos contiguos [inicio, fin) en una máscara binaria 1D."""
    events = []
    in_event = False
    start = 0
    for idx, val in enumerate(mask):
        if val > 0.5 and not in_event:
            in_event = True
            start = idx
        elif val <= 0.5 and in_event:
            in_event = False
            events.append((start, idx))
    if in_event:
        events.append((start, len(mask)))
    return events


def compute_event_overlap_iou(event_a: Tuple[int, int], event_b: Tuple[int, int]) -> float:
    """Calcula la intersección sobre la unión (IoU) entre dos intervalos."""
    start_i = max(event_a[0], event_b[0])
    end_i = min(event_a[1], event_b[1])
    intersection = max(0, end_i - start_i)
    union = (event_a[1] - event_a[0]) + (event_b[1] - event_b[0]) - intersection
    return intersection / union if union > 0 else 0.0


def compute_event_based_metrics(
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    iou_threshold: float = 0.2,
) -> Dict[str, float]:
    """
    Calcula precisión, recall y F1 basados en eventos con solapamiento IoU.
    y_true_seq y y_pred_seq son matrices (N_muestras, T_puntos).
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i in range(len(y_true_seq)):
        true_events = find_events_from_binary_mask(y_true_seq[i])
        pred_events = find_events_from_binary_mask(y_pred_seq[i])

        matched_true = set()
        matched_pred = set()

        for p_idx, p_ev in enumerate(pred_events):
            for t_idx, t_ev in enumerate(true_events):
                if t_idx in matched_true:
                    continue
                iou = compute_event_overlap_iou(p_ev, t_ev)
                if iou >= iou_threshold:
                    matched_true.add(t_idx)
                    matched_pred.add(p_idx)
                    break

        tp = len(matched_pred)
        fp = len(pred_events) - tp
        fn = len(true_events) - len(matched_true)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "event_precision": prec,
        "event_recall": rec,
        "event_f1": f1,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }
