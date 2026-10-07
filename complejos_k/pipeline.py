"""
Orquestador de experimentos y evaluación comparativa para modelos de Complejos-K.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from complejos_k.config import (
    DEVICE,
    DEFAULT_FEATURES_DATA,
    DEFAULT_RAW_DATA,
    SYNTHETIC_DATA_PATH,
    OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
)
from complejos_k.synthetic import generate_synthetic_eeg_dataset
from complejos_k.models import (
    CNNDETECTAR,
    CNNDETECTAR_MLP,
    CRNN_DETECTAR_LOCALIZAR,
    CWT_CRNN_LOCALIZAR,
    ZETA_CRNN_LOCALIZAR,
    ENSEMBLE_LOCALIZAR,
)
from complejos_k.datasets import (
    SignalDatasetDetectar,
    SignalDatasetLocalizar,
    SignalDatasetLocalizar_CWT,
    SignalDatasetLocalizar_ZETA,
    SignalDatasetLocalizar_ALL,
)
from complejos_k.trainer import run_training_loop
from complejos_k.metrics import compute_point_metrics, compute_event_based_metrics


def get_or_create_dataset(n_samples_synthetic: int = 60) -> pd.DataFrame:
    """Carga los datos existentes o genera un dataset fisiológico sintético si faltan."""
    if DEFAULT_FEATURES_DATA.is_file():
        print(f"Cargando dataset con características: {DEFAULT_FEATURES_DATA}")
        return pd.read_parquet(DEFAULT_FEATURES_DATA)
    if DEFAULT_RAW_DATA.is_file():
        print(f"Cargando dataset original: {DEFAULT_RAW_DATA}")
        return pd.read_parquet(DEFAULT_RAW_DATA)
    if SYNTHETIC_DATA_PATH.is_file():
        print(f"Cargando dataset sintético existente: {SYNTHETIC_DATA_PATH}")
        return pd.read_parquet(SYNTHETIC_DATA_PATH)

    print("No se encontró dataset local. Generando dataset EEG fisiológico sintético...")
    df = generate_synthetic_eeg_dataset(
        n_samples=n_samples_synthetic,
        k_complex_ratio=0.5,
        compute_features=True,
        output_path=SYNTHETIC_DATA_PATH,
    )
    return df


def run_experiment(
    model_name: str = "crnn",
    epochs: int = 5,
    batch_size: int = 16,
    n_synthetic_samples: int = 40,
    save_model: bool = True,
) -> Dict[str, Any]:
    """Ejecuta entrenamiento y validación de una arquitectura específica."""
    df = get_or_create_dataset(n_samples_synthetic=n_synthetic_samples)

    # Split train/val/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

    model_name = model_name.lower()
    is_localization = True

    if model_name in ("cnn", "1"):
        model = CNNDETECTAR()
        train_ds = SignalDatasetDetectar(train_df)
        val_ds = SignalDatasetDetectar(val_df)
        test_ds = SignalDatasetDetectar(test_df)
        is_localization = False
    elif model_name in ("mlp", "2"):
        model = CNNDETECTAR_MLP()
        train_ds = SignalDatasetDetectar(train_df)
        val_ds = SignalDatasetDetectar(val_df)
        test_ds = SignalDatasetDetectar(test_df)
        is_localization = False
    elif model_name in ("crnn", "seed", "3"):
        model = CRNN_DETECTAR_LOCALIZAR(num_classes=1, in_channels=1)
        train_ds = SignalDatasetLocalizar(train_df)
        val_ds = SignalDatasetLocalizar(val_df)
        test_ds = SignalDatasetLocalizar(test_df)
    elif model_name in ("cwt", "4"):
        model = CWT_CRNN_LOCALIZAR(num_classes=1)
        train_ds = SignalDatasetLocalizar_CWT(train_df)
        val_ds = SignalDatasetLocalizar_CWT(val_df)
        test_ds = SignalDatasetLocalizar_CWT(test_df)
    elif model_name in ("zeta", "5"):
        model = ZETA_CRNN_LOCALIZAR(num_classes=1)
        train_ds = SignalDatasetLocalizar_ZETA(train_df)
        val_ds = SignalDatasetLocalizar_ZETA(val_df)
        test_ds = SignalDatasetLocalizar_ZETA(test_df)
    elif model_name in ("ensemble", "all", "6"):
        model = ENSEMBLE_LOCALIZAR(num_classes=1)
        train_ds = SignalDatasetLocalizar_ALL(train_df)
        val_ds = SignalDatasetLocalizar_ALL(val_df)
        test_ds = SignalDatasetLocalizar_ALL(test_df)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ckpt_path = str(OUTPUT_DIR / f"best_model_{model_name}.pth") if save_model else None

    print(f"\nIniciando entrenamiento de '{model_name.upper()}' en {DEVICE} ({epochs} épocas)...")
    history = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        is_localization=is_localization,
        epochs=epochs,
        save_path=ckpt_path,
        device=DEVICE,
    )

    # Evaluación en test
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            preds = torch.sigmoid(model(inputs)).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)

    point_metrics = compute_point_metrics(targets_arr, preds_arr)
    event_metrics = {}
    if is_localization:
        # Extraer canales: (N, 1, 500) -> (N, 500)
        t_seq = targets_arr.squeeze(1)
        p_seq = (preds_arr.squeeze(1) >= 0.5).astype(float)
        event_metrics = compute_event_based_metrics(t_seq, p_seq)

    print(f"\n--- Resultados del Modelo {model_name.upper()} ---")
    print(f"Accuracy:  {point_metrics['accuracy']:.4f}")
    print(f"Precision: {point_metrics['precision']:.4f}")
    print(f"Recall:    {point_metrics['recall']:.4f}")
    print(f"F1-Score:  {point_metrics['f1']:.4f}")
    if event_metrics:
        print(f"Event F1:  {event_metrics['event_f1']:.4f} (IoU >= 0.2)")

    return {
        "model_name": model_name,
        "point_metrics": point_metrics,
        "event_metrics": event_metrics,
        "history": history,
    }
