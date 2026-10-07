"""
Generador de señales EEG sintéticas fisiológicamente realistas con Complejos-K y Husos de Sueño.
Permite ejecutar y validar todos los experimentos y tests sin requerir datasets externos pesados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd

from complejos_k.wavelets import compute_cwt_feature, compute_zscore_feature
from complejos_k.config import SIGNAL_LENGTH, SYNTHETIC_DATA_PATH


def generate_single_eeg_epoch(
    fs: int = 200,
    length: int = SIGNAL_LENGTH,
    inject_k_complex: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Genera un registro EEG sintético de 20 segundos (4000 puntos a 200 Hz).
    Simula EEG de Sueño Fase 2 (NREM 2).
    
    Retorna:
        signal (np.ndarray): Señal de 4000 puntos.
        labels (np.ndarray): Vector binario de 4000 puntos (1 donde hay complejo-K).
        existe_k (int): 1 si contiene complejo-K, 0 en caso contrario.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, length / fs, length, endpoint=False)

    # 1. Ruido rosa de fondo (1/f)
    white_noise = rng.normal(0, 1.0, size=length)
    freqs = np.fft.rfftfreq(length, 1.0 / fs)
    fft_vals = np.fft.rfft(white_noise)
    # Ponderación 1/sqrt(f) para ruido rosa
    fft_vals[1:] /= np.sqrt(freqs[1:])
    background = np.fft.irfft(fft_vals, n=length)
    background = (background / (np.std(background) + 1e-8)) * 15.0  # ~15 uV RMS

    # 2. Ritmos fisiológicos de fondo: Delta (0.5-3.5 Hz) y Theta (4-7 Hz)
    delta_freq = rng.uniform(0.8, 2.5)
    theta_freq = rng.uniform(4.5, 6.5)
    eeg_waves = (
        12.0 * np.sin(2.0 * np.pi * delta_freq * t + rng.uniform(0, 2 * np.pi))
        + 8.0 * np.sin(2.0 * np.pi * theta_freq * t + rng.uniform(0, 2 * np.pi))
    )

    # 3. Huso de sueño de fondo aleatorio (Sleep Spindle, 12-14 Hz)
    spindle_t0 = rng.uniform(2.0, 16.0)
    spindle_freq = rng.uniform(12.0, 14.0)
    spindle_envelope = np.exp(-((t - spindle_t0) ** 2) / (2.0 * (0.4 ** 2)))
    spindle = 18.0 * spindle_envelope * np.sin(2.0 * np.pi * spindle_freq * t)

    signal = background + eeg_waves + spindle
    labels = np.zeros(length, dtype=np.float32)
    existe_k = 0

    if inject_k_complex:
        existe_k = 1
        # Ubicación central del Complejo-K (entre 4s y 15s)
        k_center = rng.uniform(4.0, 15.0)
        k_duration = rng.uniform(0.7, 1.2)  # Duración típica 0.7 - 1.2 segundos
        k_start = max(0.0, k_center - k_duration / 2.0)
        k_end = min(length / fs, k_center + k_duration / 2.0)

        start_idx = int(k_start * fs)
        end_idx = int(k_end * fs)
        labels[start_idx:end_idx] = 1.0

        # Morfología del Complejo-K:
        # Pico negativo agudo inicial (-100 a -150 uV)
        # Pico positivo lento posterior (+70 a +110 uV)
        t_rel = t - k_center
        # Primera onda negativa (más rápida, ancho ~0.2s)
        neg_wave = -rng.uniform(90.0, 140.0) * np.exp(-((t_rel + 0.15) ** 2) / (2.0 * (0.08 ** 2)))
        # Segunda onda positiva (más ancha, ancho ~0.35s)
        pos_wave = rng.uniform(60.0, 100.0) * np.exp(-((t_rel - 0.25) ** 2) / (2.0 * (0.16 ** 2)))

        k_complex_wave = neg_wave + pos_wave
        signal += k_complex_wave

    signal = signal.astype(np.float32)
    return signal, labels, existe_k


def generate_synthetic_eeg_dataset(
    n_samples: int = 120,
    k_complex_ratio: float = 0.5,
    compute_features: bool = True,
    output_path: Optional[Union[str, Path]] = SYNTHETIC_DATA_PATH,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un DataFrame completo con señales EEG, etiquetas y características CWT/ZETA.
    """
    rng = np.random.default_rng(seed)
    signals = []
    labels_list = []
    existe_k_list = []

    print(f"Generando {n_samples} registros sintéticos EEG (4000 puntos, {k_complex_ratio:.0%} con Complejo-K)...")

    for i in range(n_samples):
        has_k = bool(rng.random() < k_complex_ratio)
        sig, lbl, ex = generate_single_eeg_epoch(inject_k_complex=has_k, rng=rng)
        signals.append(sig.tolist())
        labels_list.append(lbl.tolist())
        existe_k_list.append(ex)

    df = pd.DataFrame(
        {
            "signal": signals,
            "labels": labels_list,
            "existeK": existe_k_list,
        }
    )

    if compute_features:
        print("Calculando características CWT y Z-Score...")
        zetas = []
        cwts = []
        for s in signals:
            arr = np.array(s, dtype=np.float32)
            zetas.append(compute_zscore_feature(arr).tolist())
            cwts.append(compute_cwt_feature(arr).tolist())
        df["zeta"] = zetas
        df["cwt"] = cwts

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_p, index=False)
        print(f"Dataset sintético guardado en: {out_p}")

    return df
