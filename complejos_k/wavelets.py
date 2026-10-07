"""
Implementación de la Transformada Wavelet Continua (CWT) y normalización Z-score
compatible con SciPy 1.12+ (sin dependencias obsoletas).
"""

from __future__ import annotations

from typing import Union, Sequence, Optional
import numpy as np
from scipy.signal import convolve


def ricker(points: int, a: float) -> np.ndarray:
    """
    Genera una wavelet Ricker (Sombrero Mexicano).
    
    Args:
        points: Número de puntos de la wavelet.
        a: Parámetro de escala (ancho).
    """
    a = float(a)
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2.0
    xsq = vec ** 2
    mod = 1.0 - (xsq / wsq)
    gauss = np.exp(-xsq / (2.0 * wsq))
    return A * mod * gauss


def cwt(data: np.ndarray, wavelet_func, widths: Sequence[float]) -> np.ndarray:
    """
    Calcula la Transformada Wavelet Continua (CWT) 1D.
    
    Args:
        data: Señal 1D de entrada.
        wavelet_func: Función generadora de la wavelet (ej: ricker).
        widths: Secuencia de escalas/anchos.
    
    Returns:
        Matriz 2D de dimensiones (len(widths), len(data)).
    """
    data = np.asarray(data, dtype=np.float64)
    output = np.empty((len(widths), len(data)), dtype=np.float64)
    
    for i, width in enumerate(widths):
        # Longitud adaptativa para capturar el soporte de la wavelet
        n_points = min(10 * int(width) + 1, len(data))
        if n_points % 2 == 0:
            n_points += 1
        wavelet_data = wavelet_func(n_points, width)
        # Convolución con modo 'same'
        conv = convolve(data, wavelet_data, mode="same")
        output[i, :] = conv
        
    return output


def compute_cwt_feature(signal: np.ndarray, widths: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Calcula la CWT sobre la señal y la colapsa promediando la energía a través de las escalas.
    Retorna un vector 1D con la misma longitud que la señal.
    """
    if widths is None:
        widths = np.arange(1, 31)
    signal_arr = np.asarray(signal, dtype=np.float64)
    cwt_mat = cwt(signal_arr, ricker, widths)
    cwt_feat = np.mean(np.abs(cwt_mat), axis=0).astype(np.float32)
    return cwt_feat


def compute_zscore_feature(signal: np.ndarray) -> np.ndarray:
    """Calcula el Z-Score de la señal (media 0, desviación estándar 1)."""
    signal_arr = np.asarray(signal, dtype=np.float64)
    std = float(np.std(signal_arr))
    if std < 1e-8:
        return np.zeros_like(signal_arr, dtype=np.float32)
    mean = float(np.mean(signal_arr))
    return ((signal_arr - mean) / std).astype(np.float32)
