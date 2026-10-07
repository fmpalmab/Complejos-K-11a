"""
complejos_k - Redes neuronales convolucionales y recurrentes (CRNN) para la detección y localización de Complejos-K en EEG.
"""

from complejos_k.config import *
from complejos_k.models import (
    CNNDETECTAR,
    CNNDETECTAR_MLP,
    CRNN_DETECTAR_LOCALIZAR,
    CWT_CRNN_LOCALIZAR,
    ZETA_CRNN_LOCALIZAR,
    ENSEMBLE_LOCALIZAR,
    SEED_LOCALIZAR,
)
from complejos_k.datasets import (
    SignalDatasetDetectar,
    SignalDatasetLocalizar,
    SignalDatasetLocalizar_CWT,
    SignalDatasetLocalizar_ZETA,
    SignalDatasetLocalizar_ALL,
)
from complejos_k.wavelets import ricker, cwt, compute_cwt_feature, compute_zscore_feature
from complejos_k.synthetic import generate_synthetic_eeg_dataset, generate_single_eeg_epoch
from complejos_k.metrics import compute_point_metrics, compute_event_based_metrics
from complejos_k.pipeline import run_experiment, get_or_create_dataset

__version__ = "2.0.0"
