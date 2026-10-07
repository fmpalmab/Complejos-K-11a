"""
Compatibility shim for models.py.
Forwards to complejos_k.models.
"""

from complejos_k.models import (
    CNNDETECTAR,
    CNNDETECTAR_MLP,
    CRNN_DETECTAR_LOCALIZAR,
    CWT_CRNN_LOCALIZAR,
    ZETA_CRNN_LOCALIZAR,
    ENSEMBLE_LOCALIZAR,
    SEED_LOCALIZAR,
)

__all__ = [
    "CNNDETECTAR",
    "CNNDETECTAR_MLP",
    "CRNN_DETECTAR_LOCALIZAR",
    "CWT_CRNN_LOCALIZAR",
    "ZETA_CRNN_LOCALIZAR",
    "ENSEMBLE_LOCALIZAR",
    "SEED_LOCALIZAR",
]
