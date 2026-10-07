"""
Compatibility shim for config.py.
Forwards to complejos_k.config.
"""

from complejos_k.config import (
    DEVICE,
    RUTA_DATOS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    PATIENCE,
    NUM_RUNS,
    Nf_CNN,
    N1_CNN,
    N2_CNN,
    Nf_LOC,
    N1_LOC,
    N2_LOC,
)

__all__ = [
    "DEVICE",
    "RUTA_DATOS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "EPOCHS",
    "PATIENCE",
    "NUM_RUNS",
    "Nf_CNN",
    "N1_CNN",
    "N2_CNN",
    "Nf_LOC",
    "N1_LOC",
    "N2_LOC",
]