"""
Compatibility shim for datasets.py.
Forwards to complejos_k.datasets.
"""

from complejos_k.datasets import (
    SignalDatasetDetectar,
    SignalDatasetLocalizar,
    SignalDatasetLocalizar_CWT,
    SignalDatasetLocalizar_ZETA,
    SignalDatasetLocalizar_ALL,
)

__all__ = [
    "SignalDatasetDetectar",
    "SignalDatasetLocalizar",
    "SignalDatasetLocalizar_CWT",
    "SignalDatasetLocalizar_ZETA",
    "SignalDatasetLocalizar_ALL",
]