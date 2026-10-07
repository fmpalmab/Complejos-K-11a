"""
Datasets de PyTorch para detección y localización de Complejos-K en señales EEG.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class SignalDatasetDetectar(Dataset):
    """Dataset para la tarea de DETECCIÓN global binaria (contiene o no complejo-K)."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.signals = torch.tensor(np.array(dataframe["signal"].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(dataframe["existeK"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        signal = self.signals[idx].unsqueeze(0)  # (1, 4000)
        label = self.labels[idx].unsqueeze(0)    # (1,)
        return signal, label


class SignalDatasetLocalizar(Dataset):
    """Dataset para LOCALIZACIÓN de secuencia a secuencia (1 canal -> 500 puntos tras MaxPool)."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.signals = torch.tensor(np.array(dataframe["signal"].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(np.array(dataframe["labels"].tolist()), dtype=torch.float32)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        signal = self.signals[idx].unsqueeze(0)          # (1, 4000)
        label_raw = self.labels[idx].unsqueeze(0)        # (1, 4000)
        label_out = self.pool(label_raw)                 # (1, 500)
        return signal, label_out


class SignalDatasetLocalizar_CWT(Dataset):
    """Dataset para LOCALIZACIÓN con 2 canales: Señal + Transformada Wavelet (CWT)."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.signals = torch.tensor(np.array(dataframe["signal"].tolist()), dtype=torch.float32)
        self.cwt_features = torch.tensor(np.array(dataframe["cwt"].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(np.array(dataframe["labels"].tolist()), dtype=torch.float32)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        s = self.signals[idx].unsqueeze(0)
        c = self.cwt_features[idx].unsqueeze(0)
        stacked = torch.cat((s, c), dim=0)               # (2, 4000)
        label_out = self.pool(self.labels[idx].unsqueeze(0))  # (1, 500)
        return stacked, label_out


class SignalDatasetLocalizar_ZETA(Dataset):
    """Dataset para LOCALIZACIÓN con 2 canales: Señal + Normalización Z-Score."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.signals = torch.tensor(np.array(dataframe["signal"].tolist()), dtype=torch.float32)
        self.zeta_features = torch.tensor(np.array(dataframe["zeta"].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(np.array(dataframe["labels"].tolist()), dtype=torch.float32)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        s = self.signals[idx].unsqueeze(0)
        z = self.zeta_features[idx].unsqueeze(0)
        stacked = torch.cat((s, z), dim=0)               # (2, 4000)
        label_out = self.pool(self.labels[idx].unsqueeze(0))  # (1, 500)
        return stacked, label_out


class SignalDatasetLocalizar_ALL(Dataset):
    """Dataset para modelo ENSEMBLE con 3 canales: Señal + CWT + ZETA."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.signals = torch.tensor(np.array(dataframe["signal"].tolist()), dtype=torch.float32)
        self.cwt_features = torch.tensor(np.array(dataframe["cwt"].tolist()), dtype=torch.float32)
        self.zeta_features = torch.tensor(np.array(dataframe["zeta"].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(np.array(dataframe["labels"].tolist()), dtype=torch.float32)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        s = self.signals[idx].unsqueeze(0)
        c = self.cwt_features[idx].unsqueeze(0)
        z = self.zeta_features[idx].unsqueeze(0)
        stacked = torch.cat((s, c, z), dim=0)            # (3, 4000)
        label_out = self.pool(self.labels[idx].unsqueeze(0))  # (1, 500)
        return stacked, label_out
