"""
Arquitecturas de Redes Neuronales Convolucionales y Recurrentes (CRNN)
para Detección y Localización de Complejos-K en señales EEG.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureExtractor(nn.Module):
    """Bloque extractor convolucional jerárquico 1D de 3 etapas."""

    def __init__(self, in_channels: int = 1, Nf: int = 32) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, Nf, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(Nf, Nf, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(Nf, 2 * Nf, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(2 * Nf, 2 * Nf, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(2 * Nf, 4 * Nf, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(4 * Nf, 4 * Nf, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class CNNDETECTAR(nn.Module):
    """Modelo CRNN para detección global (contiene o no complejo-K)."""

    def __init__(
        self,
        in_channels: int = 1,
        Nf: int = 32,
        N1: int = 128,
        N2: int = 128,
        p1: float = 0.5,
        p2: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = ConvFeatureExtractor(in_channels=in_channels, Nf=Nf)
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)
        self.blstm1 = nn.LSTM(4 * Nf, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(2 * N1, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Conv1d(2 * N1, N2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(N2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        logits_seq = self.classifier(x)
        logits = torch.mean(logits_seq, dim=2)  # GAP -> (B, 1)
        return logits


class CNNDETECTAR_MLP(nn.Module):
    """Modelo de detección con clasificador MLP denso final."""

    def __init__(
        self,
        in_channels: int = 1,
        Nf: int = 32,
        N1: int = 128,
        N2: int = 128,
        p1: float = 0.5,
        p2: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = ConvFeatureExtractor(in_channels=in_channels, Nf=Nf)
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)
        self.blstm1 = nn.LSTM(4 * Nf, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(2 * N1, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.mlp_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * N1 * 500, N2),
            nn.ReLU(),
            nn.Dropout(p=p2),
            nn.Linear(N2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x)
        x = self.dropout2(x)
        logits = self.mlp_classifier(x)
        return logits


class CRNN_DETECTAR_LOCALIZAR(nn.Module):
    """Modelo secuencia a secuencia para localización temporal (1 canal de entrada)."""

    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 1,
        Nf: int = 64,
        N1: int = 256,
        N2: int = 128,
        p1: float = 0.5,
        p2: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = ConvFeatureExtractor(in_channels=in_channels, Nf=Nf)
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)
        self.blstm1 = nn.LSTM(4 * Nf, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(2 * N1, N1, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Conv1d(2 * N1, N2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(N2, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x)
        x = self.dropout2(x)
        x = x.permute(0, 2, 1)
        logits = self.classifier(x)  # (B, num_classes, 500)
        return logits


class CWT_CRNN_LOCALIZAR(CRNN_DETECTAR_LOCALIZAR):
    """Modelo de localización con 2 canales de entrada (Señal + CWT)."""

    def __init__(self, num_classes: int = 1, Nf: int = 64, N1: int = 256, **kwargs) -> None:
        super().__init__(num_classes=num_classes, in_channels=2, Nf=Nf, N1=N1, **kwargs)


class ZETA_CRNN_LOCALIZAR(CRNN_DETECTAR_LOCALIZAR):
    """Modelo de localización con 2 canales de entrada (Señal + ZETA)."""

    def __init__(self, num_classes: int = 1, Nf: int = 64, N1: int = 256, **kwargs) -> None:
        super().__init__(num_classes=num_classes, in_channels=2, Nf=Nf, N1=N1, **kwargs)


class ENSEMBLE_LOCALIZAR(CRNN_DETECTAR_LOCALIZAR):
    """Modelo de ensamble y fusión multicanal (3 canales: Señal + CWT + ZETA)."""

    def __init__(self, num_classes: int = 1, Nf: int = 64, N1: int = 256, **kwargs) -> None:
        super().__init__(num_classes=num_classes, in_channels=3, Nf=Nf, N1=N1, **kwargs)


SEED_LOCALIZAR = CRNN_DETECTAR_LOCALIZAR
