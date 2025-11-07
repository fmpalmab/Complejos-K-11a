# cwtmodel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CWT_CRNN_LOCALIZAR(nn.Module):
    """
    Implementación del modelo CRNN para LOCALIZACIÓN (secuencia a secuencia).
    Modificado para aceptar 2 canales de entrada:
    1. La señal original.
    2. La transformada CWT (Continuous Wavelet Transform) como vector.
    """
    def __init__(self, in_channels=2, num_classes=1, Nf=64, N1=256, N2=128, p1=0.2, p2=0.5):
        """
        Inicializa las capas del modelo.
        'in_channels' se establece en 2 por defecto.
        """
        super(CWT_CRNN_LOCALIZAR, self).__init__()

        # --- Bloque Convolucional (Encoder) ---
        self.conv_block1 = nn.Sequential(
            # La primera capa Conv1d ahora acepta 'in_channels' (2)
            nn.Conv1d(in_channels, Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(Nf, Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # T -> T/2 (2000)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(Nf, 2*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(2*Nf, 2*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # T/2 -> T/4 (1000)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(2*Nf, 4*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(4*Nf, 4*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) # T/4 -> T/8 (500)
        )

        # --- Bloque Recurrente ---
        self.dropout1 = nn.Dropout(p=p1)
        self.blstm1 = nn.LSTM(input_size=4*Nf, hidden_size=N1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=p2)
        self.blstm2 = nn.LSTM(input_size=2*N1, hidden_size=N1, num_layers=1, batch_first=True, bidirectional=True)

        # --- Clasificador Final (Conv1D) ---
        self.classifier = nn.Sequential(
            nn.Conv1d(2*N1, N2, kernel_size=1), # Salida del LSTM (2*N1) a N2
            nn.ReLU(),
            nn.Conv1d(N2, num_classes, kernel_size=1) # De N2 al número de clases (1)
        )

    def forward(self, x):
        # x shape esperado: (B, 2, 4000) (Señal + CWT)

        # Encoder
        x = self.conv_block1(x) # -> (B, 64, 2000)
        x = self.conv_block2(x) # -> (B, 128, 1000)
        x = self.conv_block3(x) # -> (B, 256, 500)

        # Preparación para LSTMs: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1) # -> (B, 500, 256)

        # Bloque Recurrente
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x) # -> (B, 500, 512)

        # Preparación para el clasificador: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1) # -> (B, 512, 500)

        # Clasificador
        logits = self.classifier(x) # -> (B, num_classes, 500)

        return logits