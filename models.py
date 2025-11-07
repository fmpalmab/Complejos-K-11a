# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDETECTAR(nn.Module):
    """
    Modelo CRNN para la DETECCIÓN binaria (etiqueta global).
    Colapsa la salida temporal con Global Average Pooling.
    """
    def __init__(self, in_channels=1, Nf=32, N1=128, N2=128, p1=0.5, p2=0.5):
        """
        Inicializa las capas del modelo.
        """
        super(CNNDETECTAR, self).__init__()

        # --- Extractor de Características Convolucional ---
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=Nf, out_channels=Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=Nf, out_channels=2*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*Nf, out_channels=2*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=2*Nf, out_channels=4*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=4*Nf, out_channels=4*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # --- Capas de Dropout ---
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)

        # --- Bloques Recurrentes (Bidirectional LSTMs) ---
        self.blstm1 = nn.LSTM(input_size=4*Nf, hidden_size=N1, num_layers=1, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(input_size=2*N1, hidden_size=N1, num_layers=1, batch_first=True, bidirectional=True)

        # --- Clasificador (Conv1D) ---
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2*N1, out_channels=N2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=N2, out_channels=1, kernel_size=1) # Salida de 1 canal
        )

    def forward(self, x):
        """
        Define el forward pass de la red.
        """
        # Bloques convolucionales
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Reshape para LSTMs: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        # Bloques recurrentes
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x)
        x = self.dropout2(x)

        # Reshape para Clasificador: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)

        # Clasificador
        logits_seq = self.classifier(x) # Shape: (B, 1, 500)

        # Global Average Pooling para colapsar la dimensión temporal
        logits = torch.mean(logits_seq, dim=2) # Shape: (B, 1)

        return logits


class CNNDETECTAR_MLP(nn.Module):
    """
    Modelo CRNN para DETECCIÓN binaria con un clasificador MLP final.
    """
    def __init__(self, in_channels=1, Nf=32, N1=128, N2=128, p1=0.5, p2=0.5):
        super(CNNDETECTAR_MLP, self).__init__()

        # --- Extractor de Características Convolucional (Sin cambios) ---
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=Nf, out_channels=Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=Nf, out_channels=2*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=2*Nf, out_channels=2*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=2*Nf, out_channels=4*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=4*Nf, out_channels=4*Nf, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # --- Capas de Dropout (Sin cambios) ---
        self.dropout1 = nn.Dropout(p=p1)
        self.dropout2 = nn.Dropout(p=p2)

        # --- Bloques Recurrentes (Bi-LSTM, Sin cambios) ---
        self.blstm1 = nn.LSTM(input_size=4*Nf, hidden_size=N1, batch_first=True, bidirectional=True)
        self.blstm2 = nn.LSTM(input_size=2*N1, hidden_size=N1, batch_first=True, bidirectional=True)

        # --- Clasificador MLP ---
        # Opera sobre la salida de la LSTM (2*N1 características)
        self.classifier_mlp = nn.Sequential(
            nn.Linear(in_features=2*N1, out_features=N2),
            nn.ReLU(),
            nn.Linear(in_features=N2, out_features=1) # Salida de 1 para clasificación binaria
        )

    def forward(self, x):
        """
        Define el forward pass con el nuevo clasificador MLP.
        """
        # Bloques convolucionales
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Reshape para LSTMs: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        # Bloques recurrentes
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x) # Shape: (B, 500, 2*N1)
        x = self.dropout2(x)

        # Clasificador MLP (opera en cada paso de tiempo)
        logits_seq = self.classifier_mlp(x) # Shape: (B, 500, 1)

        # Global Average Pooling (promedio sobre la dimensión temporal)
        logits = torch.mean(logits_seq, dim=1) # Shape: (B, 1)

        return logits


class CRNN_DETECTAR_LOCALIZAR(nn.Module):
    """
    Implementación del modelo CRNN para LOCALIZACIÓN (secuencia a secuencia).
    """
    def __init__(self, in_channels=1, num_classes=1, Nf=64, N1=256, N2=128, p1=0.2, p2=0.5):
        super(CRNN_DETECTAR_LOCALIZAR, self).__init__()

        # --- Bloque Convolucional (Encoder) ---
        self.conv_block1 = nn.Sequential(
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
        # x shape: (B, 1, 4000)

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