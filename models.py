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
    

class ConvMDB(nn.Module):
    """
    Implementación del Bloque Convolucional Multi-Resolución Dilatado (MDB)
    de la arquitectura SEED (Imagen c).
    
    Toma `in_channels` y `out_channels` (F) y crea 4 ramas paralelas
    con diferentes dilataciones y distribuciones de canales:
    - Rama 1 (d=8): F/8 canales
    - Rama 2 (d=4): F/8 canales
    - Rama 3 (d=2): F/4 canales
    - Rama 4 (d=1): F/2 canales
    
    Las salidas se concatenan para producir `out_channels` (F).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvMDB, self).__init__()
        
        # F (out_channels) debe ser divisible por 8
        if out_channels % 8 != 0:
            raise ValueError(f"out_channels (F) debe ser divisible por 8. Se recibió: {out_channels}")
            
        # Cálculo de canales para cada rama
        c_f8 = out_channels // 8
        c_f4 = out_channels // 4
        c_f2 = out_channels // 2
        
        # Cálculo de padding para mantener la longitud (padding='same')
        # padding = (dilation * (kernel_size - 1)) / 2
        p_d1 = (1 * (kernel_size - 1)) // 2 # padding = 1
        p_d2 = (2 * (kernel_size - 1)) // 2 # padding = 2
        p_d4 = (4 * (kernel_size - 1)) // 2 # padding = 4
        p_d8 = (8 * (kernel_size - 1)) // 2 # padding = 8

        # Rama 1 (dilation=8, F/8)
        self.branch_d8 = nn.Sequential(
            nn.Conv1d(in_channels, c_f8, kernel_size, padding=p_d8, dilation=8),
            nn.ReLU(),
            nn.Conv1d(c_f8, c_f8, kernel_size, padding=p_d8, dilation=8),
            nn.ReLU()
        )
        
        # Rama 2 (dilation=4, F/8)
        self.branch_d4 = nn.Sequential(
            nn.Conv1d(in_channels, c_f8, kernel_size, padding=p_d4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(c_f8, c_f8, kernel_size, padding=p_d4, dilation=4),
            nn.ReLU()
        )
        
        # Rama 3 (dilation=2, F/4)
        self.branch_d2 = nn.Sequential(
            nn.Conv1d(in_channels, c_f4, kernel_size, padding=p_d2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(c_f4, c_f4, kernel_size, padding=p_d2, dilation=2),
            nn.ReLU()
        )
        
        # Rama 4 (dilation=1, F/2)
        self.branch_d1 = nn.Sequential(
            nn.Conv1d(in_channels, c_f2, kernel_size, padding=p_d1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(c_f2, c_f2, kernel_size, padding=p_d1, dilation=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Ejecutar cada rama en paralelo
        out_d8 = self.branch_d8(x)
        out_d4 = self.branch_d4(x)
        out_d2 = self.branch_d2(x)
        out_d1 = self.branch_d1(x)
        
        # Concatenar las salidas en la dimensión de canales (dim=1)
        out = torch.cat([out_d8, out_d4, out_d2, out_d1], dim=1)
        return out
    

class SEED_LOCALIZAR(nn.Module):
    """
    Implementación de la arquitectura SEED para LOCALIZACIÓN (Imagen b).
    
    Utiliza los bloques ConvMDB para la extracción de características.
    Los parámetros por defecto son los mismos de CRNN_DETECTAR_LOCALIZAR
    para facilitar el intercambio.
    """
    def __init__(self, in_channels=1, num_classes=1, Nf=64, N1=256, N2=128, p1=0.2, p2=0.5):
        super(SEED_LOCALIZAR, self).__init__()
        
        # --- Etapa 1: Codificación Local (Encoder) ---
        
        # Batchnorm inicial
        self.initial_batchnorm = nn.BatchNorm1d(in_channels)
        
        # Bloque Conv 1 (estándar)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, Nf, kernel_size=3, padding=1), # k=3, d=1 -> p=1
            nn.ReLU(),
            nn.Conv1d(Nf, Nf, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # T -> T/2
        
        # Bloque Conv 2 (MDB)
        self.mdb_block2 = ConvMDB(in_channels=Nf, out_channels=2*Nf, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # T/2 -> T/4
        
        # Bloque Conv 3 (MDB)
        self.mdb_block3 = ConvMDB(in_channels=2*Nf, out_channels=4*Nf, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # T/4 -> T/8 (500)

        # --- Etapa 2: Contextualización (Recurrente) ---
        self.dropout1 = nn.Dropout(p=p1) # (drop p1)
        self.blstm1 = nn.LSTM(input_size=4*Nf, hidden_size=N1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(p=p2) # (drop p2)
        self.blstm2 = nn.LSTM(input_size=2*N1, hidden_size=N1, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(p=p2) # (segundo drop p2)

        # --- Etapa 3: Clasificación ---
        # (Idéntica a CRNN_DETECTAR_LOCALIZAR)
        self.classifier = nn.Sequential(
            nn.Conv1d(2*N1, N2, kernel_size=1), # Salida del LSTM (2*N1) a N2
            nn.ReLU(),
            nn.Conv1d(N2, num_classes, kernel_size=1) # De N2 al número de clases (1)
        )

    def forward(self, x):
        # x shape: (B, 1, 4000)
        
        # Etapa de Codificación
        x = self.initial_batchnorm(x)
        x = self.conv_block1(x) # -> (B, Nf, 4000)
        x = self.pool1(x)       # -> (B, Nf, 2000)
        
        x = self.mdb_block2(x)   # -> (B, 2*Nf, 2000)
        x = self.pool2(x)       # -> (B, 2*Nf, 1000)
        
        x = self.mdb_block3(x)   # -> (B, 4*Nf, 1000)
        x = self.pool3(x)       # -> (B, 4*Nf, 500)

        # Preparación para LSTMs: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1) # -> (B, 500, 4*Nf)

        # Etapa de Contextualización
        x = self.dropout1(x)
        x, _ = self.blstm1(x)
        x = self.dropout2(x)
        x, _ = self.blstm2(x) # -> (B, 500, 2*N1)
        x = self.dropout3(x)

        # Preparación para el clasificador: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1) # -> (B, 2*N1, 500)

        # Etapa de Clasificación
        logits = self.classifier(x) # -> (B, num_classes, 500)

        return logits
    
