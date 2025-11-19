# src/datasets.py

import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn

class SignalDatasetDetectar(Dataset):
    """
    Dataset de PyTorch para la tarea de DETECCIÓN.
    
    Esta clase toma un dataframe, extrae las columnas 'signal' y 'existeK',
    y las formatea en tensores de PyTorch.
    
    Atributos:
        signals (torch.Tensor): Tensor que contiene todas las señales.
        labels (torch.Tensor): Tensor que contiene las etiquetas binarias (0 o 1).
    """
    def __init__(self, dataframe):
        """
        Inicializa el Dataset.
        
        Args:
            dataframe (pd.DataFrame): Dataframe que contiene las columnas
                                      'signal' y 'existeK'.
        """
        self.signals = torch.tensor(np.array(dataframe['signal'].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(dataframe['existeK'].values, dtype=torch.float32)

    def __len__(self):
        """Devuelve el número total de muestras en el dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Obtiene una única muestra (señal y etiqueta) del dataset.
        
        Args:
            idx (int): El índice de la muestra a obtener.
            
        Returns:
            tuple: Una tupla conteniendo:
                   - signal (torch.Tensor): La señal con una dimensión de canal añadida.
                                            Shape: (1, 4000)
                   - label (torch.Tensor): La etiqueta con una dimensión añadida.
                                           Shape: (1)
        """
        signal = self.signals[idx].unsqueeze(0)
        label = self.labels[idx].unsqueeze(0)
        return signal, label


class SignalDatasetLocalizar(Dataset):
    """
    Dataset de PyTorch para la tarea de LOCALIZACIÓN (secuencia a secuencia).
    
    Esta clase procesa las etiquetas de 4000 puntos, aplicándoles un Max-Pooling
    para que su dimensión coincida con la salida de la red neuronal (500 puntos).
    
    Atributos:
        signals (torch.Tensor): Tensor que contiene todas las señales.
        labels (torch.Tensor): Tensor que contiene las etiquetas de secuencia originales.
        pool (nn.MaxPool1d): Capa de pooling para reducir la dimensión de las etiquetas.
    """
    def __init__(self, dataframe):
        """
        Inicializa el Dataset.
        
        Args:
            dataframe (pd.DataFrame): Dataframe que contiene las columnas
                                      'signal' y 'labels'.
        """
        self.signals = torch.tensor(np.array(dataframe['signal'].tolist()), dtype=torch.float32)
        # Cargamos las etiquetas originales de 4000 puntos
        self.labels = torch.tensor(np.array(dataframe['labels'].tolist()), dtype=torch.float32)

        # Definimos la capa de pooling que usaremos en las etiquetas.
        # Kernel y stride de 8 reducen la longitud de 4000 a 500 (4000 / 8 = 500).
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

    def __len__(self):
        """Devuelve el número total de muestras en el dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Obtiene una única muestra (señal y etiqueta de localización) del dataset.
        
        Args:
            idx (int): El índice de la muestra a obtener.
            
        Returns:
            tuple: Una tupla conteniendo:
                   - signal (torch.Tensor): La señal de entrada.
                                            Shape: (1, 4000)
                   - label_out (torch.Tensor): La etiqueta de secuencia reducida.
                                               Shape: (1, 500)
        """
        # Señal de entrada: (1, 4000)
        signal = self.signals[idx].unsqueeze(0)

        # Etiqueta original: (4000)
        label_4000 = self.labels[idx]
        
        # Añadimos dimensión de canal para el pooling: (1, 4000)
        label_4000_pooled = label_4000.unsqueeze(0)

        # Aplicamos el pooling para obtener la etiqueta de 500 puntos
        label_out = self.pool(label_4000_pooled) # -> (1, 500)

        return signal, label_out


# --- NUEVAS CLASES DE DATASET ---

class SignalDatasetLocalizar_CWT(Dataset):
    """
    Dataset para LOCALIZACIÓN con 2 canales de entrada: Señal + CWT.
    Asume que el dataframe tiene las columnas 'signal', 'cwt' y 'labels'.
    """
    def __init__(self, dataframe):
        self.signals = torch.tensor(np.array(dataframe['signal'].tolist()), dtype=torch.float32)
        self.cwt_features = torch.tensor(np.array(dataframe['cwt'].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(np.array(dataframe['labels'].tolist()), dtype=torch.float32)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)
        
        # Asegurarse que CWT tenga la misma longitud que la señal
        assert self.signals.shape == self.cwt_features.shape, "Señal y CWT deben tener la misma dimensión"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Obtener señal y cwt, añadir dimensión de canal
        signal_1ch = self.signals[idx].unsqueeze(0)      # (1, 4000)
        cwt_1ch = self.cwt_features[idx].unsqueeze(0)    # (1, 4000)
        
        # Apilar en la dimensión de canal
        stacked_input = torch.cat((signal_1ch, cwt_1ch), dim=0) # (2, 4000)

        # Procesar etiqueta (igual que antes)
        label_4000 = self.labels[idx]
        label_4000_pooled = label_4000.unsqueeze(0)
        label_out = self.pool(label_4000_pooled) # (1, 500)

        return stacked_input, label_out

class SignalDatasetLocalizar_ZETA(Dataset):
    """
    Dataset para LOCALIZACIÓN con 2 canales de entrada: Señal + Z-Score.
    Asume que el dataframe tiene las columnas 'signal', 'zeta' y 'labels'.
    """
    def __init__(self, dataframe):
        self.signals = torch.tensor(np.array(dataframe['signal'].tolist()), dtype=torch.float32)
        self.zeta_features = torch.tensor(np.array(dataframe['zeta'].tolist()), dtype=torch.float32)
        self.labels = torch.tensor(np.array(dataframe['labels'].tolist()), dtype=torch.float32)
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)

        # Asegurarse que Z-Score tenga la misma longitud que la señal
        assert self.signals.shape == self.zeta_features.shape, "Señal y Z-Score deben tener la misma dimensión"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Obtener señal y zeta, añadir dimensión de canal
        signal_1ch = self.signals[idx].unsqueeze(0)      # (1, 4000)
        zeta_1ch = self.zeta_features[idx].unsqueeze(0)  # (1, 4000)
        
        # Apilar en la dimensión de canal
        stacked_input = torch.cat((signal_1ch, zeta_1ch), dim=0) # (2, 4000)

        # Procesar etiqueta (igual que antes)
        label_4000 = self.labels[idx]
        label_4000_pooled = label_4000.unsqueeze(0)
        label_out = self.pool(label_4000_pooled) # (1, 500)

        return stacked_input, label_out