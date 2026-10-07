"""
Configuración centralizada para Complejos-K-11a.
Define dispositivos, rutas por defecto e hiperparámetros de modelos CRNN.
"""

from __future__ import annotations

import os
from pathlib import Path
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "resultados"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de datos
DEFAULT_RAW_DATA = REPO_ROOT / "ss2kc.parquet"
DEFAULT_FEATURES_DATA = REPO_ROOT / "ss2kc_features.parquet"
SYNTHETIC_DATA_PATH = DATA_DIR / "ss2kc_synthetic.parquet"

# Selección de ruta activa: si no existe el real, usar el sintético
if DEFAULT_FEATURES_DATA.is_file():
    RUTA_DATOS = str(DEFAULT_FEATURES_DATA)
elif DEFAULT_RAW_DATA.is_file():
    RUTA_DATOS = str(DEFAULT_RAW_DATA)
else:
    RUTA_DATOS = str(SYNTHETIC_DATA_PATH)

# Dispositivo de cómputo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparámetros de entrenamiento
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 15
PATIENCE = 5
NUM_RUNS = 3

# Dimensiones de la señal EEG
SIGNAL_LENGTH = 4000      # 20 segundos a 200 Hz
LOCALIZATION_LENGTH = 500  # Longitud reducida tras MaxPool1d(kernel_size=8, stride=8)

# Hiperparámetros de arquitecturas
Nf_CNN = 32
N1_CNN = 128
N2_CNN = 128
Nf_LOC = 64
N1_LOC = 256
N2_LOC = 128
p1_DROPOUT = 0.5
p2_DROPOUT = 0.5
