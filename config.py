import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
RUTA_DATOS = 'ss2kc.parquet' -> ORIGINAL # O '/content/drive/MyDrive/proyectoint/ss2kc.parquet' 
ss2kc_features.parquet -> ES EL PROCESADO.
"""
RUTA_DATOS = 'ss2kc_features.parquet' 
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EPOCHS = 40
PATIENCE = 10
NUM_RUNS = 5

# Hiperparámetros de Modelos (ejemplo)
Nf_CNN = 32
N1_CNN = 128
Nf_LOC = 64
N1_LOC = 256