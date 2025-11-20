# ensemble_model.py

import torch
import torch.nn as nn
import os

# Importamos las arquitecturas individuales
# Asegúrate de que estos archivos (models.py, cwtmodel.py, zeta.py) existen en la misma carpeta
try:
    from models import CRNN_DETECTAR_LOCALIZAR
    from cwtmodel import CWT_CRNN_LOCALIZAR
    from zeta import ZETA_CRNN_LOCALIZAR
    from config import DEVICE
except ImportError as e:
    print(f"Error importando dependencias del Ensemble: {e}")

class ENSEMBLE_LOCALIZAR(nn.Module):
    """
    Modelo de Ensamblaje (Ensemble) por Votación Mayoritaria.
    
    Combina:
    - Exp 3: CRNN Base (Entrada: Señal)
    - Exp 4: CRNN + CWT (Entrada: Señal + CWT)
    - Exp 5: CRNN + ZETA (Entrada: Señal + ZETA)
    
    Lógica:
    1. Recibe un tensor de entrada de 3 canales (Señal, CWT, ZETA).
    2. Distribuye los canales correspondientes a cada sub-modelo.
    3. Obtiene las probabilidades individuales.
    4. Binariza las salidas (detección > 0.5).
    5. Realiza votación: Si 2 o más modelos detectan un evento, el resultado es 1.
    """
    def __init__(self, in_channels=3, num_classes=1, Nf=64, N1=256, N2=128, p1=0.2, p2=0.5):
        super(ENSEMBLE_LOCALIZAR, self).__init__()
        
        # --- 1. Inicializar los Sub-Modelos ---
        # Exp 3: Modelo Base (Solo usa 1 canal: Señal)
        self.model_exp3 = CRNN_DETECTAR_LOCALIZAR(
            in_channels=1, num_classes=num_classes, Nf=Nf, N1=N1, N2=N2, p1=p1, p2=p2
        )
        
        # Exp 4: Modelo CWT (Usa 2 canales: Señal + CWT)
        self.model_exp4 = CWT_CRNN_LOCALIZAR(
            in_channels=2, num_classes=num_classes, Nf=Nf, N1=N1, N2=N2, p1=p1, p2=p2
        )
        
        # Exp 5: Modelo ZETA (Usa 2 canales: Señal + ZETA)
        self.model_exp5 = ZETA_CRNN_LOCALIZAR(
            in_channels=2, num_classes=num_classes, Nf=Nf, N1=N1, N2=N2, p1=p1, p2=p2
        )
        
        # No necesitamos parámetros entrenables propios para el ensemble por votación dura
        # pero registramos los submodelos para que .to(DEVICE) funcione en todos.

    def load_ensemble_weights(self, run_id, results_dir='resultados', map_location=None):
        """
        Carga los pesos pre-entrenados para cada sub-modelo basándose en el run_id.
        
        Nombres de archivo esperados (ajustar si tus archivos se llaman diferente):
        - Exp 3: best_model_localization.pth
        - Exp 4: best_model_localization_cwt.pth
        - Exp 5: best_model_localization_zeta.pth
        """
        if map_location is None:
            map_location = next(self.parameters()).device

        # Definir rutas (Ajusta los nombres de archivo si difieren de tu estructura)
        # NOTA: Asumo que el Exp 3 se llama 'best_model_localization_runX.pth' (sin sufijo)
        path_exp3 = os.path.join(results_dir, f'best_model_localization.pth') 
        path_exp4 = os.path.join(results_dir, f'best_model_localization_cwt.pth')
        path_exp5 = os.path.join(results_dir, f'best_model_localization_zeta.pth')

        # Cargar pesos
        try:
            self.model_exp3.load_state_dict(torch.load(path_exp3, map_location=map_location))
            print(f" [Ensemble] Cargado Exp3 desde {path_exp3}")
            
            self.model_exp4.load_state_dict(torch.load(path_exp4, map_location=map_location))
            print(f" [Ensemble] Cargado Exp4 desde {path_exp4}")
            
            self.model_exp5.load_state_dict(torch.load(path_exp5, map_location=map_location))
            print(f" [Ensemble] Cargado Exp5 desde {path_exp5}")
            
        except FileNotFoundError as e:
            print(f"Error crítico cargando pesos del ensemble: {e}")
            print("Verifica que los archivos de los Experimentos 3, 4 y 5 existen.")
            raise e

    def forward(self, x):
        """
        Forward pass con Votación Mayoritaria.
        x shape: (Batch, 3, 4000) -> [Signal, CWT, ZETA]
        """
        # --- 1. Preparación de Datos (Slicing) ---
        # Canal 0: Señal (B, 1, 4000)
        x_signal = x[:, 0:1, :] 
        
        # Canal 1: CWT (B, 1, 4000)
        x_cwt = x[:, 1:2, :]
        
        # Canal 2: ZETA (B, 1, 4000)
        x_zeta = x[:, 2:3, :]

        # Construir inputs específicos
        input_exp3 = x_signal                    # (B, 1, 4000)
        input_exp4 = torch.cat([x_signal, x_cwt], dim=1)  # (B, 2, 4000)
        input_exp5 = torch.cat([x_signal, x_zeta], dim=1) # (B, 2, 4000)

        # --- 2. Inferencia de Sub-Modelos ---
        # Obtenemos logits (Salida cruda antes de Sigmoid)
        # Shape: (B, 1, 500)
        logits3 = self.model_exp3(input_exp3)
        logits4 = self.model_exp4(input_exp4)
        logits5 = self.model_exp5(input_exp5)

        # --- 3. Obtener Predicciones Binarias ---
        # Aplicamos Sigmoid para obtener probabilidad y luego umbral 0.5
        pred3 = (torch.sigmoid(logits3) > 0.5).float()
        pred4 = (torch.sigmoid(logits4) > 0.5).float()
        pred5 = (torch.sigmoid(logits5) > 0.5).float()

        # --- 4. Votación de Mayoría (Hard Voting) ---
        # Sumamos las predicciones (0 o 1)
        vote_sum = pred3 + pred4 + pred5 # Valores posibles: 0, 1, 2, 3

        # Si la suma es >= 2, la mayoría votó positivo
        majority_vote = (vote_sum >= 3).float()

        # --- 5. Formateo de Salida ---
        # 'evaluate.py' espera logits porque suele aplicar sigmoid internamente 
        # o usa thresholds sobre probabilidades.
        # Para ser compatibles, convertimos la decisión binaria (0/1) 
        # en "pseudo-logits" muy fuertes (-10 o +10).
        # Así: sigmoid(10) ~= 1.0, sigmoid(-10) ~= 0.0
        
        output_logits = majority_vote * 20.0 - 10.0 
        # Si vote=1 -> 20-10 = 10 (Prob ~1.0)
        # Si vote=0 -> 0-10 = -10 (Prob ~0.0)

        return output_logits

    # Sobreescribimos load_state_dict para evitar errores si evaluate.py intenta cargar
    # un solo archivo .pth para todo el ensemble (lo cual no existe).
    def load_state_dict(self, state_dict, strict=True):
        print("Advertencia: 'load_state_dict' llamado en ENSEMBLE.")
        print("Este modelo carga sus pesos internamente vía 'load_ensemble_weights'. Ignorando carga externa.")
        pass