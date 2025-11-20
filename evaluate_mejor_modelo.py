# evaluate_mejor_modelo.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys

# --- 1. Importar tus módulos ---
try:
    from config import * # (DEVICE, Nf_LOC, N1_LOC, BATCH_SIZE, RUTA_DATOS)
    from models import CRNN_DETECTAR_LOCALIZAR, SEED_LOCALIZAR 
    from cwtmodel import CWT_CRNN_LOCALIZAR
    from zeta import ZETA_CRNN_LOCALIZAR
    
    # Intentar importar Ensemble
    try:
        from ensemble_model import ENSEMBLE_LOCALIZAR 
    except ImportError:
        ENSEMBLE_LOCALIZAR = None
        print("Aviso: 'ensemble_model.py' no encontrado. El modo 'ENSEMBLE' no funcionará.")

    # Importar Datasets (¡Incluyendo la nueva clase!)
    from datasets import (
        SignalDatasetLocalizar, 
        SignalDatasetLocalizar_CWT, 
        SignalDatasetLocalizar_ZETA,
        SignalDatasetLocalizar_ALL  # <--- NUEVO
    )
    
    from utils import (
        get_event_based_metrics, 
        plot_event_confusion_matrix 
    )
    
    from train import load_data 
    
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

NUM_WORKERS = 4 
PIN_MEMORY = (DEVICE.type == 'cuda')

# ==========================================
# CONFIGURACIÓN DE LA EVALUACIÓN ÚNICA
# ==========================================
MODEL_TYPE = 'ENSEMBLE'  # Opciones: 'SEED', 'CWT', 'ZETA', 'ENSEMBLE'
BEST_RUN_ID = 1          # Número de corrida a evaluar
# ==========================================

def get_experiment_config(model_type):
    """
    Devuelve la configuración según el tipo de modelo.
    """
    if model_type == 'SEED':
        return {
            'model_class': SEED_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar,
            'in_channels': 1,
            'weight_prefix': 'best_model_localization_seed_run'
        }
    elif model_type == 'CWT':
        return {
            'model_class': CWT_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_CWT,
            'in_channels': 2, 
            'weight_prefix': 'best_model_localization_cwt_run'
        }
    elif model_type == 'ZETA':
        return {
            'model_class': ZETA_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ZETA,
            'in_channels': 2, 
            'weight_prefix': 'best_model_localization_zeta_run'
        }
    elif model_type == 'ENSEMBLE':
        if ENSEMBLE_LOCALIZAR is None:
            raise ValueError("ENSEMBLE_LOCALIZAR no disponible.")
        return {
            'model_class': ENSEMBLE_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ALL, # <--- CORREGIDO: Usa el dataset de 3 canales
            'in_channels': 3,                            # <--- CORREGIDO: 3 canales de entrada
            'weight_prefix': 'best_model_ensemble_run' 
        }
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' no reconocido.")

if __name__ == '__main__':

    # Cargar configuración
    config_eval = get_experiment_config(MODEL_TYPE)
    ModelClass = config_eval['model_class']
    DatasetClass = config_eval['dataset_class']
    IN_CHANNELS = config_eval['in_channels']
    WEIGHT_PREFIX = config_eval['weight_prefix']

    # Construir ruta del modelo (Solo referencial para Ensemble)
    MODEL_PATH = f'resultados/{WEIGHT_PREFIX}.pth'
    
    print(f"--- Evaluando UN SOLO MODELO ---")
    print(f"Tipo: {MODEL_TYPE}")
    print(f"Corrida ID: {BEST_RUN_ID}")

    # --- Parámetros de Evaluación ---
    EVENT_PROB_THRESHOLD = 0.7  
    EVENT_MIN_DURATION = 15     
    EVENT_IOU_THRESHOLD = 0.2   

    # --- Cargar Datos ---
    print(f"Cargando datos desde {RUTA_DATOS}...")
    df = load_data(RUTA_DATOS)
    if df is None:
        sys.exit(1)
    
    # Validaciones de columnas para modelos complejos
    if MODEL_TYPE in ['CWT', 'ENSEMBLE'] and 'cwt' not in df.columns:
        print("Error: Falta columna 'cwt'. Ejecuta feature_engineering.py")
        sys.exit(1)
    if MODEL_TYPE in ['ZETA', 'ENSEMBLE'] and 'zeta' not in df.columns:
        print("Error: Falta columna 'zeta'. Ejecuta feature_engineering.py")
        sys.exit(1)

    print("Generando división de datos (train/val/test)...")
    df_localizar = df.copy()
    
    _, temp_df = train_test_split(
        df_localizar, test_size=0.2, random_state=42, stratify=df_localizar['existeK']
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK']
    )
    
    # Instanciar Dataset
    test_dataset = DatasetClass(test_df)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    print(f"Test loader: {len(test_df)} muestras.")

    # --- Cargar Modelo ---
    print(f"\nCargando modelo...")
    # Instanciamos el modelo con los canales correctos
    model = ModelClass(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)

    if MODEL_TYPE == 'ENSEMBLE':
        try:
            # Carga especial para Ensemble
            model.load_ensemble_weights(run_id=BEST_RUN_ID, results_dir='resultados', map_location=DEVICE)
        except Exception as e:
            print(f"Error cargando pesos del Ensemble: {e}")
            sys.exit(1)
    else:
        # Carga estándar
        if not os.path.exists(MODEL_PATH):
            print(f"Error: No existe el archivo de pesos: {MODEL_PATH}")
            sys.exit(1)
            
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Pesos cargados desde: {MODEL_PATH}")
        except RuntimeError as e:
            print(f"Error de arquitectura al cargar pesos: {e}")
            sys.exit(1)

    model.eval()

    # --- Evaluar ---
    print(f"\nCalculando métricas para Run {BEST_RUN_ID}...")
    metrics = get_event_based_metrics(
        model, 
        test_loader, 
        DEVICE,
        prob_threshold=EVENT_PROB_THRESHOLD,
        min_duration=EVENT_MIN_DURATION,
        iou_threshold=EVENT_IOU_THRESHOLD
    )

    # --- Reporte Final ---
    print("\n=======================================")
    print(f" RESULTADOS: {MODEL_TYPE} (Run {BEST_RUN_ID})")
    print("=======================================")
    print(f"Precisión: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print("---------------------------------------")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    print("=======================================")

    # --- Graficar Matriz de Confusión ---
    output_dir = 'resultados'
    filename_matrix = f'event_confusion_matrix_{MODEL_TYPE}_Run{BEST_RUN_ID}.png'
    save_path = os.path.join(output_dir, filename_matrix)
    
    plot_event_confusion_matrix(
        tp=metrics['tp'], 
        fp=metrics['fp'], 
        fn=metrics['fn'], 
        save_path=save_path
    )