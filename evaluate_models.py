# evaluate.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys

# --- 1. Importar tus módulos ---
try:
    from config import * # (DEVICE, Nf_LOC, N1_LOC, BATCH_SIZE, RUTA_DATOS, NUM_RUNS)
    # Importar Modelos Base
    from models import CRNN_DETECTAR_LOCALIZAR, SEED_LOCALIZAR 
    # Importar Modelos Nuevos (CWT, ZETA)
    from cwtmodel import CWT_CRNN_LOCALIZAR
    from zeta import ZETA_CRNN_LOCALIZAR
    
    # Intentar importar Ensemble (si existe el archivo)
    try:
        from ensemble_model import ENSEMBLE_LOCALIZAR 
    except ImportError:
        ENSEMBLE_LOCALIZAR = None
        print("Aviso: 'ensemble_model.py' no encontrado. El modo 'ENSEMBLE' no funcionará.")

    # Importar Datasets
    from datasets import (
        SignalDatasetLocalizar, 
        SignalDatasetLocalizar_CWT, 
        SignalDatasetLocalizar_ZETA
    )
    
    from utils import (
        get_event_based_metrics, 
        plot_event_confusion_matrix_with_std
    )
    
    from train import load_data 
    
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

NUM_WORKERS = 4 
PIN_MEMORY = (DEVICE.type == 'cuda')

# ==========================================
# CONFIGURACIÓN DE LA EVALUACIÓN
# Cambia esto según lo que quieras evaluar: 'SEED', 'CWT', 'ZETA', 'ENSEMBLE'
# ==========================================
MODEL_TYPE = 'CWT'  
# ==========================================

def get_experiment_config(model_type):
    """
    Devuelve la clase del modelo, la clase del dataset y el prefijo del archivo de pesos
    según el tipo de experimento seleccionado.
    """
    if model_type == 'SEED':
        return {
            'model_class': SEED_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar,
            'in_channels': 1,
            'weight_prefix': 'best_model_localization_seed'
        }
    elif model_type == 'CWT':
        return {
            'model_class': CWT_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_CWT,
            'in_channels': 2, # Señal + CWT
            'weight_prefix': 'best_model_localization_cwt' # Asegúrate que coincida con train.py
        }
    elif model_type == 'ZETA':
        return {
            'model_class': ZETA_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ZETA,
            'in_channels': 2, # Señal + Zeta
            'weight_prefix': 'best_model_localization_zeta'
        }
    elif model_type == 'ENSEMBLE':
        if ENSEMBLE_LOCALIZAR is None:
            raise ValueError("La clase ENSEMBLE_LOCALIZAR no se pudo importar.")
        return {
            'model_class': ENSEMBLE_LOCALIZAR,
            # Asumimos que el ensemble usa el dataset estándar o uno específico. 
            # Si necesita todos los features, usa SignalDatasetLocalizar_CWT (que tiene signal) 
            # o crea un DatasetEnsemble específico. Por defecto pondré el simple:
            'dataset_class': SignalDatasetLocalizar, 
            'in_channels': 1,
            'weight_prefix': 'best_model_ensemble_run'
        }
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' no reconocido.")

if __name__ == '__main__':

    # Cargar configuración del experimento
    config_eval = get_experiment_config(MODEL_TYPE)
    ModelClass = config_eval['model_class']
    DatasetClass = config_eval['dataset_class']
    IN_CHANNELS = config_eval['in_channels']
    WEIGHT_PREFIX = config_eval['weight_prefix']

    print(f"--- Evaluando configuración: {MODEL_TYPE} ---")

    # --- Parámetros de Evaluación ---
    EVENT_PROB_THRESHOLD = 0.7  
    EVENT_MIN_DURATION = 15     
    EVENT_IOU_THRESHOLD = 0.2   

    # --- Cargar Datos ---
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando datos desde {RUTA_DATOS}...")
    df = load_data(RUTA_DATOS)
    if df is None:
        sys.exit(1)
    
    # Validar columnas requeridas según el modelo
    if MODEL_TYPE == 'CWT' and 'cwt' not in df.columns:
        print("Error: El dataset no tiene la columna 'cwt'. Ejecuta feature_engineering.py primero.")
        sys.exit(1)
    if MODEL_TYPE == 'ZETA' and 'zeta' not in df.columns:
        print("Error: El dataset no tiene la columna 'zeta'. Ejecuta feature_engineering.py primero.")
        sys.exit(1)

    print("Generando división de datos (train/val/test)...")
    df_localizar = df.copy() # Copia por seguridad
    
    train_df, temp_df = train_test_split(
        df_localizar, test_size=0.2, random_state=42, stratify=df_localizar['existeK']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK']
    )
    
    # Instanciar el Dataset correcto dinámicamente
    test_dataset = DatasetClass(test_df)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    print(f"Test loader creado usando {DatasetClass.__name__}. Muestras: {len(test_df)}")

    # --- Bucle de Evaluación Múltiple ---
    print(f"\nIniciando evaluación de {NUM_RUNS} modelos tipo {MODEL_TYPE}...")
    
    all_run_metrics = []
    
    for i in range(1, NUM_RUNS + 1):
        MODEL_PATH = f'resultados/{WEIGHT_PREFIX}{i}.pth'
        
        if not os.path.exists(MODEL_PATH):
            print(f"Advertencia: No se encuentra {MODEL_PATH}. Saltando.")
            continue
            
        print(f"\n--- Evaluando Corrida {i}/{NUM_RUNS} ---")
        
        # Instanciar el Modelo correcto con los canales correctos
        model = ModelClass(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except RuntimeError as e:
            print(f"Error cargando pesos en {MODEL_PATH}: {e}")
            print("Posible causa: El modelo guardado no coincide con la arquitectura seleccionada.")
            continue
            
        model.eval()

        # Obtener métricas
        event_metrics = get_event_based_metrics(
            model, 
            test_loader, 
            DEVICE,
            prob_threshold=EVENT_PROB_THRESHOLD,
            min_duration=EVENT_MIN_DURATION,
            iou_threshold=EVENT_IOU_THRESHOLD
        )
        
        all_run_metrics.append(event_metrics)

    if not all_run_metrics:
        print("Error: No se pudo evaluar ningún modelo.")
        sys.exit(1)

    # --- Reporte Final ---
    print("\n--- Resultados Promedio ---")
    metrics_df = pd.DataFrame(all_run_metrics)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    print(f"Precisión: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall:    {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1-Score:  {mean_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")

    # Guardar matriz de confusión
    output_dir = 'resultados'
    filename_matrix = f'event_confusion_matrix_{MODEL_TYPE}_MeanDE.png'
    
    plot_event_confusion_matrix_with_std(
        tp_mean=mean_metrics['tp'], tp_std=std_metrics['tp'],
        fp_mean=mean_metrics['fp'], fp_std=std_metrics['fp'],
        fn_mean=mean_metrics['fn'], fn_std=std_metrics['fn'],
        save_path=os.path.join(output_dir, filename_matrix)
    )
    print(f"\nMatriz de confusión guardada en: {filename_matrix}")