# evaluate.py

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys

# --- 1. Importar tus módulos ---
# Asumiendo que este script está en la raíz, junto a config.py, models.py, etc.
try:
    from config import * # (DEVICE, Nf_LOC, N1_LOC, BATCH_SIZE, RUTA_DATOS)
    from models import CRNN_DETECTAR_LOCALIZAR,SEED_LOCALIZAR # (¡Importante! La arquitectura)
    from datasets import SignalDatasetLocalizar
    from utils import (get_event_based_metrics, visualizar_localizacion, plot_event_confusion_matrix,plot_event_confusion_matrix_with_std) # (La nueva función de evaluación)
    
    # Re-usamos la función de carga de train.py
    # Necesitamos importar 'load_data' desde train.py
    from train import load_data 
    
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que 'evaluate.py' esté en el directorio raíz del proyecto.")
    print("Y que todos los archivos (config.py, models.py, etc.) están presentes.")
    sys.exit(1)

NUM_WORKERS = 4 
PIN_MEMORY = (DEVICE.type == 'cuda')

if __name__ == '__main__':

    # --- 1. Parámetros de Evaluación ---
    # (¡Estos umbrales ahora se aplicarán a las 5 corridas!)
    EVENT_PROB_THRESHOLD = 0.7  # Umbral de probabilidad
    EVENT_MIN_DURATION = 15     # Duración mínima
    EVENT_IOU_THRESHOLD = 0.2   # Umbral de IoU

    # --- 2. Cargar Datos y crear Test Loader ---
    # (Esto no cambia)
    print(f"Usando dispositivo: {DEVICE}")
    print(f"Cargando datos desde {RUTA_DATOS}...")
    df = load_data(RUTA_DATOS)
    if df is None:
        print("Error al cargar datos.")
        sys.exit(1)
    
    print("Generando división de datos (train/val/test)...")
    df_localizar = df[['signal', 'labels', 'existeK']]
    train_df, temp_df = train_test_split(
        df_localizar, test_size=0.2, random_state=42, stratify=df_localizar['existeK']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK']
    )
    print(f"Total datos: {len(df)}, Usando {len(test_df)} muestras para prueba.")

    test_dataset = SignalDatasetLocalizar(test_df)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    print("Test loader creado.")

    # --- 3. Bucle de Evaluación Múltiple ---
    print(f"\nIniciando evaluación de {NUM_RUNS} modelos...")
    
    all_run_metrics = []
    
    for i in range(1, NUM_RUNS + 1):
        # (Asume que estás evaluando el Exp 6, SEED)
        MODEL_PATH = f'resultados/best_model_localization_seed_run{i}.pth'
        
        if not os.path.exists(MODEL_PATH):
            print(f"Advertencia: No se encuentra el modelo para la corrida {i} en: {MODEL_PATH}. Saltando.")
            continue
            
        print(f"\n--- Evaluando Corrida {i}/{NUM_RUNS} ---")
        print(f"Cargando modelo desde {MODEL_PATH}...")

        # Cargar el modelo de la corrida 'i'
        model = SEED_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        # Obtener métricas para este modelo
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
        print("Error: No se pudo evaluar ningún modelo. Revisa las rutas.")
        sys.exit(1)

    # --- 4. Calcular Media y Desviación Estándar ---
    print("\n--- ¡Evaluación de todas las corridas completada! ---")
    
    # Convertir la lista de diccionarios a un DataFrame de pandas
    metrics_df = pd.DataFrame(all_run_metrics)
    
    # Calcular media y DE
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()

    print("\n--- Resultados (Media ± DE) ---")
    print(f"Precisión: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall:    {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1-Score:  {mean_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")
    print("---------------------------------")
    print(f"TP:        {mean_metrics['tp']:.1f} ± {std_metrics['tp']:.1f}")
    print(f"FP:        {mean_metrics['fp']:.1f} ± {std_metrics['fp']:.1f}")
    print(f"FN:        {mean_metrics['fn']:.1f} ± {std_metrics['fn']:.1f}")

    # --- 5. Graficar Matriz de Confusión (con Media ± DE) ---
    output_dir = 'resultados' # Asumimos que es la carpeta de resultados
    
    plot_event_confusion_matrix_with_std(
        tp_mean=mean_metrics['tp'], tp_std=std_metrics['tp'],
        fp_mean=mean_metrics['fp'], fp_std=std_metrics['fp'],
        fn_mean=mean_metrics['fn'], fn_std=std_metrics['fn'],
        save_path=os.path.join(output_dir, 'event_confusion_matrix_SEED_MeanDE.png')
    )

    # (La visualización de muestras individuales ya no tiene tanto sentido aquí,
    #  ya que tenemos 5 modelos. Podríamos graficar la del "mejor" modelo,
    #  pero por ahora lo omitiremos para mantenerlo simple.)
    
    print("\n--- Evaluación promediada finalizada ---")
