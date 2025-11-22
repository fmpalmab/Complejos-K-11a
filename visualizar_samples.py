# visualize_samples.py

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- 1. Importar módulos del proyecto ---
try:
    from config import * # (DEVICE, Nf_LOC, N1_LOC, BATCH_SIZE, RUTA_DATOS)
    from models import CRNN_DETECTAR_LOCALIZAR, SEED_LOCALIZAR 
    from cwtmodel import CWT_CRNN_LOCALIZAR
    from zeta import ZETA_CRNN_LOCALIZAR
    
    try:
        from ensemble_model import ENSEMBLE_LOCALIZAR 
    except ImportError:
        ENSEMBLE_LOCALIZAR = None
        print("Aviso: 'ensemble_model.py' no encontrado. El modo 'ENSEMBLE' se omitirá.")

    from datasets import (
        SignalDatasetLocalizar, 
        SignalDatasetLocalizar_CWT, 
        SignalDatasetLocalizar_ZETA,
        SignalDatasetLocalizar_ALL
    )
    from train import load_data 
    
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

# ==========================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ==========================================
NUM_SAMPLES_TO_PLOT = 10        # Cantidad de gráficos a generar
OUTPUT_DIR = 'visualizaciones'  # Carpeta donde se guardarán
MODELS_TO_EVALUATE = ['SEED', 'CWT', 'ZETA', 'ENSEMBLE']
# ==========================================

def get_experiment_config(model_type):
    """Devuelve la configuración y nombre de archivo de pesos para cada modelo."""
    if model_type == 'SEED':
        return {
            'model_class': SEED_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar,
            'in_channels': 1,
            'weight_filename': 'best_model_localization_seed' 
        }
    elif model_type == 'CWT':
        return {
            'model_class': CWT_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_CWT,
            'in_channels': 2, 
            'weight_filename': 'best_model_localization_cwt'
        }
    elif model_type == 'ZETA':
        return {
            'model_class': ZETA_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ZETA,
            'in_channels': 2, 
            'weight_filename': 'best_model_localization_zeta'
        }
    elif model_type == 'ENSEMBLE':
        if ENSEMBLE_LOCALIZAR is None: return None 
        return {
            'model_class': ENSEMBLE_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ALL,
            'in_channels': 3,
            'weight_filename': None # Carga especial
        }
    else:
        raise ValueError(f"Modelo '{model_type}' no reconocido.")

def plot_visual_comparison(signal, gt, predictions, sample_idx, activity_score, save_dir):
    """
    Genera y guarda el gráfico comparativo para una muestra específica.
    """
    plt.figure(figsize=(14, 8))
    
    # Eje temporal (4000 puntos)
    t = np.arange(len(signal))
    
    # --- Corrección de tamaño para Ground Truth (GT) ---
    # Si el GT viene de una resolución menor (ej. 500), lo interpolamos a 4000
    if len(gt) != len(t):
        x_old = np.linspace(0, len(t)-1, len(gt))
        gt_interp = np.interp(t, x_old, gt)
        gt = (gt_interp > 0.5).astype(int)

    # --- Subplot 1: Señal + Ground Truth ---
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Señal EEG', color='black', alpha=0.6, linewidth=0.8)
    
    # Sombrear regiones donde hay Complejo K real
    if gt.max() > 0:
        plt.fill_between(t, signal.min(), signal.max(), where=(gt==1), 
                         color='green', alpha=0.3, label='Ground Truth (K-Complex)')
    
    plt.title(f"Muestra #{sample_idx} (Score Actividad: {activity_score:.0f})", fontsize=14)
    plt.ylabel("Amplitud (µV)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # --- Subplot 2: Predicciones de los Modelos ---
    plt.subplot(2, 1, 2)
    
    styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    for i, (model_name, probs) in enumerate(predictions.items()):
        # Interpolación si la predicción no es de tamaño 4000
        if len(probs) != len(t):
             x_old_probs = np.linspace(0, len(t)-1, len(probs))
             probs = np.interp(t, x_old_probs, probs)
             
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        plt.plot(t, probs, label=f"{model_name}", linestyle=style, color=color, linewidth=2, alpha=0.8)

    # Línea de umbral de decisión
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Umbral 0.5')
    
    plt.title("Probabilidad de Detección por Modelo")
    plt.xlabel("Muestras")
    plt.ylabel("Probabilidad")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar archivo
    filename = f'sample_{sample_idx}_score_{int(activity_score)}.png'
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"-> Gráfico guardado: {save_path}")

def load_all_models():
    """Carga e inicializa todos los modelos en un diccionario para uso rápido."""
    loaded_models = {}
    
    print("\n--- Cargando Modelos ---")
    for model_type in MODELS_TO_EVALUATE:
        config = get_experiment_config(model_type)
        if config is None: continue
        
        try:
            # Instanciar
            model = config['model_class'](
                num_classes=1, 
                Nf=Nf_LOC, 
                N1=N1_LOC, 
                in_channels=config['in_channels']
            ).to(DEVICE)
            
            # Cargar Pesos
            if model_type == 'ENSEMBLE':
                model.load_ensemble_weights(run_id=1, results_dir='resultados', map_location=DEVICE)
            else:
                path = f"resultados/{config['weight_filename']}.pth"
                if not os.path.exists(path):
                    print(f"Advertencia: No se encontró {path}. Saltando modelo {model_type}.")
                    continue
                model.load_state_dict(torch.load(path, map_location=DEVICE))
            
            model.eval()
            loaded_models[model_type] = {
                'model': model,
                'dataset_class': config['dataset_class']
            }
            print(f"Modelo cargado: {model_type}")
            
        except Exception as e:
            print(f"Error cargando {model_type}: {e}")
            
    return loaded_models

if __name__ == '__main__':

    # 1. Preparar Directorio
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Carpeta creada: {OUTPUT_DIR}")

    # 2. Cargar Datos
    print(f"Cargando datos desde {RUTA_DATOS}...")
    df = load_data(RUTA_DATOS)
    if df is None: sys.exit(1)

    # 3. División (Mismo random_state para consistencia)
    print("Generando set de prueba...")
    df_loc = df.copy()
    _, temp_df = train_test_split(df_loc, test_size=0.2, random_state=42, stratify=df_loc['existeK'])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK'])
    
    # 4. Cargar Modelos
    models_dict = load_all_models()
    if not models_dict:
        print("No se pudo cargar ningún modelo. Saliendo.")
        sys.exit(1)

    # 5. Seleccionar las MEJORES muestras (con más actividad K)
    print(f"\nBuscando las Top {NUM_SAMPLES_TO_PLOT} muestras con mayor actividad...")
    
    # Calcular score de actividad (suma de 1s en labels)
    test_df['activity_score'] = test_df['labels'].apply(np.sum)
    
    # Filtrar solo las que tienen K y ordenar descendente
    samples_with_k = test_df[test_df['existeK'] == 1].sort_values(by='activity_score', ascending=False)
    
    if samples_with_k.empty:
        print("Error: No se encontraron muestras con Complejos K en el test set.")
        sys.exit(1)
        
    # Tomar las top N
    top_samples = samples_with_k.head(NUM_SAMPLES_TO_PLOT)
    print(f"Se generarán gráficos para {len(top_samples)} muestras.")

    # 6. Bucle de Generación
    for idx_pos, (original_idx, row) in enumerate(top_samples.iterrows()):
        print(f"\nProcesando muestra {idx_pos+1}/{len(top_samples)} (ID Original: {original_idx})...")
        
        # Obtener el índice posicional relativo al test_df para el Dataset
        # (Dataset accede por iloc, necesitamos saber en qué fila de test_df está esta muestra)
        dataset_idx = test_df.index.get_loc(original_idx)
        
        current_signal = None
        current_gt = None
        current_preds = {}
        
        # Ejecutar inferencia con cada modelo cargado
        for m_name, m_data in models_dict.items():
            model = m_data['model']
            DatasetClass = m_data['dataset_class']
            
            # Instanciar dataset temporalmente solo para obtener el formato correcto de input
            # (Esto es rápido porque solo envuelve el dataframe, no copia datos pesados)
            ds = DatasetClass(test_df)
            
            try:
                input_tensor, label_tensor = ds[dataset_idx]
                
                # Guardar señal y GT solo la primera vez
                if current_signal is None:
                    current_signal = input_tensor[0].cpu().numpy() # Canal 0 siempre es señal
                    current_gt = label_tensor.cpu().numpy().squeeze()
                
                # Inferencia
                input_batch = input_tensor.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    output = model(input_batch)
                    probs = torch.sigmoid(output).cpu().numpy().squeeze()
                    
                current_preds[m_name] = probs
                
            except Exception as e:
                print(f"Error infiriendo con {m_name}: {e}")

        # Graficar
        if current_signal is not None:
            plot_visual_comparison(
                current_signal, 
                current_gt, 
                current_preds, 
                sample_idx=original_idx, 
                activity_score=row['activity_score'],
                save_dir=OUTPUT_DIR
            )
            
    print(f"\n¡Listo! Revisa la carpeta '{OUTPUT_DIR}' para ver las imágenes.")