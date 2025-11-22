# evaluate_all.py

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys
import random

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
        print("Aviso: 'ensemble_model.py' no encontrado. El modo 'ENSEMBLE' se omitirá si se selecciona.")

    # Importar Datasets
    from datasets import (
        SignalDatasetLocalizar, 
        SignalDatasetLocalizar_CWT, 
        SignalDatasetLocalizar_ZETA,
        SignalDatasetLocalizar_ALL
    )
    
    from utils import get_event_based_metrics
    from train import load_data 
    
except ImportError as e:
    print(f"Error importando módulos: {e}")
    sys.exit(1)

NUM_WORKERS = 4 
PIN_MEMORY = (DEVICE.type == 'cuda')

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
# Lista de modelos a evaluar
MODELS_TO_EVALUATE = ['SEED', 'CWT', 'ZETA', 'ENSEMBLE']

# Parámetros de evaluación de eventos
EVENT_PROB_THRESHOLD = 0.7  
EVENT_MIN_DURATION = 15     
EVENT_IOU_THRESHOLD = 0.2   
# ==========================================

def get_experiment_config(model_type):
    """
    Devuelve la configuración y el NOMBRE DEL ARCHIVO (sin extensión) 
    según el tipo de modelo.
    """
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
        if ENSEMBLE_LOCALIZAR is None:
            return None 
        return {
            'model_class': ENSEMBLE_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ALL,
            'in_channels': 3,
            'weight_filename': None 
        }
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' no reconocido.")

def plot_cm_percent(tp, fp, fn, metrics, model_name, save_path):
    """
    Grafica una matriz de confusión de eventos con conteos, PORCENTAJES (4 decimales)
    y muestra las métricas globales en el título.
    """
    epsilon = 1e-6
    
    recall_pct = (tp / (tp + fn + epsilon)) * 100
    miss_pct = (fn / (tp + fn + epsilon)) * 100
    fp_rate_cell = (fp / (tp + fp + epsilon)) * 100 

    matrix_data = [[tp, fn], [fp, 0]]
    
    annot_data = [
        [f"TP: {tp}\n({recall_pct:.4f}% Recall)", f"FN: {fn}\n({miss_pct:.4f}% Miss)"],
        [f"FP: {fp}\n({fp_rate_cell:.4f}% Error Pred)", "TN\n(N/A)"]
    ]
    
    prec_val = metrics['precision']
    rec_val = metrics['recall']
    f1_val = metrics['f1_score']
    
    title_text = (f"Matriz de Confusión: {model_name}\n"
                  f"Precision: {prec_val:.4f} | Recall: {rec_val:.4f} | F1-Score: {f1_val:.4f}")

    df_cm = pd.DataFrame(matrix_data,
                         index=["Real: Evento", "Real: No-Evento"],
                         columns=["Pred: Evento", "Pred: No-Evento"])
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_cm, annot=annot_data, fmt="", cmap="Blues", cbar=False,
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title(title_text, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Matriz guardada en: {save_path}")

def plot_comparison_bar_chart(results, save_dir='resultados'):
    """
    Crea un gráfico de barras comparando Precision, Recall y F1.
    """
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    precisions = [results[m]['precision'] for m in models]
    recalls = [results[m]['recall'] for m in models]
    f1s = [results[m]['f1_score'] for m in models]
    
    rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#4c72b0')
    rects2 = ax.bar(x, recalls, width, label='Recall', color='#55a868')
    rects3 = ax.bar(x + width, f1s, width, label='F1-Score', color='#c44e52')
    
    ax.set_ylabel('Puntaje (0-1)')
    ax.set_title('Comparación de Mejores Modelos (Global)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15) 
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    save_path = os.path.join(save_dir, 'comparison_best_models.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nGráfico comparativo de barras guardado en: {save_path}")

def plot_visual_comparison(signal, gt, predictions, sample_idx, save_dir='resultados'):
    """
    Genera un gráfico de línea comparando la predicción de CADA modelo.
    """
    plt.figure(figsize=(14, 8))
    
    # Eje temporal de la señal original (ej. 4000)
    t = np.arange(len(signal))
    
    # --- CORRECCIÓN DE TAMAÑO PARA GT ---
    # Si el GT (500) no calza con la señal (4000), lo redimensionamos
    if len(gt) != len(t):
        # Creamos un eje x antiguo para los 500 puntos
        x_old = np.linspace(0, len(t)-1, len(gt))
        # Interpolamos al tamaño de t (4000)
        gt_interp = np.interp(t, x_old, gt)
        # Como GT es binario (0 o 1), aplicamos umbral para mantenerlo limpio
        gt = (gt_interp > 0.5).astype(int)

    # --- Subplot 1: Señal + Ground Truth ---
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Señal EEG', color='black', alpha=0.6, linewidth=0.8)
    
    # Resaltar áreas donde GT es 1
    if gt.max() > 0:
        plt.fill_between(t, signal.min(), signal.max(), where=(gt==1), 
                         color='green', alpha=0.3, label='Ground Truth (K-Complex)')
    
    plt.title(f"Muestra de Prueba #{sample_idx} (Visualización Comparativa)", fontsize=14)
    plt.ylabel("Amplitud")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # --- Subplot 2: Curvas de Probabilidad de los Modelos ---
    plt.subplot(2, 1, 2)
    
    styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    for i, (model_name, probs) in enumerate(predictions.items()):
        # CORRECCIÓN DE TAMAÑO PARA PREDICCIONES
        if len(probs) != len(t):
             x_old_probs = np.linspace(0, len(t)-1, len(probs))
             probs = np.interp(t, x_old_probs, probs)
             
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        plt.plot(t, probs, label=f"{model_name}", linestyle=style, color=color, linewidth=2, alpha=0.8)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Umbral 0.5')
    
    plt.title("Comparación de Probabilidades Predichas")
    plt.xlabel("Muestras")
    plt.ylabel("Probabilidad / Confianza")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'comparison_sample_{sample_idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\nGráfico de visualización de muestra guardado en: {save_path}")


if __name__ == '__main__':

    print(f"--- INICIANDO EVALUACIÓN COMPARATIVA (MEJORES MODELOS) ---")
    
    # --- Cargar Datos ---
    print(f"Cargando datos desde {RUTA_DATOS}...")
    df = load_data(RUTA_DATOS)
    if df is None:
        sys.exit(1)
    
    if 'cwt' not in df.columns or 'zeta' not in df.columns:
        print("Error: Faltan columnas 'cwt' o 'zeta'. Ejecuta feature_engineering.py")
        sys.exit(1)

    # Generar división de datos
    print("Generando división de datos...")
    df_localizar = df.copy()
    _, temp_df = train_test_split(df_localizar, test_size=0.2, random_state=42, stratify=df_localizar['existeK'])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK'])
    
    print(f"Test set size: {len(test_df)}")
    
    # --- SELECCIÓN DE MUESTRA PARA VISUALIZACIÓN ---
    indices_k = [i for i, (_, row) in enumerate(test_df.iterrows()) if row['existeK'] == 1]
    
    target_idx = None
    visual_data = {'signal': None, 'gt': None, 'preds': {}}

    if indices_k:
        target_idx = random.choice(indices_k)
        print(f"\n[Visualización] Se ha seleccionado la muestra aleatoria #{target_idx} (con K) para comparar modelos.")
    else:
        print("Advertencia: No se encontraron muestras con K en el test set para visualizar.")

    all_model_results = {}

    # --- BUCLE DE EVALUACIÓN ---
    for model_type in MODELS_TO_EVALUATE:
        print(f"\n=======================================")
        print(f" EVALUANDO: {model_type}")
        print("=======================================")
        
        config_eval = get_experiment_config(model_type)
        if config_eval is None:
            print(f"Saltando {model_type} (No disponible).")
            continue

        ModelClass = config_eval['model_class']
        DatasetClass = config_eval['dataset_class']
        IN_CHANNELS = config_eval['in_channels']
        WEIGHT_FILENAME = config_eval['weight_filename']
        
        # 1. Crear Loader
        test_dataset = DatasetClass(test_df)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            pin_memory=PIN_MEMORY
        )

        # 2. Instanciar Modelo
        model = ModelClass(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)
        
        # 3. Cargar Pesos
        if model_type == 'ENSEMBLE':
            print("Cargando sub-modelos del Ensemble...")
            try:
                model.load_ensemble_weights(run_id=1, results_dir='resultados', map_location=DEVICE)
                print("Pesos Ensemble cargados correctamente.")
            except Exception as e:
                print(f"Error CRÍTICO cargando pesos del Ensemble: {e}")
                continue
        else:
            MODEL_PATH = f'resultados/{WEIGHT_FILENAME}.pth'
            if not os.path.exists(MODEL_PATH):
                print(f"Error: No existe el archivo de pesos: {MODEL_PATH}")
                continue
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                print(f"Pesos cargados desde: {MODEL_PATH}")
            except Exception as e:
                print(f"Error al cargar pesos para {model_type}: {e}")
                continue

        # 4. Calcular Métricas
        model.eval()
        metrics = get_event_based_metrics(
            model, 
            test_loader, 
            DEVICE,
            prob_threshold=EVENT_PROB_THRESHOLD,
            min_duration=EVENT_MIN_DURATION,
            iou_threshold=EVENT_IOU_THRESHOLD
        )
        all_model_results[model_type] = metrics
        
        print(f"-> Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")
        print(f"-> TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")

        # 5. Graficar Matriz
        output_dir = 'resultados'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        matrix_filename = f'confusion_matrix_percent_{model_type}.png'
        plot_cm_percent(
            metrics['tp'], 
            metrics['fp'], 
            metrics['fn'], 
            metrics,
            model_name=model_type,
            save_path=os.path.join(output_dir, matrix_filename)
        )
        
        # 6. CAPTURA VISUAL
        if target_idx is not None:
            try:
                sample_input, sample_label = test_dataset[target_idx]
                
                if visual_data['signal'] is None:
                    visual_data['signal'] = sample_input[0].cpu().numpy()
                    # Squeeze para quitar dims extra si existen
                    visual_data['gt'] = sample_label.cpu().numpy().squeeze()
                
                input_tensor = sample_input.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out_tensor = model(input_tensor)
                    prob_curve = torch.sigmoid(out_tensor).cpu().numpy().squeeze()
                
                visual_data['preds'][model_type] = prob_curve
            except Exception as e:
                print(f"Error generando visualización para {model_type}: {e}")

    # --- GRAFICAR ---
    if all_model_results:
        plot_comparison_bar_chart(all_model_results, save_dir='resultados')
    else:
        print("\nNo se obtuvieron resultados válidos para graficar barras.")
        
    if target_idx is not None and visual_data['signal'] is not None:
        plot_visual_comparison(
            visual_data['signal'], 
            visual_data['gt'], 
            visual_data['preds'], 
            target_idx, 
            save_dir='resultados'
        )
    else:
        print("\nNo se pudo generar el gráfico de visualización de muestra.")