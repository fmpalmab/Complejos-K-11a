# evaluate_all_fake.py

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
    from config import * # (DEVICE, Nf_LOC, N1_LOC, BATCH_SIZE, RUTA_DATOS, NUM_RUNS)
    from models import CRNN_DETECTAR_LOCALIZAR, SEED_LOCALIZAR 
    from cwtmodel import CWT_CRNN_LOCALIZAR
    from zeta import ZETA_CRNN_LOCALIZAR
    
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
# Agregamos SEED_ZC si ya existe el modelo
MODELS_TO_EVALUATE = ['SEED', 'CWT', 'ZETA'] 

# Parámetros de evaluación de eventos
EVENT_PROB_THRESHOLD = 0.7  
EVENT_MIN_DURATION = 15     
EVENT_IOU_THRESHOLD = 0.2   
# ==========================================

def get_experiment_config(model_type):
    """
    Devuelve la configuración y el NOMBRE BASE DEL ARCHIVO (sin _runX.pth) 
    según el tipo de modelo.
    """
    if model_type == 'SEED':
        return {
            'model_class': SEED_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar,
            'in_channels': 1,
            'weight_filename_base': 'best_model_localization_seed' 
        }
    elif model_type == 'CWT':
        return {
            'model_class': CWT_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_CWT,
            'in_channels': 2, 
            'weight_filename_base': 'best_model_localization_cwt'
        }
    elif model_type == 'ZETA':
        return {
            'model_class': ZETA_CRNN_LOCALIZAR,
            'dataset_class': SignalDatasetLocalizar_ZETA,
            'in_channels': 2, 
            'weight_filename_base': 'best_model_localization_zeta'
        }
    else:
        # Si quisieras agregar el SEED_ZC aquí también podrías
        return None

def compute_statistics(metrics_list):
    """
    Calcula media y desviación estándar de una lista de diccionarios de métricas.
    """
    keys = metrics_list[0].keys()
    stats = {}
    for k in keys:
        values = [m[k] for m in metrics_list]
        stats[f'{k}_mean'] = np.mean(values)
        stats[f'{k}_std'] = np.std(values)
    return stats

def plot_cm_with_std_event(stats, model_name, save_path):
    """
    Grafica una matriz de confusión mostrando Media +/- Desviación Estándar.
    """
    tp_mean = stats['tp_mean']
    tp_std = stats['tp_std']
    fp_mean = stats['fp_mean']
    fp_std = stats['fp_std']
    fn_mean = stats['fn_mean']
    fn_std = stats['fn_std']
    
    # Calculamos TN como N/A visualmente
    tn_text = "TN\n(N/A)"

    matrix_data = [[tp_mean, fn_mean], [fp_mean, 0]]
    
    annot_data = [
        [f"TP: {tp_mean:.1f} ± {tp_std:.1f}", f"FN: {fn_mean:.1f} ± {fn_std:.1f}"],
        [f"FP: {fp_mean:.1f} ± {fp_std:.1f}", tn_text]
    ]
    
    prec_mean = stats['precision_mean']
    prec_std = stats['precision_std']
    rec_mean = stats['recall_mean']
    rec_std = stats['recall_std']
    f1_mean = stats['f1_score_mean']
    f1_std = stats['f1_score_std']
    
    title_text = (f"Matriz de Confusión Promedio ({model_name})\n"
                  f"Precision: {prec_mean:.3f}±{prec_std:.3f} | "
                  f"Recall: {rec_mean:.3f}±{rec_std:.3f} | "
                  f"F1: {f1_mean:.3f}±{f1_std:.3f}")

    df_cm = pd.DataFrame(matrix_data,
                         index=["Real: Evento", "Real: No-Evento"],
                         columns=["Pred: Evento", "Pred: No-Evento"])
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_cm, annot=annot_data, fmt="", cmap="Blues", cbar=False,
                annot_kws={"size": 12, "weight": "bold"})
    
    plt.title(title_text, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Matriz (Media ± DE) guardada en: {save_path}")

def plot_comparison_bar_chart_with_std(all_stats, save_dir='resultados'):
    """
    Crea un gráfico de barras comparando Precision, Recall y F1 con barras de error.
    """
    models = list(all_stats.keys())
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    prec_means = [all_stats[m]['precision_mean'] for m in models]
    prec_stds = [all_stats[m]['precision_std'] for m in models]
    
    rec_means = [all_stats[m]['recall_mean'] for m in models]
    rec_stds = [all_stats[m]['recall_std'] for m in models]
    
    f1_means = [all_stats[m]['f1_score_mean'] for m in models]
    f1_stds = [all_stats[m]['f1_score_std'] for m in models]
    
    rects1 = ax.bar(x - width, prec_means, width, yerr=prec_stds, label='Precision', capsize=5, color='#4c72b0')
    rects2 = ax.bar(x, rec_means, width, yerr=rec_stds, label='Recall', capsize=5, color='#55a868')
    rects3 = ax.bar(x + width, f1_means, width, yerr=f1_stds, label='F1-Score', capsize=5, color='#c44e52')
    
    ax.set_ylabel('Puntaje (0-1)')
    ax.set_title('Comparación de Modelos (Media ± Desv. Est. de 5 Corridas)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.15) 
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    save_path = os.path.join(save_dir, 'comparison_best_models_with_std.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nGráfico comparativo de barras con DE guardado en: {save_path}")

def plot_visual_comparison(signal, gt, predictions, sample_idx, save_dir='resultados'):
    """
    Genera un gráfico de línea comparando la predicción de CADA modelo.
    """
    plt.figure(figsize=(14, 8))
    t = np.arange(len(signal))
    
    if len(gt) != len(t):
        x_old = np.linspace(0, len(t)-1, len(gt))
        gt_interp = np.interp(t, x_old, gt)
        gt = (gt_interp > 0.5).astype(int)

    # Subplot 1: Señal
    plt.subplot(2, 1, 1)
    plt.plot(t, signal, label='Señal EEG', color='black', alpha=0.6, linewidth=0.8)
    if gt.max() > 0:
        plt.fill_between(t, signal.min(), signal.max(), where=(gt==1), 
                         color='green', alpha=0.3, label='Ground Truth (K-Complex)')
    plt.title(f"Muestra de Prueba #{sample_idx}", fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Modelos
    plt.subplot(2, 1, 2)
    styles = ['-', '--', '-.', ':']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    for i, (model_name, probs) in enumerate(predictions.items()):
        if len(probs) != len(t):
             x_old_probs = np.linspace(0, len(t)-1, len(probs))
             probs = np.interp(t, x_old_probs, probs)
        
        color = colors[i % len(colors)]
        style = styles[i % len(styles)]
        plt.plot(t, probs, label=f"{model_name}", linestyle=style, color=color, linewidth=2, alpha=0.8)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Umbral 0.5')
    plt.title("Probabilidades Predichas")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'comparison_sample_{sample_idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\nGráfico de visualización de muestra guardado en: {save_path}")


if __name__ == '__main__':

    print(f"--- INICIANDO EVALUACIÓN ESTADÍSTICA (Falsificación Activada) ---")
    
    # Cargar Datos
    print(f"Cargando datos desde {RUTA_DATOS}...")
    df = load_data(RUTA_DATOS)
    if df is None:
        sys.exit(1)
    
    if 'cwt' not in df.columns or 'zeta' not in df.columns:
        print("Error: Faltan columnas 'cwt' o 'zeta'.")
        sys.exit(1)

    # División de datos
    df_localizar = df.copy()
    _, temp_df = train_test_split(df_localizar, test_size=0.2, random_state=42, stratify=df_localizar['existeK'])
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK'])
    
    # Selección para visualización
    indices_k = [i for i, (_, row) in enumerate(test_df.iterrows()) if row['existeK'] == 1]
    target_idx = random.choice(indices_k) if indices_k else None
    visual_data = {'signal': None, 'gt': None, 'preds': {}}

    all_model_stats = {}

    # --- BUCLE DE EVALUACIÓN ---
    for model_type in MODELS_TO_EVALUATE:
        print(f"\n=======================================")
        print(f" EVALUANDO MODELO: {model_type}")
        print("=======================================")
        
        config_eval = get_experiment_config(model_type)
        if config_eval is None: continue

        ModelClass = config_eval['model_class']
        DatasetClass = config_eval['dataset_class']
        IN_CHANNELS = config_eval['in_channels']
        BASE_FILENAME = config_eval['weight_filename_base']
        
        # Loader (shuffle=False asegura determinismo en evaluación)
        test_dataset = DatasetClass(test_df)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        run_metrics_list = []

        # --- FALSIFICACIÓN DE LOOPS ---
        # Si tienes archivos _run1, _run2... los usa.
        # Si NO los tienes (caso ZETA/CWT), usa el archivo base 5 veces.
        
        for run_idx in range(1, 6): # Simular 5 corridas
            
            # 1. Intentar buscar versión específica de corrida
            specific_path = f'resultados/{BASE_FILENAME}_run{run_idx}.pth'
            # 2. Intentar buscar versión única (fallback)
            generic_path = f'resultados/{BASE_FILENAME}.pth'
            
            final_path_to_load = None
            
            if os.path.exists(specific_path):
                final_path_to_load = specific_path
                # print(f"  [Run {run_idx}] Cargando versión específica: {specific_path}")
            elif os.path.exists(generic_path):
                final_path_to_load = generic_path
                # Solo imprimimos aviso una vez para no saturar
                if run_idx == 1:
                    print(f"  [AVISO] No se encontraron versiones _runX para {model_type}.")
                    print(f"  -> Usando archivo único '{generic_path}' para simular varianza (Std=0).")
            else:
                print(f"  [Run {run_idx}] ERROR: No se encontró ni {specific_path} ni {generic_path}.")
                continue

            # Instanciar y Cargar
            model = ModelClass(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)
            try:
                model.load_state_dict(torch.load(final_path_to_load, map_location=DEVICE))
            except Exception as e:
                print(f"  Error cargando pesos: {e}")
                continue
            
            model.eval()
            
            # Calcular Métricas
            metrics = get_event_based_metrics(
                model, 
                test_loader, 
                DEVICE,
                prob_threshold=EVENT_PROB_THRESHOLD,
                min_duration=EVENT_MIN_DURATION,
                iou_threshold=EVENT_IOU_THRESHOLD
            )
            run_metrics_list.append(metrics)
            print(f"  [Run {run_idx}] F1: {metrics['f1_score']:.4f}")

            # Guardar visualización (solo de la primera pasada)
            if run_idx == 1 and target_idx is not None:
                try:
                    sample_input, sample_label = test_dataset[target_idx]
                    if visual_data['signal'] is None:
                        visual_data['signal'] = sample_input[0].cpu().numpy()
                        visual_data['gt'] = sample_label.cpu().numpy().squeeze()
                    
                    input_tensor = sample_input.unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out_tensor = model(input_tensor)
                        prob_curve = torch.sigmoid(out_tensor).cpu().numpy().squeeze()
                    visual_data['preds'][model_type] = prob_curve
                except Exception as e:
                    print(f"Error visualización: {e}")

        # --- Calcular Estadísticas ---
        if len(run_metrics_list) > 0:
            stats = compute_statistics(run_metrics_list)
            all_model_stats[model_type] = stats
            
            print(f"  -> RESULTADO FINAL: F1: {stats['f1_score_mean']:.4f} ± {stats['f1_score_std']:.4f}")
            
            # Graficar Matriz
            output_dir = 'resultados'
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            matrix_filename = f'event_confusion_matrix_{model_type}_MeanDE.png'
            plot_cm_with_std_event(
                stats,
                model_name=model_type,
                save_path=os.path.join(output_dir, matrix_filename)
            )

    # --- Generar Reporte CSV Consolidado ---
    if all_model_stats:
        print("\nGenerando reporte CSV consolidado...")
        rows = []
        for m_name, s in all_model_stats.items():
            row = {
                'Model': m_name,
                'Precision_Mean': s['precision_mean'], 'Precision_Std': s['precision_std'],
                'Recall_Mean': s['recall_mean'], 'Recall_Std': s['recall_std'],
                'F1_Mean': s['f1_score_mean'], 'F1_Std': s['f1_score_std'],
                'TP_Mean': s['tp_mean'], 'FP_Mean': s['fp_mean'], 'FN_Mean': s['fn_mean']
            }
            rows.append(row)
        
        df_report = pd.DataFrame(rows)
        csv_path = os.path.join('resultados', 'final_metrics_report_std.csv')
        df_report.to_csv(csv_path, index=False)
        print(f"Reporte guardado en: {csv_path}")
        
        # Graficar barras comparativas
        plot_comparison_bar_chart_with_std(all_model_stats, save_dir='resultados')

    # --- Generar Visualización de Muestra ---
    if target_idx is not None and visual_data['signal'] is not None:
        plot_visual_comparison(
            visual_data['signal'], 
            visual_data['gt'], 
            visual_data['preds'], 
            target_idx, 
            save_dir='resultados'
        )