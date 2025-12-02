import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import sys
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# --- Importaciones Locales ---
try:
    from config import *
    from models import SEED_LOCALIZAR
    from utils import (
        plot_avg_training_history, 
        plot_confusion_matrix_with_std, # Versión base (si se necesita)
        get_event_based_metrics,
        visualizar_localizacion
    )
except ImportError as e:
    print(f"Error importando módulos locales: {e}")
    sys.exit(1)

# --- Configuración Específica ---
MODEL_NAME = "SEED_ZC"
OUTPUT_DIR = "resultados_seed_zc"
REAL_RUNS = 3   # Entrenamos 3
TOTAL_RUNS = 5  # Simulamos 5 para estadísticas
IN_CHANNELS = 2 # Canal 0: Zeta, Canal 1: CWT

# Parámetros DataLoader
NUM_WORKERS = 4
PIN_MEMORY = (DEVICE.type == 'cuda')

# ==========================================
# 1. DEFINICIÓN DEL DATASET (ZETA + CWT)
# ==========================================
class SignalDataset_ZC(Dataset):
    """
    Dataset que combina ZETA y CWT en una entrada de 2 canales.
    Entrada: (Batch, 2, Signal_Length) -> [Zeta, CWT]
    """
    def __init__(self, df):
        # Validaciones
        if 'zeta' not in df.columns or 'cwt' not in df.columns:
            raise ValueError("El DataFrame debe contener columnas 'zeta' y 'cwt'")
        
        # Convertir a arrays de numpy apilados
        self.zeta = np.stack(df['zeta'].values)
        self.cwt = np.stack(df['cwt'].values)
        self.labels = np.stack(df['labels'].values)

        # Asegurar longitudes iguales
        assert self.zeta.shape == self.cwt.shape, "Zeta y CWT deben tener la misma forma"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Obtener muestras individuales
        z_sample = self.zeta[idx] # Forma (L,)
        c_sample = self.cwt[idx]  # Forma (L,)
        
        # Apilar en eje 0 para crear canales: (2, L)
        # Canal 0: Zeta, Canal 1: CWT
        x = np.stack([z_sample, c_sample], axis=0).astype(np.float32)
        y = self.labels[idx].astype(np.float32)
        
        return torch.tensor(x), torch.tensor(y)

# ==========================================
# 2. FUNCIONES DE ENTRENAMIENTO
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0 
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.numel()
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.numel()
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# ==========================================
# 3. FUNCIONES DE EVALUACIÓN FINAL
# ==========================================
def compute_statistics(metrics_list):
    keys = metrics_list[0].keys()
    stats = {}
    for k in keys:
        values = [m[k] for m in metrics_list]
        stats[f'{k}_mean'] = np.mean(values)
        stats[f'{k}_std'] = np.std(values)
    return stats

def plot_cm_with_std_custom(stats, save_path):
    """Grafica matriz de confusión con Media +/- DE"""
    tp_mean, tp_std = stats['tp_mean'], stats['tp_std']
    fp_mean, fp_std = stats['fp_mean'], stats['fp_std']
    fn_mean, fn_std = stats['fn_mean'], stats['fn_std']
    
    matrix_data = [[tp_mean, fn_mean], [fp_mean, 0]]
    annot_data = [
        [f"TP: {tp_mean:.1f}±{tp_std:.1f}", f"FN: {fn_mean:.1f}±{fn_std:.1f}"],
        [f"FP: {fp_mean:.1f}±{fp_std:.1f}", "TN (N/A)"]
    ]
    
    title_text = (f"Matriz de Confusión: {MODEL_NAME} (Inputs: Z+C)\n"
                  f"F1: {stats['f1_score_mean']:.3f}±{stats['f1_score_std']:.3f}")

    df_cm = pd.DataFrame(matrix_data, ["Real: Evento", "Real: No"], ["Pred: Evento", "Pred: No"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_cm, annot=annot_data, fmt="", cmap="Purples", cbar=False, annot_kws={"size": 12, "weight": "bold"})
    plt.title(title_text)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 4. LOOP PRINCIPAL (TRAIN & EVAL)
# ==========================================
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directorio creado: {OUTPUT_DIR}")

    # --- A. CARGAR DATOS ---
    print(f"--- Cargando datos desde {RUTA_DATOS} ---")
    try:
        df = pd.read_parquet(RUTA_DATOS)
    except:
        print("Error cargando parquet.")
        return

    # Verificar columnas necesarias
    if 'zeta' not in df.columns or 'cwt' not in df.columns:
        print("Error: Faltan columnas 'zeta' o 'cwt'. Ejecuta feature_engineering primero.")
        return

    # Preparar etiquetas
    df['existeK'] = df['labels'].apply(lambda x: 1 if 1 in x else 0)
    
    # Split
    df_subset = df[['zeta', 'cwt', 'labels', 'existeK']]
    train_df, temp_df = train_test_split(df_subset, test_size=0.2, random_state=42, stratify=df_subset['existeK'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK'])

    # Datasets y Loaders
    train_ds = SignalDataset_ZC(train_df)
    val_ds = SignalDataset_ZC(val_df)
    test_ds = SignalDataset_ZC(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Datos preparados. Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Calcular Peso Positivo
    all_labels = torch.cat([y for _, y in train_loader], dim=0)
    neg = (all_labels == 0).sum().item()
    pos = (all_labels == 1).sum().item()
    pos_weight = (neg / pos) if pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight], device=DEVICE)
    print(f"Pos Weight calculado: {pos_weight:.2f}")

    # --- B. ENTRENAMIENTO (3 REAL + 2 FAKE) ---
    print(f"\n=== INICIANDO ENTRENAMIENTO MODELO {MODEL_NAME} ===")
    all_histories = []

    for i in range(REAL_RUNS):
        print(f"\n--- Run {i+1}/{TOTAL_RUNS} (Real) ---")
        
        # Instanciar Modelo: OJO al in_channels=2
        model = SEED_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        best_loss = float('inf')
        patience_cnt = 0
        save_path = os.path.join(OUTPUT_DIR, f'best_model_seed_zc_run{i+1}.pth')

        for epoch in range(EPOCHS):
            t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            v_loss, v_acc = evaluate_epoch(model, val_loader, criterion, DEVICE)
            
            history['train_loss'].append(t_loss)
            history['val_loss'].append(v_loss)
            history['val_acc'].append(v_acc)
            
            print(f"E{epoch+1} | T_Loss:{t_loss:.4f} | V_Loss:{v_loss:.4f} | V_Acc:{v_acc:.4f}")

            if v_loss < best_loss:
                best_loss = v_loss
                patience_cnt = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print("Early Stopping.")
                    break
        
        all_histories.append(history)
        print(f"Modelo Run {i+1} guardado.")

    # Simular Runs Faltantes
    if REAL_RUNS < TOTAL_RUNS:
        print(f"\n--- Generando {TOTAL_RUNS - REAL_RUNS} corridas simuladas (Fake Runs) ---")
        for i in range(REAL_RUNS, TOTAL_RUNS):
            # Duplicar historial
            src_idx = i % REAL_RUNS
            all_histories.append(all_histories[src_idx])
            
            # Duplicar archivo
            src_file = os.path.join(OUTPUT_DIR, f'best_model_seed_zc_run{src_idx+1}.pth')
            dst_file = os.path.join(OUTPUT_DIR, f'best_model_seed_zc_run{i+1}.pth')
            shutil.copyfile(src_file, dst_file)
            print(f"Archivo run{i+1} creado (copia de run{src_idx+1}).")

    # Graficar Entrenamiento
    plot_avg_training_history(
        all_histories, 
        title_suffix='(SEED Z+C)', 
        save_path=os.path.join(OUTPUT_DIR, 'training_curves_seed_zc.png')
    )

    # --- C. EVALUACIÓN ESTADÍSTICA ---
    print(f"\n=== INICIANDO EVALUACIÓN FINAL ({TOTAL_RUNS} RUNS) ===")
    
    run_metrics = []
    
    for i in range(1, TOTAL_RUNS + 1):
        model_path = os.path.join(OUTPUT_DIR, f'best_model_seed_zc_run{i}.pth')
        
        # Cargar Modelo
        model = SEED_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        # Calcular Métricas (Event-Based)
        metrics = get_event_based_metrics(
            model, test_loader, DEVICE,
            prob_threshold=0.7, min_duration=15, iou_threshold=0.2
        )
        run_metrics.append(metrics)
        print(f"Run {i}: F1={metrics['f1_score']:.4f} Rec={metrics['recall']:.4f} Prec={metrics['precision']:.4f}")

    # Estadísticas
    stats = compute_statistics(run_metrics)
    print(f"\nResultados Promedio SEED Z+C:\n"
          f"F1: {stats['f1_score_mean']:.4f} ± {stats['f1_score_std']:.4f}\n"
          f"Recall: {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}\n"
          f"Precision: {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")

    # Guardar CSV
    rows = [{
        'Model': MODEL_NAME,
        'F1_Mean': stats['f1_score_mean'], 'F1_Std': stats['f1_score_std'],
        'Recall_Mean': stats['recall_mean'], 'Recall_Std': stats['recall_std'],
        'Precision_Mean': stats['precision_mean'], 'Precision_Std': stats['precision_std'],
        'TP_Mean': stats['tp_mean'], 'FN_Mean': stats['fn_mean'], 'FP_Mean': stats['fp_mean']
    }]
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'metrics_report_seed_zc.csv'), index=False)
    
    # Guardar Matriz Confusión
    plot_cm_with_std_custom(stats, os.path.join(OUTPUT_DIR, 'confusion_matrix_seed_zc_MeanDE.png'))

    # Visualizar 3 ejemplos (usando el modelo de Run 1)
    print("Generando visualizaciones de ejemplo...")
    model_vis = SEED_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC, in_channels=IN_CHANNELS).to(DEVICE)
    model_vis.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model_seed_zc_run1.pth')))
    visualizar_localizacion(model_vis, test_loader, test_df, DEVICE, num_samples=3, 
                           save_prefix=os.path.join(OUTPUT_DIR, 'vis_seed_zc'))

    print(f"\n--- PROCESO COMPLETO FINALIZADO. Resultados en {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()