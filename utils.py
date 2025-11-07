# src/utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.utils import resample
import pandas as pd  # Importar pandas


def plot_training_history(history, title_suffix='', save_path=None):
    """
    Grafica las curvas de pérdida y precisión de un único entrenamiento.
    
    Args:
        history (dict): Un diccionario con listas 'train_loss', 'val_loss',
                        'train_acc', y 'val_acc'.
        title_suffix (str, optional): Sufijo para añadir a los títulos.
        save_path (str, optional): Ruta para guardar el gráfico.
    """
    plt.figure(figsize=(12, 5))

    # Gráfico de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Pérdida de Entrenamiento')
    plt.plot(history['val_loss'], label='Pérdida de Validación')
    plt.title(f'Pérdida a lo largo de las Épocas {title_suffix}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Gráfico de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Precisión de Entrenamiento')
    plt.plot(history['val_acc'], label='Precisión de Validación')
    plt.title(f'Precisión a lo largo de las Épocas {title_suffix}')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # --- AÑADIDO ---
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico de historial guardado en: {save_path}")
    plt.show()


def plot_avg_training_history(all_histories, title_suffix='', best_epoch=None, save_path=None):
    """
    Grafica la media y desviación estándar de múltiples historiales de
    entrenamiento.
    
    Args:
        all_histories (list[dict]): Lista de diccionarios de historial.
        title_suffix (str, optional): Sufijo para añadir a los títulos.
        best_epoch (int, optional): Época del mejor modelo para marcar.
        save_path (str, optional): Ruta para guardar el gráfico.
    """
    # 1. Encontrar la longitud de la corrida más corta
    min_epochs = min([len(h['train_loss']) for h in all_histories])
    if min_epochs == 0:
        print("Error: No hay datos de historial para graficar.")
        return

    # 2. Truncar todas las listas de historial
    train_loss_all = np.array([h['train_loss'][:min_epochs] for h in all_histories])
    val_loss_all = np.array([h['val_loss'][:min_epochs] for h in all_histories])
    train_acc_all = np.array([h['train_acc'][:min_epochs] for h in all_histories])
    val_acc_all = np.array([h['val_acc'][:min_epochs] for h in all_histories])

    # 3. Calcular promedio (mean) y desviación estándar (std)
    mean_train_loss = np.mean(train_loss_all, axis=0)
    std_train_loss = np.std(train_loss_all, axis=0)
    mean_val_loss = np.mean(val_loss_all, axis=0)
    std_val_loss = np.std(val_loss_all, axis=0)
    mean_train_acc = np.mean(train_acc_all, axis=0)
    std_train_acc = np.std(train_acc_all, axis=0)
    mean_val_acc = np.mean(val_acc_all, axis=0)
    std_val_acc = np.std(val_acc_all, axis=0)

    epochs_range = range(1, min_epochs + 1)

    # --- Graficar resultados ---
    plt.figure(figsize=(15, 6))

    # Gráfico de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, mean_val_loss, label='Validación (Media)', color='tab:blue', lw=2)
    plt.fill_between(epochs_range, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='tab:blue', alpha=0.3, label='Validación (DE)')
    plt.plot(epochs_range, mean_train_loss, label='Entrenamiento (Media)', color='tab:orange', lw=2)
    plt.fill_between(epochs_range, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color='tab:orange', alpha=0.3, label='Entrenamiento (DE)')
    plt.title(f'Evolución de Pérdida (Media ± DE) {title_suffix}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- AÑADIDO: Marcar el mejor epoch ---
    if best_epoch is not None and best_epoch <= min_epochs:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Mejor Época ({best_epoch})')
        plt.legend()

    # Gráfico de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, mean_val_acc, label='Validación (Media)', color='tab:blue', lw=2)
    plt.fill_between(epochs_range, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc, color='tab:blue', alpha=0.3, label='Validación (DE)')
    plt.plot(epochs_range, mean_train_acc, label='Entrenamiento (Media)', color='tab:orange', lw=2)
    plt.fill_between(epochs_range, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, color='tab:orange', alpha=0.3, label='Entrenamiento (DE)')
    plt.title(f'Evolución de Precisión (Media ± DE) {title_suffix}')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # --- AÑADIDO: Marcar el mejor epoch ---
    if best_epoch is not None and best_epoch <= min_epochs:
        plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Mejor Época ({best_epoch})')
        plt.legend()

    plt.tight_layout()
    
    # --- AÑADIDO ---
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico de historial promedio guardado en: {save_path}")
    plt.show()


def get_test_metrics(model, dataloader, device, task_type='detectar'):
    """
    Evalúa el modelo en el dataloader de prueba y devuelve métricas y etiquetas.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"--- Métricas en Conjunto de Prueba ({task_type}) ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return all_labels, all_preds


def plot_simple_confusion_matrix(model, dataloader, device, task_type='detectar', save_path=None):
    """
    Grafica una matriz de confusión simple normalizada por filas ('true').
    """
    print("Generando matriz de confusión simple...")
    all_labels, all_preds = get_test_metrics(model, dataloader, device, task_type)
    
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=["No K", "K"], yticklabels=["No K", "K"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión normalizada (por filas)")
    
    # --- AÑADIDO ---
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico de matriz de confusión guardado en: {save_path}")
    plt.show()


def plot_confusion_matrix_with_std(model, dataloader, device, n_bootstraps=1000, save_path=None):
    """
    Calcula y grafica la matriz de confusión normalizada (por filas) con 
    media y desviación estándar usando bootstrapping.
    """
    print(f"Generando matriz de confusión con bootstrapping (n={n_bootstraps})...")
    model.eval()
    all_preds = []
    all_labels = []

    # 1. Obtener todas las predicciones y etiquetas una sola vez
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 2. Realizar el bootstrapping
    bootstrapped_cms = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(all_preds)))
        boot_labels = all_labels[indices]
        boot_preds = all_preds[indices]

        cm = confusion_matrix(boot_labels, boot_preds, normalize='true')

        if cm.shape == (1, 1):
            if boot_labels[0] == 0: cm = np.array([[cm[0,0], 0], [0, 0]])
            else: cm = np.array([[0, 0], [0, cm[0,0]]])
        
        if cm.shape[0] < 2: 
            continue 

        bootstrapped_cms.append(cm)

    if not bootstrapped_cms:
        print("Error: No se pudieron generar matrices de bootstrapping válidas.")
        return

    # 3. Calcular la media y la desviación estándar
    bootstrapped_cms = np.array(bootstrapped_cms)
    cm_mean = np.mean(bootstrapped_cms, axis=0)
    cm_std = np.std(bootstrapped_cms, axis=0)

    # 4. Crear las etiquetas de texto
    annot_labels = np.empty_like(cm_mean, dtype=object)
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            annot_labels[i, j] = f"{cm_mean[i, j]:.2f} ± {cm_std[i, j]:.2f}"

    # 5. Graficar
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_mean, annot=annot_labels, fmt="", cmap="Blues",
                xticklabels=["No K", "K"], yticklabels=["No K", "K"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión normalizada (Media ± DE)")
    
    # --- AÑADIDO ---
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico de matriz de confusión (Std) guardado en: {save_path}")
    plt.show()


def visualizar_localizacion(model, dataloader, original_df, device, num_samples=3, save_prefix=None):
    """
    Grafica y guarda la señal original, la verdad (ground truth) y las predicciones
    del modelo de localización.
    """
    model.eval()
    
    with torch.no_grad():
        signals_batch, _ = next(iter(dataloader))
        signals_batch = signals_batch.to(device) 

        logits_500 = model(signals_batch)
        logits_4000 = F.interpolate(logits_500, scale_factor=8, mode='nearest')
        preds_4000_batch = (torch.sigmoid(logits_4000) > 0.5).float().cpu().numpy()
        signals_4000_batch = signals_batch.cpu().numpy()
        batch_size_real = signals_batch.size(0)
        original_labels_4000_batch = np.array(original_df['labels'].tolist()[:batch_size_real])

    print(f"\n--- Mostrando y guardando {num_samples} ejemplos de localización (Azul=0, Rojo=1) ---")

    for i in range(min(num_samples, len(signals_batch))):
        signal_to_plot = signals_4000_batch[i].squeeze()
        true_labels_to_plot = original_labels_4000_batch[i].squeeze()
        pred_labels_to_plot = preds_4000_batch[i].squeeze()

        time_axis = np.arange(4000)
        plt.figure(figsize=(16, 7))
        plt.suptitle(f"Muestra de Prueba #{i}", fontsize=16)

        plt.subplot(2, 1, 1)
        plt.scatter(time_axis, signal_to_plot, c=true_labels_to_plot, cmap='coolwarm', s=5, vmin=0, vmax=1)
        plt.title("Etiqueta Verdadera (Ground Truth)")
        plt.ylabel("Amplitud (µV)")
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.subplot(2, 1, 2)
        plt.scatter(time_axis, signal_to_plot, c=pred_labels_to_plot, cmap='coolwarm', s=5, vmin=0, vmax=1)
        plt.title("Predicción del Modelo")
        plt.xlabel("Muestra (Sample index)")
        plt.ylabel("Amplitud (µV)")
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # --- AÑADIDO ---
        if save_prefix:
            save_path = f"{save_prefix}_sample_{i}.png"
            plt.savefig(save_path)
            print(f"Visualización de muestra guardada en: {save_path}")
        plt.show()

# --- FUNCIÓN NUEVA ---
def generate_metrics_report(model, dataloader, device, save_path):
    """
    Genera un reporte de clasificación detallado (con support) y lo guarda
    como un archivo CSV.
    
    Args:
        model (nn.Module): El modelo entrenado.
        dataloader (DataLoader): El dataloader de prueba.
        device (torch.device): El dispositivo ('cuda' o 'cpu').
        save_path (str): Ruta para guardar el archivo CSV.
    """
    print("Generando reporte de métricas...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Generar el reporte
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=["No K (0)", "K (1)"], 
        output_dict=True
    )
    
    # Convertir a DataFrame y guardar
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(4) # Redondear a 4 decimales
    
    try:
        df_report.to_csv(save_path)
        print(f"Reporte de métricas guardado exitosamente en: {save_path}")
        print("\n--- Reporte de Métricas (Test Set) ---")
        print(df_report)
        print("-------------------------------------------\n")
    except Exception as e:
        print(f"Error al guardar el reporte CSV: {e}")