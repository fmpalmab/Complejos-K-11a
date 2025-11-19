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
    #plt.show()


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
    #plt.show()


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
    #plt.show()


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
    #plt.show()


def visualizar_localizacion(model, dataloader, original_df, device, num_samples=3, save_prefix=None):
    """
    Grafica y guarda la señal original, la verdad (ground truth) y las predicciones
    del modelo de localización.
    
    --- MODIFICADO ---
    Maneja tensores de entrada de 1 o 2 canales (B, C, T).
    Siempre grafica el canal 0 (índice 0) que se asume es la señal original.
    """
    model.eval()
    
    with torch.no_grad():
        signals_batch, _ = next(iter(dataloader))
        signals_batch = signals_batch.to(device) 

        logits_500 = model(signals_batch)
        logits_4000 = F.interpolate(logits_500, scale_factor=8, mode='nearest')
        preds_4000_batch = (torch.sigmoid(logits_4000) > 0.5).float().cpu().numpy()
        
        # --- MODIFICACIÓN ---
        # Seleccionar solo el canal 0 (señal original) para visualización
        # signals_batch puede ser (B, 1, 4000) o (B, 2, 4000)
        signals_4000_batch = signals_batch.cpu().numpy()
        # --------------------
        
        batch_size_real = signals_batch.size(0)
        original_labels_4000_batch = np.array(original_df['labels'].tolist()[:batch_size_real])

    print(f"\n--- Mostrando y guardando {num_samples} ejemplos de localización (Azul=0, Rojo=1) ---")

    for i in range(min(num_samples, len(signals_batch))):
        
        # --- MODIFICACIÓN ---
        # Seleccionar el canal 0 (índice 0) de la señal
        signal_to_plot = signals_4000_batch[i, 0].squeeze() 
        # --------------------
        
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
        #plt.show()

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




def post_process_output(probs, prob_threshold, min_duration):
    """
    Convierte probabilidades en una máscara binaria limpia de eventos.
    
    Args:
        probs (np.array): Array de probabilidades de salida del modelo (ej. shape 500).
        prob_threshold (float): Umbral para considerar una muestra como 'positiva'.
        min_duration (int): Duración mínima (en muestras) para que un 
                               evento sea considerado válido.
    
    Returns:
        np.array: Máscara binaria limpia (0s y 1s) con la misma forma que 'probs'.
    """
    
    # 1. Aplicar umbral de probabilidad
    binary_output = (probs > prob_threshold).astype(int)
    
    if binary_output.sum() == 0:
        return binary_output # No hay nada que procesar
        
    # 2. Filtrar eventos por duración (encontrar bloques continuos de 1s)
    cleaned_output = np.zeros_like(binary_output)
    
    # Encuentra los bordes donde cambia de 0 a 1 o de 1 a 0
    diff = np.diff(np.concatenate(([0], binary_output, [0])))
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]
    
    # Para cada evento detectado...
    for start, end in zip(start_indices, end_indices):
        duration = end - start
        
        # 3. ...aplicar el filtro de duración mínima
        if duration >= min_duration:
            cleaned_output[start:end] = 1 # Marcar este evento como válido
            
    return cleaned_output


def post_process_output(probs, prob_threshold, min_duration):
    """
    Convierte un array de probabilidades en una máscara binaria limpia de eventos.
    
    Args:
        probs (np.array): Array de probabilidades (ej. shape 500).
        prob_threshold (float): Umbral para considerar una muestra como 'positiva' (ej. 0.5).
        min_duration (int): Duración mínima (en muestras) para que un 
                               evento sea considerado válido (ej. 10).
    
    Returns:
        np.array: Máscara binaria limpia (0s y 1s).
    """
    
    # 1. Aplicar umbral de probabilidad
    binary_output = (probs > prob_threshold).astype(int)
    
    if binary_output.sum() == 0:
        return binary_output # No hay nada que procesar
        
    # 2. Encontrar bloques continuos de 1s
    cleaned_output = np.zeros_like(binary_output)
    diff = np.diff(np.concatenate(([0], binary_output, [0])))
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]
    
    # 3. Filtrar por duración mínima
    for start, end in zip(start_indices, end_indices):
        duration = end - start
        if duration >= min_duration:
            cleaned_output[start:end] = 1 # Marcar este evento como válido
            
    return cleaned_output

def get_events(binary_mask):
    """
    Convierte una máscara binaria (ej. [0,0,1,1,1,0,1,1,0]) en una
    lista de eventos [start, end] (ej. [[2, 5], [6, 8]]).
    
    Args:
        binary_mask (np.array): Array de 0s y 1s.
    
    Returns:
        list: Lista de listas, donde cada sub-lista es [start_index, end_index].
    """
    if binary_mask.sum() == 0:
        return []
        
    diff = np.diff(np.concatenate(([0], binary_mask, [0])))
    start_indices = np.where(diff == 1)[0]
    end_indices = np.where(diff == -1)[0]
    
    # Restamos 1 al end_index para que sea inclusivo, 
    # pero para IoU es mejor que sea exclusivo [start, end)
    # Así que lo dejamos como está: [2, 5] significa índices 2, 3, 4.
    return [[start, end] for start, end in zip(start_indices, end_indices)]


def calculate_iou(event_a, event_b):
    """
    Calcula el Intersection over Union (IoU) para dos eventos 1D.
    Un evento es una lista [start, end].
    
    Args:
        event_a (list): [start_a, end_a]
        event_b (list): [start_b, end_b]
        
    Returns:
        float: El valor de IoU (entre 0.0 y 1.0).
    """
    # Determinar los puntos de la intersección
    inter_start = max(event_a[0], event_b[0])
    inter_end = min(event_a[1], event_b[1])
    
    inter_duration = max(0, inter_end - inter_start)
    
    if inter_duration == 0:
        return 0.0 # No hay superposición
        
    # Calcular duraciones de cada evento
    duration_a = event_a[1] - event_a[0]
    duration_b = event_b[1] - event_b[0]
    
    # Calcular la unión
    union_duration = duration_a + duration_b - inter_duration
    
    if union_duration == 0:
        return 0.0 # Evitar división por cero
        
    iou = inter_duration / union_duration
    return iou



def get_event_based_metrics(model, dataloader, device, 
                            prob_threshold=0.5, 
                            min_duration=10, 
                            iou_threshold=0.5):
    """
    Calcula métricas de Precision, Recall y F1-Score basadas en eventos (IoU).
    
    Args:
        model (torch.nn.Module): El modelo entrenado.
        dataloader (DataLoader): El dataloader de prueba (test_loader).
        device (torch.device): 'cuda' o 'cpu'.
        prob_threshold (float): Umbral de probabilidad para post-procesamiento.
        min_duration (int): Duración mínima de evento para post-procesamiento.
        iou_threshold (float): Umbral de IoU para considerar una detección como
                                 True Positive.
    
    Returns:
        dict: Un diccionario con 'precision', 'recall', y 'f1_score'.
    """
    model.eval()
    
    total_tp = 0 # True Positives
    total_fp = 0 # False Positives
    total_fn = 0 # False Negatives

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Asumo que tu dataloader de localización devuelve (inputs, labels_500)
            # Si tu dataloader devuelve (inputs, cwt, labels) o algo así, 
            # ajusta esta línea para tomar solo 'inputs' y 'labels'.
            
            # Ejemplo si tu dataset devuelve (signal, cwt, label):
            # (inputs, _, labels) = batch_data 
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs) # Shape: (batch_size, 1, 500) o (batch_size, 500)
            
            # Convertir a probabilidades y mover a numpy
            probs = torch.sigmoid(outputs).cpu().numpy()
            true_labels = labels.cpu().numpy()
            
            # Iterar por cada señal en el lote
            for i in range(probs.shape[0]):
                prob_signal = probs[i].squeeze() # Quitar dims de 1 (ej. (1, 500) -> (500,))
                label_signal = true_labels[i].squeeze()
                
                # 1. Aplicar post-procesamiento para obtener eventos PREDICHOS
                pred_mask = post_process_output(prob_signal, prob_threshold, min_duration)
                
                # 2. Obtener listas de eventos [start, end]
                true_events = get_events(label_signal)
                pred_events = get_events(pred_mask)
                
                # 3. Lógica de conteo de TP, FP, FN
                if not true_events and not pred_events:
                    continue # No hay nada, no se suma nada.
                
                if not pred_events:
                    total_fn += len(true_events) # El modelo no predijo nada, todos son FN
                    continue
                    
                if not true_events:
                    total_fp += len(pred_events) # El modelo predijo cosas que no existen, todos son FP
                    continue

                # --- Lógica de "Matching" ---
                # Comparamos cada evento VERDADERO con todos los PREDICHOS
                
                current_tp = 0
                matched_pred_indices = set() # Para no contar un predicho 2 veces

                for true_event in true_events:
                    best_iou = 0
                    best_pred_idx = -1
                    
                    for idx, pred_event in enumerate(pred_events):
                        if idx in matched_pred_indices:
                            continue # Este ya se usó para otro TP
                            
                        iou = calculate_iou(true_event, pred_event)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_pred_idx = idx
                            
                    # Si el mejor match supera el umbral, es un TP
                    if best_iou >= iou_threshold:
                        current_tp += 1
                        matched_pred_indices.add(best_pred_idx)
                
                # --- Calcular TP, FP, FN para esta señal ---
                total_tp += current_tp
                total_fn += len(true_events) - current_tp # Verdaderos que no tuvieron match
                total_fp += len(pred_events) - len(matched_pred_indices) # Predichos que no tuvieron match

    # --- Calcular Métricas Finales (después de ver todo el test set) ---
    # Usamos "epsilon" (1e-6) para evitar división por cero
    epsilon = 1e-6 
    
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    print(f"\n--- Métricas Basadas en Eventos (IoU > {iou_threshold}) ---")
    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print("--------------------------------------------------")
    print(f"Precisión: {precision:.4f}")
    print(f"Recall (Sensibilidad): {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    
# --- ¡CAMBIO AQUÍ! ---
    # Modifica el return para que devuelva todo
    
    return {
        'precision': precision, 
        'recall': recall, 
        'f1_score': f1_score,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

def plot_event_confusion_matrix(tp, fp, fn, save_path):
    """
    Crea y guarda una matriz de confusión 2x2 basada en eventos (TP, FP, FN).
    El cuadrante TN (True Negative) se marca como 'No Aplicable' 
    porque este método no los cuenta (¡lo cual es bueno!).
    """
    
    # Creamos la matriz de datos numéricos para el color
    # (Usamos 0 para TN, ya que no nos importa su color)
    matrix_data = [
        [tp, fn],
        [fp, 0] 
    ]
    
    # Creamos la matriz de etiquetas de texto
    # Esto es lo que se escribirá dentro de cada celda
    annot_data = [
        [f"True Positive (TP)\n{tp}", f"False Negative (FN)\n{fn}"],
        [f"False Positive (FP)\n{fp}", "True Negative (TN)\n(No Aplicable)"]
    ]
    
    # Creamos un DataFrame de pandas para que Seaborn lo entienda
    df_cm = pd.DataFrame(matrix_data,
                         index=["Real: Evento", "Real: No-Evento"],
                         columns=["Pred: Evento", "Pred: No-Evento"])
                         
    # Crear la figura
    plt.figure(figsize=(10, 8))
    
    # Crear el heatmap
    # 'annot=annot_data' usa nuestras etiquetas de texto
    # 'fmt=""' le dice que no intente formatear las etiquetas como números
    # 'cbar=False' quita la barra de color (no es necesaria aquí)
    sns.heatmap(df_cm, 
                annot=annot_data, 
                fmt="", 
                cmap="Blues", 
                cbar=False,
                annot_kws={"size": 14}) # Tamaño de letra
                
    plt.title("Matriz de Confusión Basada en Eventos", fontsize=16)
    plt.ylabel("Realidad", fontsize=12)
    plt.xlabel("Predicción", fontsize=12)
    
    # Guardar la figura
    try:
        plt.savefig(save_path)
        plt.close()
        print(f"Matriz de confusión de eventos guardada en: {save_path}")
    except Exception as e:
        print(f"Error al guardar la matriz de confusión: {e}")


def plot_event_confusion_matrix_with_std(tp_mean, tp_std, fp_mean, fp_std, fn_mean, fn_std, save_path):
    """
    Crea y guarda una matriz de confusión 2x2 basada en eventos con
    Media y Desviación Estándar.
    """
    
    # --- Calcular métricas (con epsilon para evitar división por cero) ---
    epsilon = 1e-6 
    precision = tp_mean / (tp_mean + fp_mean + epsilon)
    recall = tp_mean / (tp_mean + fn_mean + epsilon)
    miss_rate = fn_mean / (tp_mean + fn_mean + epsilon)
    
    # Creamos la matriz de datos numéricos para el color
    matrix_data = [
        [tp_mean, fn_mean],
        [fp_mean, 0] 
    ]
    
    # --- NUEVO: Anotaciones con Media ± DE ---
    annot_data = [
        [f"True Positive (TP)\n{tp_mean:.1f} ± {tp_std:.1f}\n\nRecall: {recall:.2%}", 
         f"False Negative (FN)\n{fn_mean:.1f} ± {fn_std:.1f}\n\nMiss Rate: {miss_rate:.2%}"],
         
        [f"False Positive (FP)\n{fp_mean:.1f} ± {fp_std:.1f}\n\nPrecision: {precision:.2%}", 
         "True Negative (TN)\n(No Aplicable)"]
    ]
    
    df_cm = pd.DataFrame(matrix_data,
                         index=["Real: Evento", "Real: No-Evento"],
                         columns=["Pred: Evento", "Pred: No-Evento"])
                         
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, 
                annot=annot_data, 
                fmt="", 
                cmap="Blues", 
                cbar=False,
                annot_kws={"size": 14})
                
    plt.title("Matriz de Confusión Basada en Eventos (Media ± DE)", fontsize=16)
    plt.ylabel("Realidad", fontsize=12)
    plt.xlabel("Predicción", fontsize=12)
    
    try:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Matriz de confusión (Media ± DE) guardada en: {save_path}")
    except Exception as e:
        print(f"Error al guardar la matriz de confusión: {e}")