# src/train.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import sys

# Importar desde nuestros propios módulos .py
# Asume que 'src' está en el PYTHONPATH o que se ejecuta desde el directorio raíz
try:
    from config import *
    from models import CNNDETECTAR, CNNDETECTAR_MLP, CRNN_DETECTAR_LOCALIZAR
    from datasets import SignalDatasetDetectar, SignalDatasetLocalizar
    from utils import (
        plot_avg_training_history, plot_confusion_matrix_with_std, 
        visualizar_localizacion, get_test_metrics, plot_training_history
    )
except ImportError:
    print("Error: No se pudieron importar los módulos locales (config, models, datasets, utils).")
    print("Asegúrate de ejecutar este script desde el directorio raíz del proyecto")
    sys.exit(1)


# --- 1. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN ---

# --- Para Detección (etiqueta global) ---

def train_epoch_detectar(model, dataloader, criterion, optimizer, device):
    """Bucle de entrenamiento de una época para la Detección."""
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
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate_detectar(model, dataloader, criterion, device):
    """Bucle de evaluación de una época para la Detección."""
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
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


# --- Para Localización (etiqueta por punto) ---

def train_epoch_localizar(model, dataloader, criterion, optimizer, device):
    """Bucle de entrenamiento de una época para la Localización."""
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0 # Contará el total de puntos (batch_size * 500)

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device) # labels: (B, 1, 500)
        optimizer.zero_grad()
        outputs = model(inputs) # outputs: (B, 1, 500)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.numel() # Total de puntos (B * 1 * 500)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc


def evaluate_localizar(model, dataloader, criterion, device):
    """Bucle de evaluación de una época para la Localización."""
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


# --- 2. FUNCIÓN DE CARGA DE DATOS ---

def load_data(ruta):
    """Carga y pre-procesa el archivo parquet."""
    print(f"Cargando datos desde {ruta}...")
    try:
        df = pd.read_parquet(ruta)
    except Exception as e:
        print(f"Error al cargar el archivo parquet: {e}")
        print("Asegúrate de que el archivo 'ss2kc.parquet' esté en la ruta correcta.")
        return None
        
    print(f"Datos cargados. Forma: {df.shape}")
    
    # Columna 'existeK' para detección y estratificación
    if 'labels' in df.columns:
        df['existeK'] = df['labels'].apply(lambda x: 1 if 1 in x else 0)
        print("Columna 'existeK' creada.")
        print(df['existeK'].value_counts())
    else:
        print("Advertencia: La columna 'labels' no se encontró. No se pudo crear 'existeK'.")
        return None
        
    return df


# --- 3. FUNCIONES DE EXPERIMENTOS ---

def run_experiment_1_detection_cnn(df):
    """
    Corre el experimento de DETECCIÓN con el modelo CNNDETECTAR.
    Incluye 5 corridas y gráficos de media/DE.
    """
    print("\n--- INICIANDO EXPERIMENTO 1: DETECCIÓN (CNN) ---")
    
    # --- Preparación de Datos ---
    df_detectar = df[['signal', 'existeK']]
    train_df, temp_df = train_test_split(
        df_detectar, test_size=0.2, random_state=42, stratify=df_detectar['existeK']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK']
    )
    
    train_dataset = SignalDatasetDetectar(train_df)
    val_dataset = SignalDatasetDetectar(val_df)
    test_dataset = SignalDatasetDetectar(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Entrenamiento Múltiple ---
    all_histories = []
    best_global_model_path = 'best_model_cnn_detect.pth'
    global_best_val_loss = float('inf')
    
    for i in range(NUM_RUNS):
        print(f"\n--- Iniciando Corrida de Entrenamiento {i+1}/{NUM_RUNS} ---")
        model = CNNDETECTAR(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE) # Usando config
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        patience_counter = 0
        current_best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch_detectar(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = evaluate_detectar(model, val_loader, criterion, DEVICE)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Corrida {i+1}, Época {epoch+1}/{EPOCHS} | "
                  f"Loss ent: {train_loss:.4f} | Acc ent: {train_acc:.4f} | "
                  f"Loss val: {val_loss:.4f} | Acc val: {val_acc:.4f}")

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                patience_counter = 0
                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    torch.save(model.state_dict(), best_global_model_path)
                    print(f"  -> Nuevo mejor modelo global guardado en {best_global_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"--- Early stopping en la época {epoch+1} ---")
                break
        
        all_histories.append(history)

    print("\n--- Entrenamiento Múltiple Finalizado ---")
    
    # --- Evaluación y Gráficos ---
    plot_avg_training_history(all_histories, title_suffix='(CNN Detección)')
    
    print(f"Cargando el mejor modelo desde {best_global_model_path} para evaluación final...")
    best_model = CNNDETECTAR(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE)
    best_model.load_state_dict(torch.load(best_global_model_path))
    
    plot_confusion_matrix_with_std(best_model, test_loader, DEVICE)
    
    print("--- FIN EXPERIMENTO 1 ---")


def run_experiment_2_detection_mlp(df):
    """
    Corre el experimento de DETECCIÓN con el modelo CNNDETECTAR_MLP.
    Incluye 5 corridas y gráficos de media/DE.
    """
    print("\n--- INICIANDO EXPERIMENTO 2: DETECCIÓN (MLP) ---")
    
    # --- Preparación de Datos ---
    df_detectar = df[['signal', 'existeK']]
    train_df, temp_df = train_test_split(
        df_detectar, test_size=0.2, random_state=42, stratify=df_detectar['existeK']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK']
    )
    
    train_dataset = SignalDatasetDetectar(train_df)
    val_dataset = SignalDatasetDetectar(val_df)
    test_dataset = SignalDatasetDetectar(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- Entrenamiento Múltiple ---
    all_histories = []
    best_global_model_path = 'best_model_mlp_detect.pth'
    global_best_val_loss = float('inf')
    
    for i in range(NUM_RUNS):
        print(f"\n--- Iniciando Corrida de Entrenamiento {i+1}/{NUM_RUNS} ---")
        model = CNNDETECTAR_MLP(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        patience_counter = 0
        current_best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch_detectar(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = evaluate_detectar(model, val_loader, criterion, DEVICE)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Corrida {i+1}, Época {epoch+1}/{EPOCHS} | "
                  f"Loss ent: {train_loss:.4f} | Acc ent: {train_acc:.4f} | "
                  f"Loss val: {val_loss:.4f} | Acc val: {val_acc:.4f}")

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                patience_counter = 0
                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    torch.save(model.state_dict(), best_global_model_path)
                    print(f"  -> Nuevo mejor modelo global guardado en {best_global_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"--- Early stopping en la época {epoch+1} ---")
                break
        
        all_histories.append(history)

    print("\n--- Entrenamiento Múltiple Finalizado ---")
    
    # --- Evaluación y Gráficos ---
    plot_avg_training_history(all_histories, title_suffix='(MLP Detección)')
    
    print(f"Cargando el mejor modelo desde {best_global_model_path} para evaluación final...")
    best_model = CNNDETECTAR_MLP(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE)
    best_model.load_state_dict(torch.load(best_global_model_path))
    
    plot_confusion_matrix_with_std(best_model, test_loader, DEVICE)
    
    print("--- FIN EXPERIMENTO 2 ---")


def run_experiment_3_localization(df):
    """
    Corre el experimento de LOCALIZACIÓN con el modelo CRNN_DETECTAR_LOCALIZAR.
    Incluye 5 corridas y gráficos de media/DE.
    """
    print("\n--- INICIANDO EXPERIMENTO 3: LOCALIZACIÓN (CRNN) ---")
    
    # --- Preparación de Datos ---
    df_localizar = df[['signal', 'labels', 'existeK']]
    train_df, temp_df = train_test_split(
        df_localizar, test_size=0.2, random_state=42, stratify=df_localizar['existeK']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['existeK']
    )
    
    train_dataset = SignalDatasetLocalizar(train_df)
    val_dataset = SignalDatasetLocalizar(val_df)
    test_dataset = SignalDatasetLocalizar(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Cálculo de pos_weight ---
    print("Calculando peso para clases desbalanceadas (pos_weight)...")
    all_labels_500 = torch.cat([label for _, label in train_loader], dim=0)
    neg_count = (all_labels_500 == 0).sum().item()
    pos_count = (all_labels_500 == 1).sum().item()
    pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
    print(f"Peso positivo (pos_weight) calculado: {pos_weight:.2f}")
    pos_weight_tensor = torch.tensor([pos_weight], device=DEVICE)

    # --- Entrenamiento Múltiple ---
    all_histories = []
    best_global_model_path = 'best_model_localization.pth'
    global_best_val_loss = float('inf')
    
    for i in range(NUM_RUNS):
        print(f"\n--- Iniciando Corrida de Entrenamiento {i+1}/{NUM_RUNS} ---")
        model = CRNN_DETECTAR_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        patience_counter = 0
        current_best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch_localizar(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc = evaluate_localizar(model, val_loader, criterion, DEVICE)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Corrida {i+1}, Época {epoch+1}/{EPOCHS} | "
                  f"Loss ent: {train_loss:.4f} | Acc ent: {train_acc:.4f} | "
                  f"Loss val: {val_loss:.4f} | Acc val: {val_acc:.4f}")

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                patience_counter = 0
                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    torch.save(model.state_dict(), best_global_model_path)
                    print(f"  -> Nuevo mejor modelo global guardado en {best_global_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"--- Early stopping en la época {epoch+1} ---")
                break
        
        all_histories.append(history)

    print("\n--- Entrenamiento Múltiple Finalizado ---")
    
    # --- Evaluación y Gráficos ---
    plot_avg_training_history(all_histories, title_suffix='(CRNN Localización)')
    
    print(f"Cargando el mejor modelo desde {best_global_model_path} para evaluación final...")
    best_model = CRNN_DETECTAR_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC).to(DEVICE)
    best_model.load_state_dict(torch.load(best_global_model_path))
    
    # Métricas de prueba
    get_test_metrics(best_model, test_loader, DEVICE, task_type='localizar (por punto)')
    
    # Matriz de confusión con bootstrapping
    plot_confusion_matrix_with_std(best_model, test_loader, DEVICE)
    
    # Visualización de predicciones
    # Pasamos test_df (el DataFrame) para obtener las etiquetas originales de 4000 puntos
    visualizar_localizacion(best_model, test_loader, test_df, DEVICE, num_samples=3)
    
    print("--- FIN EXPERIMENTO 3 ---")


# --- 4. BLOQUE DE EJECUCIÓN PRINCIPAL ---

def main():
    # Parser para elegir el experimento desde la línea de comandos
    parser = argparse.ArgumentParser(
        description="Entrenar y evaluar modelos de detección/localización de Complejos-K."
    )
    parser.add_argument(
        '--experimento',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Número del experimento a ejecutar (1: CNN, 2: MLP, 3: Localización, 0: Todos). Default: 0"
    )
    args = parser.parse_args()

    # Cargar datos desde la ruta en config.py
    df = load_data(RUTA_DATOS)
    if df is None:
        return

    # Ejecutar el experimento seleccionado
    if args.experimento == 1:
        run_experiment_1_detection_cnn(df)
    elif args.experimento == 2:
        run_experiment_2_detection_mlp(df)
    elif args.experimento == 3:
        run_experiment_3_localization(df)
    elif args.experimento == 0:
        print("Ejecutando TODOS los experimentos...")
        run_experiment_1_detection_cnn(df)
        run_experiment_2_detection_mlp(df)
        run_experiment_3_localization(df)
        print("\n--- TODOS LOS EXPERIMENTOS HAN FINALIZADO ---")

if __name__ == "__main__":
    main()