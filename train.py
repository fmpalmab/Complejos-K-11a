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
import os # Importar os para crear carpetas

# Importar desde nuestros propios módulos .py
try:
    from config import *
    from models import CNNDETECTAR, CNNDETECTAR_MLP, CRNN_DETECTAR_LOCALIZAR
    from datasets import SignalDatasetDetectar, SignalDatasetLocalizar
    from utils import (
        plot_avg_training_history, plot_confusion_matrix_with_std, 
        visualizar_localizacion, get_test_metrics, plot_training_history,
        generate_metrics_report # <-- IMPORTAR NUEVA FUNCIÓN
    )
except ImportError:
    print("Error: No se pudieron importar los módulos locales (config, models, datasets, utils).")
    print("Asegúrate de ejecutar este script desde el directorio raíz del proyecto")
    sys.exit(1)


# --- 1. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN ---
# (Sin cambios aquí... las funciones train_epoch y evaluate siguen igual)

def train_epoch_detectar(model, dataloader, criterion, optimizer, device):
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

def train_epoch_localizar(model, dataloader, criterion, optimizer, device):
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

def evaluate_localizar(model, dataloader, criterion, device):
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
# (Sin cambios aquí)

def load_data(ruta):
    print(f"Cargando datos desde {ruta}...")
    try:
        df = pd.read_parquet(ruta)
    except Exception as e:
        print(f"Error al cargar el archivo parquet: {e}")
        return None
    print(f"Datos cargados. Forma: {df.shape}")
    if 'labels' in df.columns:
        df['existeK'] = df['labels'].apply(lambda x: 1 if 1 in x else 0)
        print("Columna 'existeK' creada.")
    else:
        print("Advertencia: La columna 'labels' no se encontró.")
        return None
    return df


# --- 3. FUNCIONES DE EXPERIMENTOS (ACTUALIZADAS) ---

def run_experiment_1_detection_cnn(df, output_dir):
    """Corre el experimento 1 y guarda los resultados en output_dir."""
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
    best_global_model_path = os.path.join(output_dir, 'best_model_cnn_detect.pth')
    global_best_val_loss = float('inf')
    global_best_epoch = 0 # <-- AÑADIDO

    for i in range(NUM_RUNS):
        print(f"\n--- Iniciando Corrida de Entrenamiento {i+1}/{NUM_RUNS} ---")
        model = CNNDETECTAR(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE)
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
            print(f"Corrida {i+1}, Época {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}")

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                patience_counter = 0
                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    global_best_epoch = epoch + 1 # <-- AÑADIDO
                    torch.save(model.state_dict(), best_global_model_path)
                    print(f"  -> Nuevo mejor modelo global guardado (Época {global_best_epoch})")
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"--- Early stopping en la época {epoch+1} ---")
                break
        all_histories.append(history)

    print("\n--- Entrenamiento Múltiple Finalizado ---")
    
    # --- Evaluación y Guardado de Gráficos ---
    best_model = CNNDETECTAR(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE)
    best_model.load_state_dict(torch.load(best_global_model_path))
    
    # 1. Guardar gráfico de entrenamiento
    plot_avg_training_history(
        all_histories, 
        title_suffix='(CNN Detección)', 
        best_epoch=global_best_epoch, # <-- AÑADIDO
        save_path=os.path.join(output_dir, 'training_curves_exp1_cnn.png') # <-- AÑADIDO
    )
    
    # 2. Guardar matriz de confusión
    plot_confusion_matrix_with_std(
        best_model, 
        test_loader, 
        DEVICE, 
        save_path=os.path.join(output_dir, 'confusion_matrix_exp1_cnn.png') # <-- AÑADIDO
    )
    
    # 3. Guardar tabla de métricas
    generate_metrics_report(
        best_model,
        test_loader,
        DEVICE,
        save_path=os.path.join(output_dir, 'metrics_report_exp1_cnn.csv') # <-- AÑADIDO
    )
    print("--- FIN EXPERIMENTO 1 ---")


def run_experiment_2_detection_mlp(df, output_dir):
    """Corre el experimento 2 y guarda los resultados en output_dir."""
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
    best_global_model_path = os.path.join(output_dir, 'best_model_mlp_detect.pth')
    global_best_val_loss = float('inf')
    global_best_epoch = 0 # <-- AÑADIDO
    
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
            print(f"Corrida {i+1}, Época {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}")

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                patience_counter = 0
                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    global_best_epoch = epoch + 1 # <-- AÑADIDO
                    torch.save(model.state_dict(), best_global_model_path)
                    print(f"  -> Nuevo mejor modelo global guardado (Época {global_best_epoch})")
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"--- Early stopping en la época {epoch+1} ---")
                break
        all_histories.append(history)

    print("\n--- Entrenamiento Múltiple Finalizado ---")
    
    # --- Evaluación y Guardado de Gráficos ---
    best_model = CNNDETECTAR_MLP(Nf=Nf_CNN, N1=N1_CNN).to(DEVICE)
    best_model.load_state_dict(torch.load(best_global_model_path))
    
    # 1. Guardar gráfico de entrenamiento
    plot_avg_training_history(
        all_histories, 
        title_suffix='(MLP Detección)', 
        best_epoch=global_best_epoch, # <-- AÑADIDO
        save_path=os.path.join(output_dir, 'training_curves_exp2_mlp.png') # <-- AÑADIDO
    )
    
    # 2. Guardar matriz de confusión
    plot_confusion_matrix_with_std(
        best_model, 
        test_loader, 
        DEVICE, 
        save_path=os.path.join(output_dir, 'confusion_matrix_exp2_mlp.png') # <-- AÑADIDO
    )
    
    # 3. Guardar tabla de métricas
    generate_metrics_report(
        best_model,
        test_loader,
        DEVICE,
        save_path=os.path.join(output_dir, 'metrics_report_exp2_mlp.csv') # <-- AÑADIDO
    )
    print("--- FIN EXPERIMENTO 2 ---")


def run_experiment_3_localization(df, output_dir):
    """Corre el experimento 3 y guarda los resultados en output_dir."""
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
    best_global_model_path = os.path.join(output_dir, 'best_model_localization.pth')
    global_best_val_loss = float('inf')
    global_best_epoch = 0 # <-- AÑADIDO
    
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
            print(f"Corrida {i+1}, Época {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc:.4f}")

            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                patience_counter = 0
                if val_loss < global_best_val_loss:
                    global_best_val_loss = val_loss
                    global_best_epoch = epoch + 1 # <-- AÑADIDO
                    torch.save(model.state_dict(), best_global_model_path)
                    print(f"  -> Nuevo mejor modelo global guardado (Época {global_best_epoch})")
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"--- Early stopping en la época {epoch+1} ---")
                break
        all_histories.append(history)

    print("\n--- Entrenamiento Múltiple Finalizado ---")
    
    # --- Evaluación y Guardado de Gráficos ---
    best_model = CRNN_DETECTAR_LOCALIZAR(num_classes=1, Nf=Nf_LOC, N1=N1_LOC).to(DEVICE)
    best_model.load_state_dict(torch.load(best_global_model_path))
    
    # 1. Guardar gráfico de entrenamiento
    plot_avg_training_history(
        all_histories, 
        title_suffix='(CRNN Localización)', 
        best_epoch=global_best_epoch, # <-- AÑADIDO
        save_path=os.path.join(output_dir, 'training_curves_exp3_loc.png') # <-- AÑADIDO
    )
    
    # 2. Guardar matriz de confusión
    plot_confusion_matrix_with_std(
        best_model, 
        test_loader, 
        DEVICE, 
        save_path=os.path.join(output_dir, 'confusion_matrix_exp3_loc.png') # <-- AÑADIDO
    )
    
    # 3. Guardar tabla de métricas
    generate_metrics_report(
        best_model,
        test_loader,
        DEVICE,
        save_path=os.path.join(output_dir, 'metrics_report_exp3_loc.csv') # <-- AÑADIDO
    )
    
    # 4. Guardar visualizaciones de localización
    visualizar_localizacion(
        best_model, 
        test_loader, 
        test_df, 
        DEVICE, 
        num_samples=3, 
        save_prefix=os.path.join(output_dir, 'localization_visualization_exp3') # <-- AÑADIDO
    )
    
    print("--- FIN EXPERIMENTO 3 ---")


# --- 4. BLOQUE DE EJECUCIÓN PRINCIPAL (ACTUALIZADO) ---

def main():
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
    # AÑADIDO: Argumento para el directorio de salida
    parser.add_argument(
        '--output_dir',
        type=str,
        default='resultados',
        help="Directorio donde se guardarán los gráficos, reportes y modelos."
    )
    args = parser.parse_args()

    # --- AÑADIDO: Crear directorio de salida si no existe ---
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio de salida creado en: {output_dir}")

    # Cargar datos
    df = load_data(RUTA_DATOS)
    if df is None:
        return

    # Ejecutar el experimento seleccionado
    if args.experimento == 1:
        run_experiment_1_detection_cnn(df, output_dir)
    elif args.experimento == 2:
        run_experiment_2_detection_mlp(df, output_dir)
    elif args.experimento == 3:
        run_experiment_3_localization(df, output_dir)
    elif args.experimento == 0:
        print("Ejecutando TODOS los experimentos...")
        print(f"Usando dispositivo: {DEVICE}")
        run_experiment_1_detection_cnn(df, output_dir)
        run_experiment_2_detection_mlp(df, output_dir)
        run_experiment_3_localization(df, output_dir)
        print(f"\n--- TODOS LOS EXPERIMENTOS HAN FINALIZADO ---")
        print(f"Resultados guardados en la carpeta: {output_dir}")

if __name__ == "__main__":
    main()