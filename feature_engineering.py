# feature_engineering.py

import pandas as pd
import numpy as np
import sys
# --- MODIFICACIÓN ---
# Volvemos a scipy.signal, asumiendo que la reinstalación lo ha arreglado
from wavelets import cwt, ricker
from tqdm import tqdm

# Importar la ruta de datos desde tu archivo de configuración
try:
    from config import RUTA_DATOS
except ImportError:
    print("Error: No se pudo encontrar 'RUTA_DATOS' en config.py")
    sys.exit(1)

def calculate_zscore(signal_array):
    """
    Calcula el Z-score para cada muestra dentro de una única señal.
    """
    mean = np.mean(signal_array)
    std = np.std(signal_array)
    
    # Evitar división por cero si la señal es plana
    if std == 0:
        return np.zeros_like(signal_array)
        
    return (signal_array - mean) / std

# --- MODIFICACIÓN ---
def calculate_cwt_feature(signal_array):
    """
    Calcula la CWT y la colapsa a un vector 1D (4000,)
    tomando la media de la magnitud a través de las escalas.
    """
    # Definimos un rango de escalas (anchos) para la wavelet.
    widths = np.arange(1, 31)
    
    # Usamos la wavelet 'ricker' (sombrero mexicano) de scipy.signal
    cwt_matrix = cwt(signal_array, ricker, widths)
    
    # La cwt_matrix tiene forma (30, 4000) -> (escalas, muestras)
    # Colapsamos las escalas tomando la media de la magnitud (energía)
    cwt_feature = np.mean(np.abs(cwt_matrix), axis=0)
    
    # Aseguramos que tenga la misma longitud que la señal original
    if len(cwt_feature) != len(signal_array):
        return np.zeros_like(signal_array)
        
    return cwt_feature

def main():
    print(f"Cargando datos originales desde: {RUTA_DATOS}")
    try:
        df = pd.read_parquet(RUTA_DATOS)
    except Exception as e:
        print(f"Error al cargar {RUTA_DATOS}: {e}")
        return

    if 'signal' not in df.columns:
        print("Error: El dataframe no tiene la columna 'signal'.")
        return

    # Inicializar tqdm para pandas
    tqdm.pandas(desc="Progreso")

    print("Calculando columna 'zeta' (Z-score)...")
    df['zeta'] = df['signal'].progress_apply(calculate_zscore)
    print("Columna 'zeta' creada.")

    print("Calculando columna 'cwt' (Wavelet con SciPy)... (Esto puede tardar)")
    df['cwt'] = df['signal'].progress_apply(calculate_cwt_feature)
    print("Columna 'cwt' creada.")

    # Definir la nueva ruta de salida
    if '.parquet' in RUTA_DATOS:
        output_path = RUTA_DATOS.replace('.parquet', '_features.parquet')
    else:
        output_path = RUTA_DATOS + '_features.parquet'

    print(f"Guardando dataframe con nuevas características en: {output_path}")
    try:
        df.to_parquet(output_path, index=False)
        print("¡Proceso completado exitosamente!")
        print("\n--- Próximo paso ---")
        print(f"Actualiza 'RUTA_DATOS' en tu archivo 'config.py' a:")
        print(f"RUTA_DATOS = '{output_path}'")
        
    except Exception as e:
        print(f"Error al guardar el nuevo archivo parquet: {e}")

if __name__ == "__main__":
    main()