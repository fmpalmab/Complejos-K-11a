# Detección y Localización de Complejos-K en Señales EEG

Este proyecto fue desarrollado para el ramo **Inteligencia Computacional (EL4106-1)**.

* **Desarrolladores:** Fernando Palma, Agustín Salgado
* **Profesor:** Pablo Estévez
* **Auxiliar:** Pablo Cornejo
* **Ayudante:** Rodrigo Catalán

---

Este proyecto implementa y evalúa tres arquitecturas de Redes Neuronales Convolucionales-Recurrentes (CRNN) para la detección y localización de Complejos-K en datos de señales de EEG.

El código está estructurado para permitir la fácil experimentación y evaluación de tres modelos distintos:
1.  **Detección (CNN):** Un modelo CRNN que clasifica una señal completa como "contiene Complejo-K" (1) o "no contiene" (0).
2.  **Detección (MLP):** Una variante del primer modelo que utiliza un clasificador MLP al final en lugar de una convolución 1x1.
3.  **Localización (CRNN):** Un modelo CRNN de secuencia-a-secuencia que predice, para cada punto de la señal (post-pooling), si es parte de un Complejo-K o no.

## 🚀 Instalación

Para configurar el entorno y ejecutar este proyecto, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/fmpalmab/Complejos-K-11a.git](https://github.com/fmpalmab/Complejos-K-11a.git)
    cd Complejos-K-11a
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    El archivo `requirements.txt` contiene todas las bibliotecas necesarias.
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Estructura del Proyecto

El código está modularizado en la carpeta `src/` para mayor claridad:

* `train.py`: El script principal para ejecutar los experimentos. Contiene los bucles de entrenamiento y evaluación, y la lógica para seleccionar qué modelo entrenar.
* `models.py`: Contiene las definiciones de las tres arquitecturas de PyTorch (`CNNDETECTAR`, `CNNDETECTAR_MLP`, `CRNN_DETECTAR_LOCALIZAR`).
* `datasets.py`: Define las clases `SignalDatasetDetectar` y `SignalDatasetLocalizar` de PyTorch para cargar y pre-procesar los datos para cada tarea.
* `utils.py`: Funciones de ayuda para graficar curvas de entrenamiento, matrices de confusión (con bootstrapping) y visualizar las predicciones de localización.
* `config.py`: Un archivo para centralizar hiperparámetros y constantes (actualmente vacío, pero listo para usarse).
* `notebooks/`: Contiene los notebooks de Jupyter (`k2.ipynb`, `ComplejosK.ipynb`) usados para la exploración inicial y el desarrollo del código.

## 📊 Datos

El modelo espera un archivo `ss2kc.parquet` ubicado según la ruta especificada en `train.py`. Este archivo debe contener al menos las columnas:
* `signal`: La señal de EEG (numpy array o lista).
* `labels`: La etiqueta de secuencia (numpy array o lista de 0s y 1s).

El script `train.py` genera automáticamente la columna `existeK` (para la tarea de detección) a partir de la columna `labels`.

## ▶️ Cómo Ejecutar los Experimentos

Puedes ejecutar los experimentos usando el script `train.py` desde la raíz del repositorio. Utiliza el argumento `--experimento` para seleccionar qué modelo entrenar.

```bash
# Ejecutar el Experimento 1 (Detección - CNN)
python train.py --experimento 1

# Ejecutar el Experimento 2 (Detección - MLP)
python train.py --experimento 2

# Ejecutar el Experimento 3 (Localización - CRNN)
python train.py --experimento 3

# Ejecutar TODOS los experimentos, uno tras otro (default)
python train.py --experimento 0
# O simplemente:
python train.py