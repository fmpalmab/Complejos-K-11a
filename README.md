# 🧠 Detección y Localización de Complejos-K en Señales EEG con Redes CRNN

[![CI](https://github.com/fmpalmab/Complejos-K-11a/actions/workflows/ci.yml/badge.svg)](https://github.com/fmpalmab/Complejos-K-11a/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Proyecto de investigación e ingeniería biomédica enfocado en la **detección binaria y localización temporal precisa de Complejos-K** en registros electroencefalográficos (EEG) de sueño en Fase 2 (NREM 2).

Desarrollado originalmente para el curso **EL4106-1 Inteligencia Computacional** (Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile).

---

## 🔬 Fundamento Fisiológico

Un **Complejo-K** es un biomarcador electrofisiológico distintivo del sueño NREM Fase 2 caracterizado por:
1. Una **deflexión negativa inicial aguda** (hiperpolarización cortical) con amplitud de pico $\le -100\ \mu\text{V}$ y duración $\approx 150 - 250\ \text{ms}$.
2. Una **onda positiva lenta posterior** con amplitud $\ge +70\ \mu\text{V}$ y duración $\approx 350 - 550\ \text{ms}$.
3. Duración total del evento $\ge 500\ \text{ms}$ a $1.2\ \text{s}$, frecuentemente asociado a un huso de sueño (*sleep spindle*, $12 - 14\ \text{Hz}$) posterior.

---

## 🏗️ Arquitectura del Repositorio

El proyecto cuenta con un paquete modular `complejos_k` bajo estándares modernos de empaquetado (PEP 517/621):

```
Complejos-K-11a/
├── .github/
│   └── workflows/
│       └── ci.yml               # Pipeline de CI (Pytest en 3.10, 3.11, 3.12)
├── complejos_k/                 # Paquete principal
│   ├── __init__.py              # Exportaciones públicas de la API
│   ├── __main__.py              # Punto de entrada ejecutable (python -m complejos_k)
│   ├── cli.py                   # Interfaz de línea de comandos unificada
│   ├── config.py                # Dispositivo (CPU/CUDA), rutas e hiperparámetros
│   ├── datasets.py              # Datasets PyTorch (Detección, Localización 1ch/2ch/3ch)
│   ├── metrics.py               # Métricas punto a punto y basadas en solapamiento de eventos (IoU)
│   ├── models.py                # Modelos PyTorch: CNN, CNN-MLP, CRNN, CWT-CRNN, ZETA-CRNN, Ensemble
│   ├── pipeline.py              # Orquestador integral de entrenamiento y evaluación
│   ├── synthetic.py             # Generador fisiológico de EEG sintético NREM2 con Complejos-K
│   ├── trainer.py               # Ciclos de entrenamiento optimizados con Early Stopping
│   └── wavelets.py              # Transformada Wavelet Continua (Ricker/Mexican Hat)
├── notebooks/                   # Notebooks originales de experimentación
├── tests/                       # Suite de pruebas automatizadas (Pytest)
│   ├── test_datasets.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_synthetic.py
│   ├── test_training_step.py
│   └── test_wavelets.py
├── pyproject.toml               # Configuración de empaquetado moderno
└── requirements.txt             # Dependencias del proyecto
```

---

## 🧠 Modelos y Arquitecturas Evaluadas

El sistema implementa 6 arquitecturas neuronales basadas en extracción convolucional multiescala y modelado de dependencias temporales con BiLSTMs:

| # | Modelo | Tipo | Entrada | Objetivo |
|:---:|---|---|:---:|---|
| **1** | `CNNDETECTAR` | CNN + BiLSTM + GAP | 1 Canal (Señal) | Clasificación binaria global (¿Existe K?) |
| **2** | `CNNDETECTAR_MLP` | CNN + BiLSTM + MLP | 1 Canal (Señal) | Clasificación binaria global |
| **3** | `CRNN_DETECTAR_LOCALIZAR` | CRNN Secuencia a Secuencia | 1 Canal (Señal) | Máscara temporal ($500$ puntos) |
| **4** | `CWT_CRNN_LOCALIZAR` | CRNN + Transformada Wavelet | 2 Canales (Señal + CWT) | Localización asistida por frecuencia |
| **5** | `ZETA_CRNN_LOCALIZAR` | CRNN + Z-Score | 2 Canales (Señal + Z-Score) | Localización normalizada por ventana |
| **6** | `ENSEMBLE_LOCALIZAR` | Ensamble Multicanal | 3 Canales (Señal + CWT + ZETA) | Fusión completa de características |

---

## ⚡ Instalación

```bash
git clone https://github.com/fmpalmab/Complejos-K-11a.git
cd Complejos-K-11a

# Instalación base
pip install -e .

# Instalación con dependencias de desarrollo y test
pip install -e ".[dev]"
```

---

## 🚀 Uso del CLI Unificado

```bash
# 1. Generar dataset sintético fisiológico (si no se dispone del archivo real ss2kc.parquet)
python -m complejos_k --synth --samples 100

# 2. Entrenar el modelo de Localización CRNN estándar
python -m complejos_k --train --model crnn --epochs 10 --batch-size 32

# 3. Entrenar y evaluar el modelo Multicanal CWT (Wavelet)
python -m complejos_k --train --model cwt --epochs 10

# 4. Entrenar el Ensamble Multicanal (Señal + CWT + ZETA)
python -m complejos_k --train --model ensemble --epochs 10

# 5. Evaluar cualquier modelo en el conjunto de prueba
python -m complejos_k --eval --model crnn
```

---

## 📈 Evaluación Fisiológica Basada en Eventos

Además de las métricas clásicas punto a punto (Accuracy, Precision, Recall, F1), el módulo `complejos_k.metrics` evalúa la detección mediante **solapamiento de eventos (IoU)**:

$$\operatorname{IoU}(E_{\text{pred}}, E_{\text{true}}) = \frac{|E_{\text{pred}} \cap E_{\text{true}}|}{|E_{\text{pred}} \cup E_{\text{true}}|}$$

Un evento detectado se califica como Verdadero Positivo ($TP$) si su solapamiento temporal supera el umbral fisiológico ($\text{IoU} \ge 0.2$).

---

## 🧪 Pruebas Automatizadas

```bash
pytest -v
```

La suite cubre formalmente la generación de wavelets, síntesis fisiológica de EEG, capas de Max-Pooling, forward passes de las 6 arquitecturas, métricas IoU y bucles de optimización.

---

## 📜 Licencia

Distribuido bajo la Licencia **MIT**. Desarrollado originalmente en el Departamento de Ingeniería Eléctrica (DIE), FCFM, Universidad de Chile.