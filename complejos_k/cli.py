"""
Interfaz de línea de comandos (CLI) unificada para Complejos-K-11a.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from complejos_k.pipeline import run_experiment, get_or_create_dataset
from complejos_k.synthetic import generate_synthetic_eeg_dataset
from complejos_k.config import SYNTHETIC_DATA_PATH


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="complejos_k",
        description="Detección y Localización de Complejos-K en EEG con Redes Neuronales CRNN",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true", help="Entrenar modelo seleccionado")
    group.add_argument("--synth", action="store_true", help="Generar dataset EEG sintético de Fase 2")
    group.add_argument("--eval", action="store_true", help="Evaluar modelo en el conjunto de prueba")

    parser.add_argument(
        "--model",
        type=str,
        default="crnn",
        choices=["cnn", "mlp", "crnn", "cwt", "zeta", "ensemble"],
        help="Arquitectura a entrenar (default: crnn)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño de batch (default: 16)")
    parser.add_argument("--samples", type=int, default=60, help="Cantidad de muestras sintéticas (default: 60)")

    args = parser.parse_args(argv)

    if args.synth:
        print(f"Generando dataset sintético con {args.samples} muestras...")
        generate_synthetic_eeg_dataset(n_samples=args.samples, output_path=SYNTHETIC_DATA_PATH)
        print("Dataset generado exitosamente.")
        return 0

    if args.eval or args.train:
        run_experiment(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_synthetic_samples=args.samples,
        )
        return 0

    # Por defecto: entrenar crnn brevemente
    run_experiment(
        model_name="crnn",
        epochs=3,
        batch_size=16,
        n_synthetic_samples=args.samples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
