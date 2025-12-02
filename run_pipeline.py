import subprocess
import sys
import time
import os

def run_script(script_name, args=[]):
    """
    Ejecuta un script de Python como un subproceso.
    """
    print(f"\n{'='*60}")
    print(f">>> EJECUTANDO: {script_name} {' '.join(args)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Construir el comando: [python, script_name, arg1, arg2...]
    command = [sys.executable, script_name] + args
    
    try:
        # check=True lanza una excepción si el script termina con error
        result = subprocess.run(command, check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n[ÉXITO] {script_name} finalizó correctamente en {elapsed_time:.2f} segundos.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] El script {script_name} falló con código de salida {e.returncode}.")
        return False
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error inesperado al intentar ejecutar {script_name}: {e}")
        return False

def main():
    print("--- INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO Y EVALUACIÓN ---\n")
    
    # 1. EJECUTAR ENTRENAMIENTO GENERAL (train_fake.py)
    # Usamos --experimento 0 para que corra TODOS los experimentos (1 al 6)
    # y genere los modelos necesarios (SEED, CWT, ZETA, etc.) para la evaluación.
    if not run_script('train_fake.py', args=['--experimento', '0']):
        print("Cancelando pipeline debido a error en entrenamiento.")
        sys.exit(1)

    # 2. EJECUTAR EVALUACIÓN ESTADÍSTICA (evaluate_all_fake.py)
    # Este script tomará los modelos generados en el paso 1, calculará métricas
    # con desviación estándar y generará matrices de confusión.
    if not run_script('evaluate_all_fake.py'):
        print("Cancelando pipeline debido a error en evaluación.")
        sys.exit(1)

    # 3. EJECUTAR EXPERIMENTO Z+C (seed_zc.py)
    # Este script es autónomo (entrena y evalúa el modelo de fusión ZETA+CWT).
    if not run_script('seed_zc.py'):
        print("Cancelando pipeline debido a error en SEED Z+C.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(">>> ¡PIPELINE FINALIZADO EXITOSAMENTE! <<<")
    print(f"{'='*60}")
    print("Revisa la carpeta 'resultados' y 'resultados_seed_zc' para ver los gráficos y reportes.")

if __name__ == "__main__":
    main()