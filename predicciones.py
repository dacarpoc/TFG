# -*- coding: utf-8 -*-
"""
Script de predicciones usando el modelo entrenado.
"""

import os
import sys
import argparse
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Importar configuración y utilidades compartidas
from config import (
    DIRECTORIO_SALIDA, UMBRAL_CICLOS_DEFAULT,
    PESO_FALSOS_POSITIVOS, configurar_logging, obtener_ruta
)
from utils import crear_derivadas, analizar_membranas, calcular_score_balanceado

warnings.filterwarnings("ignore")

# Configurar logger
logger = configurar_logging(__name__)


# ================================================================================
# FUNCIONES DE SALIDA
# ================================================================================

def guardar_predicciones(df_base: pd.DataFrame, y_real: np.ndarray, 
                         probas: np.ndarray, threshold: float, 
                         path_salida: str) -> None:
    """
    Genera un CSV con las predicciones y métricas.
    
    Args:
        df_base: DataFrame original con los datos
        y_real: Array con los valores reales binarios
        probas: Array con las probabilidades predichas
        threshold: Umbral de decisión
        path_salida: Ruta del archivo de salida
    """
    logger.info(f"Generando salida: {path_salida}")

    predicciones = (probas >= threshold).astype(int)

    df_export = df_base.copy()
    df_export.reset_index(drop=True, inplace=True)

    # Convertir a arrays si es necesario
    y_real_val = y_real.values if hasattr(y_real, 'values') else y_real
    probas_val = probas.values if hasattr(probas, 'values') else probas

    # Añadir columnas de predicción
    df_export['Target_Real'] = y_real_val
    df_export['Probabilidad'] = probas_val
    df_export['Prediccion'] = predicciones
    df_export['Acierto'] = (df_export['Target_Real'] == df_export['Prediccion']).astype(int)

    # Clasificar tipo de caso
    conditions = [
        (df_export['Target_Real'] == 1) & (df_export['Prediccion'] == 1),
        (df_export['Target_Real'] == 0) & (df_export['Prediccion'] == 1),
        (df_export['Target_Real'] == 1) & (df_export['Prediccion'] == 0),
        (df_export['Target_Real'] == 0) & (df_export['Prediccion'] == 0)
    ]
    choices = ['TP (Acierto)', 'FP (Falsa Alarma)', 'FN (Fuga Omitida)', 'TN (Normal)']
    df_export['Tipo_Caso'] = np.select(conditions, choices, default='Error')

    # Reordenar columnas
    cols_main = ['Ciclos', 'Target_Real', 'Prediccion', 'Probabilidad', 'Acierto', 'Tipo_Caso']
    cols_main = [c for c in cols_main if c in df_export.columns]
    cols_rest = [c for c in df_export.columns if c not in cols_main]
    df_export = df_export[cols_main + cols_rest]

    # Guardar CSV
    df_export.to_csv(path_salida, index=False, sep=';', decimal=',') 
    logger.info("Archivo guardado correctamente.")


# ================================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================================

def main(csv_input: str, model_file: str, threshold: float):
    """
    Función principal que ejecuta las predicciones.
    
    Args:
        csv_input: Ruta al archivo CSV de entrada
        model_file: Ruta al archivo PKL del modelo
        threshold: Umbral de decisión (obligatorio)
    """
    
    # Resolver rutas (busca en: ruta absoluta → directorio actual → output/)
    archivo_entrada = obtener_ruta(csv_input)
    archivo_modelo = obtener_ruta(model_file)
    
    logger.info("=" * 60)
    logger.info("GENERACIÓN DE PREDICCIONES")
    logger.info("=" * 60)
    
    # --------------------------------------------------------------------------
    # Carga del modelo
    # --------------------------------------------------------------------------
    
    logger.info(f"Cargando modelo desde: {archivo_modelo}")
    
    if not os.path.exists(archivo_modelo):
        raise FileNotFoundError(f"No se encuentra el archivo {archivo_modelo}")
    
    config = joblib.load(archivo_modelo)
    logger.info("Modelo cargado correctamente.")
    
    # Extraer configuración del modelo
    UMBRAL_CICLOS = UMBRAL_CICLOS_DEFAULT
    THRESHOLD = threshold
    SCALER = config.get('scaler', None)
    COLS_FEATURE = config.get('feature_cols', config.get('feature_names', None))
    MODELO = config.get('modelo', None)
    NOMBRE_MODELO = config.get('nombre_modelo', 'modelo')

    logger.info(f"Configuración del modelo — Nombre: {NOMBRE_MODELO} | Threshold: {THRESHOLD} | Umbral ciclos: {UMBRAL_CICLOS}")

    # --------------------------------------------------------------------------
    # Carga de datos
    # --------------------------------------------------------------------------
    
    logger.info(f"Cargando dataset: {archivo_entrada}")
    
    if not os.path.exists(archivo_entrada):
        raise FileNotFoundError(f"No se encuentra el archivo {archivo_entrada}")
    
    try:
        df_input = pd.read_csv(archivo_entrada) 
    except Exception as e:
        logger.warning(f"Intentando con separador punto y coma... ({e})")
        df_input = pd.read_csv(archivo_entrada, sep=';')
    
    logger.info(f"Datos cargados: {len(df_input)} filas")
    
    # Eliminar índice si existe
    if 'Indice' in df_input.columns:
        df_input.drop(columns=['Indice'], inplace=True)
    
    # --------------------------------------------------------------------------
    # Preparación de datos
    # --------------------------------------------------------------------------
    
    logger.info("Preparando datos (Feature Engineering)...")
    df_processed = crear_derivadas(df_input)
    
    # Selección de columnas
    if COLS_FEATURE:
        missing_cols = [c for c in COLS_FEATURE if c not in df_processed.columns]
        if missing_cols:
            logger.warning(f"Faltan columnas: {missing_cols}")
            for c in missing_cols:
                df_processed[c] = 0
        X_new = df_processed[COLS_FEATURE]
    else:
        X_new = df_processed
    
    # Escalado
    if SCALER:
        X_new_scaled = pd.DataFrame(SCALER.transform(X_new), columns=COLS_FEATURE)
    else:
        X_new_scaled = X_new
    
    logger.info("Datos preparados.")
    
    # --------------------------------------------------------------------------
    # Predicción
    # --------------------------------------------------------------------------
    
    logger.info("Calculando predicciones...")
    proba_final = MODELO.predict_proba(X_new_scaled)[:, 1]
    
    # --------------------------------------------------------------------------
    # Evaluación (si hay columna Ciclos)
    # --------------------------------------------------------------------------
    
    if 'Ciclos' in df_input.columns:
        y_ciclos = df_input['Ciclos'].values
        y_real_bin = (y_ciclos < UMBRAL_CICLOS).astype(int)
        pred_bin = (proba_final >= THRESHOLD).astype(int)
        
        # Análisis de membranas
        det, total_membranas = analizar_membranas(y_ciclos, pred_bin, UMBRAL_CICLOS)
        
        # Calcular métricas
        try:
            tn, fp, fn, tp = confusion_matrix(y_real_bin, pred_bin).ravel()
        except ValueError:
            fp = 0 
        
        pct_membranas = det / total_membranas if total_membranas > 0 else 0
        score = calcular_score_balanceado(pct_membranas, fp, total_membranas, PESO_FALSOS_POSITIVOS)
        
        logger.info("=" * 60)
        logger.info("RESULTADOS")
        logger.info("=" * 60)
        logger.info(f"Membranas con rotura (Total): {total_membranas}")
        logger.info(f"Membranas Detectadas:         {det}")
        logger.info(f"Falsos Positivos (Filas):     {fp}")
        logger.info(f"Score Balanceado:             {score:.2f}")
    else:
        logger.info("No se encontró columna 'Ciclos'. Generando predicciones sin métricas.")
        y_real_bin = np.zeros(len(proba_final))
    
    # --------------------------------------------------------------------------
    # Guardado
    # --------------------------------------------------------------------------
    
    logger.info("=" * 60)
    logger.info("Guardando resultados...")
    logger.info("=" * 60)
    
    nombre_dataset = os.path.splitext(os.path.basename(csv_input))[0]
    threshold_str = str(THRESHOLD).replace('.', '_')
    csv_salida = os.path.join(DIRECTORIO_SALIDA, f"predicciones_{nombre_dataset}_{NOMBRE_MODELO}_{threshold_str}.csv")
    guardar_predicciones(
        df_base=df_input,
        y_real=y_real_bin,
        probas=proba_final,
        threshold=THRESHOLD,
        path_salida=csv_salida
    )
    
    logger.info("=" * 60)
    logger.info("PREDICCIONES COMPLETADAS")
    logger.info("=" * 60)


if __name__ == "__main__":
    class ArgumentParserES(argparse.ArgumentParser):
        """ArgumentParser con mensajes en castellano."""
        def error(self, message):
            self.print_usage(sys.stderr)
            sys.stderr.write(f'Error: Se requiere especificar el archivo CSV a predecir.\n')
            sys.exit(2)
    
    parser = ArgumentParserES(
        description='Genera predicciones usando el modelo entrenado.',
        usage='python predicciones.py -d <archivo.csv> -m <modelo.pkl> -t <threshold>'
    )
    parser.add_argument(
        '-d', '--dataset',
        required=True,
        help='Archivo CSV a predecir (ruta relativa o absoluta)'
    )
    parser.add_argument(
        '-m', '--modelo',
        required=True,
        help='Archivo PKL del modelo a usar (ruta relativa o absoluta)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        required=True,
        help='Threshold de decisión (0-1)'
    )
    
    args = parser.parse_args()
    main(args.dataset, args.modelo, args.threshold)
