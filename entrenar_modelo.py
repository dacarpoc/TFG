# -*- coding: utf-8 -*-
"""
Entrena un modelo específico usando los hiperparámetros ya optimizados
guardados en output/todas_las_configs.joblib, sin necesidad de reejecutar Optuna.

Uso:
    python entrenar_modelo.py -n RandomForest_mcc
    python entrenar_modelo.py --listar
"""

import os
import sys
import ast
import argparse
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

from config import (
    CSV_ENTRENAR, CSV_VALIDAR, DIRECTORIO_SALIDA,
    RANDOM_STATE, UMBRAL_CICLOS_DEFAULT, CONFIGURACION_MEJORES_MODELOS,
    configurar_logging
)
from utils import crear_derivadas

warnings.filterwarnings('ignore')

logger = configurar_logging(__name__)


# ================================================================================
# RECREAR MODELO DESDE NOMBRE + PARAMS
# ================================================================================

def crear_modelo(nombre_base: str, params: dict):
    """
    Instancia el modelo con los parámetros dados.

    Args:
        nombre_base: 'RandomForest', 'XGBoost', etc.
        params: Diccionario de hiperparámetros

    Returns:
        Instancia del modelo configurado
    """
    if nombre_base == 'XGBoost':
        return XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric='aucpr', n_jobs=-1)
    elif nombre_base == 'LightGBM':
        return LGBMClassifier(**params, class_weight='balanced', random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    elif nombre_base == 'RandomForest':
        return RandomForestClassifier(**params, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
    elif nombre_base == 'ExtraTrees':
        return ExtraTreesClassifier(**params, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
    elif nombre_base == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(**params, class_weight='balanced', random_state=RANDOM_STATE)
    elif nombre_base == 'CatBoost':
        return CatBoostClassifier(**params, allow_writing_files=False, auto_class_weights='Balanced',
                                  random_state=RANDOM_STATE, verbose=0, thread_count=-1)
    else:
        raise ValueError(f"Modelo desconocido: '{nombre_base}'")


# ================================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================================

def main(nombre_modelo: str, listar: bool = False):
    """
    Carga hiperparámetros, prepara datos y entrena el modelo seleccionado.

    Args:
        nombre_modelo: Nombre del modelo a entrenar (ej: 'RandomForest_mcc')
        listar: Si True, muestra los modelos disponibles y sale
    """
    # --------------------------------------------------------------------------
    # Cargar configuraciones guardadas
    # --------------------------------------------------------------------------
    ruta_configs = CONFIGURACION_MEJORES_MODELOS
    if not os.path.exists(ruta_configs):
        logger.error(f"No se encuentra el archivo de configuraciones: {ruta_configs}")
        sys.exit(1)

    logger.info(f"Cargando configuraciones desde: {ruta_configs}")
    configs = joblib.load(ruta_configs)  # lista de dicts
    logger.info(f"Se encontraron {len(configs)} configuraciones guardadas.")

    # Índice por nombre de modelo
    configs_por_nombre = {c['modelo_nombre']: c for c in configs}

    if listar:
        logger.info("Modelos disponibles en el archivo de configuraciones:")
        for nombre, cfg in sorted(configs_por_nombre.items(), key=lambda x: x[1].get('score_balanceado', 0), reverse=True):
            logger.info(f"  {nombre:35} | Score: {cfg.get('score_balanceado', '?'):.2f}")
        return

    if nombre_modelo not in configs_por_nombre:
        logger.error(f"El modelo '{nombre_modelo}' no existe en las configuraciones.")
        logger.info(f"Disponibles: {sorted(configs_por_nombre.keys())}")
        sys.exit(1)

    config_elegida = configs_por_nombre[nombre_modelo]
    nombre_base = nombre_modelo.split('_')[0]

    # Los hiperparámetros se guardaron como str(), recuperar con ast.literal_eval
    hiperparametros_str = config_elegida.get('hiperparametros', '{}')
    params = ast.literal_eval(hiperparametros_str)

    logger.info("=" * 60)
    logger.info(f"Modelo seleccionado : {nombre_modelo}")
    logger.info(f"Algoritmo base      : {nombre_base}")
    logger.info(f"Score balanceado    : {config_elegida.get('score_balanceado', '?'):.2f}")
    logger.info(f"Hiperparámetros     : {params}")
    logger.info("=" * 60)

    # --------------------------------------------------------------------------
    # Carga y preparación de datos (train + validación juntos)
    # --------------------------------------------------------------------------
    logger.info("Cargando datos...")
    df_train = pd.read_csv(CSV_ENTRENAR)
    df_val   = pd.read_csv(CSV_VALIDAR)

    for df in [df_train, df_val]:
        if 'Indice' in df.columns:
            df.drop(columns=['Indice'], inplace=True)

    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    logger.info(f"Total filas: {len(df_full)} ({len(df_train)} train + {len(df_val)} validación)")

    # Feature engineering
    df_full_der = crear_derivadas(df_full)
    columnas = [c for c in df_full_der.columns if c != 'Ciclos']
    X_full = df_full_der[columnas]
    y_full_ciclos = df_full['Ciclos'].values

    # Escalado
    scaler = StandardScaler()
    X_full_esc = pd.DataFrame(scaler.fit_transform(X_full), columns=columnas)

    # Umbral y etiquetas
    UMBRAL_OPTIMO = UMBRAL_CICLOS_DEFAULT
    y_full_umbral = (y_full_ciclos < UMBRAL_OPTIMO).astype(int)
    logger.info(f"Positivos: {y_full_umbral.sum()} | Negativos: {(y_full_umbral == 0).sum()}")

    # SMOTE
    k_neighbors = min(3, y_full_umbral.sum() - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    try:
        X_bal, y_bal = smote.fit_resample(X_full_esc, y_full_umbral)
        X_bal = pd.DataFrame(X_bal, columns=columnas)
        logger.info(f"SMOTE aplicado. Nuevas filas: {len(X_bal)}")
    except ValueError as e:
        logger.warning(f"SMOTE falló ({e}), se entrena sin balanceo.")
        X_bal, y_bal = X_full_esc, y_full_umbral

    # --------------------------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------------------------
    logger.info("Entrenando modelo...")
    modelo = crear_modelo(nombre_base, params)
    modelo.fit(X_bal, y_bal)
    logger.info("Modelo entrenado correctamente.")

    # --------------------------------------------------------------------------
    # Guardado
    # --------------------------------------------------------------------------
    ruta_salida = os.path.join(DIRECTORIO_SALIDA, f"modelo_{nombre_modelo}.pkl")

    config_produccion = {
        'scaler': scaler,
        'modelo': modelo,
        'feature_cols': columnas,
        'nombre_modelo': nombre_base
    }

    joblib.dump(config_produccion, ruta_salida)
    logger.info("=" * 60)
    logger.info(f"Modelo guardado en: {ruta_salida}")
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Entrena un modelo específico usando hiperparámetros ya optimizados.',
        usage='python entrenar_modelo.py -n <nombre_modelo> [--listar]'
    )
    parser.add_argument(
        '-n', '--nombre',
        default=None,
        help='Nombre del modelo a entrenar (ej: RandomForest_mcc, XGBoost_f1...)'
    )
    parser.add_argument(
        '--listar',
        action='store_true',
        help='Muestra todos los modelos disponibles en el archivo de configuraciones y sale.'
    )

    args = parser.parse_args()

    if not args.listar and args.nombre is None:
        parser.error("Debes indicar un modelo con -n o usar --listar para ver los disponibles.")

    main(args.nombre, args.listar)
