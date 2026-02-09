# -*- coding: utf-8 -*-
"""
Funciones de utilidad compartidas por todos los módulos del proyecto.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def crear_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea parámetros derivados a partir de los originales.
    
    Args:
        df: DataFrame con las columnas originales
        
    Returns:
        DataFrame con las columnas derivadas añadidas
    """
    df = df.copy()

    # Ratios Presión
    df['ratio_PMAX_PMED_BARRIDO'] = df['PMAX_BARRIDO'] / (df['PMED_BARRIDO'] + 1e-6)
    df['ratio_PMAX_PMED_COCCION'] = df['PMAX_COCCION'] / (df['PMED_COCCION'] + 1e-6)

    # Rangos Presión
    df['diff_PMAX_PMIN_BARRIDO'] = df['PMAX_BARRIDO'] - df['PMIN_BARRIDO']
    df['diff_PMAX_PMIN_COCCION'] = df['PMAX_COCCION'] - df['PMIN_COCCION']

    # Ratios Temperatura
    df['ratio_TMAX_TMED_BARRIDO'] = df['TMAX_BARRIDO'] / (df['TMED_BARRIDO'] + 1e-6)
    df['ratio_TMAX_TMED_COCCION'] = df['TMAX_COCCION'] / (df['TMED_COCCION'] + 1e-6)

    # Rangos Temperatura
    df['diff_TMAX_TMIN_BARRIDO'] = df['TMAX_BARRIDO'] - df['TMIN_BARRIDO']
    df['diff_TMAX_TMIN_COCCION'] = df['TMAX_COCCION'] - df['TMIN_COCCION']

    # Interacciones P-T
    df['P_T_interaction_BARRIDO'] = df['PMAX_BARRIDO'] * df['TMAX_BARRIDO']
    df['P_T_interaction_COCCION'] = df['PMAX_COCCION'] * df['TMAX_COCCION']
    df['P_T_ratio_BARRIDO'] = df['PMAX_BARRIDO'] / (df['TMAX_BARRIDO'] + 1e-6)
    df['P_T_ratio_COCCION'] = df['PMAX_COCCION'] / (df['TMAX_COCCION'] + 1e-6)

    # Agregados generales
    df['PMED_general'] = (df['PMED_BARRIDO'] + df['PMED_COCCION']) / 2
    df['TMED_general'] = (df['TMED_BARRIDO'] + df['TMED_COCCION']) / 2
    df['P_range_total'] = df['diff_PMAX_PMIN_BARRIDO'] + df['diff_PMAX_PMIN_COCCION']
    df['T_range_total'] = df['diff_TMAX_TMIN_BARRIDO'] + df['diff_TMAX_TMIN_COCCION']

    # Conformación
    df['P_conformacion_ratio'] = df['PMAX_CONFORMACION'] / (df['PMED_CONFORMACION'] + 1e-6)

    # Tiempos
    df['tiempos_total'] = df['Tiempo_descompresion'] + df['Tiempo_llegada_presion'] + df['Tiempo_llegada_temp']
    df['ratio_tiempos'] = df['Tiempo_llegada_presion'] / (df['Tiempo_llegada_temp'] + 1e-6)

    # Variabilidades
    df['var_P_BARRIDO'] = df[['PMAX_BARRIDO', 'PMED_BARRIDO', 'PMIN_BARRIDO']].std(axis=1)
    df['var_T_BARRIDO'] = df[['TMAX_BARRIDO', 'TMED_BARRIDO', 'TMIN_BARRIDO']].std(axis=1)
    df['var_P_COCCION'] = df[['PMAX_COCCION', 'PMED_COCCION', 'PMIN_COCCION']].std(axis=1)
    df['var_T_COCCION'] = df[['TMAX_COCCION', 'TMED_COCCION', 'TMIN_COCCION']].std(axis=1)

    return df


def analizar_membranas(y_ciclos: np.ndarray, y_pred: np.ndarray, umbral: int) -> tuple:
    """
    Analiza cuántas membranas únicas fueron detectadas.
    
    Args:
        y_ciclos: Array con los ciclos de vida
        y_pred: Array con las predicciones binarias
        umbral: Umbral de ciclos para considerar rotura
        
    Returns:
        Tupla (membranas detectadas, total membranas con rotura)
    """
    y_target = (y_ciclos < umbral).astype(int)
    df_anal = pd.DataFrame({
        'ciclos': y_ciclos, 
        'pred': y_pred, 
        'target': y_target
    })
    
    # Detectar cambios de membrana (cuando los ciclos saltan)
    df_anal['salto'] = (df_anal['ciclos'].diff() > 5).astype(int) 
    df_anal['membrana'] = df_anal['salto'].cumsum()

    resumen = df_anal.groupby('membrana').apply(
        lambda x: pd.Series({
            'tiene_ultimos': x['target'].any(),
            'detecta_alguno': (x['pred'].astype(bool) & x['target'].astype(bool)).any()
        })
    )
    
    con_rotura = resumen[resumen['tiene_ultimos'] == True]
    detectadas = con_rotura[con_rotura['detecta_alguno'] == True]
    
    return len(detectadas), len(con_rotura)


def calcular_score_balanceado(pct_membranas: float, fp: int, total_membranas: int, 
                               peso_fp: int = 20) -> float:
    """
    Calcula un score personalizado que pondera detección de membranas con falsos positivos.
    
    Args:
        pct_membranas: Porcentaje de membranas detectadas (0-1)
        fp: Número de falsos positivos
        total_membranas: Total de membranas
        peso_fp: Peso para penalización de falsos positivos
        
    Returns:
        Score balanceado
    """
    if total_membranas == 0:
        return 0
    
    penalizacion_fp = (fp / total_membranas) * peso_fp
    score = pct_membranas * 100 - penalizacion_fp
    
    return score


def evaluar_con_umbral(y_test: np.ndarray, y_proba: np.ndarray, 
                       y_ciclos: np.ndarray, umbral: int, threshold: float) -> dict:
    """
    Evalúa predicciones con un umbral específico.
    
    Args:
        y_test: Labels reales
        y_proba: Probabilidades predichas
        y_ciclos: Ciclos de vida reales
        umbral: Umbral de ciclos
        threshold: Umbral de probabilidad
        
    Returns:
        Diccionario con métricas o None si no hay predicciones
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    if y_pred.sum() == 0 and sum(y_proba) == 0:  
        return None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    det, total = analizar_membranas(y_ciclos, y_pred, umbral)
    pct_membranas = det / total if total > 0 else 0

    return {
        'threshold': threshold,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'membranas_detectadas': det,
        'membranas_total': total,
        'pct_membranas': pct_membranas
    }
