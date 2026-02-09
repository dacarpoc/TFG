# -*- coding: utf-8 -*-
"""
Pipeline de procesamiento de datos para el proyecto de predicción de rotura de membranas.
"""

import pandas as pd
import numpy as np
import os

# Importar configuración centralizada
from config import (
    DIRECTORIO_BASE, DIRECTORIO_SALIDA, CONFIG_PRENSAS, ORDEN_PRENSAS,
    GRUPOS_ETAPAS, PARAMETROS_ESPERADOS, INDICES_ELIMINAR,
    UMBRAL_VALIDACION, UMBRAL_TESTEO, COLUMNAS_REDUNDANTES,
    configurar_logging
)

# Configurar logger
logger = configurar_logging(__name__)


# ================================================================================
# PROCESAMIENTO DE PARÁMETROS DE PRENSA
# ================================================================================

def procesar_parametros_prensa(csv_path: str) -> pd.DataFrame:
    """
    Procesa los parámetros de una prensa: pivotea datos por etapas y une temporalmente.
    
    Args:
        csv_path: Ruta al archivo CSV con parámetros
        
    Returns:
        DataFrame con los ciclos procesados o None si hay error
    """
    logger.info(f"Procesando parámetros de {os.path.basename(csv_path)}")

    # --- LECTURA ---
    try:
        df_parametros = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Archivo no encontrado: {csv_path}")
        return None
    except pd.errors.EmptyDataError:
        logger.warning(f"Archivo vacío: {csv_path}")
        return None
    except Exception as e:
        logger.error(f"Error al leer {csv_path}: {e}")
        return None

    # --- VALIDACIONES ---
    if df_parametros.empty:
        logger.warning(f"El archivo {csv_path} está vacío. Se omite.")
        return None

    required_cols = {"Timestamp", "Name", "Value"}
    if not required_cols.issubset(df_parametros.columns):
        raise ValueError(f"Faltan columnas obligatorias en {csv_path}. Se requiere: {required_cols}")

    faltantes = set(PARAMETROS_ESPERADOS) - set(df_parametros['Name'].unique())
    if faltantes:
        logger.warning(f"Faltan parámetros en {csv_path}: {faltantes}")

    # --- CONVERSIÓN DE TIPOS ---
    df_parametros["Timestamp"] = pd.to_datetime(df_parametros["Timestamp"], format='ISO8601')

    # --- PIVOTEO POR ETAPAS ---
    dfs_pivot = {}
    for nombre_grupo, variables_grupo in GRUPOS_ETAPAS.items():
        filtrado = df_parametros[df_parametros["Name"].isin(variables_grupo)]
        traspuesto = filtrado.pivot_table(index="Timestamp", columns="Name", values="Value", aggfunc='first')
        
        if traspuesto.empty:
            logger.warning(f"Etapa '{nombre_grupo}' sin datos en {csv_path}")
            dfs_pivot[nombre_grupo] = pd.DataFrame(columns=["Timestamp"] + variables_grupo)
        else:
            dfs_pivot[nombre_grupo] = traspuesto.sort_index().reset_index()

    df_barr = dfs_pivot["barrido"]
    df_coc = dfs_pivot["coccion"]
    df_conf = dfs_pivot["conformacion"]

    # --- PREPARACIÓN PARA UNIÓN ---
    df_barr = df_barr.sort_values("Timestamp").rename(columns={'Timestamp': 'Timestamp_barrido'})
    df_coc = df_coc.sort_values("Timestamp").rename(columns={'Timestamp': 'Timestamp_coccion'})
    df_conf = df_conf.sort_values("Timestamp").rename(columns={'Timestamp': 'Timestamp_conformacion'})

    # --- UNIÓN DE ETAPAS ---
    try:
        # Unir Cocción con Conformación
        df_temp = pd.merge_asof(
            df_coc, 
            df_conf, 
            left_on="Timestamp_coccion",
            right_on="Timestamp_conformacion", 
            direction="backward",  
            tolerance=pd.Timedelta("60min")  
        )
        
        # Unir Barrido con el resultado anterior
        df_ciclos = pd.merge_asof(
            df_barr,
            df_temp,
            left_on="Timestamp_barrido",
            right_on="Timestamp_coccion",
            direction="backward",  
            tolerance=pd.Timedelta("30min")  
        )
        
        # Limpiar cruces fallidos
        df_ciclos = df_ciclos.dropna(subset=['Timestamp_coccion', 'Timestamp_conformacion'])
        df_ciclos = df_ciclos.reset_index(drop=True)

    except Exception as e:
        raise RuntimeError(f"Error durante el merge en {csv_path}: {e}")

    if df_ciclos.empty:
        logger.warning(f"Resultado vacío para {csv_path} tras el cruce.")

    # --- LIMPIEZA Y CÁLCULOS ---
    if "TDesc" in df_ciclos.columns:
        df_ciclos = df_ciclos.rename(columns={"TDesc": "Tiempo_descompresion"})

    if "Tiempo_conformacion" in df_ciclos.columns:
        df_ciclos["Tiempo_conformacion"] = df_ciclos["Tiempo_conformacion"].abs()

    cols_signo = ["TMAX_BARRIDO", "TMED_BARRIDO", "TMIN_BARRIDO"]
    for c in cols_signo:
        if c in df_ciclos.columns:
            df_ciclos[c] *= -1    

    # Corregir intercambio MAX/MIN Barrido
    if set(['TMAX_BARRIDO', 'TMIN_BARRIDO']).issubset(df_ciclos.columns):
        mask = df_ciclos['TMAX_BARRIDO'] < df_ciclos['TMIN_BARRIDO']
        df_ciclos.loc[mask, ['TMAX_BARRIDO', 'TMIN_BARRIDO']] = (
            df_ciclos.loc[mask, ['TMIN_BARRIDO', 'TMAX_BARRIDO']].values
        )

    # Corregir intercambio MAX/MIN Cocción
    if set(['TMAX_COCCION', 'TMIN_COCCION']).issubset(df_ciclos.columns):
        mask2 = df_ciclos['TMAX_COCCION'] < df_ciclos['TMIN_COCCION']
        df_ciclos.loc[mask2, ['TMAX_COCCION', 'TMIN_COCCION']] = (
            df_ciclos.loc[mask2, ['TMIN_COCCION', 'TMAX_COCCION']].values
        )

    logger.info(f"Procesamiento completado: {len(df_ciclos)} ciclos")
    return df_ciclos


# ================================================================================
# AÑADIR CICLOS DE MEMBRANAS
# ================================================================================

def añadir_ciclos_membrana(df_data: pd.DataFrame, df_memb: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la información de ciclos de vida de membranas al dataset.
    
    Args:
        df_data: DataFrame con los parámetros procesados
        df_memb: DataFrame con los eventos de membranas
        
    Returns:
        DataFrame con la columna 'Ciclos' añadida
    """
    # Conversión de fechas a datetime con zona horaria UTC
    df_memb['Timestamp_Created'] = pd.to_datetime(df_memb['Timestamp_Created'], utc=True)
    df_data['Timestamp_conformacion'] = pd.to_datetime(df_data['Timestamp_conformacion'], format='mixed', utc=True)

    # Filtrar roturas de membrana y ordenar
    eventos = df_memb[df_memb['Description'] == 2].sort_values('Timestamp_Created').reset_index(drop=True)

    # Ordenar dataset objetivo por tiempo
    df_data = df_data.sort_values('Timestamp_conformacion').reset_index(drop=True)

    # Inicializar columna Ciclos
    df_data['Ciclos'] = np.nan

    logger.info(f"Procesando {len(eventos)} membranas...")

    # Lógica de cálculo de ciclos
    for i in range(len(eventos)):
        evento_actual = eventos.iloc[i]
        ts_inicio = evento_actual['Timestamp_Created']
        ciclos_inicio = evento_actual['Number of cures']

        # Determinar el timestamp de fin de una membrana e inicio de la siguiente
        ts_fin = None
        if i + 1 < len(eventos):
            ts_fin = eventos.iloc[i+1]['Timestamp_Created']

        # Filtrar las filas del dataset que corresponden a este ciclo de membrana
        if ts_fin:
            mask = (df_data['Timestamp_conformacion'] > ts_inicio) & (df_data['Timestamp_conformacion'] <= ts_fin)
        else:
            # Último evento, hasta el final
            mask = (df_data['Timestamp_conformacion'] > ts_inicio)

        indices = df_data[mask].index

        # Creamos un array con los valores decrecientes
        if len(indices) > 0:
            valores_ciclos = np.arange(ciclos_inicio, ciclos_inicio - len(indices), -1)

            # Asignar solo valores válidos (mayores o iguales a 0)
            mask_validos = valores_ciclos >= 0

            if np.any(mask_validos):
                df_data.loc[indices[mask_validos], 'Ciclos'] = valores_ciclos[mask_validos]

    # Eliminar filas con valores nulos en la columna 'Ciclos'
    df_data = df_data.dropna(subset=['Ciclos'])

    # Eliminar columnas de Timestamps 
    cols_to_drop = ['Timestamp_barrido', 'Timestamp_coccion', 'Timestamp_conformacion']
    df_data = df_data.drop(columns=[c for c in cols_to_drop if c in df_data.columns])

    return df_data


# ================================================================================
# DIVISIÓN EN CONJUNTOS
# ================================================================================

def dividir_dataset(df: pd.DataFrame, umbral_validacion: int, umbral_testeo: int) -> tuple:
    """
    Divide el dataset en conjuntos de entrenamiento, validación y testeo.
    
    Args:
        df: DataFrame completo
        umbral_validacion: Índice de separación entre entrenamiento y validación
        umbral_testeo: Índice de separación entre validación y testeo
        
    Returns:
        Tupla (entrenamiento, validacion, testeo)
    """
    testeo = df.iloc[umbral_testeo:].reset_index(drop=True)
    df_temp = df.iloc[:umbral_testeo].reset_index(drop=True)
    validacion = df_temp.iloc[umbral_validacion:].reset_index(drop=True)
    entrenamiento = df_temp.iloc[:umbral_validacion].reset_index(drop=True)

    return entrenamiento, validacion, testeo


# ================================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================================

def main():
    """Función principal que ejecuta todo el pipeline de procesamiento."""
    
    logger.info("=" * 70)
    logger.info("INICIO DEL PIPELINE DE PROCESAMIENTO")
    logger.info("=" * 70)
    
    # Crear directorio de salida si no existe
    os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)
    
    dfs_finales = []
    
    # ----------------------------------------------------------------------
    # PASOS 1 y 2: Procesar cada prensa
    # ----------------------------------------------------------------------
    
    for prensa in ORDEN_PRENSAS:
        logger.info(f"Procesando prensa {prensa}")
        
        config = CONFIG_PRENSAS[prensa]
        
        # Paso 1: Procesar parámetros
        df_ciclos = procesar_parametros_prensa(config['csv_parametros'])
        
        if df_ciclos is None:
            logger.error(f"No se pudo procesar la prensa {prensa}. Continuando...")
            continue
        
        # Paso 2: Añadir información de membranas
        logger.info(f"Añadiendo ciclos de membranas para {prensa}")
        
        try:
            df_memb = pd.read_csv(config['csv_membranas'])
            df_con_ciclos = añadir_ciclos_membrana(df_ciclos, df_memb)
            dfs_finales.append(df_con_ciclos)
            logger.info(f"Completado: {len(df_con_ciclos)} filas para {prensa}")
        except FileNotFoundError:
            logger.error(f"Archivo de membranas no encontrado para {prensa}")
            continue
        except Exception as e:
            logger.error(f"Error al procesar membranas para {prensa}: {e}")
            continue
    
    # ----------------------------------------------------------------------
    # Concatenación y filtrado
    # ----------------------------------------------------------------------
    
    logger.info("Generando dataset final consolidado...")
    
    if not dfs_finales:
        raise RuntimeError("No se generaron datos para ninguna prensa.")
    
    df_final = pd.concat(dfs_finales, ignore_index=True)
    logger.info(f"Dataset concatenado: {len(df_final)} filas")
    
    # Agregar índice inicial para referencia
    df_final["Indice_Original"] = range(len(df_final))
    
    # Eliminar filas manuales problemáticas
    # NOTA: INDICES_ELIMINAR son posiciones en el dataset concatenado (0-indexed)
    indices_validos = [idx for idx in INDICES_ELIMINAR if 0 <= idx < len(df_final)]
    if indices_validos:
        logger.info(f"Eliminando {len(indices_validos)} filas manualmente marcadas...")
        df_final = df_final.drop(indices_validos).reset_index(drop=True)
    else:
        logger.warning("Ninguno de los índices a eliminar se encontró en el rango válido del dataset.")
    
    # Mantener índice para trazabilidad pero crear uno nuevo consecutivo
    df_final["Indice"] = range(len(df_final))
    
    # ----------------------------------------------------------------------
    # Preprocesamiento final y división
    # ----------------------------------------------------------------------
    
    logger.info("Preprocesamiento final y división del dataset")
    
    # Eliminar columnas redundantes
    cols_existentes = [c for c in COLUMNAS_REDUNDANTES if c in df_final.columns]
    if cols_existentes:
        df_final = df_final.drop(columns=cols_existentes)
        logger.info(f"Columnas eliminadas: {cols_existentes}")
    
    # Eliminar columna de índice original (ya no necesaria)
    if "Indice_Original" in df_final.columns:
        df_final = df_final.drop(columns=["Indice_Original"])
    
    # Dividir dataset
    entrenamiento, validacion, testeo = dividir_dataset(
        df_final, 
        UMBRAL_VALIDACION, 
        UMBRAL_TESTEO
    )
    
    logger.info(f"Filas Entrenamiento: {len(entrenamiento)}")
    logger.info(f"Filas Validación: {len(validacion)}")
    logger.info(f"Filas Testeo: {len(testeo)}")
    
    # ----------------------------------------------------------------------
    # Guardado de resultados
    # ----------------------------------------------------------------------
    
    logger.info("Guardando resultados...")
    
    # Guardar CSVs en el directorio de salida
    entrenamiento.to_csv(os.path.join(DIRECTORIO_SALIDA, "entrenar.csv"), index=False)
    logger.info(f"Guardado: entrenar.csv")
    
    validacion.to_csv(os.path.join(DIRECTORIO_SALIDA, "validar.csv"), index=False)
    logger.info(f"Guardado: validar.csv")
    
    testeo.to_csv(os.path.join(DIRECTORIO_SALIDA, "testear.csv"), index=False)
    logger.info(f"Guardado: testear.csv")
    
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETADO")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
