# -*- coding: utf-8 -*-
"""
Configuración centralizada del proyecto de predicción de rotura de membranas.
"""

import os
import logging

# ================================================================================
# DIRECTORIOS
# ================================================================================

DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_SALIDA = os.path.join(DIRECTORIO_BASE, "output")

# ================================================================================
# ARCHIVOS DE DATOS DE ENTRADA
# ================================================================================

CONFIG_PRENSAS = {
    'P1': {
        'csv_parametros': os.path.join(DIRECTORIO_BASE, 'parametros_prensa_1.csv'),
        'csv_membranas': os.path.join(DIRECTORIO_BASE, 'excel_membranas_P1_zulu.csv')
    },
    'P2': {
        'csv_parametros': os.path.join(DIRECTORIO_BASE, 'parametros_prensa_2.csv'),
        'csv_membranas': os.path.join(DIRECTORIO_BASE, 'excel_membranas_P2_zulu.csv')
    }
}

ORDEN_PRENSAS = ['P1', 'P2']

# ================================================================================
# ARCHIVOS DE SALIDA
# ================================================================================

CSV_ENTRENAR = os.path.join(DIRECTORIO_SALIDA, "entrenar.csv")
CSV_VALIDAR = os.path.join(DIRECTORIO_SALIDA, "validar.csv")
CSV_TESTEAR = os.path.join(DIRECTORIO_SALIDA, "testear.csv")
RESULTADOS_PRELIMINAR = os.path.join(DIRECTORIO_SALIDA, "resultados_preliminar.csv")
RESULTADOS_MEJORES_MODELOS = os.path.join(DIRECTORIO_SALIDA, "registro_mejores_modelos.csv")
CONFIGURACION_MEJORES_MODELOS = os.path.join(DIRECTORIO_SALIDA, "todas_las_configs.joblib")

# ================================================================================
# PARÁMETROS DE PROCESAMIENTO
# ================================================================================

# Definir las etapas de un ciclo de cocción
GRUPOS_ETAPAS = {
    "barrido": ["PMAX_BARRIDO", "PMED_BARRIDO", "PMIN_BARRIDO", "TDesc", 
                "TMAX_BARRIDO", "TMED_BARRIDO", "TMIN_BARRIDO"],
    "coccion": ["PMAX_COCCION", "PMED_COCCION", "PMIN_COCCION", 
                "TMAX_COCCION", "TMED_COCCION", "TMIN_COCCION", 
                "TMI", "TMS", "Tiempo_llegada_presion", "Tiempo_llegada_temp"],
    "conformacion": ["PMAX_CONFORMACION", "PMED_CONFORMACION", "PMIN_CONFORMACION", 
                     "Tiempo_conformacion"]
}

PARAMETROS_ESPERADOS = [var for grupo in GRUPOS_ETAPAS.values() for var in grupo]

# Índices a eliminar manualmente (resultado de análisis previo)
# NOTA: Estos son índices del dataset concatenado final, NO índices originales
INDICES_ELIMINAR = [732, 1048, 1049, 1050, 1051, 1056, 1058, 1059, 1807, 1939, 1956, 1957]

# Umbrales de división del dataset
UMBRAL_VALIDACION = 2055
UMBRAL_TESTEO = 2728

# Columnas redundantes a eliminar en el preprocesamiento final
COLUMNAS_REDUNDANTES = ["TMI", "PMIN_CONFORMACION"]

# ================================================================================
# PARÁMETROS DE MODELADO
# ================================================================================

RANDOM_STATE = 42
UMBRALES_CICLOS = [3, 5, 7, 9]
UMBRAL_CICLOS_DEFAULT = 9  # Valor por defecto elegido después de ejecutar preliminar.ipynb
THRESHOLDS = [round(x, 2) for x in [i * 0.05 for i in range(1, 20)]]  # 0.05 a 0.95
N_TRIALS_OPTUNA = 50

# Modelos disponibles
MODELOS_DISPONIBLES = [
    'XGBoost',
    'LightGBM',
    'RandomForest',
    'HistGradientBoosting',
    'CatBoost',
    'ExtraTrees'
]

# Métricas a evaluar
METRICAS = {
    'f1': 'f1',
    'recall': 'recall',
    'mcc': 'mcc',
    'score_propio': 'Score Propio'
}

# Peso para penalización de falsos positivos en score personalizado
PESO_FALSOS_POSITIVOS = 20

# ================================================================================
# CONFIGURACIÓN DE LOGGING
# ================================================================================

def obtener_ruta(nombre_archivo: str) -> str:
    """
    Resuelve la ruta completa de un archivo.
    
    Busca el archivo en el siguiente orden:
    1. Si es ruta absoluta, la usa directamente
    2. Si existe en el directorio actual, usa esa ruta
    3. Si no, busca en DIRECTORIO_SALIDA (output/)
    
    Args:
        nombre_archivo: Nombre o ruta del archivo
        
    Returns:
        Ruta absoluta al archivo
    """
    # Si es ruta absoluta, usarla directamente
    if os.path.isabs(nombre_archivo):
        return nombre_archivo
    
    # Buscar primero en directorio actual
    if os.path.exists(nombre_archivo):
        return os.path.abspath(nombre_archivo)
    
    # Buscar en directorio de salida
    return os.path.join(DIRECTORIO_SALIDA, nombre_archivo)


def configurar_logging(nombre_modulo: str, nivel: int = logging.INFO) -> logging.Logger:
    """
    Configura y retorna un logger para el módulo especificado.
    
    Args:
        nombre_modulo: Nombre del módulo para el logger
        nivel: Nivel de logging (default: INFO)
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(nombre_modulo)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(nivel)
    
    return logger
