# Documentación Técnica del Proyecto: Predicción de Rotura de Membranas

## 1. Introducción y Objetivo
Este proyecto implementa una solución integral de **Mantenimiento Predictivo** basada en Machine Learning para prensas industriales. El objetivo principal es predecir la rotura de membranas utilizadas en el proceso de conformado, analizando las variaciones sutiles en los parámetros de telemetría (presión, temperatura y tiempos) a lo largo de los ciclos de producción.

---

## 2. Arquitectura de Archivos y Scripts

### A. Módulo de Configuración (`config.py`)
Este script actúa como la "columna vertebral" de configuración del proyecto, eliminando valores "hardcodeados" y centralizando la gestión de rutas y constantes.

*   **Definición del Proceso Industrial:** Mapea las variables de sensores a sus etapas correspondientes.

```python
# Fragmento de config.py
GRUPOS_ETAPAS = {
    "barrido": ["PMAX_BARRIDO", "PMED_BARRIDO", "PMIN_BARRIDO", "TDesc", 
                "TMAX_BARRIDO", "TMED_BARRIDO", "TMIN_BARRIDO"],
    "coccion": ["PMAX_COCCION", "PMED_COCCION", "PMIN_COCCION", 
                "TMAX_COCCION", "TMED_COCCION", "TMIN_COCCION", 
                "TMI", "TMS", "Tiempo_llegada_presion", "Tiempo_llegada_temp"],
    "conformacion": ["PMAX_CONFORMACION", "PMED_CONFORMACION", "PMIN_CONFORMACION", 
                     "Tiempo_conformacion"]
}
```

*   **Parámetros de ML:** Define modelos y métricas clave.

```python
# Fragmento de config.py
MODELOS_DISPONIBLES = [
    'XGBoost', 'LightGBM', 'RandomForest',
    'HistGradientBoosting', 'CatBoost', 'ExtraTrees'
]

METRICAS = {
    'f1': 'f1',
    'recall': 'recall',
    'mcc': 'mcc',
    'score_propio': 'Score Propio'
}
```

### B. Librería de Utilidades (`utils.py`)
Contiene la lógica matemática y de evaluación reutilizable.

**1. Feature Engineering (`crear_derivadas`)**
Transforma señales crudas en indicadores complejos (ratios, diferencias, interacciones).

```python
# Fragmento de utils.py
def crear_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratios Presión
    df['ratio_PMAX_PMED_BARRIDO'] = df['PMAX_BARRIDO'] / (df['PMED_BARRIDO'] + 1e-6)
    
    # Rangos Temperatura
    df['diff_TMAX_TMIN_COCCION'] = df['TMAX_COCCION'] - df['TMIN_COCCION']

    # Interacciones P-T
    df['P_T_interaction_COCCION'] = df['PMAX_COCCION'] * df['TMAX_COCCION']
    
    # Variabilidades (Desviación estándar intra-ciclo)
    df['var_P_COCCION'] = df[['PMAX_COCCION', 'PMED_COCCION', 'PMIN_COCCION']].std(axis=1)

    return df
```

**2. Evaluación Orientada al Activo (`analizar_membranas`)**
Evalúa el rendimiento agrupando por "Vida de Membrana" en lugar de por fila individual.

```python
# Fragmento de utils.py
def analizar_membranas(y_ciclos: np.ndarray, y_pred: np.ndarray, umbral: int) -> tuple:
    # ...
    # Detectar cambios de membrana (cuando los ciclos saltan)
    df_anal['salto'] = (df_anal['ciclos'].diff() > 5).astype(int) 
    df_anal['membrana'] = df_anal['salto'].cumsum()

    resumen = df_anal.groupby('membrana').apply(
        lambda x: pd.Series({
            'tiene_ultimos': x['target'].any(), # ¿Llegó a zona de rotura?
            # ¿El modelo predijo 1 en algún momento de la zona de peligro?
            'detecta_alguno': (x['pred'].astype(bool) & x['target'].astype(bool)).any()
        })
    )
    # ...
    return len(detectadas), len(con_rotura)
```

**3. Métrica de Negocio (`calcular_score_balanceado`)**
Penaliza los falsos positivos que detienen la producción.

```python
# Fragmento de utils.py
def calcular_score_balanceado(pct_membranas: float, fp: int, total_membranas: int, 
                               peso_fp: int = 20) -> float:
    if total_membranas == 0:
        return 0
    
    penalizacion_fp = (fp / total_membranas) * peso_fp
    score = pct_membranas * 100 - penalizacion_fp
    
    return score
```

### C. Pipeline ETL (`procesado.py`)
Convierte logs desordenados en un dataset tabular limpio.

**1. Pivoteo y Agrupación por Etapas**
Transforma datos de formato "largo" a "ancho".

```python
# Fragmento de procesado.py
for nombre_grupo, variables_grupo in GRUPOS_ETAPAS.items():
    filtrado = df_parametros[df_parametros["Name"].isin(variables_grupo)]
    # Pivoteo: Index=Timestamp, Columns=Variables
    traspuesto = filtrado.pivot_table(index="Timestamp", columns="Name", values="Value", aggfunc='first')
    dfs_pivot[nombre_grupo] = traspuesto.sort_index().reset_index()
```

**2. Alineación Temporal (`merge_asof`)**
Une las etapas (Barrido, Cocción, Conformación) que ocurren en tiempos distintos con tolerancia temporal.

```python
# Fragmento de procesado.py
# Unir Cocción con Conformación (buscando hacia atrás con tolerancia de 60min)
df_temp = pd.merge_asof(
    df_coc, 
    df_conf, 
    left_on="Timestamp_coccion",
    right_on="Timestamp_conformacion", 
    direction="backward",  
    tolerance=pd.Timedelta("60min")  
)
```

**3. Cálculo de Vida Remanente (`añadir_ciclos_membrana`)**
Genera la cuenta regresiva de ciclos hasta la rotura.

```python
# Fragmento de procesado.py
for i in range(len(eventos)):
    # ...
    # Determinar el intervalo de tiempo de esta membrana
    if ts_fin:
        mask = (df_data['Timestamp_conformacion'] > ts_inicio) & (df_data['Timestamp_conformacion'] <= ts_fin)
    
    indices = df_data[mask].index
    # Crear cuenta regresiva: [100, 99, ..., 2, 1]
    if len(indices) > 0:
        valores_ciclos = np.arange(ciclos_inicio, ciclos_inicio - len(indices), -1)
        df_data.loc[indices[mask_validos], 'Ciclos'] = valores_ciclos[mask_validos]
```

### D. Entrenamiento Final (`entrenar_modelo.py`)
Genera el artefacto para producción usando la mejor configuración encontrada.

```python
# Fragmento de entrenar_modelo.py
def main(nombre_modelo: str, listar: bool = False):
    # 1. Cargar configuración óptima (sin buscar de nuevo)
    configs = joblib.load(CONFIGURACION_MEJORES_MODELOS)
    config_elegida = configs_por_nombre[nombre_modelo]
    params = ast.literal_eval(config_elegida.get('hiperparametros', '{}'))

    # 2. Cargar y procesar TODOS los datos (Train + Valid)
    df_full = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)
    
    # 3. Entrenar modelo final
    modelo = crear_modelo(nombre_base, params)
    modelo.fit(X_bal, y_bal)

    # 4. Guardar "Paquete de Producción" (.pkl)
    config_produccion = {
        'umbral_ciclos': UMBRAL_OPTIMO,
        'threshold': threshold_elegido,
        'scaler': scaler,
        'modelo': modelo,
        'feature_cols': columnas, # Guardar nombres de columnas para evitar errores de orden
        'nombre_modelo': nombre_base
    }
    joblib.dump(config_produccion, ruta_salida)
```

### E. Inferencia (`predicciones.py`)
Script para uso del usuario final.

```python
# Fragmento de predicciones.py
def main(csv_input: str, model_file: str, threshold_override: float = None):
    # 1. Cargar modelo y configuración
    config = joblib.load(archivo_modelo)
    MODELO = config.get('modelo', None)
    SCALER = config.get('scaler', None)
    
    # 2. Cargar datos nuevos
    df_input = pd.read_csv(archivo_entrada)
    
    # 3. Replicar Feature Engineering
    df_processed = crear_derivadas(df_input)
    X_new = df_processed[COLS_FEATURE]
    
    # 4. Replicar Escalado
    if SCALER:
        X_new_scaled = pd.DataFrame(SCALER.transform(X_new), columns=COLS_FEATURE)
        
    # 5. Predecir Probabilidad
    proba_final = MODELO.predict_proba(X_new_scaled)[:, 1]
    
    # 6. Guardar resultados
    guardar_predicciones(..., probas=proba_final, ...)
```

---

## 3. Notebooks de Investigación (Jupyter)

### A. Exploración Inicial (`preliminar.ipynb`)
Itera sobre diferentes umbrales de ciclos para encontrar el punto óptimo de predicción.

```python
# Fragmento de preliminar.ipynb
# Grid Search manual sobre umbrales (3, 5, 7, 9 ciclos antes de rotura)
for UMBRAL in UMBRALES_CICLOS:
    logger.info(f"Evaluando con umbral de rotura: {UMBRAL} ciclos")
    y_train_bin = (y_train_ciclos < UMBRAL).astype(int)
    
    for nombre, modelo in MODELOS.items():
        modelo.fit(X_train_esc, y_train_bin)
        # ... evaluar y registrar scores ...
```

### B. Optimización Avanzada (`modelado.ipynb`)
Utiliza **Optuna** para buscar hiperparámetros. Define una función objetivo compleja que incluye Cross-Validation y SMOTE.

```python
# Fragmento de modelado.ipynb (Función Objetivo de Optuna)
def objetivo(trial):
    # 1. Optuna sugiere parámetros
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        # ...
    }
    
    # 2. Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for train_idx, val_idx in cv.split(X_train_full, y_train_full):
        # 3. SMOTE SOLO en entrenamiento (evitar Data Leakage)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_fold_train_bal, y_fold_train_bal = smote.fit_resample(X_fold_train, y_fold_train)

        # 4. Entrenar y validar
        model = XGBClassifier(**params) # Instancia dinámica
        model.fit(X_fold_train_bal, y_fold_train_bal)
        
        # ... acumular scores ...
        
    return np.mean(scores_cv)
```

---

# ANEXO: Detalles Técnicos de Clases y Librerías

### 1. Clases Principales

*   **`xgboost.XGBClassifier`**:
    *   Implementación optimizada de Gradient Boosting.
    *   En `modelado.ipynb`, se configura con `eval_metric='aucpr'` para optimizar curvas de precisión-recall en datos desbalanceados.
    ```python
    XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric='aucpr', n_jobs=-1)
    ```

*   **`sklearn.ensemble.RandomForestClassifier`**:
    *   Ensamble de árboles de decisión (Bagging).
    *   Se usa con `class_weight='balanced_subsample'` para ajustar pesos de clases en cada bootstrap sample.
    ```python
    RandomForestClassifier(**params, class_weight='balanced_subsample', n_jobs=-1)
    ```

*   **`optuna.study.Study`**:
    *   Orquestador de la optimización bayesiana.
    ```python
    estudio = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    estudio.optimize(objetivo, n_trials=N_TRIALS_OPTUNA)
    ```

### 2. Persistencia (Serialización)

*   **Joblib (`.joblib`):** Usado para guardar configuraciones ligeras (diccionarios).
    ```python
    # entrenar_modelo.py
    configs = joblib.load(ruta_configs) # Carga rápida
    ```
*   **Pickle (`.pkl`):** Usado para guardar el objeto del modelo binario completo.
    ```python
    # entrenar_modelo.py
    joblib.dump(config_produccion, ruta_salida) # Guarda modelo + scaler
    ```
