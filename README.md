# 🔬 Predicción de Rotura de Membranas

Sistema de Machine Learning para predecir roturas de membranas en prensas industriales basado en parámetros de ciclos de cocción.

## 📁 Estructura del Proyecto

```
TFG/
├── config.py               # Configuración centralizada (rutas, hiperparámetros, constantes)
├── utils.py                # Funciones compartidas (feature engineering, métricas, evaluación)
├── procesado.py            # Pipeline de procesamiento de datos crudos
├── preliminar.ipynb        # Búsqueda del mejor umbral de ciclos y modelo base
├── modelado.ipynb          # Optimización de hiperparámetros con Optuna
├── entrenar_modelo.py      # Entrenamiento final del modelo elegido
├── predicciones.py         # Script de inferencia y evaluación
└── output/                 # Directorio de salida (generado automáticamente)
    ├── entrenar.csv                    # Conjunto de entrenamiento
    ├── validar.csv                     # Conjunto de validación
    ├── testear.csv                     # Conjunto de testeo (no visto en modelado)
    ├── resultados_preliminar.csv       # Grilla completa del análisis preliminar
    ├── registro_mejores_modelos.csv    # Mejores configuraciones por modelo/métrica
    ├── todas_las_configs.joblib        # Hiperparámetros optimizados (para entrenar_modelo.py)
    ├── modelo_<nombre>.pkl             # Modelo entrenado listo para producción
    └── predicciones_<...>.csv          # Resultados de inferencia
```

## 📋 Requisitos

### Configuración del Entorno Virtual

1. **Crear entorno virtual:**
```bash
python3 -m venv venv
```

2. **Activar el entorno virtual:**
- **macOS/Linux:**
```bash
source venv/bin/activate
```
- **Windows:**
```cmd
venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna imbalanced-learn joblib jupyter
```

### Datos de Entrada

Colocar en el directorio raíz del proyecto:
- `parametros_prensa_1.csv` — Parámetros de ciclos de la prensa 1
- `parametros_prensa_2.csv` — Parámetros de ciclos de la prensa 2
- `excel_membranas_P1_zulu.csv` — Eventos de membranas de la prensa 1
- `excel_membranas_P2_zulu.csv` — Eventos de membranas de la prensa 2

**Formato de parámetros** (columnas relevantes): `Timestamp`, `Name`, `Value`

**Formato de membranas** (columnas relevantes): `Timestamp_Created`, `Description`, `Number of cures`

---

## 🚀 Flujo de Ejecución

### Paso 1 — Procesamiento de datos
```bash
python procesado.py
```
Procesa los CSVs de las dos prensas, une las etapas de cada ciclo (barrido, cocción, conformación), añade los ciclos de vida de cada membrana y divide el dataset en tres conjuntos:

| Fichero | Uso |
|---|---|
| `entrenar.csv` | Entrenamiento de modelos |
| `validar.csv` | Evaluación durante el modelado |
| `testear.csv` | Evaluación final (no visto en modelado) |

---

### Paso 2 — Análisis preliminar
Ejecutar el notebook **`preliminar.ipynb`**.

Entrena 6 modelos base con distintos umbrales de ciclos (3, 5, 7, 9) y thresholds de decisión, usando validación en `validar.csv` y balanceo SMOTE. Guarda la grilla completa de resultados:
- `output/resultados_preliminar.csv`

El umbral óptimo elegido (`UMBRAL_CICLOS_DEFAULT = 9`) queda fijado en `config.py`.

---

### Paso 3 — Optimización de hiperparámetros
Ejecutar el notebook **`modelado.ipynb`**.

Para cada modelo y métrica (F1, Recall, MCC y score personalizado), optimiza hiperparámetros con **Optuna** (50 trials, validación cruzada estratificada de 5 folds). Evalúa los modelos resultantes en `validar.csv` y guarda:
- `output/registro_mejores_modelos.csv` — ranking de todas las configuraciones
- `output/todas_las_configs.joblib` — hiperparámetros para uso en `entrenar_modelo.py`

---

### Paso 4 — Entrenamiento final
```bash
# Ver todos los modelos disponibles y sus scores:
python entrenar_modelo.py --listar

# Entrenar el modelo elegido:
python entrenar_modelo.py -n <nombre_modelo>
```

Carga los hiperparámetros optimizados del paso anterior, entrena sobre **train + validación completos** con SMOTE y guarda el modelo listo para producción:
- `output/modelo_<nombre_modelo>.pkl`

El `.pkl` contiene: modelo entrenado, scaler ajustado y lista de features, todo lo necesario para inferencia sin dependencias externas.

**Ejemplo:**
```bash
python entrenar_modelo.py -n "ExtraTrees_Score Propio"
```

---

### Paso 5 — Predicciones
```bash
python predicciones.py -d <archivo.csv> -m <modelo.pkl> -t <threshold>
```

| Argumento | Descripción |
|---|---|
| `-d` / `--dataset` | CSV de entrada (ruta relativa o absoluta) |
| `-m` / `--modelo` | Fichero `.pkl` del modelo a usar |
| `-t` / `--threshold` | Umbral de decisión (valor entre 0 y 1) |

El script aplica automáticamente el mismo feature engineering y escalado que en el entrenamiento. Si el CSV contiene la columna `Ciclos`, calcula métricas completas (TP, FP, FN, TN, membranas detectadas, score balanceado); si no, solo genera predicciones.

Genera: `output/predicciones_<dataset>_<modelo>_<threshold>.csv`

**Ejemplo — evaluación con datos de testeo:**
```bash
python predicciones.py -d testear.csv -m "output/modelo_ExtraTrees_Score Propio.pkl" -t 0.3
```

**Ejemplo — producción con datos nuevos:**
```bash
python predicciones.py -d /ruta/datos_nuevos.csv -m "output/modelo_ExtraTrees_Score Propio.pkl" -t 0.3
```

---

## ⚙️ Configuración

Todos los parámetros clave están centralizados en `config.py`:

| Parámetro | Descripción | Valor |
|---|---|---|
| `RANDOM_STATE` | Semilla para reproducibilidad | `42` |
| `UMBRALES_CICLOS` | Umbrales explorados en el análisis preliminar | `[3, 5, 7, 9]` |
| `UMBRAL_CICLOS_DEFAULT` | Umbral óptimo elegido tras el preliminar | `9` |
| `THRESHOLDS` | Thresholds de decisión explorados | `0.05 … 0.95` (paso 0.05) |
| `N_TRIALS_OPTUNA` | Número de trials por optimización de Optuna | `50` |
| `PESO_FALSOS_POSITIVOS` | Penalización de FP en el score personalizado | `20` |
| `MODELOS_DISPONIBLES` | Algoritmos evaluados | XGBoost, LightGBM, RandomForest, ExtraTrees, HistGradientBoosting, CatBoost |
