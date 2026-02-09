# 🔬 Predicción de Rotura de Membranas

Sistema de Machine Learning para predecir roturas de membranas en prensas industriales basado en parámetros de ciclos de cocción.

## 📁 Estructura del Proyecto

```
GIT/
├── config.py           # Configuración centralizada
├── utils.py            # Funciones compartidas
├── procesado.py        # Pipeline de procesamiento de datos
├── preliminar.ipynb    # Búsqueda del mejor umbral de ciclos
├── modelado.ipynb      # Optimización de hiperparámetros con Optuna
├── predicciones.py     # Script de inferencia
└── output/             # Directorio de salida (generado)
    ├── entrenar.csv
    ├── validar.csv
    ├── testear.csv
    ├── modelo_preliminar.pkl
    ├── modelo_final.pkl
    └── predicciones.csv
```

## 📋 Requisitos

### Dependencias
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna imbalanced-learn joblib
```

### Datos de Entrada
Colocar en el directorio raíz:
- `parametros_prensa_1.csv` - Parámetros de ciclos de prensa 1
- `parametros_prensa_2.csv` - Parámetros de ciclos de prensa 2
- `excel_membranas_P1_zulu.csv` - Eventos de membranas prensa 1
- `excel_membranas_P2_zulu.csv` - Eventos de membranas prensa 2

**Formato de parámetros**: Columnas `Timestamp`, `Name`, `Value`

**Formato de membranas**: Columnas `Timestamp_Created`, `Description`, `Number of cures`

## 🚀 Uso

### 1. Procesamiento de Datos
```bash
python procesado.py
```
Genera los CSVs de entrenamiento, validación y testeo en `output/`.

### 2. Entrenamiento Preliminar
Ejecutar el notebook `preliminar.ipynb` para encontrar el mejor umbral de ciclos y modelo base.

Guarda: `output/modelo_preliminar.pkl`

### 3. Optimización con Optuna
Ejecutar el notebook `modelado.ipynb` para optimizar hiperparámetros con múltiples métricas.

Guarda: `output/modelo_final.pkl`

### 4. Predicciones
```bash
python predicciones.py
```
Genera: `output/predicciones.csv` con probabilidades y clasificación de casos.

## ⚙️ Configuración

Todos los parámetros están centralizados en `config.py`:

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `RANDOM_STATE` | Semilla para reproducibilidad | 42 |
| `UMBRAL_CICLOS_DEFAULT` | Umbral de ciclos si no hay modelo preliminar | 9 |
| `N_TRIALS_OPTUNA` | Número de trials de optimización | 50 |
| `PESO_FALSOS_POSITIVOS` | Penalización de FP en score personalizado | 20 |

## 📊 Métricas

El sistema evalúa modelos con:
- **F1-Score**: Balance precisión/recall
- **Recall**: Detección de roturas
- **MCC**: Matthews Correlation Coefficient
- **Score Propio**: `% membranas detectadas - penalización FP`

## 📝 Modelos Soportados

- XGBoost
- LightGBM
- RandomForest
- ExtraTrees
- HistGradientBoosting
- CatBoost

## 🔄 Flujo de Datos

```
CSVs Prensas → procesado.py → entrenar/validar/testear.csv
                                      ↓
                              preliminar.ipynb → modelo_preliminar.pkl
                                      ↓
                              modelado.ipynb → modelo_final.pkl
                                      ↓
                              predicciones.py → predicciones.csv
```
