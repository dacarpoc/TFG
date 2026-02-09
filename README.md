# 🔬 Predicción de Rotura de Membranas

Sistema de Machine Learning para predecir roturas de membranas en prensas industriales basado en parámetros de ciclos de cocción.

## 📁 Estructura del Proyecto

```
GIT/
├── config.py           # Configuración centralizada
├── utils.py            # Funciones compartidas
├── procesado.py        # Pipeline de procesamiento de datos
├── preliminar.ipynb    # Búsqueda del mejor umbral de ciclos
├── modelado.ipynb      # Creación del modelo final
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

### Instalación de Jupyter Notebook

Si no se instaló con las dependencias:
```bash
pip install jupyter
```

**Para usar los notebooks:**
```bash
jupyter notebook
```
Esto abrirá una interfaz web donde puedes ejecutar `preliminar.ipynb` y `modelado.ipynb`.

### Datos de Entrada

Colocar en el directorio raíz:
- `parametros_prensa_1.csv` - Parámetros de ciclos de prensa 1
- `parametros_prensa_2.csv` - Parámetros de ciclos de prensa 2
- `excel_membranas_P1_zulu.csv` - Eventos de membranas prensa 1
- `excel_membranas_P2_zulu.csv` - Eventos de membranas prensa 2

**Formato completo de parámetros:**
```
Name,Description,Path,Timestamp,Value,Value_ID,UnitsAbbreviation,DefaultUnitsName,DefaultUnitsNameAbbreviation,Type,TypeQualifier,CategoryNames,WebId,Errors
```
**Columnas relevantes:** `Timestamp`, `Name`, `Value`

**Formato completo de membranas:**
```
Description,Timestamp_Removed,Timestamp_Created,Recipe,Press,Number of cures
```
**Columnas relevantes:** `Timestamp_Created`, `Description`, `Number of cures`

## 🚀 Uso

### 1. Procesamiento de Datos
```bash
python procesado.py
```
Genera los CSVs de entrenamiento, validación y testeo en `output/`:
- `entrenar.csv` - Usado en entrenamiento del modelo
- `validar.csv` - Usado en entrenamiento del modelo
- `testear.csv` - **NO usado en entrenamiento**, reservado para validación final

### 2. Entrenamiento Preliminar
Ejecutar el notebook `preliminar.ipynb` para encontrar el mejor umbral de ciclos y modelo base.

**Usa:** `entrenar.csv`, `validar.csv`

Guarda: `output/modelo_preliminar.pkl`

### 3. Modelado final
Ejecutar el notebook `modelado.ipynb` para optimizar hiperparámetros con múltiples métricas y obtener el modelo final.

**Usa:** `entrenar.csv`, `validar.csv`

Guarda: `output/modelo_final.pkl`

### 4. Predicciones

> **Doble propósito:** Este script sirve tanto para **validar el modelo** con datos no vistos como para **producción**.

**Uso:**
```bash
python predicciones.py <archivo.csv>
```

#### 4.1. Validación del modelo
```bash
python predicciones.py testear.csv
```
Usa `testear.csv` (dataset que **NO** fue usado en la creación del modelo) para evaluar el rendimiento final.

Genera: `output/predicciones.csv` con métricas de validación.

#### 4.2. Uso en producción
```bash
python predicciones.py datos_nuevos.csv
python predicciones.py /ruta/completa/al/archivo.csv
```

**Nota:** El script detecta automáticamente si los datos tienen la columna `Ciclos`. Si está presente, calcula métricas; si no, solo genera predicciones.

## ⚙️ Configuración

Todos los parámetros están centralizados en `config.py`:

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `RANDOM_STATE` | Semilla para reproducibilidad | 42 |
| `UMBRAL_CICLOS_DEFAULT` | Umbral de ciclos si no hay modelo preliminar | 9 |
| `N_TRIALS_OPTUNA` | Número de trials de optimización | 50 |
| `PESO_FALSOS_POSITIVOS` | Penalización de FP en score personalizado | 20 |
