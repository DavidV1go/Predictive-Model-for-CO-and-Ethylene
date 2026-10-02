# MineTwin - Módulo Predictivo para Gemelo Digital de Ventilación Minera

Este repositorio implementa el motor analítico y predictivo de regresión multisalida para mezclas de gases químicos (**Metano + Etileno** y **Monóxido de Carbono + Etileno**) a partir de lecturas de alta frecuencia (100 Hz) capturadas por una matriz de 16 sensores químicos de óxidos metálicos (MOX).

Desarrollado bajo la metodología industrial **CRISP-DM** (Fases 0 a 9) y con cumplimiento estricto de la **Regla de Oro**: todas las métricas, intervalos de confianza, contrastes de hipótesis y figuras se calculan en tiempo de ejecución a partir de los datos reales cargados.

---

## 📁 Estructura del Proyecto

```text
.
├── app.py                     # Aplicación principal de Streamlit con las 10 pestañas CRISP-DM
├── modules/                   # Módulos especializados
│   ├── __init__.py
│   ├── io_data.py             # Parser robusto de 19 columnas, submuestreo temporal y E/S
│   ├── eda.py                 # EDA: descriptivos, IQR/MAD, Spearman, VIF, ADF/KPSS, PCA
│   ├── preprocessing.py       # Ingeniería de variables, RobustScaler y partición temporal con purga
│   ├── models.py              # Catálogo de modelos, baselines, multisalida y curvas de aprendizaje
│   ├── cv.py                  # TimeSeriesSplit, métricas fold-by-fold y regla 1-SE
│   ├── tuning.py              # Optimización bayesiana de hiperparámetros con Optuna
│   └── stats_tests.py         # Diagnóstico de residuos, Friedman, Wilcoxon Holm, MBB 95%, Bland-Altman
├── data/                      # Directorio de datos reales
│   ├── Pruebas_Ch4.txt        # Mezcla Metano + Etileno a 100 Hz (19 columnas)
│   └── Pruebas_CO.txt         # Mezcla Monóxido de Carbono + Etileno a 100 Hz (19 columnas)
├── .streamlit/
│   └── config.toml            # Configuración de Streamlit (maxUploadSize = 500 MB, puerto 8501)
├── requirements.txt           # Dependencias exactas del entorno Python
└── README.md                  # Guía de instalación, ejecución y arquitectura
```

---

## 📊 Formato de Datos Reales (`./data/`)

Los archivos `.txt` contienen 19 columnas numéricas separadas por espacios/tabulaciones:
1. `col 1` = **Tiempo** en segundos (`time_s`, paso 0.01 s = 100 Hz).
2. `col 2` = **Concentración de Metano o CO** en ppm (`gas_ppm`).
3. `col 3` = **Concentración de Etileno** en ppm (`ethylene_ppm`).
4. `col 4 - 19` = **Lecturas crudas de 16 sensores** (`S01` a `S16`).

---

## ⚙️ Pasos de Instalación y Ejecución

### 1. Clonar el repositorio y crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate  # En Linux/macOS
# venv\Scripts\activate   # En Windows
```

### 2. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecutar la aplicación Streamlit
```bash
streamlit run app.py
```

La aplicación abrirá automáticamente en su navegador en `http://localhost:8501`.

---

## 🔬 Metodología CRISP-DM Implementada

- **Fase 0: Comprensión del Negocio:** Objetivos de ventilación minera, criterios cuantitativos (RMSE < 3.0 ppm, R² > 0.90, latencia < 50 ms) y limitaciones explícitas de laboratorio.
- **Fase 1: EDA:** Detección de outliers mediante IQR y MAD, correlación no paramétrica de Spearman, multicolinealidad vía VIF, pruebas de raíz unitaria ADF/KPSS y reducción dimensional con PCA.
- **Fase 2: Preparación de Datos:** Medias móviles, desv. estándar móviles, derivadas de primer orden y lags temporales; escalado `RobustScaler` estrictamente dentro de `Pipeline` y partición temporal (80% Train+CV / 20% Test) con purga de memoria ("gap").
- **Fase 3: Entrenamiento:** Catálogo de 12 modelos (Baselines, Ridge, ElasticNet, KNN, Random Forest, ExtraTrees, Gradient Boosting, XGBoost, LightGBM, SVR, MLP) y curvas de aprendizaje.
- **Fase 4: Selección del Modelo:** Regla de una desviación estándar (1-SE Rule) priorizando la parsimonia y menor complejidad estructural; reajuste y evaluación en la prueba final.
- **Fase 5: Validación Cruzada:** `TimeSeriesSplit` cronológico sin shuffle para evitar fuga temporal; reporte de estabilidad fold por fold y diagrama predicho vs observado.
- **Fase 6: Hiperparámetros:** Búsqueda bayesiana con Optuna en CV anidada temporal, historial de optimización e importancia de parámetros.
- **Fase 7: Pruebas Estadísticas Inferenciales:**
  - Diagnóstico de residuos: Shapiro-Wilk, Anderson-Darling, gráfico Q-Q, Ljung-Box, Durbin-Watson y Breusch-Pagan.
  - Comparación de modelos: Prueba de Friedman + Wilcoxon pareado con corrección de Holm-Bonferroni, Cliff's Delta y Diebold-Mariano con errores estándar HAC Newey-West.
  - Intervalos de confianza al 95% para RMSE, MAE y R² mediante Moving Block Bootstrap (MBB).
  - Prueba de permutación frente al baseline e importancia de variables por permutación con IC al 95%.
  - Concordancia: Spearman, Kendall, CCC de Lin y gráfico de Bland-Altman con límites de acuerdo.
- **Fase 8: Auditoría de Riesgos y Evaluación Global:** Trade-offs computacionales, semáforo de explosividad y detector de lecturas fuera de distribución (OOD).
- **Fase 9: Despliegue:** Serialización con `joblib`, simulador de inferencia en tiempo real para 16 canales y exportación de reportes analíticos a CSV/Excel.
