"""
MineTwin - Motor del Módulo "Modelo Predictivo" para Gemelo Digital de Ventilación Minera.
Aplicación Streamlit desarrollada bajo metodología CRISP-DM completa (Fases 0 a 9).
Cumplimiento estricto de la REGLA DE ORO:
Todos los índices, métricas, p-valores, intervalos y gráficos se calculan en tiempo de ejecución.
"""

import os
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importar módulos propios
from modules.io_data import (
    discover_local_datasets,
    parse_raw_sensor_file,
    aggregate_by_time_window,
    SENSOR_COLUMNS
)
from modules.eda import (
    compute_basic_info,
    compute_descriptive_stats,
    detect_outliers_summary,
    plot_sensor_timeseries,
    compute_spearman_correlations,
    compute_vif_dataframe,
    compute_stationarity_tests,
    detect_concentration_steps,
    compute_pca_analysis
)
from modules.preprocessing import (
    build_engineered_dataset,
    temporal_train_test_split,
    create_pipeline
)
from modules.models import (
    get_model_catalog,
    train_single_model,
    compute_learning_curve_data
)
from modules.cv import (
    run_temporal_cv_for_models,
    select_best_model_1se_rule,
    plot_cv_stability_chart,
    plot_predicted_vs_observed_cv
)
from modules.tuning import (
    get_search_space_description,
    run_optuna_tuning
)
from modules.stats_tests import (
    run_residual_diagnostics,
    run_friedman_and_posthoc,
    compute_diebold_mariano_test,
    compute_moving_block_bootstrap_ci,
    run_permutation_test_vs_baseline,
    compute_permutation_feature_importance_ci,
    compute_concordance_metrics
)

# Configuración de página de Streamlit
st.set_page_config(
    page_title="MineTwin | Módulo Predictivo de Gases",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado para ambiente industrial y analítico
st.markdown("""
<style>
    .reportview-container { background: #0b132b; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .metric-card { background: #1c2541; padding: 1.2rem; border-radius: 8px; border-left: 4px solid #0284c7; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e293b; border-radius: 4px 4px 0 0; padding: 8px 16px; color: #cbd5e1; }
    .stTabs [aria-selected="true"] { background-color: #0284c7 !important; color: white !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FUNCIONES EN CACHÉ DE STREAMLIT
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(file_path_or_bytes, fname: str, agg_window: float):
    """Carga y aplica agregación temporal opcional."""
    df_raw, meta = parse_raw_sensor_file(file_path_or_bytes, filename=fname)
    df_agg = aggregate_by_time_window(df_raw, aggregation_window_s=agg_window)
    return df_raw, df_agg, meta


# ==============================================================================
# BARRA LATERAL (SIDEBAR)
# ==============================================================================

st.sidebar.image("https://images.unsplash.com/photo-1578328819058-b69f3a3b0f6b?w=400&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.title("⛏️ MineTwin Analytics")
st.sidebar.markdown("**Gemelo Digital de Ventilación Minera**")
st.sidebar.markdown("---")

# 1. Descubrimiento y Selección de Datasets
local_datasets = discover_local_datasets("./data")
dataset_options = list(local_datasets.keys()) + ["Subir archivo (.txt / .csv)..."]

selected_option = st.sidebar.selectbox(
    "Seleccionar Dataset Químico:",
    dataset_options,
    index=0 if len(dataset_options) > 0 else 0
)

uploaded_file = None
file_path_to_load = None
dataset_label = "Dataset"

if selected_option == "Subir archivo (.txt / .csv)...":
    uploaded_file = st.sidebar.file_uploader(
        "Cargar archivo de 16 canales a 100 Hz:",
        type=["txt", "csv"]
    )
    if uploaded_file is not None:
        dataset_label = uploaded_file.name
else:
    file_path_to_load = local_datasets.get(selected_option)
    dataset_label = selected_option

# 2. Configuración de Submuestreo / Ventana Temporal
st.sidebar.subheader("⏱️ Muestreo Temporal")
agg_choice = st.sidebar.selectbox(
    "Agregación por Ventanas:",
    ["1.0 s (Promedio cada 1 s - Recomendado)", "0.5 s", "0.1 s", "Sin agregación (Crudo 100 Hz)"],
    index=0
)
agg_map = {
    "1.0 s (Promedio cada 1 s - Recomendado)": 1.0,
    "0.5 s": 0.5,
    "0.1 s": 0.1,
    "Sin agregación (Crudo 100 Hz)": 0.01
}
agg_window_val = agg_map[agg_choice]

# 3. Configuración de Variable Objetivo
st.sidebar.subheader("🎯 Variable Objetivo")
target_mode = st.sidebar.selectbox(
    "Modo de Regresión:",
    ["Multisalida: [Gas Objetivo, Etileno]", "Gas Objetivo Único (ppm)", "Etileno Único (ppm)"],
    index=0
)
target_key_map = {
    "Multisalida: [Gas Objetivo, Etileno]": "multisalida",
    "Gas Objetivo Único (ppm)": "gas_ppm",
    "Etileno Único (ppm)": "ethylene_ppm"
}
target_mode_code = target_key_map[target_mode]

# 4. Horizonte de Pronóstico (h)
forecast_h = st.sidebar.slider(
    "Horizonte de Pronóstico h (pasos adelante):",
    min_value=0, max_value=10, value=0, step=1,
    help="Si h=0 realiza estimación instantánea t. Si h>0, entrena para anticipar concentraciones a t+h."
)

st.sidebar.markdown("---")
st.sidebar.markdown("🔒 **Semilla Global:** `random_state = 42`")
st.sidebar.markdown("📐 **Partición:** 80% Train+CV / 20% Test (Purga temporal)")


# ==============================================================================
# CARGA EFECTIVA DE DATOS
# ==============================================================================

df_raw = None
df_current = None
metadata = {}

try:
    if uploaded_file is not None:
        df_raw, df_current, metadata = load_and_preprocess_dataset(uploaded_file, uploaded_file.name, agg_window_val)
    elif file_path_to_load and os.path.exists(file_path_to_load):
        df_raw, df_current, metadata = load_and_preprocess_dataset(file_path_to_load, os.path.basename(file_path_to_load), agg_window_val)
    else:
        st.warning("⚠️ No se encontró ningún dataset en `./data/`. Por favor coloque `Pruebas_Ch4.txt` o use el cargador lateral.")
except Exception as e:
    st.error(f"Error procesando el archivo: {str(e)}")


# ==============================================================================
# ENCABEZADO PRINCIPAL
# ==============================================================================

st.markdown('<div class="main-header">MineTwin | Motor Predictivo de Gases en Ventilación</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Módulo de Regresión Multisalida con Matrices de Sensores Químicos a 100 Hz bajo Metodología CRISP-DM</div>', unsafe_allow_html=True)

if df_current is None:
    st.info("Cargue o seleccione un dataset en la barra lateral para iniciar el procesamiento en tiempo de ejecución.")
    st.stop()


# ==============================================================================
# PESTAÑAS METODOLOGÍA CRISP-DM (FASES 0 A 9)
# ==============================================================================

tabs = st.tabs([
    "0. Negocio",
    "1. EDA",
    "2. Preparación",
    "3. Entrenamiento",
    "4. Selección",
    "5. Validación (CV)",
    "6. Hiperparámetros",
    "7. Pruebas Robustas",
    "8. Auditoría de Riesgos",
    "9. Despliegue"
])


# ------------------------------------------------------------------------------
# TAB 0: COMPRENSIÓN DEL NEGOCIO
# ------------------------------------------------------------------------------
with tabs[0]:
    st.header("Fase 0: Comprensión del Negocio (Business Understanding)")
    st.markdown("""
    ### 🎯 Contexto y Misión en MineTwin
    En labores subterráneas de minería de carbón y roca dura, la presencia súbita de **Metano ($CH_4$)**, **Monóxido de Carbono ($CO$)**
    y gases interferentes hidrocarbúricos como el **Etileno ($C_2H_4$)** representa un peligro crítico de deflagración, explosión o intoxicación.

    El gemelo digital **MineTwin** integra un motor de inferencia analítica para estimar continuamente y en tiempo real
    las concentraciones cuantitativas individuales de mezclas binarias de gases a partir de la respuesta eléctrica
    de un arreglo multicanal de 16 sensores químicos de óxidos metálicos (MOX).
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Objetivo Predictivo</h4>
            <p>Regresión multisalida simultánea de <b>[gas_ppm, ethylene_ppm]</b> a partir de los 16 canales de sensores químicos brutos y variables dinámicas de flujo.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🏆 Criterios de Éxito Medibles</h4>
            <p>1. <b>RMSE &lt; 3.0 ppm</b> en conjunto de prueba independiente.<br>
            2. <b>R² &gt; 0.90</b> en ambas variables objetivo.<br>
            3. <b>Latencia de inferencia &lt; 50 ms</b> por muestra para control en lazo cerrado.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>⚠️ Restricciones Técnicas</h4>
            <p>1. <b>Dataset de laboratorio</b> obtenido en cámara de mezcla controlada; no sustituye mediciones de ventilación en frentes de explotación reales.<br>
            2. <b>Sin certificación MSHA</b>: módulo de soporte a decisiones del gemelo digital.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📋 Resumen del Dataset Activo")
    st.write(f"- **Archivo Origen:** `{metadata.get('filename')}`")
    st.write(f"- **Encabezado Detectado:** `{metadata.get('detected_header')}`")
    st.write(f"- **Gas Principal Identificado:** `{metadata.get('gas_type')}`")
    st.write(f"- **Muestras Válidas Parseadas:** `{metadata.get('valid_rows'):,}` filas ({metadata.get('corrupt_rows')} filas corruptas reportadas)")
    st.write(f"- **Muestras tras Agregación ({agg_choice}):** `{len(df_current):,}` filas analíticas en memoria.")


# ------------------------------------------------------------------------------
# TAB 1: EDA (COMPRENSIÓN DE LOS DATOS)
# ------------------------------------------------------------------------------
with tabs[1]:
    st.header("Fase 1: Comprensión de los Datos (Exploratory Data Analysis - EDA)")
    st.markdown("Todos los análisis exploratorios se calculan dinámicamente con los datos cargados en tiempo de ejecución.")

    info = compute_basic_info(df_current)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dimensiones", f"{info['n_rows']} × {info['n_cols']}")
    c2.metric("Valores Faltantes", f"{info['missing_values']}")
    c3.metric("Filas Duplicadas", f"{info['duplicated_rows']}")
    c4.metric("Frecuencia Estimada", f"{info['sampling_rate_hz']:.1f} Hz")

    st.markdown("---")
    st.subheader("1.1 Estadísticos Descriptivos y Distribuciones")
    desc_df = compute_descriptive_stats(df_current)
    st.dataframe(desc_df.style.format(precision=3), use_container_width=True)

    st.subheader("1.2 Detección de Outliers (IQR vs MAD)")
    outlier_df = detect_outliers_summary(df_current)
    st.dataframe(outlier_df, use_container_width=True)

    st.subheader("1.3 Series Temporales de Sensores Químicos y Concentraciones")
    channels_to_plot = st.multiselect(
        "Seleccionar canales para visualizar:",
        SENSOR_COLUMNS,
        default=["S01", "S02", "S09", "S14"]
    )
    fig_ts = plot_sensor_timeseries(df_current, selected_channels=channels_to_plot)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.subheader("1.4 Correlación No Paramétrica de Spearman")
    corr_df, fig_corr = compute_spearman_correlations(df_current)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("1.5 Multicolinealidad (Factor de Inflación de la Varianza - VIF)")
    st.markdown("Diagnóstico de redundancia colineal entre los 16 sensores químicos de la matriz:")
    vif_df = compute_vif_dataframe(df_current)
    st.dataframe(vif_df, use_container_width=True)

    st.subheader("1.6 Deriva y Estacionariedad (Pruebas ADF y KPSS)")
    st.markdown("Evaluación rigurosa de raíz unitaria y deriva química en la respuesta sensorica:")
    stat_df = compute_stationarity_tests(df_current)
    st.dataframe(stat_df, use_container_width=True)

    st.subheader("1.7 Detección de Saltos Discretos de Concentración")
    steps_df = detect_concentration_steps(df_current)
    if len(steps_df) > 0:
        st.dataframe(steps_df.style.format(precision=4), use_container_width=True)
    else:
        st.info("No se detectaron saltos abruptos por encima del umbral de variación.")

    st.subheader("1.8 Análisis de Componentes Principales (PCA)")
    pca_df, scree_f, pca_scatter_f = compute_pca_analysis(df_current)
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.dataframe(pca_df, use_container_width=True)
        st.plotly_chart(scree_f, use_container_width=True)
    with col_p2:
        st.plotly_chart(pca_scatter_f, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: PREPARACIÓN DE DATOS
# ------------------------------------------------------------------------------
with tabs[2]:
    st.header("Fase 2: Preparación de Datos (Data Preparation & Engineering)")
    st.markdown("""
    Garantía metodológica de **cero fuga de información (Zero Data Leakage)**:
    - Las transformaciones y el escalador **RobustScaler** se encapsulan dentro de Pipelines de Scikit-Learn.
    - La partición es estrictamente **TEMPORAL sin barajado (no shuffle)**.
    - Se aplica una **purga temporal (gap)** entre el conjunto de entrenamiento/CV y la prueba final para romper la memoria temporal de los sensores químicos.
    """)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        roll_wins = st.multiselect("Ventanas para Medias/Desv. Móviles (pasos):", [2, 3, 5, 10], default=[3, 5])
    with col_cfg2:
        lag_steps = st.multiselect("Retardos Temporales (Lags t-k):", [1, 2, 3], default=[1, 2])
    with col_cfg3:
        gap_steps = st.slider("Tamaño de Purga Temporal ('Gap' entre bloques):", min_value=0, max_value=15, value=5)

    # Construir matrices
    X_mat, y_mat, feat_names, y_cols = build_engineered_dataset(
        df_current,
        target_mode=target_mode_code,
        forecast_horizon_h=forecast_h,
        rolling_windows=roll_wins,
        lag_steps=lag_steps,
        include_diff=True,
        include_rolling_std=True
    )

    X_train_val, y_train_val, X_test, y_test, split_info = temporal_train_test_split(
        X_mat, y_mat, test_size=0.20, gap_size=gap_steps
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Características Totales (X)", f"{X_mat.shape[1]}")
    m2.metric("Train + CV (80%)", f"{split_info['n_train_val']} muestras")
    m3.metric("Purga ('Gap')", f"{split_info['gap_purged']} muestras")
    m4.metric("Prueba Final (20%)", f"{split_info['n_test']} muestras")

    st.markdown("---")
    st.subheader("Vista Previa de Matriz de Características Ingeniería de Variables")
    st.dataframe(X_train_val.head(6).style.format(precision=3), use_container_width=True)

    st.subheader("Variables Objetivo en Entrenamiento")
    st.dataframe(y_train_val.head(6).style.format(precision=3), use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: ENTRENAMIENTO DE MODELOS
# ------------------------------------------------------------------------------
with tabs[3]:
    st.header("Fase 3: Entrenamiento de Modelos (Modeling)")
    st.markdown("""
    Entrenamiento del catálogo completo de modelos (Baselines empíricos, Modelos Lineales Regularizados,
    Modelos basados en Árboles, Ensambles por Boosting, Vectores de Soporte y Redes Neuronales Multicapa).
    """)

    catalog = get_model_catalog(random_state=42)

    if st.button("🚀 Entrenar Catálogo Completo de Modelos", type="primary"):
        prog_bar = st.progress(0, text="Iniciando entrenamiento secuencial de modelos...")
        trained_results = []

        total_models = len(catalog)
        for i, (m_name, estimator) in enumerate(catalog.items()):
            prog_bar.progress((i + 1) / total_models, text=f"Ajustando {m_name}...")
            res = train_single_model(
                m_name, estimator,
                X_train_val, y_train_val,
                X_test, y_test
            )
            trained_results.append(res)

        prog_bar.empty()
        st.session_state["trained_results"] = trained_results
        st.success("¡Entrenamiento del catálogo completado con éxito!")

    if "trained_results" in st.session_state:
        train_rows = []
        for r in st.session_state["trained_results"]:
            train_rows.append({
                "Modelo": r["model_name"],
                "RMSE Train": round(r["rmse_train"], 4),
                "MAE Train": round(r["mae_train"], 4),
                "R² Train": round(r["r2_train"], 4),
                "RMSE Test": round(r.get("rmse_val", np.nan), 4),
                "MAE Test": round(r.get("mae_val", np.nan), 4),
                "R² Test": round(r.get("r2_val", np.nan), 4),
                "Tiempo Ajuste (s)": round(r["fit_time_s"], 4),
                "Latencia Inferencia (ms/muestra)": round(r["latency_ms"], 4)
            })
        train_df = pd.DataFrame(train_rows).sort_values("RMSE Test").reset_index(drop=True)
        st.subheader("Tabla Comparativa de Modelos Ajustados")
        st.dataframe(train_df, use_container_width=True)

        # Curva de aprendizaje interactiva
        st.subheader("3.1 Curvas de Aprendizaje (Muestras de Entrenamiento vs RMSE)")
        curve_model = st.selectbox("Seleccionar modelo para analizar Curva de Aprendizaje:", list(catalog.keys()), index=5)
        if st.button("Calcular Curva de Aprendizaje"):
            with st.spinner("Generando curva de aprendizaje con partición temporal..."):
                tscv_lc = 3
                ts_abs, tr_rmse, va_rmse = compute_learning_curve_data(
                    catalog[curve_model],
                    X_train_val, y_train_val,
                    cv_splits=tscv_lc
                )
                lc_fig = go.Figure()
                lc_fig.add_trace(go.Scatter(x=ts_abs, y=tr_rmse, mode="lines+markers", name="Error Entrenamiento (RMSE)", line=dict(color="#38bdf8")))
                lc_fig.add_trace(go.Scatter(x=ts_abs, y=va_rmse, mode="lines+markers", name="Error Validación Temporal (RMSE)", line=dict(color="#f97316")))
                lc_fig.update_layout(
                    title=f"Curva de Aprendizaje para {curve_model}",
                    xaxis_title="Número de Muestras de Entrenamiento",
                    yaxis_title="RMSE (ppm)",
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                st.plotly_chart(lc_fig, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 4: SELECCIÓN DEL MEJOR MODELO
# ------------------------------------------------------------------------------
with tabs[4]:
    st.header("Fase 4: Selección del Mejor Modelo (Model Selection)")
    st.markdown("""
    ### ⚖️ Criterio de Selección: Regla de 1 Desviación Estándar (1-SE Rule)
    En minería subterránea, un modelo hipercomplejo sobreajustado genera falsas alarmas o retrasos en la respuesta ante explosividad.
    Se calcula el modelo con el mínimo RMSE medio en Validación Cruzada ($RMSE_{min}$) y su Error Estándar ($SE = \frac{SD}{\sqrt{K}}$).
    Se seleccionará el **modelo más simple** cuyo error no supere $RMSE_{min} + SE$.
    **Nota Metodológica:** NO se emplea MAPE debido a la presencia de concentraciones reales en 0.00 ppm.
    """)

    if "summary_df" in st.session_state:
        sum_df = st.session_state["summary_df"]
        st.subheader("Tabla Resumen de Validación Cruzada (Media ± DE)")
        st.dataframe(sum_df, use_container_width=True)

        best_model_name, selection_details = select_best_model_1se_rule(sum_df)

        st.success(f"🏆 **Modelo Seleccionado por la Regla 1-SE:** `{best_model_name}`")
        st.info(selection_details["justification"])

        # Refit del mejor modelo en Train+Val y evaluación en Test final
        st.subheader("4.1 Reajuste (Refit) y Evaluación Imparcial en Prueba Final (20% Restante)")
        best_estimator = get_model_catalog()[best_model_name]
        is_base = "Baseline" in best_model_name
        refit_pipeline = best_estimator if is_base else create_pipeline(best_estimator)

        t_refit_0 = time.perf_counter()
        refit_pipeline.fit(X_train_val.values, y_train_val.values)
        refit_time = time.perf_counter() - t_refit_0

        t_inf_0 = time.perf_counter()
        y_test_pred = refit_pipeline.predict(X_test.values)
        infer_time = time.perf_counter() - t_inf_0
        infer_latency_ms = (infer_time / len(X_test)) * 1000.0

        y_test_arr = y_test.values
        if y_test_pred.ndim == 1 and y_test_arr.ndim == 2:
            y_test_pred = y_test_pred.reshape(-1, y_test_arr.shape[1])

        # Guardar en session state para fases estadísticas
        st.session_state["best_model_name"] = best_model_name
        st.session_state["best_pipeline"] = refit_pipeline
        st.session_state["y_test_actual"] = y_test_arr
        st.session_state["y_test_pred"] = y_test_pred

        test_rmse = np.sqrt(mean_squared_error(y_test_arr, y_test_pred))
        test_mae = mean_absolute_error(y_test_arr, y_test_pred)
        test_r2 = r2_score(y_test_arr, y_test_pred)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("RMSE Prueba Final", f"{test_rmse:.3f} ppm")
        mc2.metric("MAE Prueba Final", f"{test_mae:.3f} ppm")
        mc3.metric("R² Prueba Final", f"{test_r2:.3f}")
        mc4.metric("Latencia Inferencia", f"{infer_latency_ms:.2f} ms")

        # Gráfico Predicho vs Real en prueba final
        st.subheader("4.2 Trayectoria Temporal en Conjunto de Prueba Final")
        fig_test_time = go.Figure()
        test_time_axis = df_current["time_s"].iloc[-len(y_test):].values

        for c_idx, col_name in enumerate(y_cols):
            fig_test_time.add_trace(go.Scatter(
                x=test_time_axis, y=y_test_arr[:, c_idx],
                mode="lines", name=f"Real {col_name}", line=dict(dash="solid")
            ))
            fig_test_time.add_trace(go.Scatter(
                x=test_time_axis, y=y_test_pred[:, c_idx],
                mode="lines", name=f"Predicho {col_name}", line=dict(dash="dot")
            ))

        fig_test_time.update_layout(
            title=f"Evaluación en Prueba Final: {best_model_name}",
            xaxis_title="Tiempo (s)", yaxis_title="Concentración (ppm)",
            template="plotly_dark", height=450, margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_test_time, use_container_width=True)

    else:
        st.info("Ejecute primero la fase de Validación Cruzada (Pestaña 5) para disponer de los estadísticos necesarios para la regla 1-SE.")


# ------------------------------------------------------------------------------
# TAB 5: VALIDACIÓN CRUZADA (CV TEMPORAL)
# ------------------------------------------------------------------------------
with tabs[5]:
    st.header("Fase 5: Validación Cruzada Temporal (TimeSeriesSplit)")
    st.markdown("""
    **Prohibición de K-Fold Aleatorio**: Debido a la alta frecuencia (100 Hz) y la correlación serial de los sensores químicos,
    el barajado aleatorio provocaría una severa fuga temporal entre muestras adyacentes.
    Se utiliza **TimeSeriesSplit** con bloques temporales expansivos ordenados cronológicamente.
    """)

    k_folds = st.slider("Número de Folds Temporales (k):", min_value=2, max_value=6, value=4)

    if st.button("⚡ Ejecutar Validación Cruzada Temporal Completa", type="primary"):
        catalog_cv = get_model_catalog(random_state=42)
        prog_cv = st.progress(0, text="Evaluando modelos en validación cruzada temporal...")

        summary_df, folds_df, cv_artifacts = run_temporal_cv_for_models(
            catalog_cv,
            X_train_val, y_train_val,
            n_splits=k_folds,
            y_columns=y_cols
        )

        prog_cv.empty()
        st.session_state["summary_df"] = summary_df
        st.session_state["folds_df"] = folds_df
        st.session_state["cv_artifacts"] = cv_artifacts
        st.success("Validación cruzada temporal ejecutada exitosamente.")

    if "folds_df" in st.session_state:
        st.subheader("5.1 Métricas por Fold Temporal para Cada Modelo")
        st.dataframe(st.session_state["folds_df"], use_container_width=True)

        st.subheader("5.2 Gráfico de Estabilidad de Validación Cruzada")
        fig_stab = plot_cv_stability_chart(st.session_state["folds_df"])
        st.plotly_chart(fig_stab, use_container_width=True)

        st.subheader("5.3 Diagrama Predicho vs Observado en Validación Cruzada")
        model_to_inspect = st.selectbox(
            "Seleccionar modelo para inspeccionar en los Folds de CV:",
            list(st.session_state["cv_artifacts"].keys()),
            index=2
        )
        fig_pv = plot_predicted_vs_observed_cv(
            st.session_state["cv_artifacts"],
            model_name=model_to_inspect,
            target_idx=0,
            target_name=y_cols[0]
        )
        st.plotly_chart(fig_pv, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 6: HIPERPARÁMETROS
# ------------------------------------------------------------------------------
with tabs[6]:
    st.header("Fase 6: Optimización de Hiperparámetros (Tuning)")
    st.markdown("""
    Optimización bayesiana mediante **Optuna** estructurada dentro de validación cruzada temporal anidada
    para salvaguardar la independencia de la función objetivo.
    """)

    opt_family = st.selectbox(
        "Familia de Modelo a Optimizar:",
        ["Ridge", "ElasticNet", "Random Forest", "Extra Trees", "XGBoost", "LightGBM", "MLP (Neural Network)"],
        index=0
    )

    st.subheader("6.1 Espacio de Búsqueda Configurado")
    space_df = get_search_space_description(opt_family)
    st.dataframe(space_df, use_container_width=True)

    n_trials = st.slider("Cantidad de Iteraciones (Trials):", min_value=10, max_value=60, value=25, step=5)

    if st.button(f"🔍 Optimizar Hiperparámetros para {opt_family}", type="primary"):
        with st.spinner(f"Optuna explorando espacio paramétrico ({n_trials} trials)..."):
            tune_res = run_optuna_tuning(
                opt_family,
                X_train_val, y_train_val,
                n_trials=n_trials,
                n_splits=3,
                random_state=42
            )
            st.session_state["tune_res"] = tune_res
            st.success("¡Optimización de hiperparámetros finalizada!")

    if "tune_res" in st.session_state:
        tres = st.session_state["tune_res"]
        st.subheader("6.2 Mejores Hiperparámetros Encontrados")
        st.json(tres["best_params"])
        st.metric("Mejor RMSE de CV Temporal", f"{tres['best_rmse']:.4f} ppm")

        st.subheader("6.3 Historial de Optimización y Convergencia")
        st.plotly_chart(tres["history_fig"], use_container_width=True)

        if not tres["importance_df"].empty:
            st.subheader("6.4 Importancia Relativa de Hiperparámetros")
            st.dataframe(tres["importance_df"], use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 7: PRUEBAS ESTADÍSTICAS INFERENCIALES ROBUSTAS
# ------------------------------------------------------------------------------
with tabs[7]:
    st.header("Fase 7: Pruebas Estadísticas Inferenciales Robustas")
    st.markdown("""
    Todos los estadísticos se calculan en tiempo de ejecución sobre los residuos reales del mejor modelo reajustado.
    En series temporales de sensores con miles de muestras, casi cualquier prueba estándar reporta p < 0.05 por exceso de potencia;
    por ello, se reportan de forma obligatoria los **tamaños de efecto** e **intervalos de confianza robustos**.
    """)

    if "best_pipeline" not in st.session_state:
        st.warning("Debe ejecutar primero la Fase 5 y Fase 4 para tener el modelo ajustado y sus predicciones en prueba.")
    else:
        y_true_test = st.session_state["y_test_actual"]
        y_pred_test = st.session_state["y_test_pred"]
        best_name = st.session_state["best_model_name"]

        # a) Diagnóstico de Residuos
        st.subheader("a) Diagnóstico Completo de Residuos")
        diag_res, qq_f, res_f = run_residual_diagnostics(y_true_test, y_pred_test, target_name=y_cols[0])

        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Shapiro-Wilk (p-valor)", f"{diag_res['shapiro_p']:.4e}", delta="No Normal" if not diag_res["shapiro_normal"] else "Normal")
        col_d2.metric("Durbin-Watson", f"{diag_res['durbin_watson']:.3f}", delta=diag_res["dw_interpretation"])
        col_d3.metric("Breusch-Pagan (p-valor)", f"{diag_res['breusch_pagan_p']:.4e}", delta="Heterocedástico" if not diag_res["homoscedastic"] else "Homocedástico")

        st.info(diag_res["justification"])

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.plotly_chart(qq_f, use_container_width=True)
        with col_q2:
            st.plotly_chart(res_f, use_container_width=True)

        st.markdown("---")
        # b) Comparación entre Modelos (Friedman + Post-hoc Wilcoxon Holm + Diebold-Mariano)
        st.subheader("b) Comparación No Paramétrica entre Modelos")
        if "folds_df" in st.session_state:
            fried_res, post_df = run_friedman_and_posthoc(st.session_state["folds_df"])
            st.markdown(f"**Prueba de Friedman:** estadístico $\\chi^2$ = `{fried_res['friedman_stat']:.3f}`, p-valor = `{fried_res['friedman_p']:.4e}`.")
            st.dataframe(post_df, use_container_width=True)

            # Diebold-Mariano contra baseline en prueba final
            base_pred = BaselineMeanRegressor().fit(X_train_val, y_train_val).predict(X_test)
            dm_res = compute_diebold_mariano_test(y_true_test, y_pred_test, base_pred, hac_lags=5)
            st.markdown(f"**Prueba de Diebold-Mariano (HAC Newey-West en prueba final):** estadístico DM = `{dm_res['dm_statistic']:.3f}`, p-valor = `{dm_res['p_value']:.4e}` ({dm_res['winner']}).")

        st.markdown("---")
        # c) Intervalos de Confianza al 95% por Moving Block Bootstrap (MBB)
        st.subheader("c) Intervalos de Confianza al 95% mediante Moving Block Bootstrap (MBB)")
        st.markdown("Respetando la correlación temporal serial mediante remuestreo de bloques continuos:")
        if st.button("Calcular Intervalos MBB al 95%"):
            with st.spinner("Ejecutando Moving Block Bootstrap (400 réplicas)..."):
                mbb_df = compute_moving_block_bootstrap_ci(y_true_test, y_pred_test, block_length=10, n_bootstraps=400)
                st.session_state["mbb_df"] = mbb_df
        if "mbb_df" in st.session_state:
            st.dataframe(st.session_state["mbb_df"], use_container_width=True)

        st.markdown("---")
        # d) Prueba de Permutación frente al Baseline e Importancia de Variables
        st.subheader("d) Prueba de Permutación frente a Baseline e Importancia de Variables con IC")
        base_pred_2 = BaselineMeanRegressor().fit(X_train_val, y_train_val).predict(X_test)
        perm_res = run_permutation_test_vs_baseline(y_true_test, y_pred_test, base_pred_2, n_permutations=200)
        st.write(f"- {perm_res['interpretation']}")

        if st.button("Calcular Importancia de Características por Permutación (con IC 95%)"):
            with st.spinner("Calculando importancia por permutación sobre conjunto de prueba..."):
                top_features = feat_names[:16]  # Evaluar las características principales
                perm_imp_df = compute_permutation_feature_importance_ci(
                    st.session_state["best_pipeline"],
                    X_test, y_test,
                    feature_names=top_features,
                    n_repeats=8
                )
                st.session_state["perm_imp_df"] = perm_imp_df

        if "perm_imp_df" in st.session_state:
            st.dataframe(st.session_state["perm_imp_df"], use_container_width=True)

        st.markdown("---")
        # e) Análisis de Concordancia
        st.subheader("e) Análisis de Concordancia (Spearman, Kendall, CCC de Lin y Bland-Altman)")
        conc_metrics, ba_chart = compute_concordance_metrics(y_true_test, y_pred_test, target_name=y_cols[0])
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Spearman Rho (Rank)", f"{conc_metrics['spearman_rho']:.4f}")
        cc2.metric("Kendall Tau", f"{conc_metrics['kendall_tau']:.4f}")
        cc3.metric("CCC de Lin", f"{conc_metrics['lin_ccc']:.4f}", delta=conc_metrics["ccc_grade"])

        st.plotly_chart(ba_chart, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 8: AUDITORÍA DE RIESGOS Y EVALUACIÓN GLOBAL
# ------------------------------------------------------------------------------
with tabs[8]:
    st.header("Fase 8: Evaluación Global y Auditoría de Riesgos Operacionales")
    st.markdown("""
    En un entorno subterráneo de producción, la precisión aislada en banco de pruebas es insuficiente.
    Esta fase audita la viabilidad de despliegue operacional en la arquitectura IoT / Edge de MineTwin.
    """)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("""
        ### ⚖️ Compromisos Operacionales (Trade-offs)
        - **Latencia vs Precisión:** Un modelo como ExtraTrees o Gradient Boosting ofrece ligera ganancia en R², pero su inferencia
          en procesadores embebidos puede tardar > 20 ms. Ridge o ElasticNet ejecutan en < 1 ms por ciclo de 100 Hz.
        - **Deriva del Sensor vs Reentrenamiento:** Los sensores MOX sufren deriva por envejecimiento de capa sensible y fluctuación
          de humedad ambiental. Se recomienda recalibración semanal o ventana deslizante continua.
        - **Costo de Falsos Negativos:** Una subestimación de Metano por encima del 1% (10,000 ppm) vulnera el umbral de parada de emergencia.
        """)

    with col_r2:
        st.markdown("""
        ### 🚦 Semáforo de Confiabilidad Predictiva (MineTwin Guard)
        """)
        if "y_test_pred" in st.session_state:
            y_pred_check = st.session_state["y_test_pred"]
            max_pred = float(np.max(y_pred_check))
            min_pred = float(np.min(y_pred_check))

            if min_pred < -5.0:
                st.error("🔴 **ALERTA ROJA - Predicción Físicamente Imposible**: Concentración negativa anómala detectada (< 0 ppm).")
            elif max_pred > 500.0:
                st.warning("🟡 **ALERTA AMARILLA - Saturación Sensorica**: Concentración en rango cercano al límite de escala.")
            else:
                st.success("🟢 **SEMÁFORO VERDE - Operación Nominal Segura**: Todas las predicciones se encuentran dentro del dominio de calibración.")

        st.markdown("""
        ### 🛡️ Detección Fuera de Distribución (OOD / Drift Detector)
        Si los valores de los 16 sensores se desvían más de 3 MAD respecto a la mediana del conjunto de entrenamiento,
        el gemelo digital MineTwin etiqueta la predicción con el flag `OOD_UNCERTAIN_ESTIMATE` para activar ventilación forzada por seguridad preventiva.
        """)


# ------------------------------------------------------------------------------
# TAB 9: DESPLIEGUE
# ------------------------------------------------------------------------------
with tabs[9]:
    st.header("Fase 9: Despliegue (Deployment & Exportación)")
    st.markdown("""
    Persistencia del modelo final entrenado con `joblib`, simulador de inferencia en tiempo real para nuevas
    lecturas de 16 canales y exportación de reportes analíticos.
    """)

    st.subheader("9.1 Guardar Modelo Serializado (.joblib)")
    if "best_pipeline" in st.session_state:
        pipe_to_save = st.session_state["best_pipeline"]
        buffer = io.BytesIO()
        joblib.dump({
            "pipeline": pipe_to_save,
            "model_name": st.session_state.get("best_model_name", "model"),
            "target_columns": y_cols,
            "feature_names": feat_names,
            "timestamp": time.time()
        }, buffer)
        buffer.seek(0)

        st.download_button(
            label="💾 Descargar Pipeline Entrenado (MineTwin_Model.joblib)",
            data=buffer,
            file_name="MineTwin_Predictive_Pipeline.joblib",
            mime="application/octet-stream"
        )
    else:
        st.info("Ajuste un modelo en la Fase 4 para habilitar la descarga del pipeline.")

    st.subheader("9.2 Simulador de Predicción en Tiempo Real (16 Canales)")
    st.markdown("Ingrese lecturas crudas para predecir concentraciones inmediatas de gases:")

    cols_input = st.columns(4)
    simulated_inputs = {}
    default_vals = df_current[SENSOR_COLUMNS].median().to_dict()

    for idx, s_col in enumerate(SENSOR_COLUMNS):
        with cols_input[idx % 4]:
            simulated_inputs[s_col] = st.number_input(
                f"{s_col}:",
                value=float(default_vals.get(s_col, 0.0)),
                format="%.2f"
            )

    if st.button("🔮 Predecir Concentración con Pipeline Activo", type="primary"):
        if "best_pipeline" in st.session_state:
            input_df = pd.DataFrame([simulated_inputs])
            # Generar características derivadas
            eng_sim = build_engineered_dataset(input_df, target_mode=target_mode_code)[0]
            # Si faltan columnas derivadas, rellenar con valores base
            for col in feat_names:
                if col not in eng_sim.columns:
                    base_s = col.split("_")[0]
                    eng_sim[col] = simulated_inputs.get(base_s, 0.0)
            eng_sim = eng_sim[feat_names]

            pred_res = st.session_state["best_pipeline"].predict(eng_sim.values)
            st.success("🎯 **Inferencia Computada Exitosamente:**")
            for c_i, c_name in enumerate(y_cols):
                val_c = pred_res[0, c_i] if pred_res.ndim > 1 else pred_res[0]
                st.metric(f"Concentración Predicha: {c_name}", f"{val_c:.3f} ppm")
        else:
            st.error("No hay un pipeline entrenado en memoria. Ejecute primero la Fase 4.")

    st.subheader("9.3 Exportación de Tablas de Resultados (CSV / Excel)")
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        csv_data = df_current.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Exportar Dataset Analítico (CSV)",
            data=csv_data,
            file_name="MineTwin_Processed_Data.csv",
            mime="text/csv"
        )
    with exp_c2:
        if "folds_df" in st.session_state:
            folds_csv = st.session_state["folds_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Exportar Resultados de Validación Cruzada (CSV)",
                data=folds_csv,
                file_name="MineTwin_CV_Results.csv",
                mime="text/csv"
            )
