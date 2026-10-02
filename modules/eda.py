"""
Módulo de Análisis Exploratorio de Datos (EDA) para MineTwin.
Calcula en tiempo de ejecución estadísticas descriptivas, distribuciones, outliers (IQR/MAD),
series de tiempo, correlaciones de Spearman, VIF, pruebas ADF/KPSS, saltos de concentración y PCA.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor

from modules.io_data import SENSOR_COLUMNS


def compute_basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula dimensiones, tipos de datos, valores nulos y filas duplicadas.
    """
    missing_count = int(df.isnull().sum().sum())
    dup_count = int(df.duplicated().sum())

    type_counts = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": missing_count,
        "duplicated_rows": dup_count,
        "dtypes": type_counts,
        "duration_seconds": float(df["time_s"].max() - df["time_s"].min()) if "time_s" in df else 0.0,
        "sampling_rate_hz": float(1.0 / np.median(np.diff(df["time_s"].values))) if len(df) > 1 and "time_s" in df else 0.0
    }


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula media, DE, mediana, IQR, min, max, asimetría (skewness) y curtosis.
    """
    numeric_cols = [c for c in df.columns if c != "time_s"]
    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        # MAD: Median Absolute Deviation
        med = float(series.median())
        mad = float(np.median(np.abs(series.values - med)))

        rows.append({
            "Canal / Variable": col,
            "Media": float(series.mean()),
            "Desv. Est.": float(series.std()),
            "Mínimo": float(series.min()),
            "Q25": q25,
            "Mediana": med,
            "Q75": q75,
            "Máximo": float(series.max()),
            "IQR": iqr,
            "MAD": mad,
            "Asimetría (Skew)": float(series.skew()),
            "Curtosis": float(series.kurt())
        })
    return pd.DataFrame(rows)


def detect_outliers_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica outliers por dos criterios robustos:
    1. IQR: valores < Q1 - 1.5*IQR o > Q3 + 1.5*IQR
    2. MAD: valores con |x - med| > 3 * (1.4826 * MAD)
    """
    cols = SENSOR_COLUMNS
    results = []
    n_total = len(df)

    for col in cols:
        vals = df[col].values
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25
        lower_iqr = q25 - 1.5 * iqr
        upper_iqr = q75 + 1.5 * iqr
        out_iqr = int(np.sum((vals < lower_iqr) | (vals > upper_iqr)))

        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        normal_mad = 1.4826 * mad if mad > 1e-9 else 1e-9
        out_mad = int(np.sum(np.abs(vals - med) > 3 * normal_mad))

        results.append({
            "Sensor": col,
            "Outliers IQR (n)": out_iqr,
            "Outliers IQR (%)": round(100.0 * out_iqr / n_total, 2) if n_total > 0 else 0,
            "Outliers MAD (n)": out_mad,
            "Outliers MAD (%)": round(100.0 * out_mad / n_total, 2) if n_total > 0 else 0,
            "Rango Válido IQR": f"[{lower_iqr:.2f}, {upper_iqr:.2f}]",
            "Mediana ± 3*MAD": f"[{med - 3*normal_mad:.2f}, {med + 3*normal_mad:.2f}]"
        })
    return pd.DataFrame(results)


def plot_sensor_timeseries(df: pd.DataFrame, selected_channels: List[str] = None) -> go.Figure:
    """
    Gráfico interactivo de series temporales de sensores con concentraciones de gases.
    """
    if selected_channels is None or len(selected_channels) == 0:
        selected_channels = ["S01", "S02", "S09", "S14"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Lecturas Crudas de Sensores Químicos", "Concentración de Gases (ppm)")
    )

    t = df["time_s"]
    for ch in selected_channels:
        if ch in df.columns:
            fig.add_trace(
                go.Scatter(x=t, y=df[ch], mode="lines", name=ch),
                row=1, col=1
            )

    if "gas_ppm" in df.columns:
        fig.add_trace(
            go.Scatter(x=t, y=df["gas_ppm"], mode="lines", name="Gas Objetivo (ppm)", line=dict(color="#f97316", width=2)),
            row=2, col=1
        )
    if "ethylene_ppm" in df.columns:
        fig.add_trace(
            go.Scatter(x=t, y=df["ethylene_ppm"], mode="lines", name="Etileno (ppm)", line=dict(color="#10b981", width=2)),
            row=2, col=1
        )

    fig.update_layout(
        height=550,
        hovermode="x unified",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Tiempo (segundos)", row=2, col=1)
    fig.update_yaxes(title_text="Respuesta Cruda", row=1, col=1)
    fig.update_yaxes(title_text="Concentración (ppm)", row=2, col=1)
    return fig


def compute_spearman_correlations(df: pd.DataFrame) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Calcula matriz de correlación no paramétrica de Spearman entre canales y contra gases.
    """
    cols = [c for c in df.columns if c != "time_s"]
    corr_matrix = df[cols].corr(method="spearman").round(3)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Matriz de Correlación de Spearman (Canales S01-S16 y Concentraciones)"
    )
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return corr_matrix, fig


def compute_vif_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el Factor de Inflación de la Varianza (VIF) para diagnosticar multicolinealidad.
    """
    X = df[SENSOR_COLUMNS].copy()
    # Eliminar columnas con varianza cero si existieran
    std = X.std()
    valid_cols = std[std > 1e-8].index.tolist()
    X = X[valid_cols]

    vif_data = []
    # Normalizar para estabilidad numérica en cálculo VIF
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    X_norm["constante"] = 1.0

    for i, col in enumerate(valid_cols):
        try:
            vif_val = variance_inflation_factor(X_norm.values, i)
        except Exception:
            vif_val = np.nan
        vif_data.append({
            "Canal": col,
            "VIF": round(float(vif_val), 2) if np.isfinite(vif_val) else "> 10,000",
            "Nivel Multicolinealidad": "Crítico (> 10)" if (np.isfinite(vif_val) and vif_val > 10) or not np.isfinite(vif_val) else ("Moderado (5-10)" if vif_val > 5 else "Bajo (< 5)")
        })

    return pd.DataFrame(vif_data)


def compute_stationarity_tests(df: pd.DataFrame, max_channels: int = 16) -> pd.DataFrame:
    """
    Evalúa deriva y estacionariedad mediante:
    - ADF (Dickey-Fuller Aumentado): H0 = Serie no estacionaria (raíz unitaria).
    - KPSS: H0 = Serie estacionaria en torno a la media.
    """
    results = []
    channels = SENSOR_COLUMNS[:max_channels]

    for ch in channels:
        series = df[ch].dropna().values
        if len(series) < 20 or np.std(series) < 1e-8:
            results.append({
                "Canal": ch,
                "ADF Estadístico": 0.0,
                "ADF p-valor": 1.0,
                "ADF Estacionaria?": "No (cte)",
                "KPSS Estadístico": 0.0,
                "KPSS p-valor": 1.0,
                "KPSS Estacionaria?": "No",
                "Diagnóstico Final": "Constante / Incierto"
            })
            continue

        # Prueba ADF
        try:
            adf_res = adfuller(series, maxlag=5, autolag="AIC")
            adf_stat, adf_p = float(adf_res[0]), float(adf_res[1])
            adf_stationary = adf_p < 0.05
        except Exception:
            adf_stat, adf_p, adf_stationary = np.nan, 1.0, False

        # Prueba KPSS
        try:
            kpss_res = kpss(series, regression="c", nlags="auto")
            kpss_stat, kpss_p = float(kpss_res[0]), float(kpss_res[1])
            kpss_stationary = kpss_p >= 0.05
        except Exception:
            kpss_stat, kpss_p, kpss_stationary = np.nan, 0.0, False

        if adf_stationary and kpss_stationary:
            diag = "Estacionaria Estricta"
        elif adf_stationary and not kpss_stationary:
            diag = "Estacionaria en Diferencias"
        elif not adf_stationary and kpss_stationary:
            diag = "Estacionaria en Tendencia"
        else:
            diag = "No Estacionaria (Deriva química persistente)"

        results.append({
            "Canal": ch,
            "ADF Estadístico": round(adf_stat, 3) if np.isfinite(adf_stat) else "N/A",
            "ADF p-valor": f"{adf_p:.4f}" if np.isfinite(adf_p) else "N/A",
            "ADF Estacionaria?": "Sí (p < 0.05)" if adf_stationary else "No",
            "KPSS Estadístico": round(kpss_stat, 3) if np.isfinite(kpss_stat) else "N/A",
            "KPSS p-valor": f"{kpss_p:.4f}" if np.isfinite(kpss_p) else "N/A",
            "KPSS Estacionaria?": "Sí (p ≥ 0.05)" if kpss_stationary else "No",
            "Diagnóstico Final": diag
        })

    return pd.DataFrame(results)


def detect_concentration_steps(df: pd.DataFrame, threshold: float = 1e-4) -> pd.DataFrame:
    """
    Detecta saltos o escalones discretos en las concentraciones de gas.
    """
    steps = []
    for gas_col in ["gas_ppm", "ethylene_ppm"]:
        if gas_col in df.columns:
            diffs = np.abs(np.diff(df[gas_col].values))
            jump_indices = np.where(diffs > threshold)[0] + 1
            for idx in jump_indices:
                steps.append({
                    "Variable": gas_col,
                    "Tiempo (s)": float(df["time_s"].iloc[idx]),
                    "Valor Previo (ppm)": float(df[gas_col].iloc[idx - 1]),
                    "Nuevo Valor (ppm)": float(df[gas_col].iloc[idx]),
                    "Magnitud Salto (ppm)": float(df[gas_col].iloc[idx] - df[gas_col].iloc[idx - 1])
                })
    return pd.DataFrame(steps)


def compute_pca_analysis(df: pd.DataFrame, n_components: int = 3) -> Tuple[pd.DataFrame, go.Figure, go.Figure]:
    """
    Aplica Análisis de Componentes Principales (PCA) sobre los 16 sensores.
    Retorna tabla de varianza explicada, gráfico de scree plot y dispersión 2D/3D.
    """
    X = df[SENSOR_COLUMNS].values
    # Estandarización robusta para PCA
    med = np.median(X, axis=0)
    iqr = stats.iqr(X, axis=0)
    iqr[iqr < 1e-6] = 1.0
    X_scaled = (X - med) / iqr

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)

    pca_summary = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(n_components)],
        "Varianza Explicada (%)": np.round(var_ratio * 100, 2),
        "Varianza Acumulada (%)": np.round(cum_var * 100, 2)
    })

    # Scree plot
    scree_fig = go.Figure()
    scree_fig.add_trace(go.Bar(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=var_ratio * 100,
        name="Varianza Individual (%)",
        marker_color="#0284c7"
    ))
    scree_fig.add_trace(go.Scatter(
        x=[f"PC{i+1}" for i in range(n_components)],
        y=cum_var * 100,
        name="Varianza Acumulada (%)",
        mode="lines+markers",
        line=dict(color="#f97316", width=2)
    ))
    scree_fig.update_layout(
        title="Scree Plot de Componentes Principales",
        template="plotly_dark",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    # Dispersión 2D coloreada por concentración de gas
    color_var = df["gas_ppm"] if "gas_ppm" in df.columns else df["time_s"]
    color_name = "Gas ppm" if "gas_ppm" in df.columns else "Tiempo (s)"

    scatter_fig = px.scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        color=color_var,
        labels={"x": f"PC1 ({var_ratio[0]*100:.1f}%)", "y": f"PC2 ({var_ratio[1]*100:.1f}%)", "color": color_name},
        title="Proyección 2D de Sensores en Espacio Latente (PC1 vs PC2)",
        color_continuous_scale="Viridis"
    )
    scatter_fig.update_layout(template="plotly_dark", height=420, margin=dict(l=40, r=40, t=50, b=40))

    return pca_summary, scree_fig, scatter_fig
