"""
Módulo de Preprocesamiento e Ingeniería de Variables para MineTwin.
- Extracción de características temporales: medias móviles, desv. móviles, diferencias y retardos.
- Soporte para horizonte de pronóstico h configurable.
- Partición temporal sin shuffle (80% Train+CV / 20% Prueba Final).
- Purga temporal ("gap") entre bloques para eliminar fuga de autocorrelación.
- RobustScaler estructurado para uso exclusivo dentro de Pipelines de Scikit-Learn.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from modules.io_data import SENSOR_COLUMNS


class SensorFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador Scikit-Learn para ingeniería de características de series temporales de sensores:
    - Medias móviles (rolling mean)
    - Desviaciones estándar móviles (rolling std)
    - Diferencias de primer orden (velocidad de adsorción)
    - Retardos (lags)
    """
    def __init__(
        self,
        rolling_windows: List[int] = None,
        lag_steps: List[int] = None,
        compute_diff: bool = True,
        compute_rolling_std: bool = True
    ):
        self.rolling_windows = rolling_windows or [3, 5]
        self.lag_steps = lag_steps or [1, 2]
        self.compute_diff = compute_diff
        self.compute_rolling_std = compute_rolling_std
        self.feature_names_out_: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Acepta DataFrame o ndarray numérico de 16 canales.
        Retorna matriz aumentada de características numéricas.
        """
        if isinstance(X, np.ndarray):
            cols = [f"S{i+1:02d}" for i in range(X.shape[1])]
            df_in = pd.DataFrame(X, columns=cols)
        else:
            df_in = X.copy()

        feature_dfs = [df_in]
        out_names = list(df_in.columns)

        # 1. Medias móviles
        for w in self.rolling_windows:
            rolled = df_in.rolling(window=w, min_periods=1).mean()
            rolled.columns = [f"{c}_roll_mean_{w}" for c in df_in.columns]
            feature_dfs.append(rolled)
            out_names.extend(rolled.columns)

        # 2. Desviaciones móviles
        if self.compute_rolling_std:
            for w in self.rolling_windows:
                rolled_std = df_in.rolling(window=w, min_periods=1).std().fillna(0.0)
                rolled_std.columns = [f"{c}_roll_std_{w}" for c in df_in.columns]
                feature_dfs.append(rolled_std)
                out_names.extend(rolled_std.columns)

        # 3. Diferencias de primer orden
        if self.compute_diff:
            diff_df = df_in.diff(periods=1).fillna(0.0)
            diff_df.columns = [f"{c}_diff1" for c in df_in.columns]
            feature_dfs.append(diff_df)
            out_names.extend(diff_df.columns)

        # 4. Retardos (lags)
        for lag in self.lag_steps:
            lag_df = df_in.shift(periods=lag).bfill().fillna(0.0)
            lag_df.columns = [f"{c}_lag_{lag}" for c in df_in.columns]
            feature_dfs.append(lag_df)
            out_names.extend(lag_df.columns)

        X_engineered = pd.concat(feature_dfs, axis=1)
        self.feature_names_out_ = out_names
        return X_engineered.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


def build_engineered_dataset(
    df: pd.DataFrame,
    target_mode: str = "multisalida",
    forecast_horizon_h: int = 0,
    rolling_windows: List[int] = None,
    lag_steps: List[int] = None,
    include_diff: bool = True,
    include_rolling_std: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Construye las matrices de características X y objetivo(s) y a partir del DataFrame cargado.

    Parámetros:
        - target_mode: "multisalida" [gas_ppm, ethylene_ppm], "gas_ppm", o "ethylene_ppm"
        - forecast_horizon_h: retardo a futuro (t + h). Si h > 0, predice la concentración a h pasos adelante.
    """
    if rolling_windows is None:
        rolling_windows = [3, 5]
    if lag_steps is None:
        lag_steps = [1, 2]

    # Extraer canales base
    channels = [c for c in SENSOR_COLUMNS if c in df.columns]
    X_base = df[channels].copy()

    # Generar características derivadas
    engineer = SensorFeatureEngineer(
        rolling_windows=rolling_windows,
        lag_steps=lag_steps,
        compute_diff=include_diff,
        compute_rolling_std=include_rolling_std
    )
    X_arr = engineer.transform(X_base)
    feat_names = engineer.feature_names_out_
    X_df = pd.DataFrame(X_arr, columns=feat_names, index=df.index)

    # Determinar variables objetivo
    if target_mode == "gas_ppm":
        y_cols = ["gas_ppm"]
    elif target_mode == "ethylene_ppm":
        y_cols = ["ethylene_ppm"]
    else:
        # Multisalida
        y_cols = [c for c in ["gas_ppm", "ethylene_ppm"] if c in df.columns]

    y_df = df[y_cols].copy()

    # Si se especificó pronóstico a futuro horizon_h
    if forecast_horizon_h > 0:
        y_df = y_df.shift(-forecast_horizon_h)
        # Recortar filas finales inválidas por el shift
        valid_idx = y_df.dropna().index
        X_df = X_df.loc[valid_idx]
        y_df = y_df.loc[valid_idx]

    return X_df, y_df, feat_names, y_cols


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.20,
    gap_size: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Partición TEMPORAL estricta (sin shuffle):
    - Entrenamiento + Validación interna: primeros (1 - test_size)
    - Purga temporal (gap_size): filas descartadas para evitar filtración por memoria de sensor
    - Prueba Final: último test_size (nunca visto en entrenamiento)
    """
    n_samples = len(X)
    if n_samples < 20:
        gap_size = 0

    n_test = int(np.floor(n_samples * test_size))
    split_point_end = n_samples - n_test
    split_point_train = max(0, split_point_end - gap_size)

    X_train_val = X.iloc[:split_point_train].copy()
    y_train_val = y.iloc[:split_point_train].copy()

    X_test = X.iloc[split_point_end:].copy()
    y_test = y.iloc[split_point_end:].copy()

    info = {
        "n_total": n_samples,
        "n_train_val": len(X_train_val),
        "n_test": len(X_test),
        "gap_purged": gap_size,
        "train_val_pct": round(100 * len(X_train_val) / n_samples, 1),
        "test_pct": round(100 * len(X_test) / n_samples, 1)
    }

    return X_train_val, y_train_val, X_test, y_test, info


def create_pipeline(estimator: Any) -> Pipeline:
    """
    Crea un Pipeline de Scikit-Learn con RobustScaler para evitar fuga de información.
    """
    return Pipeline([
        ("scaler", RobustScaler(quantile_range=(25.0, 75.0))),
        ("regressor", estimator)
    ])
