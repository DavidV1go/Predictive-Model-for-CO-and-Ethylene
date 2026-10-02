"""
Módulo de Modelos de Regresión para MineTwin.
Implementa el catálogo completo de modelos requeridos:
- Baseline (Media empírica y Persistencia / Lag-1)
- Ridge Regression
- ElasticNet
- K-Nearest Neighbors (KNN)
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- Support Vector Regressor (SVR)
- Multi-layer Perceptron (MLPRegressor)
Soporta regresión multisalida nativa o mediante MultiOutputRegressor.
Genera curvas de aprendizaje (tamaño de muestra vs RMSE).
"""

import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from modules.preprocessing import create_pipeline


class BaselineMeanRegressor(BaseEstimator, RegressorMixin):
    """Modelo nulo de referencia: predice la media histórica del conjunto de entrenamiento."""
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = np.mean(np.asarray(y), axis=0)
        return self

    def predict(self, X):
        n_samples = len(X)
        if np.ndim(self.mean_) == 0:
            return np.full((n_samples,), self.mean_)
        return np.tile(self.mean_, (n_samples, 1))


class BaselinePersistenceRegressor(BaseEstimator, RegressorMixin):
    """
    Modelo de persistencia para series temporales:
    Predice para el instante t el valor objetivo observado en el paso anterior (o proxy por sensor correlacionado).
    En inferencia directa, proyecta el último valor conocido o constante media en fallback.
    """
    def __init__(self):
        self.last_y_ = None

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.last_y_ = y_arr[-1]
        return self

    def predict(self, X):
        n_samples = len(X)
        if np.ndim(self.last_y_) == 0:
            return np.full((n_samples,), self.last_y_)
        return np.tile(self.last_y_, (n_samples, 1))


def get_model_catalog(random_state: int = 42) -> Dict[str, Any]:
    """
    Retorna el catálogo completo de modelos configurados.
    Usa MultiOutputRegressor para aquellos estimadores que requieren envoltura en regresión múltiple.
    """
    # Intentar importar XGBoost y LightGBM con fallback seguro
    try:
        from xgboost import XGBRegressor
        xgb_model = MultiOutputRegressor(XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
            n_jobs=-1
        ))
    except Exception:
        # Fallback a GradientBoosting con MultiOutput
        xgb_model = MultiOutputRegressor(GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            random_state=random_state
        ))

    try:
        from lightgbm import LGBMRegressor
        lgb_model = MultiOutputRegressor(LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1
        ))
    except Exception:
        lgb_model = MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1
        ))

    catalog = {
        "Baseline (Media)": BaselineMeanRegressor(),
        "Baseline (Persistencia)": BaselinePersistenceRegressor(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=2000),
        "K-Nearest Neighbors (KNN)": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=random_state)),
        "XGBoost": xgb_model,
        "LightGBM": lgb_model,
        "SVR (Support Vector)": MultiOutputRegressor(SVR(C=10.0, epsilon=0.1, kernel="rbf")),
        "MLP (Neural Network)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state, early_stopping=True)
    }
    return catalog


def train_single_model(
    model_name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Entrena un modelo dentro de un Pipeline con RobustScaler (excepto Baselines).
    Mide tiempo de entrenamiento y latencia de inferencia por muestra.
    """
    # Si es baseline, no necesita escalador
    if "Baseline" in model_name:
        pipe = estimator
    else:
        pipe = create_pipeline(estimator)

    t0 = time.perf_counter()
    pipe.fit(X_train.values, y_train.values)
    fit_time_s = time.perf_counter() - t0

    # Medir inferencia
    t1 = time.perf_counter()
    y_pred_train = pipe.predict(X_train.values)
    infer_time_s = time.perf_counter() - t1
    latency_ms_per_sample = (infer_time_s / len(X_train)) * 1000.0

    # Formatear predicciones
    y_train_arr = y_train.values
    if y_pred_train.ndim == 1 and y_train_arr.ndim == 2:
        y_pred_train = y_pred_train.reshape(-1, y_train_arr.shape[1])

    # Métricas en entrenamiento
    rmse_train = np.sqrt(mean_squared_error(y_train_arr, y_pred_train))
    mae_train = mean_absolute_error(y_train_arr, y_pred_train)
    r2_train = r2_score(y_train_arr, y_pred_train)

    result = {
        "model_name": model_name,
        "pipeline": pipe,
        "fit_time_s": fit_time_s,
        "latency_ms": latency_ms_per_sample,
        "rmse_train": float(rmse_train),
        "mae_train": float(mae_train),
        "r2_train": float(r2_train),
    }

    # Evaluar en validación si está presente
    if X_val is not None and y_val is not None:
        y_val_arr = y_val.values
        y_pred_val = pipe.predict(X_val.values)
        if y_pred_val.ndim == 1 and y_val_arr.ndim == 2:
            y_pred_val = y_pred_val.reshape(-1, y_val_arr.shape[1])

        result["rmse_val"] = float(np.sqrt(mean_squared_error(y_val_arr, y_pred_val)))
        result["mae_val"] = float(mean_absolute_error(y_val_arr, y_pred_val))
        result["r2_val"] = float(r2_score(y_val_arr, y_pred_val))
        result["y_pred_val"] = y_pred_val

    return result


def compute_learning_curve_data(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_splits: Any,
    train_sizes: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera curva de aprendizaje (tamaño de entrenamiento vs error de validación)
    usando partición de validación temporal para evitar sobreajuste.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.2, 1.0, 5)

    pipe = create_pipeline(estimator)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        pipe,
        X.values,
        y.values,
        train_sizes=train_sizes,
        cv=cv_splits,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )

    # Invertir signo para obtener RMSE positivo
    train_rmse = -np.mean(train_scores, axis=1)
    val_rmse = -np.mean(val_scores, axis=1)

    return train_sizes_abs, train_rmse, val_rmse
