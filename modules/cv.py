"""
Módulo de Validación Cruzada Temporal (CV) para MineTwin.
- Utiliza TimeSeriesSplit / Bloques Temporales (NUNCA K-Fold aleatorio debido a la alta autocorrelación).
- k folds configurables.
- Métricas desagregadas por cada fold y por cada salida [gas_ppm, ethylene_ppm].
- Implementa la Regla de Una Desviación Estándar (1-SE Rule) para seleccionar el modelo más parsimonioso.
- Genera gráficos de estabilidad por fold y diagramas de Predicho vs Observado.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

from modules.preprocessing import create_pipeline


# Jerarquía de complejidad de modelos (de más simple a más complejo)
MODEL_COMPLEXITY_RANK = {
    "Baseline (Media)": 0,
    "Baseline (Persistencia)": 1,
    "Ridge": 2,
    "ElasticNet": 3,
    "K-Nearest Neighbors (KNN)": 4,
    "SVR (Support Vector)": 5,
    "MLP (Neural Network)": 6,
    "Random Forest": 7,
    "Extra Trees": 8,
    "Gradient Boosting": 9,
    "LightGBM": 10,
    "XGBoost": 11
}


def run_temporal_cv_for_models(
    models_dict: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 4,
    y_columns: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta validación cruzada TimeSeriesSplit para todos los modelos del catálogo.

    Retorna:
        - summary_df: Tabla agregada con Media ± DE de RMSE, MAE, R² y tiempos.
        - folds_df: Detalle de métricas fold a fold para cada modelo.
        - cv_artifacts: Diccionario con predicciones y observaciones por fold para trazabilidad.
    """
    if y_columns is None:
        y_columns = list(y.columns)

    # Limitar n_splits según tamaño de muestra
    max_possible_splits = max(2, len(X) // 30)
    n_splits = min(n_splits, max_possible_splits)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_records = []
    cv_artifacts = {}

    for model_name, estimator in models_dict.items():
        cv_artifacts[model_name] = []
        is_baseline = "Baseline" in model_name

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_val_actual = X.iloc[val_idx], y.iloc[val_idx]

            # Entrenar en pipeline
            pipe = estimator if is_baseline else create_pipeline(estimator)
            pipe.fit(X_tr.values, y_tr.values)

            # Predecir
            y_pred = pipe.predict(X_va.values)
            y_val_arr = y_val_actual.values
            if y_pred.ndim == 1 and y_val_arr.ndim == 2:
                y_pred = y_pred.reshape(-1, y_val_arr.shape[1])

            # Métricas globales del fold
            rmse_fold = float(np.sqrt(mean_squared_error(y_val_arr, y_pred)))
            mae_fold = float(mean_absolute_error(y_val_arr, y_pred))
            r2_fold = float(r2_score(y_val_arr, y_pred))

            rec = {
                "Modelo": model_name,
                "Fold": fold_idx + 1,
                "N_Train": len(train_idx),
                "N_Val": len(val_idx),
                "RMSE": rmse_fold,
                "MAE": mae_fold,
                "R2": r2_fold,
            }

            # Métricas individuales por salida
            for col_i, col_name in enumerate(y_columns):
                y_true_col = y_val_arr[:, col_i]
                y_pred_col = y_pred[:, col_i]
                rec[f"RMSE_{col_name}"] = float(np.sqrt(mean_squared_error(y_true_col, y_pred_col)))
                rec[f"MAE_{col_name}"] = float(mean_absolute_error(y_true_col, y_pred_col))
                rec[f"R2_{col_name}"] = float(r2_score(y_true_col, y_pred_col))

            fold_records.append(rec)
            cv_artifacts[model_name].append({
                "fold": fold_idx + 1,
                "y_true": y_val_arr,
                "y_pred": y_pred,
                "val_indices": val_idx
            })

    folds_df = pd.DataFrame(fold_records)

    # Calcular resumen agregado (Media ± DE)
    summary_rows = []
    for model_name in models_dict.keys():
        sub = folds_df[folds_df["Modelo"] == model_name]
        k_folds = len(sub)
        rmse_mean = sub["RMSE"].mean()
        rmse_std = sub["RMSE"].std() if k_folds > 1 else 0.0
        rmse_se = rmse_std / np.sqrt(k_folds) if k_folds > 1 else 0.0

        mae_mean = sub["MAE"].mean()
        mae_std = sub["MAE"].std() if k_folds > 1 else 0.0

        r2_mean = sub["R2"].mean()
        r2_std = sub["R2"].std() if k_folds > 1 else 0.0

        row = {
            "Modelo": model_name,
            "RMSE CV (Media ± DE)": f"{rmse_mean:.3f} ± {rmse_std:.3f}",
            "MAE CV (Media ± DE)": f"{mae_mean:.3f} ± {mae_std:.3f}",
            "R² CV (Media ± DE)": f"{r2_mean:.3f} ± {r2_std:.3f}",
            "RMSE_Mean": float(rmse_mean),
            "RMSE_Std": float(rmse_std),
            "RMSE_SE": float(rmse_se),
            "MAE_Mean": float(mae_mean),
            "R2_Mean": float(r2_mean),
            "Complejidad": MODEL_COMPLEXITY_RANK.get(model_name, 99)
        }

        # Desglose por salida
        for col_name in y_columns:
            if f"RMSE_{col_name}" in sub.columns:
                m_rmse_col = sub[f"RMSE_{col_name}"].mean()
                s_rmse_col = sub[f"RMSE_{col_name}"].std() if k_folds > 1 else 0.0
                row[f"RMSE_{col_name}"] = f"{m_rmse_col:.3f} ± {s_rmse_col:.3f}"

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE_Mean").reset_index(drop=True)
    return summary_df, folds_df, cv_artifacts


def select_best_model_1se_rule(
    summary_df: pd.DataFrame
) -> Tuple[str, Dict[str, Any]]:
    """
    Selecciona el mejor modelo aplicando la Regla de Una Desviación Estándar (1-SE Rule):
    1. Encuentra el modelo con el mínimo RMSE medio: RMSE_min y su SE = SD / sqrt(K).
    2. Define el umbral: Umbral = RMSE_min + SE.
    3. Entre todos los modelos cuyo RMSE medio <= Umbral, selecciona el de MENOR complejidad.
    """
    # Encontrar modelo con menor RMSE
    best_raw_idx = summary_df["RMSE_Mean"].idxmin()
    best_raw = summary_df.loc[best_raw_idx]

    rmse_min = best_raw["RMSE_Mean"]
    se_val = best_raw["RMSE_SE"]
    threshold_1se = rmse_min + se_val

    # Candidatos dentro de 1 SE
    candidates = summary_df[summary_df["RMSE_Mean"] <= threshold_1se].copy()

    # Ordenar candidatos por menor complejidad y luego menor RMSE
    candidates = candidates.sort_values(by=["Complejidad", "RMSE_Mean"]).reset_index(drop=True)
    selected_model_row = candidates.iloc[0]
    selected_name = selected_model_row["Modelo"]

    details = {
        "best_raw_model": best_raw["Modelo"],
        "best_raw_rmse": float(rmse_min),
        "best_raw_se": float(se_val),
        "threshold_1se": float(threshold_1se),
        "selected_model": selected_name,
        "selected_rmse": float(selected_model_row["RMSE_Mean"]),
        "selected_complexity": int(selected_model_row["Complejidad"]),
        "candidates_count": len(candidates),
        "justification": (
            f"El modelo '{selected_name}' fue seleccionado aplicando la Regla de 1 Desviación Estándar. "
            f"Su RMSE medio ({selected_model_row['RMSE_Mean']:.3f}) se encuentra dentro del intervalo de tolerancia "
            f"[{rmse_min:.3f}, {threshold_1se:.3f}] del modelo con menor error absoluto ('{best_raw['Modelo']}'), "
            f"pero posee una arquitectura más parsimoniosa (complejidad {selected_model_row['Complejidad']} vs {best_raw['Complejidad']}), "
            f"lo que reduce drásticamente el riesgo de sobreajuste y la variabilidad en campo."
        )
    }

    return selected_name, details


def plot_cv_stability_chart(folds_df: pd.DataFrame) -> go.Figure:
    """
    Gráfico interactivo de estabilidad de RMSE fold a fold para cada modelo.
    """
    fig = px.line(
        folds_df,
        x="Fold",
        y="RMSE",
        color="Modelo",
        markers=True,
        title="Estabilidad de Validación Cruzada Temporal (RMSE por Fold)",
        labels={"RMSE": "RMSE de Validación", "Fold": "Fold Temporal"}
    )
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_predicted_vs_observed_cv(
    cv_artifacts: Dict[str, Any],
    model_name: str,
    target_idx: int = 0,
    target_name: str = "Gas (ppm)"
) -> go.Figure:
    """
    Gráfico de dispersión y recta identidad Predicho vs Observado sobre todos los folds de CV.
    """
    if model_name not in cv_artifacts or len(cv_artifacts[model_name]) == 0:
        return go.Figure()

    all_true, all_pred, fold_tags = [], [], []
    for art in cv_artifacts[model_name]:
        f_true = art["y_true"][:, target_idx] if art["y_true"].ndim > 1 else art["y_true"]
        f_pred = art["y_pred"][:, target_idx] if art["y_pred"].ndim > 1 else art["y_pred"]
        all_true.extend(f_true)
        all_pred.extend(f_pred)
        fold_tags.extend([f"Fold {art['fold']}"] * len(f_true))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    min_val = min(float(np.min(all_true)), float(np.min(all_pred)))
    max_val = max(float(np.max(all_true)), float(np.max(all_pred)))

    fig = go.Figure()
    # Puntos por fold
    for fold_label in sorted(list(set(fold_tags))):
        mask = [tag == fold_label for tag in fold_tags]
        fig.add_trace(go.Scatter(
            x=all_true[mask],
            y=all_pred[mask],
            mode="markers",
            name=fold_label,
            marker=dict(size=7, opacity=0.8)
        ))

    # Línea ideal y=x
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Ideal (y = x)",
        line=dict(color="#ef4444", dash="dash", width=2)
    ))

    fig.update_layout(
        title=f"Predicho vs Observado en Validación Cruzada: {model_name} [{target_name}]",
        xaxis_title="Observado Real (ppm)",
        yaxis_title="Predicho por Modelo (ppm)",
        template="plotly_dark",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
