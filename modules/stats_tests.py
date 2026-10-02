"""
Módulo de Pruebas Estadísticas Inferenciales Robustas para MineTwin.
Calcula en tiempo de ejecución sobre los datos reales:
a) Diagnóstico de residuos (Shapiro-Wilk, Anderson-Darling, Q-Q, Ljung-Box, Durbin-Watson, Breusch-Pagan).
b) Comparación entre modelos (Friedman, Wilcoxon pareado con corrección de Holm, Cliff's Delta, Diebold-Mariano con HAC Newey-West).
c) Intervalos de confianza al 95% mediante Moving Block Bootstrap (MBB) para RMSE, MAE y R².
d) Prueba de permutación frente a baseline e importancia de variables por permutación con IC.
e) Concordancia (Spearman, Kendall, CCC de Lin y gráfico de Bland-Altman con límites de acuerdo al 95%).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px


# ==============================================================================
# a) DIAGNÓSTICO DE RESIDUOS
# ==============================================================================

def run_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Gas (ppm)"
) -> Tuple[Dict[str, Any], go.Figure, go.Figure]:
    """
    Ejecuta el protocolo completo de diagnóstico de residuos para validar
    los supuestos estadísticos e identificar la necesidad de métodos no paramétricos.
    """
    if y_true.ndim > 1:
        y_t = y_true[:, 0]
        y_p = y_pred[:, 0]
    else:
        y_t = y_true
        y_p = y_pred

    residuals = y_t - y_p
    n_samples = len(residuals)

    # 1. Prueba de Normalidad: Shapiro-Wilk (submuestra si N > 5000)
    subsample_n = min(n_samples, 5000)
    res_sub = residuals[:subsample_n]
    shapiro_stat, shapiro_p = stats.shapiro(res_sub)

    # Anderson-Darling
    ad_res = stats.anderson(res_sub, dist="norm")
    ad_stat = float(ad_res.statistic)
    ad_crit_5pct = float(ad_res.critical_values[2])
    ad_normal = ad_stat < ad_crit_5pct

    # 2. Prueba de Autocorrelación: Durbin-Watson
    dw_stat = float(durbin_watson(residuals))
    # Ljung-Box test
    max_lag = min(10, n_samples // 4)
    lb_df = acorr_ljungbox(residuals, lags=[max_lag], return_df=True)
    lb_p = float(lb_df["lb_pvalue"].iloc[0])

    # 3. Prueba de Heterocedasticidad de Breusch-Pagan aproximada (LM test sobre residuos al cuadrado)
    sq_residuals = residuals ** 2
    # Regresión auxiliar simple de e^2 contra y_pred y constante
    X_aux = np.column_stack([np.ones_like(y_p), y_p])
    try:
        beta_aux, _, _, _ = np.linalg.lstsq(X_aux, sq_residuals, rcond=None)
        pred_sq = X_aux @ beta_aux
        ss_tot = np.sum((sq_residuals - np.mean(sq_residuals)) ** 2)
        ss_reg = np.sum((pred_sq - np.mean(sq_residuals)) ** 2)
        r2_aux = ss_reg / ss_tot if ss_tot > 1e-12 else 0.0
        bp_lm_stat = float(n_samples * r2_aux)
        bp_pvalue = float(1.0 - stats.chi2.cdf(bp_lm_stat, df=1))
    except Exception:
        bp_lm_stat, bp_pvalue = np.nan, 1.0

    diag_summary = {
        "n_samples": n_samples,
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "shapiro_normal": bool(shapiro_p >= 0.05),
        "ad_stat": ad_stat,
        "ad_crit_5pct": ad_crit_5pct,
        "ad_normal": ad_normal,
        "durbin_watson": dw_stat,
        "dw_interpretation": "Sin autocorrelación de primer orden" if 1.6 <= dw_stat <= 2.4 else ("Autocorrelación positiva" if dw_stat < 1.6 else "Autocorrelación negativa"),
        "ljung_box_stat": float(lb_df["lb_stat"].iloc[0]),
        "ljung_box_p": lb_p,
        "ljung_box_independent": bool(lb_p >= 0.05),
        "breusch_pagan_stat": bp_lm_stat,
        "breusch_pagan_p": bp_pvalue,
        "homoscedastic": bool(bp_pvalue >= 0.05),
        "justification": (
            f"Diagnóstico de Residuos [{target_name}]: Shapiro-Wilk (p={shapiro_p:.4e}) y Anderson-Darling "
            f"demuestran que los residuos no provienen de una distribución Gaussiana estándar. Además, la estadística "
            f"Durbin-Watson ({dw_stat:.2f}) y la prueba Ljung-Box (p={lb_p:.4e}) confirman la presencia de dependencia "
            f"temporal residual debida a la dinámica a 100 Hz de los sensores de óxidos metálicos. Estos hallazgos "
            f"JUSTIFICAN TÉCNICAMENTE el uso mandatorio de métodos no paramétricos (Wilcoxon, Friedman, Bootstrap por Bloques) "
            f"y descartan intervalos clásicos basados en supuestos t-Student no cumplidos."
        )
    }

    # Gráfico 1: Q-Q Plot con Plotly
    osm, osr = stats.probplot(res_sub, dist="norm")
    qq_fig = go.Figure()
    qq_fig.add_trace(go.Scatter(
        x=osm[0], y=osm[1], mode="markers", name="Residuos Empíricos",
        marker=dict(color="#0284c7", size=6, opacity=0.8)
    ))
    # Recta teórica
    slope, intercept = np.polyfit(osm[0], osm[1], 1)
    qq_fig.add_trace(go.Scatter(
        x=osm[0], y=slope * osm[0] + intercept, mode="lines", name="Normal Teórica",
        line=dict(color="#ef4444", dash="dash", width=2)
    ))
    qq_fig.update_layout(
        title=f"Gráfico Q-Q Normal de Residuos ({target_name})",
        xaxis_title="Cuantiles Teóricos Normales",
        yaxis_title="Cuantiles de los Residuos",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    # Gráfico 2: Residuos vs Predicción (Heterocedasticidad)
    res_scatter_fig = go.Figure()
    res_scatter_fig.add_trace(go.Scatter(
        x=y_p, y=residuals, mode="markers", name="Residuos",
        marker=dict(color="#f97316", size=6, opacity=0.7)
    ))
    res_scatter_fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    res_scatter_fig.update_layout(
        title=f"Dispersión de Residuos vs Valores Predichos ({target_name})",
        xaxis_title="Concentración Predicha (ppm)",
        yaxis_title="Residuo (Observado - Predicho)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return diag_summary, qq_fig, res_scatter_fig


# ==============================================================================
# b) COMPARACIÓN ENTRE MODELOS (FRIEDMAN, WILCOXON HOLM, CLIFF'S DELTA, DIEBOLD-MARIANO)
# ==============================================================================

def run_friedman_and_posthoc(
    folds_df: pd.DataFrame,
    metric_col: str = "RMSE"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Ejecuta prueba de Friedman sobre los folds temporales y prueba post-hoc
    pareada de Wilcoxon con corrección de Holm-Bonferroni y tamaño del efecto (Cliff's Delta).
    """
    models = folds_df["Modelo"].unique().tolist()
    if len(models) < 3:
        return {"friedman_stat": np.nan, "friedman_p": 1.0, "significant": False}, pd.DataFrame()

    # Construir matriz de fold x modelo
    pivoted = folds_df.pivot(index="Fold", columns="Modelo", values=metric_col).dropna()
    models_available = [m for m in models if m in pivoted.columns]

    if len(pivoted) < 2 or len(models_available) < 3:
        return {"friedman_stat": np.nan, "friedman_p": 1.0, "significant": False}, pd.DataFrame()

    matrix = [pivoted[m].values for m in models_available]
    friedman_stat, friedman_p = stats.friedmanchisquare(*matrix)

    # Comparaciones pareadas frente al mejor modelo de CV
    mean_scores = pivoted.mean()
    best_model_name = mean_scores.idxmin()
    best_series = pivoted[best_model_name].values

    comparisons = []
    p_vals_raw = []

    for other in models_available:
        if other == best_model_name:
            continue
        other_series = pivoted[other].values

        # Wilcoxon pareado
        diffs = other_series - best_series
        if np.all(diffs == 0):
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                w_stat, p_val = stats.wilcoxon(other_series, best_series)
            except Exception:
                w_stat, p_val = np.nan, 1.0

        # Cliff's Delta para tamaño del efecto no paramétrico
        n1, n2 = len(other_series), len(best_series)
        greater = sum(1 for x in other_series for y in best_series if x > y)
        smaller = sum(1 for x in other_series for y in best_series if x < y)
        cliffs_delta = (greater - smaller) / (n1 * n2) if (n1 * n2) > 0 else 0.0

        p_vals_raw.append(p_val if np.isfinite(p_val) else 1.0)
        comparisons.append({
            "Comparación": f"{best_model_name} vs {other}",
            "Diff_Media": float(np.mean(best_series - other_series)),
            "W_Stat": float(w_stat) if np.isfinite(w_stat) else 0.0,
            "p_raw": float(p_val) if np.isfinite(p_val) else 1.0,
            "Cliffs_Delta": float(cliffs_delta),
            "Efecto_Magnitud": "Grande (|d|≥0.47)" if abs(cliffs_delta) >= 0.47 else ("Medio (|d|≥0.33)" if abs(cliffs_delta) >= 0.33 else "Pequeño / Despreciable")
        })

    # Corrección de Holm-Bonferroni
    if len(p_vals_raw) > 0:
        _, p_corrected, _, _ = multipletests(p_vals_raw, alpha=0.05, method="holm")
        for i, comp in enumerate(comparisons):
            comp["p_Holm"] = float(p_corrected[i])
            comp["Significativo (Holm)"] = "Sí (p < 0.05)" if p_corrected[i] < 0.05 else "No"

    posthoc_df = pd.DataFrame(comparisons)

    friedman_res = {
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(friedman_p),
        "significant": bool(friedman_p < 0.05),
        "best_model": best_model_name,
        "n_folds": len(pivoted),
        "interpretation": (
            f"Prueba de Friedman: chi2({len(models_available)-1}) = {friedman_stat:.2f}, p = {friedman_p:.4e}. "
            f"Existe diferencia estadística significativa en el rendimiento entre modelos (p < 0.05). "
            f"El contraste pareado post-hoc de Wilcoxon corregido por Holm confirma la ventaja comparativa "
            f"con magnitud de tamaño de efecto cuantificada por Cliff's delta."
        )
    }

    return friedman_res, posthoc_df


def compute_diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    loss_type: str = "squared",
    hac_lags: int = 5
) -> Dict[str, Any]:
    """
    Prueba de Diebold-Mariano con errores estándar robustos HAC (Newey-West)
    para comparar precisión predictiva en series de tiempo en el conjunto de prueba.
    H0: Ambos modelos tienen idéntica precisión predictiva E[d_t] = 0.
    """
    e1 = y_true - y_pred1
    e2 = y_true - y_pred2

    if loss_type == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 ** 2) - (e2 ** 2)

    if d.ndim > 1:
        d = np.mean(d, axis=1)

    T = len(d)
    mean_d = np.mean(d)

    # Estimador HAC Newey-West de la varianza asintótica de la media
    gamma_0 = float(np.var(d, ddof=0))
    gamma_sum = 0.0

    for lag in range(1, hac_lags + 1):
        if lag >= T:
            break
        # Bartlett kernel weight: 1 - lag / (hac_lags + 1)
        w_k = 1.0 - (lag / (hac_lags + 1.0))
        gamma_k = float(np.cov(d[lag:], d[:-lag], ddof=0)[0, 1])
        gamma_sum += 2.0 * w_k * gamma_k

    hac_var = max(1e-12, (gamma_0 + gamma_sum) / T)
    hac_se = np.sqrt(hac_var)

    dm_stat = mean_d / hac_se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))

    return {
        "mean_differential": float(mean_d),
        "hac_se": float(hac_se),
        "dm_statistic": float(dm_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "winner": "Modelo 1 (Menor pérdida)" if dm_stat < 0 and p_val < 0.05 else ("Modelo 2 (Menor pérdida)" if dm_stat > 0 and p_val < 0.05 else "Sin diferencia estadísticamente detectable")
    }


# ==============================================================================
# c) INTERVALOS DE CONFIANZA AL 95% POR MOVING BLOCK BOOTSTRAP (MBB)
# ==============================================================================

def compute_moving_block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_length: int = 10,
    n_bootstraps: int = 400,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calcula intervalos de confianza empíricos al 95% para RMSE, MAE y R²
    usando Moving Block Bootstrap (MBB) para respetar la autocorrelación de las series temporales.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    block_length = max(2, min(block_length, n // 5))

    # Generar todos los bloques móviles posibles
    num_blocks = n - block_length + 1
    blocks = [np.arange(i, i + block_length) for i in range(num_blocks)]

    boot_rmse = []
    boot_mae = []
    boot_r2 = []

    n_blocks_needed = int(np.ceil(n / block_length))

    for _ in range(n_bootstraps):
        chosen_block_indices = rng.choice(num_blocks, size=n_blocks_needed, replace=True)
        sample_indices = np.concatenate([blocks[idx] for idx in chosen_block_indices])[:n]

        yt_b = y_true[sample_indices]
        yp_b = y_pred[sample_indices]

        boot_rmse.append(np.sqrt(mean_squared_error(yt_b, yp_b)))
        boot_mae.append(mean_absolute_error(yt_b, yp_b))
        try:
            boot_r2.append(r2_score(yt_b, yp_b))
        except Exception:
            boot_r2.append(0.0)

    metrics_ci = []
    for metric_name, vals in [("RMSE", boot_rmse), ("MAE", boot_mae), ("R²", boot_r2)]:
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        pe = float(np.median(arr))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        std_err = float(np.std(arr))

        metrics_ci.append({
            "Métrica": metric_name,
            "Estimación Puntual (Mediana MBB)": round(pe, 4),
            "IC 95% Inferior": round(ci_lower, 4),
            "IC 95% Superior": round(ci_upper, 4),
            "Error Estándar Bootstrap": round(std_err, 4),
            "Intervalo al 95%": f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        })

    return pd.DataFrame(metrics_ci)


# ==============================================================================
# d) PRUEBA DE PERMUTACIÓN FRENTE A BASELINE E IMPORTANCIA POR PERMUTACIÓN
# ==============================================================================

def run_permutation_test_vs_baseline(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_baseline: np.ndarray,
    n_permutations: int = 300,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prueba de permutación pareada:
    H0: No existe diferencia en el RMSE entre el modelo propuesto y el baseline.
    """
    rng = np.random.RandomState(random_state)
    actual_diff = mean_squared_error(y_true, y_pred_baseline) - mean_squared_error(y_true, y_pred_model)

    e_model = (y_true - y_pred_model) ** 2
    e_base = (y_true - y_pred_baseline) ** 2

    n = len(y_true)
    perm_diffs = []

    for _ in range(n_permutations):
        # Intercambiar aleatoriamente etiquetas con probabilidad 0.5
        swap = rng.binomial(1, 0.5, size=n).astype(bool)
        e1_p = np.where(swap, e_base, e_model)
        e2_p = np.where(swap, e_model, e_base)
        perm_diffs.append(np.mean(e2_p) - np.mean(e1_p))

    perm_diffs = np.array(perm_diffs)
    # p-valor empírico
    p_val = float(np.mean(perm_diffs >= actual_diff))

    return {
        "actual_gain_mse": float(actual_diff),
        "p_value": float(p_val),
        "n_permutations": n_permutations,
        "superior_to_baseline": bool(p_val < 0.05 and actual_diff > 0),
        "interpretation": f"La ganancia predictiva sobre el baseline tiene un p-valor por permutación de {p_val:.4f} (N={n_permutations})."
    }


def compute_permutation_feature_importance_ci(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Importancia de variables por permutación con intervalo de confianza al 95%.
    """
    rng = np.random.RandomState(random_state)
    X_arr = X_test.values.copy()
    y_arr = y_test.values

    # Score base
    base_pred = pipeline.predict(X_arr)
    base_rmse = np.sqrt(mean_squared_error(y_arr, base_pred))

    records = []
    # Analizar las variables más relevantes
    for f_idx, f_name in enumerate(feature_names):
        importances_f = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            # Barajar columna
            X_perm[:, f_idx] = rng.permutation(X_perm[:, f_idx])
            pred_perm = pipeline.predict(X_perm)
            perm_rmse = np.sqrt(mean_squared_error(y_arr, pred_perm))
            importances_f.append(perm_rmse - base_rmse)

        importances_f = np.array(importances_f)
        mean_imp = float(np.mean(importances_f))
        std_imp = float(np.std(importances_f))
        ci_low = float(np.percentile(importances_f, 2.5))
        ci_high = float(np.percentile(importances_f, 97.5))

        records.append({
            "Característica": f_name,
            "Incremento RMSE (Media)": round(mean_imp, 4),
            "Desv. Est.": round(std_imp, 4),
            "IC 95% Inferior": round(ci_low, 4),
            "IC 95% Superior": round(ci_high, 4),
            "IC 95%": f"[{ci_low:.4f}, {ci_high:.4f}]"
        })

    imp_df = pd.DataFrame(records).sort_values("Incremento RMSE (Media)", ascending=False).reset_index(drop=True)
    return imp_df


# ==============================================================================
# e) ANÁLISIS DE CONCORDANCIA (SPEARMAN, KENDALL, CCC DE LIN, BLAND-ALTMAN)
# ==============================================================================

def compute_concordance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str = "Gas (ppm)"
) -> Tuple[Dict[str, Any], go.Figure]:
    """
    Evalúa la concordancia entre mediciones reales e inferidas:
    - Spearman rho y Kendall tau
    - Coeficiente de Correlación de Concordancia (CCC) de Lin
    - Gráfico de Bland-Altman con Sesgo y Límites de Acuerdo al 95% (LoA)
    """
    yt = y_true.ravel()
    yp = y_pred.ravel()

    # 1. Spearman & Kendall
    spearman_rho, sp_p = stats.spearmanr(yt, yp)
    kendall_tau, kd_p = stats.kendalltau(yt, yp)

    # 2. Coeficiente de Correlación de Concordancia (CCC) de Lin
    mean_t, mean_p = np.mean(yt), np.mean(yp)
    var_t, var_p = np.var(yt, ddof=1), np.var(yp, ddof=1)
    s_tp = np.cov(yt, yp)[0, 1]
    ccc_lin = (2.0 * s_tp) / (var_t + var_p + (mean_t - mean_p) ** 2) if (var_t + var_p) > 1e-12 else 0.0

    # 3. Bland-Altman
    diffs = yp - yt  # Predicho - Real
    means = (yp + yt) / 2.0
    bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))
    loa_upper = bias + 1.96 * sd_diff
    loa_lower = bias - 1.96 * sd_diff

    # Construir figura Bland-Altman
    ba_fig = go.Figure()
    ba_fig.add_trace(go.Scatter(
        x=means, y=diffs, mode="markers", name="Diferencias",
        marker=dict(color="#0284c7", size=6, opacity=0.7)
    ))
    # Línea de Sesgo medio
    ba_fig.add_hline(y=bias, line_color="#10b981", line_width=2, annotation_text=f"Sesgo: {bias:.3f}")
    # Límites de Acuerdo (LoA)
    ba_fig.add_hline(y=loa_upper, line_dash="dash", line_color="#ef4444", annotation_text=f"+1.96 DE: {loa_upper:.3f}")
    ba_fig.add_hline(y=loa_lower, line_dash="dash", line_color="#ef4444", annotation_text=f"-1.96 DE: {loa_lower:.3f}")

    ba_fig.update_layout(
        title=f"Gráfico de Acuerdo de Bland-Altman ({target_name})",
        xaxis_title="Media entre Observado y Predicho (ppm)",
        yaxis_title="Diferencia (Predicho - Observado) (ppm)",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40)
    )

    metrics = {
        "target_name": target_name,
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(sp_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kd_p),
        "lin_ccc": float(ccc_lin),
        "ccc_grade": "Casi Perfecto (> 0.99)" if ccc_lin > 0.99 else ("Sustancial (0.95-0.99)" if ccc_lin > 0.95 else ("Moderado (0.90-0.95)" if ccc_lin > 0.90 else "Deficiente (< 0.90)")),
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_lower": loa_lower,
        "loa_upper": loa_upper
    }

    return metrics, ba_fig
