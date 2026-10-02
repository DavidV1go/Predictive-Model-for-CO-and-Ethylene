"""
Módulo de E/S de datos para MineTwin.
Lectura robusta, validación y agregación de lecturas de sensores químicos (16 canales).
"""

import os
import io
import re
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Nombres canónicos de las 19 columnas
SENSOR_COLUMNS = [f"S{i:02d}" for i in range(1, 17)]
ALL_COLUMNS = ["time_s", "gas_ppm", "ethylene_ppm"] + SENSOR_COLUMNS


def discover_local_datasets(data_dir: str = "./data") -> Dict[str, str]:
    """
    Busca automáticamente archivos de datos en el directorio especificado.
    """
    datasets = {}
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return datasets

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".txt") or fname.endswith(".csv"):
            fpath = os.path.join(data_dir, fname)
            # Deducir etiqueta amigable
            lower = fname.lower()
            if "ch4" in lower or "metano" in lower:
                label = f"Metano (CH4) + Etileno [{fname}]"
            elif "co" in lower:
                label = f"Monóxido de Carbono (CO) + Etileno [{fname}]"
            else:
                label = f"Dataset Químico [{fname}]"
            datasets[label] = fpath
    return datasets


def parse_raw_sensor_file(
    file_source: Any,
    filename: str = "dataset.txt"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parser robusto para archivos de sensores químicos a 100 Hz.
    Formato: primera línea encabezado texto, luego filas de 19 valores numéricos
    separados por espacios o tabulaciones.

    Retorna:
        - DataFrame con columnas [time_s, gas_ppm, ethylene_ppm, S01..S16] en float32
        - Diccionario de metadatos de validación (filas totales, corruptas, columnas detectadas)
    """
    metadata: Dict[str, Any] = {
        "filename": filename,
        "total_lines_read": 0,
        "valid_rows": 0,
        "corrupt_rows": 0,
        "detected_header": None,
        "gas_type": "gas_ppm",
        "errors": []
    }

    # Detectar si file_source es bytes, UploadedFile, string o ruta
    if isinstance(file_source, str) and os.path.exists(file_source):
        with open(file_source, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    elif hasattr(file_source, "getvalue"):
        content = file_source.getvalue().decode("utf-8", errors="replace")
        lines = content.splitlines()
    elif isinstance(file_source, bytes):
        lines = file_source.decode("utf-8", errors="replace").splitlines()
    elif isinstance(file_source, str):
        lines = file_source.splitlines()
    else:
        raise ValueError("Fuente de archivo no reconocida.")

    metadata["total_lines_read"] = len(lines)
    if len(lines) == 0:
        raise ValueError("El archivo está completamente vacío.")

    # Analizar primera línea (encabezado)
    first_line = lines[0].strip()
    metadata["detected_header"] = first_line

    if "co conc" in first_line.lower():
        metadata["gas_type"] = "CO (ppm)"
    elif "methane" in first_line.lower() or "ch4" in first_line.lower():
        metadata["gas_type"] = "Metano CH4 (ppm)"
    else:
        metadata["gas_type"] = "Gas Principal (ppm)"

    valid_data = []
    corrupt_count = 0
    errors_sample = []

    # Parsear línea por línea a partir de la fila 1
    for idx, line in enumerate(lines[1:], start=2):
        sline = line.strip()
        if not sline:
            continue

        # Separar por cualquier cantidad de espacios o tabs
        parts = re.split(r"[\s\t]+", sline)
        if len(parts) != 19:
            corrupt_count += 1
            if len(errors_sample) < 5:
                errors_sample.append(f"Fila {idx}: esperaba 19 columnas, obtuvo {len(parts)}")
            continue

        try:
            numeric_vals = [float(p) for p in parts]
            valid_data.append(numeric_vals)
        except ValueError as e:
            corrupt_count += 1
            if len(errors_sample) < 5:
                errors_sample.append(f"Fila {idx}: error numérico ({str(e)})")

    metadata["valid_rows"] = len(valid_data)
    metadata["corrupt_rows"] = corrupt_count
    metadata["errors"] = errors_sample

    if len(valid_data) == 0:
        raise ValueError(
            f"No se pudieron extraer filas válidas de {filename}. "
            f"Se encontraron {corrupt_count} filas con discrepancias."
        )

    # Crear DataFrame con float32
    arr = np.array(valid_data, dtype=np.float32)
    df = pd.DataFrame(arr, columns=ALL_COLUMNS)

    return df, metadata


def aggregate_by_time_window(
    df: pd.DataFrame,
    aggregation_window_s: float = 1.0,
    agg_func: str = "mean"
) -> pd.DataFrame:
    """
    Submuestreo o agregación temporal por ventanas fijas configurables.
    Por defecto: promedio cada 1 s.
    Si aggregation_window_s <= 0.01 o es None, retorna el dataframe original.
    """
    if aggregation_window_s <= 0.01 or len(df) == 0:
        return df.copy()

    # Asignar bins de tiempo
    t0 = df["time_s"].iloc[0]
    time_bins = np.floor((df["time_s"] - t0) / aggregation_window_s) * aggregation_window_s + t0

    df_temp = df.copy()
    df_temp["_time_bin"] = time_bins

    if agg_func == "median":
        agg_df = df_temp.groupby("_time_bin").median().reset_index(drop=True)
    else:
        agg_df = df_temp.groupby("_time_bin").mean().reset_index(drop=True)

    if "_time_bin" in agg_df.columns:
        agg_df = agg_df.drop(columns=["_time_bin"])

    return agg_df.astype(np.float32)
