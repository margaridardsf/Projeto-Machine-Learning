from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

CLUSTER_MAP = {3: 2}
CLUSTERS = [0, 1, 2, 4, 5]
CONT_COLS = ["SST", "UI", "YDOY", "sin_doy", "CHL_log", "CHL_uncertainty"]
MISSING_COLS = ["CHL_log", "CHL_uncertainty"]


def feature_names(use_chl: bool) -> List[str]:
    """Ordem fixa das colunas da matriz X (a mesma no treino e na previsão)."""
    names = ["SST", "UI", "YDOY", "sin_doy"]
    if use_chl:
        names += MISSING_COLS
    return names + [f"cluster_{c}" for c in CLUSTERS]


def doy365(dates) -> np.ndarray:
    """Dia do ano, com 366 -> 365 (anos bissextos), como no YDOY do enunciado."""
    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear.to_numpy()
    return np.minimum(doy, 365)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:                     
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ["SST", "UI", "YDOY"]:
        out[c] = _numeric(df, c)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy365(df["Date"]) / 365.0)

    chl = _numeric(df, "CHL")
    out["CHL_log"] = np.log(chl.where(chl > 0))    
    out["CHL_uncertainty"] = _numeric(df, "CHL_uncertainty")

    cluster = _numeric(df, "spatial_cluster").replace(CLUSTER_MAP)
    for c in CLUSTERS:                             
        out[f"cluster_{c}"] = (cluster == c).astype(float)
    return out


def fill_global(X: pd.DataFrame, fill_values: Dict[str, float]) -> pd.DataFrame:
    X = X.copy()
    for col, value in fill_values.items():
        if col in X.columns:
            X[col] = X[col].fillna(value)
    return X


def minmax(X: pd.DataFrame, names: List[str], lo: Dict[str, float],
           rng: Dict[str, float]) -> np.ndarray:
    A = X[names].to_numpy(dtype=float, copy=True)
    for j, name in enumerate(names):
        if name in CONT_COLS:
            A[:, j] = (A[:, j] - lo[name]) / rng[name]
    return A


def build_features(df: pd.DataFrame, model: Dict[str, Any]) -> np.ndarray:
    X = fill_global(raw_features(df), model["fill_values"])
    return minmax(X, model["feature_names"], model["min"], model["range"])


def predict_proba_class1(model: Dict[str, Any], A: np.ndarray) -> np.ndarray:
    coef = np.asarray(model["coef"], dtype=float)
    z = A @ coef + float(model["intercept"])
    p = 0.5 * (1.0 + np.tanh(0.5 * z))                
    return np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)