from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

SEASONS = ["Winter", "Spring", "Summer", "Fall"]
N_CLUSTERS = 6

# Variáveis contínuas (as únicas que são standardizadas).
CONT_COLS = ["SST", "UI", "YDOY", "sin_doy", "CHL_log", "CHL_uncertainty"]

# Ordem fixa das colunas da matriz X (a mesma no treino e na previsão).
FEATURE_NAMES = (
    [f"season_{s}" for s in SEASONS]
    + ["SST", "UI", "YDOY", "sin_doy", "CHL_log", "CHL_uncertainty"]
    + [f"cluster_{c}" for c in range(N_CLUSTERS)]
)


def doy365(dates) -> np.ndarray:
    """Dia do ano, com 366 -> 365 (anos bissextos), como no YDOY do enunciado."""
    doy = pd.DatetimeIndex(pd.to_datetime(dates)).dayofyear.to_numpy()
    return np.minimum(doy, 365)


def raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói as 19 features a partir das colunas cruas.

    Só aplica operações determinísticas (nada é aprendido aqui):
    one-hot de Season (categorias fixas) e de spatial_cluster, sin(2*pi*DOY/365)
    e log(CHL). CHL / CHL_uncertainty em falta ficam como NaN.
    Não altera o DataFrame recebido; o índice é preservado.
    """
    out = pd.DataFrame(index=df.index)

    season = df["Season"].astype(str).str.strip()
    for s in SEASONS:
        out[f"season_{s}"] = (season == s).astype(float)

    for c in ["SST", "UI", "YDOY"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    out["sin_doy"] = np.sin(2.0 * np.pi * doy365(df["Date"]) / 365.0)

    chl = pd.to_numeric(df["CHL"], errors="coerce").astype(float)
    out["CHL_log"] = np.log(chl.where(chl > 0))          # CHL <= 0 -> NaN
    out["CHL_uncertainty"] = pd.to_numeric(
        df["CHL_uncertainty"], errors="coerce").astype(float)

    cluster = pd.to_numeric(df["spatial_cluster"], errors="coerce")
    for c in range(N_CLUSTERS):
        out[f"cluster_{c}"] = (cluster == c).astype(float)

    return out[FEATURE_NAMES]


def fill_global(X: pd.DataFrame, fill_values: Dict[str, float]) -> pd.DataFrame:
    """Preenche NaN das contínuas com valores guardados (medianas do treino).

    No teste oficial todas as linhas são completas, por isso isto só serve de
    salvaguarda para que a previsão nunca falhe nem devolva NaN.
    """
    X = X.copy()
    for col, value in fill_values.items():
        if col in X.columns:
            X[col] = X[col].fillna(value)
    return X


def standardize(X: pd.DataFrame, mean: Dict[str, float],
                std: Dict[str, float]) -> np.ndarray:
    """Standardiza só as contínuas com média/desvio guardados; devolve ndarray.

    As colunas one-hot ficam 0/1. A ordem das colunas é FEATURE_NAMES.
    """
    A = X[FEATURE_NAMES].to_numpy(dtype=float, copy=True)
    for j, name in enumerate(FEATURE_NAMES):
        if name in CONT_COLS:
            A[:, j] = (A[:, j] - mean[name]) / std[name]
    return A


def build_features(df: pd.DataFrame, model: Dict[str, Any]) -> np.ndarray:
    """Matriz X (n_linhas x 19) pronta para o modelo, a partir do DataFrame cru."""
    X = raw_features(df)
    X = fill_global(X, model["fill_values"])
    return standardize(X, model["mean"], model["std"])


def predict_proba_class1(model: Dict[str, Any], A: np.ndarray) -> np.ndarray:
    """P(HAB=1) da regressão logística guardada: sigmoid(A @ coef + intercept)."""
    coef = np.asarray(model["coef"], dtype=float)
    z = A @ coef + float(model["intercept"])
    p = 0.5 * (1.0 + np.tanh(0.5 * z))                   # sigmoid numericamente estável
    return np.clip(np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
