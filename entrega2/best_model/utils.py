'''"""Auxiliary prediction functions imported by my_model.py."""

from __future__ import annotations

from typing import Any


def constant_prediction(model: Any) -> float:
    """Extract the dummy model's constant prediction."""
    return float(model["constant_prediction"])
'''

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DATE = "date"
TARGET = "chlorophyll_a_mg_m3"


def preprocess_history(history_df: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Aplica ao histórico bruto o mesmo tratamento do target feito no treino:
    log -> substituição dos outliers pelo valor guardado."""
    hist = history_df.copy()
    hist[DATE] = pd.to_datetime(hist[DATE])
    hist = hist.sort_values(DATE).reset_index(drop=True)
    y = np.log(pd.to_numeric(hist[TARGET], errors="coerce").to_numpy(dtype=float))
    bad = (y < model["outlier_lower"]) | (y > model["outlier_upper"])
    y[bad] = model["outlier_value"]
    hist[TARGET] = y
    return hist


def build_feature_row(history_df: pd.DataFrame, day_features_df: pd.DataFrame,
                      lag_spec: dict[str, list[int]], feature_cols: list[str]) -> pd.DataFrame:
    """Constrói UMA linha de features, com as mesmas colunas e ordem do treino."""
    max_lag = max(max(v) for v in lag_spec.values())

    hist = history_df.copy()
    today = day_features_df.copy()
    hist[DATE] = pd.to_datetime(hist[DATE])
    today[DATE] = pd.to_datetime(today[DATE])

    hist = hist.sort_values(DATE).tail(max_lag)
    if len(hist) < max_lag:
        raise ValueError("histórico insuficiente para os lags pedidos")
    today[TARGET] = np.nan                       # o target de hoje nunca é usado

    combined = pd.concat([hist, today], ignore_index=True, sort=False)
    t = len(combined) - 1                        # posição do dia a prever

    feats: dict[str, float] = {}
    for col, lags in lag_spec.items():
        values = pd.to_numeric(combined[col], errors="coerce").to_numpy(dtype=float)
        for lag in lags:
            if col == TARGET and lag == 0:
                raise ValueError("lag 0 do target é proibido")
            feats[f"{col}__lag_{lag}"] = values[t - lag]

    day = int(today[DATE].dt.dayofyear.iloc[0])
    feats["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    feats["annual_cos"] = np.cos(2 * np.pi * day / 365.25)

    return pd.DataFrame([feats])[feature_cols]


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Normalização explícita com os parâmetros aprendidos no treino."""
    return (np.asarray(x, dtype=float) - mean) / scale


def finalize(y_log: np.ndarray) -> np.ndarray:
    """Inverte o log e garante um valor finito e não negativo, shape (1,)."""
    y = np.exp(np.asarray(y_log, dtype=float).reshape(-1))
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(y, 0.0)