from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from utils import build_features, predict_proba_class1


def load_model(model_path: str | Path) -> Any:
    """Carrega o modelo guardado (dicionário com coeficientes e parâmetros simples)."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, day_and_lag_features_df: pd.DataFrame) -> np.ndarray:
    """Devolve a probabilidade da classe 1 (HAB) como array NumPy finito em [0, 1].

    Há um valor por linha do DataFrame recebido (shape (1,) para uma só linha).
    """
    X = build_features(day_and_lag_features_df, model)
    return np.asarray(predict_proba_class1(model, X), dtype=float).reshape(-1)
