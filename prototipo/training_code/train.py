from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from build_lagged_features import DATE, DEFAULT_LAGS, TARGET, build_lagged_frame, read_table

# ----------------------------- configuração ---------------------------------
DROP_VARS = ["par_umol_m2_s"]   # variáveis removidas (correlação alta com outras)
CORR_THRESHOLD = 0.3            # |corr| mínima com o target para manter um lag
ALPHA_GRID = np.logspace(-5, 1, 300)
CV_SPLITS = 5                   # nº de folds do TimeSeriesSplit
CV_TEST_SIZE = 365              # tamanho de cada fold de validação (dias)
MAX_ITER = 20000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path, help="ficheiro .xlsx ou .csv com os dados brutos")
    parser.add_argument("--output", required=True, type=Path,
                        help="caminho do model.pkl (se for uma pasta, grava model.pkl lá dentro)")
    args = parser.parse_args()

    # Read data and order dates
    df = read_table(args.data)

    # Apply log(TARGET)
    df[TARGET] = np.log(df[TARGET].astype(float))
    
    # Outliers treatment
    Q1, Q3 = df[TARGET].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers_mask = (df[TARGET] < lower) | (df[TARGET] > upper)
    df.loc[outliers_mask, TARGET] = np.nan
    train_clean = df[TARGET].dropna()
    df.loc[outliers_mask, TARGET] = train_clean.mode().iloc[0]

    outlier_value = float(df.loc[~outliers_mask, TARGET].round(2).mode().iloc[0])


    # 4) Filter Lags and Clean exogenous features (["par_umol_m2_s"])
    lag_spec = {k: list(v) for k, v in DEFAULT_LAGS.items() if k not in DROP_VARS}
    lagged = build_lagged_frame(df, lag_spec=lag_spec)

    lag_cols = [c for c in lagged.columns if "__lag_" in c]
    scores = lagged[lag_cols].corrwith(lagged[TARGET]).abs().fillna(0)
    selected = scores[scores >= CORR_THRESHOLD].index.tolist()
    keep = [c for c in lagged.columns if "__lag_" not in c] + selected   # date, target, annual_sin/cos, lags
    lagged = lagged[keep]

    feature_cols = [c for c in lagged.columns if c not in (DATE, TARGET)]
    X = lagged[feature_cols].to_numpy(dtype=float)
    y = lagged[TARGET].to_numpy(dtype=float)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Alpha selection
    cv = TimeSeriesSplit(n_splits=CV_SPLITS, test_size=CV_TEST_SIZE)
    lasso_cv = LassoCV(alphas=ALPHA_GRID, cv=cv, max_iter=MAX_ITER).fit(X_scaled, y)
    alpha = float(lasso_cv.alpha_)
    print(f"Alpha escolhido (LassoCV, {CV_SPLITS} folds temporais): {alpha:.6f}")

    # Final model (Lasso)
    model = Lasso(alpha=alpha, max_iter=MAX_ITER).fit(X_scaled, y)

    # Selected lags
    final_lag_spec: dict[str, list[int]] = {}
    for c in feature_cols:
        if "__lag_" in c:
            var, lag = c.split("__lag_")
            final_lag_spec.setdefault(var, []).append(int(lag))

    artifact = {
        "estimator": model,
        "feature_cols": feature_cols,
        "mean": scaler.mean,
        "scale": scaler.scale,
        "lag_spec": final_lag_spec,
        "alpha": alpha,
        "outlier_lower": float(lower),
        "outlier_upper": float(upper),
        "outlier_value": outlier_value,
    }

    out_path = args.output
    if out_path.is_dir() or out_path.suffix == "":
        out_path = out_path / "model.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)
    print(f"Modelo guardado em {out_path}")


if __name__ == "__main__":
    main()



