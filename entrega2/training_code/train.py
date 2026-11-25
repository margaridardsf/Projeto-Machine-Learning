from __future__ import annotations

import argparse
import importlib.util
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Reutiliza as funções de features de best_model/utils.py (as mesmas da previsão).
ROOT = Path(__file__).resolve().parent.parent          # raiz da submissão
DATA_NAME = "classification_task1_train.xlsx"
_UTILS_PATH = ROOT / "best_model" / "utils.py"
_spec = importlib.util.spec_from_file_location("best_model_utils", _UTILS_PATH)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)



MISSING_STRATEGIES = ["doy_window_median", "global_median", "drop_rows", "drop_features"]
C_GRID = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0]
PENALTIES = ["l2", "l1"]
CLASS_WEIGHTS = [None, "balanced"]


def configs():
    for pen in PENALTIES:
        for cw in CLASS_WEIGHTS:
            for C in C_GRID:
                yield {"penalty": pen, "C": C, "class_weight": cw}


def make_lr(cfg) -> LogisticRegression:
    solver = "lbfgs" if cfg["penalty"] == "l2" else "liblinear"
    return LogisticRegression(penalty=cfg["penalty"], C=cfg["C"],
                              class_weight=cfg["class_weight"],
                              solver=solver, max_iter=5000, random_state=0)


def p1(model, A) -> np.ndarray:
    return model.predict_proba(A)[:, list(model.classes_).index(1)]



def load_data(path: Path) -> pd.DataFrame:
    """Lê o ficheiro cru e ordena explicitamente por data (e cluster)."""
    if Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "spatial_cluster"], kind="stable").reset_index(drop=True)


def fill_doy_window(X: pd.DataFrame, dates, half_window: int) -> pd.DataFrame:
    X = X.copy()
    vals = X[U.MISSING_COLS].to_numpy(dtype=float, copy=True)
    miss = np.isnan(vals)
    rows = np.flatnonzero(miss.any(axis=1))
    if len(rows) == 0:
        return X
    doy = U.doy365(dates)
    observed = ~miss.any(axis=1)
    obs_doy, obs_vals = doy[observed], vals[observed]
    global_med = np.median(obs_vals, axis=0)
    for i in rows:
        diff = np.abs(obs_doy - doy[i])
        diff = np.minimum(diff, 365 - diff)
        in_win = diff <= half_window
        for j in range(len(U.MISSING_COLS)):
            if miss[i, j]:
                vals[i, j] = (np.median(obs_vals[in_win, j])
                              if in_win.any() else global_med[j])
    X[U.MISSING_COLS] = vals
    return X


def fit_train_state(raw_tr: pd.DataFrame, dates_tr, y_tr: np.ndarray,
                    strategy: str, half_window: int):
    names = U.feature_names(use_chl=(strategy != "drop_features"))
    cont = [c for c in names if c in U.CONT_COLS]

    both = raw_tr[U.MISSING_COLS].notna().all(axis=1).to_numpy() 
    fill_values = {}
    for c in cont:
        src = raw_tr[both] if c in U.MISSING_COLS else raw_tr
        fill_values[c] = float(src[c].median())

    if strategy == "drop_rows":
        X, y = raw_tr[both], y_tr[both]
    elif strategy == "global_median":
        X, y = U.fill_global(raw_tr, fill_values), y_tr
    elif strategy == "doy_window_median":
        X, y = fill_doy_window(raw_tr, dates_tr, half_window), y_tr
    elif strategy == "drop_features":
        X, y = raw_tr, y_tr
    else:
        raise ValueError(f"estratégia desconhecida: {strategy}")
    X = U.fill_global(X, fill_values)

    lo, rng = {}, {}
    for c in cont:
        lo[c] = float(X[c].min())
        r = float(X[c].max() - X[c].min())
        rng[c] = r if r > 0 else 1.0
    return X, y, {"feature_names": names, "fill_values": fill_values,
                  "min": lo, "range": rng}


def make_folds(df: pd.DataFrame, raw: pd.DataFrame, strategy: str, half_window: int):
    years = df["Date"].dt.year.to_numpy()
    y = df["HAB"].to_numpy(dtype=int)
    complete = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()
    folds = []
    for vy in sorted(set(years)):
        tr, va = years != vy, (years == vy) & complete
        if va.sum() == 0:
            continue
        X_tr, y_tr, st = fit_train_state(raw[tr], df["Date"][tr], y[tr], strategy, half_window)
        if len(np.unique(y_tr)) < 2:                
            continue
        X_va = U.fill_global(raw[va], st["fill_values"])
        folds.append({"year": int(vy), "y_tr": y_tr, "y_va": y[va],
                      "A_tr": U.minmax(X_tr, st["feature_names"], st["min"], st["range"]),
                      "A_va": U.minmax(X_va, st["feature_names"], st["min"], st["range"])})
    return folds


def evaluate_cfg(folds, cfg):
    ys, ps, yrs = [], [], []
    for f in folds:
        m = make_lr(cfg).fit(f["A_tr"], f["y_tr"])
        ps.append(p1(m, f["A_va"]))
        ys.append(f["y_va"])
        yrs.append(np.full(len(f["y_va"]), f["year"]))
    y, p, yr = np.concatenate(ys), np.concatenate(ps), np.concatenate(yrs)
    per_year = {int(a): roc_auc_score(y[yr == a], p[yr == a])
                for a in np.unique(yr) if np.unique(y[yr == a]).size == 2}
    return {"pooled_auc": roc_auc_score(y, p), "per_year": per_year, "oof_y": y, "oof_p": p}


# ----------------------------------------------------------------------------
def main() -> None:
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None,
                        help="ficheiro cru (.xlsx ou .csv); por defeito procura "
                             "classification_task1_train.xlsx na pasta atual, na "
                             "raiz da submissão ou em training_code/")
    parser.add_argument("--output", type=Path, default=None,
                        help="onde gravar o modelo (por defeito best_model/model.pkl "
                             "na raiz da submissão)")
    parser.add_argument("--missing", default="doy_window_median", choices=MISSING_STRATEGIES,
                        help="estratégia para CHL / CHL_uncertainty em falta")
    parser.add_argument("--doy-half-window", type=int, default=15)
    args = parser.parse_args()

    if args.data is None:
        candidates = [Path.cwd() / DATA_NAME, ROOT / DATA_NAME, Path(__file__).resolve().parent / DATA_NAME]
        found = [c for c in candidates if c.is_file()]
        if not found:
            parser.error(f"não encontrei {DATA_NAME}; indicar --data")
        args.data = found[0]
    if args.output is None:
        args.output = ROOT / "best_model" / "model.pkl"

    df = load_data(args.data)
    raw = U.raw_features(df)                  
    y = df["HAB"].to_numpy(dtype=int)
    complete = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()
    
    # escolha dos hiperparâmetros: AUC agregado out-of-fold (LOYO)
    folds = make_folds(df, raw, args.missing, args.doy_half_window)
    results, best = [], None
    for cfg in configs():
        r = evaluate_cfg(folds, cfg)
        results.append({**cfg, "pooled_auc": r["pooled_auc"],
                        "mean_year_auc": float(np.mean(list(r["per_year"].values())))})
        if best is None or r["pooled_auc"] > best["pooled_auc"]:      # empate: fica a 1ª
            best = {**r, "cfg": cfg}
    cfg = best["cfg"]
  
 
    # limiar que maximiza a Balanced Accuracy (TPR - FPR) nas previsões out-of-fold
    fpr, tpr, thr_ = roc_curve(best["oof_y"], best["oof_p"])
    threshold = float(thr_[1:][np.argmax(tpr[1:] - fpr[1:])])

    # modelo final: tudo aprendido com todos os dados de treino
    X_all, y_all, st = fit_train_state(raw, df["Date"], y, args.missing, args.doy_half_window)
    A_all = U.minmax(X_all, st["feature_names"], st["min"], st["range"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = make_lr(cfg).fit(A_all, y_all)
    assert list(final.classes_) == [0, 1]

    model = {
        "model_type": "logistic_regression",
        "feature_names": list(st["feature_names"]),
        "coef": [float(v) for v in final.coef_[0]],      # coeficientes da classe 1
        "intercept": float(final.intercept_[0]),
        "min": st["min"],                                # scaling min-max (só contínuas)
        "range": st["range"],                            # max - min
        "fill_values": st["fill_values"],                # só salvaguarda na previsão
        "threshold": threshold,                          # informativo (Balanced Accuracy)
        "missing_strategy": args.missing,
        "hyperparameters": dict(cfg),
        "cv_pooled_auc": float(best["pooled_auc"]),
    }

    assert np.allclose(final.predict_proba(A_all)[:, 1], U.predict_proba_class1(model, A_all),
                       atol=1e-9), "previsão explícita difere do sklearn"
    rows = X_all.index[complete[X_all.index.to_numpy()]]
    assert np.allclose(U.build_features(df.loc[rows], model),
                       A_all[X_all.index.get_indexer(rows)]), "X da previsão difere do treino"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle, protocol=4)
    print(f"\nModelo guardado em {args.output}")
    

if __name__ == "__main__":
    main()