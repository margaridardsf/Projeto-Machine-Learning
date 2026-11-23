from __future__ import annotations

import argparse
import importlib.util
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Reutiliza as funções de features de best_model/utils.py (as mesmas da previsão).
ROOT = Path(__file__).resolve().parent.parent          # raiz da submissão
DATA_NAME = "classification_task1_train.xlsx"
_UTILS_PATH = ROOT / "best_model" / "utils.py"
_spec = importlib.util.spec_from_file_location("best_model_utils", _UTILS_PATH)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)

warnings.filterwarnings("ignore", message=".*penalty.*")  # avisos de versões mais recentes do sklearn

MISSING_COLS = ["CHL_log", "CHL_uncertainty"]

# Grelha de hiperparâmetros (a mais simples primeiro: em empate fica a 1ª).
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]
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


# ----------------------------------------------------------------------------
# Dados
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Lê o ficheiro cru e ordena explicitamente por data (e cluster)."""
    if Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "spatial_cluster"], kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Preenchimento de CHL / CHL_uncertainty (aprendido só no treino)
# ----------------------------------------------------------------------------
def fill_doy_window(X: pd.DataFrame, dates, half_window: int) -> pd.DataFrame:
    """Mediana dos valores observados numa janela de +-half_window dias do ano.

    A distância ao longo do ano é circular. Se a janela estiver vazia usa-se a
    mediana global. Só as linhas observadas de X (o treino) são usadas.
    """
    X = X.copy()
    vals = X[MISSING_COLS].to_numpy(dtype=float, copy=True)
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
        for j in range(len(MISSING_COLS)):
            if miss[i, j]:
                vals[i, j] = (np.median(obs_vals[in_win, j])
                              if in_win.any() else global_med[j])
    X[MISSING_COLS] = vals
    return X


def fit_train_state(raw_tr: pd.DataFrame, dates_tr, y_tr: np.ndarray,
                      strategy: str, half_window: int):
    """Aprende, SÓ com o treino: preenchimento, médias e desvios.

    Devolve (X_treino_preenchido, y_treino, estado) onde estado tem
    fill_values, mean e std (dicionários simples nome -> float).
    """
    fill_values = {c: float(v) for c, v in raw_tr[U.CONT_COLS].median().items()}

    if strategy == "drop_rows":
        keep = raw_tr[MISSING_COLS].notna().all(axis=1).to_numpy()
        X, y = raw_tr[keep], y_tr[keep]
    elif strategy == "global_median":
        X, y = U.fill_global(raw_tr, fill_values), y_tr
    elif strategy == "doy_window_median":
        X = fill_doy_window(raw_tr, dates_tr, half_window)
        X, y = U.fill_global(X, fill_values), y_tr
    else:
        raise ValueError(f"estratégia desconhecida: {strategy}")

    mean = {c: float(X[c].mean()) for c in U.CONT_COLS}
    std = {}
    for c in U.CONT_COLS:
        s = float(X[c].std(ddof=0))
        std[c] = s if s > 0 else 1.0
    return X, y, {"fill_values": fill_values, "mean": mean, "std": std}


# ----------------------------------------------------------------------------
# Validação cronológica (forward chaining por ano)
# ----------------------------------------------------------------------------
def forward_chaining(df: pd.DataFrame, raw: pd.DataFrame, strategy: str,
                     half_window: int, first_val_year: int):
    dates = df["Date"]
    years = dates.dt.year.to_numpy()
    y = df["HAB"].to_numpy(dtype=int)
    complete = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()

    cfgs = list(configs())
    oof_p = [[] for _ in cfgs]
    oof_y, fold_auc = [], [[] for _ in cfgs]
    used_years = []

    for vy in sorted(set(years)):
        if vy < first_val_year:
            continue
        tr = (dates < pd.Timestamp(year=int(vy), month=1, day=1)).to_numpy()
        va = (years == vy) & complete          # validação só em linhas completas
        if va.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        X_tr, y_tr, st = fit_train_state(raw[tr], dates[tr], y[tr],
                                           strategy, half_window)
        A_tr = U.standardize(X_tr, st["mean"], st["std"])
        X_va = U.fill_global(raw[va], st["fill_values"])
        A_va = U.standardize(X_va, st["mean"], st["std"])
        used_years.append(int(vy))
        oof_y.append(y[va])
        for i, cfg in enumerate(cfgs):
            m = make_lr(cfg).fit(A_tr, y_tr)
            p = m.predict_proba(A_va)[:, list(m.classes_).index(1)]
            oof_p[i].append(p)
            if len(np.unique(y[va])) == 2:
                fold_auc[i].append(roc_auc_score(y[va], p))

    y_all = np.concatenate(oof_y)
    if len(np.unique(y_all)) < 2:
        raise RuntimeError("validação sem as duas classes; ajustar --first-val-year")

    results = []
    for i, cfg in enumerate(cfgs):
        pooled = roc_auc_score(y_all, np.concatenate(oof_p[i]))
        results.append({**cfg, "pooled_auc": pooled,
                        "mean_fold_auc": float(np.mean(fold_auc[i])) if fold_auc[i] else np.nan})
    return cfgs, results, used_years, int(len(y_all)), int(y_all.sum())


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None,
                        help="ficheiro cru (.xlsx ou .csv); por defeito procura "
                             "classification_task1_train.xlsx na pasta atual, na "
                             "raiz da submissão ou em training_code/")
    parser.add_argument("--output", type=Path, default=None,
                        help="onde gravar o modelo (por defeito best_model/model.pkl "
                             "na raiz da submissão)")
    parser.add_argument("--missing", default="doy_window_median",
                        choices=["doy_window_median", "global_median", "drop_rows"],
                        help="estratégia para CHL / CHL_uncertainty em falta")
    parser.add_argument("--doy-half-window", type=int, default=15)
    parser.add_argument("--first-val-year", type=int, default=2016)
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
    raw = U.raw_features(df)                  # determinístico: nada é aprendido
    y = df["HAB"].to_numpy(dtype=int)
    print(f"{len(df)} linhas, {int(y.sum())} positivos "
          f"({df['Date'].min().date()} a {df['Date'].max().date()}); "
          f"estratégia de missing values: {args.missing}")

    # 1) escolha dos hiperparâmetros
    cfgs, results, used_years, n_val, pos_val = forward_chaining(
        df, raw, args.missing, args.doy_half_window, args.first_val_year)
    table = pd.DataFrame(results)
    print(f"\nValidação cronológica: anos {used_years}, {n_val} linhas completas, "
          f"{pos_val} positivos")
    print(table.sort_values("pooled_auc", ascending=False).head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best_i = int(np.argmax(table["pooled_auc"].to_numpy()))   # empate: 1ª (mais simples)
    best_cfg = cfgs[best_i]
    print(f"\nEscolhido: {best_cfg}  (AUC agregado = {table['pooled_auc'][best_i]:.4f})")

    # 2) modelo final: tudo aprendido com todos os dados de treino
    X_all, y_all, st = fit_train_state(raw, df["Date"], y, args.missing,
                                         args.doy_half_window)
    A_all = U.standardize(X_all, st["mean"], st["std"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = make_lr(best_cfg).fit(A_all, y_all)
    assert list(final.classes_) == [0, 1]

    model = {
        "model_type": "logistic_regression",
        "feature_names": list(U.FEATURE_NAMES),
        "coef": [float(v) for v in final.coef_[0]],      # coeficientes da classe 1
        "intercept": float(final.intercept_[0]),
        "mean": st["mean"],
        "std": st["std"],
        "fill_values": st["fill_values"],                # só salvaguarda na previsão
        "missing_strategy": args.missing,
        "hyperparameters": {k: v for k, v in best_cfg.items()},
        "cv_pooled_auc": float(table["pooled_auc"][best_i]),
    }

    # a previsão explícita (sigmoid) tem de coincidir com a do sklearn
    p_ref = final.predict_proba(A_all)[:, 1]
    p_own = U.predict_proba_class1(model, A_all)
    assert np.allclose(p_ref, p_own, atol=1e-9), "previsão explícita difere do sklearn"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(model, handle, protocol=4)
    print(f"\nModelo guardado em {args.output}")


if __name__ == "__main__":
    main()