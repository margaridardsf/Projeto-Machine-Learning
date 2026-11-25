import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Funções Auxiliares
# ----------------------------------------------------------------------------
def _doy365(date):
    """Dia do ano com 366 -> 365 (ano bissexto), como no cos de YDOY."""
    return np.minimum(date.dt.dayofyear.to_numpy(), 365)


# --- Imputação por regressão Ridge (implementada explicitamente em NumPy) ---
RIDGE_ALPHA_GRID = (1, 10, 30, 100, 300, 1000)
RIDGE_DEFAULT_ALPHA = 100.0   # usado se só houver 1 ano em train_val


def _impute_design(d, clusters, n_harmonics, use_env):
    """
    Matriz de features do imputador (usa só variáveis sempre observadas; NUNCA o HAB):
      - harmónicas sazonais sin/cos(2π·k·DOY/365), k = 1..n_harmonics
      - dummies do spatial_cluster (já com a fusão aplicada)
      - opcionalmente SST e UI
    """
    doy = _doy365(d["Date"])
    cols = []
    for k in range(1, n_harmonics + 1):
        ang = 2 * np.pi * k * doy / 365
        cols += [np.sin(ang), np.cos(ang)]
    sc = d["spatial_cluster"].to_numpy()
    cols += [(sc == c).astype(float) for c in clusters]
    if use_env:
        cols += [d["SST"].to_numpy(float), d["UI"].to_numpy(float)]
    return np.column_stack(cols)


def _ridge_fit(X, y, alpha):
    """Ridge em forma fechada: standardiza X, intercepto não penalizado."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y_mean = y.mean()
    # lstsq (mínima norma) em vez de solve: continua válido com alpha=0, em que as
    # dummies do cluster são colineares com o intercepto (matriz singular)
    w = np.linalg.lstsq(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - y_mean), rcond=None)[0]
    return {"mu": mu, "sd": sd, "w": w, "b": y_mean, "alpha": float(alpha)}


def _ridge_predict(model, X):
    return ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def _select_alpha(X, y, years, grid):
    """Escolhe alpha por leave-one-year-out dentro de train_val (sem tocar no test)."""
    uy = np.unique(years)
    if len(uy) < 2:
        return RIDGE_DEFAULT_ALPHA
    sse = {a: 0.0 for a in grid}
    for yr in uy:
        te = years == yr
        for a in grid:
            m = _ridge_fit(X[~te], y[~te], a)
            sse[a] += np.sum((y[te] - _ridge_predict(m, X[te])) ** 2)
    return min(grid, key=lambda a: sse[a])


def _learn_fills(train_val, strategy, doy_half_window, clusters=(0, 1, 2, 4, 5),
                 impute_alpha="auto", impute_harmonics=2, impute_use_env=False):
    """Aprende, SÓ em train_val, as medianas / modelos usados na imputação."""
    fill_cols = ["CHL_log", "CHL_uncertainty"]
    obs = train_val.dropna(subset=fill_cols)
    fills = {"global": obs[fill_cols].median()}
    if strategy == "ridge_regression":
        X = _impute_design(obs, clusters, impute_harmonics, impute_use_env)
        years = obs["Date"].dt.year.to_numpy()
        fills["design"] = (tuple(clusters), impute_harmonics, impute_use_env)
        fills["ridge"], fills["bounds"] = {}, {}
        for c in fill_cols:                      # um modelo por variável a imputar
            y = obs[c].to_numpy(float)
            a = (_select_alpha(X, y, years, RIDGE_ALPHA_GRID)
                 if impute_alpha == "auto" else float(impute_alpha))
            fills["ridge"][c] = _ridge_fit(X, y, a)
            fills["bounds"][c] = (y.min(), y.max())   # previsões limitadas ao intervalo observado
    if strategy == "season_median":
        fills["season"] = obs.groupby("Season")[fill_cols].median()
    if strategy == "doy_window_median":
        fills["obs_doy"] = _doy365(obs["Date"])
        fills["obs_vals"] = obs[fill_cols].to_numpy()
        fills["half_window"] = doy_half_window
    return fills


def _apply_fills(d, strategy, fills):
    """Preenche CHL_log e CHL_uncertainty em falta (imputação em log(CHL))."""
    d = d.copy()
    fill_cols = ["CHL_log", "CHL_uncertainty"]
    miss = d[fill_cols].isna().any(axis=1).to_numpy()
    if not miss.any():
        return d

    if strategy == "global_median":
        for c in fill_cols:
            d[c] = d[c].fillna(fills["global"][c])

    elif strategy == "season_median":
        for c in fill_cols:
            by_season = d["Season"].map(fills["season"][c])
            d[c] = d[c].fillna(by_season).fillna(fills["global"][c])

    elif strategy == "ridge_regression":
        X = _impute_design(d, *fills["design"])
        for c in fill_cols:
            lo, hi = fills["bounds"][c]
            pred = np.clip(_ridge_predict(fills["ridge"][c], X), lo, hi)
            d[c] = d[c].fillna(pd.Series(pred, index=d.index))

    elif strategy == "doy_window_median":
        doy = _doy365(d["Date"])
        for i in np.flatnonzero(miss):
            diff = np.abs(fills["obs_doy"] - doy[i])
            diff = np.minimum(diff, 365 - diff)           # distância circular
            in_win = diff <= fills["half_window"]
            for j, c in enumerate(fill_cols):
                if np.isnan(d[c].iloc[i]):
                    v = (np.median(fills["obs_vals"][in_win, j])
                         if in_win.any() else fills["global"][c])
                    d.iloc[i, d.columns.get_loc(c)] = v
    return d















# ----------------------------------------------------------------------------
# Função principal
# ----------------------------------------------------------------------------
VALID_STRATEGIES = {
    "drop_rows",
    "drop_features",
    "global_median",
    "season_median",
    "doy_window_median",
    "ridge_regression",
}
VALID_SPLITS = {"recent", "recent19", "recent20", "stratified"}
VALID_SEASONS = {"one_hot", "drop"}
VALID_CV = {"loyo", "forward"}
CONT_COLS = ["SST", "UI", "YDOY", "sin_doy", "CHL_log", "CHL_uncertainty"]


def _check_args(strategy, split_test, season, test_frac, doy_half_window,
                impute_harmonics, impute_alpha):
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy inválida: {strategy!r}. Opções: {sorted(VALID_STRATEGIES)}")
    if split_test not in VALID_SPLITS:
        raise ValueError(f"split_test inválido: {split_test!r}. Opções: {sorted(VALID_SPLITS)}")
    if season not in VALID_SEASONS:
        raise ValueError(f"season inválido: {season!r}. Opções: {sorted(VALID_SEASONS)}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac deve estar entre 0 e 1 (recebido: {test_frac})")
    if doy_half_window < 0:
        raise ValueError(f"doy_half_window deve ser >= 0 (recebido: {doy_half_window})")
    if not (isinstance(impute_harmonics, (int, np.integer)) and impute_harmonics >= 0):
        raise ValueError(f"impute_harmonics deve ser um inteiro >= 0 (recebido: {impute_harmonics!r})")
    if not (impute_alpha == "auto" or (isinstance(impute_alpha, (int, float)) and impute_alpha >= 0)):
        raise ValueError(f"impute_alpha deve ser 'auto' ou um número >= 0 (recebido: {impute_alpha!r})")


def _load_and_split(split_test, test_frac, seed, merge_clusters_2_3):
    """Lê os dados, funde clusters e separa train_val / test (ainda sem features)."""
    df = pd.read_excel("classification_task1_train.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "spatial_cluster"], kind="stable").reset_index(drop=True)

    clusters = (0, 1, 2, 3, 4, 5)
    if merge_clusters_2_3:
        df["spatial_cluster"] = df["spatial_cluster"].replace({3: 2})
        clusters = (0, 1, 2, 4, 5)

    complete = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()

    if split_test == "recent":
        n_test = int(round(test_frac * len(df)))
        cutoff = df["Date"].iloc[len(df) - n_test]
        is_test = (df["Date"] >= cutoff).to_numpy()
        test_mask = is_test & complete
        rest_mask = ~is_test

    elif split_test in {"recent19", "recent20"}:
        year = {"recent19": 2019, "recent20": 2020}[split_test]
        years = df["Date"].dt.year.to_numpy()
        is_test = years == year
        test_mask = is_test & complete
        rest_mask = years < year

    elif split_test == "stratified":
        rng = np.random.default_rng(seed)
        d = df.assign(_complete=complete)
        by_date = d.groupby("Date").agg(
            n_rows=("HAB", "size"),
            all_complete=("_complete", "all"),
            hab=("HAB", "max"),
            season=("Season", "first"),
        )
        elig = by_date[by_date["all_complete"]]
        n_target_dates = int(round(test_frac * len(df) / elig["n_rows"].mean()))
        n_target_dates = min(n_target_dates, len(elig))
        chosen = []
        for _, g in elig.groupby(["season", "hab"]):
            if len(g) <= 1:
                continue
            k = int(round(n_target_dates * len(g) / len(elig)))
            k = min(max(k, 1), len(g) - 1)  # >=1 no test e >=1 no train_val
            pos = ((rng.uniform() + np.arange(k)) * len(g) / k).astype(int)
            chosen.extend(g.index[pos])
        test_mask = df["Date"].isin(chosen).to_numpy()
        rest_mask = ~test_mask

    return clusters, df[rest_mask].copy(), df[test_mask].copy()


def _build_matrices(tr, te, strategy, clusters, one_hot_cluster, sin_doy, season, scale,
                    doy_half_window, impute_alpha, impute_harmonics, impute_use_env):
    """
    Constrói as matrizes de features a partir de DataFrames brutos tr / te.
    Tudo o que é APRENDIDO (imputação, scaling) usa só `tr`; `te` só recebe a transformação.
    É usada tanto no split final train_val/test como em cada fold da validação cruzada.
    """
    tr, te = tr.copy(), te.copy()

    # transformações que não aprendem nada
    for d in (tr, te):
        d["CHL_log"] = np.log(d["CHL"])
        if sin_doy:
            d["sin_doy"] = np.sin(2 * np.pi * _doy365(d["Date"]) / 365)
        if one_hot_cluster:
            for c in clusters:
                d[f"cluster_{c}"] = (d["spatial_cluster"] == c).astype(int)

    # missing values (só em tr; te é sempre completo)
    if strategy == "drop_rows":
        tr = tr.dropna(subset=["CHL_log", "CHL_uncertainty"])
    elif strategy != "drop_features":
        fills = _learn_fills(tr, strategy, doy_half_window, clusters=clusters,
                             impute_alpha=impute_alpha, impute_harmonics=impute_harmonics,
                             impute_use_env=impute_use_env)
        tr = _apply_fills(tr, strategy, fills)

    # colunas finais
    if season == "one_hot":
        for d in (tr, te):
            for sn in ["Winter", "Spring", "Summer", "Fall"]:
                d[f"season_{sn}"] = (d["Season"] == sn).astype(int)
        cols = [f"season_{sn}" for sn in ["Winter", "Spring", "Summer", "Fall"]]
    else:
        cols = []
    cols += ["SST", "UI", "YDOY"]
    if sin_doy:
        cols.append("sin_doy")
    if strategy != "drop_features":
        cols += ["CHL_log", "CHL_uncertainty"]
    cols += ([f"cluster_{c}" for c in clusters]
             if one_hot_cluster else ["spatial_cluster"])

    X_tr, X_te = tr[cols].copy(), te[cols].copy()

    # scaling: parâmetros aprendidos só em tr
    if scale:
        cont = [c for c in CONT_COLS if c in cols]
        mu = X_tr[cont].mean()
        sd = X_tr[cont].std(ddof=0).replace(0, 1.0)
        X_tr[cont] = (X_tr[cont] - mu) / sd
        X_te[cont] = (X_te[cont] - mu) / sd

    return X_tr, tr["HAB"], X_te, te["HAB"], tr["Date"], te["Date"]


def prepare_data(strategy,
    split_test,
    one_hot_cluster=True,
    sin_doy=True,
    season="drop",
    scale=False,
    test_frac=0.15,
    doy_half_window=15,
    seed=42,
    return_dates=False,
    merge_clusters_2_3=True,
    impute_alpha="auto",
    impute_harmonics=2,
    impute_use_env=False,):
    """
    Split FINAL: train_val / test. Para escolher hiperparâmetros e estratégias use
    prepare_folds (validação leave-one-year-out dentro de train_val).

    strategy : estratégia para os missing values (CHL, CHL_uncertainty)
        "drop_rows"         – remove linhas de train_val com missing
        "drop_features"     – remove CHL_log e CHL_uncertainty
        "global_median"     – mediana global (CHL em log)
        "season_median"     – mediana por estação do ano
        "doy_window_median" – mediana numa janela de ±doy_half_window dias
        "ridge_regression"  – regressão Ridge (uma por variável), treinada só nas linhas
                              observadas de train_val, a partir de harmónicas sazonais do DOY
                              + cluster (+ SST/UI opcional). Nunca usa o HAB.

    split_test : forma de separar o conjunto de test
        "recent"     – test_frac mais recentes, sem linhas com missing
        "recent19"   – ano de 2019 completo como test; apenas anos < 2019 no treino
        "recent20"   – ano de 2020 completo como test; apenas anos < 2020 no treino
        "stratified" – ~test_frac do total, só datas completas, espalhadas no tempo
                       e por estações, preservando a mistura de classes

    one_hot_cluster, sin_doy, season, merge_clusters_2_3, impute_* : ver prepare_folds.
    scale : bool – standardiza as contínuas com média/desvio (ddof=0) de train_val.
    test_frac, seed : só para "recent" e "stratified".
    return_dates : se True, devolve também as datas de train_val e test.

    Devolve (X_train_val, y_train_val, X_test, y_test)
    ou, com return_dates=True, (..., dates_train_val, dates_test)
    """
    _check_args(strategy, split_test, season, test_frac, doy_half_window,
                impute_harmonics, impute_alpha)
    clusters, train_val, test = _load_and_split(split_test, test_frac, seed, merge_clusters_2_3)
    Xtv, ytv, Xte, yte, dtv, dte = _build_matrices(
        train_val, test, strategy, clusters, one_hot_cluster, sin_doy, season, scale,
        doy_half_window, impute_alpha, impute_harmonics, impute_use_env)
    out = (Xtv, ytv, Xte, yte)
    if return_dates:
        out += (dtv, dte)
    return out


def prepare_folds(strategy,
    split_test,
    cv="loyo",
    one_hot_cluster=True,
    sin_doy=True,
    season="drop",
    scale=True,
    test_frac=0.15,
    doy_half_window=15,
    seed=42,
    merge_clusters_2_3=True,
    impute_alpha="auto",
    impute_harmonics=2,
    impute_use_env=False,):
    """
    Validação cruzada POR ANO dentro de train_val (o test de prepare_data nunca é tocado).

    cv : "loyo"    – leave-one-year-out: cada ano de train_val é validação uma vez, treino
                     = todos os outros anos de train_val (usa o máximo de dados e de positivos)
         "forward" – expanding window: treino = anos anteriores ao ano de validação
                     (mais realista para 2021-2023, mas folds iniciais ficam com pouco treino;
                     usar só como diagnóstico)

    Em cada fold: validação = só linhas COMPLETAS do ano (como o teste oficial); imputação e
    scaling são aprendidos só no treino do fold (sem fuga de informação).
    Escolher hiperparâmetros com o AUC agregado (concatenar as previsões de todos os folds),
    porque cada ano tem poucos positivos e o AUC por ano é muito instável (2014 e 2017 nem
    têm positivos).

    Devolve uma lista de dicts: year, X_tr, y_tr, X_va, y_va, dates_va
    """
    _check_args(strategy, split_test, season, test_frac, doy_half_window,
                impute_harmonics, impute_alpha)
    if cv not in VALID_CV:
        raise ValueError(f"cv inválido: {cv!r}. Opções: {sorted(VALID_CV)}")
    clusters, train_val, _ = _load_and_split(split_test, test_frac, seed, merge_clusters_2_3)
    years = train_val["Date"].dt.year.to_numpy()
    complete = train_val[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()

    folds = []
    for Y in np.unique(years):
        va_mask = (years == Y) & complete
        tr_mask = (years != Y) if cv == "loyo" else (years < Y)
        if va_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        X_tr, y_tr, X_va, y_va, _, d_va = _build_matrices(
            train_val[tr_mask], train_val[va_mask], strategy, clusters, one_hot_cluster,
            sin_doy, season, scale, doy_half_window, impute_alpha, impute_harmonics,
            impute_use_env)
        folds.append({"year": int(Y), "X_tr": X_tr, "y_tr": y_tr, "X_va": X_va,
                      "y_va": y_va, "dates_va": d_va})
    return folds