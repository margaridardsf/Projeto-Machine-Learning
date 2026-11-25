import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
VALID_STRATEGIES = {
    "drop_rows",
    "drop_features",
    "global_median",
    "season_median",
    "doy_window_median",
    "ridge_regression",
}
# "none" = sem conjunto de test: train_val fica com TODAS as linhas (usar para CV em todos os anos
# e para o ajuste final do modelo).
VALID_SPLITS = {"recent", "recent19", "recent20", "stratified", "none"}
VALID_SEASONS = {"one_hot", "drop"}
VALID_CV = {"loyo", "forward"}

FILL_COLS = ["CHL_log", "CHL_uncertainty"]
CONT_COLS = ["SST", "UI", "YDOY", "sin_doy", "CHL_log", "CHL_uncertainty"]

RIDGE_ALPHA_GRID = (1, 10, 30, 100, 300, 1000)
RIDGE_DEFAULT_ALPHA = 100.0   # usado se só houver 1 ano disponível para escolher alpha


# ----------------------------------------------------------------------------
# Funções Auxiliares
# ----------------------------------------------------------------------------
def _doy365(date):
    """Dia do ano com 366 -> 365 (ano bissexto), como no cos de YDOY."""
    return np.minimum(date.dt.dayofyear.to_numpy(), 365)


# --- Imputação por regressão Ridge (implementada explicitamente em NumPy) ---
def _impute_design(d, clusters, n_harmonics, use_env):
    """
    Matriz de features do imputador (só variáveis sempre observadas; NUNCA o HAB):
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
    """Escolhe alpha por leave-one-year-out dentro dos dados de treino."""
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


def _learn_fills(train, strategy, doy_half_window, clusters=(0, 1, 2, 4, 5),
                 impute_alpha="auto", impute_harmonics=2, impute_use_env=False):
    """Aprende, SÓ nas linhas de `train`, as medianas / modelos usados na imputação."""
    obs = train.dropna(subset=FILL_COLS)
    fills = {"global": obs[FILL_COLS].median()}
    if strategy == "season_median":
        fills["season"] = obs.groupby("Season")[FILL_COLS].median()
    if strategy == "doy_window_median":
        fills["obs_doy"] = _doy365(obs["Date"])
        fills["obs_vals"] = obs[FILL_COLS].to_numpy()
        fills["half_window"] = doy_half_window
    if strategy == "ridge_regression":
        X = _impute_design(obs, clusters, impute_harmonics, impute_use_env)
        years = obs["Date"].dt.year.to_numpy()
        fills["design"] = (tuple(clusters), impute_harmonics, impute_use_env)
        fills["ridge"], fills["bounds"] = {}, {}
        for c in FILL_COLS:                      # um modelo por variável a imputar
            y = obs[c].to_numpy(float)
            a = (_select_alpha(X, y, years, RIDGE_ALPHA_GRID)
                 if impute_alpha == "auto" else float(impute_alpha))
            fills["ridge"][c] = _ridge_fit(X, y, a)
            fills["bounds"][c] = (y.min(), y.max())   # previsões limitadas ao intervalo observado
    return fills


def _apply_fills(d, strategy, fills):
    """Preenche CHL_log e CHL_uncertainty em falta (imputação em log(CHL))."""
    d = d.copy()
    miss = d[FILL_COLS].isna().any(axis=1).to_numpy()
    if not miss.any():
        return d

    if strategy == "global_median":
        for c in FILL_COLS:
            d[c] = d[c].fillna(fills["global"][c])

    elif strategy == "season_median":
        for c in FILL_COLS:
            by_season = d["Season"].map(fills["season"][c])
            d[c] = d[c].fillna(by_season).fillna(fills["global"][c])

    elif strategy == "ridge_regression":
        X = _impute_design(d, *fills["design"])
        for c in FILL_COLS:
            lo, hi = fills["bounds"][c]
            pred = np.clip(_ridge_predict(fills["ridge"][c], X), lo, hi)
            d[c] = d[c].fillna(pd.Series(pred, index=d.index))

    elif strategy == "doy_window_median":
        doy = _doy365(d["Date"])
        for i in np.flatnonzero(miss):
            diff = np.abs(fills["obs_doy"] - doy[i])
            diff = np.minimum(diff, 365 - diff)           # distância circular
            in_win = diff <= fills["half_window"]
            for j, c in enumerate(FILL_COLS):
                if np.isnan(d[c].iloc[i]):
                    v = (np.median(fills["obs_vals"][in_win, j])
                         if in_win.any() else fills["global"][c])
                    d.iloc[i, d.columns.get_loc(c)] = v
    return d


def _validate_args(strategy, split_test, season, test_frac, doy_half_window,
                   impute_alpha, impute_harmonics, cv="loyo"):
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy inválida: {strategy!r}. Opções: {sorted(VALID_STRATEGIES)}")
    if split_test not in VALID_SPLITS:
        raise ValueError(f"split_test inválido: {split_test!r}. Opções: {sorted(VALID_SPLITS)}")
    if season not in VALID_SEASONS:
        raise ValueError(f"season inválido: {season!r}. Opções: {sorted(VALID_SEASONS)}")
    if cv not in VALID_CV:
        raise ValueError(f"cv inválido: {cv!r}. Opções: {sorted(VALID_CV)}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac deve estar entre 0 e 1 (recebido: {test_frac})")
    if doy_half_window < 0:
        raise ValueError(f"doy_half_window deve ser >= 0 (recebido: {doy_half_window})")
    if not (isinstance(impute_harmonics, (int, np.integer)) and impute_harmonics >= 0):
        raise ValueError(f"impute_harmonics deve ser um inteiro >= 0 (recebido: {impute_harmonics!r})")
    if not (impute_alpha == "auto" or (isinstance(impute_alpha, (int, float)) and impute_alpha >= 0)):
        raise ValueError(f"impute_alpha deve ser 'auto' ou um número >= 0 (recebido: {impute_alpha!r})")


def _load_raw(merge_clusters_2_3):
    """Lê, ordena e funde clusters. A coluna _obs diz se CHL e CHL_uncertainty foram observados."""
    df = pd.read_excel("classification_task1_train.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "spatial_cluster"], kind="stable").reset_index(drop=True)

    clusters = (0, 1, 2, 3, 4, 5)
    if merge_clusters_2_3:
        df["spatial_cluster"] = df["spatial_cluster"].replace({3: 2})
        clusters = (0, 1, 2, 4, 5)

    df["_obs"] = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1)
    return df, clusters


def _split_raw(df, split_test, test_frac, seed):
    """Separa os dados de test dos restantes (train_val). Devolve (train_val, test) em bruto."""
    complete = df["_obs"].to_numpy()

    if split_test == "none":
        test_mask = np.zeros(len(df), dtype=bool)
        rest_mask = ~test_mask

    elif split_test == "recent":
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

    else:  # "stratified"
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

    return df[rest_mask].copy(), df[test_mask].copy()


def _featurize(train, evals, strategy, clusters, one_hot_cluster, sin_doy, season, scale,
               doy_half_window, impute_alpha, impute_harmonics, impute_use_env):
    """
    Constrói as features. TUDO o que é aprendido (medianas / ridge da imputação, mín/máx
    do scaling) vem só das linhas de `train`; as linhas de `evals` (lista de
    DataFrames, ex.: [val] ou [test]) apenas recebem a transformação.

    Devolve (X_train, y_train, dates_train, [X_eval...], [y_eval...], [dates_eval...]).
    """
    train = train.copy()
    evals = [e.copy() for e in evals]

    # log(CHL), sin_doy, one-hot do cluster (não aprendem nada -> igual em todos os conjuntos)
    for d in [train] + evals:
        d["CHL_log"] = np.log(d["CHL"])
        if sin_doy:
            d["sin_doy"] = np.sin(2 * np.pi * _doy365(d["Date"]) / 365)
        if one_hot_cluster:
            for c in clusters:
                d[f"cluster_{c}"] = (d["spatial_cluster"] == c).astype(int)

    # Estratégia para missing values (só em train; os conjuntos de avaliação são completos)
    if strategy == "drop_rows":
        train = train.dropna(subset=FILL_COLS)
    elif strategy != "drop_features":
        fills = _learn_fills(train, strategy, doy_half_window, clusters=clusters,
                             impute_alpha=impute_alpha, impute_harmonics=impute_harmonics,
                             impute_use_env=impute_use_env)
        train = _apply_fills(train, strategy, fills)

    # Colunas finais (as restantes são redundantes e são descartadas)
    if season == "one_hot":
        for d in [train] + evals:
            for sn in ["Winter", "Spring", "Summer", "Fall"]:
                d[f"season_{sn}"] = (d["Season"] == sn).astype(int)
        cols = [f"season_{sn}" for sn in ["Winter", "Spring", "Summer", "Fall"]]
    else:
        cols = []
    cols += ["SST", "UI", "YDOY"]
    if sin_doy:
        cols.append("sin_doy")
    if strategy != "drop_features":
        cols += FILL_COLS
    cols += ([f"cluster_{c}" for c in clusters]
             if one_hot_cluster else ["spatial_cluster"])

    for e in evals:
        assert not e[cols].isna().any().any(), "conjunto de avaliação com NaN"
    assert not train[cols].isna().any().any(), "conjunto de treino com NaN"

    X_train, X_evals = train[cols].copy(), [e[cols].copy() for e in evals]

    # Scaling das features contínuas: parâmetros aprendidos só em train
    if scale:
        cont = [c for c in CONT_COLS if c in cols]
        lo = X_train[cont].min()
        rng = (X_train[cont].max() - lo).replace(0, 1.0)
        X_train[cont] = (X_train[cont] - lo) / rng
        for X in X_evals:
            X[cont] = (X[cont] - lo) / rng

    return (X_train, train["HAB"], train["Date"],
            X_evals, [e["HAB"] for e in evals], [e["Date"] for e in evals])


# ----------------------------------------------------------------------------
# Função principal
# ----------------------------------------------------------------------------
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
    Separa train_val / test e constrói as features. Para validação cruzada dentro de train_val
    (escolha de modelo / hiperparâmetros) usar cv_folds(), que refaz a imputação e o scaling
    em cada fold.

    strategy : estratégia para os missing values (CHL, CHL_uncertainty)
        "drop_rows"         – remove linhas de train_val com missing
        "drop_features"     – remove CHL_log e CHL_uncertainty
        "global_median"     – mediana global (CHL em log)
        "season_median"     – mediana por estação do ano
        "doy_window_median" – mediana numa janela de ±doy_half_window dias
        "ridge_regression"  – regressão Ridge (uma por variável: CHL_log e CHL_uncertainty),
                              treinada só nas linhas observadas de train_val, a partir de
                              harmónicas sazonais do DOY + cluster (+ SST/UI opcional).
                              Nunca usa o HAB. Ver parâmetros impute_*.

    split_test : forma de separar o conjunto de test
        "none"       – sem test: train_val = todas as linhas (X_test/y_test vêm vazios).
                       É a opção recomendada para escolher modelo por CV em todos os anos
                       e para o ajuste final.
        "recent"     – test_frac mais recentes, sem linhas com missing
        "recent19"   – ano de 2019 completo como test; apenas anos < 2019 no treino
        "recent20"   – ano de 2020 completo como test; apenas anos < 2020 no treino
        "stratified" – ~test_frac do total, só datas completas, espalhadas no tempo
                       e por estações, preservando a mistura de classes

    one_hot_cluster : bool – spatial_cluster -> 5 colunas 0/1 (se merge_clusters_2_3=True)
                             ou 6 colunas 0/1 (se merge_clusters_2_3=False)
    sin_doy : bool – acrescenta sin(2π·DOY/365)
    season : str – tratamento de Season
        "one_hot" – 4 colunas 0/1 (season_Winter/Spring/Summer/Fall)
        "drop"    – remove Season (YDOY e sin_doy já capturam a sazonalidade)
    scale : bool – escala (min-max, [0, 1]) as variáveis contínuas (SST, UI, YDOY, sin_doy,
        CHL_log, CHL_uncertainty) com mínimo e máximo de train_val.
        Atenção: se fores dividir train_val em train/val por conta própria, usa scale=False
        e aplica o scaling dentro de cada fold – ou, melhor, usa cv_folds().
    test_frac : float – fração aproximada de dados para test (em "recent" e "stratified")
    doy_half_window : int – meia-janela em dias para "doy_window_median"
    seed : int – semente aleatória para "stratified"
    return_dates : bool – se True, devolve também as datas (Series) de train_val e test
    merge_clusters_2_3 : bool – funde o cluster espacial 3 no cluster 2
    impute_alpha : "auto" ou float – penalização do Ridge (só strategy="ridge_regression").
        "auto" escolhe, para cada variável, o melhor valor de {1,10,30,100,300,1000} por
        leave-one-year-out dentro dos dados de treino.
    impute_harmonics : int – nº de harmónicas sazonais (sin/cos de k·2π·DOY/365) no imputador
    impute_use_env : bool – se True, o imputador usa também SST e UI (por defeito False:
        na validação não melhoravam a previsão de CHL)

    Devolve:
    (X_train_val, y_train_val, X_test, y_test)
    ou, com return_dates=True:
    (X_train_val, y_train_val, X_test, y_test, dates_train_val, dates_test)
    """
    _validate_args(strategy, split_test, season, test_frac, doy_half_window,
                   impute_alpha, impute_harmonics)

    df, clusters = _load_raw(merge_clusters_2_3)
    train_val, test = _split_raw(df, split_test, test_frac, seed)

    X_tv, y_tv, d_tv, (X_te,), (y_te,), (d_te,) = _featurize(
        train_val, [test], strategy, clusters, one_hot_cluster, sin_doy, season, scale,
        doy_half_window, impute_alpha, impute_harmonics, impute_use_env)

    out = (X_tv, y_tv, X_te, y_te)
    if return_dates:
        out += (d_tv, d_te)
    return out


def cv_folds(strategy,
    split_test="none",
    cv="loyo",
    min_train_years=1,
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
    Folds de validação cruzada POR ANO dentro de train_val, sem fugas de informação.

    Em cada fold (ano Y):
      - validação = linhas COMPLETAS (CHL observado) de Y – tal como o test oficial;
      - treino    = todas as linhas dos outros anos (cv="loyo", leave-one-year-out)
                    ou só dos anos anteriores a Y (cv="forward", janela expansiva);
      - imputação (medianas / ridge), alpha do ridge e scaling são aprendidos SÓ nas
        linhas de treino do fold e depois aplicados à validação.
    Anos sem linhas completas são saltados. Anos sem nenhum HAB=1 (2014 e 2017) ficam como
    folds de validação: não têm AUC próprio, mas contam para o AUC agregado out-of-fold.

    split_test : conjunto de test que fica de fora (igual ao de prepare_data). Com "none"
        a CV usa todos os anos; com "recent20" só usa os anos <= 2019, etc.
    cv : "loyo" – treina em todos os outros anos (mais dados; recomendado para escolher modelo)
         "forward" – treina só no passado (mais realista para 2021-2023, mas folds iniciais
                     com muito pouco treino -> mais ruído)
    min_train_years : nº mínimo de anos distintos no treino (só relevante em "forward")

    Devolve uma lista de dicts com: val_year, X_train, y_train, X_val, y_val, dates_val.
    """
    _validate_args(strategy, split_test, season, test_frac, doy_half_window,
                   impute_alpha, impute_harmonics, cv)

    df, clusters = _load_raw(merge_clusters_2_3)
    train_val, _ = _split_raw(df, split_test, test_frac, seed)

    years = train_val["Date"].dt.year.to_numpy()
    obs = train_val["_obs"].to_numpy()

    folds = []
    for yr in np.unique(years):
        va_mask = (years == yr) & obs
        if not va_mask.any():
            continue
        tr_mask = (years != yr) if cv == "loyo" else (years < yr)
        if len(np.unique(years[tr_mask])) < min_train_years:
            continue
        X_tr, y_tr, _, (X_va,), (y_va,), (d_va,) = _featurize(
            train_val[tr_mask], [train_val[va_mask]], strategy, clusters, one_hot_cluster,
            sin_doy, season, scale, doy_half_window, impute_alpha, impute_harmonics,
            impute_use_env)
        folds.append({"val_year": int(yr), "X_train": X_tr, "y_train": y_tr,
                      "X_val": X_va, "y_val": y_va, "dates_val": d_va})
    return folds