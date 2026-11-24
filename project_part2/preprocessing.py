import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Funções Auxiliares
# ----------------------------------------------------------------------------
def _doy365(date):
    """Dia do ano com 366 -> 365 (ano bissexto), como no cos de YDOY."""
    return np.minimum(date.dt.dayofyear.to_numpy(), 365)


def _learn_fills(train_val, strategy, doy_half_window):
    """Aprende, SÓ em train_val, as medianas usadas na imputação."""
    fill_cols = ["CHL_log", "CHL_uncertainty"]
    obs = train_val.dropna(subset=fill_cols)
    fills = {"global": obs[fill_cols].median()}
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
def prepare_data(strategy, split_test, one_hot_cluster=True, sin_doy=True, season = "drop",
                scale = False, test_frac=0.15, doy_half_window=15, seed=42, return_dates=False, merge_clusters_2_3=True):
    """
    strategy : estratégia para os missing values (CHL, CHL_uncertainty)
        "drop_rows"         – remove linhas de train/val com missing
        "drop_features"     – remove CHL_log e CHL_uncertainty
        "global_median"     – mediana global (CHL em log)
        "season_median"     – mediana por estação do ano
        "doy_window_median" – mediana numa janela de ±doy_half_window dias

    split_test : forma de separar o conjunto de test
        "recent"     – 15% mais recentes, sem linhas com missing
        "stratified" – 15% do total, só linhas completas, espalhadas no tempo
                       e por estações, preservando a mistura de classes
    one_hot_cluster : bool – spatial_cluster -> 6 colunas 0/1
    sin_doy : bool – acrescenta sin(2π·DOY/365)
    season : str – tratamento de Season
        "one_hot" – 4 colunas 0/1 (season_Winter/Spring/Summer/Fall)
        "drop"    – remove Season (YDOY e sin_doy já capturam a sazonalidade)
    scale : bool – (por defeito False) standardiza as variáveis contínuas (SST, UI, YDOY, sin_doy,
        CHL_log, CHL_uncertainty) com média e desvio (ddof=0) de train_val.
        Não afeta Season nem as colunas one-hot.
        Usar scale=False para modelos de árvores, se se preferir.

    return_dates : bool – se True, devolve também as datas (Series) de
        train_val e de test, para permitir splits por data (ex.: train/val).

    
    Devolve:
    (X_train_val, y_train_val, X_test, y_test)

    ou, com return_dates=True,
    (X_train_val, y_train_val, X_test, y_test, dates_train_val, dates_test)
    """

    # Read data
    df = pd.read_excel("classification_task1_train.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "spatial_cluster"], kind="stable").reset_index(drop=True)

    # Fusão dos clusters
    clusters = (0, 1, 2, 3, 4, 5)
    if merge_clusters_2_3:
        df["spatial_cluster"] = df["spatial_cluster"].replace({3: 2})
        clusters = (0, 1, 2, 4, 5)

    # Separar os dados de test dos restantes dados
    if split_test == "recent":
        n_test = int(round(test_frac * len(df)))
        cutoff = df["Date"].iloc[len(df) - n_test]
        is_test = (df["Date"] >= cutoff).to_numpy()
        complete = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()
        test_mask = is_test & complete      # linhas completas do test
        rest_mask = ~is_test    

    elif split_test in {"recent20": 2020, "recent19": 2019}:
        year = {"recent20": 2020, "recent19": 2019}[split_test]
        is_test = (df["Date"].dt.year == year).to_numpy()
        complete = df[["CHL", "CHL_uncertainty"]].notna().all(axis=1).to_numpy()
        test_mask = is_test & complete     
        rest_mask = ~is_test

    else:
        rng = np.random.default_rng(seed)
        d = df.assign(_complete=df[["CHL", "CHL_uncertainty"]].notna().all(axis=1))

        by_date = d.groupby("Date").agg(n_rows=("HAB", "size"),
                                        all_complete=("_complete", "all"),
                                        hab=("HAB", "max"),
                                        season=("Season", "first"))
        elig = by_date[by_date["all_complete"]]

        n_target_dates = int(round(test_frac * len(df) / elig["n_rows"].mean()))
        n_target_dates = min(n_target_dates, len(elig))

        chosen = []
        for _, g in elig.groupby(["season", "hab"]):
            k = int(round(n_target_dates * len(g) / len(elig)))
            k = min(max(k, 1), len(g))                                        # >=1 data por estrato
            pos = ((rng.uniform() + np.arange(k)) * len(g) / k).astype(int)   # posições espaçadas
            chosen.extend(g.index[pos])

        test_mask = df["Date"].isin(chosen).to_numpy()    
        rest_mask = ~test_mask

    train_val, test = df[rest_mask].copy(), df[test_mask].copy()



    # log(CHL), sin_doy, one-hot do cluster
    # (aplicadas igualmente a train_val e test, para as colunas coincidirem) 
    # pode-se fazer isto porque não aprende nada
    train_val = train_val.copy()
    test = test.copy()
    train_val["CHL_log"] = np.log(train_val["CHL"]) 
    test["CHL_log"] = np.log(test["CHL"])                    
    if sin_doy:
        train_val["sin_doy"] = np.sin(2 * np.pi * _doy365(train_val["Date"]) / 365)
        test["sin_doy"] = np.sin(2 * np.pi * _doy365(test["Date"]) / 365)
    if one_hot_cluster:
        for c in clusters:                       
            train_val[f"cluster_{c}"] = (train_val["spatial_cluster"] == c).astype(int)
            test[f"cluster_{c}"] = (test["spatial_cluster"] == c).astype(int)


    # Estratégia para missing values (só para train_val, porque o test já é completo)
    if strategy == "drop_rows":
        train_val = train_val.dropna(subset=["CHL_log", "CHL_uncertainty"])
    elif strategy != "drop_features":
        fills = _learn_fills(train_val, strategy, doy_half_window)
        train_val = _apply_fills(train_val, strategy, fills)

    # Colunas finais (as restantes são redundantes e são descartadas)
    if season == "one_hot":
        for d in (train_val, test):
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

    assert not test[cols].isna().any().any(), "test com NaN"
    assert not train_val[cols].isna().any().any(), "train_val com NaN"


    X_train_val, X_test = train_val[cols].copy(), test[cols].copy()

    # Scaling das features contínuas: parâmetros aprendidos só em train_val
    if scale:
        cont = [c for c in ["SST", "UI", "YDOY", "sin_doy", "CHL_log", "CHL_uncertainty"] if c in cols]
        mu = X_train_val[cont].mean()
        sd = X_train_val[cont].std(ddof=0).replace(0, 1.0)
        X_train_val[cont] = (X_train_val[cont] - mu) / sd
        X_test[cont] = (X_test[cont] - mu) / sd

    out = (X_train_val, train_val["HAB"], X_test, test["HAB"])

    if return_dates:
        out += (train_val["Date"], test["Date"])
    return out