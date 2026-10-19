import os
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error)

from build_lagged_features import DATE, DEFAULT_LAGS, TARGET, build_lagged_frame, read_table

DATE = 'date'
TARGET = 'chlorophyll_a_mg_m3'

'''
------------------------------------------------------------------------------------------------------------------------------
HELPER FUNCTIONS
'''

# ---------- metrics ----------
def metrics(y_true, y_pred, verbose=True):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    if verbose:
        print(f"Metrics\n"
              f"R2: {r2}\n"
              f"MAE: {mae}\n"
              f"RMSE: {rmse}\n"
              f"MSE: {mse}\n"
              f"MAPE: {mape}")
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'MAPE': mape}


def _evaluate(model, X_test, y_test_orig, use_log):
    """Predicts, undoes the log (if used) and computes the metrics in ORIGINAL units."""
    y_pred = model.predict(X_test)
    if use_log:
        y_pred = np.exp(y_pred)
    return metrics(y_test_orig, y_pred, verbose=False)


# ---------- outliers ----------
def outliers(mode, df, train_mask, y_train):
    df_clean = df.copy()

    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # outliers only in the train rows
    outliers_mask = train_mask & (
        (df_clean[TARGET] < lower_bound) | (df_clean[TARGET] > upper_bound)
    )

    df_clean.loc[outliers_mask, TARGET] = np.nan

    # train values without outliers, to compute the mean / median / mode
    train_clean = df_clean.loc[train_mask, TARGET].dropna()

    if mode == 'Neighbours':
        df_clean[TARGET] = df_clean[TARGET].interpolate(method='linear', limit_direction='both')

    elif mode == 'Mean':
        df_clean.loc[outliers_mask, TARGET] = train_clean.mean()

    elif mode == 'Median':
        df_clean.loc[outliers_mask, TARGET] = train_clean.median()

    elif mode == 'Mode':
        df_clean.loc[outliers_mask, TARGET] = train_clean.mode().iloc[0]

    elif mode == 'SameDay':
        dates = df_clean[DATE]
        base = df_clean[TARGET].copy()
        for idx in df_clean.index[outliers_mask]:
            d = dates[idx]
            same_day_prev_years = (
                (dates.dt.month == d.month) &
                (dates.dt.day == d.day) &
                (dates.dt.year < d.year)
            )
            vals = base[same_day_prev_years].dropna()
            df_clean.loc[idx, TARGET] = vals.mean() if len(vals) > 0 else train_clean.mean()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df_clean


# ---------- removing variables ----------
def remove_features(df, features):
    df_clean = df.copy()
    if isinstance(features, str):
        features = [features]

    return df_clean.drop(columns=features, errors='ignore')


def auto_drop_by_corr(corr, high_corr=None, low_target_corr=None):
    """
    Automatic version of reading the correlation matrix by hand.
      low_target_corr: drop variables with |corr with target| below this value
      high_corr:       for pairs with |corr| above this value, keep the one more
                       correlated with the target and drop the other
    None disables a rule. Returns the list of variable names to drop.
    """
    target_corr = corr[TARGET].drop(TARGET).abs().fillna(0)
    to_drop = set()

    if low_target_corr is not None:
        to_drop |= set(target_corr[target_corr < low_target_corr].index)

    if high_corr is not None:
        kept = []
        # strongest relation with the target first, so those are the ones that survive
        for var in target_corr.sort_values(ascending=False).index:
            if var in to_drop:
                continue
            if any(abs(corr.loc[var, k]) > high_corr for k in kept):
                to_drop.add(var)
            else:
                kept.append(var)

    return sorted(to_drop)


# ---------- lag selection ----------
def crosscorr_lag_scores(X_train, y_train):
    """|correlation| between the target and each lag column (= cross-correlation at that lag),
    train rows only. Sorted from strongest to weakest."""
    lag_cols = [c for c in X_train.columns if "__lag_" in c]
    return X_train[lag_cols].corrwith(y_train).abs().fillna(0).sort_values(ascending=False)


def crosscorr_table(data):
    """data['crosscorr'] as a table: one row per variable, one column per lag (nothing truncated)."""
    cc = data['crosscorr']
    parts = cc.index.to_series().str.split("__lag_", expand=True)
    table = pd.DataFrame({'variable': parts[0].values, 'lag': parts[1].astype(int).values,
                          'corr': cc.values})
    return table.pivot(index='variable', columns='lag', values='corr').round(3)


# Extrai e devolve as lags com p-value < alpha (apenas para o ols)
def extract_lags_ols(X_train, y_train, alpha=0.05, verbose=True):
    ols_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    p_values = ols_sm.pvalues.drop('const', errors='ignore')
    significant_lags = p_values[p_values < alpha].index.tolist()

    if verbose:
        print(f"Total de coeficientes analisados: {len(p_values)}")
        print(f"Lags significativas encontradas (p < {alpha}): {len(significant_lags)}")

    return significant_lags


# Restringe os dados às lags selecionadas (funciona em qualquer df que já tenha as colunas "__lag_")
def lags_restrict(df, selected_lags):
    if isinstance(selected_lags, str):
        selected_lags = [selected_lags]
    non_lag_cols = [col for col in df.columns if "__lag_" not in col]
    cols_to_keep = non_lag_cols + [col for col in selected_lags if col in df.columns]

    return df[cols_to_keep].copy()


# ---------- scaling ----------
def scaling(mode, X_train, X_test, save_dir='scalers'):
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler(),
    }

    if mode not in scalers:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(scalers)}")

    scaler = scalers[mode]

    # fit on train only, so no test information leaks into the scaling
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # save: scalers/scaler_Standard.joblib, scalers/scaler_MinMax.joblib, ...
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(save_dir, f'scaler_{mode}.joblib'))

    return X_train_scaled, X_test_scaled, scaler


def print_scaler(scaler, feature_names=None):
    if isinstance(scaler, StandardScaler):
        params = {'mean': scaler.mean_, 'std': scaler.scale_}
    elif isinstance(scaler, MinMaxScaler):
        params = {'data_min': scaler.data_min_, 'data_max': scaler.data_max_}
    elif isinstance(scaler, RobustScaler):
        params = {'median': scaler.center_, 'IQR': scaler.scale_}
    else:
        raise ValueError("Unsupported scaler type")

    if feature_names is None:
        feature_names = getattr(scaler, 'feature_names_in_', None)

    print(f"--- {type(scaler).__name__} parameters ---")
    print(pd.DataFrame(params, index=feature_names))


# ---------- models ----------
def fit_model(model_name, X_train, y_train, tscv=None, alpha=None):
    """OLS, or Ridge/Lasso.
    alpha=None -> alpha chosen by cross-validation (RidgeCV / LassoCV, TimeSeriesSplit)
    alpha=value -> that alpha is used as it is (e.g. the one chosen by the grid search)."""
    if model_name == 'OLS':
        return LinearRegression().fit(X_train, y_train)

    if model_name not in ('Ridge', 'Lasso'):
        raise ValueError(f"Unknown model: {model_name}")

    if alpha is not None:
        est = Ridge(alpha=alpha) if model_name == 'Ridge' else Lasso(alpha=alpha, max_iter=20000, random_state=42)
        return est.fit(X_train, y_train)

    if tscv is None:
        # 4 validation blocks of 365 rows (1 year if the data is daily), always after the train part
        tscv = TimeSeriesSplit(n_splits=4, test_size=365)

    if model_name == 'Ridge':
        return RidgeCV(alphas=np.logspace(-2, 6, 400), cv=tscv,
                       scoring='neg_mean_squared_error').fit(X_train, y_train)

    return LassoCV(alphas=np.logspace(-5, 1, 300), cv=tscv,
                   max_iter=20000, random_state=42).fit(X_train, y_train)



'''
------------------------------------------------------------------------------------------------------------------------------
Passos
'''

# ---------- Passo 1 ----------

def choose_variables_train_data(df, test_year, use_log=False, outlier_mode=None):
    """
    - discards everything after test_year
    - applies the log to the whole target column right away (so it never has to be checked again)
    - removes outliers (train rows only; the IQR is computed after the log)
    - computes the correlation matrix on train rows only

    Returns a dict ("prep") for step 2:
        df          cleaned dataframe (log scale if use_log)
        df_orig     same rows, original target (used for the final metrics)
        train_mask  boolean mask on `df` (date < test_year-01-01)
        corr        correlation matrix, train rows only
    """
    df = df[df[DATE] < f'{test_year + 1}-01-01'].sort_values(DATE).reset_index(drop=True)
    df_orig = df.copy()
    train_mask = df[DATE] < f'{test_year}-01-01'

    df_prep = df.copy()
    if use_log:
        if (df_prep[TARGET] <= 0).any():
            raise ValueError("Target has values <= 0, np.log is not defined. Use log1p/expm1 instead.")
        df_prep[TARGET] = np.log(df_prep[TARGET])

    if outlier_mode is not None:
        y_train = df_prep.loc[train_mask, TARGET]
        df_prep = outliers(outlier_mode, df_prep, train_mask, y_train)

    corr = df_prep.loc[train_mask].drop(columns=DATE).corr(numeric_only=True)

    return {'df': df_prep, 'df_orig': df_orig, 'train_mask': train_mask, 'corr': corr,
            'test_year': test_year, 'use_log': use_log, 'outlier_mode': outlier_mode}


# ---------- Passo 2 ----------

def choose_variables_lags(prep, lag_spec=None, drop_mode='manual',
                          drop_vars=None, high_corr=None, low_target_corr=None):
    """
    Which variables are dropped (the two modes are exclusive):
      drop_mode='manual': drops exactly the variables in `drop_vars` (the ones you chose
                          after reading prep['corr']). The thresholds must not be given.
      drop_mode='auto':   drops variables automatically from prep['corr'] (train only):
                            high_corr:        for pairs with |corr| above this, keep the one more
                                              correlated with the target
                            low_target_corr:  drop variables with |corr with target| below this
                          None disables a rule. `drop_vars` must not be given.
                          This is the mode for the grid search.

    Then builds the lags with build_lagged_frame, removes the NaN rows created by the lags,
    rebuilds the train/test masks and splits X / y.

    Returns a dict ("data") for step 3. data['crosscorr'] is the |corr| of every lag column
    with the target (train rows only), so you can look at which lags matter.
    """
    if drop_mode == 'manual':
        if high_corr is not None or low_target_corr is not None:
            raise ValueError("drop_mode='manual' only uses drop_vars; "
                             "high_corr / low_target_corr belong to drop_mode='auto'.")
        dropped = list(drop_vars or [])
    elif drop_mode == 'auto':
        if drop_vars:
            raise ValueError("drop_mode='auto' decides by itself; drop_vars belongs to drop_mode='manual'.")
        dropped = auto_drop_by_corr(prep['corr'], high_corr, low_target_corr)
    else:
        raise ValueError("drop_mode must be 'manual' or 'auto'")
    dropped = sorted(set(dropped) - {TARGET, DATE})   # never the target or the date

    lag_spec = DEFAULT_LAGS if lag_spec is None else lag_spec
    df, df_orig, test_year = prep['df'], prep['df_orig'], prep['test_year']

    df = remove_features(df, dropped)

    # a dropped variable must also disappear from the lag spec (assumes lag_spec is a dict
    # {variable: [lags]}; if yours has another format, adapt this line)
    if isinstance(lag_spec, dict):
        lag_spec = {k: v for k, v in lag_spec.items() if k not in dropped}

    # --- lags (the first rows have NaN lags, so they are dropped)
    df_lagged = build_lagged_frame(df, lag_spec=lag_spec).dropna().reset_index(drop=True)

    # --- masks rebuilt, because rows were dropped
    train_mask = df_lagged[DATE] < f'{test_year}-01-01'
    test_mask = ~train_mask          # data after test_year was already discarded in step 1

    X = df_lagged.drop(columns=[DATE, TARGET])
    y = df_lagged[TARGET]
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # original (not logged, not cleaned) test target, matched by date
    dates_test = df_lagged.loc[test_mask, DATE]
    y_test_orig = df_orig.set_index(DATE)[TARGET].loc[dates_test].values

    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'y_test_orig': y_test_orig, 'crosscorr': crosscorr_lag_scores(X_train, y_train),
            'dropped_vars': dropped, 'drop_mode': drop_mode, 'lag_spec': lag_spec,
            'use_log': prep['use_log'], 'outlier_mode': prep['outlier_mode']}


# ---------- Passo 3 ----------

def apply_models(data, model, use_ttest=False, ttest_alpha=0.05, ttest_scope='lags',
                 lag_selection=None, selected_lags=None, corr_threshold=None,
                 scaler_mode='Standard', alpha=None, save_dir=None, tscv=None, verbose=False):
    """
    model:          'OLS', 'Ridge' or 'Lasso'

    OLS:            fit on the default lags -> metrics_before
                    if use_ttest: keep the significant features, refit -> metrics_after
                    ttest_scope='lags': only lag columns are filtered (same as lags_restrict)
                    ttest_scope='all' : every feature must be significant
    Ridge / Lasso:  scaler fitted on train only; alpha chosen by RidgeCV / LassoCV
                    (TimeSeriesSplit), or fixed if you pass alpha=... -> metrics_before
                    lag_selection (the two modes are exclusive):
                      'manual': keep the lag columns in `selected_lags` (you chose them after
                                reading data['crosscorr']), refit -> metrics_after
                      'auto':   keep the lag columns with |corr with target| >= corr_threshold,
                                refit -> metrics_after   (the mode for the grid search)
                    non-lag variables are always kept.
    use_ttest is ignored for Ridge/Lasso, lag_selection is ignored for OLS.

    The log (if used) is undone before the metrics, which are always computed against
    the original test target.
    """
    if model not in ('OLS', 'Ridge', 'Lasso'):
        raise ValueError(f"Unknown model: {model}")

    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test_orig, use_log = data['y_train'], data['y_test_orig'], data['use_log']

    def fit_eval(cols, save):
        X_tr, X_te = X_train[cols], X_test[cols]
        scaler = None
        if model != 'OLS':
            X_tr, X_te, scaler = scaling(scaler_mode, X_tr, X_te, save_dir if save else None)
        m = fit_model(model, X_tr, y_train, tscv, alpha)
        return m, scaler, _evaluate(m, X_te, y_test_orig, use_log)

    # --- which features go into the second fit (None -> no second fit)
    all_cols = list(X_train.columns)
    selected_cols, extra = None, {}

    if model == 'OLS' and use_ttest:
        significant = extract_lags_ols(X_train, y_train, ttest_alpha, verbose=verbose)
        if ttest_scope == 'lags':
            # only lag columns go to lags_restrict, which already keeps every non-lag column
            significant_lags = [c for c in significant if "__lag_" in c]
            selected_cols = list(lags_restrict(X_train, significant_lags).columns)
        else:
            selected_cols = significant
        extra['significant'] = significant

    elif model != 'OLS' and lag_selection is not None:
        if lag_selection == 'manual':
            if not selected_lags:
                raise ValueError("lag_selection='manual' needs selected_lags (list of lag column names).")
            if corr_threshold is not None:
                raise ValueError("corr_threshold belongs to lag_selection='auto'.")
            missing = [c for c in selected_lags if c not in X_train.columns]
            if missing:
                raise ValueError(f"These lag columns do not exist in the data: {missing}")
            chosen = list(selected_lags)
        elif lag_selection == 'auto':
            if selected_lags:
                raise ValueError("selected_lags belongs to lag_selection='manual'.")
            if corr_threshold is None:
                raise ValueError("lag_selection='auto' needs corr_threshold.")
            scores = data['crosscorr']
            chosen = scores[scores >= corr_threshold].index.tolist()
        else:
            raise ValueError("lag_selection must be None, 'manual' or 'auto'")
        selected_cols = list(lags_restrict(X_train, chosen).columns)

    if selected_cols is not None and len(selected_cols) == 0:
        print("Warning: selection removed every feature, skipping the second fit.")
        selected_cols = None

    # --- first fit (the scaler is only saved for the final fit)
    m, scaler, met_before = fit_eval(all_cols, save=(selected_cols is None))
    features, met_after = all_cols, None

    # --- second fit with the selected features
    if selected_cols is not None:
        m, scaler, met_after = fit_eval(selected_cols, save=True)
        features = selected_cols

    result = {
        'model_name': model,
        'model': m,
        'scaler': scaler,
        'features': features,
        'n_features_before': len(all_cols),
        'lag_spec': data['lag_spec'],
        'use_log': use_log,
        'outlier_mode': data['outlier_mode'],
        'dropped_vars': data['dropped_vars'],
        'drop_mode': data['drop_mode'],
        'lag_selection': lag_selection if model != 'OLS' else None,
        'alpha': alpha if alpha is not None else getattr(m, 'alpha_', None),   # fixed, or chosen by RidgeCV / LassoCV
        'metrics_before': met_before,
        'metrics_after': met_after,
    }
    result.update(extra)

    if model == 'Lasso':
        result['lasso_kept'] = [f for f, c in zip(features, m.coef_) if c != 0]

    return result


'''
------------------------------------------------------------------------------------------------------------------------------
output
'''

def print_results(res):
    print(f"Model: {res['model_name']} | log: {res['use_log']} | outliers: {res['outlier_mode']} "
          f"| features: {res['n_features_before']} -> {len(res['features'])}")
    print(f"Dropped variables ({res['drop_mode']}): {res['dropped_vars']}")
    if res['lag_selection'] is not None:
        print(f"Lag selection: {res['lag_selection']}")
    if res['alpha'] is not None:
        print(f"Alpha: {res['alpha']:.6g}")

    table = pd.DataFrame({'before': res['metrics_before']})
    if res['metrics_after'] is not None:
        table['after'] = pd.Series(res['metrics_after'])
    print(table)

    if 'lasso_kept' in res:
        print(f"Lasso kept {len(res['lasso_kept'])} of {len(res['features'])} features")
    if res['scaler'] is not None:
        print_scaler(res['scaler'])


def save_bundle(res, path):
    """Saves everything needed to reproduce the model on another dataset."""
    keys = ['model_name', 'model', 'scaler', 'features', 'lag_spec', 'use_log', 'dropped_vars']
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump({k: res[k] for k in keys}, path)
    print(f"Bundle saved to {path}")