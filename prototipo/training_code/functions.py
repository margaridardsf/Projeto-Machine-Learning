import numpy as np
import pandas as pd

import os
import joblib
import statsmodels.api as sm

from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from build_lagged_features import DATE, DEFAULT_LAGS, TARGET, build_lagged_frame, read_table



# Métricas para anilar a previsão feita pelo modelo
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"Metrics\n"
          f"R2: {r2}\n"
          f"MAE: {mae}\n"
          f"RMSE: {rmse}\n"
          f"MSE: {mse}\n"
          f"MAPE: {mape}")





# Extrai e devolve as lags com p-value < alpha (apenas para o ols)
def extract_lags_ols(X_train, y_train, alpha=0.05):
    ols_sm = sm.OLS(y_train, sm.add_constant(X_train)).fit()
    p_values = ols_sm.pvalues.drop('const', errors='ignore')
    significant_lags = p_values[p_values < alpha].index.tolist()

    print(f"Total de coeficientes analisados: {len(p_values)}")
    print(f"Lags significativas encontradas (p < {alpha}): {len(significant_lags)}")

    return significant_lags








# Restringe os dados às lags selecionadas
# ATENÇÃO: esta função pressupõe que o df recebido como argumento já passou previamente pela build_lagged_frame
def lags_restrict(df, selected_lags):
    if isinstance(selected_lags, str):
        selected_lags = [selected_lags]
    non_lag_cols = [col for col in df.columns if "__lag_" not in col]
    cols_to_keep = non_lag_cols + [col for col in selected_lags if col in df.columns]
    
    return df[cols_to_keep].copy()












### Esta função faz:
# atribui lags (lags default se não for escrito nada);
# aplica log à clorofila;
# limpa os outliers;
# limpa variáveis exógenas escolhidas;
# faz split tendo em conta o ano de teste test_year (maioritariamente 2023 ou 2022).

# A função devolve X_train, y_train, X_test, y_test_orig (os dados estão prontos para serem aplicados os modelos)
def data_preprocessing(df, selected_lags = None, test_year = 2023, log_y = False, clean_outliers = False, clean_features = None):
    df_clean = df.copy()
    df_orig = df.copy()
    
    # log
    if log_y:
        df_clean['chlorophyll_a_mg_m3'] = np.log(df_clean['chlorophyll_a_mg_m3'])

    # outliers
    if clean_outliers:
        train_mask_out = df_clean['date'] < f'{test_year}-01-01'
        y_train_out = df_clean.loc[train_mask_out, 'chlorophyll_a_mg_m3']

        Q1 = y_train_out.quantile(0.25)
        Q3 = y_train_out.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = train_mask_out & (
            (df_clean['chlorophyll_a_mg_m3'] < lower_bound) | 
            (df_clean['chlorophyll_a_mg_m3'] > upper_bound)
        )

        df_clean.loc[outliers_mask, 'chlorophyll_a_mg_m3'] = np.nan
        df_clean['chlorophyll_a_mg_m3'] = df_clean['chlorophyll_a_mg_m3'].interpolate(method='linear')

    # features
    if clean_features is not None:
        if isinstance(clean_features, str):
            clean_features = [clean_features]

        df_clean = df_clean.drop(columns=clean_features, errors='ignore')

    # lags
    active_lag_spec = {col: lags for col, lags in DEFAULT_LAGS.items() if col in df_clean.columns}
    df_lagged = build_lagged_frame(df_clean, lag_spec=active_lag_spec)
    if selected_lags is not None:
        df_lagged = lags_restrict(df_lagged, selected_lags)

    # split
    train_mask = df_lagged['date'] < f'{test_year}-01-01'
    test_mask = (df_lagged['date'] >= f'{test_year}-01-01') & (df_lagged['date'] <= f'{test_year}-12-31')

    X_train = df_lagged.loc[train_mask].drop(columns=['date', 'chlorophyll_a_mg_m3'], errors='ignore')
    y_train = df_lagged.loc[train_mask, 'chlorophyll_a_mg_m3']

    X_test = df_lagged.loc[test_mask].drop(columns=['date', 'chlorophyll_a_mg_m3'], errors='ignore')
    
    dates_test = df_lagged.loc[test_mask, 'date']
    y_test_orig = df_orig.loc[df_orig['date'].isin(dates_test), 'chlorophyll_a_mg_m3'].values

    return X_train, y_train, X_test, y_test_orig





def remove_features(df, features):
    df_clean = df.copy()
    if isinstance(features, str):
        features = [features]

    df_clean = df_clean.drop(columns=features, errors='ignore')

    return df_clean







