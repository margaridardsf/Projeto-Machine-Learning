
from pathlib import Path

import numpy as np
import pandas as pd

import os
import joblib


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import build_lagged_features as blf
from build_lagged_features import DATE, DEFAULT_LAGS, TARGET, build_lagged_frame, read_table

import io
from contextlib import redirect_stdout
from itertools import product







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







'''
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


def outliers(mode, df, x_train, y_train):
    df_clean = df.copy()

    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1

    # Calcula oq tirar tendo em conta o IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # ve i indice dos outliers (valores fora do IQR)
    outliers_ind = x_train & (
        (df_clean['chlorophyll_a_mg_m3'] < lower_bound) | 
        (df_clean['chlorophyll_a_mg_m3'] > upper_bound)
    )

    # poe os dados correspondentes a esses indices sendo NaN
    df_clean.loc[outliers_ind, 'chlorophyll_a_mg_m3'] = np.nan

    # makes a dataset without NaN collumns to do the median, mode, mean on the dataset without outliers
    train_clean = df_clean.loc[x_train, 'chlorophyll_a_mg_m3'].dropna()

    if mode=='Neighbours':
        # Interpola para ter a média do valor anterior e posterior
        df_clean['chlorophyll_a_mg_m3'] = df_clean['chlorophyll_a_mg_m3'].interpolate(method='linear')

    elif mode == 'Mean':
        df_clean.loc[outliers_ind, 'chlorophyll_a_mg_m3'] = train_clean.mean()

    elif mode == 'Median':
        df_clean.loc[outliers_ind, 'chlorophyll_a_mg_m3'] = train_clean.median()

    elif mode == 'Mode':
        # mode() can return several values; take the first
        df_clean.loc[outliers_ind, 'chlorophyll_a_mg_m3'] = train_clean.mode().iloc[0]

    elif mode == 'SameDay':
        dates = df_clean['date']  # already datetime
        base = df_clean['chlorophyll_a_mg_m3'].copy()
        for idx in df_clean.index[outliers_ind]:
            d = dates[idx]
            same_day_prev_years = (
                (dates.dt.month == d.month) &
                (dates.dt.day == d.day) &
                (dates.dt.year < d.year)
            )
            vals = base[same_day_prev_years].dropna()
            df_clean.loc[idx, 'chlorophyll_a_mg_m3'] = vals.mean() if len(vals) > 0 else train_clean.mean()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df_clean



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

    '''
    # keep DataFrame structure (column names and index) if the input had it
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    '''

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



'''
# use just one scaler
X_train_s, X_test_s, scaler = scaling('Standard', X_train, X_test)
print_scaler(scaler)


#use three scalers
results = {}
for mode in ['Standard', 'MinMax', 'Robust']:
    results[mode] = scaling(mode, X_train, X_test)

for mode, (_, _, scaler) in results.items():
    print_scaler(scaler)


# load a saved one later for another dataset
scaler = joblib.load('scalers/scaler_Standard.joblib')
print_scaler(scaler)                 # check the parameters
X_new_scaled = scaler.transform(X_new)   # transform only, never fit
'''

def remove_features(df, features):
    df_clean = df.copy()
    if isinstance(features, str):
        features = [features]

    df_clean = df_clean.drop(columns=features, errors='ignore')

    return df_clean







