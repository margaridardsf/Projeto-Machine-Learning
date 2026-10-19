from functions1 import choose_variables_train_data, choose_variables_lags, apply_models, print_results, crosscorr_table

import pandas as pd

from pathlib import Path

import numpy as np

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

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

import build_lagged_features as blf
from build_lagged_features import DATE, DEFAULT_LAGS, TARGET, build_lagged_frame, read_table

import io
from contextlib import redirect_stdout
from itertools import product



from functions import metrics, extract_lags_ols, lags_restrict, data_preprocessing, scaling, print_scaler

# Read data
df = pd.read_excel("chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)



prep = choose_variables_train_data(df, test_year=2023, use_log=True, outlier_mode='Mode')

data = choose_variables_lags(prep, drop_mode='auto', high_corr=0.9, low_target_corr=None)
res = apply_models(data, 'Lasso', scaler_mode='Standard',
                   lag_selection='auto', corr_threshold=0.3, alpha=0.001207)

print_results(res)