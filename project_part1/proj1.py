import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Import functions and constants from build_lagged_features.py
from build_lagged_features import (
    DATE,
    DEFAULT_LAGS,
    TARGET,
    build_lagged_frame,
    read_table,
)


# Read data
df = pd.read_excel("data/chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# Build the dataset (with default lags)
df_lagged = build_lagged_frame(df, lag_spec=DEFAULT_LAGS)

# Separate features (X) and target (y)
# Features include all lagged variables plus annual_sin and annual_cos
feature_cols = [c for c in df_lagged.columns if c not in [DATE, TARGET]]

X = df_lagged[feature_cols]
y = df_lagged[TARGET]

# Chronological Train/Test Split (Train < 2023, Test >= 2023)
train_mask = df_lagged[DATE] < "2023-01-01"

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]









