from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from build_lagged_features import DATE, DEFAULT_LAGS, TARGET, build_lagged_frame, read_table

from sklearn.preprocessing import StandardScaler


import pickle


def lags_restrict(df, selected_lags):
    if isinstance(selected_lags, str):
        selected_lags = [selected_lags]
    non_lag_cols = [col for col in df.columns if "__lag_" not in col]
    cols_to_keep = non_lag_cols + [col for col in selected_lags if col in df.columns]
    
    return df[cols_to_keep].copy()






BASE_DIR = Path(__file__).resolve().parent
df = pd.read_excel(BASE_DIR / "chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Read data
# df = pd.read_excel("chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Apply log(chlorophyll_a_mg_m3)
df['chlorophyll_a_mg_m3'] = np.log(df['chlorophyll_a_mg_m3'])


# Outliers treatment
Q1 = df['chlorophyll_a_mg_m3'].quantile(0.25)
Q3 = df['chlorophyll_a_mg_m3'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (df['chlorophyll_a_mg_m3'] < lower_bound) | (df['chlorophyll_a_mg_m3'] > upper_bound)

df.loc[outliers_mask, 'chlorophyll_a_mg_m3'] = np.nan
train_clean = df['chlorophyll_a_mg_m3'].dropna()
df.loc[outliers_mask, 'chlorophyll_a_mg_m3'] = train_clean.mode().iloc[0]


# Clean features with high correlation between themselves
df = df.drop(columns=['par_umol_m2_s'], errors='ignore')

# Filter Lags
active_lag_spec = {col: lags for col, lags in DEFAULT_LAGS.items() if col in df.columns}
df_lagged = build_lagged_frame(df, lag_spec=active_lag_spec)

lag_cols = [c for c in df_lagged.columns if "__lag_" in c]

scores = df_lagged[lag_cols].corrwith(df_lagged['chlorophyll_a_mg_m3']).abs().fillna(0)

selected_lags = scores[scores >= 0.3].index.tolist()

df = lags_restrict(df_lagged, selected_lags)


# Scaler
feature_cols = [c for c in df.columns if c not in ['date', 'chlorophyll_a_mg_m3']]
X = df[feature_cols]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Lasso
lasso = Lasso(alpha=0.001207, max_iter=10000, random_state=42)

lasso.fit(X, df['chlorophyll_a_mg_m3'])



# Guardar o modelo
output_dir = BASE_DIR.parent / "best_model"
output_dir.mkdir(parents=True, exist_ok=True)
model_pkl_path = output_dir / "model.pkl"

artifact = {
    "model": lasso,
    "scaler": scaler,
    "feature_cols": feature_cols,       # Ordem exata das colunas usadas no fit
    "active_lag_spec": active_lag_spec, # Dicionário de lags ativos
    "log_y": True                       # Flag para reverter com np.exp na previsão
}

with open(model_pkl_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Modelo guardado em: {model_pkl_path}")




'''

# processamento dos dados (experimentar várias combinações)
X_train, y_train, X_test, y_test_orig = data_preprocessing(
    df = df,                          
    selected_lags = None, 
    test_year = 2022,            
    log_y = True,                     
    clean_outliers = False,          
    clean_features = ['par_umol_m2_s']
)




scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

tscv = TimeSeriesSplit(n_splits=4, test_size=365)


# Lasso
lasso = LassoCV(
    alphas=np.logspace(-5, 1, 300), 
    cv=tscv, 
    max_iter=20000, 
    random_state=42
    )
lasso.fit(X_train_scaled, y_train)
'''