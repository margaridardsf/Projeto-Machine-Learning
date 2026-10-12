import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

'''
-----------------------------------------------------------------------------------------------------------------------
Data preparation
'''
# Read data
df = pd.read_excel("data/chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Adicionar codificação sazonal (seno e cosseno do dia do ano)
day_of_year = df["date"].dt.dayofyear
df["s_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
df["c_year"] = np.cos(2 * np.pi * day_of_year / 365.25)



# Definição de parâmetros e variáveis
target_col = "chlorophyll_a_mg_m3"
# Variáveis exógenas (inclui as medições ambientais + variáveis sazonais)
exogenous_cols = [c for c in df.columns if c not in ["date", target_col]]

p = 10  # Lags da variável alvo y (dias passados: y(t-1)...y(t-p))
q = 10  # Lags das variáveis exógenas (medições de t e passadas)

# Criação de Lags usando .shift()

# chlorophyll_df=[]
updated_df=df.copy()

# Variável Alvo (Clorofila-a): Apenas lags >= 1 (Lag 0 NUNCA é usado)
for lag in range(1, p + 1):
    updated_df=pd.concat([updated_df, df[target_col].shift(lag)], axis=1, ignore_index=True)
#    df[f"chlorophyll_lag_{lag}"] = df[target_col].shift(lag)

# Variáveis Exógenas: Lags de 0 a q (Lag 0 representa a medição do próprio dia t)
for col in exogenous_cols:
#    {col}_df=[]
    for lag in range(0, q + 1):
        if lag == 0:
            updated_df=pd.concat([updated_df, df[col]], axis=1, ignore_index=True)
#            df[f"{col}_lag_0"] = df[col]
        else:
            updated_df=pd.concat([updated_df, df[col].shift(lag)], axis=1, ignore_index=True)
#            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

# Determinar r = max(p, q) e remover as primeiras r linhas com NaNs
df_clean = updated_df.dropna().reset_index(drop=True)

print(df.describe(include = "all"))
print(df.columns)


# 5. Definição das matrizes X e Y
feature_cols = [c for c in df_clean.columns if "_lag_" in c]
X = df_clean[feature_cols]
y = df_clean[target_col]

# 6. Divisão Temporal (Treino até final de 2022; Teste em 2023)
train_mask = df_clean["date"] < "2023-01-01"
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# 7. Normalização/Padronização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Modelo de Regressão Linear (ARX)
# fit_intercept=True calcula o parâmetro 'c' (offset) automaticamente
model = LinearRegression(fit_intercept=True)
model.fit(X_train_scaled, y_train)

# 9. Previsão e Avaliação
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"--- Resultados no Conjunto de Teste (2023) ---")
print(f"RMSE : {rmse:.4f} mg/m³")
print(f"MAE  : {mae:.4f} mg/m³")
print(f"R²   : {r2:.4f}")




