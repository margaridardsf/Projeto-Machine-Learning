import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# read data
df = pd.read_excel(
"C:/Users/marga/Documents/tecnico/AAut/proj1/ML_LAB_PART_1_FILES_V2/student_release/data/chlorophyll_student_2015_2023.xlsx",
sheet_name="daily_data"
)

# order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)




# 2. Seasonal Features
day_of_year = df["date"].dt.dayofyear
df["s_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
df["c_year"] = np.cos(2 * np.pi * day_of_year / 365.25)

# 3. Create Lags
target_col = "chlorophyll_a_mg_m3"
exogenous_cols = [c for c in df.columns if c not in ["date", target_col]]

p, q = 10, 10

# Target Lags (Lag >= 1)
for lag in range(1, p + 1):
    df[f"chlorophyll_lag_{lag}"] = df[target_col].shift(lag)

# Exogenous Lags (Lag >= 0)
for col in exogenous_cols:
    for lag in range(0, q + 1):
        if lag == 0:
            df[f"{col}_lag_0"] = df[col]
        else:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

# Remove NaNs caused by shifting
df_clean = df.dropna().reset_index(drop=True)

# 4. Feature and Target Setup
feature_cols = [c for c in df_clean.columns if "_lag_" in c]
X = df_clean[feature_cols]
y = df_clean[target_col]

# 5. Chronological Train/Test Split
train_mask = df_clean["date"] < "2023-01-01"
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# 6. Feature Scaling (Crucial for regularized models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Define Models & Alpha Range to Test
alphas = np.logspace(-3, 3, 100)  # Tests values from 0.001 to 1000
tscv = TimeSeriesSplit(n_splits=5)  # Time-series safe cross-validation

models = {
    "OLS Linear Regression": LinearRegression(fit_intercept=True),
    "Ridge Regression (L2)": RidgeCV(alphas=alphas, cv=tscv),
    "Lasso Regression (L1)": LassoCV(
        alphas=alphas, cv=tscv, max_iter=10000, random_state=42
    ),
}

# 8. Train and Evaluate Each Model
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    # Print selected alpha penalty value
    if hasattr(model, "alpha_"):
        print(f"[{name}] Best Alpha: {model.alpha_:.4f}")

# 9. Print Model Comparison Table
comparison_df = pd.DataFrame(results).T
print("\n--- Model Performance Comparison (2023 Test Set) ---")
print(comparison_df.round(4))




###################




# To inspect which features Lasso selected (kept non-zero)

lasso_model = models["Lasso Regression (L1)"]
selected_features = pd.Series(
    lasso_model.coef_, index=feature_cols
).abs().sort_values(ascending=False)

print("\nTop 10 Most Important Features in Lasso:")
print(selected_features[selected_features > 0].head(10))

print(f"\nLasso zeroed out {sum(lasso_model.coef_ == 0)} out of {len(feature_cols)} features.")












