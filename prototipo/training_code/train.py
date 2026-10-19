from functions1 import choose_variables_train_data, choose_variables_lags, apply_models, print_results, crosscorr_table

import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_excel(BASE_DIR / "chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Read data
# df = pd.read_excel("chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)



prep = choose_variables_train_data(df, test_year=2023, use_log=True, outlier_mode='Mode')

data = choose_variables_lags(prep, drop_mode='auto', high_corr=0.9, low_target_corr=None)
res = apply_models(data, 'Lasso', scaler_mode='Standard',
                   lag_selection='auto', corr_threshold=0.3, alpha=0.001207)

print_results(res)