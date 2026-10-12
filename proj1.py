import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



# read data
df = pd.read_excel(
"C:/Users/marga/Documents/tecnico/AAut/proj1/ML_LAB_PART_1_FILES_V2/student_release/data/chlorophyll_student_2015_2023.xlsx",
sheet_name="daily_data"
)

# order dates
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


# Visualize data:

df.head()

print(df.describe())













