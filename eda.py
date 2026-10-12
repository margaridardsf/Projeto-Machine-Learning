import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
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


# Lista de variáveis numéricas do dataset
num_vars = [
    'chlorophyll_a_mg_m3', 'sst_c', 'par_umol_m2_s', 'nitrate_umol_l',
    'wind_speed_m_s', 'upwelling_index', 'mixed_layer_depth_m', 'salinity_psu',
    'current_speed_m_s', 'river_discharge_index', 'cloud_fraction',
    'surface_pressure_hpa', 'turbidity_ntu'
]

# Configuração de estilo do Seaborn
sns.set_theme(style="whitegrid")

# ==============================================================================
# VISUALIZAÇÃO 1: Grelha de Boxplots Individuais (Escalas Originais)
# ==============================================================================
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(num_vars):
    sns.boxplot(
        y=df[col], 
        ax=axes[i], 
        color='skyblue', 
        flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.6)
    )
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')

# Remover os subplots não utilizados na grelha 4x4 (temos 13 variáveis)
for j in range(len(num_vars), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Identificação de Outliers por Variável (Escalas Originais)', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.show()

# ==============================================================================
# VISUALIZAÇÃO 2: Boxplot Conjunto Normalizado (Z-Score)
# ==============================================================================
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[num_vars].dropna()), columns=num_vars)

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=df_scaled, 
    orient='v', 
    palette='Set3',
    flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5)
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel('Desvios-Padrão em relação à Média (Z-Score)', fontsize=12)
plt.title('Boxplot Conjunto Padronizado para Comparação da Magnitude de Outliers', fontsize=15, fontweight='bold')
plt.axhline(y=3, color='r', linestyle='--', alpha=0.7, label='Limite Z = +3')
plt.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='Limite Z = -3')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


















print("--- Resumo da Estrutura dos Dados ---")
print(f"Número de registos: {len(df)}")
print(f"Intervalo de datas: {df['date'].min().date()} até {df['date'].max().date()}")
print(f"Valores em falta por coluna:\n{df.isnull().sum()}\n")

# 2. Codificação Sazonal
day_of_year = df['date'].dt.dayofyear
df['s_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['c_year'] = np.cos(2 * np.pi * day_of_year / 365.25)

# 3. Análise da Variável Alvo (Clorofila-a)
plt.figure(figsize=(14, 5))
plt.plot(df['date'], df['chlorophyll_a_mg_m3'], color='forestgreen', linewidth=1)
plt.title('Série Temporal da Concentração de Clorofila-a (2015–2023)')
plt.xlabel('Data')
plt.ylabel('Chlorophyll-a (mg/m³)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Teste de Estacionariedade (ADF Test)
adf_result = adfuller(df['chlorophyll_a_mg_m3'].dropna(), result_object=True)
print(f"Estatística ADF: {adf_result[0]:.4f}")
print(f"p-value ADF: {adf_result[1]:.4e}")


# 4. ACF e PACF da Clorofila-a
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['chlorophyll_a_mg_m3'].dropna(), lags=30, ax=axes[0], title='ACF - Clorofila-a')
plot_pacf(df['chlorophyll_a_mg_m3'].dropna(), lags=30, ax=axes[1], title='PACF - Clorofila-a')
plt.tight_layout()
plt.show()

# 5. Matriz de Correlação entre Variáveis Exógenas e Alvo
exogenous_cols = [
    'sst_c', 'par_umol_m2_s', 'nitrate_umol_l', 'wind_speed_m_s',
    'upwelling_index', 'mixed_layer_depth_m', 'salinity_psu',
    'current_speed_m_s', 'river_discharge_index', 'cloud_fraction',
    'surface_pressure_hpa', 'turbidity_ntu'
]

corr_matrix = df[['chlorophyll_a_mg_m3'] + exogenous_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlação de Pearson')
plt.tight_layout()
plt.show()

# 6. Correlação Cruzada com Lags das Exógenas (para selecionar lags no ARX)
lags = range(0, 8)
target = df['chlorophyll_a_mg_m3']

lag_corrs = pd.DataFrame(index=exogenous_cols, columns=[f'lag_{k}' for k in lags])

for col in exogenous_cols:
    for k in lags:
        lag_corrs.loc[col, f'lag_{k}'] = target.corr(df[col].shift(k))

plt.figure(figsize=(10, 6))
sns.heatmap(lag_corrs.astype(float), annot=True, fmt=".2f", cmap='viridis')
plt.title('Correlação Cruzada entre Clorofila-a $y(t)$ e $Exógenas(t-k)$')
plt.xlabel('Atraso (Lags em dias)')
plt.tight_layout()
plt.show()