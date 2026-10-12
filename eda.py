import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

'''
-----------------------------------------------------------------------------------------------------------------------
Data preparation
'''
# Read data
df = pd.read_excel("data/chlorophyll_student_2015_2023.xlsx", sheet_name="daily_data")

# Order dates
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


'''
-----------------------------------------------------------------------------------------------------------------------
Boxplots
'''

# Boxplots Individuais
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

plt.suptitle('Boxplots per Variable', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig("images/boxplots")
plt.close()


# Boxplot Conjunto Normalizado (Z-Score)
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
plt.ylabel('Deviation', fontsize=12)
plt.title('Normalized Boxplots per Variable (Z-Score)', fontsize=15, fontweight='bold')
plt.axhline(y=3, color='r', linestyle='--', alpha=0.7, label='Limit Z = +3')
plt.axhline(y=-3, color='r', linestyle='--', alpha=0.7, label='Limit Z = -3')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("images/boxplots_normalizados")
plt.close()


'''
-----------------------------------------------------------------------------------------------------------------------
Results and Codification
'''

print("--- Resumo da Estrutura dos Dados ---")
print(f"Número de registos: {len(df)}")
print(f"Intervalo de datas: {df['date'].min().date()} até {df['date'].max().date()}")
print(f"Valores em falta por coluna:\n{df.isnull().sum()}\n")

# Codificação Sazonal
day_of_year = df['date'].dt.dayofyear
df['s_year'] = np.sin(2 * np.pi * day_of_year / 365.25)
df['c_year'] = np.cos(2 * np.pi * day_of_year / 365.25)


'''
-----------------------------------------------------------------------------------------------------------------------
Analysis of Clorofila-a 
'''

# Clorofila-a - Analise da Variável (line plot)
plt.figure(figsize=(14, 5))
plt.plot(df['date'], df['chlorophyll_a_mg_m3'], color='forestgreen', linewidth=1)
plt.title('Temporal Trend Plot for Clorofila-a (2015–2023)')
plt.xlabel('Data')
plt.ylabel('Chlorophyll-a (mg/m³)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/Clorofila-a_Temporal_Trend_Plot")
plt.close()


# Clorofila-a - Histogram
plt.figure(figsize=(10, 5))
plt.hist(
    df['chlorophyll_a_mg_m3'].dropna(), 
    bins=70, 
    color='forestgreen', 
    edgecolor='black', 
    alpha=0.7
)
plt.title('Histogram of Clorofila-a')
plt.xlabel('Chlorophyll-a (mg/m³)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/Clorofila-a_histograma")
plt.close()


# Clorofila-a - Histogram with better fit distributions
# Prepare the data
data = df['chlorophyll_a_mg_m3'].dropna()
plt.figure(figsize=(10, 5))
# Plot histogram with density=True so the total area equals 1 (matching the PDF curves)
count, bins, ignored = plt.hist(
    data, 
    bins=50, 
    density=True, 
    color='forestgreen', 
    edgecolor='black', 
    alpha=0.5,
    label=' Observed Data (Histogram)'
)
# Generate a smooth range of x values for the fitted curves
x = np.linspace(data.min(), data.max(), 200)
# Fit and plot Lognormal distribution
shape_ln, loc_ln, scale_ln = stats.lognorm.fit(data, floc=0)
pdf_lognorm = stats.lognorm.pdf(x, shape_ln, loc_ln, scale_ln)
plt.plot(x, pdf_lognorm, 'r-', linewidth=2, label='Lognormal Fit')
# Fit and plot Gamma distribution
shape_g, loc_g, scale_g = stats.gamma.fit(data, floc=0)
pdf_gamma = stats.gamma.pdf(x, shape_g, loc_g, scale_g)
plt.plot(x, pdf_gamma, 'b--', linewidth=2, label='Gamma Fit')
plt.title('Histogram for Clorofila-a with Distributions')
plt.xlabel('Chlorophyll-a (mg/m³)')
plt.ylabel('Frequancy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/Clorofila-a_histograma_distributions")
plt.close()


'''
-----------------------------------------------------------------------------------------------------------------------
Histograms
'''

# Histograms of all variables
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(num_vars):
    sns.histplot(
        df[col].dropna(), 
        ax=axes[i], 
        color='forestgreen', 
        bins=30, 
        edgecolor='black',
        alpha=0.6
    )
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')
# Remove unused subplots in the 4x4 grid
for j in range(len(num_vars), len(axes)):
    fig.delaxes(axes[j])
plt.suptitle('Histograms per Variable', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig("images/histograms")
plt.close()


# Histograms of all variables with best fit distributions
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18, 14))
axes = axes.flatten()

for i, col in enumerate(num_vars):
    data = df[col].dropna()
    # Plot density histogram
    axes[i].hist(data, bins=30, density=True, color='forestgreen', edgecolor='black', alpha=0.4)
    # Generate smooth curve points
    xmin, xmax = axes[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    try:
        # Fit Lognormal if strictly positive, else Normal
        if (data > 0).all():
            shape, loc, scale = stats.lognorm.fit(data, floc=0)
            p = stats.lognorm.pdf(x, shape, loc, scale)
        else:
            mu, std = stats.norm.fit(data)
            p = stats.norm.pdf(x, mu, std)
        axes[i].plot(x, p, 'r-', linewidth=2)
    except Exception:
        pass  # Skip if fitting fails for a specific edge-case column
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')
for j in range(len(num_vars), len(axes)):
    fig.delaxes(axes[j])
plt.suptitle('Histograms with Best-Fit Distributions per Variable', fontsize=16, fontweight='bold', y=0.99)
plt.tight_layout()
plt.savefig("images/histograms_distributions")
plt.close()


'''
-----------------------------------------------------------------------------------------------------------------------
ACF e PACF
'''

# Teste de Estacionariedade (ADF Test)
adf_result = adfuller(df['chlorophyll_a_mg_m3'].dropna(), result_object=True)
print(f"Estatística ADF: {adf_result[0]:.4f}")
print(f"p-value ADF: {adf_result[1]:.4e}")


# ACF e PACF da Clorofila-a
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(df['chlorophyll_a_mg_m3'].dropna(), lags=30, ax=axes[0], title='ACF - Clorofila-a')
plot_pacf(df['chlorophyll_a_mg_m3'].dropna(), lags=30, ax=axes[1], title='PACF - Clorofila-a')
plt.tight_layout()
plt.savefig("images/ACF_PACF")
plt.close()


'''
-----------------------------------------------------------------------------------------------------------------------
Correlation matrix and Scatter plot
'''

# Matriz de Correlacao entre Variaveis Exógenas e Alvo
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
plt.savefig("images/matrix_correlacao")
plt.close()


# Generate a pairplot matrix for all numerical variables
pairplot_fig = sns.pairplot(
    df[num_vars].dropna(), 
    diag_kind='kde', 
    plot_kws={'alpha': 0.4, 's': 10, 'color': 'forestgreen'},
    diag_kws={'color': 'forestgreen'}
)

pairplot_fig.fig.suptitle('Scatter Plot Matrix', y=1.02, fontsize=16, fontweight='bold')
pairplot_fig.savefig("images/scatterplot.png", dpi=300)
plt.close()


'''
-----------------------------------------------------------------------------------------------------------------------
Correlation with Lags
'''


# Correlacao Cruzada com Lags das Exogenas (para selecionar lags no ARX)
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
plt.savefig("images/correlacao_cruzada_lags")
plt.close()
