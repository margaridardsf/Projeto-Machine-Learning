"""
baseline_functions.py - Funções da baseline da Task 1 (previsão de HAB)
=======================================================================
Este módulo só DEFINE funções e constantes; a sua aplicação (treino dos modelos,
métricas e gráficos) está no notebook `baseline_task1.ipynb`.

Decisões de desenho (enunciado, secções 3, 6 e 7):
  * Validação CRONOLÓGICA (expanding window): para cada ano de validação treina-se
    com todos os anos anteriores. O teste final é 2021-2023.
  * A validação usa apenas linhas COMPLETAS (o teste do professor é completo).
  * Duas estratégias para os valores em falta no TREINO:
        "complete_only"  - treinar só com linhas completas;
        "median_impute"  - imputar com a mediana do treino (feito à mão, sem
                           SimpleImputer, que é proibido).
  * Pré-processamento explícito (sem Pipeline/ColumnTransformer/imputers):
    log1p(CHL), one-hot do cluster, mediana e StandardScaler ajustados SÓ ao
    treino de cada fold.
  * Desbalanceamento (~13% de positivos): class_weight="balanced" (LogReg, SVM,
    RF) e priors 0.5/0.5 (Naive Bayes). O kNN não tem equivalente.
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:                                    # dentro do Jupyter mostra tabelas formatadas
    from IPython.display import display
except ImportError:                     # fora do Jupyter faz simplesmente print
    def display(obj):
        print(obj)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
THRESHOLD = 0.5                 # limiar das métricas dependentes de limiar
VAL_YEARS = [2018, 2019, 2020]  # cada fold valida num ano e treina em todos os anteriores

CLUSTERS = [0, 1, 2, 3, 4, 5]
CONT_COLS = ["SST", "UI", "YDOY", "log_CHL", "CHL_uncertainty"]  # colunas a normalizar

MODEL_ORDER = ["kNN", "NaiveBayes", "SVM_RBF", "RandomForest", "LogReg"]
STRATEGIES = ["complete_only", "median_impute"]


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def make_model(name):
    """Devolve um modelo novo (não treinado)."""
    if name == "kNN":
        # sem class_weight: a limiar 0.5 tende a prever quase só a classe 0
        return KNeighborsClassifier(n_neighbors=15, weights="uniform")
    if name == "NaiveBayes":
        # priors iguais = equivalente ao "balanced"; as colunas one-hot não são gaussianas
        return GaussianNB(priors=[0.5, 0.5])
    if name == "SVM_RBF":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                   probability=True, random_state=RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                      class_weight="balanced_subsample",
                                      random_state=RANDOM_STATE, n_jobs=-1)
    if name == "LogReg":
        return LogisticRegression(class_weight="balanced", max_iter=2000,
                                  random_state=RANDOM_STATE)
    raise ValueError(f"Modelo desconhecido: {name}")


# --------------------------------------------------------------------------- #
# Dados e features (tudo explícito)
# --------------------------------------------------------------------------- #
def load_data(path):
    """Lê o .xlsx e ordena cronologicamente (ordenação explícita)."""
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Date", "spatial_cluster"]).reset_index(drop=True)


def build_features(df):
    """
    Matriz de features X (DataFrame) a partir do DataFrame bruto.
    - Sem Date/Day/Month/Year (o teste está fora do intervalo de anos do treino)
    - Sem Season (redundante com YDOY)
    - CHL tem cauda longa -> log1p
    - spatial_cluster é categórica -> one-hot com colunas fixas (mesma ordem sempre)
    """
    X = pd.DataFrame(index=df.index)
    X["SST"] = df["SST"]
    X["UI"] = df["UI"]
    X["YDOY"] = df["YDOY"]
    X["log_CHL"] = np.log1p(df["CHL"])
    X["CHL_uncertainty"] = df["CHL_uncertainty"]
    for c in CLUSTERS:
        X[f"cluster_{c}"] = (df["spatial_cluster"] == c).astype(float)
    return X


def prepare_data(path):
    """Lê o ficheiro e devolve {"df": dados brutos, "X": features, "y": alvo HAB}."""
    df = load_data(path)
    return {"df": df, "X": build_features(df), "y": df["HAB"]}


def describe_data(data):
    """Resumo rápido do conjunto de dados."""
    df, X, y = data["df"], data["X"], data["y"]
    print(f"Linhas: {len(df)} | prevalência HAB: {y.mean():.3f} | "
          f"linhas completas: {X.notna().all(axis=1).sum()}")


# --------------------------------------------------------------------------- #
# Pré-processamento (aprendido só no treino)
# --------------------------------------------------------------------------- #
def apply_preprocessing(X, state):
    """Aplica imputação (se existir) e normalização aprendidas no treino."""
    X = X.copy()
    if state["medians"] is not None:
        X = X.fillna(state["medians"])
    X[CONT_COLS] = state["scaler"].transform(X[CONT_COLS])
    return X.to_numpy(dtype=float)


def fit_preprocessing(X_train, y_train, strategy):
    """Aprende mediana/scaler SÓ com o treino. Devolve (X_np, y_np, state)."""
    if strategy == "complete_only":
        mask = X_train.notna().all(axis=1)
        X_train, y_train = X_train[mask], y_train[mask]
        medians = None
    elif strategy == "median_impute":
        medians = X_train[CONT_COLS].median()
        X_train = X_train.fillna(medians)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")

    scaler = StandardScaler().fit(X_train[CONT_COLS])
    state = {"medians": medians, "scaler": scaler}
    return apply_preprocessing(X_train, state), y_train.to_numpy(), state


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, proba, threshold=THRESHOLD):
    """AUC, PR-AUC e métricas dependentes de limiar + matriz de confusão."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": roc_auc_score(y_true, proba),
        "PR_AUC": average_precision_score(y_true, proba),
        "BalAcc": balanced_accuracy_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# --------------------------------------------------------------------------- #
# Validação cronológica
# --------------------------------------------------------------------------- #
def evaluate_model(model_name, strategy, data, results):
    """
    Validação cronológica de um modelo com uma estratégia de valores em falta.
    Guarda o resultado em results[(strategy, model_name)] e mostra a tabela por fold.
    """
    df, X_all, y_all = data["df"], data["X"], data["y"]
    rows, y_parts, p_parts = [], [], []

    for val_year in VAL_YEARS:
        train_idx = df["Year"] < val_year
        # validação só com linhas completas (como o teste do professor)
        val_idx = (df["Year"] == val_year) & X_all.notna().all(axis=1)

        X_tr, y_tr, state = fit_preprocessing(X_all[train_idx], y_all[train_idx], strategy)
        X_va = apply_preprocessing(X_all[val_idx], state)
        y_va = y_all[val_idx].to_numpy()

        model = make_model(model_name)            # modelo novo em cada fold
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]   # P(classe 1)

        rows.append({"val_year": val_year, "n_train": len(y_tr), "n_val": len(y_va),
                     "n_pos_val": int(y_va.sum()), **compute_metrics(y_va, proba)})
        y_parts.append(y_va)
        p_parts.append(proba)

    folds = pd.DataFrame(rows)
    results[(strategy, model_name)] = {
        "folds": folds, "y": np.concatenate(y_parts), "p": np.concatenate(p_parts)}

    print(f"{model_name}  |  estratégia: {strategy}")
    display(folds[["val_year", "n_train", "n_val", "n_pos_val", "AUC", "BalAcc",
                   "Accuracy", "Recall", "Precision", "F1"]].round(3))


# --------------------------------------------------------------------------- #
# Resumos e gráficos
# --------------------------------------------------------------------------- #
def show_strategy_metrics(strategy, results):
    """Métricas agregadas, AUC por fold e curvas ROC/PR de todos os modelos de uma estratégia."""
    names = [m for m in MODEL_ORDER if (strategy, m) in results]

    # 1) Métricas agregadas (previsões out-of-fold de todos os anos de validação)
    pooled = pd.DataFrame({m: compute_metrics(results[(strategy, m)]["y"],
                                              results[(strategy, m)]["p"]) for m in names}).T
    cols = ["AUC", "PR_AUC", "BalAcc", "Accuracy", "Recall", "Precision",
            "Specificity", "F1", "TP", "FP", "FN", "TN"]
    print(f"Métricas agregadas - {strategy} (anos {VAL_YEARS}, limiar={THRESHOLD})")
    display(pooled[cols].astype({"TP": int, "FP": int, "FN": int, "TN": int})
            .round(3).sort_values("AUC", ascending=False))

    # 2) AUC por fold + média e desvio-padrão
    auc_folds = pd.DataFrame({m: results[(strategy, m)]["folds"].set_index("val_year")["AUC"]
                              for m in names}).T
    auc_folds["média"] = auc_folds.mean(axis=1)
    auc_folds["desvio"] = auc_folds[VAL_YEARS].std(axis=1)
    print("AUC por ano de validação")
    display(auc_folds.round(3).sort_values("média", ascending=False))

    # 3) Curvas ROC e Precision-Recall
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    for m in names:
        y, p = results[(strategy, m)]["y"], results[(strategy, m)]["p"]
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax_roc.plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(y, p):.3f})")
        ax_pr.plot(rec, prec, label=f"{m} (AP={average_precision_score(y, p):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_pr.axhline(y.mean(), color="k", ls="--", alpha=0.5)
    ax_roc.set(title=f"ROC - {strategy}", xlabel="1 - Specificity", ylabel="Recall")
    ax_pr.set(title=f"Precision-Recall - {strategy}", xlabel="Recall", ylabel="Precision")
    for ax in (ax_roc, ax_pr):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    plt.show()


def show_final_comparison(results):
    """Tabela com todas as combinações (estratégia x modelo) e gráfico da AUC agregada."""
    rows = [{"estratégia": s, "modelo": m, **compute_metrics(r["y"], r["p"])}
            for (s, m), r in results.items()]
    summary = pd.DataFrame(rows)

    cols = ["AUC", "PR_AUC", "BalAcc", "Accuracy", "Recall", "Precision", "Specificity", "F1"]
    display(summary.set_index(["estratégia", "modelo"])[cols].round(3)
            .sort_values("AUC", ascending=False))

    auc_tbl = summary.pivot(index="modelo", columns="estratégia", values="AUC").loc[MODEL_ORDER]
    ax = auc_tbl.plot.bar(figsize=(9, 4), rot=0, edgecolor="k")
    ax.axhline(0.5, color="k", ls="--", alpha=0.5)
    ax.set(ylim=(0.4, 1.0), ylabel="AUC agregada", title="AUC por modelo e estratégia")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
