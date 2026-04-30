import pandas as pd
import numpy as np
import joblib
import os
from mlops_pipeline.src.ft_engineering import preprocess_data

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    recall_score
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt


# =========================
# FUNCIONES
# =========================
def build_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def summarize_classification(name, y_true, y_pred, y_prob, threshold):

    print(f"\n --- {name} | Threshold {threshold} ---")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

    roc = roc_auc_score(y_true, y_prob)
    recall_0 = recall_score(y_true, y_pred, pos_label=0)

    print(f" ROC-AUC: {roc:.4f}")
    print(f" Recall clase 0: {recall_0:.4f}")

    return roc, recall_0


# =========================
# DATA
# =========================
X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

print(" Datos listos")


# =========================
# MODELOS
# =========================
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    ))
])


# cálculo automático del peso
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos


xgb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        scale_pos_weight=scale,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    ))
])


# =========================
# ENTRENAMIENTO
# =========================
rf_model = build_model(rf_model, X_train, y_train)
xgb_model = build_model(xgb_model, X_train, y_train)

print(" Modelos entrenados")


# =========================
# PROBABILIDADES
# =========================
rf_prob = rf_model.predict_proba(X_test)[:, 1]
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]


# =========================
# THRESHOLDS
# =========================
thresholds = [0.2, 0.3, 0.5, 0.7]

results_list = []


# =========================
# EVALUACIÓN
# =========================
for t in thresholds:

    # RandomForest
    rf_pred = (rf_prob >= t).astype(int)
    rf_roc, rf_recall = summarize_classification(
        "RandomForest", y_test, rf_pred, rf_prob, t
    )

    results_list.append({
        'Modelo': 'RandomForest',
        'Threshold': t,
        'ROC-AUC': rf_roc,
        'Recall_Clase_0': rf_recall
    })

    # XGBoost
    xgb_pred = (xgb_prob >= t).astype(int)
    xgb_roc, xgb_recall = summarize_classification(
        "XGBoost", y_test, xgb_pred, xgb_prob, t
    )

    results_list.append({
        'Modelo': 'XGBoost',
        'Threshold': t,
        'ROC-AUC': xgb_roc,
        'Recall_Clase_0': xgb_recall
    })


# =========================
# TABLA RESUMEN
# =========================
results = pd.DataFrame(results_list)

print("\n TABLA RESUMEN")
print(results)


# =========================
# GRÁFICO
# =========================
plt.figure(figsize=(8,5))

for model in results['Modelo'].unique():
    subset = results[results['Modelo'] == model]
    plt.plot(subset['Threshold'], subset['Recall_Clase_0'], marker='o', label=model)

plt.title("Recall Clase 0 vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Recall Clase 0")
plt.legend()
plt.grid()

plt.show()
# =========================
# GUARDAR MODELO
# =========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(BASE_DIR, "modelo_xgboost.pkl")

joblib.dump(xgb_model, model_path)

print(f" Modelo XGBoost guardado en: {model_path}")