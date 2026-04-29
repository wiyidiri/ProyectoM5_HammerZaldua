import streamlit as st
import pandas as pd
import numpy as np
import joblib

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

from mlops_pipeline.src.ft_engineering import preprocess_data


# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Monitoring Modelo Crédito", layout="wide")
st.title("📊 Monitoreo de Modelo de Riesgo Crediticio")


# =========================
# 1. CARGAR MODELO
# =========================
@st.cache_resource
def load_model():
    return joblib.load("modelo_xgboost.pkl")

model = load_model()


# =========================
# 2. GENERAR PREDICCIONES
# =========================
def generate_predictions(model, X, threshold):

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    df = X.copy()
    df["probabilidad"] = prob
    df["prediccion"] = pred

    return df


# =========================
# 3. KS TEST
# =========================
def ks_test(train, new):

    results = {}

    for col in train.select_dtypes(include=['int64', 'float64']).columns:

        train_col = train[col].dropna()
        new_col = new[col].dropna()

        if len(train_col) == 0 or len(new_col) == 0:
            continue

        stat, p = ks_2samp(train_col, new_col)
        results[col] = p

    return pd.DataFrame.from_dict(results, orient="index", columns=["KS_pvalue"])

# =========================
# 4. PSI
# =========================
def calculate_psi(expected, actual, bins=10):

    psi_values = {}

    for col in expected.select_dtypes(include=['int64', 'float64']).columns:

        exp_col = expected[col].dropna()
        act_col = actual[col].dropna()

        # evitar columnas vacías
        if len(exp_col) == 0 or len(act_col) == 0:
            continue

        exp = np.histogram(exp_col, bins=bins)[0] / len(exp_col)
        act = np.histogram(act_col, bins=bins)[0] / len(act_col)

        psi = np.sum((act - exp) * np.log((act + 1e-6) / (exp + 1e-6)))

        psi_values[col] = psi

    return pd.DataFrame.from_dict(psi_values, orient="index", columns=["PSI"])


# =========================
# 5. JS
# =========================
def js_divergence(expected, actual):

    js_values = {}

    for col in expected.select_dtypes(include=['int64', 'float64']).columns:

        exp_col = expected[col].dropna()
        act_col = actual[col].dropna()

        if len(exp_col) == 0 or len(act_col) == 0:
            continue

        exp = np.histogram(exp_col, bins=10)[0]
        act = np.histogram(act_col, bins=10)[0]

        exp = exp / exp.sum()
        act = act / act.sum()

        js_values[col] = jensenshannon(exp, act)

    return pd.DataFrame.from_dict(js_values, orient="index", columns=["JS"])

# =========================
# 6. CHI2
# =========================
def chi_square_test(train, new):

    chi_results = {}

    for col in train.select_dtypes(include=['object']).columns:

        train_counts = train[col].value_counts()
        new_counts = new[col].value_counts()

        df = pd.concat([train_counts, new_counts], axis=1).fillna(0)

        chi2, p, _, _ = chi2_contingency(df)

        chi_results[col] = p

    return pd.DataFrame.from_dict(chi_results, orient="index", columns=["Chi2_pvalue"])


# =========================
# 7. MONITOR
# =========================
def monitor(train_df, new_df):

    ks = ks_test(train_df, new_df)
    psi = calculate_psi(train_df, new_df)
    js = js_divergence(train_df, new_df)
    chi = chi_square_test(train_df, new_df)

    report = ks.join(psi, how="outer") \
               .join(js, how="outer") \
               .join(chi, how="outer")

    return report


# =========================
# DATA
# =========================
X_train, X_test, y_train, y_test, _ = preprocess_data()


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Configuración")

threshold = st.sidebar.slider(
    "Threshold",
    0.1, 0.9, 0.5, 0.1
)

sample_size = st.sidebar.slider(
    "Tamaño muestra",
    100, len(X_test), 500, 100
)


# =========================
# SAMPLE
# =========================
new_data = X_test.sample(sample_size, random_state=42)


# =========================
# PREDICCIONES
# =========================
st.subheader("📌 Predicciones")

df_pred = generate_predictions(model, new_data, threshold)

st.dataframe(df_pred.head(20))


# =========================
# MÉTRICAS
# =========================
st.subheader("📈 Métricas")

col1, col2, col3 = st.columns(3)

col1.metric("Probabilidad promedio", round(df_pred["probabilidad"].mean(), 3))
col2.metric("% positivos", round(df_pred["prediccion"].mean()*100, 2))
col3.metric("Registros", len(df_pred))


# =========================
# DISTRIBUCIÓN
# =========================
st.subheader("📊 Distribución Probabilidades")

st.bar_chart(df_pred["probabilidad"])


# =========================
# DRIFT
# =========================
st.subheader("🚨 Data Drift")

drift = monitor(X_train, new_data)

st.dataframe(drift)


# =========================
# ALERTAS
# =========================
st.subheader("⚠️ Alertas")

psi_alert = drift[drift["PSI"] > 0.25]

if len(psi_alert) > 0:
    st.error("🚨 Drift fuerte detectado")
    st.dataframe(psi_alert)
else:
    st.success("✅ Sin drift significativo")


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("Dashboard de monitoreo de modelo 🚀")