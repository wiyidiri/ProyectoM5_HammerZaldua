import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# =========================
# CARGA DE DATOS
# =========================
def load_data():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    file_path = os.path.join(BASE_DIR, 'Base_de_datos.xlsx')
    df = pd.read_excel(file_path)
    return df


# =========================
# LIMPIEZA
# =========================
def clean_data(df):

    df = df.drop_duplicates()

    # eliminar leakage
    leakage_cols = ['puntaje','puntaje_datacredito',
    'promedio_ingresos_datacredito',
    'tendencia_ingresos']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors='ignore')

    return df


# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):

    if 'salario_cliente' in df.columns and 'cuota_pactada' in df.columns:
        df['ratio_cuota_ingreso'] = df['cuota_pactada'] / (df['salario_cliente'] + 1)

    return df


# =========================
# PIPELINE
# =========================
def build_pipeline(X):

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # (si tuvieras ordinales explícitos los defines aquí)
    ordinal_features = []

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features),
        ('ord', ordinal_pipeline, ordinal_features)
    ])

    return preprocessor


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def preprocess_data():

    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    TARGET = 'Pago_atiempo'

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_pipeline(X_train)

    return X_train, X_test, y_train, y_test, preprocessor
