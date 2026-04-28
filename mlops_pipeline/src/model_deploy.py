import pandas as pd
import joblib

from ft_engineering import preprocess_data
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# =========================
# 1. ENTRENAR Y GUARDAR MODELO
# =========================
def train_and_save_model():

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

    # cálculo de peso
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = neg / pos

    model = Pipeline([
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

    model.fit(X_train, y_train)

    # guardar modelo
    joblib.dump(model, 'modelo_xgboost.pkl')

    print("✅ Modelo guardado correctamente")


# =========================
# 2. CARGAR MODELO
# =========================
def load_model():
    model = joblib.load('modelo_xgboost.pkl')
    return model


# =========================
# 3. PREDICCIÓN
# =========================
def predict(model, new_data, threshold=0.5):

    prob = model.predict_proba(new_data)[:, 1]
    pred = (prob >= threshold).astype(int)

    return pred, prob


# =========================
# 4. EJECUCIÓN
# =========================
if __name__ == "__main__":

    # entrenar y guardar
    train_and_save_model()

    # cargar modelo
    model = load_model()

    # ejemplo con nuevos datos (simulación)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()

    sample = X_test.iloc[:5]

    pred, prob = predict(model, sample, threshold=0.5)

    print("\n📊 Predicciones:")
    print(pred)

    print("\n📊 Probabilidades:")
    print(prob)