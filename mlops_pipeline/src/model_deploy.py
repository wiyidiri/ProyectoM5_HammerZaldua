import pandas as pd
import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# =========================
# INICIALIZAR APP
# =========================
app = FastAPI(title="Modelo Riesgo Crediticio")

# =========================
# CARGAR MODELO
# =========================
MODEL_PATH = "modelo_xgboost.pkl"

if not os.path.exists(MODEL_PATH):
    print(" Modelo no encontrado, entrenando...")

    from mlops_pipeline.src.model_training_evaluation import build_model
    
    from mlops_pipeline.src.ft_engineering import preprocess_data

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data()
    model = build_model(preprocessor, X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(" Modelo entrenado y guardado")

else:
    model = joblib.load(MODEL_PATH)


# =========================
# ESQUEMA DE ENTRADA
# =========================
class Cliente(BaseModel):
    data: dict


# =========================
# ENDPOINT DE PRUEBA
# =========================
@app.get("/")
def home():
    return {"mensaje": "API de predicción activa"}


# =========================
# PREDICCIÓN (BATCH)
# =========================
@app.post("/predict")
def predict(clientes: List[Cliente]):

    # convertir a DataFrame
    data = [c.data for c in clientes]
    df = pd.DataFrame(data)

    # predicción
    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= 0.5).astype(int)

    result = []

    for i in range(len(df)):
        result.append({
            "prediccion": int(pred[i]),
            "probabilidad": float(prob[i])
        })

    return {"resultados": result}