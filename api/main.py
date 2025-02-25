import os
from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

# Ruta absoluta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Obtiene la carpeta donde está main.py
MODEL_PATH = os.path.join(BASE_DIR, "..", "modelo", "pipeline_total.gz")  # Construye la ruta absoluta

@app.get("/")
def read_root():
    return {"message": "API de predicción de ingresos"}

@app.post("/predict")
def predict_income(adults: dict):
    """Carga el modelo y el pipeline en cada petición (menos eficiente)."""
    try:
        pipeline = joblib.load(MODEL_PATH)  # Usar ruta absoluta
        df = pd.DataFrame([adults])
        prediction = pipeline.predict_proba(df)
        result = float(prediction[0][1])  # Probabilidad de ganar >50K
        return {"prediction": result}
    except FileNotFoundError:
        return {"error": f"No se encontró el archivo en {MODEL_PATH}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)