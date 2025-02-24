from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API de predicción de ingresos"}

@app.post("/predict")
def predict_income(adults: dict):
    """Carga el modelo y el pipeline en cada petición (menos eficiente)."""
    pipeline = joblib.load('../modelo/pipeline_total.gz')  # Cargar modelo en cada request
    df = pd.DataFrame([adults])
    prediction = pipeline.predict_proba(df)
    result = float(prediction[0][1])  # Probabilidad de ganar >50K

    return {"prediction": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)