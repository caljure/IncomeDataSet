import joblib
import pandas as pd

def load_model():
    """carga el modelo entrenado y preprocesa el pipeline."""
    
    path_mejor_modelo = r"C:\Users\aljur\OneDrive\Desktop\Maestria\GPCD2025\IncomeDataSet\modelo\mejor_modelo.gz"
    path_pipeline_total = r"C:\Users\aljur\OneDrive\Desktop\Maestria\GPCD2025\IncomeDataSet\modelo\pipeline_total.gz"
    
    # Load model and pipeline correctly
    model = joblib.load(path_mejor_modelo)
    pipeline = joblib.load(path_pipeline_total)
    
    return model, pipeline

def predict(data: dict):
    """recive el Df como diccionario, procesa y retorna las predicciones. """

    model, pipeline = load_model()
    
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([data])
    
    # Ensure the pipeline has a transform method before using it
    if hasattr(pipeline, "transform"):
        processed_data = pipeline.transform(df)
    else:
        raise ValueError("Pipeline does not support transformation.")
    
    # Make prediction
    prediction = model.predict(processed_data)
    return prediction.tolist()

if __name__ == "__main__":
    # Example input (replace with actual input format)
    sample_input = {
        "age": 35,
        "workclass": "Private",
        "fnlwgt": 180000,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    
    
    result = predict(sample_input)
    print(f"Prediction: {result}")