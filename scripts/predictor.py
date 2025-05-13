import joblib
import pandas as pd
import shap

def load_pipeline(model_path):
    return joblib.load(model_path)

def predict_transaction(df, model_path):
    pipeline = load_pipeline(model_path)
    probabilities = pipeline.predict_proba(df)[:, 1]
    predictions = (probabilities > 0.5).astype(int)  # default threshold
    return predictions, probabilities

def explain_predictions(df, model_path):
    pipeline = load_pipeline(model_path)
    explainer = shap.Explainer(pipeline.named_steps["classifier"], pipeline.named_steps["preprocessor"].transform(df))
    shap_values = explainer(pipeline.named_steps["preprocessor"].transform(df))
    return shap_values, df
