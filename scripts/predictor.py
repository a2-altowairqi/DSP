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

    pipeline = joblib.load(model_path)
    processed = pipeline.named_steps["preprocessor"].transform(df)

    # Pick the first base model from the stacking ensemble (e.g. Random Forest)
    base_model = pipeline.named_steps["classifier"].estimators_[0][1]  # Access 'rf'

    explainer = shap.Explainer(base_model.predict, processed)
    shap_values = explainer(processed)

    return shap_values, df
