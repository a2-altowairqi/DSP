import os
import traceback
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from scripts.training_pipeline import load_data, preprocess_data, train_model
from scripts.predictor import predict_transaction
# from scripts.utils import set_seed  # Uncomment if set_seed exists

def test_training_and_prediction():
    try:
        print("🧪 Starting model training and prediction pipeline test...")
        data_path = "data/creditcard.csv"
        model_path = "models/fraud_detection_model.pkl"

        # set_seed(42)  # Uncomment if you implement it in utils

        # Load and split data
        data = load_data(data_path)
        X, y, preprocessor = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Train model and save
        model = train_model(X_train, y_train, preprocessor)
        assert model is not None
        print("✅ Training completed and model object exists.")

        # Load model from disk for prediction test
        model = joblib.load(model_path)

        print("🧪 Testing prediction pipeline...")
        df_test = X_test.copy()
        predictions, probabilities = predict_transaction(df_test, model_path)

        assert len(predictions) == len(probabilities)
        assert all(0 <= p <= 1 for p in probabilities)
        print("✅ Prediction successful. Sample predictions:", predictions[:5])
        print("🎉 All tests passed.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_training_and_prediction()
