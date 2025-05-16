import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import StackingClassifier
import joblib
import os
import time

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

# Load data
def load_data(filepath):
    print(" Loading dataset...")
    data = pd.read_csv(filepath)
    print(f" Loaded dataset with shape: {data.shape}")
    return data

# Preprocess
def preprocess_data(data, target_column='Class'):
    print("🔧 Preprocessing data...")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    print(" Preprocessing pipeline created.")
    return X, y, preprocessor

# Train stacked model
def train_model(X, y, preprocessor, save_path="models"):
    print("⚙️ Setting up model and pipeline...")
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]

    final_estimator = LogisticRegression()
    model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5)

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    print(" Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" Training set: {X_train.shape}, Test set: {X_test.shape}")

    print("🚀 Starting model training (this may take several minutes)...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f" Model training completed in {time.time() - start:.2f} seconds.")

    print(" Generating classification report...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    print(" Saving model...")
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(pipeline, os.path.join(save_path, "fraud_detection_model.pkl"))
    print(" Model saved to 'models/fraud_detection_model.pkl'")

    return pipeline

if __name__ == "__main__":
    set_seed(42)
    data = load_data("data/creditcard.csv")
    X, y, preprocessor = preprocess_data(data)
    model = train_model(X, y, preprocessor)
