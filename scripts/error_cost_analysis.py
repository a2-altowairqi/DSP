import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load model and data
model = joblib.load('models/fraud_detection_model.pkl')
#scaler = joblib.load('models/scaler.pkl')
data = pd.read_csv('data/creditcard.csv')

# Preprocess
X = data.drop(columns=['Class'])
y = data['Class']
y_pred = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

# Cost estimation
COST_FN = 100  # Cost of missing a fraud (False Negative)
COST_FP = 50    # Cost of wrongly flagging a legit transaction (False Positive)

total_cost = (fn * COST_FN) + (fp * COST_FP)

# Output analysis
print("Confusion Matrix:")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"Estimated Cost of Errors: ${total_cost:,.2f}")

# Save misclassified transactions
data['prediction'] = y_pred
fp_cases = data[(data['Class'] == 0) & (data['prediction'] == 1)]
fn_cases = data[(data['Class'] == 1) & (data['prediction'] == 0)]

fp_cases.to_csv('data/false_positives.csv', index=False)
fn_cases.to_csv('data/false_negatives.csv', index=False)

print("False Positives and False Negatives saved.")