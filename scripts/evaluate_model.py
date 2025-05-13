import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load full pipeline and data
model = joblib.load('models/fraud_detection_model.pkl')
data = pd.read_csv('data/creditcard.csv')

# Preprocess (raw input — pipeline handles transformation)
X = data.drop(columns=['Class'])
y = data['Class']

# Stratified split (30% test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Predict on test set only
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('visuals/classification_report.csv')
print("✅ Classification report saved.")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('visuals/confusion_matrix_eval.png')
print("✅ Confusion matrix plot saved.")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('visuals/roc_curve_eval.png')
print("✅ ROC curve saved.")

# Optional: Feature importances (only valid for tree models, not pipeline-wrapped ones directly)
if hasattr(model.named_steps["classifier"], 'feature_importances_'):
    importances = model.named_steps["classifier"].feature_importances_
    features = model.named_steps["preprocessor"].transformers_[0][2]
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig('visuals/feature_importance_eval.png')
    print("✅ Feature importance plot saved.")
