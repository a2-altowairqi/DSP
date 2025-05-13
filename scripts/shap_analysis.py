import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('models/fraud_detection_model.pkl')
#scaler = joblib.load('models/scaler.pkl')  # Not used here directly if model pipeline includes it
data = pd.read_csv('data/creditcard.csv')

# Prepare features (SHAP needs original DataFrame with column names)
X = data.drop(columns=['Class'], errors='ignore')

# Initialize SHAP explainer with model pipeline and original DataFrame
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# Global Feature Importance
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.savefig('visuals/shap_summary_plot.png')

# Force plot for one instance
shap.initjs()
sample_idx = 0
force_plot = shap.plots.force(shap_values[sample_idx])
shap.save_html('visuals/force_plot_sample0.html', force_plot)

print("SHAP plots saved in 'visuals/' directory.")
