import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Load model and data
model = joblib.load('models/fraud_detection_model.pkl')
data = pd.read_csv('data/creditcard.csv')

# Prepare features (SHAP needs original DataFrame with column names)
X = data.drop(columns=['Class'], errors='ignore')

# Use a representative sample for faster SHAP computation
X_sample = X.sample(n=1000, random_state=42)

# Initialize SHAP explainer
explainer = shap.Explainer(model.predict, X_sample)
shap_values = explainer(X_sample)

# Ensure visuals directory exists
os.makedirs('visuals', exist_ok=True)

# ✅ Global Feature Importance - SHAP Summary Plot (with fix)
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.savefig('visuals/shap_summary_plot.png', bbox_inches='tight')

# ✅ Force plot for one instance (local explanation)
shap.initjs()
sample_idx = 0
single_sample = X_sample.iloc[[sample_idx]]
single_shap_value = explainer(single_sample)
force_plot = shap.plots.force(single_shap_value)
shap.save_html('visuals/force_plot_sample0.html', force_plot)

print("✅ SHAP summary and force plots saved in 'visuals/' directory.")
