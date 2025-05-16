import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traceback
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scripts.predictor import predict_transaction, explain_predictions

# App settings
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Real-Time Fraud Detection System")

# Model path
model_path = "models/fraud_detection_model.pkl"

# Upload section
uploaded_file = st.file_uploader("Upload Transaction Data (CSV)", type="csv")

if uploaded_file is not None:
    st.success("File uploaded successfully. Processing...")

    # Debug info
    st.write("File name:", uploaded_file.name)
    content = uploaded_file.read()
    st.write("File size (bytes):", len(content))
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file)

        # Predict and get probabilities
        predictions, probabilities = predict_transaction(df, model_path)

        # Classification threshold
        threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5, 0.01)
        predictions = (probabilities > threshold).astype(int)

        # Merge results
        result_df = df.copy()
        result_df["Prediction"] = predictions
        result_df["Fraud Probability"] = probabilities

        # Display prediction summary
        st.subheader("🔍 Prediction Summary")
        st.dataframe(result_df.head(500).style.applymap(
            lambda val: "background-color: red" if val == 1 else "background-color: lightgreen",
            subset=["Prediction"]
        ))

        # SHAP explanation
        st.subheader("📊 Explainability with SHAP")
        shap_values, processed_df = explain_predictions(df, model_path)

        fig = plt.figure()
        shap.summary_plot(shap_values, processed_df, show=False)
        st.pyplot(fig)

        # ✅ Local SHAP workaround: Force Plot for first sample
        try:
            st.subheader("🔎 Local Explanation (Force Plot for 1st Sample)")
            shap.initjs()
            force_plot = shap.plots.force(shap_values[0])
            os.makedirs("visuals", exist_ok=True)
            shap.save_html("visuals/force_plot_ui.html", force_plot)

            with open("visuals/force_plot_ui.html", "r", encoding="utf-8") as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=400, scrolling=True)

        except Exception as e:
            st.warning("Could not render force plot inline due to Streamlit limitations.")
            st.text(str(e))

        # Download results
        st.download_button("📥 Download Results", result_df.to_csv(index=False), file_name="predictions.csv")

    except pd.errors.EmptyDataError:
        st.error("❌ Error: Uploaded file is empty or has no columns.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        st.text(traceback.format_exc())
