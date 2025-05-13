import pandas as pd
import joblib
import time
import os
from sklearn.preprocessing import StandardScaler
from memory_profiler import memory_usage

# Load model and scaler
model = joblib.load('models/fraud_detection_model.pkl')  # Adjust path if needed
#scaler = joblib.load('models/scaler.pkl')  # Save your scaler during training

def predict_batch(input_file, output_file, chunk_size=5000):
    results = []

    def process_chunk(chunk):
        preds = model.predict(chunk)
        probs = model.predict_proba(chunk)[:, 1]
        chunk['prediction'] = preds
        chunk['fraud_probability'] = probs
        return chunk

    start_time = time.time()
    mem_usage = memory_usage((pd.read_csv, (input_file,), {'chunksize': chunk_size}))

    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {i + 1}")
        chunk = chunk.drop(columns=['Class'], errors='ignore')  # Drop label if exists
        processed = process_chunk(chunk)
        results.append(processed)

    full_results = pd.concat(results)
    full_results.to_csv(output_file, index=False)

    elapsed_time = time.time() - start_time
    print(f"Prediction complete. Time: {elapsed_time:.2f}s, Peak memory: {max(mem_usage):.2f} MiB")

if __name__ == "__main__":
    predict_batch('data/creditcard.csv', 'data/predicted_results.csv')
