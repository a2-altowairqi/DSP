import csv
import os
from datetime import datetime

def log_message(message, log_file="logs/project.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")

def save_metrics(metrics_dict, output_csv="logs/metrics_log.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    file_exists = os.path.exists(output_csv)

    with open(output_csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics_dict)
