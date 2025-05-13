import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    if 'Time' in data.columns:
        data = data.drop(['Time'], axis=1)

    if 'Amount' in data.columns:
        scaler = StandardScaler()
        data['Amount'] = scaler.fit_transform(data[['Amount']])

    # Feature Engineering
    if 'V1' in data.columns and 'V2' in data.columns:
        data['V1_V2_ratio'] = data['V1'] / (data['V2'] + 1e-5)

    if 'V3' in data.columns:
        data['V3_squared'] = data['V3'] ** 2

    if 'Amount' in data.columns and 'V4' in data.columns:
        data['Amt_V4_interaction'] = data['Amount'] * data['V4']

    return data
