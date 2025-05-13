from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

def split_data(data, target_column='Class', test_size=0.2, random_state=42):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def apply_resampling(X_train, y_train, method='SMOTE'):
    method = method.upper()
    print(f"Original distribution: {Counter(y_train)}")

    if method == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif method == 'ADASYN':
        sampler = ADASYN(random_state=42)
    elif method == 'NEARMISS':
        sampler = NearMiss()
    elif method == 'SMOTETOMEK':
        sampler = SMOTETomek(random_state=42)
    elif method == 'SMOTEENN':
        sampler = SMOTEENN(random_state=42)
    else:
        raise ValueError("Unsupported resampling method")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f"Resampled distribution: {Counter(y_resampled)}")
    return X_resampled, y_resampled
