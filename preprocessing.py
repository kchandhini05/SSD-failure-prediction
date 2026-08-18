import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):

    # Drop missing values
    data = data.dropna()

    # Select correct REAL features
    features = data[['smart_5_raw',
                     'smart_187_raw',
                     'smart_197_raw',
                     'smart_198_raw',
                     'smart_194_raw']]   # temperature

    # Target
    target = data['failure']

    # Convert to anomaly format
    target = target.apply(lambda x: -1 if x == 1 else 1)

    # Scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, target, scaler