from sklearn.ensemble import IsolationForest

def detect_anomalies(data):

    model = IsolationForest(contamination=0.05, random_state=42)

    model.fit(data)

    predictions = model.predict(data)

    return predictions