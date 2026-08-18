import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def mahalanobis_distance(X):
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    distances = []
    for x in X:
        diff = x - mean
        dist = np.sqrt(diff.T @ inv_cov @ diff)
        distances.append(dist)

    return np.array(distances)


def train_hybrid_model(X, y):

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso_pred = iso.fit_predict(X)

    # Mahalanobis
    maha = mahalanobis_distance(X)
    threshold = np.percentile(maha, 95)

    maha_pred = [-1 if d > threshold else 1 for d in maha]

    # Combine anomaly detection
    combined = []
    for i in range(len(X)):
        if iso_pred[i] == -1 or maha_pred[i] == -1:
            combined.append(-1)
        else:
            combined.append(1)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X, y)
    rf_pred = rf.predict(X)

    # Final Hybrid Output
    final = []
    for i in range(len(X)):
        if combined[i] == -1 or rf_pred[i] == -1:
            final.append(-1)
        else:
            final.append(1)

    # Metrics
    acc = accuracy_score(y, final)
    prec = precision_score(y, final)
    rec = recall_score(y, final)
    f1 = f1_score(y, final)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    return final, metrics