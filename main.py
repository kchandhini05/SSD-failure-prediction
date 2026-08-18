import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import mahalanobis

print("🚀 SSD Failure Prediction System Started")

# -----------------------------
# LOAD DATA (MEMORY SAFE)
# -----------------------------
folder_path = "data/data_Q4_2025"
all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

df_list = []

for file in all_files[:5]:   # limit files
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path, nrows=50000)  # limit rows
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)

print("Original Data:")
print(data["failure"].value_counts())

# -----------------------------
# BALANCED SAMPLING (IMPORTANT)
# -----------------------------
fail_data = data[data["failure"] == 1]
healthy_data = data[data["failure"] == 0]

healthy_sample = healthy_data.sample(n=1000, random_state=42)

data = pd.concat([fail_data, healthy_sample])

print("\nAfter Balancing:")
print(data["failure"].value_counts())

# -----------------------------
# PREPROCESSING
# -----------------------------
drop_cols = ["date", "serial_number", "model"]
data = data.drop(columns=[col for col in drop_cols if col in data.columns])

data = data.fillna(0)

y = data["failure"]

# select only RAW SMART features
features = [
    "smart_5_raw",    # Reallocated sectors
    "smart_9_raw",    # Power on hours
    "smart_187_raw",  # Reported errors
    "smart_188_raw",  # Command timeout
    "smart_197_raw",  # Pending sectors
    "smart_198_raw"   # Uncorrectable errors
]
X = data[features]

print("Using features:", features)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
X_train_md = X_train_scaled
X_test_md = X_test_scaled



# -----------------------------
# ISOLATION FOREST (optional support)
# -----------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(X_train_md)

# -----------------------------
# RANDOM FOREST (MAIN MODEL)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight={0:1, 1:20},   # 🔥 key for imbalance
    random_state=42
)

rf.fit(X_train_md, y_train)

# probability prediction
rf_probs = rf.predict_proba(X_test_md)[:, 1]

# lower threshold → detect rare failures
threshold = 0.1
rf_pred = (rf_probs > threshold).astype(int)

# -----------------------------
# FINAL PREDICTION
# -----------------------------
print("Combining Hybrid Model...")
final_pred = rf_pred

# -----------------------------
# EVALUATION
# -----------------------------
accuracy = accuracy_score(y_test, final_pred)
precision = precision_score(y_test, final_pred, zero_division=0)
recall = recall_score(y_test, final_pred, zero_division=0)
f1 = f1_score(y_test, final_pred, zero_division=0)

print("\n📊 MODEL PERFORMANCE")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# -----------------------------
# SAVE METRICS
# -----------------------------
metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ metrics.json saved successfully")

# -----------------------------
# SAVE RESULTS
# -----------------------------
results = X_test.copy()
results["failure"] = y_test.values
results["prediction"] = final_pred

results.to_csv("results.csv", index=False)

print("✅ results.csv saved successfully")
print("🎯 SYSTEM COMPLETED SUCCESSFULLY")