import os
import pandas as pd
import numpy as np
import joblib
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

print("IMPROVED CREDIT CARD FRAUD DETECTION")
print("========================================")

# ✅ Use dataset path matching the app's location
csv_path = "data/creditcard.csv"

# ✅ Check file exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")

print(f"Loading dataset from: {csv_path}")

# ✅ Safe CSV loader with delimiter detection
def safe_read_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(5000)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ","
        print(f"Detected delimiter: '{delimiter}'")

    try:
        df = pd.read_csv(path, sep=delimiter, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"CSV read failed with error: {e}, retrying with comma...")
        df = pd.read_csv(path, sep=",", engine="python", on_bad_lines="skip")

    return df

# ✅ Load dataset
df = safe_read_csv(csv_path)

# ✅ Debug dataset info
print("Dataset shape:", df.shape)
print("Dataset columns:", df.columns.tolist())
print(df.head())

# ✅ Balance the dataset for faster training
print("Balancing dataset for faster training...")
fraud = df[df['Class'] == 1]
non_fraud_count = min(len(fraud) * 10, len(df[df['Class'] == 0]))
non_fraud = df[df['Class'] == 0].sample(n=non_fraud_count, random_state=42)
df = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Using balanced sample: {len(df)} rows ({len(fraud)} fraud, {len(non_fraud)} non-fraud)")

# ✅ Ensure expected columns
expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    print(f"Missing columns detected: {missing_cols}")
    df = df[[c for c in expected_cols if c in df.columns]]

# ✅ Features and Target
X = df.drop(columns=["Class"])
y = df["Class"]

# ✅ Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Train-test split (stratify to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Define models
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# ✅ Ensemble Voting Classifier
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft",
    n_jobs=-1
)

# ✅ Train models
print("Training models...")
ensemble.fit(X_train, y_train)

# ✅ Evaluate
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# ✅ Save models and preprocessing
os.makedirs("models", exist_ok=True)

joblib.dump(ensemble, "models/ensemble.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("Models and scaler saved successfully!")

# ✅ Feature Importance (from RandomForest)
if hasattr(rf, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv("models/feature_importance.csv", index=False)
    print("Feature importance saved successfully!")
