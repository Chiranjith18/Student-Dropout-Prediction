import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# === SETTINGS ===
DATA_PATH = r"C:\Users\chira\Music\dataset.csv"
# Change this to your dataset file
TARGET_COL = "Target"                   # Change this to your target column name

# Load data
df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode categorical features in X
label_encoders = {}
for col in X.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target if categorical
if y.dtype == 'object' or y.dtype.name == 'category':
    target_le = LabelEncoder()
    y = target_le.fit_transform(y)
else:
    target_le = None

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "dropout_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save label encoders for features and target
joblib.dump(label_encoders, "feature_label_encoders.pkl")
if target_le:
    joblib.dump(target_le, "target_label_encoder.pkl")

# Save feature names order for UI input
joblib.dump(list(X.columns), "feature_names.pkl")
