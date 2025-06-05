import streamlit as st
import numpy as np
import joblib

model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("feature_label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")
target_le = joblib.load("target_label_encoder.pkl")

st.title("🎓 Student Dropout Prediction")

# Human-readable options for categorical fields
categorical_options = {
    "Marital status": {
        "1": "Single",
        "2": "Married",
        "3": "Divorced",
        "4": "Widowed"
    },
    "Application mode": {
        "1": "1st phase",
        "2": "2nd phase",
        "3": "3rd phase",
        "4": "Extra phase",
        "5": "Special",
        "6": "International",
        "7": "Transfer"
    },
    "Daytime/evening attendance": {
        "1": "Daytime",
        "0": "Evening"
    },
    "Previous qualification": {
        "1": "Secondary school",
        "2": "Bachelor",
        "3": "Other",
        "4": "Unknown"
    },
    "Nacionality": {
        "1": "Portuguese",
        "2": "European",
        "3": "Lusophone",
        "4": "Other"
    },
    "Mother's qualification": {
        "1": "Basic",
        "2": "High school",
        "3": "Bachelor",
        "4": "Master/PhD"
    },
    "Father's qualification": {
        "1": "Basic",
        "2": "High school",
        "3": "Bachelor",
        "4": "Master/PhD"
    },
    "Mother's occupation": {
        "0": "Unemployed",
        "1": "Public",
        "2": "Private",
        "3": "Self-employed"
    },
    "Father's occupation": {
        "0": "Unemployed",
        "1": "Public",
        "2": "Private",
        "3": "Self-employed"
    },
    "Displaced": {
        "0": "No",
        "1": "Yes"
    },
    "Educational special needs": {
        "0": "No",
        "1": "Yes"
    },
    "Debtor": {
        "0": "No",
        "1": "Yes"
    },
    "Tuition fees up to date": {
        "0": "No",
        "1": "Yes"
    },
    "Gender": {
        "1": "Male",
        "0": "Female"
    },
    "Scholarship holder": {
        "0": "No",
        "1": "Yes"
    },
    "International": {
        "0": "No",
        "1": "Yes"
    }
}

input_data = []

for feature in feature_names:
    if feature in categorical_options:
        options = categorical_options[feature]
        selected = st.selectbox(
            f"{feature} (choose)",
            options=list(options.keys()),
            format_func=lambda x: f"{x} - {options[x]}"
        )
        input_data.append(int(selected))
    else:
        val = st.number_input(f"{feature}", step=1.0)
        input_data.append(val)

if st.button("Predict Dropout Status"):
    try:
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        pred = model.predict(input_scaled)[0]
        label = target_le.inverse_transform([pred])[0]
        st.success(f"📊 Prediction: {label}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
