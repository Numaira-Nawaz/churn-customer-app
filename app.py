# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📞")

# --- Load model & preprocessors (assumes files are in same folder) ---
MODEL_FILE = "churn_xgb_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"   # if you saved it

@st.cache_resource(show_spinner=False)
def load_objects():
    model = joblib.load(MODEL_FILE)
    encoders = joblib.load(ENCODERS_FILE)  # dict: {col_name: LabelEncoder()}
    scaler = None
    if os.path.exists(SCALER_FILE):
        scaler = joblib.load(SCALER_FILE)
    return model, encoders, scaler

model, label_encoders, scaler = load_objects()

st.title("📞 Telecom Customer Churn Predictor")
st.write("Provide customer details and press **Predict**")

# --- Define your input fields here ---
# IMPORTANT: use the same feature names and order as your training X
CATEGORICAL_COLS = ["gender","Partner","Dependents","Contract","PaymentMethod","InternetService"]  # example
NUMERIC_COLS = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]                        # example

# You must adjust the select boxes / defaults to match your dataset's possible values.
gender = st.selectbox("Gender", ["Male","Female"])
senior = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Has Partner?", ["Yes","No"])
dependents = st.selectbox("Has Dependents?", ["Yes","No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=2000.0)
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

# Build input DataFrame with same column names as training X
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "InternetService": internet_service
}])

st.write("### Input preview")
st.dataframe(input_df.T, use_container_width=True)

# --- Preprocess: label-encode each categorical column using saved label_encoders ---
def preprocess(df):
    df_proc = df.copy()
    # Encode categorical
    for col, le in label_encoders.items():
        # if unseen category appears, map to nearest / add handling - here we try safe transform:
        try:
            df_proc[col] = le.transform(df_proc[col].astype(str))
        except Exception:
            # fallback: if new category, add it as -1 (or use mode)
            df_proc[col] = df_proc[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    # numeric columns are assumed already numeric
    # if scaler exists, apply it (ensure the scaler was fitted on same feature order)
    if scaler is not None:
        # ensure we pass the same column order used in training
        # combine numeric+encoded categorical in same order as model training
        feature_order = list(label_encoders.keys()) + NUMERIC_COLS
        arr = df_proc[feature_order].astype(float).values
        arr_scaled = scaler.transform(arr)
        return arr_scaled, feature_order
    else:
        feature_order = list(label_encoders.keys()) + NUMERIC_COLS
        return df_proc[feature_order].astype(float).values, feature_order

X_input, feature_order = preprocess(input_df)

# --- Predict ---
if st.button("🔍 Predict Churn"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    if pred == 1:
        st.error(f"⚠️ Predicted: CHURN (probability = {proba:.2%})")
    else:
        st.success(f"✅ Predicted: STAY (probability of churn = {proba:.2%})")

    st.write("Feature order used for model:")
    st.write(feature_order)
