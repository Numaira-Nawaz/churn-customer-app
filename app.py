# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📞", layout="wide")

MODEL_FILE = "churn_xgb_model.pkl"
ENCODERS_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_ORDER_FILE = "feature_order.txt"


@st.cache_resource(show_spinner=False)
def load_objects():
    model = None
    encoders = {}
    scaler = None
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
        except Exception as e:
            # If xgboost isn't installed, unpickling will fail because the class can't be imported.
            msg = str(e)
            if 'xgboost' in msg or isinstance(e, ModuleNotFoundError):
                st.error("Couldn't load model: the Python package 'xgboost' is not installed in this environment.")
                st.markdown(
                    "To fix this, install xgboost in your environment and restart the app. For example:"
                )
                st.code("pip install xgboost --prefer-binary", language='bash')
                st.markdown("If you use conda: `conda install -c conda-forge xgboost`\n\nOn hosted services (Streamlit Cloud), ensure `xgboost` is listed in `requirements.txt` before deploying.")
            else:
                st.warning(f"Couldn't load model: {e}")
    if os.path.exists(ENCODERS_FILE):
        try:
            encoders = joblib.load(ENCODERS_FILE)
        except Exception as e:
            st.warning(f"Couldn't load encoders: {e}")
    if os.path.exists(SCALER_FILE):
        try:
            scaler = joblib.load(SCALER_FILE)
        except Exception as e:
            st.info(f"No scaler loaded: {e}")
    # load feature order if available (saved by training script)
    feature_order = None
    if os.path.exists(FEATURE_ORDER_FILE):
        try:
            with open(FEATURE_ORDER_FILE, 'r') as f:
                feature_order = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            feature_order = None
    return model, encoders, scaler


model, label_encoders, scaler = load_objects()

# try to load feature order from file (train.py writes this)
TRAIN_FEATURE_ORDER = None
if os.path.exists(FEATURE_ORDER_FILE):
    try:
        with open(FEATURE_ORDER_FILE, 'r') as f:
            TRAIN_FEATURE_ORDER = [line.strip() for line in f.readlines() if line.strip()]
    except Exception:
        TRAIN_FEATURE_ORDER = None

# If the training feature order wasn't saved but the model contains feature names, try to recover them.
if TRAIN_FEATURE_ORDER is None and model is not None:
    try:
        # XGBoost sklearn wrapper stores booster with feature names
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            fn = getattr(booster, 'feature_names', None)
            if fn:
                TRAIN_FEATURE_ORDER = list(fn)
    except Exception:
        TRAIN_FEATURE_ORDER = None

st.markdown("""
<div style='display:flex;align-items:center;gap:12px'>
    <div style='font-size:36px'>📞</div>
    <div>
        <h1 style='margin:0'>Telecom Customer Churn Predictor</h1>
        <p style='margin:0;color:#9AA0A6'>Quickly predict churn probability and explore what matters.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("---")
st.markdown("""
<style>
    .stButton>button {
        background-color: #0a84ff;
        color: white;
        border-radius: 8px;
        padding: 8px 12px;
    }
    .big-prob { font-size:20px; font-weight:600 }
    .input-card { background:#0f1724; padding:16px; border-radius:12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
    .label { color:#9AA0A6; font-size:12px }
    .result-card { padding:16px; border-radius:10px; color:white; font-weight:700 }
    .result-success { background: linear-gradient(90deg,#0ea5a4,#059669); }
    .result-danger { background: linear-gradient(90deg,#ef4444,#b91c1c); }
    .small-muted { color:#9AA0A6; font-size:12px }
</style>
""", unsafe_allow_html=True)

with st.sidebar.expander("Customer details", expanded=True):
    # Use mild defaults; try to infer options from encoders if available
    def options_for(col, default):
        # Force user-friendly options for common binary fields
        if col == 'gender':
            return ["Female", "Male"]
        if col == 'SeniorCitizen':
            return ["No", "Yes"]
        if col in ('Partner', 'Dependents'):
            return ["Yes", "No"]
        # Otherwise try to infer from saved encoders, falling back to defaults
        if col in label_encoders:
            try:
                classes = list(label_encoders[col].classes_)
                # Prefer human-readable classes if present
                cls_str = [str(c).strip() for c in classes]
                return cls_str
            except Exception:
                return default
        return default

    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;justify-content:space-between;align-items:center'><div><strong>Customer Inputs</strong></div><div class='small-muted'>Fill required fields</div></div>", unsafe_allow_html=True)
    gender_choice = st.radio("Gender", options_for('gender', ["Female", "Male"]))
    st.caption("Note: Female = 0, Male = 1 (display only)")
    senior = st.selectbox("Senior Citizen", options_for('SeniorCitizen', ["No", "Yes"]))
    partner = st.selectbox("Has Partner?", options_for('Partner', ["Yes", "No"]))
    dependents = st.selectbox("Has Dependents?", options_for('Dependents', ["Yes", "No"]))
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=70.0, format="%.2f")
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=100000.0, value=2000.0, format="%.2f")
    contract = st.selectbox("Contract", options_for('Contract', ["Month-to-month", "One year", "Two year"]))
    payment_method = st.selectbox("Payment Method", options_for('PaymentMethod', ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]))
    internet_service = st.selectbox("Internet Service", options_for('InternetService', ["DSL", "Fiber optic", "No"]))
    st.markdown("</div>", unsafe_allow_html=True)

# Prepare input values: keep human-readable strings (Yes/No, Female/Male). Preprocess will convert them if model expects numeric.
input_df = pd.DataFrame([{ 
    "gender": gender_choice,
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

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Input preview")
    st.dataframe(input_df.T, use_container_width=True)
with col2:
    st.subheader("Model status")
    if model is None:
        st.error("Model not found. You can upload model/encoders/scaler below or run the training script `train.py` to create artifacts.")
        # allow uploading artifacts
        st.markdown("**Upload artifacts**")
        up_model = st.file_uploader('Upload churn_xgb_model.pkl', type=['pkl','joblib'])
        up_enc = st.file_uploader('Upload label_encoders.pkl', type=['pkl','joblib'])
        up_scaler = st.file_uploader('Upload scaler.pkl', type=['pkl','joblib'])
        if up_model is not None:
            with open(MODEL_FILE, 'wb') as f:
                f.write(up_model.getbuffer())
            st.experimental_rerun()
        if up_enc is not None:
            with open(ENCODERS_FILE, 'wb') as f:
                f.write(up_enc.getbuffer())
            st.experimental_rerun()
        if up_scaler is not None:
            with open(SCALER_FILE, 'wb') as f:
                f.write(up_scaler.getbuffer())
            st.experimental_rerun()
    else:
        st.success("Model loaded")
        # show basic model info
        try:
            st.write(f"Model: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                st.write(f"n_estimators: {getattr(model, 'n_estimators')}")
        except Exception:
            pass


def preprocess(df):
    df_proc = df.copy()
    # Normalize UI-friendly values into canonical strings/numbers before encoding
    # Map SeniorCitizen: 'Yes'/'No' -> 1/0
    if 'SeniorCitizen' in df_proc.columns:
        df_proc['SeniorCitizen'] = df_proc['SeniorCitizen'].map({"Yes": 1, "No": 0}).fillna(df_proc['SeniorCitizen'])
    # Map binary text fields for Partner/Dependents
    for col in ['Partner', 'Dependents']:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].map({"Yes": "Yes", "No": "No", 1: "Yes", 0: "No"}).fillna(df_proc[col])
    # Normalize gender as 'Female'/'Male' strings; encoders (if present) will handle mapping
    if 'gender' in df_proc.columns:
        df_proc['gender'] = df_proc['gender'].astype(str).map(lambda x: x.title() if isinstance(x, str) else x)

    # If label encoders exist and expect numeric '0'/'1' (or ints), map human-friendly inputs to those classes
    def _map_human_to_encoder(col_name, yes_is_one=True):
        if col_name not in df_proc.columns or col_name not in label_encoders:
            return
        le = label_encoders[col_name]
        classes = list(getattr(le, 'classes_', []))
        if not classes:
            return
        cls_str = [str(c).strip().lower() for c in classes]
        # if encoder already uses human labels, skip mapping
        if any(s in ('female','male','yes','no') for s in cls_str):
            return
        # if encoder classes look like '0'/'1' or 0/1, map human inputs accordingly
        if '0' in cls_str and '1' in cls_str:
            target_is_int = all(isinstance(c, int) for c in classes)
            def map_val(v):
                s = str(v).strip().lower()
                if s in ('female','f'):
                    return 0 if target_is_int else '0'
                if s in ('male','m'):
                    return 1 if target_is_int else '1'
                if s in ('yes','y'):
                    return 1 if target_is_int else '1'
                if s in ('no','n'):
                    return 0 if target_is_int else '0'
                return v
            df_proc[col_name] = df_proc[col_name].map(map_val).fillna(df_proc[col_name])

    # Apply mapping for common columns
    for col in ['gender', 'Partner', 'Dependents']:
        _map_human_to_encoder(col)
    # If training wrote a feature order, use it to guarantee exact ordering & length
    if TRAIN_FEATURE_ORDER is not None:
        feature_order = TRAIN_FEATURE_ORDER
        # transform or add each required feature
        for c in feature_order:
            if c in label_encoders:
                # column was categorical in training
                if c in df_proc.columns:
                    try:
                        # keep original categorical values for LabelEncoder compatibility
                        df_proc[c] = label_encoders[c].transform(df_proc[c].astype(str))
                    except Exception:
                        df_proc[c] = df_proc[c].astype(str).map(lambda x: label_encoders[c].transform([x])[0] if x in label_encoders[c].classes_ else -1)
                else:
                    df_proc[c] = 0
            else:
                # numeric or not encoded column
                if c not in df_proc.columns:
                    df_proc[c] = 0

        # final array follows the training feature order
        arr = df_proc[feature_order].astype(float).values
        if scaler is not None:
            arr = scaler.transform(arr)
        return arr, feature_order

    # Fallback: if no feature_order file, use encoders keys then remaining columns
    # Encode categorical columns that we have encoders for
    for col, le in label_encoders.items():
        if col in df_proc.columns:
            try:
                df_proc[col] = le.transform(df_proc[col].astype(str))
            except Exception:
                df_proc[col] = df_proc[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Determine numeric columns (everything not in encoders)
    numeric_cols = [c for c in df_proc.columns if c not in label_encoders]

    feature_order = list(label_encoders.keys()) + numeric_cols

    # ensure all required columns exist
    for c in feature_order:
        if c not in df_proc.columns:
            df_proc[c] = 0

    arr = df_proc[feature_order].astype(float).values
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr, feature_order


X_input, feature_order = preprocess(input_df)

if st.button("🔍 Predict Churn"):
    if model is None:
        st.error("No model available to make predictions. Train first with `python train.py --data path/to/data.csv`.")
    else:
        try:
            # Pre-prediction diagnostic: check feature shape vs model expectation
            provided_count = X_input.shape[1]
            expected_count = None
            if TRAIN_FEATURE_ORDER is not None:
                expected_count = len(TRAIN_FEATURE_ORDER)
            else:
                # attempt to infer from model if possible
                try:
                    if hasattr(model, 'n_features_in_'):
                        expected_count = int(getattr(model, 'n_features_in_'))
                except Exception:
                    expected_count = None

            if expected_count is not None and provided_count != expected_count:
                st.warning(f"Feature shape mismatch detected: expected {expected_count}, got {provided_count}")
                # show which features are missing if we have TRAIN_FEATURE_ORDER
                if TRAIN_FEATURE_ORDER is not None:
                    provided_names = list(input_df.columns)
                    missing = [f for f in TRAIN_FEATURE_ORDER if f not in provided_names]
                    if missing:
                        st.info("Missing features will be filled with 0 by default:")
                        st.write(missing)
                        # pad input_df with missing columns
                        for m in missing:
                            input_df[m] = 0
                        # re-run preprocess to rebuild X_input
                        X_input, feature_order = preprocess(input_df)
                        provided_count = X_input.shape[1]
                else:
                    st.info("No feature_order available to compute missing feature names. Consider retraining or providing 'feature_order.txt'.")

            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]
            # show probability with progress bar and card
            pct = int(proba * 100)
            # Styled result card
            if proba >= 0.5:
                st.markdown(f"<div class='result-card result-danger'>⚠️ Predicted: CHURN — {proba:.2%}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card result-success'>✅ Predicted: STAY — churn prob {proba:.2%}</div>", unsafe_allow_html=True)
            st.progress(pct)
            st.markdown(f"<div class='big-prob'>Probability: {proba:.2%}</div>", unsafe_allow_html=True)
            with st.expander('Feature order used for model'):
                st.write(feature_order)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
