# Telecom Churn Predictor (churn-customer-app)

This workspace contains a Streamlit app (`app.py`) for predicting telecom customer churn and a training script (`train.py`) to train and save the best XGBoost model along with preprocessing artifacts.

Files added/modified:
- `train.py` — training script (preprocessing, SMOTE, RandomizedSearchCV, saves model and preprocessors)
- `app.py` — improved Streamlit UI and robust loading of artifacts
- `requirements.txt` — updated with additional plotting and tuning packages

Quickstart (local):

1. Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model (replace with your dataset path):

```bash
python train.py --data /path/to/customer_churn_telecom_services.csv
```

This saves `churn_xgb_model.pkl`, `label_encoders.pkl`, and `scaler.pkl` in the working folder (or the directory passed via `--out-dir`).

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Tips to reach ~95% accuracy (practical guidance):
- Use the same dataset and preprocessing pipeline used during evaluation. Small mismatches in encoding/feature order will hurt results.
- Feature engineering matters: create aggregated features (avg charges per tenure), interaction terms, and use domain knowledge.
- Handle `TotalCharges` as numeric (strip spaces), and consider log-transforming skewed features.
- Use robust hyperparameter tuning: longer RandomizedSearchCV or Optuna for Bayesian tuning.
- Use stratified K-fold CV and monitor validation AUC/accuracy — don't overfit to test set.
- Try feature selection (SHAP/importance) to remove noisy features.

If you want, I can:
- Run an Optuna-based tuning job (script) that budgets more trials.
- Add example unit tests for preprocessing and a minimal CI workflow.

Contact: use issues in the repo or ask here for help running the training locally.
