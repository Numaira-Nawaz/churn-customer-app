"""train.py
Train an XGBoost classifier for telecom churn with preprocessing, SMOTE and hyperparameter tuning.
Saves: churn_xgb_model.pkl, label_encoders.pkl, scaler.pkl

Usage:
    python train.py --data /path/to/customer_churn.csv
"""
import argparse
import joblib
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV dataset")
    p.add_argument("--target", default=None, help="Target column name (auto-detected if omitted)")
    p.add_argument("--out-dir", default=".", help="Output directory for model/artifacts")
    p.add_argument("--trials", type=int, default=60, help="Optuna trials for hyperparameter search")
    return p.parse_args()


def preprocess(df, target_col=None, save_encoders=False):
    # Drop common irrelevant columns
    df = df.copy()
    df = df.drop(columns=['customerID'], errors='ignore')

    # Clean whitespace and replace blank strings
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df[c].replace({'': np.nan, 'NA': np.nan, 'NaN': np.nan}, inplace=True)

    # If target_col not provided, try common names
    if target_col is None:
        candidates = ['Churn', 'Churn Category', 'Customer Status', 'Churned', 'Is Churn']
        for t in candidates:
            if t in df.columns:
                target_col = t
                break

    if target_col is None:
        # fallback: infer binary column
        for c in df.columns:
            if df[c].nunique() == 2:
                target_col = c
                break

    if target_col is None:
        raise ValueError('Could not detect target column. Please pass --target with the column name.')

    # Fill missing values: categorical -> mode, numeric -> median
    for col in df.columns:
        if df[col].dtype == object:
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            else:
                df[col].fillna('missing', inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Basic feature engineering
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
        # ensure numeric
        try:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(df['MonthlyCharges'] * df.get('tenure', 1))
        except Exception:
            pass

    # create simple interaction features
    if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df['charges_per_month'] = df['TotalCharges'] / (df['tenure'].replace(0, 1))
        df['monthly_tenure_inter'] = df['MonthlyCharges'] * df['tenure']

    # detect categorical columns (object dtype)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    # exclude target
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    # Label encode categorical columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Create feature order and scale
    feature_order = cat_cols + num_cols
    X = df[feature_order].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = df[target_col]
    # binarize textual labels
    if y.dtype == object or not np.issubdtype(y.dtype, np.number):
        y = y.astype(str).str.strip().str.lower().map(lambda v: 1 if v in ('yes', 'y', 'true', '1', 'churn', 'churned') else 0)

    if save_encoders:
        return X_scaled, y.values, label_encoders, scaler, feature_order, target_col
    return X_scaled, y.values, label_encoders, scaler, feature_order, target_col


def objective(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'random_state': 42,
        'use_label_encoder': False,
        'verbosity': 0,
        'n_jobs': -1,
    }

    clf = xgb.XGBClassifier(**param)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return float(np.mean(scores))


def main():
    args = parse_args()
    # Allow missing/placeholder path: search common candidates if file not found
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"Provided dataset path does not exist: {data_path}")
        # try to find files with 'churn' in the name in the current workspace
        from pathlib import Path
        matches = list(Path('.').rglob('*churn*.csv'))
        if matches:
            data_path = str(matches[0])
            print(f"Found candidate dataset at: {data_path} — using it.")
        else:
            print('No dataset found automatically. Please provide the correct path to your CSV file.')
            print('Examples:')
            print('  python train.py --data ./customer_churn_telecom_services.csv')
            print('  python train.py --data /full/path/to/customer_churn_telecom_services.csv')
            import sys
            sys.exit(1)

    df = pd.read_csv(data_path)
    print('Loaded', df.shape)

    X, y, label_encoders, scaler, feature_order, target_col = preprocess(df, target_col=args.target, save_encoders=True)

    # Sanity check: ensure y has more than one class before SMOTE
    import sys
    unique_vals, counts = np.unique(y, return_counts=True)
    if len(unique_vals) < 2:
        print('\nERROR: The target variable appears to have only one class after preprocessing.')
        print('Detected unique values:', unique_vals)
        print('Counts:', counts)
        print('\nThis prevents SMOTE and model training. Possible causes and fixes:')
        print("- The target column was mis-detected. Re-run with '--target <COLUMN_NAME>' to specify the correct column.")
        print("- The target contains textual categories that were auto-mapped to a single class. Inspect the raw target values to decide how to binarize.")
        print("- You may need to remap multi-class 'Churn Category' to binary (e.g., map 'No'->0, others->1).")
        print('\nQuick check - show first 20 raw values for the detected target column:')
        try:
            print(df[target_col].astype(str).head(20).tolist())
        except Exception:
            print('Unable to print raw target values.')
        print('\nSuggested commands:')
        print('  python -c "import pandas as pd;df=pd.read_csv(\'./customer_churn_telecom_services.csv\');print(df[\'' + str(target_col) + '\'].value_counts())"')
        sys.exit(1)

    # balance with SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    print('Starting Optuna study...')
    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, X_res, y_res)
    study.optimize(func, n_trials=args.trials)

    print('Best trial:', study.best_trial.params)

    # Train final model with best params and early stopping on a validation split
    best_params = study.best_trial.params
    final_clf = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.15, random_state=42, stratify=y_res)
    final_clf.fit(X_train, y_train, early_stopping_rounds=30, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = final_clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print('Validation accuracy:', acc)
    print(classification_report(y_val, y_pred))

    # Save artifacts
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'churn_xgb_model.pkl')
    enc_path = os.path.join(out_dir, 'label_encoders.pkl')
    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    feature_path = os.path.join(out_dir, 'feature_order.txt')

    joblib.dump(final_clf, model_path)
    joblib.dump(label_encoders, enc_path)
    joblib.dump(scaler, scaler_path)
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_order))

    print(f'Saved model to {model_path}')
    print(f'Saved encoders to {enc_path}')
    print(f'Saved scaler to {scaler_path}')


if __name__ == '__main__':
    main()
