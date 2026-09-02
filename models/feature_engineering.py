"""
Feature Engineering & Preprocessing Pipeline.
Cleans raw attributes, constructs domain-specific telecom features,
and applies standard scaling and one-hot encoding for ML models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer telecom domain-specific features from raw customer data."""
    data = df.copy()
    
    # Clean numeric fields
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
    data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce").fillna(0)
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce").fillna(0.0)
    
    # 1. Average Monthly Usage vs Base Charge Ratio
    # If customer is on discount or incurred extra fees, ratio diverges from 1.0
    expected_spend = data["tenure"] * data["MonthlyCharges"]
    data["charge_to_tenure_ratio"] = np.where(data["tenure"] > 0, data["TotalCharges"] / (expected_spend + 1e-5), 1.0)
    
    # 2. Total Add-on Services Count (OnlineSecurity, TechSupport, OnlineBackup, DeviceProtection, StreamingTV, StreamingMovies)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    data["total_addon_services"] = 0
    for col in service_cols:
        if col in data.columns:
            data["total_addon_services"] += (data[col] == "Yes").astype(int)
            
    # 3. Automatic Payment Flag (Higher retention for automated payments)
    if "PaymentMethod" in data.columns:
        data["is_auto_payment"] = data["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    else:
        data["is_auto_payment"] = 0
        
    # 4. Long-Term Contract Flag
    if "Contract" in data.columns:
        data["is_long_term_contract"] = (data["Contract"] != "Month-to-month").astype(int)
    else:
        data["is_long_term_contract"] = 0
        
    # 5. Tenure Cohort Categories
    data["tenure_cohort"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 73],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"]
    ).astype(str)
    
    # 6. High Value Flag (Monthly charge above median)
    median_monthly = data["MonthlyCharges"].median() if len(data) > 0 else 70.0
    data["is_high_monthly_charges"] = (data["MonthlyCharges"] > median_monthly).astype(int)
    
    return data


def get_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_dataset(csv_path: str = None, test_size: float = 0.2, random_state: int = 42):
    """Load, engineer features, and split into train/test sets."""
    data_path = csv_path or os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_telco_data.csv")
    if not os.path.exists(data_path):
        from database.ingest_data import ingest_data
        df = ingest_data()
    else:
        df = pd.read_csv(data_path)
        
    # Apply feature engineering
    df_feat = add_engineered_features(df)
    
    # Define features and target
    target_col = "Churn"
    drop_cols = ["customerID", "Churn"] if "customerID" in df_feat.columns else ["Churn"]
    
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])
    y = (df_feat[target_col] == "Yes").astype(int)
    
    # Categorical & Numerical feature list
    numerical_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "charge_to_tenure_ratio",
        "total_addon_services", "SeniorCitizen", "is_auto_payment", "is_long_term_contract", "is_high_monthly_charges"
    ]
    categorical_cols = [c for c in X.columns if c not in numerical_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Save splits for auditability
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(os.path.join(processed_dir, "train_churn.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_churn.csv"), index=False)
    
    return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cat_cols, num_cols = prepare_dataset()
    print(f"[✓] Feature Engineering Complete.")
    print(f"    Train size: {X_train.shape[0]} samples")
    print(f"    Test size:  {X_test.shape[0]} samples")
    print(f"    Numerical Features: {num_cols}")
    print(f"    Categorical Features: {cat_cols}")
