"""
SHAP (SHapley Additive exPlanations) Module for Business Model Interpretability.
Generates global feature importance plots and individual local explanations for stakeholders.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.feature_engineering import add_engineered_features, prepare_dataset

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def get_feature_names(column_transformer):
    """Extract human-readable feature names from scikit-learn ColumnTransformer."""
    feature_names = []
    for name, trans, cols in column_transformer.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
            feature_names.extend(names)
        else:
            feature_names.extend(cols)
    return feature_names


def run_shap_analysis():
    """Compute global SHAP values and save business stakeholder visual explanations."""
    print("=" * 70)
    print("      Model Explainability & Interpretability (SHAP Analysis)")
    print("=" * 70)
    
    model_path = os.path.join(SAVED_MODELS_DIR, "churn_model.joblib")
    if not os.path.exists(model_path):
        from models.train_churn_model import train_and_evaluate_churn_models
        pipeline, _ = train_and_evaluate_churn_models()
    else:
        pipeline = joblib.load(model_path)
        
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    X_train, X_test, y_train, y_test, _, _ = prepare_dataset()
    
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
    
    print("[*] Calculating SHAP values for test instances...")
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_df)
    except Exception:
        explainer = shap.Explainer(classifier, X_test_df)
        shap_values = explainer(X_test_df).values
        
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values
        
    # Global SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_test_df, show=False, max_display=12)
    plt.title("SHAP Global Feature Impact on Customer Churn", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_summary_path = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved SHAP Beeswarm Summary: {shap_summary_path}")
    
    # Global Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_matrix, X_test_df, plot_type="bar", show=False, max_display=12)
    plt.title("SHAP Mean |Value| Feature Importance (Business Drivers)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    shap_bar_path = os.path.join(FIGURES_DIR, "shap_feature_importance_bar.png")
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved SHAP Feature Importance Bar Plot: {shap_bar_path}")
    
    # Local Explanation for an Individual Customer
    high_risk_idx = 0
    plt.figure(figsize=(10, 5))
    if hasattr(shap, "waterfall_plot") and hasattr(explainer, "expected_value"):
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        explanation = shap.Explanation(
            values=shap_matrix[high_risk_idx],
            base_values=base_val,
            data=X_test_df.iloc[high_risk_idx],
            feature_names=feature_names
        )
        shap.waterfall_plot(explanation, show=False, max_display=8)
        plt.title("SHAP Individual Customer Churn Attribution", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        waterfall_path = os.path.join(FIGURES_DIR, "shap_individual_waterfall.png")
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved SHAP Individual Waterfall Plot: {waterfall_path}")
        
    print("[OK] SHAP Model Explainability Analysis Completed.")
    print("=" * 70)


def explain_single_customer(customer_df: pd.DataFrame):
    """
    Produce top positive and negative driver factors for a single customer payload.
    Used by FastAPI and Streamlit dashboard.
    """
    model_path = os.path.join(SAVED_MODELS_DIR, "churn_model.joblib")
    if not os.path.exists(model_path):
        return {"drivers": ["Model not initialized"]}
        
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    df_feat = add_engineered_features(customer_df)
    drop_cols = ["customerID", "Churn", "Target_LTV"]
    X = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors="ignore")
    
    X_trans = preprocessor.transform(X)
    feat_names = get_feature_names(preprocessor)
    
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            vals = shap_vals[1][0]
        else:
            vals = shap_vals[0]
            
        driver_df = pd.DataFrame({
            "feature": feat_names,
            "shap_value": vals
        }).sort_values(by="shap_value", ascending=False)
        
        churn_risk_increasing = driver_df[driver_df["shap_value"] > 0].head(3).to_dict(orient="records")
        churn_risk_decreasing = driver_df[driver_df["shap_value"] < 0].tail(3).to_dict(orient="records")
        
        return {
            "risk_increasing_factors": churn_risk_increasing,
            "retention_supporting_factors": churn_risk_decreasing
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    run_shap_analysis()
