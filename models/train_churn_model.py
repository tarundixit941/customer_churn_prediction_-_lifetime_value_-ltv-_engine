"""
Churn Classification Model Training & Evaluation.
Trains Logistic Regression, Random Forest, and XGBoost.
Evaluates Precision, Recall, F1-Score, ROC-AUC, and serializes best model.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.feature_engineering import prepare_dataset, get_preprocessor

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def train_and_evaluate_churn_models():
    """Train multiple classification algorithms and compare performance."""
    print("=" * 70)
    print("      Customer Churn Prediction - Model Training & Evaluation")
    print("=" * 70)
    
    # 1. Load Data & Prepare Preprocessor
    X_train, X_test, y_train, y_test, cat_cols, num_cols = prepare_dataset()
    preprocessor = get_preprocessor(cat_cols, num_cols)
    
    # 2. Define Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.5,  # Handle churn class imbalance (~1:3)
            random_state=42,
            eval_metric="logloss"
        )
    }
    
    results = []
    trained_pipelines = {}
    roc_data = {}
    
    # 3. Train and Benchmark Each Model
    for name, clf in models.items():
        print(f"[*] Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })
        
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("                     MODEL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)
    
    # 4. Identify and Select Best Model
    best_model_name = results_df.sort_values(by=["ROC-AUC", "F1-Score"], ascending=False).iloc[0]["Model"]
    print(f"\n[SELECTED BEST MODEL]: {best_model_name}")
    best_pipeline = trained_pipelines[best_model_name]
    
    # 5. Generate Visualizations: ROC Curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_fig_path = os.path.join(FIGURES_DIR, "roc_curves_churn.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"[OK] Saved ROC Curves: {roc_fig_path}")
    
    # 6. Confusion Matrix for Best Model
    y_pred_best = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retained (No)', 'Churned (Yes)'],
                yticklabels=['Retained (No)', 'Churned (Yes)'])
    plt.title(f'Confusion Matrix ({best_model_name})', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrix_churn.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Confusion Matrix: {cm_fig_path}")
    
    # 7. Save Model Artifacts
    model_save_path = os.path.join(SAVED_MODELS_DIR, "churn_model.joblib")
    joblib.dump(best_pipeline, model_save_path)
    
    metadata = {
        "best_model_name": best_model_name,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
        "metrics": results_df.to_dict(orient="records")
    }
    joblib.dump(metadata, os.path.join(SAVED_MODELS_DIR, "churn_metadata.joblib"))
    print(f"[OK] Serialized best model saved to: {model_save_path}")
    print("=" * 70)
    
    return best_pipeline, results_df


if __name__ == "__main__":
    train_and_evaluate_churn_models()
