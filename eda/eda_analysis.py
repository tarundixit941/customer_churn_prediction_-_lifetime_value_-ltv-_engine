"""
Exploratory Data Analysis (EDA) Module.
Generates statistical analytics reports and visualizations saved into reports/figures/.
"""

import os
import sys
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

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.ingest_data import ingest_data

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("tab10")


def run_eda(csv_path: str = None):
    """Perform comprehensive EDA and save analytical figures."""
    print("=" * 60)
    print("      Executing Exploratory Data Analysis (EDA)")
    print("=" * 60)
    
    data_path = csv_path or os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_telco_data.csv")
    if not os.path.exists(data_path):
        df = ingest_data()
    else:
        df = pd.read_csv(data_path)
        
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    
    print(f"[*] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    print(f"[*] Churn Distribution:\n    No:  {churn_counts.get('No', 0)} ({churn_pct.get('No', 0):.2f}%)\n    Yes: {churn_counts.get('Yes', 0)} ({churn_pct.get('Yes', 0):.2f}%)")
    
    # 1. Churn Target Distribution Plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#3498db', '#e74c3c']
    
    sns.countplot(data=df, x='Churn', ax=ax[0], palette=colors)
    ax[0].set_title("Customer Churn Count Distribution", fontsize=13, fontweight='bold')
    ax[0].set_xlabel("Churn Status", fontsize=11)
    ax[0].set_ylabel("Customer Count", fontsize=11)
    for p in ax[0].patches:
        ax[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
                       
    ax[1].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=colors, startangle=140,
              explode=(0, 0.08), shadow=True, textprops={'fontweight': 'bold'})
    ax[1].set_title("Churn Ratio (Class Imbalance)", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    fig_path1 = os.path.join(FIGURES_DIR, "eda_churn_distribution.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig_path1}")
    
    # 2. Contract Type vs. Churn Rate
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], ax=ax, width=0.6)
    ax.set_title("Churn Rate by Contract Type (%)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xlabel("Contract Type", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    fig_path2 = os.path.join(FIGURES_DIR, "eda_contract_vs_churn.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig_path2}")
    
    # 3. Tenure Distribution by Churn
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=36, palette=['#2980b9', '#e74c3c'], ax=ax, element="step")
    ax.set_title("Tenure Distribution by Churn Status (Months)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Tenure (Months)", fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=11)
    plt.tight_layout()
    fig_path3 = os.path.join(FIGURES_DIR, "eda_tenure_vs_churn.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig_path3}")
    
    # 4. Monthly Charges Distribution & Boxplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['#3498db', '#e74c3c'], ax=ax[0])
    ax[0].set_title("Monthly Charges Boxplot by Churn", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("Monthly Charges ($)")
    
    sns.kdeplot(data=df[df['Churn'] == 'No']['MonthlyCharges'], label='Retained (No)', color='#3498db', fill=True, ax=ax[1])
    sns.kdeplot(data=df[df['Churn'] == 'Yes']['MonthlyCharges'], label='Churned (Yes)', color='#e74c3c', fill=True, ax=ax[1])
    ax[1].set_title("Monthly Charges Density", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Monthly Charges ($)")
    ax[1].legend()
    
    plt.tight_layout()
    fig_path4 = os.path.join(FIGURES_DIR, "eda_monthly_charges_vs_churn.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig_path4}")
    
    # 5. Correlation Heatmap for Numerical Attributes
    fig, ax = plt.subplots(figsize=(7, 5))
    df_num = df[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    df_num['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
    corr = df_num.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap (Numerical Features & Churn)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_path5 = os.path.join(FIGURES_DIR, "eda_correlation_matrix.png")
    plt.savefig(fig_path5, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig_path5}")
    
    # 6. Tech Support and Internet Service Impact
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    tech_churn = pd.crosstab(df['TechSupport'], df['Churn'], normalize='index') * 100
    tech_churn.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], ax=ax[0])
    ax[0].set_title("Churn Rate by Tech Support Option", fontsize=12, fontweight='bold')
    ax[0].set_ylabel("Percentage (%)")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=15)
    
    internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
    internet_churn.plot(kind='bar', stacked=False, color=['#2ecc71', '#e74c3c'], ax=ax[1])
    ax[1].set_title("Churn Rate by Internet Service Type", fontsize=12, fontweight='bold')
    ax[1].set_ylabel("Percentage (%)")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=15)
    
    plt.tight_layout()
    fig_path6 = os.path.join(FIGURES_DIR, "eda_techsupport_internet_churn.png")
    plt.savefig(fig_path6, dpi=300)
    plt.close()
    print(f"[OK] Saved: {fig_path6}")
    
    print("[OK] EDA Execution completed successfully. All figures exported.")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
