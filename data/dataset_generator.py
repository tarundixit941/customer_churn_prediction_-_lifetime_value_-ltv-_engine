"""
Data Generator & Loader for Telco Customer Churn Dataset.
Attempts to fetch the standard IBM Telco Customer Churn dataset from public source,
or generates a statistically realistic 7,043-row dataset matching the exact schema.
"""

import os
import sys
import urllib.request
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "raw", "telco_customer_churn.csv")
PUBLIC_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def download_or_generate_dataset(output_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Download official Telco dataset or generate realistic synthetic replica."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try downloading from public repository
    if not os.path.exists(output_path):
        print(f"[*] Downloading Telco Customer Churn dataset from official repository...")
        try:
            req = urllib.request.Request(PUBLIC_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response, open(output_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"[OK] Successfully downloaded dataset to {output_path}")
            df = pd.read_csv(output_path)
            return df
        except Exception as e:
            print(f"[!] Online download failed ({e}). Generating realistic 7,043-row Telco dataset offline...")
            df = _generate_synthetic_telco(7043)
            df.to_csv(output_path, index=False)
            print(f"[OK] Generated and saved dataset to {output_path}")
            return df
    else:
        print(f"[OK] Dataset already exists at {output_path}")
        return pd.read_csv(output_path)


def _generate_synthetic_telco(n_rows: int = 7043) -> pd.DataFrame:
    """Generate statistically correlated Telco Customer data."""
    np.random.seed(42)
    
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(n_rows)]
    gender = np.random.choice(["Male", "Female"], size=n_rows, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=n_rows, p=[0.838, 0.162])
    partner = np.random.choice(["Yes", "No"], size=n_rows, p=[0.483, 0.517])
    dependents = np.where(partner == "Yes", np.random.choice(["Yes", "No"], size=n_rows, p=[0.5, 0.5]), np.random.choice(["Yes", "No"], size=n_rows, p=[0.1, 0.9]))
    
    # Tenure in months (0 to 72)
    tenure = np.random.choice(
        [np.random.randint(1, 12), np.random.randint(12, 36), np.random.randint(36, 72)],
        size=n_rows,
        p=[0.4, 0.3, 0.3]
    )
    
    phone_service = np.random.choice(["Yes", "No"], size=n_rows, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.47, 0.53]))
            
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_rows, p=[0.44, 0.34, 0.22])
    
    online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies = [], [], [], [], [], []
    
    for iserv in internet_service:
        if iserv == "No":
            for lst in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]:
                lst.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.38, 0.62]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.44, 0.56]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.44, 0.56]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.39, 0.61]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.49, 0.51]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_rows,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    monthly_charges = []
    for i in range(n_rows):
        base = 20.0
        if phone_service[i] == "Yes": base += 10.0
        if multiple_lines[i] == "Yes": base += 10.0
        if internet_service[i] == "DSL": base += 25.0
        elif internet_service[i] == "Fiber optic": base += 45.0
        if online_security[i] == "Yes": base += 5.0
        if online_backup[i] == "Yes": base += 5.0
        if device_protection[i] == "Yes": base += 5.0
        if tech_support[i] == "Yes": base += 5.0
        if streaming_tv[i] == "Yes": base += 10.0
        if streaming_movies[i] == "Yes": base += 10.0
        
        noise = np.random.normal(0, 2.5)
        monthly_charges.append(round(max(18.25, min(118.75, base + noise)), 2))
        
    total_charges = []
    for i in range(n_rows):
        if tenure[i] == 0:
            total_charges.append(" ")  # Simulate whitespace missing values as in real Telco dataset
        else:
            tot = monthly_charges[i] * tenure[i] * np.random.uniform(0.95, 1.05)
            total_charges.append(str(round(tot, 2)))
            
    # Churn probability correlation calculation
    churn = []
    for i in range(n_rows):
        prob = 0.20
        if contract[i] == "Month-to-month": prob += 0.25
        elif contract[i] == "Two year": prob -= 0.18
        if internet_service[i] == "Fiber optic": prob += 0.15
        if tech_support[i] == "No": prob += 0.10
        if tenure[i] < 12: prob += 0.20
        elif tenure[i] > 48: prob -= 0.20
        if payment_method[i] == "Electronic check": prob += 0.12
        if senior_citizen[i] == 1: prob += 0.08
        
        prob = max(0.02, min(0.95, prob))
        churn.append("Yes" if np.random.rand() < prob else "No")
        
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    return df


if __name__ == "__main__":
    df = download_or_generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(df.head())
