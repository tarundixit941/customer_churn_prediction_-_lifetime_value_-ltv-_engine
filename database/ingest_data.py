"""
ETL Data Ingestion Script.
Reads raw Telco CSV, cleans standard missing values, and ingests records into SQL database.
"""

import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db_connection import engine, init_db, SessionLocal, Customer
from data.dataset_generator import download_or_generate_dataset


def ingest_data(csv_path: str = None):
    """Clean and ingest Telco dataset into the database."""
    print("=" * 60)
    print("      Starting Data Ingestion & ETL Pipeline")
    print("=" * 60)
    
    # Ensure tables exist
    init_db()
    
    # Load dataset
    if csv_path is None or not os.path.exists(csv_path):
        df = download_or_generate_dataset()
    else:
        df = pd.read_csv(csv_path)
        
    print(f"[*] Raw records loaded: {len(df)}")
    
    # 1. Clean TotalCharges (convert whitespace / nulls to numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    # For new customers with tenure = 0, TotalCharges is 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    
    # 2. Map dataframe columns to Database Schema
    db_records = []
    for _, row in df.iterrows():
        customer = Customer(
            customer_id=str(row["customerID"]),
            gender=str(row.get("gender", "Unknown")),
            senior_citizen=int(row.get("SeniorCitizen", 0)),
            partner=str(row.get("Partner", "No")),
            dependents=str(row.get("Dependents", "No")),
            tenure=int(row.get("tenure", 0)),
            phone_service=str(row.get("PhoneService", "No")),
            multiple_lines=str(row.get("MultipleLines", "No")),
            internet_service=str(row.get("InternetService", "No")),
            online_security=str(row.get("OnlineSecurity", "No")),
            online_backup=str(row.get("OnlineBackup", "No")),
            device_protection=str(row.get("DeviceProtection", "No")),
            tech_support=str(row.get("TechSupport", "No")),
            streaming_tv=str(row.get("StreamingTV", "No")),
            streaming_movies=str(row.get("StreamingMovies", "No")),
            contract=str(row.get("Contract", "Month-to-month")),
            paperless_billing=str(row.get("PaperlessBilling", "No")),
            payment_method=str(row.get("PaymentMethod", "Mailed check")),
            monthly_charges=float(row.get("MonthlyCharges", 0.0)),
            total_charges=float(row.get("TotalCharges", 0.0)),
            churn=str(row.get("Churn", "No"))
        )
        db_records.append(customer)
        
    # 3. Bulk Insert into Database
    db = SessionLocal()
    try:
        # Check existing count
        existing_count = db.query(Customer).count()
        if existing_count > 0:
            print(f"[*] Database already contains {existing_count} records. Truncating for fresh ingestion...")
            db.query(Customer).delete()
            db.commit()
            
        db.bulk_save_objects(db_records)
        db.commit()
        print(f"[OK] Successfully ingested {len(db_records)} customer records into database.")
    except Exception as e:
        db.rollback()
        print(f"[!] Error during ingestion: {e}")
        raise e
    finally:
        db.close()
        
    # Save cleaned version in processed directory
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "cleaned_telco_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"[OK] Cleaned processed dataset saved to: {processed_path}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    ingest_data()
