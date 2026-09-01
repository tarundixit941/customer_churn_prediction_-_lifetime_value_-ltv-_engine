"""
Database Connection and ORM Models for Telco Analytics.
Supports PostgreSQL (Production/Docker) with automated SQLite fallback (Local/Dev).
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func, Numeric
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_DIR = os.path.dirname(os.path.abspath(__file__))

# If no PostgreSQL URL provided or Postgres is not reachable, fallback gracefully to SQLite
if not DATABASE_URL:
    sqlite_path = os.path.join(DB_DIR, "telco_analytics.db")
    DATABASE_URL = f"sqlite:///{sqlite_path}"

try:
    engine = create_engine(DATABASE_URL, echo=False)
    with engine.connect() as conn:
        pass
except Exception as e:
    sqlite_path = os.path.join(DB_DIR, "telco_analytics.db")
    DATABASE_URL = f"sqlite:///{sqlite_path}"
    engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String(50), primary_key=True, index=True)
    gender = Column(String(10))
    senior_citizen = Column(Integer)
    partner = Column(String(5))
    dependents = Column(String(5))
    tenure = Column(Integer)
    phone_service = Column(String(5))
    multiple_lines = Column(String(25))
    internet_service = Column(String(25))
    online_security = Column(String(25))
    online_backup = Column(String(25))
    device_protection = Column(String(25))
    tech_support = Column(String(25))
    streaming_tv = Column(String(25))
    streaming_movies = Column(String(25))
    contract = Column(String(25))
    paperless_billing = Column(String(5))
    payment_method = Column(String(50))
    monthly_charges = Column(Float)
    total_charges = Column(Float)
    churn = Column(String(5))
    created_at = Column(DateTime, default=func.now())

    churn_predictions = relationship("ChurnPrediction", back_populates="customer", cascade="all, delete-orphan")
    ltv_predictions = relationship("LTVPrediction", back_populates="customer", cascade="all, delete-orphan")


class ChurnPrediction(Base):
    __tablename__ = "churn_predictions"

    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"), nullable=False)
    churn_probability = Column(Float, nullable=False)
    churn_prediction = Column(String(5), nullable=False)
    risk_level = Column(String(20), nullable=False)
    model_name = Column(String(50), default="XGBoost_Classifier_v1")
    predicted_at = Column(DateTime, default=func.now())

    customer = relationship("Customer", back_populates="churn_predictions")


class LTVPrediction(Base):
    __tablename__ = "ltv_predictions"

    ltv_id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), ForeignKey("customers.customer_id"), nullable=False)
    predicted_ltv = Column(Float, nullable=False)
    retention_priority = Column(String(20), nullable=False)
    recommended_action = Column(String(255))
    predicted_at = Column(DateTime, default=func.now())

    customer = relationship("Customer", back_populates="ltv_predictions")


def get_db():
    """Dependency helper to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables in the configured database."""
    Base.metadata.create_all(bind=engine)
    print(f"[OK] Database tables initialized using engine: {engine.url}")


if __name__ == "__main__":
    init_db()
