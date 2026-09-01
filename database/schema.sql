-- ====================================================================
-- Production Data Warehouse Schema for Customer Churn & LTV Analytics
-- PostgreSQL DDL Script
-- ====================================================================

-- 1. Customers Dimension & Facts Table
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(5),
    dependents VARCHAR(5),
    tenure INT,
    phone_service VARCHAR(5),
    multiple_lines VARCHAR(25),
    internet_service VARCHAR(25),
    online_security VARCHAR(25),
    online_backup VARCHAR(25),
    device_protection VARCHAR(25),
    tech_support VARCHAR(25),
    streaming_tv VARCHAR(25),
    streaming_movies VARCHAR(25),
    contract VARCHAR(25),
    paperless_billing VARCHAR(5),
    payment_method VARCHAR(50),
    monthly_charges NUMERIC(10, 2),
    total_charges NUMERIC(10, 2),
    churn VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Churn Prediction Scores Table
CREATE TABLE IF NOT EXISTS churn_predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id) ON DELETE CASCADE,
    churn_probability NUMERIC(5, 4) NOT NULL,
    churn_prediction VARCHAR(5) NOT NULL,
    risk_level VARCHAR(20) NOT NULL, -- Low, Medium, High
    model_name VARCHAR(50) DEFAULT 'XGBoost_Classifier_v1',
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Customer Lifetime Value (LTV) Forecasts Table
CREATE TABLE IF NOT EXISTS ltv_predictions (
    ltv_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id) ON DELETE CASCADE,
    predicted_ltv NUMERIC(10, 2) NOT NULL,
    retention_priority VARCHAR(20) NOT NULL, -- Critical, High, Normal, Low
    recommended_action VARCHAR(255),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for high-performance querying
CREATE INDEX IF NOT EXISTS idx_customers_contract ON customers(contract);
CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn);
CREATE INDEX IF NOT EXISTS idx_churn_pred_prob ON churn_predictions(churn_probability);
CREATE INDEX IF NOT EXISTS idx_ltv_predicted ON ltv_predictions(predicted_ltv);
