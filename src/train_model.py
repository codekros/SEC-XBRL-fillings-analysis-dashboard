from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("train_model")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "consolidated_kpis.parquet"
MODEL_DIR = PROJECT_ROOT / "src" / "models"
MODEL_FILE = MODEL_DIR / "distress_classifier.joblib"

FEATURE_COLUMNS = [
    "debt_to_assets",
    "operating_margin",
    "roa",
    "asset_turnover",
    "equity_multiplier",
    "sloan_ratio",
    "receivables_to_payables",
    "cash_flow_to_debt"
]

def load_and_preprocess_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Consolidated KPIs not found at {DATA_PATH}. Please run the consolidation script first.")

    logger.info("Loading consolidated dataset from %s", DATA_PATH.name)
    df = pd.read_parquet(DATA_PATH)

    df["receivables"] = df["receivables"].fillna(0.0)
    df["inventory"] = df["inventory"].fillna(0.0)
    df["payables"] = df["payables"].fillna(0.0)
    df["capex"] = df["capex"].fillna(0.0)

    working_capital = df["receivables"] + df["inventory"] - df["payables"]
    
    assets_safe = df["total_assets"].replace(0, np.nan)
    liabilities_safe = df["total_liabilities"].replace(0, np.nan)
    equity_safe = df["equity"].replace(0, np.nan)
    revenue_safe = df["revenue"].replace(0, np.nan)

    x1 = (working_capital / assets_safe).fillna(0.0)
    x2 = (df["equity"] / assets_safe).fillna(0.0)
    x3 = (df["operating_income"] / assets_safe).fillna(0.0)
    x4 = (df["equity"] / liabilities_safe).fillna(0.0)
    x5 = (df["revenue"] / assets_safe).fillna(0.0)

    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5

    high_risk_mask = (z_score < 1.8) | (df["equity"] <= 0) | ((df["net_income"] < 0) & (df["operating_cash_flow"] < 0))
    med_risk_mask = (~high_risk_mask) & (z_score >= 1.8) & (z_score <= 3.0)
    
    df["risk_label"] = 0
    df.loc[med_risk_mask, "risk_label"] = 1
    df.loc[high_risk_mask, "risk_label"] = 2

    df["debt_to_assets"] = (df["total_liabilities"] / assets_safe).fillna(0.0)
    df["operating_margin"] = (df["operating_income"] / revenue_safe).fillna(0.0)
    df["roa"] = (df["net_income"] / assets_safe).fillna(0.0)
    df["asset_turnover"] = (df["revenue"] / assets_safe).fillna(0.0)
    df["equity_multiplier"] = (df["total_assets"] / equity_safe).fillna(1.0)
    
    df["sloan_ratio"] = (
        (df["net_income"] - df["operating_cash_flow"]) / assets_safe
    ).fillna(0.0)
    
    df["receivables_to_payables"] = (df["receivables"] / df["payables"].replace(0, np.nan)).fillna(0.0)
    df["cash_flow_to_debt"] = (df["operating_cash_flow"] / liabilities_safe).fillna(0.0)

    for col in FEATURE_COLUMNS:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)

    model_df = df[FEATURE_COLUMNS + ["risk_label"]].dropna()
    logger.info("Dataset shape after feature engineering: %s", model_df.shape)
    
    class_counts = model_df["risk_label"].value_counts().to_dict()
    logger.info("Class distribution: Low Risk (0): %d, Med Risk (1): %d, High Risk (2): %d", 
                class_counts.get(0, 0), class_counts.get(1, 0), class_counts.get(2, 0))

    return model_df[FEATURE_COLUMNS], model_df["risk_label"]

def main():
    logger.info("Starting Machine Learning Model Training Pipeline")
    
    X, y = load_and_preprocess_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info("Training RandomForestClassifier with balanced class weights")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12,
        random_state=42, 
        class_weight="balanced", 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("Model Accuracy: %.2f%%", accuracy * 100)
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred, target_names=["Low Risk", "Medium Risk", "High Risk"]))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "target_names": ["Low Risk", "Medium Risk", "High Risk"]
    }
    joblib.dump(payload, MODEL_FILE)
    logger.info("Successfully saved trained model pipeline -> %s", MODEL_FILE)

if __name__ == "__main__":
    main()
