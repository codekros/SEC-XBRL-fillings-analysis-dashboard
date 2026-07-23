"""
Unit tests for Machine Learning Distress Prediction model.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = PROJECT_ROOT / "src" / "models" / "distress_classifier.joblib"

def test_model_artifact_loading():
    """
    Verify that the trained classifier joblib exists and loads correctly.
    """
    assert MODEL_FILE.exists(), f"Model file not found at {MODEL_FILE}. Please train the model first."
    
    payload = joblib.load(MODEL_FILE)
    assert "model" in payload
    assert "feature_names" in payload
    assert "target_names" in payload
    
    model = payload["model"]
    feature_names = payload["feature_names"]
    
    assert len(feature_names) == 8
    assert payload["target_names"] == ["Low Risk", "Medium Risk", "High Risk"]

def test_model_inference():
    """
    Verify that the model performs inference and outputs probabilities summing to 1.
    """
    payload = joblib.load(MODEL_FILE)
    model = payload["model"]
    feature_names = payload["feature_names"]
    
    # Create a mock input row
    mock_input = pd.DataFrame([{
        "debt_to_assets": 0.45,
        "operating_margin": 0.15,
        "roa": 0.08,
        "asset_turnover": 0.8,
        "equity_multiplier": 1.8,
        "sloan_ratio": 0.01,
        "receivables_to_payables": 1.2,
        "cash_flow_to_debt": 0.25
    }])[feature_names]
    
    # Predict risk category and probabilities
    pred = model.predict(mock_input)
    probs = model.predict_proba(mock_input)[0]
    
    assert len(pred) == 1
    assert pred[0] in [0, 1, 2] # Risk classes
    assert len(probs) == 3
    assert abs(np.sum(probs) - 1.0) < 1e-9 # Sum of probabilities must be 1.0
