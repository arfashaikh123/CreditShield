import pandas as pd
import joblib
import os

# Initialize model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgboost_model.pkl')
model = joblib.load(MODEL_PATH)

def predict_risk(input_data):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-hot encoding
        categorical_cols = [
            'person_home_ownership', 
            'loan_intent', 
            'loan_grade', 
            'cb_person_default_on_file'
        ]
        input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Align columns
        expected_features = model.get_booster().feature_names
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Predict
        prediction = model.predict(input_df[expected_features])
        proba = model.predict_proba(input_df[expected_features])[0]
        
        return {
            "prediction": int(prediction[0]),
            "confidence": float(proba.max()),
            "risk_level": "High Risk" if prediction[0] == 1 else "Low Risk"
        }
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
