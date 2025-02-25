from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model
MODEL_PATH = "model/xgboost_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        input_data = request.json

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)

        # Align columns with training data
        missing_cols = set(model.get_booster().feature_names) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[model.get_booster().feature_names]

        # Make prediction
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0]

        # Return result
        return jsonify({
            "prediction": int(prediction[0]),
            "confidence": float(max(proba)),
            "risk_level": "High Risk" if prediction[0] == 1 else "Low Risk"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)