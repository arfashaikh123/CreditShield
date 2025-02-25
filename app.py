from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load dataset and model
DATA_PATH = "credit_risk_dataset.csv"
MODEL_PATH = "xgboost_model.pkl"

# Load or train model
if not os.path.exists(MODEL_PATH):
    print("Training model...")
    data = pd.read_csv(DATA_PATH)
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xg.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Save the model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
else:
    print("Loading model...")
    model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from request
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

@app.route("/train", methods=["POST"])
def train():
    try:
        # Retrain the model
        data = pd.read_csv(DATA_PATH)
        categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        X = data.drop('loan_status', axis=1)
        y = data['loan_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = xg.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        # Save the model
        joblib.dump(model, MODEL_PATH)

        return jsonify({"message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)