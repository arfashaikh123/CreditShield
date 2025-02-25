from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_risk  # Import from predict.py
import os

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {
        "origins": [
            "https://creditshield.netlify.app",  # Your Netlify domain
            "http://localhost:3000"  # For local development
        ]
    }
})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = predict_risk(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
