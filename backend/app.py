import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load Models
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), '../models/feature_names.pkl')

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    if feature_names is None:
        return jsonify({'error': 'Feature names not loaded'}), 500
    return jsonify({'features': feature_names})

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    try:
        # Extract features in the correct order
        input_data = [float(data.get(feature, 0.0)) for feature in feature_names]
        
        # Scale the data
        input_data_scaled = scaler.transform([input_data])
        
        # Predict
        prediction = model.predict(input_data_scaled)[0]
        # Check if predict_proba is available (some models like SVC need probability=True)
        # XGBoost generally has it.
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data_scaled)[0]
            prob_healthy = float(probability[0])
            prob_parkinsons = float(probability[1])
        else:
            prob_healthy = 1.0 if prediction == 0 else 0.0
            prob_parkinsons = 1.0 if prediction == 1 else 0.0
        
        result = {
            'prediction': int(prediction),
            'probability_healthy': prob_healthy,
            'probability_parkinsons': prob_parkinsons
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
