# Parkinson’s Disease Detection using Machine Learning

## Overview
This project leverages Machine Learning to detect Parkinson’s Disease based on biomedical voice measurements. The model is trained on a dataset containing various vocal features (e.g., jitter, shimmer, pitch) and predicts whether a person has Parkinson’s Disease or is healthy. 

Recently, the project has been vastly extended to include a robust model evaluation pipeline and **NeuroPredict**, a premium, interactive web application to showcase the model.

## Features
- **Advanced Machine Learning Pipeline**: Includes data preprocessing (MinMaxScaler), K-Fold Cross-Validation, and hyperparameter tuning using `GridSearchCV`.
- **Model Comparison**: Evaluated multiple classifiers including Logistic Regression, KNN, SVM, Random Forest, and XGBoost. XGBoost was chosen as the best performing model.
- **Feature Importance Analysis**: Automatically analyzes and plots which voice features are the most critical predictors of Parkinson's Disease.
- **NeuroPredict Web App**: A beautiful, responsive, glassmorphism-styled web interface built with HTML, CSS, JavaScript, and a Python (Flask) backend to serve real-time predictions.
- **Model Serialization**: Automatically saves the trained model (`best_model.pkl`) and the data scaler (`scaler.pkl`) for rapid deployment and reuse without retraining.

## Tech Stack
- **Machine Learning**: Python, Scikit-learn, XGBoost, Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Backend**: Flask, Flask-CORS
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)

## Dataset
The dataset used contains voice measurements with a target variable (`status`):
- `1` → Parkinson’s Disease present  
- `0` → Healthy  

## How to Run the Machine Learning Pipeline
You can re-train the model, run cross-validation, and generate new feature importance plots at any time.

1. Clone the repository:
   ```bash
   git clone https://github.com/dipeshjaluria23/Parkinson-s-Disease-Detection-using-Machine-Learning.git
   cd Parkinson-s-Disease-Detection-using-Machine-Learning
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python src/train_model.py
   ```
   *This will evaluate models, tune XGBoost, and save the artifacts in the `models/` directory.*

## How to Run the NeuroPredict Web App
The project includes a ready-to-use web interface for the trained model.

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. Open your browser and go to `http://127.0.0.1:5000`
4. Use the magic wand **Auto-fill** button to load sample data and click **Analyze Data** to see the prediction and probabilities!
