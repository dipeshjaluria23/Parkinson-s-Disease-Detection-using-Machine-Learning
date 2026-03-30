# Parkinson-s-Disease-Detection-using-Machine-Learning# Parkinson’s Disease Detection using Machine Learning

## Overview
This project uses machine learning to detect Parkinson’s Disease based on biomedical voice measurements. The model is trained on a dataset containing various vocal features and predicts whether a person has Parkinson’s Disease or not.

## Features
- Data preprocessing and normalization
- Train-test split for evaluation
- Classification using XGBoost
- Model performance evaluation using accuracy score

## Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- XGBoost

## Dataset
The dataset used contains voice measurements with a target variable (`status`):
- `1` → Parkinson’s Disease present  
- `0` → Healthy  

## Workflow
1. Load dataset
2. Preprocess features (scaling)
3. Split data into training and testing sets
4. Train model using XGBoost classifier
5. Evaluate model performance

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
