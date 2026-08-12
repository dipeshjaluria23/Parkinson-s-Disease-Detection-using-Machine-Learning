import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Setup directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # 1. Load Data
    data_path = 'parkinsons.data'
    if not os.path.exists(data_path):
        data_path = '../parkinsons.data' # in case running from src/
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 2. Preprocess Data
    # 'name' is just an identifier, 'status' is the target
    features = df.drop(['name', 'status'], axis=1)
    labels = df['status']
    feature_names = features.columns.tolist()

    scaler = MinMaxScaler((-1, 1))
    X = scaler.fit_transform(features)
    y = labels.values

    # 3. Model Comparison with Cross-Validation
    print("\nEvaluating Models using 5-Fold Cross Validation...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model_name = ""
    best_score = 0
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        mean_score = np.mean(scores)
        results[name] = mean_score
        print(f"{name}: Mean Accuracy: {mean_score:.4f} (std: {np.std(scores):.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    print(f"\nBest Model: {best_model_name} with CV Accuracy: {best_score:.4f}")

    # 4. Hyperparameter Tuning for XGBoost (as it's originally used and highly effective)
    print("\nPerforming Hyperparameter Tuning for XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_xgb = grid_search.best_estimator_
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    print(f"Best Tuned XGBoost CV Accuracy: {grid_search.best_score_:.4f}")

    # 5. Final Evaluation on a Test Set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_xgb.fit(X_train, y_train)
    y_pred = best_xgb.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Test Accuracy of Tuned XGBoost: {test_acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 6. Feature Importance
    print("\nGenerating Feature Importance Plot...")
    importances = best_xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances (XGBoost)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    print("Saved feature importance plot to plots/feature_importance.png")

    # Save Models
    joblib.dump(best_xgb, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    # Save feature names for frontend dynamic form generation
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print("\nSaved best model, scaler, and feature names to 'models/' directory.")

if __name__ == "__main__":
    main()
