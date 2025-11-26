"""
Boston Housing Price Prediction - Model Training Script
This script trains a Random Forest model on the Boston Housing dataset
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from pathlib import Path

def load_boston_data():
    """
    Load Boston Housing dataset
    The dataset contains 506 samples with 13 features
    """
    # Boston Housing dataset features
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    
    # Alternative: Use sklearn's boston dataset (deprecated but still accessible)
    # or load from a CSV file
    
    # For this example, we'll create a function to load from CSV
    # You can also use: from sklearn.datasets import load_boston (deprecated)
    
    # Using a publicly available CSV version
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    
    return df

def train_model():
    """Train Random Forest model on Boston Housing data"""
    
    print("Loading Boston Housing dataset...")
    df = load_boston_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nFeatures: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Separate features and target
    # Target variable is 'medv' (Median value of homes in $1000s)
    X = df.drop('medv', axis=1)
    y = df['medv']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("Model Performance Metrics")
    print("="*50)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:.2f}k")
    print(f"MAE: ${mae:.2f}k")
    print(f"MSE: {mse:.4f}")
    print("="*50)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())
    
    # Save model
    model_path = Path(__file__).parent / 'boston_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names for reference
    feature_names_path = Path(__file__).parent / 'feature_names.pkl'
    with open(feature_names_path, 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print(f"Feature names saved to: {feature_names_path}")
    
    return model, X.columns.tolist()

if __name__ == "__main__":
    print("Boston Housing Price Prediction - Model Training")
    print("="*60)
    model, features = train_model()
    print("\n✅ Training complete!")
    print(f"\nThe model predicts house prices based on these features:")
    for i, feature in enumerate(features, 1):
        print(f"  {i}. {feature}")