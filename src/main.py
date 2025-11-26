"""
Boston Housing Price Prediction - FastAPI Backend
This script serves the trained model via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from pathlib import Path
from typing import Optional

app = FastAPI(
    title="Boston Housing Price Prediction API",
    description="API for predicting house prices in Boston area",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = Path(__file__).parent / 'boston_model.pkl'
FEATURE_NAMES_PATH = Path(__file__).parent / 'feature_names.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    print("✅ Model loaded successfully!")
    print(f"Features: {feature_names}")
except FileNotFoundError:
    print("❌ Model not found! Please run train.py first.")
    model = None
    feature_names = None

class HouseFeatures(BaseModel):
    """Input features for Boston Housing prediction"""
    crim: float = Field(..., description="Per capita crime rate by town")
    zn: float = Field(..., description="Proportion of residential land zoned for lots over 25,000 sq.ft.")
    indus: float = Field(..., description="Proportion of non-retail business acres per town")
    chas: int = Field(..., description="Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    nox: float = Field(..., description="Nitric oxides concentration (parts per 10 million)")
    rm: float = Field(..., description="Average number of rooms per dwelling")
    age: float = Field(..., description="Proportion of owner-occupied units built prior to 1940")
    dis: float = Field(..., description="Weighted distances to five Boston employment centres")
    rad: int = Field(..., description="Index of accessibility to radial highways")
    tax: float = Field(..., description="Full-value property-tax rate per $10,000")
    ptratio: float = Field(..., description="Pupil-teacher ratio by town")
    b: float = Field(..., description="1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town")
    lstat: float = Field(..., description="% lower status of the population")

    class Config:
        json_schema_extra = {
            "example": {
                "crim": 0.00632,
                "zn": 18.0,
                "indus": 2.31,
                "chas": 0,
                "nox": 0.538,
                "rm": 6.575,
                "age": 65.2,
                "dis": 4.0900,
                "rad": 1,
                "tax": 296.0,
                "ptratio": 15.3,
                "b": 396.90,
                "lstat": 4.98
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predicted_price: float
    predicted_price_formatted: str
    message: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Boston Housing Price Prediction API",
        "status": "online",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "features": feature_names
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """
    Predict house price based on input features
    Returns price in thousands of dollars
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run train.py to train the model first."
        )
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Ensure features are in correct order
        input_data = input_data[feature_names]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Format prediction
        formatted_price = f"${prediction:.2f}k (${prediction*1000:.0f})"
        
        return PredictionResponse(
            predicted_price=round(prediction, 2),
            predicted_price_formatted=formatted_price,
            message="Prediction successful"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
async def get_features():
    """Get list of required features and their descriptions"""
    if feature_names is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_descriptions = {
        "crim": "Per capita crime rate by town",
        "zn": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
        "indus": "Proportion of non-retail business acres per town",
        "chas": "Charles River dummy variable (1 if bounds river; 0 otherwise)",
        "nox": "Nitric oxides concentration (parts per 10 million)",
        "rm": "Average number of rooms per dwelling",
        "age": "Proportion of owner-occupied units built prior to 1940",
        "dis": "Weighted distances to five Boston employment centres",
        "rad": "Index of accessibility to radial highways",
        "tax": "Full-value property-tax rate per $10,000",
        "ptratio": "Pupil-teacher ratio by town",
        "b": "1000(Bk - 0.63)^2 where Bk is proportion of Black residents",
        "lstat": "% lower status of the population"
    }
    
    return {
        "features": feature_names,
        "descriptions": feature_descriptions,
        "total_features": len(feature_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)