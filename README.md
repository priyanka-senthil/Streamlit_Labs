# Boston Housing Price Prediction Lab

A complete machine learning application with FastAPI backend and Streamlit frontend for predicting house prices in the Boston area.

## 🔄 How This Differs from the Iris Dataset Lab

This lab builds upon the concepts from the Iris classification lab but introduces several key differences:

### Dataset & Problem Type
| Aspect | Iris Lab | Boston Housing Lab |
|--------|----------|-------------------|
| **Problem Type** | Classification (predict flower species) | Regression (predict house prices) |
| **Target Variable** | Categorical (3 classes: setosa, versicolor, virginica) | Continuous (price in $1000s) |
| **Number of Features** | 4 features (sepal/petal measurements) | 13 features (crime, rooms, tax, etc.) |
| **Dataset Size** | 150 samples | 506 samples |
| **Feature Types** | All numerical measurements | Mix of rates, indices, and measurements |

---

## 📁 Project Structure

```
src/
├── train.py              # Model training script
├── main.py               # FastAPI backend server
├── Dashboard.py          # Streamlit frontend
├── requirements.txt      # Python dependencies
├── test.json            # Sample input file
├── boston_model.pkl     # Trained model (generated)
└── feature_names.pkl    # Feature names (generated)
```

## 🚀 Quick Start

### Step 1: Setup Environment

Create and activate a virtual environment:

**Mac & Linux:**
```bash
python3 -m venv bostonenv
source ./bostonenv/bin/activate
```

**Windows:**
```bash
python -m venv bostonenv
.\bostonenv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model

```bash
python train.py
```

This will:
- Download the Boston Housing dataset
- Train a Random Forest model
- Save the model to `boston_model.pkl`
- Display performance metrics

**Expected Output:**
```
R² Score: 0.87-0.90
RMSE: $3-4k
MAE: $2-3k
```

### Step 4: Start FastAPI Backend

In a terminal window:
```bash
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

### Step 5: Start Streamlit Frontend

In a **new** terminal window (keep FastAPI running):
```bash
streamlit run Dashboard.py
```

The dashboard will open at: `http://localhost:8501`

## 📸 Screenshots

### 1. Streamlit Dashboard - Manual Input
<img width="1509" height="832" alt="Screenshot 2025-11-26 at 1 38 15 PM" src="https://github.com/user-attachments/assets/27b6406a-2858-4aec-963b-236b24a97eac" />

*Interactive sliders for inputting house features*

### 2. Prediction Results
<img width="1512" height="783" alt="Screenshot 2025-11-26 at 1 24 53 PM" src="https://github.com/user-attachments/assets/ea48ae33-5e9b-49a3-9b14-60ac644c1236" />

*Predicted price display with feature summary*

### 3. FastAPI Documentation

<img width="1512" height="760" alt="Screenshot 2025-11-26 at 1 31 48 PM" src="https://github.com/user-attachments/assets/d4ba3f05-aa68-4062-9f6d-82e04342cffc" />
<img width="1510" height="805" alt="Screenshot 2025-11-26 at 1 30 41 PM" src="https://github.com/user-attachments/assets/1e8074c6-9b3e-4994-9b63-047653ab715b" />

*Auto-generated interactive API documentation*

### 4. Model Training Output
<img width="855" height="536" alt="Screenshot 2025-11-26 at 1 35 27 PM" src="https://github.com/user-attachments/assets/d295c5df-62f8-48f7-8cf3-aef1193dc9d5" />

*Model performance metrics and feature importance*


## 📊 Features

### Dataset Features (13 total):
1. **CRIM** - Per capita crime rate
2. **ZN** - Proportion of residential land for large lots
3. **INDUS** - Proportion of non-retail business acres
4. **CHAS** - Charles River (1 if bounds river, 0 otherwise)
5. **NOX** - Nitric oxides concentration (air pollution)
6. **RM** - Average number of rooms per dwelling
7. **AGE** - Proportion of units built before 1940
8. **DIS** - Distance to employment centres
9. **RAD** - Highway accessibility index
10. **TAX** - Property tax rate
11. **PTRATIO** - Pupil-teacher ratio
12. **B** - Demographic proportion metric
13. **LSTAT** - % lower status population

### Target Variable:
- **MEDV** - Median value of homes in $1000s

## 🎯 How to Use

### Method 1: Manual Input (Sliders)
1. Open the Streamlit dashboard
2. Use sliders in the sidebar to adjust house features
3. Click "🔮 Predict Price"
4. View the predicted price

### Method 2: JSON File Upload
1. Create a JSON file with house features (see `test.json`)
2. Upload via the sidebar
3. Preview the data
4. Click "🔮 Predict Price"

### Sample JSON Format:
```json
{
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
```

## 🔧 API Endpoints

### FastAPI Backend Endpoints:

1. **GET /** - Health check
   ```bash
   curl http://localhost:8000/
   ```

2. **GET /health** - Detailed health status
   ```bash
   curl http://localhost:8000/health
   ```

## 📈 Model Information

- **Algorithm**: Random Forest Regressor
- **Parameters**:
  - n_estimators: 100
  - max_depth: 10
  - min_samples_split: 5
- **Dataset Size**: 506 samples
- **Train/Test Split**: 80/20

## 🐛 Troubleshooting

### Backend Offline Error
- Ensure FastAPI server is running: `uvicorn main:app --reload`
- Check if port 8000 is available
- Verify model file exists: `boston_model.pkl`

### Model Not Found Error
- Run training script first: `python train.py`
- Check if `boston_model.pkl` and `feature_names.pkl` exist

### Connection Timeout
- Check firewall settings
- Ensure both servers are running
- Verify URLs in Dashboard.py match your setup

### Import Errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

## 📝 Notes

- The Boston Housing dataset is a classic ML dataset
- Prices are in $1000s (e.g., 24.0 = $24,000)
- The model predicts median house values
- Results are estimates based on historical data

## 🎓 Learning Objectives

This lab teaches:
- ✅ Training regression models with scikit-learn
- ✅ Building REST APIs with FastAPI
- ✅ Creating interactive dashboards with Streamlit
- ✅ Serializing models with pickle
- ✅ Handling file uploads
- ✅ API communication between frontend and backend
- ✅ Error handling and validation

## 🚧 Extensions

Try these enhancements:
1. Add model comparison (try Linear Regression, XGBoost)
2. Implement data visualization of predictions
3. Add batch prediction for multiple houses
4. Create a database to store predictions
5. Deploy to cloud (Heroku, AWS, etc.)

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)

---

