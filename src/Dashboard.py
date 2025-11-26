"""
Boston Housing Price Prediction - Streamlit Dashboard
Interactive UI for predicting house prices
"""

import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

# Configuration
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FASTAPI_MODEL_LOCATION = Path(__file__).resolve().parent / 'boston_model.pkl'
LOGGER = get_logger(__name__)

# Feature ranges and defaults (based on Boston Housing dataset)
FEATURE_CONFIG = {
    'crim': {'min': 0.0, 'max': 90.0, 'default': 0.5, 'step': 0.1, 'help': 'Per capita crime rate by town'},
    'zn': {'min': 0.0, 'max': 100.0, 'default': 18.0, 'step': 1.0, 'help': 'Proportion of residential land zoned for lots over 25,000 sq.ft.'},
    'indus': {'min': 0.0, 'max': 28.0, 'default': 10.0, 'step': 0.5, 'help': 'Proportion of non-retail business acres per town'},
    'chas': {'min': 0, 'max': 1, 'default': 0, 'step': 1, 'help': 'Charles River (1 if bounds river; 0 otherwise)'},
    'nox': {'min': 0.3, 'max': 0.9, 'default': 0.5, 'step': 0.01, 'help': 'Nitric oxides concentration (parts per 10 million)'},
    'rm': {'min': 3.0, 'max': 9.0, 'default': 6.0, 'step': 0.1, 'help': 'Average number of rooms per dwelling'},
    'age': {'min': 0.0, 'max': 100.0, 'default': 65.0, 'step': 1.0, 'help': 'Proportion of units built prior to 1940'},
    'dis': {'min': 1.0, 'max': 13.0, 'default': 5.0, 'step': 0.1, 'help': 'Weighted distances to five Boston employment centres'},
    'rad': {'min': 1, 'max': 24, 'default': 4, 'step': 1, 'help': 'Index of accessibility to radial highways'},
    'tax': {'min': 180.0, 'max': 720.0, 'default': 300.0, 'step': 10.0, 'help': 'Property-tax rate per $10,000'},
    'ptratio': {'min': 12.0, 'max': 22.0, 'default': 18.0, 'step': 0.1, 'help': 'Pupil-teacher ratio by town'},
    'b': {'min': 0.0, 'max': 400.0, 'default': 390.0, 'step': 1.0, 'help': 'Proportion metric (1000(Bk - 0.63)^2)'},
    'lstat': {'min': 1.0, 'max': 38.0, 'default': 10.0, 'step': 0.5, 'help': '% lower status of the population'}
}

def check_backend_status():
    """Check if FastAPI backend is online"""
    try:
        backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=2)
        if backend_request.status_code == 200:
            return True, "Backend online ✅"
        else:
            return False, f"Problem connecting (Status: {backend_request.status_code}) 😭"
    except requests.ConnectionError:
        LOGGER.error("Backend offline - Connection Error")
        return False, "Backend offline 😱"
    except requests.Timeout:
        LOGGER.error("Backend offline - Timeout")
        return False, "Backend timeout ⏱️"
    except Exception as e:
        LOGGER.error(f"Backend error: {e}")
        return False, f"Backend error: {str(e)} 😱"

def build_sidebar():
    """Build the sidebar with input controls"""
    with st.sidebar:
        # Backend status
        status, message = check_backend_status()
        if status:
            st.success(message)
        else:
            st.error(message)
        
        st.info("Configure House Features")
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Manual Input (Sliders)", "Upload JSON File"],
            help="Choose how to input house features"
        )
        
        features = {}
        
        if input_method == "Manual Input (Sliders)":
            st.markdown("### 🏠 House Features")
            
            # Create sliders for each feature
            for feature, config in FEATURE_CONFIG.items():
                features[feature] = st.slider(
                    feature.upper(),
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['default'],
                    step=config['step'],
                    help=config['help']
                )
            
            st.session_state["input_data"] = features
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False
            
        else:  # JSON file upload
            test_input_file = st.file_uploader(
                'Upload prediction file',
                type=['json'],
                help="Upload a JSON file with house features"
            )
            
            if test_input_file:
                st.write('📄 Preview file')
                test_input_data = json.load(test_input_file)
                st.json(test_input_data)
                
                # Extract features from JSON
                if 'input_test' in test_input_data:
                    features = test_input_data['input_test']
                else:
                    features = test_input_data
                
                st.session_state["input_data"] = features
                st.session_state["IS_JSON_FILE_AVAILABLE"] = True
            else:
                st.session_state["IS_JSON_FILE_AVAILABLE"] = False
                st.info("📤 Upload a JSON file to proceed")
        
        # Predict button
        st.markdown("---")
        predict_button = st.button('🔮 Predict Price', type="primary", use_container_width=True)
        
        return predict_button

def make_prediction(features):
    """Send prediction request to backend"""
    try:
        client_input = json.dumps(features)
        
        with st.spinner('🔄 Analyzing house features...'):
            response = requests.post(
                f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                json=features,
                timeout=10
            )
        
        if response.status_code == 200:
            result = response.json()
            return True, result
        else:
            error_msg = f"Server returned status code: {response.status_code}"
            try:
                error_detail = response.json().get('detail', '')
                error_msg += f"\nDetails: {error_detail}"
            except:
                pass
            return False, error_msg
            
    except requests.Timeout:
        return False, "Request timed out. Please try again."
    except requests.ConnectionError:
        return False, "Cannot connect to backend. Make sure FastAPI server is running."
    except Exception as e:
        LOGGER.error(f"Prediction error: {e}")
        return False, f"Error: {str(e)}"

def display_feature_summary(features):
    """Display a summary of input features"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Rooms", f"{features.get('rm', 0):.1f}")
        st.metric("Crime Rate", f"{features.get('crim', 0):.2f}")
        st.metric("Age of Home", f"{features.get('age', 0):.0f}%")
    
    with col2:
        st.metric("Tax Rate", f"${features.get('tax', 0):.0f}")
        st.metric("Pupil-Teacher", f"{features.get('ptratio', 0):.1f}")
        st.metric("Lower Status %", f"{features.get('lstat', 0):.1f}%")
    
    with col3:
        st.metric("NOx Level", f"{features.get('nox', 0):.3f}")
        st.metric("Distance to Work", f"{features.get('dis', 0):.2f}")
        st.metric("Highway Access", f"{features.get('rad', 0)}")

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Boston Housing Price Predictor",
        page_icon="🏠",
        layout="wide"
    )
    
    # Build sidebar and get predict button state
    predict_button = build_sidebar()
    
    # Main content
    st.title("🏠 Boston Housing Price Predictor")
    st.markdown("""
    Predict house prices in the Boston area based on various features like location, 
    property characteristics, and neighborhood statistics.
    """)
    
    # Check if model exists
    if not FASTAPI_MODEL_LOCATION.is_file():
        st.error("⚠️ Model not found! Please run `train.py` first to train the model.")
        st.code("python train.py", language="bash")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ About"])
    
    with tab1:
        if predict_button:
            if "input_data" in st.session_state and st.session_state["input_data"]:
                features = st.session_state["input_data"]
                
                # Display feature summary
                st.markdown("### Input Features Summary")
                display_feature_summary(features)
                
                st.markdown("---")
                
                # Make prediction
                success, result = make_prediction(features)
                
                if success:
                    # Display prediction
                    st.markdown("### 💰 Predicted Price")
                    
                    predicted_price = result['predicted_price']
                    formatted_price = result['predicted_price_formatted']
                    
                    # Create impressive display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.success(f"## {formatted_price}")
                        st.markdown(f"**Predicted median value: ${predicted_price*1000:,.0f}**")
                    
                    st.balloons()
                    
                    # Additional info
                    st.info("💡 This prediction is based on historical Boston housing data and multiple property features.")
                    
                else:
                    st.error(f"❌ Prediction failed: {result}")
                    st.toast('🔴 Prediction failed. Check backend status.', icon="🔴")
            else:
                st.warning("⚠️ Please configure house features in the sidebar first.")
        else:
            st.info("👈 Configure house features in the sidebar and click 'Predict Price' to get started!")
            
            # Show example
            st.markdown("### 📝 Example Input")
            example_features = {
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
            
            col1, col2 = st.columns(2)
            with col1:
                st.json(example_features)
            with col2:
                st.markdown("""
                **Sample JSON format:**
```json
                {
                    "crim": 0.00632,
                    "zn": 18.0,
                    "indus": 2.31,
                    ...
                }
```
                """)
    
    with tab2:
        st.markdown("""
        ### About Boston Housing Dataset
        
        The Boston Housing dataset contains information about houses in the Boston area.
        It includes 506 samples with 13 features that influence house prices.
        
        #### Features:
        - **CRIM**: Per capita crime rate
        - **ZN**: Proportion of residential land zoned for large lots
        - **INDUS**: Proportion of non-retail business acres
        - **CHAS**: Charles River dummy variable
        - **NOX**: Nitric oxides concentration (air pollution)
        - **RM**: Average number of rooms per dwelling
        - **AGE**: Proportion of owner-occupied units built pre-1940
        - **DIS**: Weighted distances to employment centres
        - **RAD**: Index of accessibility to highways
        - **TAX**: Property-tax rate
        - **PTRATIO**: Pupil-teacher ratio
        - **B**: Proportion metric related to demographics
        - **LSTAT**: % lower status of the population
        
        #### How to Use:
        1. Start the FastAPI backend: `uvicorn main:app --reload`
        2. Adjust house features using sliders or upload a JSON file
        3. Click "Predict Price" to get the estimated house value
        
        #### Model:
        Random Forest Regressor trained on historical Boston housing data.
        """)

if __name__ == "__main__":
    main()