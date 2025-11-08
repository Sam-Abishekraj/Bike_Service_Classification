import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Data Loading and Caching ---

@st.cache_data
def load_data(filepath):
    """Loads the bike prediction dataset."""
    try:
        df = pd.read_csv("bike_predict.csv")
        return df
    except FileNotFoundError:
        st.error(f"Error: {filepath} not found. Please make sure the file is in the same directory.")
        return None

# --- 2. Model Training and Preprocessing ---

@st.cache_resource
def train_model(df):
    """Preprocesses data, trains models, and returns artifacts."""
    
    # --- Preprocessing ---
    
    # Identify feature types
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove the target variable 'Bike_Status' from the feature list
    if 'Bike_Status' in numerical_features:
        numerical_features.remove('Bike_Status')
    
    # Create a processed DataFrame with one-hot encoding
    df_processed = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Define X (features) and y (target)
    y = df_processed['Bike_Status']
    X = df_processed.drop('Bike_Status', axis=1)

    # --- Train-Test Split for Evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale numerical features for evaluation models
    scaler_eval = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler_eval.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler_eval.transform(X_test[numerical_features])

    # --- Train Evaluation Models (Example: Random Forest) ---
    # We'll just show the RF results for brevity in the app
    rf_eval = RandomForestClassifier(random_state=42)
    rf_eval.fit(X_train_scaled, y_train)
    y_pred_proba = rf_eval.predict_proba(X_test_scaled)[:, 1]
    
    # Get ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # --- Train Final Model on FULL Dataset ---
    # This is the model we'll use for live predictions
    
    # 1. Scale all numerical features on the full dataset
    scaler_final = StandardScaler()
    X[numerical_features] = scaler_final.fit_transform(X[numerical_features])
    
    # 2. Train the final Random Forest model
    final_model = RandomForestClassifier(random_state=42)
    final_model.fit(X, y)
    
    # Return all necessary artifacts
    return final_model, scaler_final, numerical_features, categorical_features, X.columns, fpr, tpr, auc

# --- 3. Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("Bike Maintenance Classification Project")

# Load the data
df_original = load_data('bike_predict.csv')

if df_original is not None:
    
    # Train model and get artifacts
    (
        final_model, 
        scaler_final, 
        numerical_features, 
        categorical_features, 
        final_model_columns,
        fpr, 
        tpr, 
        auc
    ) = train_model(df_original)
    
    # --- App Layout ---
    
    # --- Sidebar for Live Prediction ---
    st.sidebar.header("🔧 Make a Live Prediction")
    
    # Create input fields in the sidebar
    input_data = {}
    
    st.sidebar.subheader("Numerical Features")
    input_data['Kilometers_Driven'] = st.sidebar.slider("Kilometers Driven", 0, 50000, 15000)
    input_data['Engine_Temp'] = st.sidebar.slider("Engine Temp (°C)", 50.0, 120.0, 80.0)
    input_data['Oil_Quality'] = st.sidebar.slider("Oil Quality (Index)", 0, 100, 50)
    input_data['Battery_Voltage'] = st.sidebar.slider("Battery Voltage (V)", 9.0, 15.0, 12.5)
    input_data['Brake_Pad_Thickness'] = st.sidebar.slider("Brake Pad Thickness (mm)", 1.0, 7.0, 4.0)
    
    st.sidebar.subheader("Categorical Features")
    input_data['Chain_Condition'] = st.sidebar.selectbox("Chain Condition", options=df_original['Chain_Condition'].unique())
    input_data['Vibration_Level'] = st.sidebar.selectbox("Vibration Level", options=df_original['Vibration_Level'].unique())
    input_data['AirFilter_Condition'] = st.sidebar.selectbox("AirFilter Condition", options=df_original['AirFilter_Condition'].unique())
    
    # --- Prediction Logic ---
    if st.sidebar.button("Predict Bike Status"):
        
        # 1. Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 2. Preprocess the input data
        
        # One-hot encode categorical features
        input_processed = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
        
        # Align columns with the model's training columns
        # (Adds missing dummy columns with 0, removes extra columns)
        input_processed = input_processed.reindex(columns=final_model_columns, fill_value=0)
        
        # Scale numerical features using the *fitted* scaler
        input_processed[numerical_features] = scaler_final.transform(input_processed[numerical_features])
        
        # 3. Make prediction
        prediction = final_model.predict(input_processed)[0]
        prediction_proba = final_model.predict_proba(input_processed)[0]
        
        # 4. Display results
        st.sidebar.subheader("Prediction Result")
        if prediction == 1:
            st.sidebar.error("Status: **Needs Maintenance**")
            st.sidebar.metric(label="Probability of Needing Maintenance", value=f"{prediction_proba[1]*100:.2f}%")
        else:
            st.sidebar.success("Status: **Looks Good**")
            st.sidebar.metric(label="Probability of Needing Maintenance", value=f"{prediction_proba[1]*100:.2f}%")
            
    # --- Main Page Content ---
    
    st.header("Project Overview")
    st.write("""
    This app predicts whether a bike needs maintenance (`Bike_Status = 1`) or not (`Bike_Status = 0`)
    based on sensor readings. The model is trained on a labeled dataset using a Random Forest Classifier.
    """)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Data Sample")
        st.dataframe(df_original.head(10))
        
    with col2:
        st.subheader("Model Performance (Random Forest)")
        st.write("The model was evaluated on a 20% test split of the data.")
        
        # Display the ROC curve
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Random Forest')
        ax.legend(loc='lower right')
        ax.grid(True)
        st.pyplot(fig)
