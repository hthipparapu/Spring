import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Data Science Dashboard")
st.markdown("""
This dashboard allows you to:
- Upload and analyze your datasets
- Visualize data with interactive charts
- Perform basic statistical analysis
- Make predictions using machine learning models
""")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Display basic information
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Columns:", list(df.columns))
    
    with col2:
        st.write("Data Types:")
        st.write(df.dtypes)
    
    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Statistical Analysis
    st.subheader("Statistical Analysis")
    st.write(df.describe())
    
    # Visualization
    st.subheader("Data Visualization")
    
    # Select columns for visualization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select a column to visualize:", numeric_cols)
        
        # Create histogram
        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)
        
        # Create correlation heatmap if there are multiple numeric columns
        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, title="Correlation Heatmap")
            st.plotly_chart(fig)
    
    # Simple Prediction Model
    st.subheader("Simple Prediction Model")
    
    if len(numeric_cols) > 1:
        # Select features and target
        feature_cols = st.multiselect("Select features for prediction:", numeric_cols)
        target_col = st.selectbox("Select target variable:", numeric_cols)
        
        if feature_cols and target_col and target_col not in feature_cols:
            # Prepare data
            X = df[feature_cols]
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Display results
            st.write("Model Performance:")
            st.write(f"R² Score: {model.score(X_test_scaled, y_test):.3f}")
            
            # Plot actual vs predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()], 
                                   mode='lines', 
                                   name='Perfect Prediction'))
            fig.update_layout(title="Actual vs Predicted Values",
                            xaxis_title="Actual Values",
                            yaxis_title="Predicted Values")
            st.plotly_chart(fig)
            
            # Feature importance
            importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': abs(model.coef_)
            })
            importance = importance.sort_values('Importance', ascending=False)
            
            fig = px.bar(importance, x='Feature', y='Importance', 
                        title="Feature Importance")
            st.plotly_chart(fig)
    else:
        st.warning("Please upload a dataset with at least two numeric columns for prediction.")
else:
    st.info("Please upload a CSV file to begin analysis.") 