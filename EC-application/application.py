import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Set page config
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title(":bar_chart: Data Science Dashboard")
st.markdown("""
This dashboard allows you to:
- Upload and analyze your datasets
- Visualize data with interactive charts
- Perform basic statistical analysis
- Make predictions using machine learning models
- Get feature suggestions for more accurate predictions
- Interact with the model using custom inputs
""")

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=['csv'])

if uploaded_file is not None:
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

    # Insight Generator
    st.subheader("Automated Insight Summary")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        target = corr.abs().mean().sort_values(ascending=False).index[0]
        top_corr = corr[target].drop(target).sort_values(key=abs, ascending=False).head(3)
        st.markdown(f"- The most predictable target is **{target}**.")
        for feature, value in top_corr.items():
            direction = "positively" if value > 0 else "negatively"
            st.markdown(f"- `{feature}` is **{direction} correlated** with `{target}` (correlation = {value:.2f})")

    # Visualization
    st.subheader("Data Visualization")
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Select a column to visualize:", numeric_cols)
        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig)

        if len(numeric_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig = px.imshow(corr, title="Correlation Heatmap")
            st.plotly_chart(fig)

            st.subheader("Suggested Targets and Predictors")
            avg_corr = corr.abs().mean().sort_values(ascending=False)
            suggested_target = avg_corr.index[0]
            top_predictors = corr[suggested_target].drop(suggested_target).abs().sort_values(ascending=False).head(3)

            st.info(f"Suggested target variable: **{suggested_target}**")
            st.write(f"Top predictive features: {list(top_predictors.index)}")

            high_corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
            if not high_corr_pairs.empty and high_corr_pairs.iloc[0] > 0.9:
                st.warning("Highly correlated features detected:")
                st.write(high_corr_pairs[high_corr_pairs > 0.9])

    # Simple Prediction Model
    st.subheader("Simple Prediction Model")
    if len(numeric_cols) > 1:
        feature_cols = st.multiselect("Select features for prediction:", numeric_cols)
        target_col = st.selectbox("Select target variable:", numeric_cols)

        if feature_cols and target_col and target_col not in feature_cols:
            X = df[feature_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            r2 = model.score(X_test_scaled, y_test)
            st.write(f"R² Score: {r2:.3f}")

            if r2 < 0.5:
                st.error("Low model performance. Try different features or another target.")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions'))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                     mode='lines', name='Perfect Prediction'))
            fig.update_layout(title="Actual vs Predicted Values", xaxis_title="Actual Values", yaxis_title="Predicted Values")
            st.plotly_chart(fig)

            importance = pd.DataFrame({'Feature': feature_cols, 'Importance': abs(model.coef_)})
            importance = importance.sort_values('Importance', ascending=False)
            fig = px.bar(importance, x='Feature', y='Importance', title="Feature Importance")
            st.plotly_chart(fig)

            # Real-time input prediction
            st.subheader("🔍 Try Prediction with Custom Input")
            user_input = []
            for feature in feature_cols:
                val = st.slider(f"{feature}:", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
                user_input.append(val)

            user_input_scaled = scaler.transform([user_input])
            custom_pred = model.predict(user_input_scaled)[0]
            st.success(f"Predicted {target_col}: {custom_pred:.2f}")

    else:
        st.warning("Please upload a dataset with at least two numeric columns for prediction.")
else:
    st.info("Please upload a CSV file to begin analysis.")
