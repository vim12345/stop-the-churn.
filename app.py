
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("📊 Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Data Cleaning
    if 'customerID' in df.columns:
        df.drop(['customerID'], axis=1, inplace=True)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    # Encoding
    le = LabelEncoder()
    if 'Churn' in df.columns:
        df['Churn'] = le.fit_transform(df['Churn'])

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    # Feature and Target Split
    if 'Churn' in df.columns:
        X = df.drop('Churn', axis=1)
        y = df['Churn']
    else:
        st.error("The dataset must include a 'Churn' column for prediction.")
        st.stop()

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Display Results
    st.subheader("Model Performance")
    st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Predict on uploaded data (optional future work)
    st.success("Churn Prediction Model Trained Successfully!")
