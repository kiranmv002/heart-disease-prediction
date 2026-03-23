# ============================================================
# Heart Disease Prediction - Streamlit Web App
# Author: M V Kiran
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# -------------------- Load & Train Model --------------------
@st.cache_resource
def train_models():
    df = pd.read_csv("heart.csv")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = knn.score(X_test, y_test)

    return lr, knn, scaler, lr_acc, knn_acc

lr_model, knn_model, scaler, lr_acc, knn_acc = train_models()
# -------------------- Header --------------------
st.title("❤️ Heart Disease Prediction")
st.markdown("### Predict whether a patient has heart disease using Machine Learning")
st.markdown("---")

# -------------------- Model Info --------------------
col1, col2 = st.columns(2)
with col1:
    st.metric("Logistic Regression Accuracy", f"{lr_acc*100:.2f}%")
with col2:
    st.metric("KNN Accuracy (K=5)", f"{knn_acc*100:.2f}%")

st.markdown("---")
# -------------------- Input Form --------------------
st.subheader("📋 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 29, 77, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                      format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.slider("Resting Blood Pressure", 94, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 126, 564, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                       format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])
    thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1],
                         format_func=lambda x: "No" if x == 0 else "Yes")

with col3:
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[0, 1, 2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("No. of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x])
# -------------------- Model Selection --------------------
st.markdown("---")
st.subheader("🤖 Select Model")
model_choice = st.radio("Choose Algorithm:", ["Logistic Regression", "KNN (K=5)"], horizontal=True)


# -------------------- Predict Button --------------------
st.markdown("---")
if st.button("🔍 Predict", use_container_width=True):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)

    if model_choice == "Logistic Regression":
        prediction = lr_model.predict(input_scaled)[0]
        probability = lr_model.predict_proba(input_scaled)[0]
    else:
        prediction = knn_model.predict(input_scaled)[0]
        probability = knn_model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error("⚠️ **Heart Disease Detected!**")
        st.warning("Please consult a doctor immediately.")
    else:
        st.success("✅ **No Heart Disease Detected!**")
        st.info("Keep maintaining a healthy lifestyle!")
