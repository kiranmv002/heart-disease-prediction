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
