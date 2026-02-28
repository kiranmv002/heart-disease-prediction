# ============================================================
# Heart Disease Prediction using Logistic Regression and KNN
# Author: Kiran MV
# Dataset: heart.csv (UCI Heart Disease Dataset)
# ============================================================

# Step 1: Import Libraries
import matplotlib
matplotlib.use('Agg')  # Use this if plots don't show on screen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 2: Load Dataset (place heart.csv in the same folder)
df = pd.read_csv("heart.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Step 3: Basic EDA
print("\nDataset Info:")
print(df.info())
print("\nNull Values:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())

# Target Distribution
print("\nTarget Distribution:")
print(df['target'].value_counts())

# Step 4: Data Preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================== Logistic Regression ====================
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("\n--- Logistic Regression ---")
print("Logistic Regression Accuracy:", lr_accuracy)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion Matrix (Logistic Regression):")
print(cm_lr)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cm_logistic_regression.png")
plt.close()
print("Saved: cm_logistic_regression.png")

print("\nPredicted Labels on Test Data (Logistic Regression):")
print(y_pred_lr)

# ========================== KNN ==========================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\n--- KNN (K=5) ---")
print("KNN Accuracy (K=5):", knn_accuracy)
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion Matrix (KNN):")
print(cm_knn)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens',
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("cm_knn.png")
plt.close()
print("Saved: cm_knn.png")

print("\nPredicted Labels on Test Data (KNN):")
print(y_pred_knn)

# ================= K-value vs Accuracy =================
accuracy_scores = []
k_values = range(1, 21)
for k in k_values:
    tmp = KNeighborsClassifier(n_neighbors=k)
    tmp.fit(X_train, y_train)
    pred_k = tmp.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, pred_k))

best_k = k_values[np.argmax(accuracy_scores)]
print(f"\nBest K value: {best_k} with Accuracy: {max(accuracy_scores):.4f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_scores, marker='o', color='teal')
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title("K Value vs Accuracy (KNN)")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_k_vs_accuracy.png")
plt.close()
print("Saved: knn_k_vs_accuracy.png")

# ================= Model Comparison =================
print("\n===== Model Comparison =====")
print(f"Logistic Regression Accuracy : {lr_accuracy:.4f}")
print(f"KNN (K=5) Accuracy           : {knn_accuracy:.4f}")

models = ['Logistic Regression', f'KNN (K={best_k})']
accuracies = [lr_accuracy, max(accuracy_scores)]

plt.figure(figsize=(6, 4))
bars = plt.bar(models, accuracies, color=['steelblue', 'mediumseagreen'])
plt.ylim(0.7, 1.0)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{acc:.4f}", ha='center', fontsize=11)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()
print("Saved: model_comparison.png")

print("\n✅ All plots saved as PNG files in your project folder!")