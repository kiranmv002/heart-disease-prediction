# ❤️ Heart Disease Prediction using Machine Learning

A machine learning project to predict the presence of heart disease in patients using **Logistic Regression** and **K-Nearest Neighbors (KNN)** algorithms.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. This project uses the **UCI Heart Disease Dataset** to build and compare ML models that can predict whether a patient has heart disease based on clinical parameters.

---

## 📂 Project Structure

```
heart-disease-prediction/
├── heart_disease_prediction.py   # Main Python script
├── heart.csv                     # Dataset
├── cm_logistic_regression.png    # Confusion Matrix - Logistic Regression
├── cm_knn.png                    # Confusion Matrix - KNN
├── knn_k_vs_accuracy.png         # K Value vs Accuracy plot
├── model_comparison.png          # Model Accuracy Comparison chart
└── README.md                     # Project documentation
```

---

## 📊 Dataset

- **Source:** UCI Heart Disease Dataset (Cleveland)
- **Total Records:** 1025 rows, 14 columns
- **Missing Values:** None ✅
- **Class Balance:** Disease: 526 | No Disease: 499
- **Target Column:** `target` (1 = Heart Disease, 0 = No Heart Disease)

### Features Used:
| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by flourosopy |
| thal | Thalassemia type |

---

## ⚙️ Algorithms Used

- **Logistic Regression** — A linear model for binary classification
- **K-Nearest Neighbors (KNN)** — A non-parametric instance-based classifier

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 79.51% |
| KNN (K=5) | 83.41% |

> ✅ **KNN outperforms Logistic Regression** on this dataset with an accuracy of **83.41%**

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Kiran/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Script
```bash
python heart_disease_prediction.py
```

> Make sure `heart.csv` is in the same folder as the script.

---

## 📉 Output Plots

- Confusion Matrix for Logistic Regression
- Confusion Matrix for KNN
- K Value vs Accuracy (to find best K)
- Model Accuracy Comparison bar chart

> All plots are automatically saved as `.png` files in your project folder.

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 👨‍💻 Author

**Kiran**  
GitHub: [Kiran](https://github.com/Kiran)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
