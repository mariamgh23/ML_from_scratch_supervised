# 🚀 Machine Learning From Scratch

![Python](https://img.shields.io/badge/Python-3.11-blue)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-green)

A complete implementation of core Machine Learning algorithms **from scratch using only NumPy**, without relying on scikit-learn models.

This repository focuses on understanding how ML algorithms work internally by building them step-by-step from mathematical foundations.

---

## 🎯 Project Goals

- Understand ML algorithms at mathematical level
- Implement models using only NumPy
- Build ensemble methods (Bagging & Boosting)
- Create custom evaluation metrics
- Compare models on real datasets (Iris & Digits)

---

# 🧠 Implemented Algorithms

## 📈 Regression
- Linear Regression (Gradient Descent based)

## 📊 Classification
- Logistic Regression (Binary & One-vs-Rest)
- Linear SVM (Binary & OVR)
- Decision Tree (Gini-based)
- Random Forest (Bootstrap Aggregation)
- XGBoost (Simplified Gradient Boosting)

## 📏 Evaluation Metrics (From Scratch)
- Confusion Matrix
- Macro F1 Score

---

# 🏗 Project Structure
ML-From-Scratch/
│
├── gradient_descent.py
├── Linear_Regression.py
├── Logestic_Regression.py
├── SVM.py
├── Simple_DecisionTree.py
├── RandomForest.py
├── XGBoost.py
├── evaluation_metrics.py
│
├── iris_test_models.py
├── digits_test_models.py
│
└── README.md

---

# 📊 Datasets Used

## 🌸 Iris Dataset
- 150 samples
- 4 features
- 3 classes
- Multi-class classification

## ✍️ Digits Dataset
- 1797 samples
- 64 features (8x8 image flattened)
- 10 classes (digits 0–9)
- Multi-class classification

---

# 🔬 Model Concepts

## Logistic Regression (OVR)
- Sigmoid activation
- Cross-entropy gradient descent
- One-vs-Rest for multi-class

## Linear SVM
- Hinge loss
- Maximum margin classifier
- OVR for multi-class

## Decision Tree
- Gini impurity
- Information Gain
- Recursive splitting

## Random Forest
- Bootstrap sampling
- Majority voting
- Ensemble learning (Bagging)

## XGBoost (Simplified)
- Gradient boosting concept
- Residual learning
- Sequential tree correction

---

# ⚙️ How It Works
Input Data
↓
Train/Test Split
↓
Model Training
↓
Prediction
↓
Confusion Matrix + F1 Score

---

# ▶️ How to Run

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ML-From-Scratch.git
cd ML-From-Scratch
```
### 📊 Example Output
```
===== Logistic Regression OVR =====
Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
F1 Score: 0.93
```
### 🧮 Mathematical Foundations Covered

* Gradient Descent

* Hinge Loss

* Cross-Entropy Loss

* Gini Impurity

* Information Gain

* Bagging

* Boosting

* One-vs-Rest Strategy

### 🚀 Why This Repository Is Valuable

✔ Demonstrates deep ML understanding

✔ Shows ability to implement algorithms from first principles

✔ Covers linear, tree-based, and ensemble methods

✔ Includes evaluation metrics from scratch

✔ Strong portfolio project for ML engineering roles

### 📌 Future Improvements

* Full regression trees for true XGBoost

* Feature scaling module

* Early stopping

* Cross-validation implementation

### 🤝 Contributions

Contributions are welcome!

You can:

Improve performance

Optimize vectorization

Add new ML algorithms

Enhance documentation


### 📜 License

This project is licensed under the MIT License.

### ⭐ If You Like This Project

Give it a star ⭐ and feel free to connect!
